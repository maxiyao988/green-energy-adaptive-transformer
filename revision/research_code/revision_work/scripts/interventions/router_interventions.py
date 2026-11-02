"""Step 9 - router interventions inside trained Adaptive Daily+Weekly models (inference only, no training).

Mechanism. In scripts/python/panel_pathformer_main_v2.py the per-sample routing weights of each branch are
produced by ScaleAggregator._adaptive_weights (lines 374-378: MLP router -> softmax) and consumed in
ScaleAggregator.forward (lines 403-404: z = sum_k w_k * expert_k). FrequencyAdaptivePathFormerBranch.forward
(lines 473-478) calls the aggregator; DualFrequencyAdaptivePathFormerRegressor.forward_with_router
(lines 572-577) fuses the two branches. Each intervention replaces ONLY the tensor returned by
_adaptive_weights on daily_branch.aggregator and weekly_branch.aggregator (an instance-level override that
first calls the original method, so the learned weights stay available); experts, router parameters, fusion
head and every other parameter are untouched. Models are built with the formal Experiment 4 factory
(panel_pathformer_experiment4_multiseed._make_experiment4_model, lines 411-418), loaded strictly in eval mode
on CPU, and fed the frozen V2 tensors from panel_pathformer_main_v2._build_training_loaders (line 1326).
I0 is verified against the Step 3 replayed predictions and against the formal inference helper
(_batched_inference, lines 860-887) before any intervention is evaluated.

Interventions: I0 learned; I1 equal (1/4); I2 global TRAIN-mean routing vector; I3 per-ticker TRAIN mean;
I4 per-date cross-sectional mean of the learned weights; I5 drop one scale (8 variants) and renormalise;
I6 equal weights in one branch only (2 variants).

Usage: python -B router_interventions.py --run-label original [--checkpoints GLOB ...] [--limit N]
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import argparse
import hashlib
import re
import sys
import time
import zlib
from pathlib import Path

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "skill"))
sys.dont_write_bytecode = True
from torch.utils.data import DataLoader  # noqa: E402
from scripts.python import panel_pathformer_main_v2 as formal  # noqa: E402
from scripts.python import panel_pathformer_experiment4_multiseed as exp4  # noqa: E402
from skill_common import np, pd, rank_ic_rows, mbb_index, holm, load_pred, canon_dates  # noqa: E402

OUT_BASE = ROOT / "revision_work" / "outputs" / "interventions"
E4 = ROOT / "dataset" / "audit" / "experiment4_multiseed"
PREDS = ROOT / "revision_work" / "outputs" / "preds"
DAILY = ["daily_w_5", "daily_w_10", "daily_w_20", "daily_w_30"]
WEEKLY = ["weekly_w_2", "weekly_w_4", "weekly_w_8", "weekly_w_13"]
BOOT_B = 10000
BOOT_SEED = 20261019
INTERVENTIONS = (["I0_learned", "I1_equal", "I2_global_train_mean", "I3_ticker_train_mean", "I4_date_mean"]
                 + [f"I5_drop_{c.replace('_w_', '')}" for c in DAILY + WEEKLY] + ["I6_equal_daily_only", "I6_equal_weekly_only"])


class Writer:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root.is_relative_to(OUT_BASE.resolve()) or os.environ.get("INTERVENTIONS_ALLOW_OUT")):
            raise ValueError(f"output must be under {OUT_BASE}")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Refusing to write into non-empty {self.root}")

    def _p(self, name):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(p)
        return p

    def csv(self, name, df):
        with self._p(name).open("x") as f:
            df.to_csv(f, index=False, float_format="%.10g")

    def text(self, name, s):
        with self._p(name).open("x") as f:
            f.write(s)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class RouterOverride:
    """Instance-level replacement of ScaleAggregator._adaptive_weights: calls the original method (learned
    weights) and, if an intervention is active, returns fn(learned, rows) instead. Nothing else changes."""

    def __init__(self, agg, branch):
        self.orig = agg._adaptive_weights  # bound original method
        self.branch = branch
        self.fn = None
        self.rows = None
        agg._adaptive_weights = self

    def __call__(self, expert_outputs):
        w = self.orig(expert_outputs)
        if self.fn is None:
            return w
        out = self.fn(w, self.rows, self.branch)
        return out.to(dtype=w.dtype, device=w.device)


def run_forward(model, loader, overrides, fn):
    preds, start = [], 0
    with torch.no_grad():
        for x_d, x_w, _ in loader:
            rows = np.arange(start, start + x_d.shape[0])
            for o in overrides:
                o.fn, o.rows = fn, rows
            preds.append(model(x_d, x_w).detach().cpu())
            start += x_d.shape[0]
    for o in overrides:
        o.fn, o.rows = None, None
    return torch.cat(preds).numpy().astype(np.float32)


def make_fns(learned, tick_idx, date_idx, train_mean, tick_mean):
    """learned[branch]: (N, 4) learned weights for this split; returns intervention -> fn(w, rows, branch)."""
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    date_mean = {}
    for b in ("daily", "weekly"):
        df = pd.DataFrame(learned[b]).groupby(date_idx).transform("mean").to_numpy()
        date_mean[b] = df
    fns = {
        "I0_learned": None,
        "I1_equal": lambda w, r, b: torch.full_like(w, 0.25),
        "I2_global_train_mean": lambda w, r, b: T(np.tile(train_mean[b], (len(r), 1))),
        "I3_ticker_train_mean": lambda w, r, b: T(tick_mean[b][tick_idx[r]]),
        "I4_date_mean": lambda w, r, b: T(date_mean[b][r]),
        "I6_equal_daily_only": lambda w, r, b: torch.full_like(w, 0.25) if b == "daily" else w,
        "I6_equal_weekly_only": lambda w, r, b: torch.full_like(w, 0.25) if b == "weekly" else w,
    }
    for bi, (b, cols) in enumerate((("daily", DAILY), ("weekly", WEEKLY))):
        for k, c in enumerate(cols):
            def drop(w, r, br, b=b, k=k):
                if br != b:
                    return w
                w2 = w.clone()
                w2[:, k] = 0.0
                return w2 / w2.sum(dim=1, keepdim=True)
            fns[f"I5_drop_{c.replace('_w_', '')}"] = drop
    return fns


def panel(keys, v):
    d = keys.assign(v=v).pivot(index="anchor_date", columns="ticker", values="v").sort_index()
    return d[sorted(d.columns)].to_numpy(np.float64)


def metrics(y, p, Y, P):
    ic = rank_ic_rows(P, Y)
    return dict(mse=float(np.mean((y - p) ** 2)), mean_rank_ic=float(np.nanmean(ic)), pearson=float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan,
                direction_accuracy=float(np.mean(np.sign(p) == np.sign(y))), pred_std_over_true_std=float(np.std(p) / np.std(y))), ic


def boot_stats(b, theta):
    lo, hi = np.quantile(b, [0.025, 0.975], method="linear")
    return dict(theta=float(theta), ci_lo=float(lo), ci_hi=float(hi), ci_excludes_0=bool(lo > 0 or hi < 0),
                p_centered=float((1 + np.sum(np.abs(b - theta) >= abs(theta))) / (len(b) + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--checkpoints", nargs="+", default=[str(E4 / "checkpoints" / "adaptive_seed*_*d.pt")])
    ap.add_argument("--summary-csv", default=str(E4 / "experiment4_training_summary.csv"), help="expected checkpoint SHA-256 ('none' to skip)")
    ap.add_argument("--reference-preds", default=str(PREDS / "adaptive_dw_h{h}_s{seed}_test.parquet"))
    ap.add_argument("--out-root", default=str(OUT_BASE))
    ap.add_argument("--boot-b", type=int, default=BOOT_B)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N checkpoints (timing / dry runs)")
    a = ap.parse_args()
    W = Writer(Path(a.out_root) / a.run_label)
    import glob
    ckpts = sorted({p for g in a.checkpoints for p in glob.glob(g)})
    cells = []
    for c in ckpts:
        m = re.search(r"adaptive_seed(\d+)_(\d+)d\.pt$", c)
        if m:
            cells.append(dict(path=c, seed=int(m.group(1)), h=int(m.group(2))))
    cells = sorted(cells, key=lambda c: (c["h"], c["seed"]))[: a.limit]
    expected = {}
    if a.summary_csv != "none":
        s = pd.read_csv(a.summary_csv)
        s = s[s.mechanism == "adaptive"]
        expected = {str((ROOT / p).resolve()): h_ for p, h_ in zip(s.checkpoint_path, s.checkpoint_sha256)}
    exp4._verify_frozen_dependencies()
    ds = formal.load_frozen_panel()
    formal._formal_check_dataset_contract(ds)
    blocking, verify, met_rows, ic_series, loss_series, wsum = [], [], [], {}, {}, []
    t0 = time.time()
    for h in sorted({c["h"] for c in cells}):
        train, val, test = formal._build_training_loaders("daily_weekly", h, ds, batch_size=formal.BATCH_SIZE)
        loaders = {"train": DataLoader(train.dataset, batch_size=formal.BATCH_SIZE, shuffle=False), "val": val, "test": test}
        keys = {sp: pd.DataFrame(dict(ticker=ds.ticker[ds.split == sp], anchor_date=pd.to_datetime(ds.anchor_date[ds.split == sp]).strftime("%Y-%m-%d")))
                for sp in loaders}
        ys = {sp: ds.y[h][ds.split == sp].astype(np.float64) for sp in loaders}
        for c in [c for c in cells if c["h"] == h]:
            seed = c["seed"]
            digest = sha(c["path"])
            exp_d = expected.get(str(Path(c["path"]).resolve()))
            if exp_d is not None and digest != exp_d:
                blocking.append(f"checkpoint hash mismatch {c['path']}")
                continue
            model = exp4._make_experiment4_model("adaptive")
            model.load_state_dict(torch.load(c["path"], map_location="cpu", weights_only=True), strict=True)
            model.to(torch.device("cpu"))
            model.eval()
            # learned weights + formal-helper predictions for every split
            learned, helper_pred = {}, {}
            for sp, ld in loaders.items():
                pr, dw, ww = exp4._batched_inference(model, ld, len(ld.dataset), torch.device("cpu"))
                learned[sp], helper_pred[sp] = {"daily": dw, "weekly": ww}, pr
            ov = [RouterOverride(model.daily_branch.aggregator, "daily"), RouterOverride(model.weekly_branch.aggregator, "weekly")]
            # ---- verification of I0
            ref = load_pred(a.reference_preds.format(h=h, seed=seed))
            mine = keys["test"].assign(y_pred=helper_pred["test"]).sort_values(["anchor_date", "ticker"], kind="mergesort").reset_index(drop=True)
            d_ref = float(np.max(np.abs(mine.y_pred.to_numpy(np.float64) - ref.y_pred.to_numpy(np.float64))))
            same_keys = mine[["ticker", "anchor_date"]].equals(ref[["ticker", "anchor_date"]])
            own_i0 = run_forward(model, loaders["test"], ov, None)
            d_own = float(np.max(np.abs(own_i0.astype(np.float64) - helper_pred["test"].astype(np.float64))))
            verify.append(dict(h=h, seed=seed, checkpoint=os.path.relpath(c["path"], ROOT), sha256=digest, keys_match=same_keys,
                               max_abs_delta_vs_replayed_preds=d_ref, max_abs_delta_override_loop_vs_formal_helper=d_own))
            if not same_keys or d_ref >= 1e-7 or d_own >= 1e-7:
                blocking.append(f"h={h} seed={seed}: I0 does not reproduce saved predictions (|Δ| {d_ref:.2e}, own loop {d_own:.2e}, keys {same_keys})")
                continue
            # TRAIN routing means (per branch; per ticker)
            tk = keys["train"].ticker.to_numpy()
            tickers = sorted(set(tk))
            tmap = {t: i for i, t in enumerate(tickers)}
            train_mean = {b: learned["train"][b].mean(0) for b in ("daily", "weekly")}
            tick_mean = {b: np.stack([learned["train"][b][tk == t].mean(0) for t in tickers]) for b in ("daily", "weekly")}
            for b, cols in (("daily", DAILY), ("weekly", WEEKLY)):
                wsum.append(dict(h=h, seed=seed, branch=b, **{f"train_mean_{cc}": v for cc, v in zip(cols, train_mean[b])},
                                 test_mean_sd_across_samples=float(learned["test"][b].std(0).mean()),
                                 test_share_var_between_dates=float(pd.DataFrame(learned["test"][b]).groupby(keys["test"].anchor_date.to_numpy()).transform("mean").var().sum()
                                                                    / pd.DataFrame(learned["test"][b]).var().sum())))
            for sp in ("test", "val"):
                kk = keys[sp]
                fns = make_fns(learned[sp], np.array([tmap[t] for t in kk.ticker]), kk.anchor_date.to_numpy(), train_mean, tick_mean)
                y = ys[sp]
                Y = panel(kk, y)
                base = None
                for iv in INTERVENTIONS:
                    p = (run_forward(model, loaders[sp], ov, fns[iv]) if fns[iv] is not None else run_forward(model, loaders[sp], ov, None)).astype(np.float64)
                    P = panel(kk, p)
                    m, ic = metrics(y, p, Y, P)
                    lossd = panel(kk, (y - p) ** 2).mean(1)
                    if iv == "I0_learned":
                        base = (m, ic, lossd)
                    met_rows.append(dict(split=sp, h=h, seed=seed, intervention=iv, **m, delta_mse=m["mse"] - base[0]["mse"],
                                         delta_rank_ic=m["mean_rank_ic"] - base[0]["mean_rank_ic"]))
                    if sp == "test":
                        loss_series[(h, seed, iv)] = lossd - base[2]
                        ic_series[(h, seed, iv)] = ic - base[1]
            print(f"h={h} seed={seed} done ({time.time() - t0:.0f}s)", flush=True)
    if blocking:
        W.csv("verification.csv", pd.DataFrame(verify))
        W.text("REPORT_STEP9.md", "# REPORT STEP 9\n\n## BLOCKING\n\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        raise SystemExit("BLOCKING: " + "; ".join(blocking))
    M = pd.DataFrame(met_rows)
    # ---------------------------------------------------------------- bootstraps (TEST)
    per_seed, joint = [], []
    for h in sorted({k[0] for k in loss_series}):
        seeds = sorted({k[1] for k in loss_series if k[0] == h})
        T = len(next(iter(v for k, v in loss_series.items() if k[0] == h)))
        rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, h, h]))
        didx = mbb_index(T, h, a.boot_b, rng)  # common date blocks: paired across seeds and interventions
        rngj = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, h, h, 1]))
        sidx = rngj.integers(0, len(seeds), size=(a.boot_b, len(seeds)))
        djdx = mbb_index(T, h, a.boot_b, rngj)
        Wt = np.zeros((a.boot_b, len(seeds)))
        np.add.at(Wt, (np.repeat(np.arange(a.boot_b), len(seeds)), sidx.ravel()), 1.0 / len(seeds))
        for iv in INTERVENTIONS[1:]:
            for metric, S in (("delta_mse", loss_series), ("delta_rank_ic", ic_series)):
                Mx = np.vstack([S[(h, s, iv)] for s in seeds])
                for i, s in enumerate(seeds):
                    b = Mx[i][didx].mean(1)
                    per_seed.append(dict(h=h, seed=s, intervention=iv, metric=metric, L=h, B=a.boot_b, **boot_stats(b, Mx[i].mean())))
                bj = np.take_along_axis(Wt @ Mx, djdx, axis=1).mean(1)
                joint.append(dict(h=h, intervention=iv, metric=metric, L=h, B=a.boot_b, n_seeds=len(seeds), **boot_stats(bj, Mx.mean())))
    PS, J = pd.DataFrame(per_seed), pd.DataFrame(joint)
    J["p_holm_over_h"] = np.nan
    for _, g in J.groupby(["intervention", "metric"]):
        for i, pv in holm(dict(zip(g.index, g.p_centered))).items():
            J.loc[i, "p_holm_over_h"] = pv
    # ---------------------------------------------------------------- seed summaries
    S = M.groupby(["split", "h", "intervention"]).agg(n_seeds=("seed", "count"), mse_mean=("mse", "mean"), delta_mse_mean=("delta_mse", "mean"),
                                                     delta_mse_sd=("delta_mse", "std"), n_seeds_delta_mse_neg=("delta_mse", lambda s: int((s < 0).sum())),
                                                     rank_ic_mean=("mean_rank_ic", "mean"), delta_rank_ic_mean=("delta_rank_ic", "mean"),
                                                     delta_rank_ic_sd=("delta_rank_ic", "std"), n_seeds_delta_ic_pos=("delta_rank_ic", lambda s: int((s > 0).sum())),
                                                     pearson_mean=("pearson", "mean"), direction_accuracy_mean=("direction_accuracy", "mean"),
                                                     pred_std_ratio_mean=("pred_std_over_true_std", "mean")).reset_index()
    S["delta_mse_pct_of_I0"] = S.delta_mse_mean / (S.mse_mean - S.delta_mse_mean)
    W.csv("verification.csv", pd.DataFrame(verify))
    W.csv("metrics_by_model_intervention.csv", M)
    W.csv("seed_summary.csv", S)
    W.csv("bootstrap_per_seed.csv", PS)
    W.csv("bootstrap_joint_seed_date.csv", J)
    W.csv("routing_weight_summary.csv", pd.DataFrame(wsum))
    from report9 import build_report
    W.text("REPORT_STEP9.md", build_report(a, pd.DataFrame(verify), M, S, PS, J, pd.DataFrame(wsum), torch.get_num_threads(), torch.get_num_interop_threads()))
    print("done", W.root, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
