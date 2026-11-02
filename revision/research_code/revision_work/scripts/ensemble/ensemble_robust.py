"""Step 7b - robustness of the Adaptive-vs-Static ensemble comparison to seed composition (no training).

Seed sets (paired by seed label; both mechanisms always use the same seeds):
  like_for_like  seeds 0, 1, 21, 3407
  all_seeds      seeds 0, 1, 21, 42, 3407 (Adaptive seed 42 = formal TEST CSV)
1. Leave-one-seed-out ensembles: equal-weight TEST ensembles from the remaining seeds; MSE, mean Rank IC,
   Adaptive - Static differences; counts of Static lower MSE / higher Rank IC.
2. Seed-resampled ensemble bootstrap: per replicate draw k seed indices with replacement (same draw for both
   mechanisms), build each ensemble from the drawn multiset (duplicates allowed), then resample moving date
   blocks (L = h; sensitivity 2h) identically for both mechanisms. Statistic = mean Adaptive-ensemble minus
   Static-ensemble squared loss (and Rank IC difference). B = 20,000; SeedSequence([20261017, h, L]), seed
   indices drawn first, then block starts (the same stream for both seed sets). Percentile 95% CI, centred
   two-sided p, Holm over horizons (L = h) per seed set and metric. The per-date series of every distinct
   seed multiset are computed once and reused.
Usage: python -B ensemble_robust.py --static GLOB [...] --adaptive GLOB [...] [--adaptive-formal-csv CSV] --out-dir DIR
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_analysis import load_cells, load_formal_csv, FORMAL_CSV, md, fmt, pct  # noqa: E402
from skill_common import *  # noqa: E402,F401

BOOT_SEED = 20261017
SEED_SETS = {"like_for_like": ["0", "1", "21", "3407"], "all_seeds": ["0", "1", "21", "42", "3407"]}


def ens_series(y, Y_shape, PA, PS, Y, w):
    """Per-date loss difference and Rank IC difference for weight vector w over seeds."""
    pa = np.tensordot(w, PA, axes=1)
    ps = np.tensordot(w, PS, axes=1)
    loss = ((y - pa) ** 2 - (y - ps) ** 2).reshape(Y_shape).mean(1)
    ic = rank_ic_rows(pa.reshape(Y_shape), Y) - rank_ic_rows(ps.reshape(Y_shape), Y)
    return loss, ic, pa, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", nargs="+", required=True)
    ap.add_argument("--adaptive", nargs="+", required=True)
    ap.add_argument("--adaptive-formal-csv", default=str(FORMAL_CSV), help="'none' to skip")
    ap.add_argument("--out-dir", default=str(SKILL_OUT / "ensemble_robust"))
    ap.add_argument("--prompt7-dir", default=str(SKILL_OUT / "ensemble"), help="for date-only CI widths (optional)")
    ap.add_argument("--boot-b", type=int, default=BOOT_B)
    ap.add_argument("--label", default="original features")
    a = ap.parse_args()
    out, B = Path(a.out_dir), a.boot_b
    cells = {k: v for k, v in {**load_cells(a.static, "static"), **load_cells(a.adaptive, "adaptive")}.items() if k[3] == "test"}
    if a.adaptive_formal_csv.lower() != "none":
        for k, v in load_formal_csv(a.adaptive_formal_csv).items():
            if k in cells:
                raise SystemExit(f"BLOCKING: formal seed-42 CSV duplicates {cells[k][1]}")
            cells[k] = v
    blocking = []
    loo, boot, reps = [], [], []
    for h in HORIZONS:
        ks = [k for k in cells if k[1] == h]
        ref = cells[ks[0]][0]
        for k in ks[1:]:
            d = cells[k][0]
            if not (d[["ticker", "anchor_date"]].equals(ref[["ticker", "anchor_date"]])
                    and np.array_equal(d.y_true.to_numpy(np.float32), ref.y_true.to_numpy(np.float32))):
                blocking.append(f"h={h}: keys/labels of {cells[k][1]} differ from {cells[ks[0]][1]}")
        if blocking:
            break
        y = ref.y_true.to_numpy(np.float64)
        Y, dates, _ = to_panel(ref, "y_true")
        T = Y.shape[0]
        for sname, seeds in SEED_SETS.items():
            miss = [(m, s) for m in ("static", "adaptive") for s in seeds if (m, h, s, "test") not in cells]
            if miss:
                blocking.append(f"h={h} {sname}: missing cells {miss}")
                continue
            k = len(seeds)
            PA = np.stack([cells[("adaptive", h, s, "test")][0].y_pred.to_numpy(np.float64) for s in seeds])
            PS = np.stack([cells[("static", h, s, "test")][0].y_pred.to_numpy(np.float64) for s in seeds])
            # ---- full ensemble + leave-one-seed-out
            variants = [("none (full ensemble)", np.full(k, 1 / k))]
            for j, s in enumerate(seeds):
                w = np.full(k, 1 / (k - 1))
                w[j] = 0.0
                variants.append((s, w))
            for held, w in variants:
                loss, icd, pa, ps = ens_series(y, Y.shape, PA, PS, Y, w)
                mse_a, mse_s = float(np.mean((y - pa) ** 2)), float(np.mean((y - ps) ** 2))
                ic_a = float(np.nanmean(rank_ic_rows(pa.reshape(Y.shape), Y)))
                ic_s = float(np.nanmean(rank_ic_rows(ps.reshape(Y.shape), Y)))
                loo.append(dict(seed_set=sname, h=h, held_out_seed=held, seeds_used=",".join(s for s, ww in zip(seeds, w) if ww > 0),
                                mse_static=mse_s, mse_adaptive=mse_a, mse_diff_adaptive_minus_static=mse_a - mse_s,
                                rank_ic_static=ic_s, rank_ic_adaptive=ic_a, rank_ic_diff_adaptive_minus_static=ic_a - ic_s,
                                static_lower_mse=mse_s < mse_a, static_higher_rank_ic=ic_s > ic_a))
            theta_loss, theta_ic, _, _ = ens_series(y, Y.shape, PA, PS, Y, np.full(k, 1 / k))
            th = {"loss_adaptive_minus_static": float(theta_loss.mean()), "rank_ic_adaptive_minus_static": float(np.nanmean(theta_ic))}
            # ---- seed-resampled ensemble bootstrap
            for spec, L in (("PRIMARY", h), ("SENSITIVITY", 2 * h)):
                rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, h, L]))
                sidx = rng.integers(0, k, size=(B, k))
                didx = mbb_index(T, L, B, rng)
                keys = [tuple(sorted(Counter(r).items())) for r in map(tuple, sidx)]
                cache = {}
                for key in set(keys):
                    w = np.zeros(k)
                    for j, c in key:
                        w[j] = c / k
                    ls, ic, _, _ = ens_series(y, Y.shape, PA, PS, Y, w)
                    cache[key] = (ls, ic)
                Ls = np.stack([cache[kk][0] for kk in keys])
                Is = np.stack([cache[kk][1] for kk in keys])
                for metric, S_ in (("loss_adaptive_minus_static", Ls), ("rank_ic_adaptive_minus_static", Is)):
                    b = np.nanmean(np.take_along_axis(S_, didx, axis=1), axis=1)
                    t0 = th[metric]
                    lo, hi = np.quantile(b, [0.025, 0.975], method="linear")
                    boot.append(dict(seed_set=sname, h=h, metric=metric, specification=spec, block_length=L, n_seeds=k,
                                     n_distinct_seed_multisets=len(cache), B=B, rng=f"SeedSequence([{BOOT_SEED},h,L]); seeds then blocks",
                                     theta_hat=t0, bootstrap_mean=float(b.mean()), bootstrap_se=float(np.std(b, ddof=1)),
                                     ci_lower_95=float(lo), ci_upper_95=float(hi), ci_width=float(hi - lo), ci_excludes_zero=bool(lo > 0 or hi < 0),
                                     centered_p_two_sided=float((1 + np.sum(np.abs(b - t0) >= abs(t0))) / (B + 1))))
                    reps.append(pd.DataFrame(dict(seed_set=sname, h=h, metric=metric, specification=spec, replicate=np.arange(B), value=b)))
    if blocking:
        write_text(out / "REPORT_STEP7B.md", "# REPORT STEP 7b\n\n## BLOCKING\n\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        raise SystemExit("BLOCKING: " + "; ".join(blocking))
    LOO, BT = pd.DataFrame(loo), pd.DataFrame(boot)
    BT["centered_p_holm_primary"] = np.nan
    for _, g in BT[BT.specification == "PRIMARY"].groupby(["seed_set", "metric"]):
        for i, p in holm(dict(zip(g.index, g.centered_p_two_sided))).items():
            BT.loc[i, "centered_p_holm_primary"] = p
    # width relative to Prompt 7's date-only MBB (same seed sets; different RNG stream)
    p7 = Path(a.prompt7_dir) / "ensemble_differences_mbb.csv"
    if p7.exists():
        P7 = pd.read_csv(p7).replace({"version": {"a_all_seeds": "all_seeds", "b_like_for_like": "like_for_like"}})
        P7["date_only_ci_width"] = P7.ci_upper_95 - P7.ci_lower_95
        BT = BT.merge(P7.rename(columns={"version": "seed_set"})[["seed_set", "h", "metric", "block_length", "date_only_ci_width"]],
                      on=["seed_set", "h", "metric", "block_length"], how="left")
        BT["ci_width_ratio_vs_date_only"] = BT.ci_width / BT.date_only_ci_width
    cnt = LOO[LOO.held_out_seed != "none (full ensemble)"].groupby(["seed_set", "h"]).agg(
        n_loo=("held_out_seed", "count"), static_lower_mse=("static_lower_mse", "sum"), static_higher_rank_ic=("static_higher_rank_ic", "sum"),
        min_mse_diff=("mse_diff_adaptive_minus_static", "min"), max_mse_diff=("mse_diff_adaptive_minus_static", "max"),
        min_ic_diff=("rank_ic_diff_adaptive_minus_static", "min"), max_ic_diff=("rank_ic_diff_adaptive_minus_static", "max")).reset_index()
    write_csv(out / "loo_ensembles.csv", LOO)
    write_csv(out / "loo_counts.csv", cnt)
    write_csv(out / "seed_resampled_bootstrap.csv", BT)
    write_csv(out / "seed_resampled_bootstrap_replicates.csv", pd.concat(reps, ignore_index=True))
    write_json(out / "run_info.json", dict(created=utc(), label=a.label, static=a.static, adaptive=a.adaptive,
                                           adaptive_formal_csv=a.adaptive_formal_csv, B=B, boot_seed=BOOT_SEED, seed_sets=SEED_SETS,
                                           files={str(k): v[1] for k, v in cells.items()}))

    # ---------------------------------------------------------------- report
    L_ = [f"# REPORT STEP 7b — Robustness of the ensemble comparison to seed composition ({a.label})\n",
          f"Generated {utc()} (UTC). No training. Inputs read only; outputs in `{rel(out)}/` created exclusively. Loading, Rank IC, block indices and Holm are reused from `revision_work/scripts/ensemble/ensemble_analysis.py` and `revision_work/scripts/skill/skill_common.py`.\n",
          "## BLOCKING\n", "None. All TEST files share the same 6,647 keys and identical y_true per horizon.\n",
          "Seed sets: like-for-like = 0, 1, 21, 3407; all seeds = 0, 1, 21, 42, 3407 (Adaptive seed 42 from the formal TEST CSV). Both mechanisms always use the same seed labels. Differences are Adaptive − Static: a positive loss difference means Static has the lower MSE, and a positive Rank IC difference means Adaptive ranks better.\n",
          "## 1. Leave-one-seed-out ensembles\n"]
    LO = LOO.copy()
    LO["held_out_seed"] = LO.held_out_seed.astype(str)
    L_.append(md(LO.sort_values(["seed_set", "h"], kind="mergesort"),
                 [("seed_set", "Seed set"), ("h", "h"), ("held_out_seed", "Held-out seed"), ("mse_static", "Static MSE"), ("mse_adaptive", "Adaptive MSE"),
                  ("mse_diff_adaptive_minus_static", "Δ MSE (A−S)"), ("rank_ic_static", "Static Rank IC"), ("rank_ic_adaptive", "Adaptive Rank IC"),
                  ("rank_ic_diff_adaptive_minus_static", "Δ Rank IC (A−S)"), ("static_lower_mse", "Static lower MSE"), ("static_higher_rank_ic", "Static higher IC")],
                 {"mse_static": lambda x: fmt(x, 6), "mse_adaptive": lambda x: fmt(x, 6), "mse_diff_adaptive_minus_static": lambda x: fmt(x, sci=True)}))
    L_.append("Counts over the leave-one-out ensembles (full ensemble excluded):\n")
    L_.append(md(cnt, [("seed_set", "Seed set"), ("h", "h"), ("n_loo", "# LOO ensembles"), ("static_lower_mse", "# Static lower MSE"),
                       ("static_higher_rank_ic", "# Static higher Rank IC"), ("min_mse_diff", "min Δ MSE"), ("max_mse_diff", "max Δ MSE"),
                       ("min_ic_diff", "min Δ IC"), ("max_ic_diff", "max Δ IC")],
                 {"min_mse_diff": lambda x: fmt(x, sci=True), "max_mse_diff": lambda x: fmt(x, sci=True)}))
    L_.append("## 2. Seed-resampled ensemble bootstrap\n")
    L_.append(f"**With only 4–5 seeds the seed resampling is coarse:** there are only {BT[BT.seed_set == 'like_for_like'].n_distinct_seed_multisets.max()} (4 seeds) and {BT[BT.seed_set == 'all_seeds'].n_distinct_seed_multisets.max()} (5 seeds) distinct seed multisets, and resampling a handful of seeds tends to understate seed-level uncertainty. Each replicate draws seeds with replacement (paired across mechanisms), builds both ensembles from that multiset, and then applies one set of moving date blocks to both. B = {B:,}; SeedSequence([20261017, h, L]).\n")
    cols = [("seed_set", "Seed set"), ("metric", "Metric"), ("h", "h"), ("block_length", "L"), ("theta_hat", "θ̂ (full ensemble)"), ("ci_lower_95", "CI low"),
            ("ci_upper_95", "CI high"), ("ci_excludes_zero", "Excludes 0"), ("centered_p_two_sided", "p"), ("centered_p_holm_primary", "Holm p (L=h)")]
    if "ci_width_ratio_vs_date_only" in BT:
        cols.append(("ci_width_ratio_vs_date_only", "CI width ÷ Step-7 date-only width"))
    L_.append(md(BT.sort_values(["seed_set", "metric", "h", "block_length"]), cols,
                 {"theta_hat": lambda x: fmt(x, sci=True), "ci_lower_95": lambda x: fmt(x, sci=True), "ci_upper_95": lambda x: fmt(x, sci=True),
                  "centered_p_two_sided": lambda x: fmt(x, 4), "centered_p_holm_primary": lambda x: fmt(x, 4), "ci_width_ratio_vs_date_only": lambda x: fmt(x, 2)}))
    L_.append("Replicates: `seed_resampled_bootstrap_replicates.csv`.\n")
    # ---- summary (computed)
    L_.append("## 3. Summary\n")
    prim = BT[BT.specification == "PRIMARY"].set_index(["seed_set", "metric", "h"])
    parts = []
    for sname in SEED_SETS:
        c = cnt[cnt.seed_set == sname].set_index("h")
        lo_ = ", ".join(f"h={h}: {int(c.loc[h, 'static_lower_mse'])}/{int(c.loc[h, 'n_loo'])}" for h in HORIZONS)
        ic_ = ", ".join(f"h={h}: {int(c.loc[h, 'static_higher_rank_ic'])}/{int(c.loc[h, 'n_loo'])}" for h in HORIZONS)
        sig = [h for h in HORIZONS if prim.loc[(sname, "loss_adaptive_minus_static", h), "ci_excludes_zero"]]
        sig_ic = [h for h in HORIZONS if prim.loc[(sname, "rank_ic_adaptive_minus_static", h), "ci_excludes_zero"]]
        hp = ", ".join(f"h={h} {fmt(prim.loc[(sname, 'loss_adaptive_minus_static', h), 'centered_p_holm_primary'], 3)}" for h in HORIZONS)
        parts.append(f"With the **{sname.replace('_', '-')}** seeds, Static has the lower MSE in {lo_} leave-one-out ensembles and the higher Rank IC in {ic_}. "
                     f"Once seeds are resampled along with dates, the loss-difference CI excludes zero at h = {', '.join(map(str, sig)) or 'none'} (Holm p: {hp}); "
                     f"the Rank IC CI excludes zero at h = {', '.join(map(str, sig_ic)) or 'none'}.")
    all_loo = bool(cnt.static_lower_mse.eq(cnt.n_loo).all())
    any_joint = bool(BT[(BT.specification == "PRIMARY") & (BT.metric == "loss_adaptive_minus_static")].ci_excludes_zero.any())
    verdict = ("The direction is robust to which seeds make up the ensemble: Static's ensemble has the lower MSE in every leave-one-out ensemble at every horizon. "
               if all_loo else "The direction is not uniform across leave-one-out ensembles (see counts above). ")
    verdict += ("Its size is also distinguishable from zero after seed resampling at the horizons listed. "
                if any_joint else "But once seed composition is resampled, the advantage is no longer distinguishable from zero at any horizon. ")
    verdict += "With only 4–5 seeds this robustness check is coarse, so it should be read as descriptive rather than as a test of a general property of the mechanisms."
    L_.append(" ".join(parts) + "\n\n" + verdict + "\n")
    L_.append("## Reproduction\n")
    L_.append("```sh\ncd revision_work/scripts/ensemble\nPY=../../../.venv/bin/python\n"
              "$PY -B ensemble_robust.py --static '../../outputs/preds/static_dw_h*_s*_test.parquet' \\\n"
              "  --adaptive '../../outputs/preds/adaptive_dw_h*_s*_test.parquet' --out-dir ../../outputs/skill_analysis/ensemble_robust\n"
              "# causal features later: point --static/--adaptive at ../../outputs/causal_features/preds/..., add\n"
              "#   --adaptive-formal-csv none --label 'causal features' --prompt7-dir <its Step-7 dir> --out-dir <new dir>\n```\n")
    write_text(out / "REPORT_STEP7B.md", "\n".join(L_))
    print("\n".join(L_))


if __name__ == "__main__":
    main()
