"""Step 7 - seed ensembles and validation-based seed selection (no training, no torch).

Reuses the Step 5 primitives in revision_work/scripts/skill/skill_common.py (prediction loading and key
checks, Newey-West, per-date Rank IC, moving-block indices, Holm, exclusive writers).

Sections
  1. Equal-weight TEST ensembles per mechanism/horizon: (a) all seeds (Static 5; Adaptive 5 incl. the formal
     seed-42 CSV), (b) like-for-like seeds 0,1,21,3407. Metrics: MSE, R2_OS vs GTM, Clark-West vs GTM (NW, lag
     h-1, one-sided), TS/XS decomposition, mean Rank IC with NW t. Adaptive-ensemble minus Static-ensemble loss
     and Rank IC differences with a date-level MBB (L=h primary, 2h sensitivity, B=20,000,
     SeedSequence([20261016, h, L]); percentile 95% CI; centred two-sided p; Holm over horizons, primary L).
  2. Validation-based seed selection on seeds that have VALIDATION predictions.
Usage:
  python -B ensemble_analysis.py --static GLOB [...] --adaptive GLOB [...] [--adaptive-formal-csv CSV] --out-dir DIR
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill"))
from skill_common import *  # noqa: E402,F401  (sets single-thread env vars before numpy)

LIKE_FOR_LIKE = ("0", "1", "21", "3407")
BOOT_SEED = 20261016
FORMAL_CSV = ROOT / "dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv"


# ---------------------------------------------------------------- inputs

def load_cells(patterns, mech):
    cells = {}
    for f in expand_globs(patterns):
        info = parse_name(f)
        key = (mech, info["h"], info["seed"], info["split"])
        if key in cells:
            raise SystemExit(f"BLOCKING: duplicate cell {key}: {f}")
        cells[key] = (load_pred(f, expect_rows=SPLIT_COUNTS[info["split"]]), rel(f))
    return cells


def load_formal_csv(path, mech="adaptive", seed="42"):
    d = pd.read_csv(path)
    d = d[(d.frequency_setting == "daily_weekly") & (d.seed.astype(str) == seed)]
    out = {}
    for h, g in d.groupby(d.horizon.astype(str).str.removesuffix("d").astype(int)):
        g = g[["ticker", "anchor_date", "y_true", "y_pred"]].copy()
        g["anchor_date"] = canon_dates(g.anchor_date)
        g = g.sort_values(["anchor_date", "ticker"], kind="mergesort").reset_index(drop=True)
        if len(g) != N_TEST_ROWS:
            raise SystemExit(f"BLOCKING: formal CSV h={h} has {len(g)} rows (expected 6,647)")
        out[(mech, h, seed, "test")] = (g, rel(path) + f" [daily_weekly, seed {seed}]")
    return out


# ---------------------------------------------------------------- metrics

def metrics(y, p, Y, P, gtm, h):
    mse = float(np.mean((y - p) ** 2))
    mse_g = float(np.mean((y - gtm) ** 2))
    f = (y - gtm) ** 2 - ((y - p) ** 2 - (gtm - p) ** 2)
    cw_m, cw_se, cw_t, _ = nw_t(f.reshape(Y.shape).mean(1), h - 1)
    ybar, pbar = Y.mean(1, keepdims=True), P.mean(1, keepdims=True)
    ts = float(np.mean((ybar - pbar) ** 2))
    xs = float(np.mean(((Y - ybar) - (P - pbar)) ** 2))
    ic = rank_ic_rows(P, Y)
    ic_m, ic_se, ic_t, n_ic = nw_t(ic, h - 1)
    return dict(mse=mse, r2os_vs_gtm=1 - mse / mse_g, cw_mean=cw_m, cw_nw_t=cw_t, cw_p_one_sided=float(stats.norm.sf(cw_t)),
                mse_ts=ts, mse_xs=xs, ts_share=ts / mse, decomp_abs_error=abs(ts + xs - mse),
                mean_rank_ic=ic_m, rank_ic_nw_t=ic_t, rank_ic_n_dates=n_ic, pred_sd=float(np.std(p))), ic


def mbb_diff(series, h, B):
    out = []
    theta = float(np.mean(series))
    for spec, L in (("PRIMARY", h), ("SENSITIVITY", 2 * h)):
        rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, h, L]))
        b = series[mbb_index(len(series), L, B, rng)].mean(1)
        lo, hi = np.quantile(b, [0.025, 0.975], method="linear")
        out.append(dict(specification=spec, block_length=L, theta_hat=theta, bootstrap_se=float(np.std(b, ddof=1)),
                        ci_lower_95=float(lo), ci_upper_95=float(hi), ci_excludes_zero=bool(lo > 0 or hi < 0),
                        centered_p_two_sided=float((1 + np.sum(np.abs(b - theta) >= abs(theta))) / (B + 1))))
    return out


# ---------------------------------------------------------------- report helpers

def fmt(x, nd=4, sci=False):
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, str):
        return x
    if x is None or not np.isfinite(x):
        return "NaN"
    if sci or (x != 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.{nd}f}"


def pct(x):
    return "NaN" if not np.isfinite(x) else f"{100 * x:+.2f}%"


def md(df, cols, f=None):
    f = f or {}
    s = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.to_dict("records"):  # keeps per-column dtypes (iterrows would upcast ints to float)
        s += "| " + " | ".join(str(r[c[0]]) if isinstance(r[c[0]], (int, np.integer)) and not isinstance(r[c[0]], bool)
                               else f.get(c[0], fmt)(r[c[0]]) for c in cols) + " |\n"
    return s


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", nargs="+", required=True)
    ap.add_argument("--adaptive", nargs="+", required=True)
    ap.add_argument("--adaptive-formal-csv", default=str(FORMAL_CSV), help="'none' to skip")
    ap.add_argument("--out-dir", default=str(SKILL_OUT / "ensemble"))
    ap.add_argument("--boot-b", type=int, default=BOOT_B)
    ap.add_argument("--label", default="original features")
    a = ap.parse_args()
    out = Path(a.out_dir)
    cells = {**load_cells(a.static, "static"), **load_cells(a.adaptive, "adaptive")}
    if a.adaptive_formal_csv.lower() != "none":
        for k, v in load_formal_csv(a.adaptive_formal_csv).items():
            if k in cells:
                raise SystemExit(f"BLOCKING: formal seed-42 CSV duplicates {cells[k][1]}")
            cells[k] = v
    blocking = []
    # ---- key / label integrity per split and horizon
    for h in HORIZONS:
        for sp in ("train", "val", "test"):
            ks = [k for k in cells if k[1] == h and k[3] == sp]
            if not ks:
                continue
            ref = cells[ks[0]][0]
            for k in ks[1:]:
                d = cells[k][0]
                if not d[["ticker", "anchor_date"]].equals(ref[["ticker", "anchor_date"]]):
                    blocking.append(f"h={h} {sp}: keys of {cells[k][1]} differ from {cells[ks[0]][1]}")
                elif not np.array_equal(d.y_true.to_numpy(np.float32), ref.y_true.to_numpy(np.float32)):
                    blocking.append(f"h={h} {sp}: y_true of {cells[k][1]} differs from {cells[ks[0]][1]}")
        if not any(k[1] == h and k[3] == "train" for k in cells):
            blocking.append(f"h={h}: no TRAIN file to compute the Global Train-Mean")
    if blocking:
        write_text(out / "REPORT_STEP7.md", "# REPORT STEP 7\n\n## BLOCKING\n\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        raise SystemExit("BLOCKING: " + "; ".join(blocking))

    seeds_of = lambda mech, h, sp: sorted([k[2] for k in cells if k[0] == mech and k[1] == h and k[3] == sp], key=int)
    single, ens, diffs, val_rows, sel_rows, reps = [], [], [], [], [], []
    gtms = {}
    for h in HORIZONS:
        tr = next(cells[k][0] for k in cells if k[1] == h and k[3] == "train")
        gtm = float(np.mean(tr.y_true.to_numpy(np.float64)))
        gtms[h] = gtm
        test_ref = next(cells[k][0] for k in cells if k[1] == h and k[3] == "test")
        y = test_ref.y_true.to_numpy(np.float64)
        Y, dates, _ = to_panel(test_ref, "y_true")
        preds, ics = {}, {}
        for mech in ("static", "adaptive"):
            for s in seeds_of(mech, h, "test"):
                d, src = cells[(mech, h, s, "test")]
                p = d.y_pred.to_numpy(np.float64)
                P, _, _ = to_panel(d, "y_pred")
                m, ic = metrics(y, p, Y, P, gtm, h)
                preds[(mech, s)], ics[(mech, s)] = p, ic
                single.append(dict(h=h, mechanism=mech, seed=s, source=src, **m))
        ens_pred = {}
        for version, sel in (("a_all_seeds", None), ("b_like_for_like", LIKE_FOR_LIKE)):
            for mech in ("static", "adaptive"):
                ss = [s for s in seeds_of(mech, h, "test") if sel is None or s in sel]
                if sel is not None and sorted(ss, key=int) != sorted(sel, key=int):
                    blocking.append(f"h={h} {mech}: like-for-like seeds missing ({ss})")
                p = np.mean([preds[(mech, s)] for s in ss], axis=0)
                P = p.reshape(Y.shape)
                m, ic = metrics(y, p, Y, P, gtm, h)
                sm = pd.DataFrame([r for r in single if r["h"] == h and r["mechanism"] == mech and r["seed"] in ss])
                ens.append(dict(h=h, version=version, mechanism=mech, seeds=",".join(ss), n_seeds=len(ss), **m,
                                single_seed_mean_mse=float(sm.mse.mean()), single_seed_sd_mse=float(sm.mse.std(ddof=1)),
                                single_seed_mean_rank_ic=float(sm.mean_rank_ic.mean()), single_seed_sd_rank_ic=float(sm.mean_rank_ic.std(ddof=1)),
                                ensemble_mse_gain=float(sm.mse.mean()) - m["mse"],
                                ensemble_mse_gain_pct=(float(sm.mse.mean()) - m["mse"]) / float(sm.mse.mean()),
                                ensemble_ic_gain=m["mean_rank_ic"] - float(sm.mean_rank_ic.mean())))
                ens_pred[(version, mech)] = (p, ic)
            pa, ica = ens_pred[(version, "adaptive")]
            ps, ics_ = ens_pred[(version, "static")]
            loss_t = ((y - pa) ** 2 - (y - ps) ** 2).reshape(Y.shape).mean(1)
            for metric, series in (("loss_adaptive_minus_static", loss_t), ("rank_ic_adaptive_minus_static", ica - ics_)):
                if not np.isfinite(series).all():
                    blocking.append(f"h={h} {version} {metric}: non-finite per-date series")
                    continue
                for r in mbb_diff(series, h, a.boot_b):
                    diffs.append(dict(h=h, version=version, metric=metric, n_dates=len(series), B=a.boot_b,
                                      rng=f"SeedSequence([{BOOT_SEED},h,L])", **r))
        # ---- section 2: validation-based selection
        for mech in ("static", "adaptive"):
            vs = [s for s in seeds_of(mech, h, "val") if (mech, h, s, "test") in cells]
            for s in vs:
                dv = cells[(mech, h, s, "val")][0]
                yv = dv.y_true.to_numpy(np.float64)
                Yv, _, _ = to_panel(dv, "y_true")
                Pv, _, _ = to_panel(dv, "y_pred")
                icv = rank_ic_rows(Pv, Yv)
                st_ = next(r for r in single if r["h"] == h and r["mechanism"] == mech and r["seed"] == s)
                val_rows.append(dict(h=h, mechanism=mech, seed=s, val_mse=float(np.mean((yv - dv.y_pred.to_numpy(np.float64)) ** 2)),
                                     test_mse=st_["mse"], val_mean_rank_ic=float(np.nanmean(icv)), test_mean_rank_ic=st_["mean_rank_ic"]))
            V = pd.DataFrame([r for r in val_rows if r["h"] == h and r["mechanism"] == mech])
            if len(V) >= 2:
                rho, prho = stats.spearmanr(V.val_mse, V.test_mse)
                rho_ic, _ = stats.spearmanr(V.val_mean_rank_ic, V.test_mean_rank_ic)
                best = V.loc[V.val_mse.idxmin()]
                all_test = pd.DataFrame([r for r in single if r["h"] == h and r["mechanism"] == mech])
                sel_rows.append(dict(h=h, mechanism=mech, n_seeds_with_val=len(V), seeds_with_val=",".join(V.seed),
                                     spearman_val_test_mse=float(rho), spearman_p_descriptive=float(prho),
                                     spearman_val_test_rank_ic=float(rho_ic), selected_seed=best.seed,
                                     selected_seed_val_mse=float(best.val_mse), selected_seed_test_mse=float(best.test_mse),
                                     selected_seed_test_rank_ic=float(best.test_mean_rank_ic),
                                     seed_avg_test_mse_val_seeds=float(V.test_mse.mean()), seed_avg_test_mse_all_seeds=float(all_test.mse.mean()),
                                     seed_avg_test_rank_ic_all_seeds=float(all_test.mean_rank_ic.mean()),
                                     selected_minus_avg_test_mse=float(best.test_mse - all_test.mse.mean()),
                                     test_mse_rank_of_selected=int((V.test_mse < best.test_mse).sum()) + 1))
    # Holm over horizons (primary specification), per version and metric
    D = pd.DataFrame(diffs)
    D["centered_p_holm_primary"] = np.nan
    for (v, m), g in D[D.specification == "PRIMARY"].groupby(["version", "metric"]):
        adj = holm({i: p for i, p in zip(g.index, g.centered_p_two_sided)})
        for i, p in adj.items():
            D.loc[i, "centered_p_holm_primary"] = p
    S1, E, VAL, SEL = pd.DataFrame(single), pd.DataFrame(ens), pd.DataFrame(val_rows), pd.DataFrame(sel_rows)
    if (E.decomp_abs_error >= 1e-10).any() or (S1.decomp_abs_error >= 1e-10).any():
        blocking.append("TS + XS decomposition error >= 1e-10")
    write_csv(out / "single_seed_test_metrics.csv", S1)
    write_csv(out / "ensemble_metrics.csv", E)
    write_csv(out / "ensemble_differences_mbb.csv", D)
    write_csv(out / "validation_vs_test_by_seed.csv", VAL)
    write_csv(out / "validation_seed_selection.csv", SEL)
    write_json(out / "run_info.json", dict(created=utc(), label=a.label, static=a.static, adaptive=a.adaptive,
                                           adaptive_formal_csv=a.adaptive_formal_csv, boot_b=a.boot_b, boot_seed=BOOT_SEED,
                                           gtm=gtms, files={f"{k}": v[1] for k, v in cells.items()}, blocking=blocking))

    # ---------------------------------------------------------------- report
    L = [f"# REPORT STEP 7 — Seed ensembles and validation-based seed selection ({a.label})\n",
         f"Generated {utc()} (UTC). No training. Inputs were read only; outputs in `{rel(out)}/` were created exclusively. Metric, Newey–West, Rank IC and block-bootstrap primitives are imported from `revision_work/scripts/skill/skill_common.py` (Step 5). The MBB uses `SeedSequence([20261016, h, L])` as pre-specified, so its draws differ from the Step 5 bootstrap.\n",
         "## BLOCKING\n", ("\n".join(f"- {b}" for b in blocking) if blocking else "None. All TEST files share the same 6,647 keys and identical y_true per horizon, and so do all validation (6,290) and train files. Every TS + XS decomposition closes to < 1e-10.") + "\n",
         "Inputs: " + "; ".join(sorted(set(v[1].split(' [')[0].rsplit('_s', 1)[0] for v in cells.values()))) + f". Adaptive seed 42 enters only through the formal TEST CSV (no validation predictions exist for it). GTM = pooled TRAIN-label mean: " + ", ".join(f"h={h}: {gtms[h]:.6f}" for h in HORIZONS) + ".\n",
         "Sign conventions: R²_OS > 0 = better than GTM. Loss difference = Adaptive − Static squared loss (positive = Static better). Rank IC difference = Adaptive − Static (positive = Adaptive better).\n",
         "## 1. Seed ensembles (TEST)\n", "### 1.1 Ensemble metrics vs single-seed averages\n"]
    E2 = E.sort_values(["version", "h", "mechanism"])
    L.append(md(E2, [("version", "Version"), ("h", "h"), ("mechanism", "Mechanism"), ("seeds", "Seeds"), ("mse", "Ensemble MSE"),
                     ("single_seed_mean_mse", "Mean single-seed MSE"), ("single_seed_sd_mse", "Seed SD of MSE"), ("ensemble_mse_gain_pct", "MSE gain from ensembling"),
                     ("r2os_vs_gtm", "R²_OS vs GTM"), ("cw_nw_t", "CW t (NW)"), ("cw_p_one_sided", "CW p"), ("mse_ts", "TS"), ("mse_xs", "XS"),
                     ("mean_rank_ic", "Ensemble Rank IC"), ("rank_ic_nw_t", "IC NW t"), ("single_seed_mean_rank_ic", "Mean single-seed Rank IC")],
                {"mse": lambda x: fmt(x, 6), "single_seed_mean_mse": lambda x: fmt(x, 6), "single_seed_sd_mse": lambda x: fmt(x, sci=True),
                 "ensemble_mse_gain_pct": pct, "r2os_vs_gtm": pct, "cw_nw_t": lambda x: fmt(x, 2), "cw_p_one_sided": lambda x: fmt(x, 3),
                 "mse_ts": lambda x: fmt(x, 6), "mse_xs": lambda x: fmt(x, 6), "rank_ic_nw_t": lambda x: fmt(x, 2)}))
    L.append("Per-seed TEST metrics: `single_seed_test_metrics.csv`.\n")
    L.append("### 1.2 Adaptive-ensemble − Static-ensemble differences (date-level MBB, B = %s)\n" % f"{a.boot_b:,}")
    D2 = D.sort_values(["version", "metric", "h", "block_length"])
    L.append(md(D2, [("version", "Version"), ("metric", "Metric"), ("h", "h"), ("block_length", "L"), ("theta_hat", "θ̂"), ("ci_lower_95", "CI low"),
                     ("ci_upper_95", "CI high"), ("ci_excludes_zero", "Excludes 0"), ("centered_p_two_sided", "p"), ("centered_p_holm_primary", "Holm p (L=h)")],
                {"theta_hat": lambda x: fmt(x, sci=True), "ci_lower_95": lambda x: fmt(x, sci=True), "ci_upper_95": lambda x: fmt(x, sci=True),
                 "centered_p_two_sided": lambda x: fmt(x, 4), "centered_p_holm_primary": lambda x: fmt(x, 4)}))
    L.append("These intervals resample dates only, conditional on the trained seeds (see Step 5 A3 for why seed uncertainty is not included).\n")
    L.append("## 2. Validation-based seed selection\n")
    L.append("**n = 4–5 seeds per mechanism and horizon. The correlations and selections below are descriptive only and support no inference.**\n")
    L.append(md(VAL.sort_values(["h", "mechanism", "val_mse"]), [("h", "h"), ("mechanism", "Mechanism"), ("seed", "Seed"), ("val_mse", "VAL MSE"), ("test_mse", "TEST MSE"),
                                                                  ("val_mean_rank_ic", "VAL Rank IC"), ("test_mean_rank_ic", "TEST Rank IC")],
                {"val_mse": lambda x: fmt(x, 6), "test_mse": lambda x: fmt(x, 6)}))
    L.append(md(SEL.sort_values(["h", "mechanism"]), [("h", "h"), ("mechanism", "Mechanism"), ("n_seeds_with_val", "n"), ("spearman_val_test_mse", "Spearman(VAL MSE, TEST MSE)"),
                                                      ("spearman_val_test_rank_ic", "Spearman(VAL IC, TEST IC)"), ("selected_seed", "Lowest-VAL-MSE seed"),
                                                      ("selected_seed_test_mse", "Its TEST MSE"), ("seed_avg_test_mse_all_seeds", "Seed-avg TEST MSE (all seeds)"),
                                                      ("selected_minus_avg_test_mse", "Selected − avg"), ("test_mse_rank_of_selected", "TEST-MSE rank of selected (1=best)"),
                                                      ("selected_seed_test_rank_ic", "Its TEST Rank IC"), ("seed_avg_test_rank_ic_all_seeds", "Seed-avg TEST Rank IC")],
                {"spearman_val_test_mse": lambda x: fmt(x, 2), "spearman_val_test_rank_ic": lambda x: fmt(x, 2), "selected_seed_test_mse": lambda x: fmt(x, 6),
                 "seed_avg_test_mse_all_seeds": lambda x: fmt(x, 6), "selected_minus_avg_test_mse": lambda x: fmt(x, sci=True)}))
    # ---- summary (computed)
    L.append("## 3. Summary\n")
    sent = []
    for version in ("a_all_seeds", "b_like_for_like"):
        e = E[E.version == version].set_index(["h", "mechanism"])
        lo_mse = [f"h={h}: {'Static' if e.loc[(h, 'static'), 'mse'] < e.loc[(h, 'adaptive'), 'mse'] else 'Adaptive'}" for h in HORIZONS]
        hi_ic = [f"h={h}: {'Static' if e.loc[(h, 'static'), 'mean_rank_ic'] > e.loc[(h, 'adaptive'), 'mean_rank_ic'] else 'Adaptive'}" for h in HORIZONS]
        sig = D[(D.version == version) & (D.specification == "PRIMARY") & D.ci_excludes_zero]
        sent.append(f"- **Ensemble {version}** — lower TEST MSE: {', '.join(lo_mse)}; higher Rank IC: {', '.join(hi_ic)}. "
                    f"Date-level MBB (L=h) CIs that exclude zero: " + (", ".join(f"{r.metric} h={r.h} (Holm p {fmt(r.centered_p_holm_primary, 4)})" for _, r in sig.iterrows()) or "none") + ".")
    s1 = S1.groupby(["h", "mechanism"]).agg(mse=("mse", "mean"), ic=("mean_rank_ic", "mean")).reset_index().set_index(["h", "mechanism"])
    sent.append("- **Single-seed averages (reference)** — lower mean TEST MSE: " + ", ".join(
        f"h={h}: {'Static' if s1.loc[(h, 'static'), 'mse'] < s1.loc[(h, 'adaptive'), 'mse'] else 'Adaptive'}" for h in HORIZONS) +
                "; higher mean Rank IC: " + ", ".join(f"h={h}: {'Static' if s1.loc[(h, 'static'), 'ic'] > s1.loc[(h, 'adaptive'), 'ic'] else 'Adaptive'}" for h in HORIZONS) + ".")
    ea = E[E.version == "a_all_seeds"].set_index(["h", "mechanism"])
    gains = ", ".join(f"h={h}: Static {pct(ea.loc[(h, 'static'), 'ensemble_mse_gain_pct'])} vs Adaptive {pct(ea.loc[(h, 'adaptive'), 'ensemble_mse_gain_pct'])}" for h in HORIZONS)
    sds = ", ".join(f"h={h}: Static {fmt(ea.loc[(h, 'static'), 'single_seed_sd_mse'], sci=True)} vs Adaptive {fmt(ea.loc[(h, 'adaptive'), 'single_seed_sd_mse'], sci=True)}" for h in HORIZONS)
    bigger_gain = [h for h in HORIZONS if ea.loc[(h, 'adaptive'), 'ensemble_mse_gain_pct'] > ea.loc[(h, 'static'), 'ensemble_mse_gain_pct']]
    bigger_sd = [h for h in HORIZONS if ea.loc[(h, 'adaptive'), 'single_seed_sd_mse'] > ea.loc[(h, 'static'), 'single_seed_sd_mse']]
    sent.append(f"- **Gain from ensembling (all seeds, % of mean single-seed MSE)** — {gains}. Seed-to-seed SD of TEST MSE: {sds}. "
                f"Adaptive gains more at h = {', '.join(map(str, bigger_gain)) or 'none'} and has the larger seed SD at h = {', '.join(map(str, bigger_sd)) or 'none'}; "
                + ("the larger gain matches the larger seed variance at every horizon." if bigger_gain == bigger_sd else "the larger gain does not line up with the larger seed variance at every horizon."))
    sl = SEL.set_index(["h", "mechanism"])
    sent.append("- **Validation-based selection** — Spearman(VAL MSE, TEST MSE): " + ", ".join(
        f"h={h} Static {fmt(sl.loc[(h, 'static'), 'spearman_val_test_mse'], 2)} / Adaptive {fmt(sl.loc[(h, 'adaptive'), 'spearman_val_test_mse'], 2)}" for h in HORIZONS if (h, 'static') in sl.index and (h, 'adaptive') in sl.index) +
                ". Lower TEST MSE among the validation-selected seeds: " + ", ".join(
        f"h={h}: {'Static' if sl.loc[(h, 'static'), 'selected_seed_test_mse'] < sl.loc[(h, 'adaptive'), 'selected_seed_test_mse'] else 'Adaptive'}" for h in HORIZONS if (h, 'static') in sl.index and (h, 'adaptive') in sl.index) +
                ". With 4–5 seeds this is descriptive only.")
    npos = int((E.r2os_vs_gtm > 0).sum())
    sent.append(f"- **Against the Global Train-Mean**, {npos} of {len(E)} ensembles have positive R²_OS. "
                + ("Ensembling makes neither mechanism beat the constant forecast." if npos == 0 else "Positive cases are listed in Table 1.1."))
    L.append("\n".join(sent) + "\n")
    L.append("## Reproduction\n")
    L.append("```sh\ncd revision_work/scripts/ensemble\nPY=../../../.venv/bin/python\n"
             "# original features (this report)\n"
             "$PY -B ensemble_analysis.py \\\n  --static   '../../outputs/preds/static_dw_h*_s*_*.parquet' \\\n"
             "  --adaptive '../../outputs/preds/adaptive_dw_h*_s*_*.parquet' \\\n  --out-dir  ../../outputs/skill_analysis/ensemble\n\n"
             "# later, causal-feature predictions (new output directory; drop the formal CSV if a seed-42 parquet exists)\n"
             "$PY -B ensemble_analysis.py \\\n  --static   '../../outputs/causal_features/preds/static_dw_h*_s*_*.parquet' \\\n"
             "  --adaptive '../../outputs/causal_features/preds/adaptive_dw_h*_s*_*.parquet' \\\n"
             "  --adaptive-formal-csv none --label 'causal features' \\\n  --out-dir  ../../outputs/skill_analysis/ensemble_causal\n```\n")
    L.append("File names must follow `{mechanism}_{config}_h{h}_s{seed}_{split}.parquet`. The script stops with a BLOCKING note if keys or labels differ between files, and it refuses to overwrite an existing output.\n")
    write_text(out / "REPORT_STEP7.md", "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
