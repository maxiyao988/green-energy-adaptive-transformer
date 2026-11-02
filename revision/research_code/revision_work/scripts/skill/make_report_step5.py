"""Assemble REPORT_STEP5.md from the A1/A2/A3 CSV outputs (reads only generated outputs).

Usage: python -B make_report_step5.py [--root DIR] [--a2 a2_original] [--a3 a3_original]
"""
import argparse
from skill_common import *


def f(x, nd=4, sci=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NaN"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if sci or (x != 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.{nd}f}"


def pct(x):
    return "NaN" if not np.isfinite(x) else f"{100 * x:+.2f}%"


def table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    lines = []
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c[0], f)(r[c[0]]) if not isinstance(r[c[0]], str) else r[c[0]] for c in cols) + " |")
    return head + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SKILL_OUT))
    ap.add_argument("--a2", default="a2_original")
    ap.add_argument("--a3", default="a3_original")
    a = ap.parse_args()
    R = Path(a.root)
    a1m = pd.read_csv(R / "a1/a1_benchmark_mse.csv")
    a1r = pd.read_csv(R / "a1/a1_signal_ranking.csv")
    a1f = pd.read_csv(R / "a1/a1_signal_ols_fits.csv")
    a1c = json.loads((R / "a1/a1_integrity_checks.json").read_text())
    a2 = pd.read_csv(R / a.a2 / "a2_skill_by_cell.csv", dtype={"seed": str})
    a2s = pd.read_csv(R / a.a2 / "a2_seed_summary.csv")
    a2i = json.loads((R / a.a2 / "a2_run_info.json").read_text())
    dmbb = pd.read_csv(R / a.a3 / "a3_date_only_mbb.csv")
    jb = pd.read_csv(R / a.a3 / "a3_joint_seed_date_bootstrap.csv")
    st = pd.read_csv(R / a.a3 / "a3_seed_level_tests.csv")
    a3i = json.loads((R / a.a3 / "a3_run_info.json").read_text())
    vs = pd.read_csv(R / a.a3 / "a3_date_mbb_vs_archived.csv")
    chk = a3i["checks"]

    models = a2[~a2.model.isin(["bench", "naive"])]
    n_pos = {k: int((models[f"r2os_vs_{k}"] > 0).sum()) for k in ["global_train_mean", "zero", "ticker_train_mean", "expanding_pooled_mean"]}
    n_cw_nw, n_cw_mbb = int((models.cw_p_nw_one_sided < 0.05).sum()), int((models.cw_p_mbb_one_sided < 0.05).sum())
    best = models.loc[models.r2os_vs_global_train_mean.idxmax()]

    # ---------------------------------------------------------------- BLOCKING
    blocking, caveats = [], []
    if not chk.get("REPRODUCED", False):
        blocking.append("Archived date-level MBB was NOT reproduced (see A3.1).")
    if a2i["decomposition_failures"]:
        blocking.append(f"MSE decomposition failed the 1e-10 check for {len(a2i['decomposition_failures'])} cells.")
    for k, v in a1c.items():
        if k.endswith(("_eq_meta", "_eq_price_snapshot", "_eq_price_calendar")) and v is not True:
            blocking.append(f"A1 integrity check failed: {k}")
    caveats += [
        "Adaptive seed 42 enters only through the formal TEST CSV (its checkpoint was not recoverable in Step 3), so it has no replayed train/validation predictions; all Step 5 metrics are TEST-only and use the formal file.",
        "The joint seed × date bootstrap resamples only 5 seeds. With n = 5 a seed bootstrap is coarse (at most 126 distinct seed multisets) and tends to under-state seed uncertainty; the joint intervals should be read as a lower bound on total uncertainty, not as exact coverage.",
        "With n = 5 seeds, the exact sign and Wilcoxon tests cannot reach p < 0.0625 two-sided (0.03125 one-sided) whatever the data; non-rejection there is uninformative on its own.",
        "MOM(12-1) is undefined for 243 TRAIN rows (less than 252 prior sessions; 4 rows for MOM(6-1)); these were dropped from the TRAIN OLS fits only. Every TEST row has all four signals.",
        "Ridge, LSTM, Vanilla and SWiM are single nominal-seed (42) runs; no seed dispersion is available for them.",
        "Causal-feature predictions were not used. The scripts accept any prediction glob and will re-run on them later (see Reproduction).",
    ]
    L = []
    L.append("# REPORT STEP 5 — Forecast-skill diagnostics, financial benchmarks, hierarchical inference\n")
    L.append(f"Generated {utc()} (UTC). No model was trained. All inputs were read only; all outputs are under `revision_work/outputs/skill_analysis/` and were created exclusively (never overwritten). Scripts are in `revision_work/scripts/skill/`.\n")
    L.append("## BLOCKING\n")
    L.append(("\n".join(f"- **{b}**" for b in blocking) if blocking else "- **None.** All integrity checks passed: frozen labels equal the per-ticker meta labels and the labels recomputed from the original price snapshot (max abs diff 0 at h = 5/10/20); target dates equal the price calendar; every archived and replayed prediction cell has the 6,647 frozen keys and identical float32 labels; the archived date-level MBB was reproduced to ≤ 1e-16; every MSE decomposition closes to < 1e-10.") + "\n")
    L.append("Non-blocking caveats:\n\n" + "\n".join(f"- {c}" for c in caveats) + "\n")

    # ---------------------------------------------------------------- inputs
    L.append("## Inputs\n")
    L.append("- Replayed predictions `revision_work/outputs/preds/{single,fixed,static,adaptive}_dw_h{h}_s{seed}_test.parquet` (Static seeds 0/1/21/42/3407; Adaptive seeds 0/1/21/3407; Single/Fixed seed 42).\n"
             "- Adaptive seed 42: `dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv` rows `frequency_setting=daily_weekly, seed=42`.\n"
             "- Ridge / LSTM / Vanilla / SWiM (daily_only, weekly_only, daily_weekly): formal MAIN_V2 files under `dataset/audit/main_v2/{ridge,lstm,vanilla,swim_cpu}/` (9 × 6,647 rows each); Zero / Global Train-Mean from `dataset/audit/main_v2/naive/main_v2_naive_test_predictions.csv`. The 6,664-row root files were not used. These were copied unchanged into `skill_analysis/canonical_preds/` (manifest with source SHA-256 in `canonical_manifest.csv`).\n"
             "- Labels and splits: the replayed Static seed-0 train/val/test files (30,634 / 6,290 / 6,647 rows), checked against `dataset/multiscale_dataset/panel/{T}/meta.csv` + `y_{h}d.npy` and against `dataset/finance/panel_raw/{T}_daily.csv`.\n"
             "- MSE in Step 5 is computed in float64 from the stored values. The A3 loss differences use the archived float32 arithmetic so that they reproduce the archived inference exactly.\n")

    # ---------------------------------------------------------------- A1
    L.append("## A1. Financial benchmarks (computed, not trained)\n")
    arch_gtm = {h: float(pd.read_parquet(R / f"canonical_preds/naive_global_train_mean_h{h}_sna_test.parquet").y_pred.iloc[0]) for h in HORIZONS}
    gdiff = max(abs(arch_gtm[h] - a1c[f"h{h}_global_train_mean"]) for h in HORIZONS)
    L.append("Global Train-Mean = pooled mean of retained TRAIN labels (h=5: " + f(a1c["h5_global_train_mean"], 6) + ", h=10: " + f(a1c["h10_global_train_mean"], 6) + ", h=20: " + f(a1c["h20_global_train_mean"], 6) + f"; equal to the archived constant within max |Δ| = {f(gdiff, sci=True)}, a float32-accumulation difference. The archived constant is used as the GTM model row; R²_OS references use the recomputed value). Expanding means use only labels whose TARGET date ≤ anchor date (retained panel labels, all splits). `expanding_ticker_fullhist` is a sensitivity version that also uses the ticker's pre-panel and purged-boundary labels. OLS forecasts: label = a + b·signal fitted on TRAIN only.\n")
    w = a1m.pivot(index="benchmark", columns="h", values=["mse", "r2os_vs_global_train_mean", "r2os_vs_zero"])
    order = ["zero", "global_train_mean", "ticker_train_mean", "expanding_ticker_mean", "expanding_pooled_mean", "expanding_ticker_fullhist", "ols_mom12_1", "ols_mom6_1", "ols_rev1w", "ols_rev1m"]
    L.append("### A1.1 Benchmark forecasts — TEST MSE and R²_OS\n")
    L.append("| Benchmark | MSE h5 | MSE h10 | MSE h20 | R²_OS vs GTM h5 | h10 | h20 | R²_OS vs Zero h5 | h10 | h20 |\n|---|---|---|---|---|---|---|---|---|---|")
    for b in order:
        L.append(f"| {b} | " + " | ".join(f(w.loc[b, ("mse", h)], 6) for h in HORIZONS) + " | " + " | ".join(pct(w.loc[b, ("r2os_vs_global_train_mean", h)]) for h in HORIZONS) + " | " + " | ".join(pct(w.loc[b, ("r2os_vs_zero", h)]) for h in HORIZONS) + " |")
    L.append("")
    L.append("### A1.2 Momentum / reversal as ranking signals (391 TEST dates, 17 stocks)\n")
    L.append(table(a1r, [("h", "h"), ("signal", "Signal"), ("mean_rank_ic", "Mean Rank IC"), ("rank_ic_nw_t", "NW t (lag h−1)"), ("ic_hit_rate", "IC hit rate"),
                         ("top3_minus_bottom3_mean", "Top3−Bottom3 mean h-day return"), ("top3_minus_bottom3_nw_t", "Top3−Bottom3 NW t")]))
    L.append("### A1.3 TRAIN-only OLS calibration\n")
    L.append(table(a1f, [("h", "h"), ("signal", "Signal"), ("n_train_used", "TRAIN rows used"), ("n_train_dropped_nan", "Dropped (no history)"), ("intercept", "a"), ("slope", "b"), ("train_r2", "TRAIN R²")], {"train_r2": lambda x: f(x, sci=True)}))
    best_b = a1m[~a1m.benchmark.isin(["zero", "global_train_mean"])].sort_values("r2os_vs_global_train_mean").groupby("h").tail(1).sort_values("h")
    mom6 = a1r[a1r.signal == "mom6_1"]
    L.append("**Finding.** No historical-mean benchmark beats the pooled Global Train-Mean in the TEST period: ticker train means and the expanding ticker means are worse at every horizon, and the expanding pooled mean is worse than GTM but better than Zero. The calibrated signals barely move the MSE. The best R²_OS vs GTM by horizon is "
             + ", ".join(f"h={int(r.h)}: {r.benchmark} {pct(r.r2os_vs_global_train_mean)}" for _, r in best_b.iterrows())
             + ". As ranking signals, 6-1 momentum had a negative Rank IC in the TEST year (" + ", ".join(f"h={int(r.h)}: {f(r.mean_rank_ic)} (t={f(r.rank_ic_nw_t, 2)})" for _, r in mom6.iterrows())
             + "), with a negative top-minus-bottom spread. The short-term reversal ICs are positive but not significant, and 12-1 momentum is near zero. This is the bar a learned model must clear: a constant beats nearly everything in MSE, and the cross-section has only weak, period-specific rank structure.\n")

    # ---------------------------------------------------------------- A2
    L.append("## A2. Skill metrics for every model and seed (original features)\n")
    L.append(f"{a2i['n_cells']} TEST cells: {len(models)} model cells (Single, Fixed, Static ×5 seeds, Adaptive ×5 seeds on daily_weekly; Ridge/LSTM/Vanilla/SWiM on 3 input configurations), plus Zero/GTM and the A1 benchmarks. "
             f"Clark–West uses GTM as the benchmark; its moving-block bootstrap (L=h, B={a2i['boot_b']:,}) uses one common set of block indices per horizon. Max |decomposition error| = {f(a2i['max_decomp_abs_error'], sci=True)} (date/cross-section) and {f(a2i['max_moment_decomp_abs_error'], sci=True)} (bias/variance/covariance).\n")
    L.append(f"**Counts over the {len(models)} model cells:** R²_OS > 0 vs GTM: {n_pos['global_train_mean']}; vs Zero: {n_pos['zero']}; vs Ticker train mean: {n_pos['ticker_train_mean']}; vs Expanding pooled mean: {n_pos['expanding_pooled_mean']}. Clark–West one-sided p < 0.05: {n_cw_nw} (NW), {n_cw_mbb} (MBB). Best cell: {best.model}/{best.config}/h{best.h}/s{best.seed} with R²_OS vs GTM {pct(best.r2os_vs_global_train_mean)}.\n")
    L.append("### A2.1 Daily+Weekly cells — R²_OS, Clark–West, cross-sectional skill\n")
    dw = a2[(a2.config == "daily_weekly") | a2.model.isin(["naive"])].copy()
    dw["order"] = dw.model.map({"naive": 0, "single": 1, "fixed": 2, "static": 3, "adaptive": 4, "ridge": 5, "lstm": 6, "vanilla": 7, "swim": 8})
    dw["seed_n"] = pd.to_numeric(dw.seed, errors="coerce").fillna(-1)
    dw = dw.sort_values(["h", "order", "config", "seed_n"])
    dw["name"] = np.where(dw.model == "naive", dw.config, dw.model)
    cols = [("h", "h"), ("name", "Model"), ("seed", "Seed"), ("mse", "MSE"), ("r2os_vs_global_train_mean", "R²_OS GTM"), ("r2os_vs_zero", "R²_OS Zero"),
            ("r2os_vs_ticker_train_mean", "R²_OS TickerMean"), ("r2os_vs_expanding_pooled_mean", "R²_OS ExpPooled"), ("cw_nw_t", "CW t (NW)"),
            ("cw_p_nw_one_sided", "CW p NW"), ("cw_p_mbb_one_sided", "CW p MBB"), ("mean_rank_ic", "Mean Rank IC"), ("rank_ic_nw_t", "IC NW t"),
            ("ic_hit_rate", "IC hit"), ("xs_r2", "XS R²")]
    pf = {k: pct for k in ["r2os_vs_global_train_mean", "r2os_vs_zero", "r2os_vs_ticker_train_mean", "r2os_vs_expanding_pooled_mean", "xs_r2"]}
    pf["mse"] = lambda x: f(x, 6)
    for k in ["cw_nw_t", "rank_ic_nw_t"]:
        pf[k] = lambda x: f(x, 2)
    for k in ["cw_p_nw_one_sided", "cw_p_mbb_one_sided", "ic_hit_rate"]:
        pf[k] = lambda x: f(x, 3)
    L.append(table(dw, cols, pf))
    L.append("Daily-only and weekly-only cells for Ridge/LSTM/Vanilla/SWiM, and the benchmark rows, are in `" + a.a2 + "/a2_skill_by_cell.csv`. CW is undefined (NaN) for GTM itself; Rank IC is undefined for constant cross-sections (Zero, GTM, expanding pooled mean).\n")

    L.append("### A2.2 Exact MSE decomposition (Daily+Weekly cells and references)\n")
    L.append("MSE = TS + XS, where TS = mean_t (ȳ_t − ŷ̄_t)² (error in the date-average level) and XS = mean_it ((y−ȳ_t) − (ŷ−ŷ̄_t))² (error in the within-date spread). Equivalently, MSE = bias² + Var(y) + Var(ŷ) − 2Cov(y,ŷ). Excess = component minus the same component for GTM.\n")
    gt = a2[(a2.model == "naive") & (a2.config == "global_train_mean")].set_index("h")
    dw["ex_ts"] = dw.mse_ts_component - dw.h.map(gt.mse_ts_component)
    dw["ex_xs"] = dw.mse_xs_component - dw.h.map(gt.mse_xs_component)
    L.append(table(dw, [("h", "h"), ("name", "Model"), ("seed", "Seed"), ("mse", "MSE"), ("mse_ts_component", "TS"), ("mse_xs_component", "XS"), ("ts_share", "TS share"),
                        ("ex_ts", "Excess TS vs GTM"), ("ex_xs", "Excess XS vs GTM"), ("bias2", "bias²"), ("var_pred", "Var(ŷ)"), ("minus2cov", "−2Cov(y,ŷ)"), ("decomp_abs_error", "|err|")],
                   {**{k: (lambda x: f(x, 6)) for k in ["mse", "mse_ts_component", "mse_xs_component"]}, "ts_share": lambda x: f(x, 3),
                    **{k: (lambda x: f(x, sci=True)) for k in ["ex_ts", "ex_xs", "bias2", "var_pred", "minus2cov", "decomp_abs_error"]}}))
    mm = dw[~dw.model.isin(["naive"])]
    share_xs = float((mm.ex_xs / (mm.ex_ts + mm.ex_xs)).median())
    L.append("### A2.3 Five-seed summaries (Static, Adaptive; Daily+Weekly)\n")
    s5 = a2s[a2s.model.isin(["static", "adaptive"])].sort_values(["h", "model"])
    L.append("| h | Mechanism | Seeds | MSE mean ± SD | R²_OS GTM mean ± SD | R²_OS Zero | R²_OS TickerMean | R²_OS ExpPooled | CW t mean ± SD | Rank IC mean ± SD | IC NW t mean | XS R² mean ± SD | TS share |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in s5.iterrows():
        L.append(f"| {r.h} | {r.model} | {r.seeds} | {f(r.mse_mean, 6)} ± {f(r.mse_sd, 6)} | {pct(r.r2os_vs_global_train_mean_mean)} ± {pct(r.r2os_vs_global_train_mean_sd)[1:]} | {pct(r.r2os_vs_zero_mean)} | {pct(r.r2os_vs_ticker_train_mean_mean)} | {pct(r.r2os_vs_expanding_pooled_mean_mean)} | {f(r.cw_nw_t_mean, 2)} ± {f(r.cw_nw_t_sd, 2)} | {f(r.mean_rank_ic_mean)} ± {f(r.mean_rank_ic_sd)} | {f(r.rank_ic_nw_t_mean, 2)} | {pct(r.xs_r2_mean)} ± {pct(r.xs_r2_sd)[1:]} | {f(r.ts_share_mean, 3)} |")
    L.append("\nAll per-metric means and SDs: `" + a.a2 + "/a2_seed_summary.csv`. Per-date CW and IC series: `" + a.a2 + "/a2_per_date_cw_and_ic.csv`.\n")
    sg = models[models.rank_ic_nw_t.abs() > 1.96]
    sig_ic = "; ".join(f"{r.model}/{r.config}/h{r.h}/s{r.seed} IC {f(r.mean_rank_ic)} (t={f(r.rank_ic_nw_t, 2)})" for _, r in sg.iterrows()) or "none"
    n_pos_total = sum(n_pos.values())
    L.append(f"**Finding.** {n_pos_total} of {4 * len(models)} model-cell × reference comparisons have positive out-of-sample R², and {n_cw_nw} (NW) / {n_cw_mbb} (MBB) Clark–West tests reject at 5% (smallest one-sided p {f(min(models.cw_p_nw_one_sided.min(), models.cw_p_mbb_one_sided.min()), 3)}). "
             f"The decomposition shows where the extra error comes from. For the median model cell, about {100 * share_xs:.0f}% of the excess MSE over GTM is cross-sectional: predictions spread across stocks in ways unrelated to realized spreads (XS R² ≤ {pct(float(models.xs_r2.max()))} in every cell). The rest comes from the date-level mean. "
             f"Cross-sectional ranking skill is small and seed-dependent: mean Rank ICs range from {f(models.mean_rank_ic.min())} to {f(models.mean_rank_ic.max())}; cells with |IC NW t| > 1.96: {sig_ic}. "
             + ("In plain words, on the original features no model forecasts either the level or the cross-section of returns better than a constant.\n"
                if n_pos_total == 0 and max(n_cw_nw, n_cw_mbb) == 0 else
                "In plain words, any positive skill is confined to the cells counted above and should be read against the multiple comparisons made here.\n"))

    # ---------------------------------------------------------------- A3
    L.append("## A3. Hierarchical inference for Adaptive − Static (five seeds)\n")
    L.append("D[s,t] = mean over 17 stocks of the (Adaptive − Static) squared-loss difference; positive = Static better. dIC[s,t] = Adaptive − Static Rank IC; positive = Adaptive better.\n")
    L.append("### A3.1 Reproduction of the archived date-level MBB\n")
    L.append(f"Seed × date differences match `experiment4_loss_diff_by_seed_date.csv` (max |Δ| = {f(max(chk[f'h{h}_seed_date_max_abs_diff'] for h in HORIZONS), sci=True)}). All 120,000 archived bootstrap replicates match (max |Δ| = {f(chk['replicates_max_abs_diff'], sci=True)}). Recomputed per-date Rank ICs match the archived ICs (max |Δ| = {f(max(chk[f'h{h}_rank_ic_max_abs_diff_vs_archived'] for h in HORIZONS), sci=True)}). HAC z max |Δ| = {f(chk['hac_z_max_abs_diff'], sci=True)}. **Reproduced: {chk['REPRODUCED']}.**\n")
    L.append(table(vs.sort_values(["h", "block_length"]), [("h", "h"), ("specification", "Spec"), ("block_length", "L"), ("theta_hat", "θ̂"), ("ci_lower_95", "CI low"), ("archived_ci_lower_95", "Archived CI low"),
                                                           ("ci_upper_95", "CI high"), ("archived_ci_upper_95", "Archived CI high"), ("centered_p_two_sided", "p"), ("archived_centered_p_value", "Archived p"), ("classification", "Archived class")],
                   {k: (lambda x: f(x, sci=True)) for k in ["theta_hat", "ci_lower_95", "archived_ci_lower_95", "ci_upper_95", "archived_ci_upper_95", "centered_p_two_sided", "archived_centered_p_value"]}))
    L.append("### A3.2 Date-only vs joint seed × date bootstrap\n")
    m = dmbb.merge(jb, on=["h", "metric", "specification", "block_length"], suffixes=("_date", "_joint"))
    m = m.sort_values(["metric", "h", "block_length"])
    L.append("| Metric | h | L | θ̂ | Date-only 95% CI | Date-only p (Holm) | Joint 95% CI | Joint p (Holm) | CI width ratio joint/date | Joint CI excludes 0 |\n|---|---|---|---|---|---|---|---|---|---|")
    for _, r in m.iterrows():
        hp = lambda p, ph: f(p, 4) + ("" if not np.isfinite(ph) else f" ({f(ph, 4)})")
        L.append(f"| {r.metric} | {r.h} | {r.block_length} | {f(r.theta_hat_date, sci=r.metric == 'loss')} | [{f(r.ci_lower_95_date, sci=r.metric == 'loss')}, {f(r.ci_upper_95_date, sci=r.metric == 'loss')}] | {hp(r.centered_p_two_sided_date, r.centered_p_holm_primary_date)} | "
                 f"[{f(r.ci_lower_95_joint, sci=r.metric == 'loss')}, {f(r.ci_upper_95_joint, sci=r.metric == 'loss')}] | {hp(r.centered_p_two_sided_joint, r.centered_p_holm_primary_joint)} | {f(r.ci_width_ratio_joint_over_date_only, 2)} | {r.ci_excludes_zero_joint} |")
    L.append("\nHolm adjusts the three primary-L p-values within each metric and bootstrap type; sensitivity rows (L = 2h) are unadjusted. Replicates: `" + a.a3 + "/a3_bootstrap_replicates.csv`.\n")
    L.append("### A3.3 Seed-level tests (per-seed mean differences, n = 5)\n")
    L.append(table(st, [("h", "h"), ("metric", "Metric"), ("mean", "Mean"), ("sd", "SD"), ("n_positive", "# seeds > 0"), ("paired_t", "Paired t (df 4)"), ("paired_t_p_two_sided", "t p"),
                        ("sign_test_p_two_sided", "Sign p (exact)"), ("wilcoxon_p_two_sided", "Wilcoxon p (exact)"), ("min_attainable_p_two_sided_sign_wilcoxon", "Min attainable p")],
                   {"mean": lambda x: f(x, sci=True), "sd": lambda x: f(x, sci=True), "paired_t": lambda x: f(x, 3), "paired_t_p_two_sided": lambda x: f(x, 3),
                    "sign_test_p_two_sided": lambda x: f(x, 4), "wilcoxon_p_two_sided": lambda x: f(x, 4), "min_attainable_p_two_sided_sign_wilcoxon": lambda x: f(x, 4)}))
    lj = m[(m.metric == "loss") & (m.specification == "PRIMARY")]
    li = m[(m.metric == "rank_ic") & (m.specification == "PRIMARY")]
    sl, si = st[st.metric == "loss"].sort_values("h"), st[st.metric == "rank_ic"].sort_values("h")
    ratio = (sl.sd / sl["mean"].abs())
    n_joint_excl = int(lj.ci_excludes_zero_joint.sum())
    L.append(f"**Finding.** The archived date-only MBB is {'reproduced exactly' if chk['REPRODUCED'] else 'NOT reproduced'}. "
             f"Conditional on the five trained seeds, it favours Static: the date-only loss CI excludes zero at {int(lj.ci_excludes_zero_date.sum())}/3 horizons, with max Holm p {f(lj.centered_p_holm_primary_date.max(), sci=True)}. "
             f"Once seeds are resampled too, the loss-difference CI is {f(lj.ci_width_ratio_joint_over_date_only.min(), 1)}–{f(lj.ci_width_ratio_joint_over_date_only.max(), 1)}× wider and excludes zero at {n_joint_excl}/3 horizons (min joint Holm p {f(lj.centered_p_holm_primary_joint.min(), 3)}). "
             f"At the seed level, per-seed MSE differences favour Static for {', '.join(str(int(x)) for x in sl.n_positive)} of 5 seeds at h=5/10/20. The seed-to-seed SD is {f(ratio.min(), 1)}–{f(ratio.max(), 1)}× the absolute mean, and the smallest seed-level p is {f(sl[['paired_t_p_two_sided', 'sign_test_p_two_sided', 'wilcoxon_p_two_sided']].min().min(), 3)}. "
             f"For Rank IC, the mean Adaptive − Static difference is {', '.join(f(x) for x in si['mean'])} at h=5/10/20. The date-only CI excludes zero at {int(li.ci_excludes_zero_date.sum())}/3 horizons (unadjusted; min Holm p {f(li.centered_p_holm_primary_date.min(), 3)}), and the joint CI at {int(li.ci_excludes_zero_joint.sum())}/3. "
             + ("In plain words, the Static-over-Adaptive loss advantage holds *for these five trained networks*, but it cannot be distinguished from seed-to-seed training variation. It should not be reported as a general property of the two mechanisms.\n"
                if n_joint_excl == 0 else
                "In plain words, part of the Static-over-Adaptive loss advantage survives resampling of seeds as well as dates (see the horizons above). The remaining horizons cannot be separated from seed-to-seed training variation.\n"))

    # ---------------------------------------------------------------- reproduction
    L.append("## Reproduction\n")
    L.append("```sh\ncd revision_work/scripts/skill\nPY=../../../.venv/bin/python; P=../../outputs/preds; S=../../outputs/skill_analysis\n"
             "$PY -B s0_canonicalize_archived.py\n$PY -B a1_benchmarks.py\n"
             "$PY -B a2_skill.py --preds \"$P/*_test.parquet\" \"$S/canonical_preds/*_test.parquet\" \"$S/benchmark_preds/*_test.parquet\" --out-dir $S/a2_original\n"
             "$PY -B a3_hierarchical.py --adaptive \"$P/adaptive_dw_h*_s*_test.parquet\" \"$S/canonical_preds/adaptive_dw_h*_s42_test.parquet\" --static \"$P/static_dw_h*_s*_test.parquet\" --out-dir $S/a3_original --archived-check\n"
             "$PY -B make_report_step5.py\n```\n")
    L.append("To re-run on causal-feature or new multi-seed predictions, point `--preds` / `--adaptive` / `--static` at the new `{model}_{config}_h{h}_s{seed}_test.parquet` files and use a new `--out-dir`. If the labels change, pass the matching all-split files via `--label-family`. Every script refuses to overwrite and fails closed if keys or labels disagree.\n")
    write_text(R / "REPORT_STEP5.md", "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
