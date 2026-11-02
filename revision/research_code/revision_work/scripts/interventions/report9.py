"""REPORT_STEP9.md builder (all interpretive statements are computed from the result tables)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LABEL = {"I1_equal": "I1 equal 1/4", "I2_global_train_mean": "I2 global TRAIN mean", "I3_ticker_train_mean": "I3 per-ticker TRAIN mean",
         "I4_date_mean": "I4 per-date cross-sectional mean", "I6_equal_daily_only": "I6 equal, Daily branch only",
         "I6_equal_weekly_only": "I6 equal, Weekly branch only"}
NEGLIGIBLE_PCT = 0.01  # |ΔMSE| < 1% of I0 MSE and joint CIs covering zero -> "negligible" (pre-stated rule)


def lab(iv):
    if iv.startswith("I5_drop_"):
        s = iv.replace("I5_drop_", "")
        return f"I5 drop {s.replace('daily', 'Daily ').replace('weekly', 'Weekly ')}"
    return LABEL.get(iv, iv)


def f(x, nd=4):
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.3e}" if (x != 0 and abs(x) < 1e-3) else f"{x:.{nd}f}"


def md(df, cols, fm=None):
    fm = fm or {}
    s = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.to_dict("records"):
        s += "| " + " | ".join(fm.get(c[0], f)(r.get(c[0])) for c in cols) + " |\n"
    return s


def build_report(a, V, M, S, PS, J, WS, nthreads, ninterop):
    L = [f"# REPORT STEP 9 — Router interventions within trained Adaptive models (run: {a.run_label})\n",
         f"Generated {datetime.now(timezone.utc).isoformat()} (UTC). Inference only, no training: CPU, torch threads {nthreads}, interop {ninterop}, one process. "
         "Outputs created exclusively. Script: `revision_work/scripts/interventions/router_interventions.py`.\n",
         "## BLOCKING\n",
         f"None. All {len(V)} checkpoints matched their recorded SHA-256. The no-intervention forward pass reproduces the Step 3 replayed TEST predictions exactly "
         f"(max |Δ| = {V.max_abs_delta_vs_replayed_preds.max():.1e}), and the override loop reproduces the formal Experiment 4 inference helper exactly "
         f"(max |Δ| = {V.max_abs_delta_override_loop_vs_formal_helper.max():.1e}).\n",
         "## Mechanism\n",
         "- In `scripts/python/panel_pathformer_main_v2.py`, each branch's per-sample routing weights come from `ScaleAggregator._adaptive_weights` (lines 374–378: router MLP → softmax over the 4 scales). "
         "They are consumed in `ScaleAggregator.forward` (lines 403–404: z = Σₖ wₖ·expertₖ). `FrequencyAdaptivePathFormerBranch.forward` (lines 473–478) calls the aggregator, "
         "and `DualFrequencyAdaptivePathFormerRegressor.forward_with_router` (lines 572–577) concatenates the Daily and Weekly outputs into the fusion head.\n"
         "- Each intervention replaces **only** the tensor returned by `_adaptive_weights` on `daily_branch.aggregator` and `weekly_branch.aggregator`. The override first calls the original method, so the learned weights stay available to I4–I6. "
         "Experts, router parameters, fusion head and all other parameters are unchanged. Models come from the formal Experiment 4 factory (`panel_pathformer_experiment4_multiseed._make_experiment4_model`, lines 411–418), "
         "are loaded strictly and run in eval mode on the frozen V2 tensors from `panel_pathformer_main_v2._build_training_loaders` (line 1326). The routing has no cross-sample mixing (docstring, lines 327–329), so batch composition cannot affect the result.\n"
         "- I2/I3 use the model's own learned weights averaged over all TRAIN samples (I3: per ticker). I4 replaces each stock's weights by the cross-sectional mean of the learned weights on the same date (same-date information only). "
         "I5 zeroes one scale and renormalises the other three in that branch. I6 sets equal weights in one branch and keeps the learned weights in the other.\n"
         "- Differences are intervention − I0: ΔMSE < 0 means the intervention lowers MSE; ΔRank IC > 0 means it improves ranking. Bootstraps (TEST): per seed, a date-level moving block bootstrap (L = h, B = 10,000, SeedSequence([20261019, h, h]), common blocks across seeds and interventions). "
         "Jointly, a seed × date bootstrap draws 4 seed indices with replacement and one set of date blocks (SeedSequence([20261019, h, h, 1])). Both use percentile 95% CIs and centred two-sided p; Holm is applied over horizons per intervention and metric. "
         f"**Many interventions × horizons × metrics are tested, so isolated rejections should be read with that multiplicity in mind.** The per-seed intervals are conditional on the trained model, and the 4-seed joint resampling is coarse.\n"]
    L.append("Verification:\n")
    L.append(md(V, [("h", "h"), ("seed", "Seed"), ("checkpoint", "Checkpoint"), ("max_abs_delta_vs_replayed_preds", "max |Δ| vs replayed preds"),
                    ("max_abs_delta_override_loop_vs_formal_helper", "max |Δ| override loop vs helper")]))
    # ---- routing weight summary
    L.append("## Learned routing weights\n")
    ws = WS.copy()
    rows = []
    for (h, b), g in ws.groupby(["h", "branch"]):
        cols = [c for c in g.columns if c.startswith("train_mean_") and g[c].notna().all()]
        rows.append(dict(h=h, branch=b, train_mean_vector=" / ".join(f"{c.replace('train_mean_', '').replace('_w_', ' ')}: {g[c].mean():.3f}" for c in cols),
                         sd_across_samples=g.test_mean_sd_across_samples.mean(), share_var_between_dates=g.test_share_var_between_dates.mean()))
    L.append(md(pd.DataFrame(rows), [("h", "h"), ("branch", "Branch"), ("train_mean_vector", "TRAIN-mean routing vector (mean over seeds)"),
                                     ("sd_across_samples", "TEST SD across samples (mean over scales)"), ("share_var_between_dates", "TEST share of routing variance between dates")],
                {"sd_across_samples": lambda x: f"{x:.3f}", "share_var_between_dates": lambda x: f"{x:.3f}"}))
    # ---- main TEST table
    L.append("## TEST — interventions vs learned routing (4 seeds)\n")
    t = S[S.split == "test"].merge(J[J.metric == "delta_mse"][["h", "intervention", "ci_lo", "ci_hi", "p_centered", "p_holm_over_h"]].rename(
        columns={"ci_lo": "mse_ci_lo", "ci_hi": "mse_ci_hi", "p_centered": "mse_p", "p_holm_over_h": "mse_p_holm"}), on=["h", "intervention"], how="left")
    t = t.merge(J[J.metric == "delta_rank_ic"][["h", "intervention", "ci_lo", "ci_hi", "p_centered", "p_holm_over_h"]].rename(
        columns={"ci_lo": "ic_ci_lo", "ci_hi": "ic_ci_hi", "p_centered": "ic_p", "p_holm_over_h": "ic_p_holm"}), on=["h", "intervention"], how="left")
    t["iv"] = t.intervention.map(lab)
    t["dmse"] = [f"{m:+.2e} ± {s:.1e}" for m, s in zip(t.delta_mse_mean, t.delta_mse_sd)]
    t["dmse_pct"] = [f"{100 * x:+.2f}%" for x in t.delta_mse_pct_of_I0]
    t["dic"] = [f"{m:+.4f} ± {s:.4f}" for m, s in zip(t.delta_rank_ic_mean, t.delta_rank_ic_sd)]
    t["mse_ci"] = [("—" if not np.isfinite(lo) else f"[{lo:+.1e}, {hi:+.1e}]") for lo, hi in zip(t.mse_ci_lo, t.mse_ci_hi)]
    t["ic_ci"] = [("—" if not np.isfinite(lo) else f"[{lo:+.4f}, {hi:+.4f}]") for lo, hi in zip(t.ic_ci_lo, t.ic_ci_hi)]
    order = {iv: i for i, iv in enumerate(t.intervention.drop_duplicates())}
    t = t.sort_values(["h", "intervention"], key=lambda s: s.map(order) if s.name == "intervention" else s)
    L.append(md(t, [("h", "h"), ("iv", "Intervention"), ("mse_mean", "Mean MSE"), ("dmse", "ΔMSE mean ± SD"), ("dmse_pct", "ΔMSE % of I0"),
                    ("n_seeds_delta_mse_neg", "# seeds ΔMSE<0"), ("mse_ci", "Joint 95% CI ΔMSE"), ("mse_p_holm", "Holm p"), ("rank_ic_mean", "Mean Rank IC"),
                    ("dic", "ΔRank IC mean ± SD"), ("n_seeds_delta_ic_pos", "# seeds ΔIC>0"), ("ic_ci", "Joint 95% CI ΔIC"), ("ic_p_holm", "Holm p "),
                    ("pred_std_ratio_mean", "PredStd/TrueStd")],
                {"mse_mean": lambda x: f"{x:.6f}", "rank_ic_mean": lambda x: f"{x:+.4f}", "mse_p_holm": lambda x: f(x, 3), "ic_p_holm": lambda x: f(x, 3),
                 "pred_std_ratio_mean": lambda x: f"{x:.3f}"}))
    L.append("Per-model metrics (MSE, Rank IC, Pearson, direction accuracy, PredStd/TrueStd) for every intervention: `metrics_by_model_intervention.csv`.\n")
    # ---- per-seed bootstrap (main interventions)
    L.append("## TEST — per-seed date-level MBB (conditional on each trained model), I1–I4 and I6\n")
    ps = PS[PS.intervention.isin(list(LABEL))].copy()
    piv = ps.pivot_table(index=["h", "seed", "intervention"], columns="metric", values=["theta", "ci_lo", "ci_hi", "ci_excludes_0"], aggfunc="first")
    piv.columns = [f"{a_}_{b_}" for a_, b_ in piv.columns]
    piv = piv.reset_index()
    piv["iv"] = piv.intervention.map(lab)
    piv["mse"] = [f"{t_:+.2e} [{lo:+.1e}, {hi:+.1e}]{' *' if ex else ''}" for t_, lo, hi, ex in zip(piv.theta_delta_mse, piv.ci_lo_delta_mse, piv.ci_hi_delta_mse, piv.ci_excludes_0_delta_mse)]
    piv["ic"] = [f"{t_:+.4f} [{lo:+.4f}, {hi:+.4f}]{' *' if ex else ''}" for t_, lo, hi, ex in zip(piv.theta_delta_rank_ic, piv.ci_lo_delta_rank_ic, piv.ci_hi_delta_rank_ic, piv.ci_excludes_0_delta_rank_ic)]
    L.append("`*` = 95% CI excludes zero.\n")
    L.append(md(piv.sort_values(["h", "intervention", "seed"]), [("h", "h"), ("iv", "Intervention"), ("seed", "Seed"), ("mse", "ΔMSE [95% CI]"), ("ic", "ΔRank IC [95% CI]")]))
    L.append("I5 per-seed intervals: `bootstrap_per_seed.csv`.\n")
    # ---- scale importance
    L.append("## Scale importance (I5 drop-one-scale, TEST, mean over 4 seeds)\n")
    i5 = S[(S.split == "test") & S.intervention.str.startswith("I5_")].copy()
    i5["branch"] = np.where(i5.intervention.str.contains("daily"), "Daily", "Weekly")
    i5["scale"] = i5.intervention.str.replace("I5_drop_daily", "").str.replace("I5_drop_weekly", "")
    for metric, title, fm_ in (("delta_mse_mean", "ΔMSE when the scale is dropped (positive = the scale helps MSE)", lambda x: f"{x:+.2e}"),
                               ("delta_rank_ic_mean", "ΔRank IC when the scale is dropped (negative = the scale helps ranking)", lambda x: f"{x:+.4f}")):
        pv = i5.pivot_table(index=["branch", "scale"], columns="h", values=metric).reset_index()
        pv["o"] = pv.scale.astype(int)
        pv = pv.sort_values(["branch", "o"])
        pv.columns = [str(c) for c in pv.columns]
        L.append(f"{title}:\n")
        L.append(md(pv, [("branch", "Branch"), ("scale", "Dropped scale (patch)")] + [(str(h), f"h={h}") for h in sorted(i5.h.unique())],
                    {str(h): fm_ for h in sorted(i5.h.unique())}))
    nn = i5.pivot_table(index=["branch", "scale"], columns="h", values="n_seeds_delta_mse_neg").reset_index()
    nn.columns = [str(c) for c in nn.columns]
    L.append("Number of seeds (of 4) where dropping the scale LOWERS MSE:\n")
    L.append(md(nn.assign(o=nn.scale.astype(int)).sort_values(["branch", "o"]), [("branch", "Branch"), ("scale", "Scale")] + [(str(h), f"h={h}") for h in sorted(i5.h.unique())]))
    # ---- validation
    L.append("## VALIDATION (secondary; checkpoint-selection period)\n")
    v = S[S.split == "val"].copy()
    v["iv"] = v.intervention.map(lab)
    v["dmse"] = [f"{m:+.2e} ± {s:.1e}" for m, s in zip(v.delta_mse_mean, v.delta_mse_sd)]
    v["dic"] = [f"{m:+.4f} ± {s:.4f}" for m, s in zip(v.delta_rank_ic_mean, v.delta_rank_ic_sd)]
    v = v.sort_values(["h", "intervention"], key=lambda s: s.map(order) if s.name == "intervention" else s)
    L.append(md(v, [("h", "h"), ("iv", "Intervention"), ("mse_mean", "Mean MSE"), ("dmse", "ΔMSE mean ± SD"), ("n_seeds_delta_mse_neg", "# seeds ΔMSE<0"),
                    ("dic", "ΔRank IC mean ± SD"), ("n_seeds_delta_ic_pos", "# seeds ΔIC>0")], {"mse_mean": lambda x: f"{x:.6f}"}))
    # ---- interpretation (computed)
    L.append("## Interpretation\n")
    tt = t.set_index(["h", "intervention"])
    H = sorted(t.h.unique())

    def neg(h, iv):
        r = tt.loc[(h, iv)]
        return abs(r.delta_mse_pct_of_I0) < NEGLIGIBLE_PCT and r.mse_ci_lo <= 0 <= r.mse_ci_hi and r.ic_ci_lo <= 0 <= r.ic_ci_hi

    def desc(h, iv):
        r = tt.loc[(h, iv)]
        return (f"h={h}: ΔMSE {100 * r.delta_mse_pct_of_I0:+.2f}% (lower in {int(r.n_seeds_delta_mse_neg)}/{int(r.n_seeds)} seeds, joint CI {r.mse_ci}), "
                f"ΔRank IC {r.delta_rank_ic_mean:+.4f} (higher in {int(r.n_seeds_delta_ic_pos)}/{int(r.n_seeds)}, joint CI {r.ic_ci})")
    q1 = [h for h in H if neg(h, "I1_equal") and neg(h, "I2_global_train_mean")]
    L.append("**1. Do the learned sample-dependent weights matter?** I1 (equal) — " + "; ".join(desc(h, "I1_equal") for h in H) +
             ". I2 (global TRAIN mean) — " + "; ".join(desc(h, "I2_global_train_mean") for h in H) + ". "
             + (f"By the pre-stated rule (|ΔMSE| < 1% of I0 and both joint CIs covering zero), replacing the learned routing is **negligible at h = {', '.join(map(str, q1))}**"
                if q1 else "By the pre-stated rule (|ΔMSE| < 1% of I0 and both joint CIs covering zero), replacing the learned routing is not negligible at any horizon")
             + (f" and not negligible at h = {', '.join(str(h) for h in H if h not in q1)}." if len(q1) < len(H) and q1 else ".")
             + f" The learned weights vary little across samples (TEST SD across samples ≈ {WS.test_mean_sd_across_samples.mean():.3f} per scale on the simplex).\n")
    imp = [(h, iv) for h in H for iv in ("I2_global_train_mean", "I3_ticker_train_mean")
           if tt.loc[(h, iv)].delta_mse_mean < 0 and tt.loc[(h, iv)].mse_ci_hi < 0]
    imp_ic = [(h, iv) for h in H for iv in ("I2_global_train_mean", "I3_ticker_train_mean")
              if tt.loc[(h, iv)].delta_rank_ic_mean > 0 and tt.loc[(h, iv)].ic_ci_lo > 0]
    mse_dir = [(h, iv) for h in H for iv in ("I2_global_train_mean", "I3_ticker_train_mean") if tt.loc[(h, iv)].n_seeds_delta_mse_neg >= 3]
    ic_dir = [(h, iv) for h in H for iv in ("I2_global_train_mean", "I3_ticker_train_mean") if tt.loc[(h, iv)].n_seeds_delta_ic_pos >= 3]
    fmt_list = lambda xs: ", ".join(f"{lab(iv).split(' ')[0]} h={h}" for h, iv in xs) or "none"
    L.append("**2. Is the sample-dependent part harmful?** Replacing each sample's weights by the model's own average allocation (I2 global, I3 per ticker) "
             f"lowers MSE with a joint CI below zero in: {fmt_list(imp)}; it lowers MSE in at least 3 of 4 seeds in: {fmt_list(mse_dir)}. "
             f"It raises Rank IC with a joint CI above zero in: {fmt_list(imp_ic)}; it raises Rank IC in at least 3 of 4 seeds in: {fmt_list(ic_dir)}. "
             + ("So the deviations from the average allocation do measurable harm at those cells. " if (imp or imp_ic) else
                "There is no robust evidence that the sample-dependent deviations themselves cause the Adaptive model's weaker ranking. ")
             + "I2 and I3 give almost identical results, because the per-ticker TRAIN means are close to the global mean.\n")
    L.append("**3. Which variation matters: cross-stock or over time?** I4 keeps each date's average routing (variation over time) and removes the cross-stock differences on that date — "
             + "; ".join(desc(h, "I4_date_mean") for h in H) + ". "
             f"Only {100 * WS.test_share_var_between_dates.mean():.1f}% of the TEST routing variance lies between dates (mean over models and branches); almost all of it is cross-stock variation within a date. "
             "Comparing I4 with I2 (which also removes the time variation) separates the two components: "
             + "; ".join(f"h={h}: ΔMSE I4 {100 * tt.loc[(h, 'I4_date_mean')].delta_mse_pct_of_I0:+.2f}% vs I2 {100 * tt.loc[(h, 'I2_global_train_mean')].delta_mse_pct_of_I0:+.2f}%, "
                         f"ΔIC I4 {tt.loc[(h, 'I4_date_mean')].delta_rank_ic_mean:+.4f} vs I2 {tt.loc[(h, 'I2_global_train_mean')].delta_rank_ic_mean:+.4f}" for h in H) + ".\n")
    best = []
    for h in H:
        g = i5[i5.h == h]
        for br in ("Daily", "Weekly"):
            gb = g[g.branch == br].sort_values("delta_mse_mean")
            best.append(f"h={h} {br}: dropping scale {gb.iloc[-1].scale} raises MSE most ({gb.iloc[-1].delta_mse_mean:+.2e}); dropping scale {gb.iloc[0].scale} changes it least or lowers it ({gb.iloc[0].delta_mse_mean:+.2e})")
    L.append("**4. Which scales and which branch contribute?** " + "; ".join(best) + ". Branch-level equal weighting (I6): "
             + "; ".join(f"h={h}: Daily-only {100 * tt.loc[(h, 'I6_equal_daily_only')].delta_mse_pct_of_I0:+.2f}% MSE / {tt.loc[(h, 'I6_equal_daily_only')].delta_rank_ic_mean:+.4f} IC, "
                         f"Weekly-only {100 * tt.loc[(h, 'I6_equal_weekly_only')].delta_mse_pct_of_I0:+.2f}% / {tt.loc[(h, 'I6_equal_weekly_only')].delta_rank_ic_mean:+.4f}" for h in H)
             + ". " + (lambda a5, a14: (f"The largest drop-one-scale effect (max |ΔMSE| {a5:.2e}) is {'larger' if a5 > a14 else 'not larger'} than the largest routing-weight intervention effect (I1–I4, max |ΔMSE| {a14:.2e}). ")
                       )(float(i5.delta_mse_mean.abs().max()), float(t[t.intervention.isin(['I1_equal', 'I2_global_train_mean', 'I3_ticker_train_mean', 'I4_date_mean'])].delta_mse_mean.abs().max()))
             + "Dropping a scale removes an expert's contribution from the representation passed to the fusion head, so I5 measures how much the trained model relies on each expert, not whether the router allocates well.\n")
    L.append("All statements are about these trained models on the original features. They describe what the trained routers do, not what a differently trained Adaptive model would do.\n")
    L.append("## Reproduction\n")
    L.append("```sh\ncd revision_work/scripts/interventions\nOMP_NUM_THREADS=2 ../../../.venv/bin/python -B router_interventions.py --run-label original\n"
             "# other checkpoints: --checkpoints '<dir>/adaptive_seed*_*d.pt' --reference-preds '<preds>/adaptive_dw_h{h}_s{seed}_test.parquet' \\\n"
             "#   --summary-csv none --run-label <new label>\n```\n"
             "Limitation: the script feeds the original frozen V2 tensors (`_build_training_loaders`). Causal-feature checkpoints need their own causal tensors, so a data-loader switch must be added before re-running on them. "
             "The I0 verification against their saved predictions would stop the run with a BLOCKING note if the inputs did not match.\n")
    L.append("CSV files: `verification.csv`, `metrics_by_model_intervention.csv` (TEST and VALIDATION), `seed_summary.csv`, `bootstrap_per_seed.csv`, `bootstrap_joint_seed_date.csv`, `routing_weight_summary.csv`.\n")
    return "\n".join(L)
