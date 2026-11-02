"""REPORT_STEP8.md builder (all statements computed from the engine's tables)."""
import numpy as np
import pandas as pd

from engine import GAMMAS, GAMMA0, COSTS_BP, COST0, WMAX, LOOKBACK, HORIZONS, utc

BASE_ORDER = ["static_ens", "adaptive_ens", "static (seed mean)", "adaptive (seed mean)", "single_s42", "fixed_s42", "ridge_s42", "lstm_s42",
              "vanilla_s42", "swim_s42", "sig_mom12_1", "sig_mom6_1", "sig_rev1w", "sig_rev1m", "gtm", "EW", "MINVAR", "RISKPARITY"]


def f(x, nd=3):
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def p(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"


def md(df, cols, fm=None):
    fm = fm or {}
    s = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.to_dict("records"):
        s += "| " + " | ".join(fm.get(c[0], f)(r.get(c[0])) for c in cols) + " |\n"
    return s


def with_seed_means(M, SEEDSTAT, split, layer, cost):
    rows = M[(M.split == split) & (M.cost_bp == cost) & ((M.layer == layer) | (M.layer == "allocation benchmark"))].copy()
    rows = rows[~rows.strategy.str.match(r"^(static|adaptive)_s\d+$")]
    add = []
    for mech in ("static", "adaptive"):
        g = M[(M.split == split) & (M.cost_bp == cost) & (M.layer == layer) & M.strategy.str.match(rf"^{mech}_s\d+$")]
        for h, gg in g.groupby("h"):
            r = gg.mean(numeric_only=True).to_dict()
            r.update(split=split, h=h, strategy=f"{mech} (seed mean)", layer=layer, n_seeds=len(gg))
            add.append(r)
    out = pd.concat([rows, pd.DataFrame(add)], ignore_index=True)
    out["order"] = out.strategy.map({k: i for i, k in enumerate(BASE_ORDER)}).fillna(99)
    return out.sort_values(["h", "order"])


def build_report(a, M, SVA, GRS, SEEDSTAT, LS, QP, notes, meta, anchors, gtm, n_ffill, n_rf_missing, blocking):
    L = [f"# REPORT STEP 8 — Portfolio allocation and transaction-cost evaluation (run: {a.run_label})\n",
         f"Generated {utc()} (UTC). No training. Inputs were read only, and every output in this run directory was created exclusively. Engine: `revision_work/scripts/portfolio/engine.py`.\n",
         "## BLOCKING\n", ("\n".join(f"- {b}" for b in blocking) if blocking else
                          "None. All prediction sets share the same anchor dates and 17 tickers per split and horizon. Every anchor is on the common price calendar, and Global Train-Mean Layer 2 reproduces minimum variance (see design notes).") + "\n"]
    # ---- design
    L.append("## Design summary\n")
    L.append(
        "- **Timing.** A forecast made at the close of anchor date *t* is executed at the close of *t*+1 and held for *h* trading days, until the close of *t*+1+*h*. "
        "**The holding interval is shifted one day later than the label interval (close *t* → close *t*+*h*)**, so realised returns are not the label returns.\n"
        "- **Rebalancing.** Non-overlapping anchors: the first anchor of the split, then every *h*-th anchor. Holding periods tile without gaps, and overlapping *h*-day returns are never compounded. "
        "Between rebalances, weights drift with realised daily simple returns from adjusted closes (`dataset/finance/panel_raw`), and value compounds daily.\n"
        f"- **Risk model.** Ledoit–Wolf shrinkage covariance on the {LOOKBACK} daily returns up to and including *t*, scaled ×*h*.\n"
        f"- **Layer 1.** Long-only equal weight in the top *k* ∈ {{3, 5}} by score. The top-3 minus bottom-3 spread is a gross signal diagnostic only.\n"
        f"- **Layer 2.** Long-only mean–variance: max *w*′μ − (γ/2)*w*′Σ*w*, Σ*w* = 1, 0 ≤ *w*ᵢ ≤ {WMAX}, μᵢ = exp(ŷᵢ) − 1. Primary γ = {GAMMA0}; sensitivity γ ∈ {GAMMAS}. "
        f"Solved with SciPy SLSQP (cvxpy is not installed) as a warm start, then made exact by a primal active-set solve of the KKT system (accepted only if the KKT violation is < 1e-8); a failed solve falls back to the previous weights and is logged ({len(QP)} fallbacks, `qp_solver_log.csv`).\n"
        "- **Allocation benchmarks.** 1/N over the 17 stocks (EW), and minimum variance (same Σ and constraints).\n"
        "- **Signal baselines.** MOM(12-1), MOM(6-1), REV(1w), REV(1m), computed from prices up to *t* (Step 5 A1 code). Layer 1 ranks on the raw signal. Layer 2 uses the TRAIN-only OLS calibration from Step 5 A1 (ŷ = a + b·signal). "
        f"Global Train-Mean (ŷ = {', '.join(f'h={h}: {gtm[h]:.5f}' for h in HORIZONS)}) has constant μ, so its Layer 2 equals minimum variance (check below), and it has no Layer 1.\n"
        f"- **Costs.** Turnover = Σ|*w*_target − *w*_drifted| at each rebalance; the initial build counts as turnover 1. Proportional cost *c* ∈ {COSTS_BP} bp per unit of one-way traded value is deducted at execution. "
        "Turnover tables report the average **excluding** the initial build.\n"
        f"- **Risk-free rate.** DTB3 daily simple rate, forward-filled at most 5 days ({n_ffill} calendar days filled; {n_rf_missing} still missing).\n"
        "- **Metrics.** Annualised geometric return, volatility, Sharpe ratio (excess over DTB3), maximum drawdown, turnover, hit rate of rebalance periods (net return > 0), certainty equivalent (mean − (A/2)·var, daily, ×252) for A = 2 and 5, "
        "and alpha vs 1/N (daily OLS of excess returns on 1/N excess returns at the same cost level, Newey–West lag 5).\n"
        "- **Survivorship caveat.** The universe is the 17 stocks that survive to today (see Step 6). Absolute returns of every strategy, including 1/N, are likely biased upward. Comparisons between strategies are within the same universe, so they are less affected, but they do not generalise to a point-in-time sector universe.\n"
        "- **VALIDATION** results are shown for completeness. **The validation period was used to select checkpoints (early stopping)**, so it is not an out-of-sample period for the models. Adaptive seed 42 has no validation predictions, and Ridge/LSTM/Vanilla/SWiM are TEST-only.\n"
        "- **Static vs Adaptive bootstrap.** Circular block bootstrap of the paired daily returns (block = *h* days, B = 10,000, SeedSequence([20261018, h, h, crc32(tag)])), percentile 95% CIs. "
        "**It resamples dates only and is conditional on the trained seeds** (seed uncertainty is shown separately as across-seed SD).\n")
    L.append("Design checks and notes:\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    L.append(f"Periods: " + "; ".join(f"{sp} h={h}: anchors {ad[0]} … {ad[-1]}, {len(ad[0::h])} rebalances" for (h, sp), ad in sorted(anchors.items())) + ".\n")
    # ---- main tables
    cols = [("h", "h"), ("strategy", "Strategy"), ("ann_geo_return", "Ann. return"), ("ann_vol", "Ann. vol"), ("sharpe", "Sharpe"),
            ("max_drawdown", "Max DD"), ("avg_turnover_ex_initial", "Avg TO / rebal."), ("ann_turnover", "Ann. TO"), ("hit_rate_periods", "Hit rate"),
            ("ce_A2", "CE (A=2)"), ("ce_A5", "CE (A=5)"), ("alpha_vs_ew_ann", "α vs 1/N (ann.)"), ("alpha_vs_ew_nw_t", "α NW t")]
    fm = {"ann_geo_return": p, "ann_vol": p, "max_drawdown": p, "ce_A2": p, "ce_A5": p, "alpha_vs_ew_ann": p, "sharpe": lambda x: f(x, 2),
          "avg_turnover_ex_initial": lambda x: f(x, 2), "ann_turnover": lambda x: f(x, 1), "hit_rate_periods": lambda x: f(x, 2), "alpha_vs_ew_nw_t": lambda x: f(x, 2)}
    L.append(f"## TEST — Layer 2 (mean–variance, γ = {GAMMA0}), {COST0} bp\n")
    L.append("Seed-mean rows average each metric over the per-seed strategies (Static 5 seeds, Adaptive 5 seeds incl. formal seed 42). Ensembles average predictions before optimising.\n")
    L.append(md(with_seed_means(M, SEEDSTAT, "test", f"L2_g{GAMMA0}", COST0), cols, fm))
    for k in (3, 5):
        L.append(f"## TEST — Layer 1 (top-{k} equal weight), {COST0} bp\n")
        L.append(md(with_seed_means(M, SEEDSTAT, "test", f"L1_top{k}", COST0), cols, fm))
    # ---- cost grid
    L.append("## TEST — full cost grid (Sharpe ratio; annualised return in `metrics_all.csv`)\n")
    key = M[(M.split == "test") & (M.strategy.isin(["static_ens", "adaptive_ens", "EW", "MINVAR", "ridge_s42", "sig_rev1m"])) &
            (M.layer.isin([f"L2_g{GAMMA0}", "L1_top5", "allocation benchmark"]))].copy()
    key["row"] = key.strategy + " · " + key.layer
    pv = key.pivot_table(index=["h", "row"], columns="cost_bp", values="sharpe").reset_index()
    pv.columns = [str(c) for c in pv.columns]
    L.append(md(pv, [("h", "h"), ("row", "Strategy · layer")] + [(str(c), f"{c} bp") for c in COSTS_BP], {str(c): (lambda x: f(x, 2)) for c in COSTS_BP}))
    pv2 = key.pivot_table(index=["h", "row"], columns="cost_bp", values="ann_geo_return").reset_index()
    pv2.columns = [str(c) for c in pv2.columns]
    L.append("Annualised geometric return by cost level:\n")
    L.append(md(pv2, [("h", "h"), ("row", "Strategy · layer")] + [(str(c), f"{c} bp") for c in COSTS_BP], {str(c): p for c in COSTS_BP}))
    # ---- gamma sensitivity
    L.append(f"## TEST — γ sensitivity (Layer 2, {COST0} bp, Sharpe)\n")
    gs = M[(M.split == "test") & (M.cost_bp == COST0) & M.layer.str.startswith("L2_g") &
           M.strategy.isin(["static_ens", "adaptive_ens", "ridge_s42", "sig_rev1m", "sig_mom6_1", "gtm"])].copy()
    gs["gamma"] = gs.layer.str.replace("L2_g", "").astype(int)
    pg = gs.pivot_table(index=["h", "strategy"], columns="gamma", values="sharpe").reset_index()
    pg.columns = [str(c) for c in pg.columns]
    L.append(md(pg, [("h", "h"), ("strategy", "Strategy")] + [(str(g), f"γ={g}") for g in GAMMAS], {str(g): (lambda x: f(x, 2)) for g in GAMMAS}))
    # ---- static vs adaptive
    L.append(f"## Static vs Adaptive (TEST, {COST0} bp)\n")
    L.append("Differences are Adaptive − Static. Bootstrap CIs (dates only, conditional on the trained seeds) are for the annualised mean daily-return difference and for the Sharpe difference.\n")
    sv = SVA[SVA.split == "test"].copy()
    L.append(md(sv, [("h", "h"), ("layer", "Layer"), ("pair", "Pair"), ("sharpe_static", "Sharpe S"), ("sharpe_adaptive", "Sharpe A"), ("sharpe_diff", "Δ Sharpe"),
                     ("sharpe_diff_ci_lo", "Δ Sharpe CI lo"), ("sharpe_diff_ci_hi", "Δ Sharpe CI hi"), ("ann_geo_ret_diff", "Δ ann. return"),
                     ("mean_ret_diff_ann", "Δ mean ret (ann.)"), ("mean_ret_diff_ann_ci_lo", "CI lo"), ("mean_ret_diff_ann_ci_hi", "CI hi"),
                     ("mean_ret_diff_ann_ci_excludes_0", "Excl. 0"), ("n_seed_pairs_adaptive_higher_sharpe", "# seeds A>S")],
                {"sharpe_static": lambda x: f(x, 2), "sharpe_adaptive": lambda x: f(x, 2), "sharpe_diff": lambda x: f(x, 2), "sharpe_diff_ci_lo": lambda x: f(x, 2),
                 "sharpe_diff_ci_hi": lambda x: f(x, 2), "ann_geo_ret_diff": p, "mean_ret_diff_ann": p, "mean_ret_diff_ann_ci_lo": p, "mean_ret_diff_ann_ci_hi": p,
                 "n_seed_pairs_adaptive_higher_sharpe": lambda x: "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else str(int(x))}))
    L.append("Across-seed mean ± SD (TEST, 10 bp):\n")
    sd = SEEDSTAT[SEEDSTAT.split == "test"].copy()
    sd["Sharpe"] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(sd.sharpe_mean, sd.sharpe_std)]
    sd["Ann. return"] = [f"{100 * m:+.1f}% ± {100 * s:.1f}%" for m, s in zip(sd.ann_geo_return_mean, sd.ann_geo_return_std)]
    sd["Max DD"] = [f"{100 * m:.1f}% ± {100 * s:.1f}%" for m, s in zip(sd.max_drawdown_mean, sd.max_drawdown_std)]
    sd["Avg TO"] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(sd.avg_turnover_ex_initial_mean, sd.avg_turnover_ex_initial_std)]
    sd["α vs 1/N"] = [f"{100 * m:+.1f}% ± {100 * s:.1f}%" for m, s in zip(sd.alpha_vs_ew_ann_mean, sd.alpha_vs_ew_ann_std)]
    L.append(md(sd.sort_values(["h", "layer", "mechanism"]), [("h", "h"), ("layer", "Layer"), ("mechanism", "Mechanism"), ("sharpe_count", "n seeds"),
                                                              ("Sharpe", "Sharpe"), ("Ann. return", "Ann. return"), ("Max DD", "Max DD"), ("Avg TO", "Avg TO"), ("α vs 1/N", "α vs 1/N")]))
    L.append(f"## GRS test: zero alpha vs 1/N (Layer 2, γ = {GAMMA0}, {COST0} bp, daily)\n")
    L.append(md(GRS, [("split", "Split"), ("h", "h"), ("assets", "Assets"), ("N", "N"), ("T_days", "T"), ("GRS_F", "GRS F"), ("p_value", "p")],
                {"GRS_F": lambda x: f(x, 3), "p_value": lambda x: f(x, 4)}))
    L.append("## Top-3 minus bottom-3 spread (signal diagnostic, gross, TEST)\n")
    ls = LS[LS.split == "test"].copy()
    L.append(md(ls.sort_values(["h", "strategy"]), [("h", "h"), ("strategy", "Score"), ("n_periods", "Periods"), ("mean_period_spread", "Mean spread / period"),
                                                     ("ann_spread", "Annualised (×252/h)"), ("t_stat", "t"), ("hit_rate", "Hit rate")],
                {"mean_period_spread": p, "ann_spread": p, "t_stat": lambda x: f(x, 2), "hit_rate": lambda x: f(x, 2)}))
    L.append(f"## VALIDATION (checkpoint-selection period; not out-of-sample for the models) — Layer 2 γ = {GAMMA0} and Layer 1 top-5, {COST0} bp\n")
    vv = pd.concat([with_seed_means(M, SEEDSTAT, "val", f"L2_g{GAMMA0}", COST0).assign(lay=f"L2 γ={GAMMA0}"),
                    with_seed_means(M, SEEDSTAT, "val", "L1_top5", COST0).assign(lay="L1 top-5")])
    vv = vv[~((vv.lay == "L1 top-5") & vv.strategy.isin(["EW", "MINVAR"]))]
    L.append(md(vv.sort_values(["h", "lay", "order"]), [("h", "h"), ("lay", "Layer"), ("strategy", "Strategy"), ("ann_geo_return", "Ann. return"), ("ann_vol", "Ann. vol"),
                                                         ("sharpe", "Sharpe"), ("max_drawdown", "Max DD"), ("avg_turnover_ex_initial", "Avg TO"), ("alpha_vs_ew_ann", "α vs 1/N")], fm))
    L.append("## Cumulative returns\n")
    L.append("![TEST cumulative net returns](fig_cumulative_test_net10bp.png)\n")
    L.append("TEST period, net of 10 bp. Data behind the figure: `fig_cumulative_test_net10bp_data.csv`.\n")
    # ---- summary (computed)
    t = M[(M.split == "test") & (M.cost_bp == COST0)]
    ew = t[t.strategy == "EW"].set_index("h").sharpe
    mv = t[t.strategy == "MINVAR"].set_index("h").sharpe
    sent = []
    for h in HORIZONS:
        th = t[t.h == h]
        l2 = th[th.layer == f"L2_g{GAMMA0}"]
        beat_ew = l2[(l2.sharpe > ew[h]) & ~l2.strategy.str.match(r"^(static|adaptive)_s\d+$")].strategy.tolist()
        se, ae = l2.set_index("strategy").sharpe.get("static_ens", np.nan), l2.set_index("strategy").sharpe.get("adaptive_ens", np.nan)
        sent.append(f"h={h}: 1/N Sharpe {ew[h]:.2f}, minimum variance {mv[h]:.2f}; Static ensemble {se:.2f}, Adaptive ensemble {ae:.2f} (Layer 2, γ={GAMMA0}). "
                    f"Layer 2 strategies with a higher Sharpe than 1/N: {', '.join(beat_ew) or 'none'}.")
    s_ens = SVA[(SVA.split == "test") & (SVA.pair == "ensembles")]
    excl = s_ens[s_ens.mean_ret_diff_ann_ci_excludes_0 == True]
    seedavg = SVA[(SVA.split == "test") & SVA.pair.str.startswith("seed average")]
    sig_alpha = t[(t.alpha_vs_ew_nw_t.abs() > 1.96) & ~t.strategy.str.match(r"^(static|adaptive)_s\d+$")]
    L.append("## Summary\n")
    L.append("\n".join(f"- {s}" for s in sent) + "\n")
    L.append(f"In plain words, over the TEST period and net of {COST0} bp, the model-driven portfolios should be judged against two forecast-free allocations: 1/N and minimum variance. "
             f"Alphas versus 1/N with |NW t| > 1.96 (any layer, non-seed rows): {len(sig_alpha)} of {len(t[~t.strategy.str.match(r'^(static|adaptive)_s[0-9]+$') & (t.strategy != 'EW')])}"
             + (f" ({', '.join(sorted(set(sig_alpha.strategy + ' ' + sig_alpha.layer + ' h=' + sig_alpha.h.astype(str))))[:600]})" if len(sig_alpha) else "") + ". "
             f"For Static vs Adaptive, the ensemble return-difference CI excludes zero in {len(excl)} of {len(s_ens)} horizon-layer cells, "
             f"and across seed pairs Adaptive has the higher Sharpe in " + ", ".join(f"{r.layer} h={r.h}: {int(r.n_seed_pairs_adaptive_higher_sharpe)}/{r.pair.split('n=')[1].rstrip(')')}" for _, r in seedavg.iterrows()) + " pairs. "
             "Those date-only intervals are conditional on the trained seeds. "
             "All absolute levels come from a 17-stock universe of survivors and are likely optimistic. The differences between strategies are the relevant quantities, and the paper should present them as such.\n")
    L.append("## Reproduction\n")
    L.append("```sh\ncd revision_work/scripts/portfolio\nPY=../../../.venv/bin/python\n"
             "$PY -B engine.py --run-label original \\\n  --preds '../../outputs/preds/*_dw_h*_s*_val.parquet' '../../outputs/preds/*_dw_h*_s*_test.parquet' \\\n"
             "  --baseline-preds '../../outputs/skill_analysis/canonical_preds/*_daily_weekly_h*_s42_test.parquet'\n"
             "# causal-feature or new multi-seed predictions: new --run-label, point --preds at the new files,\n"
             "# and pass --adaptive-formal-csv none when a seed-42 Adaptive parquet is included.\n```\n")
    L.append("CSV files: `metrics_all.csv` (every strategy × layer × horizon × cost × split), `static_vs_adaptive.csv`, `seed_dispersion.csv`, `grs_tests.csv`, `long_short_diagnostic.csv`, "
             "`weights.csv` (every target weight), `rebalances.csv` (turnover and gross period return per rebalance), `daily_returns_gross_and_net10bp.csv` "
             "(net returns at any other cost follow from gross returns and turnover), and `qp_solver_log.csv`.\n")
    return "\n".join(L)


COLS = [("h", "h"), ("strategy", "Strategy"), ("ann_geo_return", "Ann. return"), ("ann_vol", "Ann. vol"), ("sharpe", "Sharpe"),
        ("max_drawdown", "Max DD"), ("avg_turnover_ex_initial", "Avg TO / rebal."), ("ann_turnover", "Ann. TO"), ("hit_rate_periods", "Hit rate"),
        ("ce_A2", "CE (A=2)"), ("ce_A5", "CE (A=5)"), ("alpha_vs_ew_ann", "α vs 1/N (ann.)"), ("alpha_vs_ew_nw_t", "α NW t")]
FM = {"ann_geo_return": p, "ann_vol": p, "max_drawdown": p, "ce_A2": p, "ce_A5": p, "alpha_vs_ew_ann": p, "sharpe": lambda x: f(x, 2),
      "avg_turnover_ex_initial": lambda x: f(x, 2), "ann_turnover": lambda x: f(x, 1), "hit_rate_periods": lambda x: f(x, 2), "alpha_vs_ew_nw_t": lambda x: f(x, 2)}


def build_report_subset(a, M, RP, notes, anchors, n_ffill, n_rf_missing, blocking, subset):
    keep = set(subset) | {"static (seed mean)", "adaptive (seed mean)"}
    sel = lambda df: df[df.strategy.isin(keep)]
    L = [f"# REPORT STEP 8 — risk-parity benchmark added (run: {a.run_label})\n",
         f"Generated {utc()} (UTC). No training. This run adds a long-only **risk parity** (equal risk contribution) allocation benchmark to the Step 8 engine. "
         f"The run `original` is unchanged; this directory `{a.run_label}` was created exclusively. Tables are restricted to {', '.join(subset)} and the Static/Adaptive seed means. "
         "Every computed strategy remains in `metrics_all.csv`.\n",
         "## BLOCKING\n", ("\n".join(f"- {b}" for b in blocking) if blocking else
                          "None. Risk contributions are equal to within 1e-6 at every decision date, and Global Train-Mean Layer 2 reproduces minimum variance.") + "\n",
         "## Design (unchanged from `original`, plus risk parity)\n",
         "- Timing: forecast at the close of *t*, executed at the close of *t*+1, held *h* trading days. Non-overlapping anchors, drifting weights, "
         f"turnover Σ|w_target − w_drifted| (initial build = 1), costs {COSTS_BP} bp per unit of one-way traded value deducted at rebalance, DTB3 risk-free rate "
         f"({n_ffill} days forward-filled, {n_rf_missing} missing), Ledoit–Wolf covariance on the trailing {LOOKBACK} daily returns up to and including *t*, scaled ×*h*.\n"
         "- **RISKPARITY**: the same Σ at each decision date. Solve min ½*y*′Σ*y* − (1/N)Σ log *y*ᵢ (strictly convex) by damped Newton with positivity-preserving backtracking, then normalise *w* = *y*/Σ*y*. "
         "At the optimum *y*ᵢ(Σ*y*)ᵢ = 1/N for every *i*, so the risk contributions RCᵢ = *w*ᵢ(Σ*w*)ᵢ / *w*′Σ*w* are all 1/N. This is verified at every date (`risk_parity_check.csv`). "
         "Long-only, fully invested, **no 0.25 cap** (none is needed or imposed). The ×*h* scaling of Σ does not change the weights.\n"
         "- Survivorship caveat: the universe is 17 stocks that survive to today, so absolute returns of every allocation are likely biased upward. Comparisons are within the same universe.\n"
         "- VALIDATION was the checkpoint-selection period for the models, so it is not out-of-sample for them.\n"]
    rc = RP.groupby(["split", "h"]).agg(n_dates=("decision_date", "count"), max_rc_dev=("max_abs_rc_deviation", "max"),
                                         max_weight=("max_weight", "max"), min_weight=("min_weight", "min"), max_newton_iter=("newton_iterations", "max")).reset_index()
    L.append("Risk-parity verification (per split and horizon, over all decision dates):\n")
    L.append(md(rc, [("split", "Split"), ("h", "h"), ("n_dates", "Decision dates"), ("max_rc_dev", "max |RC − 1/N|"), ("max_weight", "Max weight"),
                     ("min_weight", "Min weight"), ("max_newton_iter", "Max Newton iterations")],
                {"max_rc_dev": lambda x: f"{x:.1e}", "max_weight": lambda x: f(x, 4), "min_weight": lambda x: f(x, 4)}))
    L.append("Other checks: " + "; ".join(notes) + ".\n")
    for layer, title in ((f"L2_g{GAMMA0}", f"Layer 2 (mean–variance, γ = {GAMMA0})"), ("L1_top3", "Layer 1 (top-3 equal weight)"), ("L1_top5", "Layer 1 (top-5 equal weight)")):
        L.append(f"## TEST — {title}, {COST0} bp (allocation benchmarks repeated in each table)\n")
        L.append(md(sel(with_seed_means(M, None, "test", layer, COST0)), COLS, FM))
    # full cost grid
    rows = []
    for c in COSTS_BP:
        for layer in (f"L2_g{GAMMA0}", "L1_top3", "L1_top5"):
            t = sel(with_seed_means(M, None, "test", layer, c))
            t = t[(t.layer == layer) | (t.layer == "allocation benchmark")].copy()
            t["row"] = np.where(t.layer == "allocation benchmark", t.strategy, t.strategy + " · " + t.layer)
            t["cost_bp"] = c
            rows.append(t)
    G = pd.concat(rows).drop_duplicates(["h", "row", "cost_bp"])
    G["order"] = G.strategy.map({k: i for i, k in enumerate(BASE_ORDER)}).fillna(99)
    for metric, title, fmt_ in (("sharpe", "Sharpe ratio", lambda x: f(x, 2)), ("ann_geo_return", "annualised geometric return", p), ("alpha_vs_ew_ann", "alpha vs 1/N (ann.)", p)):
        pv = G.pivot_table(index=["h", "order", "row"], columns="cost_bp", values=metric).reset_index().sort_values(["h", "order", "row"])
        pv.columns = [str(c) for c in pv.columns]
        L.append(f"## TEST — full cost grid: {title}\n")
        L.append(md(pv, [("h", "h"), ("row", "Strategy")] + [(str(c), f"{c} bp") for c in COSTS_BP], {str(c): fmt_ for c in COSTS_BP}))
    L.append(f"## VALIDATION (checkpoint-selection period) — Layer 2 γ = {GAMMA0}, {COST0} bp\n")
    L.append(md(sel(with_seed_means(M, None, "val", f"L2_g{GAMMA0}", COST0)), COLS, FM))
    L.append("## Cumulative returns\n\n![TEST cumulative net returns](fig_cumulative_test_net10bp.png)\n\nTEST period, net of 10 bp. Data: `fig_cumulative_test_net10bp_data.csv`.\n")
    # summary (computed)
    t = M[(M.split == "test") & (M.cost_bp == COST0)]
    b = t[t.layer == "allocation benchmark"].pivot(index="h", columns="strategy", values="sharpe")
    a_ = t[t.layer == "allocation benchmark"].pivot(index="h", columns="strategy", values="alpha_vs_ew_ann")
    at = t[t.layer == "allocation benchmark"].pivot(index="h", columns="strategy", values="alpha_vs_ew_nw_t")
    l2 = t[t.layer == f"L2_g{GAMMA0}"].pivot(index="h", columns="strategy", values="sharpe")
    sm = with_seed_means(M, None, "test", f"L2_g{GAMMA0}", COST0).pivot_table(index="h", columns="strategy", values="sharpe")
    sent = []
    for h in HORIZONS:
        beat = [nm for nm in ("static_ens", "adaptive_ens") if l2.loc[h, nm] > b.loc[h, "RISKPARITY"]]
        sent.append(f"h={h}: Sharpe 1/N {b.loc[h, 'EW']:.2f}, minimum variance {b.loc[h, 'MINVAR']:.2f}, risk parity {b.loc[h, 'RISKPARITY']:.2f} "
                    f"(alpha vs 1/N {100 * a_.loc[h, 'RISKPARITY']:+.1f}%/yr, NW t {at.loc[h, 'RISKPARITY']:.2f}); Static ensemble {l2.loc[h, 'static_ens']:.2f}, "
                    f"Adaptive ensemble {l2.loc[h, 'adaptive_ens']:.2f}, seed means {sm.loc[h, 'static (seed mean)']:.2f} / {sm.loc[h, 'adaptive (seed mean)']:.2f}. "
                    f"Ensembles above risk parity: {', '.join(beat) or 'none'}.")
    L.append("## Summary\n")
    L.append("\n".join(f"- {x}" for x in sent) + "\n")
    vol = t[t.layer == "allocation benchmark"].pivot(index="h", columns="strategy", values="ann_vol")
    order_ok = [h for h in HORIZONS if vol.loc[h, "MINVAR"] <= vol.loc[h, "RISKPARITY"] <= vol.loc[h, "EW"]]
    L.append("Theory (Maillard, Roncalli & Teïletche, 2010) puts the risk-parity ex-ante volatility between minimum variance and 1/N. "
             "Realised TEST volatility follows that order at h = " + (", ".join(map(str, order_ok)) or "none") + " ("
             + "; ".join(f"h={h}: {100 * vol.loc[h, 'MINVAR']:.1f}% / {100 * vol.loc[h, 'RISKPARITY']:.1f}% / {100 * vol.loc[h, 'EW']:.1f}%" for h in HORIZONS)
             + " for MinVar / RiskPar / 1/N). Risk parity adds a third forecast-free reference point. "
             "The model portfolios should be read relative to all three, and absolute levels carry the survivorship caveat above.\n")
    L.append("## Reproduction\n")
    L.append("```sh\ncd revision_work/scripts/portfolio\nPY=../../../.venv/bin/python\n"
             "$PY -B engine.py --run-label original_rp --no-signals --models '(static|adaptive)_(s[0-9]+|ens)' \\\n"
             "  --report-subset EW,MINVAR,RISKPARITY,static_ens,adaptive_ens \\\n"
             "  --preds '../../outputs/preds/static_dw_h*_s*_val.parquet' '../../outputs/preds/static_dw_h*_s*_test.parquet' \\\n"
             "          '../../outputs/preds/adaptive_dw_h*_s*_val.parquet' '../../outputs/preds/adaptive_dw_h*_s*_test.parquet'\n```\n")
    return "\n".join(L)
