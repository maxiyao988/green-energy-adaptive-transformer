"""
Ridge baseline validation (Tasks A-F) on the FROZEN 17-stock, Daily+Weekly
panel. Adds naive baselines, a Ridge solver-stability check (default/'auto'
vs 'lsqr'), proper date-level cross-sectional Rank IC diagnostics, a
consolidated benchmark table, sanity checks, and a human-readable summary.

Does NOT change dataset construction, universe, split, or normalization —
all reused unchanged from panel_common.load_frozen_panel(). Does NOT
implement LSTM / Transformer / PathFormer.

Outputs (all under dataset/audit/):
  panel_naive_baseline_summary.csv
  panel_ridge_solver_stability.csv
  panel_ridge_solver_predictions.csv
  panel_ridge_rank_ic_by_date.csv
  panel_ridge_rank_ic_summary.csv
  panel_ridge_final_benchmark.csv
  panel_ridge_final_summary.txt
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import (
    AUDIT_DIR,
    FREQUENCY_SETTINGS,
    HORIZONS,
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    flatten_features,
    load_frozen_panel,
    rank_ic_by_date,
    safe_pearson_corr,
)

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]
SOLVERS_TO_COMPARE = ["auto", "lsqr"]  # "auto" = the default used by the original panel_baseline_ridge.py
AGREEMENT_CORR_THRESHOLD = 0.999

OUTPUT_FILES = [
    "panel_naive_baseline_summary.csv",
    "panel_ridge_solver_stability.csv",
    "panel_ridge_solver_predictions.csv",
    "panel_ridge_rank_ic_by_date.csv",
    "panel_ridge_rank_ic_summary.csv",
    "panel_ridge_final_benchmark.csv",
    "panel_ridge_final_summary.txt",
]


# =====================================================================
# TASK A: naive baselines (zero, train_mean)
# =====================================================================

def run_naive_baselines(ds, output_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    train_mask = ds.split == "train"
    test_mask = ds.split == "test"
    ticker_test = ds.ticker[test_mask]
    date_test = ds.anchor_date[test_mask]

    for horizon in HORIZONS:
        y = ds.y[horizon]
        y_train = y[train_mask]
        y_test = y[test_mask]

        zero_pred = np.zeros_like(y_test)
        m = evaluate_predictions(y_test, zero_pred, ticker_test, date_test)
        m.update({"model": "zero", "frequency_setting": "n/a", "horizon": f"{horizon}d"})
        rows.append(m)

        train_mean_value = float(np.mean(y_train))  # TRAIN split only, never val/test
        mean_pred = np.full_like(y_test, train_mean_value)
        m2 = evaluate_predictions(y_test, mean_pred, ticker_test, date_test)
        m2.update({
            "model": "train_mean", "frequency_setting": "n/a", "horizon": f"{horizon}d",
            "train_mean_value": train_mean_value,
        })
        rows.append(m2)

    df = pd.DataFrame(rows)
    out_dir = output_dir or AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "panel_naive_baseline_summary.csv", index=False)
    return df


# =====================================================================
# TASK B: Ridge solver-stability check (auto vs lsqr)
# =====================================================================

def fit_predict_ridge(ds, stats, frequency: str, horizon: int, solver: str) -> dict:
    normalized = apply_normalizer(ds, stats, frequency)
    X = flatten_features(frequency, normalized)
    y = ds.y[horizon]

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    best_alpha, best_val_mse, best_model, best_had_warning = None, np.inf, None, False
    for alpha in ALPHA_GRID:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = Ridge(alpha=alpha, solver=solver)
            model.fit(X[train_mask], y[train_mask])
        had_warning = any("ill-conditioned" in str(w.message).lower() for w in caught)

        val_pred = model.predict(X[val_mask])  # alpha selection uses VAL only, never test
        val_mse = float(np.mean((val_pred - y[val_mask]) ** 2))
        if val_mse < best_val_mse:
            best_alpha, best_val_mse, best_model, best_had_warning = alpha, val_mse, model, had_warning

    test_pred = best_model.predict(X[test_mask])
    return {
        "test_pred": test_pred,
        "test_mask": test_mask,
        "val_mask": val_mask,
        "train_mask": train_mask,
        "best_alpha": best_alpha,
        "val_mse": best_val_mse,
        "had_ill_conditioned_warning": best_had_warning,
    }


def run_solver_comparison(ds, stats, output_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    stability_rows = []
    prediction_frames = []
    cache = {}  # (frequency, horizon, solver) -> fit_predict_ridge() result, reused by Tasks C/D/E

    for frequency in FREQUENCY_SETTINGS:
        for horizon in HORIZONS:
            solver_results = {}
            config_row_indices = []
            for solver in SOLVERS_TO_COMPARE:
                res = fit_predict_ridge(ds, stats, frequency, horizon, solver)
                solver_results[solver] = res
                cache[(frequency, horizon, solver)] = res

                metrics = evaluate_predictions(
                    ds.y[horizon][res["test_mask"]], res["test_pred"],
                    ds.ticker[res["test_mask"]], ds.anchor_date[res["test_mask"]],
                )
                stability_rows.append({
                    "frequency_setting": frequency,
                    "horizon": f"{horizon}d",
                    "solver": solver,
                    "selected_alpha": res["best_alpha"],
                    "val_mse": res["val_mse"],
                    "test_mse": metrics["pooled_mse"],
                    "test_mae": metrics["pooled_mae"],
                    "test_corr": metrics["pooled_corr"],
                    "cross_sectional_rank_ic": metrics["cross_sectional_rank_ic"],
                    "direction_accuracy": metrics["pooled_da"],
                    "pred_std_over_true_std": metrics["pooled_pred_std_over_true_std"],
                    "had_ill_conditioned_warning": res["had_ill_conditioned_warning"],
                })
                config_row_indices.append(len(stability_rows) - 1)

                prediction_frames.append(pd.DataFrame({
                    "ticker": ds.ticker[res["test_mask"]],
                    "anchor_date": ds.anchor_date[res["test_mask"]],
                    "horizon": f"{horizon}d",
                    "frequency_setting": frequency,
                    "solver": solver,
                    "y_true": ds.y[horizon][res["test_mask"]],
                    "y_pred": res["test_pred"],
                }))

            pred_a = solver_results[SOLVERS_TO_COMPARE[0]]["test_pred"]
            pred_b = solver_results[SOLVERS_TO_COMPARE[1]]["test_pred"]
            agree_corr = safe_pearson_corr(pred_a, pred_b)
            mean_abs_diff = float(np.mean(np.abs(pred_a - pred_b)))
            max_abs_diff = float(np.max(np.abs(pred_a - pred_b)))
            for idx in config_row_indices:
                stability_rows[idx]["pred_corr_between_solvers"] = agree_corr
                stability_rows[idx]["mean_abs_diff_between_solvers"] = mean_abs_diff
                stability_rows[idx]["max_abs_diff_between_solvers"] = max_abs_diff

    stability_df = pd.DataFrame(stability_rows)
    out_dir = output_dir or AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stability_df.to_csv(out_dir / "panel_ridge_solver_stability.csv", index=False)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df.to_csv(out_dir / "panel_ridge_solver_predictions.csv", index=False)

    return stability_df, cache


def determine_formal_solver(stability_df: pd.DataFrame) -> tuple[str, bool]:
    """If lsqr agrees closely with the default ('auto') solver on every config,
    designate lsqr as formal (removes the ill-conditioning warning with no
    material change). Otherwise keep 'auto' as formal and flag for manual review."""
    min_corr = stability_df["pred_corr_between_solvers"].min()
    agrees_well = (not pd.isna(min_corr)) and (min_corr >= AGREEMENT_CORR_THRESHOLD)
    return ("lsqr", True) if agrees_well else ("auto", False)


# =====================================================================
# TASK C: proper cross-sectional Rank IC diagnostics (formal solver only)
# =====================================================================

def run_rank_ic_diagnostics(ds, cache: dict, formal_solver: str, output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_date_frames = []
    summary_rows = []

    for frequency in FREQUENCY_SETTINGS:
        for horizon in HORIZONS:
            res = cache[(frequency, horizon, formal_solver)]
            y_true = ds.y[horizon][res["test_mask"]]
            y_pred = res["test_pred"]
            ticker = ds.ticker[res["test_mask"]]
            date = ds.anchor_date[res["test_mask"]]

            ic_df = rank_ic_by_date(y_true, y_pred, ticker, date)
            ic_df["model"] = "ridge"
            ic_df["solver"] = formal_solver
            ic_df["frequency_setting"] = frequency
            ic_df["horizon"] = f"{horizon}d"
            by_date_frames.append(ic_df)

            valid = ic_df["rank_ic"].dropna()
            n_valid = len(valid)
            mean_ic = float(valid.mean()) if n_valid else float("nan")
            median_ic = float(valid.median()) if n_valid else float("nan")
            std_ic = float(valid.std()) if n_valid > 1 else float("nan")
            pos_ratio = float((valid > 0).mean()) if n_valid else float("nan")
            icir = (mean_ic / std_ic) if (n_valid > 1 and std_ic and std_ic > 1e-12) else float("nan")

            summary_rows.append({
                "model": "ridge", "solver": formal_solver, "frequency_setting": frequency, "horizon": f"{horizon}d",
                "mean_rank_ic": mean_ic, "median_rank_ic": median_ic, "rank_ic_std": std_ic,
                "positive_ic_ratio": pos_ratio, "n_valid_ic_dates": n_valid,
                "rank_icir_descriptive_not_annualized": icir,
            })

    by_date_df = pd.concat(by_date_frames, ignore_index=True)
    out_dir = output_dir or AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    by_date_df.to_csv(out_dir / "panel_ridge_rank_ic_by_date.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "panel_ridge_rank_ic_summary.csv", index=False)
    return by_date_df, summary_df


# =====================================================================
# TASK D: consolidated benchmark table
# =====================================================================

def build_final_benchmark(ds, naive_df: pd.DataFrame, cache: dict, formal_solver: str,
                           rank_ic_summary_df: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    rows = []

    for _, r in naive_df.iterrows():
        rows.append({
            "model": r["model"], "frequency_setting": "n/a", "horizon": r["horizon"],
            "solver": "n/a", "alpha": "n/a",
            "pooled_mse": r["pooled_mse"], "pooled_mae": r["pooled_mae"], "pooled_corr": r["pooled_corr"],
            "ticker_avg_mse": r["ticker_avg_mse"], "ticker_avg_mae": r["ticker_avg_mae"],
            "ticker_avg_corr": r["ticker_avg_corr"],
            "mean_rank_ic": float("nan"), "median_rank_ic": float("nan"), "rank_ic_std": float("nan"),
            "positive_ic_ratio": float("nan"), "rank_icir": float("nan"),
            "direction_accuracy": r["pooled_da"], "pred_std": r["pooled_pred_std"], "true_std": r["pooled_true_std"],
            "pred_std_true_std_ratio": r["pooled_pred_std_over_true_std"],
        })

    for frequency in FREQUENCY_SETTINGS:
        for horizon in HORIZONS:
            res = cache[(frequency, horizon, formal_solver)]
            metrics = evaluate_predictions(
                ds.y[horizon][res["test_mask"]], res["test_pred"],
                ds.ticker[res["test_mask"]], ds.anchor_date[res["test_mask"]],
            )
            ic_row = rank_ic_summary_df[
                (rank_ic_summary_df["frequency_setting"] == frequency)
                & (rank_ic_summary_df["horizon"] == f"{horizon}d")
            ].iloc[0]
            rows.append({
                "model": "ridge", "frequency_setting": frequency, "horizon": f"{horizon}d",
                "solver": formal_solver, "alpha": res["best_alpha"],
                "pooled_mse": metrics["pooled_mse"], "pooled_mae": metrics["pooled_mae"],
                "pooled_corr": metrics["pooled_corr"],
                "ticker_avg_mse": metrics["ticker_avg_mse"], "ticker_avg_mae": metrics["ticker_avg_mae"],
                "ticker_avg_corr": metrics["ticker_avg_corr"],
                "mean_rank_ic": ic_row["mean_rank_ic"], "median_rank_ic": ic_row["median_rank_ic"],
                "rank_ic_std": ic_row["rank_ic_std"], "positive_ic_ratio": ic_row["positive_ic_ratio"],
                "rank_icir": ic_row["rank_icir_descriptive_not_annualized"],
                "direction_accuracy": metrics["pooled_da"], "pred_std": metrics["pooled_pred_std"],
                "true_std": metrics["pooled_true_std"],
                "pred_std_true_std_ratio": metrics["pooled_pred_std_over_true_std"],
            })

    df = pd.DataFrame(rows)
    out_dir = output_dir or AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "panel_ridge_final_benchmark.csv", index=False)
    return df


# =====================================================================
# TASK E: sanity checks
# =====================================================================

def run_sanity_checks(ds, cache: dict, formal_solver: str) -> list[str]:
    violations = []
    expected_test_samples = int((ds.split == "test").sum())

    for frequency in FREQUENCY_SETTINGS:
        n_test = int(cache[(frequency, HORIZONS[0], formal_solver)]["test_mask"].sum())
        if n_test != expected_test_samples:
            violations.append(
                f"[MAJOR] frequency={frequency} test sample count={n_test}, expected {expected_test_samples}"
            )

    def test_pairs(frequency):
        mask = cache[(frequency, HORIZONS[0], formal_solver)]["test_mask"]
        return set(zip(ds.ticker[mask], ds.anchor_date[mask]))

    pairs = {f: test_pairs(f) for f in FREQUENCY_SETTINGS}
    if not (pairs["daily_only"] == pairs["weekly_only"] == pairs["daily_weekly"]):
        violations.append("[MAJOR] daily_only/weekly_only/daily_weekly test sample sets are NOT identical")

    test_mask0 = cache[(FREQUENCY_SETTINGS[0], HORIZONS[0], formal_solver)]["test_mask"]
    df = pd.DataFrame({"ticker": ds.ticker[test_mask0], "date": ds.anchor_date[test_mask0]})
    counts = df.groupby("date")["ticker"].nunique()
    bad_dates = counts[counts != 17]
    if len(bad_dates) > 0:
        violations.append(f"[WARNING] {len(bad_dates)} test dates do not have exactly 17 tickers")

    for (frequency, horizon, solver), res in cache.items():
        pred = res["test_pred"]
        if np.isnan(pred).any() or np.isinf(pred).any():
            violations.append(
                f"[MAJOR] NaN/Inf found in predictions for frequency={frequency}, horizon={horizon}, solver={solver}"
            )

    train_dates = ds.anchor_date[ds.split == "train"]
    val_dates = ds.anchor_date[ds.split == "val"]
    test_dates = ds.anchor_date[ds.split == "test"]
    if len(train_dates) and len(val_dates) and train_dates.max() >= val_dates.min():
        violations.append("[MAJOR] train dates overlap with/exceed val dates (leakage)")
    if len(val_dates) and len(test_dates) and val_dates.max() >= test_dates.min():
        violations.append("[MAJOR] val dates overlap with/exceed test dates (leakage)")

    return violations


# =====================================================================
# TASK F: human-readable summary
# =====================================================================

def write_final_summary(final_df: pd.DataFrame, stability_df: pd.DataFrame,
                         formal_solver: str, solvers_agree: bool, output_dir: Path | None = None) -> None:
    lines = ["RIDGE BASELINE VALIDATION SUMMARY (frozen 17-stock panel)", "=" * 72, ""]

    lines.append("1) Lowest pooled MSE frequency setting per horizon (Ridge only):")
    for horizon in HORIZONS:
        h = f"{horizon}d"
        ridge_rows = final_df[(final_df["model"] == "ridge") & (final_df["horizon"] == h)]
        best = ridge_rows.loc[ridge_rows["pooled_mse"].idxmin()]
        lines.append(f"   {h}: {best['frequency_setting']} (MSE={best['pooled_mse']:.6f})")

    lines.append("")
    lines.append("2) Highest mean cross-sectional Rank IC frequency setting per horizon (Ridge only):")
    for horizon in HORIZONS:
        h = f"{horizon}d"
        ridge_rows = final_df[(final_df["model"] == "ridge") & (final_df["horizon"] == h)]
        if ridge_rows["mean_rank_ic"].notna().any():
            best = ridge_rows.loc[ridge_rows["mean_rank_ic"].idxmax()]
            lines.append(f"   {h}: {best['frequency_setting']} (mean Rank IC={best['mean_rank_ic']:.4f})")
        else:
            lines.append(f"   {h}: no valid Rank IC available")

    lines.append("")
    lines.append("3) Does Daily+Weekly improve ranking relative to Daily-only/Weekly-only? (descriptive comparison only)")
    for horizon in HORIZONS:
        h = f"{horizon}d"
        ridge_rows = final_df[(final_df["model"] == "ridge") & (final_df["horizon"] == h)].set_index("frequency_setting")
        needed = {"daily_only", "weekly_only", "daily_weekly"}
        if needed.issubset(ridge_rows.index) and ridge_rows.loc[list(needed), "mean_rank_ic"].notna().all():
            dw = ridge_rows.loc["daily_weekly", "mean_rank_ic"]
            d = ridge_rows.loc["daily_only", "mean_rank_ic"]
            w = ridge_rows.loc["weekly_only", "mean_rank_ic"]
            better = dw > max(d, w)
            lines.append(f"   {h}: daily_weekly={dw:.4f} vs daily_only={d:.4f}, weekly_only={w:.4f} -> daily_weekly higher: {better}")
        else:
            lines.append(f"   {h}: cannot compare (missing or undefined Rank IC values)")

    lines.append("")
    lines.append("4) Does Ridge beat Zero and Train Mean on pooled MSE?")
    for horizon in HORIZONS:
        h = f"{horizon}d"
        naive_mse = final_df[(final_df["model"].isin(["zero", "train_mean"])) & (final_df["horizon"] == h)]["pooled_mse"].min()
        ridge_mse = final_df[(final_df["model"] == "ridge") & (final_df["horizon"] == h)]["pooled_mse"].min()
        lines.append(f"   {h}: best naive MSE={naive_mse:.6f}, best Ridge MSE={ridge_mse:.6f} -> Ridge beats naive: {ridge_mse < naive_mse}")

    lines.append("")
    lines.append("5) Prediction dispersion (Pred Std / True Std; <1.0 = under-dispersed, >1.0 = over-dispersed):")
    for _, r in final_df[final_df["model"] == "ridge"].iterrows():
        lines.append(f"   ridge | {r['frequency_setting']} | {r['horizon']}: ratio={r['pred_std_true_std_ratio']:.3f}")

    lines.append("")
    lines.append("6) Solver agreement (auto vs lsqr):")
    min_corr = stability_df["pred_corr_between_solvers"].min()
    max_mean_diff = stability_df["mean_abs_diff_between_solvers"].max()
    lines.append(f"   min prediction correlation across all 9 configs: {min_corr:.6f}")
    lines.append(f"   max mean_abs_diff across all 9 configs: {max_mean_diff:.6g}")
    verdict = "ESSENTIALLY IDENTICAL" if solvers_agree else "MATERIALLY DIFFERENT — needs manual review"
    lines.append(f"   -> predictions are {verdict} between solvers.")
    lines.append(f"   -> designated formal solver going forward: {formal_solver}")

    lines.append("")
    lines.append("7) Is Ridge stable enough to freeze as the formal ML baseline?")
    any_warning_with_formal = bool(stability_df[stability_df["solver"] == formal_solver]["had_ill_conditioned_warning"].any())
    lines.append(f"   Ill-conditioned warning present with formal solver ({formal_solver}): {any_warning_with_formal}")
    lines.append(f"   Solver choice materially changes predictions: {not solvers_agree}")
    stable = solvers_agree and not any_warning_with_formal
    lines.append(f"   -> Ridge numerical stability verdict: {'STABLE, can freeze as formal ML baseline' if stable else 'NEEDS FURTHER REVIEW before freezing'}")

    lines.append("")
    lines.append("IMPORTANT CAVEATS (do not over-interpret):")
    lines.append("  - A positive Rank IC alone does NOT establish predictive success; it only measures ranking/ordering information.")
    lines.append("  - MSE/MAE measure point-forecast accuracy; Corr/Rank IC/DA measure directional or ranking information;")
    lines.append("    Pred Std / True Std measures calibration/dispersion. These are three distinct dimensions and should")
    lines.append("    not be conflated into a single verdict of 'the model works'.")

    out_dir = output_dir or AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "panel_ridge_final_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# =====================================================================
# main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Ridge baseline validation on the accepted V2 panel.")
    parser.add_argument("--output-dir", type=str, default=str(AUDIT_DIR), help="Directory for output files.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen 17-stock panel (reusing panel_common; universe/split/normalization unchanged)...")
    ds = load_frozen_panel()
    stats = fit_train_only_normalizer(ds)

    print("\n=== TASK A: naive baselines (zero, train_mean) ===")
    naive_df = run_naive_baselines(ds, out_dir)

    print("\n=== TASK B: Ridge solver stability (auto vs lsqr) ===")
    stability_df, cache = run_solver_comparison(ds, stats, out_dir)
    formal_solver, solvers_agree = determine_formal_solver(stability_df)
    print(f"Min prediction correlation between solvers across all 9 configs: {stability_df['pred_corr_between_solvers'].min():.6f}")
    print(f"Designated formal solver: {formal_solver} (solvers_agree_well={solvers_agree})")

    print("\n=== TASK C: cross-sectional Rank IC diagnostics (formal solver) ===")
    _, rank_ic_summary_df = run_rank_ic_diagnostics(ds, cache, formal_solver, out_dir)

    print("\n=== TASK D: consolidated benchmark table ===")
    final_df = build_final_benchmark(ds, naive_df, cache, formal_solver, rank_ic_summary_df, out_dir)

    print("\n=== TASK E: sanity checks ===")
    violations = run_sanity_checks(ds, cache, formal_solver)
    major_violations = [v for v in violations if v.startswith("[MAJOR]")]
    if violations:
        for v in violations:
            print(v)
    else:
        print("All sanity checks passed with no violations.")
    print("(By construction: naive train_mean uses TRAIN-only stats; alpha selection uses VAL-only, never test;")
    print(" Rank IC is computed per-date cross-sectionally via rank_ic_by_date(), not pooled across all rows.)")

    if major_violations:
        print("\nMAJOR PROTOCOL VIOLATION DETECTED — stopping before writing the final summary.")
        sys.exit(1)

    print("\n=== TASK F: human-readable summary ===")
    write_final_summary(final_df, stability_df, formal_solver, solvers_agree, out_dir)

    print("\nRIDGE BASELINE VALIDATION COMPLETE")
    print(f"chosen formal Ridge solver: {formal_solver}")
    for horizon in HORIZONS:
        h = f"{horizon}d"
        ridge_rows = final_df[(final_df["model"] == "ridge") & (final_df["horizon"] == h)]
        best_mse_row = ridge_rows.loc[ridge_rows["pooled_mse"].idxmin()]
        print(f"  {h}: best MSE setting = {best_mse_row['frequency_setting']} (MSE={best_mse_row['pooled_mse']:.6f})")
        if ridge_rows["mean_rank_ic"].notna().any():
            best_ic_row = ridge_rows.loc[ridge_rows["mean_rank_ic"].idxmax()]
            print(f"  {h}: best Rank IC setting = {best_ic_row['frequency_setting']} (mean Rank IC={best_ic_row['mean_rank_ic']:.4f})")
        else:
            print(f"  {h}: no valid Rank IC available across settings")

    naive_best_by_h = final_df[final_df["model"].isin(["zero", "train_mean"])].groupby("horizon")["pooled_mse"].min()
    ridge_best_by_h = final_df[final_df["model"] == "ridge"].groupby("horizon")["pooled_mse"].min()
    beats_naive = bool((ridge_best_by_h.reindex(naive_best_by_h.index) < naive_best_by_h).all())
    print(f"  Ridge beats naive baselines on pooled MSE (all horizons): {beats_naive}")
    print(f"  Numerical instability remaining: {'no' if solvers_agree else 'YES - manual review needed'}")
    print("  Output files:")
    for f in OUTPUT_FILES:
        print(f"    {out_dir / f}")


if __name__ == "__main__":
    main()
