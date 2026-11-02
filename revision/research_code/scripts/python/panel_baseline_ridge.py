"""
Panel-ready Ridge baseline (Phase 2a, Step 1 — first model on the frozen
17-stock panel). Reuses the frozen sample index, chronological split, and
train-only normalization from panel_common.py; does not re-implement any of
that logic here.

Runs Ridge x {daily_only, weekly_only, daily_weekly} x {5d, 10d, 20d} = 9 runs.
Alpha is chosen per run via a small grid search on the val split (pooled
MSE); final metrics are reported on the test split only.

Outputs:
  dataset/audit/panel_ridge_summary_metrics.csv     9 rows: pooled + ticker-averaged metrics per (frequency, horizon)
  dataset/audit/panel_ridge_test_predictions.csv    sample-level test predictions (ticker, anchor_date, horizon, y_true, y_pred, frequency_setting, model)
"""

from __future__ import annotations

import argparse
import sys
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
)

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]
MODEL_NAME = "ridge"


def run_one(ds, stats, frequency: str, horizon: int):
    normalized = apply_normalizer(ds, stats, frequency)
    X = flatten_features(frequency, normalized)
    y = ds.y[horizon]

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    best_alpha, best_val_mse, best_model = None, np.inf, None
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha)
        model.fit(X[train_mask], y[train_mask])
        val_pred = model.predict(X[val_mask])
        val_mse = float(np.mean((val_pred - y[val_mask]) ** 2))
        if val_mse < best_val_mse:
            best_alpha, best_val_mse, best_model = alpha, val_mse, model

    test_pred = best_model.predict(X[test_mask])
    metrics = evaluate_predictions(
        y[test_mask], test_pred, ds.ticker[test_mask], ds.anchor_date[test_mask]
    )
    metrics.update({
        "model": MODEL_NAME,
        "frequency_setting": frequency,
        "horizon": f"{horizon}d",
        "best_alpha": best_alpha,
        "val_mse_at_best_alpha": best_val_mse,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
    })

    predictions_df = pd.DataFrame({
        "ticker": ds.ticker[test_mask],
        "anchor_date": ds.anchor_date[test_mask],
        "horizon": f"{horizon}d",
        "y_true": y[test_mask],
        "y_pred": test_pred,
        "frequency_setting": frequency,
        "model": MODEL_NAME,
    })

    return metrics, predictions_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Ridge baseline on the accepted V2 panel.")
    parser.add_argument("--output-dir", type=str, default=str(AUDIT_DIR), help="Directory for output files.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen 17-stock panel...")
    ds = load_frozen_panel()
    stats = fit_train_only_normalizer(ds)
    print(f"Loaded {len(ds.ticker)} pooled samples "
          f"(train={int((ds.split == 'train').sum())}, "
          f"val={int((ds.split == 'val').sum())}, "
          f"test={int((ds.split == 'test').sum())}).")

    all_metrics = []
    all_predictions = []

    for frequency in FREQUENCY_SETTINGS:
        for horizon in HORIZONS:
            print(f"Running Ridge | frequency={frequency} | horizon={horizon}d ...")
            metrics, predictions_df = run_one(ds, stats, frequency, horizon)
            all_metrics.append(metrics)
            all_predictions.append(predictions_df)
            print(
                f"  test pooled MSE={metrics['pooled_mse']:.6f}  "
                f"RankIC={metrics['cross_sectional_rank_ic']:.4f}  "
                f"pred_std/true_std={metrics['pooled_pred_std_over_true_std']:.3f}  "
                f"alpha={metrics['best_alpha']}"
            )

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(out_dir / "panel_ridge_summary_metrics.csv", index=False)

    predictions_df_all = pd.concat(all_predictions, ignore_index=True)
    predictions_df_all.to_csv(out_dir / "panel_ridge_test_predictions.csv", index=False)

    print("\nDone. Outputs:")
    print(f"  {out_dir / 'panel_ridge_summary_metrics.csv'}")
    print(f"  {out_dir / 'panel_ridge_test_predictions.csv'}")


if __name__ == "__main__":
    main()
