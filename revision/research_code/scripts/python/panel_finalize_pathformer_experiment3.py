"""Standalone immutable finalizer for the Experiment 3 PathFormer mechanism ablation.

This script performs a read-only static audit of the Experiment 3 evidence set.
It never trains, evaluates, or rewrites model artifacts. It validates the raw
training/test evidence, verifies provenance and dependency hashes, binds the
frozen Adaptive MAIN V2 daily_weekly evidence, reconstructs the required metrics
from the saved TEST predictions, and writes exactly the required final artifacts.

The script fails closed if final outputs already exist or if any validation fails.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import evaluate_predictions, rank_ic_by_date

EXPERIMENT_DIR = ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation"
MAIN_V2_DIR = ROOT / "dataset" / "audit" / "main_v2" / "pathformer"
DATASET_MANIFEST_PATH = ROOT / "dataset" / "multiscale_dataset" / "panel" / "panel_build_manifest.csv"

EXPERIMENT3_RUNNER_PATH = ROOT / "scripts" / "python" / "panel_pathformer_experiment3.py"
PANEL_COMMON_PATH = ROOT / "scripts" / "python" / "panel_common.py"
MAIN_V2_PATHFORMER_SCRIPT_PATH = ROOT / "scripts" / "python" / "panel_pathformer_main_v2.py"

FINAL_OUTPUTS = (
    EXPERIMENT_DIR / "experiment3_mechanism_comparison.csv",
    EXPERIMENT_DIR / "experiment3_run_manifest.csv",
    EXPERIMENT_DIR / "experiment3_final_summary.txt",
)
EXPECTED_TRAINING_SUMMARY_ROWS = 54
EXPECTED_SINGLE_CELLS = 48
EXPECTED_FIXED_CELLS = 3
EXPECTED_STATIC_CELLS = 3
EXPECTED_SINGLE_VALIDATION_ROWS = 301920
EXPECTED_SINGLE_SELECTION_ROWS = 3
EXPECTED_CHECKPOINT_COUNT = 54
EXPECTED_TEST_SUMMARY_ROWS = 9
EXPECTED_TEST_PREDICTION_ROWS = 59823
EXPECTED_TEST_RANK_IC_ROWS = 3519
EXPECTED_TEST_WEIGHT_ROWS = 59823
EXPECTED_ADAPTIVE_DW_SUMMARY_ROWS = 3
EXPECTED_ADAPTIVE_DW_PREDICTION_ROWS = 19941
EXPECTED_ADAPTIVE_DW_RANK_IC_ROWS = 1173
EXPECTED_ADAPTIVE_DW_ROUTER_ROWS = 19941

EXPECTED_EXPERIMENT3_RUNNER_SHA = "6f3c1f804c9610fef6f29453cc9a49b121f927fbec90e3482e2744a924aa0142"
EXPECTED_DATASET_MANIFEST_SHA = "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c"
EXPECTED_MAIN_V2_SCRIPT_SHA = "5f05ccd677359a899ca91c3baeeda3ecb6aae1a63ff75593cd31809ae27e6f77"
EXPECTED_PANEL_COMMON_SHA = "ccaba201b910b66fb5c3272bcc3aa18e04d12c098bb26100a73a3717ec6d2834"

EXPECTED_ADAPTIVE_HASHES = {
    "summary": "90e987cd87d9bdbc82c6bf587627863027878a6d170f7a6cca80569ee39de5e3",
    "predictions": "a05955a561da1b8880f1bb0878a6576852c087cc45707546d446ce96b0fabdaf",
    "rank_ic": "52b3f3475dc4db74523cfe99a09f71f2b33bd14c58b458ba2e62564ca7283298",
    "router": "bcf06150f21d35d9ec10132d248d088cc758c9782771efdf9b68849531c9814e",
}

HORIZONS = ["5d", "10d", "20d"]
HORIZON_MAP = {"5d": 5, "10d": 10, "20d": 20}
INT_TO_HORIZON_LABEL = {5: "5d", 10: "10d", 20: "20d"}
MECHANISMS = ["single", "fixed", "static", "adaptive"]
VALIDATION_MSE_RTOL = 1e-6
VALIDATION_MSE_ATOL = 1e-12


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required {label}: {path}")


def _require_not_exist(path: Path, label: str) -> None:
    if path.exists():
        raise RuntimeError(f"Final output already exists for {label}: {path}")


def _check_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def _canonical_anchor_date(value) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    _require(path, label)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{label} is empty: {path}")
    return df


def _validate_provenance_and_hashes() -> dict:
    _require(EXPERIMENT3_RUNNER_PATH, "Experiment 3 runner script")
    _require(DATASET_MANIFEST_PATH, "dataset manifest")
    _require(PANEL_COMMON_PATH, "panel_common.py")
    _require(MAIN_V2_PATHFORMER_SCRIPT_PATH, "MAIN V2 PathFormer script")
    _require(EXPERIMENT_DIR / "experiment3_training_provenance.csv", "Experiment 3 training provenance")

    hashes = {
        "experiment3_runner_sha256": _check_hash(EXPERIMENT3_RUNNER_PATH, EXPECTED_EXPERIMENT3_RUNNER_SHA, "Experiment 3 runner script"),
        "dataset_manifest_sha256": _check_hash(DATASET_MANIFEST_PATH, EXPECTED_DATASET_MANIFEST_SHA, "dataset manifest"),
        "panel_common_sha256": _check_hash(PANEL_COMMON_PATH, EXPECTED_PANEL_COMMON_SHA, "panel_common.py"),
        "main_v2_pathformer_script_sha256": _check_hash(MAIN_V2_PATHFORMER_SCRIPT_PATH, EXPECTED_MAIN_V2_SCRIPT_SHA, "MAIN V2 PathFormer script"),
    }

    provenance_df = _load_csv(EXPERIMENT_DIR / "experiment3_training_provenance.csv", "Experiment 3 training provenance")
    if len(provenance_df) != 1:
        raise RuntimeError(f"Experiment 3 provenance must contain exactly one row; found {len(provenance_df)}.")
    provenance = provenance_df.iloc[0].to_dict()
    required = {
        "run_family": "EXPERIMENT3_PATHFORMER_ABLATION",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "frequency_configuration": "daily_weekly",
        "seed": 42,
        "execution_backend": "CPU",
        "selection_metric": "validation_mse",
        "single_selection_uses_test": False,
    }
    for field, expected_value in required.items():
        if field not in provenance:
            raise RuntimeError(f"Experiment 3 provenance missing required field: {field}")
        if str(provenance[field]) != str(expected_value):
            raise RuntimeError(f"Experiment 3 provenance mismatch for {field}: expected {expected_value}, got {provenance[field]}")

    required_hash_fields = {
        "training_script_sha256": EXPECTED_EXPERIMENT3_RUNNER_SHA,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA,
        "frozen_pathformer_script_sha256": EXPECTED_MAIN_V2_SCRIPT_SHA,
        "panel_common_sha256": EXPECTED_PANEL_COMMON_SHA,
    }
    for field, expected in required_hash_fields.items():
        if field not in provenance:
            raise RuntimeError(f"Experiment 3 provenance missing hash field: {field}")
        if str(provenance[field]) != str(expected):
            raise RuntimeError(f"Experiment 3 provenance hash mismatch for {field}: expected {expected}, got {provenance[field]}")

    hashes.update({
        "run_family": provenance["run_family"],
        "temporal_status": provenance["temporal_status"],
        "dataset_contract_version": provenance["dataset_contract_version"],
        "frequency_configuration": provenance["frequency_configuration"],
        "seed": provenance["seed"],
        "execution_backend": provenance["execution_backend"],
        "selection_metric": provenance["selection_metric"],
        "single_selection_uses_test": provenance["single_selection_uses_test"],
    })
    return hashes


def _validate_training_summary(summary_df: pd.DataFrame) -> None:
    if len(summary_df) != EXPECTED_TRAINING_SUMMARY_ROWS:
        raise RuntimeError(f"Expected training summary rows = {EXPECTED_TRAINING_SUMMARY_ROWS}, found {len(summary_df)}")

    mechanism_counts = summary_df["mechanism"].value_counts().to_dict()
    if mechanism_counts.get("single", 0) != EXPECTED_SINGLE_CELLS:
        raise RuntimeError(f"Expected Single training cells = {EXPECTED_SINGLE_CELLS}, found {mechanism_counts.get('single', 0)}")
    if mechanism_counts.get("fixed", 0) != EXPECTED_FIXED_CELLS:
        raise RuntimeError(f"Expected Fixed training cells = {EXPECTED_FIXED_CELLS}, found {mechanism_counts.get('fixed', 0)}")
    if mechanism_counts.get("static", 0) != EXPECTED_STATIC_CELLS:
        raise RuntimeError(f"Expected Static training cells = {EXPECTED_STATIC_CELLS}, found {mechanism_counts.get('static', 0)}")

    required_columns = {"mechanism", "horizon", "daily_patch", "weekly_patch", "seed", "parameter_count", "validation_mse"}
    missing = required_columns - set(summary_df.columns)
    if missing:
        raise RuntimeError(f"Training summary missing columns: {sorted(missing)}")

    if len(summary_df[summary_df["mechanism"] == "single"]) != EXPECTED_SINGLE_CELLS:
        raise RuntimeError("Single training cells mismatch in summary.")

    if summary_df["seed"].nunique() != 1 or int(summary_df["seed"].iloc[0]) != 42:
        raise RuntimeError("Training summary must have seed 42 across all rows.")

    checkpoint_dir = EXPERIMENT_DIR / "checkpoints"
    if not checkpoint_dir.exists():
        raise RuntimeError("Training checkpoint directory is missing.")
    checkpoint_files = [p for p in checkpoint_dir.iterdir() if p.is_file()]
    if len(checkpoint_files) != EXPECTED_CHECKPOINT_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_CHECKPOINT_COUNT} checkpoints, found {len(checkpoint_files)}")


def _validate_single_validation_predictions(val_df: pd.DataFrame) -> None:
    if len(val_df) != EXPECTED_SINGLE_VALIDATION_ROWS:
        raise RuntimeError(f"Expected Single validation predictions = {EXPECTED_SINGLE_VALIDATION_ROWS}, found {len(val_df)}")

    required = {"mechanism", "horizon", "daily_patch", "weekly_patch", "ticker", "anchor_date", "y_true", "y_pred"}
    missing = required - set(val_df.columns)
    if missing:
        raise RuntimeError(f"Validation predictions missing columns: {sorted(missing)}")

    if (val_df["y_true"].apply(np.isfinite).to_numpy() == False).any():
        raise RuntimeError("Validation y_true contains non-finite values.")
    if (val_df["y_pred"].apply(np.isfinite).to_numpy() == False).any():
        raise RuntimeError("Validation y_pred contains non-finite values.")

    if val_df["mechanism"].nunique() != 1 or str(val_df["mechanism"].iloc[0]) != "single":
        raise RuntimeError("Validation predictions must be single-mechanism evidence only.")

    if val_df.duplicated(subset=["mechanism", "horizon", "daily_patch", "weekly_patch", "ticker", "anchor_date"]).any():
        raise RuntimeError("Duplicate validation prediction keys detected.")


def _validate_single_candidate_predictions(val_df: pd.DataFrame, horizon: str, daily_patch: int, weekly_patch: int, summary_row: pd.Series | None = None) -> float:
    subset = val_df[
        (val_df["mechanism"].astype(str) == "single")
        & (val_df["horizon"].astype(str) == horizon)
        & (val_df["daily_patch"].astype(int) == int(daily_patch))
        & (val_df["weekly_patch"].astype(int) == int(weekly_patch))
    ].copy()
    if len(subset) != 6290:
        raise RuntimeError(f"Single candidate {horizon}/d={daily_patch}/w={weekly_patch} expected 6290 rows, found {len(subset)}.")
    if subset["anchor_date"].nunique() != 370:
        raise RuntimeError(f"Single candidate {horizon}/d={daily_patch}/w={weekly_patch} expected 370 dates, found {subset['anchor_date'].nunique()}.")
    if subset["ticker"].nunique() != 17:
        raise RuntimeError(f"Single candidate {horizon}/d={daily_patch}/w={weekly_patch} expected 17 tickers, found {subset['ticker'].nunique()}.")
    if subset.duplicated(subset=["ticker", "anchor_date"]).any():
        raise RuntimeError(f"Single candidate {horizon}/d={daily_patch}/w={weekly_patch} contains duplicate ticker/date keys.")
    if not np.isfinite(subset["y_true"].to_numpy(dtype=np.float64)).all() or not np.isfinite(subset["y_pred"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Single candidate {horizon}/d={daily_patch}/w={weekly_patch} contains non-finite y_true or y_pred.")
    mse = float(np.mean((subset["y_pred"].to_numpy(dtype=np.float64) - subset["y_true"].to_numpy(dtype=np.float64)) ** 2))
    if summary_row is not None:
        stored = float(summary_row["validation_mse"])
        if not np.isclose(mse, stored, rtol=VALIDATION_MSE_RTOL, atol=VALIDATION_MSE_ATOL):
            raise RuntimeError(f"Single candidate {horizon}/d={daily_patch}/w={weekly_patch} validation MSE mismatch: reconstructed {mse}, stored {stored}.")
    return mse


def _validate_single_selection(selection_df: pd.DataFrame, summary_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    if len(selection_df) != EXPECTED_SINGLE_SELECTION_ROWS:
        raise RuntimeError(f"Expected Single selection rows = {EXPECTED_SINGLE_SELECTION_ROWS}, found {len(selection_df)}")
    required_selection_hash_columns = {"training_summary_sha256", "single_validation_predictions_sha256"}
    missing_hash_columns = required_selection_hash_columns - set(selection_df.columns)
    if missing_hash_columns:
        raise RuntimeError(f"Single selection missing required hash columns: {sorted(missing_hash_columns)}")
    if set(selection_df["horizon"].astype(str).tolist()) != set(HORIZONS):
        raise RuntimeError("Single selection must contain exactly one row for each horizon.")
    if val_df.empty:
        raise RuntimeError("Single validation predictions are empty, so selection cannot be validated.")

    expected_training_summary_sha = _sha256(EXPERIMENT_DIR / "experiment3_training_summary.csv")
    expected_validation_predictions_sha = _sha256(EXPERIMENT_DIR / "experiment3_single_validation_predictions.csv")
    source_training_hashes = selection_df["training_summary_sha256"].dropna().unique()
    source_validation_hashes = selection_df["single_validation_predictions_sha256"].dropna().unique()
    if len(source_training_hashes) != 1 or source_training_hashes[0] != expected_training_summary_sha:
        raise RuntimeError("Single selection training_summary_sha256 does not match the training summary file hash.")
    if len(source_validation_hashes) != 1 or source_validation_hashes[0] != expected_validation_predictions_sha:
        raise RuntimeError("Single selection single_validation_predictions_sha256 does not match the single validation predictions file hash.")

    single_summary = summary_df[summary_df["mechanism"] == "single"].copy()
    expected_rows = []
    for horizon in HORIZONS:
        candidates = single_summary[single_summary["horizon"] == horizon].copy()
        candidates["daily_patch"] = candidates["daily_patch"].astype(int)
        candidates["weekly_patch"] = candidates["weekly_patch"].astype(int)
        candidates["validation_mse"] = candidates["validation_mse"].astype(float)
        rows = []
        for _, row in candidates.iterrows():
            val_mse = _validate_single_candidate_predictions(
                val_df,
                str(row["horizon"]),
                int(row["daily_patch"]),
                int(row["weekly_patch"]),
                summary_row=row,
            )
            rows.append({
                "horizon": str(row["horizon"]),
                "daily_patch": int(row["daily_patch"]),
                "weekly_patch": int(row["weekly_patch"]),
                "validation_mse": val_mse,
                "checkpoint_path": str(row["checkpoint_path"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
            })
        candidate_df = pd.DataFrame(rows)
        best = candidate_df.sort_values(["validation_mse", "daily_patch", "weekly_patch"], ascending=[True, True, True], kind="mergesort").iloc[0]
        expected_rows.append({
            "horizon": str(best["horizon"]),
            "selected_daily_patch": int(best["daily_patch"]),
            "selected_weekly_patch": int(best["weekly_patch"]),
            "validation_mse": float(best["validation_mse"]),
            "selection_metric": "validation_mse",
            "tie_break_rule": "validation_mse_then_daily_patch_ascending_then_weekly_patch_ascending",
            "selected_checkpoint_path": str(best["checkpoint_path"]),
            "selected_checkpoint_sha256": str(best["checkpoint_sha256"]),
        })

    expected_selection = pd.DataFrame(expected_rows).sort_values("horizon").reset_index(drop=True)
    actual_selection = selection_df[["horizon", "selected_daily_patch", "selected_weekly_patch", "validation_mse", "selection_metric", "tie_break_rule", "selected_checkpoint_path", "selected_checkpoint_sha256"]].sort_values("horizon").reset_index(drop=True)
    for field in ["horizon", "selection_metric", "tie_break_rule", "selected_checkpoint_path", "selected_checkpoint_sha256"]:
        if not (actual_selection[field].astype(str) == expected_selection[field].astype(str)).all():
            raise RuntimeError(f"Single selection mismatch for {field} against raw validation reconstruction.")
    if not np.allclose(actual_selection["validation_mse"].to_numpy(dtype=np.float64), expected_selection["validation_mse"].to_numpy(dtype=np.float64), rtol=VALIDATION_MSE_RTOL, atol=VALIDATION_MSE_ATOL):
        raise RuntimeError("Single selection validation_mse mismatch against raw validation reconstruction.")
    if not (actual_selection["selected_daily_patch"].astype(int) == expected_selection["selected_daily_patch"].astype(int)).all():
        raise RuntimeError("Single selection daily patch mismatch against raw validation reconstruction.")
    if not (actual_selection["selected_weekly_patch"].astype(int) == expected_selection["selected_weekly_patch"].astype(int)).all():
        raise RuntimeError("Single selection weekly patch mismatch against raw validation reconstruction.")


def _validate_test_artifacts(summary_df: pd.DataFrame, pred_df: pd.DataFrame, rank_df: pd.DataFrame, weight_df: pd.DataFrame) -> None:
    if len(summary_df) != EXPECTED_TEST_SUMMARY_ROWS:
        raise RuntimeError(f"Expected TEST summary rows = {EXPECTED_TEST_SUMMARY_ROWS}, found {len(summary_df)}")
    if len(pred_df) != EXPECTED_TEST_PREDICTION_ROWS:
        raise RuntimeError(f"Expected TEST prediction rows = {EXPECTED_TEST_PREDICTION_ROWS}, found {len(pred_df)}")
    if len(rank_df) != EXPECTED_TEST_RANK_IC_ROWS:
        raise RuntimeError(f"Expected TEST RankIC rows = {EXPECTED_TEST_RANK_IC_ROWS}, found {len(rank_df)}")
    if len(weight_df) != EXPECTED_TEST_WEIGHT_ROWS:
        raise RuntimeError(f"Expected TEST scale weight rows = {EXPECTED_TEST_WEIGHT_ROWS}, found {len(weight_df)}")

    required_summary = {"mechanism", "horizon", "seed", "test_mse", "test_mae", "test_corr", "mean_rank_ic", "direction_accuracy", "pred_std_true_std_ratio"}
    missing = required_summary - set(summary_df.columns)
    if missing:
        raise RuntimeError(f"TEST summary missing columns: {sorted(missing)}")

    if pred_df.duplicated(subset=["mechanism", "horizon", "ticker", "anchor_date"]).any():
        raise RuntimeError("TEST predictions contain duplicate keys.")
    if rank_df.duplicated(subset=["mechanism", "horizon", "anchor_date"]).any():
        raise RuntimeError("TEST RankIC contains duplicate keys.")

    if pred_df["ticker"].nunique() != 17:
        raise RuntimeError("TEST predictions must have exactly 17 tickers.")
    if pred_df["anchor_date"].nunique() != 391:
        raise RuntimeError("TEST predictions must have exactly 391 anchor dates per cell.")

    if not np.isfinite(pred_df["y_true"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("TEST y_true contains non-finite values.")
    if not np.isfinite(pred_df["y_pred"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("TEST y_pred contains non-finite values.")

    for horizon in HORIZONS:
        keys = {}
        for mechanism in ("single", "fixed", "static"):
            subset = pred_df[(pred_df["mechanism"] == mechanism) & (pred_df["horizon"] == horizon)][["ticker", "anchor_date", "y_true"]].copy()
            subset = subset.sort_values(["ticker", "anchor_date"]).reset_index(drop=True)
            keys[mechanism] = subset

        base = keys["single"]
        for other in ("fixed", "static"):
            other_df = keys[other]
            if not base[["ticker", "anchor_date"]].equals(other_df[["ticker", "anchor_date"]]):
                raise RuntimeError(f"Cross-mechanism TEST key mismatch for horizon={horizon}.")
            if not np.allclose(base["y_true"].to_numpy(dtype=np.float64), other_df["y_true"].to_numpy(dtype=np.float64), rtol=0.0, atol=np.finfo(np.float32).eps, equal_nan=False):
                raise RuntimeError(f"Cross-mechanism y_true mismatch for horizon={horizon} using float32-safe tolerance.")

    for horizon in HORIZONS:
        for mechanism in ("single", "fixed", "static"):
            subset = weight_df[(weight_df["mechanism"] == mechanism) & (weight_df["horizon"] == horizon)]
            if subset.empty:
                raise RuntimeError(f"Missing scale weights for {mechanism}/{horizon}.")
            daily_cols = [c for c in subset.columns if c.startswith("daily_w_")]
            weekly_cols = [c for c in subset.columns if c.startswith("weekly_w_")]
            if not np.allclose(subset[daily_cols].to_numpy(dtype=np.float64).sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"Daily scale weights do not sum to 1 for {mechanism}/{horizon}.")
            if not np.allclose(subset[weekly_cols].to_numpy(dtype=np.float64).sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"Weekly scale weights do not sum to 1 for {mechanism}/{horizon}.")
            if (subset[daily_cols].to_numpy(dtype=np.float64) < 0).any() or (subset[weekly_cols].to_numpy(dtype=np.float64) < 0).any():
                raise RuntimeError(f"Scale weights must be non-negative for {mechanism}/{horizon}.")

    single_weights = weight_df[weight_df["mechanism"] == "single"].copy()
    if len(single_weights) == 0:
        raise RuntimeError("Missing single scale weights.")
    single_daily = single_weights[[c for c in single_weights.columns if c.startswith("daily_w_")]].to_numpy(dtype=np.float64)
    single_weekly = single_weights[[c for c in single_weights.columns if c.startswith("weekly_w_")]].to_numpy(dtype=np.float64)
    if not np.isin(single_daily, [0.0, 1.0]).all() or not np.isin(single_weekly, [0.0, 1.0]).all():
        raise RuntimeError("Single mechanism must use one-hot scale weights.")
    if not np.allclose(single_daily.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Single daily weights must sum to 1.")
    if not np.allclose(single_weekly.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Single weekly weights must sum to 1.")

    fixed_weights = weight_df[weight_df["mechanism"] == "fixed"].copy()
    fixed_daily = fixed_weights[[c for c in fixed_weights.columns if c.startswith("daily_w_")]].to_numpy(dtype=np.float64)
    fixed_weekly = fixed_weights[[c for c in fixed_weights.columns if c.startswith("weekly_w_")]].to_numpy(dtype=np.float64)
    if not np.allclose(fixed_daily, 0.25, rtol=0.0, atol=1e-9):
        raise RuntimeError("Fixed mechanism must use exactly 0.25 daily weights.")
    if not np.allclose(fixed_weekly, 0.25, rtol=0.0, atol=1e-9):
        raise RuntimeError("Fixed mechanism must use exactly 0.25 weekly weights.")

    static_weights = weight_df[weight_df["mechanism"] == "static"].copy()
    daily_cols = [c for c in static_weights.columns if c.startswith("daily_w_")]
    weekly_cols = [c for c in static_weights.columns if c.startswith("weekly_w_")]
    for horizon in HORIZONS:
        subset = static_weights[static_weights["horizon"] == horizon]
        if subset.empty:
            raise RuntimeError(f"Static weights missing for horizon={horizon}.")
        daily_unique = subset[daily_cols].drop_duplicates().reset_index(drop=True)
        weekly_unique = subset[weekly_cols].drop_duplicates().reset_index(drop=True)
        if len(daily_unique) != 1 or len(weekly_unique) != 1:
            raise RuntimeError(f"Static weights must be sample-independent for horizon={horizon}.")


def _reconstruct_test_metrics(pred_df: pd.DataFrame, rank_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mechanism in ("single", "fixed", "static"):
        for horizon in HORIZONS:
            subset = pred_df[(pred_df["mechanism"] == mechanism) & (pred_df["horizon"] == horizon)].copy()
            if subset.empty:
                raise RuntimeError(f"Missing TEST predictions for {mechanism}/{horizon}.")
            subset = subset.sort_values(["ticker", "anchor_date"]).reset_index(drop=True)
            y_true = subset["y_true"].to_numpy(dtype=np.float32)
            y_pred = subset["y_pred"].to_numpy(dtype=np.float32)
            metrics = evaluate_predictions(
                y_true,
                y_pred,
                subset["ticker"].to_numpy(),
                subset["anchor_date"].to_numpy(),
            )
            rank_subset = rank_df[(rank_df["mechanism"] == mechanism) & (rank_df["horizon"] == horizon)].copy()
            valid_rank_ic = rank_subset["rank_ic"].dropna().to_numpy(dtype=np.float64)
            mean_rank_ic = float(np.mean(valid_rank_ic)) if len(valid_rank_ic) else float("nan")
            row = {
                "mechanism": mechanism,
                "horizon": horizon,
                "seed": 42,
                "test_mse": float(metrics["pooled_mse"]),
                "test_mae": float(metrics["pooled_mae"]),
                "test_corr": float(metrics["pooled_corr"]),
                "ticker_avg_mse": float(metrics["ticker_avg_mse"]),
                "ticker_avg_mae": float(metrics["ticker_avg_mae"]),
                "ticker_avg_corr": float(metrics["ticker_avg_corr"]),
                "direction_accuracy": float(metrics["pooled_da"]),
                "pred_std": float(metrics["pooled_pred_std"]),
                "true_std": float(metrics["pooled_true_std"]),
                "pred_std_true_std_ratio": float(metrics["pooled_pred_std_over_true_std"]),
                "mean_rank_ic": mean_rank_ic,
                "rank_ic_std": float(np.std(valid_rank_ic)) if len(valid_rank_ic) else float("nan"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _validate_reconstructed_metrics(saved_summary: pd.DataFrame, pred_df: pd.DataFrame, rank_df: pd.DataFrame) -> None:
    reconstructed = _reconstruct_test_metrics(pred_df, rank_df)
    merged = saved_summary[["mechanism", "horizon", "test_mse", "test_mae", "test_corr", "direction_accuracy", "pred_std", "true_std", "pred_std_true_std_ratio", "mean_rank_ic", "ticker_avg_mse", "ticker_avg_mae", "ticker_avg_corr"]].copy()
    rec_merge = reconstructed[["mechanism", "horizon", "test_mse", "test_mae", "test_corr", "direction_accuracy", "pred_std", "true_std", "pred_std_true_std_ratio", "mean_rank_ic", "ticker_avg_mse", "ticker_avg_mae", "ticker_avg_corr"]].copy()

    merged = merged.merge(rec_merge, on=["mechanism", "horizon"], suffixes=("_saved", "_reconstructed"))
    for field in ["test_mse", "test_mae", "test_corr", "direction_accuracy", "pred_std", "true_std", "pred_std_true_std_ratio", "mean_rank_ic", "ticker_avg_mse", "ticker_avg_mae", "ticker_avg_corr"]:
        saved_values = merged[f"{field}_saved"].to_numpy(dtype=np.float64)
        rec_values = merged[f"{field}_reconstructed"].to_numpy(dtype=np.float64)
        if not np.allclose(saved_values, rec_values, rtol=1e-8, atol=1e-8, equal_nan=True):
            raise RuntimeError(f"Reconstructed TEST metric mismatch for {field}.")

    for mechanism in ("single", "fixed", "static"):
        for horizon in HORIZONS:
            pred_subset = pred_df[(pred_df["mechanism"] == mechanism) & (pred_df["horizon"] == horizon)].copy().sort_values(["ticker", "anchor_date"]).reset_index(drop=True)
            if len(pred_subset) != 6647:
                raise RuntimeError(f"Cell {mechanism}/{horizon} must contain 6647 predictions for independent RankIC reconstruction.")
            y_true = pred_subset["y_true"].to_numpy(dtype=np.float32)
            y_pred = pred_subset["y_pred"].to_numpy(dtype=np.float32)
            reconstructed_rank = rank_ic_by_date(
                y_true,
                y_pred,
                pred_subset["ticker"].to_numpy(),
                pred_subset["anchor_date"].to_numpy(),
            )
            if len(reconstructed_rank) != 391:
                raise RuntimeError(f"Reconstructed RankIC for {mechanism}/{horizon} has {len(reconstructed_rank)} dates; expected 391.")
            saved_rank = rank_df[(rank_df["mechanism"] == mechanism) & (rank_df["horizon"] == horizon)].copy()
            if len(saved_rank) != 391:
                raise RuntimeError(f"Saved RankIC for {mechanism}/{horizon} has {len(saved_rank)} rows; expected 391.")
            if list(reconstructed_rank["anchor_date"].astype(str)) != list(saved_rank["anchor_date"].astype(str)):
                raise RuntimeError(f"RankIC anchor_date mismatch for {mechanism}/{horizon}.")
            if not np.allclose(reconstructed_rank["rank_ic"].to_numpy(dtype=np.float64), saved_rank["rank_ic"].to_numpy(dtype=np.float64), rtol=1e-8, atol=1e-8, equal_nan=True):
                raise RuntimeError(f"RankIC values mismatch for {mechanism}/{horizon}.")
            saved_mean = float(saved_summary[(saved_summary["mechanism"] == mechanism) & (saved_summary["horizon"] == horizon)]["mean_rank_ic"].iloc[0])
            reconstructed_mean = float(reconstructed_rank["rank_ic"].dropna().mean())
            if not np.isclose(saved_mean, reconstructed_mean, rtol=1e-8, atol=1e-8, equal_nan=True):
                raise RuntimeError(f"RankIC mean mismatch for {mechanism}/{horizon}: saved {saved_mean}, reconstructed {reconstructed_mean}.")


def _validate_frozen_adaptive() -> dict:
    summary_path = MAIN_V2_DIR / "panel_pathformer_summary_metrics.csv"
    pred_path = MAIN_V2_DIR / "panel_pathformer_test_predictions.csv"
    rank_path = MAIN_V2_DIR / "panel_pathformer_rank_ic_by_date.csv"
    router_path = MAIN_V2_DIR / "panel_pathformer_router_weights.csv"

    _require(summary_path, "Adaptive summary")
    _require(pred_path, "Adaptive predictions")
    _require(rank_path, "Adaptive RankIC")
    _require(router_path, "Adaptive router weights")

    for name, path, expected in [
        ("summary", summary_path, EXPECTED_ADAPTIVE_HASHES["summary"]),
        ("predictions", pred_path, EXPECTED_ADAPTIVE_HASHES["predictions"]),
        ("rank_ic", rank_path, EXPECTED_ADAPTIVE_HASHES["rank_ic"]),
        ("router", router_path, EXPECTED_ADAPTIVE_HASHES["router"]),
    ]:
        _check_hash(path, expected, f"Frozen Adaptive {name}")

    summary_df = _load_csv(summary_path, "Adaptive summary")
    pred_df = _load_csv(pred_path, "Adaptive predictions")
    rank_df = _load_csv(rank_path, "Adaptive RankIC")
    router_df = _load_csv(router_path, "Adaptive router")

    adaptive_summary = summary_df[(summary_df["frequency_setting"].astype(str) == "daily_weekly") & (summary_df["horizon"].astype(int).isin([HORIZON_MAP[h] for h in HORIZONS]))].copy()
    adaptive_pred = pred_df[(pred_df["frequency_setting"].astype(str) == "daily_weekly") & (pred_df["horizon"].astype(int).isin([HORIZON_MAP[h] for h in HORIZONS]))].copy()
    adaptive_rank = rank_df[(rank_df["frequency_setting"].astype(str) == "daily_weekly") & (rank_df["horizon"].astype(int).isin([HORIZON_MAP[h] for h in HORIZONS]))].copy()
    adaptive_router = router_df[(router_df["frequency_setting"].astype(str) == "daily_weekly") & (router_df["horizon"].astype(int).isin([HORIZON_MAP[h] for h in HORIZONS]))].copy()

    if len(adaptive_summary) != EXPECTED_ADAPTIVE_DW_SUMMARY_ROWS:
        raise RuntimeError(
            f"Adaptive summary must contain exactly "
            f"{EXPECTED_ADAPTIVE_DW_SUMMARY_ROWS} daily_weekly rows, "
            f"found {len(adaptive_summary)}"
        )
    if len(adaptive_pred) != EXPECTED_ADAPTIVE_DW_PREDICTION_ROWS:
        raise RuntimeError(
            f"Adaptive predictions must contain exactly "
            f"{EXPECTED_ADAPTIVE_DW_PREDICTION_ROWS} daily_weekly rows, "
            f"found {len(adaptive_pred)}"
        )
    if len(adaptive_rank) != EXPECTED_ADAPTIVE_DW_RANK_IC_ROWS:
        raise RuntimeError(
            f"Adaptive RankIC must contain exactly "
            f"{EXPECTED_ADAPTIVE_DW_RANK_IC_ROWS} daily_weekly rows, "
            f"found {len(adaptive_rank)}"
        )
    if len(adaptive_router) != EXPECTED_ADAPTIVE_DW_ROUTER_ROWS:
        raise RuntimeError(
            f"Adaptive router weights must contain exactly "
            f"{EXPECTED_ADAPTIVE_DW_ROUTER_ROWS} daily_weekly rows, "
            f"found {len(adaptive_router)}"
        )

    return {
        "summary": adaptive_summary,
        "predictions": adaptive_pred,
        "rank_ic": adaptive_rank,
        "router": adaptive_router,
    }


def _fail_if_final_outputs_exist() -> None:
    for path in FINAL_OUTPUTS:
        _require_not_exist(path, path.name)


def _mechanism_comparison_rows(test_summary_df: pd.DataFrame, adaptive_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    current = test_summary_df.copy()
    adaptive_map = {
        INT_TO_HORIZON_LABEL[int(row["horizon"])]: row
        for _, row in adaptive_df.iterrows()
    }

    for mechanism in ["single", "fixed", "static", "adaptive"]:
        for horizon in HORIZONS:
            if mechanism == "single":
                row = current[(current["mechanism"] == mechanism) & (current["horizon"] == horizon)].iloc[0]
                phase = "reference"
                baseline = ""
                delta_mse = np.nan
                comparison_label = "Single reference"
            elif mechanism == "adaptive":
                row = adaptive_map.get(horizon)
                if row is None:
                    raise RuntimeError(f"Missing Adaptive summary row for {horizon}.")
                phase = "adaptive_vs_static"
                baseline = "static"
                baseline_row = current[(current["mechanism"] == "static") & (current["horizon"] == horizon)].iloc[0]
                delta_mse = float(row["test_mse"]) - float(baseline_row["test_mse"])
                comparison_label = "Adaptive vs Static"
            else:
                row = current[(current["mechanism"] == mechanism) & (current["horizon"] == horizon)].iloc[0]
                if mechanism == "static":
                    phase = "static_vs_fixed"
                    baseline = "fixed"
                    baseline_row = current[(current["mechanism"] == "fixed") & (current["horizon"] == horizon)].iloc[0]
                    comparison_label = "Static vs Fixed"
                else:
                    phase = "fixed_vs_single"
                    baseline = "single"
                    baseline_row = current[(current["mechanism"] == "single") & (current["horizon"] == horizon)].iloc[0]
                    comparison_label = "Fixed vs Single"
                delta_mse = float(row["test_mse"]) - float(baseline_row["test_mse"])

            records.append({
                "comparison_group": phase,
                "comparison_label": comparison_label,
                "mechanism": mechanism,
                "horizon": horizon,
                "selected_daily_patch": row.get("selected_daily_patch", np.nan),
                "selected_weekly_patch": row.get("selected_weekly_patch", np.nan),
                "test_mse": float(row["test_mse"]),
                "test_mae": float(row["test_mae"]),
                "test_corr": float(row["test_corr"]),
                "mean_rank_ic": float(row["mean_rank_ic"]),
                "direction_accuracy": float(row["direction_accuracy"]),
                "pred_std_true_std_ratio": float(row["pred_std_true_std_ratio"]),
                "delta_mse_vs_baseline": float(delta_mse) if not np.isnan(delta_mse) else np.nan,
                "baseline_mechanism": baseline,
            })

    comparison_df = pd.DataFrame(records)
    if len(comparison_df) != 12:
        raise RuntimeError(f"Mechanism comparison must contain exactly 12 rows, found {len(comparison_df)}")
    comparison_df = comparison_df.sort_values(["comparison_group", "mechanism", "horizon"], ascending=[True, True, True], kind="mergesort").reset_index(drop=True)
    return comparison_df


def _build_manifest_row(
    hashes: dict,
    adaptive: dict,
    training_summary_df: pd.DataFrame,
    training_history_df: pd.DataFrame,
    validation_pred_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    test_summary_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    test_rank_ic_df: pd.DataFrame,
    test_weight_df: pd.DataFrame,
    finalizer_sha: str,
) -> dict:
    return {
        "run_family": "EXPERIMENT3_PATHFORMER_ABLATION",
        "seed": 42,
        "frequency_configuration": "daily_weekly",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "execution_backend": "CPU",
        "selection_metric": "validation_mse",
        "single_selection_uses_test": False,
        "training_cells": 54,
        "new_test_cells": 9,
        "final_comparison_rows": 12,
        "training_provenance_sha256": _sha256(EXPERIMENT_DIR / "experiment3_training_provenance.csv"),
        "training_summary_sha256": _sha256(EXPERIMENT_DIR / "experiment3_training_summary.csv"),
        "training_history_sha256": _sha256(EXPERIMENT_DIR / "experiment3_training_history.csv"),
        "single_validation_predictions_sha256": _sha256(EXPERIMENT_DIR / "experiment3_single_validation_predictions.csv"),
        "single_selection_sha256": _sha256(EXPERIMENT_DIR / "experiment3_single_selection.csv"),
        "test_summary_sha256": _sha256(EXPERIMENT_DIR / "experiment3_test_summary_metrics.csv"),
        "test_predictions_sha256": _sha256(EXPERIMENT_DIR / "experiment3_test_predictions.csv"),
        "test_rank_ic_sha256": _sha256(EXPERIMENT_DIR / "experiment3_rank_ic_by_date.csv"),
        "scale_weights_sha256": _sha256(EXPERIMENT_DIR / "experiment3_scale_weights.csv"),
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "experiment3_runner_sha256": hashes["experiment3_runner_sha256"],
        "frozen_main_v2_pathformer_sha256": hashes["main_v2_pathformer_script_sha256"],
        "panel_common_sha256": hashes["panel_common_sha256"],
        "finalizer_sha256": finalizer_sha,
        "training_summary_rows": int(len(training_summary_df)),
        "single_training_cells": int((training_summary_df["mechanism"] == "single").sum()),
        "fixed_training_cells": int((training_summary_df["mechanism"] == "fixed").sum()),
        "static_training_cells": int((training_summary_df["mechanism"] == "static").sum()),
        "single_validation_predictions_rows": int(len(validation_pred_df)),
        "single_selection_rows": int(len(selection_df)),
        "training_history_rows": int(len(training_history_df)),
        "checkpoints": 54,
        "test_summary_rows": int(len(test_summary_df)),
        "test_prediction_rows": int(len(test_pred_df)),
        "test_rank_ic_rows": int(len(test_rank_ic_df)),
        "test_scale_weight_rows": int(len(test_weight_df)),
        "adaptive_summary_sha256": EXPECTED_ADAPTIVE_HASHES["summary"],
        "adaptive_predictions_sha256": EXPECTED_ADAPTIVE_HASHES["predictions"],
        "adaptive_rank_ic_sha256": EXPECTED_ADAPTIVE_HASHES["rank_ic"],
        "adaptive_router_sha256": EXPECTED_ADAPTIVE_HASHES["router"],
        "adaptive_source_path": str(MAIN_V2_DIR.relative_to(ROOT)),
        "comparison_output": "experiment3_mechanism_comparison.csv",
        "final_summary_output": "experiment3_final_summary.txt",
        "notes": "no training, no evaluation, adaptive evidence frozen and hash-verified only",
    }


def _build_final_summary_text(comparison_df: pd.DataFrame, adaptive_summary: pd.DataFrame) -> str:
    lines = []
    lines.append("EXPERIMENT 3 PATHFORMER MECHANISM ABLATION")
    lines.append("Static finalizer audit only. No training or evaluation executed.")
    lines.append("")
    lines.append("Validated evidence:")
    lines.append(f"- training summary rows = {EXPECTED_TRAINING_SUMMARY_ROWS}")
    lines.append(f"- Single cells = {EXPECTED_SINGLE_CELLS}")
    lines.append(f"- Fixed cells = {EXPECTED_FIXED_CELLS}")
    lines.append(f"- Static cells = {EXPECTED_STATIC_CELLS}")
    lines.append(f"- Single validation predictions = {EXPECTED_SINGLE_VALIDATION_ROWS}")
    lines.append(f"- Single selection rows = {EXPECTED_SINGLE_SELECTION_ROWS}")
    lines.append(f"- checkpoints = {EXPECTED_CHECKPOINT_COUNT}")
    lines.append(f"- TEST summary rows = {EXPECTED_TEST_SUMMARY_ROWS}")
    lines.append(f"- TEST prediction rows = {EXPECTED_TEST_PREDICTION_ROWS}")
    lines.append(f"- TEST RankIC rows = {EXPECTED_TEST_RANK_IC_ROWS}")
    lines.append(f"- scale weights rows = {EXPECTED_TEST_WEIGHT_ROWS}")
    lines.append("")
    lines.append("Hierarchy used for comparison:")
    lines.append("1. Adaptive vs Static")
    lines.append("2. Static vs Fixed")
    lines.append("3. Fixed vs Single")
    lines.append("")
    lines.append("Interpretation guardrails:")
    lines.append("- No statistical-significance claim.")
    lines.append("- No profitability claim.")
    lines.append("- No causality claim.")
    lines.append("- No multi-seed robustness claim.")
    lines.append("- No universal Adaptive superiority claim.")
    lines.append("")
    adaptive_mse_by_horizon = {}
    static_mse_by_horizon = {}
    for h in HORIZONS:
        adaptive_row = adaptive_summary[adaptive_summary["horizon"].astype(int) == HORIZON_MAP[h]].iloc[0]
        adaptive_mse_by_horizon[h] = float(adaptive_row["test_mse"])
        static_row = comparison_df[(comparison_df["mechanism"] == "static") & (comparison_df["horizon"] == h)].iloc[0]
        static_mse_by_horizon[h] = float(static_row["test_mse"])
    if all(adaptive_mse_by_horizon[h] < static_mse_by_horizon[h] for h in HORIZONS):
        lines.append("Adaptive beats Static across all three horizons but robustness/significance are not established.")
    else:
        lines.append("Seed-42 evidence shows no clear incremental benefit from sample-dependent adaptive routing over globally learned static scale aggregation.")
    lines.append("")
    lines.append("Adaptive evidence is frozen, hash-verified, and read only from the MAIN V2 PathFormer artifacts.")
    lines.append("The sample-dependent router weights are never copied into Experiment 3 raw evidence.")
    return "\n".join(lines) + "\n"


def _write_final_outputs(comparison_df: pd.DataFrame, manifest_df: pd.DataFrame, final_summary_text: str) -> None:
    comparison_df.to_csv(FINAL_OUTPUTS[0], index=False)
    manifest_df.to_csv(FINAL_OUTPUTS[1], index=False)
    FINAL_OUTPUTS[2].write_text(final_summary_text, encoding="utf-8")

    if not FINAL_OUTPUTS[0].exists() or not FINAL_OUTPUTS[1].exists() or not FINAL_OUTPUTS[2].exists():
        raise RuntimeError("One or more required final outputs were not written successfully.")


def _validate_training_history_and_checkpoints(summary_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    training_history_path = EXPERIMENT_DIR / "experiment3_training_history.csv"
    _require(training_history_path, "Experiment 3 training history")
    if len(summary_df) != EXPECTED_TRAINING_SUMMARY_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TRAINING_SUMMARY_ROWS} training summary rows, found {len(summary_df)}")
    if len(history_df) == 0:
        raise RuntimeError("Experiment 3 training history is empty.")
    for _, row in summary_df.iterrows():
        checkpoint_path = ROOT / str(row["checkpoint_path"])
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint missing for {row['mechanism']}/{row['horizon']}: {checkpoint_path}")
        if _sha256(checkpoint_path) != str(row["checkpoint_sha256"]):
            raise RuntimeError(f"Checkpoint SHA mismatch for {row['mechanism']}/{row['horizon']}: {checkpoint_path}")
    if len(history_df[history_df["mechanism"].astype(str) == "single"]) == 0:
        raise RuntimeError("Training history does not include single-cell training records.")
    if len(summary_df[summary_df["mechanism"].astype(str) == "single"]) != EXPECTED_SINGLE_CELLS:
        raise RuntimeError("Training summary single cells does not match the expected 48 cells.")


def main() -> None:
    _fail_if_final_outputs_exist()

    hashes = _validate_provenance_and_hashes()

    training_summary_path = EXPERIMENT_DIR / "experiment3_training_summary.csv"
    training_history_path = EXPERIMENT_DIR / "experiment3_training_history.csv"
    val_pred_path = EXPERIMENT_DIR / "experiment3_single_validation_predictions.csv"
    selection_path = EXPERIMENT_DIR / "experiment3_single_selection.csv"
    test_summary_path = EXPERIMENT_DIR / "experiment3_test_summary_metrics.csv"
    test_prediction_path = EXPERIMENT_DIR / "experiment3_test_predictions.csv"
    test_rank_ic_path = EXPERIMENT_DIR / "experiment3_rank_ic_by_date.csv"
    test_scale_weight_path = EXPERIMENT_DIR / "experiment3_scale_weights.csv"

    for path in [training_summary_path, training_history_path, val_pred_path, selection_path, test_summary_path, test_prediction_path, test_rank_ic_path, test_scale_weight_path]:
        _require(path, path.name)

    training_summary = _load_csv(training_summary_path, "Experiment 3 training summary")
    training_history = _load_csv(training_history_path, "Experiment 3 training history")
    val_pred = _load_csv(val_pred_path, "Experiment 3 validation predictions")
    selection = _load_csv(selection_path, "Experiment 3 single selection")
    test_summary = _load_csv(test_summary_path, "Experiment 3 TEST summary")
    test_pred = _load_csv(test_prediction_path, "Experiment 3 TEST predictions")
    test_rank_ic = _load_csv(test_rank_ic_path, "Experiment 3 TEST RankIC")
    test_weights = _load_csv(test_scale_weight_path, "Experiment 3 TEST scale weights")

    _validate_training_summary(training_summary)
    _validate_training_history_and_checkpoints(training_summary, training_history)
    _validate_single_validation_predictions(val_pred)
    _validate_single_selection(selection, training_summary, val_pred)
    _validate_test_artifacts(test_summary, test_pred, test_rank_ic, test_weights)
    _validate_reconstructed_metrics(test_summary, test_pred, test_rank_ic)

    adaptive = _validate_frozen_adaptive()

    for horizon in HORIZONS:
        frames = {}
        for mechanism in ("single", "fixed", "static"):
            df = test_pred[(test_pred["mechanism"] == mechanism) & (test_pred["horizon"] == horizon)][["ticker", "anchor_date", "y_true"]].copy()
            df = df.sort_values(["ticker", "anchor_date"]).reset_index(drop=True)
            frames[mechanism] = df
        for mechanism in ("fixed", "static"):
            if not frames["single"]["ticker"].tolist() == frames[mechanism]["ticker"].tolist():
                raise RuntimeError(f"TEST ticker mismatch across mechanisms for horizon={horizon}.")
            if not frames["single"]["anchor_date"].tolist() == frames[mechanism]["anchor_date"].tolist():
                raise RuntimeError(f"TEST anchor_date mismatch across mechanisms for horizon={horizon}.")
            if not np.allclose(
                frames["single"]["y_true"].to_numpy(dtype=np.float64),
                frames[mechanism]["y_true"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=np.finfo(np.float32).eps,
                equal_nan=False,
            ):
                raise RuntimeError(f"Persisted y_true identity mismatch across mechanisms for horizon={horizon}.")

        adaptive_subset = adaptive["predictions"][
            (adaptive["predictions"]["horizon"].astype(int) == HORIZON_MAP[horizon])
        ][["ticker", "anchor_date", "y_true"]].copy().sort_values(["ticker", "anchor_date"]).reset_index(drop=True)
        if not frames["single"]["ticker"].tolist() == adaptive_subset["ticker"].tolist():
            raise RuntimeError(f"Adaptive TEST keys differ from Experiment 3 keys for horizon={horizon}.")
        if not frames["single"]["anchor_date"].tolist() == adaptive_subset["anchor_date"].tolist():
            raise RuntimeError(f"Adaptive TEST dates differ from Experiment 3 dates for horizon={horizon}.")
        if not np.allclose(
            frames["single"]["y_true"].to_numpy(dtype=np.float64),
            adaptive_subset["y_true"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=np.finfo(np.float32).eps,
            equal_nan=False,
        ):
            raise RuntimeError(f"Adaptive y_true identity mismatch for horizon={horizon}.")

    adaptive_summary = adaptive["summary"].sort_values("horizon").reset_index(drop=True)
    comparison_df = _mechanism_comparison_rows(test_summary, adaptive_summary)

    finalizer_sha = _sha256(Path(__file__))
    manifest_row = _build_manifest_row(
        hashes,
        adaptive,
        training_summary,
        training_history,
        val_pred,
        selection,
        test_summary,
        test_pred,
        test_rank_ic,
        test_weights,
        finalizer_sha,
    )
    manifest_df = pd.DataFrame([manifest_row])
    final_summary_text = _build_final_summary_text(comparison_df, adaptive_summary)
    _write_final_outputs(comparison_df, manifest_df, final_summary_text)

    print("EXPERIMENT 3 PATHFORMER ABLATION FINALIZATION PASSED")


if __name__ == "__main__":
    main()
