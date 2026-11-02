"""Standalone immutable finalizer for the frozen Experiment 4 PathFormer multi-seed audit.

This script is read-only and non-executing in the sense of model lifecycle:
- it validates existing frozen raw evidence and seed-42 external evidence;
- it reconstructs metrics from persisted TEST predictions;
- it combines new seed evidence with frozen external seed-42 evidence;
- it writes only the required descriptive final artifacts under
  dataset/audit/experiment4_multiseed/.

It never trains, optimizes, evaluates on fresh predictions, reloads checkpoints,
reconstructs new TEST outputs, or performs statistical inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import evaluate_predictions, rank_ic_by_date

EXPERIMENT_DIR = ROOT / "dataset" / "audit" / "experiment4_multiseed"
MAIN_V2_DIR = ROOT / "dataset" / "audit" / "main_v2" / "pathformer"
EXPERIMENT3_DIR = ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation"
DATASET_MANIFEST_PATH = ROOT / "dataset" / "multiscale_dataset" / "panel" / "panel_build_manifest.csv"

EXPERIMENT4_RUNNER_PATH = ROOT / "scripts" / "python" / "panel_pathformer_experiment4_multiseed.py"
EXPERIMENT4_PROTOCOL_PATH = ROOT / "docs" / "experiment4_multiseed_protocol.md"
PANEL_COMMON_PATH = ROOT / "scripts" / "python" / "panel_common.py"
MAIN_V2_PATHFORMER_SCRIPT_PATH = ROOT / "scripts" / "python" / "panel_pathformer_main_v2.py"
EXPERIMENT3_RUNNER_PATH = ROOT / "scripts" / "python" / "panel_pathformer_experiment3.py"

FINAL_OUTPUTS = (
    EXPERIMENT_DIR / "experiment4_combined_cell_metrics.csv",
    EXPERIMENT_DIR / "experiment4_seed_pair_deltas.csv",
    EXPERIMENT_DIR / "experiment4_robustness_summary.csv",
    EXPERIMENT_DIR / "experiment4_final_summary.txt",
    EXPERIMENT_DIR / "experiment4_run_manifest.csv",
)

NEW_SEEDS = (0, 1, 21, 3407)
FROZEN_EXTERNAL_SEED = 42
ALL_SEEDS = (0, 1, 21, 42, 3407)
MECHANISMS = ("adaptive", "static")
HORIZONS = ("5d", "10d", "20d")
FREQUENCY_CONFIGURATION = "daily_weekly"

EXPECTED_RAW_HASHES = {
    "experiment4_training_provenance.csv": "3f7bc10edccf1c41f5084552178f017c4c1cb5b02c82e12e488c9b38aae0a7a8",
    "experiment4_training_summary.csv": "caaf3bdf68f579a7e677a075362c9508c5fd93ccfe28d719d0e12cdf9881672c",
    "experiment4_training_history.csv": "f02d36ef52877cb91f64a7fa75e0c7983e2365ea309b5f3f8a74232e26b3feff",
    "experiment4_test_attempt_ledger.csv": "89ae2fd232fe7e8cbce6fee787ba2c491d95d218bf1241963a5fca55c70836ba",
    "experiment4_test_summary_metrics.csv": "a93cdafe238e32c9e3973e40c6f267e52e1a21a77cdad79970b2982a1e687831",
    "experiment4_test_predictions.csv": "6b5f5ea94f1bb5f25ebd5ed09d7e4e7eb609945f97f953e7fb95116e2e866e8a",
    "experiment4_rank_ic_by_date.csv": "646b2a7cc5a0acd37bd6845a46a28b1260d1e25c4be0adabfa34b76241de3440",
    "experiment4_scale_weights.csv": "ba1fd9280341dcf68d65ffe7c52675a69d77f1c214e5458cbb17e6ffc6fdb887",
    "panel_pathformer_experiment4_multiseed.py": "2447d0c5bea09226ebf30be3105e2eeeb0c83806f0d2a95c6e3e1ba01fa2bedf",
    "experiment4_multiseed_protocol.md": "8b10e7f68bab9b9555fd080546b56ecc650b5b09cc8cd394e41ed623992e4c6b",
}

EXPECTED_DEPENDENCY_HASHES = {
    "dataset_manifest": "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c",
    "main_v2_pathformer_script": "5f05ccd677359a899ca91c3baeeda3ecb6aae1a63ff75593cd31809ae27e6f77",
    "experiment3_runner": "6f3c1f804c9610fef6f29453cc9a49b121f927fbec90e3482e2744a924aa0142",
    "panel_common": "ccaba201b910b66fb5c3272bcc3aa18e04d12c098bb26100a73a3717ec6d2834",
}

EXPECTED_SEED42_HASHES = {
    "adaptive_summary": "90e987cd87d9bdbc82c6bf587627863027878a6d170f7a6cca80569ee39de5e3",
    "adaptive_predictions": "a05955a561da1b8880f1bb0878a6576852c087cc45707546d446ce96b0fabdaf",
    "adaptive_rank_ic": "52b3f3475dc4db74523cfe99a09f71f2b33bd14c58b458ba2e62564ca7283298",
    "adaptive_router_weights": "bcf06150f21d35d9ec10132d248d088cc758c9782771efdf9b68849531c9814e",
    "static_summary": "0ada49805bf476b873e9ec2d77ea72b29ae67ecede7b342fd0dd49e9df85005b",
    "static_predictions": "95d1b06cc3ce0209e99833c29a1c3672a6d7a1a9d46d32f16ef1e48fa9846515",
    "static_rank_ic": "5e9dd5cb1da838f9f984ec424441bd1bd30663f6f70a971b55d7a6404e68d43e",
    "static_scale_weights": "219e908f0c3f75bd9b38d7fe466242a15a64d8c41b1f235ae75c91df5f6eb248",
}

EXPECTED_RAW_ROW_COUNTS = {
    "training_provenance": 1,
    "training_summary": 24,
    "training_history": 278,
    "test_attempt_ledger": 24,
    "test_summary": 24,
    "test_predictions": 159528,
    "rank_ic_by_date": 9384,
    "scale_weights": 159528,
    "checkpoints": 24,
}

EXPECTED_NEW_ROW_COUNTS = EXPECTED_RAW_ROW_COUNTS.copy()

EXPECTED_SEED42_COUNTS = {
    "adaptive_summary": 3,
    "adaptive_predictions": 19941,
    "adaptive_rank_ic": 1173,
    "adaptive_router_weights": 19941,
    "static_summary": 3,
    "static_predictions": 19941,
    "static_rank_ic": 1173,
    "static_scale_weights": 19941,
}

EXPECTED_NUMERIC_TOL = {"rtol": 1e-8, "atol": 1e-8}
Y_TRUE_RTOL = 0.0
Y_TRUE_ATOL = np.finfo(np.float32).eps

# Canonical numeric-horizon -> horizon_label mapping used to align the MAIN V2
# adaptive seed42 evidence (numeric `horizon`) with the string-labelled horizons
# used everywhere else ("5d", "10d", "20d").
HORIZON_LABEL_MAP = {5: "5d", 10: "10d", 20: "20d"}

# Explicit stored-column -> reconstructed-key mapping for the primary summary
# metrics. Every entry must exist as a column in the stored summary row and
# must have a corresponding reconstructed key; nothing here may be skipped.
STORED_TO_RECONSTRUCTED_METRIC_MAP = {
    "test_mse": "pooled_mse",
    "test_mae": "pooled_mae",
    "test_corr": "pooled_corr",
    "ticker_avg_mse": "ticker_avg_mse",
    "ticker_avg_mae": "ticker_avg_mae",
    "ticker_avg_corr": "ticker_avg_corr",
    "direction_accuracy": "direction_accuracy",
    "pred_std": "pred_std",
    "true_std": "true_std",
    "pred_std_true_std_ratio": "pred_std_true_std_ratio",
    "mean_rank_ic": "mean_rank_ic",
    "rank_ic_std": "rank_ic_std",
}

# Combined-cell-metrics columns that additionally get a mechanism-specific
# five-seed mean/SD in the robustness summary (mse is handled separately to
# preserve the pre-existing adaptive_mean_mse/static_mean_mse naming).
ADDITIONAL_ROBUSTNESS_METRICS = (
    "pooled_mae",
    "pooled_corr",
    "ticker_avg_mse",
    "ticker_avg_mae",
    "ticker_avg_corr",
    "direction_accuracy",
    "pred_std",
    "true_std",
    "pred_std_true_std_ratio",
    "mean_rank_ic",
    "rank_ic_std",
)


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


def _check_hash(path: Path, expected_hash: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected_hash:
        raise RuntimeError(f"{label} hash mismatch: expected {expected_hash}, got {actual}")
    return actual


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    _require(path, label)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{label} is empty: {path}")
    return df


def _safe_mean(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def _safe_std(series: pd.Series, ddof: int = 1) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    if len(valid) <= ddof:
        return float("nan")
    return float(valid.std(ddof=ddof))


def _assert_no_42_in_new_evidence(summary_df: pd.DataFrame) -> None:
    values = pd.to_numeric(summary_df["seed"], errors="coerce").dropna().astype(int).tolist()
    if 42 in values:
        raise RuntimeError("New Experiment 4 raw evidence contains seed42; this is forbidden.")


def _validate_raw_experiment4_files() -> dict:
    raw_files = {
        "experiment4_training_provenance.csv": EXPERIMENT_DIR / "experiment4_training_provenance.csv",
        "experiment4_training_summary.csv": EXPERIMENT_DIR / "experiment4_training_summary.csv",
        "experiment4_training_history.csv": EXPERIMENT_DIR / "experiment4_training_history.csv",
        "experiment4_test_attempt_ledger.csv": EXPERIMENT_DIR / "experiment4_test_attempt_ledger.csv",
        "experiment4_test_summary_metrics.csv": EXPERIMENT_DIR / "experiment4_test_summary_metrics.csv",
        "experiment4_test_predictions.csv": EXPERIMENT_DIR / "experiment4_test_predictions.csv",
        "experiment4_rank_ic_by_date.csv": EXPERIMENT_DIR / "experiment4_rank_ic_by_date.csv",
        "experiment4_scale_weights.csv": EXPERIMENT_DIR / "experiment4_scale_weights.csv",
        "panel_pathformer_experiment4_multiseed.py": EXPERIMENT4_RUNNER_PATH,
        "experiment4_multiseed_protocol.md": EXPERIMENT4_PROTOCOL_PATH,
    }

    for name, path in raw_files.items():
        _require(path, name)
        if name in EXPECTED_RAW_HASHES:
            _check_hash(path, EXPECTED_RAW_HASHES[name], name)

    provenance_df = _load_csv(raw_files["experiment4_training_provenance.csv"], "Experiment 4 training provenance")
    if len(provenance_df) != EXPECTED_RAW_ROW_COUNTS["training_provenance"]:
        raise RuntimeError(
            f"Expected Experiment 4 training provenance rows = {EXPECTED_RAW_ROW_COUNTS['training_provenance']}; "
            f"found {len(provenance_df)}."
        )

    provenance = provenance_df.iloc[0].to_dict()
    required_fields = {
        "run_family": "EXPERIMENT4_PATHFORMER_MULTISEED",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "frequency_configuration": "daily_weekly",
        "mechanisms": "adaptive|static",
        "new_seeds": "0|1|21|3407",
        "all_protocol_seeds": "0|1|21|42|3407",
        "horizons": "5d|10d|20d",
        "execution_backend": "CPU",
        "seed42_policy": "FROZEN_EXTERNAL_REUSE_NOT_RETRAINED",
        "train_loader_shuffle": True,
        "train_loader_explicit_generator": False,
        "rng_seed_set_before_loader_and_model_construction": True,
        "model_constructed_before_train_loader_iteration": True,
        "seed_affects": "model_initialization|dropout|global_rng_training_shuffle",
        "paired_seed_is_common_random_number_match": False,
    }
    for field_name, expected in required_fields.items():
        if field_name not in provenance:
            raise RuntimeError(f"Experiment 4 provenance missing required field: {field_name}")
        if str(provenance[field_name]) != str(expected):
            raise RuntimeError(
                f"Experiment 4 provenance mismatch for {field_name}: expected {expected}, got {provenance[field_name]}"
            )

    summary_df = _load_csv(raw_files["experiment4_training_summary.csv"], "Experiment 4 training summary")
    if len(summary_df) != EXPECTED_RAW_ROW_COUNTS["training_summary"]:
        raise RuntimeError(
            f"Expected Experiment 4 training summary rows = {EXPECTED_RAW_ROW_COUNTS['training_summary']}; "
            f"found {len(summary_df)}."
        )
    _assert_no_42_in_new_evidence(summary_df)

    if summary_df["mechanism"].nunique() != 2 or set(summary_df["mechanism"]) != {"adaptive", "static"}:
        raise RuntimeError("Experiment 4 training summary must contain exactly adaptive and static mechanisms.")

    if summary_df["horizon"].nunique() != 3 or set(summary_df["horizon"]) != {"5d", "10d", "20d"}:
        raise RuntimeError("Experiment 4 training summary must contain horizons 5d, 10d, 20d.")

    attempt_ledger = _load_csv(raw_files["experiment4_test_attempt_ledger.csv"], "Experiment 4 test attempt ledger")
    if len(attempt_ledger) != EXPECTED_RAW_ROW_COUNTS["test_attempt_ledger"]:
        raise RuntimeError(
            f"Expected Experiment 4 test attempt ledger rows = {EXPECTED_RAW_ROW_COUNTS['test_attempt_ledger']}; "
            f"found {len(attempt_ledger)}."
        )

    test_summary_df = _load_csv(raw_files["experiment4_test_summary_metrics.csv"], "Experiment 4 test summary metrics")
    if len(test_summary_df) != EXPECTED_RAW_ROW_COUNTS["test_summary"]:
        raise RuntimeError(
            f"Expected Experiment 4 test summary metrics rows = {EXPECTED_RAW_ROW_COUNTS['test_summary']}; "
            f"found {len(test_summary_df)}."
        )

    predictions_df = _load_csv(raw_files["experiment4_test_predictions.csv"], "Experiment 4 test predictions")
    if len(predictions_df) != EXPECTED_RAW_ROW_COUNTS["test_predictions"]:
        raise RuntimeError(
            f"Expected Experiment 4 test prediction rows = {EXPECTED_RAW_ROW_COUNTS['test_predictions']}; "
            f"found {len(predictions_df)}."
        )

    rank_ic_df = _load_csv(raw_files["experiment4_rank_ic_by_date.csv"], "Experiment 4 RankIC evidence")
    if len(rank_ic_df) != EXPECTED_RAW_ROW_COUNTS["rank_ic_by_date"]:
        raise RuntimeError(
            f"Expected Experiment 4 RankIC rows = {EXPECTED_RAW_ROW_COUNTS['rank_ic_by_date']}; "
            f"found {len(rank_ic_df)}."
        )

    scale_weights_df = _load_csv(raw_files["experiment4_scale_weights.csv"], "Experiment 4 scale weights")
    if len(scale_weights_df) != EXPECTED_RAW_ROW_COUNTS["scale_weights"]:
        raise RuntimeError(
            f"Expected Experiment 4 scale weight rows = {EXPECTED_RAW_ROW_COUNTS['scale_weights']}; "
            f"found {len(scale_weights_df)}."
        )

    training_history_df = _load_csv(raw_files["experiment4_training_history.csv"], "Experiment 4 training history")
    if len(training_history_df) != EXPECTED_RAW_ROW_COUNTS["training_history"]:
        raise RuntimeError(
            f"Expected Experiment 4 training history rows = {EXPECTED_RAW_ROW_COUNTS['training_history']}; "
            f"found {len(training_history_df)}."
        )

    checkpoint_count = len([p for p in (EXPERIMENT_DIR / "checkpoints").glob("*.pt")])
    if checkpoint_count != EXPECTED_RAW_ROW_COUNTS["checkpoints"]:
        raise RuntimeError(
            f"Expected {EXPECTED_RAW_ROW_COUNTS['checkpoints']} checkpoint artifacts; "
            f"found {checkpoint_count}."
        )

    return {
        "provenance": provenance,
        "provenance_rows": len(provenance_df),
        "training_summary": summary_df,
        "training_history": training_history_df,
        "test_attempt_ledger": attempt_ledger,
        "test_summary": test_summary_df,
        "test_predictions": predictions_df,
        "rank_ic": rank_ic_df,
        "scale_weights": scale_weights_df,
        "checkpoint_count": checkpoint_count,
    }


def _validate_frozen_seed42_sources() -> dict:
    _require(DATASET_MANIFEST_PATH, "dataset manifest")
    _check_hash(DATASET_MANIFEST_PATH, EXPECTED_DEPENDENCY_HASHES["dataset_manifest"], "dataset manifest")

    _require(MAIN_V2_PATHFORMER_SCRIPT_PATH, "MAIN V2 PathFormer script")
    _check_hash(MAIN_V2_PATHFORMER_SCRIPT_PATH, EXPECTED_DEPENDENCY_HASHES["main_v2_pathformer_script"], "MAIN V2 PathFormer script")

    _require(EXPERIMENT3_RUNNER_PATH, "Experiment 3 runner script")
    _check_hash(EXPERIMENT3_RUNNER_PATH, EXPECTED_DEPENDENCY_HASHES["experiment3_runner"], "Experiment 3 runner script")

    _require(PANEL_COMMON_PATH, "panel_common.py")
    _check_hash(PANEL_COMMON_PATH, EXPECTED_DEPENDENCY_HASHES["panel_common"], "panel_common.py")

    adaptive_summary = _load_csv(MAIN_V2_DIR / "panel_pathformer_summary_metrics.csv", "Adaptive seed42 summary")
    adaptive_predictions = _load_csv(MAIN_V2_DIR / "panel_pathformer_test_predictions.csv", "Adaptive seed42 predictions")
    adaptive_rank_ic = _load_csv(MAIN_V2_DIR / "panel_pathformer_rank_ic_by_date.csv", "Adaptive seed42 RankIC")
    adaptive_router = _load_csv(MAIN_V2_DIR / "panel_pathformer_router_weights.csv", "Adaptive seed42 router weights")

    static_summary = _load_csv(EXPERIMENT3_DIR / "experiment3_test_summary_metrics.csv", "Static seed42 summary")
    static_predictions = _load_csv(EXPERIMENT3_DIR / "experiment3_test_predictions.csv", "Static seed42 predictions")
    static_rank_ic = _load_csv(EXPERIMENT3_DIR / "experiment3_rank_ic_by_date.csv", "Static seed42 RankIC")
    static_scale_weights = _load_csv(EXPERIMENT3_DIR / "experiment3_scale_weights.csv", "Static seed42 scale weights")

    # Hash-check the FULL frozen source files first, before any subset filtering.
    _check_hash(MAIN_V2_DIR / "panel_pathformer_summary_metrics.csv", EXPECTED_SEED42_HASHES["adaptive_summary"], "Adaptive seed42 summary")
    _check_hash(MAIN_V2_DIR / "panel_pathformer_test_predictions.csv", EXPECTED_SEED42_HASHES["adaptive_predictions"], "Adaptive seed42 predictions")
    _check_hash(MAIN_V2_DIR / "panel_pathformer_rank_ic_by_date.csv", EXPECTED_SEED42_HASHES["adaptive_rank_ic"], "Adaptive seed42 RankIC")
    _check_hash(MAIN_V2_DIR / "panel_pathformer_router_weights.csv", EXPECTED_SEED42_HASHES["adaptive_router_weights"], "Adaptive seed42 router weights")

    _check_hash(EXPERIMENT3_DIR / "experiment3_test_summary_metrics.csv", EXPECTED_SEED42_HASHES["static_summary"], "Static seed42 summary")
    _check_hash(EXPERIMENT3_DIR / "experiment3_test_predictions.csv", EXPECTED_SEED42_HASHES["static_predictions"], "Static seed42 predictions")
    _check_hash(EXPERIMENT3_DIR / "experiment3_rank_ic_by_date.csv", EXPECTED_SEED42_HASHES["static_rank_ic"], "Static seed42 RankIC")
    _check_hash(EXPERIMENT3_DIR / "experiment3_scale_weights.csv", EXPECTED_SEED42_HASHES["static_scale_weights"], "Static seed42 scale weights")

    # Adaptive MAIN V2 canonical subset: frequency_setting == "daily_weekly",
    # seed == 42, numeric horizon in {5, 10, 20}, mapped to horizon_label.
    def _filter_adaptive_main_v2(df: pd.DataFrame) -> pd.DataFrame:
        horizon_numeric = pd.to_numeric(df["horizon"], errors="coerce")
        subset = df[
            (df["frequency_setting"].astype(str) == FREQUENCY_CONFIGURATION)
            & (pd.to_numeric(df["seed"], errors="coerce").astype("Int64") == FROZEN_EXTERNAL_SEED)
            & (horizon_numeric.isin(HORIZON_LABEL_MAP.keys()))
        ].copy()
        subset["horizon_label"] = horizon_numeric.loc[subset.index].astype(int).map(HORIZON_LABEL_MAP)
        return subset.reset_index(drop=True)

    adaptive_summary_subset = _filter_adaptive_main_v2(adaptive_summary)
    adaptive_predictions_subset = _filter_adaptive_main_v2(adaptive_predictions)
    adaptive_rank_ic_subset = _filter_adaptive_main_v2(adaptive_rank_ic)
    adaptive_router_subset = _filter_adaptive_main_v2(adaptive_router)

    # Static Experiment 3 canonical subset: mechanism == "static", seed == 42,
    # horizon already string-labelled in {"5d", "10d", "20d"}.
    def _filter_static_experiment3(df: pd.DataFrame) -> pd.DataFrame:
        subset = df[
            (df["mechanism"].astype(str) == "static")
            & (pd.to_numeric(df["seed"], errors="coerce").astype("Int64") == FROZEN_EXTERNAL_SEED)
            & (df["horizon"].astype(str).isin(HORIZONS))
        ].copy()
        subset["horizon_label"] = subset["horizon"].astype(str)
        return subset.reset_index(drop=True)

    static_summary_subset = _filter_static_experiment3(static_summary)
    static_predictions_subset = _filter_static_experiment3(static_predictions)
    static_rank_ic_subset = _filter_static_experiment3(static_rank_ic)
    static_scale_weights_subset = _filter_static_experiment3(static_scale_weights)

    # Only after filtering do we check the canonical seed42 subset counts.
    if len(adaptive_summary_subset) != EXPECTED_SEED42_COUNTS["adaptive_summary"]:
        raise RuntimeError(f"Adaptive seed42 summary subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['adaptive_summary']}, got {len(adaptive_summary_subset)}")
    if len(adaptive_predictions_subset) != EXPECTED_SEED42_COUNTS["adaptive_predictions"]:
        raise RuntimeError(f"Adaptive seed42 predictions subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['adaptive_predictions']}, got {len(adaptive_predictions_subset)}")
    if len(adaptive_rank_ic_subset) != EXPECTED_SEED42_COUNTS["adaptive_rank_ic"]:
        raise RuntimeError(f"Adaptive seed42 RankIC subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['adaptive_rank_ic']}, got {len(adaptive_rank_ic_subset)}")
    if len(adaptive_router_subset) != EXPECTED_SEED42_COUNTS["adaptive_router_weights"]:
        raise RuntimeError(f"Adaptive seed42 router subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['adaptive_router_weights']}, got {len(adaptive_router_subset)}")

    if len(static_summary_subset) != EXPECTED_SEED42_COUNTS["static_summary"]:
        raise RuntimeError(f"Static seed42 summary subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['static_summary']}, got {len(static_summary_subset)}")
    if len(static_predictions_subset) != EXPECTED_SEED42_COUNTS["static_predictions"]:
        raise RuntimeError(f"Static seed42 predictions subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['static_predictions']}, got {len(static_predictions_subset)}")
    if len(static_rank_ic_subset) != EXPECTED_SEED42_COUNTS["static_rank_ic"]:
        raise RuntimeError(f"Static seed42 RankIC subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['static_rank_ic']}, got {len(static_rank_ic_subset)}")
    if len(static_scale_weights_subset) != EXPECTED_SEED42_COUNTS["static_scale_weights"]:
        raise RuntimeError(f"Static seed42 scale-weight subset rows mismatch: expected {EXPECTED_SEED42_COUNTS['static_scale_weights']}, got {len(static_scale_weights_subset)}")

    return {
        "adaptive_summary": adaptive_summary_subset,
        "adaptive_predictions": adaptive_predictions_subset,
        "adaptive_rank_ic": adaptive_rank_ic_subset,
        "adaptive_router": adaptive_router_subset,
        "static_summary": static_summary_subset,
        "static_predictions": static_predictions_subset,
        "static_rank_ic": static_rank_ic_subset,
        "static_scale_weights": static_scale_weights_subset,
    }


def _canonicalize_anchor_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def _audit_cell_rank_ic(predictions_df: pd.DataFrame, rank_ic_df: pd.DataFrame) -> pd.DataFrame:
    """Independently reconstruct per-date RankIC from `predictions_df` and require
    exact agreement (date keys, NaN pattern, finite values) with the persisted
    `rank_ic_df` evidence. Returns the reconstructed, canonically sorted RankIC
    table so callers never need to recompute it."""
    if len(predictions_df) != 6647:
        raise RuntimeError(f"Expected exactly 6647 predictions for cell RankIC audit; found {len(predictions_df)}.")
    if len(rank_ic_df) != 391:
        raise RuntimeError(f"Expected exactly 391 persisted RankIC rows for cell audit; found {len(rank_ic_df)}.")

    pred = predictions_df.copy()
    pred["anchor_date"] = _canonicalize_anchor_date(pred["anchor_date"])
    pred["y_true"] = pred["y_true"].astype(np.float32)
    pred["y_pred"] = pred["y_pred"].astype(np.float32)

    persisted = rank_ic_df.copy()
    persisted["anchor_date"] = _canonicalize_anchor_date(persisted["anchor_date"])
    if persisted["anchor_date"].nunique() != 391:
        raise RuntimeError(f"Expected 391 unique persisted RankIC dates; found {persisted['anchor_date'].nunique()}.")

    reconstructed = rank_ic_by_date(
        pred["y_true"].to_numpy(dtype=np.float32),
        pred["y_pred"].to_numpy(dtype=np.float32),
        pred["ticker"].to_numpy(),
        pred["anchor_date"].to_numpy(),
    )

    reconstructed = reconstructed.sort_values("anchor_date").reset_index(drop=True)
    persisted = persisted.sort_values("anchor_date").reset_index(drop=True)

    if reconstructed["anchor_date"].tolist() != persisted["anchor_date"].tolist():
        raise RuntimeError("Reconstructed and persisted RankIC anchor_date keys do not match.")

    reconstructed_nan = reconstructed["rank_ic"].isna().to_numpy()
    persisted_nan = persisted["rank_ic"].isna().to_numpy()
    if not np.array_equal(reconstructed_nan, persisted_nan):
        raise RuntimeError("Reconstructed and persisted RankIC NaN pattern do not match.")

    finite_mask = ~reconstructed_nan
    if finite_mask.any():
        reconstructed_finite = reconstructed.loc[finite_mask, "rank_ic"].to_numpy(dtype=float)
        persisted_finite = persisted.loc[finite_mask, "rank_ic"].to_numpy(dtype=float)
        if not np.allclose(
            reconstructed_finite,
            persisted_finite,
            rtol=EXPECTED_NUMERIC_TOL["rtol"],
            atol=EXPECTED_NUMERIC_TOL["atol"],
        ):
            raise RuntimeError("Reconstructed and persisted RankIC finite values do not match.")

    return reconstructed


def _reconstruct_single_cell_metrics(predictions_df: pd.DataFrame, rank_ic_df: pd.DataFrame) -> dict:
    pred = predictions_df.copy()
    pred["y_true"] = pred["y_true"].astype(np.float32)
    pred["y_pred"] = pred["y_pred"].astype(np.float32)

    metrics = evaluate_predictions(
        pred["y_true"].to_numpy(dtype=np.float32),
        pred["y_pred"].to_numpy(dtype=np.float32),
        pred["ticker"].to_numpy(),
        pred["anchor_date"].to_numpy(),
    )

    # `rank_ic_df` (the persisted evidence) is actually used here: the audit
    # reconstructs RankIC independently and requires it to match persisted values.
    audited_rank_ic = _audit_cell_rank_ic(predictions_df, rank_ic_df)
    ic_values = audited_rank_ic["rank_ic"].dropna()
    return {
        "pooled_mse": float(metrics["pooled_mse"]),
        "pooled_mae": float(metrics["pooled_mae"]),
        "pooled_corr": float(metrics["pooled_corr"]),
        "ticker_avg_mse": float(metrics["ticker_avg_mse"]),
        "ticker_avg_mae": float(metrics["ticker_avg_mae"]),
        "ticker_avg_corr": float(metrics["ticker_avg_corr"]),
        "direction_accuracy": float(metrics["pooled_da"]),
        "pred_std": float(metrics["pooled_pred_std"]),
        "true_std": float(metrics["pooled_true_std"]),
        "pred_std_true_std_ratio": float(metrics["pooled_pred_std_over_true_std"]),
        "mean_rank_ic": float(ic_values.mean()) if not ic_values.empty else float("nan"),
        # Within-cell RankIC std -> ddof=0 (matches the frozen runner's np.std(valid_ic)).
        # Across-seed robustness dispersion elsewhere uses ddof=1; do not conflate the two.
        "rank_ic_std": float(ic_values.std(ddof=0)) if len(ic_values) > 0 else float("nan"),
    }


def _validate_reconstructed_cell(raw_summary_row: pd.Series, pred_df: pd.DataFrame, rank_ic_df: pd.DataFrame) -> dict:
    reconstructed = _reconstruct_single_cell_metrics(pred_df, rank_ic_df)

    for stored_column, reconstructed_key in STORED_TO_RECONSTRUCTED_METRIC_MAP.items():
        if stored_column not in raw_summary_row:
            raise RuntimeError(f"Required stored summary column missing: {stored_column}")
        stored_value = raw_summary_row[stored_column]
        reconstructed_value = reconstructed[reconstructed_key]
        stored_is_nan = bool(pd.isna(stored_value))
        reconstructed_is_nan = bool(pd.isna(reconstructed_value))
        if stored_is_nan and reconstructed_is_nan:
            continue
        if stored_is_nan != reconstructed_is_nan:
            raise RuntimeError(
                f"NaN mismatch for {stored_column}->{reconstructed_key}: "
                f"stored_is_nan={stored_is_nan}, reconstructed_is_nan={reconstructed_is_nan}."
            )
        if not np.isclose(
            float(reconstructed_value),
            float(stored_value),
            rtol=EXPECTED_NUMERIC_TOL["rtol"],
            atol=EXPECTED_NUMERIC_TOL["atol"],
        ):
            raise RuntimeError(
                f"Reconstructed metric mismatch for {stored_column}->{reconstructed_key}: "
                f"expected {stored_value}, got {reconstructed_value}."
            )

    return reconstructed


def _load_cell_predictions_for_seed_mechanism(predictions_df: pd.DataFrame, mechanism: str, horizon: str, seed: int) -> pd.DataFrame:
    subset = predictions_df[
        (predictions_df["mechanism"].astype(str) == mechanism)
        & (predictions_df["horizon"].astype(str) == horizon)
        & (predictions_df["seed"].astype(int) == int(seed))
    ].copy()
    if subset.empty:
        raise RuntimeError(f"Missing predictions for mechanism={mechanism}, horizon={horizon}, seed={seed}")
    if len(subset) != 6647:
        raise RuntimeError(f"Expected exactly 6647 predictions for cell {mechanism}/{horizon}/{seed}; found {len(subset)}.")
    if subset["ticker"].nunique() != 17:
        raise RuntimeError(f"Unexpected ticker count for cell {mechanism}/{horizon}/{seed}: {subset['ticker'].nunique()}")
    if subset["anchor_date"].nunique() != 391:
        raise RuntimeError(f"Unexpected date count for cell {mechanism}/{horizon}/{seed}: {subset['anchor_date'].nunique()}")
    return subset.reset_index(drop=True)


def _load_cell_rank_ic_for_seed_mechanism(rank_ic_df: pd.DataFrame, mechanism: str, horizon: str, seed: int) -> pd.DataFrame:
    subset = rank_ic_df[
        (rank_ic_df["mechanism"].astype(str) == mechanism)
        & (rank_ic_df["horizon"].astype(str) == horizon)
        & (rank_ic_df["seed"].astype(int) == int(seed))
    ].copy()
    if subset.empty:
        raise RuntimeError(f"Missing RankIC for mechanism={mechanism}, horizon={horizon}, seed={seed}")
    if len(subset) != 391:
        raise RuntimeError(f"Expected exactly 391 persisted RankIC rows for cell {mechanism}/{horizon}/{seed}; found {len(subset)}.")
    return subset.reset_index(drop=True)


def _seed42_pool_for_mechanism(mechanism: str, frozen_seed42_context: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if mechanism == "adaptive":
        return (
            frozen_seed42_context["adaptive_summary"],
            frozen_seed42_context["adaptive_predictions"],
            frozen_seed42_context["adaptive_rank_ic"],
            "FROZEN_MAIN_V2_SEED42",
        )
    return (
        frozen_seed42_context["static_summary"],
        frozen_seed42_context["static_predictions"],
        frozen_seed42_context["static_rank_ic"],
        "FROZEN_EXPERIMENT3_SEED42",
    )


def _build_combined_cell_metrics(raw_context: dict, frozen_seed42_context: dict) -> pd.DataFrame:
    summary_df = raw_context["test_summary"].copy()
    predictions_df = raw_context["test_predictions"].copy()
    rank_ic_df = raw_context["rank_ic"].copy()

    records = []
    for mechanism in MECHANISMS:
        for horizon in HORIZONS:
            for seed in ALL_SEEDS:
                if seed in NEW_SEEDS:
                    pred_df = _load_cell_predictions_for_seed_mechanism(predictions_df, mechanism, horizon, seed)
                    ic_df = _load_cell_rank_ic_for_seed_mechanism(rank_ic_df, mechanism, horizon, seed)
                    summary_row = summary_df[
                        (summary_df["mechanism"].astype(str) == mechanism)
                        & (summary_df["horizon"].astype(str) == horizon)
                        & (summary_df["seed"].astype(int) == int(seed))
                    ]
                    if summary_row.empty:
                        raise RuntimeError(f"Missing raw summary row for {mechanism}/{horizon}/{seed}")
                    summary_row = summary_row.iloc[0]
                    reconstructed = _validate_reconstructed_cell(summary_row, pred_df, ic_df)
                    source_label = "EXPERIMENT4_NEW"
                else:
                    # Use the already-filtered canonical seed42 context; never reload
                    # frozen CSVs here and never compare numeric horizon vs "5d"/"10d"/"20d".
                    summary_pool, pred_pool, ic_pool, source_label = _seed42_pool_for_mechanism(mechanism, frozen_seed42_context)

                    summary_rows = summary_pool[summary_pool["horizon_label"] == horizon]
                    if len(summary_rows) != 1:
                        raise RuntimeError(
                            f"Expected exactly one seed42 {mechanism} summary row for horizon={horizon}; "
                            f"found {len(summary_rows)}."
                        )
                    summary_row = summary_rows.iloc[0]

                    pred_df = pred_pool[pred_pool["horizon_label"] == horizon].copy()
                    ic_df = ic_pool[ic_pool["horizon_label"] == horizon].copy()
                    if len(pred_df) != 6647:
                        raise RuntimeError(f"Expected exactly 6647 seed42 predictions for {mechanism}/{horizon}; found {len(pred_df)}.")
                    if len(ic_df) != 391:
                        raise RuntimeError(f"Expected exactly 391 seed42 RankIC rows for {mechanism}/{horizon}; found {len(ic_df)}.")

                    reconstructed = _validate_reconstructed_cell(summary_row, pred_df, ic_df)

                row = {
                    "mechanism": mechanism,
                    "horizon": horizon,
                    "seed": int(seed),
                    "source": source_label,
                    "frequency_configuration": FREQUENCY_CONFIGURATION,
                    "execution_backend": "CPU",
                    "pooled_mse": reconstructed["pooled_mse"],
                    "pooled_mae": reconstructed["pooled_mae"],
                    "pooled_corr": reconstructed["pooled_corr"],
                    "ticker_avg_mse": reconstructed["ticker_avg_mse"],
                    "ticker_avg_mae": reconstructed["ticker_avg_mae"],
                    "ticker_avg_corr": reconstructed["ticker_avg_corr"],
                    "direction_accuracy": reconstructed["direction_accuracy"],
                    "pred_std": reconstructed["pred_std"],
                    "true_std": reconstructed["true_std"],
                    "pred_std_true_std_ratio": reconstructed["pred_std_true_std_ratio"],
                    "mean_rank_ic": reconstructed["mean_rank_ic"],
                    "rank_ic_std": reconstructed["rank_ic_std"],
                }
                records.append(row)

    combined_df = pd.DataFrame(records)
    expected = len(MECHANISMS) * len(HORIZONS) * len(ALL_SEEDS)
    if len(combined_df) != expected:
        raise RuntimeError(f"Combined metrics row count mismatch: expected {expected}, got {len(combined_df)}")
    combined_df = combined_df.sort_values(["mechanism", "horizon", "seed"]).reset_index(drop=True)
    return combined_df


def _collect_cell_ticker_dates(mechanism: str, horizon: str, seed: int, raw_context: dict, frozen_seed42_context: dict) -> pd.DataFrame:
    """Return canonicalized (ticker, anchor_date, y_true) rows for one cell, reused
    in-memory from already-loaded/-filtered evidence (no disk reload)."""
    if seed in NEW_SEEDS:
        pred_df = _load_cell_predictions_for_seed_mechanism(raw_context["test_predictions"], mechanism, horizon, seed)
    else:
        _, pred_pool, _, _ = _seed42_pool_for_mechanism(mechanism, frozen_seed42_context)
        pred_df = pred_pool[pred_pool["horizon_label"] == horizon].copy()
        if len(pred_df) != 6647:
            raise RuntimeError(f"Expected exactly 6647 seed42 predictions for {mechanism}/{horizon}; found {len(pred_df)}.")

    out = pred_df[["ticker", "anchor_date", "y_true"]].copy()
    out["anchor_date"] = _canonicalize_anchor_date(out["anchor_date"])
    out["y_true"] = out["y_true"].astype(np.float32)
    return out


def _run_pairability_audit(raw_context: dict, frozen_seed42_context: dict) -> None:
    """Dedicated read-only pairability audit run before any output writing.

    For each horizon, all 10 cells (2 mechanisms x 5 seeds) must contain exactly
    6647 predictions with identical (ticker, anchor_date) keys and identical
    y_true values (the realized test targets are shared across mechanisms/seeds)."""
    for horizon in HORIZONS:
        cell_frames = {}
        for mechanism in MECHANISMS:
            for seed in ALL_SEEDS:
                df = _collect_cell_ticker_dates(mechanism, horizon, seed, raw_context, frozen_seed42_context)
                if len(df) != 6647:
                    raise RuntimeError(f"Pairability audit: expected 6647 predictions for {mechanism}/{horizon}/{seed}; found {len(df)}.")
                cell_frames[(mechanism, seed)] = df.sort_values(["ticker", "anchor_date"]).reset_index(drop=True)

        if len(cell_frames) != len(MECHANISMS) * len(ALL_SEEDS):
            raise RuntimeError(f"Pairability audit: expected {len(MECHANISMS) * len(ALL_SEEDS)} cells for horizon={horizon}; found {len(cell_frames)}.")

        reference_key = next(iter(cell_frames))
        reference = cell_frames[reference_key]
        reference_keys = list(zip(reference["ticker"].tolist(), reference["anchor_date"].tolist()))
        reference_y_true = reference["y_true"].to_numpy(dtype=np.float32)

        for key, df in cell_frames.items():
            if key == reference_key:
                continue
            candidate_keys = list(zip(df["ticker"].tolist(), df["anchor_date"].tolist()))
            if candidate_keys != reference_keys:
                raise RuntimeError(f"Pairability audit: ticker/anchor_date keys mismatch for horizon={horizon}, cell={key}.")

            candidate_y_true = df["y_true"].to_numpy(dtype=np.float32)
            if np.array_equal(reference_y_true, candidate_y_true):
                continue
            if not np.allclose(
                reference_y_true,
                candidate_y_true,
                rtol=Y_TRUE_RTOL,
                atol=Y_TRUE_ATOL,
                equal_nan=False,
            ):
                raise RuntimeError(f"Pairability audit: y_true mismatch for horizon={horizon}, cell={key}.")




def _build_pair_delta_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    pair_rows = []
    for horizon in HORIZONS:
        for seed in ALL_SEEDS:
            adaptive_row = combined_df[
                (combined_df["mechanism"] == "adaptive")
                & (combined_df["horizon"] == horizon)
                & (combined_df["seed"] == int(seed))
            ]
            static_row = combined_df[
                (combined_df["mechanism"] == "static")
                & (combined_df["horizon"] == horizon)
                & (combined_df["seed"] == int(seed))
            ]
            if adaptive_row.empty or static_row.empty:
                raise RuntimeError(f"Missing paired seed row for horizon={horizon}, seed={seed}")
            a = adaptive_row.iloc[0]
            s = static_row.iloc[0]
            delta_mse = float(a["pooled_mse"]) - float(s["pooled_mse"])
            delta_rank_ic = float(a["mean_rank_ic"]) - float(s["mean_rank_ic"])
            pair_rows.append(
                {
                    "horizon": horizon,
                    "seed": int(seed),
                    "adaptive_mse": float(a["pooled_mse"]),
                    "static_mse": float(s["pooled_mse"]),
                    "delta_mse": delta_mse,
                    "adaptive_mean_rank_ic": float(a["mean_rank_ic"]),
                    "static_mean_rank_ic": float(s["mean_rank_ic"]),
                    "delta_mean_rank_ic": delta_rank_ic,
                }
            )
    delta_df = pd.DataFrame(pair_rows)
    if len(delta_df) != len(HORIZONS) * len(ALL_SEEDS):
        raise RuntimeError(f"Pair delta row count mismatch: expected {len(HORIZONS) * len(ALL_SEEDS)}, got {len(delta_df)}")
    return delta_df.sort_values(["horizon", "seed"]).reset_index(drop=True)


def _build_robustness_summary(combined_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        adaptive_cells = combined_df[(combined_df["mechanism"] == "adaptive") & (combined_df["horizon"] == horizon)]
        static_cells = combined_df[(combined_df["mechanism"] == "static") & (combined_df["horizon"] == horizon)]
        adaptive = adaptive_cells["pooled_mse"].astype(float)
        static = static_cells["pooled_mse"].astype(float)
        delta = pair_df[(pair_df["horizon"] == horizon)]["delta_mse"].astype(float)

        adaptive_mean = float(adaptive.mean())
        adaptive_sd = float(adaptive.std(ddof=1)) if len(adaptive) > 1 else float("nan")
        static_mean = float(static.mean())
        static_sd = float(static.std(ddof=1)) if len(static) > 1 else float("nan")
        delta_mean = float(delta.mean())
        delta_sd = float(delta.std(ddof=1)) if len(delta) > 1 else float("nan")

        adaptive_wins = int((delta < 0).sum())
        static_wins = int((delta > 0).sum())
        ties = int((delta == 0).sum())

        row = {
            "horizon": horizon,
            "adaptive_mean_mse": adaptive_mean,
            "adaptive_sd_mse": adaptive_sd,
            "static_mean_mse": static_mean,
            "static_sd_mse": static_sd,
            "mean_delta_mse": delta_mean,
            "sd_delta_mse": delta_sd,
            "adaptive_mse_wins": adaptive_wins,
            "static_mse_wins": static_wins,
            "mse_ties": ties,
        }

        # Mechanism-specific five-seed mean/SD (across-seed dispersion -> ddof=1)
        # for every remaining reconstructed metric, in addition to MSE above.
        for metric in ADDITIONAL_ROBUSTNESS_METRICS:
            row[f"adaptive_mean_{metric}"] = _safe_mean(adaptive_cells[metric].astype(float))
            row[f"adaptive_sd_{metric}"] = _safe_std(adaptive_cells[metric].astype(float), ddof=1)
            row[f"static_mean_{metric}"] = _safe_mean(static_cells[metric].astype(float))
            row[f"static_sd_{metric}"] = _safe_std(static_cells[metric].astype(float), ddof=1)

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if len(summary_df) != len(HORIZONS):
        raise RuntimeError(f"Robustness summary row count mismatch: expected {len(HORIZONS)}, got {len(summary_df)}")
    return summary_df.sort_values("horizon").reset_index(drop=True)


def _build_summary_text(combined_df: pd.DataFrame, pair_df: pd.DataFrame, robustness_df: pd.DataFrame) -> str:
    summary_lines = [
        "Experiment 4 descriptive multi-seed robustness summary",
        "",
        "- Experiment 4 uses five pre-specified seeds: 0, 1, 21, 42, 3407.",
        "- seed42 was reused from frozen external evidence and was not retrained.",
        "- delta_mse = Adaptive - Static.",
        "- Results are descriptive multi-seed robustness evidence.",
        "- No serial-dependence-aware formal inference has yet been performed.",
        "- No statistical significance claim is made.",
        "- Adaptive has greater model capacity than Static.",
        "- Router economic interpretation remains gated.",
        "",
        f"Combined cells: {len(combined_df)} rows (2 mechanisms x 3 horizons x 5 seeds).",
        f"Paired seed rows: {len(pair_df)} rows (3 horizons x 5 seeds).",
        f"Robustness summary rows: {len(robustness_df)} rows (one per horizon).",
        "",
        "Per-horizon descriptive robustness (five-seed, descriptive only):",
    ]
    for _, row in robustness_df.iterrows():
        summary_lines.extend(
            [
                f"- Horizon {row['horizon']}:",
                f"    Adaptive mean MSE: {row['adaptive_mean_mse']:.10g}",
                f"    Static mean MSE: {row['static_mean_mse']:.10g}",
                f"    Mean delta_mse (Adaptive - Static): {row['mean_delta_mse']:.10g}",
                f"    Adaptive MSE wins: {int(row['adaptive_mse_wins'])}",
                f"    Static MSE wins: {int(row['static_mse_wins'])}",
                f"    Ties: {int(row['mse_ties'])}",
                f"    Adaptive cross-seed mean Mean RankIC: {row['adaptive_mean_mean_rank_ic']:.10g}",
                f"    Static cross-seed mean Mean RankIC: {row['static_mean_mean_rank_ic']:.10g}",
            ]
        )
    return "\n".join(summary_lines) + "\n"


def _build_manifest_dataframe(
    raw_context: dict,
    frozen_seed42_context: dict,
    combined_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    combined_sha: str,
    pair_sha: str,
    robustness_sha: str,
    summary_sha: str,
) -> pd.DataFrame:
    finalizer_script_path = Path(__file__).resolve()
    finalizer_sha = _sha256(finalizer_script_path)
    protocol_sha = _check_hash(EXPERIMENT4_PROTOCOL_PATH, EXPECTED_RAW_HASHES["experiment4_multiseed_protocol.md"], "Experiment 4 protocol")
    runner_sha = _check_hash(EXPERIMENT4_RUNNER_PATH, EXPECTED_RAW_HASHES["panel_pathformer_experiment4_multiseed.py"], "Experiment 4 runner")

    hash_rows = {
        "finalizer_script_path": str(finalizer_script_path.relative_to(ROOT)),
        "finalizer_script_sha256": finalizer_sha,
        "protocol_path": str(EXPERIMENT4_PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "runner_path": str(EXPERIMENT4_RUNNER_PATH.relative_to(ROOT)),
        "runner_sha256": runner_sha,
        "dataset_manifest_path": str(DATASET_MANIFEST_PATH.relative_to(ROOT)),
        "dataset_manifest_sha256": EXPECTED_DEPENDENCY_HASHES["dataset_manifest"],
        "main_v2_pathformer_script_path": str(MAIN_V2_PATHFORMER_SCRIPT_PATH.relative_to(ROOT)),
        "main_v2_pathformer_script_sha256": EXPECTED_DEPENDENCY_HASHES["main_v2_pathformer_script"],
        "experiment3_runner_path": str(EXPERIMENT3_RUNNER_PATH.relative_to(ROOT)),
        "experiment3_runner_sha256": EXPECTED_DEPENDENCY_HASHES["experiment3_runner"],
        "panel_common_path": str(PANEL_COMMON_PATH.relative_to(ROOT)),
        "panel_common_sha256": EXPECTED_DEPENDENCY_HASHES["panel_common"],
        "experiment4_training_provenance_path": str((EXPERIMENT_DIR / "experiment4_training_provenance.csv").relative_to(ROOT)),
        "experiment4_training_provenance_sha256": EXPECTED_RAW_HASHES["experiment4_training_provenance.csv"],
        "experiment4_training_summary_path": str((EXPERIMENT_DIR / "experiment4_training_summary.csv").relative_to(ROOT)),
        "experiment4_training_summary_sha256": EXPECTED_RAW_HASHES["experiment4_training_summary.csv"],
        "experiment4_training_history_path": str((EXPERIMENT_DIR / "experiment4_training_history.csv").relative_to(ROOT)),
        "experiment4_training_history_sha256": EXPECTED_RAW_HASHES["experiment4_training_history.csv"],
        "experiment4_test_attempt_ledger_path": str((EXPERIMENT_DIR / "experiment4_test_attempt_ledger.csv").relative_to(ROOT)),
        "experiment4_test_attempt_ledger_sha256": EXPECTED_RAW_HASHES["experiment4_test_attempt_ledger.csv"],
        "experiment4_test_summary_metrics_path": str((EXPERIMENT_DIR / "experiment4_test_summary_metrics.csv").relative_to(ROOT)),
        "experiment4_test_summary_metrics_sha256": EXPECTED_RAW_HASHES["experiment4_test_summary_metrics.csv"],
        "experiment4_test_predictions_path": str((EXPERIMENT_DIR / "experiment4_test_predictions.csv").relative_to(ROOT)),
        "experiment4_test_predictions_sha256": EXPECTED_RAW_HASHES["experiment4_test_predictions.csv"],
        "experiment4_rank_ic_by_date_path": str((EXPERIMENT_DIR / "experiment4_rank_ic_by_date.csv").relative_to(ROOT)),
        "experiment4_rank_ic_by_date_sha256": EXPECTED_RAW_HASHES["experiment4_rank_ic_by_date.csv"],
        "experiment4_scale_weights_path": str((EXPERIMENT_DIR / "experiment4_scale_weights.csv").relative_to(ROOT)),
        "experiment4_scale_weights_sha256": EXPECTED_RAW_HASHES["experiment4_scale_weights.csv"],
        "adaptive_seed42_summary_path": str((MAIN_V2_DIR / "panel_pathformer_summary_metrics.csv").relative_to(ROOT)),
        "adaptive_seed42_summary_sha256": EXPECTED_SEED42_HASHES["adaptive_summary"],
        "adaptive_seed42_predictions_path": str((MAIN_V2_DIR / "panel_pathformer_test_predictions.csv").relative_to(ROOT)),
        "adaptive_seed42_predictions_sha256": EXPECTED_SEED42_HASHES["adaptive_predictions"],
        "adaptive_seed42_rank_ic_path": str((MAIN_V2_DIR / "panel_pathformer_rank_ic_by_date.csv").relative_to(ROOT)),
        "adaptive_seed42_rank_ic_sha256": EXPECTED_SEED42_HASHES["adaptive_rank_ic"],
        "adaptive_seed42_router_path": str((MAIN_V2_DIR / "panel_pathformer_router_weights.csv").relative_to(ROOT)),
        "adaptive_seed42_router_sha256": EXPECTED_SEED42_HASHES["adaptive_router_weights"],
        "static_seed42_summary_path": str((EXPERIMENT3_DIR / "experiment3_test_summary_metrics.csv").relative_to(ROOT)),
        "static_seed42_summary_sha256": EXPECTED_SEED42_HASHES["static_summary"],
        "static_seed42_predictions_path": str((EXPERIMENT3_DIR / "experiment3_test_predictions.csv").relative_to(ROOT)),
        "static_seed42_predictions_sha256": EXPECTED_SEED42_HASHES["static_predictions"],
        "static_seed42_rank_ic_path": str((EXPERIMENT3_DIR / "experiment3_rank_ic_by_date.csv").relative_to(ROOT)),
        "static_seed42_rank_ic_sha256": EXPECTED_SEED42_HASHES["static_rank_ic"],
        "static_seed42_scale_weights_path": str((EXPERIMENT3_DIR / "experiment3_scale_weights.csv").relative_to(ROOT)),
        "static_seed42_scale_weights_sha256": EXPECTED_SEED42_HASHES["static_scale_weights"],
        # Row counts are derived from validated DataFrame lengths, never from
        # counting newline characters in raw file text.
        "expected_training_provenance_rows": EXPECTED_RAW_ROW_COUNTS["training_provenance"],
        "actual_training_provenance_rows": int(raw_context["provenance_rows"]),
        "expected_training_summary_rows": EXPECTED_RAW_ROW_COUNTS["training_summary"],
        "actual_training_summary_rows": int(len(raw_context["training_summary"])),
        "expected_training_history_rows": EXPECTED_RAW_ROW_COUNTS["training_history"],
        "actual_training_history_rows": int(len(raw_context["training_history"])),
        "expected_test_attempt_ledger_rows": EXPECTED_RAW_ROW_COUNTS["test_attempt_ledger"],
        "actual_test_attempt_ledger_rows": int(len(raw_context["test_attempt_ledger"])),
        "expected_test_summary_rows": EXPECTED_RAW_ROW_COUNTS["test_summary"],
        "actual_test_summary_rows": int(len(raw_context["test_summary"])),
        "expected_test_predictions_rows": EXPECTED_RAW_ROW_COUNTS["test_predictions"],
        "actual_test_predictions_rows": int(len(raw_context["test_predictions"])),
        "expected_rank_ic_rows": EXPECTED_RAW_ROW_COUNTS["rank_ic_by_date"],
        "actual_rank_ic_rows": int(len(raw_context["rank_ic"])),
        "expected_scale_weights_rows": EXPECTED_RAW_ROW_COUNTS["scale_weights"],
        "actual_scale_weights_rows": int(len(raw_context["scale_weights"])),
        "expected_checkpoints": EXPECTED_RAW_ROW_COUNTS["checkpoints"],
        "actual_checkpoints": int(raw_context["checkpoint_count"]),
        "expected_adaptive_seed42_summary_rows": EXPECTED_SEED42_COUNTS["adaptive_summary"],
        "actual_adaptive_seed42_summary_rows": int(len(frozen_seed42_context["adaptive_summary"])),
        "expected_adaptive_seed42_predictions_rows": EXPECTED_SEED42_COUNTS["adaptive_predictions"],
        "actual_adaptive_seed42_predictions_rows": int(len(frozen_seed42_context["adaptive_predictions"])),
        "expected_adaptive_seed42_rank_ic_rows": EXPECTED_SEED42_COUNTS["adaptive_rank_ic"],
        "actual_adaptive_seed42_rank_ic_rows": int(len(frozen_seed42_context["adaptive_rank_ic"])),
        "expected_adaptive_seed42_router_rows": EXPECTED_SEED42_COUNTS["adaptive_router_weights"],
        "actual_adaptive_seed42_router_rows": int(len(frozen_seed42_context["adaptive_router"])),
        "expected_static_seed42_summary_rows": EXPECTED_SEED42_COUNTS["static_summary"],
        "actual_static_seed42_summary_rows": int(len(frozen_seed42_context["static_summary"])),
        "expected_static_seed42_predictions_rows": EXPECTED_SEED42_COUNTS["static_predictions"],
        "actual_static_seed42_predictions_rows": int(len(frozen_seed42_context["static_predictions"])),
        "expected_static_seed42_rank_ic_rows": EXPECTED_SEED42_COUNTS["static_rank_ic"],
        "actual_static_seed42_rank_ic_rows": int(len(frozen_seed42_context["static_rank_ic"])),
        "expected_static_seed42_scale_weights_rows": EXPECTED_SEED42_COUNTS["static_scale_weights"],
        "actual_static_seed42_scale_weights_rows": int(len(frozen_seed42_context["static_scale_weights"])),
        "expected_combined_cell_rows": int(len(MECHANISMS) * len(HORIZONS) * len(ALL_SEEDS)),
        "actual_combined_cell_rows": int(len(combined_df)),
        "expected_pair_rows": int(len(HORIZONS) * len(ALL_SEEDS)),
        "actual_pair_rows": int(len(pair_df)),
        "expected_robustness_rows": int(len(HORIZONS)),
        "actual_robustness_rows": int(len(robustness_df)),
        "seed_set": "0,1,21,42,3407",
        "mechanism_set": "adaptive,static",
        "horizons": "5d,10d,20d",
        "frequency_configuration": FREQUENCY_CONFIGURATION,
        "ddof_cross_seed_dispersion": "ddof=1",
        "ddof_within_cell_rank_ic_std": "ddof=0",
        "delta_mse_sign_convention": "delta_mse = Adaptive - Static",
        "statement_no_training": "No training occurred.",
        "statement_no_test_inference": "No TEST inference occurred.",
        "statement_no_statistical_inference": "No statistical inference occurred.",
        "combined_cell_metrics_sha256": combined_sha,
        "seed_pair_deltas_sha256": pair_sha,
        "robustness_summary_sha256": robustness_sha,
        "final_summary_sha256": summary_sha,
    }

    return pd.DataFrame([hash_rows])


def _write_outputs(
    combined_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    raw_context: dict,
    frozen_seed42_context: dict,
) -> dict:
    # Fail closed if ANY final output already exists, before any temp file is created.
    for target in FINAL_OUTPUTS:
        _require_not_exist(target, f"final output {target.name}")

    combined_path, pair_path, robustness_path, summary_path, manifest_path = FINAL_OUTPUTS

    run_id = uuid.uuid4().hex
    temp_paths = {
        "combined": combined_path.with_name(f"{combined_path.stem}.{run_id}.tmp{combined_path.suffix}"),
        "pair": pair_path.with_name(f"{pair_path.stem}.{run_id}.tmp{pair_path.suffix}"),
        "robustness": robustness_path.with_name(f"{robustness_path.stem}.{run_id}.tmp{robustness_path.suffix}"),
        "summary": summary_path.with_name(f"{summary_path.stem}.{run_id}.tmp{summary_path.suffix}"),
        "manifest": manifest_path.with_name(f"{manifest_path.stem}.{run_id}.tmp{manifest_path.suffix}"),
    }
    created_temp_files: list[Path] = []
    # Tracks final paths already committed by THIS invocation, so a failure
    # partway through the five renames can be rolled back precisely.
    committed_final_paths: list[Path] = []

    try:
        combined_df.to_csv(temp_paths["combined"], index=False)
        created_temp_files.append(temp_paths["combined"])

        pair_df.to_csv(temp_paths["pair"], index=False)
        created_temp_files.append(temp_paths["pair"])

        robustness_df.to_csv(temp_paths["robustness"], index=False)
        created_temp_files.append(temp_paths["robustness"])

        summary_text = _build_summary_text(combined_df, pair_df, robustness_df)
        temp_paths["summary"].write_text(summary_text, encoding="utf-8")
        created_temp_files.append(temp_paths["summary"])

        combined_sha = _sha256(temp_paths["combined"])
        pair_sha = _sha256(temp_paths["pair"])
        robustness_sha = _sha256(temp_paths["robustness"])
        summary_sha = _sha256(temp_paths["summary"])

        manifest_df = _build_manifest_dataframe(
            raw_context,
            frozen_seed42_context,
            combined_df,
            pair_df,
            robustness_df,
            combined_sha,
            pair_sha,
            robustness_sha,
            summary_sha,
        )
        manifest_df.to_csv(temp_paths["manifest"], index=False)
        created_temp_files.append(temp_paths["manifest"])

        for tmp in temp_paths.values():
            if not tmp.exists():
                raise RuntimeError(f"Temporary output file missing before commit: {tmp}")

        os.replace(temp_paths["combined"], combined_path)
        committed_final_paths.append(combined_path)
        os.replace(temp_paths["pair"], pair_path)
        committed_final_paths.append(pair_path)
        os.replace(temp_paths["robustness"], robustness_path)
        committed_final_paths.append(robustness_path)
        os.replace(temp_paths["summary"], summary_path)
        committed_final_paths.append(summary_path)
        os.replace(temp_paths["manifest"], manifest_path)
        committed_final_paths.append(manifest_path)

        # Post-commit verification: all five final files must exist, and outputs
        # 1-4 must hash to the values recorded in this run's manifest data.
        for target in FINAL_OUTPUTS:
            if not target.exists():
                raise RuntimeError(f"Post-commit verification failed: final output missing: {target}")

        post_commit_hashes = {
            combined_path: combined_sha,
            pair_path: pair_sha,
            robustness_path: robustness_sha,
            summary_path: summary_sha,
        }
        for target, expected_sha in post_commit_hashes.items():
            actual_sha = _sha256(target)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Post-commit verification failed: {target} hash mismatch "
                    f"(expected {expected_sha}, got {actual_sha})."
                )
    except Exception:
        # Delete only temp files created by this finalizer invocation; never
        # touch raw/frozen evidence or any final output not committed by this run.
        for tmp in created_temp_files:
            if tmp.exists():
                tmp.unlink()
        # Roll back only the final paths this invocation itself committed; all
        # five paths were required absent before this invocation started, so
        # anything in this list was created here and is safe to remove.
        for final_path in committed_final_paths:
            if final_path.exists():
                final_path.unlink()
        raise

    return {
        "combined_path": combined_path,
        "pair_path": pair_path,
        "robustness_path": robustness_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
    }



def _check_finalizer_no_overwrite() -> None:
    for target in FINAL_OUTPUTS:
        _require_not_exist(target, f"final output path {target.name}")


def _validate_output_contract() -> None:
    if not EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Missing Experiment 4 evidence directory: {EXPERIMENT_DIR}")
    _check_finalizer_no_overwrite()


def main() -> int:
    parser = argparse.ArgumentParser(description="Immutable finalizer for Experiment 4 PathFormer multi-seed descriptive evidence.")
    parser.add_argument("--finalize", action="store_true", help="Finalize the frozen descriptive Experiment 4 audit outputs.")
    args = parser.parse_args()

    if not args.finalize:
        print("Experiment 4 finalizer contract: verify frozen evidence, reconstruct metrics from saved TEST predictions, combine new seeds with frozen seed42 external evidence, and write the descriptive 5-seed robustness outputs. No training, inference, or statistical inference is performed.")
        return 0

    _validate_output_contract()
    raw_context = _validate_raw_experiment4_files()
    frozen_seed42_context = _validate_frozen_seed42_sources()
    combined_df = _build_combined_cell_metrics(raw_context, frozen_seed42_context)
    _run_pairability_audit(raw_context, frozen_seed42_context)
    pair_df = _build_pair_delta_table(combined_df)
    robustness_df = _build_robustness_summary(combined_df, pair_df)
    _write_outputs(combined_df, pair_df, robustness_df, raw_context, frozen_seed42_context)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
