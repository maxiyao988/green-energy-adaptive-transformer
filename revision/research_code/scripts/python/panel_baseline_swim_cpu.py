# SWiM MAIN V2 CPU formal execution path.
# This file is a CPU-only fork of the frozen formal script.
# It changes only backend execution and CPU-local governance while preserving
# the formal architecture, training semantics, and evidence contract.

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.python.panel_baseline_swim as formal
from scripts.python.panel_common import evaluate_predictions, rank_ic_by_date

D_MODEL = formal.D_MODEL
NHEAD = formal.NHEAD
NUM_BLOCKS = formal.NUM_BLOCKS
DIM_FEEDFORWARD = formal.DIM_FEEDFORWARD
DROPOUT = formal.DROPOUT
FIXED_MAX_LEN = formal.FIXED_MAX_LEN
FREQUENCY_SETTINGS = formal.FREQUENCY_SETTINGS
HORIZONS = formal.HORIZONS
BATCH_SIZE = formal.BATCH_SIZE
LEARNING_RATE = formal.LEARNING_RATE
MAX_EPOCHS = formal.MAX_EPOCHS
EARLY_STOPPING_PATIENCE = formal.EARLY_STOPPING_PATIENCE
SEED = formal.SEED
WINDOW_CONFIG = formal.WINDOW_CONFIG
PositionalEncoding = formal.PositionalEncoding
apply_normalizer = formal.apply_normalizer
assert_frozen_panel_checks = formal.assert_frozen_panel_checks
compute_rank_ic_summary = formal.compute_rank_ic_summary
count_trainable_parameters = formal.count_trainable_parameters
fit_train_only_normalizer = formal.fit_train_only_normalizer
format_elapsed = formal.format_elapsed
load_frozen_panel = formal.load_frozen_panel
prediction_scale_looks_normal = formal.prediction_scale_looks_normal
set_seed = formal.set_seed

DATASET_MANIFEST_SHA256 = formal.DATASET_MANIFEST_SHA256
MAIN_V2_SWIM_DIR = ROOT / "dataset" / "audit" / "main_v2" / "swim_cpu"
EXPECTED_SWIM_PARAM_COUNTS = formal.EXPECTED_SWIM_PARAM_COUNTS
FORMAL_SWIM_RUN_FILES = formal.FORMAL_SWIM_RUN_FILES


def compute_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_output_dir(output_dir: str | Path | None) -> Path:
    resolved = Path(output_dir) if output_dir is not None else MAIN_V2_SWIM_DIR
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    expected = MAIN_V2_SWIM_DIR.resolve()
    if resolved != expected:
        raise ValueError(f"CPU formal output directory must be exactly {expected}; got {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_cpu_training_provenance_row(row: pd.Series) -> None:
    required = {
        "run_family": "MAIN_V2",
        "model": "SWIM_STYLE_TRANSFORMER",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "dataset_audit_status": "PASS",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "formal_tickers": 17,
        "common_anchor_dates": 2563,
        "total_samples": 43571,
        "train_samples": 30634,
        "validation_samples": 6290,
        "test_samples": 6647,
        "train_dates": 1802,
        "validation_dates": 370,
        "test_dates": 391,
        "nominal_seed": 42,
        "training_script": "scripts/python/panel_baseline_swim_cpu.py",
        "output_dir": str(MAIN_V2_SWIM_DIR.resolve()),
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_swim_blocks": NUM_BLOCKS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "activation": "relu",
        "window_size_daily": 10,
        "shift_size_daily": 5,
        "window_size_weekly": 4,
        "shift_size_weekly": 2,
        "positional_encoding": "fixed_sinusoidal",
        "pooling": "masked_mean_pool_real_timesteps_only",
        "normalization": "post_ln",
        "fusion": "late_concat_fusion",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "max_norm": 1.0,
        "source_frozen_script": "scripts/python/panel_baseline_swim.py",
        "source_frozen_script_sha256": "30a6e332ed34f012a05c27134dda33e1e575715fdf7585c57d731a3de065129a",
        "execution_backend": "CPU",
    }
    for key, expected in required.items():
        if key not in row.index:
            raise ValueError(f"CPU provenance missing required field: {key}")
        actual = row[key]
        if pd.isna(actual):
            raise ValueError(f"CPU provenance field is null: {key}")
        if key in {"d_model", "nhead", "num_swim_blocks", "dim_feedforward", "batch_size", "max_epochs", "patience", "formal_tickers", "common_anchor_dates", "total_samples", "train_samples", "validation_samples", "test_samples", "train_dates", "validation_dates", "test_dates", "nominal_seed"}:
            if int(actual) != int(expected):
                raise ValueError(f"CPU provenance field {key} mismatch: expected {expected}, found {actual}")
        elif key in {"learning_rate", "dropout", "max_norm"}:
            if float(actual) != float(expected):
                raise ValueError(f"CPU provenance field {key} mismatch: expected {expected}, found {actual}")
        elif str(actual).strip() != str(expected):
            raise ValueError(f"CPU provenance field {key} mismatch: expected {expected}, found {actual}")
    if "training_script_sha256" not in row.index or pd.isna(row["training_script_sha256"]):
        raise ValueError("CPU provenance missing training_script_sha256.")


def write_cpu_training_provenance(output_dir: Path, script_path: Path) -> Path:
    output_dir = resolve_output_dir(output_dir)
    provenance_path = output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"]
    current_sha = compute_sha256(script_path)
    if provenance_path.exists():
        existing = pd.read_csv(provenance_path)
        if existing.empty:
            raise ValueError(f"CPU training provenance exists but is empty: {provenance_path}")
        row = existing.iloc[0]
        validate_cpu_training_provenance_row(row)
        stored_sha = row.get("training_script_sha256")
        if pd.notna(stored_sha) and str(stored_sha) != current_sha:
            raise ValueError(
                f"CPU provenance mismatch: stored SHA {stored_sha} differs from current script SHA {current_sha}. "
                "Refusing to overwrite CPU provenance."
            )
        return provenance_path

    row = {
        "run_family": "MAIN_V2",
        "model": "SWIM_STYLE_TRANSFORMER",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "dataset_audit_status": "PASS",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "formal_tickers": 17,
        "common_anchor_dates": 2563,
        "total_samples": 43571,
        "train_samples": 30634,
        "validation_samples": 6290,
        "test_samples": 6647,
        "train_dates": 1802,
        "validation_dates": 370,
        "test_dates": 391,
        "nominal_seed": 42,
        "training_script": "scripts/python/panel_baseline_swim_cpu.py",
        "training_script_sha256": current_sha,
        "output_dir": str(output_dir.resolve()),
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_swim_blocks": NUM_BLOCKS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "activation": "relu",
        "window_size_daily": 10,
        "shift_size_daily": 5,
        "window_size_weekly": 4,
        "shift_size_weekly": 2,
        "positional_encoding": "fixed_sinusoidal",
        "pooling": "masked_mean_pool_real_timesteps_only",
        "normalization": "post_ln",
        "fusion": "late_concat_fusion",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "max_norm": 1.0,
        "source_frozen_script": "scripts/python/panel_baseline_swim.py",
        "source_frozen_script_sha256": "30a6e332ed34f012a05c27134dda33e1e575715fdf7585c57d731a3de065129a",
        "execution_backend": "CPU",
    }
    pd.DataFrame([row]).to_csv(provenance_path, index=False)
    return provenance_path


def assert_clean_cpu_swim_namespace(output_dir: Path) -> None:
    output_dir = resolve_output_dir(output_dir)
    formal_files = [
        output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"],
        output_dir / FORMAL_SWIM_RUN_FILES["summary"],
        output_dir / FORMAL_SWIM_RUN_FILES["predictions"],
        output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"],
        output_dir / FORMAL_SWIM_RUN_FILES["history"],
        output_dir / FORMAL_SWIM_RUN_FILES["comparison"],
        output_dir / FORMAL_SWIM_RUN_FILES["manifest"],
        output_dir / FORMAL_SWIM_RUN_FILES["final_summary"],
    ]
    stale = [str(path.name) for path in formal_files if path.exists()]
    if stale:
        raise FileExistsError(
            f"CPU SWiM formal namespace is not clean at {output_dir}. Existing formal artifacts: {stale}. "
            "Refusing to overwrite formal evidence; archive or remove stale files manually before --train --all."
        )


def _cell_set_from_df_cpu(df: pd.DataFrame, freq_col: str, horizon_col: str) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(zip(df[freq_col].astype(str), df[horizon_col].astype(str)))


def validate_partial_raw_cell_set_consistency_cpu(output_dir: Path) -> set[tuple[str, str]]:
    output_dir = resolve_output_dir(output_dir)
    raw_paths = {
        "summary": output_dir / FORMAL_SWIM_RUN_FILES["summary"],
        "predictions": output_dir / FORMAL_SWIM_RUN_FILES["predictions"],
        "rank_ic": output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"],
        "history": output_dir / FORMAL_SWIM_RUN_FILES["history"],
    }
    existing = {key: path.exists() for key, path in raw_paths.items()}
    if not any(existing.values()):
        return set()
    if not all(existing.values()):
        raise ValueError(
            f"Corrupted partial CPU SWiM raw state: some raw artifacts exist but not all four. Presence={existing}. "
            "Manual inspection is required before resume."
        )

    summary_df = pd.read_csv(raw_paths["summary"])
    pred_df = pd.read_csv(raw_paths["predictions"])
    rank_ic_df = pd.read_csv(raw_paths["rank_ic"])
    hist_df = pd.read_csv(raw_paths["history"])
    artifacts = {
        "summary": summary_df,
        "predictions": pred_df,
        "rank_ic": rank_ic_df,
        "history": hist_df,
    }
    for name, df in artifacts.items():
        if df.empty:
            raise ValueError(f"Existing partial raw artifact is empty: {name}. Manual inspection is required.")
    if summary_df.duplicated(subset=["frequency_setting", "horizon"]).any():
        raise ValueError("Existing partial summary has duplicate frequency_setting/horizon cells.")
    if pred_df.duplicated(subset=["model", "ticker", "anchor_date", "horizon", "frequency_setting", "seed", "best_epoch"]).any():
        raise ValueError("Existing partial predictions have duplicate formal keys.")
    if rank_ic_df.duplicated(subset=["model", "frequency_setting", "horizon", "anchor_date"]).any():
        raise ValueError("Existing partial Rank-IC has duplicate formal keys.")
    if hist_df.duplicated(subset=["frequency_setting", "horizon", "epoch"]).any():
        raise ValueError("Existing partial history has duplicate frequency_setting/horizon/epoch keys.")

    cell_sets = {
        "summary": _cell_set_from_df_cpu(summary_df, "frequency_setting", "horizon"),
        "predictions": _cell_set_from_df_cpu(pred_df, "frequency_setting", "horizon"),
        "rank_ic": _cell_set_from_df_cpu(rank_ic_df, "frequency_setting", "horizon"),
        "history": _cell_set_from_df_cpu(hist_df, "frequency_setting", "horizon"),
    }
    base_name, base_set = next(iter(cell_sets.items()))
    for name, cs in cell_sets.items():
        if cs != base_set:
            raise ValueError(
                f"Existing partial CPU raw artifacts disagree on completed cell sets: {base_name}={sorted(base_set)} vs {name}={sorted(cs)}. Manual inspection required."
            )
    return base_set


def ensure_requested_cell_absent_cpu(output_dir: Path, frequency: str, horizon: str, seed: int = SEED) -> None:
    output_dir = resolve_output_dir(output_dir)
    summary_path = output_dir / FORMAL_SWIM_RUN_FILES["summary"]
    pred_path = output_dir / FORMAL_SWIM_RUN_FILES["predictions"]
    rank_ic_path = output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]
    history_path = output_dir / FORMAL_SWIM_RUN_FILES["history"]

    hit_summary = False
    hit_pred = False
    hit_rank_ic = False
    hit_history = False

    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        hit_summary = bool(((sdf["frequency_setting"] == frequency) & (sdf["horizon"] == horizon) & (pd.to_numeric(sdf["seed"], errors="coerce") == seed)).any())
    if pred_path.exists():
        pdf = pd.read_csv(pred_path)
        hit_pred = bool(((pdf["frequency_setting"] == frequency) & (pdf["horizon"] == horizon) & (pd.to_numeric(pdf["seed"], errors="coerce") == seed)).any())
    if rank_ic_path.exists():
        rdf = pd.read_csv(rank_ic_path)
        hit_rank_ic = bool(((rdf["frequency_setting"] == frequency) & (rdf["horizon"] == horizon)).any())
    if history_path.exists():
        hdf = pd.read_csv(history_path)
        hit_history = bool(((hdf["frequency_setting"] == frequency) & (hdf["horizon"] == horizon)).any())

    hits = {"summary": hit_summary, "predictions": hit_pred, "rank_ic": hit_rank_ic, "history": hit_history}
    if any(hits.values()) and not all(hits.values()):
        raise ValueError(
            f"Requested cell {frequency}/{horizon}/seed={seed} has inconsistent partial presence across raw artifacts: {hits}. "
            "Manual inspection is required before resume."
        )
    if all(hits.values()):
        raise ValueError(
            f"Requested cell already exists in formal raw evidence ({frequency}/{horizon}/seed={seed}). "
            "Refusing to retrain or overwrite completed evidence."
        )


def validate_summary_schema_cpu(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    required = {
        "model", "frequency_setting", "horizon", "seed", "n_trainable_parameters", "best_epoch",
        "epochs_trained", "runtime_seconds", "val_loss", "test_mse", "test_mae", "test_corr",
        "ticker_avg_mse", "ticker_avg_mae", "ticker_avg_corr", "mean_rank_ic", "median_rank_ic",
        "rank_ic_std", "positive_ic_ratio", "rank_icir", "n_valid_ic_dates", "direction_accuracy",
        "pred_std", "true_std", "pred_std_true_std_ratio", "no_extreme_scale_pathology",
    }
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"CPU summary artifact missing required columns: {missing}")

    finite_cols = [
        "n_trainable_parameters", "best_epoch", "epochs_trained", "runtime_seconds", "val_loss",
        "test_mse", "test_mae", "ticker_avg_mse", "ticker_avg_mae", "n_valid_ic_dates",
        "direction_accuracy", "pred_std", "true_std", "pred_std_true_std_ratio",
    ]
    allow_nan_cols = ["test_corr", "ticker_avg_corr", "mean_rank_ic", "median_rank_ic", "rank_ic_std", "positive_ic_ratio", "rank_icir"]

    for col in finite_cols:
        bad = pd.to_numeric(summary_df[col], errors="coerce")
        if bad.isna().any():
            raise ValueError(f"CPU summary has NaN in required-finite column: {col}")
        if np.isinf(bad.to_numpy()).any():
            raise ValueError(f"CPU summary has ±Inf in required-finite column: {col}")

    for col in allow_nan_cols:
        bad = pd.to_numeric(summary_df[col], errors="coerce")
        if np.isinf(bad.to_numpy()).any():
            raise ValueError(f"CPU summary has ±Inf in correlation/Rank-IC column: {col}")

    if summary_df["model"].nunique() != 1 or summary_df["model"].iloc[0] != "swim_transformer":
        raise ValueError("CPU summary model contract failed.")
    if summary_df["seed"].nunique() != 1 or int(summary_df["seed"].iloc[0]) != 42:
        raise ValueError("CPU summary seed contract failed.")
    if summary_df.duplicated(subset=["frequency_setting", "horizon"]).any():
        raise ValueError("CPU summary has duplicate (frequency_setting, horizon) rows.")

    for _, row in summary_df.iterrows():
        freq = str(row["frequency_setting"])
        expected = EXPECTED_SWIM_PARAM_COUNTS.get(freq, None)
        if expected is None:
            continue
        observed = int(row["n_trainable_parameters"])
        if observed != int(expected):
            raise ValueError(f"CPU parameter count contract failed for {freq}: expected {expected}, got {observed}")


def validate_prediction_rows_cpu(pred_df: pd.DataFrame) -> None:
    required = {"model", "ticker", "anchor_date", "horizon", "frequency_setting", "y_true", "y_pred", "seed", "best_epoch"}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise ValueError(f"CPU prediction artifact missing required columns: {missing}")
    if pred_df.empty:
        raise ValueError("CPU prediction artifact cannot be empty.")
    key_cols = ["model", "ticker", "anchor_date", "horizon", "frequency_setting", "seed", "best_epoch"]
    if pred_df.duplicated(subset=key_cols).any():
        raise ValueError("CPU prediction artifact contains duplicate formal keys.")
    if not np.isfinite(pd.to_numeric(pred_df["y_true"], errors="coerce").to_numpy()).all():
        raise ValueError("CPU prediction artifact contains non-finite y_true values.")
    if not np.isfinite(pd.to_numeric(pred_df["y_pred"], errors="coerce").to_numpy()).all():
        raise ValueError("CPU prediction artifact contains non-finite y_pred values.")
    if set(pred_df["model"].astype(str).tolist()) != {"swim_transformer"}:
        raise ValueError("CPU predictions model contract failed.")
    if set(pred_df["seed"].astype(int).tolist()) != {42}:
        raise ValueError("CPU predictions seed contract failed.")


def validate_rank_ic_rows_cpu(rank_ic_df: pd.DataFrame) -> None:
    required = {"model", "frequency_setting", "horizon", "anchor_date", "n_tickers", "rank_ic"}
    missing = sorted(required - set(rank_ic_df.columns))
    if missing:
        raise ValueError(f"CPU Rank-IC artifact missing required columns: {missing}")
    if rank_ic_df.empty:
        raise ValueError("CPU Rank-IC artifact cannot be empty.")
    if rank_ic_df.duplicated(subset=["model", "frequency_setting", "horizon", "anchor_date"]).any():
        raise ValueError("CPU Rank-IC artifact contains duplicate formal keys.")
    if np.isinf(pd.to_numeric(rank_ic_df["rank_ic"], errors="coerce").to_numpy()).any():
        raise ValueError("CPU Rank-IC artifact contains ±Inf rank_ic values.")
    if (pd.to_numeric(rank_ic_df["n_tickers"], errors="coerce") != 17).any():
        raise ValueError("CPU Rank-IC artifact has invalid n_tickers values.")
    if set(rank_ic_df["model"].astype(str).tolist()) != {"swim_transformer"}:
        raise ValueError("CPU Rank-IC model contract failed.")


def validate_training_history_cpu(history_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    required = {"frequency_setting", "horizon", "epoch", "train_loss", "val_loss", "elapsed_seconds"}
    missing = sorted(required - set(history_df.columns))
    if missing:
        raise ValueError(f"CPU training history artifact missing required columns: {missing}")
    if history_df.empty:
        raise ValueError("CPU training history artifact cannot be empty.")
    if not np.isfinite(pd.to_numeric(history_df["train_loss"], errors="coerce").to_numpy()).all():
        raise ValueError("CPU training history contains non-finite train_loss values.")
    if not np.isfinite(pd.to_numeric(history_df["val_loss"], errors="coerce").to_numpy()).all():
        raise ValueError("CPU training history contains non-finite val_loss values.")
    if (history_df["epoch"] < 1).any() or (history_df["epoch"] > 100).any():
        raise ValueError("CPU history epoch range contract failed.")
    if history_df.duplicated(subset=["frequency_setting", "horizon", "epoch"]).any():
        raise ValueError("CPU training history has duplicate (frequency_setting, horizon, epoch) keys.")
    if summary_df.empty:
        return
    for _, row in summary_df.iterrows():
        cell = history_df[(history_df["frequency_setting"] == row["frequency_setting"]) & (history_df["horizon"] == row["horizon"])].copy()
        if cell.empty:
            raise ValueError(f"Missing training history for CPU cell: {row['frequency_setting']} / {row['horizon']}")
        cell = cell.sort_values("epoch").reset_index(drop=True)
        epochs = cell["epoch"].astype(int).tolist()
        expected_epochs = list(range(1, int(row["epochs_trained"]) + 1))
        if epochs != expected_epochs:
            raise ValueError(f"CPU history epochs for {row['frequency_setting']} / {row['horizon']} are not contiguous 1..epochs_trained: got {epochs}, expected {expected_epochs}")
        if int(row["best_epoch"]) not in epochs:
            raise ValueError(f"CPU summary best_epoch not found in history for {row['frequency_setting']} / {row['horizon']}")
        min_loss = float(cell["val_loss"].min())
        min_loss_epochs = set(cell.loc[np.isclose(cell["val_loss"].astype(float), min_loss, rtol=1e-6, atol=1e-8), "epoch"].astype(int).tolist())
        if int(row["best_epoch"]) not in min_loss_epochs:
            raise ValueError(f"CPU summary best_epoch is not a min-val-loss epoch for {row['frequency_setting']} / {row['horizon']}")
        if not np.isclose(float(row["val_loss"]), min_loss, rtol=1e-6, atol=1e-8):
            raise ValueError(f"CPU summary val_loss does not match min validation loss for {row['frequency_setting']} / {row['horizon']}")


def validate_cross_frequency_y_true_identity_cpu(pred_df: pd.DataFrame) -> None:
    key = ["ticker", "anchor_date", "horizon"]
    grouped = pred_df.groupby(key, dropna=False)
    for _, group in grouped:
        values = group["y_true"].drop_duplicates().tolist()
        if len(values) > 1:
            raise ValueError("Cross-frequency y_true identity failed: identical horizon/ticker/date must share one target label across frequencies.")


def validate_partial_cpu_swim_evidence(output_dir: Path) -> None:
    output_dir = resolve_output_dir(output_dir)
    summary_path = output_dir / FORMAL_SWIM_RUN_FILES["summary"]
    pred_path = output_dir / FORMAL_SWIM_RUN_FILES["predictions"]
    rank_ic_path = output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]
    history_path = output_dir / FORMAL_SWIM_RUN_FILES["history"]
    if not any(path.exists() for path in [summary_path, pred_path, rank_ic_path, history_path]):
        return
    if not all(path.exists() for path in [summary_path, pred_path, rank_ic_path, history_path]):
        raise ValueError("Partial CPU SWiM raw evidence is inconsistent: some raw artifacts exist without their full set.")

    summary_df = pd.read_csv(summary_path)
    pred_df = pd.read_csv(pred_path)
    rank_ic_df = pd.read_csv(rank_ic_path)
    history_df = pd.read_csv(history_path)

    validate_summary_schema_cpu(summary_df)
    validate_prediction_rows_cpu(pred_df)
    validate_rank_ic_rows_cpu(rank_ic_df)
    validate_training_history_cpu(history_df, summary_df)
    validate_cross_frequency_y_true_identity_cpu(pred_df)


def validate_cpu_main_v2_swim_evidence(output_dir: Path) -> None:
    output_dir = resolve_output_dir(output_dir)
    summary_path = output_dir / FORMAL_SWIM_RUN_FILES["summary"]
    pred_path = output_dir / FORMAL_SWIM_RUN_FILES["predictions"]
    rank_ic_path = output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]
    history_path = output_dir / FORMAL_SWIM_RUN_FILES["history"]
    if not summary_path.exists() or not pred_path.exists() or not rank_ic_path.exists() or not history_path.exists():
        raise FileNotFoundError(f"CPU formal evidence incomplete in {output_dir}")

    summary_df = pd.read_csv(summary_path)
    pred_df = pd.read_csv(pred_path)
    rank_ic_df = pd.read_csv(rank_ic_path)
    history_df = pd.read_csv(history_path)

    if len(summary_df) != 9:
        raise ValueError(f"CPU summary validation failed: expected 9 rows, got {len(summary_df)}")
    validate_summary_schema_cpu(summary_df)
    if set(summary_df["frequency_setting"]) != {"daily_only", "weekly_only", "daily_weekly"}:
        raise ValueError("CPU summary frequencies mismatch.")
    if set(summary_df["horizon"]) != {"5d", "10d", "20d"}:
        raise ValueError("CPU summary horizons mismatch.")
    if len(pred_df) != 59823:
        raise ValueError(f"CPU prediction row count mismatch: expected 59823, got {len(pred_df)}")
    pred_cell_groups = pred_df.groupby(["frequency_setting", "horizon"])
    if len(pred_cell_groups) != 9:
        raise ValueError("CPU predictions do not cover all 9 formal cells.")
    for (freq, horizon), cell_df in pred_cell_groups:
        if len(cell_df) != 6647:
            raise ValueError(f"CPU prediction cell {freq}/{horizon} has wrong row count: expected 6647, got {len(cell_df)}")
        if cell_df["ticker"].nunique() != 17:
            raise ValueError(f"CPU prediction cell {freq}/{horizon} does not contain 17 tickers: {cell_df['ticker'].nunique()}")
        if cell_df["anchor_date"].nunique() != 391:
            raise ValueError(f"CPU prediction cell {freq}/{horizon} does not contain 391 anchor dates: {cell_df['anchor_date'].nunique()}")
        if set(cell_df["seed"].astype(int).tolist()) != {42}:
            raise ValueError(f"CPU prediction cell {freq}/{horizon} seed contract failed.")
        if set(cell_df["model"].astype(str).tolist()) != {"swim_transformer"}:
            raise ValueError(f"CPU prediction cell {freq}/{horizon} model contract failed.")
    validate_prediction_rows_cpu(pred_df)

    if len(rank_ic_df) != 3519:
        raise ValueError(f"CPU Rank-IC row count mismatch: expected 3519, got {len(rank_ic_df)}")
    if rank_ic_df.duplicated(subset=["model", "frequency_setting", "horizon", "anchor_date"]).any():
        raise ValueError("CPU Rank-IC artifact contains duplicate formal keys.")
    if (pd.to_numeric(rank_ic_df["n_tickers"], errors="coerce") != 17).any():
        raise ValueError("CPU Rank-IC n_tickers contract failed: every row must have n_tickers == 17.")
    rank_ic_cell_groups = rank_ic_df.groupby(["frequency_setting", "horizon"])
    if not np.all(rank_ic_cell_groups.size() == 391):
        raise ValueError("CPU Rank-IC per-cell row count contract failed: expected 391 rows per cell.")
    validate_rank_ic_rows_cpu(rank_ic_df)

    if len(history_df) < 1:
        raise ValueError("CPU training history artifact cannot be empty.")
    if history_df["frequency_setting"].nunique() != 3 or history_df["horizon"].nunique() != 3:
        raise ValueError("CPU history does not cover all 9 cells.")
    validate_training_history_cpu(history_df, summary_df)
    validate_cross_frequency_y_true_identity_cpu(pred_df)


def build_cpu_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    naive_path = ROOT / "dataset" / "audit" / "main_v2" / "naive" / "panel_naive_baseline_summary.csv"
    ridge_path = ROOT / "dataset" / "audit" / "main_v2" / "ridge" / "panel_ridge_summary_metrics.csv"
    lstm_path = ROOT / "dataset" / "audit" / "main_v2" / "lstm" / "panel_lstm_summary_metrics.csv"
    vanilla_path = ROOT / "dataset" / "audit" / "main_v2" / "vanilla" / "panel_vanilla_summary_metrics.csv"
    missing = [name for name, path in {"naive": naive_path, "ridge": ridge_path, "lstm": lstm_path, "vanilla": vanilla_path}.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing canonical MAIN V2 baseline artifacts required for CPU formal comparison: " + ", ".join(missing))

    naive_df = formal.canonicalize_main_v2_naive(pd.read_csv(naive_path))
    ridge_df = formal.canonicalize_main_v2_ridge(pd.read_csv(ridge_path))
    lstm_df = formal.canonicalize_main_v2_lstm_or_vanilla(pd.read_csv(lstm_path), label="lstm", model_value="lstm")
    vanilla_df = formal.canonicalize_main_v2_lstm_or_vanilla(pd.read_csv(vanilla_path), label="vanilla", model_value=None)

    out_rows = []
    for _, row in summary_df.iterrows():
        freq = row["frequency_setting"]
        horizon = row["horizon"]
        horizon_norm = str(horizon).strip().replace("d", "")
        joined = {
            "model": row["model"],
            "frequency_setting": freq,
            "horizon": horizon,
            "seed": row["seed"],
            "n_trainable_parameters": row["n_trainable_parameters"],
            "test_mse": row["test_mse"],
            "test_mae": row["test_mae"],
            "test_corr": row["test_corr"],
            "mean_rank_ic": row["mean_rank_ic"],
            "direction_accuracy": row["direction_accuracy"],
            "pred_std_true_std_ratio": row["pred_std_true_std_ratio"],
        }

        naive_matched = naive_df[naive_df["horizon"] == horizon_norm].copy()
        if naive_matched.empty:
            raise ValueError(f"Naive baseline is missing horizon {horizon}.")
        naive_matched["model"] = naive_matched["model"].astype(str).str.lower()
        zero_row = naive_matched[naive_matched["model"] == "zero"]
        train_mean_row = naive_matched[naive_matched["model"] == "train_mean"]
        if zero_row.empty or train_mean_row.empty:
            raise ValueError(f"Naive baseline missing zero/train_mean rows for horizon {horizon}.")
        joined["zero_mse"] = float(zero_row["pooled_mse"].iloc[0])
        joined["train_mean_mse"] = float(train_mean_row["pooled_mse"].iloc[0])
        joined["best_naive_mse"] = float(np.min([joined["zero_mse"], joined["train_mean_mse"]]))

        ridge_matched = ridge_df[(ridge_df["frequency_setting"] == freq) & (ridge_df["horizon"] == horizon_norm)]
        if ridge_matched.empty:
            raise ValueError(f"Ridge baseline missing row for {freq}/{horizon}.")
        joined["ridge_mse"] = float(ridge_matched["pooled_mse"].iloc[0])
        joined["ridge_mae"] = float(ridge_matched["pooled_mae"].iloc[0])
        joined["ridge_corr"] = float(ridge_matched["pooled_corr"].iloc[0])
        joined["ridge_mean_rank_ic"] = float(ridge_matched["mean_rank_ic"].iloc[0])

        lstm_matched = lstm_df[(lstm_df["frequency_setting"] == freq) & (lstm_df["horizon"] == horizon_norm)]
        if lstm_matched.empty:
            raise ValueError(f"LSTM baseline missing row for {freq}/{horizon}.")
        joined["lstm_mse"] = float(lstm_matched["test_mse"].iloc[0])
        joined["lstm_mae"] = float(lstm_matched["test_mae"].iloc[0])
        joined["lstm_corr"] = float(lstm_matched["test_corr"].iloc[0])
        joined["lstm_mean_rank_ic"] = float(lstm_matched["mean_rank_ic"].iloc[0])

        vanilla_matched = vanilla_df[(vanilla_df["frequency_setting"] == freq) & (vanilla_df["horizon"] == horizon_norm)]
        if vanilla_matched.empty:
            raise ValueError(f"Vanilla baseline missing row for {freq}/{horizon}.")
        joined["vanilla_mse"] = float(vanilla_matched["test_mse"].iloc[0])
        joined["vanilla_mae"] = float(vanilla_matched["test_mae"].iloc[0])
        joined["vanilla_corr"] = float(vanilla_matched["test_corr"].iloc[0])
        joined["vanilla_mean_rank_ic"] = float(vanilla_matched["mean_rank_ic"].iloc[0])

        out_rows.append(joined)
    return pd.DataFrame(out_rows)


def write_cpu_run_manifest(output_dir: Path, summary_df: pd.DataFrame, pred_df: pd.DataFrame, rank_ic_df: pd.DataFrame, history_df: pd.DataFrame, training_script_sha256: str, finalization_time_script_sha256: str) -> Path:
    output_dir = resolve_output_dir(output_dir)
    provenance_path = output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"]
    if not provenance_path.exists():
        raise FileNotFoundError(f"CPU training provenance missing: {provenance_path}")
    provenance_df = pd.read_csv(provenance_path)
    if provenance_df.empty:
        raise ValueError(f"CPU training provenance file is empty: {provenance_path}")
    row = provenance_df.iloc[0]
    validate_cpu_training_provenance_row(row)
    validate_cpu_main_v2_swim_evidence(output_dir)

    manifest = {
        "run_family": "MAIN_V2",
        "model": "SWIM_STYLE_TRANSFORMER",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "dataset_audit_status": "PASS",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "formal_tickers": 17,
        "common_anchor_dates": 2563,
        "nominal_seed": 42,
        "expected_cells": 9,
        "completed_cells": int(len(summary_df)),
        "summary_rows": int(len(summary_df)),
        "prediction_rows": int(len(pred_df)),
        "rank_ic_rows": int(len(rank_ic_df)),
        "training_history_rows": int(len(history_df)),
        "training_script": str(row.get("training_script", "scripts/python/panel_baseline_swim_cpu.py")),
        "training_script_sha256": str(training_script_sha256),
        "finalization_time_script_sha256": str(finalization_time_script_sha256),
        "output_dir": str(output_dir.resolve()),
        "source_frozen_script": str(row.get("source_frozen_script", "scripts/python/panel_baseline_swim.py")),
        "source_frozen_script_sha256": str(row.get("source_frozen_script_sha256", "30a6e332ed34f012a05c27134dda33e1e575715fdf7585c57d731a3de065129a")),
        "execution_backend": "CPU",
        "parent_frozen_script_sha256": str(row.get("source_frozen_script_sha256", "30a6e332ed34f012a05c27134dda33e1e575715fdf7585c57d731a3de065129a")),
        "summary_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["summary"]),
        "predictions_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["predictions"]),
        "rank_ic_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]),
        "training_history_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["history"]),
        "comparison_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["comparison"]),
        "historical_results_used_for_training": False,
        "historical_results_used_for_model_selection": False,
        "historical_results_used_for_formal_comparison": False,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_swim_blocks": NUM_BLOCKS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "activation": "relu",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "max_norm": 1.0,
    }
    manifest_path = output_dir / FORMAL_SWIM_RUN_FILES["manifest"]
    pd.DataFrame([manifest]).to_csv(manifest_path, index=False)
    return manifest_path


def write_cpu_final_summary(output_dir: Path, summary_df: pd.DataFrame, comparison_df: pd.DataFrame, training_script_sha256: str, finalization_time_script_sha256: str) -> Path:
    output_dir = resolve_output_dir(output_dir)
    lines = []
    lines.append("MAIN V2")
    lines.append("POST-TEMPORAL-FIX")
    lines.append("Dataset Contract V2")
    lines.append("SWiM-style Transformer")
    lines.append("nominal seed = 42")
    lines.append("execution backend = CPU")
    lines.append("formal namespace = dataset/audit/main_v2/swim_cpu")
    lines.append("")
    lines.append("Hashes:")
    lines.append(f"  dataset manifest SHA = {DATASET_MANIFEST_SHA256}")
    lines.append(f"  training-time CPU script SHA = {training_script_sha256}")
    lines.append(f"  finalization-time CPU script SHA = {finalization_time_script_sha256}")
    lines.append(f"  parent frozen SWiM script SHA = 30a6e332ed34f012a05c27134dda33e1e575715fdf7585c57d731a3de065129a")
    lines.append("")
    lines.append("Architecture:")
    lines.append("  - exact local + shifted-window architecture")
    lines.append("  - Daily window=10 shift=5")
    lines.append("  - Weekly window=4 shift=2")
    lines.append("  - Post-LN")
    lines.append("  - fixed sinusoidal PE")
    lines.append("  - masked mean pooling")
    lines.append("  - separate branches + late concat")
    lines.append("  - no router / adaptive scale selection")
    lines.append("")
    lines.append("Parameter counts:")
    for key, val in EXPECTED_SWIM_PARAM_COUNTS.items():
        lines.append(f"  - {key} = {val}")
    final_path = output_dir / FORMAL_SWIM_RUN_FILES["final_summary"]
    final_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final_path


def save_summary_rows_cpu(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir)
    rows = []
    for rr in results:
        rows.append({
            "model": "swim_transformer",
            "frequency_setting": rr["frequency_setting"],
            "horizon": rr["horizon"],
            "seed": rr["seed"],
            "n_trainable_parameters": rr.get("n_trainable_parameters", np.nan),
            "best_epoch": rr["best_epoch"],
            "epochs_trained": rr.get("epochs_trained", rr["best_epoch"]),
            "runtime_seconds": rr.get("runtime_seconds", np.nan),
            "val_loss": rr["val_loss"],
            "test_mse": rr["test_mse"],
            "test_mae": rr["test_mae"],
            "test_corr": rr["test_corr"],
            "ticker_avg_mse": rr["ticker_avg_mse"],
            "ticker_avg_mae": rr["ticker_avg_mae"],
            "ticker_avg_corr": rr["ticker_avg_corr"],
            "mean_rank_ic": rr["mean_rank_ic"],
            "median_rank_ic": rr["median_rank_ic"],
            "rank_ic_std": rr["rank_ic_std"],
            "positive_ic_ratio": rr["positive_ic_ratio"],
            "rank_icir": rr["rank_icir"],
            "n_valid_ic_dates": rr.get("n_valid_ic_dates", np.nan),
            "direction_accuracy": rr["direction_accuracy"],
            "pred_std": rr["pred_std"],
            "true_std": rr["true_std"],
            "pred_std_true_std_ratio": rr["pred_std_true_std_ratio"],
            "no_extreme_scale_pathology": rr.get("no_extreme_scale_pathology", np.nan),
        })
    df = pd.DataFrame(rows)
    path = output_dir / FORMAL_SWIM_RUN_FILES["summary"]
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_prediction_rows_cpu(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir)
    rows = []
    for rr in results:
        for i in range(len(rr["ticker"])):
            rows.append({
                "ticker": rr["ticker"][i],
                "anchor_date": rr["anchor_date"][i],
                "horizon": rr["horizon"],
                "frequency_setting": rr["frequency_setting"],
                "y_true": rr["y_true"][i],
                "y_pred": rr["y_pred"][i],
                "seed": rr["seed"],
                "best_epoch": rr["best_epoch"],
                "model": "swim_transformer",
            })
    df = pd.DataFrame(rows)
    path = output_dir / FORMAL_SWIM_RUN_FILES["predictions"]
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_training_history_cpu(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir)
    rows = []
    for rr in results:
        for epoch_row in rr["history"]:
            rows.append({
                "frequency_setting": epoch_row["frequency_setting"],
                "horizon": epoch_row["horizon"],
                "epoch": epoch_row["epoch"],
                "train_loss": epoch_row["train_loss"],
                "val_loss": epoch_row["val_loss"],
                "elapsed_seconds": epoch_row.get("elapsed_seconds", np.nan),
            })
    df = pd.DataFrame(rows)
    path = output_dir / FORMAL_SWIM_RUN_FILES["history"]
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_rank_ic_by_date_cpu(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir)
    rows = []
    for res in results:
        df = res["rank_ic_by_date"].copy()
        df["frequency_setting"] = res["frequency_setting"]
        df["horizon"] = res["horizon"]
        df["model"] = "swim_transformer"
        rows.append(df.reindex(columns=["model", "frequency_setting", "horizon", "anchor_date", "n_tickers", "rank_ic"]))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    path = output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]
    if append_mode and path.exists():
        out.to_csv(path, mode="a", header=False, index=False)
    else:
        out.to_csv(path, index=False)
    return out


def train_and_eval_frequency_cpu(ds, frequency: str, horizon: int, stats: dict, trace_nan: bool = False) -> dict:
    set_seed(SEED)
    device = torch.device("cpu")
    experiment_start = time.perf_counter()

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    if frequency == "daily_only":
        X_all = apply_normalizer(ds, stats, "daily_only")
        X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
        X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
        X_test = torch.tensor(X_all[test_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        y_test = ds.y[horizon][test_mask].copy()
        model = formal.SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="daily").to(device)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            return model(x_batch), y_batch

    elif frequency == "weekly_only":
        X_all = apply_normalizer(ds, stats, "weekly_only")
        X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
        X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
        X_test = torch.tensor(X_all[test_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        y_test = ds.y[horizon][test_mask].copy()
        model = formal.SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="weekly").to(device)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            return model(x_batch), y_batch

    elif frequency == "daily_weekly":
        X_daily_all, X_weekly_all = apply_normalizer(ds, stats, "daily_weekly")
        X_d_train = torch.tensor(X_daily_all[train_mask], dtype=torch.float32)
        X_d_val = torch.tensor(X_daily_all[val_mask], dtype=torch.float32)
        X_d_test = torch.tensor(X_daily_all[test_mask], dtype=torch.float32)
        X_w_train = torch.tensor(X_weekly_all[train_mask], dtype=torch.float32)
        X_w_val = torch.tensor(X_weekly_all[val_mask], dtype=torch.float32)
        X_w_test = torch.tensor(X_weekly_all[test_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        y_test = ds.y[horizon][test_mask].copy()
        model = formal.DualFrequencySWiMRegressor(
            daily_feature_dim=X_daily_all.shape[-1],
            weekly_feature_dim=X_weekly_all.shape[-1],
        ).to(device)
        train_loader = DataLoader(TensorDataset(X_d_train, X_w_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_d_val, X_w_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_daily, x_weekly, y_batch):
            x_daily = x_daily.to(device)
            x_weekly = x_weekly.to(device)
            y_batch = y_batch.to(device)
            return model(x_daily, x_weekly), y_batch

    else:
        raise ValueError(f"Unknown frequency setting: {frequency}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    history_rows = []
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0
        epoch_start = time.perf_counter()

        if frequency == "daily_weekly":
            for batch_idx, (x_d_batch, x_w_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                loss = criterion(logits, target)
                batch_size_value = y_batch.shape[0]
                train_loss_sum += float(loss.item()) * batch_size_value
                train_weight += int(batch_size_value)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                logits, target = batched_forward(x_batch, y_batch)
                loss = criterion(logits, target)
                batch_size_value = y_batch.shape[0]
                train_loss_sum += float(loss.item()) * batch_size_value
                train_weight += int(batch_size_value)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_weight = 0
        with torch.no_grad():
            if frequency == "daily_weekly":
                for x_d_batch, x_w_batch, y_batch in val_loader:
                    logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                    batch_loss = criterion(logits, target)
                    batch_size_value = y_batch.shape[0]
                    val_loss_sum += float(batch_loss.item()) * batch_size_value
                    val_weight += int(batch_size_value)
            else:
                for x_batch, y_batch in val_loader:
                    logits, target = batched_forward(x_batch, y_batch)
                    batch_loss = criterion(logits, target)
                    batch_size_value = y_batch.shape[0]
                    val_loss_sum += float(batch_loss.item()) * batch_size_value
                    val_weight += int(batch_size_value)

        train_loss = train_loss_sum / train_weight if train_weight else float("nan")
        val_loss = val_loss_sum / val_weight if val_weight else float("nan")
        if (not np.isfinite(train_loss)) or (not np.isfinite(val_loss)):
            raise RuntimeError(
                f"Non-finite loss detected for {frequency} / {horizon}d at epoch {epoch}: "
                f"train_loss={train_loss}, val_loss={val_loss}"
            )

        epochs_trained = epoch
        history_rows.append({
            "frequency_setting": frequency,
            "horizon": f"{horizon}d",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_seconds": time.perf_counter() - epoch_start,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No valid training state for {frequency}, horizon={horizon}d")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if frequency == "daily_weekly":
            test_loader = DataLoader(TensorDataset(X_d_test, X_w_test, torch.tensor(y_test, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
            preds = []
            for x_d_batch, x_w_batch, _ in test_loader:
                preds.append(model(x_d_batch.to(device), x_w_batch.to(device)).cpu().numpy())
            y_pred = np.concatenate(preds)
        else:
            test_loader = DataLoader(TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
            preds = []
            for x_batch, _ in test_loader:
                preds.append(model(x_batch.to(device)).cpu().numpy())
            y_pred = np.concatenate(preds)

    if not np.all(np.isfinite(y_pred)):
        raise RuntimeError(f"Non-finite test predictions detected for {frequency} / {horizon}d")

    sanity_metric = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        ticker=ds.ticker[test_mask],
        anchor_date=ds.anchor_date[test_mask],
    )
    ic_df = rank_ic_by_date(y_test, y_pred, ds.ticker[test_mask], ds.anchor_date[test_mask])
    rank_summary = compute_rank_ic_summary(ic_df)
    no_extreme_scale_pathology = prediction_scale_looks_normal(
        sanity_metric["pooled_pred_std"], sanity_metric["pooled_true_std"]
    )
    experiment_elapsed = time.perf_counter() - experiment_start

    return {
        "model": "swim_transformer",
        "frequency_setting": frequency,
        "horizon": f"{horizon}d",
        "seed": SEED,
        "n_trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "runtime_seconds": experiment_elapsed,
        "val_loss": best_val_loss,
        "test_mse": sanity_metric["pooled_mse"],
        "test_mae": sanity_metric["pooled_mae"],
        "test_corr": sanity_metric["pooled_corr"],
        "ticker_avg_mse": sanity_metric["ticker_avg_mse"],
        "ticker_avg_mae": sanity_metric["ticker_avg_mae"],
        "ticker_avg_corr": sanity_metric["ticker_avg_corr"],
        "direction_accuracy": sanity_metric["pooled_da"],
        "pred_std": sanity_metric["pooled_pred_std"],
        "true_std": sanity_metric["pooled_true_std"],
        "pred_std_true_std_ratio": sanity_metric["pooled_pred_std_over_true_std"],
        "mean_rank_ic": rank_summary["mean_rank_ic"],
        "median_rank_ic": rank_summary["median_rank_ic"],
        "rank_ic_std": rank_summary["rank_ic_std"],
        "positive_ic_ratio": rank_summary["positive_ic_ratio"],
        "rank_icir": rank_summary["rank_icir"],
        "n_valid_ic_dates": rank_summary["n_valid_ic_dates"],
        "y_pred": y_pred,
        "y_true": y_test,
        "ticker": ds.ticker[test_mask],
        "anchor_date": ds.anchor_date[test_mask],
        "rank_ic_by_date": ic_df,
        "history": history_rows,
        "no_extreme_scale_pathology": no_extreme_scale_pathology,
    }


def full_benchmark_cpu(ds, stats: dict, output_dir: str | Path | None = None) -> None:
    output_dir = resolve_output_dir(output_dir)

    assert_clean_cpu_swim_namespace(output_dir)
    write_cpu_training_provenance(output_dir, Path(__file__))

    for frequency in FREQUENCY_SETTINGS:
        for horizon in HORIZONS:
            horizon_int = int(horizon)
            horizon_label = f"{horizon_int}d"

            validate_partial_raw_cell_set_consistency_cpu(output_dir)
            ensure_requested_cell_absent_cpu(output_dir, frequency, horizon_label, seed=SEED)

            result = train_and_eval_frequency_cpu(ds, frequency, horizon_int, stats)

            save_summary_rows_cpu([result], output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["summary"]).exists())
            save_prediction_rows_cpu([result], output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["predictions"]).exists())
            save_training_history_cpu([result], output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["history"]).exists())
            save_rank_ic_by_date_cpu([result], output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]).exists())

            validate_partial_raw_cell_set_consistency_cpu(output_dir)
            validate_partial_cpu_swim_evidence(output_dir)

    validate_cpu_main_v2_swim_evidence(output_dir)

    provenance_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"])
    training_sha = str(provenance_df.iloc[0]["training_script_sha256"])
    current_sha = compute_sha256(Path(__file__))
    if training_sha != current_sha:
        raise RuntimeError(f"CPU provenance training SHA mismatch: stored {training_sha}, current {current_sha}")

    summary_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["summary"])
    pred_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["predictions"])
    rank_ic_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"])
    history_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["history"])

    comparison_df = build_cpu_comparison(summary_df)
    comparison_df.to_csv(output_dir / FORMAL_SWIM_RUN_FILES["comparison"], index=False)

    write_cpu_run_manifest(
        output_dir,
        summary_df,
        pred_df,
        rank_ic_df,
        history_df,
        training_script_sha256=training_sha,
        finalization_time_script_sha256=current_sha,
    )
    write_cpu_final_summary(
        output_dir,
        summary_df,
        comparison_df,
        training_script_sha256=training_sha,
        finalization_time_script_sha256=current_sha,
    )


def run_cpu_smoke_validation() -> None:
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    formal.validate_frozen_dataset_contract(ds)
    stats = fit_train_only_normalizer(ds)
    device = torch.device("cpu")

    pe = formal.PositionalEncoding(d_model=D_MODEL, max_len=formal.FIXED_MAX_LEN, dropout=0.0)
    x = torch.zeros(2, 10, D_MODEL)
    out = pe(x)
    assert out.shape == x.shape, f"unexpected PE output shape: {out.shape}"

    for T, w in [(90, 10), (28, 4)]:
        x_rand = torch.randn(3, T, 6)
        windows = formal.window_partition_1d(x_rand, w)
        recon = formal.window_reverse_1d(windows, w, B=3, T=T)
        max_err = (recon - x_rand).abs().max().item()
        assert max_err < 1e-6, f"reconstruction error too large for T={T}, w={w}: {max_err}"

    T_pad, w, shift = 20, 10, 5
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    attn_mask = formal.build_window_masks(valid_mask, w, shift)
    original = torch.arange(T_pad)
    shifted = torch.roll(original, shifts=-shift)
    expected_shifted = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4])
    assert torch.equal(shifted, expected_shifted), f"unexpected shifted index order: {shifted.tolist()}"
    win1 = attn_mask[1]
    assert torch.all(win1[:5, :5] == 0.0), "same-region valid pairs must remain exactly 0.0"
    assert torch.all(win1[5:, 5:] == 0.0), "same-region valid pairs must remain exactly 0.0"
    assert torch.all(torch.isinf(win1[:5, 5:])), "real-query cross-region pairs must be exactly -inf"
    assert torch.all(torch.isinf(win1[5:, :5])), "real-query cross-region pairs must be exactly -inf"
    for row in win1:
        assert torch.any(torch.isfinite(row)), "no real-query attention row may be entirely -inf"

    T_pad2, w2, shift2 = 28, 4, 2
    valid_mask2 = torch.ones(T_pad2, dtype=torch.bool)
    valid_mask2[24:] = False
    shifted_validity = torch.roll(valid_mask2, shifts=-shift2)
    expected_validity = torch.tensor([False, False, True, True], dtype=torch.bool)
    assert torch.equal(shifted_validity[-w2:], expected_validity), f"unexpected final weekly shifted validity: {shifted_validity[-w2:].tolist()}"
    attn_mask2 = formal.build_window_masks(valid_mask2, w2, shift2)
    last_window = attn_mask2[6]
    assert torch.all(last_window[0, 0] == 0.0), "first padded query in final weekly window must retain dummy key"
    assert torch.all(torch.isinf(last_window[0, 1:])), "first padded query must not attend real keys"
    assert torch.all(last_window[2, 2:] == 0.0), "real query at original token 0 must retain its valid same-region keys"
    assert torch.all(torch.isinf(last_window[2, :2])), "real query at original token 0 must not attend padded keys"
    assert torch.all(torch.any(torch.isfinite(last_window), dim=-1)), "every final-window query row must retain at least one finite entry"

    T, w, shift = 26, 4, 2
    pad_len = formal.compute_pad_len(T, w)
    T_pad = T + pad_len
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    valid_mask[T:] = False
    shifted_validity = torch.roll(valid_mask, shifts=-shift)
    assert torch.equal(shifted_validity[-w:], torch.tensor([False, False, True, True], dtype=torch.bool))

    X_daily = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:6], dtype=torch.float32, device=device)
    X_weekly = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:6], dtype=torch.float32, device=device)
    daily_model = formal.SingleFrequencySWiMRegressor(feature_dim=X_daily.shape[-1], frequency_key="daily").to(device)
    weekly_model = formal.SingleFrequencySWiMRegressor(feature_dim=X_weekly.shape[-1], frequency_key="weekly").to(device)
    dual_model = formal.DualFrequencySWiMRegressor(daily_feature_dim=X_daily.shape[-1], weekly_feature_dim=X_weekly.shape[-1]).to(device)
    out_daily = daily_model(X_daily)
    out_weekly = weekly_model(X_weekly)
    out_dual = dual_model(X_daily, X_weekly)
    assert out_daily.shape == (6,), f"daily_only output shape: {out_daily.shape}"
    assert out_weekly.shape == (6,), f"weekly_only output shape: {out_weekly.shape}"
    assert out_dual.shape == (6,), f"daily_weekly output shape: {out_dual.shape}"

    X_daily_grad = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:8], dtype=torch.float32, device=device)
    X_weekly_grad = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:8], dtype=torch.float32, device=device)
    y_grad = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
    model = formal.DualFrequencySWiMRegressor(daily_feature_dim=X_daily_grad.shape[-1], weekly_feature_dim=X_weekly_grad.shape[-1]).to(device)
    pred = model(X_daily_grad, X_weekly_grad)
    loss = nn.HuberLoss(delta=1.0)(pred, y_grad)
    loss.backward()
    named_params = list(model.named_parameters())
    missing = [name for name, p in named_params if p.requires_grad and p.grad is None]
    non_finite = [name for name, p in named_params if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, f"parameters with no gradient: {missing}"
    assert not non_finite, f"parameters with non-finite gradient: {non_finite}"

    X_weekly_nan = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:8], dtype=torch.float32, device=device)
    weekly_eval_model = formal.SingleFrequencySWiMRegressor(feature_dim=X_weekly_nan.shape[-1], frequency_key="weekly").to(device)
    weekly_eval_model.eval()
    with torch.no_grad():
        out_eval = weekly_eval_model(X_weekly_nan)
        seq, valid_mask_eval = weekly_eval_model.branch(X_weekly_nan, return_sequence=True)
    assert torch.isfinite(out_eval).all(), "NaN/Inf in weekly_only forward output"
    assert torch.isfinite(seq).all(), "NaN/Inf in weekly encoded sequence (pre-pool)"
    pad_positions = seq[:, ~valid_mask_eval, :]
    assert torch.allclose(pad_positions, torch.zeros_like(pad_positions), atol=1e-6), "padded positions are not exactly zero"

    daily_count = count_trainable_parameters(daily_model)
    weekly_count = count_trainable_parameters(weekly_model)
    dual_count = count_trainable_parameters(dual_model)
    assert daily_count == EXPECTED_SWIM_PARAM_COUNTS["daily_only"], f"daily_only param count mismatch: {daily_count}"
    assert weekly_count == EXPECTED_SWIM_PARAM_COUNTS["weekly_only"], f"weekly_only param count mismatch: {weekly_count}"
    assert dual_count == EXPECTED_SWIM_PARAM_COUNTS["daily_weekly"], f"daily_weekly param count mismatch: {dual_count}"

    print("CPU SWIM SMOKE VALIDATION PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWiM MAIN V2 CPU formal execution path")
    parser.add_argument("--train", action="store_true", help="Run the formal CPU SWiM benchmark.")
    parser.add_argument("--all", action="store_true", help="Run all 9 formal CPU SWiM cells.")
    parser.add_argument("--frequency", choices=FREQUENCY_SETTINGS, help="Formal CPU frequency selection: daily_only | weekly_only | daily_weekly")
    parser.add_argument("--horizon", choices=["5d", "10d", "20d"], help="Formal CPU horizon selection: 5d | 10d | 20d")
    parser.add_argument("--finalize-only", action="store_true", help="Finalize existing CPU formal evidence without retraining.")
    parser.add_argument("--output-dir", type=str, default=str(MAIN_V2_SWIM_DIR), help="CPU formal SWiM output directory.")
    args = parser.parse_args()

    if args.finalize_only and args.train:
        parser.error("--finalize-only cannot be combined with --train.")
    if args.train and not args.all and (args.frequency is None or args.horizon is None):
        parser.error("When --train is supplied, either use --all or provide both --frequency and --horizon.")
    if args.all and not args.train:
        parser.error("--all requires --train.")
    if args.frequency is not None and args.horizon is None:
        parser.error("--horizon is required when --frequency is supplied.")
    if args.horizon is not None and args.frequency is None:
        parser.error("--frequency is required when --horizon is supplied.")
    return args


def main() -> None:
    args = parse_args()
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    formal.validate_frozen_dataset_contract(ds)
    stats = fit_train_only_normalizer(ds)

    if args.finalize_only:
        output_dir = resolve_output_dir(args.output_dir)
        provenance_path = output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"]
        if not provenance_path.exists():
            raise FileNotFoundError(f"CPU training provenance missing: {provenance_path}")
        provenance_df = pd.read_csv(provenance_path)
        validate_cpu_training_provenance_row(provenance_df.iloc[0])
        training_sha = str(provenance_df.iloc[0]["training_script_sha256"])
        current_sha = compute_sha256(Path(__file__))
        if training_sha != current_sha:
            raise RuntimeError(f"CPU finalize lock failed: stored training SHA {training_sha} != current script SHA {current_sha}")

        validate_cpu_main_v2_swim_evidence(output_dir)
        summary_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["summary"])
        pred_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["predictions"])
        rank_ic_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"])
        history_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["history"])

        comparison_df = build_cpu_comparison(summary_df)
        comparison_df.to_csv(output_dir / FORMAL_SWIM_RUN_FILES["comparison"], index=False)

        write_cpu_run_manifest(
            output_dir,
            summary_df,
            pred_df,
            rank_ic_df,
            history_df,
            training_script_sha256=training_sha,
            finalization_time_script_sha256=current_sha,
        )
        write_cpu_final_summary(
            output_dir,
            summary_df,
            comparison_df,
            training_script_sha256=training_sha,
            finalization_time_script_sha256=current_sha,
        )
        return

    if args.train:
        if args.all:
            full_benchmark_cpu(ds, stats, output_dir=MAIN_V2_SWIM_DIR)
            return

        output_dir = resolve_output_dir(args.output_dir)
        write_cpu_training_provenance(output_dir, Path(__file__))
        validate_partial_raw_cell_set_consistency_cpu(output_dir)

        frequency = args.frequency
        horizon_int = int(args.horizon[:-1])
        horizon_label = args.horizon

        ensure_requested_cell_absent_cpu(output_dir, frequency, horizon_label, seed=SEED)

        result = train_and_eval_frequency_cpu(ds, frequency, horizon_int, stats)

        save_summary_rows_cpu([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["summary"]).exists())
        save_prediction_rows_cpu([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["predictions"]).exists())
        save_training_history_cpu([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["history"]).exists())
        save_rank_ic_by_date_cpu([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]).exists())

        validate_partial_raw_cell_set_consistency_cpu(output_dir)
        validate_partial_cpu_swim_evidence(output_dir)
        return

    run_cpu_smoke_validation()
    return


if __name__ == "__main__":
    main()
