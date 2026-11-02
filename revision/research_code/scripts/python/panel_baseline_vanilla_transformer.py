"""Frozen-panel Vanilla Transformer control baseline.

This script implements the conventional Transformer control under the frozen
17-stock Daily + Weekly panel protocol defined by `panel_common.py` and the
LSTM panel baseline.

Key design constraints:
- use the exact frozen panel sample index, split, targets, and normalization
- do not use the historical FSLR data loader or the old early-fusion pipeline
- use per-frequency Transformer branches with late fusion for daily_weekly
- default behavior is a smoke-validation mode only; the full 9-experiment
  benchmark is intentionally not launched in this task
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
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

from scripts.python.panel_common import (
    HORIZONS,
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    load_frozen_panel,
    rank_ic_by_date,
    safe_pearson_corr,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = ROOT / "dataset" / "audit"
MAIN_V2_VANILLA_DIR = DEFAULT_AUDIT_DIR / "main_v2" / "vanilla"
DATASET_MANIFEST_SHA256 = "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c"
AUDIT_DIR = MAIN_V2_VANILLA_DIR

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-4
D_MODEL = 64
NHEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
ACTIVATION = "relu"
FREQUENCY_SETTINGS = ["daily_only", "weekly_only", "daily_weekly"]
FIXED_MAX_LEN = 512


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonicalize_main_v2_reference_df(df: pd.DataFrame | None, required_cols: list[str], label: str) -> pd.DataFrame:
    if df is None:
        raise FileNotFoundError(f"Missing MAIN V2 {label} artifact.")
    if df.empty:
        raise ValueError(f"MAIN V2 {label} artifact is empty.")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"MAIN V2 {label} artifact is missing required columns: {missing}")
    out = df.copy()
    if "frequency" in out.columns:
        out["frequency_setting"] = out["frequency"].astype(str).str.strip().str.lower()
    if "frequency_setting" in out.columns:
        out["frequency_setting"] = out["frequency_setting"].astype(str).str.strip().str.lower()
    if "horizon" in out.columns:
        out["horizon"] = out["horizon"].astype(str).str.strip().str.replace("d$", "", regex=True)
    if "model" in out.columns:
        out["model"] = out["model"].astype(str).str.strip().str.lower()
    return out


def canonicalize_main_v2_ridge(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    required = ["frequency", "horizon", "test_mse", "test_mae", "corr", "direction_accuracy", "mean_rank_ic", "pred_std_true_std_ratio"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"Ridge MAIN V2 comparison missing required columns: {missing}")
    normalized["frequency_setting"] = normalized["frequency"].astype(str).str.strip().str.lower()
    normalized["horizon"] = normalized["horizon"].astype(str).str.strip().str.replace("d$", "", regex=True)
    normalized = normalized.rename(columns={"test_mse": "pooled_mse", "test_mae": "pooled_mae", "corr": "pooled_corr"})
    normalized = normalized[["frequency_setting", "horizon", "pooled_mse", "pooled_mae", "pooled_corr", "direction_accuracy", "mean_rank_ic", "pred_std_true_std_ratio"]].copy()
    expected_configs = {(freq, horizon) for freq in ["daily_only", "weekly_only", "daily_weekly"] for horizon in ["5", "10", "20"]}
    actual_configs = {(row["frequency_setting"], str(row["horizon"])) for _, row in normalized.iterrows()}
    if len(normalized) != 9:
        raise ValueError(f"Expected exactly 9 Ridge rows after canonicalization, found {len(normalized)}")
    if normalized.duplicated(subset=["frequency_setting", "horizon"]).any():
        dupes = normalized[normalized.duplicated(subset=["frequency_setting", "horizon"], keep=False)]
        raise ValueError(f"Ridge canonicalized data contains duplicate frequency_setting/horizon rows: {dupes.to_dict(orient='records')}")
    if actual_configs != expected_configs:
        raise ValueError(f"Ridge configs do not match expected 9 frequency x horizon settings: {sorted(actual_configs)}")
    return normalized


def canonicalize_main_v2_naive(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    required = ["model", "horizon", "pooled_mse"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"Naive MAIN V2 summary missing required columns: {missing}")
    normalized["model"] = normalized["model"].astype(str).str.strip().str.lower()
    normalized["horizon"] = normalized["horizon"].astype(str).str.strip().str.replace("d$", "", regex=True)
    formal = normalized[normalized["model"].isin(["zero", "train_mean"])].copy()
    expected = {(model, horizon) for model in ["zero", "train_mean"] for horizon in ["5", "10", "20"]}
    actual = {(row["model"], str(row["horizon"])) for _, row in formal.iterrows()}
    if len(formal) != 6:
        raise ValueError(f"Naive MAIN V2 summary must contain exactly 6 required formal rows, found {len(formal)}")
    if formal.duplicated(subset=["model", "horizon"]).any():
        dupes = formal[formal.duplicated(subset=["model", "horizon"], keep=False)]
        raise ValueError(f"Naive MAIN V2 summary contains duplicate model/horizon rows: {dupes.to_dict(orient='records')}")
    if actual != expected:
        raise ValueError(f"Naive MAIN V2 summary does not match the required formal zero/train_mean x 5/10/20 cells: {sorted(actual)}")
    return formal


def validate_summary_rows(summary_df: pd.DataFrame) -> None:
    required_cols = [
        "model",
        "frequency_setting",
        "horizon",
        "seed",
        "n_trainable_parameters",
        "best_epoch",
        "epochs_trained",
        "runtime_seconds",
        "val_loss",
        "test_mse",
        "test_mae",
        "test_corr",
        "ticker_avg_mse",
        "ticker_avg_mae",
        "ticker_avg_corr",
        "mean_rank_ic",
        "median_rank_ic",
        "rank_ic_std",
        "positive_ic_ratio",
        "rank_icir",
        "n_valid_ic_dates",
        "direction_accuracy",
        "pred_std",
        "true_std",
        "pred_std_true_std_ratio",
        "no_extreme_scale_pathology",
    ]
    missing = [col for col in required_cols if col not in summary_df.columns]
    if missing:
        raise ValueError(f"Summary artifact missing columns: {missing}")
    if len(summary_df) != 9:
        raise ValueError(f"Summary artifact must contain exactly 9 rows (3 freq x 3 horizons); found {len(summary_df)}.")
    if summary_df["model"].nunique() != 1 or str(summary_df["model"].iloc[0]).lower() != "vanilla_transformer":
        raise ValueError("Summary artifact model must be vanilla_transformer for all rows.")
    if summary_df["seed"].nunique() != 1 or int(summary_df["seed"].iloc[0]) != 42:
        raise ValueError("Summary artifact must use the nominal seed 42 for all rows.")
    if summary_df.duplicated(subset=["model", "frequency_setting", "horizon", "seed"]).any():
        raise ValueError("Summary artifact contains duplicate model/frequency/horizon/seed keys.")
    expected = {(freq, horizon) for freq in FREQUENCY_SETTINGS for horizon in ["5d", "10d", "20d"]}
    observed = {(str(row["frequency_setting"]), str(row["horizon"])) for _, row in summary_df.iterrows()}
    if observed != expected:
        raise ValueError(f"Summary artifact does not contain the expected 9 frequency/horizon cells: {sorted(observed)} != {sorted(expected)}")
    required_finite = [
        "n_trainable_parameters",
        "best_epoch",
        "epochs_trained",
        "runtime_seconds",
        "val_loss",
        "test_mse",
        "test_mae",
        "direction_accuracy",
        "pred_std",
        "true_std",
        "pred_std_true_std_ratio",
    ]
    for col in required_finite:
        vals = pd.to_numeric(summary_df[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError(f"Summary artifact contains missing or non-finite numeric values in required-finite column {col}.")
    params = pd.to_numeric(summary_df["n_trainable_parameters"], errors="coerce")
    expected_params = {
        "daily_only": 67393,
        "weekly_only": 67393,
        "daily_weekly": 142977,
    }
    for _, row in summary_df.iterrows():
        freq = str(row["frequency_setting"])
        if freq not in expected_params:
            raise ValueError(f"Unexpected frequency setting in summary: {freq}")
        want = expected_params[freq]
        have = int(row["n_trainable_parameters"])
        if have != want:
            raise ValueError(f"Parameter count mismatch for {freq}: expected {want}, found {have}.")
    allowed_nan = [
        "test_corr",
        "ticker_avg_corr",
        "mean_rank_ic",
        "median_rank_ic",
        "rank_ic_std",
        "rank_icir",
        "positive_ic_ratio",
    ]
    for col in allowed_nan:
        vals = pd.to_numeric(summary_df[col], errors="coerce")
        if vals.notna().any() and not np.isfinite(vals[vals.notna()].to_numpy()).all():
            raise ValueError(f"Summary artifact contains non-finite values in allowed-NaN metric column {col}.")


def validate_prediction_rows(pred_df: pd.DataFrame) -> None:
    required_cols = ["model", "ticker", "anchor_date", "horizon", "frequency_setting", "y_true", "y_pred", "seed", "best_epoch"]
    missing = [col for col in required_cols if col not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction artifact missing columns: {missing}")
    expected = {(freq, horizon) for freq in FREQUENCY_SETTINGS for horizon in ["5d", "10d", "20d"]}
    observed = {(str(row["frequency_setting"]), str(row["horizon"])) for _, row in pred_df.iterrows()}
    if observed != expected:
        raise ValueError(f"Prediction artifact does not contain the expected 9 frequency/horizon cells: {sorted(observed)} != {sorted(expected)}")
    if pred_df["model"].nunique() != 1 or str(pred_df["model"].iloc[0]).lower() != "vanilla_transformer":
        raise ValueError("Prediction artifact model must be vanilla_transformer for all rows.")
    if pred_df["seed"].nunique() != 1 or int(pred_df["seed"].iloc[0]) != 42:
        raise ValueError("Prediction artifact must use the nominal seed 42 for all rows.")
    if pred_df.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"]).any():
        raise ValueError("Prediction artifact contains duplicate model/frequency/horizon/ticker/date keys.")
    for (freq, horizon), group in pred_df.groupby(["frequency_setting", "horizon"]):
        if len(group) != 6647:
            raise ValueError(f"Prediction cell {freq}/{horizon} must contain exactly 6,647 rows; found {len(group)}.")
        if len(group["ticker"].unique()) != 17:
            raise ValueError(f"Prediction cell {freq}/{horizon} must contain exactly 17 unique tickers; found {len(group['ticker'].unique())}.")
        if len(group["anchor_date"].unique()) != 391:
            raise ValueError(f"Prediction cell {freq}/{horizon} must contain exactly 391 unique anchor dates; found {len(group['anchor_date'].unique())}.")
        if not np.isfinite(pd.to_numeric(group["y_true"], errors="coerce")).all():
            raise ValueError(f"Prediction cell {freq}/{horizon} contains non-finite y_true values.")
        if not np.isfinite(pd.to_numeric(group["y_pred"], errors="coerce")).all():
            raise ValueError(f"Prediction cell {freq}/{horizon} contains non-finite y_pred values.")


def validate_rank_ic_rows(rank_ic_df: pd.DataFrame) -> None:
    required_cols = ["model", "frequency_setting", "horizon", "seed", "anchor_date", "n_tickers", "rank_ic"]
    missing = [col for col in required_cols if col not in rank_ic_df.columns]
    if missing:
        raise ValueError(f"Rank-IC artifact missing columns: {missing}")
    expected = {(freq, horizon) for freq in FREQUENCY_SETTINGS for horizon in ["5d", "10d", "20d"]}
    observed = {(str(row["frequency_setting"]), str(row["horizon"])) for _, row in rank_ic_df.iterrows()}
    if observed != expected:
        raise ValueError(f"Rank-IC artifact does not contain the expected 9 frequency/horizon cells: {sorted(observed)} != {sorted(expected)}")
    if rank_ic_df["model"].nunique() != 1 or str(rank_ic_df["model"].iloc[0]).lower() != "vanilla_transformer":
        raise ValueError("Rank-IC artifact model must be vanilla_transformer for all rows.")
    if rank_ic_df["seed"].nunique() != 1 or int(rank_ic_df["seed"].iloc[0]) != 42:
        raise ValueError("Rank-IC artifact must use the nominal seed 42 for all rows.")
    if rank_ic_df.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "anchor_date"]).any():
        raise ValueError("Rank-IC artifact contains duplicate model/frequency/horizon/anchor_date keys.")
    for (freq, horizon), group in rank_ic_df.groupby(["frequency_setting", "horizon"]):
        if len(group) != 391:
            raise ValueError(f"Rank-IC cell {freq}/{horizon} must contain exactly 391 rows; found {len(group)}.")
        n_tickers = pd.to_numeric(group["n_tickers"], errors="coerce")
        if n_tickers.isna().any() or (n_tickers != 17).any():
            raise ValueError(f"Rank-IC cell {freq}/{horizon} contains n_tickers values that are not all equal to 17.")


def validate_training_history(history_df: pd.DataFrame, summary_df: pd.DataFrame | None = None) -> None:
    required_cols = ["model", "frequency_setting", "horizon", "seed", "epoch", "train_loss", "val_loss", "elapsed_seconds"]
    missing = [col for col in required_cols if col not in history_df.columns]
    if missing:
        raise ValueError(f"Training history artifact missing columns: {missing}")
    if history_df.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "epoch"]).any():
        raise ValueError("Training history artifact contains duplicate epoch keys.")
    if not np.isfinite(pd.to_numeric(history_df["epoch"], errors="coerce")).all():
        raise ValueError("Training history artifact contains non-finite epoch values.")
    if (pd.to_numeric(history_df["epoch"], errors="coerce") < 1).any():
        raise ValueError("Training history artifact contains epoch values less than 1.")
    if (pd.to_numeric(history_df["epoch"], errors="coerce") > MAX_EPOCHS).any():
        raise ValueError(f"Training history artifact contains epoch values exceeding {MAX_EPOCHS}.")
    if not np.isfinite(pd.to_numeric(history_df["train_loss"], errors="coerce")).all():
        raise ValueError("Training history artifact contains non-finite train_loss values.")
    if not np.isfinite(pd.to_numeric(history_df["val_loss"], errors="coerce")).all():
        raise ValueError("Training history artifact contains non-finite val_loss values.")
    if summary_df is not None:
        for _, row in summary_df.iterrows():
            freq = str(row["frequency_setting"])
            horizon = str(row["horizon"])
            cell_hist = history_df[(history_df["frequency_setting"] == freq) & (history_df["horizon"] == horizon)]
            if cell_hist.empty:
                raise ValueError(f"Training history is missing the {freq}/{horizon} cell.")
            best_epoch = int(row["best_epoch"])
            if not (cell_hist["epoch"] == best_epoch).any():
                raise ValueError(f"Summary best_epoch {best_epoch} missing from history for cell {freq}/{horizon}.")
            best_val = float(cell_hist.loc[cell_hist["val_loss"].idxmin(), "val_loss"])
            if not np.isclose(float(row["val_loss"]), best_val, rtol=1e-8, atol=1e-12):
                raise ValueError(f"Summary val_loss does not match the minimum history val_loss for cell {freq}/{horizon}.")
            if not np.isclose(float(cell_hist.loc[cell_hist["epoch"] == best_epoch, "val_loss"].iloc[0]), best_val, rtol=1e-8, atol=1e-12):
                raise ValueError(f"Best-epoch val_loss does not match the minimum history val_loss for cell {freq}/{horizon}.")
            epochs_trained = int(row["epochs_trained"])
            if epochs_trained != int(cell_hist["epoch"].max()):
                raise ValueError(f"Summary epochs_trained mismatch for cell {freq}/{horizon}: {epochs_trained} != max epoch {int(cell_hist['epoch'].max())}.")


def validate_cross_artifact_cell_consistency(summary_df: pd.DataFrame, pred_df: pd.DataFrame, rank_ic_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    summary_cells = set((str(r["frequency_setting"]), str(r["horizon"])) for _, r in summary_df.iterrows())
    pred_cells = set((str(r["frequency_setting"]), str(r["horizon"])) for _, r in pred_df.iterrows())
    rank_cells = set((str(r["frequency_setting"]), str(r["horizon"])) for _, r in rank_ic_df.iterrows())
    hist_cells = set((str(r["frequency_setting"]), str(r["horizon"])) for _, r in history_df.iterrows())
    if summary_cells != pred_cells or summary_cells != rank_cells or summary_cells != hist_cells:
        raise ValueError(
            "Cross-artifact cell consistency failed: summary/predictions/Rank IC/training history frequency/horizon cells do not match."
        )


def validate_cross_frequency_y_true_identity(pred_df: pd.DataFrame) -> None:
    key_cols = ["horizon", "ticker", "anchor_date"]
    y_true_key = pred_df.groupby(key_cols)["y_true"].nunique()
    mismatched = y_true_key[y_true_key != 1]
    if not mismatched.empty:
        raise ValueError("Cross-frequency y_true identity failed: the same horizon/ticker/date must map to a single target label across frequencies.")


def validate_expected_parameter_counts(summary_df: pd.DataFrame) -> None:
    params = pd.to_numeric(summary_df["n_trainable_parameters"], errors="coerce")
    if params.isna().any() or (params <= 0).any():
        raise ValueError("Expected parameter counts must be finite and strictly positive for all summary rows.")
    expected = {"daily_only": 67393, "weekly_only": 67393, "daily_weekly": 142977}
    for _, row in summary_df.iterrows():
        freq = str(row["frequency_setting"])
        got = int(row["n_trainable_parameters"])
        if expected.get(freq) != got:
            raise ValueError(f"Exact parameter count mismatch for {freq}: expected {expected.get(freq)}, found {got}.")


def validate_main_v2_vanilla_evidence(output_dir: Path) -> None:
    summary = pd.read_csv(output_dir / "panel_vanilla_summary_metrics.csv")
    pred = pd.read_csv(output_dir / "panel_vanilla_test_predictions.csv")
    rank_ic = pd.read_csv(output_dir / "panel_vanilla_rank_ic_by_date.csv")
    history = pd.read_csv(output_dir / "panel_vanilla_training_history.csv")

    validate_summary_rows(summary)
    validate_prediction_rows(pred)
    validate_rank_ic_rows(rank_ic)
    validate_training_history(history, summary)
    validate_cross_artifact_cell_consistency(summary, pred, rank_ic, history)
    validate_cross_frequency_y_true_identity(pred)
    validate_expected_parameter_counts(summary)


def validate_frozen_dataset_contract(ds: object) -> None:
    expected_tickers = [
        "AES",
        "BEP",
        "BLDP",
        "BLNK",
        "CSIQ",
        "CWEN",
        "DQ",
        "ENPH",
        "FCEL",
        "FSLR",
        "HASI",
        "JKS",
        "NEE",
        "ORA",
        "PLUG",
        "RUN",
        "SEDG",
    ]
    ticker_values = sorted(np.unique(ds.ticker).tolist())
    if len(ticker_values) != 17:
        raise RuntimeError(f"Ticker count mismatch: {len(ticker_values)} != 17")
    if ticker_values != sorted(expected_tickers):
        raise RuntimeError(f"Ticker universe mismatch: {ticker_values} != {sorted(expected_tickers)}")
    common_dates = np.unique(ds.anchor_date)
    if len(common_dates) != 2563:
        raise RuntimeError(f"Common anchor date count mismatch: {len(common_dates)} != 2563")


def resolve_output_dir(output_dir: str | Path | None) -> Path:
    resolved = Path(output_dir) if output_dir is not None else MAIN_V2_VANILLA_DIR
    resolved = resolved if resolved.is_absolute() else (ROOT / resolved)
    expected = (ROOT / "dataset" / "audit" / "main_v2" / "vanilla").resolve()
    if resolved.resolve() != expected:
        raise ValueError(
            f"Vanilla formal MAIN V2 output directory must be exactly {expected}; got {resolved.resolve()}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_training_provenance_row(row: pd.Series) -> None:
    required = {
        "run_family": "MAIN_V2",
        "model": "VANILLA_TRANSFORMER",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "dataset_audit_status": "PASS",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "nominal_seed": 42,
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "activation": "relu",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": 100,
        "patience": 10,
        "max_norm": 1.0,
    }
    for key, expected in required.items():
        if key not in row.index:
            raise ValueError(f"Training provenance missing required field: {key}")
        actual = row[key]
        if key in {"dataset_contract_version", "formal_tickers", "common_anchor_dates", "total_samples", "train_samples", "validation_samples", "test_samples", "train_dates", "validation_dates", "test_dates", "nominal_seed", "d_model", "nhead", "num_encoder_layers", "dim_feedforward", "batch_size", "max_epochs", "patience", "max_norm"}:
            if pd.notna(actual) and int(actual) != int(expected):
                raise ValueError(f"Training provenance field {key} mismatch: expected {expected}, found {actual}")
        elif pd.notna(actual) and str(actual) != str(expected):
            raise ValueError(f"Training provenance field {key} mismatch: expected {expected}, found {actual}")
    if "training_script" not in row.index or pd.isna(row["training_script"]):
        raise ValueError("Training provenance missing training script path.")
    expected_training_script = "scripts/python/panel_baseline_vanilla_transformer.py"
    if str(row["training_script"]).strip() != expected_training_script:
        raise ValueError(
            f"Training provenance training_script mismatch: expected {expected_training_script}, found {row['training_script']}"
        )
    if "output_dir" not in row.index or pd.isna(row["output_dir"]):
        raise ValueError("Training provenance missing output_dir.")
    expected_output_dir = str((ROOT / "dataset" / "audit" / "main_v2" / "vanilla").resolve())
    if str(row["output_dir"]).strip() != expected_output_dir:
        raise ValueError(f"Training provenance output_dir mismatch: expected {expected_output_dir}, found {row['output_dir']}")
    if "training_script_sha256" not in row.index or pd.isna(row["training_script_sha256"]):
        raise ValueError("Training provenance missing training_script_sha256.")


def write_training_provenance(output_dir: Path, script_path: Path) -> Path:
    provenance_path = output_dir / "main_v2_vanilla_training_provenance.csv"
    current_sha = compute_sha256(script_path)
    if provenance_path.exists():
        existing = pd.read_csv(provenance_path)
        if existing.empty:
            raise ValueError(f"Training provenance file exists but is empty: {provenance_path}")
        row = existing.iloc[0]
        validate_training_provenance_row(row)
        stored_sha = row.get("training_script_sha256")
        if pd.notna(stored_sha) and str(stored_sha) != current_sha:
            raise ValueError(
                f"Training provenance mismatch: stored SHA {stored_sha} differs from current script SHA {current_sha}. "
                f"Refusing to overwrite formal provenance."
            )
        return provenance_path

    row = {
        "run_family": "MAIN_V2",
        "model": "VANILLA_TRANSFORMER",
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
        "training_script": str(script_path.relative_to(ROOT)),
        "training_script_sha256": current_sha,
        "output_dir": str(output_dir.resolve()),
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "activation": "relu",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": 100,
        "patience": 10,
        "max_norm": 1.0,
    }
    pd.DataFrame([row]).to_csv(provenance_path, index=False)
    return provenance_path


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_elapsed(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def prediction_scale_looks_normal(pred_std: float, true_std: float) -> bool:
    if not np.isfinite(pred_std) or not np.isfinite(true_std) or true_std <= 1e-12:
        return False
    ratio = pred_std / true_std
    return 0.2 <= ratio <= 2.0


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding as a non-trainable buffer."""

    def __init__(self, d_model: int, max_len: int = FIXED_MAX_LEN, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding capacity {self.pe.size(1)}."
            )
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)


class TransformerBranch(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=FIXED_MAX_LEN, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=ACTIVATION,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.positional_encoding(h)
        h = self.encoder(h)
        return h[:, -1, :]


class SingleFrequencyTransformerRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.branch = TransformerBranch(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.branch(x)
        return self.head(rep).squeeze(-1)


class DualFrequencyTransformerRegressor(nn.Module):
    def __init__(
        self,
        daily_feature_dim: int,
        weekly_feature_dim: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.daily_branch = TransformerBranch(
            feature_dim=daily_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.weekly_branch = TransformerBranch(
            feature_dim=weekly_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        daily_vec = self.daily_branch(x_daily)
        weekly_vec = self.weekly_branch(x_weekly)
        fused = torch.cat([daily_vec, weekly_vec], dim=1)
        return self.fusion(fused).squeeze(-1)


def compute_rank_ic_summary(ic_df: pd.DataFrame) -> dict:
    valid = ic_df["rank_ic"].dropna()
    n_valid = int(len(valid))
    mean_ic = float(valid.mean()) if n_valid else float("nan")
    median_ic = float(valid.median()) if n_valid else float("nan")
    std_ic = float(valid.std()) if n_valid > 1 else float("nan")
    positive_ratio = float((valid > 0).mean()) if n_valid else float("nan")
    icir = (mean_ic / std_ic) if (n_valid > 1 and np.isfinite(std_ic) and abs(std_ic) > 1e-12) else float("nan")
    return {
        "mean_rank_ic": mean_ic,
        "median_rank_ic": median_ic,
        "rank_ic_std": std_ic,
        "positive_ic_ratio": positive_ratio,
        "n_valid_ic_dates": n_valid,
        "rank_icir": icir,
    }


def assert_sequence_orientation(ds: object) -> None:
    """Verify axis 1 is ordered oldest -> newest so final-timestep reads the latest observation."""
    if ds.X_daily.ndim != 3 or ds.X_weekly.ndim != 3:
        raise RuntimeError("Panel arrays must have shape (n_samples, window, features)")
    if ds.X_daily.shape[1] < 2 or ds.X_weekly.shape[1] < 2:
        raise RuntimeError("Panel sequences must have at least 2 timesteps along axis 1")
    if not np.isfinite(ds.X_daily).all() or not np.isfinite(ds.X_weekly).all():
        raise RuntimeError("Panel arrays contain non-finite values before training")


def assert_frozen_panel_checks(ds: object) -> None:
    validate_frozen_dataset_contract(ds)

    train_n = int(np.sum(ds.split == "train"))
    val_n = int(np.sum(ds.split == "val"))
    test_n = int(np.sum(ds.split == "test"))
    train_dates = len(np.unique(ds.anchor_date[ds.split == "train"]))
    val_dates = len(np.unique(ds.anchor_date[ds.split == "val"]))
    test_dates = len(np.unique(ds.anchor_date[ds.split == "test"]))

    if train_n != 30634:
        raise RuntimeError(f"Train sample count mismatch: {train_n} != 30634")
    if val_n != 6290:
        raise RuntimeError(f"Validation sample count mismatch: {val_n} != 6290")
    if test_n != 6647:
        raise RuntimeError(f"Test sample count mismatch: {test_n} != 6647")
    if train_dates != 1802:
        raise RuntimeError(f"Train date count mismatch: {train_dates} != 1802")
    if val_dates != 370:
        raise RuntimeError(f"Validation date count mismatch: {val_dates} != 370")
    if test_dates != 391:
        raise RuntimeError(f"Test date count mismatch: {test_dates} != 391")

    for h in [5, 10, 20]:
        if not np.isfinite(ds.y[h]).all():
            raise RuntimeError(f"Target values contain non-finite entries for horizon={h}")

    assert_sequence_orientation(ds)

    train_stats = fit_train_only_normalizer(ds)
    for freq in ["daily_only", "weekly_only"]:
        X = apply_normalizer(ds, train_stats, freq)
        if np.isnan(X).any() or np.isinf(X).any():
            raise RuntimeError(f"NaN/Inf in normalized {freq} features")
    Xd, Xw = apply_normalizer(ds, train_stats, "daily_weekly")
    if np.isnan(Xd).any() or np.isinf(Xd).any():
        raise RuntimeError("NaN/Inf in normalized daily features for daily_weekly")
    if np.isnan(Xw).any() or np.isinf(Xw).any():
        raise RuntimeError("NaN/Inf in normalized weekly features for daily_weekly")

    for ticker in np.unique(ds.ticker):
        mask = ds.ticker == ticker
        ordered_dates = pd.to_datetime(ds.anchor_date[mask])
        if not ordered_dates.is_monotonic_increasing:
            raise RuntimeError(f"Time order violation for ticker={ticker}: dates are not oldest -> newest")


def build_frequency_data(ds, stats: dict, frequency: str, horizon: int):
    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    if frequency == "daily_only":
        X_all = apply_normalizer(ds, stats, "daily_only")
        X_train = X_all[train_mask]
        X_val = X_all[val_mask]
        X_test = X_all[test_mask]
        y_train = ds.y[horizon][train_mask]
        y_val = ds.y[horizon][val_mask]
        y_test = ds.y[horizon][test_mask]
        return X_train, X_val, X_test, y_train, y_val, y_test

    if frequency == "weekly_only":
        X_all = apply_normalizer(ds, stats, "weekly_only")
        X_train = X_all[train_mask]
        X_val = X_all[val_mask]
        X_test = X_all[test_mask]
        y_train = ds.y[horizon][train_mask]
        y_val = ds.y[horizon][val_mask]
        y_test = ds.y[horizon][test_mask]
        return X_train, X_val, X_test, y_train, y_val, y_test

    if frequency == "daily_weekly":
        X_daily_all, X_weekly_all = apply_normalizer(ds, stats, "daily_weekly")
        X_d_train = X_daily_all[train_mask]
        X_d_val = X_daily_all[val_mask]
        X_d_test = X_daily_all[test_mask]
        X_w_train = X_weekly_all[train_mask]
        X_w_val = X_weekly_all[val_mask]
        X_w_test = X_weekly_all[test_mask]
        y_train = ds.y[horizon][train_mask]
        y_val = ds.y[horizon][val_mask]
        y_test = ds.y[horizon][test_mask]
        return (
            X_d_train,
            X_d_val,
            X_d_test,
            X_w_train,
            X_w_val,
            X_w_test,
            y_train,
            y_val,
            y_test,
        )

    raise ValueError(f"Unknown frequency setting: {frequency}")


def make_model(frequency: str, daily_feature_dim: int, weekly_feature_dim: int) -> nn.Module:
    if frequency == "daily_only":
        return SingleFrequencyTransformerRegressor(feature_dim=daily_feature_dim)
    if frequency == "weekly_only":
        return SingleFrequencyTransformerRegressor(feature_dim=weekly_feature_dim)
    if frequency == "daily_weekly":
        return DualFrequencyTransformerRegressor(
            daily_feature_dim=daily_feature_dim,
            weekly_feature_dim=weekly_feature_dim,
        )
    raise ValueError(f"Unknown frequency setting: {frequency}")


def smoke_test_models(ds, stats: dict, device: torch.device) -> dict:
    results = {}
    for frequency in FREQUENCY_SETTINGS:
        if frequency == "daily_only":
            X_all = apply_normalizer(ds, stats, "daily_only")
            X = torch.tensor(X_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=X.shape[-1], weekly_feature_dim=X.shape[-1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            pred = model(X)
            assert pred.shape == y.shape, f"daily_only output shape mismatch: {pred.shape} vs {y.shape}"
            loss = nn.HuberLoss(delta=1.0)(pred, y)
            loss.backward()
            grads_present = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            assert grads_present, "daily_only backprop produced no valid gradients"
            optimizer.zero_grad()
            results[frequency] = {"ok": True, "param_count": count_trainable_parameters(model)}

        elif frequency == "weekly_only":
            X_all = apply_normalizer(ds, stats, "weekly_only")
            X = torch.tensor(X_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=X.shape[-1], weekly_feature_dim=X.shape[-1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            pred = model(X)
            assert pred.shape == y.shape, f"weekly_only output shape mismatch: {pred.shape} vs {y.shape}"
            loss = nn.HuberLoss(delta=1.0)(pred, y)
            loss.backward()
            grads_present = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            assert grads_present, "weekly_only backprop produced no valid gradients"
            optimizer.zero_grad()
            results[frequency] = {"ok": True, "param_count": count_trainable_parameters(model)}

        else:
            Xd_all, Xw_all = apply_normalizer(ds, stats, "daily_weekly")
            xd = torch.tensor(Xd_all[:8], dtype=torch.float32, device=device)
            xw = torch.tensor(Xw_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=xd.shape[-1], weekly_feature_dim=xw.shape[-1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            pred = model(xd, xw)
            assert pred.shape == y.shape, f"daily_weekly output shape mismatch: {pred.shape} vs {y.shape}"
            loss = nn.HuberLoss(delta=1.0)(pred, y)
            loss.backward()
            grads_present = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            assert grads_present, "daily_weekly backprop produced no valid gradients"
            optimizer.zero_grad()
            results[frequency] = {"ok": True, "param_count": count_trainable_parameters(model)}

    return results


def train_and_eval_frequency(ds, frequency: str, horizon: int, stats: dict) -> dict:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
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
        model = SingleFrequencyTransformerRegressor(feature_dim=X_all.shape[-1]).to(device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            pred = model(x_batch.to(device))
            return pred, y_batch.to(device)

    elif frequency == "weekly_only":
        X_all = apply_normalizer(ds, stats, "weekly_only")
        X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
        X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
        X_test = torch.tensor(X_all[test_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        y_test = ds.y[horizon][test_mask].copy()
        model = SingleFrequencyTransformerRegressor(feature_dim=X_all.shape[-1]).to(device)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            pred = model(x_batch.to(device))
            return pred, y_batch.to(device)

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
        model = DualFrequencyTransformerRegressor(
            daily_feature_dim=X_daily_all.shape[-1],
            weekly_feature_dim=X_weekly_all.shape[-1],
        ).to(device)
        train_loader = DataLoader(
            TensorDataset(X_d_train, X_w_train, y_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_d_val, X_w_val, y_val),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        def batched_forward(x_daily, x_weekly, y_batch):
            pred = model(x_daily.to(device), x_weekly.to(device))
            return pred, y_batch.to(device)

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
            for x_d_batch, x_w_batch, y_batch in train_loader:
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
            for x_batch, y_batch in train_loader:
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
        history_rows.append(
            {
                "frequency_setting": frequency,
                "horizon": f"{horizon}d",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_seconds": time.perf_counter() - epoch_start,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | "
            f"best_epoch={best_epoch} | "
            f"patience={patience_counter}/{EARLY_STOPPING_PATIENCE} | "
            f"elapsed={format_elapsed(time.perf_counter() - epoch_start)}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            print(f"Best validation checkpoint = epoch {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"No valid training state for {frequency}, horizon={horizon}d")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if frequency == "daily_weekly":
            preds = []
            for x_d_batch, x_w_batch, _ in DataLoader(
                TensorDataset(X_d_test, X_w_test, torch.tensor(y_test, dtype=torch.float32)),
                batch_size=BATCH_SIZE,
                shuffle=False,
            ):
                preds.append(model(x_d_batch.to(device), x_w_batch.to(device)).cpu().numpy())
            y_pred = np.concatenate(preds)
        else:
            preds = []
            for x_batch, _ in DataLoader(
                TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32)),
                batch_size=BATCH_SIZE,
                shuffle=False,
            ):
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
        sanity_metric["pooled_pred_std"],
        sanity_metric["pooled_true_std"],
    )
    experiment_elapsed = time.perf_counter() - experiment_start

    result = {
        "model": "vanilla_transformer",
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
    return result


def merge_existing_rows(path: Path, new_df: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def save_prediction_rows(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        for i in range(len(rr["ticker"])):
            rows.append(
                {
                    "model": rr["model"],
                    "ticker": rr["ticker"][i],
                    "anchor_date": rr["anchor_date"][i],
                    "horizon": rr["horizon"],
                    "frequency_setting": rr["frequency_setting"],
                    "y_true": rr["y_true"][i],
                    "y_pred": rr["y_pred"][i],
                    "seed": rr["seed"],
                    "best_epoch": rr["best_epoch"],
                }
            )
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_vanilla_test_predictions.csv"
    if append_mode:
        merge_existing_rows(
            path,
            df,
            ["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"],
            ["model", "frequency_setting", "horizon", "anchor_date", "ticker"],
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def save_summary_rows(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        rows.append(
            {
                "model": rr["model"],
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
            }
        )
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_vanilla_summary_metrics.csv"
    if append_mode:
        merge_existing_rows(
            path,
            df,
            ["model", "frequency_setting", "horizon", "seed"],
            ["model", "frequency_setting", "horizon"],
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def save_training_history(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        for epoch_row in rr["history"]:
            rows.append(
                {
                    "model": rr["model"],
                    "frequency_setting": epoch_row["frequency_setting"],
                    "horizon": epoch_row["horizon"],
                    "seed": rr["seed"],
                    "epoch": epoch_row["epoch"],
                    "train_loss": epoch_row["train_loss"],
                    "val_loss": epoch_row["val_loss"],
                    "elapsed_seconds": epoch_row.get("elapsed_seconds", np.nan),
                }
            )
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_vanilla_training_history.csv"
    if append_mode:
        merge_existing_rows(
            path,
            df,
            ["model", "frequency_setting", "horizon", "seed", "epoch"],
            ["model", "frequency_setting", "horizon", "epoch"],
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def save_rank_ic_by_date(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for res in results:
        df = res["rank_ic_by_date"].copy()
        df["frequency_setting"] = res["frequency_setting"]
        df["horizon"] = res["horizon"]
        df["model"] = "vanilla_transformer"
        df["seed"] = res["seed"]
        columns = ["model", "frequency_setting", "horizon", "seed", "anchor_date", "n_tickers", "rank_ic"]
        rows.append(df.reindex(columns=columns))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    path = AUDIT_DIR / "panel_vanilla_rank_ic_by_date.csv"
    if append_mode:
        merge_existing_rows(
            path,
            out,
            ["model", "frequency_setting", "horizon", "seed", "anchor_date"],
            ["model", "frequency_setting", "horizon", "anchor_date"],
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out


def append_rank_ic_by_date(res: dict) -> pd.DataFrame:
    return save_rank_ic_by_date([res], append_mode=True)


def append_experiment_outputs(res: dict) -> None:
    save_summary_rows([res], append_mode=True)
    save_prediction_rows([res], append_mode=True)
    save_training_history([res], append_mode=True)
    append_rank_ic_by_date(res)


def build_baseline_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build a matched comparison table from accepted MAIN V2 panel baselines only."""
    rows = []
    naive_path = DEFAULT_AUDIT_DIR / "main_v2" / "naive" / "main_v2_naive_summary.csv"
    ridge_path = DEFAULT_AUDIT_DIR / "main_v2" / "ridge" / "main_v2_ridge_comparison.csv"
    lstm_path = DEFAULT_AUDIT_DIR / "main_v2" / "lstm" / "panel_lstm_summary_metrics.csv"

    if not naive_path.exists():
        raise FileNotFoundError(f"Missing required naive MAIN V2 artifact: {naive_path}")
    if not ridge_path.exists():
        raise FileNotFoundError(f"Missing required ridge MAIN V2 artifact: {ridge_path}")
    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing required LSTM MAIN V2 artifact: {lstm_path}")

    naive_df = canonicalize_main_v2_naive(pd.read_csv(naive_path))
    ridge_raw = pd.read_csv(ridge_path)
    ridge_df = canonicalize_main_v2_ridge(ridge_raw)
    lstm_raw = pd.read_csv(lstm_path)
    lstm_df = canonicalize_main_v2_reference_df(
        lstm_raw,
        ["model", "frequency_setting", "horizon", "seed", "test_mse", "test_mae", "test_corr", "mean_rank_ic"],
        "lstm",
    )
    if "seed" not in lstm_df.columns:
        raise ValueError("LSTM MAIN V2 reference is missing the required seed column.")
    if lstm_df.duplicated(subset=["model", "frequency_setting", "horizon", "seed"]).any():
        raise ValueError("LSTM MAIN V2 reference contains duplicate model/frequency/horizon/seed rows.")
    lstm_seed = pd.to_numeric(lstm_df["seed"], errors="coerce")
    if lstm_seed.isna().any() or (lstm_seed != 42).any():
        raise ValueError("LSTM MAIN V2 reference must be seed-42 only for all canonicalized rows.")
    expected_lstm_configs = {(freq, horizon) for freq in ["daily_only", "weekly_only", "daily_weekly"] for horizon in ["5", "10", "20"]}
    actual_lstm_configs = {(str(row["frequency_setting"]), str(row["horizon"])) for _, row in lstm_df.iterrows()}
    if len(lstm_df) != 9:
        raise ValueError(f"Expected exactly 9 canonicalized LSTM rows; found {len(lstm_df)}")
    if actual_lstm_configs != expected_lstm_configs:
        raise ValueError(f"LSTM MAIN V2 reference does not match the expected 9 frequency x horizon cells: {sorted(actual_lstm_configs)}")

    for _, row in summary_df.iterrows():
        freq = str(row["frequency_setting"])
        horizon = str(row["horizon"])
        reference_horizon = str(row["horizon"]).strip().lower().replace("d", "")
        out = {
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
            "naive_available": True,
            "ridge_available": True,
            "lstm_available": True,
        }

        matched = naive_df[naive_df["horizon"] == reference_horizon].copy()
        if not matched.empty:
            matched["model"] = matched["model"].astype(str).str.lower()
            zero_row = matched[matched["model"] == "zero"]
            train_mean_row = matched[matched["model"] == "train_mean"]
            out["zero_mse"] = float(zero_row["pooled_mse"].iloc[0]) if not zero_row.empty else np.nan
            out["train_mean_mse"] = float(train_mean_row["pooled_mse"].iloc[0]) if not train_mean_row.empty else np.nan
            naive_values = [float(v) for v in [out.get("zero_mse"), out.get("train_mean_mse")] if pd.notna(v)]
            out["best_naive_mse"] = float(np.min(naive_values)) if naive_values else np.nan
            out["naive_definition_used"] = "best_naive_mse = min(zero_mse, train_mean_mse)"
        else:
            out["zero_mse"] = np.nan
            out["train_mean_mse"] = np.nan
            out["best_naive_mse"] = np.nan
            out["naive_definition_used"] = "no_naive_baseline_match_for_horizon"

        matched = ridge_df[(ridge_df["frequency_setting"] == freq) & (ridge_df["horizon"] == reference_horizon)]
        if not matched.empty:
            out["ridge_mse"] = float(matched["pooled_mse"].iloc[0])
            out["ridge_mae"] = float(matched["pooled_mae"].iloc[0])
            out["ridge_corr"] = float(matched["pooled_corr"].iloc[0])
            out["ridge_mean_rank_ic"] = float(matched["mean_rank_ic"].iloc[0])
        else:
            out["ridge_mse"] = np.nan
            out["ridge_mae"] = np.nan
            out["ridge_corr"] = np.nan
            out["ridge_mean_rank_ic"] = np.nan

        matched = lstm_df[(lstm_df["frequency_setting"] == freq) & (lstm_df["horizon"] == reference_horizon) & (lstm_df["model"] == "lstm")]
        if not matched.empty:
            out["lstm_mse"] = float(matched["test_mse"].iloc[0])
            out["lstm_mae"] = float(matched["test_mae"].iloc[0])
            out["lstm_corr"] = float(matched["test_corr"].iloc[0])
            out["lstm_mean_rank_ic"] = float(matched["mean_rank_ic"].iloc[0])
        else:
            out["lstm_mse"] = np.nan
            out["lstm_mae"] = np.nan
            out["lstm_corr"] = np.nan
            out["lstm_mean_rank_ic"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)


def write_final_summary(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    script_path = ROOT / "scripts" / "python" / "panel_baseline_vanilla_transformer.py"
    provenance_path = AUDIT_DIR / "main_v2_vanilla_training_provenance.csv"
    summary_hash = compute_sha256(AUDIT_DIR / "panel_vanilla_summary_metrics.csv") if (AUDIT_DIR / "panel_vanilla_summary_metrics.csv").exists() else "MISSING"
    pred_hash = compute_sha256(AUDIT_DIR / "panel_vanilla_test_predictions.csv") if (AUDIT_DIR / "panel_vanilla_test_predictions.csv").exists() else "MISSING"
    rank_hash = compute_sha256(AUDIT_DIR / "panel_vanilla_rank_ic_by_date.csv") if (AUDIT_DIR / "panel_vanilla_rank_ic_by_date.csv").exists() else "MISSING"
    history_hash = compute_sha256(AUDIT_DIR / "panel_vanilla_training_history.csv") if (AUDIT_DIR / "panel_vanilla_training_history.csv").exists() else "MISSING"
    training_provenance_sha = "MISSING"
    if provenance_path.exists():
        provenance_df = pd.read_csv(provenance_path)
        if not provenance_df.empty:
            training_provenance_sha = str(provenance_df.iloc[0].get("training_script_sha256", "MISSING"))
    finalization_sha = compute_sha256(script_path)
    dataset_manifest_sha = DATASET_MANIFEST_SHA256
    lines = []
    lines.append("MAIN V2")
    lines.append("POST-TEMPORAL-FIX")
    lines.append("Dataset Contract V2")
    lines.append("Vanilla Transformer")
    lines.append("nominal seed = 42")
    lines.append("completed cells = 9 / 9")
    lines.append("")
    lines.append(f"dataset manifest SHA: {dataset_manifest_sha}")
    lines.append(f"training-time script SHA: {training_provenance_sha}")
    lines.append(f"finalization-time script SHA: {finalization_sha}")
    lines.append(f"summary artifact SHA: {summary_hash}")
    lines.append(f"prediction artifact SHA: {pred_hash}")
    lines.append(f"rank-IC artifact SHA: {rank_hash}")
    lines.append(f"training-history artifact SHA: {history_hash}")
    lines.append("")
    lines.append("Architecture:")
    lines.append("  - fixed sinusoidal positional encoding")
    lines.append("  - frequency-specific daily/weekly branches")
    lines.append("  - late fusion with concat -> Linear(128,64) -> ReLU -> Linear(64,1)")
    lines.append("  - final representation uses encoded[:, -1, :]")
    lines.append("  - no causal mask, no router, no PathFormer mechanisms")
    lines.append("")
    for horizon in ["5d", "10d", "20d"]:
        dfx = summary_df[summary_df["horizon"] == horizon]
        if dfx.empty:
            continue
        mse_row = dfx.loc[dfx["test_mse"].idxmin()]
        mse_cfg = str(mse_row["frequency_setting"])
        mse_match = comparison_df[(comparison_df["horizon"] == horizon) & (comparison_df["frequency_setting"] == mse_cfg)]

        rank_values = pd.to_numeric(dfx["mean_rank_ic"], errors="coerce")
        if rank_values.notna().any():
            rank_row = dfx.loc[rank_values.idxmax()]
            rank_cfg = str(rank_row["frequency_setting"])
            rank_value = float(rank_row["mean_rank_ic"])
            rank_line = f"{horizon}: Vanilla best mean Rank IC = {rank_cfg} ({rank_value:.6f})"
            rank_match = comparison_df[(comparison_df["horizon"] == horizon) & (comparison_df["frequency_setting"] == rank_cfg)]
        else:
            rank_line = f"{horizon}: Vanilla best mean Rank IC = undefined"
            rank_match = pd.DataFrame()

        zero_mse = np.nan
        train_mean_mse = np.nan
        ridge_mse_cfg_mse = np.nan
        ridge_mse_cfg_mae = np.nan
        ridge_mse_cfg_corr = np.nan
        ridge_mse_cfg_rank_ic = np.nan
        lstm_mse_cfg_mse = np.nan
        lstm_mse_cfg_mae = np.nan
        lstm_mse_cfg_corr = np.nan
        lstm_mse_cfg_rank_ic = np.nan
        ridge_rank_cfg_rank_ic = np.nan
        lstm_rank_cfg_rank_ic = np.nan

        if not mse_match.empty:
            mse_row_comp = mse_match.iloc[0]
            zero_mse = float(mse_row_comp["zero_mse"]) if pd.notna(mse_row_comp.get("zero_mse")) else np.nan
            train_mean_mse = float(mse_row_comp["train_mean_mse"]) if pd.notna(mse_row_comp.get("train_mean_mse")) else np.nan
            ridge_mse_cfg_mse = float(mse_row_comp["ridge_mse"]) if pd.notna(mse_row_comp.get("ridge_mse")) else np.nan
            ridge_mse_cfg_mae = float(mse_row_comp["ridge_mae"]) if pd.notna(mse_row_comp.get("ridge_mae")) else np.nan
            ridge_mse_cfg_corr = float(mse_row_comp["ridge_corr"]) if pd.notna(mse_row_comp.get("ridge_corr")) else np.nan
            ridge_mse_cfg_rank_ic = float(mse_row_comp["ridge_mean_rank_ic"]) if pd.notna(mse_row_comp.get("ridge_mean_rank_ic")) else np.nan
            lstm_mse_cfg_mse = float(mse_row_comp["lstm_mse"]) if pd.notna(mse_row_comp.get("lstm_mse")) else np.nan
            lstm_mse_cfg_mae = float(mse_row_comp["lstm_mae"]) if pd.notna(mse_row_comp.get("lstm_mae")) else np.nan
            lstm_mse_cfg_corr = float(mse_row_comp["lstm_corr"]) if pd.notna(mse_row_comp.get("lstm_corr")) else np.nan
            lstm_mse_cfg_rank_ic = float(mse_row_comp["lstm_mean_rank_ic"]) if pd.notna(mse_row_comp.get("lstm_mean_rank_ic")) else np.nan
        if not rank_match.empty:
            rank_row_comp = rank_match.iloc[0]
            ridge_rank_cfg_rank_ic = float(rank_row_comp["ridge_mean_rank_ic"]) if pd.notna(rank_row_comp.get("ridge_mean_rank_ic")) else np.nan
            lstm_rank_cfg_rank_ic = float(rank_row_comp["lstm_mean_rank_ic"]) if pd.notna(rank_row_comp.get("lstm_mean_rank_ic")) else np.nan

        lines.append(f"{horizon}: Vanilla best MSE = {mse_cfg} ({mse_row['test_mse']:.6f})")
        lines.append(f"  Same-frequency Ridge for MSE config = mse={ridge_mse_cfg_mse:.6f}, mae={ridge_mse_cfg_mae:.6f}, corr={ridge_mse_cfg_corr:.6f}, mean_rank_ic={ridge_mse_cfg_rank_ic:.6f}")
        lines.append(f"  Same-frequency LSTM for MSE config = mse={lstm_mse_cfg_mse:.6f}, mae={lstm_mse_cfg_mae:.6f}, corr={lstm_mse_cfg_corr:.6f}, mean_rank_ic={lstm_mse_cfg_rank_ic:.6f}")
        lines.append(f"  Zero MSE = {zero_mse:.6f}")
        lines.append(f"  Train-Mean MSE = {train_mean_mse:.6f}")
        if rank_values.notna().any():
            lines.append(f"{horizon}: Vanilla best mean Rank IC = {rank_cfg} ({rank_value:.6f})")
            lines.append(f"  Same-frequency Ridge mean Rank IC for Rank-IC config = {ridge_rank_cfg_rank_ic:.6f}")
            lines.append(f"  Same-frequency LSTM mean Rank IC for Rank-IC config = {lstm_rank_cfg_rank_ic:.6f}")
        else:
            lines.append(f"{horizon}: Vanilla best mean Rank IC = undefined")
        lines.append("")

    lines.append("Prediction dispersion diagnostics:")
    for _, row in summary_df.iterrows():
        lines.append(
            f"  {row['frequency_setting']} / {row['horizon']}: pred_std={float(row['pred_std']):.6f}, "
            f"true_std={float(row['true_std']):.6f}, pred_std_true_std_ratio={float(row['pred_std_true_std_ratio']):.6f}"
        )
    lines.append("")
    lines.append("PRE-TEMPORAL-FIX results were not used for formal comparison or model selection.")
    lines.append("Do not claim significance, robustness, profitability, causality, Daily+Weekly complementarity, or router success.")
    lines.append("")
    path = AUDIT_DIR / "panel_vanilla_final_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_v2_vanilla_run_manifest(output_dir: Path, summary_df: pd.DataFrame, pred_df: pd.DataFrame, rank_ic_df: pd.DataFrame, history_df: pd.DataFrame) -> Path:
    manifest_path = output_dir / "main_v2_vanilla_run_manifest.csv"
    training_provenance = output_dir / "main_v2_vanilla_training_provenance.csv"
    if not training_provenance.exists():
        raise FileNotFoundError(f"Training provenance file is required before writing the formal run manifest: {training_provenance}")
    provenance_df = pd.read_csv(training_provenance)
    if provenance_df.empty:
        raise ValueError(f"Training provenance file is empty: {training_provenance}")
    row = provenance_df.iloc[0]
    validate_training_provenance_row(row)
    stored_sha = row.get("training_script_sha256")
    if pd.isna(stored_sha):
        raise ValueError(f"Training provenance file is missing training_script_sha256: {training_provenance}")
    current_script_sha = compute_sha256(ROOT / "scripts" / "python" / "panel_baseline_vanilla_transformer.py")
    manifest = {
        "run_family": "MAIN_V2",
        "model": "VANILLA_TRANSFORMER",
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
        "completed_cells": 9,
        "predictions_per_cell": 6647,
        "total_prediction_rows": int(len(pred_df)),
        "rank_ic_dates_per_cell": 391,
        "total_rank_ic_rows": int(len(rank_ic_df)),
        "summary_rows": int(len(summary_df)),
        "training_history_rows": int(len(history_df)),
        "training_script": "scripts/python/panel_baseline_vanilla_transformer.py",
        "training_script_sha256": str(stored_sha),
        "finalization_script": "scripts/python/panel_baseline_vanilla_transformer.py",
        "finalization_script_sha256": current_script_sha,
        "artifact_dir": str(output_dir.resolve()),
        "summary_artifact_sha256": compute_sha256(output_dir / "panel_vanilla_summary_metrics.csv") if (output_dir / "panel_vanilla_summary_metrics.csv").exists() else "MISSING",
        "predictions_artifact_sha256": compute_sha256(output_dir / "panel_vanilla_test_predictions.csv") if (output_dir / "panel_vanilla_test_predictions.csv").exists() else "MISSING",
        "rank_ic_artifact_sha256": compute_sha256(output_dir / "panel_vanilla_rank_ic_by_date.csv") if (output_dir / "panel_vanilla_rank_ic_by_date.csv").exists() else "MISSING",
        "training_history_artifact_sha256": compute_sha256(output_dir / "panel_vanilla_training_history.csv") if (output_dir / "panel_vanilla_training_history.csv").exists() else "MISSING",
        "training_provenance_artifact": str(training_provenance.resolve()) if training_provenance.exists() else "MISSING",
        "historical_results_used_for_training": False,
        "historical_results_used_for_model_selection": False,
        "historical_results_used_for_formal_comparison": False,
        "naive_reference": str((DEFAULT_AUDIT_DIR / "main_v2" / "naive" / "main_v2_naive_summary.csv").resolve()),
        "ridge_reference": str((DEFAULT_AUDIT_DIR / "main_v2" / "ridge" / "main_v2_ridge_comparison.csv").resolve()),
        "lstm_reference": str((DEFAULT_AUDIT_DIR / "main_v2" / "lstm" / "panel_lstm_summary_metrics.csv").resolve()),
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "activation": "relu",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": 100,
        "patience": 10,
        "max_norm": 1.0,
    }
    pd.DataFrame([manifest]).to_csv(manifest_path, index=False)
    return manifest_path


def full_benchmark(ds, stats: dict, output_dir: Path | None = None) -> None:
    output_dir = resolve_output_dir(output_dir)
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    total_experiments = len(FREQUENCY_SETTINGS) * len(HORIZONS)
    script_start = time.perf_counter()

    print("============================================================")
    print("VANILLA TRANSFORMER PANEL BASELINE RUNSETUP")
    print(f"device = {device.type.upper()}")
    print(f"total_experiments = {total_experiments}")
    print(f"batch_size = {BATCH_SIZE}")
    print(f"learning_rate = {LEARNING_RATE}")
    print(f"max_epochs = {MAX_EPOCHS}")
    print(f"early_stopping_patience = {EARLY_STOPPING_PATIENCE}")
    print("============================================================")

    results = []
    experiment_runtimes = []
    for experiment_idx, frequency in enumerate(FREQUENCY_SETTINGS, start=1):
        for horizon in HORIZONS:
            exp_counter = (experiment_idx - 1) * len(HORIZONS) + [h for h in HORIZONS].index(horizon) + 1
            print(f"\n[Experiment {exp_counter}/{total_experiments}]")
            print(f"frequency_setting = {frequency}")
            print(f"horizon = {horizon}d")
            res = train_and_eval_frequency(ds, frequency, int(horizon), stats)
            results.append(res)
            experiment_runtimes.append(float(res["runtime_seconds"]))
            append_experiment_outputs(res)

            completed = len(experiment_runtimes)
            total_elapsed = time.perf_counter() - script_start
            mean_experiment_time = float(np.mean(experiment_runtimes)) if experiment_runtimes else 0.0
            remaining_experiments = max(total_experiments - completed, 0)
            estimated_remaining = mean_experiment_time * remaining_experiments
            print(f"Experiment runtime: {format_elapsed(res['runtime_seconds'])}")
            print(f"n_trainable_parameters = {res['n_trainable_parameters']}")
            print(f"best_epoch = {res['best_epoch']}")
            print(f"best_validation_loss = {res['val_loss']:.6f}")
            print(f"test_mse = {res['test_mse']:.6f}")
            print(f"mean_rank_ic = {res['mean_rank_ic']:.6f}")
            print(f"PredStd/TrueStd = {res['pred_std_true_std_ratio']:.4f}")
            print(f"no_extreme_scale_pathology = {res['no_extreme_scale_pathology']}")
            print(f"\nProgress: {completed}/{total_experiments} experiments completed")
            print(f"Total elapsed: {format_elapsed(total_elapsed)}")
            print(f"Estimated remaining: ~{format_elapsed(estimated_remaining)} (approximate; early stopping epochs differ)")

    summary_df = pd.read_csv(output_dir / "panel_vanilla_summary_metrics.csv")
    pred_df = pd.read_csv(output_dir / "panel_vanilla_test_predictions.csv")
    rank_ic_df = pd.read_csv(output_dir / "panel_vanilla_rank_ic_by_date.csv")
    history_df = pd.read_csv(output_dir / "panel_vanilla_training_history.csv")
    validate_summary_rows(summary_df)
    validate_prediction_rows(pred_df)
    validate_rank_ic_rows(rank_ic_df)
    validate_training_history(history_df, summary_df)
    validate_cross_artifact_cell_consistency(summary_df, pred_df, rank_ic_df, history_df)
    validate_cross_frequency_y_true_identity(pred_df)
    validate_expected_parameter_counts(summary_df)

    print("\n===========================================================")
    print("VANILLA TRANSFORMER PANEL CONTROL COMPLETE")
    print(f"Total runtime: {format_elapsed(time.perf_counter() - script_start)}")
    print(f"Completed experiments: {len(results)}/{total_experiments}")
    print("===========================================================")


def run_smoke_validation() -> None:
    MAIN_V2_VANILLA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    stats = fit_train_only_normalizer(ds)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print("FROZEN PANEL VALIDATION")
    print(f"train={int(np.sum(ds.split == 'train'))}")
    print(f"val={int(np.sum(ds.split == 'val'))}")
    print(f"test={int(np.sum(ds.split == 'test'))}")
    print(f"device={device.type.upper()}")
    print("sequence_orientation=oldest->newest (final timestep is latest historical observation)")
    print("============================================================")

    smoke_results = smoke_test_models(ds, stats, device)
    for frequency, info in smoke_results.items():
        print(f"{frequency}: smoke_ok={info['ok']} param_count={info['param_count']}")

    print("\nSmoke validation passed: forward/backward and gradient checks OK.")
    print("Formal 9-experiment training is intentionally not started in this stage.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-panel Vanilla Transformer control")
    parser.add_argument("--train", action="store_true", help="Run formal training. Requires either --all or both --frequency and --horizon.")
    parser.add_argument("--all", action="store_true", help="Run the full 9-experiment benchmark.")
    parser.add_argument("--frequency", choices=FREQUENCY_SETTINGS, help="Single-frequency training selection: daily_only | weekly_only | daily_weekly")
    parser.add_argument("--horizon", choices=["5d", "10d", "20d"], help="Single-horizon training selection: 5d | 10d | 20d")
    parser.add_argument("--output-dir", type=str, default=str(MAIN_V2_VANILLA_DIR), help="Formal Vanilla MAIN V2 output root; must be dataset/audit/main_v2/vanilla.")
    parser.add_argument("--finalize-only", action="store_true", help="Finalize existing raw MAIN V2 Vanilla evidence without retraining.")
    args = parser.parse_args()

    if args.finalize_only and args.train:
        parser.error("--finalize-only cannot be combined with --train.")

    if args.train:
        if args.all and (args.frequency is not None or args.horizon is not None):
            parser.error("Use either --all or a single --frequency/--horizon pair, not both.")
        if not args.all and (args.frequency is None or args.horizon is None):
            parser.error("When --train is supplied, either use --all or provide both --frequency and --horizon.")
        if args.frequency is not None and args.horizon is None:
            parser.error("If --frequency is supplied, --horizon must also be supplied.")
        if args.horizon is not None and args.frequency is None:
            parser.error("If --horizon is supplied, --frequency must also be supplied.")

    if args.output_dir is not None:
        resolve_output_dir(args.output_dir)

    return args


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    if args.finalize_only:
        output_dir = resolve_output_dir(args.output_dir)
        provenance_path = output_dir / "main_v2_vanilla_training_provenance.csv"
        if not provenance_path.exists():
            raise FileNotFoundError(f"Missing formal training provenance artifact for finalize-only: {provenance_path}")
        provenance_df = pd.read_csv(provenance_path)
        if provenance_df.empty:
            raise ValueError(f"Training provenance file is empty: {provenance_path}")
        validate_training_provenance_row(provenance_df.iloc[0])
        if not (output_dir / "panel_vanilla_summary_metrics.csv").exists():
            raise FileNotFoundError(f"Missing formal summary artifact for finalize-only: {output_dir / 'panel_vanilla_summary_metrics.csv'}")
        if not (output_dir / "panel_vanilla_test_predictions.csv").exists():
            raise FileNotFoundError(f"Missing formal prediction artifact for finalize-only: {output_dir / 'panel_vanilla_test_predictions.csv'}")
        if not (output_dir / "panel_vanilla_rank_ic_by_date.csv").exists():
            raise FileNotFoundError(f"Missing formal Rank-IC artifact for finalize-only: {output_dir / 'panel_vanilla_rank_ic_by_date.csv'}")
        if not (output_dir / "panel_vanilla_training_history.csv").exists():
            raise FileNotFoundError(f"Missing formal training-history artifact for finalize-only: {output_dir / 'panel_vanilla_training_history.csv'}")

        summary_df = pd.read_csv(output_dir / "panel_vanilla_summary_metrics.csv")
        pred_df = pd.read_csv(output_dir / "panel_vanilla_test_predictions.csv")
        rank_ic_df = pd.read_csv(output_dir / "panel_vanilla_rank_ic_by_date.csv")
        history_df = pd.read_csv(output_dir / "panel_vanilla_training_history.csv")

        validate_summary_rows(summary_df)
        validate_prediction_rows(pred_df)
        validate_rank_ic_rows(rank_ic_df)
        validate_training_history(history_df, summary_df)
        validate_cross_artifact_cell_consistency(summary_df, pred_df, rank_ic_df, history_df)
        validate_cross_frequency_y_true_identity(pred_df)
        validate_expected_parameter_counts(summary_df)

        comparison_df = build_baseline_comparison(summary_df)
        comparison_df.to_csv(output_dir / "main_v2_vanilla_comparison.csv", index=False)
        write_main_v2_vanilla_run_manifest(output_dir, summary_df, pred_df, rank_ic_df, history_df)
        write_final_summary(summary_df, comparison_df)
        print("Finalize-only: derived MAIN V2 Vanilla outputs regenerated without retraining.")
        return

    if args.train:
        output_dir = resolve_output_dir(args.output_dir)
        script_path = ROOT / "scripts" / "python" / "panel_baseline_vanilla_transformer.py"
        write_training_provenance(output_dir, script_path)
        ds = load_frozen_panel()
        assert_frozen_panel_checks(ds)
        stats = fit_train_only_normalizer(ds)
        if args.all:
            full_benchmark(ds, stats, output_dir=output_dir)
        else:
            frequency = args.frequency
            horizon = int(args.horizon[:-1])
            result = train_and_eval_frequency(ds, frequency, horizon, stats)
            summary_df = pd.DataFrame([{
                "model": result["model"],
                "frequency_setting": result["frequency_setting"],
                "horizon": result["horizon"],
                "seed": result["seed"],
                "n_trainable_parameters": result["n_trainable_parameters"],
                "best_epoch": result["best_epoch"],
                "epochs_trained": result["epochs_trained"],
                "runtime_seconds": result["runtime_seconds"],
                "val_loss": result["val_loss"],
                "test_mse": result["test_mse"],
                "test_mae": result["test_mae"],
                "test_corr": result["test_corr"],
                "ticker_avg_mse": result["ticker_avg_mse"],
                "ticker_avg_mae": result["ticker_avg_mae"],
                "ticker_avg_corr": result["ticker_avg_corr"],
                "mean_rank_ic": result["mean_rank_ic"],
                "median_rank_ic": result["median_rank_ic"],
                "rank_ic_std": result["rank_ic_std"],
                "positive_ic_ratio": result["positive_ic_ratio"],
                "rank_icir": result["rank_icir"],
                "direction_accuracy": result["direction_accuracy"],
                "pred_std": result["pred_std"],
                "true_std": result["true_std"],
                "pred_std_true_std_ratio": result["pred_std_true_std_ratio"],
                "no_extreme_scale_pathology": result["no_extreme_scale_pathology"],
            }])
            save_summary_rows([result], append_mode=True)
            save_prediction_rows([result], append_mode=True)
            save_training_history([result], append_mode=True)
            save_rank_ic_by_date([result], append_mode=True)
            print(f"Single-config run complete: {frequency} / {args.horizon}")
    else:
        run_smoke_validation()


if __name__ == "__main__":
    main()
