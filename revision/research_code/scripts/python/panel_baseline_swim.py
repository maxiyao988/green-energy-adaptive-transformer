"""SWiM-style temporal shifted-window Transformer control.

This script implements the structured local / shifted-window temporal attention
control under the frozen 17-stock Daily + Weekly panel protocol defined by
`panel_common.py` and the accepted MAIN V2 benchmark contract.

Design constraints (frozen for this v1; see project audit before changing):
- frequency-specific branches + late concat fusion (no early fusion, no
  cross-frequency attention, no router, no adaptive/multi-scale mechanism)
- exactly two window blocks per branch: block 1 = local window attention
  (shift=0), block 2 = shifted-window attention (shift = window_size // 2)
- Daily window_size=10, shift_size=5; Weekly window_size=4, shift_size=2
  (development baseline values; NOT derived from a search, NOT the historical
  FSLR window_size=16, which belonged to a different 160-token concatenated
  sequence and is not applicable here)
- Post-LN residual structure to stay close to the frozen panel Vanilla
  Transformer's normalization convention
- a mathematically valid shifted-window boundary mask rather than uncontrolled
  cyclic wrap-around leakage
- zero padding with an explicit validity mask threaded through padding,
  cyclic shift, window partition/reverse, and pooling
- masked mean pooling over real timesteps only
- reuses the frozen panel loader, split, normalization, training protocol,
  and evaluation functions from `panel_common.py`, and reuses several
  frequency-agnostic helpers directly from `panel_baseline_vanilla_transformer.py`
  (imported only; that file is not modified)

Formal MAIN V2 governance: the accepted output directory is fixed to
`dataset/audit/main_v2/swim`, and the script is intentionally hardened for
formal evidence locking. Default behavior is smoke/audit validation only; the
full 9-experiment benchmark is intentionally not launched by this task unless
`--train` is explicitly supplied.
"""

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

from scripts.python.panel_common import (
    HORIZONS,
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    load_frozen_panel,
    rank_ic_by_date,
)
from scripts.python.panel_baseline_vanilla_transformer import (
    BATCH_SIZE,
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    FIXED_MAX_LEN,
    FREQUENCY_SETTINGS,
    LEARNING_RATE,
    MAX_EPOCHS,
    NHEAD,
    SEED,
    PositionalEncoding,
    assert_frozen_panel_checks,
    compute_rank_ic_summary,
    count_trainable_parameters,
    format_elapsed,
    prediction_scale_looks_normal,
    set_seed,
)

# =====================================================================
# FROZEN SWiM v1 DESIGN CONSTANTS -- do not tune in this task
# =====================================================================

NUM_BLOCKS = 2
WINDOW_CONFIG = {
    "daily": {"window_size": 10, "shift_size": 5},
    "weekly": {"window_size": 4, "shift_size": 2},
}

DEBUG_NAN_MODE = False
DEBUG_NAN_CONTEXT = {"frequency": None, "batch_index": None}

DATASET_MANIFEST_SHA256 = "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c"
MAIN_V2_SWIM_DIR = ROOT / "dataset" / "audit" / "main_v2" / "swim"
EXPECTED_SWIM_PARAM_COUNTS = {
    "daily_only": 67393,
    "weekly_only": 67393,
    "daily_weekly": 142977,
}
FORMAL_SWIM_RUN_FILES = {
    "training_provenance": "main_v2_swim_training_provenance.csv",
    "summary": "panel_swim_summary_metrics.csv",
    "predictions": "panel_swim_test_predictions.csv",
    "rank_ic": "panel_swim_rank_ic_by_date.csv",
    "history": "panel_swim_training_history.csv",
    "comparison": "main_v2_swim_comparison.csv",
    "manifest": "main_v2_swim_run_manifest.csv",
    "final_summary": "panel_swim_final_summary.txt",
}
MAIN_V2_BASELINE_DIR = ROOT / "dataset" / "audit" / "main_v2"
MAIN_V2_BASELINES = {
    "naive": MAIN_V2_BASELINE_DIR / "naive" / "panel_naive_baseline_summary.csv",
    "ridge": MAIN_V2_BASELINE_DIR / "ridge" / "panel_ridge_summary_metrics.csv",
    "lstm": MAIN_V2_BASELINE_DIR / "lstm" / "panel_lstm_summary_metrics.csv",
    "vanilla": MAIN_V2_BASELINE_DIR / "vanilla" / "panel_vanilla_summary_metrics.csv",
}


def compute_sha256(path: Path) -> str:
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
        raise ValueError(f"SWiM formal MAIN V2 output directory must be exactly {expected}; got {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _normalize_horizon_label(horizon_value) -> str:
    if isinstance(horizon_value, str):
        return horizon_value
    return f"{int(horizon_value)}d"


def validate_summary_rows(summary_df: pd.DataFrame) -> None:
    required = {
        "model", "frequency_setting", "horizon", "seed", "n_trainable_parameters", "best_epoch",
        "epochs_trained", "runtime_seconds", "val_loss", "test_mse", "test_mae", "test_corr",
        "ticker_avg_mse", "ticker_avg_mae", "ticker_avg_corr", "mean_rank_ic", "median_rank_ic",
        "rank_ic_std", "positive_ic_ratio", "rank_icir", "n_valid_ic_dates", "direction_accuracy",
        "pred_std", "true_std", "pred_std_true_std_ratio", "no_extreme_scale_pathology",
    }
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"SWiM summary artifact missing required columns: {missing}")

    finite_cols = [
        "n_trainable_parameters", "best_epoch", "epochs_trained", "runtime_seconds", "val_loss",
        "test_mse", "test_mae", "ticker_avg_mse", "ticker_avg_mae", "n_valid_ic_dates",
        "direction_accuracy", "pred_std", "true_std", "pred_std_true_std_ratio",
    ]
    allow_nan_cols = ["test_corr", "ticker_avg_corr", "mean_rank_ic", "median_rank_ic", "rank_ic_std", "positive_ic_ratio", "rank_icir"]

    for col in finite_cols:
        if col in summary_df.columns:
            bad = pd.to_numeric(summary_df[col], errors="coerce")
            if bad.isna().any():
                raise ValueError(f"SWiM summary has NaN in required-finite column: {col}")
            if np.isinf(bad.to_numpy()).any():
                raise ValueError(f"SWiM summary has ±Inf in required-finite column: {col}")

    for col in allow_nan_cols:
        if col in summary_df.columns:
            bad = pd.to_numeric(summary_df[col], errors="coerce")
            if np.isinf(bad.to_numpy()).any():
                raise ValueError(f"SWiM summary has ±Inf in correlation/Rank-IC column: {col}")


def validate_prediction_rows(pred_df: pd.DataFrame) -> None:
    required = {"model", "ticker", "anchor_date", "horizon", "frequency_setting", "y_true", "y_pred", "seed", "best_epoch"}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise ValueError(f"SWiM predictions artifact missing required columns: {missing}")
    if pred_df.empty:
        raise ValueError("SWiM predictions artifact cannot be empty.")
    key_cols = ["model", "ticker", "anchor_date", "horizon", "frequency_setting", "seed", "best_epoch"]
    if pred_df.duplicated(subset=key_cols).any():
        raise ValueError("SWiM prediction artifact contains duplicate formal keys.")
    if not np.isfinite(pd.to_numeric(pred_df["y_true"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM prediction artifact contains non-finite y_true values.")
    if not np.isfinite(pd.to_numeric(pred_df["y_pred"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM prediction artifact contains non-finite y_pred values.")


def validate_rank_ic_rows(rank_ic_df: pd.DataFrame) -> None:
    required = {"model", "frequency_setting", "horizon", "anchor_date", "n_tickers", "rank_ic"}
    missing = sorted(required - set(rank_ic_df.columns))
    if missing:
        raise ValueError(f"SWiM Rank-IC artifact missing required columns: {missing}")
    if rank_ic_df.empty:
        raise ValueError("SWiM Rank-IC artifact cannot be empty.")
    if rank_ic_df.duplicated(subset=["model", "frequency_setting", "horizon", "anchor_date"]).any():
        raise ValueError("SWiM Rank-IC artifact contains duplicate formal keys.")
    if np.isinf(pd.to_numeric(rank_ic_df["rank_ic"], errors="coerce").to_numpy()).any():
        raise ValueError("SWiM Rank-IC artifact contains ±Inf rank_ic values.")
    if (pd.to_numeric(rank_ic_df["n_tickers"], errors="coerce") != 17).any():
        raise ValueError("SWiM Rank-IC artifact has invalid n_tickers values.")


def validate_training_history(history_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    required = {"frequency_setting", "horizon", "epoch", "train_loss", "val_loss", "elapsed_seconds"}
    missing = sorted(required - set(history_df.columns))
    if missing:
        raise ValueError(f"SWiM training history artifact missing required columns: {missing}")
    if history_df.empty:
        raise ValueError("SWiM training history artifact cannot be empty.")
    if not np.isfinite(pd.to_numeric(history_df["train_loss"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM training history contains non-finite train_loss values.")
    if not np.isfinite(pd.to_numeric(history_df["val_loss"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM training history contains non-finite val_loss values.")
    if history_df["epoch"].le(0).any() or history_df["epoch"].gt(100).any():
        raise ValueError("SWiM training history contains epoch outside [1, 100].")
    if history_df.duplicated(subset=["frequency_setting", "horizon", "epoch"]).any():
        raise ValueError("SWiM training history has duplicate (frequency_setting, horizon, epoch) keys.")
    history_cells = set(zip(history_df["frequency_setting"], history_df["horizon"]))
    summary_cells = set(zip(summary_df["frequency_setting"], summary_df["horizon"]))
    if history_cells != summary_cells:
        raise ValueError(f"SWiM training history cell set mismatch: history={sorted(history_cells)} summary={sorted(summary_cells)}")
    for _, row in summary_df.iterrows():
        cell = history_df[(history_df["frequency_setting"] == row["frequency_setting"]) & (history_df["horizon"] == row["horizon"])].copy()
        if cell.empty:
            raise ValueError(f"Missing training history for SWiM cell: {row['frequency_setting']} / {row['horizon']}")
        epochs = cell["epoch"].astype(int)
        if int(row["best_epoch"]) not in epochs.tolist():
            raise ValueError(f"SWiM summary best_epoch not found in history for {row['frequency_setting']} / {row['horizon']}")
        if int(row["epochs_trained"]) != int(epochs.max()):
            raise ValueError(f"SWiM summary epochs_trained mismatch for {row['frequency_setting']} / {row['horizon']}")
        min_idx = cell["val_loss"].idxmin()
        min_loss = float(cell.loc[min_idx, "val_loss"])
        min_loss_epochs = set(cell.loc[np.isclose(cell["val_loss"].astype(float), min_loss, rtol=1e-6, atol=1e-8), "epoch"].astype(int).tolist())
        if int(row["best_epoch"]) not in min_loss_epochs:
            raise ValueError(f"SWiM summary best_epoch is not a min-val-loss epoch for {row['frequency_setting']} / {row['horizon']}")
        if not np.isclose(float(row["val_loss"]), min_loss, rtol=1e-6, atol=1e-8):
            raise ValueError(f"SWiM summary val_loss does not match min validation loss for {row['frequency_setting']} / {row['horizon']}")


def validate_cross_frequency_y_true_identity(pred_df: pd.DataFrame) -> None:
    key = ["ticker", "anchor_date", "horizon"]
    grouped = pred_df.groupby(key, dropna=False)
    for _, group in grouped:
        values = group["y_true"].drop_duplicates().tolist()
        if len(values) > 1:
            raise ValueError("Cross-frequency y_true identity failed: identical horizon/ticker/date must share one target label across frequencies.")


def validate_main_v2_swim_evidence(output_dir: Path) -> None:
    summary = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["summary"])
    pred = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["predictions"])
    rank_ic = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"])
    history = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["history"])
    if len(summary) != 9:
        raise ValueError(f"SWiM summary validation failed: expected 9 rows, got {len(summary)}")
    if summary["model"].nunique() != 1 or summary["model"].iloc[0] != "swim_transformer":
        raise ValueError("SWiM summary model contract failed.")
    if summary["seed"].nunique() != 1 or int(summary["seed"].iloc[0]) != 42:
        raise ValueError("SWiM summary seed contract failed.")
    expected_cells = {"daily_only", "weekly_only", "daily_weekly"} 
    expected_horizons = {"5d", "10d", "20d"}
    cells = set((freq, hor) for freq, hor in zip(summary["frequency_setting"], summary["horizon"]))
    required_cells = {(freq, hor) for freq in ["daily_only", "weekly_only", "daily_weekly"] for hor in ["5d", "10d", "20d"]}
    if cells != required_cells:
        raise ValueError(f"SWiM summary cells mismatch: expected {sorted(required_cells)}, got {sorted(cells)}")
    if summary.duplicated(subset=["frequency_setting", "horizon"]).any():
        raise ValueError("SWiM summary has duplicate (frequency_setting, horizon) rows.")
    if set(summary["frequency_setting"]) != expected_cells:
        raise ValueError("SWiM summary frequencies mismatch.")
    if set(summary["horizon"]) != expected_horizons:
        raise ValueError("SWiM summary horizons mismatch.")
    for key, expected in EXPECTED_SWIM_PARAM_COUNTS.items():
        observed = summary.loc[summary["frequency_setting"] == key, "n_trainable_parameters"]
        if len(observed) != 3 or not np.all(observed.astype(int).to_numpy() == int(expected)):
            raise ValueError(f"SWiM parameter count contract failed for {key}: expected {expected}, got {observed.tolist()}")
    validate_summary_rows(summary)
    if len(pred) != 59823:
        raise ValueError(f"SWiM prediction row count mismatch: expected 59823, got {len(pred)}")
    pred_cell_groups = pred.groupby(["frequency_setting", "horizon"])
    if len(pred_cell_groups) != 9:
        raise ValueError("SWiM predictions do not cover all 9 formal cells.")
    per_cell = pred_cell_groups.size()
    if not np.all(per_cell == 6647):
        raise ValueError(f"SWiM predictions per-cell row count mismatch: {per_cell.to_dict()}")
    for (freq, hor), cell_df in pred_cell_groups:
        if cell_df["ticker"].nunique() != 17:
            raise ValueError(f"SWiM predictions ticker count mismatch for {freq}/{hor}")
        if cell_df["anchor_date"].nunique() != 391:
            raise ValueError(f"SWiM predictions anchor-date count mismatch for {freq}/{hor}")
        if set(cell_df["seed"].astype(int).tolist()) != {42}:
            raise ValueError(f"SWiM predictions seed contract failed for {freq}/{hor}")
        if set(cell_df["model"].astype(str).tolist()) != {"swim_transformer"}:
            raise ValueError(f"SWiM predictions model contract failed for {freq}/{hor}")
    if pred.duplicated(subset=["model", "ticker", "anchor_date", "horizon", "frequency_setting", "seed", "best_epoch"]).any():
        raise ValueError("SWiM predictions contain duplicate formal keys.")
    if not np.isfinite(pd.to_numeric(pred["y_true"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM prediction artifact contains non-finite y_true values.")
    if not np.isfinite(pd.to_numeric(pred["y_pred"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM prediction artifact contains non-finite y_pred values.")
    validate_prediction_rows(pred)
    if len(rank_ic) != 3519:
        raise ValueError(f"SWiM Rank-IC row count mismatch: expected 3519, got {len(rank_ic)}")
    if set(rank_ic["model"].astype(str).tolist()) != {"swim_transformer"}:
        raise ValueError("SWiM Rank-IC model contract failed.")
    if rank_ic.duplicated(subset=["model", "frequency_setting", "horizon", "anchor_date"]).any():
        raise ValueError("SWiM Rank-IC artifact contains duplicate formal keys.")
    per_date = rank_ic.groupby(["frequency_setting", "horizon", "anchor_date"]).size()
    if not np.all(per_date == 1):
        raise ValueError("SWiM Rank-IC row-count contract violated: expected exactly one row per (freq,horizon,date).")
    if (pd.to_numeric(rank_ic["n_tickers"], errors="coerce") != 17).any():
        raise ValueError("SWiM Rank-IC n_tickers contract failed: every row must have n_tickers == 17.")
    rank_ic_cell_groups = rank_ic.groupby(["frequency_setting", "horizon"])
    if not np.all(rank_ic_cell_groups.size() == 391):
        raise ValueError("SWiM Rank-IC per-cell row count contract failed: expected 391 rows per cell.")
    for (freq, hor), cell_df in rank_ic_cell_groups:
        if cell_df["anchor_date"].nunique() != 391:
            raise ValueError(f"SWiM Rank-IC unique-date contract failed for {freq}/{hor}")
    if np.isinf(pd.to_numeric(rank_ic["rank_ic"], errors="coerce").to_numpy()).any():
        raise ValueError("SWiM Rank-IC artifact contains ±Inf rank_ic values.")
    validate_rank_ic_rows(rank_ic)
    if len(history) < 1:
        raise ValueError("SWiM training history artifact cannot be empty.")
    if history["frequency_setting"].nunique() != 3 or history["horizon"].nunique() != 3:
        raise ValueError("SWiM history does not cover all 9 cells.")
    if (history["epoch"] < 1).any() or (history["epoch"] > 100).any():
        raise ValueError("SWiM history epoch range contract failed.")
    if not np.isfinite(pd.to_numeric(history["train_loss"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM history contains non-finite train_loss values.")
    if not np.isfinite(pd.to_numeric(history["val_loss"], errors="coerce").to_numpy()).all():
        raise ValueError("SWiM history contains non-finite val_loss values.")
    validate_training_history(history, summary)
    validate_cross_frequency_y_true_identity(pred)


def validate_frozen_dataset_contract(ds: object) -> None:
    expected_tickers = [
        "AES", "BEP", "BLDP", "BLNK", "CSIQ", "CWEN", "DQ", "ENPH",
        "FCEL", "FSLR", "HASI", "JKS", "NEE", "ORA", "PLUG", "RUN", "SEDG",
    ]
    ticker_values = sorted(np.unique(ds.ticker).tolist())
    if len(ticker_values) != 17:
        raise RuntimeError(f"Ticker count mismatch: {len(ticker_values)} != 17")
    if ticker_values != sorted(expected_tickers):
        raise RuntimeError(f"Ticker universe mismatch: {ticker_values} != {sorted(expected_tickers)}")
    common_dates = np.unique(ds.anchor_date)
    if len(common_dates) != 2563:
        raise RuntimeError(f"Common anchor date count mismatch: {len(common_dates)} != 2563")


def assert_clean_main_v2_swim_namespace(output_dir: Path) -> None:
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
            f"SWiM formal namespace is not clean at {output_dir}. Existing formal artifacts: {stale}. "
            "Refusing to overwrite formal evidence; archive or remove stale files manually before --train --all."
        )


def _cell_set_from_df(df: pd.DataFrame, freq_col: str, horizon_col: str) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(zip(df[freq_col].astype(str), df[horizon_col].astype(str)))


def validate_partial_raw_cell_set_consistency(output_dir: Path) -> set[tuple[str, str]]:
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
            f"Corrupted partial raw SWiM state: some raw artifacts exist but not all four. Presence={existing}. "
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

    cell_sets: dict[str, set[tuple[str, str]]] = {
        "summary": _cell_set_from_df(summary_df, "frequency_setting", "horizon"),
        "predictions": _cell_set_from_df(pred_df, "frequency_setting", "horizon"),
        "rank_ic": _cell_set_from_df(rank_ic_df, "frequency_setting", "horizon"),
        "history": _cell_set_from_df(hist_df, "frequency_setting", "horizon"),
    }
    base_name, base_set = next(iter(cell_sets.items()))
    for name, cs in cell_sets.items():
        if cs != base_set:
            raise ValueError(
                f"Existing partial SWiM raw artifacts disagree on completed cell sets: "
                f"{base_name}={sorted(base_set)} vs {name}={sorted(cs)}. Manual inspection required."
            )
    return base_set


def ensure_requested_cell_absent(output_dir: Path, frequency: str, horizon: str, seed: int = 42) -> None:
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

    hits = {
        "summary": hit_summary,
        "predictions": hit_pred,
        "rank_ic": hit_rank_ic,
        "history": hit_history,
    }
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
    normalized = canonicalize_main_v2_reference_df(
        df,
        required_cols=["frequency", "horizon", "test_mse", "test_mae", "corr", "direction_accuracy", "mean_rank_ic", "pred_std_true_std_ratio"],
        label="ridge",
    )
    normalized = normalized.rename(columns={"test_mse": "pooled_mse", "test_mae": "pooled_mae", "corr": "pooled_corr"})
    normalized = normalized[["frequency_setting", "horizon", "pooled_mse", "pooled_mae", "pooled_corr", "direction_accuracy", "mean_rank_ic", "pred_std_true_std_ratio"]].copy()
    expected = {(freq, horizon) for freq in ["daily_only", "weekly_only", "daily_weekly"] for horizon in ["5", "10", "20"]}
    observed = {(str(row["frequency_setting"]), str(row["horizon"])) for _, row in normalized.iterrows()}
    if len(normalized) != 9:
        raise ValueError(f"Expected exactly 9 Ridge rows after canonicalization, found {len(normalized)}")
    if normalized.duplicated(subset=["frequency_setting", "horizon"]).any():
        dupes = normalized[normalized.duplicated(subset=["frequency_setting", "horizon"], keep=False)]
        raise ValueError(f"Ridge canonicalized data contains duplicate frequency_setting/horizon rows: {dupes.to_dict(orient='records')}")
    if observed != expected:
        raise ValueError(f"Ridge configs do not match expected 9 frequency x horizon settings: {sorted(observed)}")
    return normalized


def canonicalize_main_v2_naive(df: pd.DataFrame) -> pd.DataFrame:
    normalized = canonicalize_main_v2_reference_df(df, required_cols=["model", "horizon", "pooled_mse"], label="naive")
    formal = normalized[normalized["model"].isin(["zero", "train_mean"])].copy()
    expected = {(model, horizon) for model in ["zero", "train_mean"] for horizon in ["5", "10", "20"]}
    observed = {(str(row["model"]), str(row["horizon"])) for _, row in formal.iterrows()}
    if len(formal) != 6:
        raise ValueError(f"Naive MAIN V2 summary must contain exactly 6 required formal rows, found {len(formal)}")
    if formal.duplicated(subset=["model", "horizon"]).any():
        dupes = formal[formal.duplicated(subset=["model", "horizon"], keep=False)]
        raise ValueError(f"Naive MAIN V2 summary contains duplicate model/horizon rows: {dupes.to_dict(orient='records')}")
    if observed != expected:
        raise ValueError(f"Naive MAIN V2 summary does not match the required formal zero/train_mean x 5/10/20 cells: {sorted(observed)}")
    return formal


def canonicalize_main_v2_lstm_or_vanilla(df: pd.DataFrame, label: str, model_value: str | None = None) -> pd.DataFrame:
    normalized = canonicalize_main_v2_reference_df(
        df,
        required_cols=["frequency_setting", "horizon", "test_mse", "test_mae", "test_corr", "mean_rank_ic"],
        label=label,
    )
    if model_value is not None:
        if "model" not in normalized.columns:
            raise ValueError(f"MAIN V2 {label} artifact must include model column for model filtering.")
        normalized = normalized[normalized["model"] == model_value].copy()
    expected = {(freq, horizon) for freq in ["daily_only", "weekly_only", "daily_weekly"] for horizon in ["5", "10", "20"]}
    observed = {(str(row["frequency_setting"]), str(row["horizon"])) for _, row in normalized.iterrows()}
    if len(normalized) != 9:
        raise ValueError(f"Expected exactly 9 {label} rows after canonicalization, found {len(normalized)}")
    if normalized.duplicated(subset=["frequency_setting", "horizon"]).any():
        dupes = normalized[normalized.duplicated(subset=["frequency_setting", "horizon"], keep=False)]
        raise ValueError(f"{label} canonicalized data contains duplicate frequency_setting/horizon rows: {dupes.to_dict(orient='records')}")
    if observed != expected:
        raise ValueError(f"{label} configs do not match expected 9 frequency x horizon settings: {sorted(observed)}")
    return normalized


def validate_training_provenance_row(row: pd.Series) -> None:
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
        "training_script": "scripts/python/panel_baseline_swim.py",
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
    }
    for key, expected in required.items():
        if key not in row.index:
            raise ValueError(f"Training provenance missing required field: {key}")
        actual = row[key]
        if pd.isna(actual):
            raise ValueError(f"Training provenance required field is null: {key}")
        if key in {"d_model", "nhead", "num_swim_blocks", "dim_feedforward", "batch_size", "max_epochs", "patience", "formal_tickers", "common_anchor_dates", "total_samples", "train_samples", "validation_samples", "test_samples", "train_dates", "validation_dates", "test_dates", "nominal_seed"}:
            if int(actual) != int(expected):
                raise ValueError(f"Training provenance field {key} mismatch: expected {expected}, found {actual}")
        elif key in {"learning_rate", "dropout", "max_norm"}:
            if float(actual) != float(expected):
                raise ValueError(f"Training provenance field {key} mismatch: expected {expected}, found {actual}")
        elif str(actual).strip() != str(expected):
            raise ValueError(f"Training provenance field {key} mismatch: expected {expected}, found {actual}")
    if "training_script_sha256" not in row.index or pd.isna(row["training_script_sha256"]):
        raise ValueError("Training provenance missing training_script_sha256.")


def write_training_provenance(output_dir: Path, script_path: Path) -> Path:
    provenance_path = output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"]
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
                "Refusing to overwrite formal provenance."
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
        "training_script": str(script_path.relative_to(ROOT)),
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
    }
    pd.DataFrame([row]).to_csv(provenance_path, index=False)
    return provenance_path


def _trace_rng_fingerprint_summary(prefix: str = "RNG") -> str:
    state = torch.get_rng_state()
    arr = state.detach().cpu().numpy()
    if arr.size == 0:
        return f"{prefix}=empty"
    abs_arr = np.abs(arr)
    s = float(arr.sum())
    mean = float(arr.mean())
    std = float(arr.std())
    max_abs = float(abs_arr.max())
    fingerprint = int(np.frombuffer(arr.tobytes(), dtype=np.uint64).sum() % (2**63 - 1))
    return f"{prefix}: sum={s:.6f} mean={mean:.6f} std={std:.6f} max_abs={max_abs:.6f} fingerprint={fingerprint}"


def _trace_tensor_summary(name: str, x: torch.Tensor) -> None:
    arr = x.detach().cpu().numpy()
    print(
        f"[TRACE] {name}: shape={tuple(arr.shape)} "
        f"sum={float(arr.sum()):.8f} mean={float(arr.mean()):.8f} "
        f"std={float(arr.std()):.8f} max_abs={float(np.max(np.abs(arr))):.8f}"
    )


def _trace_first_batch_fingerprint(batch_idx: int, x: torch.Tensor, y: torch.Tensor) -> None:
    x_arr = x.detach().cpu().numpy()
    y_arr = y.detach().cpu().numpy()
    print(f"[TRACE] batch={batch_idx} x: shape={tuple(x_arr.shape)} sum={float(x_arr.sum()):.8f} mean={float(x_arr.mean()):.8f} std={float(x_arr.std()):.8f} max_abs={float(np.max(np.abs(x_arr))):.8f}")
    first_5 = y_arr[:5]
    print(
        f"[TRACE] batch={batch_idx} y[:5]={first_5.tolist()} "
        f"sum={float(y_arr.sum()):.8f} mean={float(y_arr.mean()):.8f} std={float(y_arr.std()):.8f}"
    )


def _model_parameter_fingerprint(model: nn.Module) -> tuple[int, float, float]:
    param_total = sum(p.numel() for p in model.parameters())
    param_sum = float(sum(p.detach().float().sum().item() for p in model.parameters()))
    param_sq_sum = float(sum((p.detach().float() ** 2).sum().item() for p in model.parameters()))
    return param_total, param_sum, param_sq_sum


# =====================================================================
# WINDOW PARTITION / REVERSE (1D, temporal axis)
# =====================================================================


def window_partition_1d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B, T, C] -> [B * n_win, window_size, C]. Requires T % window_size == 0."""
    B, T, C = x.shape
    assert T % window_size == 0, f"T={T} is not divisible by window_size={window_size}"
    n_win = T // window_size
    return x.reshape(B, n_win, window_size, C).reshape(B * n_win, window_size, C)


def window_reverse_1d(windows: torch.Tensor, window_size: int, B: int, T: int) -> torch.Tensor:
    """[B * n_win, window_size, C] -> [B, T, C]. Exact inverse of window_partition_1d."""
    n_win = T // window_size
    C = windows.shape[-1]
    return windows.reshape(B, n_win, window_size, C).reshape(B, T, C)


def compute_pad_len(T: int, window_size: int) -> int:
    return (window_size - (T % window_size)) % window_size


# =====================================================================
# SHIFT-BOUNDARY MASK + PADDING MASK
# =====================================================================


def compute_shift_region_ids(T_pad: int, window_size: int, shift_size: int, device) -> torch.Tensor:
    """Swin-style region labeling (1D adaptation).

    Assigns a region id to every position in the *post-shift* temporal axis
    such that two positions share a region iff they were contiguous in the
    original (pre-roll) sequence. Used to build the shifted-window attention
    mask that blocks artificial wrap-around interactions created by
    `torch.roll`. This mirrors the standard Swin Transformer `img_mask`
    construction, generalized from 2D images to a 1D temporal axis.
    """
    region_id = torch.zeros(T_pad, dtype=torch.long, device=device)
    cnt = 0
    for sl in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
        region_id[sl] = cnt
        cnt += 1
    return region_id


def build_window_masks(valid_mask: torch.Tensor, window_size: int, shift_size: int) -> torch.Tensor:
    """Build an exact additive attention mask for 1D window attention.

    valid_mask: [T_pad] bool, True = real token, False = zero-padded token.
    Returns attn_mask: [n_win, window_size, window_size] float.

    REAL QUERY semantics:
      - padded keys are strictly masked as -inf;
      - keys from different shifted-window regions are strictly masked as -inf;
      - valid same-region real-key interactions remain allowed with additive mask 0.

    PADDED QUERY semantics:
      - the query output is semantically irrelevant because it is hard-zeroed
        immediately afterward;
      - to keep the row numerically valid for softmax, we allow exactly one dummy
        self key (the first key slot in the window) and mask every other key.

    This preserves the exact hard-mask semantics required by the frozen SWiM v1
    design: artificial cyclic wrap-around interactions and padded keys are
    strictly masked for valid queries, while padded-query rows remain defined
    and are later discarded by `_hard_zero`.
    """
    device = valid_mask.device
    T_pad = valid_mask.shape[0]
    n_win = T_pad // window_size

    attn_mask = torch.full((n_win, window_size, window_size), -torch.inf, dtype=torch.float32, device=device)
    if shift_size > 0:
        valid_shifted = torch.roll(valid_mask, shifts=-shift_size, dims=0)
    else:
        valid_shifted = valid_mask
    query_valid = valid_shifted.view(n_win, window_size)
    key_valid = valid_shifted.view(n_win, window_size)

    if shift_size > 0:
        region_id = compute_shift_region_ids(T_pad, window_size, shift_size, device)
        region_windows = region_id.view(n_win, window_size)
        region_block = region_windows.unsqueeze(2) != region_windows.unsqueeze(1)
    else:
        region_block = torch.zeros((n_win, window_size, window_size), dtype=torch.bool, device=device)

    allowed_real = query_valid.unsqueeze(2) & key_valid.unsqueeze(1) & ~region_block
    attn_mask.masked_fill_(allowed_real, 0.0)

    padded_rows = ~query_valid
    if padded_rows.any():
        dummy_key = torch.zeros_like(query_valid, dtype=torch.bool)
        dummy_key[:, 0] = True
        padded_allow = padded_rows.unsqueeze(2) & dummy_key.unsqueeze(1)
        attn_mask.masked_fill_(padded_allow, 0.0)

    return attn_mask


# =====================================================================
# MODEL
# =====================================================================


class SWiMWindowBlock(nn.Module):
    """One local- or shifted-window self-attention block.

    Residual / normalization structure is Post-LN, matching the frozen panel
    Vanilla Transformer's `nn.TransformerEncoderLayer` convention (the
    historical FSLR SWiM used Pre-LN; that is intentionally not reproduced):

        attn_out = WindowAttention(x)
        x = LayerNorm(x + Dropout(attn_out))
        ffn_out = FFN(x)
        x = LayerNorm(x + Dropout(ffn_out))

    Padded positions are hard-zeroed after every sub-layer. This is required
    because padded-query rows are semantically irrelevant: they are never part
    of a real-token representation and must not leak into later blocks or the
    masked mean pool. For valid queries, attention is exact-hard-masked with
    `-inf` on padded keys and cross-region wrap-around keys; for padded-query
    rows, one harmless dummy/self key remains finite so the row stays
    numerically valid for softmax and then gets replaced by exact zero via
    `_hard_zero`.
    """

    def __init__(self, d_model: int, nhead: int, window_size: int, shift_size: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _hard_zero(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Replace invalid (padded) positions with exactly 0.0.

        Uses torch.where rather than multiplying by a 0/1 mask: a degenerate
        all -inf attention row (see class docstring) produces NaN, and
        `0 * NaN == NaN` under IEEE754 -- multiplying would NOT clear it.
        torch.where genuinely replaces the value regardless of NaN/Inf.
        """
        mask = valid_mask.view(1, -1, 1).expand_as(x)
        return torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T_pad, C]; valid_mask: [T_pad] bool (True = real token)."""
        B, T_pad, C = x.shape
        w = self.window_size
        n_win = T_pad // w

        if DEBUG_NAN_MODE:
            if not torch.isfinite(x).all():
                print(f"[DEBUG_NAN] BLOCK failure before attention: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size} x_nonfinite=True")
                raise RuntimeError("Non-finite input to SWiMWindowBlock.forward")

        attn_mask_win = build_window_masks(valid_mask, w, self.shift_size)  # [n_win, w, w] float, always defined
        attn_mask = attn_mask_win.unsqueeze(0).expand(B, n_win, w, w).reshape(B * n_win, w, w)
        attn_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)
        if DEBUG_NAN_MODE:
            if not torch.isfinite(attn_mask).all() and not torch.all((attn_mask == 0.0) | torch.isinf(attn_mask)):
                print(f"[DEBUG_NAN] BAD MASK: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size} attn_mask_has_nonfinite=True")
                raise RuntimeError("Non-finite values in attention mask")
            if torch.any(torch.isinf(attn_mask).all(dim=-1) & torch.isinf(attn_mask).all(dim=-2)):
                print(f"[DEBUG_NAN] ALL-INF MASK ROW: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
                raise RuntimeError("All-inf attention row encountered in debug mode")

        residual = x
        x_shift = torch.roll(x, shifts=-self.shift_size, dims=1) if self.shift_size > 0 else x
        if DEBUG_NAN_MODE and not torch.isfinite(x_shift).all():
            print(f"[DEBUG_NAN] x_shift non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite x_shift in SWiMWindowBlock.forward")

        xw = window_partition_1d(x_shift, w)
        if DEBUG_NAN_MODE and not torch.isfinite(xw).all():
            print(f"[DEBUG_NAN] xw non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite xw in SWiMWindowBlock.forward")

        attn_out, _ = self.attn(xw, xw, xw, attn_mask=attn_mask, need_weights=False)
        if DEBUG_NAN_MODE and not torch.isfinite(attn_out).all():
            print(f"[DEBUG_NAN] attn_out non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite attention output in SWiMWindowBlock.forward")

        attn_out = window_reverse_1d(attn_out, w, B, T_pad)
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=self.shift_size, dims=1)
        attn_out = self._hard_zero(attn_out, valid_mask)  # clear the harmless-but-discarded padded-query rows
        if DEBUG_NAN_MODE and not torch.isfinite(attn_out).all():
            print(f"[DEBUG_NAN] attn_out after reverse non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite attn_out after reverse")

        x = self.norm1(residual + self.dropout1(attn_out))
        if DEBUG_NAN_MODE and not torch.isfinite(x).all():
            print(f"[DEBUG_NAN] norm1 output non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite norm1 output")
        x = self._hard_zero(x, valid_mask)

        ffn_out = self._hard_zero(self.ffn(x), valid_mask)
        if DEBUG_NAN_MODE and not torch.isfinite(ffn_out).all():
            print(f"[DEBUG_NAN] FFN output non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite FFN output")
        x = self.norm2(x + self.dropout2(ffn_out))
        if DEBUG_NAN_MODE and not torch.isfinite(x).all():
            print(f"[DEBUG_NAN] norm2 output non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite norm2 output")
        x = self._hard_zero(x, valid_mask)
        return x


class SWiMBranch(nn.Module):
    """Per-frequency SWiM encoder: projection -> PE -> 2 window blocks -> masked mean pool."""

    def __init__(
        self,
        feature_dim: int,
        frequency_key: str,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if frequency_key not in WINDOW_CONFIG:
            raise ValueError(f"Unknown frequency_key: {frequency_key}")
        cfg = WINDOW_CONFIG[frequency_key]
        self.window_size = cfg["window_size"]
        self.shift_size = cfg["shift_size"]
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=FIXED_MAX_LEN, dropout=dropout)
        assert NUM_BLOCKS == 2, "SWiM v1 requires exactly two blocks: local + shifted-window."
        self.blocks = nn.ModuleList(
            [
                SWiMWindowBlock(d_model, nhead, self.window_size, 0, dim_feedforward, dropout),
                SWiMWindowBlock(d_model, nhead, self.window_size, self.shift_size, dim_feedforward, dropout),
            ]
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        """x: [B, T, F] -> representation [B, d_model] (or the padded encoded
        sequence [B, T_pad, d_model] plus the validity mask if
        return_sequence=True, used only by audit tests)."""
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.positional_encoding(h)

        pad_len = compute_pad_len(T, self.window_size)
        T_pad = T + pad_len
        valid_mask = torch.ones(T_pad, dtype=torch.bool, device=x.device)
        if pad_len > 0:
            zeros = torch.zeros(B, pad_len, h.shape[-1], device=h.device, dtype=h.dtype)
            h = torch.cat([h, zeros], dim=1)
            valid_mask[T:] = False

        for blk in self.blocks:
            h = blk(h, valid_mask)

        if return_sequence:
            return h, valid_mask

        rep = h.sum(dim=1) / float(T)  # padded positions are exactly 0; dividing by true length T
        return rep


class SingleFrequencySWiMRegressor(nn.Module):
    def __init__(self, feature_dim: int, frequency_key: str):
        super().__init__()
        self.branch = SWiMBranch(feature_dim=feature_dim, frequency_key=frequency_key)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.branch(x)
        return self.head(rep).squeeze(-1)


class DualFrequencySWiMRegressor(nn.Module):
    def __init__(self, daily_feature_dim: int, weekly_feature_dim: int):
        super().__init__()
        self.daily_branch = SWiMBranch(feature_dim=daily_feature_dim, frequency_key="daily")
        self.weekly_branch = SWiMBranch(feature_dim=weekly_feature_dim, frequency_key="weekly")
        self.fusion = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        daily_vec = self.daily_branch(x_daily)
        weekly_vec = self.weekly_branch(x_weekly)
        fused = torch.cat([daily_vec, weekly_vec], dim=1)
        return self.fusion(fused).squeeze(-1)


def make_model(frequency: str, daily_feature_dim: int, weekly_feature_dim: int) -> nn.Module:
    if frequency == "daily_only":
        return SingleFrequencySWiMRegressor(feature_dim=daily_feature_dim, frequency_key="daily")
    if frequency == "weekly_only":
        return SingleFrequencySWiMRegressor(feature_dim=weekly_feature_dim, frequency_key="weekly")
    if frequency == "daily_weekly":
        return DualFrequencySWiMRegressor(daily_feature_dim=daily_feature_dim, weekly_feature_dim=weekly_feature_dim)
    raise ValueError(f"Unknown frequency setting: {frequency}")


# =====================================================================
# TRAIN / EVAL (mirrors panel_baseline_vanilla_transformer.py structure)
# =====================================================================


def smoke_test_models(ds, stats: dict, device: torch.device) -> dict:
    results = {}
    for frequency in FREQUENCY_SETTINGS:
        if frequency == "daily_only":
            X_all = apply_normalizer(ds, stats, "daily_only")
            X = torch.tensor(X_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=X.shape[-1], weekly_feature_dim=X.shape[-1]).to(device)
        elif frequency == "weekly_only":
            X_all = apply_normalizer(ds, stats, "weekly_only")
            X = torch.tensor(X_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=X.shape[-1], weekly_feature_dim=X.shape[-1]).to(device)
        else:
            Xd_all, Xw_all = apply_normalizer(ds, stats, "daily_weekly")
            xd = torch.tensor(Xd_all[:8], dtype=torch.float32, device=device)
            xw = torch.tensor(Xw_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=xd.shape[-1], weekly_feature_dim=xw.shape[-1]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if frequency == "daily_weekly":
            pred = model(xd, xw)
        else:
            pred = model(X)
        assert pred.shape == y.shape, f"{frequency} output shape mismatch: {pred.shape} vs {y.shape}"
        loss = nn.HuberLoss(delta=1.0)(pred, y)
        loss.backward()
        all_params = [p for p in model.parameters() if p.requires_grad]
        grads_present = all(p.grad is not None and torch.isfinite(p.grad).all() for p in all_params)
        assert grads_present, f"{frequency} backprop produced missing/non-finite gradients on some parameters"
        optimizer.zero_grad()
        results[frequency] = {"ok": True, "param_count": count_trainable_parameters(model)}
    return results


def train_and_eval_frequency(ds, frequency: str, horizon: int, stats: dict, trace_nan: bool = False) -> dict:
    set_seed(SEED)
    if trace_nan:
        print(f"[TRACE] torch.initial_seed()={torch.initial_seed()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if trace_nan:
        print(f"[TRACE] selected device={device.type.upper()}")
        print("[TRACE] DEBUG_NAN_MODE = False")
        assert DEBUG_NAN_MODE is False
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
        model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="daily").to(device)
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
        model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="weekly").to(device)
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
        model = DualFrequencySWiMRegressor(
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

    if trace_nan:
        param_total, param_sum, param_sq_sum = _model_parameter_fingerprint(model)
        print(f"[TRACE] model_total_parameters={param_total}")
        print(f"[TRACE] model_param_sum={param_sum:.12f}")
        print(f"[TRACE] model_param_sq_sum={param_sq_sum:.12f}")
        print(f"[TRACE] {_trace_rng_fingerprint_summary('torch_rng_state_before_train_loader')}")

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    history_rows = []
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        if trace_nan and epoch > 1:
            break
        model.train()
        train_loss_sum = 0.0
        train_weight = 0
        epoch_start = time.perf_counter()

        if frequency == "daily_weekly":
            for batch_idx, (x_d_batch, x_w_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                if trace_nan:
                    if not torch.isfinite(x_d_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if not torch.isfinite(x_w_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if not torch.isfinite(y_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if batch_idx < 3:
                        _trace_first_batch_fingerprint(batch_idx, x_d_batch, y_batch)

                logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                if trace_nan and not torch.isfinite(logits).all():
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=forward")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "forward", "frequency": frequency, "horizon": horizon}
                loss = criterion(logits, target)
                if trace_nan and not torch.isfinite(loss):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=loss")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "loss", "frequency": frequency, "horizon": horizon}
                batch_size_value = y_batch.shape[0]
                train_loss_sum += float(loss.item()) * batch_size_value
                train_weight += int(batch_size_value)
                loss.backward()
                if trace_nan:
                    bad_grad_names = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                bad_grad_names.append((name, "missing"))
                            elif not torch.isfinite(param.grad).all():
                                bad_grad_names.append((name, "nonfinite"))
                    if bad_grad_names:
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=backward")
                        for name, reason in bad_grad_names:
                            print(f"[TRACE] gradient failure: {name} -> {reason}")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "backward", "frequency": frequency, "horizon": horizon}
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if trace_nan and not torch.isfinite(grad_norm):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=gradient clipping")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "gradient clipping", "frequency": frequency, "horizon": horizon}
                optimizer.step()
                if trace_nan:
                    for name, param in model.named_parameters():
                        if param.requires_grad and not torch.isfinite(param).all():
                            print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=optimizer step")
                            print(f"[TRACE] parameter failure: {name}")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "optimizer step", "frequency": frequency, "horizon": horizon}
        else:
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                if trace_nan:
                    if not torch.isfinite(x_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if not torch.isfinite(y_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if batch_idx < 3:
                        _trace_first_batch_fingerprint(batch_idx, x_batch, y_batch)

                logits, target = batched_forward(x_batch, y_batch)
                if trace_nan and not torch.isfinite(logits).all():
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=forward")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "forward", "frequency": frequency, "horizon": horizon}
                loss = criterion(logits, target)
                if trace_nan and not torch.isfinite(loss):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=loss")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "loss", "frequency": frequency, "horizon": horizon}
                batch_size_value = y_batch.shape[0]
                train_loss_sum += float(loss.item()) * batch_size_value
                train_weight += int(batch_size_value)
                loss.backward()
                if trace_nan:
                    bad_grad_names = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                bad_grad_names.append((name, "missing"))
                            elif not torch.isfinite(param.grad).all():
                                bad_grad_names.append((name, "nonfinite"))
                    if bad_grad_names:
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=backward")
                        for name, reason in bad_grad_names:
                            print(f"[TRACE] gradient failure: {name} -> {reason}")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "backward", "frequency": frequency, "horizon": horizon}
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if trace_nan and not torch.isfinite(grad_norm):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=gradient clipping")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "gradient clipping", "frequency": frequency, "horizon": horizon}
                optimizer.step()
                if trace_nan:
                    for name, param in model.named_parameters():
                        if param.requires_grad and not torch.isfinite(param).all():
                            print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=optimizer step")
                            print(f"[TRACE] parameter failure: {name}")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "optimizer step", "frequency": frequency, "horizon": horizon}

        model.eval()
        val_loss_sum = 0.0
        val_weight = 0
        with torch.no_grad():
            if frequency == "daily_weekly":
                for batch_idx, (x_d_batch, x_w_batch, y_batch) in enumerate(val_loader):
                    if trace_nan:
                        if not torch.isfinite(x_d_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                        if not torch.isfinite(x_w_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                        if not torch.isfinite(y_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                    if trace_nan and not torch.isfinite(logits).all():
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    batch_loss = criterion(logits, target)
                    if trace_nan and not torch.isfinite(batch_loss):
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    batch_size_value = y_batch.shape[0]
                    val_loss_sum += float(batch_loss.item()) * batch_size_value
                    val_weight += int(batch_size_value)
            else:
                for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                    if trace_nan:
                        if not torch.isfinite(x_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                        if not torch.isfinite(y_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    logits, target = batched_forward(x_batch, y_batch)
                    if trace_nan and not torch.isfinite(logits).all():
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    batch_loss = criterion(logits, target)
                    if trace_nan and not torch.isfinite(batch_loss):
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
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

        if trace_nan:
            print(f"[TRACE] trace_train_loss={train_loss:.12f}")
            print(f"[TRACE] trace_val_loss={val_loss:.12f}")
            print("[TRACE] FORMAL TRACE EPOCH + VALIDATION BOTH FINITE")
            return {
                "trace_status": "ok",
                "frequency": frequency,
                "horizon": horizon,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

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
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | best_epoch={best_epoch} | "
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
        y_true=y_test, y_pred=y_pred, ticker=ds.ticker[test_mask], anchor_date=ds.anchor_date[test_mask]
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


def save_prediction_rows(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir) if output_dir is not None else MAIN_V2_SWIM_DIR
    rows = []
    for rr in results:
        for i in range(len(rr["ticker"])):
            rows.append(
                {
                    "ticker": rr["ticker"][i],
                    "anchor_date": rr["anchor_date"][i],
                    "horizon": rr["horizon"],
                    "frequency_setting": rr["frequency_setting"],
                    "y_true": rr["y_true"][i],
                    "y_pred": rr["y_pred"][i],
                    "seed": rr["seed"],
                    "best_epoch": rr["best_epoch"],
                    "model": rr["model"],
                }
            )
    df = pd.DataFrame(rows)
    path = output_dir / FORMAL_SWIM_RUN_FILES["predictions"]
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_summary_rows(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir) if output_dir is not None else MAIN_V2_SWIM_DIR
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
    path = output_dir / FORMAL_SWIM_RUN_FILES["summary"]
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_training_history(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir) if output_dir is not None else MAIN_V2_SWIM_DIR
    rows = []
    for rr in results:
        for epoch_row in rr["history"]:
            rows.append(
                {
                    "frequency_setting": epoch_row["frequency_setting"],
                    "horizon": epoch_row["horizon"],
                    "epoch": epoch_row["epoch"],
                    "train_loss": epoch_row["train_loss"],
                    "val_loss": epoch_row["val_loss"],
                    "elapsed_seconds": epoch_row.get("elapsed_seconds", np.nan),
                }
            )
    df = pd.DataFrame(rows)
    path = output_dir / FORMAL_SWIM_RUN_FILES["history"]
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_rank_ic_by_date(results: list[dict], output_dir: Path | None = None, append_mode: bool = False) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir) if output_dir is not None else MAIN_V2_SWIM_DIR
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


def append_experiment_outputs(res: dict, output_dir: Path | None = None) -> None:
    output_dir = resolve_output_dir(output_dir) if output_dir is not None else MAIN_V2_SWIM_DIR
    save_summary_rows([res], output_dir=output_dir, append_mode=True)
    save_prediction_rows([res], output_dir=output_dir, append_mode=True)
    save_training_history([res], output_dir=output_dir, append_mode=True)
    save_rank_ic_by_date([res], output_dir=output_dir, append_mode=True)


def build_baseline_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Comparison table against canonical MAIN V2 baselines in the formal model-specific namespaces."""
    naive_path = MAIN_V2_BASELINES["naive"]
    ridge_path = MAIN_V2_BASELINES["ridge"]
    lstm_path = MAIN_V2_BASELINES["lstm"]
    vanilla_path = MAIN_V2_BASELINES["vanilla"]

    required = {
        "naive": naive_path,
        "ridge": ridge_path,
        "lstm": lstm_path,
        "vanilla": vanilla_path,
    }
    missing = [f"{name}:{path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical MAIN V2 baseline artifacts required for formal finalization: " + ", ".join(missing)
        )

    naive_df = canonicalize_main_v2_naive(pd.read_csv(naive_path))
    ridge_df = canonicalize_main_v2_ridge(pd.read_csv(ridge_path))
    lstm_df = canonicalize_main_v2_lstm_or_vanilla(pd.read_csv(lstm_path), label="lstm", model_value="lstm")
    vanilla_df = canonicalize_main_v2_lstm_or_vanilla(pd.read_csv(vanilla_path), label="vanilla", model_value=None)

    rows = []
    for _, row in summary_df.iterrows():
        freq = row["frequency_setting"]
        horizon = row["horizon"]
        horizon_norm = str(horizon).strip().replace("d", "")
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
            "vanilla_available": True,
        }

        matched = naive_df[naive_df["horizon"] == horizon_norm].copy()
        if matched.empty:
            raise ValueError(f"Naive baseline is missing horizon {horizon}.")
        matched["model"] = matched["model"].astype(str).str.lower()
        zero_row = matched[matched["model"] == "zero"]
        train_mean_row = matched[matched["model"] == "train_mean"]
        if zero_row.empty or train_mean_row.empty:
            raise ValueError(f"Naive baseline missing zero/train_mean rows for horizon {horizon}.")
        out["zero_mse"] = float(zero_row["pooled_mse"].iloc[0])
        out["train_mean_mse"] = float(train_mean_row["pooled_mse"].iloc[0])
        naive_values = [out["zero_mse"], out["train_mean_mse"]]
        out["best_naive_mse"] = float(np.min(naive_values))

        matched = ridge_df[(ridge_df["frequency_setting"] == freq) & (ridge_df["horizon"] == horizon_norm)]
        if matched.empty:
            raise ValueError(f"Ridge baseline missing row for {freq}/{horizon}.")
        out["ridge_mse"] = float(matched["pooled_mse"].iloc[0])
        out["ridge_mae"] = float(matched["pooled_mae"].iloc[0])
        out["ridge_corr"] = float(matched["pooled_corr"].iloc[0])
        out["ridge_mean_rank_ic"] = float(matched["mean_rank_ic"].iloc[0])

        matched = lstm_df[(lstm_df["frequency_setting"] == freq) & (lstm_df["horizon"] == horizon_norm)]
        if matched.empty:
            raise ValueError(f"LSTM baseline missing row for {freq}/{horizon}.")
        out["lstm_mse"] = float(matched["test_mse"].iloc[0])
        out["lstm_mae"] = float(matched["test_mae"].iloc[0])
        out["lstm_corr"] = float(matched["test_corr"].iloc[0])
        out["lstm_mean_rank_ic"] = float(matched["mean_rank_ic"].iloc[0])

        matched = vanilla_df[(vanilla_df["frequency_setting"] == freq) & (vanilla_df["horizon"] == horizon_norm)]
        if matched.empty:
            raise ValueError(f"Vanilla baseline missing row for {freq}/{horizon}.")
        out["vanilla_mse"] = float(matched["test_mse"].iloc[0])
        out["vanilla_mae"] = float(matched["test_mae"].iloc[0])
        out["vanilla_corr"] = float(matched["test_corr"].iloc[0])
        out["vanilla_mean_rank_ic"] = float(matched["mean_rank_ic"].iloc[0])

        rows.append(out)
    return pd.DataFrame(rows)


def write_main_v2_swim_run_manifest(
    output_dir: Path,
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    history_df: pd.DataFrame,
    finalization_time_script_sha256: str,
) -> Path:
    manifest_path = output_dir / FORMAL_SWIM_RUN_FILES["manifest"]
    provenance_path = output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"]
    if not provenance_path.exists():
        raise FileNotFoundError(f"Training provenance file is required before writing the formal SWiM manifest: {provenance_path}")
    provenance_df = pd.read_csv(provenance_path)
    if provenance_df.empty:
        raise ValueError(f"Training provenance file is empty: {provenance_path}")
    row = provenance_df.iloc[0]
    validate_training_provenance_row(row)
    validate_main_v2_swim_evidence(output_dir)
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
        "completed_cells": 9,
        "summary_rows": int(len(summary_df)),
        "prediction_rows": int(len(pred_df)),
        "rank_ic_rows": int(len(rank_ic_df)),
        "training_history_rows": int(len(history_df)),
        "training_script": str(row.get("training_script", "scripts/python/panel_baseline_swim.py")),
        "training_script_sha256": str(row.get("training_script_sha256", "MISSING")),
        "finalization_time_script_sha256": finalization_time_script_sha256,
        "output_dir": str(output_dir.resolve()),
        "summary_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["summary"]) if (output_dir / FORMAL_SWIM_RUN_FILES["summary"]).exists() else "MISSING",
        "predictions_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["predictions"]) if (output_dir / FORMAL_SWIM_RUN_FILES["predictions"]).exists() else "MISSING",
        "rank_ic_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]) if (output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]).exists() else "MISSING",
        "training_history_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["history"]) if (output_dir / FORMAL_SWIM_RUN_FILES["history"]).exists() else "MISSING",
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
    }
    pd.DataFrame([manifest]).to_csv(manifest_path, index=False)
    return manifest_path


def write_final_summary(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    training_script_sha256: str,
    finalization_time_script_sha256: str,
    output_dir: Path | None = None,
) -> None:
    output_dir = resolve_output_dir(output_dir) if output_dir is not None else MAIN_V2_SWIM_DIR
    lines = []
    lines.append("MAIN V2")
    lines.append("POST-TEMPORAL-FIX")
    lines.append("Dataset Contract V2")
    lines.append("SWiM-style Transformer")
    lines.append("nominal seed = 42")
    lines.append("completed cells = 9 / 9")
    lines.append("")
    lines.append("Hashes:")
    lines.append(f"  dataset manifest SHA = {DATASET_MANIFEST_SHA256}")
    lines.append(f"  training-time script SHA = {training_script_sha256}")
    lines.append(f"  finalization-time script SHA = {finalization_time_script_sha256}")
    lines.append("  summary artifact SHA = " + compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["summary"]))
    lines.append("  prediction artifact SHA = " + compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["predictions"]))
    lines.append("  Rank-IC artifact SHA = " + compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]))
    lines.append("  training-history artifact SHA = " + compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["history"]))
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
    lines.append("  - no pre-temporal-fix evidence used")
    lines.append("")
    lines.append("Parameter counts:")
    for key, val in EXPECTED_SWIM_PARAM_COUNTS.items():
        lines.append(f"  - {key} = {val}")
    lines.append("")
    lines.append("Best MSE by horizon:")
    for horizon in ["5d", "10d", "20d"]:
        dfx = summary_df[summary_df["horizon"] == horizon]
        if not dfx.empty:
            best = dfx.loc[dfx["test_mse"].idxmin()]
            freq = str(best["frequency_setting"])
            cmp_row = comparison_df[(comparison_df["horizon"] == horizon) & (comparison_df["frequency_setting"] == freq)]
            if cmp_row.empty:
                raise ValueError(f"Comparison table missing same-frequency baseline row for best-MSE {freq}/{horizon}.")
            c = cmp_row.iloc[0]
            lines.append(f"  {horizon}: SWiM best-MSE frequency = {freq}")
            lines.append(f"    SWiM MSE/MAE/Corr/MeanRankIC = {best['test_mse']:.6f} / {best['test_mae']:.6f} / {best['test_corr']} / {best['mean_rank_ic']}")
            lines.append(f"    Ridge MSE/MAE/Corr/MeanRankIC = {c['ridge_mse']} / {c['ridge_mae']} / {c['ridge_corr']} / {c['ridge_mean_rank_ic']}")
            lines.append(f"    LSTM MSE/MAE/Corr/MeanRankIC = {c['lstm_mse']} / {c['lstm_mae']} / {c['lstm_corr']} / {c['lstm_mean_rank_ic']}")
            lines.append(f"    Vanilla MSE/MAE/Corr/MeanRankIC = {c['vanilla_mse']} / {c['vanilla_mae']} / {c['vanilla_corr']} / {c['vanilla_mean_rank_ic']}")
            lines.append(f"    Zero MSE = {c['zero_mse']}")
            lines.append(f"    Train-Mean MSE = {c['train_mean_mse']}")
    lines.append("")
    lines.append("Best mean Rank IC by horizon:")
    for horizon in ["5d", "10d", "20d"]:
        dfx = summary_df[summary_df["horizon"] == horizon]
        if not dfx.empty:
            valid = dfx["mean_rank_ic"].notna()
            if valid.any():
                best = dfx.loc[dfx["mean_rank_ic"].idxmax()]
                freq = str(best["frequency_setting"])
                cmp_row = comparison_df[(comparison_df["horizon"] == horizon) & (comparison_df["frequency_setting"] == freq)]
                if cmp_row.empty:
                    raise ValueError(f"Comparison table missing same-frequency baseline row for best-RankIC {freq}/{horizon}.")
                c = cmp_row.iloc[0]
                lines.append(f"  {horizon}: SWiM best mean Rank IC frequency = {freq} (mean_rank_ic={best['mean_rank_ic']:.6f})")
                lines.append(f"    Ridge Mean Rank IC = {c['ridge_mean_rank_ic']}")
                lines.append(f"    LSTM Mean Rank IC = {c['lstm_mean_rank_ic']}")
                lines.append(f"    Vanilla Mean Rank IC = {c['vanilla_mean_rank_ic']}")
            else:
                lines.append(f"  {horizon}: best mean Rank IC = undefined")
    lines.append("")
    lines.append("Prediction dispersion diagnostics (all 9 cells):")
    disp = summary_df[["frequency_setting", "horizon", "pred_std", "true_std", "pred_std_true_std_ratio"]].copy()
    disp["horizon_num"] = disp["horizon"].astype(str).str.replace("d", "", regex=False).astype(int)
    disp = disp.sort_values(["frequency_setting", "horizon_num"]).drop(columns=["horizon_num"])
    for _, row in disp.iterrows():
        lines.append(
            f"  {row['frequency_setting']} / {row['horizon']}: pred_std={row['pred_std']} true_std={row['true_std']} ratio={row['pred_std_true_std_ratio']}"
        )
    lines.append("")
    lines.append("PRE-TEMPORAL-FIX results were not used for formal comparison or model selection.")
    lines.append("")
    lines.append("Seed-42 SWiM evidence alone does NOT establish:")
    lines.append("- statistical significance")
    lines.append("- multi-seed robustness")
    lines.append("- profitability")
    lines.append("- causality")
    lines.append("- Daily+Weekly complementarity")
    lines.append("- causal superiority of shifted-window/local attention")
    lines.append("- adaptive routing success")
    lines.append("")
    lines.append("Important architecture-comparison caveat:")
    lines.append("SWiM and Vanilla have matched trainable parameter counts under the frozen 5-feature / d_model=64 architecture:")
    lines.append("daily_only: 67393 vs 67393")
    lines.append("weekly_only: 67393 vs 67393")
    lines.append("daily_weekly: 142977 vs 142977")
    lines.append("Therefore there is no trainable-parameter-count difference in this specific comparison.")
    lines.append("However, the architectures still differ in more than attention connectivity because:")
    lines.append("Vanilla uses final-timestep representation, whereas SWiM uses masked mean pooling.")
    path = output_dir / FORMAL_SWIM_RUN_FILES["final_summary"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def full_benchmark(ds, stats: dict, output_dir: Path | None = None) -> None:
    output_dir = resolve_output_dir(output_dir)
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    total_experiments = len(FREQUENCY_SETTINGS) * len(HORIZONS)
    script_start = time.perf_counter()

    print("============================================================")
    print("SWIM-STYLE TRANSFORMER PANEL BASELINE RUN SETUP")
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
            exp_counter = (experiment_idx - 1) * len(HORIZONS) + list(HORIZONS).index(horizon) + 1
            print(f"\n[Experiment {exp_counter}/{total_experiments}]")
            print(f"frequency_setting = {frequency}")
            print(f"horizon = {horizon}d")
            res = train_and_eval_frequency(ds, frequency, int(horizon), stats)
            results.append(res)
            experiment_runtimes.append(float(res["runtime_seconds"]))
            append_experiment_outputs(res, output_dir=output_dir)

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

    summary_df = save_summary_rows(results, output_dir=output_dir)
    save_prediction_rows(results, output_dir=output_dir)
    save_training_history(results, output_dir=output_dir)
    save_rank_ic_by_date(results, output_dir=output_dir)
    print("Formal SWiM raw evidence training completed.")

    print("\n===========================================================")
    print("SWIM-STYLE TRANSFORMER PANEL CONTROL COMPLETE")
    print(f"Total runtime: {format_elapsed(time.perf_counter() - script_start)}")
    print(f"Completed experiments: {len(results)}/{total_experiments}")
    print("===========================================================")


# =====================================================================
# SMOKE / AUDIT TESTS
# =====================================================================


def run_check(name: str, fn) -> None:
    try:
        fn()
    except AssertionError as exc:
        print(f"FAIL: {name}: {exc}")
        raise
    print(f"PASS: {name}")


def test_positional_encoding_shape() -> None:
    pe = PositionalEncoding(d_model=D_MODEL, max_len=FIXED_MAX_LEN, dropout=0.0)
    x = torch.zeros(2, 10, D_MODEL)
    out = pe(x)
    assert out.shape == x.shape, f"unexpected PE output shape: {out.shape}"


def test_window_partition_reverse() -> None:
    torch.manual_seed(0)
    for T, w in [(90, 10), (28, 4)]:  # divisible daily case; divisible post-pad weekly case
        x = torch.randn(3, T, 6)
        windows = window_partition_1d(x, w)
        recon = window_reverse_1d(windows, w, B=3, T=T)
        max_err = (recon - x).abs().max().item()
        assert max_err < 1e-6, f"reconstruction error too large for T={T}, w={w}: {max_err}"
        print(f"  window_partition/reverse T={T} w={w}: max_abs_reconstruction_error={max_err:.2e}")

    # non-divisible case: manually zero-pad, partition/reverse, then strip padding
    T, w = 26, 4
    pad_len = compute_pad_len(T, w)
    x = torch.randn(3, T, 6)
    x_padded = torch.cat([x, torch.zeros(3, pad_len, 6)], dim=1)
    windows = window_partition_1d(x_padded, w)
    recon_padded = window_reverse_1d(windows, w, B=3, T=T + pad_len)
    recon = recon_padded[:, :T, :]
    max_err = (recon - x).abs().max().item()
    assert max_err < 1e-6, f"reconstruction error too large for padded T={T}, w={w}: {max_err}"
    print(f"  window_partition/reverse (non-divisible) T={T} w={w} pad_len={pad_len}: max_abs_reconstruction_error={max_err:.2e}")


def test_shift_mask() -> None:
    """Toy example proving the shifted-window boundary mask matches the original-token ordering."""
    T_pad, w, shift = 20, 10, 5
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    attn_mask = build_window_masks(valid_mask, w, shift)

    original = torch.arange(T_pad)
    shifted = torch.roll(original, shifts=-shift)
    expected_shifted = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4])
    assert torch.equal(shifted, expected_shifted), f"unexpected shifted index order: {shifted.tolist()}"

    win0 = shifted[:w]
    win1 = shifted[w:]
    assert torch.equal(win0, torch.arange(5, 15)), f"unexpected shifted window0: {win0.tolist()}"
    assert torch.equal(win1, torch.tensor([15, 16, 17, 18, 19, 0, 1, 2, 3, 4])), f"unexpected shifted window1: {win1.tolist()}"

    n_win = T_pad // w
    assert attn_mask.shape == (n_win, w, w), f"attn_mask shape mismatch: {attn_mask.shape}"
    region_id = compute_shift_region_ids(T_pad, w, shift, device=torch.device("cpu"))
    expected = torch.tensor([0] * 10 + [1] * 5 + [2] * 5)
    assert torch.equal(region_id, expected), f"unexpected region ids: {region_id.tolist()}"

    win0_mask = attn_mask[0]
    win1_mask = attn_mask[1]
    assert torch.all(win0_mask == 0.0), "window0 should have no shift-boundary blocking"
    assert torch.all(win1_mask[:5, :5] == 0.0), "same-region (first half of window1) pairs must be allowed"
    assert torch.all(win1_mask[5:, 5:] == 0.0), "same-region (second half of window1) pairs must be allowed"
    assert torch.all(torch.isinf(win1_mask[:5, 5:])), "wrap-around region1<->region2 pairs must be masked with -inf"
    assert torch.all(torch.isinf(win1_mask[5:, :5])), "wrap-around region2<->region1 pairs must be masked with -inf"
    assert torch.all(torch.isfinite(win1_mask[win1_mask == 0.0])), "allowed keys in valid queries must remain finite"

    # Explicitly verify original-index mapping through cyclic shift + window partition.
    allowed_examples = [(15, 19), (0, 4), (19, 15), (4, 0)]
    blocked_examples = [(15, 0), (19, 4), (0, 15), (4, 19)]
    for a, b in allowed_examples:
        i, j = shifted.tolist().index(a), shifted.tolist().index(b)
        assert win1_mask[i % w, j % w] == 0.0, f"expected {a} <-> {b} to be allowed"
    for a, b in blocked_examples:
        i, j = shifted.tolist().index(a), shifted.tolist().index(b)
        assert torch.isinf(win1_mask[i % w, j % w]), f"expected {a} <-> {b} to be blocked"

    print("  original indices:", original.tolist())
    print("  shifted indices:", shifted.tolist())
    print("  window0:", win0.tolist())
    print("  window1:", win1.tolist())
    print("  representative allowed pairs:", [(15, 19), (0, 4)])
    print("  representative blocked wrap-around pairs:", [(15, 0), (19, 4)])


def test_hard_mask_exactness() -> None:
    """For real queries, all blocked attention entries are exactly -inf and no row is fully masked."""
    T_pad, w, shift = 20, 10, 5
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    attn_mask = build_window_masks(valid_mask, w, shift)
    win1 = attn_mask[1]

    assert torch.all(win1[:5, :5] == 0.0), "same-region valid pairs must remain exactly 0.0"
    assert torch.all(win1[5:, 5:] == 0.0), "same-region valid pairs must remain exactly 0.0"
    assert torch.all(torch.isinf(win1[:5, 5:])), "real-query cross-region pairs must be exactly -inf"
    assert torch.all(torch.isinf(win1[5:, :5])), "real-query cross-region pairs must be exactly -inf"
    for row in win1:
        assert torch.any(torch.isfinite(row)), "no real-query attention row may be entirely -inf"

    T_pad2, w2, shift2 = 28, 4, 2
    original_indices = torch.arange(T_pad2)
    shifted_indices = torch.roll(original_indices, shifts=-shift2)
    expected_final_window = torch.tensor([26, 27, 0, 1])
    assert torch.equal(shifted_indices[-w2:], expected_final_window), f"unexpected final weekly shifted window: {shifted_indices[-w2:].tolist()}"

    valid_mask2 = torch.ones(T_pad2, dtype=torch.bool)
    valid_mask2[24:] = False
    shifted_validity = torch.roll(valid_mask2, shifts=-shift2)
    expected_validity = torch.tensor([False, False, True, True], dtype=torch.bool)
    assert torch.equal(shifted_validity[-w2:], expected_validity), f"unexpected final weekly shifted validity: {shifted_validity[-w2:].tolist()}"

    attn_mask2 = build_window_masks(valid_mask2, w2, shift2)
    last_window = attn_mask2[6]
    assert torch.all(last_window[0, 0] == 0.0), "first padded query in final weekly window must retain dummy key"
    assert torch.all(torch.isinf(last_window[0, 1:])), "first padded query must not attend real keys"
    assert torch.all(last_window[1, 0] == 0.0), "second padded query in final weekly window must retain dummy key"
    assert torch.all(torch.isinf(last_window[1, 1:])), "second padded query must not attend real keys"
    assert torch.all(last_window[2, 2:] == 0.0), "real query at original token 0 must retain its valid same-region keys"
    assert torch.all(torch.isinf(last_window[2, :2])), "real query at original token 0 must not attend padded keys"
    assert torch.all(last_window[3, 2:] == 0.0), "real query at original token 1 must retain its valid same-region keys"
    assert torch.all(torch.isinf(last_window[3, :2])), "real query at original token 1 must not attend padded keys"
    assert torch.all(torch.any(torch.isfinite(last_window), dim=-1)), "every final-window query row must retain at least one finite entry"
    print("  exact hard-mask audit: blocked real-query entries are -inf; valid real pairs stay 0.0; no fully masked real row")
    print("  padded-key block check on final weekly window: real query rows keep only valid same-region keys available")


def test_weekly_shifted_padding_geometry() -> None:
    """Verify the exact Weekly shifted-window prompt geometry after zero-padding."""
    T, w, shift = 26, 4, 2
    pad_len = compute_pad_len(T, w)
    T_pad = T + pad_len
    assert T_pad == 28 and pad_len == 2, f"unexpected weekly padding geometry: pad_len={pad_len}, T_pad={T_pad}"

    original_indices = torch.arange(T_pad)
    shifted_indices = torch.roll(original_indices, shifts=-shift)
    last_window = shifted_indices[-w:]
    expected_window = torch.tensor([26, 27, 0, 1])
    assert torch.equal(last_window, expected_window), f"unexpected final shifted window: {last_window.tolist()}"

    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    valid_mask[T:] = False
    shifted_validity = torch.roll(valid_mask, shifts=-shift)
    expected_validity = torch.tensor([False, False, True, True], dtype=torch.bool)
    assert torch.equal(shifted_validity[-w:], expected_validity), f"unexpected final shifted validity: {shifted_validity[-w:].tolist()}"

    attn_mask = build_window_masks(valid_mask, w, shift)
    final_window_mask = attn_mask[6]
    assert torch.all(torch.isfinite(final_window_mask[0, :1])), "padded query row 0 must keep one finite dummy key"
    assert torch.all(torch.isinf(final_window_mask[0, 1:])), "padded query row 0 must not attend real keys"
    assert torch.all(torch.isfinite(final_window_mask[1, :1])), "padded query row 1 must keep one finite dummy key"
    assert torch.all(torch.isinf(final_window_mask[1, 1:])), "padded query row 1 must not attend real keys"
    assert torch.all(final_window_mask[2, 2:] == 0.0), "real query 0 must be allowed to attend real token 0 and 1"
    assert torch.all(torch.isinf(final_window_mask[2, :2])), "real query 0 must not attend padded keys"
    assert torch.all(final_window_mask[3, 2:] == 0.0), "real query 1 must be allowed to attend real token 0 and 1"
    assert torch.all(torch.isinf(final_window_mask[3, :2])), "real query 1 must not attend padded keys"
    print("  weekly shifted-padding geometry:")
    print(f"    original: 0..{T-1} then PAD{T},{T + 1}")
    print(f"    shifted last window: {last_window.tolist()}")
    print(f"    shifted validity: {shifted_validity[-w:].tolist()}")
    print("    final weekly last-window mask behavior: padded rows keep dummy key; real rows allow only same-region real keys")


def test_shifted_validity_alignment() -> None:
    """The row validity used by build_window_masks() must match actual shifted-window token validity."""
    T_pad, w, shift = 28, 4, 2
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    valid_mask[T_pad - 2:] = False
    shifted_validity = torch.roll(valid_mask, shifts=-shift)
    attn_mask = build_window_masks(valid_mask, w, shift)
    query_valid = shifted_validity.view(T_pad // w, w)

    for win_idx in range(T_pad // w):
        for q_idx in range(w):
            actual_query_valid = bool(shifted_validity[win_idx * w + q_idx])
            assert bool(query_valid[win_idx, q_idx]) == actual_query_valid, (
                f"shifted query validity mismatch at window={win_idx}, query={q_idx}: "
                f"expected {actual_query_valid}, got {bool(query_valid[win_idx, q_idx])}"
            )
            row = attn_mask[win_idx, q_idx]
            assert torch.any(torch.isfinite(row)), (
                f"mask row for shifted-window query ({win_idx}, {q_idx}) is entirely -inf"
            )
    print("  shifted query/key validity alignment: rows correspond exactly to the shifted-window tokens they represent")


def test_attention_mask_row_safety() -> None:
    """Every row passed to MultiheadAttention must keep at least one finite entry."""
    cases = [(90, 10, 5, torch.ones(90, dtype=torch.bool)), (28, 4, 2, torch.cat([torch.ones(26, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)]))]
    for T_pad, w, shift, valid_mask in cases:
        attn_mask = build_window_masks(valid_mask, w, shift)
        assert attn_mask.shape == (T_pad // w, w, w), f"unexpected attn mask shape: {attn_mask.shape}"
        for win_idx in range(attn_mask.shape[0]):
            for q_idx in range(w):
                row = attn_mask[win_idx, q_idx]
                assert torch.any(torch.isfinite(row)), (
                    f"all -inf row in {T_pad=} window={win_idx} query={q_idx}: row={row.tolist()}"
                )
    print("  row safety audit: every real/padded attention row remains finite for Daily and Weekly shifted masks")


def test_daily_shifted_window_geometry() -> None:
    """Daily T=90/window=10/shift=5 should produce the expected wrap-around boundary geometry."""
    T, w, shift = 90, 10, 5
    original = torch.arange(T)
    shifted = torch.roll(original, shifts=-shift)
    last_window = shifted[-w:]
    expected_last_window = torch.tensor([85, 86, 87, 88, 89, 0, 1, 2, 3, 4])
    assert torch.equal(last_window, expected_last_window), f"unexpected daily final shifted window: {last_window.tolist()}"
    allowed = [(85, 89), (0, 4), (89, 85), (4, 0)]
    blocked = [(85, 0), (89, 4), (0, 85), (4, 89)]
    mask = build_window_masks(torch.ones(T, dtype=torch.bool), w, shift)[8]
    for a, b in allowed:
        ia, ib = shifted.tolist().index(a), shifted.tolist().index(b)
        assert mask[ia % w, ib % w] == 0.0, f"expected {a} <-> {b} to be allowed"
    for a, b in blocked:
        ia, ib = shifted.tolist().index(a), shifted.tolist().index(b)
        assert torch.isinf(mask[ia % w, ib % w]), f"expected {a} <-> {b} to be blocked"
    print("  original daily indices:", original[:10].tolist(), "...", original[-10:].tolist())
    print("  shifted daily final window:", last_window.tolist())
    print("  representative allowed pairs:", [(85, 89), (0, 4)])
    print("  representative blocked pairs:", [(85, 0), (89, 4)])


def test_padding_mask_invariance() -> None:
    """Prove padded content cannot leak into real-token outputs or pooling."""
    torch.manual_seed(0)
    T, w, shift, d_model = 26, 4, 2, D_MODEL
    pad_len = compute_pad_len(T, w)
    T_pad = T + pad_len
    assert pad_len == 2 and T_pad == 28, f"unexpected weekly padding geometry: pad_len={pad_len}, T_pad={T_pad}"

    block1 = SWiMWindowBlock(d_model, NHEAD, w, 0, DIM_FEEDFORWARD, dropout=0.0).eval()
    block2 = SWiMWindowBlock(d_model, NHEAD, w, shift, DIM_FEEDFORWARD, dropout=0.0).eval()

    real = torch.randn(2, T, d_model)
    zero_pad = torch.zeros(2, pad_len, d_model)
    garbage_pad = torch.randn(2, pad_len, d_model) * 1e4  # deliberately extreme values

    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    valid_mask[T:] = False

    x_zero = torch.cat([real, zero_pad], dim=1)
    x_garbage = torch.cat([real, garbage_pad], dim=1)

    with torch.no_grad():
        out_zero = block2(block1(x_zero, valid_mask), valid_mask)
        out_garbage = block2(block1(x_garbage, valid_mask), valid_mask)

    assert torch.isfinite(out_zero).all(), "NaN/Inf produced with zero padding"
    assert torch.isfinite(out_garbage).all(), "NaN/Inf produced with garbage padding"

    real_diff = (out_zero[:, :T, :] - out_garbage[:, :T, :]).abs().max().item()
    assert real_diff < 1e-4, f"padded content leaked into real-token outputs: max_abs_diff={real_diff}"

    pad_zero_out = out_zero[:, T:, :]
    pad_garbage_out = out_garbage[:, T:, :]
    assert torch.allclose(pad_zero_out, torch.zeros_like(pad_zero_out), atol=1e-6), "padded positions not hard-zeroed (zero-pad case)"
    assert torch.allclose(pad_garbage_out, torch.zeros_like(pad_garbage_out), atol=1e-6), "padded positions not hard-zeroed (garbage-pad case)"

    pool_zero = out_zero.sum(dim=1) / float(T)
    pool_garbage = out_garbage.sum(dim=1) / float(T)
    pool_diff = (pool_zero - pool_garbage).abs().max().item()
    assert pool_diff < 1e-4, f"padded content leaked into masked mean pooling: max_abs_diff={pool_diff}"

    print(f"  real-token max_abs_diff (zero-pad vs garbage-pad) = {real_diff:.2e}")
    print(f"  padded-position outputs hard-zeroed in both cases -- OK")
    print(f"  pooled representation max_abs_diff (zero-pad vs garbage-pad) = {pool_diff:.2e}")


def test_forward_shapes(ds, stats: dict, device: torch.device) -> None:
    X_daily = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:6], dtype=torch.float32, device=device)
    X_weekly = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:6], dtype=torch.float32, device=device)

    daily_model = SingleFrequencySWiMRegressor(feature_dim=X_daily.shape[-1], frequency_key="daily").to(device)
    weekly_model = SingleFrequencySWiMRegressor(feature_dim=X_weekly.shape[-1], frequency_key="weekly").to(device)
    dual_model = DualFrequencySWiMRegressor(daily_feature_dim=X_daily.shape[-1], weekly_feature_dim=X_weekly.shape[-1]).to(device)

    out_daily = daily_model(X_daily)
    out_weekly = weekly_model(X_weekly)
    out_dual = dual_model(X_daily, X_weekly)

    assert out_daily.shape == (6,), f"daily_only output shape: {out_daily.shape}"
    assert out_weekly.shape == (6,), f"weekly_only output shape: {out_weekly.shape}"
    assert out_dual.shape == (6,), f"daily_weekly output shape: {out_dual.shape}"


def test_backward_gradients(ds, stats: dict, device: torch.device) -> None:
    X_daily = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:8], dtype=torch.float32, device=device)
    X_weekly = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:8], dtype=torch.float32, device=device)
    y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)

    model = DualFrequencySWiMRegressor(daily_feature_dim=X_daily.shape[-1], weekly_feature_dim=X_weekly.shape[-1]).to(device)
    pred = model(X_daily, X_weekly)
    loss = nn.HuberLoss(delta=1.0)(pred, y)
    loss.backward()

    named_params = list(model.named_parameters())
    missing = [name for name, p in named_params if p.requires_grad and p.grad is None]
    non_finite = [name for name, p in named_params if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, f"parameters with no gradient (possible unused head): {missing}"
    assert not non_finite, f"parameters with non-finite gradient: {non_finite}"
    assert len(named_params) > 0, "model has no trainable parameters"


def test_nan_inf(ds, stats: dict, device: torch.device) -> None:
    X_weekly = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:8], dtype=torch.float32, device=device)
    model = SingleFrequencySWiMRegressor(feature_dim=X_weekly.shape[-1], frequency_key="weekly").to(device)
    model.eval()
    with torch.no_grad():
        out = model(X_weekly)
        seq, valid_mask = model.branch(X_weekly, return_sequence=True)
    assert torch.isfinite(out).all(), "NaN/Inf in weekly_only forward output"
    assert torch.isfinite(seq).all(), "NaN/Inf in weekly encoded sequence (pre-pool)"
    pad_positions = seq[:, ~valid_mask, :]
    assert torch.allclose(pad_positions, torch.zeros_like(pad_positions), atol=1e-6), "padded positions are not exactly zero"


def test_parameter_counts() -> dict:
    daily_model = SingleFrequencySWiMRegressor(feature_dim=5, frequency_key="daily")
    weekly_model = SingleFrequencySWiMRegressor(feature_dim=5, frequency_key="weekly")
    dual_model = DualFrequencySWiMRegressor(daily_feature_dim=5, weekly_feature_dim=5)

    counts = {
        "daily_only": count_trainable_parameters(daily_model),
        "weekly_only": count_trainable_parameters(weekly_model),
        "daily_weekly": count_trainable_parameters(dual_model),
    }
    expected_swim_counts = EXPECTED_SWIM_PARAM_COUNTS.copy()
    for key in counts:
        ratio = counts[key] / expected_swim_counts[key]
        print(f"  {key}: swim_params={counts[key]} expected_swim={expected_swim_counts[key]} ratio={ratio:.3f}")
        assert counts[key] == expected_swim_counts[key], (
            f"SWiM parameter count mismatch for {key}: got {counts[key]}, expected {expected_swim_counts[key]}"
        )
        assert np.isclose(ratio, 1.0, atol=1e-12), f"SWiM count ratio for {key} is not 1.0: {ratio}"
    return counts


def test_deterministic_forward(ds, stats: dict, device: torch.device) -> None:
    X_daily = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:4], dtype=torch.float32, device=device)
    model = SingleFrequencySWiMRegressor(feature_dim=X_daily.shape[-1], frequency_key="daily").to(device)
    model.eval()
    with torch.no_grad():
        out1 = model(X_daily)
        out2 = model(X_daily)
    assert torch.equal(out1, out2), "eval-mode forward pass is not deterministic for identical input"


def run_smoke_validation() -> None:
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    validate_frozen_dataset_contract(ds)
    stats = fit_train_only_normalizer(ds)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("FROZEN PANEL VALIDATION (SWiM)")
    print(f"train={int(np.sum(ds.split == 'train'))}")
    print(f"val={int(np.sum(ds.split == 'val'))}")
    print(f"test={int(np.sum(ds.split == 'test'))}")
    print(f"device={device.type.upper()}")
    print("============================================================")

    print("\n--- window/shift/padding audit tests ---")
    run_check("positional_encoding_shape", test_positional_encoding_shape)
    run_check("window_partition_reverse", test_window_partition_reverse)
    run_check("shift_boundary_mask", test_shift_mask)
    run_check("hard_mask_exactness", test_hard_mask_exactness)
    run_check("weekly_shifted_padding_geometry", test_weekly_shifted_padding_geometry)
    run_check("shifted_validity_alignment", test_shifted_validity_alignment)
    run_check("attention_mask_row_safety", test_attention_mask_row_safety)
    run_check("daily_shifted_window_geometry", test_daily_shifted_window_geometry)
    run_check("padding_mask_invariance", test_padding_mask_invariance)

    print("\n--- forward/backward/gradient audit tests ---")
    run_check("forward_shapes", lambda: test_forward_shapes(ds, stats, device))
    run_check("backward_gradients", lambda: test_backward_gradients(ds, stats, device))
    run_check("nan_inf", lambda: test_nan_inf(ds, stats, device))
    run_check("deterministic_forward", lambda: test_deterministic_forward(ds, stats, device))

    print("\n--- parameter-count audit ---")
    test_parameter_counts()

    print("\n--- smoke_test_models (forward+backward per frequency) ---")
    smoke_results = smoke_test_models(ds, stats, device)
    for frequency, info in smoke_results.items():
        print(f"{frequency}: smoke_ok={info['ok']} param_count={info['param_count']}")

    print("\nSmoke validation passed: forward/backward, gradient, mask, and padding checks OK.")
    print("Formal 9-experiment training is intentionally not started in this stage.")


def print_finite_summary(name: str, x: np.ndarray) -> None:
    finite = np.isfinite(x)
    print(f"{name}: shape={x.shape} finite_fraction={finite.mean():.6f} min={np.min(x):.6e} max={np.max(x):.6e} mean={np.mean(x):.6e} std={np.std(x):.6e} max_abs={np.max(np.abs(x)):.6e}")


def check_tensor_finite(label: str, tensor: torch.Tensor, *, allow_inf_mask: bool = False) -> bool:
    if tensor.dtype.is_floating_point:
        finite = torch.isfinite(tensor)
        if allow_inf_mask:
            ok = bool(finite.all())
        else:
            ok = bool(finite.all())
        if not ok:
            bad = torch.nonzero(~finite)
            print(f"[DEBUG_NAN] {label} has non-finite entries: count={bad.shape[0]} first={bad[:5].tolist()}")
        return ok
    return True


def debug_nan_mode_daily_only() -> None:
    global DEBUG_NAN_MODE, DEBUG_NAN_CONTEXT
    set_seed(SEED)
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    stats = fit_train_only_normalizer(ds)
    mps_available = torch.backends.mps.is_available()
    device = torch.device("mps" if mps_available else "cuda" if torch.cuda.is_available() else "cpu")
    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    X_all = apply_normalizer(ds, stats, "daily_only")
    X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
    y_train = torch.tensor(ds.y[5][train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
    y_val = torch.tensor(ds.y[5][val_mask], dtype=torch.float32)

    print("[DEBUG_NAN] start daily_only / 5d diagnostics")
    print(f"[DEBUG_NAN] MPS available = {mps_available}")
    print(f"[DEBUG_NAN] selected device = {device.type.upper()}")
    print(f"[DEBUG_NAN] X_train shape={tuple(X_train.shape)} y_train shape={tuple(y_train.shape)}")
    print(f"[DEBUG_NAN] X_val shape={tuple(X_val.shape)} y_val shape={tuple(y_val.shape)}")
    print_finite_summary("X_train", X_train.detach().cpu().numpy())
    print_finite_summary("y_train", y_train.detach().cpu().numpy())
    if not np.isfinite(X_train.detach().cpu().numpy()).all():
        bad = np.argwhere(~np.isfinite(X_train.detach().cpu().numpy()))
        print("[DEBUG_NAN] X_train has NaN/Inf at indices:", bad[:10].tolist())
        raise RuntimeError("Non-finite normalized Daily training data detected before model execution")
    if not np.isfinite(y_train.detach().cpu().numpy()).all():
        bad = np.argwhere(~np.isfinite(y_train.detach().cpu().numpy()))
        print("[DEBUG_NAN] y_train has NaN/Inf at indices:", bad[:10].tolist())
        raise RuntimeError("Non-finite normalized Daily target data detected before model execution")

    model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="daily").to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            print(f"[DEBUG_NAN] initial parameter non-finite: {name}")
            raise RuntimeError(f"Non-finite initial parameter encountered: {name}")
    print("[DEBUG_NAN] initial model parameters are finite")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    total_train_batches = len(loader)
    print(f"[DEBUG_NAN] total training batches in epoch 1 = {total_train_batches}")

    DEBUG_NAN_MODE = True
    train_loss_sum = 0.0
    train_weight = 0
    completed_train_batches = 0
    train_failed = False
    failure_stage = None
    failure_batch = None

    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        DEBUG_NAN_CONTEXT["frequency"] = "daily_only"
        DEBUG_NAN_CONTEXT["batch_index"] = batch_idx
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if not torch.isfinite(x_batch).all():
            print(f"[DEBUG_NAN] non-finite input batch: batch_idx={batch_idx}")
            print_finite_summary("x_batch", x_batch.detach().cpu().numpy())
            failure_stage = "input"
            failure_batch = batch_idx
            train_failed = True
            break
        if not torch.isfinite(y_batch).all():
            print(f"[DEBUG_NAN] non-finite y_batch: batch_idx={batch_idx}")
            print_finite_summary("y_batch", y_batch.detach().cpu().numpy())
            failure_stage = "input"
            failure_batch = batch_idx
            train_failed = True
            break

        optimizer.zero_grad()
        pred = model(x_batch)
        if not torch.isfinite(pred).all():
            print(f"[DEBUG_NAN] non-finite prediction at batch {batch_idx}")
            print_finite_summary("pred", pred.detach().cpu().numpy())
            failure_stage = "forward"
            failure_batch = batch_idx
            train_failed = True
            break

        loss = criterion(pred, y_batch)
        if not torch.isfinite(loss):
            print(f"[DEBUG_NAN] non-finite loss at batch {batch_idx}: loss={loss.item()}")
            failure_stage = "loss"
            failure_batch = batch_idx
            train_failed = True
            break

        loss.backward()
        bad_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    bad_grads.append((name, "missing"))
                elif not torch.isfinite(param.grad).all():
                    bad_grads.append((name, "nonfinite"))
        if bad_grads:
            for name, reason in bad_grads:
                print(f"[DEBUG_NAN] gradient issue: {name} -> {reason}")
            failure_stage = "backward"
            failure_batch = batch_idx
            train_failed = True
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"[DEBUG_NAN] clip_grad_norm_ produced non-finite norm at batch {batch_idx}: {grad_norm}")
            failure_stage = "gradient clipping"
            failure_batch = batch_idx
            train_failed = True
            break
        optimizer.step()
        for name, param in model.named_parameters():
            if param.requires_grad and not torch.isfinite(param).all():
                print(f"[DEBUG_NAN] parameter became non-finite after optimizer.step(): {name}")
                failure_stage = "optimizer step"
                failure_batch = batch_idx
                train_failed = True
                break
        if train_failed:
            break

        batch_size_value = y_batch.shape[0]
        train_loss_sum += float(loss.item()) * batch_size_value
        train_weight += int(batch_size_value)
        completed_train_batches += 1

        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == total_train_batches:
            print(f"[DEBUG_NAN] progress {batch_idx + 1}/{total_train_batches}")

    DEBUG_NAN_MODE = False
    DEBUG_NAN_CONTEXT["frequency"] = None
    DEBUG_NAN_CONTEXT["batch_index"] = None

    if train_failed:
        print(f"[DEBUG_NAN] failure reproduced during training: batch={failure_batch} stage={failure_stage}")
        return

    if completed_train_batches != total_train_batches:
        print(f"[DEBUG_NAN] training loop terminated early: completed={completed_train_batches} total={total_train_batches}")
        return

    debug_train_loss = train_loss_sum / train_weight
    print(f"[DEBUG_NAN] debug_train_loss={debug_train_loss:.12f}")
    if not np.isfinite(debug_train_loss):
        print("[DEBUG_NAN] debug_train_loss is non-finite")
        return

    model.eval()
    val_loss_sum = 0.0
    val_weight = 0
    val_failed = False
    val_failure_stage = None
    val_failure_batch = None
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if not torch.isfinite(x_batch).all():
                print(f"[DEBUG_NAN] non-finite validation input at batch={batch_idx}")
                val_failure_stage = "input"
                val_failure_batch = batch_idx
                val_failed = True
                break
            if not torch.isfinite(y_batch).all():
                print(f"[DEBUG_NAN] non-finite validation target at batch={batch_idx}")
                val_failure_stage = "input"
                val_failure_batch = batch_idx
                val_failed = True
                break

            pred = model(x_batch)
            if not torch.isfinite(pred).all():
                print(f"[DEBUG_NAN] non-finite validation prediction at batch={batch_idx}")
                val_failure_stage = "forward"
                val_failure_batch = batch_idx
                val_failed = True
                break

            loss = criterion(pred, y_batch)
            if not torch.isfinite(loss):
                print(f"[DEBUG_NAN] non-finite validation loss at batch={batch_idx}: loss={loss.item()}")
                val_failure_stage = "loss"
                val_failure_batch = batch_idx
                val_failed = True
                break
            batch_size_value = y_batch.shape[0]
            val_loss_sum += float(loss.item()) * batch_size_value
            val_weight += int(batch_size_value)

        if val_failed:
            print(f"[DEBUG_NAN] validation failure reproduced: batch={val_failure_batch} stage={val_failure_stage}")
            return

    debug_val_loss = val_loss_sum / val_weight if val_weight else float("nan")
    print(f"[DEBUG_NAN] debug_val_loss={debug_val_loss:.12f}")
    if not np.isfinite(debug_val_loss):
        print("[DEBUG_NAN] debug_val_loss is non-finite")
        return

    print("[DEBUG_NAN] FULL FIRST TRAINING EPOCH + VALIDATION BOTH FINITE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWiM-style temporal shifted-window Transformer control")
    parser.add_argument("--debug-nan", action="store_true", help="Temporary numerical diagnostic path; no formal outputs are written.")
    parser.add_argument("--trace-formal-nan", action="store_true", help="Run the real formal single-config training path for epoch 1 with read-only numerical tracing and no formal output writes.")
    parser.add_argument("--diagnose-mps-sync", action="store_true", help="Instructional stub for MPS sync diagnosis; use --diagnose-mps-sync-variant A/B/C/D/E to run exactly one variant.")
    parser.add_argument("--diagnose-mps-sync-variant", choices=["A", "B", "C", "D", "E"], help="Run exactly one MPS sync diagnostic variant: A/B/C/D/E. Each variant is fixed to daily_only / 5d and does not write formal outputs.")
    parser.add_argument("--train", action="store_true", help="Run formal training. Requires either --all or both --frequency and --horizon.")
    parser.add_argument("--all", action="store_true", help="Run the full 9-experiment benchmark.")
    parser.add_argument("--frequency", choices=FREQUENCY_SETTINGS, help="Single-frequency training selection: daily_only | weekly_only | daily_weekly")
    parser.add_argument("--horizon", choices=["5d", "10d", "20d"], help="Single-horizon training selection: 5d | 10d | 20d")
    parser.add_argument("--output-dir", type=str, default=str(MAIN_V2_SWIM_DIR), help="Formal SWiM MAIN V2 output root; must be dataset/audit/main_v2/swim.")
    parser.add_argument("--finalize-only", action="store_true", help="Finalize existing raw SWiM MAIN V2 evidence without retraining.")
    args = parser.parse_args()

    mutually_exclusive = [
        args.debug_nan,
        args.trace_formal_nan,
        args.diagnose_mps_sync,
        args.diagnose_mps_sync_variant is not None,
        args.train,
        args.finalize_only,
    ]
    if sum(1 for flag in mutually_exclusive if flag) > 1:
        parser.error("Use only one of --debug-nan, --trace-formal-nan, --diagnose-mps-sync, --diagnose-mps-sync-variant, --train, or --finalize-only.")

    if args.trace_formal_nan and (args.frequency is None or args.horizon is None):
        parser.error("--trace-formal-nan requires both --frequency and --horizon.")

    if args.diagnose_mps_sync and args.all:
        parser.error("--diagnose-mps-sync cannot be combined with --all.")
    if args.diagnose_mps_sync_variant is not None and args.all:
        parser.error("--diagnose-mps-sync-variant cannot be combined with --all.")
    if args.finalize_only and args.all:
        parser.error("--finalize-only cannot be combined with --all.")

    if args.all and not args.train:
        parser.error("--all requires --train.")
    if args.frequency is not None and not (args.train or args.trace_formal_nan):
        parser.error("--frequency requires --train or --trace-formal-nan.")
    if args.horizon is not None and not (args.train or args.trace_formal_nan):
        parser.error("--horizon requires --train or --trace-formal-nan.")
    if args.train:
        if args.all and (args.frequency is not None or args.horizon is not None):
            parser.error("Use either --all or a single --frequency/--horizon pair, not both.")
        if not args.all and (args.frequency is None or args.horizon is None):
            parser.error("When --train is supplied, either use --all or provide both --frequency and --horizon.")
        if args.frequency is not None and args.horizon is None:
            parser.error("If --frequency is supplied, --horizon must also be supplied.")
        if args.horizon is not None and args.frequency is None:
            parser.error("If --horizon is supplied, --frequency must also be supplied.")

    if args.diagnose_mps_sync:
        print("Use --diagnose-mps-sync-variant A|B|C|D|E to run exactly one MPS sync diagnostic variant. It is fixed to daily_only / 5d and does not write formal outputs.")
    return args


def diagnose_mps_sync_variant(variant: str) -> None:
    if not torch.backends.mps.is_available():
        print("MPS unavailable — sync diagnosis cannot be performed in this environment.")
        return

    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    stats = fit_train_only_normalizer(ds)
    device = torch.device("mps")
    set_seed(SEED)

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    X_all = apply_normalizer(ds, stats, "daily_only")
    X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
    y_train = torch.tensor(ds.y[5][train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
    y_val = torch.tensor(ds.y[5][val_mask], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="daily").to(device)
    if variant == "B":
        torch.mps.synchronize()
    elif variant == "C":
        param_total, param_sum, param_sq_sum = _model_parameter_fingerprint(model)
        print(f"[DIAG] variant={variant} total_parameters={param_total}")
        print(f"[DIAG] variant={variant} param_sum={param_sum:.12f}")
        print(f"[DIAG] variant={variant} param_sq_sum={param_sq_sum:.12f}")
    elif variant == "E":
        print(f"[DIAG] variant={variant} {_trace_rng_fingerprint_summary('torch_rng_state_before_train_loader')}")

    print(f"[DIAG] variant={variant} mps_available={torch.backends.mps.is_available()} selected_device={device.type.upper()} seed={SEED} parameter_count={sum(p.numel() for p in model.parameters())} expected_train_batches={len(train_loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)
    model.train()

    if variant == "D":
        torch.mps.synchronize()

    train_loss_sum = 0.0
    train_weight = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
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
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            batch_loss = criterion(logits, y_batch)
            batch_size_value = y_batch.shape[0]
            val_loss_sum += float(batch_loss.item()) * batch_size_value
            val_weight += int(batch_size_value)

    train_loss = train_loss_sum / train_weight if train_weight else float("nan")
    val_loss = val_loss_sum / val_weight if val_weight else float("nan")
    train_loss_finite = bool(np.isfinite(train_loss))
    val_loss_finite = bool(np.isfinite(val_loss))
    print(f"[DIAG] variant={variant} completed_train_batches={len(train_loader)} train_loss={train_loss:.12f} val_loss={val_loss:.12f} train_loss_finite={train_loss_finite} val_loss_finite={val_loss_finite}")


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    if args.debug_nan:
        debug_nan_mode_daily_only()
        return
    if args.trace_formal_nan:
        ds = load_frozen_panel()
        assert_frozen_panel_checks(ds)
        stats = fit_train_only_normalizer(ds)
        frequency = args.frequency
        horizon = int(args.horizon[:-1])
        train_and_eval_frequency(ds, frequency, horizon, stats, trace_nan=True)
        return
    if args.diagnose_mps_sync_variant is not None:
        diagnose_mps_sync_variant(args.diagnose_mps_sync_variant)
        return
    if args.diagnose_mps_sync:
        print("Use --diagnose-mps-sync-variant A|B|C|D|E to run exactly one MPS sync diagnostic variant. It is fixed to daily_only / 5d and does not write formal outputs.")
        return
    if args.finalize_only:
        output_dir = resolve_output_dir(args.output_dir)
        provenance_path = output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"]
        if not provenance_path.exists():
            raise FileNotFoundError(f"Missing formal training provenance artifact for finalize-only: {provenance_path}")
        provenance_df = pd.read_csv(provenance_path)
        if provenance_df.empty:
            raise ValueError(f"Training provenance file is empty: {provenance_path}")
        validate_training_provenance_row(provenance_df.iloc[0])
        required_artifacts = [
            FORMAL_SWIM_RUN_FILES["summary"],
            FORMAL_SWIM_RUN_FILES["predictions"],
            FORMAL_SWIM_RUN_FILES["rank_ic"],
            FORMAL_SWIM_RUN_FILES["history"],
        ]
        for artifact in required_artifacts:
            if not (output_dir / artifact).exists():
                raise FileNotFoundError(f"Missing formal SWiM artifact for finalize-only: {output_dir / artifact}")

        summary_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["summary"])
        pred_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["predictions"])
        rank_ic_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"])
        history_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["history"])
        validate_main_v2_swim_evidence(output_dir)

        script_path = ROOT / "scripts" / "python" / "panel_baseline_swim.py"
        finalization_sha = compute_sha256(script_path)
        training_sha = str(provenance_df.iloc[0].get("training_script_sha256", "MISSING"))
        if training_sha == finalization_sha:
            print("Finalize-only: training and finalization used identical script content.")
        else:
            print(f"Finalize-only: training and finalization script content differ (training_sha={training_sha}, finalization_sha={finalization_sha}).")

        comparison_df = build_baseline_comparison(summary_df)
        comparison_df.to_csv(output_dir / FORMAL_SWIM_RUN_FILES["comparison"], index=False)
        write_main_v2_swim_run_manifest(
            output_dir,
            summary_df,
            pred_df,
            rank_ic_df,
            history_df,
            finalization_sha,
        )
        write_final_summary(
            summary_df,
            comparison_df,
            training_sha,
            finalization_sha,
            output_dir=output_dir,
        )
        print("Finalize-only: derived SWiM MAIN V2 outputs regenerated without retraining.")
        return

    if args.train:
        output_dir = resolve_output_dir(args.output_dir)
        script_path = ROOT / "scripts" / "python" / "panel_baseline_swim.py"
        ds = load_frozen_panel()
        assert_frozen_panel_checks(ds)
        validate_frozen_dataset_contract(ds)
        stats = fit_train_only_normalizer(ds)
        if args.all:
            assert_clean_main_v2_swim_namespace(output_dir)
            write_training_provenance(output_dir, script_path)
            full_benchmark(ds, stats, output_dir=output_dir)
        else:
            write_training_provenance(output_dir, script_path)
            frequency = args.frequency
            horizon = int(args.horizon[:-1])
            horizon_label = f"{horizon}d"
            validate_partial_raw_cell_set_consistency(output_dir)
            ensure_requested_cell_absent(output_dir, frequency, horizon_label, seed=SEED)
            result = train_and_eval_frequency(ds, frequency, horizon, stats)
            save_summary_rows([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["summary"]).exists())
            save_prediction_rows([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["predictions"]).exists())
            save_training_history([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["history"]).exists())
            save_rank_ic_by_date([result], output_dir=output_dir, append_mode=(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]).exists())
            print(f"Single-config run complete: {frequency} / {args.horizon}")
    else:
        run_smoke_validation()


if __name__ == "__main__":
    main()
