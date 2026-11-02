"""Experiment 3 — PathFormer controlled multi-scale mechanism ablation (training runner).

Scientific contract
-------------------
Frequency configuration is FIXED to daily_weekly. This experiment isolates the
contribution of the multi-scale aggregation mechanism only:

  1. single  — one Daily scale expert + one Weekly scale expert
  2. fixed   — all four Daily + all four Weekly experts, equal 0.25 weights
  3. static  — all four Daily + all four Weekly experts, one learned global
               softmax weight vector per frequency (sample-independent)
  4. adaptive — sample-dependent within-frequency router

The adaptive arm is NOT trained here. It already exists as the frozen MAIN V2
daily_weekly evidence and is consumed only by the separate finalizer.

Architecture source of truth is the frozen scripts/python/panel_pathformer_main_v2.py.
This runner imports it read-only: it never modifies, monkeypatches, or mutates
that module's globals, and never invokes its formal training/finalization paths
or writes into its MAIN V2 namespace.

Test-leakage rule (critical)
----------------------------
All 48 single candidates are trained and compared on TRAIN + VALIDATION ONLY.
Single scale selection uses validation pooled MSE with a pre-frozen tie-break
(validation_mse, then smaller Daily patch, then smaller Weekly patch). TEST is
never read during training or selection. Only the 3 validation-selected single
models, plus the 3 fixed and 3 static models, may reach TEST evaluation.

This module writes nothing when invoked with no arguments (smoke validation).
"""

from __future__ import annotations

import argparse
import hashlib
import os
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

import scripts.python.panel_pathformer_main_v2 as formal
from scripts.python.panel_common import (
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    load_frozen_panel,
    rank_ic_by_date,
)

# ----------------------------------------------------------------------------
# Frozen experiment contract
# ----------------------------------------------------------------------------

RUN_FAMILY = "EXPERIMENT3_PATHFORMER_ABLATION"
FREQUENCY_CONFIGURATION = "daily_weekly"
TRAINABLE_MECHANISMS = ("single", "fixed", "static")
ADAPTIVE_MECHANISM = "adaptive"

HORIZON_LABELS = ("5d", "10d", "20d")
HORIZON_MAP = {"5d": 5, "10d": 10, "20d": 20}

DAILY_PATCH_SIZES = tuple(formal.DAILY_PATCH_SIZES)
WEEKLY_PATCH_SIZES = tuple(formal.WEEKLY_PATCH_SIZES)

SEED = formal.SEED
EXECUTION_BACKEND = "CPU"
SELECTION_METRIC = "validation_mse"
TIE_BREAK_RULE = "validation_mse_then_daily_patch_ascending_then_weekly_patch_ascending"

N_TICKERS = 17
N_TRAIN_SAMPLES = 30634
N_VALIDATION_SAMPLES = 6290
N_TEST_SAMPLES = 6647
N_COMMON_ANCHOR_DATES = 2563
N_VALIDATION_DATES = 370
N_TEST_DATES = 391

EXPECTED_SINGLE_CELLS = len(DAILY_PATCH_SIZES) * len(WEEKLY_PATCH_SIZES) * len(HORIZON_LABELS)  # 48
EXPECTED_FIXED_CELLS = len(HORIZON_LABELS)  # 3
EXPECTED_STATIC_CELLS = len(HORIZON_LABELS)  # 3
EXPECTED_TRAINING_CELLS = EXPECTED_SINGLE_CELLS + EXPECTED_FIXED_CELLS + EXPECTED_STATIC_CELLS  # 54
EXPECTED_SINGLE_VALIDATION_ROWS = EXPECTED_SINGLE_CELLS * N_VALIDATION_SAMPLES  # 301920
EXPECTED_TEST_CELLS = 9
EXPECTED_TEST_PREDICTION_ROWS = EXPECTED_TEST_CELLS * N_TEST_SAMPLES  # 59823
EXPECTED_TEST_RANK_IC_ROWS = EXPECTED_TEST_CELLS * N_TEST_DATES  # 3519

# Verified against the frozen architecture: expert = 12896 params;
# single = 2*(192 + 12896) + 4225; fixed = 2*(192 + 4*12896) + 4225;
# static = fixed + 2*4 scale logits; adaptive = fixed + 2*4260 router params.
EXPECTED_PARAMETER_COUNTS = {"single": 30401, "fixed": 107777, "static": 107785}
EXPECTED_ADAPTIVE_PARAMETER_COUNT = 116297

DATASET_MANIFEST_PATH = formal.DATASET_MANIFEST_PATH
EXPECTED_DATASET_MANIFEST_SHA256 = "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c"
FROZEN_PATHFORMER_SCRIPT_RELATIVE = "scripts/python/panel_pathformer_main_v2.py"
EXPECTED_FROZEN_PATHFORMER_SHA256 = "5f05ccd677359a899ca91c3baeeda3ecb6aae1a63ff75593cd31809ae27e6f77"
PANEL_COMMON_RELATIVE = "scripts/python/panel_common.py"
EXPECTED_PANEL_COMMON_SHA256 = "ccaba201b910b66fb5c3272bcc3aa18e04d12c098bb26100a73a3717ec6d2834"
TRAINING_SCRIPT_RELATIVE = "scripts/python/panel_pathformer_experiment3.py"

# ----------------------------------------------------------------------------
# Output namespace (Experiment 3 only; never the MAIN V2 namespace)
# ----------------------------------------------------------------------------

EXPERIMENT_DIR = ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

PROVENANCE_PATH = EXPERIMENT_DIR / "experiment3_training_provenance.csv"
TRAINING_SUMMARY_PATH = EXPERIMENT_DIR / "experiment3_training_summary.csv"
SINGLE_VAL_PRED_PATH = EXPERIMENT_DIR / "experiment3_single_validation_predictions.csv"
SINGLE_SELECTION_PATH = EXPERIMENT_DIR / "experiment3_single_selection.csv"
TRAINING_HISTORY_PATH = EXPERIMENT_DIR / "experiment3_training_history.csv"
TEST_SUMMARY_PATH = EXPERIMENT_DIR / "experiment3_test_summary_metrics.csv"
TEST_PREDICTIONS_PATH = EXPERIMENT_DIR / "experiment3_test_predictions.csv"
TEST_RANK_IC_PATH = EXPERIMENT_DIR / "experiment3_rank_ic_by_date.csv"
TEST_SCALE_WEIGHTS_PATH = EXPERIMENT_DIR / "experiment3_scale_weights.csv"

TRAINING_ARTIFACT_PATHS = (
    TRAINING_SUMMARY_PATH,
    TRAINING_HISTORY_PATH,
    SINGLE_VAL_PRED_PATH,
)
TEST_ARTIFACT_PATHS = (
    TEST_SUMMARY_PATH,
    TEST_PREDICTIONS_PATH,
    TEST_RANK_IC_PATH,
    TEST_SCALE_WEIGHTS_PATH,
)

DAILY_WEIGHT_COLUMNS = tuple(f"daily_w_{p}" for p in DAILY_PATCH_SIZES)
WEEKLY_WEIGHT_COLUMNS = tuple(f"weekly_w_{p}" for p in WEEKLY_PATCH_SIZES)

TRAINING_SUMMARY_KEY = ["mechanism", "horizon", "daily_patch", "weekly_patch", "seed"]

# Stored validation MSE comes from the frozen float32 metric path; reconstruction from
# CSV evidence accumulates in float64, so agreement is asserted at float32-scale tolerance.
VALIDATION_MSE_RTOL = 1e-6
VALIDATION_MSE_ATOL = 1e-12


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------


def _cpu_device() -> torch.device:
    return torch.device("cpu")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_date(value) -> str:
    return formal._canonical_anchor_date(value)


def _patch_token(value) -> str:
    """Stable string form for optional patch metadata (blank for fixed/static)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(int(value))


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _append_csv(df: pd.DataFrame, path: Path, duplicate_subset: list[str]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()
    if duplicate_subset and combined.duplicated(subset=duplicate_subset).any():
        raise RuntimeError(f"Refusing to append duplicate keys to {path.name}: {duplicate_subset}")
    _atomic_write_csv(combined, path)


def _reject_adaptive(mechanism: str) -> None:
    if str(mechanism).strip().lower() == ADAPTIVE_MECHANISM:
        raise RuntimeError(
            "Adaptive has no Experiment 3 training path. It is sourced from the frozen "
            "MAIN V2 daily_weekly evidence and must never be retrained here."
        )


def _validate_mechanism(mechanism: str) -> str:
    _reject_adaptive(mechanism)
    if mechanism not in TRAINABLE_MECHANISMS:
        raise ValueError(f"Unsupported Experiment 3 mechanism: {mechanism}")
    return mechanism


def _validate_horizon_label(horizon: str) -> str:
    if horizon not in HORIZON_MAP:
        raise ValueError(f"Unsupported horizon label: {horizon}")
    return horizon


# ----------------------------------------------------------------------------
# Model construction (frozen architecture, reused not reimplemented)
# ----------------------------------------------------------------------------


def _make_experiment3_model(mechanism: str, daily_patch: int | None = None, weekly_patch: int | None = None) -> nn.Module:
    _validate_mechanism(mechanism)
    if mechanism == "single":
        if daily_patch is None or weekly_patch is None:
            raise ValueError("single mechanism requires explicit daily_patch and weekly_patch.")
        if int(daily_patch) not in DAILY_PATCH_SIZES:
            raise ValueError(f"daily_patch must be one of {DAILY_PATCH_SIZES}; got {daily_patch}")
        if int(weekly_patch) not in WEEKLY_PATCH_SIZES:
            raise ValueError(f"weekly_patch must be one of {WEEKLY_PATCH_SIZES}; got {weekly_patch}")
        model = formal._make_model(
            FREQUENCY_CONFIGURATION,
            "single",
            selected_daily_patch=int(daily_patch),
            selected_weekly_patch=int(weekly_patch),
        )
    else:
        if daily_patch is not None or weekly_patch is not None:
            raise ValueError(f"{mechanism} uses all four scales per frequency; patch arguments are not permitted.")
        model = formal._make_model(FREQUENCY_CONFIGURATION, mechanism)

    actual = formal.count_trainable_parameters(model)
    expected = EXPECTED_PARAMETER_COUNTS[mechanism]
    if actual != expected:
        raise RuntimeError(f"Parameter count mismatch for {mechanism}: expected {expected}, got {actual}")
    return model


def _one_hot_weights(selected: int, candidates: tuple[int, ...], n_rows: int) -> np.ndarray:
    weights = np.zeros((n_rows, len(candidates)), dtype=np.float64)
    weights[:, candidates.index(int(selected))] = 1.0
    return weights


# ----------------------------------------------------------------------------
# Frozen dependency / provenance governance
# ----------------------------------------------------------------------------


def _current_dependency_hashes() -> dict:
    return {
        "training_script_sha256": _sha256(ROOT / TRAINING_SCRIPT_RELATIVE),
        "dataset_manifest_sha256": _sha256(DATASET_MANIFEST_PATH),
        "frozen_pathformer_script_sha256": _sha256(ROOT / FROZEN_PATHFORMER_SCRIPT_RELATIVE),
        "panel_common_sha256": _sha256(ROOT / PANEL_COMMON_RELATIVE),
    }


def _verify_frozen_dependencies() -> dict:
    hashes = _current_dependency_hashes()
    expected = {
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "frozen_pathformer_script_sha256": EXPECTED_FROZEN_PATHFORMER_SHA256,
        "panel_common_sha256": EXPECTED_PANEL_COMMON_SHA256,
    }
    for field, expected_value in expected.items():
        if hashes[field] != expected_value:
            raise RuntimeError(
                f"Frozen dependency SHA mismatch for {field}: expected {expected_value}, got {hashes[field]}"
            )
    return hashes


def _build_provenance_record() -> dict:
    hashes = _verify_frozen_dependencies()
    return {
        "run_family": RUN_FAMILY,
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "frequency_configuration": FREQUENCY_CONFIGURATION,
        "formal_ticker_count": N_TICKERS,
        "common_anchor_dates": N_COMMON_ANCHOR_DATES,
        "train_samples": N_TRAIN_SAMPLES,
        "validation_samples": N_VALIDATION_SAMPLES,
        "test_samples": N_TEST_SAMPLES,
        "seed": SEED,
        "execution_backend": EXECUTION_BACKEND,
        "training_mechanisms": "|".join(TRAINABLE_MECHANISMS),
        "adaptive_source": "FROZEN_MAIN_V2",
        "selection_metric": SELECTION_METRIC,
        "single_selection_uses_test": False,
        "tie_break_rule": TIE_BREAK_RULE,
        "dataset_manifest_path": str(DATASET_MANIFEST_PATH.relative_to(ROOT)),
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "training_script_path": TRAINING_SCRIPT_RELATIVE,
        "training_script_sha256": hashes["training_script_sha256"],
        "frozen_pathformer_script_path": FROZEN_PATHFORMER_SCRIPT_RELATIVE,
        "frozen_pathformer_script_sha256": hashes["frozen_pathformer_script_sha256"],
        "panel_common_path": PANEL_COMMON_RELATIVE,
        "panel_common_sha256": hashes["panel_common_sha256"],
    }


def _validate_stored_provenance(provenance_df: pd.DataFrame) -> pd.Series:
    if provenance_df.empty:
        raise RuntimeError("Experiment 3 provenance file is empty.")
    if len(provenance_df) != 1:
        raise RuntimeError(f"Experiment 3 provenance must contain exactly one row; found {len(provenance_df)}.")
    current = _build_provenance_record()
    stored = provenance_df.iloc[0]
    for field, current_value in current.items():
        if field not in provenance_df.columns:
            raise RuntimeError(f"Experiment 3 provenance is missing required field: {field}")
        if str(stored[field]) != str(current_value):
            raise RuntimeError(
                f"Immutable Experiment 3 provenance mismatch for {field}: stored={stored[field]}, current={current_value}"
            )
    return stored


def _ensure_provenance() -> pd.Series:
    """Create immutable provenance on first formal write; otherwise verify it."""
    any_artifact = any(p.exists() for p in TRAINING_ARTIFACT_PATHS + TEST_ARTIFACT_PATHS)
    if not PROVENANCE_PATH.exists():
        if any_artifact:
            raise RuntimeError(
                "Experiment 3 evidence exists without provenance; refusing to assign provenance retroactively."
            )
        record = _build_provenance_record()
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_csv(pd.DataFrame([record]), PROVENANCE_PATH)
        return pd.DataFrame([record]).iloc[0]
    return _validate_stored_provenance(pd.read_csv(PROVENANCE_PATH))


def _require_clean_namespace() -> None:
    if EXPERIMENT_DIR.exists() and any(EXPERIMENT_DIR.iterdir()):
        raise RuntimeError(
            "Experiment 3 namespace is not clean; use --resume to continue an existing run "
            "instead of overwriting completed evidence."
        )


# ----------------------------------------------------------------------------
# Cell identity / completion governance
# ----------------------------------------------------------------------------


def _all_training_cells() -> list[dict]:
    cells: list[dict] = []
    for horizon in HORIZON_LABELS:
        for daily_patch in DAILY_PATCH_SIZES:
            for weekly_patch in WEEKLY_PATCH_SIZES:
                cells.append({
                    "mechanism": "single",
                    "horizon": horizon,
                    "daily_patch": int(daily_patch),
                    "weekly_patch": int(weekly_patch),
                })
    for mechanism in ("fixed", "static"):
        for horizon in HORIZON_LABELS:
            cells.append({"mechanism": mechanism, "horizon": horizon, "daily_patch": None, "weekly_patch": None})
    if len(cells) != EXPECTED_TRAINING_CELLS:
        raise RuntimeError(f"Training grid must contain exactly {EXPECTED_TRAINING_CELLS} cells; built {len(cells)}.")
    return cells


def _checkpoint_path(mechanism: str, horizon: str, daily_patch: int | None, weekly_patch: int | None) -> Path:
    if mechanism == "single":
        name = f"single_{horizon}_d{int(daily_patch)}_w{int(weekly_patch)}.pt"
    else:
        name = f"{mechanism}_{horizon}.pt"
    return CHECKPOINT_DIR / name


def _cell_mask(df: pd.DataFrame, mechanism: str, horizon: str, daily_patch, weekly_patch) -> pd.Series:
    mask = (df["mechanism"].astype(str) == mechanism) & (df["horizon"].astype(str) == horizon)
    if mechanism == "single":
        mask &= df["daily_patch"].apply(_patch_token) == _patch_token(daily_patch)
        mask &= df["weekly_patch"].apply(_patch_token) == _patch_token(weekly_patch)
    return mask


_CSV_CACHE: dict[Path, tuple[tuple[int, int], pd.DataFrame]] = {}


def _read_csv_cached(path: Path) -> pd.DataFrame:
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _CSV_CACHE.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    df = pd.read_csv(path)
    _CSV_CACHE[path] = (key, df)
    return df


def _validate_single_candidate_predictions(
    val_df: pd.DataFrame, horizon: str, daily_patch: int, weekly_patch: int
) -> float:
    """Validate one single candidate's validation evidence and return reconstructed pooled MSE."""
    label = f"single/{horizon}/d={int(daily_patch)}/w={int(weekly_patch)}"
    subset = val_df[_cell_mask(val_df, "single", horizon, daily_patch, weekly_patch)]
    if len(subset) != N_VALIDATION_SAMPLES:
        raise RuntimeError(f"{label}: expected {N_VALIDATION_SAMPLES} validation rows, found {len(subset)}.")
    if not (subset["seed"].astype(int) == SEED).all():
        raise RuntimeError(f"{label}: every validation row must be seed {SEED}.")
    if not (subset["mechanism"].astype(str) == "single").all():
        raise RuntimeError(f"{label}: every validation row must be mechanism=single.")
    if not (subset["daily_patch"].astype(int) == int(daily_patch)).all():
        raise RuntimeError(f"{label}: daily_patch metadata mismatch in validation rows.")
    if not (subset["weekly_patch"].astype(int) == int(weekly_patch)).all():
        raise RuntimeError(f"{label}: weekly_patch metadata mismatch in validation rows.")
    if subset["anchor_date"].nunique() != N_VALIDATION_DATES:
        raise RuntimeError(f"{label}: expected {N_VALIDATION_DATES} validation dates, found {subset['anchor_date'].nunique()}.")
    if subset["ticker"].nunique() != N_TICKERS:
        raise RuntimeError(f"{label}: expected {N_TICKERS} tickers, found {subset['ticker'].nunique()}.")
    if subset.duplicated(subset=["ticker", "anchor_date"]).any():
        raise RuntimeError(f"{label}: duplicate ticker/anchor_date validation keys detected.")

    y_true = subset["y_true"].to_numpy(dtype=np.float64)
    y_pred = subset["y_pred"].to_numpy(dtype=np.float64)
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise RuntimeError(f"{label}: validation predictions must be finite.")
    return float(np.mean((y_pred - y_true) ** 2))


def _cell_status(cell: dict) -> str:
    """Return 'complete' or 'absent'; raise on partial or internally inconsistent evidence."""
    mechanism = cell["mechanism"]
    horizon = cell["horizon"]
    daily_patch = cell["daily_patch"]
    weekly_patch = cell["weekly_patch"]
    label = f"{mechanism}/{horizon}/d={_patch_token(daily_patch) or '-'}/w={_patch_token(weekly_patch) or '-'}"

    summary_rows = 0
    summary_row = None
    if TRAINING_SUMMARY_PATH.exists():
        summary_df = _read_csv_cached(TRAINING_SUMMARY_PATH)
        subset = summary_df[_cell_mask(summary_df, mechanism, horizon, daily_patch, weekly_patch)]
        summary_rows = len(subset)
        if summary_rows == 1:
            summary_row = subset.iloc[0]
        elif summary_rows > 1:
            raise RuntimeError(f"Inconsistent Experiment 3 evidence: duplicate training summary rows for {label}.")

    history_rows = 0
    history_subset = None
    if TRAINING_HISTORY_PATH.exists():
        history_df = _read_csv_cached(TRAINING_HISTORY_PATH)
        history_subset = history_df[_cell_mask(history_df, mechanism, horizon, daily_patch, weekly_patch)]
        history_rows = len(history_subset)

    val_rows = 0
    if mechanism == "single" and SINGLE_VAL_PRED_PATH.exists():
        val_df = _read_csv_cached(SINGLE_VAL_PRED_PATH)
        val_rows = len(val_df[_cell_mask(val_df, mechanism, horizon, daily_patch, weekly_patch)])

    checkpoint_path = _checkpoint_path(mechanism, horizon, daily_patch, weekly_patch)
    checkpoint_exists = checkpoint_path.exists()

    signals = [summary_rows > 0, history_rows > 0, checkpoint_exists]
    if mechanism == "single":
        signals.append(val_rows > 0)
    if not any(signals):
        return "absent"
    if not all(signals):
        raise RuntimeError(
            f"Inconsistent partial Experiment 3 evidence for {label}: "
            f"summary_rows={summary_rows}, history_rows={history_rows}, "
            f"checkpoint={checkpoint_exists}, validation_rows={val_rows}. Refusing to repair or overwrite."
        )

    # Summary metadata integrity.
    if int(summary_row["seed"]) != SEED:
        raise RuntimeError(f"{label}: summary seed must be {SEED}, found {summary_row['seed']}.")
    if str(summary_row["mechanism"]) != mechanism or str(summary_row["horizon"]) != horizon:
        raise RuntimeError(f"{label}: summary mechanism/horizon metadata mismatch.")
    if _patch_token(summary_row["daily_patch"]) != _patch_token(daily_patch):
        raise RuntimeError(f"{label}: summary daily_patch metadata mismatch.")
    if _patch_token(summary_row["weekly_patch"]) != _patch_token(weekly_patch):
        raise RuntimeError(f"{label}: summary weekly_patch metadata mismatch.")
    expected_params = EXPECTED_PARAMETER_COUNTS[mechanism]
    if int(summary_row["parameter_count"]) != expected_params:
        raise RuntimeError(
            f"{label}: parameter_count must be {expected_params}, found {summary_row['parameter_count']}."
        )

    # Training-history integrity.
    epochs_trained = int(summary_row["epochs_trained"])
    if history_rows != epochs_trained:
        raise RuntimeError(f"Inconsistent Experiment 3 history length for {label}: {history_rows} != {epochs_trained}.")
    epochs = sorted(history_subset["epoch"].astype(int).tolist())
    if epochs != list(range(1, epochs_trained + 1)):
        raise RuntimeError(f"Experiment 3 history epochs must be contiguous 1..{epochs_trained} for {label}.")
    if not (history_subset["seed"].astype(int) == SEED).all():
        raise RuntimeError(f"{label}: every history row must be seed {SEED}.")
    train_losses = history_subset["train_loss"].to_numpy(dtype=np.float64)
    val_losses = history_subset["val_loss"].to_numpy(dtype=np.float64)
    if not np.isfinite(train_losses).all() or not np.isfinite(val_losses).all():
        raise RuntimeError(f"{label}: all history train_loss/val_loss values must be finite.")
    ordered_history = history_subset.sort_values("epoch")
    ordered_val = ordered_history["val_loss"].to_numpy(dtype=np.float64)
    min_val = float(ordered_val.min())
    best_epoch = int(summary_row["best_epoch"])
    if best_epoch not in set(ordered_history.loc[ordered_val == min_val, "epoch"].astype(int).tolist()):
        raise RuntimeError(f"{label}: best_epoch must be the epoch attaining the minimum validation loss.")
    if not np.isclose(float(summary_row["best_val_huber"]), min_val, rtol=1e-9, atol=1e-12):
        raise RuntimeError(f"{label}: best_val_huber must equal the minimum history validation loss.")

    # Checkpoint integrity.
    if _sha256(checkpoint_path) != str(summary_row["checkpoint_sha256"]):
        raise RuntimeError(f"Experiment 3 checkpoint SHA mismatch for {label}.")

    # Single validation-evidence integrity and independent MSE reconstruction.
    if mechanism == "single":
        reconstructed_mse = _validate_single_candidate_predictions(
            _read_csv_cached(SINGLE_VAL_PRED_PATH), horizon, int(daily_patch), int(weekly_patch)
        )
        stored_mse = float(summary_row["validation_mse"])
        if not np.isclose(reconstructed_mse, stored_mse, rtol=VALIDATION_MSE_RTOL, atol=VALIDATION_MSE_ATOL):
            raise RuntimeError(
                f"{label}: reconstructed validation MSE {reconstructed_mse} does not match stored {stored_mse}."
            )
    return "complete"


# ----------------------------------------------------------------------------
# Training / validation (never touches TEST)
# ----------------------------------------------------------------------------


def _load_dataset():
    ds = load_frozen_panel()
    formal._formal_check_dataset_contract(ds)
    return ds


def _build_train_val_loaders_only(horizon_value: int, ds, batch_size: int = None):
    """Train + Validation loaders only; TEST tensors are never materialized here.

    Mirrors the frozen daily_weekly branch of formal._build_training_loaders
    (train-only normalization, Daily+Weekly alignment, batch size, seed-42
    shuffled Train loader, unshuffled Validation loader) with the entire test
    split deliberately unreachable from the training/selection path.
    """
    batch_size = formal.BATCH_SIZE if batch_size is None else batch_size
    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    stats = fit_train_only_normalizer(ds)

    Xd_norm, Xw_norm = apply_normalizer(ds, stats, FREQUENCY_CONFIGURATION)
    Xd_train, Xd_val = Xd_norm[train_mask], Xd_norm[val_mask]
    Xw_train, Xw_val = Xw_norm[train_mask], Xw_norm[val_mask]
    y_train, y_val = ds.y[horizon_value][train_mask], ds.y[horizon_value][val_mask]

    train_set = TensorDataset(torch.from_numpy(Xd_train), torch.from_numpy(Xw_train), torch.from_numpy(y_train))
    val_set = TensorDataset(torch.from_numpy(Xd_val), torch.from_numpy(Xw_val), torch.from_numpy(y_val))

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _batched_inference(
    model: nn.Module,
    loader,
    mechanism: str,
    daily_patch: int | None,
    weekly_patch: int | None,
    expected_rows: int,
    device: torch.device,
):
    """Forward-only inference returning predictions plus per-sample scale weights."""
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    daily_chunks: list[torch.Tensor] = []
    weekly_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for x_daily, x_weekly, _ in loader:
            x_daily = x_daily.to(device)
            x_weekly = x_weekly.to(device)
            pred, daily_w, weekly_w = model.forward_with_router(x_daily, x_weekly)
            pred_chunks.append(pred.detach().cpu())
            daily_chunks.append(daily_w.detach().cpu())
            weekly_chunks.append(weekly_w.detach().cpu())

    if not pred_chunks:
        raise RuntimeError("No batches were processed during Experiment 3 inference.")
    pred_np = torch.cat(pred_chunks, dim=0).numpy().astype(np.float32)
    if pred_np.shape[0] != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} predictions, got {pred_np.shape[0]}.")

    if mechanism == "single":
        daily_np = _one_hot_weights(daily_patch, DAILY_PATCH_SIZES, expected_rows)
        weekly_np = _one_hot_weights(weekly_patch, WEEKLY_PATCH_SIZES, expected_rows)
    else:
        daily_np = torch.cat(daily_chunks, dim=0).numpy().astype(np.float64)
        weekly_np = torch.cat(weekly_chunks, dim=0).numpy().astype(np.float64)
        if daily_np.shape != (expected_rows, len(DAILY_PATCH_SIZES)):
            raise RuntimeError(f"Daily scale-weight shape mismatch: {daily_np.shape}")
        if weekly_np.shape != (expected_rows, len(WEEKLY_PATCH_SIZES)):
            raise RuntimeError(f"Weekly scale-weight shape mismatch: {weekly_np.shape}")

    return pred_np, daily_np, weekly_np


def train_and_validate_cell(mechanism: str, horizon: str, daily_patch: int | None, weekly_patch: int | None) -> dict:
    """Train one Experiment 3 cell on Train, early-stop on Validation. Never reads TEST."""
    _validate_mechanism(mechanism)
    _validate_horizon_label(horizon)
    horizon_value = HORIZON_MAP[horizon]
    device = _cpu_device()

    ds = _load_dataset()
    formal.set_seed(SEED)
    train_loader, val_loader = _build_train_val_loaders_only(horizon_value, ds, batch_size=formal.BATCH_SIZE)

    model = _make_experiment3_model(mechanism, daily_patch, weekly_patch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=formal.LEARNING_RATE)
    loss_fn = nn.HuberLoss(delta=formal.HUBER_DELTA)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    min_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, formal.MAX_EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        for x_daily, x_weekly, y_batch in train_loader:
            x_daily = x_daily.to(device)
            x_weekly = x_weekly.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_daily, x_weekly)
            loss = loss_fn(pred, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {mechanism}/{horizon}, epoch={epoch}")
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"Non-finite gradient for {name} at {mechanism}/{horizon}, epoch={epoch}")
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=formal.GRADIENT_CLIP_MAX_NORM)
            optimizer.step()
            train_loss_total += loss.item() * y_batch.shape[0]

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x_daily, x_weekly, y_batch in val_loader:
                x_daily = x_daily.to(device)
                x_weekly = x_weekly.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_daily, x_weekly)
                val_loss = loss_fn(pred, y_batch)
                if not torch.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite validation loss at {mechanism}/{horizon}, epoch={epoch}")
                val_loss_total += val_loss.item() * y_batch.shape[0]

        train_loss = train_loss_total / max(len(train_loader.dataset), 1)
        val_loss = val_loss_total / max(len(val_loader.dataset), 1)
        history.append({
            "mechanism": mechanism,
            "horizon": horizon,
            "daily_patch": daily_patch,
            "weekly_patch": weekly_patch,
            "seed": SEED,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        if val_loss < min_val_loss - 1e-12:
            min_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= formal.EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    runtime_seconds = time.time() - start_time

    val_mask = ds.split == "val"
    val_ticker = ds.ticker[val_mask]
    val_anchor = [_canonical_date(v) for v in ds.anchor_date[val_mask]]
    val_y_true = ds.y[horizon_value][val_mask]
    val_pred, _daily_w, _weekly_w = _batched_inference(
        model, val_loader, mechanism, daily_patch, weekly_patch, N_VALIDATION_SAMPLES, device
    )
    val_metrics = evaluate_predictions(val_y_true, val_pred, val_ticker, val_anchor)

    checkpoint_path = _checkpoint_path(mechanism, horizon, daily_patch, weekly_patch)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_checkpoint = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(best_state, tmp_checkpoint)
    os.replace(tmp_checkpoint, checkpoint_path)

    summary_row = {
        "mechanism": mechanism,
        "horizon": horizon,
        "daily_patch": daily_patch,
        "weekly_patch": weekly_patch,
        "seed": SEED,
        "parameter_count": formal.count_trainable_parameters(model),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(history)),
        "best_val_huber": float(min_val_loss),
        "validation_mse": float(val_metrics["pooled_mse"]),
        "validation_mae": float(val_metrics["pooled_mae"]),
        "validation_corr": float(val_metrics["pooled_corr"]),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "runtime_seconds": float(runtime_seconds),
    }

    validation_predictions = None
    if mechanism == "single":
        validation_predictions = pd.DataFrame({
            "mechanism": mechanism,
            "horizon": horizon,
            "daily_patch": int(daily_patch),
            "weekly_patch": int(weekly_patch),
            "seed": SEED,
            "ticker": val_ticker,
            "anchor_date": val_anchor,
            "y_true": val_y_true.astype(np.float32),
            "y_pred": val_pred.astype(np.float32),
        })

    return {"summary": summary_row, "history": history, "validation_predictions": validation_predictions}


def _write_training_cell_evidence(cell: dict, result: dict) -> None:
    summary_df = pd.DataFrame([result["summary"]])
    history_df = pd.DataFrame(result["history"])
    _append_csv(summary_df, TRAINING_SUMMARY_PATH, TRAINING_SUMMARY_KEY)
    _append_csv(history_df, TRAINING_HISTORY_PATH, TRAINING_SUMMARY_KEY + ["epoch"])
    if cell["mechanism"] == "single":
        predictions = result["validation_predictions"]
        if predictions is None or len(predictions) != N_VALIDATION_SAMPLES:
            raise RuntimeError("Single training cell must produce exactly 6290 validation predictions.")
        _append_csv(predictions, SINGLE_VAL_PRED_PATH, TRAINING_SUMMARY_KEY + ["ticker", "anchor_date"])


def _run_train_cell(mechanism: str, horizon: str, daily_patch: int | None, weekly_patch: int | None) -> None:
    cell = {"mechanism": mechanism, "horizon": horizon, "daily_patch": daily_patch, "weekly_patch": weekly_patch}
    _ensure_provenance()
    if _cell_status(cell) == "complete":
        raise RuntimeError(f"Experiment 3 cell already complete: {mechanism}/{horizon}; refusing to overwrite.")
    result = train_and_validate_cell(mechanism, horizon, daily_patch, weekly_patch)
    _write_training_cell_evidence(cell, result)
    print(f"EXPERIMENT 3 CELL COMPLETE: {mechanism}/{horizon}/d={_patch_token(daily_patch) or '-'}/w={_patch_token(weekly_patch) or '-'}")


def _run_train_all(resume: bool) -> None:
    if not resume:
        _require_clean_namespace()
    _ensure_provenance()
    cells = _all_training_cells()
    for cell in cells:
        status = _cell_status(cell)
        if status == "complete":
            if not resume:
                raise RuntimeError("Completed cell found without --resume; refusing to overwrite formal evidence.")
            continue
        result = train_and_validate_cell(cell["mechanism"], cell["horizon"], cell["daily_patch"], cell["weekly_patch"])
        _write_training_cell_evidence(cell, result)
    for cell in cells:
        if _cell_status(cell) != "complete":
            raise RuntimeError(f"Training finished but cell is incomplete: {cell}")
    print(f"EXPERIMENT 3 TRAINING COMPLETE: {len(cells)} cells")


# ----------------------------------------------------------------------------
# Single selection (VALIDATION ONLY)
# ----------------------------------------------------------------------------


def select_single_by_validation(single_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Select one single candidate per horizon using validation MSE only.

    Accepts validation evidence only: any test column is rejected outright.
    """
    forbidden = [c for c in single_summary_df.columns if "test" in str(c).lower()]
    if forbidden:
        raise RuntimeError(f"Single selection must not see test columns: {forbidden}")

    rows = []
    for horizon in HORIZON_LABELS:
        candidates = single_summary_df[single_summary_df["horizon"].astype(str) == horizon].copy()
        if len(candidates) != len(DAILY_PATCH_SIZES) * len(WEEKLY_PATCH_SIZES):
            raise RuntimeError(
                f"Expected {len(DAILY_PATCH_SIZES) * len(WEEKLY_PATCH_SIZES)} single candidates for {horizon}, "
                f"found {len(candidates)}."
            )
        candidates["daily_patch"] = candidates["daily_patch"].astype(int)
        candidates["weekly_patch"] = candidates["weekly_patch"].astype(int)
        candidates["validation_mse"] = candidates["validation_mse"].astype(float)
        if not np.isfinite(candidates["validation_mse"].to_numpy()).all():
            raise RuntimeError(f"Non-finite validation MSE among single candidates for {horizon}.")
        ordered = candidates.sort_values(
            by=["validation_mse", "daily_patch", "weekly_patch"], ascending=[True, True, True], kind="mergesort"
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        rows.append({
            "horizon": horizon,
            "selected_daily_patch": int(best["daily_patch"]),
            "selected_weekly_patch": int(best["weekly_patch"]),
            "validation_mse": float(best["validation_mse"]),
            "selection_rank": 1,
            "selection_metric": SELECTION_METRIC,
            "tie_break_rule": TIE_BREAK_RULE,
            "selected_checkpoint_path": str(best["checkpoint_path"]),
            "selected_checkpoint_sha256": str(best["checkpoint_sha256"]),
        })
    return pd.DataFrame(rows)


def _build_reconstructed_single_candidates(summary_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Independently recompute pooled validation MSE for all 48 single candidates."""
    single_summary = summary_df[summary_df["mechanism"].astype(str) == "single"].copy()
    if len(single_summary) != EXPECTED_SINGLE_CELLS:
        raise RuntimeError(f"Expected {EXPECTED_SINGLE_CELLS} single training rows, found {len(single_summary)}.")
    if len(val_df) != EXPECTED_SINGLE_VALIDATION_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_SINGLE_VALIDATION_ROWS} single validation rows, found {len(val_df)}.")

    rows = []
    for horizon in HORIZON_LABELS:
        for daily_patch in DAILY_PATCH_SIZES:
            for weekly_patch in WEEKLY_PATCH_SIZES:
                matched = single_summary[_cell_mask(single_summary, "single", horizon, daily_patch, weekly_patch)]
                if len(matched) != 1:
                    raise RuntimeError(
                        f"Expected exactly one summary row for single/{horizon}/d={daily_patch}/w={weekly_patch}; found {len(matched)}."
                    )
                summary_row = matched.iloc[0]
                reconstructed_mse = _validate_single_candidate_predictions(
                    val_df, horizon, int(daily_patch), int(weekly_patch)
                )
                stored_mse = float(summary_row["validation_mse"])
                if not np.isclose(reconstructed_mse, stored_mse, rtol=VALIDATION_MSE_RTOL, atol=VALIDATION_MSE_ATOL):
                    raise RuntimeError(
                        f"single/{horizon}/d={daily_patch}/w={weekly_patch}: reconstructed validation MSE "
                        f"{reconstructed_mse} does not match stored {stored_mse}."
                    )
                rows.append({
                    "mechanism": "single",
                    "horizon": horizon,
                    "daily_patch": int(daily_patch),
                    "weekly_patch": int(weekly_patch),
                    "validation_mse": reconstructed_mse,
                    "checkpoint_path": str(summary_row["checkpoint_path"]),
                    "checkpoint_sha256": str(summary_row["checkpoint_sha256"]),
                })
    return pd.DataFrame(rows)


def _reproduce_single_selection() -> pd.DataFrame:
    """Rebuild the 3-row selection from current validation evidence only."""
    summary_df = _read_csv_cached(TRAINING_SUMMARY_PATH)
    val_df = _read_csv_cached(SINGLE_VAL_PRED_PATH)
    candidates = _build_reconstructed_single_candidates(summary_df, val_df)
    selection = select_single_by_validation(candidates)
    if len(selection) != len(HORIZON_LABELS):
        raise RuntimeError(f"Single selection must contain exactly {len(HORIZON_LABELS)} rows.")
    return selection


def _run_select_single() -> None:
    _ensure_provenance()
    for cell in _all_training_cells():
        if cell["mechanism"] != "single":
            continue
        if _cell_status(cell) != "complete":
            raise RuntimeError(f"Cannot select: incomplete single cell {cell['horizon']}/d={cell['daily_patch']}/w={cell['weekly_patch']}.")

    if SINGLE_SELECTION_PATH.exists():
        raise RuntimeError("Single selection already exists; refusing to reselect or overwrite.")

    selection = _reproduce_single_selection()
    selection["training_summary_sha256"] = _sha256(TRAINING_SUMMARY_PATH)
    selection["single_validation_predictions_sha256"] = _sha256(SINGLE_VAL_PRED_PATH)
    _atomic_write_csv(selection, SINGLE_SELECTION_PATH)
    print("EXPERIMENT 3 SINGLE SELECTION COMPLETE (validation MSE only)")


# ----------------------------------------------------------------------------
# TEST evaluation of already-frozen checkpoints (no training, no reselection)
# ----------------------------------------------------------------------------


def evaluate_frozen_checkpoint_on_test(
    mechanism: str,
    horizon: str,
    daily_patch: int | None,
    weekly_patch: int | None,
    summary_row: pd.Series,
    ds,
) -> dict:
    horizon_value = HORIZON_MAP[horizon]
    device = _cpu_device()
    formal.set_seed(SEED)
    _train_loader, _val_loader, test_loader = formal._build_training_loaders(
        FREQUENCY_CONFIGURATION, horizon_value, ds, batch_size=formal.BATCH_SIZE
    )

    checkpoint_path = ROOT / str(summary_row["checkpoint_path"])
    if _sha256(checkpoint_path) != str(summary_row["checkpoint_sha256"]):
        raise RuntimeError(f"Checkpoint SHA mismatch before test evaluation: {checkpoint_path}")
    model = _make_experiment3_model(mechanism, daily_patch, weekly_patch).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_mask = ds.split == "test"
    ticker = ds.ticker[test_mask]
    anchor_date = [_canonical_date(v) for v in ds.anchor_date[test_mask]]
    y_true = ds.y[horizon_value][test_mask]

    pred_np, daily_np, weekly_np = _batched_inference(
        model, test_loader, mechanism, daily_patch, weekly_patch, N_TEST_SAMPLES, device
    )

    if mechanism == "fixed":
        if not np.allclose(daily_np, 0.25, rtol=0.0, atol=1e-9) or not np.allclose(weekly_np, 0.25, rtol=0.0, atol=1e-9):
            raise RuntimeError("Fixed mechanism must use exact 0.25 equal scale weights.")
    if mechanism == "static":
        if not np.all(daily_np == daily_np[0]) or not np.all(weekly_np == weekly_np[0]):
            raise RuntimeError("Static scale weights must be identical for every sample within a cell.")
    if (daily_np < 0).any() or (weekly_np < 0).any():
        raise RuntimeError("Scale weights must be non-negative.")
    if not np.allclose(daily_np.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Daily scale weights must sum to 1.")
    if not np.allclose(weekly_np.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Weekly scale weights must sum to 1.")

    metrics = evaluate_predictions(y_true, pred_np, ticker, anchor_date)
    rank_ic_df = rank_ic_by_date(y_true, pred_np, ticker, anchor_date)
    rank_ic_df["anchor_date"] = rank_ic_df["anchor_date"].apply(_canonical_date)
    if len(rank_ic_df) != N_TEST_DATES:
        raise RuntimeError(f"Rank-IC must contain {N_TEST_DATES} dates for {mechanism}/{horizon}.")
    valid_ic = rank_ic_df["rank_ic"].dropna().to_numpy()
    mean_rank_ic = float(np.mean(valid_ic)) if len(valid_ic) else float("nan")
    rank_ic_std = float(np.std(valid_ic)) if len(valid_ic) else float("nan")
    rank_icir = float(mean_rank_ic / rank_ic_std) if np.isfinite(rank_ic_std) and rank_ic_std > 0 else float("nan")

    rank_ic_df.insert(0, "seed", SEED)
    rank_ic_df.insert(0, "horizon", horizon)
    rank_ic_df.insert(0, "mechanism", mechanism)

    summary = {
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": SEED,
        "parameter_count": int(summary_row["parameter_count"]),
        "selected_daily_patch": daily_patch,
        "selected_weekly_patch": weekly_patch,
        "best_epoch": int(summary_row["best_epoch"]),
        "epochs_trained": int(summary_row["epochs_trained"]),
        "best_val_huber": float(summary_row["best_val_huber"]),
        "test_mse": float(metrics["pooled_mse"]),
        "test_mae": float(metrics["pooled_mae"]),
        "test_corr": float(metrics["pooled_corr"]),
        "ticker_avg_mse": float(metrics["ticker_avg_mse"]),
        "ticker_avg_mae": float(metrics["ticker_avg_mae"]),
        "ticker_avg_corr": float(metrics["ticker_avg_corr"]),
        "mean_rank_ic": mean_rank_ic,
        "median_rank_ic": float(np.median(valid_ic)) if len(valid_ic) else float("nan"),
        "rank_ic_std": rank_ic_std,
        "positive_ic_ratio": float(np.mean(valid_ic > 0)) if len(valid_ic) else float("nan"),
        "rank_icir": rank_icir,
        "n_valid_ic_dates": int(len(valid_ic)),
        "direction_accuracy": float(metrics["pooled_da"]),
        "pred_std": float(metrics["pooled_pred_std"]),
        "true_std": float(metrics["pooled_true_std"]),
        "pred_std_true_std_ratio": float(metrics["pooled_pred_std_over_true_std"]),
    }

    predictions = pd.DataFrame({
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": SEED,
        "selected_daily_patch": daily_patch,
        "selected_weekly_patch": weekly_patch,
        "ticker": ticker,
        "anchor_date": anchor_date,
        "y_true": y_true.astype(np.float32),
        "y_pred": pred_np.astype(np.float32),
    })

    weights = pd.DataFrame({
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": SEED,
        "ticker": ticker,
        "anchor_date": anchor_date,
    })
    for index, column in enumerate(DAILY_WEIGHT_COLUMNS):
        weights[column] = daily_np[:, index]
    for index, column in enumerate(WEEKLY_WEIGHT_COLUMNS):
        weights[column] = weekly_np[:, index]

    return {"summary": summary, "predictions": predictions, "rank_ic": rank_ic_df, "scale_weights": weights}


def _run_evaluate_test() -> None:
    _ensure_provenance()
    cells = _all_training_cells()
    for cell in cells:
        if _cell_status(cell) != "complete":
            raise RuntimeError(f"Cannot evaluate test: incomplete training cell {cell}.")

    if not SINGLE_SELECTION_PATH.exists():
        raise RuntimeError("Single selection is missing; run --select-single before --evaluate-test.")
    selection = pd.read_csv(SINGLE_SELECTION_PATH)
    if len(selection) != len(HORIZON_LABELS):
        raise RuntimeError(f"Single selection must contain exactly {len(HORIZON_LABELS)} rows.")
    if set(selection["horizon"].astype(str)) != set(HORIZON_LABELS):
        raise RuntimeError("Single selection must cover exactly one row per horizon.")
    if not (selection["selection_metric"].astype(str) == SELECTION_METRIC).all():
        raise RuntimeError("Every single selection row must use selection_metric=validation_mse.")
    if not (selection["tie_break_rule"].astype(str) == TIE_BREAK_RULE).all():
        raise RuntimeError("Every single selection row must record the frozen tie-break rule.")

    # Reproduce the selection from current validation evidence and fail closed on any drift.
    if str(selection["training_summary_sha256"].iloc[0]) != _sha256(TRAINING_SUMMARY_PATH):
        raise RuntimeError("Stored training_summary_sha256 does not match the current training summary artifact.")
    if str(selection["single_validation_predictions_sha256"].iloc[0]) != _sha256(SINGLE_VAL_PRED_PATH):
        raise RuntimeError("Stored single_validation_predictions_sha256 does not match the current validation artifact.")
    reproduced = _reproduce_single_selection()
    stored_sorted = selection.sort_values("horizon").reset_index(drop=True)
    reproduced_sorted = reproduced.sort_values("horizon").reset_index(drop=True)
    for field in (
        "horizon",
        "selected_daily_patch",
        "selected_weekly_patch",
        "selection_metric",
        "tie_break_rule",
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
    ):
        if not (stored_sorted[field].astype(str) == reproduced_sorted[field].astype(str)).all():
            raise RuntimeError(f"Reproduced single selection differs from stored selection for {field}.")
    if not np.allclose(
        stored_sorted["validation_mse"].to_numpy(dtype=np.float64),
        reproduced_sorted["validation_mse"].to_numpy(dtype=np.float64),
        rtol=VALIDATION_MSE_RTOL,
        atol=VALIDATION_MSE_ATOL,
    ):
        raise RuntimeError("Reproduced single selection validation_mse differs from stored selection.")

    existing = [p.name for p in TEST_ARTIFACT_PATHS if p.exists()]
    if existing:
        raise RuntimeError(f"Test evidence already exists; refusing to overwrite: {existing}")

    summary_df = _read_csv_cached(TRAINING_SUMMARY_PATH)
    ds = _load_dataset()

    test_targets: list[tuple[str, str, int | None, int | None]] = []
    for _, row in selection.iterrows():
        test_targets.append((
            "single",
            str(row["horizon"]),
            int(row["selected_daily_patch"]),
            int(row["selected_weekly_patch"]),
        ))
    for mechanism in ("fixed", "static"):
        for horizon in HORIZON_LABELS:
            test_targets.append((mechanism, horizon, None, None))
    if len(test_targets) != EXPECTED_TEST_CELLS:
        raise RuntimeError(f"Test grid must contain exactly {EXPECTED_TEST_CELLS} cells; built {len(test_targets)}.")

    summaries, predictions, rank_ics, weights = [], [], [], []
    for mechanism, horizon, daily_patch, weekly_patch in test_targets:
        mask = _cell_mask(summary_df, mechanism, horizon, daily_patch, weekly_patch)
        matched = summary_df[mask]
        if len(matched) != 1:
            raise RuntimeError(f"Expected exactly one training summary row for {mechanism}/{horizon}; found {len(matched)}.")
        result = evaluate_frozen_checkpoint_on_test(
            mechanism, horizon, daily_patch, weekly_patch, matched.iloc[0], ds
        )
        summaries.append(result["summary"])
        predictions.append(result["predictions"])
        rank_ics.append(result["rank_ic"])
        weights.append(result["scale_weights"])

    summary_out = pd.DataFrame(summaries)
    predictions_out = pd.concat(predictions, ignore_index=True)
    rank_ic_out = pd.concat(rank_ics, ignore_index=True)
    weights_out = pd.concat(weights, ignore_index=True)

    if len(summary_out) != EXPECTED_TEST_CELLS:
        raise RuntimeError(f"Test summary must contain {EXPECTED_TEST_CELLS} rows.")
    if len(predictions_out) != EXPECTED_TEST_PREDICTION_ROWS:
        raise RuntimeError(f"Test predictions must contain {EXPECTED_TEST_PREDICTION_ROWS} rows.")
    if len(rank_ic_out) != EXPECTED_TEST_RANK_IC_ROWS:
        raise RuntimeError(f"Test Rank-IC must contain {EXPECTED_TEST_RANK_IC_ROWS} rows.")
    if len(weights_out) != EXPECTED_TEST_PREDICTION_ROWS:
        raise RuntimeError(f"Test scale weights must contain {EXPECTED_TEST_PREDICTION_ROWS} rows.")

    _atomic_write_csv(summary_out, TEST_SUMMARY_PATH)
    _atomic_write_csv(predictions_out, TEST_PREDICTIONS_PATH)
    _atomic_write_csv(rank_ic_out, TEST_RANK_IC_PATH)
    _atomic_write_csv(weights_out, TEST_SCALE_WEIGHTS_PATH)
    print(f"EXPERIMENT 3 TEST EVALUATION COMPLETE: {EXPECTED_TEST_CELLS} cells")


# ----------------------------------------------------------------------------
# Smoke validation (writes nothing)
# ----------------------------------------------------------------------------


def _snapshot_namespace(directory: Path) -> dict:
    if not directory.exists():
        return {"exists": False, "children": {}}
    return {
        "exists": True,
        "children": {c.name: (_sha256(c) if c.is_file() else None) for c in directory.iterdir()},
    }


def _assert_namespace_unchanged(before: dict, after: dict, label: str) -> None:
    if before != after:
        raise RuntimeError(f"Smoke validation must leave the {label} namespace unchanged.")


def _run_smoke_validation() -> None:
    experiment_before = _snapshot_namespace(EXPERIMENT_DIR)
    main_v2_before = _snapshot_namespace(formal.FORMAL_OUTPUT_DIR)

    device = _cpu_device()
    formal.set_seed(SEED)
    batch = 2
    x_daily = torch.randn(batch, 90, 5)
    x_weekly = torch.randn(batch, 26, 5)
    y = torch.randn(batch)
    loss_fn = nn.HuberLoss(delta=formal.HUBER_DELTA)

    def _forward_backward(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred, daily_w, weekly_w = model.forward_with_router(x_daily.to(device), x_weekly.to(device))
        assert pred.shape == y.shape, (pred.shape, y.shape)
        assert torch.isfinite(pred).all()
        loss = loss_fn(pred, y)
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient: {name}"
        return pred, daily_w, weekly_w

    # A/B. Single: every allowed Daily x Weekly combination instantiates with one expert per branch.
    for daily_patch in DAILY_PATCH_SIZES:
        for weekly_patch in WEEKLY_PATCH_SIZES:
            model = _make_experiment3_model("single", daily_patch, weekly_patch)
            assert len(model.daily_branch.experts) == 1, len(model.daily_branch.experts)
            assert len(model.weekly_branch.experts) == 1, len(model.weekly_branch.experts)
            assert model.daily_branch.experts[0].patch_size == daily_patch
            assert model.weekly_branch.experts[0].patch_size == weekly_patch
            assert formal.count_trainable_parameters(model) == EXPECTED_PARAMETER_COUNTS["single"]
            _forward_backward(model)

    # C. Fixed: four experts per branch and exact equal weighting.
    fixed_model = _make_experiment3_model("fixed")
    assert len(fixed_model.daily_branch.experts) == len(DAILY_PATCH_SIZES)
    assert len(fixed_model.weekly_branch.experts) == len(WEEKLY_PATCH_SIZES)
    assert formal.count_trainable_parameters(fixed_model) == EXPECTED_PARAMETER_COUNTS["fixed"]
    _pred, fixed_daily_w, fixed_weekly_w = _forward_backward(fixed_model)
    assert torch.allclose(fixed_daily_w, torch.full_like(fixed_daily_w, 0.25), atol=1e-9, rtol=0.0)
    assert torch.allclose(fixed_weekly_w, torch.full_like(fixed_weekly_w, 0.25), atol=1e-9, rtol=0.0)
    assert torch.allclose(fixed_daily_w.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0.0)
    assert torch.allclose(fixed_weekly_w.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0.0)

    # D. Static: learned global weights, finite, non-negative, sum 1, sample-independent.
    static_model = _make_experiment3_model("static")
    assert len(static_model.daily_branch.experts) == len(DAILY_PATCH_SIZES)
    assert len(static_model.weekly_branch.experts) == len(WEEKLY_PATCH_SIZES)
    assert formal.count_trainable_parameters(static_model) == EXPECTED_PARAMETER_COUNTS["static"]
    _pred, static_daily_w, static_weekly_w = _forward_backward(static_model)
    assert torch.isfinite(static_daily_w).all() and torch.isfinite(static_weekly_w).all()
    assert (static_daily_w >= 0).all() and (static_weekly_w >= 0).all()
    assert torch.allclose(static_daily_w.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0.0)
    assert torch.allclose(static_weekly_w.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0.0)
    # Zero-initialized logits imply uniform initial weights under the frozen implementation.
    assert torch.allclose(static_daily_w, torch.full_like(static_daily_w, 0.25), atol=1e-6, rtol=0.0)
    with torch.no_grad():
        perturbed_daily = x_daily.clone()
        perturbed_daily[1] = perturbed_daily[1] * 0.0 + 99.0
        _p2, static_daily_w2, static_weekly_w2 = static_model.forward_with_router(perturbed_daily, x_weekly)
    assert torch.equal(static_daily_w, static_daily_w2), "Static weights must be sample-independent."
    assert torch.equal(static_weekly_w, static_weekly_w2), "Static weights must be sample-independent."

    # E. Adaptive is a structural reference only; it is never trained here.
    adaptive_reference = formal._make_model(FREQUENCY_CONFIGURATION, ADAPTIVE_MECHANISM)
    assert formal.count_trainable_parameters(adaptive_reference) == EXPECTED_ADAPTIVE_PARAMETER_COUNT
    for bad in (ADAPTIVE_MECHANISM, "ADAPTIVE"):
        try:
            _validate_mechanism(bad)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Adaptive must be rejected by every Experiment 3 training path.")

    # F. Grid contract.
    assert len(_all_training_cells()) == EXPECTED_TRAINING_CELLS
    assert EXPECTED_SINGLE_VALIDATION_ROWS == 301920
    assert EXPECTED_TEST_PREDICTION_ROWS == 59823
    assert EXPECTED_TEST_RANK_IC_ROWS == 3519

    # G. Synthetic validation-only selection and deterministic tie-break.
    synthetic_rows = []
    for horizon in HORIZON_LABELS:
        for daily_patch in DAILY_PATCH_SIZES:
            for weekly_patch in WEEKLY_PATCH_SIZES:
                synthetic_rows.append({
                    "mechanism": "single",
                    "horizon": horizon,
                    "daily_patch": daily_patch,
                    "weekly_patch": weekly_patch,
                    "validation_mse": 1.0,
                    "checkpoint_path": "x",
                    "checkpoint_sha256": "y",
                })
    synthetic = pd.DataFrame(synthetic_rows)
    # Unique minimum wins.
    unique_min = synthetic.copy()
    target = (unique_min["horizon"] == "5d") & (unique_min["daily_patch"] == 20) & (unique_min["weekly_patch"] == 8)
    unique_min.loc[target, "validation_mse"] = 0.5
    picked = select_single_by_validation(unique_min)
    row_5d = picked[picked["horizon"] == "5d"].iloc[0]
    assert int(row_5d["selected_daily_patch"]) == 20 and int(row_5d["selected_weekly_patch"]) == 8
    # Exact tie resolves to the smaller Daily patch, then the smaller Weekly patch.
    tied = synthetic.copy()
    tie_mask = (tied["horizon"] == "10d") & (tied["daily_patch"].isin([10, 30])) & (tied["weekly_patch"].isin([4, 13]))
    tied.loc[tie_mask, "validation_mse"] = 0.25
    picked_tie = select_single_by_validation(tied)
    row_10d = picked_tie[picked_tie["horizon"] == "10d"].iloc[0]
    assert int(row_10d["selected_daily_patch"]) == 10, row_10d["selected_daily_patch"]
    assert int(row_10d["selected_weekly_patch"]) == 4, row_10d["selected_weekly_patch"]
    assert str(row_10d["selection_metric"]) == SELECTION_METRIC
    try:
        select_single_by_validation(synthetic.assign(test_mse=0.0))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Single selection must reject any test column.")

    # H. Namespaces untouched.
    _assert_namespace_unchanged(experiment_before, _snapshot_namespace(EXPERIMENT_DIR), "Experiment 3")
    _assert_namespace_unchanged(main_v2_before, _snapshot_namespace(formal.FORMAL_OUTPUT_DIR), "MAIN V2")

    print("EXPERIMENT 3 PATHFORMER ABLATION SMOKE VALIDATION PASSED")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3 PathFormer multi-scale mechanism ablation (daily_weekly, CPU, seed 42)."
    )
    parser.add_argument("--train-cell", action="store_true", help="Train exactly one mechanism/horizon cell.")
    parser.add_argument("--train-all", action="store_true", help="Train all 54 Experiment 3 cells.")
    parser.add_argument("--resume", action="store_true", help="Skip only fully validated completed cells.")
    parser.add_argument("--select-single", action="store_true", help="Select 3 single models using validation MSE only.")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the 9 frozen cells on TEST. No training.")
    parser.add_argument("--mechanism", choices=list(TRAINABLE_MECHANISMS), default=None)
    parser.add_argument("--horizon", choices=list(HORIZON_LABELS), default=None)
    parser.add_argument("--daily-patch", type=int, choices=list(DAILY_PATCH_SIZES), default=None)
    parser.add_argument("--weekly-patch", type=int, choices=list(WEEKLY_PATCH_SIZES), default=None)
    args = parser.parse_args()

    modes = [args.train_cell, args.train_all, args.select_single, args.evaluate_test]
    if sum(bool(m) for m in modes) > 1:
        parser.error("Choose exactly one of --train-cell / --train-all / --select-single / --evaluate-test.")
    if args.resume and not args.train_all:
        parser.error("--resume is only valid with --train-all.")

    if args.train_cell:
        if args.mechanism is None or args.horizon is None:
            parser.error("--train-cell requires --mechanism and --horizon.")
        if args.mechanism == "single":
            if args.daily_patch is None or args.weekly_patch is None:
                parser.error("single requires --daily-patch and --weekly-patch.")
        else:
            if args.daily_patch is not None or args.weekly_patch is not None:
                parser.error(f"{args.mechanism} uses all four scales per frequency; patch arguments are not permitted.")
    else:
        if any(v is not None for v in (args.mechanism, args.horizon, args.daily_patch, args.weekly_patch)):
            parser.error("Cell arguments are only valid with --train-cell.")
    return args


def main() -> None:
    args = parse_args()
    if args.train_cell:
        _run_train_cell(args.mechanism, args.horizon, args.daily_patch, args.weekly_patch)
        return
    if args.train_all:
        _run_train_all(resume=args.resume)
        return
    if args.select_single:
        _run_select_single()
        return
    if args.evaluate_test:
        _run_evaluate_test()
        return
    _run_smoke_validation()


if __name__ == "__main__":
    main()
