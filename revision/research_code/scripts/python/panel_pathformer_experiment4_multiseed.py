"""Experiment 4 — PathFormer multi-seed robustness runner (24 NEW cells only).

Scientific contract
-------------------
Pre-frozen primary contrast (see docs/experiment4_multiseed_protocol.md):

    Adaptive vs Static
    x frequency_configuration = daily_weekly
    x horizons 5d / 10d / 20d
    x NEW seeds 0 / 1 / 21 / 3407

    = 24 new training cells

Seed 42 is frozen external evidence:
  - Adaptive seed 42 = frozen MAIN V2 PathFormer daily_weekly evidence
  - Static  seed 42 = frozen Experiment 3 Static daily_weekly evidence

Seed 42 is NEVER trained, evaluated, written, or combined here. This runner is
responsible only for the 24 new cells. The 5-seed robustness combination is a
separate later finalizer that runs after the 24-cell raw evidence audit.

Frozen sources (read-only)
--------------------------
Architecture and training contract come from scripts/python/panel_pathformer_main_v2.py.
This module imports it read-only: it never edits it, never mutates its globals
(including formal.SEED), never monkeypatches it, never calls its formal
CLI/finalization paths, and never writes into its namespace. The Experiment 3
runner is bound only by source hash for provenance.

Seed handling (important)
-------------------------
formal.train_one_cell(...) and the Experiment 3 train_and_validate_cell(...) are
bound to the historical formal seed-42 execution path and are therefore NOT
called here. This module reproduces that frozen training loop while accepting an
explicit seed, using formal.set_seed(seed) without mutating any frozen global.

Frozen operation order per cell: formal.set_seed(seed), then loader construction
through the frozen helper, then mechanism model construction, then Train
DataLoader iteration.

The frozen Train DataLoader uses shuffle=True with no explicit generator. Each
Experiment 4 cell calls formal.set_seed(seed) before loader and model
construction. The Train DataLoader therefore draws its shuffle trajectory from
PyTorch's global RNG when the loader iterator is consumed.

Because model construction occurs before Train DataLoader iteration, the
pre-specified cell seed affects model initialization, dropout sampling, and the
subsequent global-RNG-driven Train DataLoader shuffle trajectory.

Adaptive and Static have different module structures and consume the global RNG
differently during construction. Therefore matching the seed label does not
guarantee identical minibatch ordering or a common-random-number optimization
realization across mechanisms. The comparison is paired by the pre-specified
seed label only.

Test-leakage rule
-----------------
Training uses TRAIN only and early-stops on VALIDATION only; the TEST loader is
made unreachable in the training path. TEST evaluation is a separate explicit
CLI mode that may run only after all 24 training cells are complete, and each
cell is evaluated on TEST exactly once.

This module writes nothing when invoked with no arguments.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.python.panel_pathformer_main_v2 as formal
from scripts.python.panel_common import (
    evaluate_predictions,
    load_frozen_panel,
    rank_ic_by_date,
)

# ----------------------------------------------------------------------------
# Frozen Experiment 4 contract
# ----------------------------------------------------------------------------

RUN_FAMILY = "EXPERIMENT4_PATHFORMER_MULTISEED"
TEMPORAL_STATUS = "POST_TEMPORAL_FIX"
DATASET_CONTRACT_VERSION = 2
FREQUENCY_CONFIGURATION = "daily_weekly"

MECHANISMS = ("adaptive", "static")
FORBIDDEN_MECHANISMS = ("single", "fixed")

HORIZON_LABELS = ("5d", "10d", "20d")
HORIZON_MAP = {"5d": 5, "10d": 10, "20d": 20}

NEW_SEEDS = (0, 1, 21, 3407)
FROZEN_EXTERNAL_SEED = 42
ALL_PROTOCOL_SEEDS = (0, 1, 21, 42, 3407)
SEED42_POLICY = "FROZEN_EXTERNAL_REUSE_NOT_RETRAINED"

EXECUTION_BACKEND = "CPU"

N_TICKERS = 17
N_TRAIN_SAMPLES = 30634
N_VALIDATION_SAMPLES = 6290
N_TEST_SAMPLES = 6647
N_TEST_DATES = 391

EXPECTED_TRAINING_CELLS = len(NEW_SEEDS) * len(MECHANISMS) * len(HORIZON_LABELS)  # 24
EXPECTED_TEST_CELLS = EXPECTED_TRAINING_CELLS  # 24
EXPECTED_TEST_PREDICTION_ROWS = EXPECTED_TEST_CELLS * N_TEST_SAMPLES  # 159528
EXPECTED_TEST_RANK_IC_ROWS = EXPECTED_TEST_CELLS * N_TEST_DATES  # 9384
EXPECTED_TEST_SCALE_WEIGHT_ROWS = EXPECTED_TEST_PREDICTION_ROWS  # 159528

EXPECTED_PARAMETER_COUNTS = {"adaptive": 116297, "static": 107785}

DAILY_PATCH_SIZES = tuple(formal.DAILY_PATCH_SIZES)
WEEKLY_PATCH_SIZES = tuple(formal.WEEKLY_PATCH_SIZES)
EXPECTED_DAILY_PATCH_SIZES = (5, 10, 20, 30)
EXPECTED_WEEKLY_PATCH_SIZES = (2, 4, 8, 13)

DAILY_WEIGHT_COLUMNS = tuple(f"daily_w_{p}" for p in EXPECTED_DAILY_PATCH_SIZES)
WEEKLY_WEIGHT_COLUMNS = tuple(f"weekly_w_{p}" for p in EXPECTED_WEEKLY_PATCH_SIZES)

# Frozen optimization / architecture constants that must hold in the imported source.
EXPECTED_FROZEN_CONSTANTS = {
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "HUBER_DELTA": 1.0,
    "MAX_EPOCHS": 100,
    "EARLY_STOPPING_PATIENCE": 10,
    "GRADIENT_CLIP_MAX_NORM": 1.0,
    "INPUT_DIM": 5,
    "D_MODEL": 32,
    "NHEAD": 4,
    "DIM_FEEDFORWARD": 64,
    "DROPOUT": 0.1,
    "ROUTER_HIDDEN": 32,
    # Frozen MAIN V2 formal seed constant; imported read-only and never mutated.
    "SEED": 42,
}
EARLY_STOPPING_IMPROVEMENT_THRESHOLD = 1e-12

# ----------------------------------------------------------------------------
# Frozen dependency binding
# ----------------------------------------------------------------------------

DATASET_MANIFEST_PATH = formal.DATASET_MANIFEST_PATH
EXPECTED_DATASET_MANIFEST_SHA256 = "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c"

FROZEN_PATHFORMER_SCRIPT_RELATIVE = "scripts/python/panel_pathformer_main_v2.py"
EXPECTED_FROZEN_PATHFORMER_SHA256 = "5f05ccd677359a899ca91c3baeeda3ecb6aae1a63ff75593cd31809ae27e6f77"

FROZEN_EXPERIMENT3_SCRIPT_RELATIVE = "scripts/python/panel_pathformer_experiment3.py"
EXPECTED_FROZEN_EXPERIMENT3_SHA256 = "6f3c1f804c9610fef6f29453cc9a49b121f927fbec90e3482e2744a924aa0142"

PANEL_COMMON_RELATIVE = "scripts/python/panel_common.py"
EXPECTED_PANEL_COMMON_SHA256 = "ccaba201b910b66fb5c3272bcc3aa18e04d12c098bb26100a73a3717ec6d2834"

TRAINING_SCRIPT_RELATIVE = "scripts/python/panel_pathformer_experiment4_multiseed.py"

# ----------------------------------------------------------------------------
# Frozen seed-42 evidence (READ-ONLY; never written, never combined here)
# ----------------------------------------------------------------------------

FROZEN_ADAPTIVE_SEED42_DIR = ROOT / "dataset" / "audit" / "main_v2" / "pathformer"
FROZEN_STATIC_SEED42_DIR = ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation"

# Frozen seed-42 daily_weekly-only expectations (3 horizons x 6647 TEST samples / 391 dates).
N_FROZEN_SEED42_HORIZONS = len(HORIZON_LABELS)
N_FROZEN_SEED42_PREDICTION_ROWS = N_FROZEN_SEED42_HORIZONS * N_TEST_SAMPLES  # 19941
N_FROZEN_SEED42_RANK_IC_ROWS = N_FROZEN_SEED42_HORIZONS * N_TEST_DATES  # 1173

# Frozen Experiment 3 cross-mechanism y_true identity standard: the float32 CSV
# serialization round-trip permits no relative slack and at most one float32 ULP.
FROZEN_FLOAT32_RTOL = 0.0
FROZEN_FLOAT32_ATOL = float(np.finfo(np.float32).eps)

FROZEN_SEED42_EVIDENCE = {
    "adaptive_seed42_summary": (
        FROZEN_ADAPTIVE_SEED42_DIR / "panel_pathformer_summary_metrics.csv",
        "90e987cd87d9bdbc82c6bf587627863027878a6d170f7a6cca80569ee39de5e3",
    ),
    "adaptive_seed42_predictions": (
        FROZEN_ADAPTIVE_SEED42_DIR / "panel_pathformer_test_predictions.csv",
        "a05955a561da1b8880f1bb0878a6576852c087cc45707546d446ce96b0fabdaf",
    ),
    "adaptive_seed42_rank_ic": (
        FROZEN_ADAPTIVE_SEED42_DIR / "panel_pathformer_rank_ic_by_date.csv",
        "52b3f3475dc4db74523cfe99a09f71f2b33bd14c58b458ba2e62564ca7283298",
    ),
    "adaptive_seed42_router": (
        FROZEN_ADAPTIVE_SEED42_DIR / "panel_pathformer_router_weights.csv",
        "bcf06150f21d35d9ec10132d248d088cc758c9782771efdf9b68849531c9814e",
    ),
    "static_seed42_summary": (
        FROZEN_STATIC_SEED42_DIR / "experiment3_test_summary_metrics.csv",
        "0ada49805bf476b873e9ec2d77ea72b29ae67ecede7b342fd0dd49e9df85005b",
    ),
    "static_seed42_predictions": (
        FROZEN_STATIC_SEED42_DIR / "experiment3_test_predictions.csv",
        "95d1b06cc3ce0209e99833c29a1c3672a6d7a1a9d46d32f16ef1e48fa9846515",
    ),
    "static_seed42_rank_ic": (
        FROZEN_STATIC_SEED42_DIR / "experiment3_rank_ic_by_date.csv",
        "5e9dd5cb1da838f9f984ec424441bd1bd30663f6f70a971b55d7a6404e68d43e",
    ),
    "static_seed42_scale_weights": (
        FROZEN_STATIC_SEED42_DIR / "experiment3_scale_weights.csv",
        "219e908f0c3f75bd9b38d7fe466242a15a64d8c41b1f235ae75c91df5f6eb248",
    ),
}

# ----------------------------------------------------------------------------
# Output namespace (Experiment 4 only)
# ----------------------------------------------------------------------------

EXPERIMENT_DIR = ROOT / "dataset" / "audit" / "experiment4_multiseed"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

PROVENANCE_PATH = EXPERIMENT_DIR / "experiment4_training_provenance.csv"
TRAINING_SUMMARY_PATH = EXPERIMENT_DIR / "experiment4_training_summary.csv"
TRAINING_HISTORY_PATH = EXPERIMENT_DIR / "experiment4_training_history.csv"
TEST_ATTEMPT_LEDGER_PATH = EXPERIMENT_DIR / "experiment4_test_attempt_ledger.csv"
TEST_SUMMARY_PATH = EXPERIMENT_DIR / "experiment4_test_summary_metrics.csv"
TEST_PREDICTIONS_PATH = EXPERIMENT_DIR / "experiment4_test_predictions.csv"
TEST_RANK_IC_PATH = EXPERIMENT_DIR / "experiment4_rank_ic_by_date.csv"
TEST_SCALE_WEIGHTS_PATH = EXPERIMENT_DIR / "experiment4_scale_weights.csv"

TRAINING_ARTIFACT_PATHS = (TRAINING_SUMMARY_PATH, TRAINING_HISTORY_PATH)
TEST_OUTPUT_PATHS = (
    TEST_SUMMARY_PATH,
    TEST_PREDICTIONS_PATH,
    TEST_RANK_IC_PATH,
    TEST_SCALE_WEIGHTS_PATH,
)
TEST_ARTIFACT_PATHS = (TEST_ATTEMPT_LEDGER_PATH,) + TEST_OUTPUT_PATHS

CELL_KEY = ["mechanism", "horizon", "seed"]


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


def _assert_experiment4_path(path: Path) -> Path:
    """Every formal Experiment 4 write must resolve under the Experiment 4 namespace."""
    resolved = Path(path).resolve()
    experiment_root = EXPERIMENT_DIR.resolve()
    if resolved != experiment_root and experiment_root not in resolved.parents:
        raise RuntimeError(f"Refusing to write outside the Experiment 4 namespace: {resolved}")
    return resolved


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    _assert_experiment4_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _append_csv(df: pd.DataFrame, path: Path, duplicate_subset: list[str]) -> None:
    _assert_experiment4_path(path)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()
    if duplicate_subset and combined.duplicated(subset=duplicate_subset).any():
        raise RuntimeError(f"Refusing to append duplicate keys to {path.name}: {duplicate_subset}")
    _atomic_write_csv(combined, path)


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


# ----------------------------------------------------------------------------
# Grid validation (mechanism / horizon / seed)
# ----------------------------------------------------------------------------


def _reject_seed42(seed) -> None:
    if int(seed) == FROZEN_EXTERNAL_SEED:
        raise RuntimeError(
            "Seed 42 is frozen external evidence and has no Experiment 4 training or evaluation path. "
            "Adaptive seed 42 comes from the frozen MAIN V2 daily_weekly evidence and Static seed 42 from "
            "the frozen Experiment 3 evidence; both are read-only and are combined only by the later "
            "multi-seed finalizer."
        )


def _validate_seed(seed) -> int:
    try:
        seed_value = int(seed)
    except (TypeError, ValueError):
        raise ValueError(f"Seed must be an integer; got {seed!r}")
    _reject_seed42(seed_value)
    if seed_value not in NEW_SEEDS:
        raise ValueError(f"Seed must be one of the pre-specified new seeds {NEW_SEEDS}; got {seed_value}")
    return seed_value


def _validate_mechanism(mechanism: str) -> str:
    name = str(mechanism)
    if name in FORBIDDEN_MECHANISMS:
        raise RuntimeError(
            f"Mechanism '{name}' is not part of the Experiment 4 primary contract; only {MECHANISMS} are permitted."
        )
    if name not in MECHANISMS:
        raise ValueError(f"Unsupported Experiment 4 mechanism: {name}")
    return name


def _validate_horizon_label(horizon: str) -> str:
    label = str(horizon)
    if label not in HORIZON_MAP:
        raise ValueError(f"Unsupported horizon label: {label}")
    return label


def _all_training_cells() -> list[dict]:
    cells = [
        {"mechanism": mechanism, "horizon": horizon, "seed": int(seed)}
        for mechanism in MECHANISMS
        for horizon in HORIZON_LABELS
        for seed in NEW_SEEDS
    ]
    if len(cells) != EXPECTED_TRAINING_CELLS:
        raise RuntimeError(f"Training grid must contain exactly {EXPECTED_TRAINING_CELLS} cells; built {len(cells)}.")
    if len({(c["mechanism"], c["horizon"], c["seed"]) for c in cells}) != EXPECTED_TRAINING_CELLS:
        raise RuntimeError("Training grid contains duplicate mechanism/horizon/seed cells.")
    if any(int(c["seed"]) == FROZEN_EXTERNAL_SEED for c in cells):
        raise RuntimeError("Seed 42 must never appear in the Experiment 4 training grid.")
    return cells


def _cell_label(cell: dict) -> str:
    return f"{cell['mechanism']}/seed{int(cell['seed'])}/{cell['horizon']}"


def _checkpoint_path(mechanism: str, horizon: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"{mechanism}_seed{int(seed)}_{horizon}.pt"


def _cell_mask(df: pd.DataFrame, mechanism: str, horizon: str, seed: int) -> pd.Series:
    return (
        (df["mechanism"].astype(str) == str(mechanism))
        & (df["horizon"].astype(str) == str(horizon))
        & (df["seed"].astype(int) == int(seed))
    )


def _assert_seed42_absent_from_namespace() -> None:
    """Seed 42 evidence must never exist inside the Experiment 4 namespace."""
    if CHECKPOINT_DIR.exists():
        offending = [p.name for p in CHECKPOINT_DIR.iterdir() if f"seed{FROZEN_EXTERNAL_SEED}" in p.name]
        if offending:
            raise RuntimeError(f"Seed-42 checkpoints must not exist in the Experiment 4 namespace: {offending}")
    for path in TRAINING_ARTIFACT_PATHS + TEST_ARTIFACT_PATHS:
        if not path.exists():
            continue
        df = _read_csv_cached(path)
        if "seed" in df.columns and (df["seed"].astype(int) == FROZEN_EXTERNAL_SEED).any():
            raise RuntimeError(f"Seed-42 rows must not exist in Experiment 4 evidence: {path.name}")


# ----------------------------------------------------------------------------
# Model construction (frozen architecture, reused not reimplemented)
# ----------------------------------------------------------------------------


def _make_experiment4_model(mechanism: str) -> nn.Module:
    _validate_mechanism(mechanism)
    model = formal._make_model(FREQUENCY_CONFIGURATION, mechanism)
    actual = formal.count_trainable_parameters(model)
    expected = EXPECTED_PARAMETER_COUNTS[mechanism]
    if actual != expected:
        raise RuntimeError(f"Parameter count mismatch for {mechanism}: expected {expected}, got {actual}")
    return model


# ----------------------------------------------------------------------------
# Frozen dependency / constant / provenance governance
# ----------------------------------------------------------------------------


def _current_dependency_hashes() -> dict:
    return {
        "training_script_sha256": _sha256(ROOT / TRAINING_SCRIPT_RELATIVE),
        "dataset_manifest_sha256": _sha256(DATASET_MANIFEST_PATH),
        "frozen_pathformer_script_sha256": _sha256(ROOT / FROZEN_PATHFORMER_SCRIPT_RELATIVE),
        "frozen_experiment3_script_sha256": _sha256(ROOT / FROZEN_EXPERIMENT3_SCRIPT_RELATIVE),
        "panel_common_sha256": _sha256(ROOT / PANEL_COMMON_RELATIVE),
    }


def _verify_frozen_dependencies() -> dict:
    hashes = _current_dependency_hashes()
    expected = {
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "frozen_pathformer_script_sha256": EXPECTED_FROZEN_PATHFORMER_SHA256,
        "frozen_experiment3_script_sha256": EXPECTED_FROZEN_EXPERIMENT3_SHA256,
        "panel_common_sha256": EXPECTED_PANEL_COMMON_SHA256,
    }
    for field, expected_value in expected.items():
        if hashes[field] != expected_value:
            raise RuntimeError(
                f"Frozen dependency SHA mismatch for {field}: expected {expected_value}, got {hashes[field]}"
            )
    return hashes


def _verify_frozen_seed42_evidence() -> dict:
    """Read-only verification of the frozen seed-42 artifacts. Never rewrites them."""
    hashes: dict[str, str] = {}
    for key, (path, expected_sha) in FROZEN_SEED42_EVIDENCE.items():
        if not path.exists():
            raise RuntimeError(f"Frozen seed-42 evidence is missing: {path}")
        actual = _sha256(path)
        if actual != expected_sha:
            raise RuntimeError(
                f"Frozen seed-42 evidence SHA mismatch for {key}: expected {expected_sha}, got {actual}"
            )
        hashes[f"{key}_path"] = str(path.relative_to(ROOT))
        hashes[f"{key}_sha256"] = actual
    return hashes


def _assert_float32_y_true_identity(left: np.ndarray, right: np.ndarray, label: str) -> None:
    """Exact float32 identity, falling back only to the frozen Experiment 3 float32
    CSV-serialization tolerance (rtol=0, atol=one float32 ULP)."""
    if np.array_equal(left, right):
        return
    if np.allclose(left, right, rtol=FROZEN_FLOAT32_RTOL, atol=FROZEN_FLOAT32_ATOL, equal_nan=False):
        return
    raise RuntimeError(f"{label}: y_true vectors are not identical across the compared evidence.")


def _frozen_adaptive_seed42_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Frozen MAIN V2 rows restricted to daily_weekly / seed 42 / horizons 5, 10, 20."""
    horizon_values = [HORIZON_MAP[h] for h in HORIZON_LABELS]
    subset = df[
        (df["frequency_setting"].astype(str) == FREQUENCY_CONFIGURATION)
        & (df["seed"].astype(int) == FROZEN_EXTERNAL_SEED)
        & (df["horizon"].astype(int).isin(horizon_values))
    ].copy()
    reverse_map = {value: label for label, value in HORIZON_MAP.items()}
    subset["horizon_label"] = subset["horizon"].astype(int).map(reverse_map)
    return subset


def _frozen_static_seed42_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Frozen Experiment 3 rows restricted to static / seed 42 / horizons 5d, 10d, 20d."""
    subset = df[
        (df["mechanism"].astype(str) == "static")
        & (df["seed"].astype(int) == FROZEN_EXTERNAL_SEED)
        & (df["horizon"].astype(str).isin(HORIZON_LABELS))
    ].copy()
    subset["horizon_label"] = subset["horizon"].astype(str)
    return subset


def _sorted_prediction_keys(subset: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = subset[["ticker", "anchor_date", "y_true"]].copy()
    frame["anchor_date"] = frame["anchor_date"].apply(_canonical_date)
    if frame.duplicated(subset=["ticker", "anchor_date"]).any():
        raise RuntimeError(f"{label}: duplicate ticker/anchor_date prediction keys.")
    if len(frame) != N_TEST_SAMPLES:
        raise RuntimeError(f"{label}: expected {N_TEST_SAMPLES} prediction rows, found {len(frame)}.")
    if frame["ticker"].nunique() != N_TICKERS:
        raise RuntimeError(f"{label}: expected {N_TICKERS} tickers, found {frame['ticker'].nunique()}.")
    if frame["anchor_date"].nunique() != N_TEST_DATES:
        raise RuntimeError(f"{label}: expected {N_TEST_DATES} TEST dates, found {frame['anchor_date'].nunique()}.")
    return frame.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)


def _sorted_weight_keys(subset: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = subset[["ticker", "anchor_date"]].copy()
    frame["anchor_date"] = frame["anchor_date"].apply(_canonical_date)
    if frame.duplicated(subset=["ticker", "anchor_date"]).any():
        raise RuntimeError(f"{label}: duplicate ticker/anchor_date weight keys.")
    if len(frame) != N_TEST_SAMPLES:
        raise RuntimeError(f"{label}: expected {N_TEST_SAMPLES} weight rows, found {len(frame)}.")
    return frame.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)


def _verify_frozen_seed42_content_identity() -> None:
    """Read-only ticker/date/y_true identity audit of the frozen seed-42 evidence.

    Reads the already hash-verified frozen artifacts and validates only; nothing is
    written, copied into the Experiment 4 namespace, or combined with new-seed evidence.
    y_pred is never compared across mechanisms.
    """
    adaptive_summary = _frozen_adaptive_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["adaptive_seed42_summary"][0]))
    adaptive_pred = _frozen_adaptive_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["adaptive_seed42_predictions"][0]))
    adaptive_rank_ic = _frozen_adaptive_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["adaptive_seed42_rank_ic"][0]))
    adaptive_router = _frozen_adaptive_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["adaptive_seed42_router"][0]))

    static_summary = _frozen_static_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["static_seed42_summary"][0]))
    static_pred = _frozen_static_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["static_seed42_predictions"][0]))
    static_rank_ic = _frozen_static_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["static_seed42_rank_ic"][0]))
    static_weights = _frozen_static_seed42_subset(pd.read_csv(FROZEN_SEED42_EVIDENCE["static_seed42_scale_weights"][0]))

    expected_counts = {
        "frozen adaptive seed42 summary": (adaptive_summary, N_FROZEN_SEED42_HORIZONS),
        "frozen adaptive seed42 predictions": (adaptive_pred, N_FROZEN_SEED42_PREDICTION_ROWS),
        "frozen adaptive seed42 rank_ic": (adaptive_rank_ic, N_FROZEN_SEED42_RANK_IC_ROWS),
        "frozen adaptive seed42 router weights": (adaptive_router, N_FROZEN_SEED42_PREDICTION_ROWS),
        "frozen static seed42 summary": (static_summary, N_FROZEN_SEED42_HORIZONS),
        "frozen static seed42 predictions": (static_pred, N_FROZEN_SEED42_PREDICTION_ROWS),
        "frozen static seed42 rank_ic": (static_rank_ic, N_FROZEN_SEED42_RANK_IC_ROWS),
        "frozen static seed42 scale weights": (static_weights, N_FROZEN_SEED42_PREDICTION_ROWS),
    }
    for label, (frame, expected) in expected_counts.items():
        if len(frame) != expected:
            raise RuntimeError(f"{label}: expected {expected} daily_weekly seed-42 rows, found {len(frame)}.")
    if set(adaptive_summary["horizon_label"]) != set(HORIZON_LABELS):
        raise RuntimeError("Frozen adaptive seed-42 summary must cover exactly horizons 5d, 10d, 20d.")
    if set(static_summary["horizon_label"]) != set(HORIZON_LABELS):
        raise RuntimeError("Frozen static seed-42 summary must cover exactly horizons 5d, 10d, 20d.")

    for horizon in HORIZON_LABELS:
        adaptive_label = f"frozen adaptive seed42 {horizon}"
        static_label = f"frozen static seed42 {horizon}"
        adaptive_keys = _sorted_prediction_keys(adaptive_pred[adaptive_pred["horizon_label"] == horizon], adaptive_label)
        static_keys = _sorted_prediction_keys(static_pred[static_pred["horizon_label"] == horizon], static_label)

        for label, keys, weight_frame in (
            (adaptive_label, adaptive_keys, adaptive_router[adaptive_router["horizon_label"] == horizon]),
            (static_label, static_keys, static_weights[static_weights["horizon_label"] == horizon]),
        ):
            weight_keys = _sorted_weight_keys(weight_frame, label)
            if not weight_keys.equals(keys[["ticker", "anchor_date"]]):
                raise RuntimeError(f"{label}: weight ticker/date keys do not match the prediction keys.")

        for label, rank_ic_frame, keys in (
            (adaptive_label, adaptive_rank_ic[adaptive_rank_ic["horizon_label"] == horizon], adaptive_keys),
            (static_label, static_rank_ic[static_rank_ic["horizon_label"] == horizon], static_keys),
        ):
            dates = rank_ic_frame["anchor_date"].apply(_canonical_date)
            if len(dates) != N_TEST_DATES or dates.nunique() != N_TEST_DATES:
                raise RuntimeError(f"{label}: RankIC must contain {N_TEST_DATES} unique TEST dates.")
            if set(dates) != set(keys["anchor_date"]):
                raise RuntimeError(f"{label}: RankIC dates do not match the prediction dates.")

        if not adaptive_keys[["ticker", "anchor_date"]].equals(static_keys[["ticker", "anchor_date"]]):
            raise RuntimeError(f"Frozen seed-42 {horizon}: adaptive and static TEST keys differ.")
        _assert_float32_y_true_identity(
            adaptive_keys["y_true"].to_numpy(dtype=np.float32),
            static_keys["y_true"].to_numpy(dtype=np.float32),
            f"Frozen seed-42 {horizon}",
        )


def _verify_frozen_training_constants() -> None:
    for name, expected_value in EXPECTED_FROZEN_CONSTANTS.items():
        actual = getattr(formal, name)
        if actual != expected_value:
            raise RuntimeError(f"Frozen constant mismatch for formal.{name}: expected {expected_value}, got {actual}")
    if tuple(formal.DAILY_PATCH_SIZES) != EXPECTED_DAILY_PATCH_SIZES:
        raise RuntimeError(f"Frozen Daily scales must be {list(EXPECTED_DAILY_PATCH_SIZES)}; got {formal.DAILY_PATCH_SIZES}")
    if tuple(formal.WEEKLY_PATCH_SIZES) != EXPECTED_WEEKLY_PATCH_SIZES:
        raise RuntimeError(f"Frozen Weekly scales must be {list(EXPECTED_WEEKLY_PATCH_SIZES)}; got {formal.WEEKLY_PATCH_SIZES}")
    if formal._resolve_device() != _cpu_device():
        raise RuntimeError("Formal Experiment 4 execution backend must be CPU.")


def _build_provenance_record() -> dict:
    hashes = _verify_frozen_dependencies()
    _verify_frozen_training_constants()
    seed42 = _verify_frozen_seed42_evidence()
    _verify_frozen_seed42_content_identity()
    record = {
        "run_family": RUN_FAMILY,
        "temporal_status": TEMPORAL_STATUS,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "frequency_configuration": FREQUENCY_CONFIGURATION,
        "mechanisms": "|".join(MECHANISMS),
        "new_seeds": "|".join(str(s) for s in NEW_SEEDS),
        "all_protocol_seeds": "|".join(str(s) for s in ALL_PROTOCOL_SEEDS),
        "horizons": "|".join(HORIZON_LABELS),
        "expected_training_cells": EXPECTED_TRAINING_CELLS,
        "execution_backend": EXECUTION_BACKEND,
        "seed42_policy": SEED42_POLICY,
        "formal_ticker_count": N_TICKERS,
        "train_samples": N_TRAIN_SAMPLES,
        "validation_samples": N_VALIDATION_SAMPLES,
        "test_samples": N_TEST_SAMPLES,
        "test_dates": N_TEST_DATES,
        "dataset_manifest_path": str(DATASET_MANIFEST_PATH.relative_to(ROOT)),
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "frozen_pathformer_script_path": FROZEN_PATHFORMER_SCRIPT_RELATIVE,
        "frozen_pathformer_script_sha256": hashes["frozen_pathformer_script_sha256"],
        "frozen_experiment3_script_path": FROZEN_EXPERIMENT3_SCRIPT_RELATIVE,
        "frozen_experiment3_script_sha256": hashes["frozen_experiment3_script_sha256"],
        "panel_common_path": PANEL_COMMON_RELATIVE,
        "panel_common_sha256": hashes["panel_common_sha256"],
        "training_script_path": TRAINING_SCRIPT_RELATIVE,
        "training_script_sha256": hashes["training_script_sha256"],
        "daily_scales": "|".join(str(p) for p in EXPECTED_DAILY_PATCH_SIZES),
        "weekly_scales": "|".join(str(p) for p in EXPECTED_WEEKLY_PATCH_SIZES),
        "input_dim": formal.INPUT_DIM,
        "d_model": formal.D_MODEL,
        "nhead": formal.NHEAD,
        "dim_feedforward": formal.DIM_FEEDFORWARD,
        "dropout": formal.DROPOUT,
        "router_hidden": formal.ROUTER_HIDDEN,
        "adaptive_parameter_count": EXPECTED_PARAMETER_COUNTS["adaptive"],
        "static_parameter_count": EXPECTED_PARAMETER_COUNTS["static"],
        "optimizer": "Adam",
        "learning_rate": formal.LEARNING_RATE,
        "loss": "HuberLoss",
        "huber_delta": formal.HUBER_DELTA,
        "batch_size": formal.BATCH_SIZE,
        "max_epochs": formal.MAX_EPOCHS,
        "early_stopping_patience": formal.EARLY_STOPPING_PATIENCE,
        "early_stopping_metric": "validation_huber",
        "early_stopping_improvement_threshold": EARLY_STOPPING_IMPROVEMENT_THRESHOLD,
        "gradient_clip_max_norm": formal.GRADIENT_CLIP_MAX_NORM,
        "best_checkpoint_selection": "first_validation_huber_improvement_beyond_threshold",
        "train_loader_shuffle": True,
        "train_loader_explicit_generator": False,
        "rng_seed_set_before_loader_and_model_construction": True,
        "model_constructed_before_train_loader_iteration": True,
        "seed_affects": "model_initialization|dropout|global_rng_training_shuffle",
        "paired_seed_is_common_random_number_match": False,
        "frozen_seed42_content_identity_verified": True,
        "hyperparameter_search": False,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "test_used_for_seed_or_horizon_selection": False,
    }
    record.update(seed42)
    return record


def _validate_stored_provenance(provenance_df: pd.DataFrame) -> pd.Series:
    if provenance_df.empty:
        raise RuntimeError("Experiment 4 provenance file is empty.")
    if len(provenance_df) != 1:
        raise RuntimeError(f"Experiment 4 provenance must contain exactly one row; found {len(provenance_df)}.")
    current = _build_provenance_record()
    stored = provenance_df.iloc[0]
    for field, current_value in current.items():
        if field not in provenance_df.columns:
            raise RuntimeError(f"Experiment 4 provenance is missing required field: {field}")
        if str(stored[field]) != str(current_value):
            raise RuntimeError(
                f"Immutable Experiment 4 provenance mismatch for {field}: stored={stored[field]}, current={current_value}"
            )
    return stored


def _existing_checkpoints() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(p.name for p in CHECKPOINT_DIR.iterdir() if p.is_file())


def _ensure_provenance() -> pd.Series:
    """Create immutable provenance on first formal write; otherwise verify it."""
    existing_csv = [p.name for p in TRAINING_ARTIFACT_PATHS + TEST_ARTIFACT_PATHS if p.exists()]
    existing_checkpoints = _existing_checkpoints()
    if not PROVENANCE_PATH.exists():
        if existing_checkpoints:
            raise RuntimeError(
                "Experiment 4 checkpoint evidence exists without provenance "
                f"({existing_checkpoints}); refusing to assign provenance retroactively."
            )
        if existing_csv:
            raise RuntimeError(
                f"Experiment 4 evidence exists without provenance ({existing_csv}); "
                "refusing to assign provenance retroactively."
            )
        record = _build_provenance_record()
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_csv(pd.DataFrame([record]), PROVENANCE_PATH)
        return pd.DataFrame([record]).iloc[0]
    return _validate_stored_provenance(pd.read_csv(PROVENANCE_PATH))


def _require_clean_namespace() -> None:
    if EXPERIMENT_DIR.exists() and any(EXPERIMENT_DIR.iterdir()):
        raise RuntimeError(
            "Experiment 4 namespace already contains evidence; use --resume to continue an existing run "
            "instead of overwriting completed evidence."
        )


# ----------------------------------------------------------------------------
# Training-cell completion governance
# ----------------------------------------------------------------------------


def _reconstruct_best_validation_checkpoint(history_subset: pd.DataFrame) -> tuple[int, float]:
    """Replay the frozen checkpoint rule: update only on val_loss < best - threshold."""
    ordered = history_subset.sort_values("epoch", kind="mergesort")
    running_best = float("inf")
    best_epoch = 0
    for epoch, val_loss in zip(ordered["epoch"].astype(int), ordered["val_loss"].astype(float)):
        if val_loss < running_best - EARLY_STOPPING_IMPROVEMENT_THRESHOLD:
            running_best = float(val_loss)
            best_epoch = int(epoch)
    return best_epoch, running_best


def _cell_status(cell: dict) -> str:
    """Return 'complete' or 'absent'; raise on partial or internally inconsistent evidence."""
    mechanism = cell["mechanism"]
    horizon = cell["horizon"]
    seed = int(cell["seed"])
    _reject_seed42(seed)
    label = _cell_label(cell)

    summary_rows = 0
    summary_row = None
    if TRAINING_SUMMARY_PATH.exists():
        subset = _read_csv_cached(TRAINING_SUMMARY_PATH)
        subset = subset[_cell_mask(subset, mechanism, horizon, seed)]
        summary_rows = len(subset)
        if summary_rows == 1:
            summary_row = subset.iloc[0]
        elif summary_rows > 1:
            raise RuntimeError(f"Inconsistent Experiment 4 evidence: duplicate training summary rows for {label}.")

    history_rows = 0
    history_subset = None
    if TRAINING_HISTORY_PATH.exists():
        history_df = _read_csv_cached(TRAINING_HISTORY_PATH)
        history_subset = history_df[_cell_mask(history_df, mechanism, horizon, seed)]
        history_rows = len(history_subset)

    checkpoint_path = _checkpoint_path(mechanism, horizon, seed)
    checkpoint_exists = checkpoint_path.exists()

    signals = [summary_rows > 0, history_rows > 0, checkpoint_exists]
    if not any(signals):
        return "absent"
    if not all(signals):
        raise RuntimeError(
            f"Inconsistent partial Experiment 4 training evidence for {label}: "
            f"summary_rows={summary_rows}, history_rows={history_rows}, checkpoint={checkpoint_exists}. "
            "Refusing to repair, overwrite, or retrain automatically."
        )

    # Summary metadata integrity.
    if str(summary_row["mechanism"]) != mechanism or str(summary_row["horizon"]) != horizon:
        raise RuntimeError(f"{label}: summary mechanism/horizon metadata mismatch.")
    if int(summary_row["seed"]) != seed:
        raise RuntimeError(f"{label}: summary seed metadata mismatch.")
    expected_params = EXPECTED_PARAMETER_COUNTS[mechanism]
    if int(summary_row["parameter_count"]) != expected_params:
        raise RuntimeError(f"{label}: parameter_count must be {expected_params}, found {summary_row['parameter_count']}.")

    # Training-history integrity.
    epochs_trained = int(summary_row["epochs_trained"])
    if history_rows != epochs_trained:
        raise RuntimeError(f"{label}: history length {history_rows} does not match epochs_trained {epochs_trained}.")
    epochs = sorted(history_subset["epoch"].astype(int).tolist())
    if epochs != list(range(1, epochs_trained + 1)):
        raise RuntimeError(f"{label}: history epochs must be contiguous 1..{epochs_trained}.")
    train_losses = history_subset["train_loss"].to_numpy(dtype=np.float64)
    val_losses = history_subset["val_loss"].to_numpy(dtype=np.float64)
    if not np.isfinite(train_losses).all() or not np.isfinite(val_losses).all():
        raise RuntimeError(f"{label}: all history train_loss/val_loss values must be finite.")
    reconstructed_epoch, reconstructed_val = _reconstruct_best_validation_checkpoint(history_subset)
    if int(summary_row["best_epoch"]) != reconstructed_epoch:
        raise RuntimeError(
            f"{label}: stored best_epoch {summary_row['best_epoch']} does not match the checkpoint rule "
            f"reconstruction {reconstructed_epoch}."
        )
    if not np.isclose(float(summary_row["best_val_huber"]), reconstructed_val, rtol=1e-9, atol=1e-12):
        raise RuntimeError(
            f"{label}: stored best_val_huber {summary_row['best_val_huber']} does not match the checkpoint rule "
            f"reconstruction {reconstructed_val}."
        )

    # Checkpoint integrity.
    if str(summary_row["checkpoint_path"]) != str(checkpoint_path.relative_to(ROOT)):
        raise RuntimeError(f"{label}: stored checkpoint_path does not match the canonical cell checkpoint path.")
    if _sha256(checkpoint_path) != str(summary_row["checkpoint_sha256"]):
        raise RuntimeError(f"{label}: checkpoint SHA mismatch.")
    return "complete"


# ----------------------------------------------------------------------------
# Data loaders (frozen helper; TEST unreachable from the training path)
# ----------------------------------------------------------------------------


def _load_dataset():
    ds = load_frozen_panel()
    formal._formal_check_dataset_contract(ds)
    return ds


def _train_val_loaders_only(horizon_value: int, ds):
    """Frozen loader construction with the TEST loader dropped before training."""
    train_loader, val_loader, test_loader = formal._build_training_loaders(
        FREQUENCY_CONFIGURATION, horizon_value, ds, batch_size=formal.BATCH_SIZE
    )
    del test_loader
    if len(train_loader.dataset) != N_TRAIN_SAMPLES:
        raise RuntimeError(f"Train loader must contain {N_TRAIN_SAMPLES} samples; got {len(train_loader.dataset)}.")
    if len(val_loader.dataset) != N_VALIDATION_SAMPLES:
        raise RuntimeError(f"Validation loader must contain {N_VALIDATION_SAMPLES} samples; got {len(val_loader.dataset)}.")
    return train_loader, val_loader


def _test_loader_only(horizon_value: int, ds):
    train_loader, val_loader, test_loader = formal._build_training_loaders(
        FREQUENCY_CONFIGURATION, horizon_value, ds, batch_size=formal.BATCH_SIZE
    )
    del train_loader
    del val_loader
    if len(test_loader.dataset) != N_TEST_SAMPLES:
        raise RuntimeError(f"Test loader must contain {N_TEST_SAMPLES} samples; got {len(test_loader.dataset)}.")
    return test_loader


def _batched_inference(model: nn.Module, loader, expected_rows: int, device: torch.device):
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
        raise RuntimeError("No batches were processed during Experiment 4 inference.")
    pred_np = torch.cat(pred_chunks, dim=0).numpy().astype(np.float32)
    if pred_np.shape[0] != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} predictions, got {pred_np.shape[0]}.")

    daily_np = torch.cat(daily_chunks, dim=0).numpy().astype(np.float64)
    weekly_np = torch.cat(weekly_chunks, dim=0).numpy().astype(np.float64)
    if daily_np.shape != (expected_rows, len(DAILY_WEIGHT_COLUMNS)):
        raise RuntimeError(f"Daily scale-weight shape mismatch: {daily_np.shape}")
    if weekly_np.shape != (expected_rows, len(WEEKLY_WEIGHT_COLUMNS)):
        raise RuntimeError(f"Weekly scale-weight shape mismatch: {weekly_np.shape}")
    return pred_np, daily_np, weekly_np


# ----------------------------------------------------------------------------
# Training / validation (never touches TEST)
# ----------------------------------------------------------------------------


def train_and_validate_cell(mechanism: str, horizon: str, seed: int) -> dict:
    """Train one Experiment 4 cell on Train, early-stop on Validation. Never reads TEST."""
    mechanism = _validate_mechanism(mechanism)
    horizon = _validate_horizon_label(horizon)
    seed = _validate_seed(seed)
    horizon_value = HORIZON_MAP[horizon]
    device = _cpu_device()

    _verify_frozen_dependencies()
    _verify_frozen_training_constants()

    ds = _load_dataset()
    formal.set_seed(seed)
    train_loader, val_loader = _train_val_loaders_only(horizon_value, ds)

    model = _make_experiment4_model(mechanism).to(device)
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
                raise RuntimeError(f"Non-finite loss at {mechanism}/seed{seed}/{horizon}, epoch={epoch}")
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError(
                        f"Non-finite gradient for {name} at {mechanism}/seed{seed}/{horizon}, epoch={epoch}"
                    )
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
                    raise RuntimeError(
                        f"Non-finite validation loss at {mechanism}/seed{seed}/{horizon}, epoch={epoch}"
                    )
                val_loss_total += val_loss.item() * y_batch.shape[0]

        train_loss = train_loss_total / max(len(train_loader.dataset), 1)
        val_loss = val_loss_total / max(len(val_loader.dataset), 1)
        history.append({
            "mechanism": mechanism,
            "horizon": horizon,
            "seed": seed,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        if val_loss < min_val_loss - EARLY_STOPPING_IMPROVEMENT_THRESHOLD:
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
    val_pred, _daily_w, _weekly_w = _batched_inference(model, val_loader, N_VALIDATION_SAMPLES, device)
    val_metrics = evaluate_predictions(val_y_true, val_pred, val_ticker, val_anchor)

    checkpoint_path = _checkpoint_path(mechanism, horizon, seed)
    _assert_experiment4_path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_checkpoint = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(best_state, tmp_checkpoint)
    os.replace(tmp_checkpoint, checkpoint_path)

    summary_row = {
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": seed,
        "frequency_configuration": FREQUENCY_CONFIGURATION,
        "execution_backend": EXECUTION_BACKEND,
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
    return {"summary": summary_row, "history": history}


def _write_training_cell_evidence(result: dict) -> None:
    _append_csv(pd.DataFrame([result["summary"]]), TRAINING_SUMMARY_PATH, CELL_KEY)
    _append_csv(pd.DataFrame(result["history"]), TRAINING_HISTORY_PATH, CELL_KEY + ["epoch"])


def _run_train_cell(mechanism: str, horizon: str, seed: int, resume: bool) -> None:
    cell = {
        "mechanism": _validate_mechanism(mechanism),
        "horizon": _validate_horizon_label(horizon),
        "seed": _validate_seed(seed),
    }
    if not resume:
        _require_clean_namespace()
    _assert_seed42_absent_from_namespace()
    _ensure_provenance()
    status = _cell_status(cell)
    if status == "complete":
        if not resume:
            raise RuntimeError(f"Experiment 4 cell already complete: {_cell_label(cell)}; refusing to overwrite.")
        print(f"EXPERIMENT 4 TRAINING CELL ALREADY COMPLETE (skipped): {_cell_label(cell)}")
        return
    result = train_and_validate_cell(cell["mechanism"], cell["horizon"], cell["seed"])
    _write_training_cell_evidence(result)
    print(f"EXPERIMENT 4 TRAINING CELL COMPLETE: {_cell_label(cell)}")


def _run_train_all(resume: bool) -> None:
    if not resume:
        _require_clean_namespace()
    _assert_seed42_absent_from_namespace()
    _ensure_provenance()
    cells = _all_training_cells()
    for cell in cells:
        status = _cell_status(cell)
        if status == "complete":
            if not resume:
                raise RuntimeError("Completed cell found without --resume; refusing to overwrite formal evidence.")
            continue
        result = train_and_validate_cell(cell["mechanism"], cell["horizon"], cell["seed"])
        _write_training_cell_evidence(result)
    for cell in cells:
        if _cell_status(cell) != "complete":
            raise RuntimeError(f"Training finished but cell is incomplete: {_cell_label(cell)}")
    print(f"EXPERIMENT 4 TRAINING COMPLETE: {len(cells)} new cells (seed 42 not retrained)")


# ----------------------------------------------------------------------------
# TEST evaluation (separate mode; exactly once per new cell)
# ----------------------------------------------------------------------------


def _test_attempt_claim_count(mechanism: str, horizon: str, seed: int) -> int:
    if not TEST_ATTEMPT_LEDGER_PATH.exists():
        return 0
    ledger = _read_csv_cached(TEST_ATTEMPT_LEDGER_PATH)
    return int(_cell_mask(ledger, mechanism, horizon, seed).sum())


def _claim_test_attempt(cell: dict, summary_row: pd.Series) -> None:
    """Immutably claim the single permitted TEST attempt before any TEST loader or inference."""
    mechanism = cell["mechanism"]
    horizon = cell["horizon"]
    seed = int(cell["seed"])
    label = _cell_label(cell)

    checkpoint_path = ROOT / str(summary_row["checkpoint_path"])
    if not checkpoint_path.exists():
        raise RuntimeError(f"{label}: checkpoint missing; refusing to claim a TEST attempt.")
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != str(summary_row["checkpoint_sha256"]):
        raise RuntimeError(f"{label}: checkpoint SHA mismatch; refusing to claim a TEST attempt.")

    if _test_attempt_claim_count(mechanism, horizon, seed) != 0:
        raise RuntimeError(
            f"{label}: a TEST attempt has already been claimed. TEST is permitted exactly once per new cell; "
            "an interrupted attempt requires an explicit audited recovery decision."
        )

    claim = pd.DataFrame([{
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": seed,
        "checkpoint_path": str(summary_row["checkpoint_path"]),
        "checkpoint_sha256": checkpoint_sha,
        "attempt_number": 1,
    }])
    _append_csv(claim, TEST_ATTEMPT_LEDGER_PATH, CELL_KEY)


def _test_cell_status(cell: dict) -> str:
    """Return 'complete' or 'absent'; raise on partial or unclaimed TEST evidence."""
    mechanism = cell["mechanism"]
    horizon = cell["horizon"]
    seed = int(cell["seed"])
    _reject_seed42(seed)
    label = _cell_label(cell)

    claim_rows = _test_attempt_claim_count(mechanism, horizon, seed)
    if claim_rows > 1:
        raise RuntimeError(f"{label}: duplicate TEST attempt claims in the ledger; refusing to proceed.")

    counts: dict[str, int] = {}
    subsets: dict[str, pd.DataFrame] = {}
    for name, path in (
        ("summary", TEST_SUMMARY_PATH),
        ("predictions", TEST_PREDICTIONS_PATH),
        ("rank_ic", TEST_RANK_IC_PATH),
        ("scale_weights", TEST_SCALE_WEIGHTS_PATH),
    ):
        if path.exists():
            subset = _read_csv_cached(path)
            subset = subset[_cell_mask(subset, mechanism, horizon, seed)]
            subsets[name] = subset
            counts[name] = len(subset)
        else:
            counts[name] = 0

    output_signals = [counts[name] > 0 for name in ("summary", "predictions", "rank_ic", "scale_weights")]
    if claim_rows == 0:
        if not any(output_signals):
            return "absent"
        raise RuntimeError(
            f"{label}: TEST outputs exist without an attempt-ledger claim ({counts}). Refusing to proceed."
        )
    if not all(output_signals):
        raise RuntimeError(
            f"{label}: a TEST attempt was claimed but its outputs are missing or partial ({counts}). "
            "Refusing to repair or re-evaluate automatically."
        )

    if counts["summary"] != 1:
        raise RuntimeError(f"{label}: expected exactly 1 TEST summary row, found {counts['summary']}.")
    if counts["predictions"] != N_TEST_SAMPLES:
        raise RuntimeError(f"{label}: expected {N_TEST_SAMPLES} TEST prediction rows, found {counts['predictions']}.")
    if counts["rank_ic"] != N_TEST_DATES:
        raise RuntimeError(f"{label}: expected {N_TEST_DATES} RankIC rows, found {counts['rank_ic']}.")
    if counts["scale_weights"] != N_TEST_SAMPLES:
        raise RuntimeError(f"{label}: expected {N_TEST_SAMPLES} scale-weight rows, found {counts['scale_weights']}.")

    predictions = subsets["predictions"]
    if predictions.duplicated(subset=["ticker", "anchor_date"]).any():
        raise RuntimeError(f"{label}: duplicate ticker/anchor_date keys in TEST predictions.")
    if predictions["ticker"].nunique() != N_TICKERS:
        raise RuntimeError(f"{label}: expected {N_TICKERS} tickers in TEST predictions.")
    if predictions["anchor_date"].nunique() != N_TEST_DATES:
        raise RuntimeError(f"{label}: expected {N_TEST_DATES} TEST dates in predictions.")
    if subsets["scale_weights"].duplicated(subset=["ticker", "anchor_date"]).any():
        raise RuntimeError(f"{label}: duplicate ticker/anchor_date keys in TEST scale weights.")
    if subsets["rank_ic"]["anchor_date"].nunique() != N_TEST_DATES:
        raise RuntimeError(f"{label}: RankIC anchor dates must be unique and cover {N_TEST_DATES} dates.")

    summary_row = subsets["summary"].iloc[0]
    if int(summary_row["parameter_count"]) != EXPECTED_PARAMETER_COUNTS[mechanism]:
        raise RuntimeError(f"{label}: TEST summary parameter_count mismatch.")
    return "complete"


def evaluate_frozen_checkpoint_on_test(mechanism: str, horizon: str, seed: int, summary_row: pd.Series, ds) -> dict:
    """Forward-only TEST inference for one already-trained cell. No training, no reselection."""
    mechanism = _validate_mechanism(mechanism)
    horizon = _validate_horizon_label(horizon)
    seed = _validate_seed(seed)
    horizon_value = HORIZON_MAP[horizon]
    device = _cpu_device()

    formal.set_seed(seed)
    test_loader = _test_loader_only(horizon_value, ds)

    checkpoint_path = ROOT / str(summary_row["checkpoint_path"])
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint missing before TEST evaluation: {checkpoint_path}")
    if _sha256(checkpoint_path) != str(summary_row["checkpoint_sha256"]):
        raise RuntimeError(f"Checkpoint SHA mismatch before TEST evaluation: {checkpoint_path}")

    model = _make_experiment4_model(mechanism).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_mask = ds.split == "test"
    ticker = ds.ticker[test_mask]
    anchor_date = [_canonical_date(v) for v in ds.anchor_date[test_mask]]
    y_true = ds.y[horizon_value][test_mask]

    pred_np, daily_np, weekly_np = _batched_inference(model, test_loader, N_TEST_SAMPLES, device)

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
        raise RuntimeError(f"RankIC must contain {N_TEST_DATES} dates for {mechanism}/seed{seed}/{horizon}.")
    valid_ic = rank_ic_df["rank_ic"].dropna().to_numpy()
    mean_rank_ic = float(np.mean(valid_ic)) if len(valid_ic) else float("nan")
    rank_ic_std = float(np.std(valid_ic)) if len(valid_ic) else float("nan")
    rank_icir = float(mean_rank_ic / rank_ic_std) if np.isfinite(rank_ic_std) and rank_ic_std > 0 else float("nan")

    rank_ic_df.insert(0, "seed", seed)
    rank_ic_df.insert(0, "horizon", horizon)
    rank_ic_df.insert(0, "mechanism", mechanism)

    summary = {
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": seed,
        "frequency_configuration": FREQUENCY_CONFIGURATION,
        "execution_backend": EXECUTION_BACKEND,
        "parameter_count": int(summary_row["parameter_count"]),
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
        "checkpoint_path": str(summary_row["checkpoint_path"]),
        "checkpoint_sha256": str(summary_row["checkpoint_sha256"]),
    }

    predictions = pd.DataFrame({
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": seed,
        "ticker": ticker,
        "anchor_date": anchor_date,
        "y_true": y_true.astype(np.float32),
        "y_pred": pred_np.astype(np.float32),
    })

    weights = pd.DataFrame({
        "mechanism": mechanism,
        "horizon": horizon,
        "seed": seed,
        "ticker": ticker,
        "anchor_date": anchor_date,
    })
    for index, column in enumerate(DAILY_WEIGHT_COLUMNS):
        weights[column] = daily_np[:, index]
    for index, column in enumerate(WEEKLY_WEIGHT_COLUMNS):
        weights[column] = weekly_np[:, index]

    return {"summary": summary, "predictions": predictions, "rank_ic": rank_ic_df, "scale_weights": weights}


def _write_test_cell_evidence(result: dict) -> None:
    _append_csv(pd.DataFrame([result["summary"]]), TEST_SUMMARY_PATH, CELL_KEY)
    _append_csv(result["predictions"], TEST_PREDICTIONS_PATH, CELL_KEY + ["ticker", "anchor_date"])
    _append_csv(result["rank_ic"], TEST_RANK_IC_PATH, CELL_KEY + ["anchor_date"])
    _append_csv(result["scale_weights"], TEST_SCALE_WEIGHTS_PATH, CELL_KEY + ["ticker", "anchor_date"])


def _audit_test_evidence_identity() -> None:
    """Read-only pairability audit across all 24 new TEST cells. Aggregates nothing."""
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    weights = pd.read_csv(TEST_SCALE_WEIGHTS_PATH)
    rank_ic = pd.read_csv(TEST_RANK_IC_PATH)

    for horizon in HORIZON_LABELS:
        reference_keys = None
        reference_y_true = None
        horizon_cells = [(mechanism, seed) for mechanism in MECHANISMS for seed in NEW_SEEDS]
        if len(horizon_cells) != len(MECHANISMS) * len(NEW_SEEDS):
            raise RuntimeError(f"{horizon}: expected {len(MECHANISMS) * len(NEW_SEEDS)} TEST cells.")

        for mechanism, seed in horizon_cells:
            label = f"{mechanism}/seed{seed}/{horizon}"
            pred = predictions[_cell_mask(predictions, mechanism, horizon, seed)].copy()
            pred["anchor_date"] = pred["anchor_date"].apply(_canonical_date)
            pred_keys = _sorted_prediction_keys(pred, label)
            pred_sorted = pred.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
            y_true = pred_sorted["y_true"].to_numpy(dtype=np.float32)
            y_pred = pred_sorted["y_pred"].to_numpy(dtype=np.float32)
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                raise RuntimeError(f"{label}: TEST y_true and y_pred must be finite.")

            weight_subset = weights[_cell_mask(weights, mechanism, horizon, seed)].copy()
            weight_subset["anchor_date"] = weight_subset["anchor_date"].apply(_canonical_date)
            weight_keys = _sorted_weight_keys(weight_subset, label)
            if not weight_keys.equals(pred_keys[["ticker", "anchor_date"]]):
                raise RuntimeError(f"{label}: scale-weight keys do not match the TEST prediction keys.")
            weight_sorted = weight_subset.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
            daily_np = weight_sorted[list(DAILY_WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
            weekly_np = weight_sorted[list(WEEKLY_WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
            if not np.isfinite(daily_np).all() or not np.isfinite(weekly_np).all():
                raise RuntimeError(f"{label}: scale weights must be finite.")
            if (daily_np < 0).any() or (weekly_np < 0).any():
                raise RuntimeError(f"{label}: scale weights must be non-negative.")
            if not np.allclose(daily_np.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"{label}: Daily scale weights must sum to 1.")
            if not np.allclose(weekly_np.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"{label}: Weekly scale weights must sum to 1.")
            if mechanism == "static":
                if not np.all(daily_np == daily_np[0]) or not np.all(weekly_np == weekly_np[0]):
                    raise RuntimeError(f"{label}: Static scale weights must be sample-independent.")

            rank_ic_subset = rank_ic[_cell_mask(rank_ic, mechanism, horizon, seed)]
            rank_ic_dates = rank_ic_subset["anchor_date"].apply(_canonical_date)
            if len(rank_ic_dates) != N_TEST_DATES or rank_ic_dates.nunique() != N_TEST_DATES:
                raise RuntimeError(f"{label}: RankIC must contain {N_TEST_DATES} unique TEST dates.")
            if set(rank_ic_dates) != set(pred_keys["anchor_date"]):
                raise RuntimeError(f"{label}: RankIC dates do not match the TEST prediction dates.")

            if reference_keys is None:
                reference_keys = pred_keys[["ticker", "anchor_date"]]
                reference_y_true = y_true
                continue
            if not pred_keys[["ticker", "anchor_date"]].equals(reference_keys):
                raise RuntimeError(f"{label}: TEST keys differ from the other cells at horizon {horizon}.")
            _assert_float32_y_true_identity(y_true, reference_y_true, label)


def _run_evaluate_test(resume: bool) -> None:
    _assert_seed42_absent_from_namespace()
    _ensure_provenance()
    cells = _all_training_cells()
    for cell in cells:
        if _cell_status(cell) != "complete":
            raise RuntimeError(
                f"Cannot evaluate TEST: training cell {_cell_label(cell)} is not complete. "
                f"All {EXPECTED_TRAINING_CELLS} new training cells must be complete first."
            )

    pending = []
    for cell in cells:
        if _test_cell_status(cell) == "complete":
            if not resume:
                raise RuntimeError(
                    f"TEST evidence already exists for {_cell_label(cell)}; refusing to re-evaluate. "
                    "Use --evaluate-test --resume to continue an interrupted evaluation."
                )
            continue
        pending.append(cell)

    summary_df = _read_csv_cached(TRAINING_SUMMARY_PATH)
    if pending:
        ds = _load_dataset()
        for cell in pending:
            matched = summary_df[_cell_mask(summary_df, cell["mechanism"], cell["horizon"], cell["seed"])]
            if len(matched) != 1:
                raise RuntimeError(
                    f"Expected exactly one training summary row for {_cell_label(cell)}; found {len(matched)}."
                )
            training_row = matched.iloc[0]
            # The attempt claim is written before any TEST loader or inference for this cell.
            _claim_test_attempt(cell, training_row)
            result = evaluate_frozen_checkpoint_on_test(
                cell["mechanism"], cell["horizon"], cell["seed"], training_row, ds
            )
            _write_test_cell_evidence(result)

    for cell in cells:
        if _test_cell_status(cell) != "complete":
            raise RuntimeError(f"TEST evaluation finished but cell is incomplete: {_cell_label(cell)}")

    if len(_read_csv_cached(TEST_ATTEMPT_LEDGER_PATH)) != EXPECTED_TEST_CELLS:
        raise RuntimeError(f"TEST attempt ledger must contain exactly {EXPECTED_TEST_CELLS} rows.")
    if len(_read_csv_cached(TEST_SUMMARY_PATH)) != EXPECTED_TEST_CELLS:
        raise RuntimeError(f"TEST summary must contain exactly {EXPECTED_TEST_CELLS} rows.")
    if len(_read_csv_cached(TEST_PREDICTIONS_PATH)) != EXPECTED_TEST_PREDICTION_ROWS:
        raise RuntimeError(f"TEST predictions must contain exactly {EXPECTED_TEST_PREDICTION_ROWS} rows.")
    if len(_read_csv_cached(TEST_RANK_IC_PATH)) != EXPECTED_TEST_RANK_IC_ROWS:
        raise RuntimeError(f"TEST RankIC must contain exactly {EXPECTED_TEST_RANK_IC_ROWS} rows.")
    if len(_read_csv_cached(TEST_SCALE_WEIGHTS_PATH)) != EXPECTED_TEST_SCALE_WEIGHT_ROWS:
        raise RuntimeError(f"TEST scale weights must contain exactly {EXPECTED_TEST_SCALE_WEIGHT_ROWS} rows.")
    _audit_test_evidence_identity()
    print(f"EXPERIMENT 4 TEST EVALUATION COMPLETE: {EXPECTED_TEST_CELLS} new cells (seed 42 not re-evaluated)")


# ----------------------------------------------------------------------------
# Smoke validation (no training, no TEST evaluation, writes nothing)
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
    experiment4_before = _snapshot_namespace(EXPERIMENT_DIR)
    main_v2_before = _snapshot_namespace(FROZEN_ADAPTIVE_SEED42_DIR)
    experiment3_before = _snapshot_namespace(FROZEN_STATIC_SEED42_DIR)

    # A. Frozen dependencies, constants and read-only seed-42 evidence.
    _verify_frozen_dependencies()
    _verify_frozen_training_constants()
    _verify_frozen_seed42_evidence()
    _verify_frozen_seed42_content_identity()

    # B. Dataset Contract V2 (structural verification only).
    formal._formal_check_dataset_contract(load_frozen_panel())

    # C. Grid contract and seed governance.
    cells = _all_training_cells()
    assert len(cells) == EXPECTED_TRAINING_CELLS == 24, len(cells)
    assert EXPECTED_TEST_PREDICTION_ROWS == 159528
    assert EXPECTED_TEST_RANK_IC_ROWS == 9384
    assert FROZEN_EXTERNAL_SEED not in NEW_SEEDS
    assert set(NEW_SEEDS) | {FROZEN_EXTERNAL_SEED} == set(ALL_PROTOCOL_SEEDS)
    for bad_seed in (FROZEN_EXTERNAL_SEED, 7):
        try:
            _validate_seed(bad_seed)
        except (RuntimeError, ValueError):
            pass
        else:
            raise RuntimeError(f"Seed {bad_seed} must be rejected by every Experiment 4 path.")
    for bad_mechanism in FORBIDDEN_MECHANISMS + ("daily_only", "weekly_only"):
        try:
            _validate_mechanism(bad_mechanism)
        except (RuntimeError, ValueError):
            pass
        else:
            raise RuntimeError(f"Mechanism {bad_mechanism} must be rejected by every Experiment 4 path.")

    # D. Model construction, parameter counts and forward-only structure on synthetic tensors.
    device = _cpu_device()
    assert device == torch.device("cpu")
    formal.set_seed(NEW_SEEDS[0])
    batch = 2
    x_daily = torch.randn(batch, 90, 5)
    x_weekly = torch.randn(batch, 26, 5)
    y = torch.randn(batch)

    weights_by_mechanism = {}
    for mechanism in MECHANISMS:
        model = _make_experiment4_model(mechanism).to(device)
        model.eval()
        assert formal.count_trainable_parameters(model) == EXPECTED_PARAMETER_COUNTS[mechanism]
        assert len(model.daily_branch.experts) == len(DAILY_WEIGHT_COLUMNS)
        assert len(model.weekly_branch.experts) == len(WEEKLY_WEIGHT_COLUMNS)
        with torch.no_grad():
            pred, daily_w, weekly_w = model.forward_with_router(x_daily.to(device), x_weekly.to(device))
        assert pred.shape == y.shape, (mechanism, pred.shape, y.shape)
        assert torch.isfinite(pred).all()
        assert daily_w.shape == (batch, len(DAILY_WEIGHT_COLUMNS)), (mechanism, daily_w.shape)
        assert weekly_w.shape == (batch, len(WEEKLY_WEIGHT_COLUMNS)), (mechanism, weekly_w.shape)
        assert (daily_w >= 0).all() and (weekly_w >= 0).all()
        assert torch.allclose(daily_w.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0.0)
        assert torch.allclose(weekly_w.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0.0)
        weights_by_mechanism[mechanism] = (model, daily_w.detach(), weekly_w.detach())

    # E. Mechanism separation: Static is sample-independent, Adaptive is sample-dependent.
    perturbed_daily = x_daily.clone()
    perturbed_daily[1] = perturbed_daily[1] * 0.0 + 99.0
    static_model, static_daily_w, static_weekly_w = weights_by_mechanism["static"]
    with torch.no_grad():
        _p, static_daily_w2, static_weekly_w2 = static_model.forward_with_router(perturbed_daily, x_weekly)
    assert torch.equal(static_daily_w, static_daily_w2), "Static weights must be sample-independent."
    assert torch.equal(static_weekly_w, static_weekly_w2), "Static weights must be sample-independent."
    adaptive_model, adaptive_daily_w, _adaptive_weekly_w = weights_by_mechanism["adaptive"]
    with torch.no_grad():
        _p, adaptive_daily_w2, _w2 = adaptive_model.forward_with_router(perturbed_daily, x_weekly)
    assert torch.allclose(adaptive_daily_w[0], adaptive_daily_w2[0], atol=1e-6, rtol=0.0), (
        "Adaptive router must be sample-dependent, not batch-dependent."
    )

    # F. Namespace isolation: no formal write path leaves the Experiment 4 namespace.
    for forbidden in (FROZEN_ADAPTIVE_SEED42_DIR / "x.csv", FROZEN_STATIC_SEED42_DIR / "x.csv", ROOT / "x.csv"):
        try:
            _assert_experiment4_path(forbidden)
        except RuntimeError:
            pass
        else:
            raise RuntimeError(f"Write path outside the Experiment 4 namespace must be rejected: {forbidden}")
    for allowed in (TRAINING_SUMMARY_PATH, TEST_PREDICTIONS_PATH, _checkpoint_path("adaptive", "5d", 0)):
        _assert_experiment4_path(allowed)

    # G. Namespaces untouched.
    _assert_namespace_unchanged(experiment4_before, _snapshot_namespace(EXPERIMENT_DIR), "Experiment 4")
    _assert_namespace_unchanged(main_v2_before, _snapshot_namespace(FROZEN_ADAPTIVE_SEED42_DIR), "MAIN V2")
    _assert_namespace_unchanged(experiment3_before, _snapshot_namespace(FROZEN_STATIC_SEED42_DIR), "Experiment 3")

    print("EXPERIMENT 4 MULTI-SEED SMOKE VALIDATION PASSED")


def _print_contract_summary() -> None:
    print("Experiment 4 — PathFormer multi-seed robustness runner (writes nothing without an explicit mode)")
    print(f"  run_family                : {RUN_FAMILY}")
    print(f"  frequency_configuration   : {FREQUENCY_CONFIGURATION}")
    print(f"  mechanisms                : {', '.join(MECHANISMS)}")
    print(f"  horizons                  : {', '.join(HORIZON_LABELS)}")
    print(f"  new seeds                 : {', '.join(str(s) for s in NEW_SEEDS)}")
    print(f"  all protocol seeds        : {', '.join(str(s) for s in ALL_PROTOCOL_SEEDS)}")
    print(f"  new training cells        : {EXPECTED_TRAINING_CELLS}")
    print(f"  execution backend         : {EXECUTION_BACKEND}")
    print(f"  seed 42 policy            : {SEED42_POLICY}")
    print(f"  output namespace          : {EXPERIMENT_DIR.relative_to(ROOT)}")
    print("  modes                     : --smoke | --train-cell | --train-all | --evaluate-test")
    print("  protocol                  : docs/experiment4_multiseed_protocol.md")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 4 PathFormer multi-seed robustness runner "
            "(daily_weekly, CPU, Adaptive vs Static, new seeds 0/1/21/3407)."
        )
    )
    parser.add_argument("--smoke", action="store_true", help="Structural validation only. Writes nothing.")
    parser.add_argument("--train-cell", action="store_true", help="Train exactly one mechanism/horizon/seed cell.")
    parser.add_argument("--train-all", action="store_true", help=f"Train all {EXPECTED_TRAINING_CELLS} new cells.")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=f"Evaluate the {EXPECTED_TEST_CELLS} new frozen checkpoints on TEST. No training.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip only fully validated complete cells.")
    parser.add_argument("--mechanism", choices=list(MECHANISMS), default=None)
    parser.add_argument("--horizon", choices=list(HORIZON_LABELS), default=None)
    parser.add_argument("--seed", type=int, default=None, help=f"One of {NEW_SEEDS}. Seed 42 is rejected.")
    args = parser.parse_args()

    modes = [args.smoke, args.train_cell, args.train_all, args.evaluate_test]
    if sum(bool(m) for m in modes) > 1:
        parser.error("Choose exactly one of --smoke / --train-cell / --train-all / --evaluate-test.")
    if args.resume and not (args.train_cell or args.train_all or args.evaluate_test):
        parser.error("--resume is only valid with --train-cell, --train-all or --evaluate-test.")

    if args.train_cell:
        if args.mechanism is None or args.horizon is None or args.seed is None:
            parser.error("--train-cell requires --mechanism, --horizon and --seed.")
        if int(args.seed) == FROZEN_EXTERNAL_SEED:
            parser.error(
                "Seed 42 is frozen external evidence (Adaptive from MAIN V2, Static from Experiment 3) and must "
                "never be retrained or re-evaluated by this runner."
            )
        if int(args.seed) not in NEW_SEEDS:
            parser.error(f"--seed must be one of the pre-specified new seeds {NEW_SEEDS}.")
    elif any(v is not None for v in (args.mechanism, args.horizon, args.seed)):
        parser.error("Cell arguments are only valid with --train-cell.")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke:
        _run_smoke_validation()
        return
    if args.train_cell:
        _run_train_cell(args.mechanism, args.horizon, args.seed, resume=args.resume)
        return
    if args.train_all:
        _run_train_all(resume=args.resume)
        return
    if args.evaluate_test:
        _run_evaluate_test(resume=args.resume)
        return
    _print_contract_summary()


if __name__ == "__main__":
    main()
