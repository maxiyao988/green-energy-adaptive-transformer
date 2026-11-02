"""Standalone Experiment 4 serial-dependence-aware inference runner.

This script implements exactly the frozen protocol in
`docs/experiment4_inference_protocol.md` (SHA256 embedded and verified at
runtime below). It reads ONLY frozen persisted prediction/evidence files,
reconstructs frozen descriptive metrics as bridge audits, builds paired
Adaptive-vs-Static squared-loss differentials, aggregates them to
seed-date and date level, performs the frozen Moving Block Bootstrap
(PRIMARY) and the frozen secondary HAC analysis, and writes only the
frozen inference output family under `dataset/audit/experiment4_inference/`.

This script never trains a model, loads a checkpoint, instantiates a
neural network, generates fresh TEST predictions, modifies prior evidence,
tunes any hyperparameter, or performs router/scale-weight interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT4_DIR = ROOT / "dataset" / "audit" / "experiment4_multiseed"
MAIN_V2_DIR = ROOT / "dataset" / "audit" / "main_v2" / "pathformer"
EXPERIMENT3_DIR = ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation"
INFERENCE_DIR = ROOT / "dataset" / "audit" / "experiment4_inference"

PROTOCOL_PATH = ROOT / "docs" / "experiment4_inference_protocol.md"
FINALIZER_SCRIPT_PATH = ROOT / "scripts" / "python" / "panel_finalize_pathformer_experiment4_multiseed.py"

EXPERIMENT4_TEST_PREDICTIONS_PATH = EXPERIMENT4_DIR / "experiment4_test_predictions.csv"
ADAPTIVE_SEED42_PREDICTIONS_PATH = MAIN_V2_DIR / "panel_pathformer_test_predictions.csv"
STATIC_SEED42_PREDICTIONS_PATH = EXPERIMENT3_DIR / "experiment3_test_predictions.csv"
COMBINED_CELL_METRICS_PATH = EXPERIMENT4_DIR / "experiment4_combined_cell_metrics.csv"
SEED_PAIR_DELTAS_PATH = EXPERIMENT4_DIR / "experiment4_seed_pair_deltas.csv"
ROBUSTNESS_SUMMARY_PATH = EXPERIMENT4_DIR / "experiment4_robustness_summary.csv"

# Frozen protocol SHA256; the runner fails closed if the on-disk protocol differs.
PROTOCOL_SHA256 = "e156f24342336044015f812407423639cbce6181dd2ea19e785e52198dce09a5"

FROZEN_INPUT_HASHES = {
    EXPERIMENT4_TEST_PREDICTIONS_PATH: "6b5f5ea94f1bb5f25ebd5ed09d7e4e7eb609945f97f953e7fb95116e2e866e8a",
    ADAPTIVE_SEED42_PREDICTIONS_PATH: "a05955a561da1b8880f1bb0878a6576852c087cc45707546d446ce96b0fabdaf",
    STATIC_SEED42_PREDICTIONS_PATH: "95d1b06cc3ce0209e99833c29a1c3672a6d7a1a9d46d32f16ef1e48fa9846515",
    COMBINED_CELL_METRICS_PATH: "89cfa52907d065cfc6f94ed662f17c07cb18a3a9c19d8ce961ec15a1959d5a41",
    SEED_PAIR_DELTAS_PATH: "2f2a66689ac6abc8371565c2d8567c5ffa08259e5cb4879dd1fb306645c7f103",
    ROBUSTNESS_SUMMARY_PATH: "9571b4424ad4261e8e973bcdc528718c26a6744e1d55e5d6323f69fb57d063a3",
    FINALIZER_SCRIPT_PATH: "d9ff7b456186848ada227fba1cc75a43508d65659f28ce523e024fe5744e7c39",
    PROTOCOL_PATH: PROTOCOL_SHA256,
}

FINAL_OUTPUTS = (
    INFERENCE_DIR / "experiment4_loss_diff_by_seed_date.csv",
    INFERENCE_DIR / "experiment4_loss_diff_by_date.csv",
    INFERENCE_DIR / "experiment4_block_bootstrap_replicates.csv",
    INFERENCE_DIR / "experiment4_block_bootstrap_summary.csv",
    INFERENCE_DIR / "experiment4_hac_summary.csv",
    INFERENCE_DIR / "experiment4_inference_summary.txt",
    INFERENCE_DIR / "experiment4_inference_manifest.csv",
)

HORIZONS = ["5d", "10d", "20d"]
HORIZON_INTS = {"5d": 5, "10d": 10, "20d": 20}
SEEDS = [0, 1, 21, 42, 3407]
NEW_SEEDS = [0, 1, 21, 3407]
FROZEN_EXTERNAL_SEED = 42
MECHANISMS = ["adaptive", "static"]

N_TICKERS = 17
N_TEST_DATES = 391
N_CELL_ROWS = 6647

FREQUENCY_CONFIGURATION = "daily_weekly"
HORIZON_LABEL_MAP = {5: "5d", 10: "10d", 20: "20d"}

BOOTSTRAP_B = 20000
BOOTSTRAP_BASE_SEED = 20260826

PRIMARY_BLOCK_LENGTH = {"5d": 5, "10d": 10, "20d": 20}
SENSITIVITY_BLOCK_LENGTH = {"5d": 10, "10d": 20, "20d": 40}

HAC_MAX_LAG = {"5d": 4, "10d": 9, "20d": 19}

ALPHA = 0.05

AUDIT_RTOL = 1e-8
AUDIT_ATOL = 1e-8

FLOAT32_ATOL = np.finfo(np.float32).eps

CANONICAL_HORIZON_ORDER = ["5d", "10d", "20d"]
CANONICAL_SEED_ORDER = [0, 1, 21, 42, 3407]
CANONICAL_SPEC_ORDER = ["PRIMARY", "SENSITIVITY"]

# Frozen five-seed mean delta_mse point estimates (descriptive evidence only).
FROZEN_MEAN_DELTA_MSE = {
    "5d": 0.0003479534760117531,
    "10d": 0.0005263369530439377,
    "20d": 0.002097966521978378,
}

EXPECTED_OUTPUT_ROW_COUNTS = {
    "seed_date": len(HORIZONS) * len(SEEDS) * N_TEST_DATES,
    "date": len(HORIZONS) * N_TEST_DATES,
    "replicates": len(HORIZONS) * len(CANONICAL_SPEC_ORDER) * BOOTSTRAP_B,
    "bootstrap_summary": len(HORIZONS) * len(CANONICAL_SPEC_ORDER),
    "hac_summary": len(HORIZONS),
}


# ---------------------------------------------------------------------------
# Hashing / fail-closed helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required frozen input: {label}: {path}")


def _require_absent(path: Path, label: str) -> None:
    if path.exists():
        raise RuntimeError(f"Final inference output already exists (fail closed): {label}: {path}")


def _verify_frozen_input_hashes() -> None:
    for path, expected_hash in FROZEN_INPUT_HASHES.items():
        _require_exists(path, str(path))
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen input hash mismatch (fail closed) for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def _require_final_outputs_absent() -> None:
    for path in FINAL_OUTPUTS:
        _require_absent(path, path.name)


# ---------------------------------------------------------------------------
# Loading / canonical source extraction
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Frozen input is empty (fail closed): {path}")
    return df


def _canonicalize_anchor_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series)
    if parsed.isna().any():
        raise RuntimeError("anchor_date contains NaT after datetime parsing (fail closed).")
    return parsed


def _extract_new_experiment4_cells(pred_df: pd.DataFrame) -> pd.DataFrame:
    subset = pred_df[
        pred_df["mechanism"].astype(str).isin(MECHANISMS)
        & pred_df["seed"].astype(int).isin(NEW_SEEDS)
        & pred_df["horizon"].astype(str).isin(HORIZONS)
    ].copy()
    subset["mechanism"] = subset["mechanism"].astype(str)
    subset["horizon"] = subset["horizon"].astype(str)
    subset["seed"] = subset["seed"].astype(int)
    subset["anchor_date"] = _canonicalize_anchor_date(subset["anchor_date"])

    cell_sizes = subset.groupby(["mechanism", "horizon", "seed"]).size()
    if len(cell_sizes) != len(MECHANISMS) * len(NEW_SEEDS) * len(HORIZONS):
        raise RuntimeError(
            f"New Experiment 4 cell count mismatch (fail closed): expected "
            f"{len(MECHANISMS) * len(NEW_SEEDS) * len(HORIZONS)}, got {len(cell_sizes)}."
        )
    if (cell_sizes != N_CELL_ROWS).any():
        raise RuntimeError("New Experiment 4 cell row count mismatch (fail closed): not all cells have 6647 rows.")

    return subset[["mechanism", "horizon", "seed", "ticker", "anchor_date", "y_true", "y_pred"]].reset_index(drop=True)


def _extract_adaptive_seed42_cells(pred_df: pd.DataFrame) -> pd.DataFrame:
    horizon_numeric = pd.to_numeric(pred_df["horizon"], errors="coerce")
    subset = pred_df[
        (pred_df["frequency_setting"].astype(str) == FREQUENCY_CONFIGURATION)
        & (pd.to_numeric(pred_df["seed"], errors="coerce").astype("Int64") == FROZEN_EXTERNAL_SEED)
        & (horizon_numeric.isin(HORIZON_LABEL_MAP.keys()))
    ].copy()
    subset["horizon"] = horizon_numeric.loc[subset.index].astype(int).map(HORIZON_LABEL_MAP)
    subset["mechanism"] = "adaptive"
    subset["seed"] = int(FROZEN_EXTERNAL_SEED)
    subset["anchor_date"] = _canonicalize_anchor_date(subset["anchor_date"])

    cell_sizes = subset.groupby("horizon").size()
    if len(cell_sizes) != len(HORIZONS):
        raise RuntimeError(f"Adaptive seed42 cell count mismatch (fail closed): expected {len(HORIZONS)}, got {len(cell_sizes)}.")
    if (cell_sizes != N_CELL_ROWS).any():
        raise RuntimeError("Adaptive seed42 cell row count mismatch (fail closed): not all cells have 6647 rows.")

    return subset[["mechanism", "horizon", "seed", "ticker", "anchor_date", "y_true", "y_pred"]].reset_index(drop=True)


def _extract_static_seed42_cells(pred_df: pd.DataFrame) -> pd.DataFrame:
    subset = pred_df[
        (pred_df["mechanism"].astype(str) == "static")
        & (pd.to_numeric(pred_df["seed"], errors="coerce").astype("Int64") == FROZEN_EXTERNAL_SEED)
        & (pred_df["horizon"].astype(str).isin(HORIZONS))
    ].copy()
    subset["mechanism"] = "static"
    subset["horizon"] = subset["horizon"].astype(str)
    subset["seed"] = int(FROZEN_EXTERNAL_SEED)
    subset["anchor_date"] = _canonicalize_anchor_date(subset["anchor_date"])

    cell_sizes = subset.groupby("horizon").size()
    if len(cell_sizes) != len(HORIZONS):
        raise RuntimeError(f"Static seed42 cell count mismatch (fail closed): expected {len(HORIZONS)}, got {len(cell_sizes)}.")
    if (cell_sizes != N_CELL_ROWS).any():
        raise RuntimeError("Static seed42 cell row count mismatch (fail closed): not all cells have 6647 rows.")

    return subset[["mechanism", "horizon", "seed", "ticker", "anchor_date", "y_true", "y_pred"]].reset_index(drop=True)


def _build_canonical_predictions() -> pd.DataFrame:
    experiment4_df = _load_csv(EXPERIMENT4_TEST_PREDICTIONS_PATH)
    adaptive_seed42_df = _load_csv(ADAPTIVE_SEED42_PREDICTIONS_PATH)
    static_seed42_df = _load_csv(STATIC_SEED42_PREDICTIONS_PATH)

    new_cells = _extract_new_experiment4_cells(experiment4_df)
    adaptive_cells = _extract_adaptive_seed42_cells(adaptive_seed42_df)
    static_cells = _extract_static_seed42_cells(static_seed42_df)

    canonical = pd.concat([new_cells, adaptive_cells, static_cells], ignore_index=True)

    cell_identity = canonical[["mechanism", "horizon", "seed"]].drop_duplicates()
    if len(cell_identity) != len(MECHANISMS) * len(HORIZONS) * len(SEEDS):
        raise RuntimeError(
            f"Canonical cell identity count mismatch (fail closed): expected "
            f"{len(MECHANISMS) * len(HORIZONS) * len(SEEDS)}, got {len(cell_identity)}."
        )
    all_cell_sizes_ok = (
        canonical.groupby(["mechanism", "horizon", "seed"])
        .size()
        .eq(N_CELL_ROWS)
        .all()
    )
    if not bool(all_cell_sizes_ok):
        raise RuntimeError("Canonical cell row counts mismatch (fail closed): not all 30 cells have 6647 rows.")

    return canonical


def _get_cell(canonical_df: pd.DataFrame, mechanism: str, horizon: str, seed: int) -> pd.DataFrame:
    cell = canonical_df[
        (canonical_df["mechanism"] == mechanism)
        & (canonical_df["horizon"] == horizon)
        & (canonical_df["seed"] == int(seed))
    ].copy()
    if len(cell) != N_CELL_ROWS:
        raise RuntimeError(f"Cell {mechanism}/{horizon}/{seed} row count mismatch (fail closed): {len(cell)} != {N_CELL_ROWS}.")

    if not np.isfinite(cell["y_true"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Non-finite y_true in cell {mechanism}/{horizon}/{seed} (fail closed).")
    if not np.isfinite(cell["y_pred"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Non-finite y_pred in cell {mechanism}/{horizon}/{seed} (fail closed).")
    if cell["ticker"].isna().any():
        raise RuntimeError(f"Cell {mechanism}/{horizon}/{seed} contains null ticker keys (fail closed).")

    cell = cell.sort_values(["ticker", "anchor_date"]).reset_index(drop=True)

    if cell["ticker"].nunique() != N_TICKERS:
        raise RuntimeError(f"Cell {mechanism}/{horizon}/{seed} ticker count mismatch (fail closed): {cell['ticker'].nunique()} != {N_TICKERS}.")
    if cell["anchor_date"].nunique() != N_TEST_DATES:
        raise RuntimeError(f"Cell {mechanism}/{horizon}/{seed} date count mismatch (fail closed): {cell['anchor_date'].nunique()} != {N_TEST_DATES}.")
    if cell.duplicated(subset=["ticker", "anchor_date"]).any():
        raise RuntimeError(f"Cell {mechanism}/{horizon}/{seed} has duplicate ticker/anchor_date keys (fail closed).")

    return cell


# ---------------------------------------------------------------------------
# Bridge audits (reconstructed descriptive evidence vs frozen evidence)
# ---------------------------------------------------------------------------


def _reconstruct_pooled_mse(cell: pd.DataFrame) -> float:
    y_true32 = cell["y_true"].to_numpy(dtype=np.float32)
    y_pred32 = cell["y_pred"].to_numpy(dtype=np.float32)
    return float(np.mean((y_pred32 - y_true32) ** 2))


def _run_30_cell_bridge_audit(canonical_df: pd.DataFrame) -> dict:
    combined_metrics_df = _load_csv(COMBINED_CELL_METRICS_PATH)
    reconstructed_mse: dict = {}

    for mechanism in MECHANISMS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                cell = _get_cell(canonical_df, mechanism, horizon, seed)
                pooled_mse = _reconstruct_pooled_mse(cell)
                reconstructed_mse[(mechanism, horizon, seed)] = pooled_mse

                frozen_rows = combined_metrics_df[
                    (combined_metrics_df["mechanism"].astype(str) == mechanism)
                    & (combined_metrics_df["horizon"].astype(str) == horizon)
                    & (combined_metrics_df["seed"].astype(int) == int(seed))
                ]
                if len(frozen_rows) != 1:
                    raise RuntimeError(
                        f"30-cell bridge audit (fail closed): expected exactly one frozen row for "
                        f"{mechanism}/{horizon}/{seed}; found {len(frozen_rows)}."
                    )
                frozen_mse = float(frozen_rows.iloc[0]["pooled_mse"])
                if not np.isclose(pooled_mse, frozen_mse, rtol=AUDIT_RTOL, atol=AUDIT_ATOL):
                    raise RuntimeError(
                        f"30-cell bridge audit mismatch (fail closed) for {mechanism}/{horizon}/{seed}: "
                        f"reconstructed={pooled_mse}, frozen={frozen_mse}."
                    )

    if len(reconstructed_mse) != len(MECHANISMS) * len(HORIZONS) * len(SEEDS):
        raise RuntimeError("30-cell bridge audit produced an unexpected number of reconstructed cells (fail closed).")

    return reconstructed_mse


def _run_15_delta_bridge_audit(reconstructed_mse: dict) -> None:
    seed_pair_deltas_df = _load_csv(SEED_PAIR_DELTAS_PATH)
    checked = 0
    for horizon in HORIZONS:
        for seed in SEEDS:
            adaptive_mse = reconstructed_mse[("adaptive", horizon, seed)]
            static_mse = reconstructed_mse[("static", horizon, seed)]
            delta_mse = adaptive_mse - static_mse

            frozen_rows = seed_pair_deltas_df[
                (seed_pair_deltas_df["horizon"].astype(str) == horizon)
                & (seed_pair_deltas_df["seed"].astype(int) == int(seed))
            ]
            if len(frozen_rows) != 1:
                raise RuntimeError(
                    f"15-delta bridge audit (fail closed): expected exactly one frozen row for "
                    f"horizon={horizon}, seed={seed}; found {len(frozen_rows)}."
                )
            frozen_delta = float(frozen_rows.iloc[0]["delta_mse"])
            if not np.isclose(delta_mse, frozen_delta, rtol=AUDIT_RTOL, atol=AUDIT_ATOL):
                raise RuntimeError(
                    f"15-delta bridge audit mismatch (fail closed) for horizon={horizon}, seed={seed}: "
                    f"reconstructed={delta_mse}, frozen={frozen_delta}."
                )
            checked += 1

    if checked != len(HORIZONS) * len(SEEDS):
        raise RuntimeError("15-delta bridge audit did not check exactly 15 pairs (fail closed).")


# ---------------------------------------------------------------------------
# Pairability audit + elementwise loss differential
# ---------------------------------------------------------------------------


def _pairability_check(adaptive_cell: pd.DataFrame, static_cell: pd.DataFrame, horizon: str, seed: int) -> None:
    adaptive_keys = list(zip(adaptive_cell["ticker"].tolist(), adaptive_cell["anchor_date"].tolist()))
    static_keys = list(zip(static_cell["ticker"].tolist(), static_cell["anchor_date"].tolist()))
    if adaptive_keys != static_keys:
        raise RuntimeError(f"Pairability audit key mismatch (fail closed) for horizon={horizon}, seed={seed}.")

    y_true_adaptive = adaptive_cell["y_true"].to_numpy(dtype=np.float32)
    y_true_static = static_cell["y_true"].to_numpy(dtype=np.float32)
    if not np.array_equal(y_true_adaptive, y_true_static):
        if not np.allclose(
            y_true_adaptive,
            y_true_static,
            rtol=0.0,
            atol=FLOAT32_ATOL,
            equal_nan=False,
        ):
            raise RuntimeError(f"Pairability audit y_true mismatch (fail closed) for horizon={horizon}, seed={seed}.")

    date_counts = adaptive_cell.groupby("anchor_date").size()
    if (date_counts != N_TICKERS).any():
        raise RuntimeError(f"Pairability audit: not exactly {N_TICKERS} tickers on every date for horizon={horizon}, seed={seed}.")


def _cross_seed_date_consistency_check(canonical_df: pd.DataFrame) -> None:
    for horizon in HORIZONS:
        reference_dates = None
        for mechanism in MECHANISMS:
            for seed in SEEDS:
                cell = _get_cell(canonical_df, mechanism, horizon, seed)
                dates = sorted(cell["anchor_date"].unique().tolist())
                if reference_dates is None:
                    reference_dates = dates
                elif dates != reference_dates:
                    raise RuntimeError(
                        f"Cross-seed/mechanism anchor_date mismatch (fail closed) for horizon={horizon}, "
                        f"mechanism={mechanism}, seed={seed}."
                    )


def _build_loss_diff_by_seed_date(canonical_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            adaptive_cell = _get_cell(canonical_df, "adaptive", horizon, seed)
            static_cell = _get_cell(canonical_df, "static", horizon, seed)
            _pairability_check(adaptive_cell, static_cell, horizon, seed)

            y_true32 = adaptive_cell["y_true"].to_numpy(dtype=np.float32)
            adaptive_pred32 = adaptive_cell["y_pred"].to_numpy(dtype=np.float32)
            static_pred32 = static_cell["y_pred"].to_numpy(dtype=np.float32)

            adaptive_error = (y_true32 - adaptive_pred32) ** 2
            static_error = (y_true32 - static_pred32) ** 2
            loss_diff32 = adaptive_error - static_error
            loss_diff64 = loss_diff32.astype(np.float64)

            if not np.isfinite(adaptive_error).all():
                raise RuntimeError(f"Non-finite adaptive squared error (fail closed) for horizon={horizon}, seed={seed}.")
            if not np.isfinite(static_error).all():
                raise RuntimeError(f"Non-finite static squared error (fail closed) for horizon={horizon}, seed={seed}.")
            if not np.isfinite(loss_diff32).all():
                raise RuntimeError(f"Non-finite float32 loss differential (fail closed) for horizon={horizon}, seed={seed}.")
            if not np.isfinite(loss_diff64).all():
                raise RuntimeError(f"Non-finite float64 loss differential (fail closed) for horizon={horizon}, seed={seed}.")

            per_row = pd.DataFrame(
                {
                    "anchor_date": adaptive_cell["anchor_date"].to_numpy(),
                    "loss_diff": loss_diff64,
                }
            )
            grouped = per_row.groupby("anchor_date")["loss_diff"].agg(["mean", "count"]).reset_index()
            if (grouped["count"] != N_TICKERS).any():
                raise RuntimeError(f"Seed-date aggregation ticker-count mismatch (fail closed) for horizon={horizon}, seed={seed}.")
            if len(grouped) != N_TEST_DATES:
                raise RuntimeError(f"Seed-date aggregation date-count mismatch (fail closed) for horizon={horizon}, seed={seed}.")

            grouped["horizon"] = horizon
            grouped["seed"] = int(seed)
            grouped = grouped.rename(columns={"mean": "mean_loss_diff", "count": "n_tickers"})
            rows.append(grouped[["horizon", "seed", "anchor_date", "mean_loss_diff", "n_tickers"]])

    result = pd.concat(rows, ignore_index=True)
    if len(result) != len(HORIZONS) * len(SEEDS) * N_TEST_DATES:
        raise RuntimeError("loss_diff_by_seed_date row count mismatch (fail closed).")

    result["horizon"] = pd.Categorical(result["horizon"], categories=CANONICAL_HORIZON_ORDER, ordered=True)
    result["seed"] = pd.Categorical(result["seed"], categories=CANONICAL_SEED_ORDER, ordered=True)
    result = result.sort_values(["horizon", "seed", "anchor_date"]).reset_index(drop=True)
    result["horizon"] = result["horizon"].astype(str)
    result["seed"] = result["seed"].astype(int)
    return result


def _build_loss_diff_by_date(seed_date_df: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_date_df.groupby(["horizon", "anchor_date"])["mean_loss_diff"].agg(["mean", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "mean_loss_diff", "count": "n_seeds"})
    if (grouped["n_seeds"] != len(SEEDS)).any():
        raise RuntimeError("Date-level aggregation seed-count mismatch (fail closed).")

    per_horizon_dates = grouped.groupby("horizon").size()
    if (per_horizon_dates != N_TEST_DATES).any():
        raise RuntimeError("Date-level aggregation date-count mismatch (fail closed).")

    grouped["horizon"] = pd.Categorical(grouped["horizon"], categories=CANONICAL_HORIZON_ORDER, ordered=True)
    grouped = grouped.sort_values(["horizon", "anchor_date"]).reset_index(drop=True)
    grouped["horizon"] = grouped["horizon"].astype(str)

    if len(grouped) != len(HORIZONS) * N_TEST_DATES:
        raise RuntimeError("loss_diff_by_date row count mismatch (fail closed).")

    return grouped[["horizon", "anchor_date", "mean_loss_diff", "n_seeds"]]


def _run_final_point_estimate_audit(date_df: pd.DataFrame) -> dict:
    theta_hat_by_horizon: dict = {}
    for horizon in HORIZONS:
        series = date_df[date_df["horizon"] == horizon].sort_values("anchor_date")["mean_loss_diff"].to_numpy(dtype=np.float64)
        if len(series) != N_TEST_DATES:
            raise RuntimeError(f"Point-estimate audit series length mismatch (fail closed) for horizon={horizon}.")
        theta_hat = float(series.mean())
        frozen_value = FROZEN_MEAN_DELTA_MSE[horizon]
        if not np.isclose(theta_hat, frozen_value, rtol=AUDIT_RTOL, atol=AUDIT_ATOL):
            raise RuntimeError(
                f"Final point-estimate audit mismatch (fail closed) for horizon={horizon}: "
                f"theta_hat={theta_hat}, frozen={frozen_value}."
            )
        theta_hat_by_horizon[horizon] = theta_hat
    return theta_hat_by_horizon


# ---------------------------------------------------------------------------
# Moving Block Bootstrap (PRIMARY) + Holm helper
# ---------------------------------------------------------------------------


def _moving_block_bootstrap(series: np.ndarray, block_length: int, rng: np.random.Generator, replications: int) -> np.ndarray:
    T = series.shape[0]
    L = int(block_length)
    n_starts = T - L + 1
    if n_starts < 1:
        raise RuntimeError(f"Invalid block length {L} for series length {T} (fail closed).")

    K = math.ceil(T / L)
    starts = rng.integers(0, n_starts, size=(replications, K))
    offsets = np.arange(L)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx_flat = idx.reshape(replications, K * L)[:, :T]
    sampled = series[idx_flat]
    return sampled.mean(axis=1, dtype=np.float64)


def _mbb_classification(ci_lower: float, ci_upper: float) -> str:
    if ci_lower > 0:
        return "MBB_STATIC_LOWER_LOSS"
    if ci_upper < 0:
        return "MBB_ADAPTIVE_LOWER_LOSS"
    return "MBB_NO_DISTINGUISHABLE_DIFFERENCE"


def _holm_adjust_three(pvalues: dict) -> dict:
    if len(pvalues) != 3:
        raise RuntimeError("Holm adjustment requires exactly 3 p-values (fail closed).")
    order = sorted(pvalues.keys(), key=lambda h: pvalues[h])
    m = len(order)
    running_max = 0.0
    adjusted = {}
    for rank, horizon in enumerate(order, start=1):
        candidate = (m - rank + 1) * pvalues[horizon]
        running_max = max(running_max, candidate)
        adjusted[horizon] = min(running_max, 1.0)
    return adjusted


def _run_bootstrap(date_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_rows = []
    summary_rows = []
    primary_p_values: dict = {}

    specs = [
        ("PRIMARY", PRIMARY_BLOCK_LENGTH),
        ("SENSITIVITY", SENSITIVITY_BLOCK_LENGTH),
    ]

    bootstrap_by_spec: dict = {}

    for horizon in HORIZONS:
        series = date_df[date_df["horizon"] == horizon].sort_values("anchor_date")["mean_loss_diff"].to_numpy(dtype=np.float64)
        theta_hat = float(series.mean())

        for spec_name, block_length_map in specs:
            block_length = block_length_map[horizon]
            horizon_int = HORIZON_INTS[horizon]
            seed_sequence = np.random.SeedSequence([BOOTSTRAP_BASE_SEED, horizon_int, block_length])
            rng = np.random.default_rng(seed_sequence)

            bootstrap_means = _moving_block_bootstrap(series, block_length, rng, BOOTSTRAP_B)
            if not np.isfinite(bootstrap_means).all():
                raise RuntimeError(f"Non-finite bootstrap statistic (fail closed) for {horizon}/{spec_name}.")

            bootstrap_by_spec[(horizon, spec_name)] = bootstrap_means

            for replicate_id, value in enumerate(bootstrap_means):
                replicate_rows.append(
                    {
                        "horizon": horizon,
                        "block_length": block_length,
                        "specification": spec_name,
                        "replicate": replicate_id,
                        "bootstrap_mean_loss_diff": float(value),
                    }
                )

            bootstrap_mean = float(np.mean(bootstrap_means))
            bootstrap_se = float(np.std(bootstrap_means, ddof=1))
            ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")
            ci_lower = float(ci_lower)
            ci_upper = float(ci_upper)
            centered_p_value = (1 + int(np.sum(np.abs(bootstrap_means - theta_hat) >= abs(theta_hat)))) / (BOOTSTRAP_B + 1)
            classification = _mbb_classification(ci_lower, ci_upper)

            if spec_name == "PRIMARY":
                primary_p_values[horizon] = centered_p_value

            summary_rows.append(
                {
                    "horizon": horizon,
                    "block_length": block_length,
                    "specification": spec_name,
                    "n_dates": N_TEST_DATES,
                    "n_bootstrap": BOOTSTRAP_B,
                    "theta_hat": theta_hat,
                    "bootstrap_mean": bootstrap_mean,
                    "bootstrap_se": bootstrap_se,
                    "ci_lower_95": ci_lower,
                    "ci_upper_95": ci_upper,
                    "centered_p_value": centered_p_value,
                    "classification": classification,
                }
            )

    primary_holm = _holm_adjust_three(primary_p_values)

    for row in summary_rows:
        if row["specification"] == "PRIMARY":
            row["centered_p_value_holm_primary"] = primary_holm[row["horizon"]]
        else:
            row["centered_p_value_holm_primary"] = float("nan")

    replicates_df = pd.DataFrame(replicate_rows)
    replicates_df["horizon"] = pd.Categorical(replicates_df["horizon"], categories=CANONICAL_HORIZON_ORDER, ordered=True)
    replicates_df["specification"] = pd.Categorical(replicates_df["specification"], categories=CANONICAL_SPEC_ORDER, ordered=True)
    replicates_df = replicates_df.sort_values(["horizon", "specification", "replicate"]).reset_index(drop=True)
    replicates_df["horizon"] = replicates_df["horizon"].astype(str)
    replicates_df["specification"] = replicates_df["specification"].astype(str)

    if len(replicates_df) != len(HORIZONS) * len(specs) * BOOTSTRAP_B:
        raise RuntimeError("Bootstrap replicate row count mismatch (fail closed).")

    summary_df = pd.DataFrame(summary_rows)
    summary_df["horizon"] = pd.Categorical(summary_df["horizon"], categories=CANONICAL_HORIZON_ORDER, ordered=True)
    summary_df["specification"] = pd.Categorical(summary_df["specification"], categories=CANONICAL_SPEC_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["horizon", "specification"]).reset_index(drop=True)
    summary_df["horizon"] = summary_df["horizon"].astype(str)
    summary_df["specification"] = summary_df["specification"].astype(str)

    if len(summary_df) != len(HORIZONS) * len(specs):
        raise RuntimeError("Bootstrap summary row count mismatch (fail closed).")

    summary_df = summary_df[
        [
            "horizon",
            "block_length",
            "specification",
            "n_dates",
            "n_bootstrap",
            "theta_hat",
            "bootstrap_mean",
            "bootstrap_se",
            "ci_lower_95",
            "ci_upper_95",
            "centered_p_value",
            "centered_p_value_holm_primary",
            "classification",
        ]
    ]

    return replicates_df, summary_df


# ---------------------------------------------------------------------------
# Secondary HAC analysis
# ---------------------------------------------------------------------------


def _hac_classification(p_value_raw: float, theta_hat: float) -> str:
    if p_value_raw < ALPHA and theta_hat > 0:
        return "SECONDARY_HAC_STATIC_LOWER_LOSS"
    if p_value_raw < ALPHA and theta_hat < 0:
        return "SECONDARY_HAC_ADAPTIVE_LOWER_LOSS"
    return "SECONDARY_HAC_NO_DISTINGUISHABLE_DIFFERENCE"


def _run_hac(date_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_p_values: dict = {}

    for horizon in HORIZONS:
        series = date_df[date_df["horizon"] == horizon].sort_values("anchor_date")["mean_loss_diff"].to_numpy(dtype=np.float64)
        T = series.shape[0]
        q = HAC_MAX_LAG[horizon]

        mu = float(series.mean())
        centered = series - mu
        gamma0 = float(np.sum(centered * centered) / T)
        lrv = gamma0
        for lag in range(1, q + 1):
            gamma_l = float(np.sum(centered[lag:] * centered[:-lag]) / T)
            weight = 1.0 - lag / (q + 1)
            lrv += 2.0 * weight * gamma_l

        if not np.isfinite(lrv) or lrv <= 0:
            raise RuntimeError(f"HAC long-run variance invalid (fail closed) for horizon={horizon}: LRV={lrv}.")

        se_mean = math.sqrt(lrv / T)
        if not np.isfinite(se_mean) or se_mean <= 0:
            raise RuntimeError(f"HAC standard error invalid (fail closed) for horizon={horizon}: se_mean={se_mean}.")

        z_stat = mu / se_mean
        p_value_raw = math.erfc(abs(z_stat) / math.sqrt(2.0))
        if not np.isfinite(z_stat) or not np.isfinite(p_value_raw):
            raise RuntimeError(f"HAC statistic non-finite (fail closed) for horizon={horizon}.")

        raw_p_values[horizon] = p_value_raw
        rows.append(
            {
                "horizon": horizon,
                "n_dates": T,
                "max_lag": q,
                "theta_hat": mu,
                "long_run_variance": lrv,
                "se_mean": se_mean,
                "z_stat": z_stat,
                "p_value_raw": p_value_raw,
                "classification_raw": _hac_classification(p_value_raw, mu),
            }
        )

    holm_adjusted = _holm_adjust_three(raw_p_values)
    for row in rows:
        row["p_value_holm"] = holm_adjusted[row["horizon"]]

    hac_df = pd.DataFrame(rows)
    hac_df["horizon"] = pd.Categorical(hac_df["horizon"], categories=CANONICAL_HORIZON_ORDER, ordered=True)
    hac_df = hac_df.sort_values("horizon").reset_index(drop=True)
    hac_df["horizon"] = hac_df["horizon"].astype(str)

    if len(hac_df) != len(HORIZONS):
        raise RuntimeError("HAC summary row count mismatch (fail closed).")

    return hac_df[
        [
            "horizon",
            "n_dates",
            "max_lag",
            "theta_hat",
            "long_run_variance",
            "se_mean",
            "z_stat",
            "p_value_raw",
            "p_value_holm",
            "classification_raw",
        ]
    ]


# ---------------------------------------------------------------------------
# Output serialization + strict output-frame validators
#
# These validators are applied both to in-memory/output-ready frames and to
# the frames re-read from the serialized temporary CSV files, so a partial
# or malformed serialization can never slip past validation undetected.
# ---------------------------------------------------------------------------


def _to_output_frame_seed_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["anchor_date"])
    if dates.isna().any():
        raise RuntimeError("anchor_date contains NaT before output serialization (fail closed).")
    out["anchor_date"] = dates.dt.strftime("%Y-%m-%d")
    return out[["horizon", "seed", "anchor_date", "mean_loss_diff", "n_tickers"]]


def _to_output_frame_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["anchor_date"])
    if dates.isna().any():
        raise RuntimeError("anchor_date contains NaT before output serialization (fail closed).")
    out["anchor_date"] = dates.dt.strftime("%Y-%m-%d")
    return out[["horizon", "anchor_date", "mean_loss_diff", "n_seeds"]]


def _require_sorted_keys(keys: list, label: str) -> None:
    if keys != sorted(keys):
        raise RuntimeError(f"Output ordering contract violated (fail closed): {label}.")


def _validate_seed_date_output(df: pd.DataFrame) -> None:
    expected_cols = ["horizon", "seed", "anchor_date", "mean_loss_diff", "n_tickers"]
    if list(df.columns) != expected_cols:
        raise RuntimeError(f"seed_date output schema mismatch (fail closed): {list(df.columns)}")
    if len(df) != EXPECTED_OUTPUT_ROW_COUNTS["seed_date"]:
        raise RuntimeError(f"seed_date output row count mismatch (fail closed): {len(df)}")
    if not df["horizon"].isin(HORIZONS).all():
        raise RuntimeError("seed_date output contains invalid horizon values (fail closed).")
    if not df["seed"].isin(SEEDS).all():
        raise RuntimeError("seed_date output contains invalid seed values (fail closed).")
    if not (df["n_tickers"] == N_TICKERS).all():
        raise RuntimeError("seed_date output n_tickers mismatch (fail closed).")
    if not np.isfinite(df["mean_loss_diff"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("seed_date output mean_loss_diff non-finite (fail closed).")

    parsed_dates = pd.to_datetime(df["anchor_date"], format="%Y-%m-%d", errors="raise")
    if parsed_dates.isna().any():
        raise RuntimeError("seed_date output anchor_date invalid (fail closed).")
    if not (df["anchor_date"].astype(str) == parsed_dates.dt.strftime("%Y-%m-%d")).all():
        raise RuntimeError("seed_date output anchor_date is not exact YYYY-MM-DD (fail closed).")

    horizon_rank = {h: i for i, h in enumerate(CANONICAL_HORIZON_ORDER)}
    seed_rank = {s: i for i, s in enumerate(CANONICAL_SEED_ORDER)}
    keys = list(zip(df["horizon"].map(horizon_rank), df["seed"].map(seed_rank), df["anchor_date"].astype(str)))
    _require_sorted_keys(keys, "experiment4_loss_diff_by_seed_date.csv")


def _validate_date_output(df: pd.DataFrame) -> None:
    expected_cols = ["horizon", "anchor_date", "mean_loss_diff", "n_seeds"]
    if list(df.columns) != expected_cols:
        raise RuntimeError(f"date output schema mismatch (fail closed): {list(df.columns)}")
    if len(df) != EXPECTED_OUTPUT_ROW_COUNTS["date"]:
        raise RuntimeError(f"date output row count mismatch (fail closed): {len(df)}")
    if not df["horizon"].isin(HORIZONS).all():
        raise RuntimeError("date output contains invalid horizon values (fail closed).")
    if not (df["n_seeds"] == len(SEEDS)).all():
        raise RuntimeError("date output n_seeds mismatch (fail closed).")
    if not np.isfinite(df["mean_loss_diff"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("date output mean_loss_diff non-finite (fail closed).")

    parsed_dates = pd.to_datetime(df["anchor_date"], format="%Y-%m-%d", errors="raise")
    if parsed_dates.isna().any():
        raise RuntimeError("date output anchor_date invalid (fail closed).")
    if not (df["anchor_date"].astype(str) == parsed_dates.dt.strftime("%Y-%m-%d")).all():
        raise RuntimeError("date output anchor_date is not exact YYYY-MM-DD (fail closed).")

    horizon_rank = {h: i for i, h in enumerate(CANONICAL_HORIZON_ORDER)}
    keys = list(zip(df["horizon"].map(horizon_rank), df["anchor_date"].astype(str)))
    _require_sorted_keys(keys, "experiment4_loss_diff_by_date.csv")


def _validate_replicates_output(df: pd.DataFrame) -> None:
    expected_cols = ["horizon", "block_length", "specification", "replicate", "bootstrap_mean_loss_diff"]
    if list(df.columns) != expected_cols:
        raise RuntimeError(f"replicates output schema mismatch (fail closed): {list(df.columns)}")
    if len(df) != EXPECTED_OUTPUT_ROW_COUNTS["replicates"]:
        raise RuntimeError(f"replicates output row count mismatch (fail closed): {len(df)}")
    if not df["horizon"].isin(HORIZONS).all():
        raise RuntimeError("replicates output contains invalid horizon values (fail closed).")
    if not df["specification"].isin(CANONICAL_SPEC_ORDER).all():
        raise RuntimeError("replicates output contains invalid specification values (fail closed).")
    if not np.isfinite(df["bootstrap_mean_loss_diff"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("replicates output bootstrap_mean_loss_diff non-finite (fail closed).")

    for horizon in HORIZONS:
        expected_block_lengths = {"PRIMARY": PRIMARY_BLOCK_LENGTH[horizon], "SENSITIVITY": SENSITIVITY_BLOCK_LENGTH[horizon]}
        for spec_name, expected_length in expected_block_lengths.items():
            group = df[(df["horizon"] == horizon) & (df["specification"] == spec_name)]
            if len(group) != BOOTSTRAP_B:
                raise RuntimeError(f"replicates output group size mismatch (fail closed) for {horizon}/{spec_name}.")
            if not (group["block_length"] == expected_length).all():
                raise RuntimeError(f"replicates output block_length mismatch (fail closed) for {horizon}/{spec_name}.")
            replicate_ids = group["replicate"].to_numpy(dtype=np.int64)
            if len(np.unique(replicate_ids)) != BOOTSTRAP_B:
                raise RuntimeError(f"replicates output duplicate/missing replicate IDs (fail closed) for {horizon}/{spec_name}.")
            if not np.array_equal(np.sort(replicate_ids), np.arange(BOOTSTRAP_B)):
                raise RuntimeError(f"replicates output replicate IDs are not exactly 0..19999 (fail closed) for {horizon}/{spec_name}.")

    horizon_rank = {h: i for i, h in enumerate(CANONICAL_HORIZON_ORDER)}
    spec_rank = {s: i for i, s in enumerate(CANONICAL_SPEC_ORDER)}
    keys = list(
        zip(
            df["horizon"].map(horizon_rank),
            df["specification"].map(spec_rank),
            df["replicate"].astype(np.int64),
        )
    )
    _require_sorted_keys(keys, "experiment4_block_bootstrap_replicates.csv")


_VALID_MBB_CLASSIFICATIONS = {"MBB_STATIC_LOWER_LOSS", "MBB_ADAPTIVE_LOWER_LOSS", "MBB_NO_DISTINGUISHABLE_DIFFERENCE"}
_VALID_HAC_CLASSIFICATIONS = {
    "SECONDARY_HAC_STATIC_LOWER_LOSS",
    "SECONDARY_HAC_ADAPTIVE_LOWER_LOSS",
    "SECONDARY_HAC_NO_DISTINGUISHABLE_DIFFERENCE",
}


def _validate_bootstrap_summary_output(df: pd.DataFrame) -> None:
    expected_cols = [
        "horizon",
        "block_length",
        "specification",
        "n_dates",
        "n_bootstrap",
        "theta_hat",
        "bootstrap_mean",
        "bootstrap_se",
        "ci_lower_95",
        "ci_upper_95",
        "centered_p_value",
        "centered_p_value_holm_primary",
        "classification",
    ]
    if list(df.columns) != expected_cols:
        raise RuntimeError(f"bootstrap_summary output schema mismatch (fail closed): {list(df.columns)}")
    if len(df) != EXPECTED_OUTPUT_ROW_COUNTS["bootstrap_summary"]:
        raise RuntimeError(f"bootstrap_summary output row count mismatch (fail closed): {len(df)}")

    required_finite_cols = ["theta_hat", "bootstrap_mean", "bootstrap_se", "ci_lower_95", "ci_upper_95", "centered_p_value"]
    for col in required_finite_cols:
        if not np.isfinite(df[col].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"bootstrap_summary output {col} non-finite (fail closed).")

    if not (df["n_dates"] == N_TEST_DATES).all():
        raise RuntimeError("bootstrap_summary output n_dates mismatch (fail closed).")
    if not (df["n_bootstrap"] == BOOTSTRAP_B).all():
        raise RuntimeError("bootstrap_summary output n_bootstrap mismatch (fail closed).")
    if not df["classification"].isin(_VALID_MBB_CLASSIFICATIONS).all():
        raise RuntimeError("bootstrap_summary output contains an invalid classification (fail closed).")

    for _, row in df.iterrows():
        horizon = row["horizon"]
        spec = row["specification"]
        if horizon not in HORIZONS or spec not in CANONICAL_SPEC_ORDER:
            raise RuntimeError("bootstrap_summary output contains an invalid horizon/specification (fail closed).")
        expected_length = PRIMARY_BLOCK_LENGTH[horizon] if spec == "PRIMARY" else SENSITIVITY_BLOCK_LENGTH[horizon]
        if int(row["block_length"]) != expected_length:
            raise RuntimeError(f"bootstrap_summary output block_length mismatch (fail closed) for {horizon}/{spec}.")

        recomputed_classification = _mbb_classification(float(row["ci_lower_95"]), float(row["ci_upper_95"]))
        if recomputed_classification != row["classification"]:
            raise RuntimeError(f"bootstrap_summary output classification mismatch (fail closed) for {horizon}/{spec}.")

        holm_value = row["centered_p_value_holm_primary"]
        if spec == "PRIMARY":
            if pd.isna(holm_value) or not np.isfinite(float(holm_value)):
                raise RuntimeError(f"bootstrap_summary output centered_p_value_holm_primary must be finite for PRIMARY (fail closed): {horizon}.")
        else:
            if not pd.isna(holm_value):
                raise RuntimeError(f"bootstrap_summary output centered_p_value_holm_primary must be NaN for SENSITIVITY (fail closed): {horizon}.")

    horizon_rank = {h: i for i, h in enumerate(CANONICAL_HORIZON_ORDER)}
    spec_rank = {s: i for i, s in enumerate(CANONICAL_SPEC_ORDER)}
    keys = list(zip(df["horizon"].map(horizon_rank), df["specification"].map(spec_rank)))
    _require_sorted_keys(keys, "experiment4_block_bootstrap_summary.csv")


def _validate_hac_summary_output(df: pd.DataFrame) -> None:
    expected_cols = [
        "horizon",
        "n_dates",
        "max_lag",
        "theta_hat",
        "long_run_variance",
        "se_mean",
        "z_stat",
        "p_value_raw",
        "p_value_holm",
        "classification_raw",
    ]
    if list(df.columns) != expected_cols:
        raise RuntimeError(f"hac_summary output schema mismatch (fail closed): {list(df.columns)}")
    if len(df) != EXPECTED_OUTPUT_ROW_COUNTS["hac_summary"]:
        raise RuntimeError(f"hac_summary output row count mismatch (fail closed): {len(df)}")

    required_finite_cols = ["theta_hat", "long_run_variance", "se_mean", "z_stat", "p_value_raw", "p_value_holm"]
    for col in required_finite_cols:
        if not np.isfinite(df[col].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"hac_summary output {col} non-finite (fail closed).")

    if not (df["n_dates"] == N_TEST_DATES).all():
        raise RuntimeError("hac_summary output n_dates mismatch (fail closed).")
    if not df["classification_raw"].isin(_VALID_HAC_CLASSIFICATIONS).all():
        raise RuntimeError("hac_summary output contains an invalid classification_raw (fail closed).")

    for _, row in df.iterrows():
        horizon = row["horizon"]
        if horizon not in HORIZONS:
            raise RuntimeError("hac_summary output contains an invalid horizon (fail closed).")
        if int(row["max_lag"]) != HAC_MAX_LAG[horizon]:
            raise RuntimeError(f"hac_summary output max_lag mismatch (fail closed) for {horizon}.")
        recomputed_classification = _hac_classification(float(row["p_value_raw"]), float(row["theta_hat"]))
        if recomputed_classification != row["classification_raw"]:
            raise RuntimeError(f"hac_summary output classification_raw mismatch (fail closed) for {horizon}.")

    horizon_rank = {h: i for i, h in enumerate(CANONICAL_HORIZON_ORDER)}
    keys = list(df["horizon"].map(horizon_rank))
    _require_sorted_keys(keys, "experiment4_hac_summary.csv")


def _validate_summary_text_output(actual_text: str, expected_text: str) -> None:
    if not actual_text:
        raise RuntimeError("Inference summary text is empty (fail closed).")
    if actual_text != expected_text:
        raise RuntimeError("Inference summary text does not exactly match the deterministic in-memory summary (fail closed).")


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _validate_manifest_output(df: pd.DataFrame, expected_output_hashes: dict, expected_row_counts: dict) -> None:
    if len(df) != 1:
        raise RuntimeError(f"Manifest must contain exactly one row (fail closed): found {len(df)}.")
    row = df.iloc[0]

    if str(row["protocol_sha256"]) != PROTOCOL_SHA256:
        raise RuntimeError("Manifest protocol_sha256 mismatch (fail closed).")
    if not _as_bool(row["no_training"]):
        raise RuntimeError("Manifest no_training must be True (fail closed).")
    if not _as_bool(row["no_checkpoint_loading"]):
        raise RuntimeError("Manifest no_checkpoint_loading must be True (fail closed).")
    if not _as_bool(row["no_fresh_test_predictions"]):
        raise RuntimeError("Manifest no_fresh_test_predictions must be True (fail closed).")
    if not _as_bool(row["no_router_analysis"]):
        raise RuntimeError("Manifest no_router_analysis must be True (fail closed).")

    row_count_fields = {
        "output_1_loss_diff_by_seed_date_rows": expected_row_counts["seed_date"],
        "output_2_loss_diff_by_date_rows": expected_row_counts["date"],
        "output_3_block_bootstrap_replicates_rows": expected_row_counts["replicates"],
        "output_4_block_bootstrap_summary_rows": expected_row_counts["bootstrap_summary"],
        "output_5_hac_summary_rows": expected_row_counts["hac_summary"],
    }
    for field, expected_value in row_count_fields.items():
        if int(row[field]) != expected_value:
            raise RuntimeError(f"Manifest row count field mismatch (fail closed): {field}.")

    for field, expected_hash in expected_output_hashes.items():
        if str(row[field]) != expected_hash:
            raise RuntimeError(f"Manifest output hash field mismatch (fail closed): {field}.")


# ---------------------------------------------------------------------------
# Summary text + manifest
# ---------------------------------------------------------------------------

_MBB_NL_STATEMENTS = {
    "MBB_STATIC_LOWER_LOSS": (
        "Static has statistically lower expected squared forecast loss than "
        "Adaptive under the primary date-block bootstrap specification."
    ),
    "MBB_ADAPTIVE_LOWER_LOSS": (
        "Adaptive has statistically lower expected squared forecast loss than "
        "Static under the primary date-block bootstrap specification."
    ),
    "MBB_NO_DISTINGUISHABLE_DIFFERENCE": (
        "No statistically distinguishable difference in expected squared forecast "
        "loss is established under the primary date-block bootstrap specification."
    ),
}


def _build_summary_text(bootstrap_summary_df: pd.DataFrame, hac_df: pd.DataFrame) -> str:
    lines = [
        "Experiment 4 serial-dependence-aware inference summary",
        "",
        f"Protocol: docs/experiment4_inference_protocol.md",
        f"Protocol SHA256: {PROTOCOL_SHA256}",
        "Inference is conditional on the five pre-specified model seeds 0, 1, 21, 42, 3407.",
        "Sign convention: delta_mse = Adaptive MSE - Static MSE (theta_hat > 0 => Static lower loss; theta_hat < 0 => Adaptive lower loss).",
        "Primary method: Moving Block Bootstrap (MBB) on the chronological 391-date series per horizon.",
        f"Primary block lengths (L=h): {PRIMARY_BLOCK_LENGTH}",
        f"Sensitivity block lengths (L=2h): {SENSITIVITY_BLOCK_LENGTH}",
        f"Bootstrap replications: B={BOOTSTRAP_B}, base RNG seed={BOOTSTRAP_BASE_SEED}",
        "Confidence interval: two-sided 95% percentile bootstrap CI.",
        "HAC (Newey-West-style, Bartlett-weighted) is a SECONDARY diagnostic and never overrides the PRIMARY MBB conclusion.",
        "",
    ]

    for horizon in HORIZONS:
        primary_row = bootstrap_summary_df[
            (bootstrap_summary_df["horizon"] == horizon) & (bootstrap_summary_df["specification"] == "PRIMARY")
        ].iloc[0]
        sensitivity_row = bootstrap_summary_df[
            (bootstrap_summary_df["horizon"] == horizon) & (bootstrap_summary_df["specification"] == "SENSITIVITY")
        ].iloc[0]
        hac_row = hac_df[hac_df["horizon"] == horizon].iloc[0]

        lines.append(f"Horizon {horizon}:")
        lines.append(
            f"  PRIMARY: theta_hat={primary_row['theta_hat']:.10g}, "
            f"CI=[{primary_row['ci_lower_95']:.10g}, {primary_row['ci_upper_95']:.10g}], "
            f"centered_p_raw={primary_row['centered_p_value']:.10g}, "
            f"centered_p_holm_primary={primary_row['centered_p_value_holm_primary']:.10g}, "
            f"classification={primary_row['classification']}"
        )
        lines.append(f"  {_MBB_NL_STATEMENTS[primary_row['classification']]}")
        lines.append(
            f"  SENSITIVITY: CI=[{sensitivity_row['ci_lower_95']:.10g}, {sensitivity_row['ci_upper_95']:.10g}], "
            f"centered_p_raw={sensitivity_row['centered_p_value']:.10g}, "
            f"classification={sensitivity_row['classification']}"
        )
        if primary_row["classification"] != sensitivity_row["classification"]:
            lines.append("  the inference is sensitive to the pre-specified block-length choice.")

        lines.append(
            f"  HAC (SECONDARY): z={hac_row['z_stat']:.10g}, p_raw={hac_row['p_value_raw']:.10g}, "
            f"p_holm={hac_row['p_value_holm']:.10g}, classification_raw={hac_row['classification_raw']}"
        )
        if primary_row["classification"].replace("MBB_", "") != hac_row["classification_raw"].replace("SECONDARY_HAC_", ""):
            lines.append("  MBB (PRIMARY) and HAC (SECONDARY) classifications disagree for this horizon; the disagreement is reported without selecting either as authoritative.")

        lines.append("")

    lines.extend(
        [
            "Adaptive has greater parameter capacity than Static; this capacity caveat is retained.",
            "Router / scale-weight interpretation remains gated until after this inference stage is finalized.",
            "This inference applies only to expected squared forecast loss differences; it does not establish "
            "profitability, economic utility, causality, universal architecture superiority, universal failure of "
            "Adaptive routing, or router economic regime discovery.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_manifest(
    canonical_row_count: int,
    seed_date_rows: int,
    date_rows: int,
    replicate_rows: int,
    bootstrap_summary_rows: int,
    hac_rows: int,
    output_hashes: dict,
) -> pd.DataFrame:
    record = {
        "run_family": "EXPERIMENT4_SERIAL_DEPENDENCE_AWARE_INFERENCE",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract": "Dataset Contract V2",
        "primary_contrast": "adaptive_vs_static",
        "frequency_configuration": FREQUENCY_CONFIGURATION,
        "horizons": ",".join(HORIZONS),
        "seeds": ",".join(str(s) for s in SEEDS),
        "n_tickers": N_TICKERS,
        "n_test_dates": N_TEST_DATES,
        "inference_scope": "conditional_on_five_prespecified_seeds",
        "loss_definition": "squared_error_adaptive_minus_squared_error_static",
        "sign_convention": "delta_mse_equals_adaptive_mse_minus_static_mse",
        "primary_method": "moving_block_bootstrap",
        "primary_block_lengths": str(PRIMARY_BLOCK_LENGTH),
        "sensitivity_block_lengths": str(SENSITIVITY_BLOCK_LENGTH),
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "confidence_level": 0.95,
        "ci_method": "two_sided_percentile_bootstrap",
        "bootstrap_se_ddof": 1,
        "quantile_method": "linear",
        "HAC_role": "secondary",
        "HAC_lags": str(HAC_MAX_LAG),
        "HAC_kernel": "bartlett",
        "alpha": ALPHA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": PROTOCOL_SHA256,
        "canonical_prediction_rows": canonical_row_count,
        "no_training": True,
        "no_checkpoint_loading": True,
        "no_fresh_test_predictions": True,
        "no_router_analysis": True,
        "output_1_loss_diff_by_seed_date_rows": seed_date_rows,
        "output_2_loss_diff_by_date_rows": date_rows,
        "output_3_block_bootstrap_replicates_rows": replicate_rows,
        "output_4_block_bootstrap_summary_rows": bootstrap_summary_rows,
        "output_5_hac_summary_rows": hac_rows,
    }

    for path, expected_hash in FROZEN_INPUT_HASHES.items():
        key = path.stem.replace("-", "_")
        record[f"frozen_input_{key}_path"] = str(path.relative_to(ROOT))
        record[f"frozen_input_{key}_sha256"] = expected_hash

    record.update(output_hashes)

    return pd.DataFrame([record])


# ---------------------------------------------------------------------------
# Transactional output writing
# ---------------------------------------------------------------------------


def _write_outputs(
    seed_date_df: pd.DataFrame,
    date_df: pd.DataFrame,
    replicates_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    hac_df: pd.DataFrame,
    canonical_row_count: int,
) -> None:
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    (
        loss_diff_by_seed_date_path,
        loss_diff_by_date_path,
        block_bootstrap_replicates_path,
        block_bootstrap_summary_path,
        hac_summary_path,
        inference_summary_path,
        inference_manifest_path,
    ) = FINAL_OUTPUTS

    run_id = uuid.uuid4().hex
    temp_paths = {
        "seed_date": loss_diff_by_seed_date_path.with_name(f"{loss_diff_by_seed_date_path.stem}.{run_id}.tmp{loss_diff_by_seed_date_path.suffix}"),
        "date": loss_diff_by_date_path.with_name(f"{loss_diff_by_date_path.stem}.{run_id}.tmp{loss_diff_by_date_path.suffix}"),
        "replicates": block_bootstrap_replicates_path.with_name(f"{block_bootstrap_replicates_path.stem}.{run_id}.tmp{block_bootstrap_replicates_path.suffix}"),
        "bootstrap_summary": block_bootstrap_summary_path.with_name(f"{block_bootstrap_summary_path.stem}.{run_id}.tmp{block_bootstrap_summary_path.suffix}"),
        "hac_summary": hac_summary_path.with_name(f"{hac_summary_path.stem}.{run_id}.tmp{hac_summary_path.suffix}"),
        "summary_text": inference_summary_path.with_name(f"{inference_summary_path.stem}.{run_id}.tmp{inference_summary_path.suffix}"),
        "manifest": inference_manifest_path.with_name(f"{inference_manifest_path.stem}.{run_id}.tmp{inference_manifest_path.suffix}"),
    }
    final_paths = {
        "seed_date": loss_diff_by_seed_date_path,
        "date": loss_diff_by_date_path,
        "replicates": block_bootstrap_replicates_path,
        "bootstrap_summary": block_bootstrap_summary_path,
        "hac_summary": hac_summary_path,
        "summary_text": inference_summary_path,
        "manifest": inference_manifest_path,
    }

    committed_final_paths: list[Path] = []

    try:
        # Explicit YYYY-MM-DD output serialization; internal computation above
        # used datetime objects for chronological sorting, never lexical strings.
        seed_date_output = _to_output_frame_seed_date(seed_date_df)
        date_output = _to_output_frame_date(date_df)

        # Validate the in-memory/output-ready frames before any serialization.
        _validate_seed_date_output(seed_date_output)
        _validate_date_output(date_output)
        _validate_replicates_output(replicates_df)
        _validate_bootstrap_summary_output(bootstrap_summary_df)
        _validate_hac_summary_output(hac_df)

        summary_text = _build_summary_text(bootstrap_summary_df, hac_df)

        seed_date_output.to_csv(temp_paths["seed_date"], index=False)
        date_output.to_csv(temp_paths["date"], index=False)
        replicates_df.to_csv(temp_paths["replicates"], index=False)
        bootstrap_summary_df.to_csv(temp_paths["bootstrap_summary"], index=False)
        hac_df.to_csv(temp_paths["hac_summary"], index=False)
        temp_paths["summary_text"].write_text(summary_text, encoding="utf-8")

        for key in ("seed_date", "date", "replicates", "bootstrap_summary", "hac_summary", "summary_text"):
            if not temp_paths[key].exists():
                raise RuntimeError(f"Temporary inference output missing before commit: {temp_paths[key]}")

        # Re-read the SERIALIZED temporary CSVs and validate the round-tripped
        # contents; validating only the pre-serialization frames is not sufficient.
        _validate_seed_date_output(pd.read_csv(temp_paths["seed_date"]))
        _validate_date_output(pd.read_csv(temp_paths["date"]))
        _validate_replicates_output(pd.read_csv(temp_paths["replicates"]))
        _validate_bootstrap_summary_output(pd.read_csv(temp_paths["bootstrap_summary"]))
        _validate_hac_summary_output(pd.read_csv(temp_paths["hac_summary"]))
        _validate_summary_text_output(temp_paths["summary_text"].read_text(encoding="utf-8"), summary_text)

        output_hashes = {
            "output_1_loss_diff_by_seed_date_sha256": _sha256_file(temp_paths["seed_date"]),
            "output_2_loss_diff_by_date_sha256": _sha256_file(temp_paths["date"]),
            "output_3_block_bootstrap_replicates_sha256": _sha256_file(temp_paths["replicates"]),
            "output_4_block_bootstrap_summary_sha256": _sha256_file(temp_paths["bootstrap_summary"]),
            "output_5_hac_summary_sha256": _sha256_file(temp_paths["hac_summary"]),
            "output_6_inference_summary_sha256": _sha256_file(temp_paths["summary_text"]),
        }

        manifest_df = _build_manifest(
            canonical_row_count=canonical_row_count,
            seed_date_rows=len(seed_date_output),
            date_rows=len(date_output),
            replicate_rows=len(replicates_df),
            bootstrap_summary_rows=len(bootstrap_summary_df),
            hac_rows=len(hac_df),
            output_hashes=output_hashes,
        )
        _validate_manifest_output(manifest_df, output_hashes, EXPECTED_OUTPUT_ROW_COUNTS)

        manifest_df.to_csv(temp_paths["manifest"], index=False)
        if not temp_paths["manifest"].exists():
            raise RuntimeError(f"Temporary manifest missing before commit: {temp_paths['manifest']}")
        _validate_manifest_output(pd.read_csv(temp_paths["manifest"]), output_hashes, EXPECTED_OUTPUT_ROW_COUNTS)

        for tmp in temp_paths.values():
            if not tmp.exists():
                raise RuntimeError(f"Temporary inference output missing before commit: {tmp}")

        os.replace(temp_paths["seed_date"], final_paths["seed_date"])
        committed_final_paths.append(final_paths["seed_date"])
        os.replace(temp_paths["date"], final_paths["date"])
        committed_final_paths.append(final_paths["date"])
        os.replace(temp_paths["replicates"], final_paths["replicates"])
        committed_final_paths.append(final_paths["replicates"])
        os.replace(temp_paths["bootstrap_summary"], final_paths["bootstrap_summary"])
        committed_final_paths.append(final_paths["bootstrap_summary"])
        os.replace(temp_paths["hac_summary"], final_paths["hac_summary"])
        committed_final_paths.append(final_paths["hac_summary"])
        os.replace(temp_paths["summary_text"], final_paths["summary_text"])
        committed_final_paths.append(final_paths["summary_text"])
        os.replace(temp_paths["manifest"], final_paths["manifest"])
        committed_final_paths.append(final_paths["manifest"])

        for path in FINAL_OUTPUTS:
            if not path.exists():
                raise RuntimeError(f"Post-commit verification failed: final output missing: {path}")

        post_commit_hashes = {
            final_paths["seed_date"]: output_hashes["output_1_loss_diff_by_seed_date_sha256"],
            final_paths["date"]: output_hashes["output_2_loss_diff_by_date_sha256"],
            final_paths["replicates"]: output_hashes["output_3_block_bootstrap_replicates_sha256"],
            final_paths["bootstrap_summary"]: output_hashes["output_4_block_bootstrap_summary_sha256"],
            final_paths["hac_summary"]: output_hashes["output_5_hac_summary_sha256"],
            final_paths["summary_text"]: output_hashes["output_6_inference_summary_sha256"],
        }
        for path, expected_sha in post_commit_hashes.items():
            actual_sha = _sha256_file(path)
            if actual_sha != expected_sha:
                raise RuntimeError(f"Post-commit hash mismatch (fail closed) for {path}.")

        # Post-commit: re-validate the committed outputs, not just their hashes.
        _validate_seed_date_output(pd.read_csv(final_paths["seed_date"]))
        _validate_date_output(pd.read_csv(final_paths["date"]))
        _validate_replicates_output(pd.read_csv(final_paths["replicates"]))
        _validate_bootstrap_summary_output(pd.read_csv(final_paths["bootstrap_summary"]))
        _validate_hac_summary_output(pd.read_csv(final_paths["hac_summary"]))
        _validate_summary_text_output(final_paths["summary_text"].read_text(encoding="utf-8"), summary_text)
        _validate_manifest_output(pd.read_csv(final_paths["manifest"]), output_hashes, EXPECTED_OUTPUT_ROW_COUNTS)
    except Exception:
        # Delete every temp path that currently exists, regardless of which
        # write step reached completion before the exception was raised.
        for tmp in temp_paths.values():
            if tmp.exists():
                tmp.unlink()
        # Roll back only final paths committed by THIS invocation; never touch
        # any other pre-existing/frozen evidence.
        for final_path in committed_final_paths:
            if final_path.exists():
                final_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_inference() -> None:
    _require_final_outputs_absent()
    _verify_frozen_input_hashes()

    canonical_df = _build_canonical_predictions()
    reconstructed_mse = _run_30_cell_bridge_audit(canonical_df)
    _run_15_delta_bridge_audit(reconstructed_mse)
    _cross_seed_date_consistency_check(canonical_df)

    seed_date_df = _build_loss_diff_by_seed_date(canonical_df)
    date_df = _build_loss_diff_by_date(seed_date_df)
    _run_final_point_estimate_audit(date_df)

    replicates_df, bootstrap_summary_df = _run_bootstrap(date_df)
    hac_df = _run_hac(date_df)

    _write_outputs(
        seed_date_df=seed_date_df,
        date_df=date_df,
        replicates_df=replicates_df,
        bootstrap_summary_df=bootstrap_summary_df,
        hac_df=hac_df,
        canonical_row_count=len(canonical_df),
    )

    print("EXPERIMENT 4 SERIAL-DEPENDENCE-AWARE INFERENCE COMPLETE")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 4 serial-dependence-aware inference runner (frozen protocol; read-only frozen evidence)."
    )
    parser.add_argument("--run-inference", action="store_true", help="Run the frozen Experiment 4 inference protocol.")
    args = parser.parse_args()

    if not args.run_inference:
        print(
            "Experiment 4 inference runner contract: verify frozen protocol/input hashes, reconstruct descriptive "
            "evidence as bridge audits, build paired Adaptive-vs-Static squared-loss differentials, run the frozen "
            "Moving Block Bootstrap (PRIMARY) and secondary HAC analysis, and write the frozen inference output "
            "family under dataset/audit/experiment4_inference/. No training, checkpoint loading, or fresh TEST "
            "inference is performed. Use --run-inference to execute."
        )
        return 0

    _run_inference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
