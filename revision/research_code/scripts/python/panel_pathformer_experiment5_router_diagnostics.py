"""Experiment 5 router diagnostics runner.

Experiment 5 is diagnostics only:
- no model execution;
- no checkpoint loading;
- no fresh TEST prediction generation;
- no statistical hypothesis testing;
- frozen protocol SHA must match before execution.

Protocol SHA256: 0aa176bac8d21220a6cf22045ad64557bfd8dcec20fc0d3ad47638d70be429fe

This runner reads only already-persisted frozen TEST evidence and writes only to
`dataset/audit/experiment5_router_diagnostics/`.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "docs" / "experiment5_router_diagnostics_protocol.md"
PROTOCOL_SHA256 = "0aa176bac8d21220a6cf22045ad64557bfd8dcec20fc0d3ad47638d70be429fe"

OUTPUT_DIR = ROOT / "dataset" / "audit" / "experiment5_router_diagnostics"

HORIZONS = ["5d", "10d", "20d"]
HORIZON_ORDER = {"5d": 0, "10d": 1, "20d": 2}
FREQUENCY_ORDER = {"daily": 0, "weekly": 1}
PATCH_ORDER_MAP = {5: 0, 10: 1, 20: 2, 30: 3, 2: 0, 4: 1, 8: 2, 13: 3}
SEEDS = [0, 1, 21, 42, 3407]
NEW_SEEDS = [0, 1, 21, 3407]
MECHANISMS = ["adaptive", "static"]
FREQUENCIES = ["daily", "weekly"]
DAILY_PATCHES = [5, 10, 20, 30]
WEEKLY_PATCHES = [2, 4, 8, 13]
N_TICKERS = 17
N_TEST_DATES = 391
N_CELL_ROWS = 6647
N_WEIGHT_CELLS = 30
N_CANONICAL_WEIGHT_ROWS = 199410
WEIGHT_SUM_ATOL = 1e-6
STATIC_COMPONENT_RANGE_TOL = 1e-10
LOSS_BRIDGE_RTOL = 1e-8
LOSS_BRIDGE_ATOL = 1e-8
FLOAT32_EPS = np.finfo(np.float32).eps
RANGE_TOL = 1e-9
N_SAMPLE_METRICS_ROWS = 199410
N_CROSS_SEED_STABILITY_ROWS = 39882
ANCHOR_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

FROZEN_MEAN_DELTA_MSE = {
    "5d": 0.0003479534760117531,
    "10d": 0.0005263369530439377,
    "20d": 0.002097966521978378,
}

# Frozen exact-column schemas for Outputs 1-7. Validators require exact column-list equality.
EXACT_OUTPUT_1_COLUMNS = [
    "horizon", "seed", "ticker", "anchor_date", "frequency",
    "patch_1", "patch_2", "patch_3", "patch_4",
    "weight_1", "weight_2", "weight_3", "weight_4",
    "entropy", "normalized_entropy", "effective_scales", "max_weight", "dominant_patch",
    "tv_to_cell_mean", "tv_to_static", "loss_diff",
]
EXACT_OUTPUT_2_COLUMNS = [
    "horizon", "seed", "frequency", "n_samples", "n_tickers", "n_dates",
    "mean_weight_1", "mean_weight_2", "mean_weight_3", "mean_weight_4",
    "sd_weight_1", "sd_weight_2", "sd_weight_3", "sd_weight_4",
    "sample_dependence_index", "within_ticker_temporal_tv", "within_date_crosssectional_tv",
    "normalized_entropy_mean", "normalized_entropy_sd", "normalized_entropy_median", "normalized_entropy_p10", "normalized_entropy_p90",
    "effective_scales_mean", "effective_scales_sd", "effective_scales_median", "effective_scales_p10", "effective_scales_p90",
    "max_weight_mean", "max_weight_sd", "max_weight_median", "max_weight_p10", "max_weight_p90",
    "static_weight_1", "static_weight_2", "static_weight_3", "static_weight_4",
    "tv_cell_mean_to_static",
    "tv_to_static_mean", "tv_to_static_sd", "tv_to_static_median", "tv_to_static_p10", "tv_to_static_p90",
]
EXACT_OUTPUT_3_COLUMNS = [
    "horizon", "seed", "frequency", "patch_size", "dominant_count", "dominant_share",
]
EXACT_OUTPUT_4_COLUMNS = [
    "horizon", "ticker", "anchor_date", "frequency",
    "n_seeds", "mean_pairwise_seed_tv", "max_pairwise_seed_tv",
    "dominant_agreement_share", "dominant_unanimous",
]
EXACT_OUTPUT_5_COLUMNS = [
    "horizon", "frequency", "n_samples",
    "mean_pairwise_seed_tv_mean", "mean_pairwise_seed_tv_sd", "mean_pairwise_seed_tv_median", "mean_pairwise_seed_tv_p90",
    "max_pairwise_seed_tv_mean", "max_pairwise_seed_tv_median", "max_pairwise_seed_tv_p90",
    "dominant_agreement_share_mean", "dominant_agreement_share_median", "dominant_unanimous_share",
]
EXACT_OUTPUT_6_COLUMNS = [
    "horizon", "seed", "frequency", "n_samples", "n_dates",
    "sample_level_spearman_rho", "sample_level_correlation_defined",
    "date_level_spearman_rho", "date_level_correlation_defined",
]
EXACT_OUTPUT_7_COLUMNS = [
    "horizon", "frequency",
    "sample_level_rho_mean", "sample_level_rho_sd", "sample_level_rho_median",
    "sample_level_n_defined", "sample_level_n_positive", "sample_level_n_negative", "sample_level_n_zero",
    "date_level_rho_mean", "date_level_rho_sd", "date_level_rho_median",
    "date_level_n_defined", "date_level_n_positive", "date_level_n_negative", "date_level_n_zero",
]

# Documented frozen inputs and hashes.
FROZEN_INPUTS = {
    "dataset/audit/experiment4_multiseed/experiment4_scale_weights.csv": {
        "path": ROOT / "dataset" / "audit" / "experiment4_multiseed" / "experiment4_scale_weights.csv",
        "sha256": "ba1fd9280341dcf68d65ffe7c52675a69d77f1c214e5458cbb17e6ffc6fdb887",
    },
    "dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv": {
        "path": ROOT / "dataset" / "audit" / "experiment4_multiseed" / "experiment4_test_predictions.csv",
        "sha256": "6b5f5ea94f1bb5f25ebd5ed09d7e4e7eb609945f97f953e7fb95116e2e866e8a",
    },
    "dataset/audit/main_v2/pathformer/panel_pathformer_router_weights.csv": {
        "path": ROOT / "dataset" / "audit" / "main_v2" / "pathformer" / "panel_pathformer_router_weights.csv",
        "sha256": "bcf06150f21d35d9ec10132d248d088cc758c9782771efdf9b68849531c9814e",
    },
    "dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv": {
        "path": ROOT / "dataset" / "audit" / "main_v2" / "pathformer" / "panel_pathformer_test_predictions.csv",
        "sha256": "a05955a561da1b8880f1bb0878a6576852c087cc45707546d446ce96b0fabdaf",
    },
    "dataset/audit/experiment3_pathformer_ablation/experiment3_scale_weights.csv": {
        "path": ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation" / "experiment3_scale_weights.csv",
        "sha256": "219e908f0c3f75bd9b38d7fe466242a15a64d8c41b1f235ae75c91df5f6eb248",
    },
    "dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv": {
        "path": ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation" / "experiment3_test_predictions.csv",
        "sha256": "95d1b06cc3ce0209e99833c29a1c3672a6d7a1a9d46d32f16ef1e48fa9846515",
    },
    "dataset/audit/experiment4_inference/experiment4_inference_manifest.csv": {
        "path": ROOT / "dataset" / "audit" / "experiment4_inference" / "experiment4_inference_manifest.csv",
        "sha256": "93801b778f7ef1d60794b2777ffa5b983cb1bc03a2ef425df5551dbe7358f89e",
    },
    "dataset/audit/experiment4_inference/experiment4_inference_summary.txt": {
        "path": ROOT / "dataset" / "audit" / "experiment4_inference" / "experiment4_inference_summary.txt",
        "sha256": "3f3e2f67097ba2859374832ca06386e03e6a0620776a9919559a1d3ab0c0c35b",
    },
    "dataset/audit/experiment4_multiseed/experiment4_seed_pair_deltas.csv": {
        "path": ROOT / "dataset" / "audit" / "experiment4_multiseed" / "experiment4_seed_pair_deltas.csv",
        "sha256": "2f2a66689ac6abc8371565c2d8567c5ffa08259e5cb4879dd1fb306645c7f103",
    },
}

FINAL_OUTPUTS = (
    OUTPUT_DIR / "experiment5_router_sample_metrics.csv",
    OUTPUT_DIR / "experiment5_router_cell_summary.csv",
    OUTPUT_DIR / "experiment5_dominant_scale_usage.csv",
    OUTPUT_DIR / "experiment5_cross_seed_stability_by_sample.csv",
    OUTPUT_DIR / "experiment5_cross_seed_stability_summary.csv",
    OUTPUT_DIR / "experiment5_loss_association_by_seed.csv",
    OUTPUT_DIR / "experiment5_loss_association_summary.csv",
    OUTPUT_DIR / "experiment5_router_diagnostics_summary.txt",
    OUTPUT_DIR / "experiment5_router_diagnostics_manifest.csv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required {label}: {path}")


def _require_absent(path: Path, label: str) -> None:
    if path.exists():
        raise RuntimeError(f"Fail closed: final output already exists for {label}: {path}")


def _verify_protocol() -> None:
    _require_exists(PROTOCOL_PATH, "protocol")
    actual = _sha256_file(PROTOCOL_PATH)
    if actual != PROTOCOL_SHA256:
        raise RuntimeError(
            "Fail closed: protocol SHA mismatch. "
            f"Expected {PROTOCOL_SHA256}, found {actual}."
        )


def _verify_frozen_inputs() -> None:
    for rel_path, meta in FROZEN_INPUTS.items():
        path = meta["path"]
        expected = meta["sha256"]
        _require_exists(path, rel_path)
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Fail closed: frozen input hash mismatch for {rel_path}: "
                f"expected {expected}, got {actual}"
            )


def _require_final_outputs_absent() -> None:
    for path in FINAL_OUTPUTS:
        _require_absent(path, path.name)


def _canonicalize_anchor_date(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="raise")
    if out.isna().any():
        raise RuntimeError("anchor_date contains NaT after datetime parsing.")
    return out


def _coerce_anchor_date(series: pd.Series, *, serialized: bool) -> pd.Series:
    if not serialized:
        return _canonicalize_anchor_date(series)
    str_values = series.astype(str)
    if not str_values.str.match(ANCHOR_DATE_REGEX).all():
        raise RuntimeError("anchor_date does not match the required ^\\d{4}-\\d{2}-\\d{2}$ serialized format.")
    parsed = pd.to_datetime(str_values, format="%Y-%m-%d", errors="raise")
    if parsed.isna().any():
        raise RuntimeError("anchor_date contains NaT after serialized datetime parsing.")
    round_trip = parsed.dt.strftime("%Y-%m-%d")
    if not (round_trip.to_numpy() == str_values.to_numpy()).all():
        raise RuntimeError("anchor_date serialized round-trip mismatch.")
    return parsed


def _coerce_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isnull().any():
        raise RuntimeError("Boolean-equivalent column contains an unrecognized value.")
    return mapped.to_numpy(dtype=bool)


def _sort_key_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
    return out


def _validate_numeric_vector(vec: np.ndarray, *, lower: float, upper: float, atol: float) -> None:
    if vec.shape[0] == 0:
        raise RuntimeError("Empty numeric vector encountered.")
    if not np.all(np.isfinite(vec)):
        raise RuntimeError("Vector contains non-finite entries.")
    if np.any(vec < lower - atol) or np.any(vec > upper + atol):
        raise RuntimeError(f"Vector out of range [{lower}, {upper}] with tolerance {atol}.")


def _tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise RuntimeError("TV distance requires matching vector shapes.")
    return float(0.5 * np.sum(np.abs(a - b), dtype=np.float64))


def _entropy_metrics(w: np.ndarray) -> tuple[float, float, float, float]:
    w = np.asarray(w, dtype=np.float64)
    if w.shape[0] != 4:
        raise RuntimeError("Entropy metrics require four-scale vectors.")
    positive = w[w > 0.0]
    h = float(-np.sum(positive * np.log(positive), dtype=np.float64))
    h_norm = float(h / math.log(4.0))
    n_eff = float(math.exp(h))
    max_weight = float(np.max(w))
    # dominant scale index/patch is frequency-dependent and resolved by the caller.
    return h, h_norm, n_eff, max_weight


def _average_rank(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="average")
    return ranks.astype(np.float64)


def _manual_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise RuntimeError("manual_spearman requires matching shapes.")
    if x.size == 0:
        return float("nan"), False
    rx = _average_rank(pd.Series(x)).to_numpy(dtype=np.float64)
    ry = _average_rank(pd.Series(y)).to_numpy(dtype=np.float64)
    rx_centered = rx - np.mean(rx, dtype=np.float64)
    ry_centered = ry - np.mean(ry, dtype=np.float64)
    denom = math.sqrt(float(np.sum(rx_centered ** 2, dtype=np.float64) * np.sum(ry_centered ** 2, dtype=np.float64)))
    if denom == 0.0:
        return float("nan"), False
    rho = float(np.sum(rx_centered * ry_centered, dtype=np.float64) / denom)
    if not np.isfinite(rho):
        return float("nan"), False
    if rho < -1.0 - 1e-12 or rho > 1.0 + 1e-12:
        raise RuntimeError(f"manual Spearman rho out of range: {rho}")
    return rho, True


def _mean_sd_median_p90(values: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean_v = float(np.mean(arr, dtype=np.float64))
    sd_v = float(np.std(arr, ddof=1, dtype=np.float64)) if arr.size > 1 else float("nan")
    median_v = float(np.median(arr))
    p90 = float(np.quantile(arr, 0.90, method="linear"))
    return mean_v, sd_v, median_v, p90


def _mean_median_p90(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean_v = float(np.mean(arr, dtype=np.float64))
    median_v = float(np.median(arr))
    p90 = float(np.quantile(arr, 0.90, method="linear"))
    return mean_v, median_v, p90


def _summarize_defined_rhos(rhos: np.ndarray) -> dict:
    arr = np.asarray(rhos, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    n_defined = int(valid.size)
    if n_defined == 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "median": float("nan"),
            "n_defined": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
        }
    if n_defined == 1:
        val = float(valid[0])
        return {
            "mean": val,
            "sd": float("nan"),
            "median": val,
            "n_defined": 1,
            "n_positive": int(val > 0.0),
            "n_negative": int(val < 0.0),
            "n_zero": int(val == 0.0),
        }
    pos = np.sum(valid > 0.0)
    neg = np.sum(valid < 0.0)
    zero = np.sum(valid == 0.0)
    mean_v = float(np.mean(valid, dtype=np.float64))
    sd_v = float(np.std(valid, ddof=1, dtype=np.float64))
    median_v = float(np.median(valid))
    return {
        "mean": mean_v,
        "sd": sd_v,
        "median": median_v,
        "n_defined": n_defined,
        "n_positive": int(pos),
        "n_negative": int(neg),
        "n_zero": int(zero),
    }


def _build_weight_table() -> pd.DataFrame:
    # 1) Experiment 4 new-seed scale weights.
    exp4 = pd.read_csv(ROOT / "dataset" / "audit" / "experiment4_multiseed" / "experiment4_scale_weights.csv")
    exp4 = exp4[
        exp4["mechanism"].isin(MECHANISMS)
        & exp4["seed"].isin(NEW_SEEDS)
        & exp4["horizon"].isin(HORIZONS)
    ].copy()
    exp4["mechanism"] = exp4["mechanism"].astype(str)
    exp4["horizon"] = exp4["horizon"].astype(str)
    exp4["seed"] = exp4["seed"].astype(int)
    exp4["anchor_date"] = _canonicalize_anchor_date(exp4["anchor_date"])
    exp4 = exp4[[
        "mechanism", "horizon", "seed", "ticker", "anchor_date",
        "daily_w_5", "daily_w_10", "daily_w_20", "daily_w_30",
        "weekly_w_2", "weekly_w_4", "weekly_w_8", "weekly_w_13",
    ]].copy()

    # 2) Adaptive seed42 router weights.
    adaptive42 = pd.read_csv(ROOT / "dataset" / "audit" / "main_v2" / "pathformer" / "panel_pathformer_router_weights.csv")
    adaptive42 = adaptive42[
        adaptive42["frequency_setting"].astype(str).eq("daily_weekly")
        & adaptive42["seed"].astype(int).eq(42)
        & pd.to_numeric(adaptive42["horizon"], errors="coerce").isin([5, 10, 20])
    ].copy()
    adaptive42["horizon"] = pd.to_numeric(adaptive42["horizon"], errors="coerce").astype(int).map({5: "5d", 10: "10d", 20: "20d"})
    adaptive42["mechanism"] = "adaptive"
    adaptive42["seed"] = 42
    adaptive42["anchor_date"] = _canonicalize_anchor_date(adaptive42["anchor_date"])
    adaptive42 = adaptive42[[
        "mechanism", "horizon", "seed", "ticker", "anchor_date",
        "daily_w_5", "daily_w_10", "daily_w_20", "daily_w_30",
        "weekly_w_2", "weekly_w_4", "weekly_w_8", "weekly_w_13",
    ]].copy()

    # 3) Static seed42 weights.
    static42 = pd.read_csv(ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation" / "experiment3_scale_weights.csv")
    static42 = static42[
        static42["mechanism"].astype(str).eq("static")
        & static42["seed"].astype(int).eq(42)
        & static42["horizon"].isin(HORIZONS)
    ].copy()
    static42["mechanism"] = "static"
    static42["seed"] = static42["seed"].astype(int)
    static42["horizon"] = static42["horizon"].astype(str)
    static42["anchor_date"] = _canonicalize_anchor_date(static42["anchor_date"])
    static42 = static42[[
        "mechanism", "horizon", "seed", "ticker", "anchor_date",
        "daily_w_5", "daily_w_10", "daily_w_20", "daily_w_30",
        "weekly_w_2", "weekly_w_4", "weekly_w_8", "weekly_w_13",
    ]].copy()

    df = pd.concat([exp4, adaptive42, static42], ignore_index=True)
    df = _sort_key_df(df)
    expected_cells = len(MECHANISMS) * len(HORIZONS) * len(SEEDS)
    cell_counts = df.groupby(["mechanism", "horizon", "seed"]).size()
    if len(cell_counts) != expected_cells:
        raise RuntimeError(f"Canonical weight table cell count mismatch: expected {expected_cells}, got {len(cell_counts)}")
    if not np.all(cell_counts.to_numpy() == N_CELL_ROWS):
        raise RuntimeError("Each canonical weight cell must have exactly 6647 rows.")
    if len(df) != N_CANONICAL_WEIGHT_ROWS:
        raise RuntimeError(f"Canonical weight rows mismatch: expected {N_CANONICAL_WEIGHT_ROWS}, got {len(df)}")

    # Key checks.
    for horizon in HORIZONS:
        for seed in SEEDS:
            adaptive_keys = df[(df["mechanism"] == "adaptive") & (df["horizon"] == horizon) & (df["seed"] == seed)][["ticker", "anchor_date"]].copy()
            static_keys = df[(df["mechanism"] == "static") & (df["horizon"] == horizon) & (df["seed"] == seed)][["ticker", "anchor_date"]].copy()
            if not adaptive_keys.reset_index(drop=True).equals(static_keys.reset_index(drop=True)):
                raise RuntimeError(f"Adaptive and Static key mismatch for horizon={horizon}, seed={seed}.")
            for mechanism in MECHANISMS:
                by_seed = df[(df["mechanism"] == mechanism) & (df["horizon"] == horizon) & (df["seed"] == seed)][["ticker", "anchor_date"]].copy()
                by_seed = by_seed.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
                if by_seed.duplicated(subset=["ticker", "anchor_date"]).any():
                    raise RuntimeError(f"Duplicate ticker/date key in {mechanism} {horizon} seed={seed}.")
                if by_seed["ticker"].nunique() != N_TICKERS:
                    raise RuntimeError(f"Unexpected ticker count for {mechanism} {horizon} seed={seed}.")
                dates = by_seed["anchor_date"].dt.normalize()
                if dates.nunique() != N_TEST_DATES:
                    raise RuntimeError(f"Unexpected date count for {mechanism} {horizon} seed={seed}.")
                per_date = by_seed.groupby("anchor_date").size()
                if not np.all(per_date.to_numpy() == N_TICKERS):
                    raise RuntimeError(f"Expected exactly 17 observations per date for {mechanism} {horizon} seed={seed}.")
            # Require identical ordered TEST keys across all seeds within horizon.
            ordered_keys = []
            for seed_chk in SEEDS:
                ordered_keys.append(
                    df[(df["horizon"] == horizon) & (df["seed"] == seed_chk)][["ticker", "anchor_date"]]
                    .sort_values(["ticker", "anchor_date"], kind="mergesort")
                    .reset_index(drop=True)
                )
            ref = ordered_keys[0].reset_index(drop=True)
            for idx in range(1, len(ordered_keys)):
                if not ref.equals(ordered_keys[idx].reset_index(drop=True)):
                    raise RuntimeError(f"Ordered TEST keys differ across seeds for horizon={horizon}.")

    # Weight validity.
    weight_cols = [
        "daily_w_5", "daily_w_10", "daily_w_20", "daily_w_30",
        "weekly_w_2", "weekly_w_4", "weekly_w_8", "weekly_w_13",
    ]
    for col in weight_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing required weight column: {col}")
    for idx, row in df.iterrows():
        daily = row[["daily_w_5", "daily_w_10", "daily_w_20", "daily_w_30"]].to_numpy(dtype=np.float64)
        weekly = row[["weekly_w_2", "weekly_w_4", "weekly_w_8", "weekly_w_13"]].to_numpy(dtype=np.float64)
        for vec, label in [(daily, "daily"), (weekly, "weekly")]:
            if not np.all(np.isfinite(vec)):
                raise RuntimeError(f"Non-finite weight vector in {label} for row {idx}.")
            if np.any(vec < 0.0) or np.any(vec > 1.0):
                raise RuntimeError(f"Weight out of [0,1] in {label} for row {idx}.")
            if abs(float(np.sum(vec, dtype=np.float64)) - 1.0) > WEIGHT_SUM_ATOL:
                raise RuntimeError(f"Weight sum out of tolerance in {label} for row {idx}: {np.sum(vec)}")
    return df


def _build_static_reference_vectors(weights_df: pd.DataFrame) -> dict:
    ref = {}
    for horizon in HORIZONS:
        for seed in SEEDS:
            for frequency in FREQUENCIES:
                subset = weights_df[
                    (weights_df["mechanism"] == "static")
                    & (weights_df["horizon"] == horizon)
                    & (weights_df["seed"] == seed)
                ].copy()
                vector_cols = DAILY_PATCHES if frequency == "daily" else WEEKLY_PATCHES
                vec_names = [f"daily_w_{p}" for p in DAILY_PATCHES] if frequency == "daily" else [f"weekly_w_{p}" for p in WEEKLY_PATCHES]
                arr = subset[vec_names].to_numpy(dtype=np.float64)
                if arr.shape[0] != N_CELL_ROWS:
                    raise RuntimeError(f"Static reference cell has wrong number of rows: {horizon} {seed} {frequency} -> {arr.shape[0]}")
                comp_range = np.max(arr, axis=0) - np.min(arr, axis=0)
                if np.any(comp_range > STATIC_COMPONENT_RANGE_TOL):
                    raise RuntimeError(f"Static sample-independence check failed for {horizon} {seed} {frequency}.")
                s = np.mean(arr, axis=0, dtype=np.float64)
                if not np.all(np.isfinite(s)):
                    raise RuntimeError(f"Static reference vector is non-finite for {horizon} {seed} {frequency}.")
                if np.any(s < 0.0) or np.any(s > 1.0):
                    raise RuntimeError(f"Static reference vector outside [0,1] for {horizon} {seed} {frequency}.")
                if abs(float(np.sum(s, dtype=np.float64)) - 1.0) > WEIGHT_SUM_ATOL:
                    raise RuntimeError(f"Static reference vector sum invalid for {horizon} {seed} {frequency}.")
                ref[(horizon, seed, frequency)] = s
    return ref


def _build_prediction_table() -> pd.DataFrame:
    adaptive_pred = pd.read_csv(ROOT / "dataset" / "audit" / "main_v2" / "pathformer" / "panel_pathformer_test_predictions.csv")
    static_pred = pd.read_csv(ROOT / "dataset" / "audit" / "experiment3_pathformer_ablation" / "experiment3_test_predictions.csv")
    exp4_pred = pd.read_csv(ROOT / "dataset" / "audit" / "experiment4_multiseed" / "experiment4_test_predictions.csv")

    # Source rules: Experiment 4 new seeds + seed42 external reuse.
    adaptive_seed42 = adaptive_pred[
        adaptive_pred["frequency_setting"].astype(str).eq("daily_weekly")
        & adaptive_pred["seed"].astype(int).eq(42)
        & pd.to_numeric(adaptive_pred["horizon"], errors="coerce").isin([5, 10, 20])
    ].copy()
    adaptive_seed42["horizon"] = pd.to_numeric(adaptive_seed42["horizon"], errors="coerce").astype(int).map({5: "5d", 10: "10d", 20: "20d"})
    adaptive_seed42["mechanism"] = "adaptive"
    adaptive_seed42["seed"] = 42
    adaptive_seed42["anchor_date"] = _canonicalize_anchor_date(adaptive_seed42["anchor_date"])

    static_seed42 = static_pred[
        static_pred["mechanism"].astype(str).eq("static")
        & static_pred["seed"].astype(int).eq(42)
        & static_pred["horizon"].isin(HORIZONS)
    ].copy()
    static_seed42["mechanism"] = "static"
    static_seed42["seed"] = 42
    static_seed42["anchor_date"] = _canonicalize_anchor_date(static_seed42["anchor_date"])

    exp4_new = exp4_pred[
        exp4_pred["mechanism"].isin(MECHANISMS)
        & exp4_pred["seed"].isin(NEW_SEEDS)
        & exp4_pred["horizon"].isin(HORIZONS)
    ].copy()
    exp4_new["mechanism"] = exp4_new["mechanism"].astype(str)
    exp4_new["horizon"] = exp4_new["horizon"].astype(str)
    exp4_new["seed"] = exp4_new["seed"].astype(int)
    exp4_new["anchor_date"] = _canonicalize_anchor_date(exp4_new["anchor_date"])

    pred_df = pd.concat([exp4_new, adaptive_seed42, static_seed42], ignore_index=True)
    pred_df = pred_df[["mechanism", "horizon", "seed", "ticker", "anchor_date", "y_true", "y_pred"]].copy()
    pred_df = pred_df.sort_values(["mechanism", "horizon", "seed", "ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)

    cell_sizes = pred_df.groupby(["mechanism", "horizon", "seed"]).size()
    if len(cell_sizes) != len(MECHANISMS) * len(HORIZONS) * len(SEEDS):
        raise RuntimeError("Prediction table has wrong number of cells.")
    if not np.all(cell_sizes.to_numpy() == N_CELL_ROWS):
        raise RuntimeError("Every prediction cell must have 6647 rows.")

    for horizon in HORIZONS:
        for seed in SEEDS:
            adaptive = pred_df[(pred_df["mechanism"] == "adaptive") & (pred_df["horizon"] == horizon) & (pred_df["seed"] == seed)].copy()
            static = pred_df[(pred_df["mechanism"] == "static") & (pred_df["horizon"] == horizon) & (pred_df["seed"] == seed)].copy()
            adaptive = adaptive.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
            static = static.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
            if not adaptive[["ticker", "anchor_date"]].reset_index(drop=True).equals(static[["ticker", "anchor_date"]].reset_index(drop=True)):
                raise RuntimeError(f"Adaptive and Static prediction keys differ for {horizon} seed {seed}.")
            if adaptive["ticker"].nunique() != N_TICKERS:
                raise RuntimeError(f"Unexpected ticker count in predictions for {horizon} seed {seed}.")
            if adaptive["anchor_date"].dt.normalize().nunique() != N_TEST_DATES:
                raise RuntimeError(f"Unexpected test date count in predictions for {horizon} seed {seed}.")
            if not np.all(adaptive.groupby("anchor_date").size().to_numpy() == N_TICKERS):
                raise RuntimeError(f"Prediction rows do not have exactly 17 per date for {horizon} seed {seed}.")
            if not np.all(np.isfinite(adaptive["y_true"])) or not np.all(np.isfinite(adaptive["y_pred"])):
                raise RuntimeError(f"Non-finite prediction values for adaptive {horizon} seed {seed}.")
            if not np.all(np.isfinite(static["y_true"])) or not np.all(np.isfinite(static["y_pred"])):
                raise RuntimeError(f"Non-finite prediction values for static {horizon} seed {seed}.")
            adaptive_y_true32 = adaptive["y_true"].to_numpy(dtype=np.float32)
            static_y_true32 = static["y_true"].to_numpy(dtype=np.float32)
            if not np.array_equal(adaptive_y_true32, static_y_true32):
                if not np.allclose(adaptive_y_true32, static_y_true32, rtol=0.0, atol=np.finfo(np.float32).eps, equal_nan=False):
                    raise RuntimeError(f"y_true float32 pairability failed for {horizon} seed {seed}.")
    return pred_df


def _build_loss_diffs(pred_df: pd.DataFrame) -> pd.DataFrame:
    combined = pred_df.copy()
    combined = combined.sort_values(["mechanism", "horizon", "seed", "ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
    loss_rows = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            adaptive = combined[(combined["mechanism"] == "adaptive") & (combined["horizon"] == horizon) & (combined["seed"] == seed)].copy().sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
            static = combined[(combined["mechanism"] == "static") & (combined["horizon"] == horizon) & (combined["seed"] == seed)].copy().sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
            if not adaptive[["ticker", "anchor_date"]].reset_index(drop=True).equals(static[["ticker", "anchor_date"]].reset_index(drop=True)):
                raise RuntimeError(f"Loss bridge key mismatch for {horizon} seed {seed}.")
            y_true32 = adaptive["y_true"].astype(np.float32).to_numpy(dtype=np.float32)
            adaptive_pred32 = adaptive["y_pred"].astype(np.float32).to_numpy(dtype=np.float32)
            static_pred32 = static["y_pred"].astype(np.float32).to_numpy(dtype=np.float32)
            adaptive_err = (y_true32 - adaptive_pred32) ** 2
            static_err = (y_true32 - static_pred32) ** 2
            loss_diff32 = adaptive_err - static_err
            loss_diff = loss_diff32.astype(np.float64)
            if not np.all(np.isfinite(adaptive_err)) or not np.all(np.isfinite(static_err)) or not np.all(np.isfinite(loss_diff)):
                raise RuntimeError(f"Non-finite loss bridge values for {horizon} seed {seed}.")
            adaptive = adaptive.assign(loss_diff=loss_diff)
            loss_rows.append(adaptive[["mechanism", "horizon", "seed", "ticker", "anchor_date", "loss_diff"]])
    loss_df = pd.concat(loss_rows, ignore_index=True)
    return loss_df


def _build_loss_bridge_audit(loss_df: pd.DataFrame) -> None:
    delta_file = pd.read_csv(ROOT / "dataset" / "audit" / "experiment4_multiseed" / "experiment4_seed_pair_deltas.csv")
    if len(delta_file) != 15:
        raise RuntimeError(f"Frozen delta file must contain exactly 15 rows; found {len(delta_file)}.")
    expected_keys = [
        (horizon, seed) for horizon in HORIZONS for seed in SEEDS
    ]
    present = set((row["horizon"], row["seed"]) for _, row in delta_file.iterrows())
    if set(expected_keys) != present:
        raise RuntimeError("Frozen delta file missing or contains unexpected horizon/seed keys.")
    for horizon in HORIZONS:
        for seed in SEEDS:
            values = loss_df[(loss_df["horizon"] == horizon) & (loss_df["seed"] == seed)]["loss_diff"].to_numpy(dtype=np.float64)
            reconstructed = float(np.mean(values, dtype=np.float64))
            ref = float(delta_file[(delta_file["horizon"] == horizon) & (delta_file["seed"] == seed)]["delta_mse"].iloc[0])
            if not np.isclose(reconstructed, ref, rtol=LOSS_BRIDGE_RTOL, atol=LOSS_BRIDGE_ATOL):
                raise RuntimeError(f"Loss bridge reconstruction mismatch for {horizon} seed {seed}: {reconstructed} vs {ref}")
        per_horizon = [
            float(loss_df[(loss_df["horizon"] == horizon) & (loss_df["seed"] == seed)]["loss_diff"].mean())
            for seed in SEEDS
        ]
        mean_delta = float(np.mean(per_horizon, dtype=np.float64))
        if not np.isclose(mean_delta, FROZEN_MEAN_DELTA_MSE[horizon], rtol=LOSS_BRIDGE_RTOL, atol=LOSS_BRIDGE_ATOL):
            raise RuntimeError(f"Frozen mean delta_mse mismatch for horizon {horizon}: {mean_delta} vs {FROZEN_MEAN_DELTA_MSE[horizon]}")


def _build_adaptive_sample_metrics(weights_df: pd.DataFrame, loss_df: pd.DataFrame, static_ref: dict) -> pd.DataFrame:
    records = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            cell = weights_df[
                (weights_df["mechanism"] == "adaptive")
                & (weights_df["horizon"] == horizon)
                & (weights_df["seed"] == seed)
            ].copy()
            cell = cell.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)

            # Build the loss lookup for this horizon/seed cell once (aligned by identical key order)
            # instead of re-filtering the full loss table for every individual sample row.
            loss_cell = loss_df[
                (loss_df["mechanism"] == "adaptive")
                & (loss_df["horizon"] == horizon)
                & (loss_df["seed"] == seed)
            ].copy()
            loss_cell = loss_cell.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
            if len(loss_cell) != N_CELL_ROWS:
                raise RuntimeError(f"Loss cell row count invalid for {horizon} seed {seed}: {len(loss_cell)}")
            if not loss_cell[["ticker", "anchor_date"]].equals(cell[["ticker", "anchor_date"]]):
                raise RuntimeError(f"Loss bridge key mismatch for {horizon} seed {seed}.")
            loss_diff_arr = loss_cell["loss_diff"].to_numpy(dtype=np.float64)

            ticker_arr = cell["ticker"].to_numpy()
            anchor_date_arr = cell["anchor_date"].to_numpy()

            for frequency in FREQUENCIES:
                if frequency == "daily":
                    vec_cols = [f"daily_w_{p}" for p in DAILY_PATCHES]
                    patch_order = DAILY_PATCHES
                else:
                    vec_cols = [f"weekly_w_{p}" for p in WEEKLY_PATCHES]
                    patch_order = WEEKLY_PATCHES
                arr = cell[vec_cols].to_numpy(dtype=np.float64)
                means = np.mean(arr, axis=0, dtype=np.float64)
                tv_to_cell_mean = np.array([
                    _tv_distance(arr[i], means)
                    for i in range(arr.shape[0])
                ], dtype=np.float64)
                static_vec = static_ref[(horizon, seed, frequency)]
                for i in range(arr.shape[0]):
                    w = arr[i]
                    h, h_norm, n_eff, max_w = _entropy_metrics(w)
                    dominant_idx = int(np.argmax(w))
                    dominant_patch = patch_order[dominant_idx]
                    tv_to_static = _tv_distance(w, static_vec)
                    loss_diff = float(loss_diff_arr[i])
                    rec = {
                        "horizon": horizon,
                        "seed": int(seed),
                        "ticker": ticker_arr[i],
                        "anchor_date": anchor_date_arr[i],
                        "frequency": frequency,
                        "patch_1": patch_order[0],
                        "patch_2": patch_order[1],
                        "patch_3": patch_order[2],
                        "patch_4": patch_order[3],
                        "weight_1": float(w[0]),
                        "weight_2": float(w[1]),
                        "weight_3": float(w[2]),
                        "weight_4": float(w[3]),
                        "entropy": float(h),
                        "normalized_entropy": float(h_norm),
                        "effective_scales": float(n_eff),
                        "max_weight": float(max_w),
                        "dominant_patch": int(dominant_patch),
                        "tv_to_cell_mean": float(tv_to_cell_mean[i]),
                        "tv_to_static": float(tv_to_static),
                        "loss_diff": float(loss_diff),
                    }
                    records.append(rec)
    out = pd.DataFrame(records)
    if list(out.columns) != EXACT_OUTPUT_1_COLUMNS:
        raise RuntimeError("Adaptive sample metrics schema mismatch.")
    out["_horizon_order"] = out["horizon"].map(HORIZON_ORDER)
    out["_frequency_order"] = out["frequency"].map(FREQUENCY_ORDER)
    out = out.sort_values(
        ["_horizon_order", "seed", "ticker", "anchor_date", "_frequency_order"],
        kind="mergesort",
    ).drop(columns=["_horizon_order", "_frequency_order"]).reset_index(drop=True)
    if len(out) != 199410:
        raise RuntimeError(f"Adaptive sample metrics row count incorrect: expected 199410, got {len(out)}")
    return out


def _build_cell_summary(weights_df: pd.DataFrame, sample_df: pd.DataFrame, static_ref: dict) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            for frequency in FREQUENCIES:
                adaptive = sample_df[
                    (sample_df["horizon"] == horizon)
                    & (sample_df["seed"] == seed)
                    & (sample_df["frequency"] == frequency)
                ].copy()
                if len(adaptive) != N_CELL_ROWS:
                    raise RuntimeError(f"Cell summary has wrong number of sample rows: {horizon} {seed} {frequency} -> {len(adaptive)}")
                cell_mean_w = adaptive[["weight_1", "weight_2", "weight_3", "weight_4"]].mean(axis=0, skipna=False).to_numpy(dtype=np.float64)
                static_vec = static_ref[(horizon, seed, frequency)]
                tv_cell_mean_to_static = _tv_distance(cell_mean_w, static_vec)
                tv_values = adaptive["tv_to_static"].to_numpy(dtype=np.float64)
                if len(tv_values) != N_CELL_ROWS:
                    raise RuntimeError(f"TV sample list wrong size for {horizon} {seed} {frequency}.")
                n_tickers = adaptive["ticker"].nunique()
                n_dates = adaptive["anchor_date"].nunique()
                if n_tickers != N_TICKERS or n_dates != N_TEST_DATES:
                    raise RuntimeError(f"Ticker/date counts invalid for cell summary: {horizon} {seed} {frequency}")
                w1 = adaptive["weight_1"].to_numpy(dtype=np.float64)
                w2 = adaptive["weight_2"].to_numpy(dtype=np.float64)
                w3 = adaptive["weight_3"].to_numpy(dtype=np.float64)
                w4 = adaptive["weight_4"].to_numpy(dtype=np.float64)

                # Within-ticker temporal variation (Q1.12.2) and within-date cross-sectional variation (Q1.12.3).
                ticker_tvs = []
                for _, grp in adaptive.groupby("ticker"):
                    vecs = grp[["weight_1", "weight_2", "weight_3", "weight_4"]].to_numpy(dtype=np.float64)
                    mean_vec = np.mean(vecs, axis=0, dtype=np.float64)
                    ticker_tvs.append(float(np.mean([_tv_distance(vec, mean_vec) for vec in vecs], dtype=np.float64)))
                within_ticker_temporal_tv = float(np.mean(ticker_tvs, dtype=np.float64)) if ticker_tvs else float("nan")
                date_tvs = []
                for _, grp in adaptive.groupby("anchor_date"):
                    vecs = grp[["weight_1", "weight_2", "weight_3", "weight_4"]].to_numpy(dtype=np.float64)
                    mean_vec = np.mean(vecs, axis=0, dtype=np.float64)
                    date_tvs.append(float(np.mean([_tv_distance(vec, mean_vec) for vec in vecs], dtype=np.float64)))
                within_date_crosssectional_tv = float(np.mean(date_tvs, dtype=np.float64)) if date_tvs else float("nan")

                row = {
                    "horizon": horizon,
                    "seed": int(seed),
                    "frequency": frequency,
                    "n_samples": int(len(adaptive)),
                    "n_tickers": int(n_tickers),
                    "n_dates": int(n_dates),
                    "mean_weight_1": float(np.mean(w1, dtype=np.float64)),
                    "mean_weight_2": float(np.mean(w2, dtype=np.float64)),
                    "mean_weight_3": float(np.mean(w3, dtype=np.float64)),
                    "mean_weight_4": float(np.mean(w4, dtype=np.float64)),
                    "sd_weight_1": float(np.std(w1, ddof=1, dtype=np.float64)) if len(w1) > 1 else float("nan"),
                    "sd_weight_2": float(np.std(w2, ddof=1, dtype=np.float64)) if len(w2) > 1 else float("nan"),
                    "sd_weight_3": float(np.std(w3, ddof=1, dtype=np.float64)) if len(w3) > 1 else float("nan"),
                    "sd_weight_4": float(np.std(w4, ddof=1, dtype=np.float64)) if len(w4) > 1 else float("nan"),
                    "sample_dependence_index": float(np.mean(adaptive["tv_to_cell_mean"].to_numpy(dtype=np.float64), dtype=np.float64)),
                    "within_ticker_temporal_tv": within_ticker_temporal_tv,
                    "within_date_crosssectional_tv": within_date_crosssectional_tv,
                    "normalized_entropy_mean": float(np.mean(adaptive["normalized_entropy"].to_numpy(dtype=np.float64), dtype=np.float64)),
                    "normalized_entropy_sd": float(np.std(adaptive["normalized_entropy"].to_numpy(dtype=np.float64), ddof=1, dtype=np.float64)) if len(adaptive) > 1 else float("nan"),
                    "normalized_entropy_median": float(np.median(adaptive["normalized_entropy"].to_numpy(dtype=np.float64))),
                    "normalized_entropy_p10": float(np.quantile(adaptive["normalized_entropy"].to_numpy(dtype=np.float64), 0.10, method="linear")),
                    "normalized_entropy_p90": float(np.quantile(adaptive["normalized_entropy"].to_numpy(dtype=np.float64), 0.90, method="linear")),
                    "effective_scales_mean": float(np.mean(adaptive["effective_scales"].to_numpy(dtype=np.float64), dtype=np.float64)),
                    "effective_scales_sd": float(np.std(adaptive["effective_scales"].to_numpy(dtype=np.float64), ddof=1, dtype=np.float64)) if len(adaptive) > 1 else float("nan"),
                    "effective_scales_median": float(np.median(adaptive["effective_scales"].to_numpy(dtype=np.float64))),
                    "effective_scales_p10": float(np.quantile(adaptive["effective_scales"].to_numpy(dtype=np.float64), 0.10, method="linear")),
                    "effective_scales_p90": float(np.quantile(adaptive["effective_scales"].to_numpy(dtype=np.float64), 0.90, method="linear")),
                    "max_weight_mean": float(np.mean(adaptive["max_weight"].to_numpy(dtype=np.float64), dtype=np.float64)),
                    "max_weight_sd": float(np.std(adaptive["max_weight"].to_numpy(dtype=np.float64), ddof=1, dtype=np.float64)) if len(adaptive) > 1 else float("nan"),
                    "max_weight_median": float(np.median(adaptive["max_weight"].to_numpy(dtype=np.float64))),
                    "max_weight_p10": float(np.quantile(adaptive["max_weight"].to_numpy(dtype=np.float64), 0.10, method="linear")),
                    "max_weight_p90": float(np.quantile(adaptive["max_weight"].to_numpy(dtype=np.float64), 0.90, method="linear")),
                    "static_weight_1": float(static_vec[0]),
                    "static_weight_2": float(static_vec[1]),
                    "static_weight_3": float(static_vec[2]),
                    "static_weight_4": float(static_vec[3]),
                    "tv_cell_mean_to_static": float(tv_cell_mean_to_static),
                    "tv_to_static_mean": float(np.mean(tv_values, dtype=np.float64)),
                    "tv_to_static_sd": float(np.std(tv_values, ddof=1, dtype=np.float64)) if len(tv_values) > 1 else float("nan"),
                    "tv_to_static_median": float(np.median(tv_values)),
                    "tv_to_static_p10": float(np.quantile(tv_values, 0.10, method="linear")),
                    "tv_to_static_p90": float(np.quantile(tv_values, 0.90, method="linear")),
                }
                rows.append(row)
    out = pd.DataFrame(rows)
    if list(out.columns) != EXACT_OUTPUT_2_COLUMNS:
        raise RuntimeError("Cell summary schema mismatch.")
    out["_horizon_order"] = out["horizon"].map(HORIZON_ORDER)
    out["_frequency_order"] = out["frequency"].map(FREQUENCY_ORDER)
    out = out.sort_values(
        ["_horizon_order", "seed", "_frequency_order"], kind="mergesort"
    ).drop(columns=["_horizon_order", "_frequency_order"]).reset_index(drop=True)
    if len(out) != 30:
        raise RuntimeError(f"Cell summary row count incorrect: expected 30, got {len(out)}")
    return out


def _build_dominant_usage(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            for frequency in FREQUENCIES:
                cell = sample_df[(sample_df["horizon"] == horizon) & (sample_df["seed"] == seed) & (sample_df["frequency"] == frequency)].copy()
                patch_order = DAILY_PATCHES if frequency == "daily" else WEEKLY_PATCHES
                for patch in patch_order:
                    count = int((cell["dominant_patch"] == patch).sum())
                    share = float(count / N_CELL_ROWS)
                    rows.append({
                        "horizon": horizon,
                        "seed": int(seed),
                        "frequency": frequency,
                        "patch_size": int(patch),
                        "dominant_count": count,
                        "dominant_share": share,
                    })
    out = pd.DataFrame(rows)
    if list(out.columns) != EXACT_OUTPUT_3_COLUMNS:
        raise RuntimeError("Dominant scale usage schema mismatch.")
    if len(out) != 120:
        raise RuntimeError(f"Dominant scale usage row count incorrect: expected 120, got {len(out)}")
    out["_horizon_order"] = out["horizon"].map(HORIZON_ORDER)
    out["_frequency_order"] = out["frequency"].map(FREQUENCY_ORDER)
    out["_patch_order"] = out["patch_size"].map(PATCH_ORDER_MAP)
    out = out.sort_values(
        ["_horizon_order", "seed", "_frequency_order", "_patch_order"], kind="mergesort"
    ).drop(columns=["_horizon_order", "_frequency_order", "_patch_order"]).reset_index(drop=True)
    for horizon in HORIZONS:
        for seed in SEEDS:
            for frequency in FREQUENCIES:
                grp = out[(out["horizon"] == horizon) & (out["seed"] == seed) & (out["frequency"] == frequency)]
                total = float(
                    np.sum(
                        grp["dominant_share"].to_numpy(dtype=np.float64),
                        dtype=np.float64,
                    )
                )
                if abs(total - 1.0) > 1e-12:
                    raise RuntimeError(f"Dominant share totals must sum to 1 within 1e-12 for {horizon} {seed} {frequency}: {total}")
    return out


def _build_cross_seed_stability(sample_df: pd.DataFrame) -> pd.DataFrame:
    weight_cols = ["weight_1", "weight_2", "weight_3", "weight_4"]
    rows = []
    grouped = sample_df.groupby(["horizon", "ticker", "anchor_date", "frequency"], sort=False)
    for key, grp in grouped:
        horizon, ticker, anchor_date, frequency = key
        if len(grp) != len(SEEDS):
            raise RuntimeError(f"Cross-seed sample missing seeds for {key}: {len(grp)}")
        grp_sorted = grp.sort_values("seed", kind="mergesort")
        if grp_sorted["seed"].tolist() != SEEDS:
            raise RuntimeError(f"Cross-seed sample seed set mismatch for {key}.")
        vecs = grp_sorted[weight_cols].to_numpy(dtype=np.float64)
        dominant_by_seed = grp_sorted["dominant_patch"].to_numpy()
        pairwise = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                pairwise.append(_tv_distance(vecs[i], vecs[j]))
        mean_pairwise = float(np.mean(pairwise, dtype=np.float64))
        max_pairwise = float(np.max(np.asarray(pairwise, dtype=np.float64)))
        counts: dict = {}
        for v in dominant_by_seed:
            counts[int(v)] = counts.get(int(v), 0) + 1
        agreement = max(counts.values()) / 5.0
        unanimity = 1 if agreement == 1.0 else 0
        rows.append({
            "horizon": horizon,
            "ticker": ticker,
            "anchor_date": anchor_date,
            "frequency": frequency,
            "n_seeds": 5,
            "mean_pairwise_seed_tv": mean_pairwise,
            "max_pairwise_seed_tv": max_pairwise,
            "dominant_agreement_share": float(agreement),
            "dominant_unanimous": int(unanimity),
        })
    out = pd.DataFrame(rows)
    if len(out) != 39882:
        raise RuntimeError(f"Cross-seed stability row count incorrect: expected 39882, got {len(out)}")
    if list(out.columns) != EXACT_OUTPUT_4_COLUMNS:
        raise RuntimeError("Cross-seed stability schema mismatch.")
    out["_horizon_order"] = out["horizon"].map(HORIZON_ORDER)
    out["_frequency_order"] = out["frequency"].map(FREQUENCY_ORDER)
    out = out.sort_values(
        ["_horizon_order", "ticker", "anchor_date", "_frequency_order"], kind="mergesort"
    ).drop(columns=["_horizon_order", "_frequency_order"]).reset_index(drop=True)
    return out


def _build_cross_seed_summary(stability_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for frequency in FREQUENCIES:
            sub = stability_df[(stability_df["horizon"] == horizon) & (stability_df["frequency"] == frequency)].copy()
            if len(sub) != N_CELL_ROWS:
                raise RuntimeError(f"Cross-seed summary sample count invalid for {horizon} {frequency}: {len(sub)}")
            mean_pairwise = sub["mean_pairwise_seed_tv"].to_numpy(dtype=np.float64)
            max_pairwise = sub["max_pairwise_seed_tv"].to_numpy(dtype=np.float64)
            agreement = sub["dominant_agreement_share"].to_numpy(dtype=np.float64)
            mean_v, sd_v, median_v, p90 = _mean_sd_median_p90(mean_pairwise)
            max_mean, max_median, max_p90 = _mean_median_p90(max_pairwise)
            rows.append({
                "horizon": horizon,
                "frequency": frequency,
                "n_samples": int(len(sub)),
                "mean_pairwise_seed_tv_mean": mean_v,
                "mean_pairwise_seed_tv_sd": sd_v,
                "mean_pairwise_seed_tv_median": median_v,
                "mean_pairwise_seed_tv_p90": p90,
                "max_pairwise_seed_tv_mean": max_mean,
                "max_pairwise_seed_tv_median": max_median,
                "max_pairwise_seed_tv_p90": max_p90,
                "dominant_agreement_share_mean": float(np.mean(agreement, dtype=np.float64)),
                "dominant_agreement_share_median": float(np.median(agreement)),
                "dominant_unanimous_share": float(np.mean(sub["dominant_unanimous"].to_numpy(dtype=np.float64), dtype=np.float64)),
            })
    out = pd.DataFrame(rows)
    if list(out.columns) != EXACT_OUTPUT_5_COLUMNS:
        raise RuntimeError("Cross-seed summary schema mismatch.")
    if len(out) != 6:
        raise RuntimeError(f"Cross-seed summary row count incorrect: expected 6, got {len(out)}")
    return out


def _build_loss_association_by_seed(sample_df: pd.DataFrame, loss_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            for frequency in FREQUENCIES:
                adaptive = sample_df[(sample_df["horizon"] == horizon) & (sample_df["seed"] == seed) & (sample_df["frequency"] == frequency)].copy()
                x = adaptive["tv_to_static"].to_numpy(dtype=np.float64)
                y = adaptive["loss_diff"].to_numpy(dtype=np.float64)
                rho, defined = _manual_spearman(x, y)
                if x.size != y.size or x.size != N_CELL_ROWS:
                    raise RuntimeError(f"Loss association sample size invalid for {horizon} {seed} {frequency}: {x.size}")
                # date-level mean pairs across 391 dates.
                date_groups = []
                for date, grp in adaptive.groupby("anchor_date"):
                    tv_mean = np.mean(grp["tv_to_static"].to_numpy(dtype=np.float64), dtype=np.float64)
                    loss_mean = np.mean(grp["loss_diff"].to_numpy(dtype=np.float64), dtype=np.float64)
                    date_groups.append((tv_mean, loss_mean))
                if len(date_groups) != N_TEST_DATES:
                    raise RuntimeError(f"Date-level loss association count invalid for {horizon} {seed} {frequency}: {len(date_groups)}")
                date_x = np.array([pair[0] for pair in date_groups], dtype=np.float64)
                date_y = np.array([pair[1] for pair in date_groups], dtype=np.float64)
                date_rho, date_defined = _manual_spearman(date_x, date_y)
                rows.append({
                    "horizon": horizon,
                    "seed": int(seed),
                    "frequency": frequency,
                    "n_samples": int(len(adaptive)),
                    "n_dates": int(len(date_groups)),
                    "sample_level_spearman_rho": float(rho) if defined else float("nan"),
                    "sample_level_correlation_defined": bool(defined),
                    "date_level_spearman_rho": float(date_rho) if date_defined else float("nan"),
                    "date_level_correlation_defined": bool(date_defined),
                })
    out = pd.DataFrame(rows)
    if list(out.columns) != EXACT_OUTPUT_6_COLUMNS:
        raise RuntimeError("Loss association by seed schema mismatch.")
    if len(out) != 30:
        raise RuntimeError(f"Loss association by seed row count incorrect: expected 30, got {len(out)}")
    return out


def _build_loss_association_summary(loss_assoc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for frequency in FREQUENCIES:
            sub = loss_assoc_df[(loss_assoc_df["horizon"] == horizon) & (loss_assoc_df["frequency"] == frequency)].copy()
            sample_rhos = sub["sample_level_spearman_rho"].to_numpy(dtype=np.float64)
            date_rhos = sub["date_level_spearman_rho"].to_numpy(dtype=np.float64)
            sample_summary = _summarize_defined_rhos(sample_rhos)
            date_summary = _summarize_defined_rhos(date_rhos)
            rows.append({
                "horizon": horizon,
                "frequency": frequency,
                "sample_level_rho_mean": sample_summary["mean"],
                "sample_level_rho_sd": sample_summary["sd"],
                "sample_level_rho_median": sample_summary["median"],
                "sample_level_n_defined": sample_summary["n_defined"],
                "sample_level_n_positive": sample_summary["n_positive"],
                "sample_level_n_negative": sample_summary["n_negative"],
                "sample_level_n_zero": sample_summary["n_zero"],
                "date_level_rho_mean": date_summary["mean"],
                "date_level_rho_sd": date_summary["sd"],
                "date_level_rho_median": date_summary["median"],
                "date_level_n_defined": date_summary["n_defined"],
                "date_level_n_positive": date_summary["n_positive"],
                "date_level_n_negative": date_summary["n_negative"],
                "date_level_n_zero": date_summary["n_zero"],
            })
    out = pd.DataFrame(rows)
    if list(out.columns) != EXACT_OUTPUT_7_COLUMNS:
        raise RuntimeError("Loss association summary schema mismatch.")
    if len(out) != 6:
        raise RuntimeError(f"Loss association summary row count incorrect: expected 6, got {len(out)}")
    for _, row in out.iterrows():
        if row["sample_level_n_defined"] != row["sample_level_n_positive"] + row["sample_level_n_negative"] + row["sample_level_n_zero"]:
            raise RuntimeError("Sample-level sign count arithmetic invalid.")
        if row["date_level_n_defined"] != row["date_level_n_positive"] + row["date_level_n_negative"] + row["date_level_n_zero"]:
            raise RuntimeError("Date-level sign count arithmetic invalid.")
        if row["sample_level_n_defined"] < 0 or row["sample_level_n_defined"] > 5:
            raise RuntimeError("Sample-level n_defined out of range.")
        if row["date_level_n_defined"] < 0 or row["date_level_n_defined"] > 5:
            raise RuntimeError("Date-level n_defined out of range.")
    return out


def _assert_canonical_order(df: pd.DataFrame, key_cols: list[str], sort_cols: list[str], order_maps: dict) -> None:
    order_check = df[key_cols].copy()
    temp_cols = []
    sort_keys = []
    for col in sort_cols:
        if col in order_maps:
            temp_col = f"_order_{col}"
            order_check[temp_col] = order_check[col].map(order_maps[col])
            temp_cols.append(temp_col)
            sort_keys.append(temp_col)
        else:
            sort_keys.append(col)
    expected_order = order_check.sort_values(sort_keys, kind="mergesort").drop(columns=temp_cols).reset_index(drop=True)
    actual_order = order_check.drop(columns=temp_cols).reset_index(drop=True)
    if not actual_order.equals(expected_order):
        raise RuntimeError("Output rows are not in the required canonical order.")


def _validate_output_1(sample_df: pd.DataFrame, *, serialized: bool = False) -> None:
    if list(sample_df.columns) != EXACT_OUTPUT_1_COLUMNS:
        raise RuntimeError("Output 1 columns do not exactly match the frozen schema.")
    if len(sample_df) != N_SAMPLE_METRICS_ROWS:
        raise RuntimeError("Output 1 row count mismatch.")
    if set(sample_df["horizon"].unique().tolist()) != set(HORIZONS):
        raise RuntimeError("Output 1 horizon values invalid.")
    if set(sample_df["seed"].unique().tolist()) != set(SEEDS):
        raise RuntimeError("Output 1 seed values invalid.")
    if set(sample_df["frequency"].unique().tolist()) != set(FREQUENCIES):
        raise RuntimeError("Output 1 frequency values invalid.")
    if sample_df["ticker"].isnull().any():
        raise RuntimeError("Output 1 contains null ticker values.")

    anchor_date = _coerce_anchor_date(sample_df["anchor_date"], serialized=serialized)
    if anchor_date.isnull().any():
        raise RuntimeError("Output 1 contains null anchor_date values.")

    check_df = sample_df.copy()
    check_df["anchor_date"] = anchor_date

    key_df = check_df[["horizon", "seed", "ticker", "anchor_date", "frequency"]]
    if key_df.duplicated().any():
        raise RuntimeError("Output 1 contains duplicate horizon/seed/ticker/anchor_date/frequency keys.")

    cell_counts = check_df.groupby(["horizon", "seed", "frequency"]).size()
    if len(cell_counts) != len(HORIZONS) * len(SEEDS) * len(FREQUENCIES):
        raise RuntimeError("Output 1 horizon/seed/frequency grid incomplete: expected 30 cells.")
    if not np.all(cell_counts.to_numpy() == N_CELL_ROWS):
        raise RuntimeError("Output 1 every horizon/seed/frequency cell must have exactly 6647 rows.")

    for (horizon, seed, frequency), grp in check_df.groupby(["horizon", "seed", "frequency"]):
        if grp["ticker"].nunique() != N_TICKERS:
            raise RuntimeError(f"Output 1 ticker count invalid for {horizon} {seed} {frequency}.")
        if grp["anchor_date"].nunique() != N_TEST_DATES:
            raise RuntimeError(f"Output 1 date count invalid for {horizon} {seed} {frequency}.")
        per_date = grp.groupby("anchor_date").size()
        if not np.all(per_date.to_numpy() == N_TICKERS):
            raise RuntimeError(f"Output 1 rows-per-date invalid for {horizon} {seed} {frequency}.")

    _assert_canonical_order(
        check_df, ["horizon", "seed", "ticker", "anchor_date", "frequency"],
        ["horizon", "seed", "ticker", "anchor_date", "frequency"],
        {"horizon": HORIZON_ORDER, "frequency": FREQUENCY_ORDER},
    )

    weight_cols = ["weight_1", "weight_2", "weight_3", "weight_4"]
    weights = sample_df[weight_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("Output 1 weights contain non-finite values.")
    if np.any(weights < -RANGE_TOL) or np.any(weights > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 1 weights out of [0,1].")
    row_sums = np.sum(weights, axis=1, dtype=np.float64)
    if np.any(np.abs(row_sums - 1.0) > WEIGHT_SUM_ATOL):
        raise RuntimeError("Output 1 weight rows do not sum to 1 within 1e-6.")

    is_daily = (sample_df["frequency"] == "daily").to_numpy()
    patch_arr = sample_df[["patch_1", "patch_2", "patch_3", "patch_4"]].to_numpy(dtype=np.int64)
    if not np.all(patch_arr[is_daily] == np.array(DAILY_PATCHES, dtype=np.int64)):
        raise RuntimeError("Output 1 daily patch columns do not match the canonical daily patch order.")
    if not np.all(patch_arr[~is_daily] == np.array(WEEKLY_PATCHES, dtype=np.int64)):
        raise RuntimeError("Output 1 weekly patch columns do not match the canonical weekly patch order.")

    entropy = sample_df["entropy"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(entropy)) or np.any(entropy < -RANGE_TOL) or np.any(entropy > math.log(4.0) + RANGE_TOL):
        raise RuntimeError("Output 1 entropy out of [0, ln(4)].")

    normalized_entropy = sample_df["normalized_entropy"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(normalized_entropy)) or np.any(normalized_entropy < -RANGE_TOL) or np.any(normalized_entropy > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 1 normalized_entropy out of [0,1].")

    effective_scales = sample_df["effective_scales"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(effective_scales)) or np.any(effective_scales < 1.0 - RANGE_TOL) or np.any(effective_scales > 4.0 + RANGE_TOL):
        raise RuntimeError("Output 1 effective_scales out of [1,4].")

    max_weight = sample_df["max_weight"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(max_weight)) or np.any(max_weight < 0.25 - RANGE_TOL) or np.any(max_weight > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 1 max_weight out of [0.25,1].")

    tv_to_cell_mean = sample_df["tv_to_cell_mean"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(tv_to_cell_mean)) or np.any(tv_to_cell_mean < -RANGE_TOL) or np.any(tv_to_cell_mean > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 1 tv_to_cell_mean out of [0,1].")

    tv_to_static = sample_df["tv_to_static"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(tv_to_static)) or np.any(tv_to_static < -RANGE_TOL) or np.any(tv_to_static > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 1 tv_to_static out of [0,1].")

    loss_diff = sample_df["loss_diff"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(loss_diff)):
        raise RuntimeError("Output 1 loss_diff contains non-finite values.")

    expected_max_weight = np.max(weights, axis=1)
    if np.any(np.abs(expected_max_weight - max_weight) > RANGE_TOL):
        raise RuntimeError("Output 1 max_weight does not match max(weight_1..weight_4).")

    dominant_idx = np.argmax(weights, axis=1)
    dominant_patch_actual = sample_df["dominant_patch"].to_numpy(dtype=np.int64)
    daily_patches = np.array(DAILY_PATCHES, dtype=np.int64)
    weekly_patches = np.array(WEEKLY_PATCHES, dtype=np.int64)
    expected_dominant_patch = np.where(is_daily, daily_patches[dominant_idx], weekly_patches[dominant_idx])
    if not np.array_equal(dominant_patch_actual, expected_dominant_patch):
        raise RuntimeError("Output 1 dominant_patch does not equal the first-canonical argmax of the weight vector.")


def _validate_output_2(cell_summary: pd.DataFrame) -> None:
    if list(cell_summary.columns) != EXACT_OUTPUT_2_COLUMNS:
        raise RuntimeError("Output 2 columns do not exactly match the frozen schema.")
    if len(cell_summary) != 30:
        raise RuntimeError("Output 2 row count mismatch.")
    if set(cell_summary["horizon"].unique().tolist()) != set(HORIZONS):
        raise RuntimeError("Output 2 horizon values invalid.")
    if set(cell_summary["seed"].unique().tolist()) != set(SEEDS):
        raise RuntimeError("Output 2 seed values invalid.")
    if set(cell_summary["frequency"].unique().tolist()) != set(FREQUENCIES):
        raise RuntimeError("Output 2 frequency values invalid.")

    key_df = cell_summary[["horizon", "seed", "frequency"]]
    if key_df.duplicated().any():
        raise RuntimeError("Output 2 contains duplicate horizon/seed/frequency keys.")
    if len(key_df) != len(HORIZONS) * len(SEEDS) * len(FREQUENCIES):
        raise RuntimeError("Output 2 horizon/seed/frequency grid incomplete.")

    _assert_canonical_order(
        cell_summary, ["horizon", "seed", "frequency"], ["horizon", "seed", "frequency"],
        {"horizon": HORIZON_ORDER, "frequency": FREQUENCY_ORDER},
    )

    for horizon in HORIZONS:
        for seed in SEEDS:
            for frequency in FREQUENCIES:
                subset = cell_summary[(cell_summary["horizon"] == horizon) & (cell_summary["seed"] == seed) & (cell_summary["frequency"] == frequency)]
                if subset.empty:
                    raise RuntimeError(f"Missing cell summary row for {horizon} {seed} {frequency}.")

    if not np.all(cell_summary["n_samples"].to_numpy() == N_CELL_ROWS):
        raise RuntimeError("Output 2 n_samples must equal 6647 everywhere.")
    if not np.all(cell_summary["n_tickers"].to_numpy() == N_TICKERS):
        raise RuntimeError("Output 2 n_tickers must equal 17 everywhere.")
    if not np.all(cell_summary["n_dates"].to_numpy() == N_TEST_DATES):
        raise RuntimeError("Output 2 n_dates must equal 391 everywhere.")

    numeric_cols = [c for c in EXACT_OUTPUT_2_COLUMNS if c not in ("horizon", "seed", "frequency", "n_samples", "n_tickers", "n_dates")]
    numeric_arr = cell_summary[numeric_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric_arr)):
        raise RuntimeError("Output 2 must not contain NaN/inf in any scientific numeric field.")

    mean_weights = cell_summary[["mean_weight_1", "mean_weight_2", "mean_weight_3", "mean_weight_4"]].to_numpy(dtype=np.float64)
    if np.any(mean_weights < -RANGE_TOL) or np.any(mean_weights > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 2 mean_weight_* out of [0,1].")
    if np.any(np.abs(np.sum(mean_weights, axis=1) - 1.0) > WEIGHT_SUM_ATOL):
        raise RuntimeError("Output 2 mean_weight_* rows do not sum to 1 within 1e-6.")

    static_weights = cell_summary[["static_weight_1", "static_weight_2", "static_weight_3", "static_weight_4"]].to_numpy(dtype=np.float64)
    if np.any(static_weights < -RANGE_TOL) or np.any(static_weights > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 2 static_weight_* out of [0,1].")
    if np.any(np.abs(np.sum(static_weights, axis=1) - 1.0) > WEIGHT_SUM_ATOL):
        raise RuntimeError("Output 2 static_weight_* rows do not sum to 1 within 1e-6.")

    sd_arr = cell_summary[["sd_weight_1", "sd_weight_2", "sd_weight_3", "sd_weight_4"]].to_numpy(dtype=np.float64)
    if np.any(sd_arr < -RANGE_TOL):
        raise RuntimeError("Output 2 sd_weight_* must be >= 0.")

    tv_cols = [
        "sample_dependence_index", "within_ticker_temporal_tv", "within_date_crosssectional_tv",
        "tv_cell_mean_to_static", "tv_to_static_mean", "tv_to_static_median", "tv_to_static_p10", "tv_to_static_p90",
    ]
    tv_arr = cell_summary[tv_cols].to_numpy(dtype=np.float64)
    if np.any(tv_arr < -RANGE_TOL) or np.any(tv_arr > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 2 TV-based metrics out of [0,1].")
    if np.any(cell_summary["tv_to_static_sd"].to_numpy(dtype=np.float64) < -RANGE_TOL):
        raise RuntimeError("Output 2 tv_to_static_sd must be >= 0.")

    entropy_arr = cell_summary[["normalized_entropy_mean", "normalized_entropy_median", "normalized_entropy_p10", "normalized_entropy_p90"]].to_numpy(dtype=np.float64)
    if np.any(entropy_arr < -RANGE_TOL) or np.any(entropy_arr > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 2 normalized entropy summary fields out of [0,1].")
    if np.any(cell_summary["normalized_entropy_sd"].to_numpy(dtype=np.float64) < -RANGE_TOL):
        raise RuntimeError("Output 2 normalized_entropy_sd must be >= 0.")

    eff_arr = cell_summary[["effective_scales_mean", "effective_scales_median", "effective_scales_p10", "effective_scales_p90"]].to_numpy(dtype=np.float64)
    if np.any(eff_arr < 1.0 - RANGE_TOL) or np.any(eff_arr > 4.0 + RANGE_TOL):
        raise RuntimeError("Output 2 effective scale summary fields out of [1,4].")
    if np.any(cell_summary["effective_scales_sd"].to_numpy(dtype=np.float64) < -RANGE_TOL):
        raise RuntimeError("Output 2 effective_scales_sd must be >= 0.")

    max_w_arr = cell_summary[["max_weight_mean", "max_weight_median", "max_weight_p10", "max_weight_p90"]].to_numpy(dtype=np.float64)
    if np.any(max_w_arr < 0.25 - RANGE_TOL) or np.any(max_w_arr > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 2 max-weight summary fields out of [0.25,1].")
    if np.any(cell_summary["max_weight_sd"].to_numpy(dtype=np.float64) < -RANGE_TOL):
        raise RuntimeError("Output 2 max_weight_sd must be >= 0.")


def _validate_output_3(dom_usage: pd.DataFrame) -> None:
    if list(dom_usage.columns) != EXACT_OUTPUT_3_COLUMNS:
        raise RuntimeError("Output 3 columns do not exactly match the frozen schema.")
    if len(dom_usage) != 120:
        raise RuntimeError("Output 3 row count mismatch.")
    if set(dom_usage["horizon"].unique().tolist()) != set(HORIZONS):
        raise RuntimeError("Output 3 horizon values invalid.")
    if set(dom_usage["seed"].unique().tolist()) != set(SEEDS):
        raise RuntimeError("Output 3 seed values invalid.")
    if set(dom_usage["frequency"].unique().tolist()) != set(FREQUENCIES):
        raise RuntimeError("Output 3 frequency values invalid.")

    key_df = dom_usage[["horizon", "seed", "frequency", "patch_size"]]
    if key_df.duplicated().any():
        raise RuntimeError("Output 3 contains duplicate horizon/seed/frequency/patch_size keys.")
    if len(key_df) != len(HORIZONS) * len(SEEDS) * len(FREQUENCIES) * 4:
        raise RuntimeError("Output 3 horizon/seed/frequency/patch_size grid incomplete.")

    _assert_canonical_order(
        dom_usage, ["horizon", "seed", "frequency", "patch_size"], ["horizon", "seed", "frequency", "patch_size"],
        {"horizon": HORIZON_ORDER, "frequency": FREQUENCY_ORDER, "patch_size": PATCH_ORDER_MAP},
    )

    dominant_count_raw = dom_usage["dominant_count"].to_numpy(dtype=np.float64)
    if not np.all(dominant_count_raw == np.round(dominant_count_raw)):
        raise RuntimeError("Output 3 dominant_count must be integer-equivalent.")

    for (horizon, seed, frequency), grp in dom_usage.groupby(["horizon", "seed", "frequency"]):
        expected_patches = set(DAILY_PATCHES if frequency == "daily" else WEEKLY_PATCHES)
        if set(grp["patch_size"].tolist()) != expected_patches:
            raise RuntimeError(f"Output 3 patch set invalid for {horizon} {seed} {frequency}.")
        counts = grp["dominant_count"].to_numpy(dtype=np.float64)
        if np.any(counts < 0) or np.any(counts > N_CELL_ROWS):
            raise RuntimeError(f"Output 3 dominant_count out of [0,6647] for {horizon} {seed} {frequency}.")
        if int(np.sum(counts)) != N_CELL_ROWS:
            raise RuntimeError(f"Output 3 dominant_count values do not sum to 6647 for {horizon} {seed} {frequency}.")
        shares = grp["dominant_share"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(shares)) or np.any(shares < -RANGE_TOL) or np.any(shares > 1.0 + RANGE_TOL):
            raise RuntimeError(f"Output 3 dominant_share out of [0,1] for {horizon} {seed} {frequency}.")
        expected_shares = counts / float(N_CELL_ROWS)
        if np.any(np.abs(shares - expected_shares) > RANGE_TOL):
            raise RuntimeError(f"Output 3 dominant_share does not equal dominant_count/6647 for {horizon} {seed} {frequency}.")
        if abs(float(np.sum(shares, dtype=np.float64)) - 1.0) > 1e-12:
            raise RuntimeError(f"Output 3 dominant_share values do not sum to 1 within 1e-12 for {horizon} {seed} {frequency}.")


def _validate_output_4(stability_df: pd.DataFrame, *, serialized: bool = False) -> None:
    if list(stability_df.columns) != EXACT_OUTPUT_4_COLUMNS:
        raise RuntimeError("Output 4 columns do not exactly match the frozen schema.")
    if len(stability_df) != N_CROSS_SEED_STABILITY_ROWS:
        raise RuntimeError("Output 4 row count mismatch.")
    if set(stability_df["horizon"].unique().tolist()) != set(HORIZONS):
        raise RuntimeError("Output 4 horizon values invalid.")
    if set(stability_df["frequency"].unique().tolist()) != set(FREQUENCIES):
        raise RuntimeError("Output 4 frequency values invalid.")

    anchor_date = _coerce_anchor_date(stability_df["anchor_date"], serialized=serialized)
    if anchor_date.isnull().any():
        raise RuntimeError("Output 4 contains null anchor_date values.")

    check_df = stability_df.copy()
    check_df["anchor_date"] = anchor_date

    key_df = check_df[["horizon", "ticker", "anchor_date", "frequency"]]
    if key_df.duplicated().any():
        raise RuntimeError("Output 4 contains duplicate horizon/ticker/anchor_date/frequency keys.")

    cell_counts = check_df.groupby(["horizon", "frequency"]).size()
    if len(cell_counts) != len(HORIZONS) * len(FREQUENCIES):
        raise RuntimeError("Output 4 horizon/frequency grid incomplete.")
    if not np.all(cell_counts.to_numpy() == N_CELL_ROWS):
        raise RuntimeError("Output 4 every horizon/frequency cell must have exactly 6647 rows.")

    for (horizon, frequency), grp in check_df.groupby(["horizon", "frequency"]):
        if grp["ticker"].nunique() != N_TICKERS:
            raise RuntimeError(f"Output 4 ticker count invalid for {horizon} {frequency}.")
        if grp["anchor_date"].nunique() != N_TEST_DATES:
            raise RuntimeError(f"Output 4 date count invalid for {horizon} {frequency}.")
        per_date = grp.groupby("anchor_date").size()
        if not np.all(per_date.to_numpy() == N_TICKERS):
            raise RuntimeError(f"Output 4 rows-per-date invalid for {horizon} {frequency}.")

    _assert_canonical_order(
        check_df, ["horizon", "ticker", "anchor_date", "frequency"], ["horizon", "ticker", "anchor_date", "frequency"],
        {"horizon": HORIZON_ORDER, "frequency": FREQUENCY_ORDER},
    )

    if not np.all(stability_df["n_seeds"].to_numpy() == 5):
        raise RuntimeError("Output 4 n_seeds must be 5 everywhere.")

    mean_tv = stability_df["mean_pairwise_seed_tv"].to_numpy(dtype=np.float64)
    max_tv = stability_df["max_pairwise_seed_tv"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(mean_tv)) or np.any(mean_tv < -RANGE_TOL) or np.any(mean_tv > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 4 mean_pairwise_seed_tv out of [0,1].")
    if not np.all(np.isfinite(max_tv)) or np.any(max_tv < -RANGE_TOL) or np.any(max_tv > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 4 max_pairwise_seed_tv out of [0,1].")
    if np.any(max_tv < mean_tv - RANGE_TOL):
        raise RuntimeError("Output 4 max_pairwise_seed_tv must be >= mean_pairwise_seed_tv.")

    agreement = stability_df["dominant_agreement_share"].to_numpy(dtype=np.float64)
    allowed = np.array([0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    if not np.all(np.isin(agreement, allowed)):
        raise RuntimeError("Output 4 dominant_agreement_share values invalid.")

    unanimous_raw = stability_df["dominant_unanimous"].to_numpy(dtype=np.float64)
    if not np.all((unanimous_raw == 0.0) | (unanimous_raw == 1.0)):
        raise RuntimeError("Output 4 dominant_unanimous values invalid.")
    expected_unanimous = (agreement == 1.0).astype(np.float64)
    if not np.array_equal(unanimous_raw, expected_unanimous):
        raise RuntimeError("Output 4 dominant_unanimous is not logically equivalent to dominant_agreement_share == 1.0.")


def _validate_output_5(summary_df: pd.DataFrame) -> None:
    if list(summary_df.columns) != EXACT_OUTPUT_5_COLUMNS:
        raise RuntimeError("Output 5 columns do not exactly match the frozen schema.")
    if len(summary_df) != 6:
        raise RuntimeError("Output 5 row count mismatch.")
    if set(summary_df["horizon"].unique().tolist()) != set(HORIZONS):
        raise RuntimeError("Output 5 horizon values invalid.")
    if set(summary_df["frequency"].unique().tolist()) != set(FREQUENCIES):
        raise RuntimeError("Output 5 frequency values invalid.")

    key_df = summary_df[["horizon", "frequency"]]
    if key_df.duplicated().any():
        raise RuntimeError("Output 5 contains duplicate horizon/frequency keys.")
    if len(key_df) != len(HORIZONS) * len(FREQUENCIES):
        raise RuntimeError("Output 5 horizon/frequency grid incomplete.")

    _assert_canonical_order(
        summary_df, ["horizon", "frequency"], ["horizon", "frequency"],
        {"horizon": HORIZON_ORDER, "frequency": FREQUENCY_ORDER},
    )

    if not np.all(summary_df["n_samples"].to_numpy() == N_CELL_ROWS):
        raise RuntimeError("Output 5 n_samples must equal 6647 everywhere.")

    numeric_cols = [c for c in EXACT_OUTPUT_5_COLUMNS if c not in ("horizon", "frequency", "n_samples")]
    numeric_arr = summary_df[numeric_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric_arr)):
        raise RuntimeError("Output 5 must not contain NaN/inf in any diagnostic field.")

    tv_cols = [
        "mean_pairwise_seed_tv_mean", "mean_pairwise_seed_tv_median", "mean_pairwise_seed_tv_p90",
        "max_pairwise_seed_tv_mean", "max_pairwise_seed_tv_median", "max_pairwise_seed_tv_p90",
    ]
    tv_arr = summary_df[tv_cols].to_numpy(dtype=np.float64)
    if np.any(tv_arr < -RANGE_TOL) or np.any(tv_arr > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 5 TV-based summary fields out of [0,1].")
    if np.any(summary_df["mean_pairwise_seed_tv_sd"].to_numpy(dtype=np.float64) < -RANGE_TOL):
        raise RuntimeError("Output 5 mean_pairwise_seed_tv_sd must be >= 0.")

    agreement_arr = summary_df[["dominant_agreement_share_mean", "dominant_agreement_share_median", "dominant_unanimous_share"]].to_numpy(dtype=np.float64)
    if np.any(agreement_arr < -RANGE_TOL) or np.any(agreement_arr > 1.0 + RANGE_TOL):
        raise RuntimeError("Output 5 dominant agreement/unanimity fields out of [0,1].")


def _validate_output_6(loss_assoc_df: pd.DataFrame) -> None:
    if list(loss_assoc_df.columns) != EXACT_OUTPUT_6_COLUMNS:
        raise RuntimeError("Output 6 columns do not exactly match the frozen schema.")
    if len(loss_assoc_df) != 30:
        raise RuntimeError("Output 6 row count mismatch.")
    if set(loss_assoc_df["horizon"].unique().tolist()) != set(HORIZONS):
        raise RuntimeError("Output 6 horizon values invalid.")
    if set(loss_assoc_df["seed"].unique().tolist()) != set(SEEDS):
        raise RuntimeError("Output 6 seed values invalid.")
    if set(loss_assoc_df["frequency"].unique().tolist()) != set(FREQUENCIES):
        raise RuntimeError("Output 6 frequency values invalid.")

    key_df = loss_assoc_df[["horizon", "seed", "frequency"]]
    if key_df.duplicated().any():
        raise RuntimeError("Output 6 contains duplicate horizon/seed/frequency keys.")
    if len(key_df) != len(HORIZONS) * len(SEEDS) * len(FREQUENCIES):
        raise RuntimeError("Output 6 horizon/seed/frequency grid incomplete.")

    _assert_canonical_order(
        loss_assoc_df, ["horizon", "seed", "frequency"], ["horizon", "seed", "frequency"],
        {"horizon": HORIZON_ORDER, "frequency": FREQUENCY_ORDER},
    )

    if not np.all(loss_assoc_df["n_samples"].to_numpy() == N_CELL_ROWS):
        raise RuntimeError("Output 6 n_samples must equal 6647 everywhere.")
    if not np.all(loss_assoc_df["n_dates"].to_numpy() == N_TEST_DATES):
        raise RuntimeError("Output 6 n_dates must equal 391 everywhere.")

    sample_defined = _coerce_bool(loss_assoc_df["sample_level_correlation_defined"])
    date_defined = _coerce_bool(loss_assoc_df["date_level_correlation_defined"])
    sample_rho = loss_assoc_df["sample_level_spearman_rho"].to_numpy(dtype=np.float64)
    date_rho = loss_assoc_df["date_level_spearman_rho"].to_numpy(dtype=np.float64)

    for rho, defined, label in ((sample_rho, sample_defined, "sample_level"), (date_rho, date_defined, "date_level")):
        if not np.all(np.isfinite(rho[defined])):
            raise RuntimeError(f"Output 6 {label}_spearman_rho must be finite where defined.")
        if np.any(rho[defined] < -1.0 - RANGE_TOL) or np.any(rho[defined] > 1.0 + RANGE_TOL):
            raise RuntimeError(f"Output 6 {label}_spearman_rho out of [-1,1] where defined.")
        if not np.all(np.isnan(rho[~defined])):
            raise RuntimeError(f"Output 6 {label}_spearman_rho must be NaN where undefined.")


def _validate_output_7(summary_df: pd.DataFrame) -> None:
    if list(summary_df.columns) != EXACT_OUTPUT_7_COLUMNS:
        raise RuntimeError("Output 7 columns do not exactly match the frozen schema.")
    if len(summary_df) != 6:
        raise RuntimeError("Output 7 row count mismatch.")
    if set(summary_df["horizon"].unique().tolist()) != set(HORIZONS):
        raise RuntimeError("Output 7 horizon values invalid.")
    if set(summary_df["frequency"].unique().tolist()) != set(FREQUENCIES):
        raise RuntimeError("Output 7 frequency values invalid.")

    key_df = summary_df[["horizon", "frequency"]]
    if key_df.duplicated().any():
        raise RuntimeError("Output 7 contains duplicate horizon/frequency keys.")
    if len(key_df) != len(HORIZONS) * len(FREQUENCIES):
        raise RuntimeError("Output 7 horizon/frequency grid incomplete.")

    _assert_canonical_order(
        summary_df, ["horizon", "frequency"], ["horizon", "frequency"],
        {"horizon": HORIZON_ORDER, "frequency": FREQUENCY_ORDER},
    )

    for prefix in ("sample_level", "date_level"):
        n_defined = summary_df[f"{prefix}_n_defined"].to_numpy(dtype=np.float64)
        n_positive = summary_df[f"{prefix}_n_positive"].to_numpy(dtype=np.float64)
        n_negative = summary_df[f"{prefix}_n_negative"].to_numpy(dtype=np.float64)
        n_zero = summary_df[f"{prefix}_n_zero"].to_numpy(dtype=np.float64)
        for arr, label in ((n_defined, "n_defined"), (n_positive, "n_positive"), (n_negative, "n_negative"), (n_zero, "n_zero")):
            if not np.all(arr == np.round(arr)):
                raise RuntimeError(f"Output 7 {prefix}_{label} must be integer-equivalent.")
            if np.any(arr < 0) or np.any(arr > 5):
                raise RuntimeError(f"Output 7 {prefix}_{label} out of [0,5].")
        if not np.array_equal(n_defined, n_positive + n_negative + n_zero):
            raise RuntimeError(f"Output 7 {prefix} sign-count arithmetic invalid.")

        mean_v = summary_df[f"{prefix}_rho_mean"].to_numpy(dtype=np.float64)
        median_v = summary_df[f"{prefix}_rho_median"].to_numpy(dtype=np.float64)
        sd_v = summary_df[f"{prefix}_rho_sd"].to_numpy(dtype=np.float64)

        zero_mask = n_defined == 0
        one_mask = n_defined == 1
        many_mask = n_defined >= 2

        if not np.all(np.isnan(mean_v[zero_mask])) or not np.all(np.isnan(sd_v[zero_mask])) or not np.all(np.isnan(median_v[zero_mask])):
            raise RuntimeError(f"Output 7 {prefix} rows with n_defined==0 must have NaN mean/sd/median.")
        if not np.all(n_positive[zero_mask] == 0) or not np.all(n_negative[zero_mask] == 0) or not np.all(n_zero[zero_mask] == 0):
            raise RuntimeError(f"Output 7 {prefix} rows with n_defined==0 must have zero sign counts.")

        if not np.all(np.isfinite(mean_v[one_mask])) or not np.all(np.isfinite(median_v[one_mask])):
            raise RuntimeError(f"Output 7 {prefix} rows with n_defined==1 must have finite mean/median.")
        if np.any(np.abs(mean_v[one_mask] - median_v[one_mask]) > RANGE_TOL):
            raise RuntimeError(f"Output 7 {prefix} rows with n_defined==1 must have mean == median.")
        if not np.all(np.isnan(sd_v[one_mask])):
            raise RuntimeError(f"Output 7 {prefix} rows with n_defined==1 must have NaN sd.")

        if not np.all(np.isfinite(mean_v[many_mask])) or not np.all(np.isfinite(median_v[many_mask])) or not np.all(np.isfinite(sd_v[many_mask])):
            raise RuntimeError(f"Output 7 {prefix} rows with n_defined>=2 must have finite mean/median/sd.")
        if np.any(sd_v[many_mask] < -RANGE_TOL):
            raise RuntimeError(f"Output 7 {prefix} rows with n_defined>=2 must have sd >= 0.")

        defined_mask = one_mask | many_mask
        if np.any(mean_v[defined_mask] < -1.0 - RANGE_TOL) or np.any(mean_v[defined_mask] > 1.0 + RANGE_TOL):
            raise RuntimeError(f"Output 7 {prefix}_rho_mean out of [-1,1].")
        if np.any(median_v[defined_mask] < -1.0 - RANGE_TOL) or np.any(median_v[defined_mask] > 1.0 + RANGE_TOL):
            raise RuntimeError(f"Output 7 {prefix}_rho_median out of [-1,1].")


VALIDATORS = {
    "output_1": _validate_output_1,
    "output_2": _validate_output_2,
    "output_3": _validate_output_3,
    "output_4": _validate_output_4,
    "output_5": _validate_output_5,
    "output_6": _validate_output_6,
    "output_7": _validate_output_7,
}
SERIALIZED_AWARE_OUTPUTS = {"output_1", "output_4"}


def _validate_serialized_temp_output(key: str, path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Temporary output missing after serialization: {path}")
    validator = VALIDATORS.get(key)
    if validator is None:
        raise RuntimeError(f"No validator registered for serialized output key: {key}")
    df = pd.read_csv(path)
    if key in SERIALIZED_AWARE_OUTPUTS:
        validator(df, serialized=True)
    else:
        validator(df)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if "anchor_date" in out.columns:
        out["anchor_date"] = pd.to_datetime(out["anchor_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _build_summary_text(cell_summary: pd.DataFrame, dom_usage: pd.DataFrame, cross_summary: pd.DataFrame, assoc_summary: pd.DataFrame) -> str:
    lines = []
    lines.append("Experiment 5 router diagnostics summary")
    lines.append("Interpretation remains descriptive and comparative only; it does not override the final Experiment 4 inference result.")
    lines.append("")
    for horizon in HORIZONS:
        for frequency in FREQUENCIES:
            cell = cell_summary[(cell_summary["horizon"] == horizon) & (cell_summary["frequency"] == frequency)].copy()
            if len(cell) != len(SEEDS) or set(cell["seed"].tolist()) != set(SEEDS):
                raise RuntimeError(f"Human summary requires exactly the five canonical seed rows for {horizon} {frequency}.")

            cross_rows = cross_summary[(cross_summary["horizon"] == horizon) & (cross_summary["frequency"] == frequency)]
            if len(cross_rows) != 1:
                raise RuntimeError(f"Human summary requires exactly one Output 5 row for {horizon} {frequency}.")
            cross = cross_rows.iloc[0]

            assoc_rows = assoc_summary[(assoc_summary["horizon"] == horizon) & (assoc_summary["frequency"] == frequency)]
            if len(assoc_rows) != 1:
                raise RuntimeError(f"Human summary requires exactly one Output 7 row for {horizon} {frequency}.")
            assoc = assoc_rows.iloc[0]

            patch_order = DAILY_PATCHES if frequency == "daily" else WEEKLY_PATCHES
            mean_alloc = {
                patch_order[0]: float(cell["mean_weight_1"].mean()),
                patch_order[1]: float(cell["mean_weight_2"].mean()),
                patch_order[2]: float(cell["mean_weight_3"].mean()),
                patch_order[3]: float(cell["mean_weight_4"].mean()),
            }

            dom_grp = dom_usage[(dom_usage["horizon"] == horizon) & (dom_usage["frequency"] == frequency)].copy()
            avg_dom = {}
            for patch in patch_order:
                patch_rows = dom_grp[dom_grp["patch_size"] == patch]
                if len(patch_rows) != len(SEEDS) or set(patch_rows["seed"].tolist()) != set(SEEDS):
                    raise RuntimeError(f"Human summary requires exactly the five canonical seed rows for patch {patch} in {horizon} {frequency}.")
                avg_dom[patch] = float(patch_rows["dominant_share"].mean())

            lines.append(f"{horizon} / {frequency}")
            lines.append(f"  mean scale allocation (patch size -> mean weight): {mean_alloc}")
            lines.append(f"  sample_dependence_index: {float(cell['sample_dependence_index'].mean())}")
            lines.append(f"  within_ticker_temporal_tv: {float(cell['within_ticker_temporal_tv'].mean())}")
            lines.append(f"  within_date_crosssectional_tv: {float(cell['within_date_crosssectional_tv'].mean())}")
            lines.append(f"  normalized_entropy_mean: {float(cell['normalized_entropy_mean'].mean())}")
            lines.append(f"  effective_scales_mean: {float(cell['effective_scales_mean'].mean())}")
            lines.append(f"  max_weight_mean: {float(cell['max_weight_mean'].mean())}")
            lines.append(f"  dominant_scale_shares_mean (patch size -> share): {avg_dom}")
            lines.append(f"  tv_cell_mean_to_static: {float(cell['tv_cell_mean_to_static'].mean())}")
            lines.append(f"  tv_to_static_mean: {float(cell['tv_to_static_mean'].mean())}")
            lines.append(f"  cross_seed_mean_pairwise_seed_tv: {float(cross['mean_pairwise_seed_tv_mean'])}")
            lines.append(f"  dominant_agreement_share: {float(cross['dominant_agreement_share_mean'])}")
            lines.append(f"  sample_level_rho_mean: {float(assoc['sample_level_rho_mean'])} (n_defined={int(assoc['sample_level_n_defined'])})")
            lines.append(f"  date_level_rho_mean: {float(assoc['date_level_rho_mean'])} (n_defined={int(assoc['date_level_n_defined'])})")
            lines.append("")
    lines.append("Experiment 5 does not override Experiment 4 inference.")
    lines.append("Branch-specific TV/loss associations use the same joint D+W forecast loss and do not identify causal or separable Daily or Weekly contributions.")
    lines.append("Interpretation remains descriptive and comparative only.")
    return "\n".join(lines) + "\n"


def _build_manifest(sample_df: pd.DataFrame, cell_summary: pd.DataFrame, dom_usage: pd.DataFrame, cross_seed_stability: pd.DataFrame, cross_summary: pd.DataFrame, loss_assoc_df: pd.DataFrame, assoc_summary: pd.DataFrame, summary_text: str, output_hashes: dict[str, str]) -> pd.DataFrame:
    manifest = {
        "run_family": "EXPERIMENT5_ROUTER_DIAGNOSTICS",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract": "V2",
        "diagnostic_scope": "ADAPTIVE_ROUTER_FAILURE_ANALYSIS",
        "mechanisms": "adaptive,static",
        "frequency_configuration": "daily_weekly",
        "frequencies": "daily,weekly",
        "horizons": "5d,10d,20d",
        "seeds": "0,1,21,42,3407",
        "n_tickers": 17,
        "n_test_dates": 391,
        "n_samples_per_cell": 6647,
        "daily_patch_order": "5,10,20,30",
        "weekly_patch_order": "2,4,8,13",
        "distance_metric": "TOTAL_VARIATION",
        "entropy_log_base": "NATURAL_LOG",
        "entropy_normalization": "LN_4",
        "effective_scale_definition": "EXP_ENTROPY",
        "dominant_tie_rule": "FIRST_CANONICAL_SCALE",
        "sample_dependence_definition": "MEAN_TV_TO_CELL_MEAN",
        "temporal_variation_definition": "EQUAL_WEIGHT_TICKER_MEAN_TV",
        "crosssectional_variation_definition": "EQUAL_WEIGHT_DATE_MEAN_TV",
        "cross_seed_definition": "PAIRWISE_TV_ACROSS_FIVE_SEEDS",
        "loss_association_definition": "MANUAL_SPEARMAN_ON_TV_TO_STATIC_VS_LOSS_DIFF",
        "spearman_tie_rule": "AVERAGE_RANKS",
        "no_training": True,
        "no_checkpoint_loading": True,
        "no_fresh_test_predictions": True,
        "no_hypothesis_testing": True,
        "no_regime_analysis": True,
        "no_subset_model_selection": True,
        "protocol_path": "docs/experiment5_router_diagnostics_protocol.md",
        "protocol_sha256": PROTOCOL_SHA256,
        "frozen_input_1_path": "dataset/audit/experiment4_multiseed/experiment4_scale_weights.csv",
        "frozen_input_1_sha256": FROZEN_INPUTS["dataset/audit/experiment4_multiseed/experiment4_scale_weights.csv"]["sha256"],
        "frozen_input_2_path": "dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv",
        "frozen_input_2_sha256": FROZEN_INPUTS["dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv"]["sha256"],
        "frozen_input_3_path": "dataset/audit/main_v2/pathformer/panel_pathformer_router_weights.csv",
        "frozen_input_3_sha256": FROZEN_INPUTS["dataset/audit/main_v2/pathformer/panel_pathformer_router_weights.csv"]["sha256"],
        "frozen_input_4_path": "dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv",
        "frozen_input_4_sha256": FROZEN_INPUTS["dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv"]["sha256"],
        "frozen_input_5_path": "dataset/audit/experiment3_pathformer_ablation/experiment3_scale_weights.csv",
        "frozen_input_5_sha256": FROZEN_INPUTS["dataset/audit/experiment3_pathformer_ablation/experiment3_scale_weights.csv"]["sha256"],
        "frozen_input_6_path": "dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv",
        "frozen_input_6_sha256": FROZEN_INPUTS["dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv"]["sha256"],
        "frozen_input_7_path": "dataset/audit/experiment4_inference/experiment4_inference_manifest.csv",
        "frozen_input_7_sha256": FROZEN_INPUTS["dataset/audit/experiment4_inference/experiment4_inference_manifest.csv"]["sha256"],
        "frozen_input_8_path": "dataset/audit/experiment4_inference/experiment4_inference_summary.txt",
        "frozen_input_8_sha256": FROZEN_INPUTS["dataset/audit/experiment4_inference/experiment4_inference_summary.txt"]["sha256"],
        "frozen_input_9_path": "dataset/audit/experiment4_multiseed/experiment4_seed_pair_deltas.csv",
        "frozen_input_9_sha256": FROZEN_INPUTS["dataset/audit/experiment4_multiseed/experiment4_seed_pair_deltas.csv"]["sha256"],
        "output_1_rows": int(len(sample_df)),
        "output_2_rows": int(len(cell_summary)),
        "output_3_rows": int(len(dom_usage)),
        "output_4_rows": int(len(cross_seed_stability)),
        "output_5_rows": int(len(cross_summary)),
        "output_6_rows": int(len(loss_assoc_df)),
        "output_7_rows": int(len(assoc_summary)),
        "output_8_rows": 1,
        "output_1_sha256": output_hashes["output_1"],
        "output_2_sha256": output_hashes["output_2"],
        "output_3_sha256": output_hashes["output_3"],
        "output_4_sha256": output_hashes["output_4"],
        "output_5_sha256": output_hashes["output_5"],
        "output_6_sha256": output_hashes["output_6"],
        "output_7_sha256": output_hashes["output_7"],
        "output_8_sha256": output_hashes["output_8"],
    }
    return pd.DataFrame([manifest])


GOVERNANCE_FLAGS = (
    "no_training", "no_checkpoint_loading", "no_fresh_test_predictions",
    "no_hypothesis_testing", "no_regime_analysis", "no_subset_model_selection",
)

# Frozen manifest literals (Section 34 of the protocol). Excludes protocol/frozen-input
# hashes and row counts, which are checked separately below against their own sources of truth.
EXPECTED_MANIFEST_LITERALS = {
    "run_family": "EXPERIMENT5_ROUTER_DIAGNOSTICS",
    "temporal_status": "POST_TEMPORAL_FIX",
    "dataset_contract": "V2",
    "diagnostic_scope": "ADAPTIVE_ROUTER_FAILURE_ANALYSIS",
    "mechanisms": "adaptive,static",
    "frequency_configuration": "daily_weekly",
    "frequencies": "daily,weekly",
    "horizons": "5d,10d,20d",
    "seeds": "0,1,21,42,3407",
    "n_tickers": 17,
    "n_test_dates": 391,
    "n_samples_per_cell": 6647,
    "daily_patch_order": "5,10,20,30",
    "weekly_patch_order": "2,4,8,13",
    "distance_metric": "TOTAL_VARIATION",
    "entropy_log_base": "NATURAL_LOG",
    "entropy_normalization": "LN_4",
    "effective_scale_definition": "EXP_ENTROPY",
    "dominant_tie_rule": "FIRST_CANONICAL_SCALE",
    "sample_dependence_definition": "MEAN_TV_TO_CELL_MEAN",
    "temporal_variation_definition": "EQUAL_WEIGHT_TICKER_MEAN_TV",
    "crosssectional_variation_definition": "EQUAL_WEIGHT_DATE_MEAN_TV",
    "cross_seed_definition": "PAIRWISE_TV_ACROSS_FIVE_SEEDS",
    "loss_association_definition": "MANUAL_SPEARMAN_ON_TV_TO_STATIC_VS_LOSS_DIFF",
    "spearman_tie_rule": "AVERAGE_RANKS",
    "protocol_path": "docs/experiment5_router_diagnostics_protocol.md",
    "protocol_sha256": PROTOCOL_SHA256,
}

EXPECTED_MANIFEST_ROW_COUNTS = {
    "output_1_rows": N_SAMPLE_METRICS_ROWS,
    "output_2_rows": 30,
    "output_3_rows": 120,
    "output_4_rows": N_CROSS_SEED_STABILITY_ROWS,
    "output_5_rows": 6,
    "output_6_rows": 30,
    "output_7_rows": 6,
    "output_8_rows": 1,
}


def _validate_manifest(manifest_df: pd.DataFrame, expected_output_hashes: dict[str, str]) -> None:
    if len(manifest_df) != 1:
        raise RuntimeError("Manifest must contain exactly one row.")
    row = manifest_df.iloc[0]

    for field, expected in EXPECTED_MANIFEST_LITERALS.items():
        if row[field] != expected:
            raise RuntimeError(f"Manifest field {field} does not match the frozen literal: expected {expected}, got {row[field]}.")

    for flag in GOVERNANCE_FLAGS:
        value = row[flag]
        if isinstance(value, (bool, np.bool_)):
            normalized = bool(value)
        elif isinstance(value, str):
            normalized_str = value.strip().lower()
            if normalized_str == "true":
                normalized = True
            elif normalized_str == "false":
                normalized = False
            else:
                raise RuntimeError(f"Manifest governance flag {flag} has an unrecognized value: {value}.")
        else:
            raise RuntimeError(f"Manifest governance flag {flag} has an unrecognized type: {type(value)}.")
        if not normalized:
            raise RuntimeError(f"Manifest governance flag {flag} must be true. Fail closed.")

    for idx, (rel_path, meta) in enumerate(FROZEN_INPUTS.items(), start=1):
        path_col = f"frozen_input_{idx}_path"
        sha_col = f"frozen_input_{idx}_sha256"
        if row[path_col] != rel_path:
            raise RuntimeError(f"Manifest {path_col} does not match the registered frozen input path: expected {rel_path}, got {row[path_col]}.")
        if row[sha_col] != meta["sha256"]:
            raise RuntimeError(f"Manifest {sha_col} does not match the frozen input hash for {rel_path}.")

    for field, expected in EXPECTED_MANIFEST_ROW_COUNTS.items():
        if int(row[field]) != expected:
            raise RuntimeError(f"Manifest {field} does not match the required row count: expected {expected}, got {row[field]}.")

    for key, expected_hash in expected_output_hashes.items():
        col = f"{key}_sha256"
        if row[col] != expected_hash:
            raise RuntimeError(f"Manifest {col} does not match the expected validated output hash.")


def _run_transactional_commit() -> None:
    _verify_protocol()
    _verify_frozen_inputs()
    _require_final_outputs_absent()

    # Build outputs in memory.
    weight_df = _build_weight_table()
    static_ref = _build_static_reference_vectors(weight_df)
    loss_df = _build_loss_diffs(_build_prediction_table())
    _build_loss_bridge_audit(loss_df)
    sample_df = _build_adaptive_sample_metrics(weight_df, loss_df, static_ref)
    _validate_output_1(sample_df)
    cell_summary = _build_cell_summary(weight_df, sample_df, static_ref)
    _validate_output_2(cell_summary)
    dominants = _build_dominant_usage(sample_df)
    _validate_output_3(dominants)
    stability = _build_cross_seed_stability(sample_df)
    _validate_output_4(stability)
    cross_summary = _build_cross_seed_summary(stability)
    _validate_output_5(cross_summary)
    assoc_by_seed = _build_loss_association_by_seed(sample_df, loss_df)
    _validate_output_6(assoc_by_seed)
    assoc_summary = _build_loss_association_summary(assoc_by_seed)
    _validate_output_7(assoc_summary)

    summary_text = _build_summary_text(cell_summary, dominants, cross_summary, assoc_summary)
    output_frames = {
        "output_1": sample_df,
        "output_2": cell_summary,
        "output_3": dominants,
        "output_4": stability,
        "output_5": cross_summary,
        "output_6": assoc_by_seed,
        "output_7": assoc_summary,
    }
    OUTPUT_MAP = {
        "output_1": OUTPUT_DIR / "experiment5_router_sample_metrics.csv",
        "output_2": OUTPUT_DIR / "experiment5_router_cell_summary.csv",
        "output_3": OUTPUT_DIR / "experiment5_dominant_scale_usage.csv",
        "output_4": OUTPUT_DIR / "experiment5_cross_seed_stability_by_sample.csv",
        "output_5": OUTPUT_DIR / "experiment5_cross_seed_stability_summary.csv",
        "output_6": OUTPUT_DIR / "experiment5_loss_association_by_seed.csv",
        "output_7": OUTPUT_DIR / "experiment5_loss_association_summary.csv",
        "output_8": OUTPUT_DIR / "experiment5_router_diagnostics_summary.txt",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Predeclare the complete planned set of temp paths before writing anything, so
    # cleanup on any failure covers every planned artifact of this run.
    pid_suffix = os.getpid()
    planned_temp_paths: dict[str, Path] = {key: OUTPUT_DIR / f".tmp_{key}_{pid_suffix}.csv" for key in output_frames}
    planned_temp_paths["output_8"] = OUTPUT_DIR / f".tmp_output_8_{pid_suffix}.txt"
    planned_temp_paths["manifest"] = OUTPUT_DIR / f".tmp_manifest_{pid_suffix}.csv"
    temp_paths: list[Path] = list(planned_temp_paths.values())
    committed: list[Path] = []

    try:
        output_hashes: dict[str, str] = {}
        for key, df in output_frames.items():
            temp_path = planned_temp_paths[key]
            _write_csv(df, temp_path)
            # Validate the exact serialized bytes before trusting them, then hash the validated file.
            _validate_serialized_temp_output(key, temp_path)
            output_hashes[key] = _sha256_file(temp_path)

        summary_temp = planned_temp_paths["output_8"]
        with summary_temp.open("w", encoding="utf-8") as fh:
            fh.write(summary_text)
        with summary_temp.open("r", encoding="utf-8") as fh:
            summary_readback = fh.read()
        if summary_readback != summary_text:
            raise RuntimeError("Summary text readback mismatch.")
        output_hashes["output_8"] = _sha256_file(summary_temp)

        manifest_df = _build_manifest(sample_df, cell_summary, dominants, stability, cross_summary, assoc_by_seed, assoc_summary, summary_text, output_hashes)
        manifest_temp = planned_temp_paths["manifest"]
        _write_csv(manifest_df, manifest_temp)
        manifest_readback = pd.read_csv(manifest_temp)
        _validate_manifest(manifest_readback, output_hashes)

        # Secure commit: rename each validated temp file into place, tracking exactly what this run committed.
        for key, final_path in OUTPUT_MAP.items():
            os.replace(planned_temp_paths[key], final_path)
            committed.append(final_path)
        os.replace(manifest_temp, OUTPUT_DIR / "experiment5_router_diagnostics_manifest.csv")
        committed.append(OUTPUT_DIR / "experiment5_router_diagnostics_manifest.csv")

        # Post-commit verification.
        for path in FINAL_OUTPUTS:
            if not path.exists():
                raise RuntimeError(f"Final output missing after commit: {path}")
        sample_final = pd.read_csv(OUTPUT_DIR / "experiment5_router_sample_metrics.csv")
        cell_final = pd.read_csv(OUTPUT_DIR / "experiment5_router_cell_summary.csv")
        dom_final = pd.read_csv(OUTPUT_DIR / "experiment5_dominant_scale_usage.csv")
        stab_final = pd.read_csv(OUTPUT_DIR / "experiment5_cross_seed_stability_by_sample.csv")
        cross_final = pd.read_csv(OUTPUT_DIR / "experiment5_cross_seed_stability_summary.csv")
        assoc_seed_final = pd.read_csv(OUTPUT_DIR / "experiment5_loss_association_by_seed.csv")
        assoc_summary_final = pd.read_csv(OUTPUT_DIR / "experiment5_loss_association_summary.csv")
        _validate_output_1(sample_final, serialized=True)
        _validate_output_2(cell_final)
        _validate_output_3(dom_final)
        _validate_output_4(stab_final, serialized=True)
        _validate_output_5(cross_final)
        _validate_output_6(assoc_seed_final)
        _validate_output_7(assoc_summary_final)
        summary_final = (OUTPUT_DIR / "experiment5_router_diagnostics_summary.txt").read_text(encoding="utf-8")
        if summary_final != summary_text:
            raise RuntimeError("Final summary text mismatch.")

        # Confirm hashes recorded in manifest match actual output files, using the same
        # complete manifest validator that was used pre-commit.
        actual_output_hashes = {
            "output_1": _sha256_file(OUTPUT_DIR / "experiment5_router_sample_metrics.csv"),
            "output_2": _sha256_file(OUTPUT_DIR / "experiment5_router_cell_summary.csv"),
            "output_3": _sha256_file(OUTPUT_DIR / "experiment5_dominant_scale_usage.csv"),
            "output_4": _sha256_file(OUTPUT_DIR / "experiment5_cross_seed_stability_by_sample.csv"),
            "output_5": _sha256_file(OUTPUT_DIR / "experiment5_cross_seed_stability_summary.csv"),
            "output_6": _sha256_file(OUTPUT_DIR / "experiment5_loss_association_by_seed.csv"),
            "output_7": _sha256_file(OUTPUT_DIR / "experiment5_loss_association_summary.csv"),
            "output_8": _sha256_file(OUTPUT_DIR / "experiment5_router_diagnostics_summary.txt"),
        }
        manifest_final = pd.read_csv(OUTPUT_DIR / "experiment5_router_diagnostics_manifest.csv")
        _validate_manifest(manifest_final, actual_output_hashes)
        print("EXPERIMENT 5 ROUTER DIAGNOSTICS COMPLETE")
    except Exception:
        # Roll back only artifacts produced by this run: leftover temp files and any
        # final outputs this run itself committed. Never touch pre-existing evidence.
        for temp in temp_paths:
            if temp.exists():
                temp.unlink()
        for final_path in committed:
            if final_path.exists():
                final_path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 5 router diagnostics read-only audit.")
    parser.add_argument("--run-diagnostics", action="store_true", help="Run the deterministic frozen read-only Experiment 5 diagnostics.")
    args = parser.parse_args()
    if not args.run_diagnostics:
        parser.print_help()
        return 0
    _run_transactional_commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
