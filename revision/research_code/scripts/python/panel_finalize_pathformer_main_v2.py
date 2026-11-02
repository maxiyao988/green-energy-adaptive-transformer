"""Standalone immutable finalizer for the frozen PathFormer MAIN V2 CPU evidence.

This is NOT training and NOT a modification of the frozen training script at
scripts/python/panel_pathformer_main_v2.py. It only reads existing raw
evidence, re-validates it, builds the accepted baseline comparison, and
writes the finalization-only artifacts (comparison, manifest, final summary).

Rationale: the original in-module finalizer's cross-frequency y_true identity
check used bitwise np.array_equal, which fails on a float32 CSV round-trip
representation difference (observed max abs diff ~2.9e-08, well under one
float32 machine epsilon ~1.19e-07). This is a serialization artifact, not a
real target mismatch (dataset/model/training/router/metrics are unaffected).
This standalone finalizer replaces ONLY that specific numeric comparison with
a float32-aware tolerance (rtol=0, atol=np.finfo(np.float32).eps) while
preserving exact ticker/date key-set equality and every other validation
unchanged. It never modifies raw evidence, provenance, or any previously
failed/archived run directory.

The frozen module is imported (not copied) and is safe to import because it
guards training/CLI execution behind `if __name__ == "__main__": main()`.
This finalizer never monkeypatches the frozen module or mutates its globals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.python.panel_pathformer_main_v2 as formal

TRAINING_SCRIPT_RELATIVE = "scripts/python/panel_pathformer_main_v2.py"
FINALIZER_SCRIPT_RELATIVE = "scripts/python/panel_finalize_pathformer_main_v2.py"

# Only the numeric cross-frequency y_true comparison is relaxed to this bound,
# to absorb float32 CSV round-trip representation only.
FLOAT32_ATOL = float(np.finfo(np.float32).eps)


def _require_immutable_provenance() -> pd.Series:
    """Read-only: provenance must already exist, be immutable, and match the current frozen script."""
    if not formal.FORMAL_PROVENANCE_PATH.exists():
        raise RuntimeError("Formal provenance is missing; standalone finalization requires immutable provenance to already exist.")
    provenance_df = pd.read_csv(formal.FORMAL_PROVENANCE_PATH)
    if provenance_df.empty:
        raise RuntimeError("Formal provenance file is empty.")
    if len(provenance_df) != 1:
        raise RuntimeError(f"Formal provenance must contain exactly one row; found {len(provenance_df)}.")
    formal._validate_stored_provenance(provenance_df)
    provenance_row = provenance_df.iloc[0]

    current_training_script_sha256 = formal._compute_sha256(ROOT / TRAINING_SCRIPT_RELATIVE)
    stored_training_script_sha256 = str(provenance_row["training_script_sha256"])
    if current_training_script_sha256 != stored_training_script_sha256:
        raise RuntimeError(
            "Training script SHA mismatch: formal training evidence belongs to a different "
            f"training-script version (stored={stored_training_script_sha256}, current={current_training_script_sha256})."
        )
    return provenance_row


def _validate_cross_frequency_target_identity_float32_safe(pred_df: pd.DataFrame) -> None:
    """Float32-safe replacement for formal._validate_cross_frequency_target_identity.

    Exact ticker/canonical-date key-set equality is preserved unchanged. Only
    the numeric y_true comparison is relaxed from bitwise np.array_equal to
    rtol=0, atol=np.finfo(np.float32).eps.
    """
    for horizon in formal.MAIN_HORIZONS:
        frames = {}
        for frequency in formal.FREQUENCY_SETTINGS:
            subset = pred_df[
                (pred_df["frequency_setting"].astype(str) == frequency)
                & (pred_df["horizon"].astype(int) == int(horizon))
            ].copy()
            if subset.empty:
                raise RuntimeError(f"Missing predictions for cross-frequency identity check: {frequency}/{horizon}.")
            subset["key"] = subset["ticker"].astype(str) + "|" + subset["anchor_date"].apply(formal._canonical_anchor_date)
            if subset["key"].duplicated().any():
                raise RuntimeError(f"Duplicate ticker/anchor_date keys detected for {frequency}/{horizon}.")
            frames[frequency] = subset.set_index("key")["y_true"].astype(np.float64)

        reference_frequency = formal.FREQUENCY_SETTINGS[0]
        reference_keys = set(frames[reference_frequency].index)
        for frequency, series in frames.items():
            if len(series) != 6647:
                raise RuntimeError(f"Expected exactly 6647 ticker/date keys for {frequency}/{horizon}, found {len(series)}.")
            if set(series.index) != reference_keys:
                raise RuntimeError(f"Cross-frequency ticker/anchor_date key set mismatch for horizon={horizon} at frequency={frequency}.")

        reference_series = frames[reference_frequency].sort_index()
        if reference_series.isna().any():
            raise RuntimeError(f"Reference y_true contains NaN for horizon={horizon}; NaN forbidden in identity check.")
        for frequency in formal.FREQUENCY_SETTINGS[1:]:
            other_series = frames[frequency].sort_index()
            if other_series.isna().any():
                raise RuntimeError(f"y_true for {frequency}/{horizon} contains NaN; NaN forbidden in identity check.")

            reference_values = reference_series.to_numpy()
            other_values = other_series.to_numpy()
            if reference_values.shape != other_values.shape:
                raise RuntimeError(f"Cross-frequency y_true length mismatch for horizon={horizon}: {reference_frequency} vs {frequency}.")

            max_abs_diff = float(np.max(np.abs(reference_values - other_values))) if len(reference_values) else 0.0
            is_close = bool(np.allclose(reference_values, other_values, rtol=0.0, atol=FLOAT32_ATOL, equal_nan=False))
            if not is_close or max_abs_diff > FLOAT32_ATOL:
                raise RuntimeError(
                    f"Cross-frequency y_true mismatch for horizon={horizon}: {reference_frequency} vs {frequency}: "
                    f"max_abs_diff={max_abs_diff}, FLOAT32_ATOL={FLOAT32_ATOL}"
                )


def _validate_full_evidence_float32_safe() -> dict[str, pd.DataFrame]:
    """Mirrors formal._validate_full_main_v2_evidence, except the cross-frequency
    y_true check is float32-safe instead of bitwise np.array_equal."""
    formal._combined_partial_validation()

    missing = [name for name, path in formal.RAW_EVIDENCE_PATHS.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing raw evidence files for full validation: {missing}")

    summary_df = pd.read_csv(formal.FORMAL_SUMMARY_PATH)
    pred_df = pd.read_csv(formal.FORMAL_PRED_PATH)
    rank_ic_df = pd.read_csv(formal.FORMAL_RANK_IC_PATH)
    history_df = pd.read_csv(formal.FORMAL_HISTORY_PATH)
    router_df = pd.read_csv(formal.FORMAL_ROUTER_PATH)

    if len(summary_df) != 9:
        raise RuntimeError(f"Expected exactly 9 summary rows, found {len(summary_df)}.")
    if len(pred_df) != 59823:
        raise RuntimeError(f"Expected exactly 59823 prediction rows, found {len(pred_df)}.")
    if len(rank_ic_df) != 3519:
        raise RuntimeError(f"Expected exactly 3519 Rank-IC rows, found {len(rank_ic_df)}.")
    if len(router_df) != 59823:
        raise RuntimeError(f"Expected exactly 59823 router rows, found {len(router_df)}.")

    completed_cells = set(
        (str(row["model"]), str(row["frequency_setting"]), int(row["horizon"]), int(row["seed"]))
        for _, row in summary_df.iterrows()
    )
    if len(completed_cells) != 9:
        raise RuntimeError(f"Expected exactly 9 completed formal cells, found {len(completed_cells)}.")
    if completed_cells != formal._formal_cell_grid():
        raise RuntimeError(f"Completed cell grid mismatch: {sorted(completed_cells)}")

    for _, summary_row in summary_df.iterrows():
        frequency = str(summary_row["frequency_setting"])
        horizon = int(summary_row["horizon"])
        seed = int(summary_row["seed"])
        model = str(summary_row["model"])

        cell_pred_df = pred_df[
            (pred_df["model"].astype(str) == model)
            & (pred_df["frequency_setting"].astype(str) == frequency)
            & (pred_df["horizon"].astype(int) == horizon)
            & (pred_df["seed"].astype(int) == seed)
        ].copy()
        cell_rank_ic_df = rank_ic_df[
            (rank_ic_df["model"].astype(str) == model)
            & (rank_ic_df["frequency_setting"].astype(str) == frequency)
            & (rank_ic_df["horizon"].astype(int) == horizon)
            & (rank_ic_df["seed"].astype(int) == seed)
        ].copy()
        cell_router_df = router_df[
            (router_df["model"].astype(str) == model)
            & (router_df["frequency_setting"].astype(str) == frequency)
            & (router_df["horizon"].astype(int) == horizon)
            & (router_df["seed"].astype(int) == seed)
        ].copy()

        if len(cell_pred_df) != 6647:
            raise RuntimeError(f"Expected exactly 6647 prediction rows for {frequency}/{horizon}, found {len(cell_pred_df)}.")
        if len(cell_rank_ic_df) != 391:
            raise RuntimeError(f"Expected exactly 391 Rank-IC rows for {frequency}/{horizon}, found {len(cell_rank_ic_df)}.")
        if len(cell_router_df) != 6647:
            raise RuntimeError(f"Expected exactly 6647 router rows for {frequency}/{horizon}, found {len(cell_router_df)}.")
        if cell_pred_df["ticker"].nunique() != 17:
            raise RuntimeError(f"Expected exactly 17 tickers in predictions for {frequency}/{horizon}.")
        if cell_pred_df["anchor_date"].nunique() != 391:
            raise RuntimeError(f"Expected exactly 391 anchor dates in predictions for {frequency}/{horizon}.")

        reconstructed_metrics = formal._reconstruct_cell_summary_metrics(cell_pred_df)
        reconstructed_rank_ic = formal._reconstruct_cell_rank_ic(cell_pred_df)
        formal._compare_reconstructed_rank_ic(reconstructed_rank_ic, cell_rank_ic_df, frequency, horizon)
        reconstructed_rank_ic_stats = formal._reconstruct_rank_ic_summary_stats(reconstructed_rank_ic)
        formal._validate_cell_metric_reconstruction(summary_row, reconstructed_metrics, reconstructed_rank_ic_stats, frequency, horizon)

    _validate_cross_frequency_target_identity_float32_safe(pred_df)

    return {
        "summary": summary_df,
        "predictions": pred_df,
        "rank_ic": rank_ic_df,
        "history": history_df,
        "router": router_df,
    }


def _require_finalization_outputs_absent() -> None:
    existing = [
        str(path)
        for path in [formal.FORMAL_COMPARISON_PATH, formal.FORMAL_MANIFEST_PATH, formal.FORMAL_FINAL_SUMMARY_PATH]
        if path.exists()
    ]
    if existing:
        raise RuntimeError("Finalization outputs already exist; refusing to overwrite: " + "; ".join(existing))


def _build_manifest_row(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    history_df: pd.DataFrame,
    router_df: pd.DataFrame,
    provenance_row: pd.Series,
    artifact_shas: dict[str, str],
    finalizer_script_sha256: str,
) -> dict:
    manifest_row = formal._build_final_manifest_row(
        summary_df, pred_df, rank_ic_df, history_df, router_df, provenance_row, artifact_shas
    )
    manifest_row["finalization_script"] = FINALIZER_SCRIPT_RELATIVE
    manifest_row["finalization_script_sha256"] = finalizer_script_sha256
    manifest_row["cross_frequency_y_true_validation"] = "exact_keys_float32_epsilon_values"
    manifest_row["cross_frequency_y_true_atol"] = FLOAT32_ATOL
    return manifest_row


def _build_final_summary_text(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    provenance_row: pd.Series,
    artifact_shas: dict[str, str],
    finalizer_script_sha256: str,
) -> str:
    base_text = formal._build_final_summary_text(summary_df, comparison_df, provenance_row, artifact_shas)
    extra_lines = [
        "",
        "Cross-frequency target identity was validated using exact ticker/date keys and",
        f"absolute tolerance equal to one float32 machine epsilon ({FLOAT32_ATOL}) to account",
        "solely for CSV floating-point round-trip representation. Raw target evidence was not",
        "modified.",
        "",
        f"finalization_script_sha256 = {finalizer_script_sha256}",
    ]
    return base_text + "\n".join(extra_lines) + "\n"


def _finalize_pathformer_main_v2() -> None:
    provenance_row = _require_immutable_provenance()

    raw = _validate_full_evidence_float32_safe()
    summary_df = raw["summary"]
    pred_df = raw["predictions"]
    rank_ic_df = raw["rank_ic"]
    history_df = raw["history"]
    router_df = raw["router"]

    _require_finalization_outputs_absent()

    baselines = formal._load_accepted_main_v2_baselines()
    comparison_df = formal._build_matched_main_v2_comparison(summary_df, baselines)
    formal._validate_comparison_contract(comparison_df)

    comparison_df.to_csv(formal.FORMAL_COMPARISON_PATH, index=False)
    if not formal.FORMAL_COMPARISON_PATH.exists():
        raise RuntimeError("Formal comparison artifact was not written successfully.")

    artifact_shas = {
        "summary": formal._compute_sha256(formal.FORMAL_SUMMARY_PATH),
        "predictions": formal._compute_sha256(formal.FORMAL_PRED_PATH),
        "rank_ic": formal._compute_sha256(formal.FORMAL_RANK_IC_PATH),
        "history": formal._compute_sha256(formal.FORMAL_HISTORY_PATH),
        "router": formal._compute_sha256(formal.FORMAL_ROUTER_PATH),
        "comparison": formal._compute_sha256(formal.FORMAL_COMPARISON_PATH),
    }
    finalizer_script_sha256 = formal._compute_sha256(ROOT / FINALIZER_SCRIPT_RELATIVE)

    manifest_row = _build_manifest_row(
        summary_df, pred_df, rank_ic_df, history_df, router_df, provenance_row, artifact_shas, finalizer_script_sha256
    )
    pd.DataFrame([manifest_row]).to_csv(formal.FORMAL_MANIFEST_PATH, index=False)

    final_summary_text = _build_final_summary_text(
        summary_df, comparison_df, provenance_row, artifact_shas, finalizer_script_sha256
    )
    formal.FORMAL_FINAL_SUMMARY_PATH.write_text(final_summary_text, encoding="utf-8")

    print("PATHFORMER MAIN V2 STANDALONE FINALIZATION PASSED")


def main() -> None:
    _finalize_pathformer_main_v2()


if __name__ == "__main__":
    main()
