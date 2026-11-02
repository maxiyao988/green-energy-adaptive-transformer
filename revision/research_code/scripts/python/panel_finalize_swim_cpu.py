from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.python.panel_baseline_swim as formal
from scripts.python.panel_baseline_swim_cpu import (
    DATASET_MANIFEST_SHA256,
    EXPECTED_SWIM_PARAM_COUNTS,
    FORMAL_SWIM_RUN_FILES,
    MAIN_V2_SWIM_DIR,
    compute_sha256,
    resolve_output_dir,
    validate_cpu_main_v2_swim_evidence,
    validate_cpu_training_provenance_row,
)

TRAINING_SCRIPT_RELATIVE = "scripts/python/panel_baseline_swim_cpu.py"
FINALIZER_SCRIPT_RELATIVE = "scripts/python/panel_finalize_swim_cpu.py"
FROZEN_SCRIPT_RELATIVE = "scripts/python/panel_baseline_swim.py"


def _read_provenance_row(provenance_path: Path) -> pd.Series:
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing CPU training provenance: {provenance_path}")
    provenance_df = pd.read_csv(provenance_path)
    if provenance_df.empty:
        raise ValueError(f"CPU training provenance is empty: {provenance_path}")
    if len(provenance_df) != 1:
        raise ValueError(f"Expected exactly one training provenance row in {provenance_path}, found {len(provenance_df)}.")
    row = provenance_df.iloc[0]
    validate_cpu_training_provenance_row(row)
    return row


def _compute_runtime_sha(path: Path) -> str:
    return compute_sha256(path)


def _require_training_script_lock(provenance_row: pd.Series, cpu_script_path: Path) -> str:
    stored_sha = str(provenance_row["training_script_sha256"]).strip()
    current_sha = _compute_runtime_sha(cpu_script_path)
    if stored_sha != current_sha:
        raise RuntimeError(
            "Training provenance SHA mismatch: "
            f"stored={stored_sha}, current={current_sha}, script={cpu_script_path}"
        )
    return stored_sha


def _load_main_v2_baselines() -> dict[str, pd.DataFrame]:
    naive_path = ROOT / "dataset" / "audit" / "main_v2" / "naive" / "panel_naive_baseline_summary.csv"
    ridge_path = ROOT / "dataset" / "audit" / "main_v2" / "ridge" / "main_v2_ridge_comparison.csv"
    lstm_path = ROOT / "dataset" / "audit" / "main_v2" / "lstm" / "panel_lstm_summary_metrics.csv"
    vanilla_path = ROOT / "dataset" / "audit" / "main_v2" / "vanilla" / "panel_vanilla_summary_metrics.csv"

    missing = []
    for name, path in {
        "naive": naive_path,
        "ridge": ridge_path,
        "lstm": lstm_path,
        "vanilla": vanilla_path,
    }.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError("Missing accepted MAIN V2 baseline artifacts: " + "; ".join(missing))

    naive_df = formal.canonicalize_main_v2_naive(pd.read_csv(naive_path))
    ridge_df = formal.canonicalize_main_v2_ridge(pd.read_csv(ridge_path))
    lstm_df = formal.canonicalize_main_v2_lstm_or_vanilla(
        pd.read_csv(lstm_path), label="lstm", model_value="lstm"
    )
    vanilla_df = formal.canonicalize_main_v2_lstm_or_vanilla(
        pd.read_csv(vanilla_path), label="vanilla", model_value=None
    )

    return {"naive": naive_df, "ridge": ridge_df, "lstm": lstm_df, "vanilla": vanilla_df}


def _validate_summary_cell_contract(summary_df: pd.DataFrame) -> None:
    if len(summary_df) != 9:
        raise ValueError(f"Expected exactly 9 SWiM summary rows, found {len(summary_df)}")
    cells = set(zip(summary_df["frequency_setting"].astype(str), summary_df["horizon"].astype(str)))
    expected = {
        (freq, horizon)
        for freq in ["daily_only", "weekly_only", "daily_weekly"]
        for horizon in ["5d", "10d", "20d"]
    }
    if cells != expected:
        raise ValueError(f"Summary cell mismatch: expected {sorted(expected)}, got {sorted(cells)}")


def _build_final_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    _validate_summary_cell_contract(summary_df)
    baseline_frames = _load_main_v2_baselines()
    out_rows = []
    for _, row in summary_df.iterrows():
        freq = str(row["frequency_setting"]).strip()
        horizon = str(row["horizon"]).strip()
        horizon_norm = horizon.replace("d", "")

        joined = {
            "model": str(row["model"]).strip(),
            "frequency_setting": freq,
            "horizon": horizon,
            "seed": int(row["seed"]),
            "n_trainable_parameters": int(row["n_trainable_parameters"]),
            "test_mse": float(row["test_mse"]),
            "test_mae": float(row["test_mae"]),
            "test_corr": float(row["test_corr"]),
            "mean_rank_ic": float(row["mean_rank_ic"]),
            "direction_accuracy": float(row["direction_accuracy"]),
            "pred_std_true_std_ratio": float(row["pred_std_true_std_ratio"]),
        }

        naive_df = baseline_frames["naive"]
        naive_matched = naive_df[naive_df["horizon"] == horizon_norm].copy()
        if naive_matched.empty:
            raise ValueError(f"Naive baseline missing horizon {horizon}.")
        naive_matched["model"] = naive_matched["model"].astype(str).str.lower()
        zero_row = naive_matched[naive_matched["model"] == "zero"]
        train_mean_row = naive_matched[naive_matched["model"] == "train_mean"]
        if zero_row.empty or train_mean_row.empty:
            raise ValueError(f"Naive baseline missing zero/train_mean rows for {horizon}.")
        joined["zero_mse"] = float(zero_row["pooled_mse"].iloc[0])
        joined["train_mean_mse"] = float(train_mean_row["pooled_mse"].iloc[0])
        joined["best_naive_mse"] = float(np.min([joined["zero_mse"], joined["train_mean_mse"]]))

        ridge_df = baseline_frames["ridge"]
        ridge_matched = ridge_df[(ridge_df["frequency_setting"] == freq) & (ridge_df["horizon"] == horizon_norm)]
        if ridge_matched.empty:
            raise ValueError(f"Ridge baseline missing row for {freq}/{horizon}.")
        joined["ridge_mse"] = float(ridge_matched["pooled_mse"].iloc[0])
        joined["ridge_mae"] = float(ridge_matched["pooled_mae"].iloc[0])
        joined["ridge_corr"] = float(ridge_matched["pooled_corr"].iloc[0])
        joined["ridge_mean_rank_ic"] = float(ridge_matched["mean_rank_ic"].iloc[0])

        lstm_df = baseline_frames["lstm"]
        lstm_matched = lstm_df[(lstm_df["frequency_setting"] == freq) & (lstm_df["horizon"] == horizon_norm)]
        if lstm_matched.empty:
            raise ValueError(f"LSTM baseline missing row for {freq}/{horizon}.")
        joined["lstm_mse"] = float(lstm_matched["test_mse"].iloc[0])
        joined["lstm_mae"] = float(lstm_matched["test_mae"].iloc[0])
        joined["lstm_corr"] = float(lstm_matched["test_corr"].iloc[0])
        joined["lstm_mean_rank_ic"] = float(lstm_matched["mean_rank_ic"].iloc[0])

        vanilla_df = baseline_frames["vanilla"]
        vanilla_matched = vanilla_df[(vanilla_df["frequency_setting"] == freq) & (vanilla_df["horizon"] == horizon_norm)]
        if vanilla_matched.empty:
            raise ValueError(f"Vanilla baseline missing row for {freq}/{horizon}.")
        joined["vanilla_mse"] = float(vanilla_matched["test_mse"].iloc[0])
        joined["vanilla_mae"] = float(vanilla_matched["test_mae"].iloc[0])
        joined["vanilla_corr"] = float(vanilla_matched["test_corr"].iloc[0])
        joined["vanilla_mean_rank_ic"] = float(vanilla_matched["mean_rank_ic"].iloc[0])

        out_rows.append(joined)

    comparison_df = pd.DataFrame(out_rows)
    if len(comparison_df) != 9:
        raise ValueError(f"Expected exactly 9 comparison rows, found {len(comparison_df)}")
    if comparison_df.duplicated(subset=["frequency_setting", "horizon"]).any():
        raise ValueError("Comparison has duplicate frequency_setting/horizon rows.")
    return comparison_df


def _build_final_summary_text(summary_df: pd.DataFrame, comparison_df: pd.DataFrame, training_script_sha256: str, finalizer_sha256: str, parent_frozen_sha256: str) -> str:
    lines: list[str] = []
    lines.append("MAIN V2")
    lines.append("POST-TEMPORAL-FIX")
    lines.append("Dataset Contract V2")
    lines.append("SWiM-style Transformer")
    lines.append("seed 42")
    lines.append("execution backend CPU")
    lines.append("training CPU script SHA = " + training_script_sha256)
    lines.append("finalization script SHA = " + finalizer_sha256)
    lines.append("parent frozen SWiM script SHA = " + parent_frozen_sha256)
    lines.append("")
    lines.append("Parameter counts:")
    for key, value in EXPECTED_SWIM_PARAM_COUNTS.items():
        lines.append(f"  - {key} = {value}")
    lines.append("")
    lines.append("SWiM 9-cell summary:")
    for _, row in summary_df.iterrows():
        lines.append(
            "  - "
            f"{row['frequency_setting']} / {row['horizon']}: "
            f"MSE={float(row['test_mse'])}, "
            f"MeanRankIC={float(row['mean_rank_ic'])}, "
            f"PredStd/TrueStd={float(row['pred_std_true_std_ratio'])}"
        )
    lines.append("")
    lines.append("Matched baseline comparison:")
    for _, row in comparison_df.iterrows():
        lines.append(
            "  - "
            f"{row['frequency_setting']} / {row['horizon']}: "
            f"SWiM_MSE={float(row['test_mse'])}, "
            f"NaiveBest={float(row['best_naive_mse'])}, "
            f"Ridge_MSE={float(row['ridge_mse'])}, "
            f"LSTM_MSE={float(row['lstm_mse'])}, "
            f"Vanilla_MSE={float(row['vanilla_mse'])}"
        )
    lines.append("")
    lines.append(
        "This is descriptive seed-42 benchmark evidence for the fixed CPU formal pipeline; "
        "it is not significance testing or multi-seed robustness evidence."
    )
    return "\n".join(lines) + "\n"


def _write_finalization_artifacts(output_dir: Path, finalizer_script_path: Path) -> None:
    output_dir = resolve_output_dir(output_dir)
    provenance_path = output_dir / FORMAL_SWIM_RUN_FILES["training_provenance"]
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing training provenance for finalization: {provenance_path}")

    provenance_row = _read_provenance_row(provenance_path)
    training_script_sha256 = _require_training_script_lock(
        provenance_row, ROOT / TRAINING_SCRIPT_RELATIVE
    )
    parent_frozen_sha256 = str(provenance_row["source_frozen_script_sha256"]).strip()

    validate_cpu_main_v2_swim_evidence(output_dir)
    summary_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["summary"])
    pred_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["predictions"])
    rank_ic_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"])
    history_df = pd.read_csv(output_dir / FORMAL_SWIM_RUN_FILES["history"])

    comparison_df = _build_final_comparison(summary_df)

    comparison_path = output_dir / FORMAL_SWIM_RUN_FILES["comparison"]
    comparison_df.to_csv(comparison_path, index=False)

    finalizer_sha256 = _compute_runtime_sha(finalizer_script_path)
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
        "training_script": TRAINING_SCRIPT_RELATIVE,
        "training_script_sha256": str(training_script_sha256),
        "finalization_script": FINALIZER_SCRIPT_RELATIVE,
        "finalization_script_sha256": str(finalizer_sha256),
        "output_dir": str(output_dir.resolve()),
        "source_frozen_script": FROZEN_SCRIPT_RELATIVE,
        "source_frozen_script_sha256": str(parent_frozen_sha256),
        "execution_backend": "CPU",
        "summary_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["summary"]),
        "predictions_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["predictions"]),
        "rank_ic_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["rank_ic"]),
        "training_history_artifact_sha256": compute_sha256(output_dir / FORMAL_SWIM_RUN_FILES["history"]),
        "comparison_artifact_sha256": compute_sha256(comparison_path),
        "historical_results_used_for_training": False,
        "historical_results_used_for_model_selection": False,
        "historical_results_used_for_formal_comparison": False,
        "d_model": formal.D_MODEL,
        "nhead": formal.NHEAD,
        "num_swim_blocks": formal.NUM_BLOCKS,
        "dim_feedforward": formal.DIM_FEEDFORWARD,
        "dropout": formal.DROPOUT,
        "activation": "relu",
        "batch_size": formal.BATCH_SIZE,
        "learning_rate": formal.LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": formal.MAX_EPOCHS,
        "patience": formal.EARLY_STOPPING_PATIENCE,
        "max_norm": 1.0,
    }
    manifest_path = output_dir / FORMAL_SWIM_RUN_FILES["manifest"]
    pd.DataFrame([manifest]).to_csv(manifest_path, index=False)

    final_summary_path = output_dir / FORMAL_SWIM_RUN_FILES["final_summary"]
    final_summary_path.write_text(
        _build_final_summary_text(summary_df, comparison_df, training_script_sha256, finalizer_sha256, parent_frozen_sha256),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the existing SWiM MAIN V2 CPU evidence without retraining.")
    parser.add_argument("--output-dir", type=str, default=str(MAIN_V2_SWIM_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    finalizer_script_path = ROOT / FINALIZER_SCRIPT_RELATIVE
    _write_finalization_artifacts(output_dir, finalizer_script_path)
    print("SWIM MAIN V2 CPU FINALIZATION PASSED")


if __name__ == "__main__":
    main()
