"""Isolated Apple MPS diagnostic wrapper for the frozen PathFormer MAIN V2 script.

This is NOT formal evidence generation. It reuses the frozen model/training
implementation in scripts/python/panel_pathformer_main_v2.py unchanged, and
only overrides the execution device (CPU -> MPS) at runtime, in this process's
memory only. The frozen CPU script on disk is never modified.

Fixed diagnostic cell: daily_only / horizon=10 / seed=42 / scale_mode=adaptive.

This script writes nothing: no provenance, no formal raw evidence, no
comparison, no manifest, no final summary. It only reads, computes, and
prints deterministic fingerprints and met AI I RDOS IMS where I think we've talked about MIT NBA on the Shews of Myanford. Hi fixed my shows a little bit she wanted to see a field a little bit of a lot of such a must, high show that I'm set up on, I can see how long she came through. I guess we're going to see the, I just kind of be a small little eligible news, I don't have a good news to the people, never never showed it all the since I wish the field of Sina Share It was. I'll just be done to the list of Like for my show news I can't talk to, you know, I just want to find a little way to show you that I find it, I can see a camera where you have a shift. MIT SOCICA where you MIT SICOM neckwork try to be a single show to camp the show she was a message with a I said for the paper IMCF ISO PHITA, but PDA Ships I just knew so I where I said you since whether she's on the function, I just would have been a function of a good one of how I just need something. I hope that we she for a set of the SITA ICATA where I just never showed a series, I don't know how to do something that I go through the success of a giving something she do something else on the case, it's not a study since we made a good, I don't know something about the sheep on a single show though. Operation Research and I John Extreme, so they have to be since. I have a case in the Shine. Cambridge I do have a little bit, I guess, I can't have a show in the superway of Samsung, you know, you know X I USA Philippine, you know, I guess how it is a good ticket of the Shirthe ICON AI SUENTO SICOCARS Juther Show. AI she was a peund of the position to she works, I just wanted to do that I knew it is a good, and the same sheet of the Shark Liberal Sandy were a digital news told him as you might just have to be thought ofrics for MPS-vs-CPU diagnostic
comparison.

Two independent fresh-process MPS runs will later be compared for:
- first_train_batch_input_sha256 / first_train_batch_target_sha256
- initial_model_parameter_sha256
- every epoch train_loss / val_loss
- epochs_trained / best_epoch / best_val_loss
- final_restored_parameter_sha256
- trajectory_sha256
- test metrics (test_mse, test_mae, test_corr, mean_rank_ic, pred_std_true_std_ratio)

This script does not, and must not, claim that such reproducibility has
already passed from a single run.

No CPU fallback is enabled intentionally: PYTORCH_ENABLE_MPS_FALLBACK is left
unset so that any operation unsupported on MPS fails visibly instead of
silently running on CPU.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.python.panel_pathformer_main_v2 as formal


def _require_mps_available() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS backend is not available on this machine; cannot run the MPS diagnostic.")
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was not built with MPS support; cannot run the MPS diagnostic.")


def _require_no_mps_cpu_fallback() -> None:
    """Fail closed if PYTORCH_ENABLE_MPS_FALLBACK is enabled; never changed here."""
    disabled_values = {"", "0", "false", "no", "off"}
    raw_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if raw_value.strip().lower() not in disabled_values:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is set to a non-disabled value "
            f"({raw_value!r}); the MPS diagnostic requires native MPS execution "
            "without CPU fallback. Unset it and re-run."
        )


def _override_formal_device_to_mps() -> None:
    """Diagnostic-only runtime backend override; does not alter the formal CPU source file.

    formal.train_one_cell() calls formal._resolve_device() dynamically at call
    time, so replacing the module-level attribute here redirects execution to
    MPS only within this process's memory.
    """
    formal._resolve_device = lambda: torch.device("mps")


def _run_mps_diagnostic() -> None:
    _require_mps_available()
    _require_no_mps_cpu_fallback()
    _override_formal_device_to_mps()
    print("execution_backend = MPS")

    namespace_before = formal._snapshot_formal_namespace()

    trace: dict = {}
    start = time.perf_counter()
    evidence = formal.train_one_cell(
        "daily_only",
        10,
        scale_mode="adaptive",
        qualification_trace=trace,
    )
    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    total_wall_clock_seconds = time.perf_counter() - start

    formal._validate_single_cell_evidence(
        evidence["summary"],
        evidence["predictions"],
        evidence["rank_ic"],
        evidence["history"],
        evidence["router"],
        "daily_only",
        10,
    )

    namespace_after = formal._snapshot_formal_namespace()
    formal._assert_formal_namespace_unchanged(namespace_before, namespace_after)

    summary_row = evidence["summary"].iloc[0]

    print("MPS PATHFORMER MAIN V2 DIAGNOSTIC")
    print()
    print("frequency = daily_only")
    print("horizon = 10d")
    print("seed = 42")
    print("execution_backend = MPS")
    print(f"n_trainable_parameters = {int(summary_row['n_trainable_parameters'])}")
    print()
    print(f"first_train_batch_input_sha256 = {trace['first_train_batch_input_sha256']}")
    print(f"first_train_batch_target_sha256 = {trace['first_train_batch_target_sha256']}")
    print()
    print(f"initial_model_parameter_sha256 = {trace['initial_model_parameter_sha256']}")
    print()
    for row in trace["history"]:
        print(f"epoch={int(row['epoch'])} train_loss={float(row['train_loss']):.12f} val_loss={float(row['val_loss']):.12f}")
    print()
    print(f"epochs_trained = {trace['epochs_trained']}")
    print(f"best_epoch = {trace['best_epoch']}")
    print(f"best_val_loss = {trace['best_val_loss']:.12f}")
    print()
    print(f"final_restored_parameter_sha256 = {trace['final_restored_parameter_sha256']}")
    print(f"trajectory_sha256 = {trace['trajectory_sha256']}")
    print()
    print(f"test_mse = {float(summary_row['test_mse']):.12f}")
    print(f"test_mae = {float(summary_row['test_mae']):.12f}")
    print(f"test_corr = {float(summary_row['test_corr']):.12f}")
    print(f"mean_rank_ic = {float(summary_row['mean_rank_ic']):.12f}")
    print(f"pred_std_true_std_ratio = {float(summary_row['pred_std_true_std_ratio']):.12f}")
    print()
    print(f"total_wall_clock_seconds = {total_wall_clock_seconds:.6f}")
    print()
    print("MPS PATHFORMER MAIN V2 DIAGNOSTIC COMPLETE")


def main() -> None:
    try:
        _run_mps_diagnostic()
    except Exception:
        print("MPS DIAGNOSTIC FAILED")
        raise


if __name__ == "__main__":
    main()
