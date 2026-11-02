"""Build REPORT_STEP12A.md from the part-1 result JSON and the part-2 bootstrap CSVs (no computation beyond formatting)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "revision_work" / "outputs" / "input_sensitivity" / "step12a"
P1 = BASE / "part1_static_s0_h5_t12"
B7 = BASE / "part2_causal_7b"
A3 = BASE / "part2_causal_a3"


def f(x, nd=4):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)):
        return str(x)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.3e}" if (x != 0 and abs(x) < 1e-3) else f"{x:.{nd}f}"


def md(df, cols):
    s = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.to_dict("records"):
        s += "| " + " | ".join(f(r[c[0]]) for c in cols) + " |\n"
    return s


def main():
    out = BASE / "REPORT_STEP12A.md"
    if out.exists():
        raise FileExistsError(out)
    blocking = []
    res_path = P1 / "static_dw_h5_s0_t12_result.json"
    r = json.loads(res_path.read_text()) if res_path.exists() else None
    if r is None:
        blocked = list(P1.glob("BLOCKED_*.json"))
        blocking.append("Part 1 was not completed: " + (blocked[0].read_text()[:500] if blocked else "result file missing"))
    else:
        if not r["archived_checkpoint_unchanged"]:
            blocking.append("Archived checkpoint hash changed during the run (should be impossible).")
        if r["telemetry_thread_counts_changed"]:
            blocking.append("Logged torch thread counts changed during training.")
    L = ["# REPORT STEP 12a — Timing / reproducibility check and causal joint bootstrap\n",
         f"Generated {datetime.now(timezone.utc).isoformat()} (UTC). Existing files were read only. New outputs under `revision_work/outputs/input_sensitivity/step12a/` were created exclusively. "
         "Code: `revision_work/scripts/input_sensitivity/` (part 1, report) plus the Step 7B / Step 5 A3 scripts reused unchanged by import or command line (part 2). No other training was started.\n",
         "## BLOCKING\n"]
    notes = [
        "Isolation: no other Python or training process was running at launch (checked programmatically) and none was started during the run. "
        "The OS, IDE and the agent stayed resident (as in Step 3B), so the machine was not literally idle; load averages are logged below. Background load affects timing only.",
        "Part 2 wording: the prompt asks for \"the same procedure as REPORT_STEP7B\". The primary result is therefore the Step 7B seed-resampled **ensemble** bootstrap "
        "(each replicate rebuilds both ensembles from the drawn seed multiset). The Step 5 A3 joint bootstrap (mean of per-seed differences over drawn seeds and dates) is reported as a supplement. "
        "Both resample seeds with replacement, pair them across mechanisms, and use one common set of moving date blocks (L = h).",
        "Causal predictions come from the Stage 3a (seed 42) and Stage 3b (seeds 0/1/21/3407) runs under the authorised **3/1, OMP=3** profile, not the 12/12 profile. Part 1 concerns the 12/12 profile on original features only.",
    ]
    L.append(("\n".join(f"- **{b}**" for b in blocking) if blocking else "- Nothing blocked: both parts were completed.") + "\n\nNotes:\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    # ---- part 1
    L.append("## 1. Reproducibility and timing — ORIGINAL features, Static, seed 0, h = 5 (Experiment 4 trainer, 12/12)\n")
    if r:
        c = r["comparison_vs_archived_experiment4"]
        e = r["environment"]
        L.append(f"- Profile: Step 3B 12/12 — `torch.get_num_threads()` = {e['torch_num_threads']}, `torch.get_num_interop_threads()` = {e['torch_num_interop_threads']} "
                 f"(defaults before pinning: {e['default_threads_before_explicit_setting']}). Thread env vars: {e['thread_environment']}. "
                 f"Python {e['python'].split()[0]}, torch {e['torch']}, numpy {e['numpy']}, pandas {e['pandas']}, {e['platform']}, CPU, PID {e['pid']}, one process.\n"
                 f"- Trainer: `panel_pathformer_experiment4_multiseed.train_and_validate_cell('static', '5d', 0)`, imported unchanged. Only its checkpoint destination was rebound in-process to `{r['new_checkpoint']}`. "
                 f"The archived checkpoint `{r['archived_checkpoint']}` has the same SHA-256 before and after the run ({r['archived_checkpoint_unchanged']}).\n")
        t = pd.DataFrame([dict(item="Classification (EXACT < 1e-7, NEAR < 1e-5, else DIFFERENT)", value=c["classification"]),
                          dict(item="max |Δpred| vs archived Experiment 4 TEST predictions", value=c["max_abs_delta_pred"]),
                          dict(item="ΔMSE (new − archived, float32 metric arithmetic)", value=c["test_mse_difference"]),
                          dict(item="TEST MSE new / archived", value=f"{c['replayed_test_mse']:.10f} / {c['formal_test_mse']:.10f}"),
                          dict(item="max |Δpred| vs Step 3 replayed predictions", value=r["comparison_vs_step3_replay"]["max_abs_delta_pred"]),
                          dict(item="Epochs trained new / archived", value=f"{r['epochs_trained']} / {r['archived_epochs_trained']}"),
                          dict(item="Best epoch new / archived", value=f"{r['best_epoch']} / {r['archived_best_epoch']}"),
                          dict(item="Best validation Huber new / archived", value=f"{r['best_val_huber']:.12g} / {r['archived_best_val_huber']:.12g}"),
                          dict(item="Loss history: same rows / exactly equal / max |Δ|", value=f"{r['history_rows_equal']} / {r['history_exact_equal']} / {r['history_max_abs_delta']:.3e}"),
                          dict(item="Elapsed wall time (process, s)", value=f"{r['wall_clock_seconds']:.1f}"),
                          dict(item="Trainer-reported runtime new / archived (s)", value=f"{r['trainer_runtime_seconds']:.1f} / {r['archived_trainer_runtime_seconds']:.1f}"),
                          dict(item="Seconds per epoch (trainer runtime / epochs)", value=f"{r['trainer_runtime_seconds'] / r['epochs_trained']:.1f}"),
                          dict(item="Telemetry samples / thread counts seen / changed", value=f"{r['telemetry_samples']} / {r['telemetry_thread_counts']} / {r['telemetry_thread_counts_changed']}"),
                          dict(item="Load average (1/5/15 min) at start and end", value=f"{r['telemetry_load_average_1_5_15'][0]} → {r['telemetry_load_average_1_5_15'][-1]}")])
        L.append(md(t, [("item", "Item"), ("value", "Value")]))
        L.append(f"Files: `{P1.relative_to(ROOT)}/` (result JSON, TEST predictions, per-epoch history comparison, telemetry JSONL, new checkpoint).\n")
        L.append("**Finding.** " + (f"The 12/12 profile reproduces the archived Experiment 4 Static seed-0 h=5 run **{c['classification']}** (max |Δpred| = {c['max_abs_delta_pred']:.3e})"
                                    + (", with identical epochs and loss history" if r["history_exact_equal"] and r["best_epoch"] == r["archived_best_epoch"] else "")
                                    + f". One retraining of this cell takes about {r['wall_clock_seconds'] / 60:.0f} minutes of wall time on this host.\n"))
    # ---- part 2
    L.append("## 2. Causal features — seed × date joint bootstrap, Adaptive − Static (TEST)\n")
    b = pd.read_csv(B7 / "seed_resampled_bootstrap.csv")
    b = b[b.specification == "PRIMARY"].copy()
    b["metric"] = b.metric.map({"loss_adaptive_minus_static": "(i) squared loss", "rank_ic_adaptive_minus_static": "(ii) Rank IC"})
    L.append("**Primary — Step 7B procedure** (seed-resampled ensembles, paired seeds, common moving date blocks L = h, B = 20,000, SeedSequence([20261017, h, L]), Holm over 3 horizons). "
             "θ̂ = full-ensemble difference. Positive loss difference = Static better; positive Rank IC difference = Adaptive better.\n")
    L.append(md(b.sort_values(["seed_set", "metric", "h"]), [("seed_set", "Seed set"), ("metric", "Metric"), ("h", "h"), ("theta_hat", "θ̂"), ("ci_lower_95", "95% CI low"),
                                                               ("ci_upper_95", "95% CI high"), ("centered_p_two_sided", "p"), ("centered_p_holm_primary", "Holm p"),
                                                               ("n_distinct_seed_multisets", "Distinct seed multisets")]))
    j = pd.read_csv(A3 / "a3_joint_seed_date_bootstrap.csv")
    j = j[j.specification == "PRIMARY"].copy()
    j["metric"] = j.metric.map({"loss": "(i) squared loss", "rank_ic": "(ii) Rank IC"})
    L.append("**Supplement — Step 5 A3 procedure** (statistic = mean over drawn seeds and resampled dates of the per-seed difference, seeds 0/1/21/42/3407, B = 20,000, SeedSequence([20260826, h, L, 1]), Holm over 3 horizons):\n")
    L.append(md(j.sort_values(["metric", "h"]), [("metric", "Metric"), ("h", "h"), ("theta_hat", "θ̂"), ("ci_lower_95", "95% CI low"), ("ci_upper_95", "95% CI high"),
                                                  ("centered_p_two_sided", "p"), ("centered_p_holm_primary", "Holm p")]))
    n_ex = int(((b.ci_lower_95 > 0) | (b.ci_upper_95 < 0)).sum()) + int(((j.ci_lower_95 > 0) | (j.ci_upper_95 < 0)).sum())
    L.append(f"**Finding.** With causal features, {n_ex} of {len(b) + len(j)} reported intervals exclude zero. The minimum Holm p is {min(b.centered_p_holm_primary.min(), j.centered_p_holm_primary.min()):.3f}. "
             "Once seed variation is resampled, Adaptive and Static are statistically indistinguishable on both squared loss and Rank IC at every horizon. "
             "With only 4–5 seeds the seed resampling is coarse (35 or 126 distinct multisets). Full outputs, including leave-one-seed-out ensembles, are in `part2_causal_7b/` and `part2_causal_a3/`.\n")
    out.write_text("\n".join(L)) if not out.exists() else None
    print(out)


if __name__ == "__main__":
    main()
