"""Build REPORT_STEP12A_PART1.md (12a supplement: sequential vs parallel 12/12 runs). Formatting only."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "revision_work" / "outputs" / "input_sensitivity" / "step12a"
SEQ = BASE / "part1_static_s0_h5_t12" / "static_dw_h5_s0_t12_result.json"
PAR = BASE / "part1b_parallel"
JOBS = {"static_s0_h5": "static_dw_h5_s0_t12", "adaptive_s42_h10": "adaptive_dw_h10_s42_t12", "static_s1_h5": "static_dw_h5_s1_t12"}
REF_SEQ_ADAPTIVE_H10 = ROOT / "revision_work" / "outputs" / "repro_threads" / "runs" / "adaptive_dw_h10_s42_t12.json"


def tel_line(t):
    return (f"{t['event']} {t['utc'][11:19]} UTC: threads {t['torch_num_threads']}/{t['torch_num_interop_threads']}, "
            f"OMP_NUM_THREADS={t['environment']['OMP_NUM_THREADS']}, OMP_DYNAMIC={t['environment']['OMP_DYNAMIC']}, MKL_DYNAMIC={t['environment']['MKL_DYNAMIC']}, "
            f"load {', '.join(f'{x:.2f}' for x in t['load_average_1_5_15'])}")


def main():
    out = BASE / "REPORT_STEP12A_PART1.md"
    if out.exists():
        raise FileExistsError(out)
    blocking = []
    seq = json.loads(SEQ.read_text()) if SEQ.exists() else None
    if seq is None:
        blocking.append("1a sequential result missing.")
    lau = json.loads((PAR / "launcher.json").read_text()) if (PAR / "launcher.json").exists() else None
    if lau is None:
        blocking.append("1b launcher record missing (parallel runs not completed).")
    par = {}
    for job, stem in JOBS.items():
        p = PAR / job / f"{stem}_result.json"
        if p.exists():
            par[job] = json.loads(p.read_text())
        else:
            blocking.append(f"1b parallel run {job} produced no result (see {job}.log).")
    for job, r in par.items():
        if r.get("telemetry_thread_counts_changed"):
            blocking.append(f"{job}: logged thread counts changed during training.")
        if r.get("archived_checkpoint_unchanged") is False:
            blocking.append(f"{job}: archived checkpoint hash changed.")
    if seq and not seq.get("archived_checkpoint_unchanged", True):
        blocking.append("1a: archived checkpoint hash changed.")
    L = ["# REPORT STEP 12a — Part 1 (supplement): sequential vs parallel 12/12 retraining\n",
         f"Generated {datetime.now(timezone.utc).isoformat()} (UTC). This replaces Part 1 of Prompt 12a. Part 2 (causal joint bootstrap) is in `REPORT_STEP12A.md` and was not rerun. "
         "Existing files were read only; outputs under `revision_work/outputs/input_sensitivity/step12a/` were created exclusively. "
         "Code: `revision_work/scripts/input_sensitivity/retrain_static_s0_h5_t12.py` (1a), `retrain_cell_t12.py` + `parallel_launch_t12.py` (1b), this report builder. "
         "The trainers are imported unchanged; Experiment 4 checkpoint writes were redirected in-process to the output directory.\n",
         "## BLOCKING\n",
         ("\n".join(f"- **{b}**" for b in blocking) if blocking else "- Nothing blocked: 1a, 1b and 1c were all completed.") + "\n",
         "Notes:\n\n"
         "- Profile for every run: Step 3B 12/12 (CPU; torch intra-op 12, inter-op 12; OMP_NUM_THREADS unset; OMP_DYNAMIC=FALSE; MKL_DYNAMIC=FALSE; Python 3.12.1, torch 2.2.0, numpy 1.26.4). Each process sets this itself before importing torch.\n"
         "- Isolation: 1a ran alone, after a programmatic check that no other Python interpreter was running. 1b's three processes ran together and nothing else was launched. OS, IDE and agent stayed resident (as in Step 3B). "
         "The parallel launch used `caffeinate -i` to prevent idle sleep. **Keep the Mac on power and awake during such runs.**\n"
         "- A first 1a launch stopped before training because the isolation check matched its own parent shell's command line. The check was fixed to match executable names only. No training or write had happened (`part1_static_s0_h5_t12/BLOCKED_other_processes.json`).\n"
         "- Classification: EXACT if max |Δpred| < 1e-7, NEAR if < 1e-5, otherwise DIFFERENT (Step 3/3B convention). ΔMSE = new − archived, float32 metric arithmetic.\n"]
    # 1a
    L.append("## 1a. Sequential — ORIGINAL Static, seed 0, h = 5 (Experiment 4 `train_and_validate_cell`), alone\n")
    if seq:
        c = seq["comparison_vs_archived_experiment4"]
        L.append("| Item | Value |\n|---|---|\n"
                 f"| Classification vs archived Experiment 4 TEST predictions | **{c['classification']}** |\n"
                 f"| max \\|Δpred\\| | {c['max_abs_delta_pred']:.3e} |\n| ΔMSE | {c['test_mse_difference']:.3e} |\n"
                 f"| Epochs trained (new / archived) | {seq['epochs_trained']} / {seq['archived_epochs_trained']} |\n"
                 f"| Best epoch (new / archived) | {seq['best_epoch']} / {seq['archived_best_epoch']} |\n"
                 f"| Loss history exactly equal / max \\|Δ\\| | {seq['history_exact_equal']} / {seq['history_max_abs_delta']:.3e} |\n"
                 f"| Wall time, seconds | {seq['wall_clock_seconds']:.1f} |\n"
                 f"| Trainer runtime new / archived, seconds | {seq['trainer_runtime_seconds']:.1f} / {seq['archived_trainer_runtime_seconds']:.1f} |\n"
                 f"| Threads at start/end | {seq['telemetry_thread_counts']} (changed: {seq['telemetry_thread_counts_changed']}) |\n"
                 f"| Load average start → end | {seq['telemetry_load_average_1_5_15'][0]} → {seq['telemetry_load_average_1_5_15'][-1]} |\n"
                 f"| Archived checkpoint unchanged | {seq['archived_checkpoint_unchanged']} |\n")
    # 1b
    L.append("## 1b. Parallel — three 12/12 processes launched together\n")
    if lau:
        L.append(f"Launched {lau['launched_utc']}, all finished {lau['finished_utc']}. **Parallel wall time (launch → last exit): {lau['parallel_wall_seconds']:.1f} s.**\n")
        L.append("| Run | Trainer | PID | Return code | Classification | max \\|Δpred\\| | ΔMSE | Epochs new / archived | Best epoch new / archived | Process wall s |\n|---|---|---|---|---|---|---|---|---|---|")
        for j in lau["jobs"]:
            r = par.get(j["name"])
            if r:
                c = r["comparison"]
                L.append(f"| {j['name']} | {r['trainer']} | {j['pid']} | {j['returncode']} | **{c['classification']}** | {c['max_abs_delta_pred']:.3e} | {c['test_mse_difference']:.3e} | "
                         f"{r['epochs_trained']} / {r['archived_epochs_trained']} | {r['best_epoch']} / {r['archived_best_epoch']} | {j['process_wall_seconds']:.1f} |")
            else:
                L.append(f"| {j['name']} | — | {j['pid']} | {j['returncode']} | NOT DONE | — | — | — | — | {j['process_wall_seconds']:.1f} |")
        L.append("\nArchived references: Static seeds 0/1 → `dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv`; Adaptive seed 42 h10 → `dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv` (Step 3B's EXACT reference).\n")
        L.append("Thread / environment / load telemetry at start and end of each parallel process:\n")
        for job, r in par.items():
            ts = [t for t in r["telemetry"] if t["event"] in ("start", "end")]
            L.append(f"- **{job}**: " + "; ".join(tel_line(t) for t in ts) + f" (thread counts changed during run: {r['telemetry_thread_counts_changed']}).")
        L.append("")
    # 1c
    L.append("## 1c. Sequential vs parallel throughput\n")
    if seq and lau and len(par) == 3:
        seq_times = {"static_s0_h5": (seq["wall_clock_seconds"], "1a (this step)")}
        if REF_SEQ_ADAPTIVE_H10.exists():
            ref = json.loads(REF_SEQ_ADAPTIVE_H10.read_text())
            seq_times["adaptive_s42_h10"] = (float(ref["wall_clock_seconds"]), "Step 3B sequential 12/12 run")
        rows = []
        for j in lau["jobs"]:
            s = seq_times.get(j["name"])
            rows.append((j["name"], s, j["process_wall_seconds"]))
        per_epoch_seq = seq["wall_clock_seconds"] / seq["epochs_trained"]
        L.append("| Run | Epochs | Sequential wall s (source) | Parallel process wall s | Slow-down (parallel ÷ sequential) |\n|---|---|---|---|---|")
        est = {}
        for name, s, pw in rows:
            ep = par[name]["epochs_trained"]
            if s:
                est[name] = s[0]
                src = f"{s[0]:.1f} ({s[1]})"
            else:
                est[name] = per_epoch_seq * ep  # same architecture and horizon as 1a; scaled by its own epoch count
                src = f"{est[name]:.1f} (estimated: 1a seconds per epoch {per_epoch_seq:.1f} × {ep} epochs; not measured alone)"
            L.append(f"| {name} | {ep} | {src} | {pw:.1f} | {pw / est[name]:.2f}× |")
        seq_total = sum(est.values())
        ratio = seq_total / lau["parallel_wall_seconds"]
        all_exact = all(par[j]["comparison"]["classification"] == "EXACT" for j in par)
        short_end = sorted(j["process_wall_seconds"] for j in lau["jobs"])
        L.append(f"\nSequential total for the three runs: {seq_total:.1f} s (Static s1 epoch-scaled from 1a). Parallel wall time: {lau['parallel_wall_seconds']:.1f} s. "
                 f"**Implied throughput ratio (sequential ÷ parallel) = {ratio:.2f}×.** "
                 f"Every parallel run EXACT: **{all_exact}**. "
                 f"The jobs were unbalanced: two finished after about {short_end[0]:.0f}–{short_end[1]:.0f} s, and Static s0 (22 epochs) then ran alone for the last {short_end[2] - short_end[1]:.0f} s. "
                 "The parallel wall time is set by the longest job, so this ratio understates what equally sized jobs would gain.\n")
        L.append("**Finding.** " + (f"Running three 12/12 processes at once gives {ratio:.2f}× the throughput of running them one after another"
                                    + (" and leaves every result bit-for-bit identical to its archived reference (EXACT). Parallel 12/12 execution is therefore safe for reproducibility on this host and"
                                       if all_exact else ", but not every run stays EXACT, so parallel execution changes numerics on this host and")
                                    + (" worthwhile for throughput." if ratio > 1.05 else " gives no material throughput gain.")) + "\n")
    out.write_text("\n".join(L))
    print(out)


if __name__ == "__main__":
    main()
