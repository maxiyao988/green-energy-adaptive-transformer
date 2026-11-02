"""Step 12a part 1 - one training run: ORIGINAL-feature Static, seed 0, h=5, Experiment 4 trainer, 12/12 profile.

Numerical profile = the Step 3B "12/12" profile that reproduced MAIN V2 Adaptive seed 42 exactly
(revision_work/outputs/repro_threads/REPORT_STEP3B.md): imported via repro_threads_common, which pops
OMP_NUM_THREADS and sets OMP_DYNAMIC=FALSE / MKL_DYNAMIC=FALSE before Torch is imported, and configure() pins
torch intra-op 12 / inter-op 12; CPU; one training process. Telemetry reuses repro_threads_common.Telemetry
(start, every >= 5 minutes from a main-thread forward pre-hook, end).

Read-only guarantee: Experiment 4's train_and_validate_cell writes only its checkpoint (the CSV evidence is
written by a separate function that is not called). In this process only, exp4._checkpoint_path is rebound to
revision_work/outputs/input_sensitivity/... and exp4._assert_experiment4_path additionally admits that
directory; the archived checkpoint's SHA-256 is verified unchanged before and after. No source file is edited.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "revision_work" / "scripts" / "causal"))
import repro_threads_common as rtc  # noqa: E402  (sets the 12/12 environment before torch is imported)
import step3_common as prior  # noqa: E402

OUTDIR = ROOT / "revision_work" / "outputs" / "input_sensitivity" / "step12a" / "part1_static_s0_h5_t12"
prior.OUT = OUTDIR
rtc.OUT = OUTDIR
from repro_threads_common import torch, np, pd, formal, exp4, sha, utc, rel, write_json, write_csv, compare, formal_subset, COUNTS  # noqa: E402

E4 = ROOT / "dataset" / "audit" / "experiment4_multiseed"
ARCHIVED_CKPT = E4 / "checkpoints" / "static_seed0_5d.pt"
STEM = "static_dw_h5_s0_t12"


def other_python_processes():
    """Other Python interpreters (by executable name, not command-line text), excluding this process and its parent."""
    import subprocess
    out = subprocess.run(["ps", "-axo", "pid,ppid,pcpu,comm"], capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    me, parent = os.getpid(), os.getppid()
    rows = []
    for line in out:
        f = line.strip().split(None, 3)
        if len(f) == 4 and int(f[0]) not in (me, parent) and "python" in Path(f[3]).name.lower():
            rows.append(dict(pid=int(f[0]), ppid=int(f[1]), cpu_pct=float(f[2]), executable=f[3][:300]))
    return rows


def main():
    import time
    environment = rtc.configure()
    others = other_python_processes()
    if others:
        write_json(OUTDIR / f"BLOCKED_other_processes_{os.getpid()}.json", others)
        raise SystemExit(f"BLOCKING: other Python processes are running: {others}")
    exp4._verify_frozen_dependencies()
    archived_before = sha(ARCHIVED_CKPT)
    ckpt_dir = OUTDIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    new_ckpt = ckpt_dir / f"static_seed0_5d_t12.pt"
    if new_ckpt.exists():
        raise FileExistsError(new_ckpt)
    # ---- process-local rebinding of the checkpoint destination only
    old_path_fn, old_assert, old_factory = exp4._checkpoint_path, exp4._assert_experiment4_path, exp4._make_experiment4_model
    exp4._checkpoint_path = lambda mechanism, horizon, seed: new_ckpt
    exp4._assert_experiment4_path = lambda p: Path(p).resolve() if Path(p).resolve().is_relative_to(OUTDIR.resolve()) else old_assert(p)
    tel = rtc.Telemetry(STEM)

    def factory(mechanism):
        m = old_factory(mechanism)
        m.register_forward_pre_hook(lambda module, inputs: tel.sample("progress"))  # observes only; no tensors touched
        return m
    exp4._make_experiment4_model = factory
    tel.sample("start", force=True)
    start_utc, t0 = utc(), time.perf_counter()
    try:
        result = exp4.train_and_validate_cell("static", "5d", 0)
    finally:
        exp4._checkpoint_path, exp4._assert_experiment4_path, exp4._make_experiment4_model = old_path_fn, old_assert, old_factory
    wall = time.perf_counter() - t0
    end_utc = utc()
    tel.sample("end", force=True)
    tel.close()
    archived_after = sha(ARCHIVED_CKPT)
    # ---- TEST inference from the restored best state (formal helper), comparison with archived Experiment 4
    ds = formal.load_frozen_panel()
    formal._formal_check_dataset_contract(ds)
    _, _, test_loader = formal._build_training_loaders("daily_weekly", 5, ds, batch_size=formal.BATCH_SIZE)
    model = old_factory("static")
    model.load_state_dict(torch.load(new_ckpt, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    pred, dw, ww = exp4._batched_inference(model, test_loader, COUNTS["test"], torch.device("cpu"))
    mask = ds.split == "test"
    new = pd.DataFrame(dict(ticker=ds.ticker[mask], anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime("%Y-%m-%d"),
                            split="test", y_true=ds.y[5][mask], y_pred=pred))
    old = formal_subset(E4 / "experiment4_test_predictions.csv", "static", 5, 0)
    cmp_e4 = compare(new, old)
    cmp_replay = compare(new, pd.read_parquet(ROOT / "revision_work/outputs/preds/static_dw_h5_s0_test.parquet"))
    write_csv(OUTDIR / f"{STEM}_test_predictions.csv", new)
    # ---- history / epochs
    H = pd.DataFrame(result["history"])
    oldH = pd.read_csv(E4 / "experiment4_training_history.csv", float_precision="round_trip")
    oldH = oldH[(oldH.mechanism == "static") & (oldH.horizon == "5d") & (oldH.seed == 0)].sort_values("epoch")
    mH = H[["epoch", "train_loss", "val_loss"]].merge(oldH[["epoch", "train_loss", "val_loss"]], on="epoch", how="outer", suffixes=("_new", "_archived"))
    mH["train_abs_delta"] = (mH.train_loss_new - mH.train_loss_archived).abs()
    mH["val_abs_delta"] = (mH.val_loss_new - mH.val_loss_archived).abs()
    write_csv(OUTDIR / f"{STEM}_history_comparison.csv", mH)
    oldS = pd.read_csv(E4 / "experiment4_training_summary.csv")
    oldS = oldS[(oldS.mechanism == "static") & (oldS.horizon == "5d") & (oldS.seed == 0)].iloc[0]
    s = result["summary"]
    tele = [__import__("json").loads(l) for l in (OUTDIR / f"telemetry/{STEM}.jsonl").read_text().splitlines()]
    out = dict(mechanism="static", horizon=5, seed=0, input_config="daily_weekly", features="original (frozen V2)",
               trainer="panel_pathformer_experiment4_multiseed.train_and_validate_cell", profile="Step 3B 12/12",
               started_utc=start_utc, completed_utc=end_utc, wall_clock_seconds=wall, trainer_runtime_seconds=float(s["runtime_seconds"]),
               archived_trainer_runtime_seconds=float(oldS.runtime_seconds),
               epochs_trained=int(s["epochs_trained"]), archived_epochs_trained=int(oldS.epochs_trained),
               best_epoch=int(s["best_epoch"]), archived_best_epoch=int(oldS.best_epoch),
               best_val_huber=float(s["best_val_huber"]), archived_best_val_huber=float(oldS.best_val_huber),
               history_rows_equal=bool(len(H) == len(oldH)), history_exact_equal=bool(((mH.train_abs_delta == 0) & (mH.val_abs_delta == 0)).all()),
               history_max_abs_delta=float(np.nanmax(np.r_[mH.train_abs_delta.to_numpy(), mH.val_abs_delta.to_numpy()])),
               comparison_vs_archived_experiment4=cmp_e4, comparison_vs_step3_replay=cmp_replay,
               new_checkpoint=rel(new_ckpt), new_checkpoint_sha256=sha(new_ckpt), archived_checkpoint=rel(ARCHIVED_CKPT),
               archived_checkpoint_sha256_before=archived_before, archived_checkpoint_sha256_after=archived_after,
               archived_checkpoint_unchanged=archived_before == archived_after,
               environment=environment, telemetry_samples=len(tele),
               telemetry_thread_counts=sorted({(r["torch_num_threads"], r["torch_num_interop_threads"]) for r in tele}),
               telemetry_thread_counts_changed=bool(tel.changed),
               telemetry_load_average_1_5_15=[r["load_average_1_5_15"] for r in tele],
               other_python_processes_at_start=others)
    write_json(OUTDIR / f"{STEM}_result.json", out)
    print("RESULT", cmp_e4["classification"], cmp_e4["max_abs_delta_pred"], wall, flush=True)


if __name__ == "__main__":
    main()
