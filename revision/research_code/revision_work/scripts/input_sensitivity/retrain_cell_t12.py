"""Step 12a-supplement worker: retrain ONE original-feature cell under the Step 3B 12/12 profile and compare it
with its archived TEST predictions.

  --trainer exp4    panel_pathformer_experiment4_multiseed.train_and_validate_cell(mechanism, f"{h}d", seed)
                    (checkpoint destination rebound in-process to the output dir; archived checkpoint hash verified)
                    archived predictions: dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv
  --trainer main_v2 panel_pathformer_main_v2.train_one_cell('daily_weekly', h, 'adaptive') (seed 42 inside; no disk
                    writes); archived predictions: dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv
Profile: repro_threads_common (OMP_NUM_THREADS unset, OMP_DYNAMIC/MKL_DYNAMIC=FALSE before torch import) and
configure() -> torch intra-op 12 / inter-op 12, CPU. Telemetry: repro_threads_common.Telemetry (start, >= 5-minute
progress from a main-thread forward pre-hook, end).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "revision_work" / "scripts" / "causal"))
import repro_threads_common as rtc  # noqa: E402  (12/12 environment before torch import)
import step3_common as prior  # noqa: E402
from repro_threads_common import torch, np, pd, formal, exp4, sha, utc, rel, write_json, write_csv, compare, formal_subset, COUNTS  # noqa: E402

BASE = ROOT / "revision_work" / "outputs" / "input_sensitivity" / "step12a"
E4 = ROOT / "dataset" / "audit" / "experiment4_multiseed"
MAIN = ROOT / "dataset" / "audit" / "main_v2" / "pathformer"


def other_python_processes(allowed):
    out = subprocess.run(["ps", "-axo", "pid,ppid,pcpu,comm"], capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    skip = {os.getpid(), os.getppid(), *allowed}
    rows = []
    for line in out:
        f = line.strip().split(None, 3)
        if len(f) == 4 and int(f[0]) not in skip and "python" in Path(f[3]).name.lower():
            rows.append(dict(pid=int(f[0]), ppid=int(f[1]), cpu_pct=float(f[2]), executable=f[3][:300]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", choices=["exp4", "main_v2"], required=True)
    ap.add_argument("--mechanism", choices=["static", "adaptive"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--h", type=int, choices=[5, 10, 20], required=True)
    ap.add_argument("--out-subdir", required=True)
    ap.add_argument("--allowed-pids", default="", help="comma list of sibling worker PIDs (parallel mode)")
    a = ap.parse_args()
    if a.trainer == "main_v2" and (a.mechanism != "adaptive" or a.seed != 42):
        raise SystemExit("main_v2 trainer is only used for Adaptive seed 42")
    out = BASE / a.out_subdir
    prior.OUT = out
    rtc.OUT = out
    stem = f"{a.mechanism}_dw_h{a.h}_s{a.seed}_t12"
    environment = rtc.configure()
    allowed = {int(x) for x in a.allowed_pids.split(",") if x}
    others = [p for p in other_python_processes(allowed) if p["ppid"] != os.getppid()]  # siblings share the launcher as parent
    if others:
        write_json(out / f"BLOCKED_other_processes_{os.getpid()}.json", others)
        raise SystemExit(f"BLOCKING: other Python processes are running: {others}")
    tel = rtc.Telemetry(stem)
    hooks = []

    def observed(factory):
        def make(*args, **kwargs):
            m = factory(*args, **kwargs)
            hooks.append(m.register_forward_pre_hook(lambda module, inputs: tel.sample("progress")))  # observation only
            return m
        return make
    tel.sample("start", force=True)
    start_utc, t0 = utc(), time.perf_counter()
    archived_ckpt = None
    if a.trainer == "exp4":
        exp4._verify_frozen_dependencies()
        archived_ckpt = E4 / "checkpoints" / f"{a.mechanism}_seed{a.seed}_{a.h}d.pt"
        before = sha(archived_ckpt)
        new_ckpt = out / "checkpoints" / f"{a.mechanism}_seed{a.seed}_{a.h}d_t12.pt"
        new_ckpt.parent.mkdir(parents=True, exist_ok=True)
        if new_ckpt.exists():
            raise FileExistsError(new_ckpt)
        old = (exp4._checkpoint_path, exp4._assert_experiment4_path, exp4._make_experiment4_model)
        exp4._checkpoint_path = lambda mechanism, horizon, seed: new_ckpt
        exp4._assert_experiment4_path = lambda p: Path(p).resolve() if Path(p).resolve().is_relative_to(out.resolve()) else old[1](p)
        exp4._make_experiment4_model = observed(old[2])
        try:
            result = exp4.train_and_validate_cell(a.mechanism, f"{a.h}d", a.seed)
        finally:
            exp4._checkpoint_path, exp4._assert_experiment4_path, exp4._make_experiment4_model = old
        wall = time.perf_counter() - t0
        after = sha(archived_ckpt)
        ds = formal.load_frozen_panel()
        formal._formal_check_dataset_contract(ds)
        _, _, test_loader = formal._build_training_loaders("daily_weekly", a.h, ds, batch_size=formal.BATCH_SIZE)
        model = old[2](a.mechanism)
        model.load_state_dict(torch.load(new_ckpt, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        pred, _, _ = exp4._batched_inference(model, test_loader, COUNTS["test"], torch.device("cpu"))
        mask = ds.split == "test"
        new = pd.DataFrame(dict(ticker=ds.ticker[mask], anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime("%Y-%m-%d"),
                                split="test", y_true=ds.y[a.h][mask], y_pred=pred))
        archived_preds = E4 / "experiment4_test_predictions.csv"
        oldS = pd.read_csv(E4 / "experiment4_training_summary.csv")
        oldS = oldS[(oldS.mechanism == a.mechanism) & (oldS.horizon == f"{a.h}d") & (oldS.seed == a.seed)].iloc[0]
        epochs, best = int(result["summary"]["epochs_trained"]), int(result["summary"]["best_epoch"])
        arch_epochs, arch_best = int(oldS.epochs_trained), int(oldS.best_epoch)
        extra = dict(new_checkpoint=rel(new_ckpt), new_checkpoint_sha256=sha(new_ckpt), archived_checkpoint=rel(archived_ckpt),
                     archived_checkpoint_unchanged=before == after, trainer_runtime_seconds=float(result["summary"]["runtime_seconds"]),
                     archived_trainer_runtime_seconds=float(oldS.runtime_seconds))
    else:
        old_factory = formal._make_model
        formal._make_model = observed(old_factory)
        try:
            result = formal.train_one_cell("daily_weekly", a.h, "adaptive")
        finally:
            formal._make_model = old_factory
        wall = time.perf_counter() - t0
        new = result["predictions"].copy()
        archived_preds = MAIN / "panel_pathformer_test_predictions.csv"
        hist = pd.DataFrame(result["history"])
        oldH = pd.read_csv(MAIN / "panel_pathformer_training_history.csv")
        oldH = oldH[(oldH.frequency_setting == "daily_weekly") & (oldH.horizon.astype(str).str.removesuffix("d").astype(int) == a.h) & (oldH.seed == 42)]
        summ = result["summary"].iloc[0].to_dict() if hasattr(result["summary"], "iloc") else result["summary"]
        epochs, arch_epochs = len(hist), len(oldH)
        best = int(summ.get("best_epoch", -1))
        oldS = formal_subset(MAIN / "panel_pathformer_summary_metrics.csv", "adaptive", a.h, 42).iloc[0]
        arch_best = int(oldS.get("best_epoch", -1))
        extra = dict(trainer_runtime_seconds=float(summ.get("training_time_seconds", summ.get("runtime_seconds", float("nan")))))
    for hnd in hooks:
        hnd.remove()
    tel.sample("end", force=True)
    tel.close()
    old_preds = formal_subset(archived_preds, a.mechanism, a.h, a.seed)
    cmp = compare(new, old_preds)
    write_csv(out / f"{stem}_test_predictions.csv", new[["ticker", "anchor_date", "y_true", "y_pred"]])
    tele = [json.loads(l) for l in (out / f"telemetry/{stem}.jsonl").read_text().splitlines()]
    res = dict(trainer=a.trainer, mechanism=a.mechanism, seed=a.seed, h=a.h, features="original (frozen V2)", profile="Step 3B 12/12",
               started_utc=start_utc, completed_utc=utc(), wall_clock_seconds=wall, epochs_trained=epochs, archived_epochs_trained=arch_epochs,
               best_epoch=best, archived_best_epoch=arch_best, archived_predictions=rel(archived_preds), comparison=cmp,
               environment=environment, pid=os.getpid(), parent_pid=os.getppid(), allowed_sibling_pids=sorted(allowed),
               telemetry=[dict(event=r["event"], utc=r["utc"], torch_num_threads=r["torch_num_threads"], torch_num_interop_threads=r["torch_num_interop_threads"],
                               environment=r["environment"], load_average_1_5_15=r["load_average_1_5_15"]) for r in tele],
               telemetry_thread_counts_changed=bool(tel.changed), **extra)
    write_json(out / f"{stem}_result.json", res)
    print("RESULT", stem, cmp["classification"], cmp["max_abs_delta_pred"], f"{wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
