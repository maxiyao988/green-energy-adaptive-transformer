"""Step 12a-supplement 1b: launch THREE 12/12 training processes at the same time and wait for all of them.

Before launching: refuses to start if any other Python interpreter is running. Keeps the Mac from idle sleep
with `caffeinate -i -w <launcher pid>` (ends automatically with the launcher). Each worker
(retrain_cell_t12.py) sets the 12/12 profile itself and logs threads, env vars and load averages at start/end.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "revision_work" / "outputs" / "input_sensitivity" / "step12a" / "part1b_parallel"
PY = str(ROOT / ".venv" / "bin" / "python")
JOBS = [
    ("static_s0_h5", ["--trainer", "exp4", "--mechanism", "static", "--seed", "0", "--h", "5"]),
    ("adaptive_s42_h10", ["--trainer", "main_v2", "--mechanism", "adaptive", "--seed", "42", "--h", "10"]),
    ("static_s1_h5", ["--trainer", "exp4", "--mechanism", "static", "--seed", "1", "--h", "5"]),
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def python_processes():
    out = subprocess.run(["ps", "-axo", "pid,ppid,comm"], capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    me, parent = os.getpid(), os.getppid()
    return [l.strip() for l in out if len(l.split(None, 2)) == 3 and int(l.split()[0]) not in (me, parent)
            and "python" in Path(l.split(None, 2)[2]).name.lower()]


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    others = python_processes()
    if others:
        with (OUT / "BLOCKED_launcher.json").open("x") as f:
            json.dump(others, f, indent=2)
        raise SystemExit(f"BLOCKING: other Python processes running: {others}")
    caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    procs = []
    t_launch, utc_launch = time.perf_counter(), utc()
    for name, args in JOBS:
        log = (OUT / f"{name}.log").open("x")
        p = subprocess.Popen([PY, "-B", "-W", "ignore", str(HERE / "retrain_cell_t12.py"), *args, "--out-subdir", f"part1b_parallel/{name}"],
                             stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        procs.append(dict(name=name, args=args, pid=p.pid, proc=p, start_mono=time.perf_counter(), start_utc=utc(), log=log))
    while any(x["proc"].poll() is None for x in procs):
        for x in procs:
            if x["proc"].poll() is not None and "end_mono" not in x:
                x["end_mono"], x["end_utc"] = time.perf_counter(), utc()
        time.sleep(2)
    for x in procs:
        if "end_mono" not in x:
            x["end_mono"], x["end_utc"] = time.perf_counter(), utc()
        x["log"].close()
    t_end = time.perf_counter()
    rec = dict(launched_utc=utc_launch, finished_utc=utc(), parallel_wall_seconds=t_end - t_launch, caffeinate_pid=caff.pid, launcher_pid=os.getpid(),
               jobs=[dict(name=x["name"], args=x["args"], pid=x["pid"], returncode=x["proc"].returncode, start_utc=x["start_utc"], end_utc=x["end_utc"],
                          process_wall_seconds=x["end_mono"] - x["start_mono"]) for x in procs])
    with (OUT / "launcher.json").open("x") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
