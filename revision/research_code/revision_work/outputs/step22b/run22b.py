"""User-launched Step22B state machine; no numerical imports in the scheduler."""
import sys
sys.dont_write_bytecode = True
from io22b_v2 import *
import argparse
import fcntl
import signal
import subprocess
import time
import traceback

WORKER = OUT / 'worker22b_v2.py'
ACTIVE = {}

def event(kind, **data):
    js(OUT / 'events' / (stamp() + '.json'), dict(utc=utc(), kind=kind, **data))

def command_status():
    # Status is read-only: it does not import Torch or verify/open TEST artifacts.
    rows = []
    for path in sorted((OUT / 'fits').glob('*/attempt_*/COMPLETE.json')):
        data = read(path)
        rows.append(data)
    gate = [r for r in rows if r['job'].get('gate')]
    grid = [r for r in rows if not r['job'].get('gate')]
    decision = read(OUT / 'TIMING_GATE.json') if (OUT / 'TIMING_GATE.json').exists() else None
    total = 120 if decision and decision['include_fixed'] else 90
    print(f"Step22B: gate fits={len(gate)}; grid terminal={len(grid)}/{total}; complete={sum(r['status']=='COMPLETE' for r in grid)}; failed={sum(r['status']!='COMPLETE' for r in grid)}", flush=True)
    for name in ['REPRODUCIBILITY_GATE.json', 'TIMING_GATE.json', 'ALL_TRAINING_FINISHED.json', 'ANALYSIS_COMPLETE.json']:
        print(name, 'present' if (OUT / name).exists() else 'pending')
    result = subprocess.run(['ps', '-Ao', 'pid,etime,pcpu,command'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if str(OUT) in line and any(n in line for n in ['worker22b_v2.py', 'run22b.py', 'analyze22b.py']):
            print(line)
    reports = sorted(OUT.glob('STOP_REPORT_*.md'))
    if reports:
        print('Latest stop report:', reports[-1])
    if (OUT / 'REPORT_STEP22B.md').exists():
        print('Final report:', OUT / 'REPORT_STEP22B.md')

def status_snapshot():
    rows = []
    for path in sorted((OUT / 'fits').glob('*/attempt_*/COMPLETE.json')):
        d = read(path)
        rows.append(dict(job_id=job_id(d['job']), **d['job'], status=d['status'], epochs=d.get('epochs'),
                         best_epoch=d.get('best_epoch'), ended_utc=d.get('ended_utc'), record=rel(path)))
    for pid, item in ACTIVE.items():
        rows.append(dict(job_id=job_id(item['job']), **item['job'], status='RUNNING', pid=pid, started_utc=item['started']))
    folder = target(OUT / 'status' / stamp())
    with target(folder / 'status.csv').open('x', newline='') as handle:
        fields = ['job_id', 'setting', 'model', 'h', 'seed', 'gate', 'status', 'pid', 'started_utc', 'ended_utc', 'epochs', 'best_epoch', 'record']
        writer = csvlib.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def failed_record(item, code):
    folder = item['folder']
    if (folder / 'COMPLETE.json').exists():
        return
    error = f'Worker exit {code}; no terminal model record. See worker.log and UNHANDLED files.'
    paths = [p for p in folder.iterdir() if p.is_file()]
    js(folder / 'COMPLETE.json', dict(job=item['job'], status='FAILED', error=error, protocol_sha256=check_protocol(),
       epochs=None, best_epoch=None, started_utc=item['started'], ended_utc=utc(),
       elapsed_utc_seconds=elapsed(item['started'], utc()), artifacts=[artifact(p) for p in paths]))

def queue(jobs):
    pending = [job for job in jobs if completion(job)[0] is None]
    while pending or ACTIVE:
        while pending and len(ACTIVE) < 3:
            job = pending.pop(0)
            folder = target(OUT / 'fits' / job_id(job) / ('attempt_' + stamp()))
            folder.mkdir(parents=True)
            handle = target(folder / 'worker.log').open('x')
            started = utc()
            process = subprocess.Popen([sys.executable, '-B', str(WORKER), 'train', json.dumps(job), str(folder)],
                       cwd=ROOT, env=environment(), stdout=handle, stderr=subprocess.STDOUT)
            ACTIVE[process.pid] = dict(process=process, handle=handle, job=job, folder=folder, started=started)
            event('training_launch', job=job, pid=process.pid, folder=rel(folder))
            status_snapshot()
        for pid, item in list(ACTIVE.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['handle'].close()
            failed_record(item, code)
            del ACTIVE[pid]
            path, data = completion(item['job'])
            event('training_terminal', job=item['job'], record=rel(path), status=data['status'])
            print('TERMINAL', job_id(item['job']), data['status'], len(ACTIVE), 'active', len(pending), 'pending', flush=True)
            status_snapshot()
        if ACTIVE:
            time.sleep(.5)

def helper(script, *args):
    log = target(OUT / 'logs' / (stamp() + '_' + script + '.log'))
    with log.open('x') as handle:
        result = subprocess.run([sys.executable, '-B', str(OUT / script), *args], cwd=ROOT,
                     env=environment(), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} failed; read {log}')

def timing_gate(pilots):
    if (OUT / 'TIMING_GATE.json').exists():
        d = read(OUT / 'TIMING_GATE.json')
        assert d['protocol_sha256'] == check_protocol()
        if not d['proceed']:
            raise RuntimeError('STOP: registered timing gate exceeded 14h')
        return d
    results = [completion(job)[1] for job in pilots]
    if any(r is None or r['status'] != 'COMPLETE' for r in results):
        raise RuntimeError('STOP: timing pilot failed or degenerate; no projection allowed')
    start = read(OUT / 'TIMING_PILOT_START.json')['utc']
    end = max(r['ended_utc'] for r in results)
    seconds = elapsed(start, end)
    core, full = seconds * 30, seconds * 40
    d = dict(protocol_sha256=check_protocol(), pilot_jobs=pilots, started_utc=start, ended_utc=end,
             pilot_elapsed_utc_seconds=seconds, core_runs=90, workers=3, core_projection_hours=core / 3600,
             with_fixed_runs=120, with_fixed_projection_hours=full / 3600,
             include_fixed=full <= 12 * 3600, proceed=core <= 14 * 3600,
             formula='pilot batch UTC elapsed * (total runs / 3); gate fit excluded')
    js(OUT / 'TIMING_GATE.json', d)
    if not d['proceed']:
        raise RuntimeError('STOP: core projection >14h; TEST remains locked')
    return d

def execute():
    check_protocol()
    # Lock bytes are never rewritten; flock operates on a read-only existing descriptor on restart.
    lockpath = OUT / 'QUEUE_LOCK'
    if not lockpath.exists():
        text(lockpath, 'Immutable lock inode; ownership is the live OS flock, not these bytes.\n')
    lock = lockpath.open('r')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another Step22B queue is already active')
    if (OUT / 'ANALYSIS_COMPLETE.json').exists():
        print('Already finished:', OUT / 'REPORT_STEP22B.md')
        return
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    try:
        event('queue_start', pid=os.getpid(), profile='CPU12/12; OMP unset; dynamic off')
        if not (OUT / 'T0_SOURCES.json').exists():
            # Registry/manifests only; TEST arrays/predictions are not read.
            rows = source_rows()
            js(OUT / 'T0_SOURCES.json', dict(protocol_sha256=check_protocol(), sources=rows))
        if not (OUT / 'ALL_TRAINING_FINISHED.json').exists():
            gate = spec('T0', 'static', 5, 0, True)
            queue([gate])
            if not (OUT / 'REPRODUCIBILITY_GATE.json').exists():
                helper('gates22b.py')
            reproducibility = read(OUT / 'REPRODUCIBILITY_GATE.json')
            if not reproducibility['passed']:
                raise RuntimeError('STOP: T0 validation reproducibility gate failed')
            pilots = [spec('T1', model, 20, 0) for model in ['static', 'adaptive', 'lstm']]
            if not (OUT / 'TIMING_PILOT_START.json').exists():
                js(OUT / 'TIMING_PILOT_START.json', dict(utc=utc(), jobs=pilots, protocol_sha256=check_protocol()))
            queue(pilots)
            decision = timing_gate(pilots)
            models = ['static', 'adaptive', 'lstm'] + (['fixed'] if decision['include_fixed'] else [])
            jobs = [spec(setting, model, h, seed) for setting in ['T1', 'T2'] for seed in SEEDS
                    for h in [20, 10, 5] for model in models]
            if not (OUT / 'GRID_PLAN.json').exists():
                js(OUT / 'GRID_PLAN.json', dict(protocol_sha256=check_protocol(), jobs=jobs))
            assert read(OUT / 'GRID_PLAN.json')['jobs'] == jobs
            queue(jobs)
            saved = []
            for job in jobs:
                path, result = completion(job)
                assert result is not None
                saved.append(dict(job=job, record=rel(path), status=result['status']))
            js(OUT / 'ALL_TRAINING_FINISHED.json', dict(protocol_sha256=check_protocol(), utc=utc(), runs=saved))
            event('all_training_finished', runs=len(jobs))
        barrier()
        plan = read(OUT / 'GRID_PLAN.json')['jobs']
        models = sorted(set(j['model'] for j in plan))
        evaluation = [spec('T0', model, h, seed) for model in models for h in HS for seed in SEEDS] + plan
        event('evaluation_phase_start', cells=len(evaluation))
        # Sequential TEST workers: no new training can occur after the barrier.
        for job in evaluation:
            folder = target(OUT / 'evaluation' / job_id(job))
            finished = sorted(folder.glob('attempt_*/COMPLETE.json'))
            if finished:
                assert len(finished) == 1
                verify_record(finished[0])
                continue
            attempt = target(folder / ('attempt_' + stamp()))
            attempt.mkdir(parents=True)
            with target(attempt / 'worker.log').open('x') as handle:
                result = subprocess.run([sys.executable, '-B', str(WORKER), 'evaluate', json.dumps(job), str(attempt)],
                             cwd=ROOT, env=environment(), stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Evaluation failed; no training will restart: ' + str(attempt))
        helper('analyze22b.py')
        event('queue_complete')
        print('Done:', OUT / 'REPORT_STEP22B.md', flush=True)
        subprocess.run(['/usr/bin/afplay', '/System/Library/Sounds/Glass.aiff'], check=False)
    except BaseException as error:
        # Interrupts preserve partial artifacts; never mark interrupted work terminal or overwrite it.
        for item in ACTIVE.values():
            if item['process'].poll() is None:
                item['process'].terminate()
        for item in ACTIVE.values():
            item['process'].wait()
            item['handle'].close()
        ACTIVE.clear()
        stop = OUT / ('STOP_REPORT_' + stamp() + '.md')
        text(stop, '# BLOCKING\n\n' + repr(error) + '\n\nProtocol SHA-256: ' + check_protocol() +
             '\n\nNo TEST evaluation is permitted before the all-training barrier. Existing artifacts were retained.\n\n```\n' + traceback.format_exc() + '\n```\n')
        print('STOP:', error, '\nReport:', stop, flush=True)
        raise
    finally:
        awake.terminate()
        awake.wait()
        lock.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'status'])
    args = parser.parse_args()
    if args.command == 'status':
        command_status()
    else:
        execute()
