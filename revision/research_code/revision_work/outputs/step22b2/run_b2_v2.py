"""Step22B2 orchestration using the imported Step22B three-worker scheduler."""
import sys
sys.dont_write_bytecode = True
import io_b2 as io
from io_b2 import *
sys.path.insert(0, str(OLD))
import run22b_v3 as q
import subprocess
import fcntl
import traceback
q.__dict__.update(io.exports())
q.WORKER = OUT / 'worker_b2_v2.py'

def status():
    rows = [read(p) for p in (OUT / 'fits').glob('*/attempt_*/COMPLETE.json')]
    print('Step22B2 T3:', len(rows), '/45 terminal;', sum(r['status'] == 'COMPLETE' for r in rows), 'complete;', sum(r['status'] != 'COMPLETE' for r in rows), 'failed')
    for name in ['SOURCE_HASH_GATE.json', 'TIMING_GATE.json', 'ALL_TRAINING_FINISHED.json', 'ANALYSIS_COMPLETE.json']:
        p = OUT / name
        print(name, 'present' if p.exists() else 'pending')
    ps = subprocess.run(['ps', '-Ao', 'pid,etime,pcpu,command'], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        if str(OUT) in line and any(t in line for t in ['worker_b2', 'run_b2', 'analyze_b2']):
            print(line)
    for p in sorted(OUT.glob('STOP_REPORT_*.md'))[-1:]:
        print('Latest stop:', p)
    if (OUT / 'REPORT_STEP22B2.md').exists():
        print('Report:', OUT / 'REPORT_STEP22B2.md')

def source_gate():
    checks = check_old_sources()
    passed = all(r['unchanged'] for r in checks)
    if not (OUT / 'SOURCE_HASH_GATE.json').exists():
        js(OUT / 'SOURCE_HASH_GATE.json', dict(utc=utc(), protocol_sha256=check_protocol(), passed=passed,
           original_protocol_sha256=read(OLD / 'RNG_MANIFEST.json')['protocol_sha256'], checks=checks,
           no_test_values_read=True, new_training_required_for_this_gate=False))
    assert read(OUT / 'SOURCE_HASH_GATE.json')['passed'] and passed, 'STOP: Step22B T1/T2 source hashes changed'

def timing(pilots):
    if (OUT / 'TIMING_GATE.json').exists():
        d = read(OUT / 'TIMING_GATE.json')
        assert d['protocol_sha256'] == check_protocol()
        assert d['proceed'], 'STOP: T3 timing projection >10h'
        return d
    records = [completion(j)[1] for j in pilots]
    assert all(r is not None and r['status'] == 'COMPLETE' for r in records), 'STOP: timing pilot failed; projection undefined'
    for job in pilots:
        cp, rec = completion(job)
        verify_record(cp.parent / 'ADAPTER_COMPLETE.json')
        assert rec['lr'] == 3e-5 and rec['delta'] == {5:.098083,10:.138578,20:.195560}[job['h']]
    start = read(OUT / 'TIMING_PILOT_START.json')['utc']
    end = read(OUT / 'TIMING_PILOT_END.json')['utc']
    seconds = elapsed(start, end)
    d = dict(protocol_sha256=check_protocol(), started_utc=start, ended_utc=end, pilot_elapsed_utc_seconds=seconds,
             workers=3, total_runs=45, projected_hours=15 * seconds / 3600, threshold_hours=10,
             proceed=15 * seconds <= 10 * 3600, formula='actual pilot batch UTC elapsed * (45/3)')
    js(OUT / 'TIMING_GATE.json', d)
    assert d['proceed'], 'STOP: T3 timing projection >10h'
    print('TIMING GATE PASS; projected hours', d['projected_hours'], flush=True)
    return d

def run():
    check_protocol()
    lockpath = OUT / 'QUEUE_LOCK'
    if not lockpath.exists():
        text(lockpath, 'Immutable lock file; ownership is a live OS flock.\n')
    lock = lockpath.open('r')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'ANALYSIS_COMPLETE.json').exists():
        verify_record(OUT / 'ANALYSIS_COMPLETE.json')
        print('Already finished:', OUT / 'REPORT_STEP22B2.md')
        return
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    try:
        source_gate()
        q.event('queue_start', pid=os.getpid(), profile='CPU12/12; dynamic off; OMP unset',
                power=subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout)
        if not (OUT / 'ALL_TRAINING_FINISHED.json').exists():
            pilots = [spec('T3', m, 20, 0) for m in ['static', 'adaptive', 'lstm']]
            if not (OUT / 'TIMING_PILOT_START.json').exists():
                js(OUT / 'TIMING_PILOT_START.json', dict(utc=utc(), protocol_sha256=check_protocol(), jobs=pilots))
            q.queue(pilots)
            if not (OUT / 'TIMING_PILOT_END.json').exists():
                js(OUT / 'TIMING_PILOT_END.json', dict(utc=utc(), protocol_sha256=check_protocol()))
            timing(pilots)
            plan = [spec('T3', model, h, seed) for seed in SEEDS for h in [20, 10, 5] for model in ['static', 'adaptive', 'lstm']]
            if not (OUT / 'GRID_PLAN.json').exists():
                js(OUT / 'GRID_PLAN.json', dict(protocol_sha256=check_protocol(), jobs=plan))
            assert read(OUT / 'GRID_PLAN.json')['jobs'] == plan
            q.queue(plan)
            records = []
            for job in plan:
                p, d = completion(job)
                assert d is not None
                if d['status'] == 'COMPLETE':
                    verify_record(p.parent / 'ADAPTER_COMPLETE.json')
                    assert d['lr'] == 3e-5 and d['delta'] == {5:.098083,10:.138578,20:.195560}[job['h']]
                records.append(dict(job=job, status=d['status'], record=rel(p)))
            js(OUT / 'ALL_TRAINING_FINISHED.json', dict(utc=utc(), protocol_sha256=check_protocol(), runs=records))
            q.event('all_training_finished', runs=45)
        barrier()
        q.event('test_evaluation_start')
        for job in read(OUT / 'GRID_PLAN.json')['jobs']:
            folder = target(OUT / 'evaluation' / job_id(job))
            done = sorted(folder.glob('attempt_*/COMPLETE.json'))
            if done:
                assert len(done) == 1
                verify_record(done[0])
                continue
            attempt = target(folder / ('attempt_' + stamp()))
            attempt.mkdir(parents=True)
            with target(attempt / 'worker.log').open('x') as log:
                result = subprocess.run([sys.executable, '-B', str(q.WORKER), 'evaluate', json.dumps(job), str(attempt)],
                             cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
            assert result.returncode == 0, 'Evaluation failed: ' + str(attempt)
            verify_record(attempt / 'COMPLETE.json')
        q.helper('analyze_b2_v2.py')
        verify_record(OUT / 'ANALYSIS_COMPLETE.json')
        q.event('queue_complete')
        print('COMPLETE:', OUT / 'REPORT_STEP22B2.md', flush=True)
        subprocess.run(['/usr/bin/afplay', '/System/Library/Sounds/Glass.aiff'], check=False)
    except BaseException as error:
        for r in q.ACTIVE.values():
            if r['process'].poll() is None:
                r['process'].terminate()
        for r in q.ACTIVE.values():
            r['process'].wait()
            r['handle'].close()
        q.ACTIVE.clear()
        body = '# BLOCKING\n\n' + repr(error) + '\n\nProtocol SHA: `' + check_protocol() + '`.\n\nAll existing files retained.\n\n```text\n' + traceback.format_exc() + '\n```\n'
        p = OUT / ('STOP_REPORT_' + stamp() + '.md')
        text(p, body)
        if ('source hashes changed' in str(error) or 'timing projection' in str(error) or 'timing pilot failed' in str(error)) and not (OUT / 'REPORT_STEP22B2.md').exists():
            text(OUT / 'REPORT_STEP22B2.md', body)
        print('STOP:', p, flush=True)
        raise
    finally:
        awake.terminate()
        awake.wait()
        lock.close()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['run', 'status'])
    a = p.parse_args()
    if a.command == 'status':
        status()
    else:
        run()
