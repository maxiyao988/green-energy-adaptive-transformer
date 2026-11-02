"""Resumable saturated three-worker queue with an explicit Q0 gate and AC telemetry."""
from fullrun14a_io import *
import fcntl
import subprocess
import sys
import time


def power():
    return subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True, check=True).stdout.strip()


def snapshot_protected():
    skipped = {'.git', '.venv', 'node_modules', '__pycache__', '_packages', '.ipynb_checkpoints'}
    result = []
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skipped and not d.startswith('_deps')
                   and (Path(folder) / d).absolute() != OUT]
        for name in names:
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink():
                result.append(dict(path=rel(path), bytes=path.stat().st_size, sha256=sha(path)))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lockpath = OUT / 'queue.lock'
    if lockpath.exists():
        lock = lockpath.open('rb')
    else:
        lock = target(lockpath).open('x+')
        lock.write('Step14A scheduler lock; immutable after creation.\n')
        lock.flush()
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    session_id = unique()
    session_path = OUT / 'sessions' / session_id
    session_path.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    started = utc()
    assert 'AC Power' in power(), 'Connect the Mac to AC power before starting'
    caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    environment = profile_environment()
    jsonout(session_path / 'session.json', dict(started_utc=started, scheduler_pid=os.getpid(),
        caffeinate_pid=caffeinate.pid, caffeinate_command=['/usr/bin/caffeinate','-i','-w',str(os.getpid())],
        max_workers=3, profile='CPU12/12; OMP unset; dynamic controls FALSE', environment={k:environment.get(k) for k in ENV_KEYS},
        power=power(), source=rel(Path(__file__)), source_sha256=sha(__file__), n_jobs=86))
    if not (OUT / 'protected_inputs.json').exists():
        jsonout(OUT / 'protected_inputs.json', dict(utc=utc(), files=snapshot_protected()))
    protected = json.loads((OUT / 'protected_inputs.json').read_text())['files']
    if not (OUT / 'preflight.json').exists():
        with target(session_path / 'preflight.log').open('x') as log:
            code = subprocess.call([str(PYTHON), '-B', str(HERE / 'fullrun14a_worker.py'), '--preflight'], cwd=ROOT,
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError('Preflight failed; see new session/preflight.log')
    preflight = json.loads((OUT / 'preflight.json').read_text())
    assert preflight['passed']
    for dependency in preflight['dependencies']:
        assert sha(ROOT / dependency['path']) == dependency['sha256'], dependency
    plan = jobs()
    jsonout(session_path / 'queue_plan.json', plan)
    status_path = OUT / 'status.csv' if not (OUT / 'status.csv').exists() else session_path / 'status.csv'
    status_file = target(status_path).open('x', newline='')
    fields = ['utc','job_id','queue','input_set','mechanism','horizon','seed','status','pid','returncode',
              'wall_seconds','epochs','best_epoch','q0_classification','completion','message']
    writer = csv.DictWriter(status_file, fieldnames=fields)
    writer.writeheader()
    status_file.flush()
    jsonout(session_path / 'status_reference.json', dict(status_csv=rel(status_path),
        semantics='New append-only journal; exclusive full status snapshots after each completed/reused/blocked run. Old journals are read only.'))
    events = target(session_path / 'events.jsonl').open('x')
    states = {j['job_id']: dict(**j, status='PENDING') for j in plan}
    done, failed, active, pending, power_samples = {}, {}, [], [], []
    counter = 0

    def event(kind, **data):
        record = dict(utc=utc(), event=kind, **data)
        events.write(json.dumps(record) + '\n')
        events.flush()
        print(kind, record, flush=True)

    def status(spec, label, record=None, path=None, **extra):
        nonlocal counter
        row = dict(utc=utc(), **spec, status=label, pid='', returncode='', wall_seconds='', epochs='',
                   best_epoch='', q0_classification='', completion='', message='')
        if record:
            row.update(pid=record['pid'], wall_seconds=record['wall_clock_seconds'], epochs=record['epochs'],
                       best_epoch=record['best_epoch'], completion=rel(path),
                       q0_classification=(record.get('q0') or {}).get('classification',''))
        row.update(extra)
        writer.writerow(row)
        status_file.flush()
        os.fsync(status_file.fileno())
        states[spec['job_id']] = dict(**spec, status=label, completion=row['completion'], message=row['message'])
        counter += 1
        csvout(session_path / 'status_snapshots' / f'{counter:04d}' / 'status.csv',
               [dict(j, status=states[j['job_id']]['status'], completion=states[j['job_id']].get('completion',''),
                     message=states[j['job_id']].get('message','')) for j in plan])

    for spec in plan:
        path, record = completed(spec)
        if record:
            done[spec['job_id']] = (path, record)
            status(spec, 'SKIPPED_VERIFIED', record, path, message='All saved artifacts and dependencies verified on resume')
        else:
            pending.append(spec)

    def gate():
        q0_jobs = [j for j in plan if j['queue'] == 'Q0']
        if any(j['job_id'] in failed for j in q0_jobs):
            return False
        if any(j['job_id'] in done and done[j['job_id']][1]['q0']['classification'] != 'EXACT' for j in q0_jobs):
            return False
        if all(j['job_id'] in done for j in q0_jobs):
            return True
        return None

    next_power = 0.
    while pending or active:
        if caffeinate.poll() is not None:
            raise RuntimeError('caffeinate exited before the queue completed')
        if time.perf_counter() >= next_power:
            power_samples.append(dict(utc=utc(), power=power(), active_pids=[r['process'].pid for r in active]))
            next_power = time.perf_counter() + 60.
        outcome = gate()
        if outcome is False:
            for spec in list(pending):
                if spec['queue'] in ['Q2', 'Q3']:
                    status(spec, 'BLOCKED_Q0', message='Q0 DIFFERENT or failed; awaiting user reply')
                    pending.remove(spec)
        while len(active) < 3:
            eligible = next((j for j in pending if j['queue'] in ['Q0','Q1'] or outcome is True), None)
            if eligible is None:
                break
            pending.remove(eligible)
            attempt = OUT / 'runs' / eligible['job_id'] / ('attempt_' + unique())
            attempt.mkdir(parents=True, exist_ok=False)
            logfile = target(attempt / 'worker.log').open('x')
            command = [str(PYTHON), '-B', str(HERE / 'fullrun14a_worker.py'), '--job-id', eligible['job_id'], '--attempt', str(attempt)]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=logfile, stderr=subprocess.STDOUT)
            active.append(dict(process=process, log=logfile, job=eligible, attempt=attempt, start=time.perf_counter()))
            states[eligible['job_id']]['status'] = 'RUNNING'
            event('LAUNCH', **eligible, pid=process.pid, active_workers=len(active), attempt=rel(attempt), q0_gate=outcome)
        for item in list(active):
            code = item['process'].poll()
            if code is None:
                continue
            spec, attempt = item['job'], item['attempt']
            item['log'].close()
            active.remove(item)
            path = attempt / 'COMPLETE.json'
            try:
                if code:
                    raise RuntimeError(f'Worker exit {code}; see {rel(attempt / "worker.log")}')
                record = verify_completion(path, spec)
                done[spec['job_id']] = (path, record)
                status(spec, 'COMPLETE', record, path, returncode=code)
                event('EXIT', **spec, pid=item['process'].pid, returncode=code,
                      wall_seconds=time.perf_counter()-item['start'], completion=rel(path), q0=record['q0'], active_workers=len(active))
            except (AssertionError, OSError, ValueError, KeyError, RuntimeError) as exc:
                failed[spec['job_id']] = str(exc)
                status(spec, 'FAILED', pid=item['process'].pid, returncode=code, message=str(exc))
                event('FAILED', **spec, error=str(exc), returncode=code, attempt=rel(attempt), active_workers=len(active))
        if active:
            time.sleep(1)
        elif pending:
            raise RuntimeError('No eligible jobs and no active workers; inspect Q0 state')
    power_samples.append(dict(utc=utc(), power=power(), active_pids=[]))
    status_file.close()
    events.close()
    jsonout(session_path / 'power_events.json', power_samples)
    changed = []
    for item in protected:
        path = ROOT / item['path']
        if not path.is_file() or sha(path) != item['sha256']:
            changed.append(dict(path=item['path'], expected_sha256=item['sha256'], actual_sha256=sha(path) if path.is_file() else None))
    jsonout(session_path / 'protected_inputs_verification.json', dict(completed_utc=utc(), n_files=len(protected), passed=not changed, changed=changed))
    jsonout(session_path / 'queue_result.json', dict(started_utc=started, completed_utc=utc(),
        wall_clock_seconds=time.perf_counter()-start, q0_gate=gate(), n_complete=len(done), failures=failed,
        blocked=[j for j in plan if states[j['job_id']]['status']=='BLOCKED_Q0'],
        status_csv=rel(status_path), states=list(states.values())))
    with target(session_path / 'report.log').open('x') as log:
        report_code = subprocess.call([str(PYTHON), '-B', str(HERE / 'fullrun14a_report.py'), str(session_path)],
                                      cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if report_code:
        raise RuntimeError(f'Report generation failed; see {rel(session_path / "report.log")}')
    jsonout(session_path / 'FINISHED.json', dict(utc=utc(), n_complete=len(done), n_failed=len(failed), q0_gate=gate(),
        report_manifest=rel(session_path/'report_manifest.json'), caffeinate_alive_through_report=caffeinate.poll() is None))
    print('QUEUE FINISHED', utc(), len(done), 'Q0', gate(), flush=True)
    caffeinate.terminate()
    caffeinate.wait(timeout=10)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


if __name__ == '__main__':
    main()
