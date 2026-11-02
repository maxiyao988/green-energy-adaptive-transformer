"""Standard-library-only Step14A queue contracts and exclusive output helpers."""
import csv
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'revision_work/outputs/fullrun'
PYTHON = ROOT / '.venv/bin/python'
HERE = Path(__file__).resolve().parent
SEEDS = [42, 0, 1, 21, 3407]
HORIZONS = [20, 10, 5]
COUNTS = {'train': 30634, 'val': 6290, 'test': 6647}
ENV_KEYS = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_DYNAMIC', 'MKL_DYNAMIC']
WEIGHTS = ['daily_w_5', 'daily_w_10', 'daily_w_20', 'daily_w_30',
           'weekly_w_2', 'weekly_w_4', 'weekly_w_8', 'weekly_w_13']
EXACT_THRESHOLD = 1e-7


def utc():
    return datetime.now(timezone.utc).isoformat()


def unique():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]


def rel(path):
    return str(Path(path).relative_to(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target(path):
    path = Path(path).absolute()
    if not path.is_relative_to(OUT) or path.resolve() != path:
        raise ValueError(f'Output outside exclusive fullrun namespace: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing overwrite: {path}')
    return path


def jsonout(path, value):
    with target(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def textout(path, value):
    with target(path).open('x') as handle:
        handle.write(value)


def csvout(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with target(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile_environment():
    env = dict(os.environ)
    # Match the Step3B environment, including the absence of BLAS overrides.
    for key in ENV_KEYS[:5]:
        env.pop(key, None)
    env.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    return env


def job(input_set, mechanism, h, seed, queue):
    return dict(job_id=f'{input_set}_{mechanism}_dw_h{h}_s{seed}', input_set=input_set,
                mechanism=mechanism, horizon=h, seed=seed, queue=queue)


def jobs():
    result = [job('O', 'fixed', 5, 42, 'Q0'), job('O', 'single', 5, 42, 'Q0')]
    for seed in SEEDS:
        for h in HORIZONS:
            result += [job('C', mechanism, h, seed, 'Q1') for mechanism in ['static', 'adaptive']]
            if seed != 42:
                result += [job('O', mechanism, h, seed, 'Q2') for mechanism in ['single', 'fixed']]
            result += [job('C', mechanism, h, seed, 'Q3') for mechanism in ['single', 'fixed']]
    assert len(result) == 86 and len({row['job_id'] for row in result}) == 86
    return result


def verify_completion(path, expected_job=None):
    """Resume only immutable, fully validated artifacts whose hashes still match."""
    path = Path(path)
    record = json.loads(path.read_text())
    if expected_job is not None:
        for key in ['job_id', 'input_set', 'mechanism', 'horizon', 'seed', 'queue']:
            assert record[key] == expected_job[key], (key, path)
    assert record['validated'] and record['loader_generator_seed'] == 42
    assert record['thread_counts_start'] == [12, 12] == record['thread_counts_end']
    assert not record['thread_counts_changed']
    assert record['row_counts'] == COUNTS
    assert record['environment_start']['OMP_NUM_THREADS'] is None
    assert record['environment_start']['OMP_DYNAMIC'] == 'FALSE'
    assert record['environment_start']['MKL_DYNAMIC'] == 'FALSE'
    for row in record['artifacts']:
        artifact = ROOT / row['path']
        assert artifact.is_relative_to(OUT) and sha(artifact) == row['sha256'], artifact
    for row in record['dependencies']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    assert record['checkpoint'] in [row['path'] for row in record['artifacts']]
    return record


def completed(expected_job):
    for path in sorted((OUT / 'runs' / expected_job['job_id']).glob('attempt_*/COMPLETE.json')):
        try:
            return path, verify_completion(path, expected_job)
        except (AssertionError, OSError, ValueError, KeyError):
            continue
    return None, None


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows]) + '\n'
