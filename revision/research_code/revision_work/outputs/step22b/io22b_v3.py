"""Standard-library-only I/O and source bindings for immutable Step22B runs."""
import csv as csvlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SEEDS = [0, 1, 21, 42, 3407]
HS = [5, 10, 20]
COUNTS = dict(train=30634, val=6290, test=6647)
ENV_KEYS = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_DYNAMIC', 'MKL_DYNAMIC']
WEIGHTS = ['daily_w_5', 'daily_w_10', 'daily_w_20', 'daily_w_30',
           'weekly_w_2', 'weekly_w_4', 'weekly_w_8', 'weekly_w_13']

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]

def elapsed(a, b):
    return (dt.datetime.fromisoformat(b) - dt.datetime.fromisoformat(a)).total_seconds()

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))

def target(path):
    path = Path(path).absolute()
    if not path.is_relative_to(OUT) or path.resolve() != path:
        raise ValueError('Output outside exclusive namespace: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def text(path, value):
    with target(path).open('x') as handle:
        handle.write(value)

def js(path, value):
    text(path, json.dumps(value, indent=2, allow_nan=False,
         default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')

def csv(path, frame):
    with target(path).open('x', newline='') as handle:
        frame.to_csv(handle, index=False, float_format='%.17g', na_rep='N/A')

def read(path):
    return json.loads(Path(path).read_text())

def check_protocol():
    manifest = read(OUT / 'RNG_MANIFEST.json')
    if sha(OUT / 'PROTOCOL_STEP22B.md') != manifest['protocol_sha256']:
        raise RuntimeError('STOP: immutable protocol hash mismatch')
    for item in manifest['implementation']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise RuntimeError('STOP: registered implementation changed: ' + item['path'])
    for item in manifest.get('source_manifests', []):
        if sha(ROOT / item['path']) != item['sha256']:
            raise RuntimeError('STOP: registered source manifest changed: ' + item['path'])
    return manifest['protocol_sha256']

def environment():
    env = dict(os.environ)
    for key in ENV_KEYS[:5]:
        env.pop(key, None)
    env.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    return env

def spec(setting, model, h, seed, gate=False):
    return dict(setting=setting, model=model, h=int(h), seed=int(seed), gate=bool(gate))

def job_id(job):
    return ('gate_' if job.get('gate') else '') + f"{job['setting']}_{job['model']}_h{job['h']}_s{job['seed']}"

def artifact(path):
    return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)

def verify_record(path):
    data = read(path)
    if data['protocol_sha256'] != check_protocol():
        raise RuntimeError('Wrong protocol: ' + str(path))
    for item in data['artifacts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise RuntimeError('Artifact verification failed: ' + item['path'])
    return data

def completion(job):
    found = sorted((OUT / 'fits' / job_id(job)).glob('attempt_*/COMPLETE.json'))
    if len(found) > 1:
        raise RuntimeError('Multiple terminal attempts: ' + job_id(job))
    return (found[0], verify_record(found[0])) if found else (None, None)

def source_rows():
    with (ROOT / 'revision_work/outputs/analysis16/registry.csv').open(newline='') as handle:
        rows = list(csvlib.DictReader(handle))
    rows = [r for r in rows if r['input_set'] == 'C' and r['window'] == 'W3'
            and r['config'] == 'daily_weekly' and
            ((r['group'] == 'core' and r['model'] in ['static', 'adaptive', 'fixed']) or
             (r['group'] == 'neural' and r['model'] == 'lstm'))]
    if len(rows) != 60:
        raise RuntimeError('Expected all 60 C T0 source cells, got ' + str(len(rows)))
    for r in rows:
        r['h'], r['seed'] = int(r['h']), int(r['seed'])
        folder = (ROOT / r['source']).parent
        if r['model'] == 'lstm' and r['seed'] == 42:
            # The Step16 prediction is the B5 copy; history/VAL belong to 14c-day.
            day = ROOT / 'revision_work/outputs/fullrun/sessions/14c_day_20261017T050034Z_12db13f7/runs'
            found = list((day / f"B4_C_lstm_daily_weekly_h{r['h']}_s42").glob('attempt_*/COMPLETE.json'))
            if len(found) != 1:
                raise RuntimeError('Ambiguous 14c-day LSTM seed42 source: ' + str(found))
            folder = found[0].parent
        r['history'] = rel(folder / 'history.csv')
        r['val'] = rel(folder / 'val.parquet')
        r['complete'] = rel(folder / 'COMPLETE.json')
        for key in ['history', 'val', 'complete']:
            if not (ROOT / r[key]).is_file():
                raise RuntimeError('Required T0 source missing: ' + r[key])
        # Read only metadata/history/VAL here, never TEST predictions or metrics.
        old = read(ROOT / r['complete'])
        r['best_epoch'] = int(old['best_epoch'])
        if old['thread_counts_start'] != [12, 12] or old['thread_counts_end'] != [12, 12]:
            raise RuntimeError('T0 numerical profile not 12/12: ' + r['id'])
    return rows

def barrier():
    path = OUT / 'ALL_TRAINING_FINISHED.json'
    if not path.exists():
        raise RuntimeError('TEST remains locked: training barrier absent')
    data = read(path)
    assert data['protocol_sha256'] == check_protocol()
    decision = read(OUT / 'TIMING_GATE.json')
    expected = 120 if decision['include_fixed'] else 90
    assert decision['proceed'] and read(OUT / 'REPRODUCIBILITY_GATE.json')['passed']
    assert len(data['runs']) == expected
    planned = read(OUT / 'GRID_PLAN.json')['jobs']
    assert len(planned) == expected
    assert {job_id(r['job']) for r in data['runs']} == {job_id(j) for j in planned}
    for r in data['runs']:
        saved = verify_record(ROOT / r['record'])
        assert saved['job'] == r['job']
    return data
