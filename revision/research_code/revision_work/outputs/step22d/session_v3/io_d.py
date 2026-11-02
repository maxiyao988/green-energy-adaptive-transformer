"""Bind unchanged Step22B I/O helper bytecode into the new exclusive namespace."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import types

_NEW_OUT = Path(__file__).resolve().parent
_ROOT = _NEW_OUT.parents[3]
_OLD = _ROOT / 'revision_work/outputs/step22b'
_spec = importlib.util.spec_from_file_location('_step22b_io_original', _OLD / 'io22b_v3.py')
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
for _name, _value in list(vars(_original).items()):
    if not _name.startswith('_'):
        globals()[_name] = _value
OUT, ROOT, OLD = _NEW_OUT, _ROOT, _OLD
for _name, _value in list(vars(_original).items()):
    if isinstance(_value, types.FunctionType) and _value.__module__ == _original.__name__:
        _clone = types.FunctionType(_value.__code__, globals(), _name, _value.__defaults__, _value.__closure__)
        _clone.__kwdefaults__ = _value.__kwdefaults__
        _clone.__annotations__ = _value.__annotations__
        globals()[_name] = _clone

def check_protocol():
    m = read(OUT / 'RNG_MANIFEST.json')
    assert sha(OUT / 'PROTOCOL_STEP22D.md') == m['protocol_sha256'], 'Immutable protocol changed'
    for item in m['implementation'] + m['source_manifests']:
        assert sha(ROOT / item['path']) == item['sha256'], 'Registered file changed: ' + item['path']
    return m['protocol_sha256']

def check_old_sources():
    m = read(OLD / 'RNG_MANIFEST.json')
    rows = []
    for item in m['implementation'] + m['source_manifests']:
        actual = sha(ROOT / item['path'])
        rows.append(dict(path=item['path'], expected_sha256=item['sha256'], actual_sha256=actual, unchanged=actual == item['sha256']))
    rows.append(dict(path=rel(OLD / 'PROTOCOL_STEP22B.md'), expected_sha256=m['protocol_sha256'],
                     actual_sha256=sha(OLD / 'PROTOCOL_STEP22B.md'), unchanged=sha(OLD / 'PROTOCOL_STEP22B.md') == m['protocol_sha256']))
    return rows

B2 = ROOT / 'revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041'

def counts(window):
    return read(B2 / 'design.json')['windows'][window]['counts']

def spec(setting, window, h, seed):
    sd = read(OUT / 'SIGMAS.json')['values'][window][str(h)]
    return dict(setting=setting, window=window, model='lstm', h=int(h), seed=int(seed), gate=False,
                learning_rate=1e-4 if setting=='T0' else 3e-5, delta=1.0 if setting=='T0' else sd)

def job_id(job):
    return f"{job['setting']}_lstm_{job['window']}_h{job['h']}_s{job['seed']}"

def barrier():
    path=OUT/'ALL_TRAINING_FINISHED.json'
    assert path.exists(), 'TEST remains locked until all60 fits have terminal records'
    d=read(path);assert d['protocol_sha256']==check_protocol()
    assert read(OUT/'DATA_GATE.json')['passed'] and read(OUT/'TIMING_GATE.json')['proceed']
    plan=read(OUT/'GRID_PLAN.json')['jobs'];assert len(plan)==len(d['runs'])==60
    assert {job_id(j) for j in plan}=={job_id(r['job']) for r in d['runs']}
    for r in d['runs']:
        saved=verify_record(ROOT/r['record']);assert saved['job']==r['job']
    return d

def exports():
    return {k: globals()[k] for k in ['OUT', 'ROOT', 'OLD', 'B2', 'counts', 'SEEDS', 'HS', 'COUNTS', 'ENV_KEYS', 'WEIGHTS',
        'utc', 'stamp', 'elapsed', 'sha', 'rel', 'target', 'text', 'js', 'csv', 'read', 'check_protocol', 'environment',
        'spec', 'job_id', 'artifact', 'verify_record', 'completion', 'source_rows', 'barrier']}
