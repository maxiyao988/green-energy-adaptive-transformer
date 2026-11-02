"""Import Step22D I/O bytecode into an exclusive Step22E namespace."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,types
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
D=ROOT/'revision_work/outputs/step22d/session_v3'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
_d=module('_22e_original_io_d',D/'io_d.py')
for _n,_v in vars(_d).items():
    if not _n.startswith('_'):globals()[_n]=_v
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
D=ROOT/'revision_work/outputs/step22d/session_v3'
for _n,_v in vars(_d).items():
    if isinstance(_v,types.FunctionType) and not _n.startswith('_'):
        _c=types.FunctionType(_v.__code__,globals(),_n,_v.__defaults__,_v.__closure__)
        _c.__kwdefaults__=_v.__kwdefaults__;_c.__annotations__=_v.__annotations__;globals()[_n]=_c

def check_protocol():
    m=read(OUT/'RNG_MANIFEST.json')
    assert sha(OUT/'PROTOCOL_STEP22E.md')==m['protocol_sha256'],'Immutable Step22E protocol changed'
    for a in m['implementation']+m['source_manifests']:
        assert sha(ROOT/a['path'])==a['sha256'],'Registered source changed: '+a['path']
    return m['protocol_sha256']

def spec(model,window,h,seed):
    assert model in ['static','adaptive'] and window in ['W1','W2'] and int(h) in HS and int(seed) in SEEDS
    a=read(D/'SIGMAS.json')['values'][window][str(h)]
    b=read(OUT/'SIGMAS.json')['values'][window][str(h)]
    assert a==b
    return dict(setting='T3',model=model,window=window,h=int(h),seed=int(seed),gate=False,learning_rate=3e-5,delta=a)

def job_id(job):return f"T3_{job['model']}_{job['window']}_h{job['h']}_s{job['seed']}"

def attempts(job):
    paths=sorted((OUT/'fits'/job_id(job)).glob('attempt_*/COMPLETE.json'))
    return [(p,verify_record(p)) for p in paths]

def completion(job):
    good=[(p,r) for p,r in attempts(job) if r['status']=='COMPLETE']
    assert len(good)<=1,'Successful cell was trained twice: '+job_id(job)
    return good[0] if good else (None,None)

def latest_terminal(job):
    found=attempts(job);return found[-1] if found else (None,None)

def failed_attempts(job):return sum(r['status']!='COMPLETE' for _,r in attempts(job))

def barrier():
    p=OUT/'ALL_TRAINING_FINISHED.json';assert p.exists(),'TEST locked until all60 cells finish'
    d=read(p);assert d['protocol_sha256']==check_protocol()
    assert read(OUT/'DATA_GATE.json')['passed'] and read(OUT/'TIMING_GATE.json')['proceed']
    plan=read(OUT/'GRID_PLAN.json')['jobs'];assert len(plan)==len(d['runs'])==60
    assert {job_id(j) for j in plan}=={job_id(r['job']) for r in d['runs']}
    for row in d['runs']:
        cp,r=completion(row['job']);assert cp is not None and rel(cp)==row['record']
        assert r['status']=='COMPLETE' and failed_attempts(row['job'])<2
    return d

def exports():
    keys=['OUT','ROOT','OLD','B2','counts','SEEDS','HS','COUNTS','ENV_KEYS','WEIGHTS','utc','stamp','elapsed','sha','rel',
          'target','text','js','csv','read','check_protocol','environment','spec','job_id','artifact','verify_record',
          'completion','source_rows','barrier']
    return {k:globals()[k] for k in keys}
