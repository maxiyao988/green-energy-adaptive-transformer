"""Step22B exclusive I/O helpers imported into the Step22F namespace."""
import sys,types,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
OLD=ROOT/'revision_work/outputs/step22b';E=ROOT/'revision_work/outputs/step22e'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
_original=module('_22f_io_b',OLD/'io22b_v3.py')
for _n,_v in vars(_original).items():
    if not _n.startswith('_'):globals()[_n]=_v
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
OLD=ROOT/'revision_work/outputs/step22b';E=ROOT/'revision_work/outputs/step22e'
for _n,_v in vars(_original).items():
    if isinstance(_v,types.FunctionType) and not _n.startswith('_'):
        _c=types.FunctionType(_v.__code__,globals(),_n,_v.__defaults__,_v.__closure__);_c.__kwdefaults__=_v.__kwdefaults__;globals()[_n]=_c
TENSOR_BASE=ROOT/'revision_work/outputs/causal_features';B2=TENSOR_BASE
VARIANTS=['Static-R10','Static-R100','Adaptive-R100']
BOOT_SEEDS=[42,0,1,21,3407]
def check_protocol():
    m=read(OUT/'RNG_MANIFEST.json');assert sha(OUT/'PROTOCOL_STEP22F.md')==m['protocol_sha256']
    for a in m['implementation']+m['source_manifests']:assert sha(ROOT/a['path'])==a['sha256'],'Registered source changed: '+a['path']
    return m['protocol_sha256']
def spec(variant,h,seed):
    assert variant in VARIANTS and int(h) in HS and int(seed) in SEEDS
    return dict(setting=variant,variant=variant,model='static' if variant.startswith('Static') else 'adaptive',h=int(h),seed=int(seed),gate=False,window='W3',learning_rate=1e-4,delta=1.,multiplier=10 if variant=='Static-R10' else 100)
def job_id(j):return f"{j['variant']}_h{j['h']}_s{j['seed']}"
def source_rows():
    rows=_original.source_rows();rows=[r for r in rows if r['model'] in ['fixed','static','adaptive']]
    assert len(rows)==45 and len({(r['model'],r['h'],r['seed']) for r in rows})==45
    for m in ['fixed','static','adaptive']:
        for h in HS:assert {r['seed'] for r in rows if r['model']==m and r['h']==h}==set(SEEDS)
    return rows
def attempts(j):return [(p,verify_record(p)) for p in sorted((OUT/'fits'/job_id(j)).glob('attempt_*/COMPLETE.json'))]
def completion(j):
    good=[(p,r) for p,r in attempts(j) if r['status']=='COMPLETE'];assert len(good)<=1
    return good[0] if good else (None,None)
def latest_terminal(j):
    a=attempts(j);return a[-1] if a else (None,None)
def failed_attempts(j):return sum(r['status']!='COMPLETE' for _,r in attempts(j))
def barrier():
    p=OUT/'ALL_TRAINING_FINISHED.json';assert p.exists(),'TEST locked until all45 fits succeed'
    a=read(p);assert a['protocol_sha256']==check_protocol()
    plan=read(OUT/'GRID_PLAN.json')['jobs'];assert len(plan)==len(a['runs'])==45
    assert {job_id(j) for j in plan}=={job_id(r['job']) for r in a['runs']}
    assert read(OUT/'DATA_GATE.json')['passed'] and read(OUT/'TIMING_GATE.json')['proceed']
    for row in a['runs']:
        cp,r=completion(row['job']);assert cp is not None and rel(cp)==row['record'] and failed_attempts(row['job'])<2
    return a
def exports():
    names=['OUT','ROOT','OLD','B2','SEEDS','HS','COUNTS','ENV_KEYS','WEIGHTS','utc','stamp','elapsed','sha','rel','target','text','js','csv','read','check_protocol','environment','spec','job_id','artifact','verify_record','completion','source_rows','barrier']
    return {n:globals()[n] for n in names}
