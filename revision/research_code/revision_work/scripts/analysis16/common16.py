"""Step16 exclusive, content-verified I/O and imports. No training entrypoint."""
import os,sys,json,hashlib,importlib.util,datetime
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'revision_work/outputs/analysis16'
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
for d in ['skill','ensemble','portfolio','causal']:
 sys.path.insert(0,str(ROOT/'revision_work/scripts'/d))
import numpy as np,pandas as pd
SEEDS=[42,0,1,21,3407];HS=[5,10,20];B=20000;BASE=20261019
FULL=ROOT/'revision_work/outputs/fullrun';SESSION=FULL/'sessions'
DAY=SESSION/'14c_day_20261017T050034Z_12db13f7'
B2=SESSION/'14c_B2_20261017T101438Z_1a8d5041'
B2S=SESSION/'14c_B2_supplement_20261017T162556Z_bfb4b1be'
B3=SESSION/'14c_B3_20261017T201559Z_f47c5c3c'
B5=SESSION/'14c_B5_20261018T142713Z_6908ba80'
FIX=SESSION/'14c_B5fix_20261019T055730Z_6da4918c'
FROZEN=ROOT/'revision_work/outputs/frozen'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def path(p):return ROOT/p

def put(name,payload):
 p=(OUT/name).resolve();assert p.is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 data=payload.encode() if isinstance(payload,str) else payload
 if p.exists():
  assert sha(p)==hashlib.sha256(data).hexdigest(),f'Existing artifact differs; never overwrite: {p}'
 else:
  with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return p

def js(name,obj):return put(name,json.dumps(obj,indent=2,default=str,sort_keys=True,allow_nan=False)+'\n')
def csv(name,d):return put(name,d.to_csv(index=False,float_format='%.17g'))
def pq(name,d):return put(name,d.to_parquet(index=False))
def complete(name,files,meta=None):return js(name,dict(files=[dict(path=rel(p),sha256=sha(p)) for p in files],meta=meta or {}))
def verified(name):
 p=OUT/name
 if not p.exists():return False
 j=json.loads(p.read_text());assert all(sha(ROOT/r['path'])==r['sha256'] for r in j['files']),f'Bad resume: {p}'
 return True

def canon(d):
 d=d.copy();d['anchor_date']=pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d')
 return d.sort_values(['anchor_date','ticker'],kind='mergesort').reset_index(drop=True)
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
