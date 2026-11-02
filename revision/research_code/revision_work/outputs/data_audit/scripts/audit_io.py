"""Step 13 exclusive output and low-load HTTP provenance helpers."""
import os,sys,json,hashlib,time,urllib.request,urllib.error,csv
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['OMP_DYNAMIC']='FALSE';os.environ['MKL_DYNAMIC']='FALSE'
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'revision_work/outputs/data_audit'
UA = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")
def utc():return datetime.now(timezone.utc).isoformat()
def target(name):
 p=(OUT/name).resolve()
 if not p.is_relative_to(OUT.resolve()):raise ValueError(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 return p

def text(name,s):
 with target(name).open('x') as f:f.write(s)
def js(name,d):text(name,json.dumps(d,indent=2,allow_nan=False,default=str)+'\n')
def rows(name,rs,columns=None):
 import pandas as pd
 d=rs if isinstance(rs,pd.DataFrame) else pd.DataFrame(rs,columns=columns)
 with target(name).open('x',newline='') as f:d.to_csv(f,index=False,float_format='%.17g')
 return d

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
_last=0
_requests=[]
def fetch(url,label):
 global _last
 time.sleep(max(0,.30-(time.monotonic()-_last)))
 start=utc();_last=time.monotonic()
 req=urllib.request.Request(url,headers={'User-Agent':UA,'Accept-Encoding':'identity'})
 try:
  with urllib.request.urlopen(req,timeout=45) as r:body=r.read();code=r.status;headers=dict(r.headers);final=r.geturl()
 except urllib.error.HTTPError as e:body=e.read();code=e.code;headers=dict(e.headers);final=e.geturl()
 except Exception as e:body=str(e).encode();code=0;headers={};final=url
 path=target('sources/'+label)
 with path.open('xb') as f:f.write(body)
 meta=dict(url=url,final_url=final,requested_utc=start,status=code,user_agent=UA,bytes=len(body),sha256=sha(path),path=str(path.relative_to(ROOT)),headers=headers)
 js('sources/'+label+'.meta.json',meta);_requests.append(meta)
 print('HTTP',code,label,len(body),flush=True)
 return code,body,meta
