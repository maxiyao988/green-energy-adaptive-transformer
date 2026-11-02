"""Step17 exclusive I/O and imported scientific primitives; no trainer import."""
import os,sys,json,hashlib,datetime,io
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revision_work/outputs/analysis17';A16=ROOT/'revision_work/outputs/analysis16'
for d in ['skill','ensemble']:
 sys.path.insert(0,str(ROOT/'revision_work/scripts'/d))
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
from skill_common import np,pd,mbb_index,holm,rank_ic_rows,nw_t,reference_forecasts,load_meta
from a2_skill import cell_metrics
from a3_hierarchical import series_stats
from ensemble_robust import ens_series
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
from functools import lru_cache
HORIZONS=[5,10,20];SEEDS=[0,1,21,42,3407];GRID=np.arange(11,dtype=float)/10;B=20000;BASE=20261019
FULL=ROOT/'revision_work/outputs/fullrun';SESSIONS=FULL/'sessions';DAY=SESSIONS/'14c_day_20261017T050034Z_12db13f7'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def put(name,payload):
 p=(OUT/name).resolve();assert p.is_relative_to(OUT.resolve());p.parent.mkdir(exist_ok=True,parents=True);data=payload.encode() if isinstance(payload,str) else payload
 if p.exists():assert sha(p)==hashlib.sha256(data).hexdigest(),f'Never overwrite: {p}'
 else:
  with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return p

def csv(name,d):return put(name,d.to_csv(index=False,float_format='%.17g'))
def js(name,d):return put(name,json.dumps(d,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n')
def pq(name,d):return put(name,d.to_parquet(index=False))
def canon(d):
 d=d.copy();d['anchor_date']=pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d');return d.sort_values(['anchor_date','ticker'],kind='mergesort').reset_index(drop=True)
def complete(name,files,meta=None):return js(name,dict(meta=meta or {},files=[dict(path=rel(p),sha256=sha(p)) for p in files]))
def verified(name):
 p=OUT/name
 if not p.exists():return False
 j=json.loads(p.read_text());assert all(sha(ROOT/r['path'])==r['sha256'] for r in j['files']),p
 return True
@lru_cache(None)
def labels(w,h):return pd.read_parquet(A16/f'labels/{w}_h{h}.parquet')
@lru_cache(None)
def split_labels(w,h,sp):return canon(labels(w,h).query('split==@sp'))
def combine(p,m,l):
 if l==0:return np.full(len(p),m,dtype=float)
 if l==1:return np.asarray(p,dtype=float).copy()
 return m+l*(np.asarray(p,dtype=float)-m)
def ensemble(P,y):
 P=np.asarray(P,float);return ens_series(y,(len(y)//17,17),P,P,y.reshape(-1,17),np.full(len(P),1/len(P)))[2]
def group_id(w,inp,model,h):return f'{w}_{inp}_{model}_h{h}'
