import os,sys,json,hashlib,datetime,csv
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step24';sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd
READ={}
def read(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;b=p.read_bytes();READ[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b
def js(p):return json.loads(read(p))
def df(p):
 read(p);return pd.read_csv(ROOT/p if not Path(p).is_absolute() else p,float_precision='round_trip')
def pq(p):
 read(p);return pd.read_parquet(ROOT/p if not Path(p).is_absolute() else p)
def write(name,f):
 with (OUT/name).open('x',newline='') as h:f.to_csv(h,index=False,float_format='%.17g')
def jwrite(name,x):
 with (OUT/name).open('x') as h:json.dump(x,h,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x),allow_nan=True)
def source_manifest(name):write(name,pd.DataFrame([dict(path=k,sha256=v) for k,v in READ.items()]))
def canon(f):
 f=f.copy();f.anchor_date=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d');return f.sort_values(['anchor_date','ticker'],kind='stable').reset_index(drop=True)
