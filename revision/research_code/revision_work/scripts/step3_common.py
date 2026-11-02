"""Shared evidence I/O. Existing repository files are read only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'revision_work/outputs'
sys.path.insert(0,str(ROOT))
# Only pyarrow is installed here; retain the original numpy/pandas/torch environment.
sys.path.append(str(OUT/'_deps_pyarrow15'))
sys.path.append(str(OUT/'_deps'))
import hashlib,json,platform,os
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
MAIN=ROOT/'dataset/audit/main_v2/pathformer'
E3=ROOT/'dataset/audit/experiment3_pathformer_ablation'
E4=ROOT/'dataset/audit/experiment4_multiseed'
WEIGHTS=['daily_w_5','daily_w_10','daily_w_20','daily_w_30','weekly_w_2','weekly_w_4','weekly_w_8','weekly_w_13']
COUNTS={'train':30634,'val':6290,'test':6647}
def utc():return datetime.now(timezone.utc).isoformat()
def rel(p):return str(Path(p).relative_to(ROOT))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def target(p):
 p=Path(p)
 if not p.is_relative_to(OUT):raise ValueError(f'Output outside permitted directory: {p}')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(f'Refusing overwrite: {p}')
 return p

def write_json(p,obj):
 with target(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def write_csv(p,df):
 with target(p).open('x',newline='') as f:df.to_csv(f,index=False)
def write_parquet(p,df):
 payload=df.to_parquet(index=False,engine='pyarrow')
 with target(p).open('xb') as f:f.write(payload)
def threads(n=3):
 torch.set_num_threads(n);torch.set_num_interop_threads(1)
 return dict(torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,python=sys.version,platform=platform.platform(),device='cpu')
def horizon(x):return int(str(x).removesuffix('d'))
def key_sorted(d):
 d=d.copy();d['anchor_date']=pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d')
 if d.duplicated(['ticker','anchor_date']).any():raise ValueError('Duplicate stock/date keys')
 return d.sort_values(['ticker','anchor_date'],kind='mergesort').reset_index(drop=True)
def mse(d):
 y=d.y_true.to_numpy(np.float32);p=d.y_pred.to_numpy(np.float32)
 return float(np.mean((y-p)**2))
def compare(new,old):
 a=key_sorted(new);b=key_sorted(old)
 if not a[['ticker','anchor_date']].equals(b[['ticker','anchor_date']]):raise ValueError('Prediction stock/date key mismatch')
 if not np.array_equal(a.y_true.to_numpy(np.float32),b.y_true.to_numpy(np.float32)):raise ValueError('Saved/current labels differ')
 delta=float(np.max(np.abs(a.y_pred.to_numpy(np.float64)-b.y_pred.to_numpy(np.float64))))
 return dict(n_rows=len(a),max_abs_delta_pred=delta,replayed_test_mse=mse(a),formal_test_mse=mse(b),test_mse_difference=mse(a)-mse(b),classification='EXACT' if delta<1e-7 else 'NEAR' if delta<1e-5 else 'DIFFERENT')
def formal_subset(path,mechanism,h,seed):
 d=pd.read_csv(path);mask=d.horizon.map(horizon).eq(h)
 if 'mechanism' in d:mask &= d.mechanism.eq(mechanism)
 if 'seed' in d:mask &= d.seed.eq(seed)
 if 'frequency_setting' in d:mask &= d.frequency_setting.eq('daily_weekly')
 return d[mask].copy()
