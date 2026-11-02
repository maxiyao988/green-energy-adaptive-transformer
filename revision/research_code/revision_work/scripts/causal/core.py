"""Exclusive Step 4 I/O and formal V2 imports. No existing file is writable."""
import sys,os
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'revision_work/outputs/causal_features'
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd,torch,json,hashlib
from datetime import datetime,timezone
from scripts.python import panel_common as common
from scripts.python import panel_pathformer_main_v2 as formal
from scripts.python import panel_pathformer_experiment3 as exp3
from scripts.python import panel_pathformer_experiment4_multiseed as exp4
from scripts.python import panel_build_multiscale_dataset as builder
CUTOFF='2026-07-07';START='2015-08-05';CHANNELS=['open','high','low','close','volume']
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
 if not p.is_relative_to(OUT):raise ValueError('Output outside new causal namespace')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError('Refusing overwrite: '+str(p))
 return p
def csvout(name,data,columns=None):
 d=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data,columns=columns)
 with target(OUT/name).open('x',newline='') as f:d.to_csv(f,index=False)
 return d
def jsonout(name,d):
 with target(OUT/name).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def npout(name,a):
 with target(OUT/name).open('xb') as f:np.save(f,a,allow_pickle=False)
def parquet(name,d):
 payload=d.to_parquet(index=False,engine='pyarrow')
 with target(OUT/name).open('xb') as f:f.write(payload)
def threads(n=3):
 torch.set_num_threads(n);torch.set_num_interop_threads(1)
 return dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),device='cpu')
def load_causal():
 meta=pd.read_csv(OUT/'tensors/sample_index.csv')
 return common.PanelDataset(ticker=meta.ticker.to_numpy(),anchor_date=pd.to_datetime(meta.anchor_date).to_numpy(),split=meta.split.to_numpy(),X_daily=np.load(OUT/'tensors/X_daily_causal_raw.npy'),X_weekly=np.load(OUT/'tensors/X_weekly_causal_raw.npy'),y={h:np.load(OUT/f'tensors/y_h{h}_frozen.npy') for h in [5,10,20]})
def metrics(y,p,ticker,dates):return common.evaluate_predictions(y,p,ticker,dates)
def fmt_table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in row)+' |' for row in rows])+'\n'
