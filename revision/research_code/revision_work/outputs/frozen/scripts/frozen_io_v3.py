"""Step15 standard-library exclusive I/O, resume, profile, and provenance helpers."""
import os,sys,json,csv,hashlib,uuid
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'revision_work/outputs/frozen'
SEEDS=[42,0,1,21,3407]
HORIZONS=[5,10,20]
COUNTS={'train':30634,'val':6290,'test':6647}
WEIGHTS=['daily_w_5','daily_w_10','daily_w_20','daily_w_30','weekly_w_2','weekly_w_4','weekly_w_8','weekly_w_13']
ENV_KEYS=['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC']
def utc():return datetime.now(timezone.utc).isoformat()
def uid():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:6]
def rel(p):return str(Path(p).relative_to(ROOT))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def target(p):
 p=Path(p).absolute()
 if not p.is_relative_to(OUT) or p.resolve()!=p:raise ValueError('Write outside frozen namespace: '+str(p))
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 return p

def text(p,s):
 with target(p).open('x') as f:f.write(s)
def jout(p,d):text(p,json.dumps(d,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+chr(10))
def cout(p,d):
 with target(p).open('x',newline='') as f:d.to_csv(f,index=False,float_format='%.17g')
def pout(p,d):
 with target(p).open('xb') as f:f.write(d.to_parquet(index=False))
def nout(p,a):
 import numpy as np
 with target(p).open('xb') as f:np.save(f,a,allow_pickle=False)
def profile_env():
 e=dict(os.environ)
 for k in ENV_KEYS[:5]:e.pop(k,None)
 e.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 return e

def artifact(p):return dict(path=rel(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def validate_complete(p):
 j=json.loads(Path(p).read_text());assert j['validated']
 for a in j['artifacts']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
 return j

def finished(base):
 for p in sorted(Path(base).glob('attempt_*/COMPLETE.json')):
  try:return p,validate_complete(p)
  except (OSError,ValueError,AssertionError,KeyError):pass
 return None,None

def build_registry():
 import pandas as pd
 sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
 import fullrun14a_io as old
 statuses=pd.read_csv(ROOT/'revision_work/outputs/fullrun/status.csv').fillna('')
 sources={}
 for r in statuses.itertuples():
  if r.status=='COMPLETE' and r.mechanism in ['fixed','adaptive']:
   j=old.verify_completion(ROOT/r.completion)
   sources[(r.input_set,r.mechanism,int(r.horizon),int(r.seed))]=j
 e3=pd.read_csv(ROOT/'dataset/audit/experiment3_pathformer_ablation/experiment3_training_summary.csv')
 e4=pd.read_csv(ROOT/'dataset/audit/experiment4_multiseed/experiment4_training_summary.csv')
 result=[]
 for backbone,mech in [('S1','fixed'),('S2','adaptive')]:
  for inp in ['O','C']:
   for seed in SEEDS:
    for h in HORIZONS:
     key=(inp,mech,h,seed);base=dict(id=f'{backbone}_{inp}_h{h}_s{seed}',backbone=backbone,input_set=inp,mechanism=mech,horizon=h,seed=seed)
     if key in sources:
      j=sources[key];attempt=(ROOT/j['checkpoint']).parent
      base.update(provenance='14a',checkpoint=j['checkpoint'],checkpoint_sha256=j['checkpoint_sha256'],references={sp:rel(attempt/f'{sp}.parquet') for sp in COUNTS},formal_reference=rel(attempt/'test.parquet'),formal_reference_type='parquet')
     elif mech=='fixed':
      r=e3[(e3.mechanism=='fixed')&(e3.horizon==f'{h}d')].iloc[0]
      base.update(provenance='Exp3',checkpoint=r.checkpoint_path,checkpoint_sha256=r.checkpoint_sha256,references={sp:f'revision_work/outputs/preds/fixed_dw_h{h}_s42_{sp}.parquet' for sp in COUNTS},formal_reference='dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv',formal_reference_type='csv')
     elif seed==42:
      j=json.loads((ROOT/f'revision_work/outputs/repro_threads/runs/adaptive_dw_h{h}_s42_t12.json').read_text());assert j['classification']=='EXACT'
      base.update(provenance='Step3B',checkpoint=j['checkpoint'],checkpoint_sha256=j['checkpoint_sha256'],references={sp:f'revision_work/outputs/repro_threads/preds/adaptive_dw_h{h}_s42_t12_{sp}.parquet' for sp in COUNTS},formal_reference='dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv',formal_reference_type='csv')
     else:
      r=e4[(e4.mechanism=='adaptive')&(e4.horizon==f'{h}d')&(e4.seed==seed)].iloc[0]
      base.update(provenance='Exp4',checkpoint=r.checkpoint_path,checkpoint_sha256=r.checkpoint_sha256,references={sp:f'revision_work/outputs/preds/adaptive_dw_h{h}_s{seed}_{sp}.parquet' for sp in COUNTS},formal_reference='dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv',formal_reference_type='csv')
     assert sha(ROOT/base['checkpoint'])==base['checkpoint_sha256'],base
     base['formal_reference_sha256']=sha(ROOT/base['formal_reference'])
     base['reference_sha256']={sp:sha(ROOT/p) for sp,p in base['references'].items()}
     result.append(base)
 assert len(result)==60
 jout(OUT/'backbone_registry.json',result)
 cout(OUT/'backbone_provenance.csv',pd.DataFrame(result).drop(columns=['references','reference_sha256']))
 return result
