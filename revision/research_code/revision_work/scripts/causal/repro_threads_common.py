"""Shared read-only references and exclusive I/O for the 12/12 reproducibility test."""
import os,sys
sys.dont_write_bytecode=True
os.environ.pop('OMP_NUM_THREADS',None)
os.environ['OMP_DYNAMIC']='FALSE';os.environ['MKL_DYNAMIC']='FALSE'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'revision_work/scripts'))
import step3_common as prior
prior.OUT=ROOT/'revision_work/outputs/repro_threads'
from step3_common import *
from scripts.python import panel_pathformer_main_v2 as formal
from scripts.python import panel_pathformer_experiment3 as exp3
from scripts.python import panel_pathformer_experiment4_multiseed as exp4

def configure():
 before=dict(intra=torch.get_num_threads(),interop=torch.get_num_interop_threads())
 torch.set_num_threads(12);torch.set_num_interop_threads(12)
 assert 'OMP_NUM_THREADS' not in os.environ
 return dict(default_threads_before_explicit_setting=before,torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC']},parallel_info=torch.__config__.parallel_info(),python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),device='cpu',pid=os.getpid())
def csvout(name,d):
 d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)
 write_csv(OUT/name,d);return d

def get_history(mechanism,h):
 p=MAIN/'panel_pathformer_training_history.csv' if mechanism=='adaptive' else E3/'experiment3_training_history.csv'
 d=pd.read_csv(p,float_precision='round_trip');mask=d.horizon.map(horizon).eq(h)&d.seed.eq(42)
 if 'frequency_setting' in d:mask &= d.frequency_setting.eq('daily_weekly')
 if 'mechanism' in d:mask &= d.mechanism.eq(mechanism)
 return d[mask].sort_values('epoch').copy(),p

def history_check(stem,mechanism,h,new):
 old,path=get_history(mechanism,h);n=pd.read_csv(OUT/f'runs/{stem}_history.csv',float_precision='round_trip')
 a=n[['epoch','train_loss','val_loss']].rename(columns={'train_loss':'new_train_loss','val_loss':'new_val_loss'});b=old[['epoch','train_loss','val_loss']].rename(columns={'train_loss':'formal_train_loss','val_loss':'formal_val_loss'});m=a.merge(b,on='epoch',how='outer',validate='one_to_one',indicator=True).sort_values('epoch')
 m['train_abs_delta']=abs(m.new_train_loss-m.formal_train_loss);m['val_abs_delta']=abs(m.new_val_loss-m.formal_val_loss);m['train_exact_equal']=m.new_train_loss==m.formal_train_loss;m['val_exact_equal']=m.new_val_loss==m.formal_val_loss;m['both_exact_equal']=m.train_exact_equal&m.val_exact_equal;m['both_within_1e_12']=(m.train_abs_delta<1e-12)&(m.val_abs_delta<1e-12)
 csvout(f'comparisons/{stem}_history_by_epoch.csv',m)
 return dict(formal_history=rel(path),new_history_rows=len(n),formal_history_rows=len(old),history_rows_equal=len(n)==len(old),history_exact_equal=bool(m.both_exact_equal.all()),history_within_1e_12=bool(m.both_within_1e_12.all()),history_max_train_abs_delta=float(m.train_abs_delta.max()),history_max_val_abs_delta=float(m.val_abs_delta.max()),first_differing_epoch=int(m.loc[~m.both_exact_equal,'epoch'].iloc[0]) if (~m.both_exact_equal).any() else None)

def export_new_checkpoint(stem,mechanism,h,checkpoint):
 ds=formal.load_frozen_panel();formal._formal_check_dataset_contract(ds)
 model=exp3._make_experiment3_model('static',None,None) if mechanism=='static' else exp4._make_experiment4_model('adaptive')
 model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.to(formal._resolve_device());model.eval()
 from torch.utils.data import DataLoader
 train,val,test=formal._build_training_loaders('daily_weekly',h,ds,batch_size=formal.BATCH_SIZE)
 assert train.generator.initial_seed()==42
 loaders={'train':DataLoader(train.dataset,batch_size=formal.BATCH_SIZE,shuffle=False),'val':val,'test':test};files=[];frames={}
 for split,loader in loaders.items():
  count=COUNTS[split];mask=ds.split==split;assert len(loader.dataset)==count
  if mechanism=='static':p,dw,ww=exp3._batched_inference(model,loader,'static',None,None,count,formal._resolve_device())
  else:p,dw,ww=exp4._batched_inference(model,loader,count,formal._resolve_device())
  d=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ds.y[h][mask],y_pred=p))
  for i,col in enumerate(WEIGHTS):d[col]=(dw if i<4 else ww)[:,i%4]
  assert np.allclose(dw.sum(1),1,rtol=0,atol=1e-6) and np.allclose(ww.sum(1),1,rtol=0,atol=1e-6)
  if mechanism=='static':assert np.ptp(d[WEIGHTS].to_numpy(),axis=0).max()<1e-7
  assert len(d)==count and not d.duplicated(['ticker','anchor_date']).any() and np.isfinite(d[['y_true','y_pred']+WEIGHTS]).all().all()
  path=OUT/f'preds/{stem}_{split}.parquet';write_parquet(path,d);check=pd.read_parquet(path);pd.testing.assert_frame_equal(d,check)
  files.append(dict(mechanism=mechanism,horizon=h,seed=42,split=split,n_rows=count,path=rel(path),sha256=sha(path),checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint)));frames[split]=check
 csvout(f'runs/{stem}_prediction_files.csv',files)
 return ds,loaders,frames

class Telemetry:
 def __init__(self,stem):
  import time
  self.time=time;self.deadline=time.perf_counter();self.file=target(OUT/f'telemetry/{stem}.jsonl').open('x');self.count=0;self.changed=False
 def sample(self,event,force=False):
  import subprocess
  now=self.time.perf_counter()
  if not force and now<self.deadline:return
  intra=torch.get_num_threads();inter=torch.get_num_interop_threads();self.changed |= intra!=12 or inter!=12
  result=subprocess.run(['ps','-axo','pid,ppid,pcpu,comm'],capture_output=True,text=True,check=True)
  processes=[]
  for line in result.stdout.splitlines()[1:]:
   fields=line.strip().split(None,3)
   if len(fields)==4:processes.append(dict(pid=int(fields[0]),ppid=int(fields[1]),cpu_pct=float(fields[2]),command=fields[3]))
  record=dict(utc=utc(),event=event,training_pid=os.getpid(),torch_num_threads=intra,torch_num_interop_threads=inter,environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC']},load_average_1_5_15=list(os.getloadavg()),top_cpu_processes=sorted(processes,key=lambda p:p['cpu_pct'],reverse=True)[:20])
  self.file.write(json.dumps(record)+'\n');self.file.flush();self.count+=1;self.deadline=now+300
  print('TELEMETRY',record['utc'],event,'threads',intra,inter,'load',record['load_average_1_5_15'],flush=True)
 def close(self):self.file.close()
