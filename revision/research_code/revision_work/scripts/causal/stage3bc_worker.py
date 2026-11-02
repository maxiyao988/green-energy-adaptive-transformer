"""Formal Experiment4 multiseed causal runs and formal seed42 as-of-level runs."""
import os,sys,time,argparse,types,subprocess
sys.dont_write_bytecode=True
os.environ.update(OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
from core import *
from torch.utils.data import DataLoader

class Telemetry:
 def __init__(self,prefix,stem):
  self.file=target(OUT/f'{prefix}/telemetry/{stem}.jsonl').open('x');self.deadline=0;self.changed=False;self.count=0
 def sample(self,event,force=False):
  now=time.perf_counter()
  if not force and now<self.deadline:return
  intra=torch.get_num_threads();inter=torch.get_num_interop_threads();self.changed|=(intra!=3 or inter!=1);rows=[]
  for line in subprocess.run(['ps','-axo','pid,ppid,pcpu,comm'],capture_output=True,text=True,check=True).stdout.splitlines()[1:]:
   v=line.strip().split(None,3)
   if len(v)==4:rows.append(dict(pid=int(v[0]),ppid=int(v[1]),cpu_pct=float(v[2]),command=v[3]))
  data=dict(utc=utc(),event=event,pid=os.getpid(),torch_num_threads=intra,torch_num_interop_threads=inter,environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC']},load_average_1_5_15=list(os.getloadavg()),top_cpu_processes=sorted(rows,key=lambda r:r['cpu_pct'],reverse=True)[:20]);self.file.write(json.dumps(data)+'\n');self.file.flush();self.deadline=now+300;self.count+=1;print('TELEMETRY',data['utc'],event,intra,inter,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['stage3b','stage3c']);p.add_argument('--mechanism',required=True,choices=['static','adaptive']);p.add_argument('--horizon',required=True,type=int,choices=[5,10,20]);p.add_argument('--seed',required=True,type=int);a=p.parse_args();stage=a.stage;mech=a.mechanism;h=a.horizon;seed=a.seed
 assert (stage=='stage3b' and seed in [0,1,21,3407]) or (stage=='stage3c' and seed==42)
 env=threads(3);env['thread_environment']={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC']};exp3._verify_frozen_dependencies();exp4._verify_frozen_dependencies()
 if stage=='stage3b':ds=load_causal()
 else:
  frozen=common.load_frozen_panel();ds=common.PanelDataset(frozen.ticker,frozen.anchor_date,frozen.split,np.load(OUT/'stage3c/tensors/X_daily_asof_raw.npy'),np.load(OUT/'stage3c/tensors/X_weekly_asof_raw.npy'),frozen.y)
 formal._formal_check_dataset_contract(ds);stem=f'{mech}_dw_h{h}_s{seed}';checkpoint=OUT/f'{stage}/checkpoints/{stem}.pt';target(checkpoint);target(OUT/f'{stage}/runs/{stem}.json');start=time.perf_counter();stamp=utc();telemetry=Telemetry(stage,stem);telemetry.sample('start',True)
 owner=exp4 if stage=='stage3b' else (exp3 if mech=='static' else formal);factory_name='_make_experiment4_model' if stage=='stage3b' else ('_make_experiment3_model' if mech=='static' else '_make_model');factory=getattr(owner,factory_name);oldload=owner.load_frozen_panel;models=[];handles=[];phase={'last':None,'epoch':0};val={'n':0,'sse':0.};valmse=[]
 def capture(*args,**kwargs):
  model=factory(*args,**kwargs);models.append(model)
  def progress(module,inputs):
   telemetry.sample('five_minute');status='TRAIN' if module.training else 'VALIDATION'
   if status!=phase['last']:
    if status=='TRAIN':phase['epoch']+=1
    phase['last']=status;print('PHASE',utc(),stage,stem,phase['epoch'],status,flush=True)
  handles.append(model.register_forward_pre_hook(progress));return model
 original_huber=torch.nn.HuberLoss;original_save=torch.save
 def observed_huber(*args,**kwargs):
  criterion=original_huber(*args,**kwargs)
  def observe(module,inputs,output):
   if not models[0].training:
    pred,y=inputs;diff=pred.detach()-y.detach();val['sse']+=float((diff*diff).sum(dtype=torch.float64));val['n']+=len(y)
    if val['n']==COUNTS['val']:valmse.append(val['sse']/val['n']);val.update(n=0,sse=0.);print('VAL_MSE',utc(),stem,len(valmse),valmse[-1],flush=True)
    elif val['n']>COUNTS['val']:raise RuntimeError('Validation rows exceed frozen count')
  handles.append(criterion.register_forward_hook(observe));return criterion
 setattr(owner,factory_name,capture);owner.load_frozen_panel=lambda:ds;torch.nn.HuberLoss=observed_huber
 saves_checkpoint=stage=='stage3b' or mech=='static';oldos=owner.os if saves_checkpoint else None;oldpath=owner._checkpoint_path if saves_checkpoint else None;old_exp4dir=exp4.EXPERIMENT_DIR
 if saves_checkpoint:
  owner._checkpoint_path=lambda *args,**kwargs:checkpoint
  if stage=='stage3b':exp4.EXPERIMENT_DIR=OUT/stage
  def save_exclusive(obj,path,*args,**kwargs):
   with target(Path(path)).open('xb') as f:original_save(obj,f,*args,**kwargs)
  def publish_exclusive(src,dst):
   with Path(src).open('rb') as source,target(Path(dst)).open('xb') as destination:
    for b in iter(lambda:source.read(1024*1024),b''):destination.write(b)
  torch.save=save_exclusive;owner.os=types.SimpleNamespace(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')});owner.os.replace=publish_exclusive
 print('START',utc(),stage,stem,env,flush=True)
 try:
  if stage=='stage3b':result=exp4.train_and_validate_cell(mech,f'{h}d',seed)
  elif mech=='static':result=exp3.train_and_validate_cell('static',f'{h}d',None,None)
  else:result=formal.train_one_cell('daily_weekly',h,'adaptive')
 finally:
  setattr(owner,factory_name,factory);owner.load_frozen_panel=oldload;torch.nn.HuberLoss=original_huber;torch.save=original_save;exp4.EXPERIMENT_DIR=old_exp4dir
  if saves_checkpoint:owner.os=oldos;owner._checkpoint_path=oldpath
  for handle in handles:handle.remove()
 assert len(models)==1;model=models[0];model.eval();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 if saves_checkpoint:
  saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert all(torch.equal(saved[k],v) for k,v in state.items())
 else:
  with target(checkpoint).open('xb') as f:original_save(state,f)
 history=pd.DataFrame(result['history']);assert len(history)==len(valmse);history['train_huber']=history.train_loss;history['val_huber']=history.val_loss;history['val_mse']=valmse;history['loss_type']='Huber delta=1';csvout(f'{stage}/history/{stem}.csv',history);summary=result['summary'].iloc[0].to_dict() if isinstance(result['summary'],pd.DataFrame) else result['summary'];csvout(f'{stage}/runs/{stem}_trainer_summary.csv',[summary])
 if mech=='static':jsonout(f'{stage}/weights/{stem}_logits.json',{k:v.tolist() for k,v in state.items() if 'logit' in k})
 # Replay the serialized checkpoint into a fresh imported model, exactly as Task B.
 replay=exp4._make_experiment4_model(mech) if stage=='stage3b' else (exp3._make_experiment3_model('static',None,None) if mech=='static' else formal._make_model('daily_weekly','adaptive'));replay.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);replay.eval();train,validation,test=formal._build_training_loaders('daily_weekly',h,ds);assert train.generator.initial_seed()==42;met=[];files=[]
 for split,loader in [('train',DataLoader(train.dataset,batch_size=formal.BATCH_SIZE,shuffle=False)),('val',validation),('test',test)]:
  mask=ds.split==split;count=COUNTS[split]
  if stage=='stage3b':pred,dw,ww=exp4._batched_inference(replay,loader,count,torch.device('cpu'))
  else:pred,dw,ww=exp3._batched_inference(replay,loader,mech,None,None,count,torch.device('cpu'))
  frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ds.y[h][mask],y_pred=pred))
  for j,c in enumerate(WEIGHTS):frame[c]=(dw if j<4 else ww)[:,j%4]
  assert len(frame)==count and not frame.duplicated(['ticker','anchor_date']).any() and np.isfinite(frame[['y_true','y_pred']+WEIGHTS]).all().all();assert np.allclose(dw.sum(1),1,atol=1e-6) and np.allclose(ww.sum(1),1,atol=1e-6)
  if mech=='static':assert np.ptp(frame[WEIGHTS].to_numpy(),axis=0).max()<1e-7
  path=f'{stage}/preds/{stem}_{split}.parquet';parquet(path,frame);pd.testing.assert_frame_equal(frame,pd.read_parquet(OUT/path));files.append(dict(path=rel(OUT/path),n_rows=count,split=split,sha256=sha(OUT/path)));met.append(dict(feature_set='causal_relative' if stage=='stage3b' else 'asof_levels',mechanism=mech,horizon=h,seed=seed,split=split,**metrics(frame.y_true.to_numpy(),pred,frame.ticker.to_numpy(),frame.anchor_date.to_numpy())))
 csvout(f'{stage}/runs/{stem}_metrics.csv',met);csvout(f'{stage}/runs/{stem}_files.csv',files);telemetry.sample('end',True);telemetry.file.close();jsonout(f'{stage}/runs/{stem}.json',dict(stage=stage,feature_set='causal_relative' if stage=='stage3b' else 'asof_levels',mechanism=mech,horizon=h,seed=seed,started_utc=stamp,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,environment=env,best_epoch=int(summary['best_epoch']),stop_epoch=int(summary['epochs_trained']),checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),logged_thread_counts_changed=telemetry.changed,telemetry_samples=telemetry.count,training_entrypoint='experiment4.train_and_validate_cell' if stage=='stage3b' else ('experiment3.train_and_validate_cell' if mech=='static' else 'main_v2.train_one_cell')));print('COMPLETE',utc(),stage,stem,flush=True)
if __name__=='__main__':main()
