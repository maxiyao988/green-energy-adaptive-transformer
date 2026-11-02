"""B3 orchestration; complete model and training loops are imported unchanged."""
from b3_io import *
from b3_data import base,common,formal,core,load_panel,contract,fit_bound_stats,apply_bound_stats
from b3_models import make_model,FrequencyInterface
from b3_contracts import dependencies,verify_completion
import torch,numpy as np,pandas as pd,time,types,argparse
from contextlib import ExitStack
exp3,exp4=base.exp3,base.exp4

def run(spec,attempt):
 base.check_profile();started=utc();clock=time.perf_counter();deps=dependencies();ds=load_panel(spec['input_set']);contract(ds)
 freq,h,mechanism,seed=spec['frequency_setting'],spec['horizon'],spec['mechanism'],spec['seed']
 main_path=seed==42 and mechanism=='adaptive';owner=formal if main_path else exp3 if seed==42 else exp4
 dual_interface=not main_path
 factory_name='_make_model' if owner is formal else '_make_experiment3_model' if owner is exp3 else '_make_experiment4_model'
 checkpoint=attempt/'best_state.pt';telemetry=base.Telemetry(attempt/'telemetry.jsonl');telemetry.sample('start',True)
 models=[];handles=[];val_mse=[];phase=dict(last=None,epoch=0,n=0,sse=0.)
 epoch_stream=target(attempt/'epoch_progress.jsonl').open('x')
 original_huber=torch.nn.HuberLoss;original_save=torch.save
 def capture(*args,**kwargs):
  model=make_model(freq,mechanism,dual_interface=dual_interface);models.append(model)
  def progress(module,inputs):
   telemetry.sample('five_minute');label='TRAIN' if module.training else 'VALIDATION'
   if label!=phase['last']:
    if label=='TRAIN':phase['epoch']+=1
    phase['last']=label;r=dict(utc=utc(),epoch=phase['epoch'],phase=label);epoch_stream.write(json.dumps(r)+'\n');epoch_stream.flush();print('PHASE',spec['job_id'],r,flush=True)
  handles.append(model.register_forward_pre_hook(progress));return model
 def huber(*a,**kw):
  criterion=original_huber(*a,**kw)
  def observe(module,inputs,output):
   if not models[0].training:
    p,y=inputs;phase['sse']+=float(((p.detach()-y.detach())**2).sum(dtype=torch.float64));phase['n']+=len(y)
    if phase['n']==COUNTS['val']:val_mse.append(phase['sse']/phase['n']);phase.update(n=0,sse=0.)
    assert phase['n']<COUNTS['val']
  handles.append(criterion.register_forward_hook(observe));return criterion
 def bind(stack,obj,key,value):
  before=getattr(obj,key);setattr(obj,key,value);stack.callback(setattr,obj,key,before)
 def save_exclusive(obj,path,*args,**kwargs):
  with target(Path(path)).open('xb') as f:original_save(obj,f,*args,**kwargs)
 def publish_exclusive(source,destination):
  with Path(source).open('rb') as inp,target(Path(destination)).open('xb') as out:
   for b in iter(lambda:inp.read(1024*1024),b''):out.write(b)
 # The unchanged formal loaders call these imported binding names. Their source,
 # batch order, generator42 and complete training/selection loops are retained.
 with ExitStack() as stack:
  bind(stack,formal,'_formal_check_dataset_contract',contract)
  for obj in [formal,exp3]:
   bind(stack,obj,'fit_train_only_normalizer',fit_bound_stats);bind(stack,obj,'apply_normalizer',apply_bound_stats)
  bind(stack,owner,'load_frozen_panel',lambda:ds);bind(stack,owner,factory_name,capture);bind(stack,torch.nn,'HuberLoss',huber)
  if owner is not formal:
   bind(stack,owner,'_checkpoint_path',lambda *a,**kw:checkpoint)
   bind(stack,exp4,'EXPERIMENT_DIR',attempt);bind(stack,torch,'save',save_exclusive)
   osview=types.SimpleNamespace(**{k:getattr(os,k) for k in dir(os) if not k.startswith('__')});osview.replace=publish_exclusive;bind(stack,owner,'os',osview)
  if owner is formal:result=formal.train_one_cell(freq,h,'adaptive');entry='main_v2.train_one_cell (formal single/dual-resolution path)'
  elif owner is exp3:result=exp3.train_and_validate_cell(mechanism,f'{h}d',None,None);entry='experiment3.train_and_validate_cell (formal factory bound to requested resolution)'
  else:result=exp4.train_and_validate_cell(mechanism,f'{h}d',seed);entry='experiment4.train_and_validate_cell (formal factory bound to requested resolution)'
 for handle in handles:handle.remove()
 epoch_stream.close();assert len(models)==1 and phase['n']==0
 model=models[0];summary=result['summary'].iloc[0].to_dict() if isinstance(result['summary'],pd.DataFrame) else result['summary'];history=pd.DataFrame(result['history'])
 assert len(history)==len(val_mse)==summary['epochs_trained'] and history.epoch.tolist()==list(range(1,len(history)+1))
 history['train_huber']=history.train_loss;history['val_huber']=history.val_loss;history['val_mse']=val_mse;history['loss_type']='Huber delta1'
 assert np.isfinite(history[['train_huber','val_huber','val_mse']]).all().all()
 state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 if owner is formal:
  with target(checkpoint).open('xb') as f:torch.save(state,f)
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved.keys()==state.keys() and all(torch.equal(saved[k],state[k]) for k in state)
 replay=make_model(freq,mechanism,dual_interface=dual_interface);replay.load_state_dict(saved,strict=True);replay.eval()
 view=FrequencyInterface(replay,freq) if main_path and freq!='daily_weekly' else replay
 for name,data in [('history.csv',history),('trainer_summary.csv',pd.DataFrame([summary]))]:
  with target(attempt/name).open('x',newline='') as f:data.to_csv(f,index=False)
 if mechanism=='static':
  logits={k:v.tolist() for k,v in saved.items() if 'logit' in k};assert len(logits)==(2 if freq=='daily_weekly' else 1);jsonout(attempt/'static_logits.json',logits)
 with ExitStack() as stack:
  bind(stack,formal,'fit_train_only_normalizer',fit_bound_stats);bind(stack,formal,'apply_normalizer',apply_bound_stats)
  train,val,test=formal._build_training_loaders('daily_weekly',h,ds)
 assert train.generator.initial_seed()==42
 loaders={'train':torch.utils.data.DataLoader(train.dataset,batch_size=formal.BATCH_SIZE,shuffle=False),'val':val,'test':test};metrics=[]
 for split,loader in loaders.items():
  mask=ds.split==split;p,wd,ww=exp4._batched_inference(view,loader,COUNTS[split],torch.device('cpu'))
  frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ds.y[h][mask],y_pred=p))
  for active,weights,columns in [(freq!='weekly_only',wd,WEIGHTS[:4]),(freq!='daily_only',ww,WEIGHTS[4:])]:
   if not active:assert not weights.any();continue
   assert np.allclose(weights.sum(1),1,rtol=0,atol=1e-6)
   if mechanism=='static':assert np.ptp(weights,axis=0).max()<1e-7
   for k,v in zip(columns,weights.T):frame[k]=v
  assert len(frame)==COUNTS[split] and not frame.duplicated(['ticker','anchor_date']).any()
  assert np.array_equal(frame.y_true.to_numpy(),ds.y[h][mask]) and np.array_equal(frame.ticker.to_numpy(),ds.ticker[mask])
  assert np.array_equal(pd.to_datetime(frame.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy())
  assert np.isfinite(frame.select_dtypes(include='number')).all().all()
  with target(attempt/f'{split}.parquet').open('xb') as f:f.write(frame.to_parquet(index=False,engine='pyarrow'))
  pd.testing.assert_frame_equal(frame,pd.read_parquet(attempt/f'{split}.parquet'))
  metrics.append(dict(split=split,**common.evaluate_predictions(frame.y_true.to_numpy(),p,ds.ticker[mask],ds.anchor_date[mask])))
 with target(attempt/'metrics.csv').open('x',newline='') as f:pd.DataFrame(metrics).to_csv(f,index=False)
 telemetry.sample('end',True);telemetry.handle.close();assert all(x['thread_counts']==[12,12] for x in telemetry.rows)
 for d in deps:assert sha(ROOT/d['path'])==d['sha256'],d['path']
 r=dict(**spec,validated=True,started_utc=started,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-clock,pid=os.getpid(),training_entrypoint=entry,loader_generator_seed=42,epochs=int(summary['epochs_trained']),best_epoch=int(summary['best_epoch']),formal_training_seconds=float(summary['runtime_seconds']),parameter_count=formal.count_trainable_parameters(replay),row_counts=COUNTS,checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),strict_checkpoint_reload=True,mask_transport_checked=True,causal_reference_policy=json.loads((EXP/'causal_reference_policy.json').read_text())['policy'],thread_counts_start=telemetry.rows[0]['thread_counts'],thread_counts_end=telemetry.rows[-1]['thread_counts'],thread_counts_changed=False,environment_start=telemetry.rows[0]['environment'],environment_end=telemetry.rows[-1]['environment'],load_average_start=telemetry.rows[0]['load_average_1_5_15'],load_average_end=telemetry.rows[-1]['load_average_1_5_15'],power_start=telemetry.rows[0]['power'],power_end=telemetry.rows[-1]['power'],telemetry_samples=len(telemetry.rows),q0=None,metrics=metrics,dependencies=deps,artifacts=[artifact(p) for p in sorted(attempt.iterdir()) if p.is_file() and p.name!='worker.log'])
 jsonout(attempt/'COMPLETE.json',r);verify_completion(attempt/'COMPLETE.json',spec);print('B3_COMPLETE',spec['job_id'],r['wall_clock_seconds'],flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--job-id');ap.add_argument('--attempt');a=ap.parse_args()
 if a.preflight:assert json.loads((EXP/'preflight.json').read_text())['passed'];return
 spec=spec_for(a.job_id);attempt=Path(a.attempt).absolute();assert attempt.is_relative_to(EXP/'runs'/spec['job_id'])
 try:run(spec,attempt)
 except BaseException as e:
  if not (attempt/'FAILED.json').exists():jsonout(attempt/'FAILED.json',dict(utc=utc(),job_id=spec['job_id'],error=repr(e)))
  raise
if __name__=='__main__':main()
