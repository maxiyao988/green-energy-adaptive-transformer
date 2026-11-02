"""Import-only trainer adapters: same-patch Fixed and causal formal benchmarks."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
from day_io_v3 import *
# This imported worker sets intra12/inter12 immediately after importing Torch.
import fullrun14a_worker_v2 as base
import torch,numpy as np,pandas as pd,time,types,argparse,math
from contextlib import ExitStack
from scripts.python import panel_baseline_lstm as lstm
from scripts.python import panel_baseline_vanilla_transformer as vanilla
from scripts.python import panel_baseline_swim_cpu as swim
from scripts.python import panel_baseline_ridge as ridge
common,formal,exp3,exp4,core=base.common,base.formal,base.exp3,base.exp4,base.causal_core
DataLoader=torch.utils.data.DataLoader
TensorDataset=torch.utils.data.TensorDataset
FACTORY=formal._make_model
EXP3_FACTORY=exp3._make_experiment3_model
BASE_DEPS=base.dependencies

def deps(inp):
 rows=BASE_DEPS(inp)+extra_dependencies()
 ps=[Path(m.__file__) for m in [lstm,vanilla,swim,swim.formal,ridge]]+[ROOT/'revision_work/outputs/fullrun/preflight.json']
 if inp=='C':ps +=[core.OUT/f'tensors/X_{b}_causal_standardized.npy' for b in ['daily','weekly']]
 rows += [dict(path=rel(p),sha256=sha(p)) for p in ps]
 return list({r['path']:r for r in rows}.values())

def same_factory(mechanism='fixed',daily_patch=None,weekly_patch=None):
 assert mechanism=='fixed' and daily_patch is None and weekly_patch is None
 d,w=formal.DAILY_PATCH_SIZES,formal.WEEKLY_PATCH_SIZES
 try:
  formal.DAILY_PATCH_SIZES=[10]*4;formal.WEEKLY_PATCH_SIZES=[4]*4
  return EXP3_FACTORY('fixed',None,None)
 finally:formal.DAILY_PATCH_SIZES,formal.WEEKLY_PATCH_SIZES=d,w

def causal_check(ds):
 raw=core.load_causal();original=common.load_frozen_panel()
 rows=[]
 for f in ['ticker','anchor_date','split','X_daily','X_weekly']:
  assert np.array_equal(getattr(ds,f),getattr(raw,f)),f
 for f in ['ticker','split']:assert np.array_equal(getattr(original,f),getattr(ds,f)),f
 assert np.array_equal(pd.to_datetime(original.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date).to_numpy()),'anchor_date values'
 for h in [5,10,20]:assert np.array_equal(original.y[h],ds.y[h])
 normal=common.apply_normalizer(ds,common.fit_train_only_normalizer(ds),'daily_weekly')
 for branch,arr in zip(['daily','weekly'],normal):
  ref=np.load(core.OUT/f'tensors/X_{branch}_causal_standardized.npy',mmap_mode='r')
  assert np.array_equal(arr,ref),branch
  rows.append(dict(branch=branch,shape=list(arr.shape),dtype=str(arr.dtype),bitwise_equal_stage3a_and_14a_C_binding=True,array_sha256=__import__('hashlib').sha256(arr.tobytes()).hexdigest()))
 archived=json.loads((ROOT/'revision_work/outputs/fullrun/preflight.json').read_text());assert archived['causal_standardization_bitwise_equal']
 for d in archived['dependencies']:
  assert sha(ROOT/d['path'])==d['sha256'],d
 return dict(passed=True,raw_core_binding_bitwise_equal=True,keys_labels_equal=True,normalized=rows,formal14a_preflight_sha256=sha(ROOT/'revision_work/outputs/fullrun/preflight.json'))

def preflight():
 base.check_profile();exp3._verify_frozen_dependencies();exp4._verify_frozen_dependencies();exp4._verify_frozen_training_constants()
 ds=core.load_causal();formal._formal_check_dataset_contract(ds);assert {s:int((ds.split==s).sum()) for s in COUNTS}==COUNTS
 check=causal_check(ds)
 torch.manual_seed(42);diverse=EXP3_FACTORY('fixed',None,None);same=same_factory();assert formal.count_trainable_parameters(diverse)==formal.count_trainable_parameters(same)==107777
 observed=[];handles=[]
 for branch,patch,T in [('daily',10,90),('weekly',4,26)]:
  b=getattr(same,branch+'_branch');assert b.patch_sizes==[patch]*4 and len(b.experts)==4
  for i,e in enumerate(b.experts):
   original=e.patchify
   def wrapped(self,h,valid_mask=None,original=original,branch=branch,i=i,patch=patch):
    result=original(h,valid_mask);assert self.patch_size==patch
    # patchify returns batch x number_of_patches x patch_size x d_model plus occupancy mask.
    assert result[0].shape[1]==math.ceil(h.shape[1]/patch) and result[0].shape[2]==patch
    observed.append(dict(branch=branch,expert=i+1,patch_size=self.patch_size,input_shape=list(h.shape),patchified_shape=list(result[0].shape)))
    return result
   e.patchify=types.MethodType(wrapped,e)
 same.eval();norm=common.apply_normalizer(ds,common.fit_train_only_normalizer(ds),'daily_weekly')
 with torch.no_grad():prediction,wd,ww=same.forward_with_router(torch.from_numpy(norm[0][:2]),torch.from_numpy(norm[1][:2]))
 assert len(observed)==8 and torch.equal(wd,torch.full((2,4),.25)) and torch.equal(ww,torch.full((2,4),.25))
 assert 'AC Power' in base.power()
 rows=[dict(model='same_patch_fixed',parameters=107777,daily_patches='10,10,10,10',weekly_patches='4,4,4,4'),dict(model='formal_diverse_fixed',parameters=107777,daily_patches='5,10,20,30',weekly_patches='2,4,8,13')]
 csvout(EXP/'pretrain_parameter_counts.csv',rows);csvout(EXP/'pretrain_patch_forward_check.csv',observed)
 textout(EXP/'PRETRAIN_CHECKS.md','# Pre-training checks\n\nBoth same-patch Fixed and formal diverse-patch Fixed have107,777 trainable parameters. Forward hooks on all8 imported expert.patchify calls confirm Daily10 in every Daily expert and Weekly4 in every Weekly expert, with unchanged expert count, d_model, attention/fusion architecture and equal weighting.\n\nC inputs match core.load_causal() and the Stage3a standardized tensors bitwise, the same binding certified in14a preflight. B1 runs first; then one C LSTM Daily+Weekly h5 pilot runs alone. Remaining B4 runs start only after the end-to-end pilot passes. Baseline global-RNG shuffling is preserved; B1/Adaptive use independent generator42.\n')
 jsonout(EXP/'preflight.json',dict(passed=True,utc=utc(),dependencies=deps('C'),parameter_counts=rows,forward_check=observed,causal_input_check=check,thread_counts=[12,12],environment={k:os.environ.get(k) for k in ENV_KEYS},power=base.power()))
 print('PREFLIGHT_PASS',rows,flush=True)

class InferenceView(torch.nn.Module):
 """Output/argument adapter only; uses the imported model's original forward."""
 def __init__(self,model,freq,router):super().__init__();self.inner=model;self.freq=freq;self.router=router
 def forward_with_router(self,xd,xw):
  args=(xd,xw) if self.freq=='daily_weekly' else (xd,)
  if self.router:
   p,w=self.inner.forward_with_router(*args);z=torch.zeros_like(w)
   return (p,w,z) if self.freq=='daily_only' else (p,z,w)
  p=self.inner(*args);z=torch.zeros((len(p),4),dtype=p.dtype,device=p.device);return p,z,z

def inference(model,ds,spec):
 freq,h=spec['frequency_setting'],spec['horizon'];norm=common.apply_normalizer(ds,common.fit_train_only_normalizer(ds),freq)
 outputs={};metrics=[]
 view=InferenceView(model,freq,spec['mechanism']=='adaptive');view.eval()
 for split,n in COUNTS.items():
  mask=ds.split==split;y=torch.from_numpy(ds.y[h][mask]);xs=[torch.from_numpy(x[mask]) for x in norm] if isinstance(norm,tuple) else [torch.from_numpy(norm[mask]),torch.empty((n,0))]
  loader=DataLoader(TensorDataset(*xs,y),batch_size=formal.BATCH_SIZE,shuffle=False)
  p,wd,ww=exp4._batched_inference(view,loader,n,torch.device('cpu'))
  frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ds.y[h][mask],y_pred=p))
  if spec['mechanism']=='adaptive':
   active=wd if freq=='daily_only' else ww;assert np.allclose(active.sum(1),1,atol=1e-6,rtol=0)
   for i,k in enumerate(DWEIGHTS if freq=='daily_only' else WWEIGHTS):frame[k]=active[:,i]
  assert len(frame)==n and not frame.duplicated(['ticker','anchor_date']).any()
  assert np.isfinite(frame.select_dtypes(include='number')).all().all()
  assert np.array_equal(frame.y_true.to_numpy(),ds.y[h][mask]);assert np.array_equal(frame.ticker.to_numpy(),ds.ticker[mask])
  outputs[split]=frame
  metrics.append(dict(split=split,**common.evaluate_predictions(ds.y[h][mask],p,ds.ticker[mask],ds.anchor_date[mask])))
 return outputs,metrics

def run_b4(spec,attempt):
 base.check_profile();start=time.perf_counter();started=utc();ds=core.load_causal();check=causal_check(ds);dependencies=deps('C')
 telemetry=base.Telemetry(attempt/'telemetry.jsonl');telemetry.sample('start',True)
 modelname,freq,h=spec['mechanism'],spec['frequency_setting'],spec['horizon'];models=[];handles=[];val_mse=[];observed_loaders=[]
 phase=dict(last=None,epoch=0,n=0,sse=0.)
 epoch_stream=target(attempt/'epoch_progress.jsonl').open('x')
 stats=common.fit_train_only_normalizer(ds);normal=common.apply_normalizer(ds,stats,freq)
 def watch_model(model):
  models.append(model)
  def hook(module,inputs):
   telemetry.sample('five_minute');state='TRAIN' if module.training else 'VALIDATION_OR_INFERENCE'
   if state!=phase['last']:
    if state=='TRAIN':phase['epoch']+=1
    phase['last']=state;epoch_stream.write(json.dumps(dict(utc=utc(),epoch=phase['epoch'],phase=state))+chr(10));epoch_stream.flush()
  handles.append(model.register_forward_pre_hook(hook));return model
 def capture(original):
  def call(*a,**kw):return watch_model(original(*a,**kw))
  return call
 original_huber=torch.nn.HuberLoss
 def huber(*a,**kw):
  criterion=original_huber(*a,**kw)
  def observe(module,inputs,output):
   if not models[0].training:
    p,y=inputs;phase['sse']+=float(((p.detach()-y.detach())**2).sum(dtype=torch.float64));phase['n']+=len(y)
    if phase['n']==COUNTS['val']:
     val_mse.append(phase['sse']/phase['n']);phase.update(n=0,sse=0.)
    assert phase['n']<COUNTS['val']
  handles.append(criterion.register_forward_hook(observe));return criterion
 def checked_loader(dataset,*a,**kw):
  loader=DataLoader(dataset,*a,**kw)
  if len(dataset) in COUNTS.values():
   split=next(s for s,n in COUNTS.items() if n==len(dataset));mask=ds.split==split
   expected=[x[mask] for x in normal] if isinstance(normal,tuple) else [normal[mask]]
   tensors=dataset.tensors;assert len(tensors)==len(expected)+1
   for actual,ref in zip(tensors[:-1],expected):assert np.array_equal(actual.numpy(),ref)
   assert np.array_equal(tensors[-1].numpy(),ds.y[h][mask]);assert loader.generator is None
   observed_loaders.append(dict(split=split,rows=len(dataset),bitwise_equal_causal_features=True,generator=None,shuffle=bool(kw.get('shuffle',False))))
  return loader
 def bind(stack,obj,key,value):
  old=getattr(obj,key);setattr(obj,key,value);stack.callback(setattr,obj,key,old)
 checkpoint=attempt/('best_model.pkl' if modelname=='ridge' else 'best_state.pt')
 if modelname=='ridge':
  original_ridge=ridge.Ridge;candidates=[]
  def capture_ridge(*a,**kw):
   model=original_ridge(*a,**kw);candidates.append(model);return model
  with ExitStack() as stack:
   bind(stack,ridge,'Ridge',capture_ridge);summary,archived_test=ridge.run_one(ds,stats,freq,h)
  assert len(candidates)==4 and [m.alpha for m in candidates]==[.1,1.,10.,100.]
  best=next(m for m in candidates if m.alpha==summary['best_alpha']);X=common.flatten_features(freq,normal);ys=ds.y[h];history=[]
  for m in candidates:
   row=dict(alpha=float(m.alpha),history_kind='alpha candidate; no neural epochs',epoch=None)
   for split in ['train','val']:
    mask=ds.split==split;p=m.predict(X[mask]);delta=p-ys[mask]
    row[split+'_huber']=float(torch.nn.functional.huber_loss(torch.from_numpy(p),torch.from_numpy(ys[mask]),delta=1))
    if split=='val':row['val_mse']=float(np.mean(delta**2))
   history.append(row)
  with target(checkpoint).open('xb') as f:pickle.dump(best,f,protocol=5)
  with checkpoint.open('rb') as f:replay=pickle.load(f)
  assert np.array_equal(best.coef_,replay.coef_) and best.intercept_==replay.intercept_
  frames={};metrics=[]
  for split,n in COUNTS.items():
   mask=ds.split==split;p=replay.predict(X[mask]);frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ys[mask],y_pred=p));frames[split]=frame
   metrics.append(dict(split=split,**common.evaluate_predictions(ys[mask],p,ds.ticker[mask],ds.anchor_date[mask])))
  reference=archived_test.y_pred.to_numpy();best_epoch=epochs=0;runtime=0.;entry='panel_baseline_ridge.run_one; unchanged four-alpha validation-MSE selection';nparams=int(replay.coef_.size+1)
  check['actual_trainer_input']='Imported run_one uses common.apply_normalizer/flatten_features on checked causal dataset'
 else:
  owner=formal if modelname=='adaptive' else {'lstm':lstm,'vanilla':vanilla,'swim':swim}[modelname]
  with ExitStack() as stack:
   bind(stack,torch.nn,'HuberLoss',huber)
   if modelname=='adaptive':
    bind(stack,formal,'load_frozen_panel',lambda:ds);bind(stack,formal,'_make_model',capture(FACTORY))
    result=formal.train_one_cell(freq,h,'adaptive');summary=result['summary'].iloc[0].to_dict();reference=result['predictions'].y_pred.to_numpy();history=pd.DataFrame(result['history']);entry='main_v2.train_one_cell'
   else:
    # CPU-only backend selection; original trainer loop and its global-RNG DataLoader are unchanged.
    bind(stack,torch.backends.mps,'is_available',lambda:False);bind(stack,torch.cuda,'is_available',lambda:False)
    if modelname=='lstm':
     for name in ['LSTMRegressor','DualFrequencyLSTM']:bind(stack,lstm,name,capture(getattr(lstm,name)))
     bind(stack,torch.utils.data,'DataLoader',checked_loader)
    elif modelname=='vanilla':
     for name in ['SingleFrequencyTransformerRegressor','DualFrequencyTransformerRegressor']:bind(stack,vanilla,name,capture(getattr(vanilla,name)))
     bind(stack,vanilla,'DataLoader',checked_loader)
    else:
     for name in ['SingleFrequencySWiMRegressor','DualFrequencySWiMRegressor']:bind(stack,swim.formal,name,capture(getattr(swim.formal,name)))
     bind(stack,swim,'DataLoader',checked_loader)
    function=owner.train_and_eval_frequency_cpu if modelname=='swim' else owner.train_and_eval_frequency
    result=function(ds,freq,h,stats);summary={k:v for k,v in result.items() if not isinstance(v,(np.ndarray,pd.DataFrame,list,dict))};reference=result['y_pred'];history=pd.DataFrame(result['history']);entry=owner.__name__+'.'+function.__name__
  for handle in handles:handle.remove()
  assert len(models)==1;model=models[0];assert all(p.device.type=='cpu' for p in model.parameters())
  epochs=int(summary['epochs_trained']);best_epoch=int(summary['best_epoch']);runtime=float(summary['runtime_seconds']);assert len(history)==len(val_mse)==epochs and phase['n']==0
  history['train_huber']=history.train_loss;history['val_huber']=history.val_loss;history['val_mse']=val_mse;history['loss_type']='Huber delta1'
  state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  with target(checkpoint).open('xb') as f:torch.save(state,f)
  replay=FACTORY(freq,'adaptive') if modelname=='adaptive' else (lstm.make_model(freq,5,5) if modelname=='lstm' else vanilla.make_model(freq,5,5) if modelname=='vanilla' else swim.formal.make_model(freq,5,5))
  replay.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);replay.eval();frames,metrics=inference(replay,ds,spec);nparams=formal.count_trainable_parameters(replay)
  if modelname!='adaptive':
   assert {r['split'] for r in observed_loaders}==set(COUNTS);check['actual_trainer_loaders']=observed_loaders
  else:check['actual_trainer_input']='Imported main_v2 loaders normalize this exact checked causal dataset; independent generator42'
 delta=float(np.max(np.abs(frames['test'].y_pred.to_numpy(np.float64)-np.asarray(reference,dtype=np.float64))));assert delta<1e-7,delta
 comparison=dict(classification='EXACT',threshold=1e-7,max_abs_delta_pred=delta,comparison='strict restored-checkpoint replay vs same-run formal trainer TEST')
 for split,frame in frames.items():
  mask=ds.split==split;assert len(frame)==COUNTS[split] and not frame.duplicated(['ticker','anchor_date']).any()
  assert np.array_equal(frame.y_true.to_numpy(),ds.y[h][mask]) and np.array_equal(frame.ticker.to_numpy(),ds.ticker[mask])
  assert np.array_equal(pd.to_datetime(frame.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy())
  assert np.isfinite(frame.select_dtypes(include='number')).all().all()
  p=attempt/f'{split}.parquet'
  with target(p).open('xb') as f:f.write(frame.to_parquet(index=False,engine='pyarrow'))
  pd.testing.assert_frame_equal(frame,pd.read_parquet(p))
 with target(attempt/'history.csv').open('x',newline='') as f:pd.DataFrame(history).to_csv(f,index=False)
 with target(attempt/'trainer_summary.csv').open('x',newline='') as f:pd.DataFrame([summary]).to_csv(f,index=False)
 with target(attempt/'metrics.csv').open('x',newline='') as f:pd.DataFrame(metrics).to_csv(f,index=False)
 jsonout(attempt/'causal_input_check.json',check);jsonout(attempt/'test_replay.json',comparison)
 telemetry.sample('end',True);telemetry.handle.close();epoch_stream.close();assert all(r['thread_counts']==[12,12] for r in telemetry.rows)
 for d in dependencies:assert sha(ROOT/d['path'])==d['sha256']
 record=dict(**spec,validated=True,pid=os.getpid(),started_utc=started,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,epochs=epochs,best_epoch=best_epoch,formal_training_seconds=runtime,parameter_count=nparams,training_entrypoint=entry,loader_generator_seed=42 if modelname=='adaptive' else None,shuffle_protocol='independent generator42' if modelname=='adaptive' else 'formal baseline global RNG seed42' if modelname!='ridge' else 'not applicable',row_counts=COUNTS,checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),thread_counts_start=telemetry.rows[0]['thread_counts'],thread_counts_end=telemetry.rows[-1]['thread_counts'],thread_counts_changed=False,environment_start=telemetry.rows[0]['environment'],environment_end=telemetry.rows[-1]['environment'],load_average_start=telemetry.rows[0]['load_average_1_5_15'],load_average_end=telemetry.rows[-1]['load_average_1_5_15'],telemetry_samples=len(telemetry.rows),metrics=metrics,test_replay=comparison,causal_input_check=check,q0=None,dependencies=dependencies,artifacts=[artifact(p) for p in sorted(attempt.iterdir()) if p.is_file() and p.name!='worker.log'])
 jsonout(attempt/'COMPLETE.json',record);verify_completion(attempt/'COMPLETE.json',spec);print('COMPLETE',spec['job_id'],record['wall_clock_seconds'],flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--job-id');ap.add_argument('--attempt');a=ap.parse_args()
 if a.preflight:return preflight()
 spec=spec_for(a.job_id);attempt=Path(a.attempt).absolute();assert attempt.is_relative_to(EXP/'runs'/spec['job_id'])
 try:
  if spec['block']=='B1':
   base.OUT=EXP;base.dependencies=deps;base.WEIGHTS=SAME_WEIGHTS;base.verify_completion=verify_completion;exp3._make_experiment3_model=same_factory
   base.run(spec,attempt)
  else:run_b4(spec,attempt)
 except BaseException as e:
  if not (attempt/'FAILED.json').exists():jsonout(attempt/'FAILED.json',dict(utc=utc(),job_id=spec['job_id'],error=repr(e)))
  raise
if __name__=='__main__':main()
