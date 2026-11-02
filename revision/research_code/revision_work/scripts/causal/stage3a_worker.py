"""Run imported formal V2 trainers with original-snapshot causal tensors substituted in memory."""
import os,sys,time,argparse,pickle,types
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='3'
os.environ['OPENBLAS_NUM_THREADS']='3'
os.environ['MKL_NUM_THREADS']='3'
from core import *
from torch.utils.data import DataLoader

def save_state(path,state):
 with target(Path(path)).open('xb') as f:torch_save_original(state,f)
torch_save_original=torch.save

def export(ds,mechanism,h,freq,model):
 loaders=formal._build_training_loaders(freq,h,ds)
 rows=[]
 for split,loader in zip(COUNTS,loaders):
  mask=ds.split==split
  if split=='train':loader=DataLoader(loader.dataset,batch_size=formal.BATCH_SIZE,shuffle=False)
  pred,dw,ww=exp3._batched_inference(model,loader,mechanism,None,None,COUNTS[split],torch.device('cpu'))
  frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ds.y[h][mask],y_pred=pred))
  for j,c in enumerate(WEIGHTS):frame[c]=(dw if j<4 else ww)[:,j%4]
  if mechanism=='static':assert np.max(np.ptp(frame[WEIGHTS].to_numpy(),axis=0))<1e-7
  name=f'preds/{mechanism}_dw_h{h}_s42_{split}.parquet';parquet(name,frame)
  reread=pd.read_parquet(OUT/name);pd.testing.assert_frame_equal(frame,reread)
  rows.append(dict(mechanism=mechanism,horizon=h,frequency_setting=freq,split=split,n_rows=len(frame),**metrics(frame.y_true.to_numpy(),pred,frame.ticker.to_numpy(),frame.anchor_date.to_numpy())))
 return rows

def neural(mechanism,h):
 env=threads(3);exp3._verify_frozen_dependencies();ds=load_causal();formal._formal_check_dataset_contract(ds)
 gate=json.loads((OUT/'precision_gate_decision.json').read_text());assert gate['stage3a_authorized_by_diagnostic']
 prior=json.loads((ROOT/'revision_work/outputs/task_a_processes.json').read_text());assert all(x['returncode']==0 for x in prior['processes'])
 cell=f'{mechanism}_dw_h{h}_s42';check=OUT/f'checkpoints/{cell}.pt';target(check);target(OUT/f'runs/{cell}.json')
 start=time.perf_counter();startutc=utc();captured=[];handles=[];phase={'epoch':0,'last':None};valacc={'sse':0.,'n':0};valmse=[]
 factory_owner=formal if mechanism=='adaptive' else exp3;factory_name='_make_model' if mechanism=='adaptive' else '_make_experiment3_model';factory=getattr(factory_owner,factory_name)
 def capture(*a,**kw):
  model=factory(*a,**kw);captured.append(model)
  def progress(module,inputs):
   status='train' if module.training else 'val'
   if status!=phase['last']:
    if status=='train':phase['epoch']+=1
    phase['last']=status;print('PHASE',utc(),cell,phase['epoch'],status,flush=True)
  handles.append(model.register_forward_pre_hook(progress));return model
 original_huber=torch.nn.HuberLoss
 def observed_huber(*a,**kw):
  criterion=original_huber(*a,**kw)
  def observe(module,args,output):
   if not captured[0].training:
    pred,y=args;diff=pred.detach()-y.detach();valacc['sse']+=float((diff*diff).sum(dtype=torch.float64));valacc['n']+=len(y)
    if valacc['n']==COUNTS['val']:
     valmse.append(valacc['sse']/valacc['n']);print('VAL_MSE',utc(),cell,len(valmse),valmse[-1],flush=True);valacc.update(sse=0.,n=0)
    elif valacc['n']>COUNTS['val']:raise RuntimeError('Unexpected validation row count')
  handles.append(criterion.register_forward_hook(observe));return criterion
 oldload=factory_owner.load_frozen_panel;oldos=exp3.os;oldcheckpoint=exp3.CHECKPOINT_DIR
 setattr(factory_owner,factory_name,capture);factory_owner.load_frozen_panel=lambda:ds;torch.nn.HuberLoss=observed_huber
 if mechanism=='static':
  exp3.CHECKPOINT_DIR=OUT/'trainer_checkpoints'
  def exclusive_publish(src,dst):
   with Path(src).open('rb') as f,target(Path(dst)).open('xb') as g:
    for b in iter(lambda:f.read(1024*1024),b''):g.write(b)
   # The newly created temporary file is deliberately retained; no file is renamed/deleted.
  exp3.os=types.SimpleNamespace(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')});exp3.os.replace=exclusive_publish
  torch.save=lambda state,path:save_state(path,state)
 try:
  print('START',utc(),cell,env,flush=True)
  result=formal.train_one_cell('daily_weekly',h,'adaptive') if mechanism=='adaptive' else exp3.train_and_validate_cell('static',f'{h}d',None,None)
 finally:
  setattr(factory_owner,factory_name,factory);factory_owner.load_frozen_panel=oldload;torch.nn.HuberLoss=original_huber;torch.save=torch_save_original;exp3.os=oldos;exp3.CHECKPOINT_DIR=oldcheckpoint
  for handle in handles:handle.remove()
 assert len(captured)==1
 model=captured[0];model.eval();save_state(check,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
 history=pd.DataFrame(result['history']);assert len(valmse)==len(history);history['train_huber']=history.train_loss;history['val_huber']=history.val_loss;history['val_mse']=valmse;history['loss_type']='Huber delta=1';csvout(f'history/{cell}.csv',history)
 summary=result['summary'].iloc[0].to_dict() if mechanism=='adaptive' else result['summary'];csvout(f'runs/{cell}_trainer_summary.csv',[summary])
 if mechanism=='static':jsonout(f'weights/{cell}_logits.json',{k:v.detach().cpu().tolist() for k,v in model.state_dict().items() if 'logit' in k})
 outmetrics=export(ds,mechanism,h,'daily_weekly',model);csvout(f'runs/{cell}_metrics.csv',outmetrics)
 metadata=dict(mechanism=mechanism,horizon=h,seed=42,started_utc=startutc,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,environment=env,best_epoch=int(summary['best_epoch']),stop_epoch=int(summary['epochs_trained']),checkpoint=rel(check),checkpoint_sha256=sha(check),n_epochs=len(history),val_mse_observer='sum of squared float32 residuals accumulated in float64; no extra model forward calls',input_snapshot='ORIGINAL only; frozen labels/splits/anchors',formal_training_entrypoint='main_v2.train_one_cell' if mechanism=='adaptive' else 'experiment3.train_and_validate_cell')
 jsonout(f'runs/{cell}.json',metadata);print('COMPLETE',utc(),metadata,flush=True)

def ridge_runs():
 from scripts.python import panel_baseline_ridge as ridge
 env=threads(3);ds=load_causal();stats=common.fit_train_only_normalizer(ds);original=ridge.Ridge
 for freq in ['daily_only','weekly_only','daily_weekly']:
  for h in [5,10,20]:
   start=time.perf_counter();started=utc();cell=f'ridge_{freq}_h{h}_s42';target(OUT/f'checkpoints/{cell}.pkl');models=[]
   def capture(*a,**kw):
    model=original(*a,**kw);models.append(model);return model
   ridge.Ridge=capture
   try:result,preds=ridge.run_one(ds,stats,freq,h)
   finally:ridge.Ridge=original
   model=next(m for m in models if m.alpha==result['best_alpha']);assert len(models)==4
   with target(OUT/f'checkpoints/{cell}.pkl').open('xb') as f:pickle.dump(model,f)
   x=common.flatten_features(freq,common.apply_normalizer(ds,stats,freq));grid=[];mets=[]
   for candidate in models:
    train=ds.split=='train';val=ds.split=='val';tp=candidate.predict(x[train]);vp=candidate.predict(x[val]);loss=lambda v:float(np.where(np.abs(v)<=1,.5*v*v,np.abs(v)-.5).mean())
    grid.append(dict(alpha=candidate.alpha,train_huber=loss(tp-ds.y[h][train]),val_huber=loss(vp-ds.y[h][val]),val_mse=float(np.mean((vp-ds.y[h][val])**2)),selected=bool(candidate is model)))
   csvout(f'history/{cell}_alpha_grid.csv',grid);csvout(f'history/{cell}.csv',[],columns=['epoch','train_huber','val_huber','val_mse'])
   for split in COUNTS:
    mask=ds.split==split;p=model.predict(x[mask]);frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ds.y[h][mask],y_pred=p));assert len(frame)==COUNTS[split]
    name=f'preds/{cell}_{split}.parquet';parquet(name,frame);pd.testing.assert_frame_equal(frame,pd.read_parquet(OUT/name));mets.append(dict(mechanism='ridge',horizon=h,frequency_setting=freq,split=split,n_rows=len(frame),**metrics(frame.y_true.to_numpy(),p,frame.ticker.to_numpy(),frame.anchor_date.to_numpy())))
   csvout(f'runs/{cell}_metrics.csv',mets);csvout(f'runs/{cell}_trainer_summary.csv',[result]);meta=dict(mechanism='ridge',horizon=h,seed=42,frequency_setting=freq,started_utc=started,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,environment=env,best_alpha=model.alpha,best_epoch=None,stop_epoch=None,history='No epochs for Ridge; four-alpha grid saved instead',checkpoint=rel(OUT/f'checkpoints/{cell}.pkl'))
   jsonout(f'runs/{cell}.json',meta);print('COMPLETE',utc(),meta,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mechanism',choices=['static','adaptive','ridge'],required=True);p.add_argument('--horizon',type=int,choices=[5,10,20]);a=p.parse_args()
 if a.mechanism=='ridge':ridge_runs()
 else:neural(a.mechanism,a.horizon)
