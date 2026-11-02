"""Rolling-origin cell adapter; delegates complete training loops to formal imports."""
from b2_io import *
from b2_data import base,common,formal,core,load_panel,window_contract
import numpy as np,pandas as pd,torch,time,argparse
from scripts.python import panel_baseline_ridge as ridge
ORIGINAL_DEPS=base.dependencies

def deps(inp,window):
 original=ORIGINAL_DEPS(inp)
 tensor=json.loads((EXP/'tensor_manifest.json').read_text())['files'];prefix=rel(EXP/'tensors'/window)+'/'
 return list({r['path']:r for r in original+extra_dependencies()+[artifact(Path(ridge.__file__))]+[r for r in tensor if r['path'].startswith(prefix)]}.values())

def run_reference(spec,attempt):
 base.check_profile();start=time.perf_counter();started=utc();w=spec['window'];h=spec['horizon'];kind=spec['mechanism'];ds=load_panel(w,spec['input_set']);window_contract(ds,w);ct=counts(w);dependencies=deps(spec['input_set'],w)
 telemetry=base.Telemetry(attempt/'telemetry.jsonl');telemetry.sample('start',True);history=[];frames={};metrics=[]
 stats=common.fit_train_only_normalizer(ds);X=common.flatten_features('daily_weekly',common.apply_normalizer(ds,stats,'daily_weekly'));train=ds.split=='train';val=ds.split=='val';ys=ds.y[h]
 if kind=='ridge':
  constructor=ridge.Ridge;models=[]
  def capture(*a,**kw):
   m=constructor(*a,**kw);models.append(m);return m
  ridge.Ridge=capture
  try:summary,reference=ridge.run_one(ds,stats,'daily_weekly',h)
  finally:ridge.Ridge=constructor
  assert len(models)==4 and [m.alpha for m in models]==[.1,1.,10.,100.]
  chosen=next(m for m in models if m.alpha==summary['best_alpha']);checkpoint=attempt/'best_model.pkl'
  for model in models:
   tp=model.predict(X[train]);vp=model.predict(X[val]);history.append(dict(alpha=float(model.alpha),epoch=None,history_kind='four-alpha validation-MSE selection; no neural epochs',train_huber=float(torch.nn.functional.huber_loss(torch.from_numpy(tp),torch.from_numpy(ys[train]),delta=1)),val_huber=float(torch.nn.functional.huber_loss(torch.from_numpy(vp),torch.from_numpy(ys[val]),delta=1)),val_mse=float(np.mean((vp-ys[val])**2))))
  with target(checkpoint).open('xb') as f:pickle.dump(chosen,f,protocol=5)
  with checkpoint.open('rb') as f:model=pickle.load(f)
  assert np.array_equal(model.coef_,chosen.coef_) and model.intercept_==chosen.intercept_
  predict=lambda mask:model.predict(X[mask]);original_test=reference.y_pred.to_numpy();nparams=int(model.coef_.size+1)
 else:
  mean=np.float32(0.) if kind=='zero' else np.float32(ys[train].mean());checkpoint=attempt/'constant_predictor.json'
  jsonout(checkpoint,dict(model=kind,value=float(mean),window=w,horizon=h,construction='zero constant' if kind=='zero' else 'mean of this window TRAIN labels in frozen float32 arithmetic',n_train=int(train.sum()),trained_optimizer=False))
  value=np.float32(json.loads(checkpoint.read_text())['value']);assert value==mean
  predict=lambda mask:np.full_like(ys[mask],value);original_test=np.full_like(ys[ds.split=='test'],mean);nparams=0 if kind=='zero' else 1
  tp=predict(train);vp=predict(val);history.append(dict(epoch=None,history_kind='constant construction; no neural training',train_huber=float(torch.nn.functional.huber_loss(torch.from_numpy(tp),torch.from_numpy(ys[train]),delta=1)),val_huber=float(torch.nn.functional.huber_loss(torch.from_numpy(vp),torch.from_numpy(ys[val]),delta=1)),val_mse=float(np.mean((vp-ys[val])**2))))
  summary=dict(model=kind,window=w,horizon=h,train_mean=float(mean),n_train=int(train.sum()),no_optimization=True)
 for split,n in ct.items():
  mask=ds.split==split;pred=predict(mask);frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ys[mask],y_pred=pred));frames[split]=frame
  assert len(frame)==n and not frame.duplicated(['ticker','anchor_date']).any() and np.isfinite(frame[['y_true','y_pred']]).all().all()
  assert np.array_equal(frame.y_true.to_numpy(),ys[mask]) and np.array_equal(frame.ticker.to_numpy(),ds.ticker[mask]);assert np.array_equal(pd.to_datetime(frame.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy())
  metrics.append(dict(split=split,**common.evaluate_predictions(ys[mask],pred,ds.ticker[mask],ds.anchor_date[mask])))
  with target(attempt/f'{split}.parquet').open('xb') as f:f.write(frame.to_parquet(index=False,engine='pyarrow'))
  pd.testing.assert_frame_equal(frame,pd.read_parquet(attempt/f'{split}.parquet'))
 delta=float(np.max(np.abs(frames['test'].y_pred.to_numpy(np.float64)-np.asarray(original_test,np.float64))));assert delta==0.
 for name,data in [('history.csv',history),('metrics.csv',metrics),('trainer_summary.csv',[summary])]:
  with target(attempt/name).open('x',newline='') as f:pd.DataFrame(data).to_csv(f,index=False)
 telemetry.sample('end',True);telemetry.handle.close()
 for d in dependencies:assert sha(ROOT/d['path'])==d['sha256']
 r=dict(**spec,validated=True,pid=os.getpid(),started_utc=started,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,epochs=0,best_epoch=0,training_entrypoint='panel_baseline_ridge.run_one' if kind=='ridge' else 'deterministic constant constructor using window TRAIN labels only',loader_generator_seed=None,row_counts=ct,checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),thread_counts_start=telemetry.rows[0]['thread_counts'],thread_counts_end=telemetry.rows[-1]['thread_counts'],thread_counts_changed=False,environment_start=telemetry.rows[0]['environment'],environment_end=telemetry.rows[-1]['environment'],load_average_start=telemetry.rows[0]['load_average_1_5_15'],load_average_end=telemetry.rows[-1]['load_average_1_5_15'],telemetry_samples=len(telemetry.rows),parameter_count=nparams,metrics=clean(metrics),test_replay=dict(classification='EXACT',max_abs_delta_pred=delta),q0=None,dependencies=dependencies,artifacts=[artifact(p) for p in sorted(attempt.iterdir()) if p.is_file() and p.name!='worker.log'])
 jsonout(attempt/'COMPLETE.json',r);verify_completion(attempt/'COMPLETE.json',spec);print('REFERENCE_COMPLETE',spec['job_id'],r['wall_clock_seconds'],flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--job-id');ap.add_argument('--attempt');a=ap.parse_args()
 if a.preflight:
  assert json.loads((EXP/'preflight.json').read_text())['passed'];return
 spec=spec_for(a.job_id);attempt=Path(a.attempt).absolute();assert attempt.is_relative_to(EXP/'runs'/spec['job_id'])
 try:
  if spec['block']=='neural':
   w=spec['window'];base.OUT=EXP;base.COUNTS=counts(w);base.dataset=lambda inp:load_panel(w,inp);base.dependencies=lambda inp:deps(inp,w);base.verify_completion=verify_completion
   # Only split contract bindings change. Original W3 invariants were checked unchanged during preparation.
   formal._formal_check_dataset_contract=lambda ds:window_contract(ds,w);base.exp4.N_TRAIN_SAMPLES=counts(w)['train']
   base.run(spec,attempt)
  else:run_reference(spec,attempt)
 except BaseException as e:
  if not (attempt/'FAILED.json').exists():jsonout(attempt/'FAILED.json',dict(utc=utc(),job_id=spec['job_id'],error=repr(e)))
  raise
if __name__=='__main__':main()
