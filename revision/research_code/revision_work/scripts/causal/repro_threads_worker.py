"""One CPU training process: imported formal V2 path, 12 intra-op / 12 inter-op, OMP unset."""
import os,sys,time,argparse,types
sys.dont_write_bytecode=True
os.environ.pop('OMP_NUM_THREADS',None)
from repro_threads_common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--mechanism',required=True,choices=['adaptive','static']);p.add_argument('--horizon',required=True,type=int,choices=[5,10,20]);a=p.parse_args();mechanism=a.mechanism;h=a.horizon
 environment=configure();exp3._verify_frozen_dependencies();exp4._verify_frozen_dependencies()
 stem=f'{mechanism}_dw_h{h}_s42_t12';checkpoint=OUT/f'checkpoints/{stem}.pt'
 target(checkpoint);target(OUT/f'runs/{stem}.json');assert (OUT/'isolation_condition.json').is_file()
 start=time.perf_counter();startutc=utc();telemetry=Telemetry(stem);telemetry.sample('start',force=True);captured=[];handles=[];phase={'epoch':0,'last':None}
 owner=formal if mechanism=='adaptive' else exp3;factory_name='_make_model' if mechanism=='adaptive' else '_make_experiment3_model';factory=getattr(owner,factory_name)
 def capture(*args,**kwargs):
  model=factory(*args,**kwargs);captured.append(model)
  def progress(module,inputs):
   telemetry.sample('five_minute');current='TRAIN' if module.training else 'VALIDATION'
   if current!=phase['last']:
    if current=='TRAIN':phase['epoch']+=1
    phase['last']=current;print('PHASE',utc(),stem,phase['epoch'],current,flush=True)
  handles.append(model.register_forward_pre_hook(progress));return model
 setattr(owner,factory_name,capture)
 oldsave=torch.save;oldos=exp3.os;oldpath=exp3._checkpoint_path
 if mechanism=='static':
  exp3._checkpoint_path=lambda *args,**kwargs:checkpoint
  def save_exclusively(obj,destination,*args,**kwargs):
   with target(Path(destination)).open('xb') as f:oldsave(obj,f,*args,**kwargs)
  def publish_exclusively(src,dst):
   with Path(src).open('rb') as source,target(Path(dst)).open('xb') as destination:
    for block in iter(lambda:source.read(1024*1024),b''):destination.write(block)
   # Retain the newly created .tmp rather than renaming or deleting anything.
  torch.save=save_exclusively;exp3.os=types.SimpleNamespace(**{name:getattr(os,name) for name in dir(os) if not name.startswith('__')});exp3.os.replace=publish_exclusively
 print('START',utc(),stem,environment,flush=True)
 try:
  result=formal.train_one_cell('daily_weekly',h,'adaptive') if mechanism=='adaptive' else exp3.train_and_validate_cell('static',f'{h}d',None,None)
 finally:
  setattr(owner,factory_name,factory);torch.save=oldsave;exp3.os=oldos;exp3._checkpoint_path=oldpath
  for handle in handles:handle.remove()
 assert len(captured)==1;model=captured[0];state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 if mechanism=='adaptive':
  with target(checkpoint).open('xb') as f:oldsave(state,f)
 else:
  loaded=torch.load(checkpoint,map_location='cpu',weights_only=True);assert all(torch.equal(v,loaded[k]) for k,v in state.items())
 history=result['history'];csvout(f'runs/{stem}_history.csv',history);summary=result['summary'].iloc[0].to_dict() if mechanism=='adaptive' else result['summary'];csvout(f'runs/{stem}_trainer_summary.csv',[summary])
 if mechanism=='adaptive':
  for name in ['predictions','rank_ic','router']:csvout(f'runs/{stem}_{name}.csv',result[name])
 history_result=history_check(stem,mechanism,h,history)
 source=MAIN/'panel_pathformer_test_predictions.csv' if mechanism=='adaptive' else E3/'experiment3_test_predictions.csv';old=formal_subset(source,mechanism,h,42)
 ds,loaders,frames=export_new_checkpoint(stem,mechanism,h,checkpoint);comparison=compare(frames['test'],old)
 old_s=formal_subset(MAIN/'panel_pathformer_summary_metrics.csv' if mechanism=='adaptive' else E3/'experiment3_training_summary.csv',mechanism,h,42).iloc[0]
 if mechanism=='adaptive':
  w=key_sorted(result['router']);oldw=key_sorted(formal_subset(MAIN/'panel_pathformer_router_weights.csv','adaptive',h,42));assert w[['ticker','anchor_date']].equals(oldw[['ticker','anchor_date']]);weight_delta=float(np.abs(w[WEIGHTS].to_numpy()-oldw[WEIGHTS].to_numpy()).max())
  direct=compare(frames['test'],result['predictions']);csvout(f'comparisons/{stem}_own_checkpoint_replay.csv',[direct])
 else:
  reference_path=ROOT/old_s.checkpoint_path;assert sha(reference_path)==old_s.checkpoint_sha256
  reference=exp3._make_experiment3_model('static',None,None);reference.load_state_dict(torch.load(reference_path,map_location='cpu',weights_only=True),strict=True);reference.to(formal._resolve_device());reference.eval()
  prediction,dw,ww=exp3._batched_inference(reference,loaders['test'],'static',None,None,COUNTS['test'],formal._resolve_device());mask=ds.split=='test';ref=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split='test',y_true=ds.y[h][mask],y_pred=prediction))
  for i,col in enumerate(WEIGHTS):ref[col]=(dw if i<4 else ww)[:,i%4]
  write_parquet(OUT/f'preds/formal_static_dw_h{h}_s42_checkpoint_t12_test.parquet',ref)
  replay=compare(ref,old);versus_checkpoint=compare(frames['test'],ref);csvout(f'comparisons/{stem}_formal_checkpoint_replay.csv',[dict(reference_checkpoint=rel(reference_path),**replay)]);csvout(f'comparisons/{stem}_versus_formal_checkpoint.csv',[versus_checkpoint]);weight_delta=float(np.abs(frames['test'][WEIGHTS].to_numpy()-ref[WEIGHTS].to_numpy()).max())
  write_json(OUT/f'runs/{stem}_static_logits.json',{k:v.tolist() for k,v in state.items() if 'logit' in k})
 metadata=dict(mechanism=mechanism,horizon=h,seed=42,input_config='daily_weekly',started_utc=startutc,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,formal_training_loop_seconds=float(summary['runtime_seconds']),environment=environment,checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),best_epoch=int(summary['best_epoch']),formal_best_epoch=int(old_s.best_epoch),stop_epoch=int(summary['epochs_trained']),formal_stop_epoch=int(old_s.epochs_trained),best_epoch_equal=bool(summary['best_epoch']==old_s.best_epoch),stop_epoch_equal=bool(summary['epochs_trained']==old_s.epochs_trained),router_max_abs_delta=weight_delta,restored_state_fingerprint=formal._state_dict_sha256(model),source_predictions=rel(source),observer='Same delegated factory-reference and phase-only forward pre-hook as recover_adaptive42.py; no extra training forward/backward/RNG draws',**history_result,**comparison)
 if mechanism=='static':metadata.update(comparison_vs_formal_checkpoint=versus_checkpoint,formal_checkpoint_replay=replay)
 telemetry.sample('end',force=True);telemetry.close();metadata.update(telemetry_samples=telemetry.count,logged_thread_counts_changed=telemetry.changed)
 write_json(OUT/f'runs/{stem}.json',metadata);print('COMPLETE',utc(),metadata,flush=True)
if __name__=='__main__':main()
