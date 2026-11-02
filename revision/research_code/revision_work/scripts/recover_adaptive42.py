"""Run the unchanged formal train_one_cell; retain its model reference to save restored state."""
import os,sys,time,argparse
sys.dont_write_bytecode=True
# Set OMP before importing torch. Launcher also supplies this environment.
os.environ['OMP_NUM_THREADS']='3'
from step3_common import *
from scripts.python import panel_pathformer_main_v2 as formal
from scripts.python import panel_pathformer_experiment3 as exp3

def main():
 p=argparse.ArgumentParser();p.add_argument('--horizon',type=int,choices=[5,10,20],required=True);args=p.parse_args();h=args.horizon
 checkpoint=OUT/'checkpoints'/f'adaptive_dw_h{h}_s42.pt'
 for path in [checkpoint,OUT/'task_a'/f'recovery_h{h}.json']:
  if path.exists():raise FileExistsError(f'Refusing to retrain/overwrite completed recovery: {path}')
 environment=threads(3);exp3._verify_frozen_dependencies()
 print('START',utc(),'horizon',h,environment,flush=True)
 started=time.perf_counter();started_utc=utc();captured=[];epoch_status={'epoch':0,'phase':None};handles=[]
 original_factory=formal._make_model
 # Pure observer: delegates architecture construction unchanged, captures a reference,
 # and reports train/validation transitions without touching tensors or RNG state.
 def observer_factory(*args,**kwargs):
  model=original_factory(*args,**kwargs);captured.append(model)
  def progress(module,inputs):
   phase='TRAIN' if module.training else 'VALIDATION'
   if phase!=epoch_status['phase']:
    if phase=='TRAIN':epoch_status['epoch']+=1
    epoch_status['phase']=phase
    print('PHASE',utc(),'h',h,'epoch',epoch_status['epoch'],phase,flush=True)
  handles.append(model.register_forward_pre_hook(progress))
  return model
 formal._make_model=observer_factory
 try:
  # This is the full original MAIN V2 training function: same data, loaders, seed,
  # optimizer, loss, patience, checkpoint restoration and test evaluator.
  result=formal.train_one_cell('daily_weekly',h,'adaptive')
 finally:
  formal._make_model=original_factory
  for handle in handles:handle.remove()
 if len(captured)!=1:raise RuntimeError('Expected exactly one original model instance')
 model=captured[0]
 state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 with target(checkpoint).open('xb') as f:torch.save(state,f)
 for name in ['summary','predictions','rank_ic','history','router']:
  write_csv(OUT/'task_a'/f'adaptive_dw_h{h}_s42_{name}.csv',result[name])
 old=formal_subset(MAIN/'panel_pathformer_test_predictions.csv','adaptive',h,42)
 comparison=compare(result['predictions'],old)
 new_weights=key_sorted(result['router']);old_weights=key_sorted(formal_subset(MAIN/'panel_pathformer_router_weights.csv','adaptive',h,42))
 if not new_weights[['ticker','anchor_date']].equals(old_weights[['ticker','anchor_date']]):raise ValueError('Router key mismatch')
 weight_delta=float(np.max(np.abs(new_weights[WEIGHTS].to_numpy()-old_weights[WEIGHTS].to_numpy())))
 old_summary=formal_subset(MAIN/'panel_pathformer_summary_metrics.csv','adaptive',h,42).iloc[0]
 s=result['summary'].iloc[0]
 old_hist=formal_subset(MAIN/'panel_pathformer_training_history.csv','adaptive',h,42).sort_values('epoch')
 new_hist=result['history'].sort_values('epoch')
 trajectory_delta=float(np.max(np.abs(old_hist[['train_loss','val_loss']].to_numpy()-new_hist[['train_loss','val_loss']].to_numpy()))) if len(old_hist)==len(new_hist) else None
 metadata=dict(horizon=h,seed=42,input_config='daily_weekly',mechanism='adaptive',environment=environment,started_utc=started_utc,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-started,formal_training_loop_seconds=float(s.runtime_seconds),checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),best_epoch=int(s.best_epoch),formal_best_epoch=int(old_summary.best_epoch),best_epoch_equal=bool(s.best_epoch==old_summary.best_epoch),stop_epoch=int(s.epochs_trained),formal_stop_epoch=int(old_summary.epochs_trained),stop_epoch_equal=bool(s.epochs_trained==old_summary.epochs_trained),router_max_abs_delta=weight_delta,history_max_abs_delta=trajectory_delta,restored_state_fingerprint=formal._state_dict_sha256(model),observer='in-process model-factory reference capture plus phase-only forward pre-hook; original source/trainer unchanged; no extra RNG draws',**comparison)
 write_json(OUT/'task_a'/f'recovery_h{h}.json',metadata)
 print('COMPLETE',utc(),metadata,flush=True)
if __name__=='__main__':main()
