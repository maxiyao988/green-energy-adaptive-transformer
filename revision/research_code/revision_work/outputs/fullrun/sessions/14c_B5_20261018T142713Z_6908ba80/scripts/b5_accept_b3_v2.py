"""Post-B3 audit, before any B5 numerical job. Imports formal metrics only."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
from b5_io import *
import fcntl,subprocess,math
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
import numpy as np,pandas as pd
sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
from scripts.python import panel_common as common
from scripts.python import panel_pathformer_experiment4_multiseed as exp4

def locked(path):
 if not path.exists():return False
 with path.open('rb') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(f,fcntl.LOCK_UN);return False

def main():
 assert not locked(B3/'queue.lock'),'B3 scheduler still running'
 ps=subprocess.run(['ps','-axo','pid,command'],capture_output=True,text=True,check=True).stdout
 assert not any(str(B3) in line and 'fullrun14a_worker_v2.py' in line and '--job-id ' in line for line in ps.splitlines()),'B3 workers still running'
 final=json.loads((B3/'FINAL_VALIDATION.json').read_text())
 assert final['cells']==180 and final['all_cells_verified'] and final['protected_files_unchanged'] and final['all_keys_labels_counts_verified']
 assert sha(ROOT/final['report'])==final['report_sha256']
 assert any((p/'FINISHED.json').exists() for p in (B3/'control').iterdir()),'B3 outer queue has not finished'
 sys.path.insert(0,str(B3/'scripts'))
 import b3_contracts as bc
 # Cache only identical path/size/mtime versions during this read-only audit.
 original_sha=bc.sha;cache={}
 def cached_sha(p):
  p=Path(p);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
  if key not in cache:cache[key]=original_sha(p)
  return cache[key]
 bc.sha=cached_sha
 ds=common.load_frozen_panel();plan=json.loads((B3/'experiment.json').read_text())['plan'];rows=[]
 for spec in plan:
  cp,r=bc.completed({k:spec[k] for k in bc.KEYS});assert r,spec
  history=pd.read_csv(cp.parent/'history.csv');assert len(history)==r['epochs']
  assert np.isfinite(history[['train_huber','val_huber','val_mse']]).all().all()
  assert exp4._reconstruct_best_validation_checkpoint(history)[0]==r['best_epoch']
  state=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True)
  assert isinstance(state,dict) and state and all(torch.isfinite(v).all().item() for v in state.values() if isinstance(v,torch.Tensor))
  telemetry=[json.loads(l) for l in (cp.parent/'telemetry.jsonl').read_text().splitlines() if l]
  assert len(telemetry)>=2 and all(t['thread_counts']==[12,12] for t in telemetry)
  assert all(t['environment']['OMP_NUM_THREADS'] is None and t['environment']['OMP_DYNAMIC']==t['environment']['MKL_DYNAMIC']=='FALSE' for t in telemetry)
  for split,n in COUNTS.items():
   f=pd.read_parquet(cp.parent/f'{split}.parquet');mask=ds.split==split
   assert len(f)==n and not f.duplicated(['ticker','anchor_date']).any()
   assert (f.split==split).all() and np.array_equal(f.ticker.to_numpy(),ds.ticker[mask])
   assert np.array_equal(pd.to_datetime(f.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy())
   assert np.array_equal(f.y_true.to_numpy(),ds.y[spec['horizon']][mask])
   assert np.isfinite(f.select_dtypes(include='number')).all().all()
   if split=='test':
    scores=common.evaluate_predictions(f.y_true.to_numpy(),f.y_pred.to_numpy(),f.ticker.to_numpy(),f.anchor_date.to_numpy())
    recorded=next(v for v in r['metrics'] if v['split']=='test')
    md=abs(scores['pooled_mse']-recorded['pooled_mse']);rd=abs(scores['cross_sectional_rank_ic']-recorded['cross_sectional_rank_ic']);assert md<1e-12 and rd<1e-12,(spec,md,rd)
  rows.append(dict(job_id=spec['job_id'],checkpoint=rel(ROOT/r['checkpoint']),epochs=r['epochs'],best_epoch=r['best_epoch'],artifacts_hashes_passed=True,checkpoint_finite=True,recorded_strict_reload=r['strict_checkpoint_reload'],keys_labels_counts_passed=True,telemetry_passed=True,test_mse_recompute_delta=md,rank_ic_recompute_delta=rd))
  if len(rows)%20==0:print('B3_ACCEPTANCE_CELLS',len(rows),flush=True)
 assert len(rows)==180
 for session in (B3/'sessions').iterdir():
  if (session/'queue_result.json').exists():
   q=json.loads((session/'queue_result.json').read_text());assert not q['failures'] and not q['blocked']
   p=json.loads((session/'protected_inputs_verification.json').read_text());assert p['passed']
 for filename,n in [('summary_5seed.csv',36),('summary_3seed.csv',36),('all_cell_metrics.csv',180)]:assert len(pd.read_csv(B3/filename))==n
 csvout(EXP/'B3_acceptance_cells.csv',rows)
 textout(EXP/'B3_ACCEPTANCE.md','# BLOCKING\n\nNone. B3 completed all 180 cells before this audit and before B5 training.\n\nVerified all completion/dependency/artifact hashes; 540 Parquets have the formal row counts, ordered keys and frozen labels; histories are finite and best epochs match the imported formal validation-Huber improvement rule; all checkpoint tensors are finite and prior strict-reload attestations verified by artifact hash; all telemetry samples retain CPU12/12 and required environment. TEST MSE and mean Rank IC recomputed with imported formal metric code agree within 1e-12. Final report hash, summary row counts and source-protection checks passed. This audit does not perform a new inference replay of all 180 checkpoints; their original strict reload/inference outputs are preserved and hash verified.\n')
 jsonout(EXP/'B3_ACCEPTANCE.json',dict(utc=utc(),passed=True,cells=180,parquets=540,test_metrics_recomputed=True,checkpoint_tensors_finite=True,prior_strict_checkpoint_reloads_hash_verified=True,all_telemetry_samples_12_12=True,report=artifact(ROOT/final['report']),evidence=[artifact(EXP/'B3_acceptance_cells.csv'),artifact(EXP/'B3_ACCEPTANCE.md')]))
 print('B3_ACCEPTANCE_PASS',flush=True)
if __name__=='__main__':main()
