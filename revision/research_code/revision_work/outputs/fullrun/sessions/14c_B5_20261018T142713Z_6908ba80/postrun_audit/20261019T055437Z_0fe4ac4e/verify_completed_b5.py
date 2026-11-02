"""Read-only post-run B5 acceptance; no training and no new model inference."""
import os,sys,json,csv,math,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent;EXP=OUT.parents[1];ROOT=EXP.parents[4]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP/'scripts'));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import b5_io as io
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
import numpy as np,pandas as pd
from scripts.python import panel_common as common
from scripts.python import panel_baseline_lstm as lstm
from scripts.python import panel_baseline_vanilla_transformer as vanilla
from scripts.python import panel_baseline_swim_cpu as swim
sys.path.insert(0,str(io.DAY/'scripts'))
import day_report_v2 as summary_code
original_sha=io.sha;cache={}
def checked_sha(p):
 p=Path(p);s=p.stat();key=(str(p.resolve()),s.st_size,s.st_mtime_ns)
 if key not in cache:cache[key]=original_sha(p)
 return cache[key]
io.sha=checked_sha

def verify_frame(frame,ds,h,split,ordered=True):
 mask=ds.split==split
 assert len(frame)==io.COUNTS[split] and not frame.duplicated(['ticker','anchor_date']).any()
 assert (frame.split==split).all() and np.isfinite(frame.select_dtypes(include='number')).all().all()
 dates=pd.to_datetime(frame.anchor_date).to_numpy();expected_dates=pd.to_datetime(ds.anchor_date[mask]).to_numpy()
 if ordered:
  assert np.array_equal(frame.ticker.to_numpy(),ds.ticker[mask]) and np.array_equal(dates,expected_dates)
  assert np.array_equal(frame.y_true.to_numpy(),ds.y[h][mask])
 else:
  expected=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=expected_dates,y_true=ds.y[h][mask])).sort_values(['ticker','anchor_date']).reset_index(drop=True)
  actual=frame.assign(anchor_date=dates).sort_values(['ticker','anchor_date']).reset_index(drop=True)
  assert np.array_equal(actual[['ticker','anchor_date']].to_numpy(),expected[['ticker','anchor_date']].to_numpy())
  assert np.array_equal(actual.y_true.to_numpy(),expected.y_true.to_numpy())

def metric(frame):return common.evaluate_predictions(frame.y_true.to_numpy(),frame.y_pred.to_numpy(),frame.ticker.to_numpy(),frame.anchor_date.to_numpy())
def close(a,b,tol=1e-12):assert abs(float(a)-float(b))<tol,(a,b)
def main():
 final=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert final['queued_cells']==72 and final['pilot_cells']==1 and final['all_cells_verified']
 assert checked_sha(ROOT/final['report'])==final['report_sha256']
 ds=common.load_frozen_panel();rows=[];actual_metrics=[];telemetry_count=0;parquets=0;worst_replay=0.
 for spec in [io.PILOT]+io.PLAN:
  cp,r=io.completed(spec);assert r,spec
  history=pd.read_csv(cp.parent/'history.csv',float_precision='round_trip');assert len(history)==r['epochs']
  assert np.array_equal(history.epoch.to_numpy(),np.arange(1,r['epochs']+1))
  assert np.isfinite(history[['train_huber','val_huber','val_mse']]).all().all()
  assert np.array_equal(history.train_huber.to_numpy(),history.train_loss.to_numpy()) and np.array_equal(history.val_huber.to_numpy(),history.val_loss.to_numpy())
  assert int(history.loc[history.val_loss.idxmin(),'epoch'])==r['best_epoch']
  if 'seed' in history.columns:assert (history.seed==spec['seed']).all()
  check=r['input_check'];assert check['keys_labels_equal']
  assert {v['split'] for v in check['actual_trainer_loaders']}==set(io.COUNTS)
  assert all(v['bitwise_equal_input_features'] and v['generator'] is None for v in check['actual_trainer_loaders'])
  telemetry=[json.loads(x) for x in (cp.parent/'telemetry.jsonl').read_text().splitlines() if x]
  assert len(telemetry)>=2 and telemetry[0]['event']=='start' and telemetry[-1]['event']=='end'
  for t in telemetry:
   assert t['thread_counts']==[12,12] and t['environment']['OMP_NUM_THREADS'] is None
   assert t['environment']['OMP_DYNAMIC']==t['environment']['MKL_DYNAMIC']=='FALSE'
  telemetry_count+=len(telemetry)
  model=({'lstm':lstm.make_model,'vanilla':vanilla.make_model,'swim':swim.formal.make_model}[spec['mechanism']])('daily_weekly',5,5)
  state=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True)
  model.load_state_dict(state,strict=True);model.eval()
  assert all(torch.isfinite(v).all().item() for v in state.values())
  assert sum(p.numel() for p in model.parameters() if p.requires_grad)==r['parameter_count']
  delta=0.
  for split in io.COUNTS:
   f=pd.read_parquet(cp.parent/f'{split}.parquet');verify_frame(f,ds,spec['horizon'],split);parquets+=1
   if split=='test':
    m=metric(f);old=next(x for x in r['metrics'] if x['split']=='test')
    close(m['pooled_mse'],old['pooled_mse']);close(m['cross_sectional_rank_ic'],old['cross_sectional_rank_ic'])
    delta=abs(m['pooled_mse']-old['pooled_mse'])
    if spec['seed']!=42:actual_metrics.append(dict(block='B5',input_set=spec['input_set'],model=spec['mechanism'],frequency_setting='daily_weekly',horizon=spec['horizon'],seed=spec['seed'],test_mse=m['pooled_mse'],mean_rank_ic=m['cross_sectional_rank_ic']))
  worst_replay=max(worst_replay,r['test_replay']['max_abs_delta_pred'])
  rows.append(dict(job_id=spec['job_id'],epochs=r['epochs'],best_epoch=r['best_epoch'],global_rng_seed=r['global_rng_seed'],checkpoint_strict_reload=True,history_passed=True,all_three_splits_keys_labels_passed=True,all_telemetry_passed=True,telemetry_samples=len(telemetry),recorded_same_run_max_delta_pred=r['test_replay']['max_abs_delta_pred'],test_mse_recomputed_delta=delta))
  if len(rows)%10==0:print('VERIFIED_NEW_CELLS',len(rows),flush=True)
 assert len(rows)==73 and parquets==219
 provenance=json.loads((EXP/'reference_provenance.json').read_text())
 for a in provenance['sources']+[provenance['four_mechanism_summary']]:assert checked_sha(ROOT/a['path'])==a['sha256']
 refs=pd.read_csv(EXP/'seed42_metrics.csv',float_precision='round_trip');assert len(refs)==18
 for _,r in refs.iterrows():
  assert r.seed==42 and checked_sha(ROOT/r.prediction)==r.prediction_sha256
  f=pd.read_parquet(ROOT/r.prediction);verify_frame(f,ds,int(r.horizon),'test',ordered=False);m=metric(f)
  close(m['pooled_mse'],r.test_mse);close(m['cross_sectional_rank_ic'],r.mean_rank_ic)
  actual_metrics.append(dict(block='B5',input_set=r.input_set,model=r.model,frequency_setting='daily_weekly',horizon=int(r.horizon),seed=42,test_mse=m['pooled_mse'],mean_rank_ic=m['cross_sectional_rank_ic']))
 assert len(actual_metrics)==90
 for key,g in pd.DataFrame(actual_metrics).groupby(['input_set','model','horizon']):assert len(g)==5 and set(g.seed)==set(io.SEEDS)
 four=pd.read_csv(ROOT/provenance['four_mechanism_summary']['path'],float_precision='round_trip')
 expected_side=list(actual_metrics)
 for _,r in four.iterrows():expected_side.append(dict(block='B5',input_set=r.input_set,model=r.mechanism,frequency_setting='daily_weekly',horizon=int(r.horizon),seed=int(r.seed),test_mse=r.test_mse,mean_rank_ic=r.mean_rank_ic))
 for filename,expected,n in [('benchmark_summary_5seed.csv',actual_metrics,18),('side_by_side_summary_5seed.csv',expected_side,42)]:
  calculated=summary_code.summarize(expected);saved=pd.read_csv(EXP/filename,float_precision='round_trip');assert len(calculated)==len(saved)==n
  for c in calculated:
   match=saved[(saved.input_set==c['input_set'])&(saved.model==c['model'])&(saved.horizon==c['horizon'])];assert len(match)==1
   r=match.iloc[0];assert r.n_seeds==5
   for field in ['test_mse_mean','test_mse_sd','mean_rank_ic_mean','mean_rank_ic_sd']:close(c[field],r[field])
 for filename,expected in [('benchmark_cell_metrics.csv',actual_metrics),('side_by_side_per_seed.csv',expected_side)]:
  saved=pd.read_csv(EXP/filename,float_precision='round_trip');assert len(saved)==len(expected)
  for c in expected:
   match=saved[(saved.input_set==c['input_set'])&(saved.model==c['model'])&(saved.horizon==c['horizon'])&(saved.seed==c['seed'])];assert len(match)==1
   for field in ['test_mse','mean_rank_ic']:close(c[field],match.iloc[0][field])
 phases=[]
 for phase in (EXP/'sessions').iterdir():
  if not (phase/'queue_result.json').exists():continue
  q=json.loads((phase/'queue_result.json').read_text());p=json.loads((phase/'protected_inputs_verification.json').read_text())
  assert not q['failures'] and not q['blocked'] and p['passed']
  assert json.loads((phase/'FINISHED.json').read_text())['caffeinate_alive_through_report']
  phases.append(q['n_complete'])
 assert sorted(phases)==[1,72]
 pilot=json.loads((EXP/'PILOT_COMPARISON.json').read_text());assert pilot['classification']=='DIFFERENT' and pilot['summary_uses_archived_seed42']
 assert 'DIFFERENT' in (ROOT/final['report']).read_text().split('# Step14c-B5')[0]
 io.csvout(OUT/'cell_checks.csv',rows)
 body='# BLOCKING\n\nThe required O LSTM seed42 h5 historical replay remains DIFFERENT (max prediction delta 0.02687845751643181; MSE delta +6.23287633061409e-05). It is retained in the original report as requested; its cause is not established. This does not prevent acceptance of completed queue artifacts.\n\n# Post-run acceptance\n\nPASS: 72 queued cells plus 1 pilot; 73 checkpoints strictly reloaded into the imported original factories with finite tensors and matching parameter counts; 219 prediction Parquets verified against the frozen ordered keys, split counts and labels; per-epoch histories/best epochs checked; every telemetry record retains 12/12 and required environment; every saved artifact/dependency hash verified. Original same-run trainer TEST replays are all EXACT, and their artifacts are unchanged; this audit does not perform a new forward-inference replay. TEST MSE and mean Rank IC recomputed for all 73 new cells and 18 reused seed42 cells agree with recorded scores. Five-seed mean and sample SD tables (18 benchmark and 42 side-by-side groups) recomputed with the imported summary helper and agree within 1e-12. Seed42 diagnostic predictions are excluded from the five-seed summaries. Phase source-protection checks and report SHA256 passed.\n'
 io.textout(OUT/'ACCEPTANCE.md',body)
 io.jsonout(OUT/'ACCEPTANCE.json',dict(utc=io.utc(),passed=True,new_cells=73,queued_cells=72,pilot_cells=1,strict_checkpoint_reloads=73,verified_prediction_parquets=219,verified_reused_seed42_cells=18,telemetry_samples=telemetry_count,all_artifact_dependency_hashes_passed=True,all_keys_labels_counts_passed=True,test_metrics_recomputed=True,five_seed_summaries_recomputed=True,benchmark_summary_groups=18,side_by_side_summary_groups=42,maximum_recorded_same_run_replay_delta=worst_replay,historical_pilot_classification='DIFFERENT',historical_pilot_cause='unverified',report=io.artifact(ROOT/final['report']),evidence=[io.artifact(OUT/'cell_checks.csv'),io.artifact(OUT/'ACCEPTANCE.md')]))
 print('B5_POSTRUN_ACCEPTANCE_PASS',73,219,18,42,'maximum recorded replay delta',worst_replay,flush=True)
if __name__=='__main__':main()
