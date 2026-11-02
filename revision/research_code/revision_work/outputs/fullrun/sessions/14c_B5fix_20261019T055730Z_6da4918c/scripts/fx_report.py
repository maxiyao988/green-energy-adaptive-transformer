"""B5fix archive comparisons and two five-seed O-only summaries; no training."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
from fx_io import *
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd
from scripts.python import panel_common as common
sys.path.insert(0,str(B5/'scripts'))
import b5_io as oldio
sys.path.insert(0,str(DAY/'scripts'))
import day_report_v2 as summary_code

def canonical(frame):
 f=frame.copy();f['anchor_date']=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d');assert not f.duplicated(['ticker','anchor_date']).any()
 return f.sort_values(['ticker','anchor_date']).reset_index(drop=True)
def metrics(f):return common.evaluate_predictions(f.y_true.to_numpy(),f.y_pred.to_numpy(),f.ticker.to_numpy(),f.anchor_date.to_numpy())
def check_frame(f,ds,h,split):
 mask=ds.split==split;assert len(f)==COUNTS[split] and (f.split==split).all()
 assert np.array_equal(f.ticker.to_numpy(),ds.ticker[mask]) and np.array_equal(pd.to_datetime(f.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy())
 assert np.array_equal(f.y_true.to_numpy(),ds.y[h][mask]) and not f.duplicated(['ticker','anchor_date']).any()
 assert np.isfinite(f.select_dtypes(include='number')).all().all()
def profile(r,cp):
 ts=[json.loads(x) for x in (cp.parent/'telemetry.jsonl').read_text().splitlines() if x]
 assert len(ts)>=2
 for t in ts:
  assert t['thread_counts']==[12,12] and t['environment']['OMP_NUM_THREADS'] is None
  assert t['environment']['OMP_DYNAMIC']==t['environment']['MKL_DYNAMIC']=='FALSE'
 assert r['seed_calls']==[r['seed']] and r['global_rng_seed']==r['seed'] and r['loader_generator_seed'] is None
 return len(ts)
def row(model,h,seed,m,cp,pred,origin,execution_profile='CPU12/12'):
 return dict(block='B5FIX',input_set='O',model=model,frequency_setting='daily_weekly',horizon=h,seed=seed,test_mse=m['pooled_mse'],mean_rank_ic=m['cross_sectional_rank_ic'],completion=rel(cp) if cp else '',prediction=rel(pred),provenance=origin,profile=execution_profile)

def generate(session):
 session=Path(session);q=json.loads((session/'queue_result.json').read_text());assert q['n_complete']==9 and not q['failures'] and not q['blocked']
 report=FULLRUN/'REPORT_STEP14C_B5FIX.md'
 if (EXP/'FINAL_VALIDATION.json').exists():
  v=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert sha(ROOT/v['report'])==v['report_sha256']
  jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(report),artifact(EXP/'FINAL_VALIDATION.json')],verified_previous_report=True));return
 ds=common.load_frozen_panel();ref=json.loads((EXP/'reference_manifest.json').read_text());refs={(r['model'],int(r['horizon'])):r for r in ref['references']}
 primary=[];reference=[];comparisons=[];telemetry=[];files=[]
 for spec in PLAN:
  cp,r=completed(spec);assert r,spec;model,h=spec['mechanism'],spec['horizon'];n=profile(r,cp)
  history=pd.read_csv(cp.parent/'history.csv',float_precision='round_trip');assert len(history)==r['epochs'] and int(history.loc[history.val_loss.idxmin(),'epoch'])==r['best_epoch']
  assert np.isfinite(history[['train_huber','val_huber','val_mse']]).all().all()
  for split in COUNTS:check_frame(pd.read_parquet(cp.parent/f'{split}.parquet'),ds,h,split)
  f=pd.read_parquet(cp.parent/'test.parquet');m=metrics(f);recorded=next(x for x in r['metrics'] if x['split']=='test')
  assert abs(m['pooled_mse']-recorded['pooled_mse'])<1e-12 and abs(m['cross_sectional_rank_ic']-recorded['cross_sectional_rank_ic'])<1e-12
  ar=refs[(model,h)]
  for a in [ar['local_copy'],ar['source_copy']]:assert sha(ROOT/a['path'])==a['sha256']
  assert sha(ROOT/ar['original_csv'])==ar['original_csv_sha256']
  old=pd.read_parquet(ROOT/ar['local_copy']['path']);a,b=canonical(f),canonical(old)
  pd.testing.assert_frame_equal(a[['ticker','anchor_date','split','y_true']],b[['ticker','anchor_date','split','y_true']],check_exact=True)
  oldm=metrics(old);assert abs(oldm['pooled_mse']-ar['archived_test_mse'])<1e-12 and abs(oldm['cross_sectional_rank_ic']-ar['archived_mean_rank_ic'])<1e-12
  delta=float(np.max(np.abs(a.y_pred.to_numpy(np.float64)-b.y_pred.to_numpy(np.float64))))
  comparisons.append(dict(input_set='O',model=model,horizon=h,seed=42,classification='EXACT' if delta<1e-7 else 'DIFFERENT',max_abs_delta_pred=delta,delta_mse=m['pooled_mse']-oldm['pooled_mse'],new_12_12_test_mse=m['pooled_mse'],archived_test_mse=oldm['pooled_mse'],delta_mean_rank_ic=m['cross_sectional_rank_ic']-oldm['cross_sectional_rank_ic'],new_mean_rank_ic=m['cross_sectional_rank_ic'],archived_mean_rank_ic=oldm['cross_sectional_rank_ic'],training_reused=r['training_reused'],epochs=r['epochs'],best_epoch=r['best_epoch'],checkpoint=r['checkpoint'],completion=rel(cp),archived_original_csv=ar['original_csv']))
  primary.append(row(model,h,42,m,cp,cp.parent/'test.parquet','B5 pilot reused' if r['training_reused'] else 'B5FIX new seed42 training'))
  reference.append(row(model,h,42,oldm,None,ROOT/ar['local_copy']['path'],'archived MAIN V2 seed42','archived profile; not rerun'))
  telemetry.append(dict(job_id=spec['job_id'],training_reused=r['training_reused'],started_utc=r['started_utc'],completed_utc=r['completed_utc'],training_wall_seconds=r['wall_clock_seconds'],epochs=r['epochs'],best_epoch=r['best_epoch'],threads_start=str(r['thread_counts_start']),threads_end=str(r['thread_counts_end']),environment_start=json.dumps(r['environment_start']),environment_end=json.dumps(r['environment_end']),telemetry_samples=n,load_average_start=str(r['load_average_start']),load_average_end=str(r['load_average_end'])))
  files += [dict(job_id=spec['job_id'],**a) for a in r['artifacts']]
 # Reuse exactly the 36 B5 O cells for seeds0/1/21/3407; no C cells are loaded/trained.
 for seed in [0,1,21,3407]:
  for h in [5,10,20]:
   for model in ['lstm','vanilla','swim']:
    spec=oldio.spec_for(f'B5_O_{model}_dw_h{h}_s{seed}');cp,r=oldio.completed(spec);assert r;profile(r,cp)
    f=pd.read_parquet(cp.parent/'test.parquet');check_frame(f,ds,h,'test');m=metrics(f);saved=next(x for x in r['metrics'] if x['split']=='test')
    assert abs(m['pooled_mse']-saved['pooled_mse'])<1e-12 and abs(m['cross_sectional_rank_ic']-saved['cross_sectional_rank_ic'])<1e-12
    x=row(model,h,seed,m,cp,cp.parent/'test.parquet','B5 original other-seed CPU12/12')
    primary.append(x);reference.append(dict(x))
 assert len(primary)==len(reference)==45
 for data in [primary,reference]:
  for key,g in pd.DataFrame(data).groupby(['model','horizon']):assert len(g)==5 and set(g.seed)==set(SEEDS)
 unified=summary_code.summarize(primary);archived=summary_code.summarize(reference);assert len(unified)==len(archived)==9
 lookup={(r['model'],r['horizon']):r for r in archived};paired=[]
 for r in unified:
  old=lookup[(r['model'],r['horizon'])];x=dict(r)
  for k in ['test_mse_mean','test_mse_sd','mean_rank_ic_mean','mean_rank_ic_sd']:x['archived_seed42_'+k]=old[k]
  x['delta_five_seed_test_mse_mean']=r['test_mse_mean']-old['test_mse_mean'];x['delta_five_seed_mean_rank_ic']=r['mean_rank_ic_mean']-old['mean_rank_ic_mean'];paired.append(x)
 for name,data in [('seed42_archive_comparison.csv',comparisons),('O_uniform12_per_seed.csv',primary),('O_archived42_reference_per_seed.csv',reference),('O_uniform12_summary_5seed.csv',unified),('O_archived42_reference_summary_5seed.csv',archived),('O_summary_side_by_side.csv',paired),('queue_telemetry.csv',telemetry),('artifact_files.csv',files)]:csvout(EXP/name,data)
 protected=json.loads((session/'protected_inputs_verification.json').read_text());assert protected['passed']
 ev=[json.loads(x) for x in (session/'events.jsonl').read_text().splitlines() if x];launches=[x for x in ev if x['event']=='LAUNCH']
 assert len(launches)==8 and all(x['job_id']!=REUSE_ID for x in launches) and max(x['active_workers'] for x in launches)==3
 states={x['job_id']:x for x in q['states']};assert states[REUSE_ID]['status']=='SKIPPED_VERIFIED'
 n_different=sum(x['classification']=='DIFFERENT' for x in comparisons);n_exact=9-n_different
 blocking=(f'Historical archived MAIN V2 replay remains DIFFERENT in {n_different}/9 cells; see the table below. The causes are not established. All nine seed42 cells and all 36 reused other-seed cells pass the recorded CPU12/12 profile, artifact, label/key/count and same-run replay checks, so the primary five-seed O summary now uses one observed numerical profile. The historical reference is retained separately.' if n_different else 'None. All9 archived MAIN V2 comparisons are EXACT and both five-seed summaries have verified artifacts.')
 body='\n\n'.join(['# BLOCKING',blocking,'# Step14c-B5FIX — O seed42 at CPU12/12',f'Completed {utc()} UTC; exclusive session `{rel(EXP)}`. Original B5 report and all existing outputs remain read only. This report is the new uniform-profile O benchmark summary.','## Training and reuse',f'Nine seed42 model/horizon cells: eight newly trained with the exact imported B5 worker/14c-day baseline entrypoints and one verified B5 O LSTM h5 pilot reused. The reused checkpoint was strictly loaded and TRAIN/VAL/TEST re-inferred with the imported formal batched inference helper; all three predictions matched its saved B5 outputs bitwise. Its original training timestamps, history and telemetry are retained and marked reused, with no repeat training. Provenance: `{rel(EXP/"preflight.json")}` and its reused attempt/reuse_provenance.json.','O is the same frozen V2 input binding and train-only standardization as B5, verified by standardized tensor hashes. Formal model classes/factories, original set_seed(42), original global-RNG DataLoaders(generator=None), optimizer/dropout/batch/Huber/early-stopping/validation-best restoration are unchanged. No C model, other seed, patch search or other experiment was trained.','The imported14a scheduler refills to3 CPU workers while at least3 jobs remain, using intra12/inter12, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE before Torch import; BLAS overrides remain unset. AC power is checked and caffeinate covers preflight/reuse, queue and reporting. New journals/checkpoints/predictions/history/telemetry are exclusive, with hash-verified resume. All nine cells have30634/6290/6647 TRAIN/VAL/TEST rows, identical frozen ordered keys and labels, finite outputs, saved best checkpoints and per-epoch train/val Huber plus val MSE. The same-run restored-checkpoint TEST replays are EXACT.','## Seed42 vs archived MAIN V2',f'EXACT={n_exact}; DIFFERENT={n_different}. EXACT means max|delta prediction| <1e-7; no NEAR category. Delta MSE is new12/12 minus archived. References are the verified B5 copies reconstructed at original float32 neural-prediction precision from archived MAIN V2 CSVs (SWiM: swim_cpu), consistent with B5 comparisons; original CSV paths and hashes are retained in reference_manifest.json. No historical discrepancy is attributed to a cause without evidence.',table(['model','h','reused','classification','max absolute prediction delta','MSE delta','new TEST MSE','archived TEST MSE'],[[r['model'],r['horizon'],r['training_reused'],r['classification'],r['max_abs_delta_pred'],r['delta_mse'],r['new_12_12_test_mse'],r['archived_test_mse']] for r in comparisons]),'## Primary: all five seeds at CPU12/12','Five seeds42/0/1/21/3407. Seed42 uses the nine cells above; other seeds reuse the original36 O B5 cells. Every telemetry sample for all45 cells was checked for12/12 and the required environment. Formal TEST pooled MSE and mean date-level cross-sectional Spearman Rank IC are summarized as mean +/- sample SD(ddof1), without ensembling or new inferential tests.',summary_code.format_summary(unified),'## Original-paper reference: archived seed42','Same four B5 other-seed cells per group; seed42 uses the archived MAIN V2 values. This table preserves the original-paper reference and is not labeled as a uniform12/12 table.',summary_code.format_summary(archived),'## Runtime and files',f'Queue wall time before reporting={q["wall_clock_seconds"]:.3f} seconds. Eight worker launches; the h5 LSTM is SKIPPED_VERIFIED. All{protected["n_files"]} protected original files unchanged. Reused-cell training wall time/timestamps refer to its original B5 run; new queue runtime does not retrain it.',f'[Nine comparisons]({EXP}/seed42_archive_comparison.csv), [uniform-profile per-seed values]({EXP}/O_uniform12_per_seed.csv), [primary summary]({EXP}/O_uniform12_summary_5seed.csv), [archived42 reference per-seed values]({EXP}/O_archived42_reference_per_seed.csv), [reference summary]({EXP}/O_archived42_reference_summary_5seed.csv), [side-by-side summary]({EXP}/O_summary_side_by_side.csv), [telemetry]({EXP}/queue_telemetry.csv), [artifact hashes]({EXP}/artifact_files.csv).'])+'\n'
 textout(EXP/'REPORT_STEP14C_B5FIX.md',body);textout(report,body)
 jsonout(EXP/'FINAL_VALIDATION.json',dict(utc=utc(),cells=9,newly_trained=8,reused=1,worker_launches=8,checkpoints_available=9,prediction_parquets_available=27,all_cells_verified=True,all_keys_labels_counts_verified=True,all_same_run_replays_exact=True,reused_pilot_three_split_replay_bitwise_exact=True,uniform_profile_O_cells=45,reused_other_seed_B5_cells=36,primary_summary_groups=9,archived_reference_summary_groups=9,archive_EXACT=n_exact,archive_DIFFERENT=n_different,maximum_workers=3,protected_inputs_passed=True,report=rel(report),report_sha256=sha(report)))
 jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(report),artifact(EXP/'FINAL_VALIDATION.json')]))
 print('B5FIX_FINAL_COMPLETE',utc(),n_exact,n_different,flush=True)
if __name__=='__main__':generate(Path(sys.argv[1]))
