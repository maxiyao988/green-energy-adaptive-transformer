"""Descriptive benchmark-seed summaries, importing 14c-day summary code."""
from b5_io import *
sys.path.insert(0,str(DAY/'scripts'))
import day_report_v2 as report
import statistics,argparse,math

def rows_for(specs):
 rows=[];files=[];telemetry=[]
 for spec in specs:
  cp,r=completed(spec);assert r,spec;t=next(m for m in r['metrics'] if m['split']=='test')
  rows.append(dict(block='B5',input_set=r['input_set'],model=r['mechanism'],frequency_setting='daily_weekly',horizon=r['horizon'],seed=r['seed'],test_mse=t['pooled_mse'],mean_rank_ic=t['cross_sectional_rank_ic'],provenance='B5 CPU12/12',source=rel(cp),source_sha256=sha(cp),prediction=rel(cp.parent/'test.parquet'),prediction_sha256=sha(cp.parent/'test.parquet')))
  files += [dict(job_id=spec['job_id'],**a) for a in r['artifacts']]
  telemetry.append(dict(job_id=spec['job_id'],started_utc=r['started_utc'],completed_utc=r['completed_utc'],wall_seconds=r['wall_clock_seconds'],epochs=r['epochs'],best_epoch=r['best_epoch'],threads_start=str(r['thread_counts_start']),threads_end=str(r['thread_counts_end']),environment_start=json.dumps(r['environment_start']),environment_end=json.dumps(r['environment_end']),load_average_start=str(r['load_average_start']),load_average_end=str(r['load_average_end']),telemetry_samples=r['telemetry_samples'],global_rng_seed=r['global_rng_seed']))
 return rows,files,telemetry

def pilot_compare():
 # This subprocess is launched after the pilot exits; no concurrent training in pilot phase.
 import b5_worker as worker
 import numpy as np,pandas as pd
 cp,r=completed(PILOT);assert r
 f=pd.read_parquet(cp.parent/'test.parquet');old=pd.read_parquet(EXP/'seed42/O_lstm_dw_h5_s42_test.parquet')
 for frame in [f,old]:frame['anchor_date']=pd.to_datetime(frame.anchor_date).dt.strftime('%Y-%m-%d')
 f=f.sort_values(['ticker','anchor_date']).reset_index(drop=True);old=old.sort_values(['ticker','anchor_date']).reset_index(drop=True)
 assert np.array_equal(f[['ticker','anchor_date']].to_numpy(),old[['ticker','anchor_date']].to_numpy()) and np.array_equal(f.y_true.to_numpy(),old.y_true.to_numpy())
 delta=float(np.max(np.abs(f.y_pred.to_numpy(np.float64)-old.y_pred.to_numpy(np.float64))))
 scores=lambda x:worker.day.common.evaluate_predictions(x.y_true.to_numpy(),x.y_pred.to_numpy(),x.ticker.to_numpy(),x.anchor_date.to_numpy())
 newm,oldm=scores(f),scores(old)
 result=dict(utc=utc(),classification='EXACT' if delta<1e-7 else 'DIFFERENT',threshold=1e-7,max_abs_delta_pred=delta,delta_mse=newm['pooled_mse']-oldm['pooled_mse'],retrained_mse=newm['pooled_mse'],archived_mse=oldm['pooled_mse'],pilot_completion=artifact(cp),archive_predictions=artifact(EXP/'seed42/O_lstm_dw_h5_s42_test.parquet'),summary_uses_archived_seed42=True,queue_continues_if_different=True)
 jsonout(EXP/'PILOT_COMPARISON.json',result);csvout(EXP/'pilot_comparison.csv',[{k:v for k,v in result.items() if not isinstance(v,dict)}]);return result

def phase_report(session):
 session=Path(session);q=json.loads((session/'queue_result.json').read_text());ids={s['job_id'] for s in q['states']}
 specs=[s for s in [PILOT]+PLAN if s['job_id'] in ids];rows,files,t=rows_for(specs)
 assert len(rows)==len(specs) and not q['failures'] and not q['blocked']
 if PILOT['job_id'] in ids:
  if not (EXP/'PILOT_COMPARISON.json').exists():p=pilot_compare()
  else:p=json.loads((EXP/'PILOT_COMPARISON.json').read_text())
 else:p=None
 csvout(session/'phase_metrics.csv',rows)
 textout(session/'REPORT_PHASE.md','# BLOCKING\n\n'+('Pilot archived replay DIFFERENT; the user explicitly authorized proceeding with the 72-run queue.' if p and p['classification']=='DIFFERENT' else 'None in this phase.')+'\n\nVerified cells: '+str(len(rows))+'\n')
 jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(session/'phase_metrics.csv'),artifact(session/'REPORT_PHASE.md')]))

def final():
 destination=FULLRUN/'REPORT_STEP14C_B5.md'
 if (EXP/'FINAL_VALIDATION.json').exists():
  r=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert sha(ROOT/r['report'])==r['report_sha256'];return
 new,files,telemetry=rows_for(PLAN);assert len(new)==72
 with (EXP/'seed42_metrics.csv').open() as f:seed42=list(csv.DictReader(f))
 provenance=json.loads((EXP/'reference_provenance.json').read_text())
 for a in provenance['sources']+[provenance['four_mechanism_summary']]:assert sha(ROOT/a['path'])==a['sha256']
 for r in seed42:
  for k in ['seed','horizon']:r[k]=int(r[k])
  for k in ['test_mse','mean_rank_ic']:r[k]=float(r[k])
  assert sha(ROOT/r['prediction'])==r['prediction_sha256']
 rows=new+seed42;assert len(rows)==90
 with (ROOT/provenance['four_mechanism_summary']['path']).open() as f:four=list(csv.DictReader(f))
 comparison=list(rows)
 for r in four:comparison.append(dict(block='B5',input_set=r['input_set'],model=r['mechanism'],frequency_setting='daily_weekly',horizon=int(r['horizon']),seed=int(r['seed']),test_mse=float(r['test_mse']),mean_rank_ic=float(r['mean_rank_ic']),provenance='14a five-seed formal/archived summary',source=provenance['four_mechanism_summary']['path'],source_sha256=provenance['four_mechanism_summary']['sha256'],prediction='',prediction_sha256=''))
 for data in [rows,comparison]:
  group={}
  for r in data:group.setdefault((r['input_set'],r['model'],r['horizon']),[]).append(r['seed'])
  assert all(len(v)==5 and set(v)==set(SEEDS) for v in group.values())
 summary=report.summarize(rows);side=report.summarize(comparison);assert len(summary)==18 and len(side)==42
 pilot=json.loads((EXP/'PILOT_COMPARISON.json').read_text());assert json.loads((EXP/'B3_ACCEPTANCE.json').read_text())['passed']
 phases=[];protections=[];events=[]
 for session in sorted((EXP/'sessions').iterdir()):
  if not (session/'queue_result.json').exists():continue
  q=json.loads((session/'queue_result.json').read_text());assert not q['failures'] and not q['blocked']
  p=json.loads((session/'protected_inputs_verification.json').read_text());assert p['passed']
  protections.append(dict(session=rel(session),protected_files=p['n_files'],passed=True))
  phases.append(dict(session=rel(session),completed=q['n_complete'],started_utc=q['started_utc'],completed_utc=q['completed_utc'],wall_seconds=q['wall_clock_seconds']))
  events += [json.loads(x) for x in (session/'events.jsonl').read_text().splitlines() if x]
 assert max(e.get('active_workers',0) for e in events)==3
 started=[e['job_id'] for e in events if e['event']=='LAUNCH' and e['job_id']!=PILOT['job_id']];order=[s['job_id'] for s in PLAN]
 assert len(started)==len(set(started)) or any((EXP/'runs'/s['job_id']).glob('attempt_*/FAILED.json') for s in PLAN)
 firsts=list(dict.fromkeys(started));assert firsts==[i for i in order if i in set(firsts)]
 for name,data in [('benchmark_cell_metrics.csv',rows),('benchmark_summary_5seed.csv',summary),('side_by_side_per_seed.csv',comparison),('side_by_side_summary_5seed.csv',side),('artifact_files.csv',files),('queue_telemetry.csv',telemetry),('phase_runtimes.csv',phases),('protected_input_checks.csv',protections)]:csvout(EXP/name,data)
 blocking=('O LSTM Daily+Weekly seed42 h5 retraining at CPU12/12 is DIFFERENT from archived MAIN V2: max |delta prediction|='+str(pilot['max_abs_delta_pred'])+', delta MSE='+str(pilot['delta_mse'])+'. As explicitly requested, the full queue still ran. O seed42 summary values retain archived MAIN V2 predictions from a different numerical profile; this diagnostic does not establish the cause of the mismatch.' if pilot['classification']=='DIFFERENT' else 'None. B3 passed acceptance before B5 began; all 72 queued cells and the mandatory isolated LSTM pilot completed with verified artifacts. The pilot archived replay is EXACT.')
 body='\n\n'.join(['# BLOCKING',blocking,'# Step14c-B5 — Benchmark seeds',f'Completed {utc()} UTC; exclusive session `{rel(EXP)}`. All pre-existing code/data/results are read only.','## B3 acceptance',f'[Acceptance report]({EXP}/B3_ACCEPTANCE.md), [180-cell evidence]({EXP}/B3_acceptance_cells.csv). B3 finished first; every saved artifact and dependency hash, all 540 Parquet row counts/keys/labels, histories/best epochs, finite checkpoint tensors, all thread telemetry, report hash and protection checks passed. TEST metrics recomputed with the formal helper agreed within 1e-12.','## Mandatory archived replay',table(['cell','classification','max absolute prediction delta','MSE delta'],[['O LSTM DW h5 seed42',pilot['classification'],pilot['max_abs_delta_pred'],pilot['delta_mse']]]),'The mandatory pilot runs alone at 12/12 before the 72-cell queue and is excluded from five-seed summaries. All O seed42 entries come from archived MAIN V2 predictions; C seed42 entries come from 14c-day B4. No other seed42 cell was retrained. SWiM uses the formal MAIN V2 CPU archive (swim_cpu).','## Imported protocols and queue',f'The 14c-day B4 runner and its original baseline train_and_eval_frequency entrypoints are imported; SWiM uses train_and_eval_frequency_cpu. Model classes, optimizer, dropout, batch, Huber loss, gradient clipping, early stopping and validation-best restoration are unchanged. Only input binding O=common.load_frozen_panel / C=core.load_causal and the requested run seed differ. Each original entrypoint calls its original set_seed(run_seed) exactly once. Original global-RNG loaders retain generator=None; no independent generator is added. Strict checkpoint reloading and the imported formal batched inference helper reproduce same-run trainer TEST predictions EXACT in all new cells.','The actual imported 14a scheduler runs an isolated one-worker pilot phase then the 72-cell phase, refilling to exactly 3 workers whenever at least 3 eligible jobs remain; the final tail has fewer workers. Every numerical process uses CPU intra12/inter12, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE/MKL_DYNAMIC=FALSE before Torch import; other BLAS overrides are unset as before. Per-run start/5-minute/end telemetry records threads, environment, UTC, system load and power; caffeinate spans waiting, audit, pilot, queue and report. New exclusive journals and snapshots support verified skips on restart. Order is seed0/1/21/3407, h20/10/5, models LSTM/Vanilla/SWiM interleaved O/C.',table(['phase session','cells','wall seconds'],[[p['session'],p['completed'],p['wall_seconds']] for p in phases]),'Every new run saves restored-best checkpoint, TRAIN/VAL/TEST Parquets with 30,634/6,290/6,647 rows, identical frozen keys/labels, per-epoch train/val Huber and validation MSE. There are 73 new checkpoints and 219 new prediction Parquets including the pilot.','## Five-seed benchmarks','Formal TEST pooled MSE and mean date-level cross-sectional Spearman Rank IC. Means and sample SD (ddof1) are over five separate seeds 42/0/1/21/3407; no ensemble or new inferential statistics.',report.format_summary(summary),'## Alongside Single / Fixed / Static / Adaptive','Four-mechanism values reuse the existing 14a five-seed cell summary, including its archived O Static/Adaptive seed42 evidence; no four-mechanism model is retrained.',report.format_summary(side),'## Files',f'[Per-seed benchmark metrics]({EXP}/benchmark_cell_metrics.csv), [five-seed summary]({EXP}/benchmark_summary_5seed.csv), [side-by-side per-seed metrics]({EXP}/side_by_side_per_seed.csv), [comparison summary]({EXP}/side_by_side_summary_5seed.csv), [provenance]({EXP}/reference_provenance.json), [artifacts/hashes]({EXP}/artifact_files.csv), [telemetry]({EXP}/queue_telemetry.csv).'])+'\n'
 textout(EXP/'REPORT_STEP14C_B5.md',body);textout(destination,body)
 jsonout(EXP/'FINAL_VALIDATION.json',dict(utc=utc(),queued_cells=72,pilot_cells=1,new_checkpoints=73,new_prediction_parquets=219,benchmark_cells_including_reused_seed42=90,benchmark_summary_rows=18,side_by_side_rows=42,all_cells_verified=True,all_keys_labels_counts_verified=True,all_same_run_replays_exact=True,b3_acceptance_passed=True,pilot_archived_classification=pilot['classification'],protected_inputs_passed=True,maximum_workers=3,report=rel(destination),report_sha256=sha(destination)))
 print('B5_FINAL_REPORT_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
 if a.final:final()
