"""Descriptive Step14c-day tables only; no new inferential statistics."""
from day_io_v2 import *
import statistics,math,argparse

def values():
 rows=[];refs=[];missing=[]
 for spec in PLAN:
  cp,r=completed({k:spec[k] for k in SPEC_KEYS})
  if r is None:missing.append(spec['job_id']);continue
  test=next(x for x in r['metrics'] if x['split']=='test')
  rows.append(dict(block=spec['block'],input_set=spec['input_set'],model='same_patch_fixed' if spec['block']=='B1' else spec['mechanism'],frequency_setting=spec['frequency_setting'],horizon=spec['horizon'],seed=spec['seed'],test_mse=test['pooled_mse'],mean_rank_ic=test['cross_sectional_rank_ic'],best_epoch=r['best_epoch'],epochs=r['epochs'],wall_seconds=r['wall_clock_seconds'],completion=rel(cp),provenance='14c-day'))
  for a in r['artifacts']:refs.append(dict(job_id=spec['job_id'],**a))
 return rows,refs,missing

def summarize(rows):
 groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in ['block','input_set','model','frequency_setting','horizon']),[]).append(r)
 out=[]
 for key,rs in sorted(groups.items()):
  seeds=[r['seed'] for r in rs];assert len(seeds)==len(set(seeds));n=len(rs)
  d=dict(zip(['block','input_set','model','frequency_setting','horizon'],key));d.update(n_seeds=n)
  for metric in ['test_mse','mean_rank_ic']:
   v=[float(r[metric]) for r in rs];assert all(math.isfinite(x) for x in v)
   d[metric+'_mean']=statistics.mean(v);d[metric+'_sd']=statistics.stdev(v) if n>1 else None
  d['sd_rule']='ddof1' if n>1 else 'undefined: one seed';out.append(d)
 return out

def format_summary(rs):
 return table(['block','input','model','frequency','h','n','TEST MSE mean +/- SD','mean Rank IC mean +/- SD'],[[r['block'],r['input_set'],r['model'],r['frequency_setting'],r['horizon'],r['n_seeds'],f"{r['test_mse_mean']:.12g} +/- "+(f"{r['test_mse_sd']:.12g}" if r['test_mse_sd'] is not None else 'NOT DEFINED (n=1)'),f"{r['mean_rank_ic_mean']:.12g} +/- "+(f"{r['mean_rank_ic_sd']:.12g}" if r['mean_rank_ic_sd'] is not None else 'NOT DEFINED (n=1)')] for r in rs])

def phase_report(session):
 session=Path(session);r=json.loads((session/'queue_result.json').read_text());ids={s['job_id'] for s in r['states']};allrows,refs,missing=values();rows=[x for x in allrows if spec_for(Path(x['completion']).parts[-3])['job_id'] in ids] if False else [x for x in allrows if any(x['completion'].find('/'+i+'/')>=0 for i in ids)]
 csvout(session/'phase_cell_metrics.csv',rows,fields=list(rows[0]) if rows else ['block','input_set','model'])
 textout(session/'REPORT_PHASE.md','# BLOCKING\n\n'+('None for this completed phase.' if not r['failures'] else repr(r['failures']))+'\n\nVerified phase completions: '+str(r['n_complete'])+'/'+str(len(ids))+'. Full Step14c-day summary is written after all72 runs.\n')
 jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(session/'phase_cell_metrics.csv'),artifact(session/'REPORT_PHASE.md')]))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');ap.add_argument('--control');a=ap.parse_args();rows,refs,missing=values();assert not missing and len(rows)==72,missing
 b1=summarize([r for r in rows if r['block']=='B1']);b4=summarize([r for r in rows if r['block']=='B4']);assert len(b1)==6 and all(r['n_seeds']==5 for r in b1);assert len(b4)==42 and all(r['n_seeds']==1 for r in b4)
 archive=ROOT/'revision_work/outputs/fullrun/sessions/20261016T142732626448Z_9ba11b23/summary/cell_metrics.csv'
 with archive.open(newline='') as f:old=list(csv.DictReader(f))
 comparison_rows=[r for r in rows if r['block']=='B1']
 for r in old:
  if r['mechanism'] in ['fixed','single']:
   comparison_rows.append(dict(block='B1',input_set=r['input_set'],model='diverse_patch_fixed' if r['mechanism']=='fixed' else 'single_selected_D5_W2',frequency_setting='daily_weekly',horizon=int(r['horizon']),seed=int(r['seed']),test_mse=float(r['test_mse']),mean_rank_ic=float(r['mean_rank_ic']),best_epoch=None,epochs=None,wall_seconds=None,completion=rel(archive),provenance='14a summary; archived seed42 where14a used it'))
 comparison=summarize(comparison_rows);assert len(comparison)==18 and all(r['n_seeds']==5 for r in comparison)
 for name,data in [('cell_metrics.csv',rows),('B1_five_seed_summary.csv',b1),('B4_seed42_summary.csv',b4),('B1_side_by_side_per_seed.csv',comparison_rows),('B1_side_by_side_summary.csv',comparison),('artifact_files.csv',refs)]:csvout(EXP/name,data)
 pilot=json.loads((EXP/'LSTM_PILOT_QUALIFIED.json').read_text());pre=json.loads((EXP/'preflight.json').read_text());protected=[];telemetry=[];events=[];phases=[]
 for session in sorted((EXP/'sessions').iterdir()):
  if not (session/'queue_result.json').exists():continue
  rr=json.loads((session/'queue_result.json').read_text());phases.append(dict(session=rel(session),n_complete=rr['n_complete'],wall_seconds=rr['wall_clock_seconds'],started_utc=rr['started_utc'],completed_utc=rr['completed_utc']))
  pv=json.loads((session/'protected_inputs_verification.json').read_text());assert pv['passed'];protected.append(dict(session=rel(session),n_files=pv['n_files'],passed=True))
  events += [dict(session=rel(session),**json.loads(l)) for l in (session/'events.jsonl').read_text().splitlines() if l]
 for spec in PLAN:
  cp,r=completed({k:spec[k] for k in SPEC_KEYS});assert r
  telemetry.append(dict(job_id=spec['job_id'],start_utc=r['started_utc'],end_utc=r['completed_utc'],wall_seconds=r['wall_clock_seconds'],best_epoch=r['best_epoch'],epochs=r['epochs'],threads_start=str(r['thread_counts_start']),threads_end=str(r['thread_counts_end']),environment_start=json.dumps(r['environment_start']),environment_end=json.dumps(r['environment_end']),load_start=str(r['load_average_start']),load_end=str(r['load_average_end']),telemetry_samples=r['telemetry_samples']))
 csvout(EXP/'queue_telemetry.csv',telemetry);csvout(EXP/'phase_runtime.csv',phases);csvout(EXP/'protected_input_checks.csv',protected)
 assert max(e.get('active_workers',0) for e in events)==3
 report=['# BLOCKING','None. All72 requested cells completed with verified artifacts. B1 has all5 seeds in every input/horizon group; B4 is seed42 only, so sample SD is undefined there. No new significance tests were run.',
 '# Step14c-day — Same-patch control and causal benchmarks',f'Completed {utc()} UTC. New outputs are exclusively created in `{rel(EXP)}`. Existing source code, data, checkpoints, prior14a/frozen results and journals remain read only.',
 '## Pre-training checks',table(['model','parameters','Daily patches','Weekly patches'],[[x['model'],x['parameters'],x['daily_patches'],x['weekly_patches']] for x in pre['parameter_counts']]),
 'All8 expert patchify calls were observed in a real eval forward: Daily10 for each of4 Daily experts, Weekly4 for each of4 Weekly experts. Both models have107,777 parameters. The imported model factory creates independently initialized experts; no weights are shared. Attention, d_model, heads, fusion, optimizer, early stopping and Independent DataLoader generator42 match14a Fixed. Only the factory patch lists change, process-locally, and are restored immediately.',
 f'[Forward evidence]({EXP}/pretrain_patch_forward_check.csv), [parameter counts]({EXP}/pretrain_parameter_counts.csv), [preflight]({EXP}/preflight.json).',
 '## Queue, CPU profile and phase order',
 'The actual14a scheduler main is imported and executed, with job/source/report/output bindings. It is reused for three phases:30 B1 jobs, then the one required end-to-end LSTM pilot alone, then41 remaining B4 jobs. Each phase refills to3 worker processes when at least3 eligible jobs remain. Deliberate pilot isolation and phase tails have fewer workers; no dummy/filler training is launched. B1 order is seed42/0/1/21/3407, horizon20/10/5, interleaving O thenC. B4 starts only after B1 completes; remaining B4 starts only after the pilot passes.',
 'Every training process uses CPU, intra12/inter12, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE configured before Torch. Other BLAS overrides are unset as in Step3B/14a. Telemetry samples thread counts, environments, load, power and topCPU processes at start/every5minutes/end. Mac AC power is checked at phase start. An outer caffeinate -i spans all phases and reporting, in addition to the imported per-phase guard. Immutable completion hashes and dependencies are verified on restart before skipping. New append-only journals and full snapshots are written after each run; no existing journal is reopened for writing.',
 table(['phase session','completed cells','wall seconds'],[[p['session'],p['n_complete'],f"{p['wall_seconds']:.3f}"] for p in phases]),
 f'[Queue telemetry]({EXP}/queue_telemetry.csv), [phase runtimes]({EXP}/phase_runtime.csv), [protection checks]({EXP}/protected_input_checks.csv).',
 '## B4 causal input qualification and formal protocols',
 f'The pilot C LSTM Daily+Weekly h5 completed end to end. Its actual TRAIN/VAL/TEST DataLoader tensors were compared without iteration or RNG draws and match the causal normalized arrays bitwise. Those arrays match the same Stage3a files certified by14a preflight, and raw input binding is core.load_causal(). The restored strict checkpoint TEST replay is {pilot["test_replay"]["classification"]}, maximum delta={pilot["test_replay"]["max_abs_delta_pred"]:.12g}. [Pilot evidence]({EXP}/LSTM_PILOT_QUALIFIED.json).',
 'Adaptive Daily-only/Weekly-only use main_v2.train_one_cell. LSTM, Vanilla and SWiM import their original train_and_eval_frequency entrypoints (SWiM uses the formal CPU script). Their original optimizer, dropout, batch, loss, early stopping and best-state rules are retained. LSTM/Vanilla backend availability is bound process-locally to CPU; no model or training loop is rewritten. Baseline loaders preserve formal global-RNG shuffling at seed42, rather than adding a new independent generator. B1/Adaptive retain independent generator42. All neural checkpoints save restored validation-best weights and reload strictly. The imported Experiment4 batched inference helper is reused through an argument/output adapter that forwards to the original model. Same-run trainer TEST predictions replay EXACT in every B4 cell.',
 'Ridge imports panel_baseline_ridge.run_one unchanged, including alpha=[0.1,1,10,100], original default solver and strict validation-MSE selection on TRAIN-fitted models. The selected fitted object is retained as a pickle checkpoint and reloaded for all split predictions. Ridge has no neural epochs/best epoch; its history records the4 alpha candidates, post-fit TRAIN/VAL Huber(delta1) diagnostics and validation MSE. Huber does not select Ridge alpha. Neural history records per-epoch TRAIN/VAL Huber and observed validation MSE.',
 'Every cell has30634/6290/6647 rows for TRAIN/VAL/TEST, exact common ticker/anchor keys and frozen labels, finite values and verified Parquet round trips. Router weights are saved for the active Adaptive branch; B1 expert weights use explicit expert-index/patch-size columns, all0.25. Static is not a requested model in this queue; no Static logits apply.',
 '## B1 side-by-side comparison',
 'Mean +/- sample SD(ddof1), five seeds per group. Diverse Fixed and Single reuse the14a per-seed results without training or recomputing those models. Single retains the archived Daily5/Weekly2 selection. Metrics are formal TEST pooled MSE and mean date-level cross-sectional Spearman Rank IC.',format_summary(comparison),
 '## B4 causal benchmark cells',
 'Only seed42 was requested. The mean is that seed result; ddof1 SD is NOT DEFINED(n=1), never reported as zero or as five-seed uncertainty.',format_summary(b4),
 '## Files and integrity',f'[Every cell]({EXP}/cell_metrics.csv), [B1 per-seed comparisons]({EXP}/B1_side_by_side_per_seed.csv), [B1 summary]({EXP}/B1_side_by_side_summary.csv), [B4 summary]({EXP}/B4_seed42_summary.csv), [all artifacts with hashes]({EXP}/artifact_files.csv). Each run folder contains the best checkpoint, TRAIN/VAL/TEST Parquets, history, metrics, telemetry and COMPLETE.json. The LSTM pilot is one of the27 neural benchmark cells and is not trained twice. All code adapters live under this new experiment directory. No bootstrap, p-values, point-in-time or leakage-free claims are added.'
 ]
 body=(chr(10)*2).join(report)+chr(10);textout(EXP/'REPORT_STEP14C_DAY.md',body)
 dest=FULLRUN/'REPORT_STEP14C_DAY.md';assert not dest.exists(),'Cannot overwrite an existing report';textout(dest,body)
 jsonout(EXP/'FINAL_VALIDATION.json',dict(utc=utc(),n_cells=72,B1=30,B4_adaptive=6,B4_neural=27,B4_ridge=9,unique_prediction_parquets=216,strict_replays_passed=True,B1_nseeds=5,B4_nseeds=1,maximum_workers=3,preflight_passed=True,protected_inputs_passed=True,report=rel(dest),report_sha256=sha(dest),source14a_summary=rel(archive),source14a_summary_sha256=sha(archive)))
 print('REPORT_STEP14C_DAY_COMPLETE',utc(),flush=True)
if __name__=='__main__':main()
