"""Descriptive rolling-origin summary, with immutable W3 reuse; no inferential tests."""
from b2_io import *
import statistics

def number(x):
 if x in [None,'','nan','NaN']:return None
 v=float(x);return v if math.isfinite(v) else None

def display(rows):
 def val(m,s):return 'UNDEFINED (constant predictor)' if m is None else f'{m:.12g} +/- '+('UNDEFINED (n=1)' if s is None else f'{s:.12g}')
 return table(['window','input','model','h','seeds','TEST MSE mean +/- SD','mean Rank IC mean +/- SD','MSE / own Train-Mean MSE mean +/- SD'],[[r['window'],r['input_set'],r['model'],r['horizon'],r['n_seeds'],val(r['test_mse_mean'],r['test_mse_sd']),val(r['mean_rank_ic_mean'],r['mean_rank_ic_sd']),val(r['mse_relative_train_mean_mean'],r['mse_relative_train_mean_sd'])] for r in rows])

def generate(session):
 session=Path(session);ledger=json.loads((session/'queue_result.json').read_text());assert ledger['n_complete']==108 and not ledger['failures'] and not ledger['blocked']
 with (EXP/'global_train_mean_references.csv').open(newline='') as f:denom={(r['window'],int(r['horizon'])):r for r in csv.DictReader(f)}
 rows=[];files=[];telemetry=[];static=[]
 for spec in PLAN:
  cp,r=completed({k:spec[k] for k in KEYS});assert r,spec;test=next(m for m in r['metrics'] if m['split']=='test');base=float(denom[(spec['window'],spec['horizon'])]['test_mse_global_train_mean']);mse=test['pooled_mse']
  rows.append(dict(window=spec['window'],input_set=spec['input_set'],model=spec['mechanism'],horizon=spec['horizon'],seed=spec['seed'],test_mse=mse,mean_rank_ic=test['cross_sectional_rank_ic'],mse_relative_train_mean=mse/base,R2_OS_train_mean=1-mse/base,provenance=r['training_entrypoint'],predictions=rel(cp.parent/'test.parquet')))
  for ar in r['artifacts']:files.append(dict(job_id=spec['job_id'],**ar))
  telemetry.append(dict(job_id=spec['job_id'],window=spec['window'],input_set=spec['input_set'],model=spec['mechanism'],horizon=spec['horizon'],seed=spec['seed'],started_utc=r['started_utc'],completed_utc=r['completed_utc'],wall_seconds=r['wall_clock_seconds'],best_epoch=r['best_epoch'],epochs=r['epochs'],threads_start=str(r['thread_counts_start']),threads_end=str(r['thread_counts_end']),environment_start=json.dumps(r['environment_start']),environment_end=json.dumps(r['environment_end']),load_start=str(r['load_average_start']),load_end=str(r['load_average_end']),telemetry_samples=r['telemetry_samples'],completion=rel(cp)))
  if spec['mechanism']=='static':
   p=cp.parent/'static_logits.json';assert p.exists();static.append(dict(job_id=spec['job_id'],path=rel(p),sha256=sha(p)))
  if spec['mechanism']=='train_mean':assert mse==base
 with (EXP/'W3_reference_metrics.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   r['horizon']=int(r['horizon']);r['seed']=int(r['seed'])
   for field in ['test_mse','mean_rank_ic','mse_relative_train_mean','R2_OS_train_mean']:r[field]=number(r[field])
   rows.append(r)
 assert len(rows)==162
 source=json.loads((EXP/'W3_provenance.json').read_text());assert len(source['entries'])==54 and source['no_retraining']
 for e in source['entries']:
  assert sha(ROOT/e['source'])==e['source_sha256'];assert sha(ROOT/e['copied_test_predictions']['path'])==e['copied_test_predictions']['sha256']
 groups={}
 for r in rows:groups.setdefault((r['window'],r['input_set'],r['model'],r['horizon']),[]).append(r)
 summaries=[]
 for key,rs in sorted(groups.items()):
  model=key[2];expected=3 if model in ['static','adaptive'] else 1;assert len(rs)==expected and len({r['seed'] for r in rs})==expected
  if expected==3:assert {r['seed'] for r in rs}=={42,0,1}
  row=dict(zip(['window','input_set','model','horizon'],key));row['n_seeds']=expected
  for metric in ['test_mse','mean_rank_ic','mse_relative_train_mean','R2_OS_train_mean']:
   values=[number(r[metric]) for r in rs]
   if all(v is None for v in values):mu=sd=None
   else:assert all(v is not None for v in values);mu=statistics.mean(values);sd=statistics.stdev(values) if expected>1 else None
   row[metric+'_mean']=mu;row[metric+'_sd']=sd
  summaries.append(row)
 assert len(summaries)==90
 for name,data in [('all_cell_metrics.csv',rows),('window_summary.csv',summaries),('artifact_files.csv',files),('queue_telemetry.csv',telemetry),('static_logit_files.csv',static)]:csvout(EXP/name,data)
 protected=json.loads((session/'protected_inputs_verification.json').read_text());assert protected['passed'],protected
 ds=design();ranges=[]
 with (EXP/'design_split_ranges.csv').open(newline='') as f:ranges=list(csv.DictReader(f))
 preflight=json.loads((EXP/'preflight.json').read_text());assert preflight['passed']
 for d in preflight['dependencies']:assert sha(ROOT/d['path'])==d['sha256']
 events=[json.loads(line) for line in (session/'events.jsonl').read_text().splitlines() if line];assert max(e.get('active_workers',0) for e in events)==3
 text=['# BLOCKING','None. All72 rolling-window neural runs and36 reference cells completed. W3 was reused without training. All split, tensor, source, count/key/label and protection checks passed. Static/Adaptive summaries use exactly seeds42/0/1; each deterministic/Ridge reference is one fitted/constructed predictor, with undefined ddof1 seed SD.',
 '# Step14c-B2 — Rolling-origin evaluation',f'Completed {utc()} UTC. Exclusive new output namespace: `{rel(EXP)}`. Existing code/data/checkpoints/results and old journals were read only. The original14a scheduler main, complete formal trainers/factories, original panel-record/weekly builders, causal functions and metric code are imported. No model or training loop was reimplemented.',
 '## Design check written BEFORE training',f'The immutable [pre-training design report]({EXP}/DESIGN_REPORT_STEP14C_B2.md), [design JSON]({EXP}/design.json) and [complete split CSV]({EXP}/design_split_ranges.csv) were created during preparation, before the queue launched. Its SHA256 is `{sha(EXP/"DESIGN_REPORT_STEP14C_B2.md")}`.',
 table(['window','h','part','first anchor','last anchor','anchors','rows'],[[r['window'],r['horizon'],r['split'],r['first_anchor'],r['last_anchor'],r['n_anchors'],r['n_rows']] for r in ranges]),
 'The indexing calendar is the original17-ticker common eligible2603-anchor calendar BEFORE split-specific purging. It includes the40 eligible dates omitted only by W3 purges; those are not permanently excluded from earlier windows. All noninitial W3 boundaries move back411 for W2 and822 for W1; TRAIN starts at the same earliest eligible anchor and expands. VAL has370 anchors; TEST391; both gaps20. The W1/W2/W3 TEST sets are pairwise disjoint. The dates/counts above apply to both O/C and all horizons. W3 round-trips bitwise to the existing original and causal panel bindings and is never retrained.',
 f'[Index assignments]({EXP}/design_calendar.csv), [test nonoverlap]({EXP}/design_test_nonoverlap.csv), [per-ticker horizon-label boundary checks]({EXP}/purge_label_boundary_checks.csv). The boundary check verifies each last TRAIN target date precedes the first VAL anchor and each last VAL target date precedes the first TEST anchor.',
 '## Input construction and train-only standardization',
 'O selects unchanged sample windows/labels from imported original V2 panel bundles, including anchors purged only in W3. C is rebuilt separately for each window with the imported Stage3a load_prices/windows/transform functions on the ORIGINAL snapshot. Both Daily90 and Weekly26 windows match original float32 inputs before transformation; actual weekly availability remains unchanged and source endpoints do not exceed anchors. Rebuilt W3 causal tensors equal core.load_causal() bitwise. C per-anchor formulas do not depend on split membership; its standardization changes with each window TRAIN.',
 'For every window/input, common.fit_train_only_normalizer recomputes ticker x resolution x channel mean/std over that window TRAIN window entries, ddof0, retaining the original near-zero-std guard. No W3 statistics or VAL/TEST rows enter W1/W2 fitting. The imported normalizer is used again by each original trainer. New raw/standardized tensors, frozen labels, keys and170 normalization rows per window/input are retained and hash-bound.',
 f'[Normalizer statistics]({EXP}/normalization_stats.csv), [tensor manifest]({EXP}/tensor_manifest.json), [window/availability reconstruction checks]({EXP}/feature_availability_checks.csv), [preflight]({EXP}/preflight.json).',
 '## Trainers, execution and artifacts',
 'Seed42 Static uses Experiment3 train_and_validate_cell; seed42 Adaptive uses main_v2.train_one_cell; seeds0/1 use Experiment4 train_and_validate_cell. The same original loops/factories retain Adam1e-4, batch32, Huber delta1, clip1, max100, patience10, validation-best selection/restoration and independent DataLoader generator42. Only process-local dataset/count/date-contract bindings change for the rolling windows; Experiment4 train-count assertion is rebound to the window design. The original formal W3 contract is checked unchanged during preparation, and the rolling validator enforces the new manifest, universe, shapes, keys, labels, counts and dates. VAL/TEST counts remain6290/6647.',
 'Every process uses CPU intra12/inter12 with OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE before importing Torch; other Step3B BLAS overrides are unset. The imported14a scheduler refills3 workers as cells finish, in seed42/0/1 and horizon20/10/5 order, interleaving W1/W2 and O/C, Static then Adaptive. The36 requested Ridge/constant reference cells follow the neural queue. Tail underfill is logged; no filler jobs are created. AC is required before launch; caffeinate -i covers waiting, training, validation and reporting. Telemetry records start/end UTC, wall time, epochs/best epoch, threads, env, load and topCPU processes, at start/every5minutes/end.',
 'Every new cell saves a best-state checkpoint or fitted Ridge pickle/constant JSON, TRAIN/VAL/TEST predictions, applicable per-sample router weights and Static logits, history and immutable COMPLETE.json. Restored neural states are strictly reloaded before imported batched inference. Ridge uses imported run_one with the original4 alphas0.1/1/10/100 and validation-MSE choice; no TRAIN+VAL refit is added. Constant baselines use0 or each window own float32 TRAIN-label mean. Neural history includes train/val Huber and val MSE; Ridge history lists alpha candidates with post-fit losses; constants have no neural epochs. All predictions have design-matched rows, exact ticker/date keys and frozen labels, finite values and Parquet round trips.',
 f'New cell artifacts:324 prediction Parquets,72 restored neural checkpoints,12 selected Ridge pickles,24 constant JSON checkpoints and36 Static-logit files. [All artifact paths/hashes]({EXP}/artifact_files.csv), [telemetry]({EXP}/queue_telemetry.csv), [Static logits]({EXP}/static_logit_files.csv), [status journal reference]({session}/status_reference.json), [launch/exit events]({session}/events.jsonl), [queue result]({session}/queue_result.json), [protected-input verification]({session}/protected_inputs_verification.json). Queue wall time before reporting={ledger["wall_clock_seconds"]:.3f} seconds; {protected["n_files"]} protected pre-existing files unchanged.',
 '## W3 reference provenance — no retraining',
 'W3 O Static/Adaptive reuse the archived formal TEST predictions: Experiment3 Static seed42, MAIN V2 Adaptive seed42, and Experiment4 seeds0/1. W3 C Static/Adaptive reuse14a12/12 outputs for the same3 seeds. W3 O Ridge uses its archived MAIN V2 predictions; W3 C Ridge uses14c-day12/12 Daily+Weekly predictions. Zero and Global Train-Mean reuse archived deterministic predictions for both input sets, after verifying that O/C labels and TRAIN means are identical.54 W3 cells have source hashes, copied canonical TEST rows and exact key/label checks. W3 models are not trained or inferred anew.',
 f'[W3 full source provenance]({EXP}/W3_provenance.json), [W3 per-cell metrics]({EXP}/W3_reference_metrics.csv).',
 '## Descriptive comparison with W3',
 'Each cell score uses imported formal pooled TEST MSE and mean within-date cross-sectional Spearman Rank IC. Static/Adaptive show mean +/- sample SD(ddof1) over exactly seeds42/0/1, without averaging predictions into an ensemble. Ridge, Zero and Train-Mean have1 predictor per window/input/horizon; their seed SD is undefined, not0. Rank IC of a constant predictor is undefined and is not replaced by0.',
 'Relative MSE means TEST_MSE(model) / TEST_MSE(Global Train-Mean of the SAME window and horizon). Below1 favors the model. The companion R2_OS_train_mean=1-relative_MSE is retained in CSV. The denominator uses each window own TRAIN-label mean, never W3 mean for W1/W2. Means/SDs of relative MSE are computed across the same paired seed scores. No bootstrap, p-values, significance tests, point-in-time or leakage-free claims are added.',
 display(summaries),
 f'[Full precision90-row summary]({EXP}/window_summary.csv), [all162 per-cell scores]({EXP}/all_cell_metrics.csv), [window-specific Train-Mean denominators]({EXP}/global_train_mean_references.csv).'
 ]
 body=(chr(10)*2).join(text)+chr(10);textout(EXP/'REPORT_STEP14C_B2.md',body);rootreport=FULLRUN/'REPORT_STEP14C_B2.md';textout(rootreport,body)
 jsonout(EXP/'FINAL_VALIDATION.json',dict(utc=utc(),new_neural_runs=72,new_reference_cells=36,W3_reused_cells=54,W3_training_runs=0,all_metric_rows=162,summary_rows=90,new_prediction_parquets=324,W3_canonical_TEST_parquets=54,test_ranges_disjoint=True,all_horizon_target_boundary_checks=True,train_only_normalization_per_window=True,all_new_cells_verified=True,maximum_workers=3,profile='CPU12/12',protected_files=protected['n_files'],protected_files_unchanged=True,report=rel(rootreport),report_sha256=sha(rootreport)))
 jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(EXP/'window_summary.csv'),artifact(EXP/'all_cell_metrics.csv'),artifact(rootreport),artifact(EXP/'FINAL_VALIDATION.json')]))
 print('REPORT_STEP14C_B2_COMPLETE',utc(),flush=True)
if __name__=='__main__':
 import sys;generate(sys.argv[1])
