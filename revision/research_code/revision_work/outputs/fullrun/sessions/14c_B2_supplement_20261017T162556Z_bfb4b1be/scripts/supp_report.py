"""Five-seed rolling-origin summary plus unchanged three-seed comparison."""
from supp_io import *
from b2_report import display,number
import statistics

def summarize(rows,seeds):
 groups={}
 for r in rows:
  if r['model'] in ['static','adaptive'] and r['seed'] not in seeds:continue
  groups.setdefault((r['window'],r['input_set'],r['model'],r['horizon']),[]).append(r)
 result=[]
 for key,rs in sorted(groups.items()):
  neural=key[2] in ['static','adaptive'];expected=len(seeds) if neural else 1
  assert len(rs)==expected and len({r['seed'] for r in rs})==expected,(key,len(rs))
  if neural:assert {r['seed'] for r in rs}==set(seeds)
  row=dict(zip(['window','input_set','model','horizon'],key));row['n_seeds']=expected
  for metric in ['test_mse','mean_rank_ic','mse_relative_train_mean','R2_OS_train_mean']:
   values=[number(r[metric]) for r in rs]
   if all(v is None for v in values):mu=sd=None
   else:
    assert all(v is not None for v in values);mu=statistics.mean(values);sd=statistics.stdev(values) if expected>1 else None
   row[metric+'_mean']=mu;row[metric+'_sd']=sd
  result.append(row)
 assert len(result)==90
 return result

def generate(session):
 session=Path(session);ledger=json.loads((session/'queue_result.json').read_text());assert ledger['n_complete']==156 and not ledger['failures'] and not ledger['blocked']
 rootreport=FULLRUN/'REPORT_STEP14C_B2_5SEED.md'
 if (EXP/'FINAL_VALIDATION.json').exists():
  v=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert sha(rootreport)==v['report_sha256']
  jsonout(session/'report_manifest.json',dict(utc=utc(),verified_existing_report=True,files=[artifact(rootreport),artifact(EXP/'FINAL_VALIDATION.json')]))
  return
 with (OLD/'global_train_mean_references.csv').open(newline='') as f:denom={(r['window'],int(r['horizon'])):float(r['test_mse_global_train_mean']) for r in csv.DictReader(f)}
 rows=[];files=[];telemetry=[];new_records=[]
 for spec in PLAN:
  cp,r=completed({k:spec[k] for k in KEYS});assert r
  t=next(m for m in r['metrics'] if m['split']=='test');mse=t['pooled_mse'];d=denom[(spec['window'],spec['horizon'])]
  rows.append(dict(window=spec['window'],input_set=spec['input_set'],model=spec['mechanism'],horizon=spec['horizon'],seed=spec['seed'],test_mse=mse,mean_rank_ic=t['cross_sectional_rank_ic'],mse_relative_train_mean=mse/d,R2_OS_train_mean=1-mse/d,provenance=r['training_entrypoint'],predictions=rel(cp.parent/'test.parquet')))
  if spec['mechanism']=='train_mean':assert mse==d
  for a in r['artifacts']:files.append(dict(job_id=spec['job_id'],cell_origin='original B2' if spec['job_id'] in OLD_IDS else 'supplement',**a))
  if spec['job_id'] in OLD_IDS:continue
  new_records.append(r)
  telemetry.append(dict(job_id=spec['job_id'],window=spec['window'],input_set=spec['input_set'],model=spec['mechanism'],horizon=spec['horizon'],seed=spec['seed'],started_utc=r['started_utc'],completed_utc=r['completed_utc'],wall_seconds=r['wall_clock_seconds'],epochs=r['epochs'],best_epoch=r['best_epoch'],threads_start=str(r['thread_counts_start']),threads_end=str(r['thread_counts_end']),environment_start=json.dumps(r['environment_start']),environment_end=json.dumps(r['environment_end']),load_start=str(r['load_average_start']),load_end=str(r['load_average_end']),telemetry_samples=r['telemetry_samples'],completion=rel(cp)))
 assert len(new_records)==48
 with (EXP/'W3_reference_metrics.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   r['horizon']=int(r['horizon']);r['seed']=int(r['seed'])
   for field in ['test_mse','mean_rank_ic','mse_relative_train_mean','R2_OS_train_mean']:r[field]=number(r[field])
   rows.append(r)
 assert len(rows)==234
 five=summarize(rows,[42,0,1,21,3407]);three=summarize(rows,[42,0,1])
 # This exact numerical equality establishes that every original three-seed cell/score was kept.
 with (OLD/'window_summary.csv').open(newline='') as f:previous=list(csv.DictReader(f))
 for a,b in zip(three,previous):
  assert all(str(a[k])==b[k] for k in ['window','input_set','model','horizon','n_seeds'])
  for k in a:
   if k not in ['window','input_set','model','horizon','n_seeds']:assert number(a[k])==number(b[k]),(k,a,b)
 frozen=json.loads((EXP/'ORIGINAL_REUSE.json').read_text())
 for a in frozen['original_tree_files']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
 w3=json.loads((EXP/'W3_provenance.json').read_text());assert len(w3['entries'])==78 and w3['no_retraining']
 for e in w3['entries']:
  assert sha(ROOT/e['source'])==e['source_sha256'];a=e['copied_test_predictions'];assert sha(ROOT/a['path'])==a['sha256']
 protected=json.loads((session/'protected_inputs_verification.json').read_text());assert protected['passed']
 events=[json.loads(x) for x in (session/'events.jsonl').read_text().splitlines() if x]
 launches=[e for e in events if e['event']=='LAUNCH'];assert all(e['seed'] in [21,3407] and e['job_id'] not in OLD_IDS for e in launches)
 assert max(e.get('active_workers',0) for e in events)==3
 assert [e['job_id'] for e in launches]==[s['job_id'] for s in NEW] or len(launches)<48 # Resume can skip verified supplementary runs.
 for name,data in [('all_cell_metrics_5seed.csv',rows),('window_summary_5seed.csv',five),('window_summary_3seed.csv',three),('artifact_files.csv',files),('supplement_telemetry.csv',telemetry)]:csvout(EXP/name,data)
 ranges=[]
 with (OLD/'design_split_ranges.csv').open(newline='') as f:ranges=list(csv.DictReader(f))
 text=['# BLOCKING','None. All48 supplementary neural cells completed and verified; the original108 cells (including all36 Ridge/constant baselines) were verified and skipped. All five seeds are present for Static and Adaptive in W1/W2/W3. The three-seed summary is numerically identical to the preserved original report/CSV.',
 '# Step14c-B2 supplement — five-seed rolling-origin evaluation',f'Completed {utc()} UTC. New exclusive output session: `{rel(EXP)}`. The original [three-seed report]({FULLRUN}/REPORT_STEP14C_B2.md) remains unchanged.',
 '## Design and exact reuse',f'The already-built W1/W2 tensors, per-window TRAIN-only normalizers, labels, weekly availability, purge masks and original design are imported by the same B2 load_panel/window_contract binding. No window or tensor was rebuilt. The source [tensor manifest]({OLD}/tensor_manifest.json) SHA256 is `{sha(OLD/"tensor_manifest.json")}`. [Original pre-training design]({OLD}/DESIGN_REPORT_STEP14C_B2.md).',
 table(['window','h','part','first anchor','last anchor','anchors','rows'],[[r['window'],r['horizon'],r['split'],r['first_anchor'],r['last_anchor'],r['n_anchors'],r['n_rows']] for r in ranges]),
 'TEST sets remain pairwise disjoint; each window retains370 VAL anchors,391 TEST anchors and20-anchor TRAIN/VAL and VAL/TEST purges. TRAIN has16660 rows in W1,23647 in W2 and30634 in W3; VAL/TEST have6290/6647. Original split definitions, ddof0 standardization and horizon-label timing checks are unchanged.',
 f'All108 previous B2 cells were verified before launch and marked SKIPPED_VERIFIED by the imported14a scheduler. The complete original B2 tree ({len(frozen["original_tree_files"])} files, plus the root report as listed) was hashed before launch and verified unchanged after completion. [Reuse/protection evidence]({EXP}/ORIGINAL_REUSE.json). No seed42/0/1 cell or Ridge/Zero/Train-Mean predictor was retrained or re-inferred.',
 '## Supplement execution and artifacts',
 'Exactly48 new Static/Adaptive Daily+Weekly cells use the complete imported B2/14a worker and Experiment4 train_and_validate_cell. The only additions are seeds21 and3407. Hyperparameters, factory, optimizer, Huber loss, validation-best selection/restoration, early stopping and independent DataLoader generator42 are unchanged. Process-local window contract/count bindings are identical to the initial B2 adapter.',
 'Order is seed21 then3407; horizon20/10/5; interleaved W1/W2 and O/C, Static then Adaptive. The imported14a scheduler maintains at most3 workers and refills each free slot. CPU intra12/inter12, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE are set before importing Torch. Per-run start/end and five-minute telemetry records threads, env vars, power, load averages and top CPU processes. caffeinate -i stays active through reporting.',
 f'New saved artifacts:48 restored best-state checkpoints,144 TRAIN/VAL/TEST prediction Parquets (including applicable8 router/Static weights),24 Static-logit JSONs,48 histories with train/val Huber and val MSE, and per-run telemetry. Strict checkpoint reload and formal batched inference, finite values, counts, exact keys/labels, router sums and Parquet round trips are verified by the unchanged imported14a worker. [Artifact list/hashes]({EXP}/artifact_files.csv), [supplement telemetry]({EXP}/supplement_telemetry.csv), [status journal]({session}/status_reference.json), [launch/exit events]({session}/events.jsonl).',
 f'Queue wall time including original-cell verification/skipping, before report generation: {ledger["wall_clock_seconds"]:.3f} seconds. Protected pre-existing source/data files unchanged: {protected["n_files"]}; original B2 tree unchanged: true. [Queue result]({session}/queue_result.json), [source protection]({session}/protected_inputs_verification.json).',
 '## W3 reference provenance',
 'The54 previously verified W3 cells are retained.24 additional W3 neural references cover seeds21/3407: O Static/Adaptive use archived formal Experiment4 TEST predictions; C Static/Adaptive use verified14a12/12 predictions. W3 O Adaptive seed42 remains the archived MAIN V2 predictions. W3 is not trained or inferred again. All78 W3 cells have6647 canonical TEST keys/labels and verified source/copy hashes.',
 f'[Complete W3 source provenance and URLs/paths]({EXP}/W3_provenance.json), [W3 per-cell scores]({EXP}/W3_reference_metrics.csv).',
 '## Five-seed descriptive comparison',
 'Static/Adaptive use exactly seeds42/0/1/21/3407. TEST MSE and mean within-date cross-sectional Spearman Rank IC use the imported formal metric code; mean +/- sample SD(ddof1) are over individual seed scores, not ensemble predictions. Ridge/Zero/Train-Mean retain one previously fitted/constructed predictor per cell, so seed SD is undefined. Constant-predictor Rank IC is undefined, not zero. Relative MSE is TEST MSE divided by the Global Train-Mean TEST MSE from the same window and horizon; R2_OS=1-relative MSE is also in CSV. Each baseline uses its own window TRAIN mean. No inferential statistics or point-in-time/leakage-free claims are added.',
 display(five),f'[Full precision five-seed summary]({EXP}/window_summary_5seed.csv), [all234 cell scores]({EXP}/all_cell_metrics_5seed.csv), [unchanged baseline denominators]({OLD}/global_train_mean_references.csv).',
 '## Three-seed comparison — seeds42/0/1',
 'This restricted90-row table matches the original B2 summary exactly, including every finite score and undefined entry.',display(three),f'[Full precision three-seed comparison]({EXP}/window_summary_3seed.csv).']
 body='\n\n'.join(text)+'\n';textout(EXP/'REPORT_STEP14C_B2_5SEED.md',body);textout(rootreport,body)
 jsonout(EXP/'FINAL_VALIDATION.json',dict(utc=utc(),new_neural_runs=48,new_reference_runs=0,original_B2_cells_verified_skipped=108,all_W1_W2_cells=156,W3_reused_cells=78,W3_training_runs=0,all_metric_rows=234,five_seed_summary_rows=90,three_seed_summary_rows=90,three_seed_summary_matches_original_exactly=True,new_prediction_parquets=144,new_checkpoints=48,maximum_workers=3,profile='CPU12/12',all_new_cells_verified=True,original_tree_unchanged=True,protected_files=protected['n_files'],protected_files_unchanged=True,report=rel(rootreport),report_sha256=sha(rootreport)))
 jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(rootreport),artifact(EXP/'window_summary_5seed.csv'),artifact(EXP/'window_summary_3seed.csv'),artifact(EXP/'FINAL_VALIDATION.json')]))
 print('REPORT_STEP14C_B2_5SEED_COMPLETE',utc(),flush=True)
if __name__=='__main__':
 import sys
 generate(sys.argv[1])
