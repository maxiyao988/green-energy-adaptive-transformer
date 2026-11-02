"""Descriptive B3 summaries against frozen Daily90+Weekly26 comparators."""
from b3_io import *
from b3_contracts import completed
import statistics

def summarize(rows,seeds):
 groups={}
 for r in rows:
  if int(r['seed']) in seeds:groups.setdefault((r['input_set'],r['configuration'],r['mechanism'],int(r['horizon'])),[]).append(r)
 out=[]
 for k,rs in sorted(groups.items()):
  assert len(rs)==len(seeds) and {int(r['seed']) for r in rs}==set(seeds)
  d=dict(zip(['input_set','configuration','mechanism','horizon'],k));d['n_seeds']=len(seeds)
  for metric in ['test_mse','mean_rank_ic']:
   vals=[float(r[metric]) for r in rs];d[metric+'_mean']=statistics.mean(vals);d[metric+'_sd_ddof1']=statistics.stdev(vals)
  out.append(d)
 return out

def generate(session):
 session=Path(session);q=json.loads((session/'queue_result.json').read_text());assert q['n_complete']==180 and not q['failures'] and not q['blocked']
 report=FULLRUN/'REPORT_STEP14C_B3.md'
 if (EXP/'FINAL_VALIDATION.json').exists():
  v=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert sha(report)==v['report_sha256'];jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(report),artifact(EXP/'FINAL_VALIDATION.json')],verified_previous_summary=True));return
 rows=[];telemetry=[];files=[]
 for spec in PLAN:
  cp,r=completed({k:spec[k] for k in KEYS});assert r,spec;t=next(x for x in r['metrics'] if x['split']=='test')
  rows.append(dict(input_set=spec['input_set'],configuration=spec['frequency_setting'],mechanism=spec['mechanism'],horizon=spec['horizon'],seed=spec['seed'],test_mse=t['pooled_mse'],mean_rank_ic=t['cross_sectional_rank_ic'],prediction_file=rel(cp.parent/'test.parquet'),checkpoint=r['checkpoint'],best_epoch=r['best_epoch'],epochs=r['epochs'],parameter_count=r['parameter_count'],trainer=r['training_entrypoint']))
  telemetry.append({k:r[k] for k in ['job_id','started_utc','completed_utc','wall_clock_seconds','epochs','best_epoch','thread_counts_start','thread_counts_end','environment_start','environment_end','load_average_start','load_average_end','telemetry_samples']})
  for a in r['artifacts']:files.append(dict(job_id=spec['job_id'],**a))
 with (EXP/'baseline_cell_metrics.csv').open() as f:b=list(csv.DictReader(f))
 baseline=[dict(input_set=r['input_set'],configuration='formal_daily90_weekly26',mechanism=r['model'],horizon=int(r['horizon']),seed=int(r['seed']),test_mse=float(r['test_mse']),mean_rank_ic=float(r['mean_rank_ic'])) for r in b]
 assert len(baseline)==60
 five=summarize(rows,[42,0,1,21,3407]);three=summarize(rows,[42,0,1]);bf=summarize(baseline,[42,0,1,21,3407]);bt=summarize(baseline,[42,0,1]);assert len(five)==len(three)==36 and len(bf)==12
 def paired(summary,reference):
  ref={(r['input_set'],r['mechanism'],r['horizon']):r for r in reference};out=[]
  for r in summary:
   x=dict(r);base=ref[(r['input_set'],r['mechanism'],r['horizon'])]
   for k in ['test_mse_mean','test_mse_sd_ddof1','mean_rank_ic_mean','mean_rank_ic_sd_ddof1']:x['formal_daily90_weekly26_'+k]=base[k]
   out.append(x)
  return out
 comparisons=paired(five,bf);comparison3=paired(three,bt)
 for name,data in [('all_cell_metrics.csv',rows),('summary_5seed.csv',comparisons),('summary_3seed.csv',comparison3),('baseline_summary_5seed.csv',bf),('queue_telemetry.csv',telemetry),('artifact_files.csv',files)]:csvout(EXP/name,data)
 prov=json.loads((EXP/'baseline_provenance.json').read_text());assert len(prov['entries'])==60
 assert sha(ROOT/prov['source_metric_csv']['path'])==prov['source_metric_csv']['sha256']
 for e in prov['entries']:
  assert sha(ROOT/e['source'])==e['source_sha256'];a=e['copied_test_predictions'];assert sha(ROOT/a['path'])==a['sha256']
 protect=json.loads((session/'protected_inputs_verification.json').read_text());assert protect['passed']
 events=[json.loads(s) for s in (session/'events.jsonl').read_text().splitlines() if s];assert max(e.get('active_workers',0) for e in events)==3
 started=[e['job_id'] for e in events if e['event']=='LAUNCH'];ids=[s['job_id'] for s in PLAN];assert started==[i for i in ids if i in set(started)]
 assert json.loads((EXP/'PADDING_CHECKS.json').read_text())['passed']
 policy=json.loads((EXP/'causal_reference_policy.json').read_text())
 def view(rs):
  return table(['input','configuration','mechanism','h','seeds','B3 TEST MSE mean +/- SD','formal D90+W26 TEST MSE mean +/- SD','B3 mean Rank IC mean +/- SD','formal D90+W26 mean Rank IC mean +/- SD'],[[r['input_set'],r['configuration'],r['mechanism'],r['horizon'],r['n_seeds'],f"{r['test_mse_mean']:.12g} +/- {r['test_mse_sd_ddof1']:.12g}",f"{r['formal_daily90_weekly26_test_mse_mean']:.12g} +/- {r['formal_daily90_weekly26_test_mse_sd_ddof1']:.12g}",f"{r['mean_rank_ic_mean']:.12g} +/- {r['mean_rank_ic_sd_ddof1']:.12g}",f"{r['formal_daily90_weekly26_mean_rank_ic_mean']:.12g} +/- {r['formal_daily90_weekly26_mean_rank_ic_sd_ddof1']:.12g}"] for r in rs])
 text=['# BLOCKING','None. All180 requested B3 cells completed and passed artifact, input, parameter, thread, count/key/label and source-protection checks. The C reference rule was explicitly resolved before training; see the immutable design report.',
 '# Step14c-B3 — matched historical coverage',f'Completed {utc()} UTC. Exclusive session `{rel(EXP)}`. The B2 queue and its seed21/3407 supplement finished before B3. Existing files were read only. [Baseline provenance]({EXP}/baseline_provenance.json).',
 '## Immutable pre-training design',
 (EXP/'DESIGN_REPORT_STEP14C_B3.md').read_text(),
 '## Training paths and verification',
 'All180 models use imported main_v2 model classes, scale experts, routers/static logits and heads. Daily patches remain5/10/20/30; Weekly2/4/8/13. Daily127 is an input-length/coverage change; Weekly26 underlying source bars are unchanged. Single-resolution models instantiate only the requested branch. No model or optimizer/training loop was reimplemented. The external mask and parameter-free signature adapters are described in the pre-training design, saved as importable new scripts for strict checkpoint reconstruction.',
 'Seed42 Adaptive uses main_v2.train_one_cell at the requested resolution; seed42 Static uses Experiment3; seeds0/1/21/3407 use Experiment4. Original batch32, Adam1e-4, Huber delta1, clip1, max100, patience10, improvement rule, validation selection/restoration and independent DataLoader generator42 remain unchanged. The imported14a scheduler refills at most3 worker processes at CPU intra12/inter12, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE. Seeds42/0/1 are queued before21/3407; within seed h20/10/5, interleaved configurations and O/C. Tail underfill is recorded; caffeinate covers the complete queue and reporting.',
 f'Artifacts:180 best-state checkpoints;540 TRAIN/VAL/TEST prediction Parquets;90 Static-logit JSONs;180 histories containing train/val Huber and val MSE; start/five-minute/end telemetry. Each checkpoint is strictly reloaded with its original factory/adapters. Predictions use the imported formal batched inference helper. Active-branch router weights sum to1; Static weights are constant; inactive-branch weights are not reported as learned routers. All files have43571 formal keys split30634/6290/6647, unchanged labels and finite predictions; Parquet round trips pass. [Artifacts/hashes]({EXP}/artifact_files.csv), [per-cell scores]({EXP}/all_cell_metrics.csv), [telemetry]({EXP}/queue_telemetry.csv), [status]({session}/status_reference.json).',
 f'Queue wall time before reporting={q["wall_clock_seconds"]:.3f} seconds. All{protect["n_files"]} protected pre-existing files unchanged. [Queue result]({session}/queue_result.json), [source verification]({session}/protected_inputs_verification.json).',
 '## Five-seed descriptive comparison',
 'Each score uses the imported formal pooled TEST MSE and mean within-date cross-sectional Spearman Rank IC. Mean +/- sample SD(ddof1) are over seeds42/0/1/21/3407, without ensembling. The formal comparator uses the same input set, mechanism, horizon and seeds: O Static/Adaptive archived Experiment3/MAIN V2/Experiment4 predictions (Adaptive original seed42 remains archived MAIN V2); C uses14a12/12 predictions. No comparator model is retrained or re-inferred. No bootstrap, p-value, hypothesis test, point-in-time or leakage-free claim is added.',
 ('The selected B3 C end-of-completed-week reference differs from the comparator Stage3a anchor-Close reference, in addition to matched Daily history. This difference follows the explicit user resolution and must be retained when interpreting C comparisons.' if policy['policy']=='completed_week_close' else 'The C anchor-Close denominator exception applies to34510 samples, as explicitly selected; only bar coverage is matched. Strict exclusion of post-week reference prices is not claimed.'),
 view(comparisons),f'[Five-seed full precision summary]({EXP}/summary_5seed.csv), [formal baseline summary]({EXP}/baseline_summary_5seed.csv).',
 '## Three-seed set — seeds42/0/1',view(comparison3),f'[Three-seed full precision summary]({EXP}/summary_3seed.csv).']
 body='\n\n'.join(text)+'\n';textout(EXP/'REPORT_STEP14C_B3.md',body);textout(report,body)
 jsonout(EXP/'FINAL_VALIDATION.json',dict(utc=utc(),cells=180,checkpoints=180,prediction_parquets=540,static_logit_files=90,five_seed_summary_rows=36,three_seed_summary_rows=36,baseline_cells=60,trained_baselines=0,all_cells_verified=True,strict_checkpoint_reload=True,all_keys_labels_counts_verified=True,padding_adapter_passed=True,maximum_workers=3,profile='CPU12/12',causal_reference_policy=policy['policy'],protected_files=protect['n_files'],protected_files_unchanged=True,report=rel(report),report_sha256=sha(report)))
 jsonout(session/'report_manifest.json',dict(utc=utc(),files=[artifact(report),artifact(EXP/'summary_5seed.csv'),artifact(EXP/'FINAL_VALIDATION.json')]))
 print('REPORT_STEP14C_B3_COMPLETE',utc(),flush=True)
if __name__=='__main__':
 import sys
 generate(sys.argv[1])
