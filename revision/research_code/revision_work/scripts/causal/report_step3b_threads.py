"""Final thread-reproduction report after the requested sequential/queued ordering."""
from repro_threads_common import *

def val(x):
 if x is None:return 'NOT FOUND'
 if isinstance(x,(float,np.floating)):return f'{x:.12g}' if np.isfinite(x) else 'N/A'
 return str(x)
def table(d,columns=None):
 if columns:d=d[columns]
 return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(val(x).replace('|','/').replace('\n',' ') for x in row)+' |' for row in d.itertuples(index=False,name=None)])+'\n'
def main():
 processes=json.loads((OUT/'processes.json').read_text());assert all(p['returncode']==0 for p in processes['processes']);runs=[json.loads((OUT/f"runs/{p['mechanism']}_dw_h{p['horizon']}_s42_t12.json").read_text()) for p in processes['processes']];first=next(r for r in runs if r['mechanism']=='adaptive' and r['horizon']==10);prior=json.loads((ROOT/'revision_work/outputs/task_a/recovery_h10.json').read_text())
 queue=json.loads((ROOT/'revision_work/outputs/causal_features/stage3bc_queue/processes.json').read_text());status=json.loads((OUT/'stage3bc_queue_status.json').read_text());assert status['returncode']==0
 assert processes['processes'][0]['completed_utc']<=status['started_utc']
 assert status['completed_utc']<=processes['processes'][1]['started_utc']
 for previous,current in zip(processes['processes'],processes['processes'][1:]):assert previous['completed_utc']<=current['started_utc']
 events=sorted([(p['started_utc'],1) for p in queue['processes']]+[(p['completed_utc'],-1) for p in queue['processes']]);active=maximum=0
 for _,delta in events:active+=delta;maximum=max(maximum,active)
 assert maximum<=4;write_json(OUT/'execution_order_verification.json',dict(h10_before_queue=True,queue_before_remaining_t12_runs=True,t12_runs_sequential=True,max_queue_workers=maximum,passed=True))
 for r in runs:r['utc_elapsed_seconds']=(datetime.fromisoformat(r['completed_utc'])-datetime.fromisoformat(r['started_utc'])).total_seconds()
 telemetry=[];top=[]
 for r in runs:
  stem=f"{r['mechanism']}_dw_h{r['horizon']}_s42_t12"
  for line in (OUT/f'telemetry/{stem}.jsonl').read_text().splitlines():
   item=json.loads(line);row=dict(mechanism=r['mechanism'],horizon=r['horizon'],event=item['event'],utc=item['utc'],torch_num_threads=item['torch_num_threads'],torch_num_interop_threads=item['torch_num_interop_threads'],load_avg_1=item['load_average_1_5_15'][0],load_avg_5=item['load_average_1_5_15'][1],load_avg_15=item['load_average_1_5_15'][2],**item['environment']);row.update({k:('UNSET' if v is None else v) for k,v in item['environment'].items()});telemetry.append(row)
   for i,p in enumerate(item['top_cpu_processes']):top.append(dict(mechanism=r['mechanism'],horizon=r['horizon'],event=item['event'],utc=item['utc'],rank=i+1,**p))
 tel=csvout('thread_and_load_telemetry.csv',telemetry);topdf=csvout('top_cpu_processes.csv',top)
 comparisons=csvout('reproduction_comparison.csv',[{k:r[k] for k in ['mechanism','horizon','classification','max_abs_delta_pred','test_mse_difference','replayed_test_mse','formal_test_mse','history_exact_equal','history_within_1e_12','history_max_train_abs_delta','history_max_val_abs_delta','first_differing_epoch','best_epoch','formal_best_epoch','stop_epoch','formal_stop_epoch','router_max_abs_delta','wall_clock_seconds','utc_elapsed_seconds','logged_thread_counts_changed']} for r in runs])
 files=pd.concat([pd.read_csv(OUT/f"runs/{r['mechanism']}_dw_h{r['horizon']}_s42_t12_prediction_files.csv") for r in runs],ignore_index=True);csvout('prediction_files.csv',files)
 before=json.loads((OUT/'protected_inputs_before.json').read_text());errors=[];hashcount=0
 for old in before['files']:
  p=ROOT/old['path']
  if not p.is_file():errors.append(dict(path=old['path'],problem='missing'));continue
  stat=p.stat()
  if stat.st_size!=old['size'] or stat.st_mtime_ns!=old['mtime_ns']:errors.append(dict(path=old['path'],problem='metadata changed'))
  if 'sha256' in old:
   hashcount+=1
   if sha(p)!=old['sha256']:errors.append(dict(path=old['path'],problem='SHA256 changed'))
 integrity=dict(completed_utc=utc(),protected_files=len(before['files']),hash_verified_files=hashcount,passed=not errors,problems=errors,excluded='.git, .venv, dependency installation trees and bytecode caches');write_json(OUT/'protected_inputs_verification.json',integrity)
 changed=any(r['logged_thread_counts_changed'] for r in runs)
 if first['classification'] in ['EXACT','NEAR']:
  other=[r for r in runs if r['horizon']!=10];matched=[r['horizon'] for r in runs if r['classification'] in ['EXACT','NEAR']]
  conclusion=f"Adaptive h10 at 12/12 is {first['classification']} (max |Δpred|={first['max_abs_delta_pred']:.12g}, ΔMSE={first['test_mse_difference']:.12g}), whereas the prior 3/1 run was DIFFERENT (max |Δpred|={prior['max_abs_delta_pred']:.12g}). The requested 12/12 profile is therefore sufficient to reproduce h10 within the declared threshold on this host; qualifying horizons in this test: {matched}. This supports a thread-configuration explanation, but does not identify intra-op versus inter-op as the sole component and does not authenticate the missing historical runtime provenance. Background application load is used only for timing interpretation, not as an explanation of numerical differences when logged thread counts remain fixed. For all new training after this explicitly authorized queue, use the pinned 12 intra-op / 12 inter-op, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE, MKL_DYNAMIC=FALSE, CPU and one training process at a time; record exact versions, seeds, loader generator, thread counts and checkpoints. The separately authorized Step4 Stage3b/3c queue used its explicitly requested 3/1, OMP=3 profile and must remain labeled as that profile rather than silently merged with 12/12 evidence."
 else:
  static=next(r for r in runs if r['mechanism']=='static');phrase='Static is reproducible within tolerance' if static['classification'] in ['EXACT','NEAR'] else 'neither of the two tested historical training cells is reproducible within tolerance'
  conclusion=f"Adaptive h10 remains DIFFERENT at 12/12 (max |Δpred|={first['max_abs_delta_pred']:.12g}, ΔMSE={first['test_mse_difference']:.12g}); the Static h10 control is {static['classification']} (max |Δpred|={static['max_abs_delta_pred']:.12g}), so {phrase}. Restoring audit-default threads is not sufficient to explain the Adaptive mismatch; unarchived historical execution details remain unverified. A faithful replay of the historical Static checkpoint, reported separately, distinguishes training reproducibility from checkpoint/inference reproducibility. Background application load is used only for timing interpretation, not as a cause of numerical differences while thread counts remain fixed. All new training after this explicitly authorized queue must explicitly pin CPU, 12 intra-op / 12 inter-op, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE, MKL_DYNAMIC=FALSE and one training process at a time, while archiving versions/RNG/loader/checkpoint evidence; this is a controlled standard, not a proven reconstruction of the unknown historical profile. The user-authorized Step4 Stage3b/3c queue is separately labeled 3/1, OMP=3 and is not silently treated as the same numerical protocol."
 block=[]
 if first['classification']=='DIFFERENT':block.append('Adaptive seed42 h10 historical checkpoint remains NOT RECOVERED; h5/h20 recovery was not launched under the requested branch.')
 for run in runs:
  if run['classification']=='DIFFERENT' and not (run['mechanism']=='adaptive' and run['horizon']==10):block.append(f"{run['mechanism']} h{run['horizon']} was DIFFERENT; its training was not historically reproduced. See its per-cell comparison.")
 if changed:block.append('Thread counts changed in at least one telemetry record; see telemetry and per-run JSON.')
 if errors:block.append('Protected-input verification FAILED: '+json.dumps(errors))
 if not block:block.append('No execution blocker. Qualification of each saved state follows its own EXACT/NEAR/DIFFERENT comparison; a DIFFERENT result is not promoted as a recovered formal checkpoint.')
 sections=['# BLOCKING\n']+[f'- {b}\n' for b in block]+['# Step 3B — Thread reproducibility test\n',f'Completed {utc()} UTC. All new outputs are under `revision_work/outputs/repro_threads/`; all new code is under `revision_work/scripts/causal/`. Existing snapshots, code, tensors, checkpoints and results were read only. Protected inputs: {len(before["files"])} files; {hashcount} hashes; PASS={integrity["passed"]}. [Verification](protected_inputs_verification.json).\n',
 '## Conclusion\n',conclusion+'\n',
 '## Execution and ordering\n',
 'The user allowed resident applications, requiring exactly one training process and no other training/inference jobs for the 12/12 tests. Adaptive h10 ran first and alone. Only after it exited did the 30-cell Step4 Stage3b+3c queue run with at most4 workers, Torch3 / inter-op1 / OMP3. After that queue exited, the conditional h5→h20 12/12 recoveries or Static h10 12/12 control ran sequentially. No training or checkpoint-inference overlap occurs between these profiles. [Process ledger](processes.json), [order/concurrency verification](execution_order_verification.json), [queue status](stage3bc_queue_status.json), [queue ledger](../causal_features/stage3bc_queue/processes.json).\n',
 'The unchanged `main_v2.train_one_cell("daily_weekly",h,"adaptive")` training function is imported and called exactly as in `revision_work/scripts/recover_adaptive42.py`, with the same delegated factory-reference and phase-only observer; required telemetry adds read-only observations without extra training forward/backward passes or RNG draws. Static uses unchanged Experiment3 `train_and_validate_cell`. Original frozen V2 features, labels, normalization, splits, CPU, seed42, generator42, Adam1e-4, batch32, Huberδ1, clip1, patience10 and max100 are retained. Static serialization is redirected only to new exclusive paths; its new `.tmp` is retained.\n',
 'Audit defaults are current-host measurements, not historical provenance: [AUDIT_REPORT.md:116](../../../revision_audit/AUDIT_REPORT.md) reports12/12, but historical hardware/thread/version metadata are NOT FOUND. [recover_adaptive42.py](../../scripts/recover_adaptive42.py), [main V2 trainer:1611](<project_root>/scripts/python/panel_pathformer_main_v2.py:1611), [Experiment3 trainer:622](<project_root>/scripts/python/panel_pathformer_experiment3.py:622).\n',
 '## Prediction, loss-history and epoch comparison\n',table(comparisons),
 'Classification: EXACT if max|Δpred|<1e-7; NEAR if<1e-5; otherwise DIFFERENT. Prediction differences use float64 representations; pooled MSE uses original float32 arithmetic. ΔMSE=new−formal. History exact equality uses float64 CSV round-trip parsing and exact value equality, with an additional labeled1e-12 numeric check; epoch absence is not treated as equality. Monotonic elapsed and UTC elapsed seconds are both reported so a host-sleep/timekeeping gap, if any, is not silently removed.\n']
 for r in runs:
  stem=f"{r['mechanism']}_dw_h{r['horizon']}_s42_t12";sections += [f'### {r["mechanism"]}, h={r["horizon"]}: every epoch\n',table(pd.read_csv(OUT/f'comparisons/{stem}_history_by_epoch.csv',float_precision='round_trip'))]
  if r['mechanism']=='static':sections += ['Historical Static checkpoint replay at12/12, against archived formal CSV:\n',table(pd.read_csv(OUT/f'comparisons/{stem}_formal_checkpoint_replay.csv')),'Retrained Static versus predictions re-inferred directly from the historical checkpoint:\n',table(pd.read_csv(OUT/f'comparisons/{stem}_versus_formal_checkpoint.csv'))]
 sections += ['## Thread controls and background telemetry\n',
 'OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE are set before importing Torch. OMP_NUM_THREADS remains absent. Every run logs `torch.get_num_threads()`, `torch.get_num_interop_threads()`, all seven selected environment controls, system load average(1/5/15 minutes), and top20 CPU processes at start, every5 minutes (main-thread forward observer, within one batch of the deadline), and end. `MKL not found` on this host is retained in each environment record. OS/services and the agent remain resident. Background activity affects timing only; it is not invoked as a numerical cause unless the observed thread counts change.\n',table(tel),
 'Full top20 process list for every sample: [top_cpu_processes.csv](top_cpu_processes.csv); raw records: `telemetry/*.jsonl`. Top5 for each snapshot follows:\n',table(topdf[topdf['rank']<=5]),
 '## Checkpoints and all-split replay\n',
 'States are saved after the original trainer restores its best state, then strictly reloaded into imported models in eval mode. TaskB formal batched helpers generate ordered TRAIN and unshuffled VAL/TEST outputs. Every file contains ticker, anchor_date, split, y_true, y_pred plus Daily/Weekly router weights; Static weights are checked constant. Readback, unique keys, finite values, simplex weights and row counts are verified. Expected counts: train30,634 / val6,290 / test6,647 per cell. Nonqualifying checkpoints/Parquets are retained only as rerun diagnostics.\n',table(files),
 '## Limitations\n',
 'No authenticated historical thread/runtime manifest was found. A successful training reproduction supports the tested numerical profile; a checkpoint replay alone proves inference consistency, not recovery of the optimizer trajectory. A failure of both tested cells is not a proof that every historical run on this machine is irreproducible. No undocumented historical causes are asserted. The frozen data are ex-post snapshots and are not claimed to be point-in-time.\n']
 with target(OUT/'REPORT_STEP3B.md').open('x') as f:f.write('\n'.join(sections))
 # Inventory only closed, stable output files; no redirected active log is included.
 manifest=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name not in ['launcher.log','report_step3b.log']:manifest.append(dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p)))
 write_json(OUT/'MANIFEST_STEP3B.json',dict(completed_utc=utc(),files=manifest,new_source=[dict(path=rel(p),sha256=sha(p)) for p in sorted((ROOT/'revision_work/scripts/causal').glob('repro_threads_*.py'))],report_source=rel(Path(__file__)),report_source_sha256=sha(Path(__file__)),background_load_interpretation='Timing only unless thread counts changed'))
 print('STEP3B REPORT COMPLETE');print(comparisons.to_string(index=False));print('Protected inputs',integrity['passed'])
if __name__=='__main__':main()
