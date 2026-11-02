"""Step4 final five-seed and as-of-level diagnostics. No training or inference."""
from core import *
import re
SEEDS=[0,1,21,42,3407]

def number(x):
 if x is None:return 'N/A'
 if isinstance(x,(float,np.floating)):return f'{x:.10g}' if np.isfinite(x) else 'N/A'
 return str(x)
def table(d,cols=None):
 if cols:d=d[cols]
 return fmt_table(list(d.columns),[[number(x) for x in row] for row in d.itertuples(index=False,name=None)])
def canonical(d):
 d=d.copy();d['anchor_date']=pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d');assert not d.duplicated(['ticker','anchor_date']).any();return d.sort_values(['ticker','anchor_date']).reset_index(drop=True)
def old_predictions():
 rows={};paths={'static':ROOT/'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv','adaptive':ROOT/'dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv'};extra=pd.read_csv(ROOT/'dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv')
 for mech,path in paths.items():
  base=pd.read_csv(path)
  if 'mechanism' in base:base=base[base.mechanism==mech]
  if 'frequency_setting' in base:base=base[base.frequency_setting=='daily_weekly']
  for h in [5,10,20]:
   for seed in SEEDS:
    data=base if seed==42 else extra[extra.mechanism==mech];mask=data.horizon.astype(str).str.removesuffix('d').astype(int).eq(h)&data.seed.eq(seed);d=canonical(data[mask]);assert len(d)==6647;rows[(mech,h,seed)]=d
 return rows

def main():
 qpath=OUT/'stage3bc_queue/processes.json';q=json.loads(qpath.read_text());assert len(q['processes'])==30 and all(p['returncode']==0 for p in q['processes'])
 assert not (OUT/'REPORT_STEP4_FINAL.md').exists()
 frozen=common.load_frozen_panel();expected=canonical(pd.DataFrame(dict(ticker=frozen.ticker[frozen.split=='test'],anchor_date=frozen.anchor_date[frozen.split=='test'])));old=old_predictions();preds={};metricsrows=[];runs=[];artifact_checks=[]
 # Original five-seed predictions and all three re-estimation feature sets.
 for (mech,h,seed),frame in old.items():preds[('original_levels',mech,h,seed)]=frame
 for h in [5,10,20]:
  for mech in ['static','adaptive']:
   for seed in SEEDS:
    prefix='' if seed==42 else 'stage3b/';stem=f'{mech}_dw_h{h}_s{seed}';d=canonical(pd.read_parquet(OUT/f'{prefix}preds/{stem}_test.parquet'));preds[('causal_relative',mech,h,seed)]=d
   stem=f'{mech}_dw_h{h}_s42';d=canonical(pd.read_parquet(OUT/f'stage3c/preds/{stem}_test.parquet'));preds[('asof_levels',mech,h,42)]=d
 for (feature,mech,h,seed),d in preds.items():
  assert len(d)==6647 and d[['ticker','anchor_date']].equals(expected[['ticker','anchor_date']]);y=d.y_true.to_numpy(np.float32);p=d.y_pred.to_numpy(np.float32);base=old[(mech,h,seed if feature!='asof_levels' else 42)];assert np.array_equal(y,base.y_true.to_numpy(np.float32));m=metrics(y,p,d.ticker.to_numpy(),d.anchor_date.to_numpy());tm=np.float32(frozen.y[h][frozen.split=='train'].mean());mse_tm=float(np.mean((y-tm)**2));mse_zero=float(np.mean(y*y));metricsrows.append(dict(feature_set=feature,mechanism=mech,horizon=h,seed=seed,R2_OS_trainmean=1-m['pooled_mse']/mse_tm,R2_OS_zero=1-m['pooled_mse']/mse_zero,**m))
 metricsdf=pd.DataFrame(metricsrows);assert len(metricsdf)==66
 for stage in ['stage3b','stage3c']:
  for path in sorted((OUT/f'{stage}/runs').glob('*.json')):
   r=json.loads(path.read_text());r['utc_elapsed_seconds']=(datetime.fromisoformat(r['completed_utc'])-datetime.fromisoformat(r['started_utc'])).total_seconds();runs.append(r);assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256'];stem=path.stem;history=pd.read_csv(OUT/f'{stage}/history/{stem}.csv');assert len(history)==r['stop_epoch'];assert history.epoch.tolist()==list(range(1,len(history)+1));assert np.isfinite(history[['train_huber','val_huber','val_mse']]).all().all();best=float('inf');epoch=0
   for e,v in zip(history.epoch,history.val_huber):
    if v<best-1e-12:best=v;epoch=int(e)
   assert epoch==r['best_epoch'];saved=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True);model=exp4._make_experiment4_model(r['mechanism']) if stage=='stage3b' else (exp3._make_experiment3_model('static',None,None) if r['mechanism']=='static' else formal._make_model('daily_weekly','adaptive'));model.load_state_dict(saved,strict=True);assert all(torch.isfinite(v).all() for v in saved.values())
   for split,count in COUNTS.items():
    frame=pd.read_parquet(OUT/f'{stage}/preds/{stem}_{split}.parquet');mask=frozen.split==split;assert len(frame)==count and frame.split.eq(split).all() and not frame.duplicated(['ticker','anchor_date']).any();assert np.array_equal(frame.ticker.to_numpy(),frozen.ticker[mask]);assert np.array_equal(pd.to_datetime(frame.anchor_date).to_numpy(),pd.to_datetime(frozen.anchor_date[mask]).to_numpy());assert np.array_equal(frame.y_true.to_numpy(),frozen.y[r['horizon']][mask]);assert np.isfinite(frame[['y_true','y_pred']+WEIGHTS]).all().all()
    if r['mechanism']=='static':assert np.ptp(frame[WEIGHTS].to_numpy(),axis=0).max()<1e-7
    artifact_checks.append(dict(stage=stage,mechanism=r['mechanism'],horizon=r['horizon'],seed=r['seed'],split=split,n_rows=count,path=rel(OUT/f'{stage}/preds/{stem}_{split}.parquet'),sha256=sha(OUT/f'{stage}/preds/{stem}_{split}.parquet'),passed=True))
 assert len(runs)==30 and len(artifact_checks)==90
 metric_names=['pooled_mse','pooled_mae','pooled_corr','pooled_da','cross_sectional_rank_ic','pooled_pred_std_over_true_std','R2_OS_trainmean','R2_OS_zero'];summary=[]
 multi=metricsdf[metricsdf.feature_set!='asof_levels']
 for (feature,mech,h),g in multi.groupby(['feature_set','mechanism','horizon']):
  assert len(g)==5 and set(g.seed)==set(SEEDS);row=dict(feature_set=feature,mechanism=mech,horizon=h,n_seeds=5)
  for name in metric_names:row[name+'_mean']=float(g[name].mean());row[name+'_sd_ddof1']=float(g[name].std(ddof=1));row[name+'_n_valid']=int(g[name].notna().sum())
  summary.append(row)
 summarydf=pd.DataFrame(summary);display_summary=[]
 for _,row in summarydf.sort_values(['horizon','mechanism','feature_set']).iterrows():
  view={k:row[k] for k in ['feature_set','mechanism','horizon']}
  for name,label in zip(metric_names,['MSE','MAE','Pearson','DA','Mean Rank IC','PredStd/TrueStd','R2 trainmean','R2 zero']):view[label]=f"{number(row[name+'_mean'])} ± {number(row[name+'_sd_ddof1'])} (n={int(row[name+'_n_valid'])})"
  display_summary.append(view)
 display_summary=pd.DataFrame(display_summary);wins=[];seed_dates=[];date_rows=[];paired_summary=[]
 for feature in ['original_levels','causal_relative']:
  for h in [5,10,20]:
   g=multi[(multi.feature_set==feature)&(multi.horizon==h)].pivot(index='seed',columns='mechanism',values='pooled_mse');wins.append(dict(feature_set=feature,horizon=h,adaptive_seed_wins=int((g.adaptive<g.static).sum()),static_seed_wins=int((g.static<g.adaptive).sum()),ties=int((g.static==g.adaptive).sum()),adaptive_minus_static_mean=float((g.adaptive-g.static).mean()),adaptive_minus_static_sd_ddof1=float((g.adaptive-g.static).std(ddof=1))))
   for seed in SEEDS:
    a=preds[(feature,'adaptive',h,seed)];s=preds[(feature,'static',h,seed)];assert a[['ticker','anchor_date']].equals(s[['ticker','anchor_date']]);assert np.array_equal(a.y_true.to_numpy(np.float32),s.y_true.to_numpy(np.float32));y=a.y_true.to_numpy(np.float32).astype(float);pA=a.y_pred.to_numpy(np.float32).astype(float);pS=s.y_pred.to_numpy(np.float32).astype(float);df=pd.DataFrame(dict(anchor_date=a.anchor_date,ticker=a.ticker,loss_difference=(pA-y)**2-(pS-y)**2));assert df.groupby('anchor_date').size().eq(17).all()
    for date,v in df.groupby('anchor_date').loss_difference.mean().items():seed_dates.append(dict(feature_set=feature,horizon=h,seed=seed,anchor_date=date,stock_mean_adaptive_minus_static=float(v),n_tickers=17))
 sd=pd.DataFrame(seed_dates)
 for (feature,h,date),g in sd.groupby(['feature_set','horizon','anchor_date']):assert len(g)==5;date_rows.append(dict(feature_set=feature,horizon=h,anchor_date=date,seed_mean_stock_mean_loss_difference=float(g.stock_mean_adaptive_minus_static.mean()),n_seeds=5,n_tickers=17))
 dates=pd.DataFrame(date_rows)
 for (feature,h),g in dates.groupby(['feature_set','horizon']):
  v=g.seed_mean_stock_mean_loss_difference;assert len(v)==391;paired_summary.append(dict(feature_set=feature,horizon=h,n_dates=len(v),mean=float(v.mean()),std_ddof1=float(v.std(ddof=1)),median=float(v.median()),p05=float(v.quantile(.05)),p95=float(v.quantile(.95)),fraction_dates_adaptive_lower_loss=float((v<0).mean())))
 # Three feature sets at the same nominal seed42 (same 3/1 profile for causal/asof).
 single=metricsdf[metricsdf.seed==42].sort_values(['horizon','mechanism','feature_set']);contrasts=[]
 for h in [5,10,20]:
  for mech in ['static','adaptive']:
   g=single[(single.horizon==h)&(single.mechanism==mech)].set_index('feature_set');orig=g.loc['original_levels'];causal=g.loc['causal_relative'];asof=g.loc['asof_levels'];rel_asof=float(asof.pooled_mse/orig.pooled_mse-1);rel_causal=float(causal.pooled_mse/orig.pooled_mse-1)
   contrasts.append(dict(horizon=h,mechanism=mech,original_mse=orig.pooled_mse,asof_mse=asof.pooled_mse,causal_mse=causal.pooled_mse,asof_relative_mse_change=rel_asof,causal_relative_mse_change=rel_causal,asof_minus_original_rank_ic=float(asof.cross_sectional_rank_ic-orig.cross_sectional_rank_ic),causal_minus_original_rank_ic=float(causal.cross_sectional_rank_ic-orig.cross_sectional_rank_ic),descriptive_asof_flag='WORSE >=2%' if rel_asof>=.02 else 'BETTER >=2%' if rel_asof<=-.02 else 'WITHIN +/-2%'))
 contrastsdf=pd.DataFrame(contrasts)
 thread_controls=[]
 for h in [5,10,20]:
  prior_path=ROOT/f'revision_work/outputs/task_a/adaptive_dw_h{h}_s42_predictions.csv';d=canonical(pd.read_csv(prior_path));meta=json.loads((ROOT/f'revision_work/outputs/task_a/recovery_h{h}.json').read_text());assert len(d)==6647;g=single[(single.horizon==h)&(single.mechanism=='adaptive')].set_index('feature_set');rerun=metrics(d.y_true.to_numpy(np.float32),d.y_pred.to_numpy(np.float32),d.ticker.to_numpy(),d.anchor_date.to_numpy())['pooled_mse'];thread_controls.append(dict(horizon=h,archived_original_mse=float(g.loc['original_levels'].pooled_mse),original_3_1_rerun_mse=rerun,original_rerun_status=meta['classification'],causal_3_1_mse=float(g.loc['causal_relative'].pooled_mse),asof_3_1_mse=float(g.loc['asof_levels'].pooled_mse),causal_vs_original_rerun_relative_change=float(g.loc['causal_relative'].pooled_mse/rerun-1),asof_vs_original_rerun_relative_change=float(g.loc['asof_levels'].pooled_mse/rerun-1)))
 # Required telemetry, and freeze numerical interpretation to observed counts.
 telemetry=[]
 for r in runs:
  stem=f"{r['mechanism']}_dw_h{r['horizon']}_s{r['seed']}"
  for line in (OUT/f"{r['stage']}/telemetry/{stem}.jsonl").read_text().splitlines():
   e=json.loads(line);telemetry.append(dict(stage=r['stage'],mechanism=r['mechanism'],horizon=r['horizon'],seed=r['seed'],utc=e['utc'],event=e['event'],torch_num_threads=e['torch_num_threads'],torch_num_interop_threads=e['torch_num_interop_threads'],load_avg_1=e['load_average_1_5_15'][0],load_avg_5=e['load_average_1_5_15'][1],load_avg_15=e['load_average_1_5_15'][2],**{k:('UNSET' if v is None else v) for k,v in e['environment'].items()}))
 teldf=pd.DataFrame(telemetry);repro=json.loads((ROOT/'revision_work/outputs/repro_threads/runs/adaptive_dw_h10_s42_t12.json').read_text());repro_report=ROOT/'revision_work/outputs/repro_threads/REPORT_STEP3B.md';counts_changed=any(r['logged_thread_counts_changed'] for r in runs)
 # Complete validation before creating derived outputs, all exclusively.
 csvout('stage3bc_queue/all_seed_metrics.csv',metricsdf);csvout('stage3bc_queue/five_seed_summary.csv',summarydf);csvout('stage3bc_queue/seed_wins.csv',wins);csvout('stage3bc_queue/paired_loss_by_seed_date.csv',sd);csvout('stage3bc_queue/paired_loss_by_date.csv',dates);csvout('stage3bc_queue/paired_loss_summary.csv',paired_summary);csvout('stage3bc_queue/seed42_three_feature_metrics.csv',single);csvout('stage3bc_queue/asof_contrasts.csv',contrastsdf);csvout('stage3bc_queue/adaptive_s42_thread_control.csv',thread_controls);csvout('stage3bc_queue/artifact_checks.csv',artifact_checks);csvout('stage3bc_queue/thread_and_load_telemetry.csv',teldf)
 runtimes=pd.DataFrame([{k:r.get(k) for k in ['stage','mechanism','horizon','seed','best_epoch','stop_epoch','wall_clock_seconds','utc_elapsed_seconds','checkpoint','checkpoint_sha256','logged_thread_counts_changed']} for r in runs]);csvout('stage3bc_queue/runtimes.csv',runtimes)
 old_interim=OUT/'REPORT_STEP4_INTERIM_CORRECTED.md';risks=old_interim.read_text().split('## Risk → CHECKED / MITIGATED / REMAINS UNVERIFIED',1)[1].strip()
 risks=risks.replace('Stage3c NOT RUN.','Stage3c now completed; results are diagnostic, not independent proof of leakage.').replace('this does not prove the archived V2 score was caused by leakage','three-way comparisons do not alone prove that the archived V2 score was caused by leakage')
 blocks=[]
 if repro['classification']=='DIFFERENT':blocks.append('Adaptive seed42 h10 remains historically unreproduced even at12/12; nominal seed42 cross-feature contrasts cannot be treated as exact reconstruction of the archived training trajectory.')
 else:blocks.append('Adaptive h10 reproduces at12/12, while this queue explicitly uses3/1 as authorized. Original-to-new feature effects are potentially confounded by the differing numerical thread profile; compare causal versus asof under the shared3/1 profile, and do not silently claim all original/new differences are solely features.')
 if counts_changed:blocks.append('At least one3/1 run logged a changed thread count; see telemetry. Background load otherwise affects timing only.')
 blocks += ['The original1e-6 cross-snapshot test remains FAILED; the documented1e-4 numerical-precision tolerance authorized Stage3a and this continuation. Point-in-time vendor vintages, historical security identity and delisted-universe coverage remain unverified.','The as-of-level reconstruction uses the requested c(t) formula, not authenticated historical quotes. BEP’s BEPC distribution and vendor adjustment revisions limit causal interpretation.']
 sections=['# BLOCKING\n']+['- '+x+'\n' for x in blocks]+['# Step4 final — Stage3b and Stage3c\n',f'Completed {utc()} UTC. User continuation explicitly authorized the30-cell queue after Step3B h10; 24 causal multiseed +6 asof seed42 cells completed. No pre-existing file was overwritten. Code: `revision_work/scripts/causal/`; outputs remain in this namespace. [Interim report](REPORT_STEP4_INTERIM_CORRECTED.md) retains Parts1/2, the original failed threshold, precision diagnostic and Stage3a. [Step3B report](../repro_threads/REPORT_STEP3B.md).\n',
 '## Training protocol and ordering\n',
 'Step3B Adaptive h10 ran alone first at12/12, OMP_NUM_THREADS unset, both dynamic controlsFALSE. This queue then ran at most4 training/inference workers with Torch3 / inter-op1 / OMP3, BLAS caps3 and both dynamic controlsFALSE. Only after queue completion did the requested conditional12/12 recoveries or Static control run. System load averages and top20 processes were recorded at start, every5 minutes and end. Background load is used for timing only unless recorded thread counts change.\n',
 'Stage3b imports Experiment4 train_and_validate_cell for Static/Adaptive seeds0,1,21,3407, reusing formal model constructors and DataLoader generator42. Stage3c seed42 imports main_v2 for Adaptive and Experiment3 for Static, as Stage3a. Only the process-local input dataset and new checkpoint destination are rebound; the model and training loop are not reimplemented. The original sources and archived results stay unchanged. Adam1e-4, batch32, Huberδ1, clipping1, validation-Huber best-state selection, patience10, max100 and CPU are retained. [Worker source](<project_root>/revision_work/scripts/causal/stage3bc_worker.py:20), [asof builder](<project_root>/revision_work/scripts/causal/build_stage3c_asof.py:7).\n',
 '## Stage3b — Five-seed comparison\n',
 'Each feature set uses seeds0,1,21,42,3407. Causal seed42 comes from the preserved Stage3a results; the other24 cells are new. Original seed42 Static is Experiment3, seed42 Adaptive is MAIN V2, and the other original seeds are Experiment4. SD uses ddof=1 across five seeds. Metrics are recomputed from canonical6647-row test predictions with the imported V2 metric functions and identical frozen labels.\n',table(display_summary),
 '### Seed wins and Adaptive−Static MSE\n',table(pd.DataFrame(wins)),
 '### Paired date-level loss differences\n',
 'For each feature/horizon/seed/date compute the mean of (Adaptive squared error−Static squared error) across17 stocks. Then average those five seed-specific stock means at each date, leaving391 ordered dates per horizon. Report the distribution across dates; negative values favor Adaptive. Date observations are dependent because targets overlap, so these are descriptive distributions, not independent-sample significance tests. Float64 paired-loss accumulation can differ by tiny rounding from the original float32 pooled-MSE arithmetic.\n',table(pd.DataFrame(paired_summary)),
 '[Stock-mean loss by seed/date](stage3bc_queue/paired_loss_by_seed_date.csv), [seed-averaged date losses](stage3bc_queue/paired_loss_by_date.csv), [all66 cell metrics](stage3bc_queue/all_seed_metrics.csv).\n',
 '## Stage3c — Ex-post levels / asof levels / causal relatives\n',
 'For each retained anchor t, divide every original frozen OHLC value in its Daily and Weekly windows by NEW-snapshot Part1 c(t). Volume stays exactly V2. Use the same original frozen labels/splits/windows and recompute train-only ticker×resolution×channel statistics over training window entries (ddof0). This is the requested formula-based counterfactual and is not claimed to be true point-in-time data. Saved raw/standardized tensors, anchor factors and170 normalization statistics are under `stage3c/`; [checks](stage3c/feature_checks.json).\n',table(single,['feature_set','mechanism','horizon','seed','pooled_mse','pooled_mae','pooled_corr','pooled_da','cross_sectional_rank_ic','pooled_pred_std_over_true_std','R2_OS_trainmean','R2_OS_zero']),table(contrastsdf),
 'The ±2% MSE band above is a declared descriptive aid for “approximately equal,” not a significance threshold. Similar asof/ex-post performance is consistent with genuine level information but does not prove it; clearly worse asof performance is consistent with ex-post-adjustment dependence but is not alone proof of leakage. Thread-profile differences, unrecovered historical optimizer trajectories, c(t)’s vendor assumptions and security identity remain alternative limitations. Causal-versus-asof new results share the same3/1 protocol and seed42 training entrypoints.\n',
 '### Existing3/1 Adaptive seed42 numerical control (no new training)\n',table(pd.DataFrame(thread_controls)),
 'These existing Step3A original-feature reruns are DIFFERENT from the formal paper results and are never substituted for them. They provide separately labeled3/1 controls for Adaptive seed42, using the same frozen data and formal main_v2 trainer. Dynamic-control history was not archived for that older probe. Original-feature3/1 retraining controls for Static and the other four seeds are NOT FOUND, so this supplement does not isolate the feature effect for the full five-seed comparison.\n',
 '## Runtime, epochs, artifacts and telemetry\n',table(runtimes),f'Queue wall time: {q["wall_clock_seconds"]:.3f}s. [Process ledger](stage3bc_queue/processes.json). All30 selected-best checkpoints load strictly; all90 new Parquets pass key/date/label/row-count/finite checks: train30,634 / val6,290 / test6,647 per cell. Each neural history includes train/val Huber and val MSE; router weights cover all splits, with Static logits saved separately. [Artifact list and hashes](stage3bc_queue/artifact_checks.csv), [all histories](stage3bc_queue/runtimes.csv).\n',
 f'12/12 Adaptive h10 reproduction result: **{repro["classification"]}**, max|Δpred|={repro["max_abs_delta_pred"]:.12g}, ΔMSE={repro["test_mse_difference"]:.12g}. Complete cause/thread-policy discussion and all required12/12 telemetry are in [REPORT_STEP3B](../repro_threads/REPORT_STEP3B.md). This final report preserves the explicitly requested3/1 profile for its30 new cells.\n',table(teldf),
 'Full top20 process snapshots remain in each `stage3b/telemetry/*.jsonl` and `stage3c/telemetry/*.jsonl`. [Telemetry table](stage3bc_queue/thread_and_load_telemetry.csv). Final protected-input verification, covering the preserved previous reports, snapshots, sources and results, is also in [Step3B verification](../repro_threads/protected_inputs_verification.json).\n',
 '## Risk → CHECKED / MITIGATED / REMAINS UNVERIFIED\n',risks+'\n']
 with target(OUT/'REPORT_STEP4_FINAL.md').open('x') as f:f.write('\n'.join(sections))
 manifest=[dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((OUT/'stage3b').rglob('*')) if p.is_file()]+[dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((OUT/'stage3c').rglob('*')) if p.is_file()]
 jsonout('stage3bc_queue/MANIFEST_FINAL.json',dict(completed_utc=utc(),report=rel(OUT/'REPORT_STEP4_FINAL.md'),report_sha256=sha(OUT/'REPORT_STEP4_FINAL.md'),files=manifest,new_source=[dict(path=rel(p),sha256=sha(p)) for p in sorted((ROOT/'revision_work/scripts/causal').glob('*stage3*c*.py'))]))
 print('STEP4 FINAL COMPLETE',utc(),'30 runs / 90 Parquets');print(pd.DataFrame(wins).to_string(index=False));print(contrastsdf.to_string(index=False))
if __name__=='__main__':main()
