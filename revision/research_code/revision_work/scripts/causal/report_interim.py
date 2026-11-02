"""Aggregate Stage 3a, verify immutable inputs and write the final interim report once."""
from core import *
import platform,re,pickle

def number(x):
 if x is None:return 'N/A'
 if isinstance(x,(bool,np.bool_)):return str(bool(x))
 if isinstance(x,(int,np.integer)):return str(x)
 if isinstance(x,(float,np.floating)):return f'{x:.10g}' if np.isfinite(x) else 'N/A'
 return str(x)
def table(d,cols=None):
 if cols:d=d[cols]
 return fmt_table(list(d.columns),[[number(x) for x in row] for row in d.itertuples(index=False,name=None)])
def read(n):return pd.read_csv(OUT/n)
def cite(path,line):return f'[{Path(path).name}:{line}]({ROOT/path}:{line})'
def code_ref(path,needle):
 lines=(ROOT/path).read_text().splitlines();line=next(i+1 for i,s in enumerate(lines) if needle in s);return cite(path,line)
def original():
 paths={'ridge':'dataset/audit/main_v2/ridge/panel_ridge_test_predictions.csv','static':'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv','adaptive':'dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv'}
 summaries={'static':pd.read_csv(ROOT/'dataset/audit/experiment3_pathformer_ablation/experiment3_test_summary_metrics.csv'),'adaptive':pd.read_csv(ROOT/'dataset/audit/main_v2/pathformer/panel_pathformer_summary_metrics.csv')}
 rows=[];expected=load_causal();emask=expected.split=='test';keys=pd.DataFrame(dict(ticker=expected.ticker[emask],anchor_date=pd.to_datetime(expected.anchor_date[emask]).strftime('%Y-%m-%d')))
 for mechanism,path in paths.items():
  d=pd.read_csv(ROOT/path);d['horizon']=d.horizon.astype(str).str.removesuffix('d').astype(int)
  if 'seed' in d:d=d[d.seed==42]
  if 'mechanism' in d:d=d[d.mechanism==mechanism]
  if 'frequency_setting' not in d:d['frequency_setting']='daily_weekly'
  if mechanism!='ridge':d=d[d.frequency_setting=='daily_weekly']
  for (freq,h),g in d.groupby(['frequency_setting','horizon']):
   assert len(g)==6647
   actual=g.copy();actual['anchor_date']=pd.to_datetime(actual.anchor_date).dt.strftime('%Y-%m-%d');joined=keys.assign(expected_y=expected.y[h][emask]).merge(actual[['ticker','anchor_date','y_true']],on=['ticker','anchor_date'],validate='one_to_one');assert len(joined)==6647;assert np.array_equal(joined.expected_y.to_numpy(),joined.y_true.to_numpy(np.float32))
   met=metrics(g.y_true.to_numpy(np.float32),g.y_pred.to_numpy(np.float32),g.ticker.to_numpy(),pd.to_datetime(g.anchor_date).to_numpy())
   best=stop=None
   if mechanism!='ridge':
    s=summaries[mechanism];m=s.horizon.astype(str).str.removesuffix('d').astype(int).eq(h)&s.seed.eq(42)
    if 'mechanism' in s:m &= s.mechanism.eq(mechanism)
    if 'frequency_setting' in s:m &= s.frequency_setting.eq(freq)
    ss=s[m];assert len(ss)==1;best=int(ss.iloc[0].best_epoch);stop=int(ss.iloc[0].epochs_trained)
   rows.append(dict(feature_set='original_levels',mechanism=mechanism,frequency_setting=freq,horizon=h,seed=42,split='test',best_epoch=best,stop_epoch=stop,source=path,**met))
 return rows

def main():
 processes=json.loads((OUT/'stage3a_processes.json').read_text());retry=json.loads((OUT/'stage3a_retry_processes.json').read_text())
 assert {(p['mechanism'],p['horizon'],p['returncode']) for p in processes['processes'] if p['returncode']!=0}=={('static',5,-15),('static',10,-15)}
 assert all(p['returncode']==0 for p in retry['processes'])
 total_elapsed=processes['parallel_wall_clock_seconds']+retry['parallel_wall_clock_seconds']
 jsonout('stage3a_all_processes.json',dict(first_batch=processes,retry_batch=retry,combined_monotonic_seconds=total_elapsed,interrupted_attempts=2,successful_cells=15))
 gate=json.loads((OUT/'precision_gate_decision.json').read_text());assert gate['stage3a_authorized_by_diagnostic']
 ds=load_causal();rows=original();runs=[]
 for p in sorted((OUT/'runs').glob('*.json')):
  m=json.loads(p.read_text());m.setdefault('frequency_setting','daily_weekly');runs.append(m);stem=p.stem;d=read(f'runs/{stem}_metrics.csv');test=d[d.split=='test'].iloc[0].to_dict()
  rows.append(dict(feature_set='causal_relative',seed=42,best_epoch=m.get('best_epoch'),stop_epoch=m.get('stop_epoch'),source=f'revision_work/outputs/causal_features/runs/{stem}_metrics.csv',**test))
 assert len(runs)==15 and len(rows)==30
 baseline=[]
 for h in [5,10,20]:
  train=ds.y[h][ds.split=='train'];y=ds.y[h][ds.split=='test'];tm=np.float32(train.mean());zero=float(np.mean(y*y));mean=float(np.mean((y-tm)**2));baseline.append(dict(horizon=h,train_mean=float(tm),test_mse_train_mean=mean,test_mse_zero=zero))
 b=csvout('reference_test_baselines.csv',baseline).set_index('horizon');out=pd.DataFrame(rows)
 out['test_mse_over_trainmean']=[r.pooled_mse/b.loc[r.horizon,'test_mse_train_mean'] for r in out.itertuples()];out['test_mse_over_zero']=[r.pooled_mse/b.loc[r.horizon,'test_mse_zero'] for r in out.itertuples()];out['R2_OS_trainmean']=1-out.test_mse_over_trainmean;out['R2_OS_zero']=1-out.test_mse_over_zero
 out=csvout('stage3a_side_by_side_metrics.csv',out.sort_values(['mechanism','frequency_setting','horizon','feature_set']));flags=[]
 for h in [5,10,20]:
  d=out[(out.horizon==h)&(out.frequency_setting=='daily_weekly')].set_index(['feature_set','mechanism']);oldS=d.loc[('original_levels','static')];oldA=d.loc[('original_levels','adaptive')];newS=d.loc[('causal_relative','static')];newA=d.loc[('causal_relative','adaptive')];rs=newS.pooled_mse/oldS.pooled_mse-1;ra=newA.pooled_mse/oldA.pooled_mse-1;oldDelta=oldA.pooled_mse-oldS.pooled_mse;newDelta=newA.pooled_mse-newS.pooled_mse
  unchanged=bool(np.sign(oldDelta)==np.sign(newDelta) and abs(rs)<.02 and abs(ra)<.02);becamepositive=bool((oldS.R2_OS_trainmean<=0<newS.R2_OS_trainmean) or (oldA.R2_OS_trainmean<=0<newA.R2_OS_trainmean));icS=newS.cross_sectional_rank_ic-oldS.cross_sectional_rank_ic;icA=newA.cross_sectional_rank_ic-oldA.cross_sectional_rank_ic;improved=bool(rs<=-.02 or ra<=-.02 or becamepositive or icS>=.02 or icA>=.02);worsened=bool(rs>=.02 or ra>=.02);reversed_=bool(np.sign(oldDelta)!=np.sign(newDelta))
  classification='REVERSED' if reversed_ else 'WORSENED' if worsened else 'IMPROVED' if improved else 'UNCHANGED' if unchanged else 'NO RULE MATCH'
  flags.append(dict(horizon=h,classification=classification,original_static_mse=oldS.pooled_mse,causal_static_mse=newS.pooled_mse,static_relative_mse_change=rs,original_adaptive_mse=oldA.pooled_mse,causal_adaptive_mse=newA.pooled_mse,adaptive_relative_mse_change=ra,original_adaptive_minus_static=oldDelta,causal_adaptive_minus_static=newDelta,static_rank_ic_change=icS,adaptive_rank_ic_change=icA,R2_OS_became_positive=becamepositive,flag_unchanged=unchanged,flag_improved=improved,flag_worsened=worsened,flag_reversed=reversed_))
 flagdf=csvout('interim_classification.csv',flags)
 prefixes=[]
 for h in [5,10]:
  def logged(name):
   text=(OUT/name).read_text();return {int(ep):float(v) for ep,v in re.findall(r'VAL_MSE \S+ static_dw_h'+str(h)+r'_s42 (\d+) ([^\s]+)',text)}
  oldlog=logged(f'logs/static_{h}.log');newlog=logged(f'logs/static_{h}_retry.log')
  for ep in sorted(set(oldlog)&set(newlog)):prefixes.append(dict(mechanism='static',horizon=h,epoch=ep,interrupted_val_mse=oldlog[ep],retry_val_mse=newlog[ep],abs_delta=abs(oldlog[ep]-newlog[ep])))
 prefixdf=csvout('static_retry_prefix_comparison.csv',prefixes)
 runtime=csvout('stage3a_runtimes.csv',[{k:r.get(k) for k in ['mechanism','frequency_setting','horizon','seed','best_epoch','stop_epoch','best_alpha','wall_clock_seconds','started_utc','completed_utc','checkpoint']} for r in runs])
 # Validate saved checkpoints, histories, all prediction keys/labels and constant Static weights.
 checks=[];files=[]
 for r in runs:
  mechanism=r['mechanism'];h=r['horizon'];freq=r.get('frequency_setting','daily_weekly');stem=f'{mechanism}_dw_h{h}_s42' if mechanism!='ridge' else f'ridge_{freq}_h{h}_s42'
  check=ROOT/r['checkpoint'];assert check.is_file()
  if 'checkpoint_sha256' in r:assert sha(check)==r['checkpoint_sha256']
  if mechanism=='ridge':
   with check.open('rb') as f:estimator=pickle.load(f)
   assert estimator.alpha==r['best_alpha'];assert np.isfinite(estimator.coef_).all();assert estimator.n_features_in_=={'daily_only':450,'weekly_only':130,'daily_weekly':580}[freq]
  if mechanism!='ridge':
   state=torch.load(check,map_location='cpu');model=exp3._make_experiment3_model('static',None,None) if mechanism=='static' else formal._make_model('daily_weekly','adaptive');model.load_state_dict(state,strict=True);model.eval();assert all(torch.isfinite(v).all() for v in state.values())
   hist=read(f'history/{stem}.csv');assert len(hist)==r['stop_epoch'];assert hist.epoch.tolist()==list(range(1,len(hist)+1));assert np.isfinite(hist[['train_huber','val_huber','val_mse']]).all().all();best=float('inf');chosen=0
   for ep,v in zip(hist.epoch,hist.val_huber):
    if v<best-1e-12:best=v;chosen=int(ep)
   assert chosen==r['best_epoch']
   if mechanism=='static':assert json.loads((OUT/f'weights/{stem}_logits.json').read_text())
  for split in COUNTS:
   p=OUT/f'preds/{stem}_{split}.parquet';frame=pd.read_parquet(p);mask=ds.split==split;assert len(frame)==COUNTS[split];assert frame.split.eq(split).all();assert not frame.duplicated(['ticker','anchor_date']).any();assert np.array_equal(frame.ticker.to_numpy(),ds.ticker[mask]);assert np.array_equal(pd.to_datetime(frame.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy());assert np.array_equal(frame.y_true.to_numpy(),ds.y[h][mask]);assert np.isfinite(frame[['y_true','y_pred']]).all().all()
   if mechanism!='ridge':
    assert np.isfinite(frame[WEIGHTS]).all().all();assert np.allclose(frame[WEIGHTS[:4]].sum(1),1,atol=1e-6);assert np.allclose(frame[WEIGHTS[4:]].sum(1),1,atol=1e-6)
    if mechanism=='static':assert np.ptp(frame[WEIGHTS].to_numpy(),axis=0).max()<1e-7
   files.append(dict(path=rel(p),mechanism=mechanism,frequency_setting=freq,horizon=h,split=split,n_rows=len(frame),sha256=sha(p)))
  checks.append(dict(cell=stem,checkpoint=rel(check),checkpoint_sha256=sha(check),all_split_keys_labels_finite_pass=True,history_and_best_epoch_pass=True if mechanism!='ridge' else 'N/A: non-iterative Ridge'))
 assert len(files)==45
 csvout('stage3a_artifact_checks.csv',checks);csvout('stage3a_prediction_files.csv',files)
 # Every protected file predating this task: metadata plus previously recorded selected hashes.
 before=json.loads((OUT/'protected_inputs_before.json').read_text());problems=[];hashcount=0
 for old in before:
  p=ROOT/old['path']
  if not p.is_file():problems.append(dict(path=old['path'],problem='missing'));continue
  s=p.stat()
  if s.st_size!=old['size'] or s.st_mtime_ns!=old['mtime_ns']:problems.append(dict(path=old['path'],problem='metadata changed'))
  if 'sha256' in old:
   hashcount+=1
   if sha(p)!=old['sha256']:problems.append(dict(path=old['path'],problem='SHA256 changed'))
 integrity=dict(completed_utc=utc(),protected_files=len(before),hash_verified_files=hashcount,passed=not problems,problems=problems);jsonout('protected_inputs_verification.json',integrity)
 tests=json.loads((OUT/'feature_tests.json').read_text());e=json.loads((OUT/'precision_diagnostic_evidence.json').read_text());steps=read('ratio_step_context.csv');labels=read('label_impact_summary.csv');spikes=read('spike_reversals.csv');shifts=read('close_level_shift_summary.csv');fac=read('adjustment_factor_summary.csv');fs=read('adjustment_factor_split_summary.csv');precision=read('snapshot_csv_precision_through_cutoff.csv');pdist=read('precision_distributions.csv');assoc=read('precision_action_association.csv');compat=read('precision_adjustment_rounding_compatibility.csv');top=read('precision_top20_raw_and_standardized.csv')
 txt=['# BLOCKING\n',
 '**No remaining execution blocker for Stage 3a under the user-authorized diagnostic tolerance.** The original cross-snapshot unit test is **FAILED at 1e-6** and is retained unchanged in `feature_tests.json`. Following the requested diagnostic, the additional result is **PASS at numerical precision (documented tolerance)**: standardized max outside the two step spans must be **<1e-4**. This is a documented relaxation, not a retroactive pass of the original test.\n',
 'Attribution limitation: original-feature Adaptive seed-42 checkpoint recovery in Step 3 was DIFFERENT for all three horizons. Stage 3a uses the same imported trainer and current environment, but its change relative to the archived original-feature results cannot be attributed solely to features with exact historical reproducibility. Historical Adaptive seed-42 checkpoint: **NOT FOUND**. See [REPORT_STEP3.md](../REPORT_STEP3.md).\n',
 'Operational issue, corrected: the new exclusive checkpoint-save adapter initially had reversed arguments for two Static workers (h5/h10). These two attempts were interrupted before any checkpoint/output was written, then retrained with the corrected adapter. Interrupted logs and timings are retained; 15 successful cells and two aborted attempts are distinguished. Ridge emitted nine ill-conditioned-matrix LinAlgWarnings during the formal alpha grid (rcond approximately6.9e-9..3.1e-8); warnings are retained in logs/ridge_all.log. No solver/dtype/grid change was made, so Ridge numerical conditioning remains a comparison limitation.\n',
 'Archived original action columns/independently authenticated point-in-time vendor vintages: **NOT FOUND**. Corporate-action identity and true contemporaneous adjustment conventions are not established by this exercise. No claim of point-in-time data is made.\n',
 '# Step 4 interim — Parts 1/2 and Stage 3a\n',f'Completed {utc()} UTC. **Stage 3b and Stage 3c were NOT RUN.** The requested 9 Ridge + 6 neural seed-42 cells completed; two initial Static attempts were interrupted for the adapter correction and are separately recorded. New code is in `revision_work/scripts/causal/`; all outputs are here. All serialization uses exclusive creation and refuses existing destinations. Formal source files, raw snapshots, frozen tensors, existing checkpoints and results were read only. Protected-input check: **{integrity["passed"]}**, {len(before)} files, {hashcount} selected SHA-256 comparisons; [verification](protected_inputs_verification.json).\n',
 '## Protocol and construction\n',
 f'Original source snapshot: `dataset/finance/panel_raw/{{TICKER}}_daily.csv`; new vendor snapshot: `data_raw_revision/prices/{{TICKER}}.csv`, restricted to dates ≤2026-07-07. Both requested audit reports and all 14 prior corporate-action support CSVs were read; [catalog with hashes](prior_corporate_support_catalog.csv). Weekly reconstruction delegates to {cite("scripts/python/panel_build_multiscale_dataset.py",39)}. Its Monday week grouping uses the last actual trading session as `available_date`; the selection is `available_date <= anchor`. The weekly feature denominator is the **daily anchor close**, including when the anchor is later than the last completed weekly bar.\n',
 f'Price features are log(OHLC_tau / daily Close_anchor), and volume is log1p(V_tau) minus its same-window mean. Original float64 source arithmetic is cast to float32 before the unchanged train-only normalizer at {cite("scripts/python/panel_common.py",125)}. Statistics pool all training window entries per ticker/resolution/channel with ddof=0. Model shapes remain Daily 90×5 / Weekly 26×5. No labels, sample keys, chronological splits or the frozen 20-anchor boundary purge were rebuilt or changed for training.\n',
 f'Adaptive calls {cite("scripts/python/panel_pathformer_main_v2.py",1611)} and Static calls {cite("scripts/python/panel_pathformer_experiment3.py",622)}; a process-local data-loader binding supplies original-snapshot causal tensors. Factory observers delegate the original model constructors unchanged and retain the restored best model. The Huber observer records validation MSE from existing forward outputs; it adds no forward calls, backward calls or RNG draws. Static checkpoint publishing copies only newly generated files to exclusive destinations and retains its new `.tmp`; no existing path is replaced. Ridge delegates its four-alpha selection to {cite("scripts/python/panel_baseline_ridge.py",49)}.\n',
 f'Adam lr=1e-4, batch=32, Huber delta=1, gradient clip=1, max100 epochs, patience10, validation-Huber improvement threshold1e-12, seed42 and explicit Train DataLoader generator seed42, CPU; see {cite("scripts/python/panel_pathformer_main_v2.py",1632)} and {cite("scripts/python/panel_pathformer_main_v2.py",1371)}. At most four training workers ran concurrently, each Torch3 / interop1 / OMP3 (BLAS caps also3). Each run records UTC timestamps and monotonic elapsed seconds.\n',
 'Implementation references: '+', '.join([code_ref('revision_work/scripts/causal/build_and_audit.py','def transform('),code_ref('revision_work/scripts/causal/build_and_audit.py','for h in [5,10,20]:'),code_ref('revision_work/scripts/causal/build_and_audit.py','future=np.cumprod'),code_ref('revision_work/scripts/causal/precision_diagnostic.py','def main('),code_ref('revision_work/scripts/causal/precision_decision.py','def main('),code_ref('revision_work/scripts/causal/stage3a_worker.py','def neural('),code_ref('revision_work/scripts/causal/stage3a_worker.py','def ridge_runs(')])+'.\n',
 '## Part 1 — Snapshot consequences\n### Ratio steps\n',table(steps,['ticker','step_date','split','ratio_before','ratio_after','relative_ratio_step','new_same_date_dividend','new_same_date_split','step_minus_dividend_prediction']),
 'CWEN 2026-06-01 and NEE 2026-06-05 are both TEST dates. Their ratio jumps match −dividend / previous vendor Close to approximately 3.2e-8 and 1.3e-8 respectively. Most likely explanation: the June dividends were adjusted differently between snapshots. This is an inference; old action records are NOT FOUND, so it is not an authenticated revision history. Full ±5-session action records are in [ratio_step_nearby_actions.csv](ratio_step_nearby_actions.csv). The values .468 and .623 are vendor-reported.\n',
 '### Label impact (diagnostic only; training labels unchanged)\n',table(labels),
 'All 130,713 retained anchor×horizon labels were rebuilt from NEW Adj Close. [Every comparison](label_comparison_all_anchors.csv), [every affected anchor/date and split](label_affected_anchors.csv). This includes small >1e-6 numeric changes beyond CWEN/NEE. Rows in the latter file are not automatically evidence of economically meaningful revisions.\n',
 table(read('saved_predictions_new_label_mse.csv')),
 'The archived seed-42 predictions above are held fixed; only the evaluation labels change in this counterfactual. Static remains lower-MSE than Adaptive for each horizon.\n',
 '### Spike-reversal check\n',table(spikes[spikes.in_requested_range],['ticker','spike_date','reversal_date','return_spike','return_reversal','original_previous_close','original_spike_close','original_reversal_close','old_values_identical','old_returns_match_1e_6']),
 f'{len(spikes)} cases in the full overlapping history; {int(spikes.in_requested_range.sum())} within 2015-08-05..2026-07-07. [All cases and three-date values](spike_reversals.csv). All six in-range cases also occur in the original snapshot. Vendor Close is already split-normalized; a discontinuity previously called “unexplained by split” is not evidence of a bad tick. Independent trade/source validation remains unverified.\n',
 '### Ex-post adjustment factor\n',
 'c(t)=AdjClose_t/[Close_t × product of vendor split ratios dated strictly after t]. The product includes all later reported events through the new download date; values plotted are limited to 2015-08-05..2026-07-07. c can be below1, so adjusted levels do not always exceed the formula-based as-traded level. This requested algebraic reconstruction is not an authenticated historical traded-price series, particularly for BEP’s BEPC distribution.\n',table(fac,['ticker','min_c','max_c']),
 '![Adjustment factor](fig_adjustment_factor.png)\n',
 'For each ticker/resolution/split, the requested standardized shift is (c−1) × V2 mean_close_train / std_close_train. Full 102-row table follows and is also saved in [adjustment_factor_split_summary.csv](adjustment_factor_split_summary.csv); [all dates and split products](adjustment_factors_by_date.csv). These are arithmetic diagnostics, not causal attribution of model behavior.\n',table(fs,['ticker','resolution','split','min_c','max_c','implied_shift_min','implied_shift_max']),
 '### Original level-feature distribution shift\n',table(shifts),
 table(read('original_last_observation_distribution.csv').query("split == 'test' and fraction_outside_train_last_range > 0.5"),['ticker','resolution','channel','fraction_outside_train_last_range','fraction_outside_train_all_entry_range']),
 'Only BLNK has >50% of test last-observation close values outside the observed training last-observation close range (Daily ~74.2%, Weekly ~76.2%). “Outside range” is not the same as |z|>3; BLNK’s latter fraction is zero. FSLR has large positive standardized test levels, despite some values still being inside training extrema. All 510 ticker/resolution/channel/split mean/std(ddof0)/|z|>3 entries and range fractions are in [original_last_observation_distribution.csv](original_last_observation_distribution.csv).\n',
 '![Original level distributions](fig_level_shift.png)\n',
 '## Part 2 — Causal feature tests and precision diagnostic\n',
 f'Original-source windows reproduce the frozen V2 float32 windows exactly: **PASS**, all34 ticker/resolution checks. Controlled common price rescaling: **PASS**. No look-ahead: **PASS**, all87,142 source dependencies, with every source end and daily reference date ≤anchor. Shapes/counts/finiteness/frozen labels: **PASS**, 43,571 rows; train30,634 / val6,290 / test6,647. Saved raw/standardized tensors and index are under `tensors/`; [train-only statistics](causal_normalization_stats.csv).\n',
 f'Original cross-snapshot test: **FAILED at 1e-6**. Outside the12 dependency spans containing a genuine ratio step, raw max={tests["cross_snapshot_raw_max_outside_steps"]:.12g}, standardized max={tests["cross_snapshot_standardized_max_outside_steps"]:.12g}. Raw failed windows={tests["n_raw_violations_outside_steps"]}, standardized failed windows={tests["n_standardized_violations_outside_steps"]}. Overall including steps: raw max={tests["cross_snapshot_raw_max_overall"]:.12g}, standardized max={tests["cross_snapshot_standardized_max_overall"]:.12g}. The excluded span runs from earliest underlying source date through the daily anchor reference, including the gap after a completed weekly bar.\n',
 '[All differing windows](invariance_differing_windows.csv), [all windows and both normalizer comparisons](invariance_all_windows.csv), [strict violations outside steps](invariance_violations_outside_steps.csv), [unit test record](feature_tests.json). The same ORIGINAL causal training statistics are used for the primary cross-snapshot comparison; independently refitted NEW statistics are supplemental only.\n',
 '### User-requested diagnostic, before any training\n',
 'Distribution below is over every non-step window element, including exact zeros, 25,267,700 observations. Full max/99.9th/99th/median tables, per ticker × channel × resolution and for both element errors and within-window maxima: [precision_distributions.csv](precision_distributions.csv).\n',
 fmt_table(['Scale','Max','99.9th','99th','Median'],[[scale]+[number(e[k][q]) for q in ['max','p99_9','p99','median']] for scale,k in [('raw','raw_distribution'),('standardized','standardized_distribution')]]),
 'Top20 distinct windows by standardized error (and another top20 by raw error) with exact worst channel, adjusted source/anchor prices, new vendor Close, source dates and nearby action types: [precision_top20_raw_and_standardized.csv](precision_top20_raw_and_standardized.csv).\n',table(top[top.ranking=='standardized'],['ticker','resolution','anchor_date','channel','source_start','source_end','max_standardized_window','original_source_value','new_adjusted_source_value','original_anchor_close','new_anchor_close','nearby_events']),
 'The standardized top20 are NEE windows, mainly with adjusted source closes near30.9 versus an overall NEE source-close median62.5 (about the19th percentile). They are lower-price observations, not penny-price anomalies. Three top20 source spans and one top20 anchor are within±5 sessions of an action; none is close to the June2026 CWEN/NEE steps. Low price association is compatible with relative quantization amplification but does not establish its cause. [Per-ticker price-level comparison](precision_price_level_check.csv).\n',
 'Across all tickers the naive event enrichment is ~2.7× because only dividend-paying tickers have nonzero revisions; this is ticker confounding. Within each affected ticker, source/anchor enrichment is at most1.0255×, with rates near their same-ticker baseline or lower. These descriptive ratios support absence of event-date concentration; they are not significance tests. The diagnostic uses a declared 10% descriptive enrichment screen (max ratio<1.1) solely to operationalize “not concentrated.”\n',table(assoc[assoc.ticker!='all']),
 'Tail sensitivity by ticker and resolution is additionally saved in [precision_tail_action_sensitivity.csv](precision_tail_action_sensitivity.csv). Small top20/top1% tails can show local enrichment (for example AES); overlapping windows repeat the same source dates, so these are not independent event counts. The gate screen uses all original-threshold-violating windows and the global worst20, not a claim that every possible tail selection has zero association. This limits the strength of the numerical attribution.\n',
 'CSV precision is decimal text parsed as float64. Each ticker/channel’s min/median/max stored decimal places, fraction numerically representable as float32 and residual are in [snapshot_csv_precision_through_cutoff.csv](snapshot_csv_precision_through_cutoff.csv). Original adjusted Close and NEW OHLC/Adj Close are numerically float32-valued, whereas original adjusted O/H/L are double-precision products. Many printed decimal digits therefore reflect a binary float32 value, not equivalent independent price accuracy.\n',
 table(precision[((precision.snapshot=='ORIGINAL')&(precision.channel=='close'))|((precision.snapshot=='NEW')&(precision.channel=='Adj Close'))],['ticker','snapshot','channel','decimal_places_min','decimal_places_median','decimal_places_max','fraction_exact_float32','max_float32_rounding_residual']),
 'Numerical explanation: all four daily OHLC ratios agree within8.8e-16, identifying a common date-specific adjustment multiplier rather than altered within-day price geometry. All volume values and dividend-free ticker features agree exactly. Residual log multipliers after removing only the two documented steps lie within a conservative conditional accumulated float32-factor rounding envelope (gamma_n=n·u/(1−n·u), u=eps32/2, n=later dividend counts in both snapshots+4 endpoint operations). Final CSV serialization alone has an illustrative four-endpoint bound4·eps32≈4.77e-7 and is insufficient for the largest1.53e-6 raw feature error. The cumulative-envelope compatibility supports a numerical-precision explanation; Yahoo’s internal adjustment arithmetic and exact historical operations are **UNVERIFIED**. It does not prove rounding as the unique cause.\n',table(compat,['ticker','max_daily_OHLC_log_scale_spread','max_piecewise_close_log_scale_residual','max_residual_over_conditional_cumulative_rounding_envelope','fraction_residuals_inside_envelope']),
 f'**Decision: {gate["status"]}.** All three user conditions are supported: max standardized error2.04e-5<1e-4; no within-ticker action-date concentration; discrepancies compatible with adjustment-factor precision. Revised tolerance1e-4 applies only outside the explicitly documented step-dependent windows. It is approximately0.01% of one standardized unit; observed max is about0.00204% of one standardized unit. The original1e-6 failure remains recorded. Training below uses **ORIGINAL-snapshot causal features only**. [Decision record](precision_gate_decision.json), [raw evidence](precision_diagnostic_evidence.json).\n',
 '## Stage 3a — Results\n',
 'MSE, MAE, Pearson correlation, direction accuracy, mean cross-sectional Rank IC and PredStd/TrueStd use the imported V2 metric functions. Original-feature metrics are recomputed from the required archived prediction files in float32; labels and all6647 keys match. Epochs come from archived formal summaries. Ridge has no training epochs: N/A, with four-alpha validation-MSE selection tables rather than fictitious neural histories.\n',
 table(out,['mechanism','frequency_setting','horizon','feature_set','pooled_mse','pooled_mae','pooled_corr','pooled_da','cross_sectional_rank_ic','pooled_pred_std_over_true_std','best_epoch','stop_epoch']),
 '### Zero / Global Train-Mean skill\n',table(b.reset_index()),table(out,['mechanism','frequency_setting','horizon','feature_set','test_mse_over_trainmean','test_mse_over_zero','R2_OS_trainmean','R2_OS_zero']),
 'Global Train-Mean uses one pooled training-label mean per horizon, unchanged across feature sets. R²_OS(train-mean)=1−MSE_model/MSE_trainmean; zero-relative R² is shown separately. Negative values mean worse MSE than that reference. [Full side-by-side metrics](stage3a_side_by_side_metrics.csv).\n',
 '### Interim descriptive classification\n',
 'Rules can overlap. Declared reporting priority: REVERSED → WORSENED → IMPROVED → UNCHANGED; every underlying flag and raw value remains in [interim_classification.csv](interim_classification.csv). These are descriptive review flags for seed42, not statistical significance or five-seed evidence. “R²_OS becomes positive” is interpreted as crossing from≤0 to>0 relative to Global Train-Mean.\n',table(flagdf),
 '### Runtime and saved artifacts\n',table(runtime),f'Bounded-process elapsed: {total_elapsed:.3f}s across first batch plus retry batch; [all process records](stage3a_all_processes.json). Per-run times include setup, serialization and all-split inference. Trainer-only runtime is also retained in each trainer summary.\n',
 'Adapter correction replay: the validation-MSE prefix from each interrupted Static attempt is compared with the corresponding completed retry in [static_retry_prefix_comparison.csv](static_retry_prefix_comparison.csv). Maximum absolute prefix difference='+number(prefixdf.abs_delta.max())+'. No additional training was performed for this comparison.\n',
 '45 prediction Parquets: each has ticker, anchor_date, split, y_true, y_pred, with all8 per-sample router weights for Adaptive and constant weights for Static. Static logits are additionally in `weights/*_logits.json`. All15 selected checkpoints are in `checkpoints/`; neural state dictionaries are restored best states; Ridge pickles hold the selected fitted estimators. Each neural history contains train/val Huber(delta1) plus validation MSE, from unchanged training forward outputs. Ridge `history/*_alpha_grid.csv` saves all4 trials; its epoch CSV is empty by design.\n',table(pd.DataFrame(files),['mechanism','frequency_setting','horizon','split','n_rows','path']),
 '[Artifact/hash checks](stage3a_artifact_checks.csv), [file inventory](stage3a_prediction_files.csv), [final manifest and package versions](MANIFEST_STEP4.json). All45 Parquets pass readback, key order, unique keys, frozen labels, finiteness and exact requested row counts. Neural checkpoint states load strictly into imported models; epoch count and restored-best metadata agree with history minima. All source paths/line references refer to unchanged original code unless inside the permitted new namespace. Failed/interrupted no-training diagnostic logs are retained, alongside successful retry logs.\n',
 '**STOP after Stage 3a.** No Stage 3b multiseed runs and no Stage 3c as-of-level training have been launched.\n',
 'Issuer context: [Brookfield 2020 distribution notice](https://www.sec.gov/Archives/edgar/data/1791863/000179186320000006/bepcq22020-ex992.htm) documents one BEPC share per four BEP units. [Clearway Q2 2026 filing](https://www.sec.gov/Archives/edgar/data/1567683/000162828026053421/cwen-20260630.htm) documents the May1,2026 ClassA-to-ClassC one-for-one conversion and CWEN.A delisting; CWEN continues. These facts partially clarify identity but do not validate the historical vendor security mapping.\n',
 '## Risk → CHECKED / MITIGATED / REMAINS UNVERIFIED\n',
 fmt_table(['Risk','Status','Evidence and remaining limit'],[
 ['Retroactive dividend rescaling','CHECKED / MITIGATED','Snapshot ratio and label consequences measured; relative-price features remove common price multipliers. True contemporaneous vendor vintages remain unverified.'],
 ['Split / reverse-split adjustment','CHECKED / REMAINS UNVERIFIED','Vendor events and formula-based factors inspected; Yahoo Close already split-normalized. Nominal traded-price reconstruction and economic-event identity are not authenticated.'],
 ['Vendor revisions (CWEN/NEE)','CHECKED','June1/June5 TEST steps align with dividend adjustments; impact on labels/prediction MSE measured. Exact vendor revision chronology unverified.'],
 ['Volume adjustment convention','CHECKED partially / REMAINS UNVERIFIED','Volume is exactly equal across these two snapshots and unchanged for training; demeaned log1p removes within-window log level. True historical split-volume convention not independently authenticated.'],
 ['Ex-post adjustment leakage into level features','MITIGATED','Relative OHLC features are invariant to common retroactive price multipliers at documented precision; this does not prove the archived V2 score was caused by leakage. Stage3c NOT RUN.'],
 ['Non-stationarity of level features','CHECKED / MITIGATED','Original train/test distributions quantified; absolute price/volume levels removed in causal features. Return/volatility/regime non-stationarity remains.'],
 ['Historical security identity','REMAINS UNVERIFIED','BEP July30,2020 vendor split1.251 corresponds to BEPC distribution, not a literal share split. BEP partnership vs BEPC corporation; CWEN ClassC vs historical CWEN.A ClassA need security-master mapping.'],
 ['Point-in-time vendor vintages','REMAINS UNVERIFIED','Original authenticated timestamp/action archive NOT FOUND; current data are ex-post snapshots. Causal construction does not make the data point-in-time.'],
 ['Survivorship / delisted securities','REMAINS UNVERIFIED','Frozen17-name universe retained; no historical membership or delisted-security coverage reconstructed.']]),
]
 if problems:txt.insert(1,'Protected-input verification FAILED: '+json.dumps(problems)+'.\n')
 with target(OUT/'REPORT_STEP4_INTERIM.md').open('x') as f:f.write('\n'.join(txt))
 print('REPORT COMPLETE',utc());print(flagdf.to_string(index=False));print('INTEGRITY',integrity)
if __name__=='__main__':main()
