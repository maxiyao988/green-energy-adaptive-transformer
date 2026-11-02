"""Post-barrier Step22A evaluation summaries and imported Step16 joint tests."""
import os,sys
sys.dont_write_bytecode=True
from io22 import *
sys.path.insert(0,str(ROOT/'revision_work/scripts/analysis16'))
from statistics16_v3 import s
import numpy as np,pandas as pd,platform
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')

def read(p):return pd.read_csv(p,float_precision='round_trip')
def finite_summary(x):
 v=np.asarray(x,float)
 return float(v.mean()) if np.isfinite(v).all() else np.nan
def finite_sd(x):
 v=np.asarray(x,float)
 return float(v.std(ddof=1)) if len(v)==5 and np.isfinite(v).all() else np.nan
def md(d):
 def c(v):
  if pd.isna(v):return 'N/A'
  if isinstance(v,(float,np.floating)):return f'{v:.6g}'
  return str(v).replace('|',' / ').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(c(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)])+'\n'
def main():
 check_protocol();clock=time.perf_counter();started=utc()
 assert (OUT/'phases/remaining_seeds.json').exists() and (OUT/'phases/final_test_evaluation.json').exists()
 bs=bindings();select=json.loads((OUT/'SELECTED_RATES.json').read_text());chosen={(j['input_set'],j['backbone'],j['h'],j['variant']):j['lr'] for j in select['cells']}
 fitrows=[];history=[];selected_fits=[]
 for p in sorted((OUT/'fits').glob('*/COMPLETE.json')):
  j=verify_job(p);fitrows.append({k:j[k] for k in ['id','backbone_id','input_set','backbone','h','seed','variant','lr','status','error','eligible','validation_mse','best_epoch','epochs','best_val_huber','started_utc','ended_utc','wall_seconds','initial_trainable_sha256','trainable_parameters','test_reads']})
  selected=j['lr']==chosen[(j['input_set'],j['backbone'],j['h'],j['variant'])]
  hd=read(p.parent/'history.csv')
  for key in ['input_set','backbone','h','seed','variant','lr','status']:hd[key]=j[key]
  hd['selected_for_final']=selected;hd['best_epoch']=j['best_epoch'];history.append(hd)
  if selected:selected_fits.append(j)
 assert len(fitrows)==252 and len(selected_fits)==180
 fits=pd.DataFrame(fitrows);hist=pd.concat(history,ignore_index=True);tune=fits[fits.seed.eq(0)].copy();assert len(tune)==108
 csv(OUT/'all_training_runs.csv',fits);csv(OUT/'tuning_trials.csv',tune);csv(OUT/'selected_learning_rates.csv',pd.DataFrame(select['cells']))
 csv(OUT/'learning_curves_all_trials.csv',hist);csv(OUT/'learning_curves_selected.csv',hist[hist.selected_for_final])
 selected_frame=pd.DataFrame([{k:j[k] for k in ['input_set','backbone','h','seed','variant','lr','best_epoch','epochs','status']} for j in selected_fits])
 csv(OUT/'best_epochs_per_seed.csv',selected_frame)
 distribution=selected_frame.groupby(['input_set','backbone','h','variant','lr','best_epoch'],dropna=False).size().rename('n_seeds').reset_index();csv(OUT/'best_epoch_distribution.csv',distribution)
 epochmeans=hist[hist.selected_for_final].groupby(['input_set','backbone','h','variant','lr','epoch'],as_index=False).agg(n_seeds=('seed','nunique'),train_huber_mean=('train_loss','mean'),val_huber_mean=('val_loss','mean'))
 csv(OUT/'learning_mean_by_epoch.csv',epochmeans)
 metrics=[];routers=[];filelist=[];panels={};raw_router=[];integrity=[]
 for b in bs:
  for v in ['fixed']+VARIANTS:
   folder=OUT/'evaluation'/f"{b['id']}_{v}";j=verify_job(folder/'COMPLETE.json');meta=dict(input_set=b['input_set'],backbone=b['backbone'],h=b['horizon'],seed=b['seed'],variant=v)
   metrics.append(dict(**meta,lr=j['lr'],status=j['status'],mse=j['mse'],mean_rank_ic=j['mean_rank_ic'],da=j['da'],n_zero_realized=j['n_zero_realized'],n_valid=j['n_valid'],test_inference_passes=j['test_inference_passes']))
   d=read(folder/'date_metrics.csv').sort_values('anchor_date');assert len(d)==391 and d.n_tickers.eq(17).all();panels[tuple(meta[k] for k in ['input_set','backbone','h','seed','variant'])]=d
   r=read(folder/'routing_diagnostics.csv')
   frame=read(folder/'predictions_and_weights.csv');assert len(frame)==6647 and not frame.duplicated(['ticker','anchor_date']).any()
   for key,value in meta.items():r[key]=value
   # Preserve raw float32 SD residuals alongside mathematically constant routing diagnostics.
   old=pd.read_parquet(ROOT/b['old'][v]['predictions']['test']['path'],columns=WEIGHTS)
   for ix,rr in r.iterrows():
    a=frame[WEIGHTS].to_numpy(np.float32) if rr.regime=='tuned' else old[WEIGHTS].to_numpy(np.float32)
    z=a[:,:4] if rr.branch=='daily' else a[:,4:]
    r.loc[ix,'raw_float32_mean_sample_sd']=float(z.std(0).mean()) if np.isfinite(z).all() else np.nan
   routers.append(r)
   integrity.append(dict(**meta,rows=len(frame),dates=frame.anchor_date.nunique(),tickers=frame.ticker.nunique(),n_valid=int((frame.y_true!=0).sum()),status=j['status'],test_inference_passes=j['test_inference_passes']))
   for a in j['artifacts']:filelist.append(dict(**meta,**a))
 M=pd.DataFrame(metrics);assert len(M)==240 and M.groupby(['input_set','backbone','h','variant']).seed.nunique().eq(5).all()
 for meta,g in M.groupby(['input_set','backbone','h']):assert g.n_valid.nunique()==1 and g.n_zero_realized.nunique()==1
 csv(OUT/'test_metrics_per_seed.csv',M);csv(OUT/'prediction_files.csv',pd.DataFrame(filelist));csv(OUT/'prediction_integrity.csv',pd.DataFrame(integrity))
 groups=['input_set','backbone','h','variant']
 summary=M.groupby(groups,as_index=False).agg(n_seeds=('seed','nunique'),n_complete=('status',lambda a:int(a.eq('COMPLETE').sum())),mse_mean=('mse',finite_summary),mse_sd=('mse',finite_sd),rank_ic_mean=('mean_rank_ic',finite_summary),rank_ic_sd=('mean_rank_ic',finite_sd),da_mean=('da',finite_summary),da_sd=('da',finite_sd))
 csv(OUT/'five_seed_summary.csv',summary);csv(OUT/'TABLE7_TUNED.csv',summary)
 # Stored Step16 primary metrics and Step21 zero-excluded DA; never use archived all-row DA.
 old=read(ROOT/'revision_work/outputs/analysis16/B_forecast_skill.csv');old=old[old.group.eq('frozen')].rename(columns={'config':'backbone','model':'variant'})
 assert len(old)==240
 oldda=read(ROOT/'revision_work/outputs/step21/frozen_primary_da_per_seed.csv').rename(columns={'model':'variant'})
 old=old[groups+['seed','mse','mean_rank_ic','id']].merge(oldda[groups+['seed','da']],on=groups+['seed'],validate='one_to_one')
 csv(OUT/'Step16_reference_per_seed.csv',old)
 oldsummary=old.groupby(groups,as_index=False).agg(n_seeds=('seed','nunique'),mse_mean=('mse',finite_summary),mse_sd=('mse',finite_sd),rank_ic_mean=('mean_rank_ic',finite_summary),rank_ic_sd=('mean_rank_ic',finite_sd),da_mean=('da',finite_summary),da_sd=('da',finite_sd))
 csv(OUT/'S6_STEP16_REFERENCE.csv',oldsummary)
 side=summary.merge(oldsummary,on=groups,suffixes=('_tuned','_Step16'),validate='one_to_one');csv(OUT/'tuned_vs_Step16.csv',side)
 R=pd.concat(routers,ignore_index=True);csv(OUT/'router_diagnostics_per_seed.csv',R)
 RS=R.groupby(groups+['branch','regime'],as_index=False).agg(n_seeds=('seed','nunique'),mean_abs_deviation_mean=('mean_abs_deviation_from_equal',finite_summary),mean_abs_deviation_seed_sd=('mean_abs_deviation_from_equal',finite_sd),sample_sd_mean=('mean_sample_sd',finite_summary),sample_sd_seed_sd=('mean_sample_sd',finite_sd),between_date_share_mean=('between_date_share',finite_summary),between_date_share_seed_sd=('between_date_share',finite_sd),n_finite_between_date_shares=('between_date_share','count'),n_constant_seeds=('constant_to_1e7','sum'))
 csv(OUT/'router_diagnostics_summary.csv',RS)
 joint=[];difference_rows=[];drawchecks=[];directchecks=[]
 for h in HS:
  _,di,si=s.draws(h,5,391)
  with np.load(ROOT/f'revision_work/outputs/analysis16/rng/draws_h{h}.npz') as z:
   assert np.array_equal(di,z['joint_date_indices']) and np.array_equal(si,z['seed_indices'])
  drawchecks.append(dict(h=h,seed_indices_equal=True,date_indices_equal=True,B=20000,L=h,entropy=[20261019,h,h,1]))
 for inp in ['O','C']:
  for bk in ['S1','S2']:
   for left,right in [('adaptive','constant_gate'),('static','adaptive')]:
    for metric,col in [('mse','mse'),('rank_ic','rank_ic'),('da','n_correct')]:
     for h in HS:
      A=[panels[(inp,bk,h,seed,left)] for seed in SEEDS];Z=[panels[(inp,bk,h,seed,right)] for seed in SEEDS]
      dates=A[0].anchor_date.to_numpy()
      assert all(np.array_equal(dates,d.anchor_date.to_numpy()) for d in A+Z)
      X=np.ascontiguousarray(np.stack([a[col].to_numpy(float)-z[col].to_numpy(float) for a,z in zip(A,Z)]))
      valid_test=True;reason=''
      if not np.isfinite(X).any(axis=1).all():valid_test=False;reason='at least one paired seed has no finite outcomes; not silently dropped'
      if valid_test:
       theta,boot,_=s.distributions(X,h)
       if metric=='da':
        den=np.stack([a.n_valid.to_numpy(float) for a in A]);assert np.array_equal(den,np.stack([z.n_valid.to_numpy(float) for z in Z]));dpoint,dboot,_=s.distributions(den,h);theta/=dpoint;boot/=dboot;v=X.sum(1)/den.sum(1)
       else:v=np.nanmean(X,axis=1)
       if not np.isfinite(boot).all():valid_test=False;reason='undefined resampled statistic retained as N/A'
      if valid_test:
       stats=s.series_stats(boot,theta);wins=int((v<0 if metric=='mse' else v>0).sum());losses=int((v>0 if metric=='mse' else v<0).sum());ties=int((v==0).sum())
       _,di,si=s.draws(h,5,391);direct=[]
       for i in range(16):
        if metric=='da':vv=X[si[i]][:,di[i]].sum()/den[si[i]][:,di[i]].sum()
        else:vv=np.nanmean(X[si[i]][:,di[i]],axis=1).mean()
        direct.append(vv)
       err=float(np.max(np.abs(np.array(direct)-boot[:16])));assert err<1e-12
       directchecks.append(dict(input_set=inp,backbone=bk,h=h,contrast=left+'-minus-'+right,metric=metric,max_abs_error=err))
      else:stats=dict(theta_hat=np.nan,ci_lower_95=np.nan,ci_upper_95=np.nan,centered_p_two_sided=np.nan);wins=losses=ties=0;v=np.full(5,np.nan)
      meta=dict(input_set=inp,backbone=bk,h=h,contrast=left+'-minus-'+right,metric=metric,B=20000,L=h,n_seeds=5,n_dates=391,n_missing_seed_dates=int((~np.isfinite(X)).sum()),test_defined=valid_test,undefined_reason=reason,seed_wins=wins,seed_losses=losses,seed_ties=ties,positive_favors_first=metric!='mse')
      joint.append(dict(**meta,**stats,holm_input_p=stats['centered_p_two_sided'] if valid_test else 1.))
      for i,seed in enumerate(SEEDS):
       for date,value in zip(dates,X[i]):difference_rows.append(dict(**{k:meta[k] for k in ['input_set','backbone','h','contrast','metric']},seed=seed,anchor_date=date,paired_difference=value,aggregation='correct-count numerator' if metric=='da' else 'paired within-date mean loss / IC',n_valid=int(A[i].set_index('anchor_date').loc[date,'n_valid']) if metric=='da' else 17))
 J=s.corrected(pd.DataFrame(joint),['input_set','backbone','contrast','metric'],p='holm_input_p');assert len(J)==72
 csv(OUT/'joint_bootstrap_tests.csv',J);csv(OUT/'paired_seed_date_differences.csv',pd.DataFrame(difference_rows))
 csv(OUT/'seed_wins.csv',J[['input_set','backbone','h','contrast','metric','seed_wins','seed_losses','seed_ties']])
 js(OUT/'BOOTSTRAP_VERIFICATION.json',dict(draws=drawchecks,direct_checks=directchecks,all_match=True))
 # Mean curves contain observed histories only; no interpolation after stopping.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import ScalarFormatter
 plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 palette={'static':'#0072B2','adaptive':'#D55E00','constant_gate':'#009E73'}
 for h in HS:
  fig,axes=plt.subplots(2,2,figsize=(16/2.54,12/2.54),layout='constrained')
  for i,bk in enumerate(['S1','S2']):
   for j,inp in enumerate(['O','C']):
    ax=axes[i,j];g=epochmeans[(epochmeans.h==h)&(epochmeans.backbone==bk)&(epochmeans.input_set==inp)]
    for v in VARIANTS:
     q=g[g.variant==v];ax.plot(q.epoch,q.train_huber_mean,color=palette[v],label=v+' train');ax.plot(q.epoch,q.val_huber_mean,color=palette[v],linestyle='--',label=v+' val')
    ax.text(.01,1.03,f'({chr(97+i*2+j)}) {bk}, {inp}',transform=ax.transAxes);ax.set_xlabel('Epoch');ax.set_ylabel('Huber loss');ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.set_xlim(left=1)
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
  for fmt in ['png','pdf']:
   with target(OUT/f'learning_curves_h{h}.{fmt}').open('xb') as fh:fig.savefig(fh,format=fmt,dpi=300)
  plt.close(fig)
 # Phase barriers, thread settings and source protection are part of final verification.
 phase_rows=[]
 for p in sorted((OUT/'phases').glob('*.json')):
  phase=json.loads(p.read_text());phase_rows.append(dict(phase=p.stem,started_utc=phase['started_utc'],ended_utc=phase['ended_utc'],wall_seconds=phase['wall_seconds'],tasks=phase['n_tasks']))
 phaseframe=pd.DataFrame(phase_rows).sort_values('started_utc');csv(OUT/'phase_timestamps.csv',phaseframe)
 phases={r['phase']:r for r in phase_rows}
 assert phases['cache']['ended_utc']<=phases['validation_replay']['started_utc']
 assert phases['validation_replay']['ended_utc']<=phases['test_replay']['started_utc']
 assert phases['test_replay']['ended_utc']<=phases['tuning']['started_utc']
 assert phases['tuning']['ended_utc']<=phases['remaining_seeds']['started_utc']
 assert phases['remaining_seeds']['ended_utc']<=phases['final_test_evaluation']['started_utc']
 assert select['locked_utc']<phases['final_test_evaluation']['started_utc']
 assert fits.test_reads.eq(0).all()
 for j in selected_fits:
  assert j['profile']['torch_num_threads']==j['profile']['torch_num_interop_threads']==12
  assert j['profile']['environment']['OMP_NUM_THREADS'] is None
  assert j['profile']['environment']['OMP_DYNAMIC']==j['profile']['environment']['MKL_DYNAMIC']=='FALSE'
 replay=pd.concat([read(p) for p in sorted((OUT/'replay').glob('*.csv'))],ignore_index=True);assert len(replay)==480 and replay.passed.all();csv(OUT/'cache_replay_verification.csv',replay)
 frozen_evidence=[dict(source='revision_work/outputs/frozen/scripts/frozen_worker_v3.py',lines='76;116-143;165-167',sha256=sha(FR/'scripts/frozen_worker_v3.py'),result='eval-mode caching; train override and per-forward frozen eval assertions')]
 protected=json.loads((OUT/'PROTECTED_INPUTS.json').read_text())
 for p,digest in protected.items():assert sha(ROOT/p)==digest,p
 failures=fits[fits.status.ne('COMPLETE')]
 blocking=[]
 if len(failures):blocking.append(f'{len(failures)} divergent/failed training trials are retained with reasons; affected final cells are unavailable and never dropped.')
 if not J.test_defined.all():blocking.append(f'{int((~J.test_defined).sum())} paired tests are undefined; conservative Holm input1 is retained, raw p N/A.')
 validation=dict(status='PASS' if not blocking else 'COMPLETED_WITH_REPORTED_LIMITATIONS',protocol_sha256=check_protocol(),frozen_mode_evidence=frozen_evidence,cache_manifests=60,replay_cells_validation=240,replay_cells_test=240,replay_max_abs_difference=float(replay.max_abs_prediction_difference.max()),replay_performance_metrics_computed=False,training_runs=252,tuning_trials=108,selected_rate_groups=36,final_cells=240,summary_groups=48,joint_tests=72,holm_families=24,all_training_test_reads_zero=True,phase_order_verified=True,all_final_test_inference_once_for_complete_cells=bool(M[M.status.eq('COMPLETE')].test_inference_passes.eq(1).all()),protected_files_unchanged=len(protected),analysis_started_utc=started,analysis_ended_utc=utc(),analysis_wall_seconds=time.perf_counter()-clock,package_versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),reporting_rule='tuned results replace Table7 regardless of direction; Step16 reference moves to S6; existing manuscripts untouched')
 js(OUT/'VALIDATION_STEP22A.json',validation)
 display=summary[groups+['n_seeds','n_complete']].copy()
 for label,mu,sd in [('TEST MSE','mse_mean','mse_sd'),('Rank IC','rank_ic_mean','rank_ic_sd'),('DA (%)','da_mean','da_sd')]:
  scale=100 if label=='DA (%)' else 1
  display[label]=[f'{a*scale:.6g} ± {b*scale:.6g}' if np.isfinite(a) and np.isfinite(b) else 'N/A (five seeds not all defined)' for a,b in zip(summary[mu],summary[sd])]
 side_display=side[groups].copy()
 for regime in ['tuned','Step16']:
  for label,mu,sd in [('MSE','mse_mean','mse_sd'),('IC','rank_ic_mean','rank_ic_sd'),('DA (%)','da_mean','da_sd')]:
   scale=100 if label=='DA (%)' else 1;side_display[regime+' '+label]=[f'{a*scale:.6g} ± {b*scale:.6g}' if np.isfinite(a) and np.isfinite(b) else 'N/A' for a,b in zip(side[mu+'_'+regime],side[sd+'_'+regime])]
 columns=['input_set','backbone','h','contrast','metric','theta_hat','ci_lower_95','ci_upper_95','centered_p_two_sided','p_holm_3h','seed_wins','seed_losses','seed_ties']
 report=['# REPORT STEP22A — Frozen-aggregator learning-rate grid (Phase A only)','','## BLOCKING','', '\n'.join(blocking) if blocking else 'None. All prescribed cells, phase barriers, replay tolerances, thread settings and protected-source checks passed.','',
  '## Immutable protocol and reporting commitment','',f"Protocol SHA-256: `{check_protocol()}`. [Protocol](PROTOCOL_STEP22A.md), [RNG manifest](RNG_MANIFEST.json). The tuned results replace manuscript Table7 regardless of direction; Step16 lr1e-4 results move to Supplementary S6. New CSVs are [TABLE7_TUNED.csv](TABLE7_TUNED.csv) and [S6_STEP16_REFERENCE.csv](S6_STEP16_REFERENCE.csv). Existing manuscripts/workbooks are unchanged. No Phase B or head-retrained model was run.",'',
  '## Frozen modules, cache and replay','',
  'Step15 frozen_worker_v3.py:76 uses eval during expert caching; lines116–143 freeze input projections, positional encoding, all eight experts and the head, including an override that keeps them eval during routing fits; lines165–167 assert this per forward. Step22 imports this configuration and the formal models/head; it does not reimplement them. All frozen parameter fingerprints remain unchanged. Source evidence and hash are in VALIDATION_STEP22A.json.','',
  'All60 existing expert caches were reused read-only after full hash/shape/value verification. Adaptive input vectors are zero-copy branch-cache views; constant-input vectors are TRAIN-only means. The original differentiable aggregation/head forward reconstructs the head inputs for each set of routing weights. No expert or backbone training was performed. New cache manifests identify every tensor and source.','',
  'Each of240 VALIDATION cells was replayed first, followed by240 TEST cells. Only stored predictions were compared, using dummy targets and reading no realized-return column. The outputs were PASS/FAIL plus max absolute difference; no MSE, IC, DA or other performance metric was calculated during replay. All passed <=1e-6; maximum difference '+f"{validation['replay_max_abs_difference']:.6g}. Full results: [cache_replay_verification.csv](cache_replay_verification.csv). Replay affected only the go/stop gate.",'',
  '## Read order, timing and numerical profile','',md(phaseframe),
  'Cache binding/hash reads precede the cache-pool phase and are timestamped separately in events/. Overall training start/end and the final reporting stage are also timestamped there. Each backbone cache manifest, replay row, fit completion, evaluation completion and telemetry stream contains UTC timestamps. The verified phase order is cache → all VALIDATION replay → all TEST replay → all training → final TEST inference/metrics/statistics. TEST file-audit guards were active in training; all252 fits record zero TEST reads. The training worker pool exited before the final evaluation pool started. Selected rates were locked before TEST evaluation.','',
  'Three workers refill as jobs finish; every model worker uses CPU intra12/inter12, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE, MKL_DYNAMIC=FALSE, with actual counts/environment/load averages logged in telemetry/. caffeinate -i remains active through the queue. Resident applications were not closed. Per-fit wall times and start/end UTC are in all_training_runs.csv.','',
  'The unchanged formal main_v2 training-prefix AST is reused through restored-best-state assignment, with its TEST suffix omitted. Only cached loaders, the imported frozen model factory, learning rate and max50 epochs are rebound. Adam defaults, batch32, raw-target Huber delta1, gradient clip1, DataLoader generator42, patience10 and >1e-12 Huber improvement rule are unchanged. Formal source: [FORMAL_TRAINING_PREFIX.txt](FORMAL_TRAINING_PREFIX.txt).','',
  '## All tuning trials and selected rates','',
  'Seed0 alone selects the LR with lowest restored-best-Huber-checkpoint VALIDATION MSE; exact ties favor the lower rate. All three rates were fit from the same initialization. Seed0 is reused, then seeds1/21/42/3407 are trained at the locked rate. 108 tuning trials +144 remaining-seed fits =252. No rate was chosen by TEST.','',
  md(tune[['input_set','backbone','h','variant','lr','validation_mse','best_epoch','epochs','status']].sort_values(['input_set','backbone','h','variant','lr'])),
  'Selected rates:', '',md(pd.DataFrame(select['cells']).drop(columns=['seed0_job'])),
  'Complete trial records (including status/errors): [tuning_trials.csv](tuning_trials.csv) and [all_training_runs.csv](all_training_runs.csv). Every history and restored-best checkpoint is retained under fits/.','',
  '## TEST results — all cells','',md(display),
  'Five-seed arithmetic mean ± sample SD(ddof1), computed per seed before summarizing; these are not metrics of mean forecasts. DA excludes exactly zero realized returns (h5/10/20:46/27/20 excluded). No seed or cell was dropped; missing complete-five-seed metrics remain N/A. Full precision: [test_metrics_per_seed.csv](test_metrics_per_seed.csv), [five_seed_summary.csv](five_seed_summary.csv).','',
  '## Paired joint seed×date bootstrap','',
  'Adaptive−constant gate is primary; Static−Adaptive is the other prespecified contrast. MSE differences <0 favor the first arm; IC/DA differences >0 favor it. DA differences are fractions, not percentage points, in this table. B20,000,L=h, paired seeds[0,1,21,42,3407], common noncircular overlapping date blocks, SeedSequence([20261019,h,h,1]). All seed/date draws match Step16 exactly. Reused Step16 contiguous-matrix distributions, percentile95% intervals and centered two-sided +1 p; Holm across three horizons separately within contrast×metric×input×backbone (24 families,72 tests). Seed wins/losses/ties appear for every row. No tests against Step16 are performed.','',
  'DA resamples correct-count differences and valid-pair counts together to retain the pooled Step21 estimand; Rank IC uses the formal within-date cross-section and Step16 paired-valid date handling. Undefined tests are shown rather than silently excluded, with conservative1 only for their Holm input.','',md(J[columns]),
  '[joint_bootstrap_tests.csv](joint_bootstrap_tests.csv), [paired_seed_date_differences.csv](paired_seed_date_differences.csv), [seed_wins.csv](seed_wins.csv), [BOOTSTRAP_VERIFICATION.json](BOOTSTRAP_VERIFICATION.json).','',
  '## Router-weight movement vs Step16','',
  'Movement is mean absolute deviation from0.25. Within-model SD averages NumPy population SD over the four scale weights. Between-date share follows the stored Step19/Step9 definition: sum of sample variances of date-mean weights broadcast to stock–date rows divided by sum of sample variances of sample weights. For numerically constant routing (peak-to-peak<1e-7), report SD0 and between-date share N/A; raw float32 SD and variance residuals are retained so summation noise is visible. This describes routing, without an additional significance test.','',
  md(RS[RS.variant.ne('fixed')][groups+['branch','regime','mean_abs_deviation_mean','sample_sd_mean','between_date_share_mean','n_constant_seeds']]),
  '[router_diagnostics_per_seed.csv](router_diagnostics_per_seed.csv), [router_diagnostics_summary.csv](router_diagnostics_summary.csv).','',
  '## Learning curves and best epochs','',md(distribution),
  'All252 trial histories: [learning_curves_all_trials.csv](learning_curves_all_trials.csv). Final180 fitted cells: [learning_curves_selected.csv](learning_curves_selected.csv), [best_epochs_per_seed.csv](best_epochs_per_seed.csv), [best_epoch_distribution.csv](best_epoch_distribution.csv). Mean curves retain only seeds observed at each epoch; n_seeds is recorded in [learning_mean_by_epoch.csv](learning_mean_by_epoch.csv), with no extrapolation after stopping.','',
  '[h5 PNG](learning_curves_h5.png) / [PDF](learning_curves_h5.pdf); [h10 PNG](learning_curves_h10.png) / [PDF](learning_curves_h10.pdf); [h20 PNG](learning_curves_h20.png) / [PDF](learning_curves_h20.pdf). Solid=train Huber; dashed=validation Huber.','',
  '## Tuned vs Step16 — descriptive only','',md(side_display),
  'The Step16 MSE/Rank IC values are copied from its frozen-primary skill registry; zero-excluded DA is copied from Step21. No new hypothesis test compares tuned and old models. [tuned_vs_Step16.csv](tuned_vs_Step16.csv), [Step16_reference_per_seed.csv](Step16_reference_per_seed.csv).','',
  '## Saved forecasts, weights and verification','',
  'Every final cell has per-stock/per-date TEST predictions and all eight routing weights in evaluation/<backbone>_<variant>/predictions_and_weights.csv. Companion date_metrics.csv, routing_by_date.csv and routing_diagnostics.csv retain date/seed summaries. [prediction_files.csv](prediction_files.csv) lists every path/hash; [prediction_integrity.csv](prediction_integrity.csv) verifies rows/keys and the single final inference pass. Every training checkpoint is saved as fits/<job>/best_aggregator.pt, bound to its immutable backbone and protocol hash.','',
  '[VALIDATION_STEP22A.json](VALIDATION_STEP22A.json), [PROTECTED_INPUTS.json](PROTECTED_INPUTS.json), per-worker telemetry/ and immutable phase/status/events records document execution. All protected source hashes and the protocol hash are unchanged. All new outputs were exclusively created.','']
 text(OUT/'REPORT_STEP22A.md','\n'.join(report));js(OUT/'ANALYSIS_COMPLETE.json',dict(utc=utc(),report=artifact(OUT/'REPORT_STEP22A.md'),status=validation['status'],protocol_sha256=check_protocol()))
 print('STEP22A COMPLETE',validation['status'],'240 cells,72 joint comparisons; report',rel(OUT/'REPORT_STEP22A.md'),flush=True)
if __name__=='__main__':main()
