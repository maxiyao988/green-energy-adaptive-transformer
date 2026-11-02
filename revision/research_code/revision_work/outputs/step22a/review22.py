"""Independent saved-artifact checks and a new, reviewed timing disclosure."""
from io22 import *
import re
sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd

def read(p):return pd.read_csv(p,float_precision='round_trip')
def md(d):
 def c(v):
  if pd.isna(v):return 'N/A'
  if isinstance(v,(float,np.floating)):return f'{v:.6g}'
  return str(v).replace('|',' / ')
 return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(c(v) for v in r)+' |' for r in d.itertuples(index=False,name=None)])+'\n'
def main():
 protocol=check_protocol();M=read(OUT/'test_metrics_per_seed.csv');F=read(OUT/'all_training_runs.csv');S=read(OUT/'five_seed_summary.csv');J=read(OUT/'joint_bootstrap_tests.csv');B=bindings()
 assert len(M)==240 and len(F)==252 and F.status.eq('COMPLETE').all() and M.status.eq('COMPLETE').all()
 keys=['input_set','backbone','h','seed','variant'];base_labels={};checks=[];max_mse=max_da=max_ic=0.
 for b in B:
  for variant in ['fixed']+VARIANTS:
   folder=OUT/'evaluation'/f"{b['id']}_{variant}";d=read(folder/'predictions_and_weights.csv').sort_values(['anchor_date','ticker']).reset_index(drop=True)
   row=M[(M.input_set==b['input_set'])&(M.backbone==b['backbone'])&(M.h==b['horizon'])&(M.seed==b['seed'])&(M.variant==variant)].iloc[0]
   assert len(d)==6647 and not d.duplicated(['ticker','anchor_date']).any()
   y=d.y_true.to_numpy(float);p=d.y_pred.to_numpy(float);v=y!=0
   if b['horizon'] not in base_labels:base_labels[b['horizon']]=d[['ticker','anchor_date','y_true']]
   assert d[['ticker','anchor_date','y_true']].equals(base_labels[b['horizon']])
   mse=float(np.mean((p-y)**2));da=float((((p>0)&(y>0))|((p<0)&(y<0)))[v].mean())
   dates=read(folder/'date_metrics.csv');ic=float(dates.rank_ic.mean())
   max_mse=max(max_mse,abs(mse-row.mse));max_da=max(max_da,abs(da-row.da));max_ic=max(max_ic,abs(ic-row.mean_rank_ic))
   assert abs(mse-row.mse)<2e-15 and abs(da-row.da)<2e-15 and abs(ic-row.mean_rank_ic)<2e-15
   w=d[WEIGHTS].to_numpy(float);assert np.isfinite(w).all() and w.min()>=0 and w.max()<=1
   sumerr=float(np.max(np.abs(w[:,:4].sum(1)-1)));sumerr2=float(np.max(np.abs(w[:,4:].sum(1)-1)));assert max(sumerr,sumerr2)<1e-6
   checks.append(dict(input_set=b['input_set'],backbone=b['backbone'],h=b['horizon'],seed=b['seed'],variant=variant,rows=len(d),identical_keys_and_labels=True,da_nonzero_pairs=int(v.sum()),da_equal=True,mse_equal=True,rank_ic_date_mean_equal=True,weight_sum_max_error=max(sumerr,sumerr2)))
 for row in S.itertuples():
  a=M[(M.input_set==row.input_set)&(M.backbone==row.backbone)&(M.h==row.h)&(M.variant==row.variant)]
  assert len(a)==5
  for col,mu,sd in [('mse','mse_mean','mse_sd'),('mean_rank_ic','rank_ic_mean','rank_ic_sd'),('da','da_mean','da_sd')]:
   assert abs(a[col].mean()-getattr(row,mu))<2e-15
   assert abs(a[col].std(ddof=1)-getattr(row,sd))<2e-15
 for key,g in J.groupby(['input_set','backbone','contrast','metric']):
  assert sorted(g.h.tolist())==HS
  order=np.argsort(g.holm_input_p.to_numpy());raw=g.holm_input_p.to_numpy()[order]
  adjusted=np.minimum(1,np.maximum.accumulate(raw*np.array([3,2,1])))
  assert np.max(np.abs(adjusted-g.p_holm_3h.to_numpy()[order]))<2e-15
 # Same-seed Adaptive/constant-gate initialization and unchanged initialization across LR trials.
 for bid,g in F.groupby('backbone_id'):
  for variant,t in g.groupby('variant'):assert t.initial_trainable_sha256.nunique()==1
  assert g[g.variant.isin(['adaptive','constant_gate'])].initial_trainable_sha256.nunique()==1
  for j in g.itertuples():
   complete=verify_job(OUT/'fits'/j.id/'COMPLETE.json');assert complete['frozen_unchanged']
   pr=complete['profile'];assert pr['torch_num_threads']==pr['torch_num_interop_threads']==12
   assert pr['environment']['OMP_NUM_THREADS'] is None
   assert pr['environment']['OMP_DYNAMIC']==pr['environment']['MKL_DYNAMIC']=='FALSE'
   h=read(OUT/'fits'/j.id/'history.csv');assert len(h)==j.epochs and len(h)<=50
   assert list(h.epoch)==list(range(1,len(h)+1));best=h[h.epoch.eq(j.best_epoch)].iloc[0]
   assert abs(best.val_loss-j.best_val_huber)<1e-14
 # Prove the number of active fit intervals and count actual model-worker PIDs.
 training_pids=set();starts=[];ends=[]
 for p in (OUT/'telemetry').glob('*.jsonl'):
  for line in p.read_text().splitlines():
   r=json.loads(line)
   if r['event']=='training_start':training_pids.add(r['pid']);starts.append(r)
   if r['event']=='training_end':ends.append(r)
 assert len(training_pids)==3 and len(starts)==len(ends)==252
 sweep=sorted([(r['utc'],1) for r in starts]+[(r['utc'],-1) for r in ends],key=lambda x:(x[0],x[1]))
 active=maximum=0
 for _,change in sweep:active+=change;maximum=max(maximum,active)
 assert active==0 and maximum==3
 csv(OUT/'FINAL_PREDICTION_CHECKS.csv',pd.DataFrame(checks))
 # Preserve both independent time measurements; never relabel perf_counter as UTC elapsed.
 phase=read(OUT/'phase_timestamps.csv').rename(columns={'wall_seconds':'monotonic_seconds'})
 phase['elapsed_utc_seconds']=(pd.to_datetime(phase.ended_utc,utc=True)-pd.to_datetime(phase.started_utc,utc=True)).dt.total_seconds()
 phase['utc_minus_monotonic_seconds']=phase.elapsed_utc_seconds-phase.monotonic_seconds
 phase=phase[['phase','started_utc','ended_utc','elapsed_utc_seconds','monotonic_seconds','utc_minus_monotonic_seconds','tasks']]
 csv(OUT/'phase_timestamps_reviewed.csv',phase)
 f=F.rename(columns={'wall_seconds':'monotonic_seconds'}).copy();f['elapsed_utc_seconds']=(pd.to_datetime(f.ended_utc,utc=True)-pd.to_datetime(f.started_utc,utc=True)).dt.total_seconds();csv(OUT/'all_training_runs_reviewed.csv',f)
 events=[json.loads(p.read_text()) for p in sorted((OUT/'events').glob('*.json'))]
 eventframe=pd.DataFrame([dict(event=r['event'],utc=r['utc']) for r in events]);csv(OUT/'event_timestamps.csv',eventframe)
 sleep=[]
 for line in (OUT/'SYSTEM_SLEEP_EVENTS.log').read_text().splitlines():
  m=re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \+0800 (Sleep|Wake|DarkWake) +\t(.*)',line)
  if m:sleep.append(dict(local_time_UTC_plus_08=m[1],event=m[2],description=m[3]))
 csv(OUT/'sleep_events.csv',pd.DataFrame(sleep));assert any(r['event']=='Sleep' for r in sleep)
 # All pre-existing protected content and every protocol byte remain unchanged.
 protected=json.loads((OUT/'PROTECTED_INPUTS.json').read_text())
 for path,digest in protected.items():assert sha(ROOT/path)==digest,path
 assert check_protocol()==protocol
 table=md(phase.drop(columns=['utc_minus_monotonic_seconds']))
 report=(OUT/'REPORT_STEP22A.md').read_text()
 pattern=r'\| phase \| started_utc \| ended_utc \| wall_seconds \| tasks \|.*?(?=\nCache binding/hash)'
 report,n=re.subn(pattern,table+'\n',report,flags=re.S);assert n==1
 disclosure=('**Timing / infrastructure disclosure:** macOS recorded Clamshell Sleep and Maintenance Sleep during the queue while the caffeinate assertion remained active. All 252 fits resumed/completed; no cell was rerun or dropped. The original files label `time.perf_counter()` durations as `wall_seconds`; Python reports this clock as `mach_absolute_time()`. This reviewed report preserves those numbers as `monotonic_seconds` and separately reports actual UTC elapsed seconds. No numerical cause is inferred from sleep; every recorded thread check stayed 12/12. There is no unresolved model/data/selection blocker. See [sleep_events.csv](sleep_events.csv), [SYSTEM_SLEEP_EVENTS.log](SYSTEM_SLEEP_EVENTS.log) and the reviewed timing files.\n\n')
 anchor='None. All prescribed cells, phase barriers, replay tolerances, thread settings and protected-source checks passed.\n\n'
 assert anchor in report;report=report.replace(anchor,anchor+disclosure,1)
 milestone=eventframe[eventframe.event.isin(['SUPERVISOR_START','CACHE_BINDING_START','CACHE_BINDING_END','ALL_TRAINING_START','ALL_TRAINING_END','ANALYSIS_START','ANALYSIS_END','SUPERVISOR_END'])]
 insertion=('Additional phase milestones (UTC):\n\n'+md(milestone)+'\n[phase_timestamps_reviewed.csv](phase_timestamps_reviewed.csv) contains UTC elapsed and raw monotonic durations; [all_training_runs_reviewed.csv](all_training_runs_reviewed.csv) does the same for every fit. [event_timestamps.csv](event_timestamps.csv) retains all phase events.\n\n')
 report=report.replace('## All tuning trials and selected rates',insertion+'## All tuning trials and selected rates',1)
 report=report.replace('caffeinate -i remains active through the queue','caffeinate -i was held active through the queue')
 report+='\n## Final independent review\n\nAll240 saved forecast cells,48 five-seed summaries,72 Holm values,252 histories/checkpoints and initialization pairing passed independent checks. The actual training pool used exactly3 model-worker PIDs, with maximum3 active fit intervals. No additional model inference, training, hypothesis test, seed selection or rate selection was performed in review. All three learning-curve figures were visually reviewed. The immutable initial report is retained; this copy adds the timing disclosure and audited UTC elapsed times without changing any statistical table. See [FINAL_REVIEW_STEP22A.json](FINAL_REVIEW_STEP22A.json).\n'
 text(OUT/'REPORT_STEP22A_FINAL.md',report)
 review=dict(status='PASS_WITH_TIMING_DISCLOSURE',protocol_sha256=protocol,prediction_cells_checked=240,five_seed_groups_checked=48,joint_tests_holm_checked=72,fit_histories_checkpoints_checked=252,adaptive_constant_initialization_pairing=True,exact_model_worker_pids=sorted(training_pids),maximum_parallel_fits=maximum,max_mse_check_error=max_mse,max_da_check_error=max_da,max_rank_ic_check_error=max_ic,protected_inputs_unchanged=len(protected),system_sleep_recorded=True,all_training_completed_without_rerun=True,statistical_tables_changed=False,new_model_inference_or_tests=False,final_report=rel(OUT/'REPORT_STEP22A_FINAL.md'),reviewed_utc=utc())
 js(OUT/'FINAL_REVIEW_STEP22A.json',review)
 js(OUT/'COMPLETE_STEP22A.json',dict(utc=utc(),protocol_sha256=protocol,final_report=artifact(OUT/'REPORT_STEP22A_FINAL.md'),tables=[artifact(OUT/n) for n in ['TABLE7_TUNED.csv','S6_STEP16_REFERENCE.csv','joint_bootstrap_tests.csv','tuning_trials.csv','prediction_files.csv']],review=artifact(OUT/'FINAL_REVIEW_STEP22A.json')))
 print(json.dumps(review,indent=2))
if __name__=='__main__':main()
