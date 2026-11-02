"""Independent artifact, portfolio constraint, bootstrap replay and thread-sensitivity checks."""
from statistics16_v2 import s
from statistics16 import *
from threadpoolctl import threadpool_limits
import io

def run():
 if verified('verification.COMPLETE.json'):return
 files=[];checks=[];R=pd.read_csv(OUT/'registry.csv');sources=R[['source','sha256']].drop_duplicates();assert not sources.source.duplicated().any()
 for r in sources.itertuples():assert sha(ROOT/r.source)==r.sha256,r.source
 checks.append(dict(check='registered_prediction_source_hashes_unchanged',passed=True,n=len(sources)))
 markers=list(OUT.rglob('*.COMPLETE.json'))+list(OUT.glob('*/**/COMPLETE.json'));markers=sorted(set(markers))
 for p in markers:assert verified(str(p.relative_to(OUT))),p
 checks.append(dict(check='all_existing_completion_artifacts_hash_valid',passed=True,n=len(markers)))
 # Frozen Step15 primary joint distribution replay: same draws, source functions and paired date panels.
 D=pd.read_csv(FROZEN/'paired_seed_date_differences.csv',float_precision='round_trip');J=pd.read_csv(FROZEN/'paired_seed_date_bootstrap.csv',float_precision='round_trip');J=J[J.setting.eq('head_frozen')&J.metric.isin(['mse','rank_ic'])];replays=[];thread=[]
 with threadpool_limits(limits=1):
  for r in J.itertuples():
   g=D[(D.input_set==r.input_set)&(D.backbone==r.backbone)&(D.setting==r.setting)&(D.horizon==r.horizon)&(D.contrast==r.contrast)&(D.metric==r.metric)];X=g.pivot(index='seed',columns='anchor_date',values='paired_difference').sort_index().to_numpy(float);theta,bj,bd=distributions(X,r.horizon);met=series_stats(bj,theta)
   delta=max(abs(theta-r.theta),abs(met['ci_lower_95']-r.ci_lo),abs(met['ci_upper_95']-r.ci_hi));peq=met['centered_p_two_sided']==r.p_centered;assert delta<1e-12 and peq,(r,delta,met)
   replays.append(dict(input_set=r.input_set,backbone=r.backbone,h=r.horizon,contrast=r.contrast,metric=r.metric,max_stat_CI_delta=delta,p_identical=peq))
   if r.contrast=='adaptive_minus_static':
    with threadpool_limits(limits=12):
     theta2,b2,_=distributions(X,r.horizon);m2=series_stats(b2,theta2)
    dd=float(np.max(np.abs(bj-b2)));assert dd<1e-12 and m2['centered_p_two_sided']==met['centered_p_two_sided'];thread.append(dict(input_set=r.input_set,backbone=r.backbone,h=r.horizon,metric=r.metric,threads_compared='1 vs12',max_abs_bootstrap_draw_delta=dd,p_identical=True))
 checks.append(dict(check='Step15_primary_joint_bootstrap_replay',passed=True,n=len(replays)));checks.append(dict(check='bootstrap_thread_sensitivity',passed=True,n=len(thread)))
 weights=[]
 for p in OUT.glob('G/*/weights.csv'):
  W=pd.read_csv(p);cols=[c for c in W if c in __import__('engine').TICKERS];a=W[cols].to_numpy(float);err=float(np.max(np.abs(a.sum(1)-1)));assert err<1e-8 and a.min()>-1e-8
  mv=W.strategy.str.contains('L2_');maxw=float(W.loc[mv,cols].to_numpy().max());assert maxw<=.25+1e-8
  top=W.strategy.str.contains('L1_top5');ta=W.loc[top,cols].to_numpy();assert np.all(np.sum(ta>1e-10,axis=1)==5) and np.max(np.abs(ta.max(1)-.2))<1e-10
  weights.append(dict(path=rel(p),n_rebalances=len(W),max_sum_error=err,max_mv_weight=maxw,top5_exact=True))
 checks.append(dict(check='portfolio_weights_sum_cap_and_top5',passed=True,n=len(weights)))
 V=pd.read_csv(OUT/'H_I0_verification.csv');assert len(V)==30 and V.classification.eq('EXACT').all() and V.cache_i0_max_delta.lt(1e-7).all()
 P=pd.read_csv(OUT/'B_PRIMARY_forecast_skill.csv');H=pd.read_csv(OUT/'H_intervention_metrics.csv');joined=H[(H.split=='test')&(H.intervention=='I0_learned')].merge(P[P.group.eq('core')&P.model.eq('adaptive')],on=['input_set','h','seed'],suffixes=('_H','_B'));assert len(joined)==30;md=float(np.max(np.abs(joined.mse_H-joined.mse_B)));icd=float(np.max(np.abs(joined.mean_rank_ic_H-joined.mean_rank_ic_B)));assert md<1e-10 and icd<1e-10
 checks.append(dict(check='H_I0_metrics_vs_B_primary_Adaptive',passed=True,n=30,max_mse_delta=md,max_ic_delta=icd))
 # Save the actual common draws as well as entropy and draw-order metadata.
 for h in HS:
  W,di,si=draws(h);buf=io.BytesIO();np.savez_compressed(buf,seed_indices=si,joint_date_indices=di,date_only_indices=dates(h),skill_CW_IC_indices=dates(h,391,2));files.append(put(f'rng/draws_h{h}.npz',buf.getvalue()))
 files += [csv('verification_checks.csv',pd.DataFrame(checks)),csv('verification_frozen_bootstrap_replay.csv',pd.DataFrame(replays)),csv('verification_thread_sensitivity.csv',pd.DataFrame(thread)),csv('verification_portfolio_constraints.csv',pd.DataFrame(weights)),csv('protected_prediction_sources.csv',sources.rename(columns={'sha256':'sha256_before_and_after'})),js('RNG_MANIFEST.json',dict(B=B,model_seed_order=sorted(SEEDS),joint_seed_date=dict(entropy='[20261019,h,h,1]',draw_order='seed indices then noncircular overlapping date blocks',source='Step5 a3_hierarchical.py:116-122; Step15 missing-IC branch'),date_only=dict(entropy='[20261019,h,h,0]'),skill_CW_IC=dict(entropy='[20261019,h,h,2]'),ensemble=dict(entropy='[20261017,h,h]',draw_order='paired seed multisets then shared date blocks; imported Step7B source'),block='L=h; ceil(T/L) overlapping non-circular blocks, truncate T=391',B5fix_original_submission_reference_not_primary=True))]
 complete('verification.COMPLETE.json',files,dict(all_passed=True,new_training_runs=0,bootstrap_replay_cells=len(replays),thread_comparisons=len(thread)))
 print('VERIFICATION COMPLETE',len(checks),flush=True)
if __name__=='__main__':run()
