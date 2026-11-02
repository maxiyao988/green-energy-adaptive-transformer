"""Independent Step17 verification; no new hypotheses or CW calculation."""
from common17 import *
from compute17_v2 import scores,read

def run():
 if verified('VERIFICATION_COMPLETE.json'):return
 assert verified('SELECTION_COMPLETE.json') and verified('COMPUTE_COMPLETE.json');T=read(OUT/'test_results.csv');C=read(OUT/'base_cells.csv').fillna('');S=read(OUT/'selected_lambdas.csv');V=read(OUT/'validation_curves.csv');CR=read(OUT/'CW_REFERENCE_REGISTRY.csv').set_index('cell_id');checks=[];files=[]
 for file in ['sources.csv','CW_reference_sources.csv']:
  for r in read(OUT/file).itertuples():assert sha(ROOT/r.path)==r.sha256,r.path
 gate=json.loads((OUT/'PRETEST_GATE_FINAL.json').read_text());start=json.loads((OUT/'TEST_COMPUTE_START.json').read_text());assert gate['no_STEP17_TEST_metric_computed_yet'];assert start['pretest_gate_sha256']==sha(OUT/'PRETEST_GATE_FINAL.json')
 chronology=[OUT/'PROTOCOL_STEP17.md',OUT/'validation_curves.csv',OUT/'PROTOCOL_STEP17_AMENDMENT_01.md',OUT/'PRETEST_GATE_FINAL.json',OUT/'TEST_COMPUTE_START.json'];assert all(a.stat().st_mtime_ns<=b.stat().st_mtime_ns for a,b in zip(chronology,chronology[1:])),chronology
 assert gate['selection_sha256']==sha(OUT/'selected_lambdas.csv') and gate['curves_sha256']==sha(OUT/'validation_curves.csv') and gate['amendment_sha256']==sha(OUT/'PROTOCOL_STEP17_AMENDMENT_01.md')
 checks.append(dict(check='preregistration_and_user_amendment_before_TEST_metrics',passed=True,n=1))
 for r in S.itertuples():
  g=V[(V.group_id==r.group_id)&(V.used_for_selection==True)].sort_values('lambda_value');assert len(g)==11
  if r.status=='ready':
   k=int(np.argmin(g.val_mse.to_numpy(float)));assert float(g.iloc[k].lambda_value)==r.lambda_selected
  else:assert g.val_mse.isna().all() and pd.isna(r.lambda_selected)
 checks.append(dict(check='all84_VAL_only_selections_and_smallest_exact_tie',passed=True,n=len(S)))
 for (regime,gid),g in T.groupby(['regime','group_id']):
  if regime=='PRIMARY':expected=S.set_index('group_id').loc[gid].lambda_selected;assert g.lambda_used.isna().all() if pd.isna(expected) else g.lambda_used.eq(expected).all()
  else:assert g.lambda_used.eq(.5).all()
 checks.append(dict(check='one_ensemble_selected_lambda_applied_to_all_members',passed=True,n=168))
 for regime,g in T.groupby('regime'):
  assert len(g)==414;want=holm(dict(zip(g.index,g.loss_p_for_Holm)))
  for ix,p in want.items():assert p==T.loc[ix,'loss_p_Holm'],(regime,ix,p,T.loc[ix,'loss_p_Holm'])
  z=g[g.lambda_used.eq(0)];assert z.loss_p_raw.isna().all() and z.loss_p_for_Holm.eq(1).all() and z.loss_p_Holm.eq(1).all() and z.loss_ci_lower_95.isna().all() and z.loss_ci_upper_95.isna().all() and z.mean_rank_ic.isna().all() and z.r2os_vs_global_train_mean.eq(0).all()
 checks.append(dict(check='exactly2_Holm_families_414_each_including_all_placeholders',passed=True,n=828))
 for r in T.itertuples():
  if r.cw_status=='unchanged by shrinkage; corrected in Step16':
   src=CR.loc[r.cell_id]
   for col in ['cw_p_nw_raw','cw_p_mbb_raw','cw_p_nw_holm_Step16','cw_p_mbb_holm_Step16']:assert getattr(r,col)==src[col],(r.cell_id,col)
  elif r.lambda_used==0:
   assert all(pd.isna(getattr(r,col)) for col in ['cw_p_nw_raw','cw_p_mbb_raw','cw_p_nw_holm_Step16','cw_p_mbb_holm_Step16'])
 checks.append(dict(check='CW_reference_copy_bitwise_numeric_identity_no_recorrection',passed=True,n=int(T.cw_status.eq('unchanged by shrinkage; corrected in Step16').sum())))
 # Validate every saved combined forecast against its exact Prompt16 base, without fitting any model.
 base={}
 for r in C[C.member!='ensemble'].itertuples():base[r.cell_id]=pd.read_parquet(ROOT/r.canonical).y_pred.to_numpy(float)
 for gid,g in C.groupby('group_id'):
  if 'ensemble' in set(g.member):
   row=g.iloc[0];y=split_labels(row.window,int(row.h),'test').y_true.to_numpy(float);base[gid+'_ensemble']=ensemble(np.stack([base[gid+'_'+str(s)] for s in SEEDS]),y)
 metriccheck=[]
 for r in T[T.status.eq('evaluated')].itertuples():
  p=OUT/f'predictions/{r.regime}_{r.cell_id}_test.parquet';d=canon(pd.read_parquet(p));ref=split_labels(r.window,int(r.h),'test');assert len(d)==6647 and d[['ticker','anchor_date']].equals(ref[['ticker','anchor_date']]);assert np.array_equal(d.y_true.to_numpy(np.float32),ref.y_true.to_numpy(np.float32));pr=d.y_pred.to_numpy(float);expected=combine(base[r.cell_id],r.training_mean,r.lambda_used);assert np.array_equal(pr,expected),r.cell_id
  assert p.stat().st_mtime_ns>= (OUT/'TEST_COMPUTE_START.json').stat().st_mtime_ns
  y=ref.y_true.to_numpy(float);m=scores(y,pr,r.training_mean);assert m['mse']==r.test_mse and m['r2os_vs_global_train_mean']==r.r2os_vs_global_train_mean and m['r2os_vs_zero']==r.r2os_vs_zero
  ic=rank_ic_rows(pr.reshape(391,17),y.reshape(391,17));mu=float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan;assert (pd.isna(mu) and pd.isna(r.mean_rank_ic)) or mu==r.mean_rank_ic
  metriccheck.append(dict(regime=r.regime,cell_id=r.cell_id,rows=6647,prediction_bitwise_match=True,scores_bitwise_match=True))
 checks.append(dict(check='all825_prediction_keys_labels_exact_combination_and_metrics',passed=True,n=len(metriccheck)));assert len(metriccheck)==825
 inv=read(OUT/'invariance_checks.csv');assert not inv.rank_ic_check.str.startswith('FAILED').any();assert inv.loss_mse_identity_abs_error.max()<1e-12;checks.append(dict(check='positive_lambda_RankIC_and_loss_identity',passed=True,n=len(inv),max_loss_mse_error=float(inv.loss_mse_identity_abs_error.max())))
 # Replay one pre-existing raw test per window/horizon/regime. This is verification, not a new hypothesis.
 D=read(OUT/'date_level_results.csv');rep=[]
 for (regime,w,h),g in T[T.test_performed==True].groupby(['regime','window','h']):
  r=g.iloc[0];dd=D[(D.regime==regime)&(D.cell_id==r.cell_id)].sort_values('anchor_date');x=dd.loss_difference.to_numpy(float);di=np.load(OUT/f'rng/{w}_h{h}.npz')['date_indices'];theta=float(x.mean());v=series_stats(x[di].mean(1),theta)
  assert v['centered_p_two_sided']==r.loss_p_raw and abs(v['ci_lower_95']-r.loss_ci_lower_95)<1e-15 and abs(v['ci_upper_95']-r.loss_ci_upper_95)<1e-15
  rep.append(dict(regime=regime,window=w,h=h,cell_id=r.cell_id,p_identical=True,CI_identical_within_1e15=True))
 checks.append(dict(check='18_window_horizon_regime_MBB_replays',passed=True,n=len(rep)));assert len(rep)==18
 assert 'torch' not in sys.modules
 files += [csv('verification_checks.csv',pd.DataFrame(checks)),csv('verification_prediction_cells.csv',pd.DataFrame(metriccheck)),csv('verification_bootstrap_replays.csv',pd.DataFrame(rep)),js('VERIFICATION.json',dict(utc=utc(),all_passed=True,new_training_runs=0,new_inference_runs=0,new_CW_tests=0,new_CW_corrections=0,torch_imported=False,protocol_and_amendment_before_TEST=True,positive_lambda_rank_IC_check=True,combined_prediction_files=825,blocked_PRIMARY_cells=3,lambda0_PRIMARY_cells=45))]
 complete('VERIFICATION_COMPLETE.json',files,dict(all_passed=True));print('STEP17 VERIFICATION COMPLETE',len(checks),flush=True)
if __name__=='__main__':run()
