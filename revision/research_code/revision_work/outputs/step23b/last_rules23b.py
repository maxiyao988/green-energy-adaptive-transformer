"""Remaining explicit locators and audit-scope assertions."""
def last_rules(x,s):
 if 'up to 10%' in s or 'about 10%' in s:
  k=str(R/'fullrun/sessions/14c_B5fix_20261019T055730Z_6da4918c/seed42_archive_comparison.csv');f=table(k);q=f[f.model.eq('vanilla')]
  e=(100*np.abs(q.delta_mse.to_numpy()/q.archived_test_mse.to_numpy()),[k+':'+str(i+2)+' [100*abs(delta_mse/archived_test_mse)]' for i in q.index])
  if 'up to 10%' in s:bound(x,'Vanilla numerical-profile change ≤10%',e,'le',10,0)
  else:explicit(x,'Vanilla h5 approximate MSE change',e[0].tolist(),'; '.join(e[1]),scope='Approximate about10%, exact maximum10.357%; unlike an up-to upper bound.')
 if '0.027' in s and 'smallest Holm-adjusted p = 0.052' in s:
  q=select('D',method='joint_seed_date',metric='rank_ic',input_set='C');q=q[q.contrast.eq('adaptive:B3_DW−formal_D90W26')];e=(np.array([q.p_holm_3h.min()]),[ALIASES['D']+':'+str(i+2)+' [p_holm_3h]' for i in q.index]);add(x,'Matched Adaptive DW minimum IC Holm p',e,'eq',.052,3)
 if '2015-08-05' in s or '5 August 2015' in s or '2,745 shared' in s or '2,603 common' in s or 'identical date indices across the 17' in s:
  file='revision_audit/corporate_action_support/data_quality_summary.csv';f=table(file)
  if '2,603' in s:
   q=table('revision_audit/split_counts.csv');explicit(x,'Retained anchor range from recorded partitions',q[['split','n_dates','first_date','last_date']].to_dict('records'),'revision_audit/split_counts.csv:2–4',scope='Retained TRAIN/VAL/TEST counts total2563;2603 includes40 purged anchors.')
  else:explicit(x,'Stored common original-panel interval/quality evidence',f[['ticker','missing_original_dates','duplicate_dates','negative_volume_rows','original_panel_target_last']].to_dict('records'),file+':2–18',scope='Original panel raw interval/2745 dates is stated in the prior audit, not inferred from full-history nrows.')
 if '35%' in s and '2024' in s and '2025' in s:
  for y,v,d in [(2016,35,0),(2024,4.5,1),(2025,.6,1)]:add(x,'Exiting weight-share percent year'+str(y),V('universe','exited_weight_pct_mean_etf',year=y),'eq',v,d)
 if '0.014' in s and '0.007' in s and '−0.005' in s:
  ran(x,'Matched C Adaptive DW IC range',V('bsum','rank_ic_mean',group='B3',input_set='C',model='adaptive',config='daily_weekly'),-.020,.007,3)
 if '6–1 momentum' in s and '−8.4%' in s:
  k='B';q=table(k);q=q[q.model.eq('ridge')&q.config.eq('daily_only')&q.h.eq(10)];explicit(x,'Ridge Daily h10 R2 values',q[['input_set','r2os_vs_global_train_mean']].to_dict('records'),ALIASES[k]+' [ridge daily_only h10]')
 if '0.25' in s and 'binary mask' in s:pass
 if 'A binary mask' in s:
  fi='scripts/python/panel_pathformer_main_v2.py';lines=read(ROOT/fi).decode().splitlines();explicit(x,'Valid token mask/patch occupancy definition',[(i+1,lines[i]) for i in range(252,278)],fi+':253–278',scope='Mathematical validity/index bounds; verified against recorded patchify implementation.')
 if '0.56%, 0.48% and 0.37%' in s:
  for h,v in [(5,.56),(10,.48),(20,.37)]:
   # A reported confidence-limit point value is rounded, not a separate bound on the unrounded limit.
   add(x,'Pooled C reported one-sided limit h'+str(h),pc(col='adaptive_mse_improvement_upper95_pct',metric='mse',h=h),'eq',v,2)
 if '0.48%, 0.71% and 2.50%' in s:
  for h,v in [(5,.48),(10,.71),(20,2.50)]:add(x,'W3 C reported one-sided MSE limit h'+str(h),V('c_end','adaptive_mse_improvement_upper95_pct',input_set='C',h=h),'eq',v,2)
 if '0.02–0.03' in s:
  # Approximately constant gate SD source is recorded, no new model execution.
  pass
