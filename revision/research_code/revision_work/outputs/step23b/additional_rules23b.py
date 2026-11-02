"""Additional source lookups; executed inside the strict-audit module's namespace."""
def additional(x,s):
 cid=x['candidate_id']
 if 'squared bias' in s and '0.02%' in s:
  k=str(R/'skill_analysis/a2_original/a2_skill_by_cell.csv');f=table(k);f=f[f.model.eq('global_train_mean')]
  bound(x,'Train-Mean squared bias share <.02%',(100*f.bias2.to_numpy()/f.mse.to_numpy(),[k+':'+str(i+2)+' [100*bias2/mse]' for i in f.index]),'lt',.02,2)
 if '+0.09% to +0.13%' in s:
  ran(x,'Calibrated one-month reversal R2 percent',V('a1','r2os_vs_global_train_mean',100,benchmark='ols_rev1m'),.09,.13,2)
 if 'SWiM-style Transformer’s 5-day MSE was below' in s:
  e=V('bsum','mse_mean',group='neural',input_set='C',model='swim',h=5);m=V('a1','mse',benchmark='global_train_mean',h=5)
  add(x,'Five-seed C SWiM h5 MSE excess %',(100*(e[0]/m[0]-1),e[1]+m[1]),'eq',.6,1)
  k=str(R/'analysis16/B_PRIMARY_forecast_skill.csv');f=table(k);q=f[(f.model=='swim')&(f.input_set=='C')&(f.h==5)&(f.seed==42)&(f.config=='daily_weekly')];bound(x,'C SWiM seed42 R2 above0',(q.r2os_vs_global_train_mean.to_numpy(),[k+':'+str(i+2) for i in q.index]),'gt',0,None)
 if 'epochs 12–13' in s:
  k=str(R/'manuscript19/learning_mean_by_epoch.csv');f=table(k);printcols=False
  q=f[f.input_set.eq('C')&f.mechanism.eq('static')&f.horizon.eq(20)&f.epoch.isin([12,13])]
  c=next(c for c in ['n_seeds','n_histories','n_runs'] if c in q);add(x,'C Static h20 epochs12/13 have one run',(q[c].to_numpy(),[k+':'+str(i+2)+' ['+c+']' for i in q.index]),'eq',1,0)
 if 'smallest p = 0.375' in s:
  e=join(*[V('a3seed',c,metric='loss') for c in ['paired_t_p_two_sided','sign_test_p_two_sided','wilcoxon_p_two_sided']]);add(x,'Minimum loss seed-level paired p',(np.array([e[0].min()]),e[1]),'eq',.375,3)
 if 'one of 96 cases' in s:
  f=select('H',method='joint_seed_date');q=f[f.contrast.str.startswith('I5_')];sig=q[q.p_holm_3h<.05];explicit(x,'I5 joint MSE/IC tests, exactly one adjusted p<.05',dict(n_tests=len(q),n_rejections=len(sig),stored_significant_rows=sig[['input_set','contrast','h','metric','theta_hat','p_holm_3h','seed_wins']].to_dict('records')),ALIASES['H']+' [joint I5 rows]')
 if '9 and 4 dates' in s:
  q=select('subgroup',window=['W1','W3'],input_set='C',model='static',h=5);q=q[q.subgroup.str.contains('low',case=False)];explicit(x,'Low-vol dates with ≥5stocks (cutoffs depend on h as stored)',q[['window','h','n_dates_at_least_5_stocks']].to_dict('records'),ALIASES['subgroup']+' [Static h5 low-vol]')
 if 'within 3×10' in s:pass
 if '1.001–1.027' in s:
  k=str(ROOT/'revision_audit/corporate_action_support/snapshot_ratio_summary.csv');f=table(k);pay=['AES','BEP','CWEN','HASI','NEE','ORA','FSLR']
  # Last FSLR may not be dividend payer; identify from stored cash-dividend audit counts.
  cash=table(str(R/'data_audit/A_cash_dividend_spotchecks.csv'));pay=cash[cash['count']>0].ticker.tolist();q=f[f.ticker.isin(pay)];ran(x,'Dividend-payer median original/new adjusted-price factor',(q.window_ratio_median.to_numpy(),[k+':'+str(i+2)+' [window_ratio_median]' for i in q.index]),1.001,1.027,3)
  q=f[~f.ticker.isin(pay)];add(x,'Non-payer price factor=1',(q.window_ratio_median.to_numpy(),[k+':'+str(i+2)+' [window_ratio_median]' for i in q.index]),'eq',1,None)
 if '266 cash-dividend' in s:
  k=str(R/'data_audit/A_cash_dividend_spotchecks.csv');f=table(k);q=f[f['count']>0];explicit(x,'Cash dividend count/amount rounding',[dict(ticker=r.ticker,count=int(r.count),largest_vendor_dividend=float(r.largest_vendor_dividend),issuer_amount=float(r.issuer_gross_amount),matches_round3=bool(r.matches_after_rounding_issuer_to_3_decimals)) for r in q.itertuples()],k+' [all cash-paying tickers]',scope='Cash amount spot-check only; issuer ex-dates are NOT FOUND for some rows and are not asserted verified.')
 if '30 to 360' in s or '30–360' in s:
  k=str(R/'causal_features/adjustment_factor_split_summary.csv');f=table(k);q=f[f.split.eq('train')&f.ticker.eq('FCEL')];a=1/q.max_c.to_numpy();b=1/q.min_c.to_numpy();ran(x,'TRAIN FCEL ex-post/as-of price factor',(np.r_[a,b],[k+':'+str(i+2)+' [1/max_c,1/min_c]' for i in q.index]),30,360,0)
  cash=table(str(R/'data_audit/A_cash_dividend_spotchecks.csv'));pay=cash[cash['count']>0].ticker.tolist();q=f[f.split.eq('train')&f.ticker.isin(pay)];ran(x,'TRAIN dividend-payer adjustment factors',(np.r_[q.min_c.to_numpy(),q.max_c.to_numpy()],[k+':'+str(i+2)+' [min_c,max_c]' for i in q.index]),.19,.99,2)
 if '168 fund-quarters' in s:
  k=str(R/'universe/coverage_by_quarter.csv');f=table(k);explicit(x,'Historical fund-quarter coverage',dict(n_rows=len(f),quarters=sorted(f.quarter.unique().tolist()),n_obtained=int(f.obtained.sum())),k+':2–'+str(len(f)+1))
 if 'at least one of these funds' in s:
  f=table('universe');names=set()
  for z in f.study_tickers_member.dropna():names.update(str(z).split(';'))
  # Source stores comma-separated tickers: retain exact full source fields, no heuristic count.
  explicit(x,'Universe membership recorded per year',f[['year','n_study_tickers_member','study_tickers_member','n_nonpanel_still_listed']].to_dict('records'),ALIASES['universe']+' [all years; membership source]')
 if '0.014–0.027' in s:ran(x,'Formal C S/A mean Rank IC',V('bsum','rank_ic_mean',group='core',input_set='C',model=SA),.014,.027,3)
 if 'at least 20 subsequent' in s or '90 trading-day' in s:
  f='scripts/python/panel_pathformer_main_v2.py';a=docs_evidence(f,r'DAILY.*90|WEEKLY.*26|HORIZONS|MAX_HORIZON|DAILY_LEN|WEEKLY_LEN|daily_len|weekly_len')
  explicit(x,'Formal window/label availability specification',a,'; '.join(f+':'+str(i) for i,t in a))
 if re.fullmatch(r'\d{4}-\d\d-\d\d to \d{4}-\d\d-\d\d',s):
  k=str(ROOT/'revision_audit/split_counts.csv');f=table(k);pairs=[str(r.first_date)+' to '+str(r.last_date) for r in f.itertuples()];pairs.append(str(f.first_date.min())+' to '+str(f.last_date.max()))
  # Earlier rolling windows have a separate design manifest.
  files=[('revision_work/outputs/fullrun/REPORT_STEP14C_B2_5SEED.md',re.escape(s[:10])+'|'+re.escape(s[-10:])),('revision_work/outputs/fullrun/REPORT_STEP14C_B2.md',re.escape(s[:10])+'|'+re.escape(s[-10:]))]
  hits=[]
  for fi,pat in files:
   if (ROOT/fi).exists():hits.extend([(fi,i,t) for i,t in docs_evidence(fi,pat)])
  if s in pairs:explicit(x,'Exact partition/combined date interval',pairs,k+':2–4')
  elif hits:explicit(x,'Recorded rolling design dates',hits,'; '.join(fi+':'+str(i) for fi,i,t in hits))
 if '2021–2026' in s or 'three test windows cover' in s:
  fi='revision_work/outputs/fullrun/REPORT_STEP14C_B2_5SEED.md';a=docs_evidence(fi,r'2021-08-06|2023-02-24|2023-03-27|2024-10-14|2024-11-12|2026-06-05');explicit(x,'Stored rolling TEST ranges',a,'; '.join(fi+':'+str(i) for i,t in a))
 if '30,401; Fixed 107,777' in s:
  k=str(ROOT/'revision_audit/parameter_counts.csv');f=table(k);f=f[f.input_config.eq('daily_weekly')];explicit(x,'Formal trainable parameter counts',f[['mechanism','n_trainable_parameters']].to_dict('records'),k+' [Daily+Weekly]')
 if '3×10−5' in s and ('Note: T3 uses' in s or 'δ = 0.098' in s):
  fi='revision_work/outputs/step22b2/PROTOCOL_STEP22B2.md';a=docs_evidence(fi,r'3e-5|0.098|0.138|0.195|T3');explicit(x,'Recorded T3 setting',a,'; '.join(fi+':'+str(i) for i,t in a))
 if '8.4%' in s and '−1.2%' in s:pass
 if 'at least 0.12' in s:pass
 if '0–20 bp' in s:ran(x,'Stored transaction cost grid',V('G','cost_bp'),0,20,0)
 if '0≤wi≤0.25' in s:
  fi='revision_work/scripts/portfolio/run_portfolio.py'
  # Find recorded optimizer cap in Step8 source metadata/report, no optimization run.
  fi='revision_work/outputs/portfolio/original/REPORT_STEP8.md';a=docs_evidence(fi,r'0.25|25%|cap|long.only');explicit(x,'Recorded portfolio constraints',a,'; '.join(fi+':'+str(i) for i,t in a))
 if 'Values below 1' in s:explicit(x,'MSE-ratio reference definition',{'ratio':'model MSE / same-window Train-Mean MSE','threshold':1},ALIASES['d_summary']+' [r2os_train_mean_mean; ratio=1-R2]',scope='Mathematical interpretation of the stored ratio, not a test.')
 if 'Sharpe ratio than 1/N at every horizon' in s or 'risk-parity benchmark' in s:
  k=str(R/'portfolio/original_rp/metrics_all.csv');f=table(k);q=f[f.split.eq('test')&f.cost_bp.eq(10)&f.strategy.isin(['EW','MINVAR','RISKPARITY'])];explicit(x,'Stored forecast-free TEST Sharpe comparisons',q[['h','strategy','sharpe']].to_dict('records'),k+' [test,10bp,allocation benchmarks]')
 if 'only 4 alphas' in s or '4 of 270' in s:
  k=str(R/'portfolio/original/metrics_all.csv');f=table(k);c='alpha_vs_ew_nw_t';q=f[f.split.eq('test')&f.cost_bp.eq(10)&f[c].notna()];explicit(x,'O strategy-layer-horizon alpha t threshold',dict(n_combinations=len(q),n_abs_t_gt_1_96=int((q[c].abs()>1.96).sum()),significant_rows=q[q[c].abs()>1.96][['h','strategy','layer',c]].to_dict('records')),k+' [test,10bp,non-null alpha_vs_ew_nw_t]')
 if 'effective scales' in s and ('3.8' in s or '3.80' in s) and '3.87' not in s:ran(x,'Approximate O effective-scale range at one decimal',ro('effective_scales_mean'),3.8,3.9,1)
 if any(p in s for p in ['0.02–0.03','39–45%','40–45%','about 2%']):
  k='Hrouter';f=table(k)
  # These are original S2 backbone routers, not newly trained frozen aggregators.
  q=f[f.id.str.startswith('S2_')].copy()
  grouped=q.groupby(['input_set','h','branch'],sort=True)
  means=grouped[['test_mean_sd_across_samples','share_variance_between_dates']].mean();refs=[ALIASES[k]+':'+str(i+2) for i in q.index]
  if '0.02–0.03' in s:ran(x,'End-to-end mean router sample SD',(means.test_mean_sd_across_samples.to_numpy(),refs),.02,.03,2)
  if '39–45%' in s or '40–45%' in s:ran(x,'C between-date variance share percent',(100*means.loc['C'].share_variance_between_dates.to_numpy(),refs),40 if '40–45%' in s else 39,45,0)
  if 'about 2%' in s:ran(x,'O between-date variance share rounded percent',(100*means.loc['O'].share_variance_between_dates.to_numpy(),refs),2,2,0)
 if 'within 3×10' in s:pass
 if 'up to 10%' in s or 'about 10%' in s:
  # Stored B5FIX comparison CSV; do not retrain.
  files=list((R/'fullrun').glob('**/*comparison*.csv'))
  # A separate manifest below records the exact selected source; no prediction files opened.
  pass
