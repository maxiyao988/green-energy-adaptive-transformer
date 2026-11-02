"""Explicit user decisions: n=1 provenance, Weekly reuse, ensemble skill, >=5-stock subgroup IC."""
from statistics16_v3 import s
from statistics16 import *
UP='clarified/'
def flags(d,family):
 out=[]
 for r in d.itertuples():
  for ref in ['global_train_mean','zero','ticker_train_mean','expanding_pooled_mean']:
   value=getattr(r,'r2os_vs_'+ref)
   if value>0:
    tested=ref=='global_train_mean';out.append(dict(id=r.id,family=family,input_set=r.input_set,model=r.model,config=r.config,h=r.h,seed=r.seed,metric='r2os_vs_'+ref,value=value,test_label='CW vs Train-Mean' if tested else 'descriptive, not tested',raw_p=r.cw_p_mbb_one_sided if tested else np.nan,p_holm_all_cells=r.cw_p_mbb_one_sided_holm_all_cells if tested else np.nan,raw_p_NW=r.cw_p_nw_one_sided if tested else np.nan,p_NW_holm_all_cells=r.cw_p_nw_one_sided_holm_all_cells if tested else np.nan))
  if np.isfinite(r.rank_ic_p_mbb) and r.rank_ic_p_mbb<.05:out.append(dict(id=r.id,family=family,input_set=r.input_set,model=r.model,config=r.config,h=r.h,seed=r.seed,metric='rank_ic',value=r.mean_rank_ic,test_label='Rank IC, two-sided centered MBB',raw_p=r.rank_ic_p_mbb,p_holm_all_cells=r.rank_ic_p_mbb_holm_all_cells,raw_p_NW=np.nan,p_NW_holm_all_cells=np.nan))
 return pd.DataFrame(out)
def adjust(d):
 for p in ['cw_p_nw_one_sided','cw_p_mbb_one_sided','rank_ic_p_mbb']:
  d[p+'_holm_all_cells']=np.nan
  for ix,v in holm({ix:float(v) for ix,v in d[p].items() if np.isfinite(v)}).items():d.loc[ix,p+'_holm_all_cells']=v
 return d

def run():
 if verified(UP+'COMPLETE.json'):return
 a=Analysis();old=pd.read_csv(OUT/'B_forecast_skill.csv',float_precision='round_trip');drop=[c for c in old if c.endswith('_holm_all_cells') or c in ['n_seeds','mse_mean','mse_sd','rank_ic_mean','rank_ic_sd','cw_seed_wins','rank_ic_seed_wins']];raw=old.drop(columns=drop);checks=[];replacement=[];files=[]
 for h in HS:
  orig=a.group('single_resolution','O','adaptive',h,config='weekly_only');b3=a.group('B3','O','adaptive',h,config='weekly_only');delta=float(np.max(np.abs(a.pred(orig[42])-a.pred(b3[42]))));exact=delta<1e-7
  checks.append(dict(input_set='O',model='adaptive',h=h,original_seed=42,B3_seed=42,max_abs_delta_pred=delta,classification='EXACT' if exact else 'DIFFERENT',reuse_five_seeds=exact,rule='ONLY O Weekly26; never C or Daily; original labels, keys and weekly windows unchanged'))
  if exact:
   target=raw[(raw.group=='single_resolution')&(raw.input_set=='O')&(raw.model=='adaptive')&(raw.config=='weekly_only')&(raw.h==h)];assert len(target)==1;raw=raw.drop(target.index)
   src=raw[(raw.group=='B3')&(raw.input_set=='O')&(raw.model=='adaptive')&(raw.config=='weekly_only')&(raw.h==h)].copy();assert len(src)==5
   for _,r in src.iterrows():
    z=r.to_dict();z['id']=f'single_resolution_O_adaptive_weekly_only_{h}_{int(r.seed)}_W3';z['alias_of']=r.id;z['group']='single_resolution';replacement.append(z)
 raw=pd.concat([raw,pd.DataFrame(replacement)],ignore_index=True);raw['alias_of']=raw.get('alias_of',pd.Series('',index=raw.index)).fillna('');raw=adjust(raw)
 keys=['group','input_set','model','config','h'];summary=raw.groupby(keys,as_index=False).agg(n_seeds=('seed','nunique'),mse_mean=('mse','mean'),mse_sd=('mse','std'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'),cw_seed_wins=('cw_mean',lambda x:int((x>0).sum())),rank_ic_seed_wins=('mean_rank_ic',lambda x:int((x>0).sum())))
 summary.loc[summary.n_seeds.eq(1),['mse_sd','rank_ic_sd','cw_seed_wins','rank_ic_seed_wins']]=np.nan
 raw=raw.merge(summary,on=keys,validate='many_to_one');raw['conditioning']='five training seeds';raw.loc[raw.n_seeds.eq(1),'conditioning']='conditional on seed 42; training randomness not reflected';raw.loc[raw.group.eq('ridge'),'conditioning']='deterministic Ridge protocol; seed SD and seed wins not applicable'
 raw['numerical_provenance']='12/12 formal/14a/B5FIX/Step15 verified source; see registry'
 raw.loc[raw.group.eq('single_resolution')&raw.input_set.eq('C'),'numerical_provenance']='C: 14c-day 12/12'
 raw.loc[raw.group.eq('single_resolution')&raw.input_set.eq('O')&raw.n_seeds.eq(1),'numerical_provenance']='O: archived; profile not verified (Step12A did not verify this single-resolution path)'
 raw.loc[raw.alias_of.ne(''),'numerical_provenance']='O Weekly26: B3 12/12 five seeds; seed42 EXACT vs formal Weekly-only in this Step16 check; not a Step12A single-resolution verification'
 ff=flags(raw,'single_run_all_cells');files += [csv(UP+'B_forecast_skill.csv',raw),csv(UP+'B_seed_summary.csv',summary),csv(UP+'B_flagged_cells.csv',ff),csv(UP+'B_weekly_reuse_check.csv',pd.DataFrame(checks)),csv(UP+'B_cell_provenance.csv',raw[['id','group','input_set','model','config','h','seed','n_seeds','conditioning','numerical_provenance','alias_of']])]
 # Thirty five-seed ensemble forecasts form their OWN all-cell Holm families.
 ens=[]
 for p in sorted((OUT/'F_ensembles').glob('*.parquet')):
  inp,model,hh,_=p.stem.split('_');h=int(hh[1:]);d=canon(pd.read_parquet(p));ref=a.test('W3',h);assert d[['ticker','anchor_date']].equals(ref[['ticker','anchor_date']]) and np.array_equal(d.y_true.to_numpy(np.float32),ref.y_true.to_numpy(np.float32));y=d.y_true.to_numpy(float);pr=d.y_pred.to_numpy(float);Y=y.reshape(391,17);refs,gtm=a.refs('W3',h);m,_,ic=cell_metrics(y,pr,refs,h,dates(h,391,2),Y,pr.reshape(Y.shape),gtm);m.pop('r2os_vs_expanding_ticker_mean',None);mean,se,t,n=nw_t(ic,h);bb=np.nanmean(ic[dates(h,391,2)],axis=1);m.update(rank_ic_nw_se=se,rank_ic_nw_t=t,rank_ic_nw_lag=h,rank_ic_n_dates=n,rank_ic_p_mbb=series_stats(bb,float(np.nanmean(ic)))['centered_p_two_sided'])
  ens.append(dict(id=f'ensemble_{inp}_{model}_h{h}',group='ensemble',input_set=inp,model=model,config='daily_weekly',h=h,seed='ensemble',n_member_seeds=5,n_forecast_cells=1,ensemble_mse_sd=np.nan,ensemble_seed_wins=np.nan,conditioning='equal-weight five-seed ensemble; B tests conditional on fixed member forecasts; seed-composition uncertainty in F',source=rel(p),**m))
 en=adjust(pd.DataFrame(ens));assert len(en)==30;ef=flags(en,'ensemble_all_cells_separate');files += [csv(UP+'B_ensemble_skill.csv',en),csv(UP+'B_ensemble_flags.csv',ef)]
 # Requested C subgroup updates: >=5 stocks/date for IC, counts and by-stock forecast bias.
 from engine import load_prices
 _,ret=load_prices();vol=ret.rolling(60,min_periods=60).std(ddof=1);cuts=pd.read_csv(OUT/'C_volatility_cutpoints.csv');out=[]
 for w in ['W1','W2','W3']:
  for h in HS:
   d=a.test(w,h);y=d.y_true.to_numpy(float);refs,_=a.refs(w,h);vv=np.array([vol.loc[r.anchor_date,r.ticker] for r in d.itertuples()]);reg=np.empty(len(d),dtype=object)
   for t in sorted(d.ticker.unique()):
    c=cuts[(cuts.window==w)&(cuts.h==h)&(cuts.ticker==t)].iloc[0];mask=d.ticker.eq(t).to_numpy();reg[mask]=np.where(vv[mask]<=c.q33,'low',np.where(vv[mask]<=c.q67,'mid','high'))
   ad=sorted(d.anchor_date.unique());half=np.where(d.anchor_date.isin(ad[:195]),'first_half','second_half')
   groups=[('stock',t,d.ticker.eq(t).to_numpy()) for t in sorted(d.ticker.unique())]+[('volatility',r,reg==r) for r in ['low','mid','high']]+[('subperiod',r,half==r) for r in ['first_half','second_half']]
   for inp in ['O','C']:
    for model in ['static','adaptive','ridge','train_mean','zero']:
     ids=a.group('rolling' if w!='W3' else ('core' if model in ['static','adaptive'] else 'ridge'),inp,model,h,window=w) if model not in ['train_mean','zero'] else {-1:None}
     for seed,id in ids.items():
      pr=a.pred(id) if id else refs['global_train_mean' if model=='train_mean' else 'zero']
      for typ,label,mask in groups:
       dd=d.loc[mask,['anchor_date']].copy();ct=dd.groupby('anchor_date').size();eligible=ct[ct>=5].index;ic=[]
       if typ!='stock':
        for date in eligible:
         mm=mask&d.anchor_date.eq(date).to_numpy();ic.append(rank_ic_rows(pr[mm][None,:],y[mm][None,:])[0])
       valid=int(np.isfinite(ic).sum());mse=float(np.mean((y[mask]-pr[mask])**2));base=float(np.mean((y[mask]-refs['global_train_mean'][mask])**2))
       out.append(dict(window=w,input_set=inp,model=model,h=h,seed=seed,subgroup_type=typ,subgroup=label,n_rows=int(mask.sum()),n_dates_with_group=int(len(ct)),n_dates_at_least_5_stocks=int(len(eligible)),n_dates_fewer_than_5_stocks=int((ct<5).sum()),n_dates_rank_ic_valid=valid,mse=mse,mse_relative_train_mean=mse/base,mean_rank_ic=float(np.nanmean(ic)) if valid else np.nan,mean_forecast_bias=float(pr[mask].mean()-y[mask].mean()) if typ=='stock' else np.nan,rank_ic_status='not applicable (Rank IC is defined cross-sectionally per date)' if typ=='stock' else ('N/A: no dates with at least5 stocks' if not len(eligible) else ('N/A: constant forecast/undefined ranks on eligible dates' if not valid else 'date-level cross-sectional IC on dates with at least5 stocks'))))
   print('CLARIFIED C',w,h,flush=True)
 S=pd.DataFrame(out);summ=S.groupby(['window','input_set','model','h','subgroup_type','subgroup'],as_index=False).agg(n_seeds=('seed','nunique'),n_rows=('n_rows','first'),n_dates_with_group=('n_dates_with_group','first'),n_dates_at_least_5_stocks=('n_dates_at_least_5_stocks','first'),n_dates_fewer_than_5_stocks=('n_dates_fewer_than_5_stocks','first'),n_dates_rank_ic_valid_min=('n_dates_rank_ic_valid','min'),mse_mean=('mse','mean'),mse_sd=('mse','std'),relative_mse_mean=('mse_relative_train_mean','mean'),relative_mse_sd=('mse_relative_train_mean','std'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'),mean_forecast_bias_pooled_seeds=('mean_forecast_bias','mean'),mean_forecast_bias_seed_sd=('mean_forecast_bias','std'))
 summ.loc[summ.n_seeds.eq(1),['mse_sd','relative_mse_sd','rank_ic_sd','mean_forecast_bias_seed_sd']]=np.nan
 files += [csv(UP+'C_subgroups_per_seed.csv',S),csv(UP+'C_subgroups_summary.csv',summ)]
 decisions=dict(utc=utc(),user_clarifications='Explicit reply received after provisional report; supersedes prior defaults',R2='Only positive R2 vs GTM attaches CW-vs-TrainMean p; all other positive-R2 flags descriptive not tested and p=NA',single_seed='All n=1 cells included in Holm family; no SD or seed wins; conditional on seed42; numerical provenance explicit',weekly='O Adaptive Weekly-only h5/10/20 all EXACT vsB3 seed42; all five B3 seeds reused. Never C/Daily.',ensembles='30 equal-weight forecasts; own all-ensemble Holm family separate from738 single-run experiment cells; no extra model-vs-model ensemble contrasts',subgroups='IC restricted to dates with>=5 stocks; dates count supplied; per-stock IC N/A, per-stock mean forecast bias byseed and pooledseeds, no tests',multiplicity='All738 displayed single-run experimental cells retained in Holm including15 formal-Weekly aliases; conservative repeated-control counting is explicit. All30 ensemble forecasts form a separate Holm family.',single_run_cells=len(raw),ensemble_cells=len(en))
 files.append(js(UP+'DECISIONS.json',decisions));complete(UP+'COMPLETE.json',files,dict(single_cells=len(raw),ensemble_cells=len(en),new_training_runs=0));print('CLARIFIED COMPLETE',len(raw),len(en),flush=True)
if __name__=='__main__':run()
