"""Parts 1/2: original-input diagnostics, snapshot consequences and causal feature tests."""
import os,sys
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1'
from core import *
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def load_prices(ticker,new=False):
 if not new:d=pd.read_csv(ROOT/f'dataset/finance/panel_raw/{ticker}_daily.csv')
 else:
  d=pd.read_csv(ROOT/f'data_raw_revision/prices/{ticker}.csv').rename(columns={'Date':'datetime','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
 d['datetime']=pd.to_datetime(d.datetime);return d[d.datetime<=CUTOFF].sort_values('datetime').reset_index(drop=True)

def adjusted(new):
 d=new.copy();factor=d.adj_close/d.close
 for c in ['open','high','low']:d[c]=d[c]*factor
 d['close']=d.adj_close
 return d

def windows(d,anchors):
 weekly=builder._reconstruct_weekly_from_daily(d)
 di=pd.DatetimeIndex(d.datetime).get_indexer(anchors)
 wi=np.searchsorted(weekly.available_date.to_numpy(),anchors,side='right')-1
 if (di<89).any() or (wi<25).any():raise ValueError('Window history NOT FOUND')
 xd=sliding_window_view(d[CHANNELS].to_numpy(),90,axis=0).transpose(0,2,1)[di-89]
 xw=sliding_window_view(weekly[CHANNELS].to_numpy(),26,axis=0).transpose(0,2,1)[wi-25]
 starts={'daily':d.datetime.to_numpy()[di-89],'weekly':weekly.week_start.to_numpy()[wi-25]}
 ends={'daily':d.datetime.to_numpy()[di],'weekly':weekly.available_date.to_numpy()[wi]}
 anchor_close=d.close.to_numpy()[di]
 return (xd,xw),starts,ends,anchor_close,di

def transform(x,anchor):
 if (x[:,:,:4]<=0).any() or (anchor<=0).any() or (x[:,:,4]<0).any():raise ValueError('Invalid feature source values')
 out=np.empty_like(x,dtype=np.float64)
 out[:,:,:4]=np.log(x[:,:,:4]/anchor[:,None,None])
 v=np.log1p(x[:,:,4]);out[:,:,4]=v-v.mean(axis=1,keepdims=True)
 return out

def main():
 threads(1);exp3._verify_frozen_dependencies()
 ds=common.load_frozen_panel();formal._formal_check_dataset_contract(ds)
 oldstats=common.fit_train_only_normalizer(ds)
 normalized=common.apply_normalizer(ds,oldstats,'daily_weekly')
 rawd=np.empty(ds.X_daily.shape,np.float64);raww=np.empty(ds.X_weekly.shape,np.float64)
 newd=np.empty_like(rawd);neww=np.empty_like(raww)
 steps=pd.read_csv(ROOT/'revision_audit/corporate_action_support/snapshot_ratio_significant_changes.csv')
 labelrows=[];labelsummary=[];steprows=[];actionrows=[];spikes=[];factors=[];factorsummary=[];splitfactors=[];distributions=[];range_summary=[];dependency=[];roundtrip=[];controlled=[]
 figfac,axfac=plt.subplots(5,4,figsize=(17,14),layout='constrained');fighist,axhist=plt.subplots(5,4,figsize=(17,14),layout='constrained')
 sourcejumps=pd.read_csv(ROOT/'revision_audit/corporate_action_support/price_discontinuities.csv')
 for k,t in enumerate(common.FROZEN_TICKERS):
  mask=ds.ticker==t;idx=np.flatnonzero(mask);a=pd.to_datetime(ds.anchor_date[mask]).to_numpy(dtype='datetime64[ns]');splits=ds.split[mask]
  old=load_prices(t);new=load_prices(t,True);newadj=adjusted(new)
  ow,starts,ends,ref,di=windows(old,a);nw,nstarts,nends,nref,ni=windows(newadj,a)
  for resolution,x,newx,originaltensor,dest,newdest in [('daily',ow[0],nw[0],ds.X_daily[mask],rawd,newd),('weekly',ow[1],nw[1],ds.X_weekly[mask],raww,neww)]:
   if not np.array_equal(x.astype(np.float32),originaltensor):raise ValueError(f'Original window reconstruction differs: {t}/{resolution}')
   roundtrip.append(dict(ticker=t,resolution=resolution,original_window_float32_exact=True))
   z=transform(x,ref);zn=transform(newx,nref)
   dest[idx]=z;newdest[idx]=zn
   # Pure arithmetic invariance control isolates the builder from vendor revisions/noise.
   multiplied=x.copy();multiplied[:,:,:4]*=7.125
   synthetic=transform(multiplied,ref*7.125)
   controlled.append(dict(ticker=t,resolution=resolution,max_abs_difference=float(np.max(np.abs(z-synthetic))),passed=bool(np.max(np.abs(z-synthetic))<1e-6)))
   relevant=steps[steps.ticker==t].Date.tolist()
   for j,gidx in enumerate(idx):
    start=starts[resolution][j];end=ends[resolution][j]
    if end>a[j] or start>end:raise ValueError('LOOKAHEAD in source window')
    includes=any(start<=np.datetime64(s)<=a[j] for s in relevant)
    dependency.append(dict(row_index=int(gidx),ticker=t,anchor_date=pd.Timestamp(a[j]).strftime('%Y-%m-%d'),split=splits[j],resolution=resolution,source_start=str(pd.Timestamp(start).date()),source_end=str(pd.Timestamp(end).date()),reference_close_date=str(pd.Timestamp(a[j]).date()),contains_ratio_step_in_dependency_span=includes,no_lookahead=True))
  for h in [5,10,20]:
   if (ni+h>=len(new)).any():raise ValueError('New label target date NOT FOUND')
   ny=np.log(new.adj_close.to_numpy()[ni+h]/new.adj_close.to_numpy()[ni]);fy=ds.y[h][mask].astype(np.float64);delta=ny-fy
   labelsummary.append(dict(ticker=t,horizon=h,n_anchors=len(a),max_abs_delta=float(np.abs(delta).max()),n_abs_delta_gt_1e_6=int((np.abs(delta)>1e-6).sum())))
   for j in range(len(a)):labelrows.append(dict(ticker=t,horizon=h,anchor_date=str(pd.Timestamp(a[j]).date()),split=splits[j],frozen_label=float(fy[j]),new_adj_close_label=float(ny[j]),delta=float(delta[j]),affected=bool(abs(delta[j])>1e-6)))
  for _,s in steps[steps.ticker==t].iterrows():
   date=pd.Timestamp(s.Date);i=new.index[new.datetime==date][0];subset=new.iloc[max(0,i-5):i+6]
   split=next(iter(ds.split[(ds.ticker==t)&(ds.anchor_date==np.datetime64(date))]),'NOT AN ANCHOR')
   action_subset=subset[(subset.Dividends!=0)|(subset['Stock Splits']!=0)]
   for _,event in action_subset.iterrows():actionrows.append(dict(ticker=t,step_date=s.Date,event_date=str(event.datetime.date()),offset_sessions=int(new.index[new.datetime==event.datetime][0]-i),new_dividend=float(event.Dividends),new_split=float(event['Stock Splits']),original_actions='NOT FOUND: original OHLCV snapshot has no action columns'))
   dividend=float(new.loc[i,'Dividends']);prior=float(new.loc[i-1,'close']);relative=float(s.ratio/s.previous_ratio-1)
   steprows.append(dict(ticker=t,step_date=s.Date,split=split,ratio_before=float(s.previous_ratio),ratio_after=float(s.ratio),relative_ratio_step=relative,new_same_date_dividend=dividend,new_same_date_split=float(new.loc[i,'Stock Splits']),prior_vendor_close=prior,dividend_predicted_ratio_step=-dividend/prior,step_minus_dividend_prediction=relative+dividend/prior,likely_cause='dividend adjustment differs between snapshots (inference; archived old actions NOT FOUND)'))
  # Price spike/reversal check uses original numeric values and also compares returns.
  close=new.close.to_numpy();returns=close[1:]/close[:-1]-1;old_index=old.set_index('datetime')
  for _,event in sourcejumps[sourcejumps.ticker==t].iterrows():
   ii=new.index[new.datetime==pd.Timestamp(event.Date)]
   if len(ii)!=1:continue
   i=int(ii[0])
   if i<1 or i+1>=len(new):continue
   r0,r1=float(returns[i-1]),float(returns[i])
   if abs(r0)>.25 and abs(r1)>.20 and r0*r1<0:
    dates=new.datetime.iloc[i-1:i+2];ov=old_index.reindex(dates).close.to_numpy();nv=close[i-1:i+2]
    oldret=ov[1:]/ov[:-1]-1
    spikes.append(dict(ticker=t,previous_date=str(dates.iloc[0].date()),spike_date=str(dates.iloc[1].date()),reversal_date=str(dates.iloc[2].date()),return_spike=r0,return_reversal=r1,in_requested_range=START<=event.Date<=CUTOFF,old_values_identical=bool(np.array_equal(ov,nv)),old_returns_match_1e_6=bool(np.allclose(oldret,[r0,r1],rtol=0,atol=1e-6)),original_previous_close=float(ov[0]),original_spike_close=float(ov[1]),original_reversal_close=float(ov[2]),new_previous_close=float(nv[0]),new_spike_close=float(nv[1]),new_reversal_close=float(nv[2])))
  # Undo all later vendor-coded share-ratio events, strictly excluding a split on date t.
  full=pd.read_csv(ROOT/f'data_raw_revision/prices/{t}.csv');sv=full['Stock Splits'].where(full['Stock Splits']!=0,1).to_numpy(np.float64)
  future=np.cumprod(sv[::-1])[::-1]/sv;future=future[:len(new)]
  astraded=new.close.to_numpy()*future;c=new.adj_close.to_numpy()/astraded
  f=pd.DataFrame(dict(ticker=t,Date=new.datetime.dt.strftime('%Y-%m-%d'),vendor_Close=new.close,Adj_Close=new.adj_close,future_split_product=future,AsTraded_formula=astraded,c=c))
  f=f[f.Date.between(START,CUTOFF)];factors.extend(f.to_dict('records'))
  factorsummary.append(dict(ticker=t,min_c=float(f.c.min()),max_c=float(f.c.max()),formula='AdjClose/(Close*product strictly later vendor split ratios)',special_event_caveat='BEP 2020-07-30 factor 1.251 is a BEPC distribution, not a literal split' if t=='BEP' else 'volume/security identity not independently verified'))
  ca=pd.Series(c,index=new.datetime).reindex(pd.to_datetime(a)).to_numpy()
  for resolution in ['daily','weekly']:
   mu,sd=oldstats[(t,resolution)];scale=float(mu[3]/sd[3])
   for split in ['train','val','test']:
    v=ca[splits==split]
    splitfactors.append(dict(ticker=t,resolution=resolution,split=split,n_anchors=len(v),min_c=float(v.min()),max_c=float(v.max()),v2_train_mean_close=float(mu[3]),v2_train_std_close=float(sd[3]),implied_shift_min=float(((v-1)*scale).min()),implied_shift_max=float(((v-1)*scale).max())))
  for resolution,Z,X in [('daily',normalized[0][mask],ds.X_daily[mask]),('weekly',normalized[1][mask],ds.X_weekly[mask])]:
   last=Z[:,-1,:];train=last[splits=='train'];lo=train.min(0);hi=train.max(0)
   all_train=Z[splits=='train'].reshape(-1,5);alo=all_train.min(0);ahi=all_train.max(0)
   for split in ['train','val','test']:
    vals=last[splits==split]
    for ch,name in enumerate(CHANNELS):distributions.append(dict(ticker=t,resolution=resolution,channel=name,split=split,n_anchors=len(vals),mean=float(vals[:,ch].mean()),std_ddof0=float(vals[:,ch].std()),fraction_abs_z_gt3=float((np.abs(vals[:,ch])>3).mean()),fraction_outside_train_last_range=float(((vals[:,ch]<lo[ch])|(vals[:,ch]>hi[ch])).mean()),fraction_outside_train_all_entry_range=float(((vals[:,ch]<alo[ch])|(vals[:,ch]>ahi[ch])).mean()),train_last_min=float(lo[ch]),train_last_max=float(hi[ch])))
   test=last[splits=='test',3]
   range_summary.append(dict(ticker=t,resolution=resolution,test_mean_z=float(test.mean()),test_fraction_abs_z_gt3=float((np.abs(test)>3).mean()),test_fraction_outside_training_last_range=float(((test<lo[3])|(test>hi[3])).mean()),test_fraction_outside_training_all_entry_range=float(((test<alo[3])|(test>ahi[3])).mean()),mostly_outside_last_range=bool((((test<lo[3])|(test>hi[3])).mean())>.5)))
   bins=np.linspace(min(last[splits=='train',3].min(),test.min()),max(last[splits=='train',3].max(),test.max()),45)
   for split,color in [('train','#2864ad'),('test','#d75b25')]:axhist.flat[k].hist(last[splits==split,3],bins=bins,density=True,histtype='step',color=color,linestyle='-' if resolution=='daily' else '--',linewidth=1,label=f'{resolution} {split}')
  axhist.flat[k].set_title(t);axhist.flat[k].set_xlabel('V2 standardized close, last observation');axhist.flat[k].set_ylabel('Density')
  axfac.flat[k].plot(pd.to_datetime(f.Date),f.c,linewidth=.9);axfac.flat[k].set_yscale('log');axfac.flat[k].axhline(1,color='gray',linestyle='--',linewidth=.6);axfac.flat[k].set_title(f'{t}: c={f.c.min():.3g}..{f.c.max():.3g}');axfac.flat[k].tick_params(axis='x',labelrotation=25)
  print('BUILT',utc(),t,flush=True)
 for ax in list(axfac.flat)[17:]+list(axhist.flat)[17:]:ax.set_visible(False)
 handles,labels=axhist.flat[0].get_legend_handles_labels();fighist.legend(handles,labels,loc='outside lower center',ncol=4)
 figfac.suptitle('Ex-post adjustment factor c(t), as defined by requested vendor-split formula',fontsize=15);fighist.suptitle('Original V2 close-level inputs: train versus test',fontsize=15)
 figfac.savefig(target(OUT/'fig_adjustment_factor.png'),dpi=160);fighist.savefig(target(OUT/'fig_level_shift.png'),dpi=160);plt.close('all')
 labeldf=csvout('label_comparison_all_anchors.csv',labelrows);csvout('label_impact_summary.csv',labelsummary);csvout('label_affected_anchors.csv',labeldf[labeldf.affected]);csvout('ratio_step_context.csv',steprows);csvout('ratio_step_nearby_actions.csv',actionrows);csvout('spike_reversals.csv',spikes);csvout('adjustment_factors_by_date.csv',factors);csvout('adjustment_factor_summary.csv',factorsummary);csvout('adjustment_factor_split_summary.csv',splitfactors);csvout('original_last_observation_distribution.csv',distributions);csvout('close_level_shift_summary.csv',range_summary);csvout('window_dependencies.csv',dependency);csvout('original_window_reconstruction_checks.csv',roundtrip);csvout('controlled_scale_invariance.csv',controlled)
 impact=[]
 for mech,path in [('static',ROOT/'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv'),('adaptive',ROOT/'dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv')]:
  pred=pd.read_csv(path)
  if 'mechanism' in pred:pred=pred[pred.mechanism==mech]
  if 'frequency_setting' in pred:pred=pred[pred.frequency_setting=='daily_weekly']
  for h in [5,10,20]:
   p=pred[(pred.horizon.astype(str).str.removesuffix('d').astype(int)==h)&(pred.seed==42)]
   m=p.merge(labeldf[(labeldf.horizon==h)&(labeldf.split=='test')],on=['ticker','anchor_date'],validate='one_to_one')
   if len(m)!=6647:raise ValueError('Saved forecast/label alignment mismatch')
   p32=m.y_pred.to_numpy(np.float32);oldy=m.frozen_label.to_numpy(np.float32);newy=m.new_adj_close_label.to_numpy(np.float32)
   before=float(np.mean((p32-oldy)**2));after=float(np.mean((p32-newy)**2))
   impact.append(dict(mechanism=mech,horizon=h,seed=42,n_test=len(m),old_test_mse=before,new_label_test_mse=after,mse_change=after-before,relative_mse_change=after/before-1))
 csvout('saved_predictions_new_label_mse.csv',impact)
 # Persist only original-snapshot causal inputs, frozen labels/splits/anchors.
 cds=common.PanelDataset(ds.ticker,ds.anchor_date,ds.split,rawd.astype(np.float32),raww.astype(np.float32),ds.y)
 ns=common.PanelDataset(ds.ticker,ds.anchor_date,ds.split,newd.astype(np.float32),neww.astype(np.float32),ds.y)
 stats=common.fit_train_only_normalizer(cds);nstats=common.fit_train_only_normalizer(ns)
 cz=common.apply_normalizer(cds,stats,'daily_weekly');nz_same=common.apply_normalizer(ns,stats,'daily_weekly');nz_own=common.apply_normalizer(ns,nstats,'daily_weekly')
 formal._formal_check_dataset_contract(cds)
 npout('tensors/X_daily_causal_raw.npy',cds.X_daily);npout('tensors/X_weekly_causal_raw.npy',cds.X_weekly);npout('tensors/X_daily_causal_standardized.npy',cz[0]);npout('tensors/X_weekly_causal_standardized.npy',cz[1])
 for h in [5,10,20]:npout(f'tensors/y_h{h}_frozen.npy',ds.y[h])
 csvout('tensors/sample_index.csv',pd.DataFrame(dict(ticker=ds.ticker,anchor_date=pd.to_datetime(ds.anchor_date).strftime('%Y-%m-%d'),split=ds.split)))
 statrows=[]
 for (t,r),(mu,sd) in stats.items():
  for i,ch in enumerate(CHANNELS):statrows.append(dict(ticker=t,resolution=r,channel=ch,mean=float(mu[i]),std=float(sd[i])))
 csvout('causal_normalization_stats.csv',statrows)
 inv=[];dep=pd.DataFrame(dependency).set_index(['row_index','resolution'])
 for r,oldraw,newraw,origz,newz,newzo in [('daily',rawd,newd,cz[0],nz_same[0],nz_own[0]),('weekly',raww,neww,cz[1],nz_same[1],nz_own[1])]:
  mdiff=np.max(np.abs(oldraw-newraw),axis=(1,2));zdiff=np.max(np.abs(origz-newz),axis=(1,2));odiff=np.max(np.abs(origz-newzo),axis=(1,2))
  for i in range(len(ds.ticker)):
   d=dep.loc[(i,r)]
   inv.append(dict(ticker=ds.ticker[i],anchor_date=str(pd.Timestamp(ds.anchor_date[i]).date()),split=ds.split[i],resolution=r,contains_ratio_step=bool(d.contains_ratio_step_in_dependency_span),max_raw_difference=float(mdiff[i]),max_standardized_difference_same_train_stats=float(zdiff[i]),max_standardized_difference_own_train_stats=float(odiff[i]),raw_pass_lt1e_6=bool(mdiff[i]<1e-6),standardized_pass_lt1e_6=bool(zdiff[i]<1e-6)))
 inv=csvout('invariance_all_windows.csv',inv);csvout('invariance_differing_windows.csv',inv[inv.max_raw_difference>0]);csvout('invariance_violations_outside_steps.csv',inv[(~inv.contains_ratio_step)&(~inv.raw_pass_lt1e_6|~inv.standardized_pass_lt1e_6)])
 excluded=inv.contains_ratio_step;outside=inv[~excluded]
 tests=dict(completed_utc=utc(),original_windows_match_v2_exactly=all(x['original_window_float32_exact'] for x in roundtrip),controlled_price_rescaling_invariance_pass=all(x['passed'] for x in controlled),no_lookahead=all(x['no_lookahead'] for x in dependency),shape_contract_pass=True,split_counts={s:int((ds.split==s).sum()) for s in COUNTS},finite=bool(np.isfinite(cz[0]).all() and np.isfinite(cz[1]).all()),labels_frozen=True,cross_snapshot_raw_max_overall=float(inv.max_raw_difference.max()),cross_snapshot_raw_max_outside_steps=float(outside.max_raw_difference.max()),cross_snapshot_standardized_max_overall=float(inv.max_standardized_difference_same_train_stats.max()),cross_snapshot_standardized_max_outside_steps=float(outside.max_standardized_difference_same_train_stats.max()),n_raw_violations_outside_steps=int((~outside.raw_pass_lt1e_6).sum()),n_standardized_violations_outside_steps=int((~outside.standardized_pass_lt1e_6).sum()),n_excluded_step_windows=int(excluded.sum()),invariance_exclusion='Source dependency span from earliest underlying daily/weekly observation through the daily anchor Close reference, inclusive; covers partial-current-week reference gaps.')
 tests['strict_all_required_tests_pass']=bool(tests['original_windows_match_v2_exactly'] and tests['no_lookahead'] and tests['finite'] and not tests['n_raw_violations_outside_steps'] and not tests['n_standardized_violations_outside_steps'])
 jsonout('feature_tests.json',tests);print(json.dumps(tests,indent=2),flush=True)
if __name__=='__main__':main()
