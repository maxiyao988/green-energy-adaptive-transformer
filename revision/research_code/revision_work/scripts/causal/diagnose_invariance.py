"""Locate strict cross-snapshot failures; do not alter feature definitions or saved tensors."""
from core import *
from build_and_audit import load_prices,adjusted,windows,transform

def main():
 ds=common.load_frozen_panel();old=load_causal();stats=common.fit_train_only_normalizer(old)
 deps=pd.read_csv(OUT/'window_dependencies.csv').set_index(['ticker','anchor_date','resolution']);rows=[];worst=[];catalog=[]
 for p in sorted((ROOT/'revision_audit/corporate_action_support').glob('*.csv')):
  d=pd.read_csv(p);catalog.append(dict(path=rel(p),n_rows=len(d),columns='|'.join(d.columns),sha256=sha(p)))
 csvout('prior_corporate_support_catalog.csv',catalog)
 for t in common.FROZEN_TICKERS:
  mask=ds.ticker==t;a=pd.to_datetime(ds.anchor_date[mask]).to_numpy(dtype='datetime64[ns]')
  o=load_prices(t);n=adjusted(load_prices(t,True));ow,starts,ends,ref,_=windows(o,a);nw,_,_,nref,_=windows(n,a)
  for r,x,nx in [('daily',ow[0],nw[0]),('weekly',ow[1],nw[1])]:
   of=transform(x,ref);nf=transform(nx,nref);error=np.abs(of-nf)
   excluded=np.array([bool(deps.loc[(t,str(pd.Timestamp(date).date()),r)].contains_ratio_step_in_dependency_span) for date in a]);mu,sd=stats[(t,r)]
   for c,ch in enumerate(CHANNELS):
    diff=error[:,:,c];outside=diff[~excluded];j=np.unravel_index(np.argmax(outside),outside.shape);sample=np.flatnonzero(~excluded)[j[0]];element=j[1]
    rows.append(dict(ticker=t,resolution=r,channel=ch,max_raw_difference_outside_steps=float(outside.max()),max_raw_difference_overall=float(diff.max()),approx_standardized_error_bound=float(outside.max()/sd[c]),n_windows_raw_diff_ge1e_6=int((outside.max(1)>=1e-6).sum()),n_windows_exact_nonzero=int((outside.max(1)>0).sum()),train_std=float(sd[c])))
    worst.append(dict(ticker=t,resolution=r,channel=ch,anchor_date=str(pd.Timestamp(a[sample]).date()),window_position=int(element),old_source_value=float(x[sample,element,c]),new_adjusted_source_value=float(nx[sample,element,c]),old_anchor_close=float(ref[sample]),new_anchor_close=float(nref[sample]),old_feature=float(of[sample,element,c]),new_feature=float(nf[sample,element,c]),absolute_difference=float(outside.max())))
 csvout('invariance_channel_diagnostics.csv',rows);csvout('invariance_worst_observations.csv',worst)
 print(pd.DataFrame(rows).sort_values('max_raw_difference_outside_steps',ascending=False).head(12).to_string(index=False))
 print('volume max outside',max(r['max_raw_difference_outside_steps'] for r in rows if r['channel']=='volume'))
if __name__=='__main__':main()
