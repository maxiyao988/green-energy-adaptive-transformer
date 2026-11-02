"""Precision and corporate-action localization diagnostics; creates only new artifacts."""
from core import *
from build_and_audit import load_prices,adjusted,windows,transform
import csv,decimal

def quant(v):
 v=np.asarray(v).ravel();return dict(max=float(v.max()),p99_9=float(np.quantile(v,.999)),p99=float(np.quantile(v,.99)),median=float(np.median(v)),n=int(len(v)))
def near(dates,events):
 return np.array([bool(np.any(np.abs(events-i)<=5)) for i in dates]) if len(events) else np.zeros(len(dates),bool)
def main():
 ds=common.load_frozen_panel();cds=load_causal();stats=common.fit_train_only_normalizer(cds)
 dep=pd.read_csv(OUT/'window_dependencies.csv');steps=pd.read_csv(OUT/'ratio_step_context.csv');summ=[];tops=[];associations=[];precision=[];rounding=[];globalraw=[];globalz=[]
 for t in common.FROZEN_TICKERS:
  m=ds.ticker==t;a=pd.to_datetime(ds.anchor_date[m]).to_numpy(dtype='datetime64[ns]');o=load_prices(t);n=load_prices(t,True);na=adjusted(n);ow,_,_,ref,di=windows(o,a);nw,_,_,nr,ni=windows(na,a)
  raw_source=np.abs(o.set_index('datetime').loc[n.datetime,'close'].to_numpy()-o.set_index('datetime').loc[n.datetime,'close'].to_numpy().astype(np.float32).astype(float))
  for name,path,cols in [('ORIGINAL',ROOT/f'dataset/finance/panel_raw/{t}_daily.csv',CHANNELS[:4]),('NEW',ROOT/f'data_raw_revision/prices/{t}.csv',['Open','High','Low','Close','Adj Close'])]:
   with path.open() as f:rows=list(csv.DictReader(f))
   for ch in cols:
    vals=[r[ch] for r in rows];dec=[max(0,-decimal.Decimal(v).as_tuple().exponent) for v in vals];v=np.array(vals,float);err=np.abs(v-v.astype(np.float32).astype(float))
    precision.append(dict(ticker=t,snapshot=name,channel=ch,n_rows=len(v),decimal_places_min=min(dec),decimal_places_median=float(np.median(dec)),decimal_places_max=max(dec),fraction_exact_float32=float((err==0).mean()),max_float32_rounding_residual=float(err.max()),storage='CSV decimal text parsed float64'))
  oldweekly=builder._reconstruct_weekly_from_daily(o);oldweekly_end=oldweekly.available_date.to_numpy()
  ev=np.flatnonzero((n.Dividends!=0)|(n['Stock Splits']!=0));st=pd.DatetimeIndex(n.datetime).get_indexer(pd.to_datetime(steps[steps.ticker==t].step_date));all_events=np.unique(np.r_[ev,st])
  for r,x,nx in [('daily',ow[0],nw[0]),('weekly',ow[1],nw[1])]:
   ex=dep[(dep.ticker==t)&(dep.resolution==r)].contains_ratio_step_in_dependency_span.to_numpy(bool);valid=np.flatnonzero(~ex);old=transform(x,ref);new=transform(nx,nr);err=np.abs(old-new);mu,sd=stats[(t,r)]
   # Match saved model-input dtype/normalization exactly.
   oz=((old.astype(np.float32)-mu)/sd).astype(np.float32);nz=((new.astype(np.float32)-mu)/sd).astype(np.float32);ze=np.abs(oz-nz)
   globalraw.append(err[~ex].reshape(-1));globalz.append(ze[~ex].reshape(-1))
   for c,ch in enumerate(CHANNELS):
    for scale,arr in [('raw',err),('standardized',ze)]:
     summ.append(dict(ticker=t,resolution=r,channel=ch,scale=scale,unit='all window elements',**quant(arr[~ex,:,c])))
     summ.append(dict(ticker=t,resolution=r,channel=ch,scale=scale,unit='maximum within each window',**quant(arr[~ex,:,c].max(1))))
   # Each window appears once, ranked by its worst standardized element.
   for j in valid:
    pos,c=np.unravel_index(np.argmax(ze[j]),ze[j].shape)
    if r=='daily':sourcepos=int(di[j]-89+pos);srcdate=o.datetime.iloc[sourcepos];nv=na.set_index('datetime').loc[srcdate];ov=o.iloc[sourcepos];source_start=srcdate;source_end=srcdate
    else:
     wi=np.searchsorted(oldweekly_end,a[j],side='right')-1;bar=oldweekly.iloc[wi-25+pos];srcdate=bar.available_date;source_start=bar.week_start;source_end=srcdate;ov=bar;nv=None
    startidx=int(np.searchsorted(n.datetime.to_numpy(),np.datetime64(source_start)));endidx=int(np.searchsorted(n.datetime.to_numpy(),np.datetime64(source_end)))
    nearsource=bool(any(np.any(np.abs(all_events-i)<=5) for i in range(startidx,endidx+1)));nearanchor=bool(near([ni[j]],all_events)[0]);maxraw=float(err[j].max());maxz=float(ze[j].max())
    tops.append(dict(ticker=t,resolution=r,anchor_date=str(pd.Timestamp(a[j]).date()),channel=CHANNELS[c],window_position=int(pos),source_start=str(pd.Timestamp(source_start).date()),source_end=str(pd.Timestamp(source_end).date()),raw_difference_at_worst_standardized_element=float(err[j,pos,c]),max_raw_window=maxraw,max_standardized_window=maxz,original_source_value=float(x[j,pos,c]),new_adjusted_source_value=float(nx[j,pos,c]),original_source_close=float(x[j,pos,3]),new_adjusted_source_close=float(nx[j,pos,3]),original_anchor_close=float(ref[j]),new_anchor_close=float(nr[j]),near_action_or_step_source_pm5=nearsource,near_action_or_step_anchor_pm5=nearanchor))
   print(t,r,flush=True)
  # Pure final float32 price quantization: two endpoints in each snapshot, conservatively 4 * eps.
  rounding.append(dict(ticker=t,max_original_close_distance_from_float32=float(raw_source.max()),float32_epsilon=float(np.finfo(np.float32).eps),conservative_four_endpoint_relative_log_error_bound=float(4*np.finfo(np.float32).eps),note='This bounds final float32 price serialization only, not unknown upstream adjustment arithmetic.'))
 d=csvout('precision_window_details.csv',tops);csvout('precision_top20_windows.csv',d.nlargest(20,'max_standardized_window'));csvout('precision_distributions.csv',summ);csvout('snapshot_csv_precision.csv',precision);csvout('final_float32_rounding_bounds.csv',rounding)
 for scope,frame in [('all',d)]+[(t,d[d.ticker==t]) for t in common.FROZEN_TICKERS]:
  high=frame.max_standardized_window>=1e-6
  for loc in ['source','anchor']:
   col=f'near_action_or_step_{loc}_pm5';base=float(frame[col].mean());rate=float(frame.loc[high,col].mean()) if high.any() else None
   associations.append(dict(ticker=scope,location=loc,n_windows=len(frame),n_large_difference_windows=int(high.sum()),all_windows_near_fraction=base,large_difference_windows_near_fraction=rate,concentration_ratio=(rate/base if base and rate is not None else None)))
 csvout('precision_action_association.csv',associations)
 gr=np.concatenate(globalraw);gz=np.concatenate(globalz)
 evidence=dict(completed_utc=utc(),original_threshold=1e-6,proposed_standardized_tolerance=1e-4,raw_distribution=quant(gr),standardized_distribution=quant(gz),below_revised_max=bool(gz.max()<1e-4),final_float32_rounding_bound=float(4*np.finfo(np.float32).eps),raw_exceeds_final_rounding_bound=bool(gr.max()>4*np.finfo(np.float32).eps),diagnostic_status='REQUIRES_EVIDENCE_REVIEW',note='Upstream vendor adjustment precision is undocumented; final CSV float32 serialization alone may not explain discrepancies.')
 jsonout('precision_diagnostic_evidence.json',evidence);print(json.dumps(evidence,indent=2));print(pd.DataFrame(associations).head(2).to_string(index=False));print(d.nlargest(20,'max_standardized_window').to_string(index=False))
if __name__=='__main__':main()
