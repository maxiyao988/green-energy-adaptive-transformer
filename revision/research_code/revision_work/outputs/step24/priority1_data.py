"""Raw-price/sample reconstruction and future perturbation, no project imports or training."""
from review24_common import *
CHANNELS=['open','high','low','close','volume'];TICKERS=['AES','BEP','BLDP','BLNK','CSIQ','CWEN','DQ','ENPH','FCEL','FSLR','HASI','JKS','NEE','ORA','PLUG','RUN','SEDG'];B2=ROOT/'revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041';B3=ROOT/'revision_work/outputs/fullrun/sessions/14c_B3_20261017T201559Z_f47c5c3c';checks=[];examples=[];ranges=[]
def check(name,yes,delta=0.,**meta):checks.append(dict(check=name,passed=bool(yes),max_abs_difference=float(delta),**meta))
def loadnp(p):read(p);return np.load(p,mmap_mode='r',allow_pickle=False)
def weekly(d):
 v=d.copy();v['week']=v.datetime.dt.to_period('W-SUN').dt.start_time;g=v.groupby('week',sort=True);o=g[CHANNELS].agg(dict(open='first',high='max',low='min',close='last',volume='sum'));o['available']=g.datetime.max();o['first_session']=g.datetime.min();return o.reset_index()
def feat(d,t):
 w=weekly(d);av=w[w.available<=t].tail(26);dw=d[d.datetime<=t].tail(90);assert len(av)==26 and len(dw)==90;p0=float(dw.close.iloc[-1]);out=[]
 for f in [dw,av]:
  a=f[CHANNELS].to_numpy(float);z=np.empty_like(a);z[:,:4]=np.log(a[:,:4]/p0);v=np.log1p(a[:,4]);z[:,4]=v-np.mean(v);out.append((a,z))
 return out,dw,av
# Independently intersect per-ticker eligible anchor lists, before any purge.
meta={t:df(f'dataset/multiscale_dataset/panel/{t}/meta.csv') for t in TICKERS};sets=[set(pd.to_datetime(m.anchor_date).dt.normalize()) for m in meta.values()];cal=pd.DatetimeIndex(sorted(set.intersection(*sets)));n=len(cal);tr=int(.7*n);va=int(.85*n);check('Raw eligible intersection has2603 anchors',n==2603)
design=js(B2/'design.json');norm=df(B2/'normalization_stats.csv');windowtests={}
for name,shift in [('W1',822),('W2',411),('W3',0)]:
 bounds={'train':(0,tr-20-shift),'purge_train_val':(tr-20-shift,tr-shift),'val':(tr-shift,va-20-shift),'purge_val_test':(va-20-shift,va-shift),'test':(va-shift,n-shift)}
 for sp,(lo,hi) in bounds.items():
  dates=cal[lo:hi];stored=design['windows'][name]['ranges'][sp];check('Boundary '+name+' '+sp,dates[0].strftime('%Y-%m-%d')==stored['first'] and dates[-1].strftime('%Y-%m-%d')==stored['last'] and len(dates)==stored['n_anchors']);ranges.append(dict(window=name,split=sp,n_dates=len(dates),n_rows=len(dates)*17,first=str(dates[0].date()),last=str(dates[-1].date()),purge_size=len(dates) if sp.startswith('purge') else None))
 windowtests[name]=set(cal[slice(*bounds['test'])]);keys=df(B2/'tensors'/name/'sample_index.csv');labels={h:loadnp(B2/'tensors'/name/f'y_h{h}.npy') for h in [5,10,20]}
 for inp in ['O','C']:
  for res in ['daily','weekly']:
   raw=loadnp(B2/'tensors'/name/inp/f'X_{res}_raw.npy');z=loadnp(B2/'tensors'/name/inp/f'X_{res}_standardized.npy')
   # Every ticker's train-only stats; no VAL/TEST row participates.
   for t in TICKERS:
    ix=np.flatnonzero(keys.ticker.eq(t).to_numpy()&keys.split.eq('train').to_numpy());flat=np.asarray(raw[ix]).reshape(-1,5);mu=np.mean(flat,axis=0);sd=np.std(flat,axis=0,ddof=0);sd=np.where(sd<1e-8,1.,sd)
    stored=norm[(norm.window==name)&(norm.input_set==inp)&(norm.ticker==t)&(norm.resolution==res)].set_index('channel').loc[CHANNELS]
    error=max(float(np.max(np.abs(mu-stored['mean'].to_numpy()))),float(np.max(np.abs(sd-stored['std'].to_numpy()))));check('TRAIN-only normalization mean/std',error<=1e-10,error,window=name,input=inp,resolution=res,ticker=t)
    rows=np.flatnonzero(keys.ticker.eq(t).to_numpy());take=rows[[0,len(rows)//2,-1]];rec=(np.asarray(raw[take])-mu)/sd;delta=float(np.max(np.abs(rec-z[take])));check('Stored standardized tensor samples',delta<=1e-10,delta,window=name,input=inp,resolution=res,ticker=t)
 # Rebuild features and labels at three split-midpoint anchors for three representative issuers.
 for t in ['AES','CWEN','FCEL']:
  prices=df(f'dataset/finance/panel_raw/{t}_daily.csv');prices.datetime=pd.to_datetime(prices.datetime);prices=prices.sort_values('datetime').reset_index(drop=True)
  for sp in ['train','val','test']:
   rows=np.flatnonzero(keys.ticker.eq(t).to_numpy()&keys.split.eq(sp).to_numpy());ix=int(rows[len(rows)//2]);anchor=pd.Timestamp(keys.anchor_date.iloc[ix]);f,dw,ww=feat(prices,anchor)
   altered=prices.copy();after=altered.datetime>anchor;altered.loc[after,['open','high','low','close']]*=np.linspace(2,7,int(after.sum()))[:,None];altered.loc[after,'volume']*=3
   f2,_,_=feat(altered,anchor)
   for ri,res in enumerate(['daily','weekly']):
    for inp,fi in [('O',0),('C',1)]:
     raw=loadnp(B2/'tensors'/name/inp/f'X_{res}_raw.npy');delta=float(np.max(np.abs(f[ri][fi].astype(np.float32)-raw[ix])));check('Independent raw feature sample',delta<=1e-10,delta,window=name,input=inp,resolution=res,ticker=t,anchor=str(anchor.date()))
     perturb=float(np.max(np.abs(f[ri][fi]-f2[ri][fi])));check('After-anchor price perturbation invariance',perturb==0,perturb,window=name,input=inp,resolution=res,ticker=t,anchor=str(anchor.date()))
   pos=int(np.flatnonzero(prices.datetime.eq(anchor))[0]);p0=float(prices.close.iloc[pos])
   for h in [5,10,20]:
    y=np.float32(np.log(float(prices.close.iloc[pos+h])/p0));delta=abs(float(y)-float(labels[h][ix]));check('Close(t) to Close(t+h) label alignment',delta==0,delta,window=name,ticker=t,h=h,anchor=str(anchor.date()))
   examples.append(dict(window=name,split=sp,ticker=t,anchor=str(anchor.date()),daily_start=str(dw.datetime.min().date()),daily_end=str(dw.datetime.max().date()),weekly_start=str(ww.first_session.min().date()),weekly_end=str(ww.available.max().date()),standardizer_fit_cutoff=design['windows'][name]['ranges']['train']['last'],causal_reference_date=str(anchor.date()),feature_max_date_not_after_anchor=ww.available.max()<=anchor))
 for sp,nextsp in [('train','val'),('val','test')]:
  last=cal[bounds[sp][1]-1];first=cal[bounds[nextsp][0]]
  for t in TICKERS:
   prices=df(f'dataset/finance/panel_raw/{t}_daily.csv');a=pd.to_datetime(prices.datetime);i=int(np.flatnonzero(a.eq(last))[0]);check('Purge h20 target strictly before next split',a.iloc[i+20]<first,window=name,ticker=t,split=sp)
for a,b in [('W1','W2'),('W1','W3'),('W2','W3')]:check('TEST calendars nonoverlap '+a+'/'+b,len(windowtests[a]&windowtests[b])==0)
# Matched coverage: all saved keys/coverage metadata; sample raw rebuilt values and denominator.
design3=df(B3/'design_windows.csv');valid=loadnp(B3/'tensors/valid_daily.npy');key3=df(B3/'tensors/sample_index.csv');policy=js(B3/'causal_reference_policy.json');check('B3 fixed127/valid122..127',valid.shape==(43571,127) and valid.sum(1).min()==122 and valid.sum(1).max()==127)
for sp in ['train','val','test']:
 ix=int(np.flatnonzero(key3.split.eq(sp).to_numpy())[len(key3[key3.split.eq(sp)])//2]);r=design3.iloc[ix];t=r.ticker;prices=df(f'dataset/finance/panel_raw/{t}_daily.csv');prices.datetime=pd.to_datetime(prices.datetime);anchor=pd.Timestamp(r.anchor_date);weeks=weekly(prices);weeks=weeks[weeks.available<=anchor].tail(26);lo=weeks.first_session.min();hi=weeks.available.max();d=prices[prices.datetime.between(lo,hi)];v=d[CHANNELS].to_numpy(float);denom=float(d.close.iloc[-1]);z=np.empty_like(v);z[:,:4]=np.log(v[:,:4]/denom);lv=np.log1p(v[:,4]);z[:,4]=lv-lv.mean()
 for inp,arr in [('O',v),('C',z)]:
  key='O' if inp=='O' else policy['tensor_key'];stored=loadnp(B3/'tensors'/key/'X_daily_raw.npy');delta=float(np.max(np.abs(stored[ix,:len(v)]-arr.astype(np.float32))));check('B3 matched sample rebuilt',delta==0,delta,input=inp,split=sp)
 check('B3 no unfinished-week bar enters either branch',hi<=anchor and len(d)==int(valid[ix].sum()) and pd.Timestamp(r.daily_first_session)==lo and pd.Timestamp(r.daily_last_session)==hi,split=sp)
write('PRIORITY1_DATA_CHECKS.csv',pd.DataFrame(checks));write('RECOMPUTED_SPLIT_BOUNDARIES.csv',pd.DataFrame(ranges));write('FEATURE_PERTURBATION_EXAMPLES.csv',pd.DataFrame(examples));source_manifest('SOURCES_DATA.csv');print('DATA',len(checks),'checks; failures',sum(not r['passed'] for r in checks));print([r for r in checks if not r['passed']][:10])
