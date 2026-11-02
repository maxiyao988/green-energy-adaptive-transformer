"""Variable-length matched-window binding using imported builders/transforms/stats."""
from b3_io import *
from b3_design import base,feature_module
import numpy as np,pandas as pd
common,formal,core=base.common,base.formal,base.causal_core

def npout(path,x):
 with target(path).open('xb') as f:np.save(f,x,allow_pickle=False)

def make_valid_stats(ds,valid):
 # Preserve the imported normalizer's float32 arithmetic, ddof0 and std guard.
 # Compact only real TRAIN entries for Daily; never let padding enter fit.
 stats=common.fit_train_only_normalizer(ds)
 for ticker in np.unique(ds.ticker):
  rows=(ds.ticker==ticker)&(ds.split=='train');flat=ds.X_daily[rows][valid[rows]];n=len(flat)
  compact=common.PanelDataset(np.full(n,ticker),np.zeros(n,dtype='datetime64[ns]'),np.full(n,'train'),flat[:,None,:],np.zeros((n,1,5),dtype=flat.dtype),{})
  stats[(ticker,'daily')]=common.fit_train_only_normalizer(compact)[(ticker,'daily')]
 return stats

def load_panel(inp):
 key=inp
 if inp=='C':key=json.loads((EXP/'causal_reference_policy.json').read_text())['tensor_key']
 folder=EXP/'tensors';meta=pd.read_csv(folder/'sample_index.csv')
 ds=common.PanelDataset(meta.ticker.to_numpy(),pd.to_datetime(meta.anchor_date).to_numpy(),meta.split.to_numpy(),np.load(folder/key/'X_daily_raw.npy'),np.load(folder/key/'X_weekly_raw.npy'),{h:np.load(folder/f'y_h{h}.npy') for h in [5,10,20]})
 ds.valid_daily=np.load(folder/'valid_daily.npy');ds.input_set_key=key
 return ds

def fit_bound_stats(ds):
 assert hasattr(ds,'input_set_key')
 data=json.loads((EXP/'normalization_stats.json').read_text())[ds.input_set_key]
 return {(r['ticker'],r['resolution']):(np.array(r['mean'],np.float32),np.array(r['std'],np.float32)) for r in data}

def apply_bound_stats(ds,stats,frequency):
 out=common.apply_normalizer(ds,stats,frequency)
 if frequency=='weekly_only':return out
 if frequency=='daily_only':out[~ds.valid_daily]=0
 else:out[0][~ds.valid_daily]=0
 return out

def contract(ds):
 # Keep the original manifest/date/universe/count/label guard, substituting only
 # the original frozen Daily view for its obsolete hardcoded90 shape check.
 frozen=common.load_frozen_panel()
 view=common.PanelDataset(ds.ticker,ds.anchor_date,ds.split,frozen.X_daily,ds.X_weekly,ds.y)
 formal_original_contract(view)
 assert ds.X_daily.shape==(43571,127,5) and ds.valid_daily.shape==(43571,127)
 assert ds.valid_daily.dtype==np.bool_ and ds.valid_daily.sum(1).min()==122 and ds.valid_daily.sum(1).max()==127
 assert np.array_equal(ds.valid_daily,np.arange(127)[None,:]<ds.valid_daily.sum(1)[:,None])
 for f in ['ticker','split']:assert np.array_equal(getattr(ds,f),getattr(frozen,f))
 assert np.array_equal(pd.to_datetime(ds.anchor_date).to_numpy(),pd.to_datetime(frozen.anchor_date).to_numpy())
 for h in [5,10,20]:assert np.array_equal(ds.y[h],frozen.y[h])
 assert np.isfinite(ds.X_daily).all() and np.isfinite(ds.X_weekly).all() and (ds.X_daily[~ds.valid_daily]==0).all()

formal_original_contract=formal._formal_check_dataset_contract

def prepare():
 base.check_profile();fb=feature_module();frozen=common.load_frozen_panel();causal=core.load_causal();formal_original_contract(frozen);formal_original_contract(causal)
 design=pd.read_csv(EXP/'design_windows.csv');assert len(design)==43571
 assert np.array_equal(design.ticker.to_numpy(),frozen.ticker) and np.array_equal(pd.to_datetime(design.anchor_date).to_numpy(),pd.to_datetime(frozen.anchor_date).to_numpy())
 valid=np.arange(127)[None,:]<design.valid_daily_sessions.to_numpy()[:,None]
 raw={k:np.zeros((43571,127,5),np.float32) for k in ['O','C_anchor','C_end']}
 weekly={'O':frozen.X_weekly,'C_anchor':causal.X_weekly,'C_end':np.empty_like(causal.X_weekly)}
 for ticker in sorted(common.FROZEN_TICKERS):
  selected=np.flatnonzero(frozen.ticker==ticker);g=design.iloc[selected];d=fb.load_prices(ticker)
  anchors=pd.to_datetime(frozen.anchor_date[selected]).to_numpy();source=d[core.CHANNELS].to_numpy();dates=pd.DatetimeIndex(d.datetime)
  lo=dates.get_indexer(pd.to_datetime(g.daily_first_session));hi=dates.get_indexer(pd.to_datetime(g.daily_last_session))+1;di=dates.get_indexer(anchors)
  assert (lo>=0).all() and (hi>lo).all()
  ow,_,_,anchor_ref,_=fb.windows(d,anchors);end_ref=d.close.to_numpy()[hi-1]
  weekly['C_end'][selected]=fb.transform(ow[1],end_ref).astype(np.float32)
  for n in sorted(g.valid_daily_sessions.unique()):
   local=np.flatnonzero(g.valid_daily_sessions.to_numpy()==n);ix=selected[local]
   windows=np.lib.stride_tricks.sliding_window_view(source,int(n),axis=0).transpose(0,2,1)[lo[local]]
   assert np.array_equal(hi[local]-lo[local],np.full(len(local),n))
   raw['O'][ix,:n]=windows.astype(np.float32)
   raw['C_anchor'][ix,:n]=fb.transform(windows,anchor_ref[local]).astype(np.float32)
   raw['C_end'][ix,:n]=fb.transform(windows,end_ref[local]).astype(np.float32)
  assert np.array_equal(ow[0].astype(np.float32),frozen.X_daily[selected])
  print('TENSORS_BUILT',ticker,flush=True)
 folder=EXP/'tensors';meta=pd.DataFrame(dict(ticker=frozen.ticker,anchor_date=pd.to_datetime(frozen.anchor_date).strftime('%Y-%m-%d'),split=frozen.split))
 with target(folder/'sample_index.csv').open('x',newline='') as f:meta.to_csv(f,index=False)
 npout(folder/'valid_daily.npy',valid)
 for h in [5,10,20]:npout(folder/f'y_h{h}.npy',frozen.y[h])
 stats_all={};normal_rows=[];checks=[]
 for key in ['O','C_anchor','C_end']:
  ds=common.PanelDataset(frozen.ticker,frozen.anchor_date,frozen.split,raw[key],weekly[key],frozen.y);ds.valid_daily=valid
  stats=make_valid_stats(ds,valid);z=common.apply_normalizer(ds,stats,'daily_weekly');z[0][~valid]=0
  assert np.isfinite(z[0]).all() and np.isfinite(z[1]).all() and z[0][valid].any(axis=-1).all()
  stats_all[key]=[]
  for (ticker,res),(mu,sd) in stats.items():
   row=dict(ticker=ticker,resolution=res,mean=mu.tolist(),std=sd.tolist());stats_all[key].append(row)
   for j,ch in enumerate(common.FEATURES):normal_rows.append(dict(tensor_key=key,ticker=ticker,resolution=res,channel=ch,mean=float(mu[j]),std=float(sd[j]),ddof=0,padding_excluded_from_fit=True))
  for branch,un,standard in zip(['daily','weekly'],[raw[key],weekly[key]],z):
   npout(folder/key/f'X_{branch}_raw.npy',un);npout(folder/key/f'X_{branch}_standardized.npy',standard)
  if key=='C_anchor':
   assert np.array_equal(weekly[key],causal.X_weekly)
   assert np.array_equal(z[1],np.load(core.OUT/'tensors/X_weekly_causal_standardized.npy'))
  checks.append(dict(tensor_key=key,rows=43571,shape_daily=list(ds.X_daily.shape),shape_weekly=list(ds.X_weekly.shape),valid_mask_exact=True,padding_zero_after_standardization=True,real_tokens_never_all_zero=True,weekly_binding_bitwise_unchanged=key in ['O','C_anchor'],reference_rule='anchor-date Close' if key=='C_anchor' else 'last completed-week source Close' if key=='C_end' else 'original adjusted levels'))
 jsonout(EXP/'normalization_stats.json',stats_all);csvout(EXP/'normalization_stats.csv',normal_rows)
 paths=sorted(p for p in folder.rglob('*') if p.is_file())
 jsonout(EXP/'tensor_manifest.json',dict(utc=utc(),files=[artifact(p) for p in paths],checks=checks,no_training=True,causal_policy_pending=True,normalization='imported common.fit_train_only_normalizer with real TRAIN Daily entries compacted; imported apply_normalizer; standardized padding then zeroed; ddof0 and original std guard'))
 print('B3_CANDIDATE_TENSORS_PREPARED',utc(),flush=True)
if __name__=='__main__':prepare()
