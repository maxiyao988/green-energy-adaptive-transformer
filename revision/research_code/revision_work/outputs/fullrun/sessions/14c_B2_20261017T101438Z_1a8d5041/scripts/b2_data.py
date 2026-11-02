"""Split design and tensor preparation; imported original and causal builders only."""
from b2_io import *
import fullrun14a_worker_v2 as base
import numpy as np,pandas as pd,torch,importlib,hashlib
common,formal,core=base.common,base.formal,base.causal_core

def feature_module():
 # The imported research module has legacy import-time OMP/MPL assignments. Keep this job's profile and output namespace.
 cls=type(os.environ);original=cls.__setitem__
 def setter(self,key,value):
  if key in ENV_KEYS[:5]:return
  if key=='MPLCONFIGDIR':value=str(EXP/'matplotlib_cache')
  return original(self,key,value)
 cls.__setitem__=setter
 try:module=importlib.import_module('build_and_audit')
 finally:cls.__setitem__=original
 base.check_profile();return module

def npout(path,array):
 with target(path).open('xb') as f:np.save(f,array,allow_pickle=False)

def load_panel(window,inp):
 folder=EXP/'tensors'/window;meta=pd.read_csv(folder/'sample_index.csv')
 return common.PanelDataset(meta.ticker.to_numpy(),pd.to_datetime(meta.anchor_date).to_numpy(),meta.split.to_numpy(),np.load(folder/inp/'X_daily_raw.npy'),np.load(folder/inp/'X_weekly_raw.npy'),{h:np.load(folder/f'y_h{h}.npy') for h in [5,10,20]})

def window_contract(ds,window):
 d=design()['windows'][window];c=d['counts'];n=sum(c.values())
 assert len(ds.ticker)==len(ds.anchor_date)==len(ds.split)==len(ds.X_daily)==len(ds.X_weekly)==n
 assert ds.X_daily.shape==(n,90,5) and ds.X_weekly.shape==(n,26,5)
 assert set(ds.ticker)==set(common.FROZEN_TICKERS)
 assert sha(formal.DATASET_MANIFEST_PATH)==formal.DATASET_MANIFEST_SHA256
 f=pd.DataFrame(dict(ticker=ds.ticker,anchor_date=pd.to_datetime(ds.anchor_date),split=ds.split));assert not f.duplicated(['ticker','anchor_date']).any()
 assert f.groupby('anchor_date').ticker.nunique().eq(17).all()
 for sp,count in c.items():
  mask=ds.split==sp;dates=pd.to_datetime(ds.anchor_date[mask]);assert int(mask.sum())==count
  assert dates.nunique()==d['ranges'][sp]['n_anchors']
  assert dates.min().strftime('%Y-%m-%d')==d['ranges'][sp]['first'] and dates.max().strftime('%Y-%m-%d')==d['ranges'][sp]['last']
 for h in [5,10,20]:assert ds.y[h].shape==(n,) and np.isfinite(ds.y[h]).all()
 assert np.isfinite(ds.X_daily).all() and np.isfinite(ds.X_weekly).all()
 meta=pd.read_csv(EXP/'tensors'/window/'sample_index.csv')
 assert np.array_equal(ds.ticker,meta.ticker.to_numpy()) and np.array_equal(ds.split,meta.split.to_numpy())
 assert np.array_equal(pd.to_datetime(ds.anchor_date).to_numpy(),pd.to_datetime(meta.anchor_date).to_numpy())
 for h in [5,10,20]:assert np.array_equal(ds.y[h],np.load(EXP/'tensors'/window/f'y_h{h}.npy'))

def prepare():
 base.check_profile();base.exp3._verify_frozen_dependencies();base.exp4._verify_frozen_dependencies();base.exp4._verify_frozen_training_constants()
 records=common.load_all_tickers();_,detail=common.build_tradeoff_curve(records);retained=sorted(detail[17]['retained']);assert retained==sorted(common.FROZEN_TICKERS)
 split=common.compute_split_for_dates(detail[17]['common_dates']);calendar=pd.DatetimeIndex(split['sorted_dates']);assert len(calendar)==2603
 train_raw=len(split['train_dates']);val_raw=len(split['val_dates']);assert (train_raw,val_raw,len(split['test_dates']))==(1822,390,391)
 original=common.load_frozen_panel();causal=core.load_causal();formal._formal_check_dataset_contract(original);formal._formal_check_dataset_contract(causal)
 # Select unchanged raw sample bundles from the original imported panel-record loader, including dates purged only in W3.
 tickers=[];dates=[];daily=[];weekly=[];labels={h:[] for h in [5,10,20]}
 for ticker in retained:
  rec=records[ticker];ix=pd.DatetimeIndex(rec['anchor_date_norm']).get_indexer(calendar);assert (ix>=0).all()
  tickers.extend([ticker]*len(calendar));dates.extend(calendar.to_numpy());daily.append(rec['X_daily'][ix]);weekly.append(rec['X_weekly'][ix])
  for h in labels:labels[h].append(rec[f'y_{h}d'][ix])
 full=common.PanelDataset(np.asarray(tickers),np.asarray(dates),np.full(len(tickers),'unset'),np.concatenate(daily).astype(np.float32),np.concatenate(weekly).astype(np.float32),{h:np.concatenate(v).astype(np.float32) for h,v in labels.items()})
 windows={};rows=[];calendar_rows=[]
 for window,multiple in [('W1',2),('W2',1),('W3',0)]:
  shift=411*multiple;bounds=dict(train=(0,train_raw-20-shift),purge_train_val=(train_raw-20-shift,train_raw-shift),val=(train_raw-shift,train_raw+val_raw-20-shift),purge_val_test=(train_raw+val_raw-20-shift,train_raw+val_raw-shift),test=(train_raw+val_raw-shift,len(calendar)-shift))
  codes=np.full(len(calendar),'exclude',dtype='U16');ranges={}
  for part,(lo,hi) in bounds.items():
   assert 0<=lo<hi<=len(calendar);codes[lo:hi]=part;selected=calendar[lo:hi]
   ranges[part]=dict(first=selected[0].strftime('%Y-%m-%d'),last=selected[-1].strftime('%Y-%m-%d'),n_anchors=len(selected),lo_index=lo,hi_index_exclusive=hi)
   for h in [5,10,20]:rows.append(dict(window=window,horizon=h,split=part,first_anchor=ranges[part]['first'],last_anchor=ranges[part]['last'],n_anchors=len(selected),n_tickers=17,n_rows=len(selected)*17,used_by_models=part in ['train','val','test']))
  assert ranges['val']['n_anchors']==370 and ranges['test']['n_anchors']==391
  assert ranges['purge_train_val']['n_anchors']==ranges['purge_val_test']['n_anchors']==20
  windows[window]=dict(shift_anchors=shift,ranges=ranges,counts={sp:ranges[sp]['n_anchors']*17 for sp in ['train','val','test']})
  for i,date in enumerate(calendar):calendar_rows.append(dict(window=window,calendar_index=i,anchor_date=date.strftime('%Y-%m-%d'),assignment=codes[i]))
 tests={w:set(calendar[x['ranges']['test']['lo_index']:x['ranges']['test']['hi_index_exclusive']]) for w,x in windows.items()}
 nonoverlap=[]
 for a,b in [('W1','W2'),('W1','W3'),('W2','W3')]:
  overlap=tests[a]&tests[b];assert not overlap;nonoverlap.append(dict(window_a=a,window_b=b,overlapping_test_anchors=0,passed=True))
 csvout(EXP/'design_split_ranges.csv',rows);csvout(EXP/'design_calendar.csv',calendar_rows);csvout(EXP/'design_test_nonoverlap.csv',nonoverlap)
 jsonout(EXP/'design.json',dict(utc=utc(),calendar_n=2603,calendar_first=calendar[0].strftime('%Y-%m-%d'),calendar_last=calendar[-1].strftime('%Y-%m-%d'),calendar_basis='original17-ticker eligible common anchors before split-specific purging; same W3 boundaries',shift=411,windows=windows,test_ranges_disjoint=True,all_horizons_share_splits=True))
 # Pre-training report is immutable; the final report includes this exact design and timestamp.
 textout(EXP/'DESIGN_REPORT_STEP14C_B2.md','# BLOCKING\n\nNone in the design checks. No training has run.\n\n# Rolling-origin design before training\n\n'+table(['window','h','split','first','last','anchors','rows'],[[r['window'],r['horizon'],r['split'],r['first_anchor'],r['last_anchor'],r['n_anchors'],r['n_rows']] for r in rows])+'\nW1/W2/W3 TEST sets are pairwise disjoint. Each window has20-anchor TRAIN/VAL and VAL/TEST purges. The original eligible2603-anchor calendar is retained before window-specific purging; the40 dates purged in W3 are not permanently deleted from earlier windows. TRAIN expands from the common first anchor. W3 matches the archived split and will not be retrained. Training, if authorized by the passed tensor checks and AC power, starts only after this report and preflight exist.\n')
 fb=feature_module();normal_rows=[];boundary=[];rebuilt=[];tensor_paths=[];availability=[]
 for window,d in windows.items():
  code=np.full(len(calendar),'exclude',dtype='U16')
  for sp in ['train','val','test']:
   r=d['ranges'][sp];code[r['lo_index']:r['hi_index_exclusive']]=sp
  codes=np.tile(code,len(retained));keep=codes!='exclude';idx=np.flatnonzero(keep)
  ods=common.PanelDataset(full.ticker[idx],full.anchor_date[idx],codes[idx],full.X_daily[idx],full.X_weekly[idx],{h:full.y[h][idx] for h in labels})
  cd=np.empty_like(ods.X_daily);cw=np.empty_like(ods.X_weekly)
  for ticker in retained:
   mask=ods.ticker==ticker;anchors=pd.to_datetime(ods.anchor_date[mask]).to_numpy(dtype='datetime64[ns]');price=fb.load_prices(ticker);ow,starts,ends,ref,di=fb.windows(price,anchors)
   assert np.array_equal(ow[0].astype(np.float32),ods.X_daily[mask]);assert np.array_equal(ow[1].astype(np.float32),ods.X_weekly[mask])
   cd[mask]=fb.transform(ow[0],ref).astype(np.float32);cw[mask]=fb.transform(ow[1],ref).astype(np.float32)
   for resolution in ['daily','weekly']:
    assert (ends[resolution]<=anchors).all();availability.append(dict(window=window,ticker=ticker,resolution=resolution,n_anchors=len(anchors),original_windows_float32_exact=True,source_end_not_after_anchor=True))
   local_splits=ods.split[mask]
   for h in [5,10,20]:
    # This validates unchanged source labels and actual per-security target timing after split movement.
    actual=np.log(price.close.to_numpy()[di+h]/price.close.to_numpy()[di]).astype(np.float32)
    assert np.array_equal(actual,ods.y[h][mask]),(window,ticker,h,'labels')
    for before,after in [('train','val'),('val','test')]:
     last=int(np.flatnonzero(local_splits==before)[-1]);end=pd.Timestamp(price.datetime.iloc[di[last]+h]);nextdate=pd.Timestamp(anchors[local_splits==after][0]);assert end<nextdate
     boundary.append(dict(window=window,ticker=ticker,horizon=h,before_split=before,after_split=after,last_before_anchor=str(pd.Timestamp(anchors[last]).date()),last_target_date=str(end.date()),first_after_anchor=str(nextdate.date()),passed=True))
  cds=common.PanelDataset(ods.ticker,ods.anchor_date,ods.split,cd,cw,ods.y)
  if window=='W3':
   for f in ['ticker','split','X_daily','X_weekly']:
    assert np.array_equal(getattr(ods,f),getattr(original,f)),f
   assert np.array_equal(pd.to_datetime(ods.anchor_date).to_numpy(),pd.to_datetime(original.anchor_date).to_numpy())
   for f in ['X_daily','X_weekly']:assert np.array_equal(getattr(cds,f),getattr(causal,f)),('W3 causal',f)
   for h in labels:assert np.array_equal(ods.y[h],original.y[h]) and np.array_equal(cds.y[h],causal.y[h])
   formal._formal_check_dataset_contract(ods);formal._formal_check_dataset_contract(cds)
  folder=EXP/'tensors'/window;meta=pd.DataFrame(dict(ticker=ods.ticker,anchor_date=pd.to_datetime(ods.anchor_date).strftime('%Y-%m-%d'),split=ods.split))
  with target(folder/'sample_index.csv').open('x',newline='') as f:meta.to_csv(f,index=False)
  tensor_paths.append(folder/'sample_index.csv')
  for h in labels:npout(folder/f'y_h{h}.npy',ods.y[h]);tensor_paths.append(folder/f'y_h{h}.npy')
  for inp,ds in [('O',ods),('C',cds)]:
   window_contract(ds,window);stats=common.fit_train_only_normalizer(ds);z=common.apply_normalizer(ds,stats,'daily_weekly')
   for branch,raw,standardized in zip(['daily','weekly'],[ds.X_daily,ds.X_weekly],z):
    for kind,array in [('raw',raw),('standardized',standardized)]:
     path=folder/inp/f'X_{branch}_{kind}.npy';npout(path,array);tensor_paths.append(path)
   for (ticker,res),(mu,sd) in stats.items():
    for j,ch in enumerate(common.FEATURES):normal_rows.append(dict(window=window,input_set=inp,ticker=ticker,resolution=res,channel=ch,mean=float(mu[j]),std=float(sd[j]),ddof=0,train_anchors_per_ticker=d['ranges']['train']['n_anchors']))
   rebuilt.append(dict(window=window,input_set=inp,row_counts=d['counts'],weekly_availability_unchanged=True,labels_unchanged=True,train_only_normalizer='imported common.fit_train_only_normalizer; ddof0 over TRAIN window entries',W3_bitwise_equal_original_binding=window=='W3'))
  print('WINDOW_PREPARED',window,d['counts'],flush=True)
 csvout(EXP/'normalization_stats.csv',normal_rows);csvout(EXP/'purge_label_boundary_checks.csv',boundary);csvout(EXP/'feature_availability_checks.csv',availability)
 jsonout(EXP/'tensor_manifest.json',dict(utc=utc(),files=[artifact(p) for p in tensor_paths],checks=rebuilt,original_W3_bitwise_equal=True,causal_W3_bitwise_equal=True,causal_rebuilt_by='imported build_and_audit.load_prices/windows/transform on ORIGINAL snapshot only'))
 jsonout(EXP/'preflight.json',dict(utc=utc(),passed=True,dependencies=[artifact(p) for p in [Path(formal.__file__),Path(common.__file__),Path(core.__file__),Path(fb.__file__),Path(base.__file__),EXP/'design.json',EXP/'tensor_manifest.json']]+extra_dependencies(),windows=windows,weekly_availability_unchanged=True,original_labels_unchanged=True,all_purge_target_checks_pass=True,W3_original_and_causal_roundtrip_bitwise_equal=True,thread_counts=[12,12],environment={k:os.environ.get(k) for k in ENV_KEYS}))
 print('DESIGN_AND_TENSORS_PASS',utc(),flush=True)
if __name__=='__main__':prepare()
