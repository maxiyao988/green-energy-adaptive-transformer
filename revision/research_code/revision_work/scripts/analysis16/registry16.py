"""Manifest from explicit completed source tables, followed by full key/label validation."""
from common16 import *
from functools import lru_cache
@lru_cache(None)
def frame(file):return pd.read_csv(ROOT/file,float_precision='round_trip')
def read_entry(r):
 p=ROOT/r['source'];d=pd.read_parquet(p) if p.suffix=='.parquet' else frame(r['source']).copy()
 if p.suffix=='.csv':
  for col,val in [('mechanism',r['model']),('frequency_setting',r['config']),('seed',r['seed'])]:
   if col in d:d=d[d[col].astype(str).eq(str(val))]
  hc='horizon' if 'horizon' in d else 'h';d=d[d[hc].astype(str).str.removesuffix('d').astype(int).eq(int(r['h']))]
  if 'split' in d:d=d[d.split.eq('test')]
 return canon(d[['ticker','anchor_date','y_true','y_pred']])
def add(rows,group,inp,model,h,seed,src,config='daily_weekly',window='W3',provenance='',**extra):
 rows.append(dict(group=group,input_set=inp,model=model,config=config,h=int(h),seed=int(seed),window=window,source=rel(ROOT/src),provenance=provenance,**extra))
def build():
 if verified('registry.COMPLETE.json'):return pd.read_csv(OUT/'registry.csv').fillna('')
 rows=[]
 a=SESSION/'20261016T142732626448Z_9ba11b23/summary/prediction_sources.csv'
 for r in pd.read_csv(a).itertuples():add(rows,'core',r.input_set,r.mechanism,r.horizon,r.seed,r.predictions,provenance='14a / formally archived EXACT12 per Step12A')
 for r in pd.read_csv(DAY/'cell_metrics.csv').itertuples():
  if r.block=='B1':add(rows,'samepatch',r.input_set,'same_patch_fixed',r.horizon,r.seed,str(Path(r.completion).parent/'test.parquet'),provenance='14c-day B1')
  elif r.frequency_setting!='daily_weekly' or r.model=='ridge':add(rows,'single_resolution' if r.model!='ridge' else 'ridge',r.input_set,r.model,r.horizon,r.seed,str(Path(r.completion).parent/'test.parquet'),config=r.frequency_setting,provenance='14c-day B4')
 for inp,source in [('O',FIX/'O_uniform12_per_seed.csv'),('C',B5/'benchmark_cell_metrics.csv')]:
  for r in pd.read_csv(source).itertuples():
   if r.input_set==inp:add(rows,'neural',inp,r.model,r.horizon,r.seed,r.prediction,provenance='B5FIX uniform12' if inp=='O' else 'B5/14c-day CPU12')
 # Original single-resolution controls retain available seed42 only; they are explicitly separate from DW B5FIX.
 for model in ['lstm','vanilla','swim','ridge']:
  for cfg in ['daily_only','weekly_only']+(['daily_weekly'] if model=='ridge' else []):
   for h in HS:
    src=f'revision_work/outputs/skill_analysis/canonical_preds/{model}_{cfg}_h{h}_s42_test.parquet'
    add(rows,'ridge' if model=='ridge' else 'single_resolution','O',model,h,42,src,config=cfg,provenance='archived single-resolution MAIN V2 (not uniform DW benchmark family)')
 src='dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv'
 for cfg in ['daily_only','weekly_only']:
  for h in HS:add(rows,'single_resolution','O','adaptive',h,42,src,config=cfg,provenance='archived MAIN V2 single-resolution')
 for r in pd.read_csv(B3/'all_cell_metrics.csv').itertuples():add(rows,'B3',r.input_set,r.mechanism,r.horizon,r.seed,r.prediction_file,config=r.configuration,provenance='14c-B3 CPU12')
 for r in pd.read_csv(B2S/'all_cell_metrics_5seed.csv').itertuples():
  if r.window!='W3':add(rows,'rolling',r.input_set,r.model,r.horizon,r.seed,r.predictions,window=r.window,provenance='B2/B2 supplement CPU12')
 for r in pd.read_csv(FROZEN/'prediction_files.csv').itertuples():
  if r.setting=='head_frozen' and r.split=='test':add(rows,'frozen',r.input_set,r.variant,r.horizon,r.seed,r.path,config=r.backbone,provenance='Step15 primary head_frozen')
 R=pd.DataFrame(rows);assert not R.duplicated(['group','input_set','model','config','h','seed','window']).any()
 R['id']=['_'.join(map(str,[r.group,r.input_set,r.model,r.config,r.h,r.seed,r.window])) for r in R.itertuples()]
 from skill_common import load_pred
 refs={};label_files=[]
 for window in ['W1','W2','W3']:
  if window=='W3':
   for h in HS:
    parts=[]
    for sp in ['train','val','test']:
     p=ROOT/f'revision_work/outputs/preds/static_dw_h{h}_s0_{sp}.parquet';d=canon(pd.read_parquet(p));d['split']=sp;parts.append(d[['ticker','anchor_date','split','y_true']]);label_files.append(p)
    refs[window,h]=pd.concat(parts,ignore_index=True)
  else:
   p=B2/f'tensors/{window}/sample_index.csv';ix=pd.read_csv(p);label_files.append(p)
   for h in HS:
    py=B2/f'tensors/{window}/y_h{h}.npy';d=ix[['ticker','anchor_date','split']].copy();d['y_true']=np.load(py);refs[window,h]=canon(d[d.split.isin(['train','val','test'])]);label_files.append(py)
 files=[];checks=[]
 for (w,h),d in refs.items():files.append(pq(f'labels/{w}_h{h}.parquet',d))
 for i,r in R.iterrows():
  d=read_entry(r);ref=canon(refs[r.window,r.h].query("split=='test'"));assert len(d)==6647 and not d.duplicated(['ticker','anchor_date']).any(),r.id
  assert d[['ticker','anchor_date']].equals(ref[['ticker','anchor_date']]),r.id+' keys'
  assert np.array_equal(d.y_true.to_numpy(np.float32),ref.y_true.to_numpy(np.float32)),r.id+' labels'
  assert np.isfinite(d[['y_true','y_pred']]).all().all(),r.id
  p=pq(f'canonical/{r.id}.parquet',d);files.append(p);R.loc[i,'canonical']=rel(p);R.loc[i,'sha256']=sha(ROOT/r.source)
  checks.append(dict(id=r.id,rows=len(d),dates=d.anchor_date.nunique(),keys_equal=True,labels_float32_equal=True,finite=True,source_sha256=sha(ROOT/r.source),canonical_sha256=sha(p)))
 files.extend([csv('registry.csv',R),csv('prediction_checks.csv',pd.DataFrame(checks)),csv('protected_label_sources.csv',pd.DataFrame([dict(path=rel(p),sha256=sha(p)) for p in sorted(set(label_files))]))])
 complete('registry.COMPLETE.json',files,dict(cells=len(R),training_runs=0))
 print('REGISTRY VERIFIED',len(R),R.groupby('group').size().to_dict(),flush=True)
 return R
if __name__=='__main__':build()
