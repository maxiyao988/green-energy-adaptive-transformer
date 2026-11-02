"""Post-lock prediction only; no estimator fitted."""
import phase_b26 as b
np,pd=b.np,b.pd
from pathlib import Path
import pickle,json,time,csv,sys,os,traceback
ROOT,OUT=b.ROOT,b.OUT

def main():
 b.check_protocol();lock=json.loads((OUT/'SELECTION_LOCK_SHA256.json').read_text());assert b.sha(OUT/'SELECTION_LOCK.csv')==lock['sha256'] and lock['n_cells']==33
 b.js('TEST_UNLOCK.json',dict(lock_sha256=lock['sha256'],n_selections=33,utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),before_first_TEST_deserialization=True))
 # Preserve exclusive writes while permitting post-barrier inference.
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
  if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
   assert p.is_relative_to(OUT) and flags&os.O_EXCL,'Exclusive output boundary '+str(p)
 sys.addaudithook(audit)
 inv=list(csv.DictReader((OUT/'INVENTORY_STEP26.csv').open()));current=None;checks=[]
 for r in inv:
  key=r['window'],r['input_set']
  if key!=current:
   f=b.B2/'tensors'/key[0];meta=pd.read_csv(f/'sample_index.csv');ds=b.common.PanelDataset(meta.ticker.to_numpy(),pd.to_datetime(meta.anchor_date).to_numpy(),meta.split.to_numpy(),np.load(f/key[1]/'X_daily_raw.npy'),np.load(f/key[1]/'X_weekly_raw.npy'),{h:np.load(f/f'y_h{h}.npy') for h in b.HS});stats=b.common.fit_train_only_normalizer(ds);current=key
  h=int(r['h']);mask=ds.split=='test';X=b.common.flatten_features(r['config'],b.common.apply_normalizer(ds,stats,r['config']));p=OUT/'cells'/r['cell_id']
  with (p/'selected_model.pkl').open('rb') as handle:m=pickle.load(handle)
  with (p/'original_model.pkl').open('rb') as handle:original=pickle.load(handle)
  y=ds.y[h][mask].astype(float);dates=ds.anchor_date[mask]
  if r['target']=='cross_sectional':
   yc=np.empty_like(y)
   for day in np.unique(dates):
    mm=dates==day;yc[mm]=y[mm]-np.mean(y[mm],dtype=np.float64)
  else:yc=y
  frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(dates).strftime('%Y-%m-%d'),y_true=yc,y_raw=y,y_pred=m.predict(X[mask]),y_original_grid=original.predict(X[mask])))
  source=ROOT/(r['canonical'] or r['test_source']);old=pd.read_parquet(source) if source.suffix=='.parquet' else pd.read_csv(source)
  old.anchor_date=pd.to_datetime(old.anchor_date).dt.strftime('%Y-%m-%d');old=old[['ticker','anchor_date','y_true','y_pred']].rename(columns={'y_pred':'y_old','y_true':'y_old_true'})
  frame=frame.merge(old,on=['ticker','anchor_date'],validate='one_to_one').sort_values(['anchor_date','ticker']).reset_index(drop=True);assert len(frame)==6647
  assert np.max(np.abs(frame.y_true-frame.y_old_true))<1e-14
  d=float(np.max(np.abs(frame.y_original_grid.astype(float)-frame.y_old.astype(float))));checks.append(dict(cell_id=r['cell_id'],max_abs_delta=d,classification='EXACT' if d<1e-7 else 'NEAR' if d<1e-5 else 'DIFFERENT',deferred_W3_O=r['window']=='W3' and r['input_set']=='O'))
  with (p/'test_step26.csv').open('x') as handle:frame.to_csv(handle,index=False,float_format='%.17g')
  print('TEST_PREDICTED',r['cell_id'],checks[-1]['classification'],flush=True)
 b.csvout('ORIGINAL_GRID_TEST_REPRODUCTION.csv',checks);b.js('PREDICTION_COMPLETE.json',dict(n_cells=len(checks),n_test_rows=6647,TEST_unlock_sha256=b.sha(OUT/'TEST_UNLOCK.json')))
if __name__=='__main__':
 try:main()
 except BaseException:b.js('PREDICTION_BLOCKING.json',dict(error=traceback.format_exc()));raise
