"""Requested Stage3c diagnostic levels from frozen V2 inputs and Part1 c(anchor)."""
import os,sys
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='3';os.environ['OMP_DYNAMIC']='FALSE';os.environ['MKL_DYNAMIC']='FALSE'
from core import *

def main():
 threads(3);ds=common.load_frozen_panel();formal._formal_check_dataset_contract(ds)
 factors=pd.read_csv(OUT/'adjustment_factors_by_date.csv');factors=factors.rename(columns={'Date':'anchor_date'})
 index=pd.DataFrame(dict(ticker=ds.ticker,anchor_date=pd.to_datetime(ds.anchor_date).strftime('%Y-%m-%d'),split=ds.split));aligned=index.merge(factors[['ticker','anchor_date','c']],on=['ticker','anchor_date'],validate='one_to_one',how='left',sort=False);c=aligned.c.to_numpy(float);assert len(c)==43571 and np.isfinite(c).all() and (c>0).all()
 arrays=[]
 for r,x in [('daily',ds.X_daily),('weekly',ds.X_weekly)]:
  a=x.astype(np.float64).copy();a[:,:,:4]/=c[:,None,None];a=a.astype(np.float32);assert np.array_equal(a[:,:,4],x[:,:,4]);assert np.isfinite(a).all() and (a[:,:,:4]>0).all();npout(f'stage3c/tensors/X_{r}_asof_raw.npy',a);arrays.append(a)
 cds=common.PanelDataset(ds.ticker,ds.anchor_date,ds.split,arrays[0],arrays[1],ds.y);stats=common.fit_train_only_normalizer(cds);z=common.apply_normalizer(cds,stats,'daily_weekly');rows=[]
 for r,a in zip(['daily','weekly'],z):assert np.isfinite(a).all();npout(f'stage3c/tensors/X_{r}_asof_standardized.npy',a)
 for (t,r),(mu,sd) in stats.items():
  for i,ch in enumerate(CHANNELS):rows.append(dict(ticker=t,resolution=r,channel=ch,mean=float(mu[i]),std=float(sd[i])))
 csvout('stage3c/asof_anchor_factors.csv',aligned);csvout('stage3c/asof_normalization_stats.csv',rows)
 jsonout('stage3c/feature_checks.json',dict(completed_utc=utc(),formula='Original frozen adjusted OHLC_tau / NEW-snapshot Part1 c(anchor); original Volume unchanged',counts={s:int((ds.split==s).sum()) for s in COUNTS},shapes=[list(a.shape) for a in arrays],finite=True,positive_prices=True,volume_exactly_v2=True,frozen_labels_splits_anchors=True,denominator='Daily anchor c(t) for both Daily and Weekly',factor_source=rel(OUT/'adjustment_factors_by_date.csv'),factor_source_sha256=sha(OUT/'adjustment_factors_by_date.csv'),caveat='Formula-based diagnostic only; not authenticated point-in-time data. BEP distribution/security identity and historical volume convention remain unverified.'))
 print('ASOF FEATURES COMPLETE',utc(),flush=True)
if __name__=='__main__':main()
