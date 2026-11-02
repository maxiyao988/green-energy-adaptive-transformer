"""Independent W3 metrics: math.fsum squared errors and scipy Spearman per date."""
import os,sys,math,csv,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
import numpy as np,pandas as pd
from scipy.stats import spearmanr
S=pd.read_csv(OUT/'TEST_METRICS_STEP26.csv',float_precision='round_trip');I=pd.read_csv(OUT/'INVENTORY_STEP26.csv');rows=[]
for r in I[I.window=='W3'].itertuples():
 f=pd.read_csv(OUT/'cells'/r.cell_id/'test_step26.csv',float_precision='round_trip');y=f.y_true.to_numpy(float)
 for arm,col in [('old','y_old'),('new','y_pred')]:
  p=f[col].to_numpy(float);mse=math.fsum(float(a-b)**2 for a,b in zip(y,p))/len(y);ics=[]
  for d,g in f.groupby('anchor_date',sort=True):
   a=g[col].to_numpy(float);b=g.y_true.to_numpy(float)
   ics.append(float(spearmanr(a,b).statistic) if a.std()>=1e-12 and b.std()>=1e-12 else float('nan'))
  ic=float(np.nanmean(ics)) if np.isfinite(ics).any() else float('nan');expected=S[(S.cell_id==r.cell_id)&(S.arm==arm)].iloc[0];dm=abs(mse-expected.mse);di=abs(ic-expected.mean_rank_ic) if np.isfinite(ic) and np.isfinite(expected.mean_rank_ic) else 0 if np.isnan(ic) and np.isnan(expected.mean_rank_ic) else float('inf')
  rows.append(dict(cell_id=r.cell_id,arm=arm,mse_independent=mse,mse_discrepancy=dm,rank_ic_independent=ic,rank_ic_discrepancy=di,passed=dm<1e-12 and di<1e-12,n_dates=len(ics)))
with (OUT/'INDEPENDENT_W3_TEST_METRICS.csv').open('x') as f:pd.DataFrame(rows).to_csv(f,index=False,float_format='%.17g')
with (OUT/'INDEPENDENT_TEST_VALIDATION.json').open('x') as f:json.dump(dict(n_cells=21,n_arm_checks=len(rows),all_pass=all(x['passed'] for x in rows),max_mse_discrepancy=max(x['mse_discrepancy'] for x in rows),max_rank_ic_discrepancy=max(x['rank_ic_discrepancy'] for x in rows)),f,indent=2)
print('INDEPENDENT_W3_TEST',all(x['passed'] for x in rows))
