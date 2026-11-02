"""Original Step17 shrinkage grid selected using only new TRAIN/VAL forecasts."""
import os,sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'revision_work/scripts/analysis17'))
import common17 as c
np,pd=c.np,c.pd
assert (OUT/'SELECTION_LOCK_SHA256.json').exists()
assert not (OUT/'TEST_UNLOCK.json').exists()
rows=[];curves=[]
I=pd.read_csv(OUT/'INVENTORY_STEP26.csv').fillna('')
for r in I[(I.target=='raw')&(I.config=='daily_weekly')].itertuples():
 p=OUT/'cells'/r.cell_id;tr=pd.read_csv(p/'train.csv',float_precision='round_trip');va=pd.read_csv(p/'val.csv',float_precision='round_trip');m=float(tr.y_true.to_numpy(float).mean());y=va.y_true.to_numpy(float);v=va.y_pred.to_numpy(float);objective=[]
 for l in c.GRID:
  loss=float(np.mean((y-c.combine(v,m,float(l)))**2));objective.append(loss);curves.append(dict(cell_id=r.cell_id,lambda_value=l,val_mse=loss))
 k=int(np.argmin(objective));rows.append(dict(cell_id=r.cell_id,window=r.window,input_set=r.input_set,h=r.h,training_mean=m,selected_lambda=float(c.GRID[k]),val_mse=objective[k]))
for name,data in [('SHRINKAGE_SELECTION_STEP26.csv',rows),('SHRINKAGE_VAL_CURVES_STEP26.csv',curves)]:
 with (OUT/name).open('x') as f:pd.DataFrame(data).to_csv(f,index=False,float_format='%.17g')
with (OUT/'SHRINKAGE_PRETEST_GATE.json').open('x') as f:json.dump(dict(n_cells=len(rows),selection_sha256=hashlib.sha256((OUT/'SHRINKAGE_SELECTION_STEP26.csv').read_bytes()).hexdigest(),TEST_reads=0),f,indent=2)
print('SHRINKAGE_PRETEST_GATE',len(rows))
