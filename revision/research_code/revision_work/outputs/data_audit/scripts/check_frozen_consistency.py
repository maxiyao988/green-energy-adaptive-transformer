"""Final frozen-input consistency checks, no training/inference."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
import core,numpy as np,pandas as pd,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
ds=core.common.load_frozen_panel();O=core.common.apply_normalizer(ds,core.common.fit_train_only_normalizer(ds),'daily_weekly')
C={r:np.load(ROOT/f'revision_work/outputs/causal_features/tensors/X_{r}_causal_standardized.npy',mmap_mode='r') for r in ['daily','weekly']}
index={(str(t),str(pd.Timestamp(d).date())):i for i,(t,d) in enumerate(zip(ds.ticker,ds.anchor_date))}
f=pd.read_csv(OUT/'C_event_window_features_annotated.csv');out=[]
for row in f.itertuples():
 i=index.get((row.ticker,row.anchor_date))
 if i is None:continue
 j=3 if row.channel=='Close' else 4;k=0 if row.resolution=='daily' else 1
 for family,a in [('original',O[k]),('causal',C[row.resolution])]:
  expected_last=float(a[i,-1,j]);expected_mean=float(a[i,:,j].mean(dtype=np.float64));out.append(dict(ticker=row.ticker,event_date=row.event_date,anchor_date=row.anchor_date,resolution=row.resolution,channel=row.channel,family=family,last_abs_difference=abs(getattr(row,family+'_last')-expected_last),mean_abs_difference=abs(getattr(row,family+'_window_mean')-expected_mean)))
r=rows('C_reconstruction_vs_frozen.csv',out);assert max(r.last_abs_difference.max(),r.mean_abs_difference.max())<1e-12
labeldiff=[];d=pd.read_csv(OUT/'D_input_comparisons.csv');d=d[d.quantity.str.startswith('label_')]
for row in d.itertuples():
 h=int(row.quantity.split('_')[1][1:]);i=index[(row.ticker,row.decision_date)];labeldiff.append(dict(split=row.split,decision_h=row.h,decision_date=row.decision_date,ticker=row.ticker,label_h=h,original_recomputed=np.float32(row.original),asof_recomputed=np.float32(row.asof),frozen_label=ds.y[h][i],original_exact=bool(np.float32(row.original)==ds.y[h][i]),asof_exact=bool(np.float32(row.asof)==ds.y[h][i])))
l=rows('D_labels_vs_frozen.csv',labeldiff);assert l.original_exact.all() and l.asof_exact.all()
js('frozen_consistency.json',dict(C_retained_channel_rows=len(r)//2,C_original_max_abs_last=float(r[r.family=='original'].last_abs_difference.max()),C_causal_max_abs_last=float(r[r.family=='causal'].last_abs_difference.max()),C_max_abs_mean=float(r.mean_abs_difference.max()),D_label_values=len(l),D_original_exact_frozen=bool(l.original_exact.all()),D_asof_exact_frozen=bool(l.asof_exact.all()),torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS'],nice=os.getpriority(os.PRIO_PROCESS,0)))
print('Frozen consistency verified',len(r)//2,'C channel rows;',len(l),'D label values',flush=True)
