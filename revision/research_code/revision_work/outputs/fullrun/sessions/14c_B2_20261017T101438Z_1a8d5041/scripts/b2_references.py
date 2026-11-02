"""Read-only W3 reuse and naive denominators; no model training or checkpoint inference."""
from b2_io import *
from b2_data import common,load_panel,window_contract
import numpy as np,pandas as pd
import fullrun14a_io as oldio
from report_stage3bc_final import old_predictions

def prepare_references():
 ds=load_panel('W3','O');window_contract(ds,'W3');mask=ds.split=='test';key=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'))).sort_values(['ticker','anchor_date']).reset_index(drop=True)
 entries=[];metricrows=[];denominators=[]
 for w in ['W1','W2','W3']:
  d=load_panel(w,'O');c=load_panel(w,'C')
  for h in [5,10,20]:
   assert np.array_equal(d.y[h],c.y[h]);tm=np.float32(d.y[h][d.split=='train'].mean());yt=d.y[h][d.split=='test'];mse=float(np.mean((yt-tm)**2));assert mse>0
   denominators.append(dict(window=w,horizon=h,n_train=int((d.split=='train').sum()),training_mean=float(tm),test_mse_global_train_mean=mse,test_mse_zero=float(np.mean(yt**2)),arithmetic='frozen float32 labels/mean/MSE; TRAIN labels only for fitted constant'))
 csvout(EXP/'global_train_mean_references.csv',denominators)
 denom={(r['window'],r['horizon']):r['test_mse_global_train_mean'] for r in denominators}
 def add(inp,model,h,seed,frame,source,provenance):
  x=frame.copy();x['anchor_date']=pd.to_datetime(x.anchor_date).dt.strftime('%Y-%m-%d');x=x.sort_values(['ticker','anchor_date']).reset_index(drop=True)
  assert len(x)==6647 and x[['ticker','anchor_date']].equals(key);assert np.array_equal(x.y_true.to_numpy(np.float32),pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),y_true=ds.y[h][mask])).sort_values(['ticker','anchor_date']).y_true.to_numpy(np.float32))
  y=x.y_true.to_numpy(np.float32);p=x.y_pred.to_numpy(np.float32);assert np.isfinite(y).all() and np.isfinite(p).all()
  if model=='zero':assert np.array_equal(p,np.zeros_like(y))
  if model=='train_mean':assert np.array_equal(p,np.full_like(y,np.float32(ds.y[h][ds.split=='train'].mean())))
  out=pd.DataFrame(dict(ticker=x.ticker,anchor_date=x.anchor_date,split='test',y_true=y,y_pred=p));dest=EXP/'W3_reused_predictions'/f'W3_{inp}_{model}_h{h}_s{seed}_test.parquet'
  with target(dest).open('xb') as f:f.write(out.to_parquet(index=False,engine='pyarrow'))
  m=common.evaluate_predictions(y,p,out.ticker.to_numpy(),out.anchor_date.to_numpy());metricrows.append(dict(window='W3',input_set=inp,model=model,horizon=h,seed=seed,test_mse=m['pooled_mse'],mean_rank_ic=m['cross_sectional_rank_ic'],mse_relative_train_mean=m['pooled_mse']/denom[('W3',h)],R2_OS_train_mean=1-m['pooled_mse']/denom[('W3',h)],provenance=provenance,predictions=rel(dest)))
  entries.append(dict(window='W3',input_set=inp,model=model,horizon=h,seed=seed,source=rel(source),source_sha256=sha(source),copied_test_predictions=artifact(dest),n_rows=6647,keys_labels_identical=True,provenance=provenance,no_retraining=True))
 old=old_predictions()
 for (model,h,seed),frame in old.items():
  if seed not in [42,0,1] or model not in ['static','adaptive']:continue
  source=ROOT/('dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv' if seed==42 and model=='adaptive' else 'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv' if seed==42 else 'dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv')
  add('O',model,h,seed,frame,source,'archived formal W3 predictions; no retraining')
 for model in ['static','adaptive']:
  for seed in [42,0,1]:
   for h in [5,10,20]:
    cp,r=oldio.completed(oldio.job('C',model,h,seed,'Q1'));assert r
    source=cp.parent/'test.parquet';add('C',model,h,seed,pd.read_parquet(source),source,'14a12/12 C W3 predictions; no retraining')
 source=ROOT/'dataset/audit/main_v2/ridge/panel_ridge_test_predictions.csv';ridge=pd.read_csv(source);ridge['h']=ridge.horizon.astype(str).str.removesuffix('d').astype(int)
 for h in [5,10,20]:add('O','ridge',h,42,ridge[ridge.frequency_setting.eq('daily_weekly')&ridge.h.eq(h)],source,'archived MAIN V2 Ridge W3')
 day=FULLRUN/'sessions/14c_day_20261017T050034Z_12db13f7'
 for h in [5,10,20]:
  cps=list((day/'runs'/f'B4_C_ridge_daily_weekly_h{h}_s42').glob('attempt_*/COMPLETE.json'));assert len(cps)==1;cp=cps[0];r=json.loads(cp.read_text());assert r['validated']
  for x in r['artifacts']+r['dependencies']:assert sha(ROOT/x['path'])==x['sha256']
  source=cp.parent/'test.parquet';add('C','ridge',h,42,pd.read_parquet(source),source,'14c-day12/12 causal Ridge W3; no retraining')
 source=ROOT/'dataset/audit/main_v2/naive/main_v2_naive_test_predictions.csv';naive=pd.read_csv(source);naive['h']=naive.horizon.astype(str).str.removesuffix('d').astype(int)
 for inp in ['O','C']:
  for h in [5,10,20]:
   for model,name in [('zero','zero'),('train_mean','global_train_mean')]:add(inp,model,h,42,naive[naive.model.eq(name)&naive.h.eq(h)],source,'archived W3 deterministic baseline; O/C labels and TRAIN mean identical')
 assert len(entries)==54 and len(metricrows)==54
 with target(EXP/'W3_reference_metrics.csv').open('x',newline='') as f:pd.DataFrame(metricrows).to_csv(f,index=False)
 jsonout(EXP/'W3_provenance.json',dict(utc=utc(),n_cells=54,no_retraining=True,entries=entries))
 print('W3_REFERENCES_VERIFIED',len(entries),flush=True)
if __name__=='__main__':prepare_references()
