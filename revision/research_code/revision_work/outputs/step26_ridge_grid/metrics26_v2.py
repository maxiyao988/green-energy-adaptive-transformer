"""Post-lock Ridge-only metrics through original Step5/16 functions."""
import os,sys,json,pickle,csv,hashlib,ast,copy,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'revision_work/scripts/analysis16'))
from statistics16_v3 import s
np,pd=s.np,s.pd
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
A16=ROOT/'revision_work/outputs/analysis16'

def write(name,d):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:d.to_csv(f,index=False,float_format='%.17g')
def js(name,d):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2,default=str)
def frame(r):return pd.read_csv(OUT/'cells'/r['cell_id']/'test_step26.csv',float_precision='round_trip')
def main():
 assert (OUT/'PREDICTION_COMPLETE.json').exists()
 start=time.perf_counter();inv=pd.read_csv(OUT/'INVENTORY_STEP26.csv').fillna('');a=s.Analysis();records=[];series=[];ref_cache={};frames={}
 for r in inv.to_dict('records'):
  f=frame(r);frames[r['cell_id']]=f;w,h=r['window'],int(r['h']);y=f.y_true.to_numpy(float);Y=y.reshape(391,17)
  refs,gtm=a.refs(w,h)
  if r['target']=='cross_sectional':
   train=pd.read_csv(OUT/'cells'/r['cell_id']/'train.csv',float_precision='round_trip');m=float(train.y_true.mean());refs={k:np.zeros(len(y)) for k in s.cell_metrics.__globals__['REFS']};refs['global_train_mean']=np.full(len(y),m);gtm=m
  for arm,col in [('old','y_old'),('new','y_pred')]:
   p=f[col].to_numpy(float);P=p.reshape(Y.shape);z,cw,ic=s.cell_metrics(y,p,refs,h,s.dates(h,391,2),Y,P,gtm);mu,se,t,n=s.nw_t(ic,h);bb=np.nanmean(ic[s.dates(h,391,2)],axis=1) if np.isfinite(ic).any() else np.full(20000,np.nan)
   z.update(rank_ic_nw_se=se,rank_ic_nw_t=t,rank_ic_nw_lag=h,rank_ic_p_nw_two_sided=float(2*s.stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan,rank_ic_p_mbb=s.series_stats(bb,float(np.nanmean(ic)))['centered_p_two_sided'] if np.isfinite(ic).any() else np.nan,relative_mse_train_mean=1-z['r2os_vs_global_train_mean'],da_excluding_zero=float(np.mean(np.sign(p[y!=0])==np.sign(y[y!=0]))),n_nonzero_target=int((y!=0).sum()))
   meta={k:r[k] for k in ['cell_id','window','input_set','config','h','target']};records.append(dict(**meta,arm=arm,**z));series.append(pd.DataFrame(dict(cell_id=r['cell_id'],arm=arm,window=w,input_set=r['input_set'],config=r['config'],h=h,target=r['target'],anchor_date=f.anchor_date.unique(),cw_f=cw,rank_ic=ic)))
  print('METRICS',r['cell_id'],flush=True)
 M=pd.DataFrame(records);D=pd.concat(series,ignore_index=True);write('TEST_METRICS_STEP26.csv',M);write('DATE_SERIES_STEP26.csv',D)
 # Original all738 experimental-cell Holm family: retain non-Ridge raw statistics verbatim.
 changes=[]
 for filename in ['clarified/B_forecast_skill_final.csv','B_PRIMARY_forecast_skill.csv']:
  p=A16/filename
  if not p.exists():continue
  old=pd.read_csv(p,float_precision='round_trip');new=old.copy(deep=True)
  for r in inv[(inv.window=='W3')&(inv.target=='raw')].to_dict('records'):
   q=M[(M.cell_id==r['cell_id'])&(M.arm=='new')].iloc[0];idx=new.index[new.id==r['cell_id']];assert len(idx)==1
   for c in M.columns:
    if c in new.columns and c not in ['cell_id','arm','target']:new.loc[idx,c]=q[c]
  for col in ['cw_p_nw_one_sided','cw_p_mbb_one_sided','rank_ic_p_mbb']:
   adj=col+'_holm_all_cells';new[adj]=np.nan
   for i,v in s.holm({i:float(v) for i,v in new[col].items() if np.isfinite(v)}).items():new.loc[i,adj]=v
   for i in new.index:
    changes.append(dict(family=filename,id=new.loc[i,'id'],model=new.loc[i,'model'],test=col,old_raw_p=old.loc[i,col],new_raw_p=new.loc[i,col],old_holm=old.loc[i,adj],new_holm=new.loc[i,adj],old_reject=bool(old.loc[i,adj]<.05),new_reject=bool(new.loc[i,adj]<.05),decision_changed=bool((old.loc[i,adj]<.05)!=(new.loc[i,adj]<.05))))
  nmask=new.model!='ridge';rawcols=[c for c in old.columns if not c.endswith('_holm_all_cells') and c not in ['mse_mean','mse_sd','rank_ic_mean','rank_ic_sd']];pd.testing.assert_frame_equal(new.loc[nmask,rawcols],old.loc[nmask,rawcols],check_dtype=False)
  write('propagated/'+p.stem+('_primary' if filename.startswith('B_PRIMARY') else '')+'_step26.csv',new)
 write('HOLM_COMPARISON_STEP26.csv',pd.DataFrame(changes))
 # Rolling-window and subgroup values, using exact original clarified16 run block with model list bound to Ridge.
 source=ROOT/'revision_work/scripts/analysis16/clarified16.py';tree=ast.parse(source.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run');startnode=next(i for i,n in enumerate(run.body) if isinstance(n,ast.ImportFrom) and n.module=='engine');endnode=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='decisions' for t in n.targets));body=copy.deepcopy(run.body[startnode:endnode])
 for n in ast.walk(ast.Module(body=body,type_ignores=[])):
  if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='model':n.iter=ast.List(elts=[ast.Constant('ridge')],ctx=ast.Load())
 original_pred=a.pred
 def pred(id):
  if id in frames:return frames[id].y_pred.to_numpy(float)
  raise RuntimeError('Non-Ridge reevaluation forbidden '+id)
 a.pred=pred
 def csvsave(name,d):write('propagated/'+Path(name).stem+'_ridge_step26.csv',d);return OUT/'propagated'/Path(name).name
 env=dict(vars(s),a=a,OUT=A16,UP='',csv=csvsave,files=[]);exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source),'exec'),env)
 js('METRICS_COMPLETE.json',dict(cells=len(inv),arms=2,seconds=time.perf_counter()-start,non_Ridge_evaluations=0,Holm_decision_changes=sum(x['decision_changed'] for x in changes)))
if __name__=='__main__':
 try:main()
 except BaseException:js('METRICS_BLOCKING.json',dict(error=traceback.format_exc()));raise
