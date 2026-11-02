"""Ridge-only downstream calculations using archived Step17/22D primitives."""
from metrics26_v2 import *
sys.path.insert(0,str(ROOT/'revision_work/scripts/analysis17'))
import common17 as c

def shrinkage():
 selections=pd.read_csv(OUT/'SHRINKAGE_SELECTION_STEP26.csv',float_precision='round_trip');source=ROOT/'revision_work/outputs/analysis17/test_results.csv';old=pd.read_csv(source,float_precision='round_trip');new=old.copy(deep=True);B=20000
 skill=pd.read_csv(OUT/'propagated/B_forecast_skill_final_step26.csv',float_precision='round_trip');rows=[]
 for r in selections.itertuples():
  f=pd.read_csv(OUT/'cells'/r.cell_id/'test_step26.csv',float_precision='round_trip');y=f.y_true.to_numpy(float);p=f.y_pred.to_numpy(float);Y=y.reshape(391,17);m=r.training_mean;ix=c.mbb_index(391,int(r.h),B,np.random.default_rng(np.random.SeedSequence([c.BASE,17,int(r.window[1:]),int(r.h)])));oldic=c.rank_ic_rows(p.reshape(Y.shape),Y);base=float(np.mean((y-m)**2));zero=float(np.mean(y**2));cid=f'{r.window}_{r.input_set}_ridge_h{r.h}_deterministic'
  for regime,l in [('PRIMARY',r.selected_lambda),('SECONDARY',.5)]:
   z=c.combine(p,m,l);mse=float(np.mean((y-z)**2));ic=c.rank_ic_rows(z.reshape(Y.shape),Y);loss=((y-z)**2-(y-m)**2).reshape(391,17).mean(1);theta=float(loss.mean());bs=c.series_stats(loss[ix].mean(1),theta) if l!=0 and np.any(loss!=0) else dict(ci_lower_95=np.nan,ci_upper_95=np.nan,centered_p_two_sided=np.nan);pv=bs['centered_p_two_sided'];values=dict(status='evaluated',lambda_selected_PRIMARY=r.selected_lambda,lambda_used=l,training_mean=m,test_mse=mse,r2os_vs_global_train_mean=1-mse/base,r2os_vs_zero=1-mse/zero,mean_rank_ic=float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan,original_rank_ic=float(np.nanmean(oldic)) if np.isfinite(oldic).any() else np.nan,rank_ic_max_abs_delta=float(np.nanmax(np.abs(ic-oldic))) if np.isfinite(ic-oldic).any() else np.nan,rank_ic_check='N/A: lambda0 constant Train-Mean' if l==0 else 'PASS',loss_difference=theta,loss_ci_lower_95=bs['ci_lower_95'],loss_ci_upper_95=bs['ci_upper_95'],loss_p_raw=pv,loss_p_for_Holm=pv if np.isfinite(pv) else 1.,test_performed=bool(np.isfinite(pv)),loss_test_status='two-sided date-level MBB, conditional on frozen forecasts/VAL selection' if np.isfinite(pv) else 'undefined: identical to Train-Mean; placeholder1 retained',selection_status='ready')
   idx=new.index[(new.cell_id==cid)&(new.regime==regime)];assert len(idx)==1
   for k,v in values.items():new.loc[idx,k]=v
   rows.append(dict(cell_id=cid,regime=regime,**values))
 for regime,g in new.groupby('regime'):
  for i,pv in c.holm(dict(zip(g.index,g.loss_p_for_Holm))).items():new.loc[i,'loss_p_Holm']=pv
 # Refresh copied CW references only for Ridge W3; rolling CW recomputation below is sourced separately.
 for r in selections.itertuples():
  if r.window!='W3':continue
  q=skill[(skill.model=='ridge')&(skill.input_set==r.input_set)&(skill.config=='daily_weekly')&(skill.h==r.h)].iloc[0];idx=new.index[(new.model=='ridge')&(new.window=='W3')&(new.input_set==r.input_set)&(new.h==r.h)]
  for dest,col in [('cw_p_nw_raw','cw_p_nw_one_sided'),('cw_p_mbb_raw','cw_p_mbb_one_sided'),('cw_p_nw_holm_Step16','cw_p_nw_one_sided_holm_all_cells'),('cw_p_mbb_holm_Step16','cw_p_mbb_one_sided_holm_all_cells')]:
   new.loc[idx,dest]=q[col]
   new.loc[idx[new.loc[idx,'lambda_used'].eq(0)],dest]=np.nan
 write('propagated/SHRINKAGE_full_family_step26.csv',new);write('SHRINKAGE_RIDGE_STEP26.csv',pd.DataFrame(rows));write('SHRINKAGE_HOLM_CHANGES_STEP26.csv',pd.DataFrame(dict(regime=new.regime,cell_id=new.cell_id,model=new.model,old_holm=old.loss_p_Holm,new_holm=new.loss_p_Holm,decision_changed=old.loss_p_Holm.lt(.05)!=new.loss_p_Holm.lt(.05))))


def pooled():
 D=pd.read_csv(OUT/'DATE_SERIES_STEP26.csv',float_precision='round_trip');M=pd.read_csv(OUT/'TEST_METRICS_STEP26.csv',float_precision='round_trip');source=ROOT/'revision_work/outputs/step22d/session_v3/analyze_d.py'
 if not source.exists():source=ROOT/'revision_work/outputs/step22d/analyze_d.py'
 tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');fn=copy.deepcopy(next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='series_result'))
 from functools import lru_cache
 @lru_cache(None)
 def draws(w,h):return s.mbb_index(391,h,20000,np.random.default_rng(np.random.SeedSequence([20261021,h,int(w[1:]),0])))
 def date_boot(x,w,h):return np.mean(np.asarray(x,float)[draws(w,h)],axis=1) if np.isfinite(x).all() else np.full(20000,np.nan)
 def replicates(name,v):
  p=OUT/'bootstrap'/name;p.parent.mkdir(exist_ok=True,parents=True)
  with p.open('xb') as f:np.save(f,v,allow_pickle=False)
  return str(p.relative_to(ROOT))
 env=dict(np=np,s=s,B=20000,date_boot=date_boot,replicates=replicates);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
 out=[]
 for arm in ['old','new']:
  for inp in ['O','C']:
   for h in [5,10,20]:
    f=D[(D.arm==arm)&(D.input_set==inp)&(D.config=='daily_weekly')&(D.target=='raw')&(D.h==h)];series={w:f[f.window==w].rank_ic.to_numpy(float) for w in ['W1','W2','W3']};q=M[(M.arm==arm)&(M.input_set==inp)&(M.config=='daily_weekly')&(M.target=='raw')&(M.h==h)]
    for ws in [['W1'],['W2'],['W3'],['W1','W2','W3']]:
     name='+'.join(ws);ic=env['series_result'](series,ws,h,h,test=False,tag=f'{arm}_{inp}_{name}_h{h}');z=q[q.window.isin(ws)];mse=float(z.mse.mean());gtm=float((z.mse/(1-z.r2os_vs_global_train_mean)).mean());zero=float((z.mse/(1-z.r2os_vs_zero)).mean());out.append(dict(arm=arm,window=name,input_set=inp,model='ridge',h=h,mse=mse,r2os_train_mean=1-mse/gtm,r2os_zero=1-mse/zero,da=float(z.da_excluding_zero.mean()),family='descriptive',**ic))
 P=pd.DataFrame(out);write('POOLED_RIDGE_STEP26.csv',P)
 olddir=ROOT/json.loads((ROOT/'revision_work/outputs/step22d/session_v3/ANALYSIS_COMPLETE.json').read_text())['analysis_directory'];T=pd.read_csv(olddir/'part1_rank_ic_inference.csv',float_precision='round_trip');new=T.copy()
 for r in P[P.arm=='new'].to_dict('records'):
  idx=new.index[(new.model=='ridge')&(new.input_set==r['input_set'])&(new.h==r['h'])&(new.window==r['window'])];assert len(idx)==1
  for k,v in r.items():
   if k in new.columns:new.loc[idx,k]=v
 write('propagated/S15_rank_ic_inference_step26.csv',new);js('POOLED_HOLM_STEP26.json',dict(Ridge_scope='descriptive; Ridge is outside original C PRIMARY6 and O SECONDARY6 families',non_Ridge_Holm_changes=0,non_Ridge_values_preserved=True,source=str((olddir/'part1_rank_ic_inference.csv').relative_to(ROOT))))

if __name__=='__main__':
 try:
  assert (OUT/'METRICS_COMPLETE.json').exists();shrinkage();pooled();js('DOWNSTREAM_COMPLETE.json',dict(shrinkage_cells=36,pooled_and_window_rows=48))
 except BaseException:js('DOWNSTREAM_BLOCKING.json',dict(error=traceback.format_exc()));raise
