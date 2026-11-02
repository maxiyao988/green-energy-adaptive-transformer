"""Primary-source separation and pooled rolling CW using imported Step5 f-statistics."""
from statistics16_v2 import s
from statistics16 import *
import a2_skill
# Existing centered one-sided CW p expression, reused without changing sidedness or +1 correction.
tree=ast.parse(inspect_source := __import__('inspect').getsource(a2_skill.cell_metrics))
expr=next(k.value for n in ast.walk(tree) if isinstance(n,ast.Call) for k in n.keywords if k.arg=='cw_p_mbb_one_sided')
PCODE=compile(ast.Expression(body=expr),__import__('inspect').getsourcefile(a2_skill),'eval')
def run():
 if verified('reconciliation.COMPLETE.json'):return
 B0=pd.read_csv(OUT/'B_forecast_skill.csv');arch=B0.input_set.eq('O')&B0.group.eq('single_resolution')&B0.model.isin(['lstm','vanilla','swim']);P=B0[~arch].copy();R=B0[arch].copy()
 for p in ['cw_p_nw_one_sided','cw_p_mbb_one_sided','rank_ic_p_mbb']:
  P[p+'_holm_all_cells']=np.nan
  for ix,v in holm({ix:float(v) for ix,v in P[p].items() if np.isfinite(v)}).items():P.loc[ix,p+'_holm_all_cells']=v
 flags=[]
 for r in P.itertuples():
  for ref in ['global_train_mean','zero','ticker_train_mean','expanding_pooled_mean']:
   val=getattr(r,'r2os_vs_'+ref)
   if val>0:flags.append(dict(id=r.id,input_set=r.input_set,model=r.model,config=r.config,h=r.h,seed=r.seed,metric='r2os_vs_'+ref,value=val,raw_p=r.cw_p_mbb_one_sided,p_holm_all_cells=r.cw_p_mbb_one_sided_holm_all_cells,p_scope='CW vs GTM only; R2 vs other references is descriptive'))
  if np.isfinite(r.rank_ic_p_mbb) and r.rank_ic_p_mbb<.05:flags.append(dict(id=r.id,input_set=r.input_set,model=r.model,config=r.config,h=r.h,seed=r.seed,metric='rank_ic',value=r.mean_rank_ic,raw_p=r.rank_ic_p_mbb,p_holm_all_cells=r.rank_ic_p_mbb_holm_all_cells,p_scope='two-sided MBB zero-IC test'))
 files=[csv('B_PRIMARY_forecast_skill.csv',P),csv('B_PRIMARY_flagged_cells.csv',pd.DataFrame(flags)),csv('B_original_submission_reference_single_resolution.csv',R)]
 # Baseline source references are never pooled into the uniform12 primary neural family.
 scope=P.groupby(['group','input_set','model','config','h'],as_index=False).agg(n_seeds=('seed','nunique'),mse_mean=('mse','mean'),mse_sd=('mse','std'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'))
 files.append(csv('B_PRIMARY_seed_summary.csv',scope));a=Analysis();out=[]
 for w in ['W1','W2','W3']:
  for inp in ['O','C']:
   for h in HS:
    d=a.test(w,h);y=d.y_true.to_numpy(float);Y=y.reshape(-1,17);refs,gtm=a.refs(w,h)
    for model in ['static','adaptive']:
     g=a.group('rolling' if w!='W3' else 'core',inp,model,h,window=w)
     for refname in ['global_train_mean','zero']:
      frows=[]
      for seed in sorted(SEEDS):
       p=a.pred(g[seed]);rr=dict(refs);rr['global_train_mean']=refs[refname];_,f,_=cell_metrics(y,p,rr,h,dates(h,391,2),Y,p.reshape(Y.shape),float(refs[refname][0]));frows.append(f)
      X=np.stack(frows);theta,bj,bd=distributions(X,h);values=X.mean(1);meta=dict(window=w,input_set=inp,model=model,h=h,reference=refname,n_seeds=5,seed_wins=int((values>0).sum()),seed_CW_mean=float(values.mean()),seed_CW_sd=float(values.std(ddof=1)),B=B,L=h)
      m,se,t,n=nw_t(X.mean(0),h-1);out.append(dict(**meta,method='Newey_West',theta_hat=m,cw_nw_t=t,p_value=float(stats.norm.sf(t))))
      for method,boot in [('joint_seed_date',bj),('date_only',bd)]:out.append(dict(**meta,method=method,theta_hat=theta,p_value=eval(PCODE,dict(np=np,boot=boot,m=theta)),**{k:v for k,v in series_stats(boot,theta).items() if k not in ['theta_hat','centered_p_two_sided','ci_excludes_zero']}))
 Q=corrected(pd.DataFrame(out),['window','input_set','model','reference','method'],p='p_value',target='p_holm_3h');files.append(csv('C_rolling_pooled_CW.csv',Q))
 files.append(js('scope_decisions.json',dict(unanswered_optional_questions_default='recommended choices: R2 descriptive + CW-vs-GTM p only; n=1 single-resolution; F only C Adaptive−Static contrast; stock date-level IC undefined',primary_skill_cells=len(P),original_submission_reference_single_resolution_cells=len(R),reason_for_reference_separation='O neural DO/WO have only archived seed42 and no B5FIX uniform12 replacement; excluded from primary B Holm family. Raw requested diagnostics remain separately labelled reference, never mixed with primary.',B_initial_raw_inventory='B_forecast_skill.csv contains all available single-resolution inventory; use B_PRIMARY* for primary claims',single_seed_cells='n=1, ddof1 SD undefined, no invented seeds; Ridge deterministic n=1')))
 complete('reconciliation.COMPLETE.json',files,dict(new_tests=len(Q),reference_tests_already_counted_in_B=int(R[['cw_p_nw_one_sided','cw_p_mbb_one_sided','rank_ic_p_mbb']].notna().sum().sum())))
if __name__=='__main__':run()
