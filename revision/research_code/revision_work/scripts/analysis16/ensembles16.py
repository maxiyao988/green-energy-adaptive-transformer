"""Step7/7B ensemble procedures, using imported functions and unmodified source blocks."""
from statistics16_v2 import s
from statistics16 import *
from collections import Counter
import inspect,textwrap
# Source segment is imported Step7B's multiset bootstrap, stopping before test rows are appended.
import ensemble_robust as er
source=inspect.getsource(er.main);tree=ast.parse(textwrap.dedent(source));loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(spec, L)')
body=[]
for node in loop.body:
 if isinstance(node,ast.For) and ast.unparse(node.target)=='(metric, S_)':break
 body.append(node)
ECODE=compile(ast.Module(body=body,type_ignores=[]),inspect.getsourcefile(er),'exec');ESOURCE='\n'.join(ast.get_source_segment(source,n) for n in body)
def interval(x):
 d=series_stats(x,float(np.mean(x)));return {k:v for k,v in d.items() if k not in ['centered_p_two_sided','ci_excludes_zero','theta_hat']}
def run():
 if verified('F.COMPLETE.json'):return
 a=Analysis();loo=[];ci=[];tests=[];files=[];put('ensemble_source_reused.py.txt',ESOURCE)
 for inp,models in [('C',['single','fixed','static','adaptive','lstm','vanilla','swim']),('O',['lstm','vanilla','swim'])]:
  for h in HS:
   ref=a.test('W3',h);y=ref.y_true.to_numpy(float);Y=y.reshape(-1,17);gtm=a.refs('W3',h)[1];order=sorted(SEEDS)
   for model in models:
    group='neural' if model in ['lstm','vanilla','swim'] else 'core';g=a.group(group,inp,model,h);PA=a.arrays(g);perfect=np.broadcast_to(y,PA.shape);full=None
    for omit in [None]+order:
     w=np.full(5,.2) if omit is None else np.array([0. if seed==omit else .25 for seed in order]);ls,_,pa,_=ens_series(y,Y.shape,PA,perfect,Y,w);met,ic=ensemble_metrics(y,pa,Y,pa.reshape(Y.shape),gtm,h)
     row=dict(input_set=inp,model=model,h=h,held_out_seed='none' if omit is None else str(omit),mse=met['mse'],mean_rank_ic=met['mean_rank_ic'],r2os_vs_gtm=met['r2os_vs_gtm'],mse_ts=met['mse_ts'],mse_xs=met['mse_xs']);loo.append(row)
     if omit is None:
      full=row;files.append(pq(f'F_ensembles/{inp}_{model}_h{h}_test.parquet',ref.assign(y_pred=pa)))
    # The second arm is y itself, so the existing loss-difference helper returns absolute squared loss;
    # its IC equals1 on every nonconstant label cross-section. No reference-model test is added.
    env=dict(np=np,Counter=Counter,ens_series=ens_series,rank_ic_rows=rank_ic_rows,mbb_index=mbb_index,BOOT_SEED=er.BOOT_SEED,h=h,L=h,k=5,B=B,T=len(Y),y=y,Y=Y,PA=PA,PS=perfect)
    exec(ECODE,env);did=env['didx'];LS=env['Ls'];IS=env['Is']+1
    for metric,vals,theta in [('mse',LS,full['mse']),('rank_ic',IS,full['mean_rank_ic'])]:
     bb=np.nanmean(np.take_along_axis(vals,did,axis=1),axis=1);ci.append(dict(input_set=inp,model=model,h=h,metric=metric,theta_hat=theta,B=B,L=h,n_seeds=5,rng=f'SeedSequence([{er.BOOT_SEED},{h},{h}])',**interval(bb)))
    print('ENSEMBLE',inp,model,h,flush=True)
   if inp=='C':
    PA=a.arrays(a.group('core','C','adaptive',h));PS=a.arrays(a.group('core','C','static',h));w=np.full(5,.2);tl,ti,_,_=ens_series(y,Y.shape,PA,PS,Y,w)
    env=dict(np=np,Counter=Counter,ens_series=ens_series,rank_ic_rows=rank_ic_rows,mbb_index=mbb_index,BOOT_SEED=er.BOOT_SEED,h=h,L=h,k=5,B=B,T=len(Y),y=y,Y=Y,PA=PA,PS=PS);exec(ECODE,env)
    for metric,arr,theta in [('squared_loss',env['Ls'],float(tl.mean())),('rank_ic',env['Is'],float(np.nanmean(ti)))]:
     bj=np.nanmean(np.take_along_axis(arr,env['didx'],axis=1),axis=1);ts=tl if metric=='squared_loss' else ti;bd=np.nanmean(ts[dates(h)],axis=1);v=[]
     for i in range(5):
      delta=ens_series(y,Y.shape,PA,PS,Y,np.eye(5)[i])[0 if metric=='squared_loss' else 1];v.append(float(np.nanmean(delta)))
     for method,bb in [('seed_resampled_ensemble',bj),('date_only_ensemble',bd)]:tests.append(dict(input_set='C',contrast='adaptive−static',h=h,metric=metric,method=method,B=B,L=h,n_seeds=5,seed_wins=int((np.array(v)>0 if metric=='rank_ic' else np.array(v)<0).sum()),seed_delta_mean=float(np.mean(v)),seed_delta_sd=float(np.std(v,ddof=1)),**series_stats(bb,theta)))
 T=corrected(pd.DataFrame(tests),['input_set','contrast','metric','method']);files += [csv('F_ensemble_leave_one_out.csv',pd.DataFrame(loo)),csv('F_seed_resampled_intervals.csv',pd.DataFrame(ci)),csv('F_C_adaptive_static_tests.csv',T)]
 complete('F.COMPLETE.json',files,dict(tests=len(T),seed_resampling_rng=er.BOOT_SEED))
if __name__=='__main__':run()
