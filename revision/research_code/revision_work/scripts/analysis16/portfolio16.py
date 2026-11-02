"""Step8 engine execution, retaining only requested strategies/statistics."""
from statistics16_v2 import s
from statistics16 import *
import engine as e
import inspect,copy
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
# Execute original strategy construction, optimizer, drift, cost and metric loop by import/source adaptation.
# Remove top3 and its long/short t test; retain top5, all original gamma settings and allocation benchmarks.
tree=ast.parse(inspect.getsource(e.main));outer=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(h, sp)' and 'sorted(anchors.items())'==ast.unparse(n.iter))
outer=copy.deepcopy(outer)
for node in ast.walk(outer):
 if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='k' and isinstance(node.iter,ast.Tuple) and ast.unparse(node.iter)=='(3, 5)':node.iter.elts=[ast.Constant(value=5)]
 if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='name' and ast.unparse(node.iter)=='names':
  new=[]
  for q in node.body:
   if isinstance(q,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='spread' for t in q.targets):break
   new.append(q)
  node.body=new
CODE=compile(ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[])),inspect.getsourcefile(e),'exec')
def no_alpha(net,per_net,to,rf,h,bench_ex=None):return e.metrics(net,per_net,to,rf,h,None)
def run():
 if verified('G.COMPLETE.json'):return
 a=Analysis();C,Rdf=e.load_prices();cal=list(Rdf.index);pos={d:i for i,d in enumerate(cal)};R=Rdf.to_numpy();rf_s,nfill,nmissing=e.load_rf(cal);rf=rf_s.to_numpy();files=[];allmetrics=[];grsrows=[];problems=[]
 for inp in ['C','O']:
  for h in HS:
   mark=f'G/{inp}_h{h}/COMPLETE.json'
   if verified(mark):
    allmetrics.append(pd.read_csv(OUT/f'G/{inp}_h{h}/metrics.csv'));grsrows.extend(json.loads((OUT/f'G/{inp}_h{h}/grs.json').read_text()));continue
   ref=a.test('W3',h);ad=sorted(ref.anchor_date.unique());sets={};models=['single','fixed','static','adaptive','lstm','vanilla','swim'] if inp=='C' else ['lstm','vanilla','swim']
   for model in models:
    g=a.group('neural' if model in ['lstm','vanilla','swim'] else 'core',inp,model,h);mat=a.arrays(g)
    for seed,id in g.items():sets[(model+'_s'+str(seed),h,'test')]=pd.DataFrame(a.pred(id).reshape(391,17),index=ad,columns=e.TICKERS)
    _,_,pa,_=ens_series(ref.y_true.to_numpy(float),(391,17),mat,mat,ref.y_true.to_numpy(float).reshape(391,17),np.full(5,.2));sets[(model+'_ens',h,'test')]=pd.DataFrame(pa.reshape(391,17),index=ad,columns=e.TICKERS)
   if inp=='C':
    g=a.group('ridge',inp,'ridge',h);sets[('ridge',h,'test')]=pd.DataFrame(a.pred(next(iter(g.values()))).reshape(391,17),index=ad,columns=e.TICKERS)
   gtm={h:a.refs('W3',h)[1]};sets[('gtm',h,'test')]=pd.DataFrame(gtm[h],index=ad,columns=e.TICKERS)
   env=dict(vars(e));env.update(sets=sets,anchors={(h,'test'):ad},a=types.SimpleNamespace(models=None),cal=cal,pos=pos,R=R,rf=rf,gtm=gtm,notes=[],blocking=[],qp_log=[],rp_rows=[],daily_rows=[],reb_rows=[],met_rows=[],ls_rows=[],w_rows=[],cum_rows=[],sims={},metrics=no_alpha)
   exec(CODE,env);assert not env['blocking'],env['blocking'];M=pd.DataFrame(env['met_rows']);M['input_set']=inp;M['provenance']='Step16 imported Step8 engine, uniform12 predictions';allmetrics.append(M)
   gr=[];sims=env['sims'];days,gg,ex,to,per=sims['test',h,'EW'];factor=e.apply_costs(days,gg,ex,to,per,e.COST0/1e4)[0]-rf[days]
   for label,members in [('all_seeds',[k[0] for k in sets if '_s' in k[0]]+(['ridge'] if inp=='C' else [])),('ensembles',[m+'_ens' for m in models]+(['ridge'] if inp=='C' else []))]:
    # Explicit grouping matches Step8's all-seed and ensemble GRS families; gamma5/cost10 only.
    excess=[]
    for member in sorted(members):
     d,g,x,t,p=sims['test',h,member+'|L2_g5'];excess.append(e.apply_costs(d,g,x,t,p,.001)[0]-rf[d])
    E=np.column_stack(excess)
    try:
     stat,pv,T,N=e.grs(E,factor)
     # GRS requires a nonsingular residual covariance; do not substitute an inverse or regularizer.
     design=np.column_stack([np.ones(len(factor)),factor]);res=E-design@np.linalg.lstsq(design,E,rcond=None)[0];rank=int(np.linalg.matrix_rank(res));assert rank==E.shape[1],f'residual rank {rank} < {E.shape[1]}'
     if stat<0 or not np.isfinite(stat):raise ArithmeticError('invalid GRS statistic')
     row=dict(input_set=inp,h=h,group=label,N=N,T=T,GRS_F=stat,p_value=pv,cost_bp=10,layer='L2_g5',status='computed')
    except (np.linalg.LinAlgError,AssertionError,ArithmeticError) as err:
     row=dict(input_set=inp,h=h,group=label,N=E.shape[1],T=len(E),GRS_F=None,p_value=None,cost_bp=10,layer='L2_g5',status='UNDEFINED: '+str(err));problems.append(row)
    gr.append(row)
   grsrows+=gr
   ff=[csv(f'G/{inp}_h{h}/metrics.csv',M),js(f'G/{inp}_h{h}/grs.json',gr),csv(f'G/{inp}_h{h}/weights.csv',pd.DataFrame(env['w_rows'])),csv(f'G/{inp}_h{h}/rebalances.csv',pd.DataFrame(env['reb_rows'])),csv(f'G/{inp}_h{h}/daily_returns.csv',pd.concat(env['daily_rows'],ignore_index=True)),csv(f'G/{inp}_h{h}/risk_parity_check.csv',pd.DataFrame(env['rp_rows'])),csv(f'G/{inp}_h{h}/qp_solver_log.csv',pd.DataFrame(env['qp_log'],columns=['split','h','strategy','date','message','action']))]
   complete(mark,ff,dict(n_models=len(sets),n_strategies=len(M)//4,missing_rf=nmissing,rf_ffill=nfill));print('PORTFOLIO',inp,h,len(M),flush=True)
 primary=pd.concat(allmetrics,ignore_index=True);old=pd.read_csv(ROOT/'revision_work/outputs/portfolio/original_rp/metrics_all.csv');old=old[(old.split=='test')&(old.layer.isin(['allocation benchmark','L1_top5']+['L2_g'+str(g) for g in e.GAMMAS]))].copy();old['input_set']='O';old['provenance']='existing Step8 O reference; not recomputed; no mixing with uniform12 neural set'
 old=old.drop(columns=[c for c in ['alpha_vs_ew_nw_t','alpha_vs_ew_ann','beta_vs_ew'] if c in old])
 gr=pd.DataFrame(grsrows);gr=corrected(gr,['input_set','group'],p='p_value',target='p_holm_3h')
 files=[csv('G_portfolio_primary.csv',primary),csv('G_portfolio_with_existing_O.csv',pd.concat([primary,old],ignore_index=True)),csv('G_GRS.csv',gr),js('G_settings.json',dict(execution='anchor t -> close t+1, hold h sessions; nonoverlapping, Step8 simulation/drift/cost code',lookback=252,covariance='LedoitWolf times h',gamma=e.GAMMAS,cap=e.WMAX,costs=e.COSTS_BP,grs_groups='all_seeds and ensembles per input/h; includes Ridge for C',no_new_alpha_or_longshort_tests=True,undefined_grs=gr[gr.status.ne('computed')].to_dict('records')))]
 complete('G.COMPLETE.json',files,dict(tests=int(gr.p_value.notna().sum())))
if __name__=='__main__':run()
