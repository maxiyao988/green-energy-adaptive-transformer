"""Reuse original Step8 strategy loop for Ridge only; no non-Ridge model evaluations."""
from metrics26_v2 import *
import engine as e,types,inspect
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

def main():
 assert (OUT/'PREDICTION_COMPLETE.json').exists();source=Path(inspect.getsourcefile(e));tree=ast.parse(inspect.getsource(e.main));outer=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='((h, sp), ad)' and ast.unparse(n.iter)=='sorted(anchors.items())'))
 # Keep original allocation/simulation/cost code and original top3/top5/gamma grid.
 # Skip the non-Ridge benchmark metrics; their simulated EW book is required for Ridge alpha.
 for n in ast.walk(outer):
  if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='sname' and ast.unparse(n.iter)=='strategies':
   if any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='m' for t in x.targets) for x in n.body):
    n.body.insert(0,ast.parse("if not sname.startswith('ridge'): continue").body[0])
 code=compile(ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[])),str(source),'exec');C,Rdf=e.load_prices();cal=list(Rdf.index);pos={d:i for i,d in enumerate(cal)};R=Rdf.to_numpy();rf_s,nfill,nmissing=e.load_rf(cal);rf=rf_s.to_numpy();inv=pd.read_csv(OUT/'INVENTORY_STEP26.csv').fillna('');allm=[]
 for r in inv[(inv.window=='W3')&(inv.config=='daily_weekly')&(inv.target=='raw')].to_dict('records'):
  f=pd.read_csv(OUT/'cells'/r['cell_id']/'test_step26.csv',float_precision='round_trip');h=int(r['h']);ad=list(f.anchor_date.unique());name='ridge_s42';target=r['target'];inp=r['input_set']
  for arm,col in [('old','y_old'),('new','y_pred')]:
   P=pd.DataFrame(f[col].to_numpy(float).reshape(391,17),index=ad,columns=e.TICKERS);env=dict(vars(e));env.update(sets={(name,h,'test'):P},anchors={(h,'test'):ad},a=types.SimpleNamespace(models=None),cal=cal,pos=pos,R=R,rf=rf,gtm={},notes=[],blocking=[],qp_log=[],rp_rows=[],daily_rows=[],reb_rows=[],met_rows=[],ls_rows=[],w_rows=[],cum_rows=[],sims={})
   exec(code,env);assert not env['blocking'],env['blocking'];M=pd.DataFrame(env['met_rows']);assert M.strategy.eq(name).all();M['arm']=arm;M['input_set']=inp;M['target']=target;M['cell_id']=r['cell_id'];allm.append(M);prefix=f'portfolio/{r["cell_id"]}/{arm}'
   write(prefix+'_metrics_step26.csv',M);write(prefix+'_weights_step26.csv',pd.DataFrame(env['w_rows']).query("strategy.str.startswith('ridge')"));write(prefix+'_rebalances_step26.csv',pd.DataFrame(env['reb_rows']).query("strategy.str.startswith('ridge')"));write(prefix+'_daily_returns_step26.csv',pd.concat(env['daily_rows'],ignore_index=True));write(prefix+'_qp_log_step26.csv',pd.DataFrame(env['qp_log']));write(prefix+'_long_short_step26.csv',pd.DataFrame(env['ls_rows']))
   print('RIDGE_PORTFOLIO',r['cell_id'],arm,len(M),flush=True)
 P=pd.concat(allm,ignore_index=True);write('PORTFOLIO_RIDGE_STEP26.csv',P)
 old=pd.read_csv(ROOT/'revision_work/outputs/portfolio/original/metrics_all.csv',float_precision='round_trip');new=old.copy();q=P[(P.arm=='new')&(P.input_set=='O')&(P.target=='raw')]
 for r in q.to_dict('records'):
  idx=new.index[(new.split==r['split'])&(new.h==r['h'])&(new.strategy==r['strategy'])&(new.layer==r['layer'])&(new.cost_bp==r['cost_bp'])];assert len(idx)==1,(r['strategy'],r['h'],r['layer'],r['cost_bp'])
  for k,v in r.items():
   if k in new:new.loc[idx,k]=v
 write('propagated/S10_S11_price_level_full_step26.csv',new)
 counts=[]
 for arm,t in [('old',old),('new',new)]:
  z=t[(t.split=='test')&(t.cost_bp==10)&t.alpha_vs_ew_nw_t.notna()];counts.append(dict(arm=arm,n_combinations=len(z),n_abs_t_gt_1_96=int(z.alpha_vs_ew_nw_t.abs().gt(1.96).sum())))
 write('PRICE_LEVEL_480_ALPHA_COUNT_STEP26.csv',pd.DataFrame(counts));js('PORTFOLIO_COMPLETE.json',dict(rows=len(P),alpha_counts=counts,settings='original Step8 full top3/top5/gamma/cost grid, original EW comparison, lag5 alpha; no non-Ridge model evaluated',source=str(source.relative_to(ROOT))))
if __name__=='__main__':
 try:main()
 except BaseException:js('PORTFOLIO_BLOCKING_V3.json',dict(error=traceback.format_exc()));raise
