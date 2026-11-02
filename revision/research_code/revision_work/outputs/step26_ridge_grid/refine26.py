"""Complete source-consistent rolling CW, CS portfolio, and table propagation inputs."""
from metrics26_v2 import *
import engine,types

def main():
 assert (OUT/'METRICS_COMPLETE.json').exists();M=pd.read_csv(OUT/'TEST_METRICS_STEP26.csv',float_precision='round_trip');I=pd.read_csv(OUT/'INVENTORY_STEP26.csv').fillna('');a=s.Analysis()
 # Rolling CW: exact original Step16 cell_metrics, h-1 NW, common original date draws, original 3h Holm.
 old=pd.read_csv(A16/'C_rolling_CW.csv',float_precision='round_trip');new=old.copy();rr=[]
 for r in I[(I.target=='raw')&(I.config=='daily_weekly')].to_dict('records'):
  f=frame(r);y=f.y_true.to_numpy(float);p=f.y_pred.to_numpy(float);Y=y.reshape(391,17);refs,gtm=a.refs(r['window'],int(r['h']))
  for ref in ['global_train_mean','zero']:
   br=refs[ref];rrefs=dict(refs);rrefs['global_train_mean']=br;z,_,_=s.cell_metrics(y,p,rrefs,int(r['h']),s.dates(int(r['h']),391,2),Y,p.reshape(Y.shape),float(br[0]));idx=new.index[(new.model=='ridge')&(new.window==r['window'])&(new.input_set==r['input_set'])&(new.h==r['h'])&(new.reference==ref)];assert len(idx)==1
   v=dict(cw_mean=z['cw_mean'],cw_nw_t=z['cw_nw_t'],p_nw=z['cw_p_nw_one_sided'],p_mbb=z['cw_p_mbb_one_sided'])
   for k,x in v.items():new.loc[idx,k]=x
   if 'cw_mean_across_seeds' in new:new.loc[idx,'cw_mean_across_seeds']=v['cw_mean']
   if 'seed_wins' in new:new.loc[idx,'seed_wins']=int(v['cw_mean']>0)
 for p in ['p_nw','p_mbb']:
  for _,g in new[new.model=='ridge'].groupby(['window','input_set','model','seed','reference']):
   for i,v in s.holm({i:float(v) for i,v in g[p].items() if np.isfinite(v)}).items():new.loc[i,p+'_holm_3h']=v
 write('propagated/C_rolling_CW_final_step26.csv',new)
 # Correct deterministic Ridge summary fields (no non-Ridge raw values changed).
 b=pd.read_csv(OUT/'propagated/B_forecast_skill_final_step26.csv',float_precision='round_trip')
 for i,r in b[b.model=='ridge'].iterrows():
  b.loc[i,'mse_mean']=r.mse;b.loc[i,'rank_ic_mean']=r.mean_rank_ic
 write('propagated/B_forecast_skill_final_complete_step26.csv',b)
 # Preserve every existing subgroup row; replace Ridge only by source-extracted computation.
 for kind in ['per_seed','summary']:
  old=pd.read_csv(A16/f'clarified/C_subgroups_{kind}.csv',float_precision='round_trip');fresh=pd.read_csv(OUT/f'propagated/C_subgroups_{kind}_ridge_step26.csv',float_precision='round_trip');keys=['window','input_set','model','h','subgroup_type','subgroup']+(['seed'] if kind=='per_seed' else []);new=old.copy()
  for r in fresh.to_dict('records'):
   mask=np.ones(len(new),bool)
   for k in keys:mask&=new[k].eq(r[k]).to_numpy()
   assert mask.sum()==1
   for k,v in r.items():
    if k in new:new.loc[mask,k]=v
  write(f'propagated/C_subgroups_{kind}_full_step26.csv',new)
 # Refresh Step17 rolling CW references from the recomputed original 3h families.
 sh=pd.read_csv(OUT/'propagated/SHRINKAGE_full_family_step26.csv',float_precision='round_trip');Q=pd.read_csv(OUT/'propagated/C_rolling_CW_final_step26.csv',float_precision='round_trip');curves=pd.read_csv(OUT/'SHRINKAGE_VAL_CURVES_STEP26.csv',float_precision='round_trip')
 for r in I[(I.target=='raw')&(I.config=='daily_weekly')].to_dict('records'):
  idx=sh.index[(sh.model=='ridge')&(sh.window==r['window'])&(sh.input_set==r['input_set'])&(sh.h==r['h'])]
  for i in idx:
   q=curves[(curves.cell_id==r['cell_id'])&np.isclose(curves.lambda_value,sh.loc[i,'lambda_used'])];assert len(q)==1;sh.loc[i,'val_mse_at_used_lambda']=q.iloc[0].val_mse
   sh.loc[i,'validation_curve_file']='SHRINKAGE_VAL_CURVES_STEP26.csv';sh.loc[i,'validation_curve_filter']=r['cell_id']
   if r['window']=='W3':continue
   q=Q[(Q.model=='ridge')&(Q.window==r['window'])&(Q.input_set==r['input_set'])&(Q.h==r['h'])&(Q.reference=='global_train_mean')].iloc[0]
   for dest,col in [('cw_p_nw_raw','p_nw'),('cw_p_mbb_raw','p_mbb'),('cw_p_nw_holm_Step16','p_nw_holm_3h'),('cw_p_mbb_holm_Step16','p_mbb_holm_3h')]:sh.loc[i,dest]=q[col] if sh.loc[i,'lambda_used']!=0 else np.nan
 write('propagated/SHRINKAGE_final_step26.csv',sh)
 # CS top5/long-short portfolios: original Step22G loop copied intact; Ridge only, old and new.
 source=ROOT/'revision_work/outputs/step22g/analyze_g_v3.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');loop=copy.deepcopy(next(n for n in fn.body if isinstance(n,ast.For) and ast.unparse(n.target)=='((arm, h), f)'))
 C,Rdf=engine.load_prices();cal=list(Rdf.index);pos={d:i for i,d in enumerate(cal)};R=Rdf.to_numpy();rf,filled,missing=engine.load_rf(cal);ensembles={}
 for r in I[I.target=='cross_sectional'].to_dict('records'):
  f=frame(r)
  for arm,col in [('old','y_old'),('new','y_pred')]:ensembles['Ridge-CS_'+arm,int(r['h'])]=f[['ticker','anchor_date','y_true']].assign(y_pred=f[col])
 env=dict(np=np,pd=pd,engine=engine,cal=cal,pos=pos,R=R,rf=rf,ensemble_frames=ensembles,portfolio=[],dailyrows=[],weightrows=[],rebalances=[]);exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(source),'exec'),env)
 for name,key in [('CS_PORTFOLIO_METRICS_STEP26.csv','portfolio'),('CS_PORTFOLIO_DAILY_STEP26.csv','dailyrows'),('CS_PORTFOLIO_WEIGHTS_STEP26.csv','weightrows'),('CS_PORTFOLIO_REBALANCES_STEP26.csv','rebalances')]:write(name,pd.DataFrame(env[key]))
 js('REFINEMENT_COMPLETE.json',dict(rolling_CW_replaced_Ridge_cells=36,CS_portfolio_rows=len(env['portfolio']),non_Ridge_raw_metrics_recomputed=0))
if __name__=='__main__':
 try:main()
 except BaseException:js('REFINEMENT_BLOCKING.json',dict(error=traceback.format_exc()));raise
