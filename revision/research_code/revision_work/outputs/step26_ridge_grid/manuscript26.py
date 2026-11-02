"""Step18/19 table exporter and exact Step19 Figure3 block, rebased to Step26 results."""
from metrics26_v2 import *
import csv,math,shutil
DEST=OUT/'manuscript';DEST.mkdir()
source=ROOT/'revision_work/outputs/manuscript19/prepare_step19_v2.py';tree=ast.parse(source.read_text());env=dict(csv=csv,math=math,OUT=DEST,artifacts=[])
for name in ['num','fmt','one','export_csv']:
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
def export(name,df):env['export_csv'](name+'_step26.csv',df.to_dict('records') if isinstance(df,pd.DataFrame) else df)
def read(p):return pd.read_csv(p,float_precision='round_trip',keep_default_na=False)
def val(x):
 try:return float(x)
 except:return np.nan

def main():
 M=pd.read_csv(OUT/'TEST_METRICS_STEP26.csv',float_precision='round_trip');S=pd.read_csv(OUT/'SELECTION_LOCK.csv',float_precision='round_trip');N=M[M.arm=='new'];B=read(OUT/'propagated/B_forecast_skill_final_complete_step26.csv');summary=read(A16/'clarified/B_seed_summary.csv')
 for i,r in summary[summary.model=='ridge'].iterrows():
  q=N[(N.window=='W3')&(N.target=='raw')&(N.input_set==r.input_set)&(N.config==r.config)&(N.h==int(r.h))].iloc[0];summary.loc[i,'mse_mean']=q.mse;summary.loc[i,'rank_ic_mean']=q.mean_rank_ic
 # Main Table4: original saved summaries, deterministic Ridge fields replaced only.
 modelgroup={'single':'core','fixed':'core','same_patch_fixed':'samepatch','static':'core','adaptive':'core','lstm':'neural','vanilla':'neural','swim':'neural','ridge':'ridge'};t4=[]
 for inp in ['O','C']:
  for h in [5,10,20]:
   for model,group in modelgroup.items():
    q=summary[(summary.input_set==inp)&(summary.h.astype(int)==h)&(summary.model==model)&(summary.group==group)&(summary.config=='daily_weekly')];assert len(q)==1;t4.append(q.iloc[0].to_dict())
 export('Table4',pd.DataFrame(t4));export('Table2_Ridge_penalties',S)
 # S3 financial references including all Ridge configurations.
 t=read(ROOT/'revision_work/outputs/manuscript19/S3_table.csv')
 for i,r in t[t.model=='ridge'].iterrows():
  q=N[(N.window=='W3')&(N.target=='raw')&(N.input_set==r.input_set)&(N.config==r.configuration)&(N.h==int(r.horizon))].iloc[0]
  for col,src in [('test_mse','mse'),('r2_os_vs_train_mean','r2os_vs_global_train_mean'),('cw_p_nw_vs_train_mean','cw_p_nw_one_sided'),('mean_rank_ic','mean_rank_ic'),('rank_ic_nw_t','rank_ic_nw_t')]:t.loc[i,col]=q[src]
 export('S3',t);export('S4',B[(B.group=='single_resolution')|((B.model=='ridge')&(B.config!='daily_weekly'))])
 # S8 retains original record types and all non-Ridge records.
 t=read(ROOT/'revision_work/outputs/manuscript19/S8_table.csv');CW=read(OUT/'propagated/C_rolling_CW_final_step26.csv')
 for i,r in t[t.model=='ridge'].iterrows():
  h=int(val(r.horizon)) if np.isfinite(val(r.horizon)) else int(r.h)
  if r.record_type=='five_seed_rolling_metrics':
   q=N[(N.window==r.window)&(N.target=='raw')&(N.input_set==r.input_set)&(N.config=='daily_weekly')&(N.h==h)].iloc[0];t.loc[i,'mse_ratio_mean']=q.relative_mse_train_mean
  else:
   q=CW[(CW.window==r.window)&(CW.model=='ridge')&(CW.input_set==r.input_set)&(CW.h.astype(int)==h)&(CW.reference==r.reference)].iloc[0]
   for k in q.index:
    if k in t and k not in ['record_type','source_file','source_csv_line']:t.loc[i,k]=q[k]
 export('S8',t);export('S9',read(OUT/'propagated/C_subgroups_summary_full_step26.csv'))
 P=pd.read_csv(OUT/'PORTFOLIO_RIDGE_STEP26.csv',float_precision='round_trip');PN=P[(P.arm=='new')&(P.target=='raw')]
 for name in ['S10','S11']:
  t=read(ROOT/f'revision_work/outputs/manuscript19/{name}_table.csv')
  for i,r in t[t.strategy.astype(str).str.startswith('ridge')].iterrows():
   q=PN[(PN.input_set==r.input_set)&(PN.h==int(r.h))&(PN.layer==r.layer)&(PN.cost_bp==int(r.cost_bp))];assert len(q)==1
   for k,v in q.iloc[0].items():
    if k in t and k not in ['strategy','input_set']:t.loc[i,k]=v
  extra=PN[PN.input_set=='O'].copy();extra['source_file']='PORTFOLIO_RIDGE_STEP26.csv';extra['provenance']='Step26 Ridge O; original Step8 strategy grid; added because original S10/S11 source omitted O Ridge'
  export(name,pd.concat([t,extra],ignore_index=True))
 export('S12',read(OUT/'propagated/SHRINKAGE_final_step26.csv'))
 # All Ridge DA cells; source Step21 conventions exclude exactly-zero realized returns.
 da=read(ROOT/'revision_work/outputs/step21/all_da_per_seed.csv')
 export('S13_Ridge',N[['cell_id','window','input_set','config','h','target','da_excluding_zero','n_nonzero_target']]);export('S15',read(OUT/'propagated/S15_rank_ic_inference_step26.csv'));export('S15_Ridge_metrics',read(OUT/'POOLED_RIDGE_STEP26.csv'));export('S18_Ridge',N[N.target=='cross_sectional']);export('S18_Ridge_portfolios',read(OUT/'CS_PORTFOLIO_METRICS_STEP26.csv'))
 # Figure3 rebuild exact full-precision original Step19 ratio construction (no non-Ridge re-evaluation).
 fn=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='f3' for t in n.targets));end=next(i for i,n in enumerate(tree.body[fn+1:],fn+1) if isinstance(n,ast.Assert));block=copy.deepcopy(tree.body[fn:end+1]);ratioenv=dict(env,seed_summary=read(A16/'clarified/B_seed_summary.csv').to_dict('records'),gtm=read(ROOT/'revision_work/outputs/skill_analysis/a1/a1_benchmark_mse.csv').to_dict('records'),MODELS=tuple(modelgroup),M18='',A16='revision_work/outputs/analysis16/')
 # Source annotations required by original selector; original source line numbers preserved for audit.
 for name in ['seed_summary','gtm']:
  for i,r in enumerate(ratioenv[name],2):r['_line']=i;r['_source']='original Step19 source'
 exec(compile(ast.fix_missing_locations(ast.Module(body=block,type_ignores=[])),str(source),'exec'),ratioenv);f3=ratioenv['f3']
 for r in f3:
  if r['model']=='ridge':
   q=N[(N.window=='W3')&(N.target=='raw')&(N.input_set==r['input_set'])&(N.config=='daily_weekly')&(N.h==r['horizon'])].iloc[0];r.update(test_mse_mean=q.mse,mse_ratio=q.relative_mse_train_mean,summary_source='Step26 selected Ridge',numerical_provenance='Step26 original deterministic Ridge path; W3 O archived original-grid TEST replay DIFFERENT')
 export('Figure3_data',f3)
 # Use existing valid font cache via exclusive copy; prevent global cache writes.
 cache=OUT/'mpl_step26';cache.mkdir();oldcache=ROOT/'revision_work/outputs/manuscript19/matplotlib_cache_prepare_step19_v2/fontlist-v330.json'
 with (cache/'fontlist-v330.json').open('xb') as f:f.write(oldcache.read_bytes())
 os.environ['MPLCONFIGDIR']=str(cache)
 start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Import) and any(x.name=='matplotlib' for x in n.names));stop=next(i for i,n in enumerate(tree.body[start:],start) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='save' and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value=='F3_complete')
 body=copy.deepcopy(tree.body[start:stop+1])
 for n in ast.walk(ast.Module(body=body,type_ignores=[])):
  if isinstance(n,ast.Constant) and n.value=='F3_complete':n.value='Figure3_step26'
 env.update(f3=f3,MODELS=tuple(modelgroup),OUT=DEST,artifacts=[],figdata=[])
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source),'exec'),env)
 js('MANUSCRIPT_ARTIFACTS_COMPLETE.json',dict(table_exporter='original Step19 export_csv',figure_code='original Step19 complete Figure3 block',artifacts=sorted(str(p.relative_to(ROOT)) for p in DEST.iterdir()),font=env['font']))
if __name__=='__main__':
 try:main()
 except BaseException:js('MANUSCRIPT_BLOCKING.json',dict(error=traceback.format_exc()));raise
