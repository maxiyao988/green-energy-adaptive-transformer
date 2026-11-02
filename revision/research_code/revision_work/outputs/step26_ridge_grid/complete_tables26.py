"""Full affected DA/CS source views and explicit old-new downstream audit tables."""
from metrics26_v2 import *
def compare(old,new,keys,fields,name):
 old=old.copy();new=new.copy();keep=[c for c in fields if c in old and c in new];z=old[keys+keep].merge(new[keys+keep],on=keys,validate='one_to_one',suffixes=('_old','_new'));write(name,z);return z

def main():
 M=pd.read_csv(OUT/'TEST_METRICS_STEP26.csv',float_precision='round_trip');N=M[M.arm=='new'];I=pd.read_csv(OUT/'INVENTORY_STEP26.csv')
 compare(M[M.arm=='old'],N,['cell_id','window','input_set','config','h','target'],[c for c in M if c not in ['cell_id','window','input_set','config','h','target','arm']],'OLD_NEW_TEST_METRICS_STEP26.csv')
 for kind in ['summary','per_seed']:
  old=pd.read_csv(A16/f'clarified/C_subgroups_{kind}.csv',float_precision='round_trip');new=pd.read_csv(OUT/f'propagated/C_subgroups_{kind}_full_step26.csv',float_precision='round_trip');keys=['window','input_set','model','h','subgroup_type','subgroup']+(['seed'] if kind=='per_seed' else []);compare(old[old.model=='ridge'],new[new.model=='ridge'],keys,[c for c in new if c not in keys],f'OLD_NEW_SUBGROUPS_{kind.upper()}_STEP26.csv')
 old=pd.read_csv(A16/'C_rolling_CW.csv',float_precision='round_trip');new=pd.read_csv(OUT/'propagated/C_rolling_CW_final_step26.csv',float_precision='round_trip');compare(old[old.model=='ridge'],new[new.model=='ridge'],['window','input_set','model','h','seed','reference'],['cw_mean','cw_nw_t','p_nw','p_mbb','p_nw_holm_3h','p_mbb_holm_3h'],'OLD_NEW_ROLLING_CW_STEP26.csv')
 old=pd.read_csv(ROOT/'revision_work/outputs/analysis17/test_results.csv',float_precision='round_trip');new=pd.read_csv(OUT/'propagated/SHRINKAGE_final_step26.csv',float_precision='round_trip');compare(old[old.model=='ridge'],new[new.model=='ridge'],['regime','cell_id'],['lambda_used','test_mse','r2os_vs_global_train_mean','r2os_vs_zero','mean_rank_ic','loss_difference','loss_p_raw','loss_p_Holm','loss_ci_lower_95','loss_ci_upper_95','cw_p_nw_raw','cw_p_mbb_raw','cw_p_nw_holm_Step16','cw_p_mbb_holm_Step16'],'OLD_NEW_SHRINKAGE_STEP26.csv')
 P=pd.read_csv(OUT/'PORTFOLIO_RIDGE_STEP26.csv',float_precision='round_trip');compare(P[P.arm=='old'],P[P.arm=='new'],['cell_id','input_set','target','split','h','strategy','layer','cost_bp'],[c for c in P if c not in ['cell_id','input_set','target','split','h','strategy','layer','cost_bp','arm']],'OLD_NEW_PORTFOLIOS_STEP26.csv')
 P=pd.read_csv(OUT/'POOLED_RIDGE_STEP26.csv',float_precision='round_trip');compare(P[P.arm=='old'],P[P.arm=='new'],['window','input_set','model','h'],['mse','r2os_train_mean','r2os_zero','da','mean','nw_se','ci95_lower','ci95_upper'],'OLD_NEW_POOLED_STEP26.csv')
 # Full Step21 table: replace each existing Ridge row, preserving every other row verbatim.
 da=pd.read_csv(ROOT/'revision_work/outputs/step21/all_da_per_seed.csv',float_precision='round_trip');new=da.copy()
 for i,r in new[new.model=='ridge'].iterrows():
  q=N[N.cell_id==r.id];assert len(q)==1;z=q.iloc[0];f=pd.read_csv(OUT/'cells'/r.id/'test_step26.csv',float_precision='round_trip');mask=f.y_true.to_numpy(float)!=0;p=f.y_pred.to_numpy(float);nvalid=int(mask.sum());correct=int((np.sign(p[mask])==np.sign(f.y_true.to_numpy(float)[mask])).sum())
  for k,v in dict(da=correct/nvalid,n_valid_pairs=nvalid,n_correct=correct,n_zero_realized_excluded=len(f)-nvalid,n_zero_forecasts_on_retained_pairs=int((p[mask]==0).sum()),prediction_file=str((OUT/'cells'/r.id/'test_step26.csv').relative_to(ROOT))).items():new.loc[i,k]=v
 write('manuscript/S13_full_step26.csv',new);summ=pd.read_csv(ROOT/'revision_work/outputs/step21/all_da_summary.csv',float_precision='round_trip')
 for i,r in summ[summ.model=='ridge'].iterrows():
  q=new[(new.model=='ridge')&(new.scope==r.scope)&(new.input_set==r.input_set)&(new.config==r.config)&(new.h==r.h)];assert len(q)==1
  for k in ['da_mean','da_min','da_max']:summ.loc[i,k]=q.iloc[0].da
 write('manuscript/S13_summary_full_step26.csv',summ)
 # Full CS metric table; all neural results and inferential families remain copied.
 d=ROOT/json.loads((ROOT/'revision_work/outputs/step22g/ANALYSIS_COMPLETE.json').read_text())['analysis_directory'];cs=pd.read_csv(d/'test_metrics_per_seed.csv',float_precision='round_trip')
 for i,r in cs[cs.arm=='Ridge-CS'].iterrows():
  q=N[(N.target=='cross_sectional')&(N.h==r.h)].iloc[0]
  for k,v in dict(mse=q.mse,mean_rank_ic=q.mean_rank_ic,da=q.da_excluding_zero,r2os=q.r2os_vs_zero,r2os_cs_zero=q.r2os_vs_zero).items():
   if k in cs:cs.loc[i,k]=v
 write('manuscript/S18_full_metrics_step26.csv',cs)
 csp=pd.read_csv(d/'portfolio_metrics.csv',float_precision='round_trip');newp=pd.read_csv(OUT/'CS_PORTFOLIO_METRICS_STEP26.csv',float_precision='round_trip')
 for i,r in csp[csp.arm=='Ridge-CS'].iterrows():
  q=newp[(newp.arm=='Ridge-CS_new')&(newp.h==r.h)&(newp.portfolio==r.portfolio)].iloc[0]
  for k,v in q.items():
   if k in csp and k!='arm':csp.loc[i,k]=v
 write('manuscript/S18_full_portfolios_step26.csv',csp)
 # Paired report values for CS portfolios.
 oldp=newp[newp.arm=='Ridge-CS_old'].drop(columns='arm');newp=newp[newp.arm=='Ridge-CS_new'].drop(columns='arm');compare(oldp,newp,['h','portfolio','cost_bp'],['sharpe','ann_turnover','max_drawdown','ce_A2','ce_A5'],'OLD_NEW_CS_PORTFOLIOS_STEP26.csv')
 js('COMPARISON_TABLES_COMPLETE.json',dict(full_DA_rows=len(new),full_CS_metric_rows=len(cs),full_CS_portfolio_rows=len(csp)))
if __name__=='__main__':
 try:main()
 except BaseException:js('COMPARISON_TABLES_BLOCKING.json',dict(error=traceback.format_exc()));raise
