"""OLS HAC reference and stored original-O GRS normalization impact; no new tests."""
from review24_common import *
import independent24 as ind
import ast,math
from scipy import stats
sys.path.insert(0,str(OUT/'deps'))
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
p='revision_work/scripts/portfolio/engine.py';tree=ast.parse(read(p));ns=dict(np=np,pd=pd,stats=stats)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['ols_nw','grs']],type_ignores=[]),p,'exec'),ns)
rng=np.random.default_rng(20261022);x=rng.normal(size=391);y=.01+.8*x+rng.normal(size=391);C=[]
for lag in [0,4,5,9,10,19,20]:
 b,se,u=ns['ols_nw'](y,x,lag);fit=sm.OLS(y,np.column_stack([np.ones(len(x)),x])).fit();rs=np.sqrt(np.diag(cov_hac(fit,nlags=lag,use_correction=False)));d=float(np.max(abs(np.r_[b,se]-np.r_[fit.params,rs])));C.append(dict(check='Alpha OLS HAC versus statsmodels',lag=lag,max_abs_difference=d,passed=d<=1e-10))
old=ROOT/'revision_work/outputs/portfolio/original_rp';D=df(old/'daily_returns_gross_and_net10bp.csv');G=df(old/'grs_tests.csv');RF=df('data_raw_revision/rf/DTB3.csv');RF=RF.set_index('Date').daily_simple_rate;RF=RF.reindex(sorted(set(RF.index)|set(D.date))).sort_index().ffill(limit=5)
impact=[];alpha=[];M=df(old/'metrics_all.csv')
for r in G[G.split.eq('test')].itertuples():
 d=D[D.split.eq('test')&D.h.eq(r.h)];f=d[d.strategy.eq('EW')].sort_values('date');rf=RF.loc[f.date].to_numpy(float);fac=f.net_10bp.to_numpy(float)-rf;members=r.members.split(',');Y=np.column_stack([d[d.strategy.eq(n+'|L2_g5')].sort_values('date').net_10bp.to_numpy(float)-rf for n in members]);z,pv=ind.grs(Y,fac);z0,p0,T,N=ns['grs'](Y,fac)
 impact.append(dict(input_set='O Step8 original',h=r.h,group=r.assets,N=N,T=T,stored_F=r.GRS_F,stored_p=r.p_value,reference_MLE_factor_F=z,reference_MLE_factor_p=pv,delta_F=z-r.GRS_F,delta_p=pv-r.p_value,rejection_changed=bool((pv<.05)!=(r.p_value<.05)),production_replay_F_error=z0-r.GRS_F,production_replay_p_error=p0-r.p_value,precision_note='Original Step8 daily/summary CSV rounded to10 significant digits'))
for h in [5,10,20]:
 d=D[D.split.eq('test')&D.h.eq(h)];f=d[d.strategy.eq('EW')].sort_values('date');rf=RF.loc[f.date].to_numpy(float);fac=f.net_10bp.to_numpy(float)-rf
 for r in M[M.split.eq('test')&M.h.eq(h)&M.cost_bp.eq(10)&M.alpha_vs_ew_nw_t.notna()].itertuples():
  name=r.strategy+'|'+r.layer;g=d[d.strategy.eq(name)].sort_values('date')
  if g.empty:g=d[d.strategy.eq(r.strategy)].sort_values('date')
  fit=sm.OLS(g.net_10bp.to_numpy(float)-rf,np.column_stack([np.ones(len(f)),fac])).fit();se=np.sqrt(np.diag(cov_hac(fit,nlags=5,use_correction=False)));t=float(fit.params[0]/se[0]);a=float(fit.params[0]*252);alpha.append(dict(h=h,strategy=r.strategy,layer=r.layer,stored_alpha_ann=r.alpha_vs_ew_ann,independent_alpha_ann=a,abs_alpha_delta=abs(a-r.alpha_vs_ew_ann),stored_t=r.alpha_vs_ew_nw_t,independent_t=t,abs_t_delta=abs(t-r.alpha_vs_ew_nw_t),significance_changed=bool((abs(t)>1.96)!=(abs(r.alpha_vs_ew_nw_t)>1.96)),precision_note='Step8 CSV10 significant-digit quantization; 1e-10 differences retained'))
write('ALPHA_HAC_CHECKS.csv',pd.DataFrame(C));write('ORIGINAL_O_GRS_NORMALIZATION_IMPACT.csv',pd.DataFrame(impact));write('ORIGINAL_O_ALPHA_REFERENCE.csv',pd.DataFrame(alpha));source_manifest('SOURCES_ALPHA_GRS_REFERENCE.csv');print('HAC',C);print('Old O GRS max delta p',max(abs(x['delta_p']) for x in impact),'rejection changes',sum(x['rejection_changed'] for x in impact));print('Alpha rows',len(alpha),'max tdelta',max(x['abs_t_delta'] for x in alpha),'rejection changes',sum(x['significance_changed'] for x in alpha))
