"""Independent portfolio replay and normalization checks; no optimization/training or result updates."""
from review24_common import *
import independent24 as ind
import ast,math
from scipy import stats
from sklearn.covariance import LedoitWolf
TICKERS=['AES','BEP','BLDP','BLNK','CSIQ','CWEN','DQ','ENPH','FCEL','FSLR','HASI','JKS','NEE','ORA','PLUG','RUN','SEDG'];checks=[];impact=[];periods=[]
def ck(name,x,y,tol=1e-10,**meta):
 a=np.asarray(x,float);b=np.asarray(y,float);d=float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else 0.;checks.append(dict(check=name,passed=bool(d<=tol and np.array_equal(np.isnan(a),np.isnan(b))),max_abs_difference=d,tolerance=tol,**meta))
raw={}
for t in TICKERS:
 f=df(f'dataset/finance/panel_raw/{t}_daily.csv');raw[t]=f.set_index('datetime').close
C=pd.DataFrame(raw).dropna().sort_index();returns=C/C.shift(1)-1;returns=returns.iloc[1:];cal=list(returns.index);pos={d:i for i,d in enumerate(cal)};R=returns.to_numpy(float)
rf=df('data_raw_revision/rf/DTB3.csv');rf=rf.set_index('Date').daily_simple_rate;rf=rf.reindex(sorted(set(rf.index)|set(cal))).sort_index().ffill(limit=5).reindex(cal)
# Production functions under test, isolated from CLI/import-time side effects.
p='revision_work/scripts/portfolio/engine.py';tree=ast.parse(read(p));ns=dict(np=np,pd=pd,math=math,stats=stats,ANN=252)
exec(compile(ast.Module(body=[q for q in tree.body if isinstance(q,ast.FunctionDef) and q.name in ['grs','risk_parity','ols_nw','simulate','apply_costs','metrics']],type_ignores=[]),p,'exec'),ns)
rng=np.random.default_rng(20261022)
# With N=1, the exact finite-sample alpha F must equal the OLS intercept t squared.
factor=rng.normal(.15,.1,size=60);y=.04+.8*factor+rng.normal(0,.1,size=60);X=np.column_stack([np.ones(60),factor]);coef=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@coef;v=res@res/(60-2)*np.linalg.inv(X.T@X);t2=coef[0]**2/v[0,0];indf,indp=ind.grs(y[:,None],factor);prodf,prodp,_,_=ns['grs'](y[:,None],factor);ck('GRS independent N1 equals intercept t squared',indf,t2);ck('GRS production N1 equals intercept t squared',prodf,t2)
for sd in [np.array([1.,2.,3.]),np.arange(1.,18.)]:
 S=np.diag(sd*sd);w,err,it=ns['risk_parity'](S);want=1/sd;want/=want.sum();ck('Risk parity diagonal inverse-vol solution n'+str(len(sd)),w,want);ck('Risk parity RC convergence n'+str(len(sd)),err,0.,1e-10)
# Independent self-financing signed-stock + implicit idle-cash recurrence.
def replay(dates,targets,h):
 ex=np.array([pos[d]+1 for d in dates]);end=min(ex[-1]+h,len(cal)-1);days=np.arange(ex[0],end+1);out=np.zeros(len(days));turn=[];previous=np.zeros(17);period=[]
 for i,(e,w) in enumerate(zip(ex,targets)):
  turn.append(np.linalg.norm(w-previous,ord=1));stop=ex[i+1] if i+1<len(ex) else end;wealth=1.;holdings=w.astype(float).copy();cash=1-float(w.sum())
  for t in range(e+1,stop+1):
   before=wealth;holdings*=1+R[t];wealth=cash+holdings.sum();out[t-ex[0]]=wealth/before-1
  previous=holdings/wealth;period.append(wealth-1)
 turn=np.array(turn);net=out.copy()
 for i,e in enumerate(ex):net[e-ex[0]]=(1+out[e-ex[0]])*(1-.001*turn[i])-1
 return days,out,net,turn,np.array(period),ex
for inp in ['O','C']:
 for h in [5,10,20]:
  folder=ROOT/f'revision_work/outputs/analysis16/G/{inp}_h{h}';W=df(folder/'weights.csv');D=df(folder/'daily_returns.csv');RB=df(folder/'rebalances.csv');MS=df(folder/'metrics.csv');nstrategies=0
  for strategy,w in W.groupby('strategy',sort=False):
   w=w.sort_values('decision_date');days,gross,net,to,per,ex=replay(list(w.decision_date),w[TICKERS].to_numpy(float),h);d=D[D.strategy.eq(strategy)].sort_values('date');r=RB[RB.strategy.eq(strategy)].sort_values('decision_date');assert list(d.date)==[cal[i] for i in days]
   ck('Portfolio next-close daily gross',gross,d.gross.to_numpy(float),input=inp,h=h,strategy=strategy);ck('Portfolio drifted L1 turnover',to,r.turnover.to_numpy(float),input=inp,h=h,strategy=strategy);ck('Portfolio costs including initial',net,d.net_10bp.to_numpy(float),input=inp,h=h,strategy=strategy)
   m=MS[(MS.strategy==strategy.split('|')[0])&(MS.layer==strategy.split('|')[1])&(MS.cost_bp==10)] if '|' in strategy else MS[(MS.strategy==strategy)&(MS.cost_bp==10)]
   if len(m)==1:
    z=m.iloc[0];excess=net-rf.to_numpy(float)[days];nav=np.cumprod(1+net);values=[nav[-1]**(252/len(net))-1,np.std(net,ddof=1)*np.sqrt(252),np.mean(excess)/np.std(excess,ddof=1)*np.sqrt(252),np.min(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1),to[1:].mean(),252*(net.mean()-2.5*net.var(ddof=1))];ck('Portfolio return/vol/Sharpe/DD/turnover/CE',values,[z.ann_geo_return,z.ann_vol,z.sharpe,z.max_drawdown,z.avg_turnover_ex_initial,z.ce_A5],input=inp,h=h,strategy=strategy)
   nstrategies+=1
  periods.append(dict(input_set=inp,h=h,forecast_first=W.decision_date.min(),forecast_last=W.decision_date.max(),execution_first=D.date.min(),realized_last=D.date.max(),n_realized_dates=D.date.nunique(),n_portfolios=nstrategies))
  # Timestamp/covariance checks at representative decisions; no future price may enter LW fit.
  decisions=sorted(W.decision_date.unique());dates=[decisions[0],decisions[len(decisions)//2],decisions[-1]]
  for d in dates:
   idx=pos[d];sample=R[idx-251:idx+1];S=LedoitWolf().fit(sample).covariance_*h;copy=R.copy();copy[idx+1:]*=100;S2=LedoitWolf().fit(copy[idx-251:idx+1]).covariance_*h;ck('After-decision perturbation covariance',S,S2,input=inp,h=h,anchor=d)
   z=W[(W.strategy=='RISKPARITY')&(W.decision_date==d)].iloc[0][TICKERS].to_numpy(float);rc=z*(S@z)/(z@S@z);ck('Stored risk parity equal contributions',rc,np.full(17,1/17),1e-10,input=inp,h=h,anchor=d)
  # Same stored GRS groups and hypotheses; exact-normalization reference only.
  grs=df('revision_work/outputs/analysis16/G_GRS.csv');factor=D[D.strategy.eq('EW')].sort_values('date');factor_ex=factor.net_10bp.to_numpy(float)-rf.loc[factor.date].to_numpy(float)
  allnames=sorted(s for s in D.strategy.unique() if s.endswith('|L2_g5'))
  for label in ['all_seeds','ensembles']:
   names=[s for s in allnames if ('_s' in s.split('|')[0] if label=='all_seeds' else s.split('|')[0].endswith('_ens')) or s.split('|')[0]=='ridge'];E=np.column_stack([D[D.strategy.eq(s)].sort_values('date').net_10bp.to_numpy(float)-rf.loc[factor.date].to_numpy(float) for s in names]);a,pv=ind.grs(E,factor_ex);a0,p0,T,N=ns['grs'](E,factor_ex);stored=grs[(grs.input_set==inp)&(grs.h==h)&(grs.group==label)].iloc[0];ck('GRS production replay stored',[a0,p0],[stored.GRS_F,stored.p_value],input=inp,h=h,group=label)
   impact.append(dict(input_set=inp,h=h,group=label,N=N,T=T,stored_F=a0,stored_p=p0,reference_MLE_factor_F=a,reference_MLE_factor_p=pv,delta_F=a-a0,delta_p=pv-p0,rejection_changed=bool((pv<.05)!=(p0<.05)),reference='exact N1=t^2 and MLE factor covariance normalization; no new hypothesis family'))
# Step22G signed-book replay, all24 descriptive portfolios.
G=ROOT/'revision_work/outputs/step22g';j=js(G/'ANALYSIS_COMPLETE.json');gd=ROOT/j['analysis_directory'];W=df(gd/'portfolio_weights.csv');D=df(gd/'portfolio_daily_returns.csv');RB=df(gd/'portfolio_rebalances.csv')
for (arm,h,mode),w in W.groupby(['arm','h','portfolio']):
 w=w.sort_values('decision_date');days,gross,net,to,per,ex=replay(list(w.decision_date),w[TICKERS].to_numpy(float),h);dd=D[(D.arm==arm)&(D.h==h)&(D.portfolio==mode)].sort_values('date');r=RB[(RB.arm==arm)&(RB.h==h)&(RB.portfolio==mode)].sort_values('decision_date');ck('CS portfolio signed-book self-financing gross',gross,dd.gross_return.to_numpy(float),arm=arm,h=h,mode=mode);ck('CS full-book initial/drift turnover',to,r.turnover.to_numpy(float),arm=arm,h=h,mode=mode);ck('CS initial turnover 1 or2',to[0],2. if mode=='top5_minus_bottom5' else 1.,arm=arm,h=h,mode=mode);ck('CS portfolio net after10bp',net,dd.net_return.to_numpy(float),arm=arm,h=h,mode=mode)
write('PRIORITY2_PORTFOLIO_CHECKS.csv',pd.DataFrame(checks));write('GRS_NORMALIZATION_IMPACT.csv',pd.DataFrame(impact));write('PORTFOLIO_REALIZED_DATE_RANGES.csv',pd.DataFrame(periods));source_manifest('SOURCES_PORTFOLIO.csv');print('PORTFOLIO',len(checks),'checks; failures',sum(not c['passed'] for c in checks));print([c for c in checks if not c['passed']][:5]);print('GRS max absolute p impact',max(abs(r['delta_p']) for r in impact),'rej changes',sum(r['rejection_changed'] for r in impact))
