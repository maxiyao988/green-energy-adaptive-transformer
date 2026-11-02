"""All-cell independent replay for Tables4/9/10/11 and Step16 skill arithmetic."""
from review24_common import *
import independent24 as ind
import math
from scipy.stats import norm
A=ROOT/'revision_work/outputs/analysis16';registry=df(A/'registry.csv').set_index('id');B=df(A/'clarified/B_forecast_skill_final.csv');storedsum=df(A/'B_PRIMARY_seed_summary.csv');checks=[];metrics=[];P={};LAB={};date_ic={};losses={}
def ck(name,x,y,tol=1e-10,**meta):
 aa=np.asarray(x,float);bb=np.asarray(y,float);support=np.array_equal(np.isnan(aa),np.isnan(bb));d=float(np.nanmax(np.abs(aa-bb))) if np.isfinite(aa-bb).any() else 0.;checks.append(dict(check=name,passed=bool(support and d<=tol),max_abs_difference=d,nan_support_equal=support,tolerance=tol,**meta))
def labels(w,h):
 if (w,h) not in LAB:
  f=canon(pq(A/f'labels/{w}_h{h}.parquet'));test=f[f.split=='test'].reset_index(drop=True);m=float(f[f.split=='train'].y_true.to_numpy(float).mean());LAB[w,h]=(test,m)
 return LAB[w,h]
def prediction(id):
 if id not in P:
  r=registry.loc[id];f=canon(pq(ROOT/r.canonical));P[id]=f.y_pred.to_numpy(float)
 return P[id]
# Full738-cell skill family: each canonical forecast, aliases resolved explicitly.
for row in B.itertuples():
 ident=row.alias_of if isinstance(row.alias_of,str) and row.alias_of else row.id;d,m=labels('W3',row.h);p=prediction(ident);y=d.y_true.to_numpy(float);mm,l,ic=ind.metrics(y,p,(391,17),m);metrics.append(dict(id=row.id,**{k:getattr(row,k) for k in ['group','input_set','model','config','h','seed']},**mm));losses[row.id]=l;date_ic[row.id]=ic
 for col,val in [('mse',mm['mse']),('mean_rank_ic',mm['rank_ic']),('r2os_vs_global_train_mean',mm['r2_train_mean']),('r2os_vs_zero',mm['r2_zero'])]:ck('Step16 '+col,val,getattr(row,col),id=row.id)
 e=y-p;ts=np.mean(e.reshape(391,17),axis=1);tsmse=float(np.mean(ts**2));xsmse=float(np.mean((e.reshape(391,17)-ts[:,None])**2));ck('MSE decomposition TS/XS',[tsmse,xsmse],[row.mse_ts_component,row.mse_xs_component],id=row.id)
 var_y=np.var(y);var_p=np.var(p);cov=np.mean((y-y.mean())*(p-p.mean()));bias=(y.mean()-p.mean())**2;ck('MSE moment decomposition',bias+var_y+var_p-2*cov,row.mse,id=row.id)
 if np.isfinite(row.cw_mean):
  f,ser,r=ind.cw(y,p,m,(391,17),row.h);ck('CW mean/SE/t/p NW',[r['mean'],r['se'],r['t'],r['p_one']],[row.cw_mean,row.cw_nw_se,row.cw_nw_t,row.cw_p_nw_one_sided],id=row.id)
 if np.isfinite(ic).any():
  r=ind.nw(ic,row.h);ck('IC NW lagh mean/SE/t',[r['mean'],r['se'],r['t']],[row.mean_rank_ic,row.rank_ic_nw_se,row.rank_ic_nw_t],id=row.id)
 if len(metrics)%150==0:print('HEADLINES cells',len(metrics),flush=True)
M=pd.DataFrame(metrics);S=M.groupby(['group','input_set','model','config','h']).agg(n_seeds=('seed','nunique'),mse_mean=('mse','mean'),mse_sd=('mse','std'),rank_ic_mean=('rank_ic','mean'),rank_ic_sd=('rank_ic','std')).reset_index();write('INDEPENDENT_STEP16_METRICS_738.csv',M);write('INDEPENDENT_TABLE4_SUMMARY.csv',S)
for r in S[S.group.isin(['core','samepatch','neural','ridge'])].itertuples():
 q=storedsum[(storedsum.group==r.group)&(storedsum.input_set==r.input_set)&(storedsum.model==r.model)&(storedsum.config==r.config)&(storedsum.h==r.h)];assert len(q)==1
 z=q.iloc[0]
 for col in ['mse_mean','mse_sd','rank_ic_mean','rank_ic_sd']:
  if r.n_seeds==1 and col.endswith('_sd'):continue
  ck('Table4 source summary '+col,getattr(r,col),z[col],cell=str((r.group,r.input_set,r.model,r.config,r.h)))
# Independently inspect the738-member Holm promise with undefined raw p assigned1, and existing finite-only implementation.
holmrows=[]
for col in ['cw_p_nw_one_sided','cw_p_mbb_one_sided','rank_ic_p_mbb']:
 raw=B[col].to_numpy(float);valid=np.isfinite(raw);stored=B[col+'_holm_all_cells'].to_numpy(float);finite=ind.holm(raw[valid]);full=ind.holm(np.where(valid,raw,1.))
 ck('Holm matches stored finite subset '+col,finite,stored[valid],family_n=int(valid.sum()))
 for i in np.flatnonzero(valid):
  holmrows.append(dict(id=B.id.iloc[i],method=col,finite_family_size=int(valid.sum()),promised_cell_count=len(B),stored_holm=float(stored[i]),independent_all738_holm=float(full[i]),delta=float(full[i]-stored[i]),rejection_changed=bool((full[i]<.05)!=(stored[i]<.05))))
write('HOLM_738_FAMILY_AUDIT.csv',pd.DataFrame(holmrows))
# Bootstrap draws/reference are independent of project helper code.
BBOOT=20000;draw={}
def draws(h,base=20261019,tag=1):
 k=(h,base,tag)
 if k not in draw:
  rng=np.random.default_rng(np.random.SeedSequence([base,h,h,tag]));si=rng.integers(0,5,size=(BBOOT,5)) if tag==1 else None;di=ind.mbb(391,h,BBOOT,rng);draw[k]=si,di
 return draw[k]
def bootstats(X,h,base=20261019):
 theta=float(np.mean(X));si,di=draws(h,base,1);joint=ind.joint_from_draws(X,si,di);dd=draws(h,base,0)[1];date=X.mean(0)[dd].mean(1)
 def info(bb):return dict(theta=theta,lo=float(np.quantile(bb,.025)),hi=float(np.quantile(bb,.975)),p=float((1+np.count_nonzero(np.abs(bb-theta)>=abs(theta)))/(len(bb)+1)))
 return info(joint),info(date)
def group(w,inp,model,h):
 g='rolling' if w!='W3' else 'core';q=registry[(registry['group']==g)&(registry.window==w)&(registry.input_set==inp)&(registry.model==model)&(registry.h==h)&(registry.config=='daily_weekly')].sort_values('seed');assert len(q)==5;return np.stack([prediction(i) for i in q.index])
CSTORE=df(A/'C_paired_tests.csv');ASTORE=df(A/'A_paired_tests.csv');T9=[];T10=[];ROLL=[]
for w in ['W1','W2','W3']:
 for inp in ['O','C']:
  for h in [5,10,20]:
   d,m=labels(w,h);y=d.y_true.to_numpy(float);pa,pz=group(w,inp,'static',h),group(w,inp,'adaptive',h);base=float(np.mean((y-m)**2));lm=(pa-y)**2;rm=(pz-y)**2;X=(lm-rm).reshape(5,391,17).mean(2);bj,bd=bootstats(X,h)
   for method,info in [('joint_seed_date',bj),('date_only',bd)]:
    r=CSTORE[(CSTORE.window==w)&(CSTORE.input_set==inp)&(CSTORE.h==h)&(CSTORE.metric=='squared_loss')&(CSTORE.method==method)].iloc[0];ck('Table10 stored contrast '+method,[info['theta'],info['lo'],info['hi'],info['p']],[r.theta_hat,r.ci_lower_95,r.ci_upper_95,r.centered_p_two_sided],window=w,input=inp,h=h)
   for model,loss in [('static',lm),('adaptive',rm)]:T10.append(dict(window=w,input_set=inp,model=model,h=h,mse=float(loss.mean()),mean_relative_mse=float(loss.mean()/base),seed_sd_relative_mse=float(np.mean(loss,axis=1).std(ddof=1)/base)))
   T10.append(dict(window=w,input_set=inp,model='zero',h=h,mse=float(np.mean(y*y)),mean_relative_mse=float(np.mean(y*y)/base)))
   ROLL.append(dict(window=w,input_set=inp,h=h,p_raw=bj['p'],estimate=bj['theta'],ci_lo=bj['lo'],ci_hi=bj['hi'],static_wins=int((lm.mean(1)<rm.mean(1)).sum())))
   if w=='W3':
    # Table9 O retains Step5 RNG20260826; C uses Step16 RNG20261019. Contrast is Adaptive−Static.
    oj,od=bootstats(-X,h,20260826 if inp=='O' else 20261019)
    for method,inf in [('joint_seed_date',oj),('date_only',od)]:T9.append(dict(input_set=inp,h=h,method=method,seed_wins_static=int((X.mean(1)<0).sum()),**inf))
for inp in ['O','C']:
 for method in ['joint_seed_date','date_only']:
  rows=[r for r in T9 if r['input_set']==inp and r['method']==method];adj=ind.holm([r['p'] for r in rows])
  for r,v in zip(rows,adj):r['p_holm_3h']=float(v)
for w in ['W1','W2','W3']:
 for inp in ['O','C']:
  rows=[r for r in ROLL if r['window']==w and r['input_set']==inp];adj=ind.holm([r['p_raw'] for r in rows])
  for r,v in zip(rows,adj):r['p_holm_3h']=float(v);z=CSTORE[(CSTORE.window==w)&(CSTORE.input_set==inp)&(CSTORE.h==r['h'])&(CSTORE.metric=='squared_loss')&(CSTORE.method=='joint_seed_date')].iloc[0];ck('Table10 Holm',v,z.p_holm_3h,window=w,input=inp,h=r['h'])
write('INDEPENDENT_TABLE9.csv',pd.DataFrame(T9));write('INDEPENDENT_TABLE10_METRICS.csv',pd.DataFrame(T10));write('INDEPENDENT_TABLE10_TESTS.csv',pd.DataFrame(ROLL))
# Held-out Table11: reconstruct ensembles from all5 source prediction arrays, pool SSE, NW by window.
E=ROOT/'revision_work/outputs/step22e';ej=js(E/'ANALYSIS_COMPLETE.json');dest=ROOT/ej['analysis_directory'];ps=df(dest/'prediction_sources.csv');ES=df(dest/'ensemble_summary.csv') if (dest/'ensemble_summary.csv').exists() else df(dest/'ensemble_metrics.csv');ICSTORE=df(dest/'E1_rank_ic.csv') if (dest/'E1_rank_ic.csv').exists() else None
T11=[];cwrows=[]
for model in ['static','adaptive','lstm']:
 for h in [5,10,20]:
  per={};cws={};ics={};ens={}
  for w in ['W1','W2']:
   q=ps[(ps.setting=='T3')&(ps.model==model)&(ps.window==w)&(ps.h==h)].sort_values('seed');assert len(q)==5
   fs=[canon(df(ROOT/p)) if str(p).endswith('.csv') else canon(pq(ROOT/p)) for p in q.predictions];d,m=labels(w,h);y=d.y_true.to_numpy(float);pp=np.stack([z.y_pred.to_numpy(float) for z in fs]);ep=pp.mean(0);met,l,ei=ind.metrics(y,ep,(391,17),m);per[w]=dict(met=met,n=len(y),sse=float(np.sum((ep-y)**2)),gtm_sse=float(np.sum((y-m)**2)),zero_sse=float(np.sum(y*y)));cws[w]=ind.cw(y,ep,m,(391,17),h)[1];ics[w]=np.mean([ind.rank_ic(p.reshape(391,17),y.reshape(391,17)) for p in pp],axis=0)
  total_sse=sum(a['sse'] for a in per.values());r2=1-total_sse/sum(a['gtm_sse'] for a in per.values());r0=1-total_sse/sum(a['zero_sse'] for a in per.values());ic=np.concatenate(list(ics.values())).mean();nw=ind.pooled_nw(list(cws.values()),h-1)
  row=dict(model=model,h=h,r2_W1=per['W1']['met']['r2_train_mean'],r2_W2=per['W2']['met']['r2_train_mean'],r2_W1_W2=r2,r2_zero_W1_W2=r0,mean_rank_ic=ic,cw_mean=nw['mean'],cw_se=nw['se'],cw_p_nw=nw['p_one']);T11.append(row)
  z=ES[(ES.setting=='T3')&(ES.model==model)&ES.h.eq(h)&ES.window.eq('W1+W2')].iloc[0];ck('Table11 pooled ensemble R2 and zero R2',[r2,r0],[z.r2os_train_mean,z.r2os_zero],model=model,h=h)
for models in [['static','adaptive'],['lstm']]:
 rows=[r for r in T11 if r['model'] in models];adj=ind.holm([r['cw_p_nw'] for r in rows])
 for r,p in zip(rows,adj):r['cw_p_nw_holm']=float(p)
write('INDEPENDENT_TABLE11.csv',pd.DataFrame(T11));write('PRIORITY1_HEADLINE_CHECKS.csv',pd.DataFrame(checks));source_manifest('SOURCES_HEADLINES.csv');print('HEADLINE done',len(checks),'checks; failures',sum(not r['passed'] for r in checks));print([r for r in checks if not r['passed']][:6])
