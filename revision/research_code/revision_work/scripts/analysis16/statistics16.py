"""Imported Step5/7/7B/9/15 statistics; fixed, declared families only."""
from common16 import *
from skill_common import mbb_index,holm,rank_ic_rows,nw_t,to_panel,reference_forecasts,load_meta,stable_int
from a2_skill import cell_metrics
from a3_hierarchical import series_stats
from ensemble_analysis import metrics as ensemble_metrics
from ensemble_robust import ens_series
from functools import lru_cache
import ast,types,collections
from scipy import stats
# Earlier modules put draw/application code inside main(). Reuse those source statements verbatim
# rather than writing a second bootstrap implementation. The exact AST excerpts are archived.
def extract(file,start,end):
 s=(ROOT/file).read_text();lines=s.splitlines();block='\n'.join(lines[start-1:end])+'\n';import textwrap
 return compile(textwrap.dedent(block),str(ROOT/file),'exec'),block
JCODE,JSOURCE=extract('revision_work/scripts/skill/a3_hierarchical.py',116,124)
# The imported Step15 missing-IC branch preserves the calendar and averages paired-valid dates per sampled seed.
ftree=ast.parse((FROZEN/'scripts/frozen_report.py').read_text())
missing=next(n for n in ast.walk(ftree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='n_missing')
MCODE=compile(ast.Module(body=[missing],type_ignores=[]),str(FROZEN/'scripts/frozen_report.py'),'exec')
MCODE_SOURCE=ast.get_source_segment((FROZEN/'scripts/frozen_report.py').read_text(),missing)
@lru_cache(None)
def draws(h,n=5,T=391,tag=1):
 # Source lines116-124 are the formal joint resampling draws and application; caller M is supplied.
 env=dict(np=np,mbb_index=mbb_index,BOOT_BASE_SEED=BASE,h=h,L=h,B=B,nS=n,N_TEST_DATES=T,M=np.zeros((n,T)))
 exec(JCODE,env);return env['W'],env['didx'],env['sidx']
@lru_cache(None)
def dates(h,T=391,tag=0):
 return mbb_index(T,h,B,np.random.default_rng(np.random.SeedSequence([BASE,h,h,tag])))
def distributions(X,h):
 X=np.asarray(X,float);n,T=X.shape;wt,didx,_=draws(h,n,T);theta=float(np.nanmean(X,axis=1).mean());n_missing=int((~np.isfinite(X)).sum())
 env=dict(np=np,X=X,wt=wt,didx=didx,n_missing=n_missing,missing=[],inp='',bk='',setting='',h=h,contrast='',metric='')
 exec(MCODE,env);bj=env['boot'];di=dates(h,T)
 # Same imported Step15 handling, now with equal deterministic seed weights (conditional on observed seeds).
 env.update(wt=np.full((B,n),1/n),didx=di);exec(MCODE,env)
 return theta,bj,env['boot']
def bootrow(X,h,**meta):
 theta,bj,bd=distributions(X,h);v=np.nanmean(X,axis=1);good=meta.get('metric')=='rank_ic'
 out=[]
 for kind,b in [('joint_seed_date',bj),('date_only',bd)]:out.append(dict(**meta,h=h,method=kind,n_seeds=len(X),n_dates=X.shape[1],L=h,B=B,seed_wins=int((v>0 if good else v<0).sum()),seed_delta_mean=float(np.mean(v)),seed_delta_sd=float(np.std(v,ddof=1)) if len(v)>1 else np.nan,**series_stats(b,theta)))
 return out

def corrected(d,family,p='centered_p_two_sided',target='p_holm_3h'):
 d=d.copy();d[target]=np.nan
 for _,g in d.groupby(family,dropna=False):
  for ix,v in holm({ix:float(pv) for ix,pv in g[p].items() if np.isfinite(pv)}).items():d.loc[ix,target]=v
 return d

class Analysis:
 def __init__(self):
  self.R=pd.read_csv(OUT/'registry.csv').fillna('');self.by={r.id:r._asdict() for r in self.R.itertuples(index=False)};self.meta=load_meta(sorted(pd.read_parquet(OUT/'labels/W3_h5.parquet').ticker.unique()))
 @lru_cache(None)
 def labels(self,w,h):return pd.read_parquet(OUT/f'labels/{w}_h{h}.parquet')
 @lru_cache(None)
 def test(self,w,h):return canon(self.labels(w,h).query("split=='test'"))
 @lru_cache(None)
 def refs(self,w,h):return reference_forecasts(self.labels(w,h),self.meta,self.test(w,h)[['ticker','anchor_date']],h)
 @lru_cache(None)
 def pred(self,id):return pd.read_parquet(ROOT/self.by[id]['canonical']).y_pred.to_numpy(float)
 def group(self,group,inp,model,h,config='daily_weekly',window='W3'):
  d=self.R.query('group==@group and input_set==@inp and model==@model and h==@h and config==@config and window==@window');assert len(d)>0,(group,inp,model,h,config,window)
  return {int(r.seed):r.id for r in d.itertuples()}
 def arrays(self,g):return np.stack([self.pred(g[s]) for s in sorted(g)])
 def paired(self,section,inp,contrast,h,A,Z,window='W3',detail=''):
  assert set(A)==set(Z)==set(SEEDS),(section,inp,contrast,h,set(A),set(Z));d=self.test(window,h);y=d.y_true.to_numpy(float);Y=y.reshape(-1,17);PA=self.arrays(A);PS=self.arrays(Z);loss=[];ic=[];n=len(A)
  for i in range(n):
   w=np.eye(n)[i];l,c,_,_=ens_series(y,Y.shape,PA,PS,Y,w);loss.append(l);ic.append(c)
  out=[]
  for metric,X in [('squared_loss',np.stack(loss)),('rank_ic',np.stack(ic))]:
   z=bootrow(X,h,section=section,input_set=inp,window=window,contrast=contrast,detail=detail,metric=metric)
   # Put both arms' mean/SD next to every test; seed differences retain pairing.
   va=np.mean((PA-y)**2,axis=1) if metric=='squared_loss' else np.array([np.nanmean(rank_ic_rows(p.reshape(Y.shape),Y)) for p in PA]);vz=np.mean((PS-y)**2,axis=1) if metric=='squared_loss' else np.array([np.nanmean(rank_ic_rows(p.reshape(Y.shape),Y)) for p in PS])
   for r in z:r.update(first_mean=float(va.mean()),first_sd=float(va.std(ddof=1)),second_mean=float(vz.mean()),second_sd=float(vz.std(ddof=1)))
   out+=z
  return out
 def contrasts(self):
  if verified('contrasts.COMPLETE.json'):return
  rows=[]
  for inp in ['O','C']:
   for h in HS:
    arms={m:self.group('core',inp,m,h) for m in ['single','fixed','static','adaptive']};arms['same_patch_fixed']=self.group('samepatch',inp,'same_patch_fixed',h)
    for a,b in [('single','fixed'),('single','same_patch_fixed'),('same_patch_fixed','fixed'),('fixed','static'),('static','adaptive')]:rows+=self.paired('A',inp,a+'−'+b,h,arms[a],arms[b])
    for w in ['W1','W2','W3']:
     a=self.group('rolling' if w!='W3' else 'core',inp,'static',h,window=w);b=self.group('rolling' if w!='W3' else 'core',inp,'adaptive',h,window=w);rows+=self.paired('C',inp,'static−adaptive',h,a,b,window=w)
    for mech in ['static','adaptive']:
     a={cfg:self.group('B3',inp,mech,h,config=cfg) for cfg in ['daily_only','weekly_only','daily_weekly']}
     for left,right in [('daily_only','weekly_only'),('daily_only','daily_weekly'),('weekly_only','daily_weekly')]:rows+=self.paired('D',inp,mech+':'+left+'−'+right,h,a[left],a[right])
     rows+=self.paired('D',inp,mech+':B3_DW−formal_D90W26',h,a['daily_weekly'],arms[mech],detail='coverage + exclusion of unfinished current week'+(' + C reference-price change' if inp=='C' else ''))
    for model in ['lstm','vanilla','swim']:
     for mech in ['static','adaptive']:rows+=self.paired('E',inp,model+'−'+mech,h,self.group('neural',inp,model,h),arms[mech])
   print('CONTRASTS',inp,flush=True)
  d=corrected(pd.DataFrame(rows),['section','input_set','window','contrast','metric','method']);files=[]
  for sec in ['A','C','D','E']:files.append(csv(f'{sec}_paired_tests.csv',d[d.section.eq(sec)]))
  complete('contrasts.COMPLETE.json',files,dict(tests=len(d)))
 def skill(self):
  if verified('B.COMPLETE.json'):return
  rows=[];files=[]
  for r in self.R[self.R.group.ne('rolling')].itertuples():
   name=f'B_cells/{r.id}.csv';mark=f'B_cells/{r.id}.COMPLETE.json'
   if verified(mark):rows.append(pd.read_csv(OUT/name).iloc[0].to_dict());continue
   d=self.test('W3',r.h);y=d.y_true.to_numpy(float);p=self.pred(r.id);Y=y.reshape(-1,17);P=p.reshape(Y.shape);refs,gtm=self.refs('W3',r.h)
   m,f,ic=cell_metrics(y,p,refs,r.h,dates(r.h,391,2),Y,P,gtm)
   mean,se,t,n=nw_t(ic,r.h);m.update(rank_ic_nw_lag=r.h,rank_ic_nw_se=se,rank_ic_nw_t=t,rank_ic_n_dates=n)
   valid=np.isfinite(ic)
   if valid.any():
    b=np.nanmean(ic[dates(r.h,391,2)],axis=1);m.update(rank_ic_p_mbb=series_stats(b,float(np.nanmean(ic)))['centered_p_two_sided'])
   else:m.update(rank_ic_p_mbb=np.nan)
   # No extra R2 tests. CW p refers only to GTM. Exclude the unrequested expanding-ticker reference output.
   m.pop('r2os_vs_expanding_ticker_mean',None)
   row=dict(id=r.id,group=r.group,input_set=r.input_set,model=r.model,config=r.config,h=r.h,seed=r.seed,window='W3',**m);rows.append(row)
   cp=csv(name,pd.DataFrame([row]));dp=csv(f'B_cells/{r.id}_date.csv',pd.DataFrame(dict(anchor_date=sorted(d.anchor_date.unique()),cw_f=f,rank_ic=ic)));complete(mark,[cp,dp]);files += [cp,dp]
   if len(rows)%60==0:print('SKILL',len(rows),flush=True)
  R=pd.DataFrame(rows)
  for p in ['cw_p_nw_one_sided','cw_p_mbb_one_sided','rank_ic_p_mbb']:
   R[p+'_holm_all_cells']=np.nan
   for ix,v in holm({ix:float(v) for ix,v in R[p].items() if np.isfinite(v)}).items():R.loc[ix,p+'_holm_all_cells']=v
  by=['group','input_set','model','config','h'];summary=R.groupby(by).agg(n_seeds=('seed','nunique'),mse_mean=('mse','mean'),mse_sd=('mse','std'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'),cw_seed_wins=('cw_mean',lambda x:int((x>0).sum())),rank_ic_seed_wins=('mean_rank_ic',lambda x:int((x>0).sum()))).reset_index()
  R=R.merge(summary,on=by,validate='many_to_one');flag=[]
  for r in R.itertuples():
   for ref in ['global_train_mean','zero','ticker_train_mean','expanding_pooled_mean']:
    val=getattr(r,'r2os_vs_'+ref)
    if val>0:flag.append(dict(id=r.id,metric='r2os_vs_'+ref,value=val,p_scope='CW vs Global Train-Mean only; not a p-value for this R2/reference',raw_p=r.cw_p_mbb_one_sided,p_holm_all_cells=r.cw_p_mbb_one_sided_holm_all_cells))
   if np.isfinite(r.rank_ic_p_mbb) and r.rank_ic_p_mbb<.05:flag.append(dict(id=r.id,metric='rank_ic',value=r.mean_rank_ic,p_scope='two-sided centered MBB vs zero IC',raw_p=r.rank_ic_p_mbb,p_holm_all_cells=r.rank_ic_p_mbb_holm_all_cells))
  files=[csv('B_forecast_skill.csv',R),csv('B_seed_summary.csv',summary),csv('B_flagged_cells.csv',pd.DataFrame(flag))]
  complete('B.COMPLETE.json',files,dict(cells=len(R),tests=int(R[['cw_p_nw_one_sided','cw_p_mbb_one_sided','rank_ic_p_mbb']].notna().sum().sum())))
 def rolling(self):
  if verified('C_diagnostics.COMPLETE.json'):return
  from engine import load_prices
  C,returns=load_prices();vol=returns.rolling(60,min_periods=60).std(ddof=1);rows=[];cw=[];cut=[]
  for w in ['W1','W2','W3']:
   for h in HS:
    d=self.test(w,h);y=d.y_true.to_numpy(float);Y=y.reshape(-1,17);refs,gtm=self.refs(w,h);train=self.labels(w,h).query("split=='train'")
    vv=np.array([vol.loc[r.anchor_date,r.ticker] for r in d.itertuples()]);terc=[]
    for t in sorted(d.ticker.unique()):
     vals=vol.loc[sorted(train[train.ticker.eq(t)].anchor_date.unique()),t];assert vals.notna().all();lo,hi=np.quantile(vals,[1/3,2/3]);cut.append(dict(window=w,h=h,ticker=t,q33=lo,q67=hi,n_train=len(vals)))
     mask=d.ticker.eq(t).to_numpy();reg=np.where(vv[mask]<=lo,'low',np.where(vv[mask]<=hi,'mid','high'));terc.extend(zip(np.flatnonzero(mask),reg))
    regime=np.empty(len(d),dtype=object)
    for ix,rr in terc:regime[ix]=rr
    dates_=sorted(d.anchor_date.unique());half=np.where(d.anchor_date.isin(dates_[:len(dates_)//2]),'first_half','second_half')
    for inp in ['O','C']:
     for model in ['static','adaptive','ridge','train_mean','zero']:
      g=self.group('rolling' if w!='W3' else ('core' if model in ['static','adaptive'] else 'ridge'),inp,model,h,window=w) if model not in ['train_mean','zero'] else {-1:None}
      for seed,ident in g.items():
       p=self.pred(ident) if ident else refs['global_train_mean' if model=='train_mean' else 'zero'];P=p.reshape(Y.shape)
       if model in ['static','adaptive','ridge']:
        for refname in ['global_train_mean','zero']:
         br=refs['global_train_mean'] if refname=='global_train_mean' else refs['zero'];rr=dict(refs);rr['global_train_mean']=br
         m,f,ic=cell_metrics(y,p,rr,h,dates(h,391,2),Y,P,float(br[0]));cw.append(dict(window=w,input_set=inp,model=model,h=h,seed=seed,reference=refname,cw_mean=m['cw_mean'],cw_nw_t=m['cw_nw_t'],p_nw=m['cw_p_nw_one_sided'],p_mbb=m['cw_p_mbb_one_sided']))
       # Stock IC is temporal rank correlation, explicitly labelled; regime/subperiod IC stays date-level cross-sectional.
       groups=[('stock',t,d.ticker.eq(t).to_numpy()) for t in sorted(d.ticker.unique())]+[('volatility',v,regime==v) for v in ['low','mid','high']]+[('subperiod',v,half==v) for v in ['first_half','second_half']]
       for typ,label,mask in groups:
        MSE=float(np.mean((y[mask]-p[mask])**2));base=float(np.mean((y[mask]-refs['global_train_mean'][mask])**2));ics=[]
        if typ=='stock':
         # A stock contributes one observation per date, so cross-sectional Rank IC is undefined; no temporal substitution.
         meanic=np.nan
        else:
         for date in sorted(set(d.anchor_date[mask])):
          mm=mask&d.anchor_date.eq(date).to_numpy()
          if mm.sum()>=2:ics.append(rank_ic_rows(p[mm][None,:],y[mm][None,:])[0])
         meanic=float(np.nanmean(ics)) if np.isfinite(ics).any() else np.nan
        rows.append(dict(window=w,input_set=inp,model=model,h=h,seed=seed,subgroup_type=typ,subgroup=label,n_rows=int(mask.sum()),mse=MSE,mse_relative_train_mean=MSE/base,mean_rank_ic=meanic,rank_ic_definition='undefined: one stock per date' if typ=='stock' else 'mean date-level cross-sectional IC within subgroup'))
    print('ROLLING DESCRIPTIVES',w,h,flush=True)
  Q=pd.DataFrame(cw)
  for p in ['p_nw','p_mbb']:Q=corrected(Q,['window','input_set','model','seed','reference'],p=p,target=p+'_holm_3h')
  summary=Q.groupby(['window','input_set','model','h','reference']).agg(n_seeds=('seed','nunique'),cw_mean=('cw_mean','mean'),cw_sd=('cw_mean','std'),seed_wins=('cw_mean',lambda x:int((x>0).sum()))).reset_index();Q=Q.merge(summary,on=['window','input_set','model','h','reference'],suffixes=('','_across_seeds'))
  D=pd.DataFrame(rows);S=D.groupby(['window','input_set','model','h','subgroup_type','subgroup'],as_index=False).agg(n_seeds=('seed','nunique'),n_rows=('n_rows','first'),mse_mean=('mse','mean'),mse_sd=('mse','std'),relative_mse_mean=('mse_relative_train_mean','mean'),relative_mse_sd=('mse_relative_train_mean','std'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'))
  files=[csv('C_rolling_CW.csv',Q),csv('C_subgroups_per_seed.csv',D),csv('C_subgroups_summary.csv',S),csv('C_volatility_cutpoints.csv',pd.DataFrame(cut))];complete('C_diagnostics.COMPLETE.json',files,dict(tests=int(Q[['p_nw','p_mbb']].notna().sum().sum())))

def main():
 a=Analysis();put('bootstrap_source_joint.py.txt',JSOURCE);put('bootstrap_source_missing.py.txt',MCODE_SOURCE)
 a.contrasts();a.skill();a.rolling()
if __name__=='__main__':main()
