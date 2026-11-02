"""Supplemental complete saved-replicate replay for Step22D's stratified statistics."""
from review24_common import *
import independent24 as ind
D=ROOT/js('revision_work/outputs/step22d/session_v3/ANALYSIS_COMPLETE.json')['analysis_directory'];IC=df(D/'part1_rank_ic_inference.csv');CT=df(D/'part1_static_adaptive_contrasts.csv');CF=df(D/'confirmatory_tests.csv');C=[];draws={};cache={}
def ck(name,a,b,**meta):
 a=np.asarray(a,float);b=np.asarray(b,float);dd=float(np.nanmax(abs(a-b))) if np.isfinite(a-b).any() else 0.;C.append(dict(check=name,passed=bool(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and dd<=1e-10),max_abs_difference=dd,**meta))
def load(p):read(p);return np.load(p,allow_pickle=False)
for h in [5,10,20]:
 for code,w in enumerate(['W1','W2','W3'],1):
  z=load(D/f'draws/{w}_h{h}.npz');di=z['date_indices'];si=z['seed_indices'];ck('Saved within-window MBB indices independent replay',di,ind.mbb(391,h,20000,np.random.default_rng(np.random.SeedSequence([20261021,h,code,0]))),window=w,h=h);ck('Saved paired-seed draws independent replay',si,np.random.default_rng(np.random.SeedSequence([20261021,h,code,1])).integers(0,5,size=(20000,5)),window=w,h=h);draws[w,h]=(di,si)
for r in IC.itertuples():
 key=(r.input_set,r.model,r.h)
 if key not in cache:cache[key]=load(D/f'inputs/part1_{r.input_set}_{r.model}_h{r.h}_ic.npz')
 z=cache[key];ws=r.window.split('+');bb=np.mean([z[w][draws[w,r.h][0]].mean(1) for w in ws],axis=0);stored=load(ROOT/r.bootstrap);ck('Pooled/component date bootstrap all20000',bb,stored,input=r.input_set,model=r.model,h=r.h,window=r.window);lo,hi=np.quantile(bb,[.025,.975]);ck('Pooled/component IC percentile interval',[lo,hi],[r.ci95_lower,r.ci95_upper],input=r.input_set,model=r.model,h=r.h,window=r.window)
 if r.model!='ridge':
  nw=ind.pooled_nw([z[w] for w in ws],r.h);p=(1+np.count_nonzero(abs(bb-nw['mean'])>=abs(nw['mean'])))/20001;ck('Pooled/component IC NW and two-sided MBB',[nw['mean'],nw['se'],nw['p_two'],p],[r.mean,r.nw_se,r.p_nw,r.p_mbb],input=r.input_set,model=r.model,h=r.h,window=r.window)
for r in CT.itertuples():
 key=(r.input_set,r.h,r.metric)
 if key not in cache:cache[key]=load(D/f'inputs/part1_{r.input_set}_static-minus-adaptive_h{r.h}_{r.metric}.npz')
 z=cache[key];ws=r.window.split('+');bb=np.mean([ind.joint_from_draws(z[w],draws[w,r.h][1],draws[w,r.h][0]) for w in ws],axis=0);theta=np.concatenate([z[w] for w in ws],axis=1).mean();lo,hi=np.quantile(bb,[.025,.975]);pv=(1+np.count_nonzero(abs(bb-theta)>=abs(theta)))/20001;sgn=1 if r.metric=='mse' else -1;up=np.quantile(sgn*bb,.95);ck('Stratified paired-seed/date all20000',bb,load(ROOT/r.bootstrap),input=r.input_set,h=r.h,metric=r.metric,window=r.window);ck('Stratified contrast estimate/interval/p/bound',[theta,lo,hi,pv,up],[r.estimate,r.ci95_lower,r.ci95_upper,r.p_mbb,r.adaptive_improvement_upper95],input=r.input_set,h=r.h,metric=r.metric,window=r.window)
for r in CF.itertuples():
 tag='ic' if r.family=='CONFIRMATORY_RANK_IC' else 'cw';z=load(D/f'inputs/confirm_{tag}_h{r.h}.npz');ws=['W1','W2'];bb=np.mean([z[w][draws[w,r.h][0]].mean(1) for w in ws],axis=0);nw=ind.pooled_nw([z[w] for w in ws],r.nw_lag);mu=nw['mean'];pv=(1+np.count_nonzero(bb-mu>=mu))/20001 if tag=='cw' else (1+np.count_nonzero(abs(bb-mu)>=abs(mu)))/20001;pn=nw['p_one'] if tag=='cw' else nw['p_two'];ck('Held-out confirmatory stratified all20000',bb,load(ROOT/r.bootstrap),h=r.h,metric=tag);ck('Held-out NW / MBB reference',[mu,nw['se'],pn,pv],[r.mean,r.nw_se,r.p_nw,r.p_mbb],h=r.h,metric=tag)
for family in ['PRIMARY','SECONDARY']:
 g=IC[IC.family.eq(family)]
 for method in ['nw','mbb']:ck('Pooled IC separate Holm '+family+' '+method,ind.holm(g['p_'+method]),g['p_'+method+'_holm'])
for family,g in CF.groupby('family'):
 for method in ['nw','mbb']:ck('Held-out separate Holm '+family+' '+method,ind.holm(g['p_'+method]),g['p_'+method+'_holm'])
write('STRATIFIED_REPLAY_CHECKS.csv',pd.DataFrame(C));source_manifest('SOURCES_STRATIFIED_REPLAY.csv');print('STRATIFIED',len(C),'failed',sum(not c['passed'] for c in C));print([c for c in C if not c['passed']][:8])
