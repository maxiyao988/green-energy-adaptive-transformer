"""Independent Step7B replay: seed composition changes forecasts before loss/IC evaluation."""
from review24_common import *
import independent24 as ind
R=df('revision_work/outputs/analysis16/registry.csv');root=ROOT/'revision_work/outputs/skill_analysis/ensemble_robust';L=df(root/'loo_ensembles.csv');S=df(root/'seed_resampled_bootstrap.csv');B=df(root/'seed_resampled_bootstrap_replicates.csv');C=[]
def ck(n,a,b,**meta):
 a=np.asarray(a,float);b=np.asarray(b,float);d=float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else 0.;C.append(dict(check=n,passed=bool(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and d<=1e-10),max_abs_difference=d,**meta))
for h in [5,10,20]:
 lab=canon(pq(f'revision_work/outputs/analysis16/labels/W3_h{h}.parquet'));y=lab[lab.split.eq('test')].y_true.to_numpy(float);P={}
 for model in ['static','adaptive']:
  for seed in [0,1,21,42,3407]:
   q=R[R.group.eq('core')&R.input_set.eq('O')&R.model.eq(model)&R.h.eq(h)&R.seed.eq(seed)].iloc[0];P[model,seed]=canon(pq(ROOT/q.canonical)).y_pred.to_numpy(float)
 for r in L[L.h.eq(h)].itertuples():
  seeds=[int(s) for s in r.seeds_used.split(',')];out={}
  for model in ['static','adaptive']:
   p=np.mean([P[model,s] for s in seeds],axis=0);out[model]=(np.mean((y-p)**2),np.nanmean(ind.rank_ic(p.reshape(391,17),y.reshape(391,17))))
  ck('Leave-one-seed-out / full equal-weight ensemble',[*out['static'],*out['adaptive']],[r.mse_static,r.rank_ic_static,r.mse_adaptive,r.rank_ic_adaptive],h=h,seed_set=r.seed_set,held_out=str(r.held_out_seed))
 for seedset,seeds in [('like_for_like',[0,1,21,3407]),('all_seeds',[0,1,21,42,3407])]:
  PA=np.stack([P['adaptive',s] for s in seeds]);PS=np.stack([P['static',s] for s in seeds]);k=len(seeds)
  for spec,length in [('PRIMARY',h),('SENSITIVITY',2*h)]:
   rng=np.random.default_rng(np.random.SeedSequence([20261017,h,length]));si=rng.integers(0,k,size=(20000,k));di=ind.mbb(391,length,20000,rng);counts=np.stack([(si==s).sum(1) for s in range(k)],axis=1);unique,inverse=np.unique(counts,axis=0,return_inverse=True);bl=np.empty(20000);bi=np.empty(20000)
   for j,co in enumerate(unique):
    w=co/k;pa=np.sum(PA*w[:,None],axis=0);ps=np.sum(PS*w[:,None],axis=0);ld=((y-pa)**2-(y-ps)**2).reshape(391,17).mean(1);ic=ind.rank_ic(pa.reshape(391,17),y.reshape(391,17))-ind.rank_ic(ps.reshape(391,17),y.reshape(391,17));ix=np.flatnonzero(inverse==j);bl[ix]=ld[di[ix]].mean(1);bi[ix]=np.nanmean(ic[di[ix]],axis=1)
   for metric,bb in [('loss_adaptive_minus_static',bl),('rank_ic_adaptive_minus_static',bi)]:
    q=B[B.seed_set.eq(seedset)&B.h.eq(h)&B.specification.eq(spec)&B.metric.eq(metric)].sort_values('replicate');assert len(q)==20000;ck('Stored seed-composition bootstrap all20000',bb,q.value,h=h,seed_set=seedset,specification=spec,metric=metric)
    z=S[S.seed_set.eq(seedset)&S.h.eq(h)&S.specification.eq(spec)&S.metric.eq(metric)].iloc[0];lo,hi=np.quantile(bb,[.025,.975]);ck('Seed-composition percentile interval',[lo,hi],[z.ci_lower_95,z.ci_upper_95],h=h,seed_set=seedset,specification=spec,metric=metric)
   print('ENSEMBLE',h,seedset,spec,flush=True)
write('ENSEMBLE_REPLAY_CHECKS.csv',pd.DataFrame(C));source_manifest('SOURCES_ENSEMBLE_REPLAY.csv');print('ENSEMBLE checks',len(C),'failed',sum(not c['passed'] for c in C));print([c for c in C if not c['passed']][:10])
