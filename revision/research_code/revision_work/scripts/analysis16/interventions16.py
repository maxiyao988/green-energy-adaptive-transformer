"""Step9 functions with Step13 C binding and Step15 verified caches; no fit/training."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
from common16 import *
import time,types,subprocess
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(FROZEN/'scripts'))
# Preserve the requested numerical profile while importing earlier low-thread adapters.
envclass=type(os.environ);oldset=envclass.__setitem__;tn=torch.set_num_threads;ti=torch.set_num_interop_threads
protected={'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'}
def guard(self,k,v):
 if k not in protected:oldset(self,k,v)
envclass.__setitem__=guard;torch.set_num_threads=lambda n:None;torch.set_num_interop_threads=lambda n:None
try:
 fw=module('step15_cache_worker',FROZEN/'scripts/frozen_worker_v3.py')
 iv=module('step9_suite',ROOT/'revision_work/scripts/interventions/router_interventions.py')
finally:envclass.__setitem__=oldset;torch.set_num_threads=tn;torch.set_num_interop_threads=ti
# No alternate model implementation: original factory, formal helper, Step15 branch cache binding,
# Step9 RouterOverride/make_fns/run_forward/metrics. Step13 data binding is fw.dataset -> core.load_causal.
def profile():
 assert torch.get_num_threads()==torch.get_num_interop_threads()==12 and os.environ.get('OMP_NUM_THREADS') is None
 return dict(utc=utc(),pid=os.getpid(),threads=[12,12],env={k:os.environ.get(k) for k in sorted(protected|{'OMP_DYNAMIC','MKL_DYNAMIC'})},load=os.getloadavg())
def run():
 reg=json.loads((FROZEN/'backbone_registry.json').read_text())
 for spec in [r for r in reg if r['backbone']=='S2']:
  ident=spec['id'];marker=f'H/{ident}/COMPLETE.json'
  if verified(marker):print('H SKIP',ident,flush=True);continue
  started=time.perf_counter();files=[];pro=[profile()];assert sha(ROOT/spec['checkpoint'])==spec['checkpoint_sha256']
  ds=fw.dataset(spec['input_set']);h=spec['horizon'];seed=spec['seed'];inp=spec['input_set']
  m=fw.model_for('adaptive');m.load_state_dict(torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=True),strict=True);m.cpu().eval()
  _,_,test=fw.FORMAL_LOAD('daily_weekly',h,ds,batch_size=32)
  full,dw,ww=fw.exp4._batched_inference(m,test,6647,torch.device('cpu'))
  mine=fw.keys(ds,'test',h).assign(y_pred=full);check=fw.comparison(mine,fw.reference(spec,'test'))
  assert check['classification']=='EXACT',('STOP I0 full',ident,check)
  cp,cj=fw.finished(FROZEN/'backbones'/ident/'cache');assert cp
  assert cj['checkpoint_sha256']==spec['checkpoint_sha256'] and all(r['classification']=='EXACT' for r in cj['replay'])
  cache=cp.parent
  for ar in cj['artifacts']:assert sha(ROOT/ar['path'])==ar['sha256']
  fw.bind_cache(m);loaders={};keys={};learned={};preds={}
  for sp in ['train','val','test']:
   a=np.load(cache/f'{sp}_experts.npy');k=fw.keys(ds,sp,h);keys[sp]=k
   ld=DataLoader(TensorDataset(torch.from_numpy(a[:,:4].copy()),torch.from_numpy(a[:,4:].copy()),torch.from_numpy(k.y_true.to_numpy(np.float32).copy())),batch_size=32,shuffle=False)
   loaders[sp]=ld;pr,dw,ww=fw.exp4._batched_inference(m,ld,len(k),torch.device('cpu'));learned[sp]={'daily':dw,'weekly':ww};preds[sp]=pr
   assert fw.comparison(k.assign(y_pred=pr),fw.reference(spec,sp))['classification']=='EXACT',(ident,sp)
  ov=[iv.RouterOverride(m.daily_branch.aggregator,'daily'),iv.RouterOverride(m.weekly_branch.aggregator,'weekly')]
  own=iv.run_forward(m,loaders['test'],ov,None);dmax=float(np.max(np.abs(own.astype(float)-full.astype(float))));assert dmax<1e-7
  verification=dict(id=ident,input_set=inp,h=h,seed=seed,**check,cache_i0_max_delta=dmax,**profile(),checkpoint=spec['checkpoint'],checkpoint_sha256=spec['checkpoint_sha256'])
  files.append(js(f'H/{ident}/I0_verification.json',verification))
  tk=keys['train'].ticker.to_numpy();tickers=sorted(set(tk));tmap={t:i for i,t in enumerate(tickers)}
  trmean={b:learned['train'][b].mean(0) for b in ['daily','weekly']};tickmean={b:np.stack([learned['train'][b][tk==t].mean(0) for t in tickers]) for b in ['daily','weekly']}
  metrics=[];dated=[];routers=[]
  for b in ['daily','weekly']:
   z=learned['test'][b];total=float(pd.DataFrame(z).var().sum());between=float(pd.DataFrame(z).groupby(keys['test'].anchor_date.to_numpy()).transform('mean').var().sum())
   routers.append(dict(id=ident,input_set=inp,h=h,seed=seed,branch=b,train_mean=json.dumps(trmean[b].tolist()),test_mean_sd_across_samples=float(z.std(0).mean()),share_variance_between_dates=between/total if total else None))
  for sp in ['test','val']:
   k=keys[sp];y=k.y_true.to_numpy(float);Y=iv.panel(k,y);fns=iv.make_fns(learned[sp],np.array([tmap[t] for t in k.ticker]),k.anchor_date.to_numpy(),trmean,tickmean);base=None
   for name in iv.INTERVENTIONS:
    p=iv.run_forward(m,loaders[sp],ov,fns[name]).astype(float);P=iv.panel(k,p);met,ic=iv.metrics(y,p,Y,P);loss=iv.panel(k,(y-p)**2).mean(1)
    if name=='I0_learned':base=(met,ic,loss)
    metrics.append(dict(input_set=inp,h=h,seed=seed,split=sp,intervention=name,**met,delta_mse=met['mse']-base[0]['mse'],delta_rank_ic=met['mean_rank_ic']-base[0]['mean_rank_ic']))
    if sp=='test':
     dates=sorted(k.anchor_date.unique())
     dated.extend(dict(input_set=inp,h=h,seed=seed,intervention=name,anchor_date=d,loss_difference=float(l),rank_ic_difference=float(c)) for d,l,c in zip(dates,loss-base[2],ic-base[1]))
    files.append(pq(f'H/{ident}/predictions/{name}_{sp}.parquet',k.assign(y_pred=p)))
   pro.append(profile())
  files += [csv(f'H/{ident}/metrics.csv',pd.DataFrame(metrics)),csv(f'H/{ident}/date_differences.csv',pd.DataFrame(dated)),csv(f'H/{ident}/routers.csv',pd.DataFrame(routers)),js(f'H/{ident}/telemetry.json',pro)]
  complete(marker,files,dict(wall_seconds=time.perf_counter()-started,training_runs=0,cache_source=rel(cp),source_sha256=sha(cp)))
  print('H COMPLETE',ident,round(time.perf_counter()-started,2),flush=True)
if __name__=='__main__':run()
