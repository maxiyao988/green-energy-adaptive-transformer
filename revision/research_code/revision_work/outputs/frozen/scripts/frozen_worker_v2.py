"""Step15 single CPU numerical worker; formal trainers and model classes are imported unchanged."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
from frozen_io_v3 import *
import time,types,copy,argparse,subprocess,traceback
import numpy as np,pandas as pd
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
from scripts.python import panel_common as common
from scripts.python import panel_pathformer_main_v2 as formal
from scripts.python import panel_pathformer_experiment3 as exp3
from scripts.python import panel_pathformer_experiment4_multiseed as exp4
import core
from torch.utils.data import DataLoader,TensorDataset

DATA={};FORMAL_LOAD=formal._build_training_loaders;FACTORY=formal._make_model

def profile():
 assert torch.get_num_threads()==12 and torch.get_num_interop_threads()==12
 assert os.environ.get('OMP_NUM_THREADS') is None
 assert os.environ['OMP_DYNAMIC']==os.environ['MKL_DYNAMIC']=='FALSE'
 return dict(torch_num_threads=12,torch_num_interop_threads=12,environment={k:os.environ.get(k) for k in ENV_KEYS})
class Telemetry:
 def __init__(self,path):self.f=target(path).open('x');self.deadline=0
 def log(self,event,force=False):
  now=time.perf_counter()
  if not force and now<self.deadline:return
  d=dict(utc=utc(),event=event,pid=os.getpid(),**profile(),load_average=os.getloadavg(),power=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout.strip(),top_processes=subprocess.run(['ps','-axo','pid,ppid,pcpu,comm','-r'],capture_output=True,text=True).stdout.splitlines()[:21])
  self.f.write(json.dumps(d)+chr(10));self.f.flush();self.deadline=now+300
  print('TELEMETRY',event,d['utc'],d['load_average'],flush=True)
 def close(self):self.f.close()

def dataset(inp):
 if inp not in DATA:
  DATA[inp]=core.load_causal() if inp=='C' else common.load_frozen_panel()
  formal._formal_check_dataset_contract(DATA[inp])
 return DATA[inp]
def canonical(d):
 d=d.copy();d['anchor_date']=pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d')
 return d.sort_values(['ticker','anchor_date'],kind='mergesort').reset_index(drop=True)
def keys(ds,sp,h):
 m=ds.split==sp
 return pd.DataFrame(dict(ticker=ds.ticker[m],anchor_date=pd.to_datetime(ds.anchor_date[m]).strftime('%Y-%m-%d'),split=sp,y_true=ds.y[h][m]))
def reference(spec,sp,formal_source=False):
 p=ROOT/(spec['formal_reference'] if formal_source else spec['references'][sp])
 assert sha(p)==(spec['formal_reference_sha256'] if formal_source else spec['reference_sha256'][sp])
 if p.suffix=='.parquet':d=pd.read_parquet(p)
 else:
  d=pd.read_csv(p,float_precision='round_trip')
  h=pd.to_numeric(d.horizon.astype(str).str.replace('d','',regex=False));d=d[(h==spec['horizon'])&(d.seed==spec['seed'])]
  if 'mechanism' in d:d=d[d.mechanism==spec['mechanism']]
  if 'frequency_setting' in d:d=d[d.frequency_setting=='daily_weekly']
 return canonical(d)
def comparison(frame,ref):
 a=canonical(frame);b=canonical(ref);ks=['ticker','anchor_date']
 samekeys=len(a)==len(b) and a[ks].equals(b[ks]);assert samekeys,'Replay key mismatch'
 assert np.array_equal(a.y_true.to_numpy(np.float32),b.y_true.to_numpy(np.float32)),'Replay labels mismatch'
 diff=float(np.max(np.abs(a.y_pred.to_numpy(float)-b.y_pred.to_numpy(float))))
 y=a.y_true.to_numpy(float)
 return dict(max_abs_delta_pred=diff,delta_mse=float(np.mean((a.y_pred.to_numpy(float)-y)**2)-np.mean((b.y_pred.to_numpy(float)-y)**2)),classification='EXACT' if diff<1e-7 else 'DIFFERENT',identical_keys=True,identical_labels=True)
def model_for(mech):return FACTORY('daily_weekly','adaptive') if mech=='adaptive' else exp3._make_experiment3_model(mech,None,None)
def cached_branch(self,x):return self.aggregator(list(x.unbind(dim=1)))
def bind_cache(m):
 for b in [m.daily_branch,m.weekly_branch]:b.forward=types.MethodType(cached_branch,b)
 return m

def make_cache(spec,tele):
 base=OUT/'backbones'/spec['id']/'cache';done,j=finished(base)
 if done:
  assert j['checkpoint_sha256']==spec['checkpoint_sha256'];print('CACHE_SKIP',spec['id'],flush=True);return done.parent,j
 attempt=base/('attempt_'+uid());ds=dataset(spec['input_set']);h=spec['horizon'];start=time.perf_counter()
 assert sha(ROOT/spec['checkpoint'])==spec['checkpoint_sha256']
 model=model_for(spec['mechanism']);model.load_state_dict(torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=True),strict=True);model.cpu().eval()
 train,val,test=FORMAL_LOAD('daily_weekly',h,ds,batch_size=formal.BATCH_SIZE)
 assert train.generator.initial_seed()==42
 loaders={'train':DataLoader(train.dataset,batch_size=32,shuffle=False),'val':val,'test':test}
 replay=[];paths=[];means=None
 for sp,loader in loaders.items():
  expert_chunks={'daily':[],'weekly':[]};handles=[]
  for branch,name in [(model.daily_branch,'daily'),(model.weekly_branch,'weekly')]:
   def capture(mod,args,name=name):
    tele.log('cache_'+spec['id']+'_'+sp)
    values=args[0];assert len(values)==4 and all(v.shape[1]==32 for v in values)
    expert_chunks[name].append(torch.stack([v.detach().cpu() for v in values],dim=1))
   handles.append(branch.aggregator.register_forward_pre_hook(capture))
  pred,dw,ww=exp4._batched_inference(model,loader,COUNTS[sp],torch.device('cpu'))
  for handle in handles:handle.remove()
  ar=np.concatenate([torch.cat(expert_chunks['daily']).numpy(),torch.cat(expert_chunks['weekly']).numpy()],axis=1)
  assert ar.shape==(COUNTS[sp],8,32) and ar.dtype==np.float32 and np.isfinite(ar).all()
  npy=attempt/f'{sp}_experts.npy';nout(npy,ar);paths.append(npy)
  k=keys(ds,sp,h);frame=k.assign(y_pred=pred)
  for q,col in enumerate(WEIGHTS):frame[col]=(dw if q<4 else ww)[:,q%4]
  savedcheck=comparison(frame,reference(spec,sp));replay.append(dict(split=sp,comparison='checkpoint_vs_saved_parquet',**savedcheck))
  if sp=='test':replay.append(dict(split=sp,comparison='checkpoint_vs_formal_TEST',**comparison(frame,reference(spec,sp,True))))
  # Existing class/fusion/router on the serialized expert cache. No projection/expert executes twice.
  original=[model.daily_branch.forward,model.weekly_branch.forward];bind_cache(model)
  cd=torch.from_numpy(ar[:,:4].copy());cw=torch.from_numpy(ar[:,4:].copy());yy=torch.from_numpy(k.y_true.to_numpy(np.float32).copy())
  c_loader=DataLoader(TensorDataset(cd,cw,yy),batch_size=32,shuffle=False)
  cp,dd,ww2=exp4._batched_inference(model,c_loader,COUNTS[sp],torch.device('cpu'))
  model.daily_branch.forward,model.weekly_branch.forward=original
  dmax=float(np.max(np.abs(cp.astype(float)-pred.astype(float))));wmax=max(float(np.max(np.abs(dd-dw))),float(np.max(np.abs(ww2-ww))))
  replay.append(dict(split=sp,comparison='cache_vs_full_checkpoint',max_abs_delta_pred=dmax,delta_mse=0. if np.array_equal(cp,pred) else float(np.mean((cp-k.y_true.to_numpy())**2)-np.mean((pred-k.y_true.to_numpy())**2)),classification='EXACT' if dmax<1e-7 else 'DIFFERENT',identical_keys=True,identical_labels=True,bitwise_predictions=np.array_equal(cp,pred),max_abs_delta_weights=wmax))
  p=attempt/f'backbone_{sp}.parquet';pout(p,frame);paths.append(p)
  if sp=='train':means=ar.mean(axis=0,dtype=np.float64).astype(np.float32)
  print('CACHE_SPLIT',spec['id'],sp,'max_delta',savedcheck['max_abs_delta_pred'],'cache',dmax,flush=True)
  if any(r['classification']!='EXACT' for r in replay):
   cout(attempt/'REPLAY_FAILED.csv',pd.DataFrame(replay));raise RuntimeError('STOP: non-EXACT backbone replay '+spec['id'])
 mp=attempt/'train_mean_experts.npy';nout(mp,means);paths.append(mp)
 rp=attempt/'cache_replay.csv';cout(rp,pd.DataFrame(replay));paths.append(rp)
 j=dict(validated=True,backbone_id=spec['id'],checkpoint_sha256=spec['checkpoint_sha256'],provenance=spec['provenance'],shape_per_sample=[8,32],dtype='float32',row_counts=COUNTS,train_mean_reduction='float64 mean of TRAIN cache only, cast to float32',wall_seconds=time.perf_counter()-start,profile=profile(),replay=replay,artifacts=[artifact(p) for p in paths])
 jout(attempt/'COMPLETE.json',j);print('CACHE_COMPLETE',spec['id'],j['wall_seconds'],flush=True);return attempt,j

def configure_variant(spec,variant,setting,means):
 mode='fixed' if variant=='fixed' else ('static' if variant=='static' else 'adaptive')
 m=model_for(mode);state=torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=True)
 freeze={k:v for k,v in state.items() if '.aggregator.' not in k};missing,unexpected=m.load_state_dict(freeze,strict=False)
 assert not unexpected and all('.aggregator.' in k for k in missing)
 for p in m.parameters():p.requires_grad_(False)
 for b in [m.daily_branch,m.weekly_branch]:
  for p in b.aggregator.parameters():p.requires_grad_(variant!='fixed')
 if setting=='head_retrained':
  for p in m.fusion.parameters():p.requires_grad_(True)
 if variant=='constant_gate':
  for b,mu in [(m.daily_branch,means[:4]),(m.weekly_branch,means[4:])]:
   agg=b.aggregator;agg.register_buffer('fixed_train_mean',torch.from_numpy(mu.copy()));original=agg._adaptive_weights
   def constant(expert_outputs,agg=agg,original=original):
    return original([agg.fixed_train_mean[q].view(1,-1).expand_as(expert_outputs[q]) for q in range(4)])
   agg._adaptive_weights=constant
 original_train=m.train
 def train(self,mode=True):
  original_train(mode)
  for b in [self.daily_branch,self.weekly_branch]:
   b.input_proj.eval();b.positional_encoding.eval();b.experts.eval()
  if setting=='head_frozen':self.fusion.eval()
  return self
 m.train=types.MethodType(train,m);bind_cache(m);m.eval()
 if variant=='static':assert all(torch.equal(b.aggregator.scale_logits,torch.zeros(4)) for b in [m.daily_branch,m.weekly_branch])
 expected=(0 if variant=='fixed' else 8 if variant=='static' else 8520)+(4225 if setting=='head_retrained' else 0)
 assert formal.count_trainable_parameters(m)==expected,(variant,setting,formal.count_trainable_parameters(m))
 return m

def param_sha(m,trainable):
 d={k:v.detach().cpu() for k,v in m.named_parameters() if v.requires_grad==trainable}
 h=__import__('hashlib').sha256()
 for k,v in d.items():h.update(k.encode());h.update(v.numpy().tobytes())
 return h.hexdigest()

def fit_variant(spec,cache,variant,setting,tele):
 base=OUT/'backbones'/spec['id']/setting/variant;done,j=finished(base)
 if done:print('VARIANT_SKIP',spec['id'],setting,variant,flush=True);return j
 attempt=base/('attempt_'+uid());start=time.perf_counter();stamp=utc();h=spec['horizon'];ds=dataset(spec['input_set']);mean=np.load(cache/'train_mean_experts.npy');loaders={}
 for sp in COUNTS:
  a=np.load(cache/f'{sp}_experts.npy');y=ds.y[h][ds.split==sp];td=TensorDataset(torch.from_numpy(a[:,:4].copy()),torch.from_numpy(a[:,4:].copy()),torch.from_numpy(y.copy()))
  if sp=='train':loaders[sp]=DataLoader(td,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42))
  else:loaders[sp]=DataLoader(td,batch_size=32,shuffle=False)
 assert loaders['train'].generator.initial_seed()==42
 models=[];frozen_before={};init={};mse_hist=[];sse={'n':0,'v':0.};hooks=[]
 def construct(*args,**kwargs):
  m=configure_variant(spec,variant,setting,mean);models.append(m);frozen_before['sha']=param_sha(m,False);init['trainable_sha256']=param_sha(m,True)
  init['parameters']={k:p.numel() for k,p in m.named_parameters() if p.requires_grad};init['initial_router_hash']=__import__('hashlib').sha256(b''.join(p.detach().numpy().tobytes() for b in [m.daily_branch,m.weekly_branch] for p in b.aggregator.parameters())).hexdigest()
  def check(mod,args):
   tele.log(spec['id']+' '+setting+' '+variant)
   assert all(not b.input_proj.training and not b.experts.training for b in [m.daily_branch,m.weekly_branch])
   if setting=='head_frozen':assert not m.fusion.training
  hooks.append(m.register_forward_pre_hook(check));return m
 original_load,original_factory,original_seed=formal.load_frozen_panel,formal._make_model,formal.SEED;original_build=formal._build_training_loaders;original_huber=torch.nn.HuberLoss
 def watched_huber(*args,**kwargs):
  loss=original_huber(*args,**kwargs)
  def observe(mod,args,out):
   if not models[0].training:
    pred,y=args;diff=pred.detach()-y.detach();sse['v']+=float((diff*diff).sum(dtype=torch.float64));sse['n']+=len(y)
    if sse['n']==6290:mse_hist.append(sse['v']/sse['n']);sse.update(n=0,v=0.)
    assert sse['n']<6290
  hooks.append(loss.register_forward_hook(observe));return loss
 try:
  formal.SEED=spec['seed'];formal.load_frozen_panel=lambda:ds;formal._make_model=construct;formal._build_training_loaders=lambda *args,**kwargs:(loaders['train'],loaders['val'],loaders['test']);torch.nn.HuberLoss=watched_huber
  if variant=='fixed':
   formal.set_seed(spec['seed']);m=construct();hist=pd.DataFrame(columns=['epoch','train_loss','val_loss']);summary=dict(best_epoch=0,epochs_trained=0,val_loss=None)
  else:
   result=formal.train_one_cell('daily_weekly',h,'adaptive');m=models[0];hist=pd.DataFrame(result['history']);summary=result['summary'].iloc[0].to_dict() if isinstance(result['summary'],pd.DataFrame) else result['summary']
   assert len(hist)==len(mse_hist);hist['val_mse']=mse_hist
 finally:
  formal.SEED=original_seed;formal.load_frozen_panel=original_load;formal._make_model=original_factory;formal._build_training_loaders=original_build;torch.nn.HuberLoss=original_huber
  for handle in hooks:handle.remove()
 assert param_sha(m,False)==frozen_before['sha'],'Frozen parameters changed'
 m.eval();ntrain=formal.count_trainable_parameters(m);paths=[];testmetrics=None
 if len(hist):hist['train_huber']=hist.train_loss;hist['val_huber']=hist.val_loss;hist['loss_type']='Huber delta=1'
 p=attempt/'history.csv';cout(p,hist);paths.append(p)
 # Only learned routing/head tensors and the constant-input buffers are serialized; backbone/cache hashes bind the rest.
 checkpoint=dict(trainable_state={k:v.detach().cpu().clone() for k,v in m.named_parameters() if v.requires_grad},constant_train_mean=torch.from_numpy(mean.copy()) if variant=='constant_gate' else None,backbone=spec['checkpoint'],backbone_sha256=spec['checkpoint_sha256'],variant=variant,setting=setting,best_epoch=int(summary['best_epoch']),head_initialization='backbone trained head, retained at initialization',router_initialization='fresh formal seeded factory; Static zeros')
 cp=attempt/'best_routing_head.pt'
 with target(cp).open('xb') as f:torch.save(checkpoint,f)
 paths.append(cp)
 for sp,ld in loaders.items():
  if sp=='train':ld=DataLoader(ld.dataset,batch_size=32,shuffle=False)
  pred,dw,ww=exp4._batched_inference(m,ld,COUNTS[sp],torch.device('cpu'));frame=keys(ds,sp,h).assign(y_pred=pred)
  for q,col in enumerate(WEIGHTS):frame[col]=(dw if q<4 else ww)[:,q%4]
  assert len(frame)==COUNTS[sp] and np.isfinite(frame[['y_true','y_pred']+WEIGHTS]).all().all()
  assert np.allclose(dw.sum(1),1,atol=1e-6) and np.allclose(ww.sum(1),1,atol=1e-6)
  if variant in ['fixed','static','constant_gate']:assert max(np.ptp(dw,axis=0).max(),np.ptp(ww,axis=0).max())<1e-7,'Sample-dependent constant routing'
  p=attempt/f'{sp}.parquet';pout(p,frame);paths.append(p)
  if sp=='test':
   metrics=common.evaluate_predictions(frame.y_true.to_numpy(),pred,frame.ticker.to_numpy(),frame.anchor_date.to_numpy());testmetrics=dict(test_mse=metrics['pooled_mse'],mean_rank_ic=metrics['cross_sectional_rank_ic'],direction_accuracy=metrics['pooled_da'])
   dates=common.rank_ic_by_date(frame.y_true.to_numpy(),pred,frame.ticker.to_numpy(),frame.anchor_date.to_numpy())
   loss=frame.assign(mse_date=(frame.y_true.to_numpy(float)-pred.astype(float))**2,da_date=(np.sign(frame.y_true.to_numpy())==np.sign(pred)).astype(float)).groupby('anchor_date')[['mse_date','da_date']].mean().reset_index()
   dates=dates.merge(loss,on='anchor_date',validate='one_to_one');p=attempt/'test_date_metrics.csv';cout(p,dates);paths.append(p)
 j=dict(validated=True,backbone_id=spec['id'],input_set=spec['input_set'],backbone=spec['backbone'],provenance=spec['provenance'],horizon=h,seed=spec['seed'],setting=setting,variant=variant,trainable_parameters=ntrain,trainable_parameter_names=init['parameters'],initial_router_hash=init['initial_router_hash'],frozen_parameters_sha256=frozen_before['sha'],frozen_parameters_unchanged=True,started_utc=stamp,ended_utc=utc(),wall_seconds=time.perf_counter()-start,best_epoch=int(summary['best_epoch']),epochs=int(summary['epochs_trained']),train_loader_generator_seed=42,profile=profile(),metrics=testmetrics,cache_complete=rel(cache/'COMPLETE.json'),checkpoint=rel(cp),artifacts=[artifact(p) for p in paths])
 jout(attempt/'COMPLETE.json',j);print('VARIANT_COMPLETE',spec['id'],setting,variant,j['epochs'],j['wall_seconds'],testmetrics,flush=True);return j

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',required=True);ap.add_argument('--probe',action='store_true');args=ap.parse_args();session=Path(args.session);tele=Telemetry(session/'telemetry.jsonl');tele.log('start',True);registry=json.loads((OUT/'backbone_registry.json').read_text());started=utc()
 exp3._verify_frozen_dependencies();exp4._verify_frozen_dependencies();exp4._verify_frozen_training_constants()
 # Qualify every non14a source before using any of them for a routing experiment.
 for spec in registry:
  if spec['provenance']!='14a':make_cache(spec,tele)
 jout(session/'NON14A_QUALIFIED.json',dict(utc=utc(),n_sources=17,classification='EXACT',all_strict_load=True))
 completions=[]
 for spec in registry:
  cache,cj=make_cache(spec,tele)
  for setting in (['head_frozen','head_retrained'] if spec['backbone']=='S1' else ['head_frozen']):
   for variant in ['fixed','static','constant_gate','adaptive']:
    if setting=='head_retrained' and variant=='fixed':continue
    j=fit_variant(spec,cache,variant,setting,tele);completions.append(j)
    snap=session/'status'/f'{len(completions):04d}_{uid()}.csv';cout(snap,pd.DataFrame([{k:r[k] for k in ['backbone_id','setting','variant','best_epoch','epochs','wall_seconds','trainable_parameters']} for r in completions]))
  if args.probe:break
 tele.log('end',True);tele.close();jout(session/'WORKER_COMPLETE.json',dict(utc=utc(),started_utc=started,n_variants=len(completions),probe=args.probe,profile=profile()));print('WORKER_COMPLETE',len(completions),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  p=OUT/'failures'/('FAIL_'+uid()+'.txt');text(p,traceback.format_exc());print(traceback.format_exc(),flush=True);raise
