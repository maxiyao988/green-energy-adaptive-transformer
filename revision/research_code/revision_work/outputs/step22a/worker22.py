"""Imported Step15 configuration and unchanged formal training-prefix AST."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
from io22 import *
sys.path.insert(0,str(FR/'scripts'))
import frozen_worker_v3 as f
torch=f.torch;np=f.np;pd=f.pd
import ast,copy,traceback,types,subprocess
STAGE='idle';TELE=None;PREPARED=False
DENY=set()
def init():
 global TELE,DENY
 check_protocol();f.profile()
 for b in bindings():
  DENY.update(str((ROOT/p).resolve()) for p in [b['cache']['test']['experts'],b['cache']['test']['keys']])
  for old in b['old'].values():DENY.add(str((ROOT/old['predictions']['test']['path']).resolve()))
 def audit(event,args):
  if STAGE=='train' and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.fsdecode(args[0]);resolved=str(Path(p).absolute())
   if resolved in DENY or resolved.endswith('test_experts.npy') or resolved.endswith('backbone_test.parquet'):
    raise RuntimeError('STOP: forbidden TEST read during training: '+resolved)
 sys.addaudithook(audit)
 TELE=target(OUT/'telemetry'/f'worker_{os.getpid()}.jsonl').open('x');tele('worker_start')
def tele(event,**extra):
 profile=f.profile()
 d=dict(utc=utc(),pid=os.getpid(),stage=STAGE,event=event,**profile,load_average=os.getloadavg(),**extra)
 if TELE:TELE.write(json.dumps(d)+'\n');TELE.flush()
 return d
def raw_keys(b,sp,labels=True):
 cols=['ticker','anchor_date','split']+(['y_true'] if labels else [])
 d=pd.read_parquet(ROOT/b['cache'][sp]['keys'],columns=cols)
 d.anchor_date=pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d')
 assert len(d)==COUNTS[sp] and d.split.eq(sp).all() and not d.duplicated(['ticker','anchor_date']).any()
 return d
def loader(b,sp,shuffle=False,dummy=False):
 a=np.load(ROOT/b['cache'][sp]['experts'],allow_pickle=False)
 k=raw_keys(b,sp,not dummy)
 y=np.zeros(len(k),np.float32) if dummy else k.y_true.to_numpy(np.float32).copy()
 td=f.TensorDataset(torch.from_numpy(a[:,:4].copy()),torch.from_numpy(a[:,4:].copy()),torch.from_numpy(y))
 kw=dict(batch_size=32,shuffle=shuffle,num_workers=0)
 if shuffle:kw['generator']=torch.Generator().manual_seed(42)
 return f.DataLoader(td,**kw),k
def configured(b,v):
 f.formal.set_seed(b['seed'])
 means=np.load(ROOT/b['mean']['path'],allow_pickle=False)
 return f.configure_variant(b,v,'head_frozen',means)
def load_state(m,p):
 j=torch.load(p,map_location='cpu',weights_only=True)
 parameters=dict(m.named_parameters());expected={k for k,p in parameters.items() if p.requires_grad}
 assert set(j['trainable_state'])==expected
 with torch.no_grad():
  for k,v in j['trainable_state'].items():parameters[k].copy_(v)
 if j.get('constant_train_mean') is not None:
  for branch,mu in [(m.daily_branch,j['constant_train_mean'][:4]),(m.weekly_branch,j['constant_train_mean'][4:])]:
   assert torch.equal(branch.aggregator.fixed_train_mean,mu)
 return m.cpu().eval()
def cache_task(b):
 global STAGE
 STAGE='cache';start=utc();clock=time.perf_counter();tele('cache_start',backbone=b['id'])
 meta=[]
 for sp in COUNTS:
  p=ROOT/b['cache'][sp]['experts'];assert sha(p)==b['cache'][sp]['experts_sha256']
  a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==(COUNTS[sp],8,32) and a.dtype==np.float32 and np.isfinite(a).all()
  # Exact router inputs: branch-major concatenation, preserving four expert order.
  dr=a[:,:4].reshape(len(a),128);wr=a[:,4:].reshape(len(a),128)
  assert np.shares_memory(dr,a) and np.shares_memory(wr,a)
  assert np.array_equal(dr,np.concatenate([a[:,i] for i in range(4)],axis=1))
  assert np.array_equal(wr,np.concatenate([a[:,i] for i in range(4,8)],axis=1))
  keys=raw_keys(b,sp);assert sha(ROOT/b['cache'][sp]['keys'])==b['cache'][sp]['keys_sha256']
  if sp=='train':mu=a.mean(axis=0,dtype=np.float64).astype(np.float32)
  meta.append(dict(split=sp,shape=list(a.shape),dtype=str(a.dtype),expert_source=b['cache'][sp]['experts'],sha256=sha(p),router_daily_shape=list(dr.shape),router_weekly_shape=list(wr.shape),zero_copy_views=True,keys_source=b['cache'][sp]['keys']))
 stored=np.load(ROOT/b['mean']['path']);assert np.array_equal(mu,stored)
 result=dict(backbone_id=b['id'],cache_started_utc=start,cache_ended_utc=utc(),wall_seconds=time.perf_counter()-clock,head_checkpoint=b['checkpoint'],head_checkpoint_sha256=b['checkpoint_sha256'],train_mean=b['mean'],cache=meta,profile=f.profile())
 js(OUT/'cache_manifests'/f"{b['id']}.json",result);tele('cache_end',backbone=b['id']);return result
def replay_task(task):
 global STAGE
 b,sp=task;STAGE=sp+'_replay';start=utc();clock=time.perf_counter();tele('replay_start',backbone=b['id'],split=sp)
 ld,k=loader(b,sp,dummy=True);rows=[]
 for v in ['fixed']+VARIANTS:
  old=b['old'][v];assert sha(ROOT/old['checkpoint'])==old['checkpoint_sha256']
  m=load_state(configured(b,v),ROOT/old['checkpoint'])
  p,_,_=f.exp4._batched_inference(m,ld,COUNTS[sp],torch.device('cpu'))
  ref=pd.read_parquet(ROOT/old['predictions'][sp]['path'],columns=['ticker','anchor_date','y_pred'])
  ref.anchor_date=pd.to_datetime(ref.anchor_date).dt.strftime('%Y-%m-%d')
  mine=k[['ticker','anchor_date']].assign(y_pred=p)
  aa=f.canonical(mine);rr=f.canonical(ref)
  assert aa[['ticker','anchor_date']].equals(rr[['ticker','anchor_date']])
  delta=float(np.max(np.abs(aa.y_pred.to_numpy(float)-rr.y_pred.to_numpy(float))))
  row=dict(backbone_id=b['id'],input_set=b['input_set'],backbone=b['backbone'],h=b['horizon'],seed=b['seed'],variant=v,split=sp,passed=delta<=1e-6,max_abs_prediction_difference=delta,started_utc=start,ended_utc=utc())
  rows.append(row)
 csv(OUT/'replay'/f"{sp}_{b['id']}.csv",pd.DataFrame(rows));tele('replay_end',backbone=b['id'],split=sp)
 if not all(r['passed'] for r in rows):raise RuntimeError('STOP: cached aggregator replay failed '+b['id'])
 return dict(started_utc=start,ended_utc=utc(),wall_seconds=time.perf_counter()-clock,rows=rows)

def training_function():
 global PREPARED,TRAIN_CODE
 if PREPARED:return TRAIN_CODE
 file=ROOT/'scripts/python/panel_pathformer_main_v2.py';tree=ast.parse(file.read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_cell'))
 stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='test_mask' for t in n.targets))
 body=node.body[:stop]
 ret=ast.parse("return dict(status='COMPLETE', model=model, history=history, best_epoch=best_epoch, epochs=len(history), best_val_loss=min_val_loss, runtime_seconds=runtime_seconds)").body[0]
 handler=ast.ExceptHandler(type=ast.Name(id='Exception',ctx=ast.Load()),name='training_error',body=ast.parse("return dict(status='DIVERGED', error=repr(training_error), captured=locals())").body)
 node.body=[ast.Try(body=body+[ret],handlers=[handler],orelse=[],finalbody=[])];node.name='train_cached_prefix'
 mod=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));TRAIN_CODE=compile(mod,str(file),'exec');PREPARED=True
 return TRAIN_CODE
def train_task(task):
 global STAGE
 b,v,lr=task;STAGE='train';check_protocol();folder=job_path(b,v,lr)
 if (folder/'COMPLETE.json').exists():return verify_job(folder/'COMPLETE.json')
 start=utc();clock=time.perf_counter();tele('training_start',job=job_id(b,v,lr))
 tr,tk=loader(b,'train',shuffle=True);vl,vk=loader(b,'val');models=[];frozen=[];initial=[];val_mse=[];hooks=[];seen={'n':0,'s':0.};last_tick=[0.]
 def construct(*a,**kw):
  m=f.configure_variant(b,v,'head_frozen',np.load(ROOT/b['mean']['path']));models.append(m);frozen.append(f.param_sha(m,False));initial.append(f.param_sha(m,True))
  def guard(mod,args):
   assert all(not branch.input_proj.training and not branch.positional_encoding.training and not branch.experts.training for branch in [m.daily_branch,m.weekly_branch])
   assert not m.fusion.training
   if time.monotonic()-last_tick[0]>300:tele('training_heartbeat',job=job_id(b,v,lr));last_tick[0]=time.monotonic()
  hooks.append(m.register_forward_pre_hook(guard));return m
 orig_huber=torch.nn.HuberLoss
 def huber(*a,**kw):
  loss=orig_huber(*a,**kw)
  def observe(mod,args,value):
   if not models[0].training:
    p,y=args;diff=p.detach()-y.detach();seen['s']+=float((diff*diff).sum(dtype=torch.float64));seen['n']+=len(y)
    if seen['n']==COUNTS['val']:val_mse.append(seen['s']/seen['n']);seen.update(n=0,s=0.)
  hooks.append(loss.register_forward_hook(observe));return loss
 env=dict(f.formal.__dict__);env.update(load_frozen_panel=lambda:None,_formal_check_dataset_contract=lambda ds:None,_build_training_loaders=lambda *a,**kw:(tr,vl,None),_make_model=construct,SEED=b['seed'],LEARNING_RATE=lr,MAX_EPOCHS=50,EARLY_STOPPING_PATIENCE=10)
 exec(training_function(),env)
 try:
  torch.nn.HuberLoss=huber
  result=env['train_cached_prefix']('daily_weekly',b['horizon'],'adaptive')
 finally:
  torch.nn.HuberLoss=orig_huber
  for h in hooks:h.remove()
 if result['status']=='DIVERGED':
  captured=result.pop('captured');m=captured.get('model');hist=captured.get('history',[]);best=int(captured.get('best_epoch',0));loss=float(captured.get('min_val_loss',float('inf')));epochs=len(hist)
  if m is not None and best:m.load_state_dict(captured['best_state'])
 else:m=result.pop('model');hist=result.pop('history');best=result['best_epoch'];loss=result['best_val_loss'];epochs=result['epochs']
 assert m is not None and f.param_sha(m,False)==frozen[0]
 m.eval();history=pd.DataFrame(hist,columns=['frequency_setting','horizon','seed','epoch','train_loss','val_loss']);history['val_mse']=val_mse[:len(history)];history['loss_type']='Huber delta=1, raw target'
 csv(folder/'history.csv',history)
 checkpoint=dict(trainable_state={k:p.detach().cpu().clone() for k,p in m.named_parameters() if p.requires_grad},constant_train_mean=torch.from_numpy(np.load(ROOT/b['mean']['path']).copy()) if v=='constant_gate' else None,backbone=b['checkpoint'],backbone_sha256=b['checkpoint_sha256'],variant=v,setting='head_frozen',best_epoch=best,seed=b['seed'],lr=lr,protocol_sha256=check_protocol())
 with target(folder/'best_aggregator.pt').open('xb') as fh:torch.save(checkpoint,fh)
 restored_mse=None
 if best:
  predictions,_,_=f.exp4._batched_inference(m,vl,COUNTS['val'],torch.device('cpu'))
  restored_mse=float(np.mean((predictions.astype(float)-vk.y_true.to_numpy(float))**2))
 j=dict(id=job_id(b,v,lr),backbone_id=b['id'],input_set=b['input_set'],backbone=b['backbone'],h=b['horizon'],seed=b['seed'],variant=v,lr=lr,status=result['status'],error=result.get('error'),eligible=result['status']=='COMPLETE' and restored_mse is not None and np.isfinite(restored_mse),validation_mse=restored_mse,best_epoch=best,epochs=epochs,best_val_huber=loss if np.isfinite(loss) else None,started_utc=start,ended_utc=utc(),wall_seconds=time.perf_counter()-clock,profile=f.profile(),initial_trainable_sha256=initial[0],frozen_sha256=frozen[0],frozen_unchanged=True,trainable_parameters=f.formal.count_trainable_parameters(m),test_reads=0,artifacts=[artifact(folder/'history.csv'),artifact(folder/'best_aggregator.pt')])
 js(folder/'COMPLETE.json',j);tele('training_end',job=j['id'],status=j['status'],epochs=epochs);print('FIT',j['id'],j['status'],epochs,restored_mse,flush=True);return j

def evaluation_task(task):
 global STAGE
 b,v,lr=task;STAGE='final_test_evaluation';check_protocol();tele('evaluation_start',backbone=b['id'],variant=v);start=utc();clock=time.perf_counter()
 folder=OUT/'evaluation'/f"{b['id']}_{v}"
 if (folder/'COMPLETE.json').exists():return verify_job(folder/'COMPLETE.json')
 m=configured(b,v);status='COMPLETE'
 if v!='fixed':
  j=verify_job(job_path(b,v,lr)/'COMPLETE.json');status=j['status']
  if status=='COMPLETE':load_state(m,job_path(b,v,lr)/'best_aggregator.pt')
 ld,k=loader(b,'test')
 if status=='COMPLETE':
  p,dw,ww=f.exp4._batched_inference(m,ld,COUNTS['test'],torch.device('cpu'))
 else:p=np.full(len(k),np.nan);dw=ww=np.full((len(k),4),np.nan)
 frame=k.assign(y_pred=p)
 for i,name in enumerate(WEIGHTS):frame[name]=(dw if i<4 else ww)[:,i%4]
 for key,value in dict(input_set=b['input_set'],backbone=b['backbone'],h=b['horizon'],seed=b['seed'],variant=v,learning_rate=lr,status=status).items():frame[key]=value
 csv(folder/'predictions_and_weights.csv',frame)
 date=k[['ticker','anchor_date']].copy();y=k.y_true.to_numpy(float);valid=y!=0
 if status=='COMPLETE':
  ranks=f.common.rank_ic_by_date(y,p,k.ticker.to_numpy(),k.anchor_date.to_numpy())
  loss=(p.astype(float)-y)**2;hits=(np.sign(p)==np.sign(y))&valid
  date['squared_loss']=loss;date['n_valid']=valid.astype(int);date['n_correct']=hits.astype(int)
  date=date.groupby('anchor_date',as_index=False).agg(mse=('squared_loss','mean'),n_valid=('n_valid','sum'),n_correct=('n_correct','sum')).merge(ranks,on='anchor_date',validate='one_to_one')
  mse=float(loss.mean());rank=float(ranks.rank_ic.mean()) if ranks.rank_ic.notna().any() else None;da=float(hits.sum()/valid.sum())
 else:
  date=k.groupby('anchor_date',as_index=False).size().rename(columns={'size':'n_tickers'});date['mse']=np.nan;date['rank_ic']=np.nan;date['n_valid']=k.assign(v=valid).groupby('anchor_date').v.sum().to_numpy();date['n_correct']=np.nan;mse=rank=da=None
 csv(folder/'date_metrics.csv',date)
 router=[];route_dates=[];old=b['old'][v]['predictions']['test'];reference=pd.read_parquet(ROOT/old['path'],columns=['ticker','anchor_date']+WEIGHTS)
 assert reference[['ticker','anchor_date']].assign(anchor_date=lambda d:pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d')).equals(k[['ticker','anchor_date']])
 for regime,weights in [('tuned',np.concatenate([dw,ww],axis=1)),('Step16_lr1e-4',reference[WEIGHTS].to_numpy())]:
  for branch,z in [('daily',weights[:,:4]),('weekly',weights[:,4:])]:
   finite=np.isfinite(z).all();constant=bool(np.max(np.ptp(z,axis=0))<1e-7) if finite else False
   if finite:
    d=pd.DataFrame(z);by=d.groupby(k.anchor_date.to_numpy()).transform('mean');total=float(d.var().sum());between=float(by.var().sum());share=between/total if total>0 and not constant else None
    mad=float(np.abs(z-.25).mean());sd=float(z.std(0).mean()) if not constant else 0.
   else:mad=sd=share=None;total=between=None
   router.append(dict(regime=regime,branch=branch,mean_abs_deviation_from_equal=mad,mean_sample_sd=sd,between_date_share=share,total_variance=total,between_date_variance=between,constant_to_1e7=constant))
   if finite:
    grouped=pd.DataFrame(z,columns=[branch+'_w'+str(i) for i in range(4)]).assign(anchor_date=k.anchor_date.to_numpy()).groupby('anchor_date',as_index=False).mean();grouped['regime']=regime;grouped['branch']=branch;route_dates.append(grouped)
 csv(folder/'routing_diagnostics.csv',pd.DataFrame(router));csv(folder/'routing_by_date.csv',pd.concat(route_dates,ignore_index=True) if route_dates else pd.DataFrame())
 j=dict(input_set=b['input_set'],backbone=b['backbone'],h=b['horizon'],seed=b['seed'],variant=v,lr=lr,status=status,mse=mse,mean_rank_ic=rank,da=da,n_zero_realized=int((~valid).sum()),n_valid=int(valid.sum()),started_utc=start,ended_utc=utc(),wall_seconds=time.perf_counter()-clock,profile=f.profile(),test_inference_passes=int(status=='COMPLETE'),artifacts=[artifact(folder/name) for name in ['predictions_and_weights.csv','date_metrics.csv','routing_diagnostics.csv','routing_by_date.csv']])
 js(folder/'COMPLETE.json',j);tele('evaluation_end',backbone=b['id'],variant=v);return j
