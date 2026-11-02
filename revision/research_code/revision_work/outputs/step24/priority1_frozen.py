"""Inference-only sample cache checks; no optimizer.step, backward or fit entrypoint."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step24';os.environ['TMPDIR']=str(OUT/'scratch');(OUT/'scratch').mkdir(exist_ok=False)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
sys.path.insert(0,str(ROOT/'revision_work/outputs/frozen/scripts'));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import frozen_worker_v3 as worker
import torch,numpy as np,pandas as pd,json,hashlib
# Imported module source under review is used only as a reconstruction adapter; independent checks own comparisons.
from review24_common import read,js,df,pq,write,source_manifest
# Fail closed on any attempted source or cache write by imported code.
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0])).absolute().resolve();mode=args[1];flags=args[2] if len(args)>2 else 0
  writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not path.is_relative_to(OUT.resolve()):raise RuntimeError('Review forbids external write: '+str(path))
sys.addaudithook(guard)
checks=[];details=[];FR=ROOT/'revision_work/outputs/frozen';B2=ROOT/'revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041/tensors/W3';specs=js(FR/'backbone_registry.json');bindings=js(ROOT/'revision_work/outputs/step22a/BINDINGS.json');predfiles=df(ROOT/'revision_work/outputs/step22a/prediction_files.csv')
def check(name,passed,delta=0.,**meta):checks.append(dict(check=name,passed=bool(passed),max_abs_difference=float(delta),**meta))
for inp,bk,h in [('O','S1',5),('C','S1',20),('O','S2',20),('C','S2',5)]:
 spec=next(s for s in specs if s['input_set']==inp and s['backbone']==bk and s['horizon']==h and s['seed']==0);b=next(s for s in bindings if s['id']==spec['id']);base=worker.model_for(spec['mechanism']);read(ROOT/spec['checkpoint']);base.load_state_dict(torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=True),strict=True);base.eval()
 meta=df(B2/'sample_index.csv');raw=np.load(B2/inp/'X_daily_standardized.npy',mmap_mode='r');week=np.load(B2/inp/'X_weekly_standardized.npy',mmap_mode='r')
 for sp in ['train','val','test']:
  ids=np.flatnonzero(meta.split.eq(sp).to_numpy())[:32];xd=torch.from_numpy(np.array(raw[ids]));xw=torch.from_numpy(np.array(week[ids]));observed=[];hooks=[]
  for branch in [base.daily_branch,base.weekly_branch]:hooks.append(branch.aggregator.register_forward_pre_hook(lambda mod,args:observed.append(torch.stack(args[0],dim=1).detach().cpu())))
  with torch.inference_mode():p,wd,ww=base.forward_with_router(xd,xw)
  for hook in hooks:hook.remove()
  fresh=torch.cat(observed,dim=1).numpy();cache=np.load(ROOT/b['cache'][sp]['experts'],mmap_mode='r');read(ROOT/b['cache'][sp]['experts']);delta=float(np.max(np.abs(fresh-cache[:32])));check('Fresh experts equal cache sample',delta<=1e-10,delta,backbone_id=b['id'],split=sp)
 mu=np.load(ROOT/b['mean']['path']);read(ROOT/b['mean']['path'])
 for v in ['fixed','static','constant_gate','adaptive']:
  m=worker.configure_variant(spec,v,'head_frozen',mu);m.train(True);frozen=[(name,p) for name,p in m.named_parameters() if not p.requires_grad];trainable=[(name,p) for name,p in m.named_parameters() if p.requires_grad]
  modes=all(not branch.input_proj.training and not branch.positional_encoding.training and not branch.experts.training for branch in [m.daily_branch,m.weekly_branch]) and not m.fusion.training
  check('Frozen projections/experts/head eval after train(True)',modes,backbone_id=b['id'],variant=v);check('Only aggregator requires_grad',all('.aggregator.' in n for n,p in trainable),backbone_id=b['id'],variant=v)
  n=sum(p.numel() for n,p in trainable);check('Aggregator parameter count',n==dict(fixed=0,static=8,constant_gate=8520,adaptive=8520)[v],backbone_id=b['id'],variant=v)
  # The actual formal trainer constructs Adam(model.parameters()), not a filtered parameter iterator.
  allparams=sum(p.numel() for p in m.parameters());details.append(dict(backbone_id=b['id'],variant=v,setting='head_frozen',trainable_parameters=n,formal_optimizer_parameter_count=allparams,frozen_parameters_present_in_optimizer=allparams-n,requires_grad_frozen_false=all(not p.requires_grad for _,p in frozen),optimizer_state_for_frozen='No gradient => Adam skips; no optimizer step executed in review'))
  if v=='constant_gate':
   with torch.inference_mode():
    ca=torch.from_numpy(np.array(cache[:32,:4]));cw=torch.from_numpy(np.array(cache[:32,4:]));a=m.daily_branch.aggregator._adaptive_weights(list(ca.unbind(1)));z=m.daily_branch.aggregator._adaptive_weights(list((ca*10).unbind(1)))
   delta=float(torch.max(torch.abs(a-z)));check('Constant gate input ignores sample expert changes',delta==0,delta,backbone_id=b['id'])
  # Restore selected Step22A weights and compare TEST cached sample with stored TEST predictions.
  old=v=='fixed'
  if old:cp=ROOT/b['old'][v]['checkpoint']
  else:
   rate=json.loads((ROOT/'revision_work/outputs/step22a/SELECTED_RATES.json').read_text());rates=rate
   # Locate the actual selected best-aggregator checkpoint via prediction/evaluation record.
   row=predfiles[(predfiles.input_set==inp)&(predfiles.backbone==bk)&(predfiles.h==h)&(predfiles.seed==0)&(predfiles.variant==v)].iloc[0];ev=js((ROOT/row.path).parent/'COMPLETE.json');lr=float(ev['lr']);jid=f"{b['id']}_{v}_lr{lr:.0e}";cp=ROOT/'revision_work/outputs/step22a/fits'/jid/'best_aggregator.pt'
  read(cp);state=torch.load(cp,map_location='cpu',weights_only=True);parameters=dict(m.named_parameters());check('Checkpoint trains exactly declared parameter set',set(state['trainable_state'])=={name for name,p in parameters.items() if p.requires_grad},backbone_id=b['id'],variant=v)
  with torch.no_grad():
   for name,val in state['trainable_state'].items():parameters[name].copy_(val)
  m.eval();ca=torch.from_numpy(np.array(cache[:32,:4]));cw=torch.from_numpy(np.array(cache[:32,4:]));
  with torch.inference_mode():p,_,_=m.forward_with_router(ca,cw)
  row=predfiles[(predfiles.input_set==inp)&(predfiles.backbone==bk)&(predfiles.h==h)&(predfiles.seed==0)&(predfiles.variant==v)].iloc[0];saved=df(ROOT/row.path);keys=pq(ROOT/b['cache']['test']['keys'])[['ticker','anchor_date']].iloc[:32].copy();keys.anchor_date=pd.to_datetime(keys.anchor_date).dt.strftime('%Y-%m-%d');saved.anchor_date=pd.to_datetime(saved.anchor_date).dt.strftime('%Y-%m-%d');ref=keys.merge(saved,on=['ticker','anchor_date'],validate='one_to_one');delta=float(np.max(np.abs(p.numpy().astype(float)-ref.y_pred.to_numpy(float))));check('Selected22A aggregator replay sample',delta<=1e-10,delta,backbone_id=b['id'],variant=v)
# Factories and same-patch control: no training.
formal=worker.formal
for v in ['single','fixed','static','adaptive']:
 m=formal._make_model('daily_weekly',v,5,2);n=sum(p.numel() for p in m.parameters() if p.requires_grad);check('Formal model parameter count '+v,n==dict(single=30401,fixed=107777,static=107785,adaptive=116297)[v])
origd,origw=formal.DAILY_PATCH_SIZES,formal.WEEKLY_PATCH_SIZES
try:
 formal.DAILY_PATCH_SIZES=[10]*4;formal.WEEKLY_PATCH_SIZES=[4]*4;m=formal._make_model('daily_weekly','fixed');n=sum(p.numel() for p in m.parameters() if p.requires_grad);check('Same-patch Fixed parameter count',n==107777);check('Same-patch sizes all daily10 weekly4',all(e.patch_size==10 for e in m.daily_branch.experts) and all(e.patch_size==4 for e in m.weekly_branch.experts))
finally:formal.DAILY_PATCH_SIZES,formal.WEEKLY_PATCH_SIZES=origd,origw
write('PRIORITY1_FROZEN_CHECKS.csv',pd.DataFrame(checks));write('FROZEN_OPTIMIZER_MEMBERSHIP.csv',pd.DataFrame(details));source_manifest('SOURCES_FROZEN.csv');print('FROZEN',len(checks),'checks;',sum(not q['passed'] for q in checks),'failures');print([q for q in checks if not q['passed']])
