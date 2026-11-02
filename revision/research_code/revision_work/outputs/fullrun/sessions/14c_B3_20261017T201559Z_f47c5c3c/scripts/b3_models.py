"""Imported formal models with non-parametric input/attention-mask wiring only."""
from b3_io import *
import fullrun14a_worker_v2 as base
import torch
formal=base.formal
FACTORY=formal._make_model

class FrequencyInterface(torch.nn.Module):
 """Adapt the formal single-resolution signature to imported dual-input trainers."""
 def __init__(self,model,freq):super().__init__();self.inner=model;self.frequency_setting=freq
 def forward(self,xd,xw):
  return self.inner(xd if self.frequency_setting=='daily_only' else xw)
 def forward_with_router(self,xd,xw):
  p,w=self.inner.forward_with_router(xd if self.frequency_setting=='daily_only' else xw)
  inactive=torch.zeros_like(w)
  return (p,w,inactive) if self.frequency_setting=='daily_only' else (p,inactive,w)

def wire_daily_mask(model):
 """Thread valid tokens into original expert.forward; exclude empty patch keys.

 Stored standardized padding is exactly zero. The builder proves every real token
 is nonzero and valid tokens form a contiguous prefix, so this is an exact mask
 transport through the unchanged formal TensorDataset/DataLoader signatures.
 No parameter, attention formula, aggregation rule or prediction head is added.
 """
 for branch in [m for m in model.modules() if isinstance(m,formal.FrequencyAdaptivePathFormerBranch) and m.frequency=='daily']:
  context={}
  def branch_pre(module,args,context=context):
   x=args[0];valid=x.ne(0).any(dim=-1)
   override=getattr(module,'_b3_valid_mask_override',None)
   if override is not None:valid=override.to(device=x.device,dtype=torch.bool)
   assert valid.shape==x.shape[:2] and valid.any(dim=1).all()
   context['valid']=valid
   return (x.masked_fill(~valid.unsqueeze(-1),0),)+args[1:]
  branch.register_forward_pre_hook(branch_pre)
  for expert in branch.experts:
   state={}
   def expert_pre(module,args,kwargs,context=context,state=state):
    valid=context['valid'];kwargs=dict(kwargs,valid_mask=valid)
    pad=(-valid.shape[1])%module.patch_size
    occupied=torch.nn.functional.pad(valid,(0,pad),value=False).reshape(valid.shape[0],-1,module.patch_size).any(-1)
    state['occupied']=occupied
    return args,kwargs
   expert.register_forward_pre_hook(expert_pre,with_kwargs=True)
   def intra_pre(module,args,kwargs):
    # The dummy key is discarded by the original valid-token mask/pooling. It
    # only avoids undefined all-masked softmax for completely empty patches.
    invalid=kwargs['key_padding_mask'];empty=invalid.all(dim=1)
    if empty.any():
     invalid=invalid.clone();invalid[empty,0]=False;kwargs=dict(kwargs,key_padding_mask=invalid)
    return args,kwargs
   expert.intra_attn.register_forward_pre_hook(intra_pre,with_kwargs=True)
   def inter_pre(module,args,kwargs,state=state):return args,dict(kwargs,key_padding_mask=~state['occupied'])
   expert.inter_attn.register_forward_pre_hook(inter_pre,with_kwargs=True)
 return model

def make_model(freq,mechanism,dual_interface=False):
 model=wire_daily_mask(FACTORY(freq,mechanism))
 return FrequencyInterface(model,freq) if dual_interface and freq!='daily_weekly' else model

def padding_checks():
 formal.set_seed(42);rows=[]
 # Padded and cropped branches have the same parameters and original positional
 # indices for every real token. Compare each imported patch expert and full model.
 for mechanism in ['static','adaptive']:
  for freq in ['daily_only','daily_weekly']:
   model=make_model(freq,mechanism);model.eval();branch=model.branch if freq=='daily_only' else model.daily_branch
   for length in range(122,128):
    real=torch.randn(2,length,5);x=torch.nn.functional.pad(real,(0,0,0,127-length));w=torch.randn(2,26,5)
    with torch.no_grad():
     p=model(x) if freq=='daily_only' else model(x,w)
     q=model(real) if freq=='daily_only' else model(real,w)
    delta=float((p-q).abs().max());assert torch.isfinite(p).all() and delta<1e-6,(mechanism,freq,length,delta)
    branch._b3_valid_mask_override=torch.arange(127)[None,:].expand(2,-1)<length
    noisy=x.clone();noisy[:,length:]=torch.randn_like(noisy[:,length:])*100
    with torch.no_grad():r=model(noisy) if freq=='daily_only' else model(noisy,w)
    del branch._b3_valid_mask_override
    perturb=float((p-r).abs().max());assert perturb==0.,(freq,length,perturb)
    model.train();model.zero_grad(set_to_none=True);p=model(x) if freq=='daily_only' else model(x,w);p.square().mean().backward()
    finite=all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters());assert finite
    model.eval();rows.append(dict(mechanism=mechanism,frequency_setting=freq,valid_length=length,padded_length=127,max_abs_delta_vs_cropped=delta,max_abs_delta_after_padding_perturbation=perturb,finite_forward_backward=bool(finite),trainable_parameters=formal.count_trainable_parameters(model)))
 for freq in ['daily_only','weekly_only','daily_weekly']:
  for mechanism in ['static','adaptive']:
   model=make_model(freq,mechanism);original=FACTORY(freq,mechanism)
   assert formal.count_trainable_parameters(model)==formal.count_trainable_parameters(original)
 return rows

if __name__=='__main__':
 rows=padding_checks();csvout(EXP/'padding_adapter_checks.csv',rows)
 jsonout(EXP/'PADDING_CHECKS.json',dict(utc=utc(),passed=True,n_checks=len(rows),tolerance_vs_cropped=1e-6,padding_perturbation_delta_exact_zero=True,finite_forward_backward=True,parameter_counts_unchanged=True,implementation='Imported formal models plus forward hooks for explicit valid-token propagation, empty intra-patch dummy keys and inter-patch key masking; no learned model reimplementation'))
 print('PADDING_ADAPTER_PASS',len(rows),flush=True)
