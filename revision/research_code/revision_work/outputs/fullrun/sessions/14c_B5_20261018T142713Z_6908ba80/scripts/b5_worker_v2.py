"""Import the 14c-day B4 runner; change only input binding and explicit run seed."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
from b5_io import *
sys.path.insert(0,str(DAY/'scripts'))
# Ensure cached day_io imports cannot change resolution of the original worker.
sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
import day_worker_v2 as day
assert Path(day.base.__file__).resolve()==ROOT/'revision_work/scripts/causal/fullrun14a_worker_v2.py'
assert Path(day.__file__).resolve()==DAY/'scripts/day_worker_v2.py'
import torch,numpy as np,pandas as pd,argparse
from contextlib import ExitStack
ORIGINAL_CAUSAL=day.core.load_causal
ORIGINAL_CHECK=day.causal_check
ORIGINAL_DEPS=day.deps

def dependencies(inp):
 ds=ORIGINAL_DEPS(inp)+[artifact(EXP/'experiment.json')]+[artifact(p) for p in sorted(HERE.rglob('*.py'))]
 return list({d['path']:d for d in ds}.values())

def input_check(ds,inp):
 if inp=='C':r=ORIGINAL_CHECK(ds)
 else:
  frozen=day.common.load_frozen_panel()
  for field in ['ticker','split','X_daily','X_weekly']:assert np.array_equal(getattr(ds,field),getattr(frozen,field)),field
  assert np.array_equal(pd.to_datetime(ds.anchor_date).to_numpy(),pd.to_datetime(frozen.anchor_date).to_numpy())
  for h in [5,10,20]:assert np.array_equal(ds.y[h],frozen.y[h])
  r=dict(passed=True,binding='common.load_frozen_panel',frozen_original_raw_tensors_bitwise_equal=True,keys_labels_equal=True)
  normal=day.common.apply_normalizer(ds,day.common.fit_train_only_normalizer(ds),'daily_weekly')
  r['normalized']=[dict(branch=b,shape=list(a.shape),dtype=str(a.dtype),array_sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest()) for b,a in zip(['daily','weekly'],normal)]
 r.update(passed=True,input_set=inp)
 assert {s:int((ds.split==s).sum()) for s in COUNTS}==COUNTS
 return r

def run(spec,attempt):
 ds=ORIGINAL_CAUSAL() if spec['input_set']=='C' else day.common.load_frozen_panel()
 day.formal._formal_check_dataset_contract(ds)
 check=input_check(ds,spec['input_set']);seed=int(spec['seed']);calls=[]
 owner={'lstm':day.lstm,'vanilla':day.vanilla,'swim':day.swim}[spec['mechanism']]
 original_seed=owner.set_seed;original_jsonout=day.jsonout
 def seed_once(value):
  assert value==seed,(value,seed);calls.append(value);return original_seed(value)
 def clean(value):
  if isinstance(value,dict):return {('bitwise_equal_input_features' if k=='bitwise_equal_causal_features' else k):clean(v) for k,v in value.items()}
  if isinstance(value,list):return [clean(v) for v in value]
  return value
 def save(path,value):
  path=Path(path)
  if path.name=='causal_input_check.json':path=path.with_name('input_check.json');value=clean(value)
  if path.name=='COMPLETE.json':
   assert calls==[seed],calls
   value=dict(value,global_rng_seed=seed,seed_calls=list(calls),shuffle_protocol=f'formal baseline global RNG seed{seed}; no independent DataLoader generator')
   value['input_check']=clean(value.pop('causal_input_check'))
  return original_jsonout(path,value)
 def bind(stack,obj,key,value):
  old=getattr(obj,key);setattr(obj,key,value);stack.callback(setattr,obj,key,old)
 with ExitStack() as stack:
  bind(stack,day,'EXP',EXP);bind(stack,day,'deps',lambda ignored:dependencies(spec['input_set']))
  bind(stack,day,'causal_check',lambda actual:dict(check));bind(stack,day.core,'load_causal',lambda:ds)
  bind(stack,day,'verify_completion',verify_completion);bind(stack,day,'jsonout',save)
  bind(stack,owner,'SEED',seed);bind(stack,owner,'set_seed',seed_once)
  if spec['mechanism']=='swim':bind(stack,day.swim.formal,'SEED',seed)
  day.run_b4(spec,attempt)

def preflight():
 day.base.check_profile();checks=[]
 for inp in ['O','C']:
  ds=ORIGINAL_CAUSAL() if inp=='C' else day.common.load_frozen_panel();day.formal._formal_check_dataset_contract(ds)
  checks.append(input_check(ds,inp))
 assert json.loads((EXP/'B3_ACCEPTANCE.json').read_text())['passed']
 jsonout(EXP/'preflight.json',dict(utc=utc(),passed=True,dependencies=dependencies('C'),input_checks=checks,thread_counts=[torch.get_num_threads(),torch.get_num_interop_threads()],environment={k:os.environ.get(k) for k in ENV_KEYS},global_rng_protocol='original entrypoint set_seed(run_seed); original generator=None loaders'))
 print('B5_PREFLIGHT_PASSED',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--job-id');p.add_argument('--attempt');a=p.parse_args()
 if a.preflight:return preflight()
 spec=spec_for(a.job_id);attempt=Path(a.attempt).absolute();assert attempt.is_relative_to(EXP/'runs'/spec['job_id'])
 try:run(spec,attempt)
 except BaseException as e:
  if not (attempt/'FAILED.json').exists():jsonout(attempt/'FAILED.json',dict(utc=utc(),job_id=spec['job_id'],error=repr(e)))
  raise
if __name__=='__main__':main()
