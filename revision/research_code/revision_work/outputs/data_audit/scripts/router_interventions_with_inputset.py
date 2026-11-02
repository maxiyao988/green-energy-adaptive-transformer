"""Non-destructive input-set adapter for the original router intervention suite.

Smoke (the only operation executed for Step13):
  nice -n 10 python -B router_interventions_with_inputset.py --input-set causal --smoke-only
Future suite interface (NOT executed in this audit): original script's arguments, plus --input-set.
All outputs are restricted to data_audit/router_interventions/<run-label>/; existing writers stay exclusive.
"""
import os,sys,re,types,runpy,importlib.util,argparse
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--input-set',choices=['original','causal'],required=True);p.add_argument('--smoke-only',action='store_true');a,rest=p.parse_known_args()
 if a.smoke_only:
  script=Path(__file__).with_name('router_interventions_inputset_v2.py');sys.argv=[str(script),'--input-set',a.input_set]+rest
  runpy.run_path(str(script),run_name='__main__');return
 sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
 spec=importlib.util.spec_from_file_location('original_intervention_suite',ROOT/'revision_work/scripts/interventions/router_interventions.py');iv=importlib.util.module_from_spec(spec)
 klass=type(os.environ);orig_set=klass.__setitem__
 def capped(self,key,value):
  if key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):value='1'
  return orig_set(self,key,value)
 klass.__setitem__=capped
 try:spec.loader.exec_module(iv)
 finally:klass.__setitem__=orig_set
 iv.OUT_BASE=OUT/'router_interventions'
 if '--out-root' not in rest:rest+=['--out-root',str(iv.OUT_BASE)]
 # Writer validation continues to prohibit outputs outside the new Step13 namespace.
 if a.input_set=='causal':
  import core
  iv.formal.load_frozen_panel=core.load_causal
  base=ROOT/'revision_work/outputs/causal_features'
  if '--checkpoints' not in rest:rest+=['--checkpoints',str(base/'checkpoints/adaptive_dw_h*_s*.pt'),str(base/'stage3b/checkpoints/adaptive_dw_h*_s*.pt')]
  if '--summary-csv' not in rest:rest+=['--summary-csv','none']
  # Adapt only the original checkpoint-name parser; model factories and inference are unchanged imports.
  oldsearch=re.search
  def search(pattern,string,*args,**kwargs):
   if pattern==r'adaptive_seed(\d+)_(\d+)d\.pt$':
    found=oldsearch(r'adaptive_dw_h(\d+)_s(\d+)\.pt$',str(string))
    if found:string=f'adaptive_seed{found[2]}_{found[1]}d.pt'
   return oldsearch(pattern,string,*args,**kwargs)
  iv.re=types.SimpleNamespace(**{k:getattr(re,k) for k in dir(re) if not k.startswith('__')});iv.re.search=search
  # The original suite formats one reference template. Resolve its h/seed tokens to the two
  # existing causal directories without copying or changing saved predictions.
  if '--reference-preds' not in rest:
   rest+=['--reference-preds',str(OUT/'causal_reference_h{h}_s{seed}.parquet')]
   oldload=iv.load_pred
   def load_ref(path,*args,**kwargs):
    m=re.search(r'causal_reference_h(\d+)_s(\d+)\.parquet$',str(path))
    if m:
     h,seed=map(int,m.groups());b=base if seed==42 else base/'stage3b';path=b/f'preds/adaptive_dw_h{h}_s{seed}_test.parquet'
    return oldload(path,*args,**kwargs)
   iv.load_pred=load_ref
 sys.argv=[str(ROOT/'revision_work/scripts/interventions/router_interventions.py')]+rest
 iv.main()
if __name__=='__main__':main()
