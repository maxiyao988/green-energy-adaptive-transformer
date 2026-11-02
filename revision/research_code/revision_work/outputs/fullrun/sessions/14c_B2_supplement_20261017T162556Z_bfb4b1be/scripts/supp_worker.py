"""Reuse the complete B2/14a worker and formal Experiment4 trainer."""
from supp_io import *
import b2_worker as original_worker
from b2_data import base,formal,load_panel,window_contract
import argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--job-id');ap.add_argument('--attempt');a=ap.parse_args()
 if a.preflight:assert json.loads((EXP/'preflight.json').read_text())['passed'];return
 spec=spec_for(a.job_id);assert spec['job_id'] not in OLD_IDS and spec in NEW
 attempt=Path(a.attempt).absolute();assert attempt.is_relative_to(EXP/'runs'/spec['job_id'])
 w=spec['window']
 try:
  base.OUT=EXP;base.COUNTS=counts(w);base.dataset=lambda inp:load_panel(w,inp)
  base.dependencies=lambda inp:list({d['path']:d for d in original_worker.deps(inp,w)+extra_dependencies()}.values())
  base.verify_completion=verify_completion
  formal._formal_check_dataset_contract=lambda ds:window_contract(ds,w)
  base.exp4.N_TRAIN_SAMPLES=counts(w)['train']
  base.run(spec,attempt)
 except BaseException as e:
  if not (attempt/'FAILED.json').exists():jsonout(attempt/'FAILED.json',dict(utc=utc(),job_id=spec['job_id'],error=repr(e)))
  raise
if __name__=='__main__':main()
