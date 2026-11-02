"""Exclusive B5 contracts; standard library only."""
import os,sys,json,csv
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=HERE.parents[5]
sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
from fullrun14a_io import target,jsonout,textout,csvout,utc,unique,sha,rel,profile_environment,ENV_KEYS,COUNTS,PYTHON,table
FULLRUN=ROOT/'revision_work/outputs/fullrun'
CONFIG=json.loads((EXP/'experiment.json').read_text());PLAN=CONFIG['plan'];PILOT=CONFIG['pilot'];DAY=ROOT/CONFIG['day_session'];B3=ROOT/CONFIG['wait_for_b3']
KEYS=['job_id','input_set','mechanism','horizon','seed','queue']
SEEDS=[42,0,1,21,3407]
def artifact(p):
 p=Path(p);return dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p))
def spec_for(i):return next(s for s in [PILOT]+PLAN if s['job_id']==i)
def jobs(phase='main'):return [{k:r[k] for k in KEYS} for r in ([PILOT] if phase=='pilot' else PLAN)]
def verify_completion(path,expected_job=None):
 r=json.loads(Path(path).read_text());spec=spec_for(r['job_id']);assert r['validated']
 for k in KEYS:assert r[k]==spec[k]
 if expected_job:
  for k in KEYS:assert r[k]==expected_job[k]
 assert r['row_counts']==COUNTS and r['loader_generator_seed'] is None
 assert r['global_rng_seed']==r['seed'] and r['seed_calls']==[r['seed']]
 assert r['thread_counts_start']==r['thread_counts_end']==[12,12] and not r['thread_counts_changed']
 for env in [r['environment_start'],r['environment_end']]:assert env['OMP_NUM_THREADS'] is None and env['OMP_DYNAMIC']==env['MKL_DYNAMIC']=='FALSE'
 assert r['test_replay']['classification']=='EXACT' and r['test_replay']['max_abs_delta_pred']<1e-7
 assert r['input_check']['passed'] and r['input_check']['input_set']==r['input_set']
 for a in r['artifacts']+r['dependencies']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
 assert r['checkpoint'] in [a['path'] for a in r['artifacts']]
 return r
def completed(spec):
 for p in sorted((EXP/'runs'/spec['job_id']).glob('attempt_*/COMPLETE.json')):
  try:return p,verify_completion(p,spec)
  except (AssertionError,OSError,ValueError,KeyError):continue
 return None,None
