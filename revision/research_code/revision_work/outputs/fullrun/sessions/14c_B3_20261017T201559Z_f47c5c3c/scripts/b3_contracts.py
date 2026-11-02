"""Resume contracts for B3; no existing evidence is modified."""
from b3_io import *
def dependencies():
 manifest=json.loads((EXP/'tensor_manifest.json').read_text())
 paths=[EXP/n for n in ['experiment.json','causal_reference_policy.json','tensor_manifest.json','normalization_stats.json','PADDING_CHECKS.json','baseline_provenance.json']]+sorted(HERE.rglob('*.py'))
 original=json.loads((EXP/'design_diagnostics.json').read_text())['dependencies']
 return list({d['path']:d for d in original+manifest['files']+[artifact(p) for p in paths]}.values())
def verify_completion(path,expected_job=None):
 r=json.loads(Path(path).read_text());assert r['validated'];spec=spec_for(r['job_id'])
 if expected_job:
  for k in KEYS:assert r[k]==expected_job[k]
 assert r['frequency_setting']==spec['frequency_setting'] and r['row_counts']==COUNTS
 assert r['thread_counts_start']==r['thread_counts_end']==[12,12] and not r['thread_counts_changed']
 for env in [r['environment_start'],r['environment_end']]:assert env['OMP_NUM_THREADS'] is None and env['OMP_DYNAMIC']==env['MKL_DYNAMIC']=='FALSE'
 assert r['loader_generator_seed']==42 and r['strict_checkpoint_reload'] and r['mask_transport_checked']
 assert r['causal_reference_policy']==json.loads((EXP/'causal_reference_policy.json').read_text())['policy']
 for a in r['artifacts']+r['dependencies']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
 assert r['checkpoint'] in [a['path'] for a in r['artifacts']]
 return r
def completed(spec):
 for p in sorted((EXP/'runs'/spec['job_id']).glob('attempt_*/COMPLETE.json')):
  try:return p,verify_completion(p,spec)
  except (AssertionError,OSError,ValueError,KeyError):continue
 return None,None
