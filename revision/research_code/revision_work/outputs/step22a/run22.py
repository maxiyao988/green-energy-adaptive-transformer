"""Three-worker supervisor with strict global phase barriers and immutable journals."""
from io22 import *
import csv as csvlib,subprocess,traceback,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed

def initialize_worker():
 import worker22
 worker22.init()
def dispatch(kind,arg):
 import worker22
 return getattr(worker22,kind+'_task')(arg)
def event(name,**data):
 stamp=utc();text(OUT/'events'/f"{time.time_ns()}_{name}.json",json.dumps(dict(utc=stamp,event=name,**data),indent=2)+'\n');print(name,stamp,flush=True);return stamp
def phase(pool,name,kind,args):
 done=OUT/'phases'/f'{name}.json'
 if done.exists():return json.loads(done.read_text())['results']
 start=event(name+'_START');clock=time.perf_counter();results=[]
 tasks={pool.submit(dispatch,kind,a):a for a in args}
 for i,future in enumerate(as_completed(tasks),1):
  result=future.result();results.append(result)
  # One new status snapshot per completion, including replay's flag/max-delta only.
  status=target(OUT/'status'/f'{name}_{i:04d}.json');js(status,dict(utc=utc(),phase=name,completed=i,total=len(args),last_result=result))
  print('PROGRESS',name,i,'/',len(args),flush=True)
 end=event(name+'_END');js(done,dict(started_utc=start,ended_utc=end,wall_seconds=time.perf_counter()-clock,n_tasks=len(args),results=results));return results
def main():
 check_protocol()
 if (OUT/'REPORT_STEP22A.md').exists():print('Report already exists; refusing any rerun/overwrite.');return
 launch=event('SUPERVISOR_START',pid=os.getpid(),power=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout,free_bytes=__import__('shutil').disk_usage(ROOT).free)
 wake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);event('AWAKE',pid=wake.pid)
 try:
  event('CACHE_BINDING_START')
  bs=bindings() if (OUT/'BINDINGS.json').exists() else prepare_bindings()
  event('CACHE_BINDING_END')
  # Archive exactly the original training source prefix before any fit.
  if not (OUT/'FORMAL_TRAINING_PREFIX.txt').exists():
   import ast
   file=ROOT/'scripts/python/panel_pathformer_main_v2.py';source=file.read_text();tree=ast.parse(source);n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_cell');cut=next(v.lineno for v in n.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='test_mask' for t in v.targets));text(OUT/'FORMAL_TRAINING_PREFIX.txt','\n'.join(source.splitlines()[n.lineno-1:cut-1])+'\n')
  with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
   phase(pool,'cache','cache',bs)
   phase(pool,'validation_replay','replay',[(b,'val') for b in bs])
   phase(pool,'test_replay','replay',[(b,'test') for b in bs])
   event('ALL_TRAINING_START')
   seed0=[b for b in bs if b['seed']==0]
   tuning=phase(pool,'tuning','train',[(b,v,lr) for b in seed0 for v in VARIANTS for lr in GRID])
   selections=[]
   for b in seed0:
    for v in VARIANTS:
     trials=[j for j in tuning if j['backbone_id']==b['id'] and j['variant']==v]
     assert len(trials)==3 and len({j['initial_trainable_sha256'] for j in trials})==1
     eligible=[j for j in trials if j['eligible']]
     if not eligible:raise RuntimeError('STOP: all learning rates failed '+b['id']+' '+v)
     win=min(eligible,key=lambda j:(j['validation_mse'],j['lr']))
     selections.append(dict(input_set=b['input_set'],backbone=b['backbone'],h=b['horizon'],variant=v,lr=win['lr'],seed0_job=win['id'],validation_mse=win['validation_mse']))
   if not (OUT/'SELECTED_RATES.json').exists():js(OUT/'SELECTED_RATES.json',dict(locked_utc=utc(),selection='seed0 restored-best-Huber checkpoint validation MSE; ties lower LR',cells=selections,protocol_sha256=check_protocol()))
   else:assert json.loads((OUT/'SELECTED_RATES.json').read_text())['cells']==selections
   chosen={(j['input_set'],j['backbone'],j['h'],j['variant']):j['lr'] for j in selections}
   remaining=phase(pool,'remaining_seeds','train',[(b,v,chosen[(b['input_set'],b['backbone'],b['horizon'],v)]) for b in bs if b['seed']!=0 for v in VARIANTS])
   assert len(tuning)==108 and len(remaining)==144
   event('ALL_TRAINING_END',fits=252,test_reads_during_training=0)
  # New processes for TEST: training workers and their runtime states no longer exist.
  with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
   evaluations=phase(pool,'final_test_evaluation','evaluation',[(b,v,None if v=='fixed' else chosen[(b['input_set'],b['backbone'],b['horizon'],v)]) for b in bs for v in ['fixed']+VARIANTS])
  assert len(evaluations)==240
  event('ANALYSIS_START')
  p=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(OUT/'analyze22.py')],cwd=ROOT,env=profile_env(),check=True)
  event('ANALYSIS_END',returncode=p.returncode)
  protected=json.loads((OUT/'PROTECTED_INPUTS.json').read_text())
  for path,h in protected.items():assert sha(ROOT/path)==h,path
  js(OUT/'FINAL_SOURCE_PROTECTION.json',dict(utc=utc(),files_checked=len(protected),unchanged=True,protocol_sha256=check_protocol()))
  event('SUPERVISOR_END',status='COMPLETE',report=rel(OUT/'REPORT_STEP22A.md'))
  subprocess.run(['/usr/bin/afplay','/System/Library/Sounds/Glass.aiff'])
 except Exception:
  message=traceback.format_exc();event('BLOCKING',error=message)
  if not (OUT/'REPORT_STEP22A.md').exists():text(OUT/'REPORT_STEP22A_BLOCKING.md','# BLOCKING\n\n'+message+'\nNo later phase was started. The immutable protocol and completed artifacts are retained.\n')
  print(message,flush=True);raise
 finally:
  wake.terminate();wake.wait()
if __name__=='__main__':main()
