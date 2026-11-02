"""Wait for B3, audit it, then execute the imported14a B5 phases."""
from b5_io import *
import fcntl,subprocess,time,traceback
import fullrun14a_queue_v2 as queue

def locked(path):
 if not path.exists():return False
 with path.open('rb') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(f,fcntl.LOCK_UN);return False

def other_workers():
 ps=subprocess.run(['ps','-axo','pid,command'],capture_output=True,text=True,check=True).stdout
 return [l.strip() for l in ps.splitlines() if 'fullrun14a_worker_v2.py' in l and '--job-id ' in l]

def sound(success):
 p='/System/Library/Sounds/Glass.aiff' if success else '/System/Library/Sounds/Basso.aiff'
 r=subprocess.run(['/usr/bin/afplay',p],capture_output=True,text=True)
 return dict(utc=utc(),path=p,returncode=r.returncode,stderr=r.stderr)

def main():
 path=EXP/'supervisor.lock'
 try:
  with path.open('x') as f:f.write('B5 supervisor lock; immutable after creation.\n')
 except FileExistsError:pass
 lock=path.open('rb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 control=EXP/'control'/unique();control.mkdir(parents=True,exist_ok=False)
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);start=time.perf_counter();started=utc()
 jsonout(control/'START.json',dict(utc=started,pid=os.getpid(),caffeinate_pid=awake.pid,task='PROMPT14c-B5',wait_for_B3=rel(B3),max_workers=3,n_queued=72,n_pilot=1))
 def execute(script,label,args=()):
  with target(control/(label+'.log')).open('x') as log:
   code=subprocess.call([str(PYTHON),'-B',str(HERE/script),*args],cwd=ROOT,env=profile_environment(),stdout=log,stderr=subprocess.STDOUT)
  assert code==0,f'{label} exited {code}; see {rel(control/(label+".log"))}'
 try:
  if (EXP/'FINAL_VALIDATION.json').exists():
   f=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert f['all_cells_verified'] and sha(ROOT/f['report'])==f['report_sha256'];print('B5_ALREADY_COMPLETE',flush=True);return
  # No Torch imported and no training started in this waiting supervisor.
  last_log=0.
  while True:
   assert awake.poll() is None,'caffeinate exited'
   finished=(B3/'FINAL_VALIDATION.json').exists() and any((p/'FINISHED.json').exists() for p in (B3/'control').iterdir())
   workers=other_workers();busy=locked(B3/'queue.lock') or bool(workers)
   if finished and not busy:break
   failures=list((B3/'control').glob('*/FAILED.json'))
   if failures and not locked(B3/'queue.lock') and not workers and not finished:raise RuntimeError('B3 stopped with a failure; B5 not started. '+str(failures))
   if time.perf_counter()-last_log>=300:
    jsonout(control/'wait_snapshots'/(unique()+'.json'),dict(utc=utc(),B3_final_present=(B3/'FINAL_VALIDATION.json').exists(),B3_finished=finished,other_workers=workers))
    print('WAITING_FOR_B3',utc(),'workers',len(workers),flush=True);last_log=time.perf_counter()
   time.sleep(30)
  if not (EXP/'B3_ACCEPTANCE.json').exists():execute('b5_accept_b3_v2.py','b3_acceptance')
  acceptance=json.loads((EXP/'B3_ACCEPTANCE.json').read_text());assert acceptance['passed']
  for a in acceptance['evidence']+[acceptance['report']]:assert sha(ROOT/a['path'])==a['sha256']
  if not (EXP/'preflight.json').exists():execute('b5_worker.py','b5_preflight',['--preflight'])
  if not (EXP/'reference_provenance.json').exists():execute('b5_references.py','seed42_references')
  original_snapshot=queue.snapshot_protected;original_json=queue.jsonout
  def snapshot():
   previous=queue.OUT;queue.OUT=FULLRUN
   try:return original_snapshot()
   finally:queue.OUT=previous
  queue.OUT=EXP;queue.HERE=HERE/'entry';queue.completed=completed;queue.verify_completion=verify_completion;queue.snapshot_protected=snapshot
  phases=[]
  for phase in ['pilot','main']:
   plan=jobs(phase);queue.jobs=lambda plan=plan:plan
   def adapted(path,value,phase=phase,plan=plan):
    if Path(path).name=='session.json':value=dict(value,n_jobs=len(plan),task='PROMPT14c-B5',phase=phase,scheduler_reused_by_import=True)
    return original_json(path,value)
   queue.jsonout=adapted
   with target(control/(phase+'_power_wait.jsonl')).open('x') as power_log:
    while True:
     power=queue.power();power_log.write(json.dumps(dict(utc=utc(),power=power))+'\n');power_log.flush()
     if 'AC Power' in power:break
     assert awake.poll() is None;print('WAITING_FOR_AC',utc(),flush=True);time.sleep(30)
   assert not other_workers(),'Other training workers present before B5 phase'
   before=set((EXP/'sessions').glob('*')) if (EXP/'sessions').exists() else set()
   queue.main();new=set((EXP/'sessions').glob('*'))-before;assert len(new)==1;session=new.pop();result=json.loads((session/'queue_result.json').read_text())
   assert result['n_complete']==len(plan) and not result['failures'] and not result['blocked']
   phases.append(dict(phase=phase,session=rel(session),completed=result['n_complete']))
   jsonout(control/(phase+'_COMPLETE.json'),phases[-1])
   if phase=='pilot':assert (EXP/'PILOT_COMPARISON.json').exists()
  execute('b5_report.py','final_report',['--final'])
  final=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert final['queued_cells']==72 and final['all_cells_verified']
  jsonout(control/'FINISHED.json',dict(utc=utc(),started_utc=started,wall_seconds=time.perf_counter()-start,phases=phases,caffeinate_alive_through_report=awake.poll() is None))
  jsonout(control/'completion_sound.json',sound(True));print('B5_FINISHED',utc(),flush=True)
 except BaseException as e:
  jsonout(control/'FAILED.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()))
  textout(control/'BLOCKING.md','# BLOCKING\n\n'+repr(e)+'\n\nNo completed artifacts were overwritten. See logs in this control directory.\n')
  jsonout(control/'failure_sound.json',sound(False));raise
 finally:
  if awake.poll() is None:awake.terminate();awake.wait(timeout=10)
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
