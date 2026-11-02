"""Actual imported14a scheduler: nine cells, one verified reuse, eight new trainings."""
from fx_io import *
import subprocess,time,traceback,fcntl,shlex
import fullrun14a_queue_v2 as queue

def workers():
 r=subprocess.run(['ps','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True);out=[]
 for line in r.stdout.splitlines():
  x=line.split(None,2)
  if len(x)!=3:continue
  try:args=shlex.split(x[2])
  except ValueError:continue
  if '--job-id' in args and any(a.endswith('/fullrun14a_worker_v2.py') for a in args):out.append(line.strip())
 return out

def sound(ok):
 p='/System/Library/Sounds/Glass.aiff' if ok else '/System/Library/Sounds/Basso.aiff';r=subprocess.run(['/usr/bin/afplay',p],capture_output=True,text=True)
 return dict(utc=utc(),sound=p,returncode=r.returncode,stderr=r.stderr)

def main():
 lp=EXP/'supervisor.lock'
 try:
  with lp.open('x') as f:f.write('B5FIX exclusive supervisor lock.\n')
 except FileExistsError:pass
 lock=lp.open('rb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 control=EXP/'control'/unique();control.mkdir(parents=True,exist_ok=False);awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);start=time.perf_counter();started=utc()
 jsonout(control/'START.json',dict(utc=started,pid=os.getpid(),caffeinate_pid=awake.pid,task='PROMPT14c-B5fix',planned_cells=9,new_training=8,reused=1,maximum_workers=3))
 try:
  if (EXP/'FINAL_VALIDATION.json').exists():
   r=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert sha(ROOT/r['report'])==r['report_sha256'];print('B5FIX_ALREADY_COMPLETE',flush=True);return
  assert not workers(),'Other training workers present; no B5FIX training started'
  with target(control/'power_wait.jsonl').open('x') as log:
   while True:
    power=queue.power();log.write(json.dumps(dict(utc=utc(),power=power))+'\n');log.flush()
    if 'AC Power' in power:break
    assert awake.poll() is None;print('WAITING_FOR_AC',utc(),flush=True);time.sleep(30)
  if not (EXP/'preflight.json').exists():
   with target(control/'preflight.log').open('x') as log:
    code=subprocess.call([str(PYTHON),'-B',str(HERE/'fx_worker.py'),'--preflight'],cwd=ROOT,env=profile_environment(),stdout=log,stderr=subprocess.STDOUT)
   assert code==0,'B5FIX preflight failed; inspect '+rel(control/'preflight.log')
  cp,reused=completed(spec_for(REUSE_ID));assert reused and reused['training_reused'],'Verified pilot reuse is mandatory; never train h5 LSTM twice'
  original_snapshot=queue.snapshot_protected;original_json=queue.jsonout
  def snapshot():
   old=queue.OUT;queue.OUT=FULLRUN
   try:return original_snapshot()
   finally:queue.OUT=old
  queue.OUT=EXP;queue.HERE=HERE/'entry';queue.jobs=jobs;queue.completed=completed;queue.verify_completion=verify_completion;queue.snapshot_protected=snapshot
  def adapted(path,value):
   if Path(path).name=='session.json':value=dict(value,n_jobs=9,n_new_training=8,n_reused=1,task='PROMPT14c-B5fix',scheduler_reused_by_import=True)
   return original_json(path,value)
  queue.jsonout=adapted
  before=set((EXP/'sessions').glob('*')) if (EXP/'sessions').exists() else set();queue.main();new=set((EXP/'sessions').glob('*'))-before;assert len(new)==1;session=new.pop();q=json.loads((session/'queue_result.json').read_text())
  assert q['n_complete']==9 and not q['failures'] and not q['blocked']
  v=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert v['newly_trained']==8 and v['reused']==1 and v['all_cells_verified']
  jsonout(control/'FINISHED.json',dict(utc=utc(),started_utc=started,wall_seconds=time.perf_counter()-start,session=rel(session),cells=9,new_training=8,reused=1,caffeinate_alive_through_report=awake.poll() is None))
  jsonout(control/'completion_sound.json',sound(True));print('B5FIX_FINISHED',utc(),flush=True)
 except BaseException as e:
  jsonout(control/'FAILED.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()));textout(control/'BLOCKING.md','# BLOCKING\n\n'+repr(e)+'\n');jsonout(control/'failure_sound.json',sound(False));raise
 finally:
  if awake.poll() is None:awake.terminate();awake.wait(timeout=10)
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
