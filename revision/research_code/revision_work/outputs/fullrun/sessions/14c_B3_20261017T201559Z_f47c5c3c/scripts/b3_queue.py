"""Original14a scheduler with B3 plan/contracts; no training before final preflight."""
from b3_io import *
from b3_contracts import completed,verify_completion
import fullrun14a_queue_v2 as queue
import subprocess,time,traceback

def sound(ok=True):
 p='/System/Library/Sounds/Glass.aiff' if ok else '/System/Library/Sounds/Basso.aiff';r=subprocess.run(['/usr/bin/afplay',p],capture_output=True,text=True)
 return dict(utc=utc(),path=p,returncode=r.returncode,stderr=r.stderr)

def main():
 assert (EXP/'preflight.json').exists(),'C reference choice and final preflight are required before training'
 control=EXP/'control'/unique();control.mkdir(parents=True,exist_ok=False);awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);start=time.perf_counter();started=utc()
 original_json,original_snapshot=queue.jsonout,queue.snapshot_protected
 def snapshot():
  before=queue.OUT;queue.OUT=FULLRUN
  try:return original_snapshot()
  finally:queue.OUT=before
 queue.OUT=EXP;queue.HERE=HERE;queue.jobs=jobs;queue.completed=completed;queue.verify_completion=verify_completion;queue.snapshot_protected=snapshot
 def adapted(path,value):
  if Path(path).name=='session.json':value=dict(value,n_jobs=180,task='PROMPT14c-B3',scheduler_reused_by_import=True)
  return original_json(path,value)
 queue.jsonout=adapted
 jsonout(control/'START.json',dict(utc=started,pid=os.getpid(),caffeinate_pid=awake.pid,n_jobs=180,max_workers=3,source=rel(Path(queue.__file__)),source_sha256=sha(queue.__file__)))
 try:
  assert json.loads((EXP/'preflight.json').read_text())['passed']
  with target(control/'power_wait.jsonl').open('x') as f:
   while True:
    power=queue.power();f.write(json.dumps(dict(utc=utc(),power=power))+'\n');f.flush()
    if 'AC Power' in power:break
    assert awake.poll() is None;print('WAITING_FOR_AC',utc(),flush=True);time.sleep(30)
  before=set((EXP/'sessions').glob('*')) if (EXP/'sessions').exists() else set();queue.main();added=set((EXP/'sessions').glob('*'))-before;assert len(added)==1;session=added.pop();q=json.loads((session/'queue_result.json').read_text())
  assert q['n_complete']==180 and not q['failures'] and not q['blocked']
  jsonout(control/'FINISHED.json',dict(utc=utc(),started_utc=started,wall_seconds=time.perf_counter()-start,complete=180,session=rel(session),caffeinate_alive_through_report=awake.poll() is None));jsonout(control/'completion_sound.json',sound(True));print('B3_FINISHED',utc(),flush=True)
 except BaseException as e:
  jsonout(control/'FAILED.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()));jsonout(control/'failure_sound.json',sound(False));raise
 finally:
  if awake.poll() is None:awake.terminate();awake.wait(timeout=10)
if __name__=='__main__':main()
