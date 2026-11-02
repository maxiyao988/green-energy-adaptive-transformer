"""Reuse imported14a scheduler directly with rolling-origin job/contract bindings."""
from b2_io import *
import subprocess,time,traceback
import fullrun14a_queue_v2 as queue

def main():
 control=EXP/'control'/unique();control.mkdir(parents=True,exist_ok=False);caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);started=utc();clock=time.perf_counter()
 original_json=queue.jsonout;original_snapshot=queue.snapshot_protected
 def snapshot():
  old=queue.OUT;queue.OUT=FULLRUN
  try:return original_snapshot()
  finally:queue.OUT=old
 queue.OUT=EXP;queue.HERE=HERE;queue.jobs=jobs;queue.completed=completed;queue.verify_completion=verify_completion;queue.snapshot_protected=snapshot
 def adapted(path,value):
  if Path(path).name=='session.json':value=dict(value,n_jobs=len(PLAN),n_neural=72,n_reference=36,task='PROMPT14c-B2',scheduler_reused_by_import=True)
  return original_json(path,value)
 queue.jsonout=adapted
 jsonout(control/'START.json',dict(utc=started,pid=os.getpid(),caffeinate_pid=caffeinate.pid,n_jobs=108,n_neural=72,n_reference=36,max_workers=3,source=rel(Path(queue.__file__)),source_sha256=sha(queue.__file__),environment={k:profile_environment().get(k) for k in ENV_KEYS}))
 try:
  assert json.loads((EXP/'preflight.json').read_text())['passed'];assert json.loads((EXP/'W3_provenance.json').read_text())['no_retraining']
  with target(control/'power_wait.jsonl').open('x') as powerlog:
   while True:
    power=queue.power();powerlog.write(json.dumps(dict(utc=utc(),power=power))+chr(10));powerlog.flush()
    if 'AC Power' in power:break
    print('WAITING_FOR_AC',utc(),flush=True);assert caffeinate.poll() is None;time.sleep(30)
  before=set((EXP/'sessions').glob('*')) if (EXP/'sessions').exists() else set();queue.main();after=set((EXP/'sessions').glob('*'));new=after-before;assert len(new)==1;session=new.pop();r=json.loads((session/'queue_result.json').read_text())
  assert r['n_complete']==108 and not r['failures'] and not r['blocked'],r
  assert (FULLRUN/'REPORT_STEP14C_B2.md').exists()
  jsonout(control/'FINISHED.json',dict(utc=utc(),started_utc=started,wall_seconds=time.perf_counter()-clock,n_complete=108,n_neural=72,n_reference=36,window3_retraining=False,session=rel(session),caffeinate_alive_through_report=caffeinate.poll() is None));jsonout(control/'completion_sound.json',sound(True));print('STEP14C_B2_FINISHED',utc(),flush=True)
 except BaseException as e:
  jsonout(control/'FAILED.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()))
  report=FULLRUN/'REPORT_STEP14C_B2.md'
  if not report.exists():textout(report,'# BLOCKING\n\nQueue stopped: '+repr(e)+'\n\nSee '+rel(control)+' and the immutable pre-training design report '+rel(EXP/'DESIGN_REPORT_STEP14C_B2.md')+'.\nNo prior files were overwritten.\n')
  jsonout(control/'failure_sound.json',sound(False));raise
 finally:
  if caffeinate.poll() is None:caffeinate.terminate();caffeinate.wait(timeout=10)
if __name__=='__main__':main()
