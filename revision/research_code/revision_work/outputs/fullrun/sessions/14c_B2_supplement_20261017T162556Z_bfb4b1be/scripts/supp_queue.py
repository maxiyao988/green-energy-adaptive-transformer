"""Append48 seeds to the imported B2/14a queue without touching old journals."""
from supp_io import *
import subprocess,time,traceback
import fullrun14a_queue_v2 as queue

def main():
 control=EXP/'control'/unique();control.mkdir(parents=True,exist_ok=False)
 caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);started=utc();clock=time.perf_counter()
 original_json=queue.jsonout;original_snapshot=queue.snapshot_protected
 def snapshot():
  previous=queue.OUT;queue.OUT=FULLRUN
  try:return original_snapshot()
  finally:queue.OUT=previous
 queue.OUT=EXP;queue.HERE=HERE;queue.jobs=jobs;queue.completed=completed;queue.verify_completion=verify_completion;queue.snapshot_protected=snapshot
 def adapted(path,value):
  if Path(path).name=='session.json':value=dict(value,n_jobs=156,n_existing_completed=108,n_new_neural=48,task='PROMPT14c-B2-supplement',scheduler_reused_by_import=True)
  return original_json(path,value)
 queue.jsonout=adapted
 jsonout(control/'START.json',dict(utc=started,pid=os.getpid(),caffeinate_pid=caffeinate.pid,n_jobs=156,n_existing=108,n_new=48,max_workers=3,source=rel(Path(queue.__file__)),source_sha256=sha(queue.__file__)))
 try:
  assert json.loads((EXP/'preflight.json').read_text())['passed']
  with target(control/'power_wait.jsonl').open('x') as log:
   while True:
    power=queue.power();log.write(json.dumps(dict(utc=utc(),power=power))+'\n');log.flush()
    if 'AC Power' in power:break
    print('WAITING_FOR_AC',utc(),flush=True);assert caffeinate.poll() is None;time.sleep(30)
  before=set((EXP/'sessions').glob('*')) if (EXP/'sessions').exists() else set();queue.main();created=set((EXP/'sessions').glob('*'))-before;assert len(created)==1;session=created.pop()
  result=json.loads((session/'queue_result.json').read_text());assert result['n_complete']==156 and not result['failures'] and not result['blocked']
  assert (EXP/'FINAL_VALIDATION.json').exists()
  jsonout(control/'FINISHED.json',dict(utc=utc(),started_utc=started,wall_seconds=time.perf_counter()-clock,new_neural_complete=48,original_cells_skipped=108,session=rel(session),caffeinate_alive_through_report=caffeinate.poll() is None))
  jsonout(control/'completion_sound.json',sound(True));print('B2_SUPPLEMENT_FINISHED',utc(),flush=True)
 except BaseException as e:
  jsonout(control/'FAILED.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc()));jsonout(control/'failure_sound.json',sound(False));raise
 finally:
  if caffeinate.poll() is None:caffeinate.terminate();caffeinate.wait(timeout=10)
if __name__=='__main__':main()
