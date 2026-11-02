"""Run the actual imported14a scheduler in B1 / gated pilot / B4 phases."""
from day_io_v2 import *
import subprocess,time,traceback
import fullrun14a_queue_v2 as queue

def main():
 outer_id=unique();control=EXP/'control'/outer_id;control.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter();started=utc();caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);original_json=queue.jsonout;original_snapshot=queue.snapshot_protected
 def snapshot():
  old=queue.OUT;queue.OUT=FULLRUN
  try:return original_snapshot()
  finally:queue.OUT=old
 queue.OUT=EXP;queue.HERE=HERE;queue.completed=completed;queue.verify_completion=verify_completion;queue.snapshot_protected=snapshot
 results=[]
 jsonout(control/'START.json',dict(utc=started,pid=os.getpid(),caffeinate_pid=caffeinate.pid,command=['caffeinate','-i','-w',str(os.getpid())],scheduler_source=rel(Path(queue.__file__)),scheduler_sha256=sha(queue.__file__),profile_environment={k:profile_environment().get(k) for k in ENV_KEYS},n_jobs=72,max_training_workers=3,phase_order=['b1','pilot','b4']))
 try:
  for phase in ['b1','pilot','b4']:
   plan=phase_jobs(phase);queue.jobs=lambda plan=plan:plan
   def adapted_json(path,value,phase=phase,plan=plan):
    if Path(path).name=='session.json':value=dict(value,n_jobs=len(plan),phase=phase,task='PROMPT14c-day',scheduler_reused_by_import=True)
    return original_json(path,value)
   queue.jsonout=adapted_json
   before=set((EXP/'sessions').glob('*')) if (EXP/'sessions').exists() else set()
   queue.main()
   after=set((EXP/'sessions').glob('*'));new=after-before;assert len(new)==1;session=new.pop();r=json.loads((session/'queue_result.json').read_text())
   assert r['n_complete']==len(plan) and not r['failures'] and not r['blocked'],r
   results.append(dict(phase=phase,session=rel(session),n_complete=r['n_complete']))
   jsonout(control/f'{phase}_COMPLETE.json',results[-1])
   if phase=='pilot':
    cp,c=completed(plan[0]);assert c and c['causal_input_check']['passed'] and c['test_replay']['classification']=='EXACT'
    jsonout(EXP/'LSTM_PILOT_QUALIFIED.json',dict(utc=utc(),completion=rel(cp),sha256=sha(cp),actual_loader_bitwise_checks=c['causal_input_check']['actual_trainer_loaders'],test_replay=c['test_replay']))
  with target(control/'final_report.log').open('x') as log:
   code=subprocess.call([str(PYTHON),'-B',str(HERE/'day_report.py'),'--final','--control',str(control)],cwd=ROOT,env=profile_environment(),stdout=log,stderr=subprocess.STDOUT)
  assert code==0,f'Final report failed: {rel(control/"final_report.log")}'
  jsonout(control/'FINISHED.json',dict(utc=utc(),started_utc=started,wall_seconds=time.perf_counter()-start,n_complete=72,phases=results,caffeinate_alive_through_report=caffeinate.poll() is None))
  jsonout(control/'completion_sound.json',sound(True));print('STEP14C_DAY_FINISHED',utc(),72,flush=True)
 except BaseException as e:
  jsonout(control/'FAILED.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc(),phases_completed=results))
  report=FULLRUN/'REPORT_STEP14C_DAY.md'
  if not report.exists():
   textout(report,'# BLOCKING\n\nQueue stopped: '+repr(e)+'\n\nControl log: '+rel(control)+'\nAll completed artifacts are preserved; no prior output was overwritten.\n')
  jsonout(control/'failure_sound.json',sound(False));raise
 finally:
  if caffeinate.poll() is None:caffeinate.terminate();caffeinate.wait(timeout=10)
if __name__=='__main__':main()
