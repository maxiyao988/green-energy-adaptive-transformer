"""One numerical process for Step15; resumable immutable outputs and awake assertion."""
import os,sys,subprocess,time,json
sys.dont_write_bytecode=True
from frozen_io_v3 import *
for k in ENV_KEYS[:5]:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
def main():
 if not (OUT/'backbone_registry.json').exists():build_registry()
 session=OUT/'sessions'/uid();session.mkdir(parents=True,exist_ok=False)
 command=[str(ROOT/'.venv/bin/python'),'-B',str(OUT/'scripts/frozen_worker_v3.py'),'--session',str(session)]
 log=target(session/'worker.log').open('x');start=time.perf_counter();p=subprocess.Popen(command,cwd=ROOT,env=profile_env(),stdout=log,stderr=subprocess.STDOUT)
 wake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 jout(session/'launch.json',dict(utc=utc(),supervisor_pid=os.getpid(),worker_pid=p.pid,caffeinate_pid=wake.pid,command=command,single_numerical_worker=True,profile='CPU12/12,OMP unset,dynamic FALSE',power=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout,process_snapshot=subprocess.run(['ps','-axo','pid,ppid,pcpu,command'],capture_output=True,text=True).stdout,source_authorization='User explicitly authorized Exp3/Exp4/Step3B sources to complete60 backbones; stop if any non-EXACT replay'))
 print('LAUNCHED',p.pid,rel(session),flush=True)
 try:
  rc=p.wait();log.close();jout(session/'exit.json',dict(utc=utc(),returncode=rc,wall_seconds=time.perf_counter()-start))
  if rc==0:
   rlog=target(session/'report.log').open('x');rp=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(OUT/'scripts/frozen_report.py'),'--session',str(session)],cwd=ROOT,env=profile_env(),stdout=rlog,stderr=subprocess.STDOUT);rlog.close()
   jout(session/'report_exit.json',dict(utc=utc(),returncode=rp.returncode));print('REPORT_EXIT',rp.returncode,flush=True)
  else:
   target_report=OUT/'REPORT_STEP15.md'
   if not target_report.exists():text(target_report,'# BLOCKING'+chr(10)*2+'Worker stopped; no further routing runs were launched. See `'+rel(session/'worker.log')+'` and immutable failure records. No backbone was retrained.'+chr(10))
 finally:wake.terminate();wake.wait()
if __name__=='__main__':main()
