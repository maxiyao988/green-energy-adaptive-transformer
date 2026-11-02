"""Lightweight durable supervision; never reads prediction/label contents or changes a protocol."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,datetime,time,subprocess,fcntl,uuid
BASE=Path(__file__).resolve().parent
SESSION=BASE/'session_v3'
sys.path.insert(0,str(SESSION))
import io_d as io

def write(folder,kind,**data):
    p=folder/(io.stamp()+'_'+kind+'.json')
    with p.open('x') as f:json.dump(dict(utc=io.utc(),kind=kind,**data),f,indent=2)

def main():
    folder=BASE/'monitor'/io.stamp();folder.mkdir(parents=True)
    lockpath=BASE/'MONITOR_LOCK'
    try:
        with lockpath.open('x') as f:f.write('Immutable inode; OS flock owns the monitor.\n')
    except FileExistsError:pass
    lock=lockpath.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(folder,'monitor_start',pid=os.getpid(),interval_seconds=60,session=io.rel(SESSION),protocol_sha256=io.check_protocol())
    restarts=0;idle=0
    while True:
        try:
            ps=subprocess.run(['ps','-Ao','pid=,etime=,pcpu=,command='],capture_output=True,text=True,check=True).stdout.splitlines()
            queue=[l for l in ps if str(SESSION/'run_d_v4.py') in l and ' run' in l]
            workers=[l for l in ps if str(SESSION/'worker_d.py') in l and (' train ' in l or ' evaluate ' in l)]
            analysis=[l for l in ps if str(SESSION/'analyze_d_v3.py') in l]
            rows=[io.read(p) for p in (SESSION/'fits').glob('*/attempt_*/COMPLETE.json')]
            ev=list((SESSION/'evaluation').glob('*/attempt_*/COMPLETE.json'))
            done=SESSION/'ANALYSIS_COMPLETE.json'
            stopped=sorted(SESSION.glob('STOP_REPORT_*.md'))
            details=dict(terminal_fits=len(rows),complete_fits=sum(r['status']=='COMPLETE' for r in rows),
              failed_fits=sum(r['status']!='COMPLETE' for r in rows),evaluated_cells=len(ev),queue_processes=queue,
              worker_processes=workers,analysis_processes=analysis,all_training_finished=(SESSION/'ALL_TRAINING_FINISHED.json').exists(),
              report_complete=done.exists(),stop_reports=[io.rel(p) for p in stopped],
              load_average=list(os.getloadavg()),power=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout,
              free_bytes=os.statvfs(SESSION).f_bavail*os.statvfs(SESSION).f_frsize)
            write(folder,'health',**details)
            if done.exists():
                io.verify_record(done)
                write(folder,'verified_completion',analysis_complete=io.rel(done),report=io.rel(SESSION/'REPORT_STEP22D.md'))
                print('Step22D complete and artifacts verified.',flush=True);return
            if stopped:
                # No silent retry of a recorded gate/training failure and no specification change.
                write(folder,'needs_repair',latest_stop=io.rel(stopped[-1]),automatic_restart=False)
                subprocess.run(['/usr/bin/afplay','/System/Library/Sounds/Basso.aiff'],check=False)
                print('STOP requires review:',stopped[-1],flush=True);return
            if len(workers)>3:
                write(folder,'concurrency_violation',workers=workers)
                raise RuntimeError('More than three Step22D workers observed; no new process launched')
            if not queue and not workers and not analysis:
                idle+=1
                if idle>=2:
                    assert restarts<3,'Unexpected interruptions repeated3 times; review required'
                    io.check_protocol()
                    log=folder/('resume_'+io.stamp()+'.log')
                    with log.open('x') as handle:
                        p=subprocess.Popen([sys.executable,'-B',str(SESSION/'run_d_v4.py'),'run'],cwd=io.ROOT,
                          env=io.environment(),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    restarts+=1;idle=0
                    write(folder,'safe_resume',pid=p.pid,log=io.rel(log),restart_number=restarts,
                      conditions='No controller, worker or analysis process; no STOP record; immutable protocol hashes verified')
            else:idle=0
        except Exception as error:
            write(folder,'monitor_error',error=repr(error));raise
        time.sleep(60)

if __name__=='__main__':main()
