"""Imported Step22B three-worker scheduler, immutable W1/W2 bindings and TEST barrier."""
import sys
sys.dont_write_bytecode=True
import io_d as io
from io_d import *
sys.path.insert(0,str(OLD))
import run22b_v3 as q
import subprocess,fcntl,time,traceback
q.__dict__.update(io.exports())
q.WORKER=OUT/'worker_d.py'

def snapshot():
    rows=[]
    for p in sorted((OUT/'fits').glob('*/attempt_*/COMPLETE.json')):
        r=read(p);rows.append(dict(job_id=job_id(r['job']),**r['job'],status=r['status'],epochs=r.get('epochs'),best_epoch=r.get('best_epoch'),ended_utc=r.get('ended_utc'),record=rel(p)))
    for pid,r in q.ACTIVE.items():
        rows.append(dict(job_id=job_id(r['job']),**r['job'],status='RUNNING',pid=pid,started_utc=r['started']))
    fields=['job_id','setting','window','model','h','seed','gate','learning_rate','delta','status','pid','started_utc','ended_utc','epochs','best_epoch','record']
    with target(OUT/'status'/stamp()/'status.csv').open('x',newline='') as f:
        w=csvlib.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
q.status_snapshot=snapshot

def status():
    rows=[read(p) for p in (OUT/'fits').glob('*/attempt_*/COMPLETE.json')]
    print('Step22D LSTM:',len(rows),'/60 terminal;',sum(r['status']=='COMPLETE' for r in rows),'complete;',sum(r['status']!='COMPLETE' for r in rows),'failed',flush=True)
    for name in ['DATA_GATE.json','TIMING_GATE.json','ALL_TRAINING_FINISHED.json','ANALYSIS_COMPLETE.json']:
        print(name,'present' if (OUT/name).exists() else 'pending')
    if (OUT/'TIMING_GATE.json').exists():print('Pilot projection (hours):',read(OUT/'TIMING_GATE.json')['projected_hours'])
    for line in subprocess.run(['ps','-Ao','pid,etime,pcpu,command'],capture_output=True,text=True).stdout.splitlines():
        if str(OUT) in line and any(t in line for t in ['worker_d.py','run_d.py','analyze_d.py']):print(line)
    for p in sorted(OUT.glob('STOP_REPORT_*.md'))[-1:]:print('Latest stop:',p)
    if (OUT/'REPORT_STEP22D.md').exists():print('Report:',OUT/'REPORT_STEP22D.md')

def wait_ac():
    announced=False
    while True:
        power=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True,check=True).stdout
        if "'AC Power'" in power:
            q.event('ac_power_verified',snapshot=power);return
        if not announced:
            print('Waiting for AC power; no training launched.',flush=True)
            q.event('waiting_for_ac',snapshot=power);announced=True
        time.sleep(30)

def timing(pilots):
    if (OUT/'TIMING_GATE.json').exists():
        d=read(OUT/'TIMING_GATE.json');assert d['proceed'],'STOP: timing projection >3h';return d
    for j in pilots:
        p,r=completion(j)
        assert r is not None and r['status']=='COMPLETE','STOP: timing pilot failed; projection undefined'
        verify_record(p.parent/'ADAPTER_COMPLETE.json')
    a=read(OUT/'TIMING_PILOT_START.json');b=read(OUT/'TIMING_PILOT_END.json')
    seconds=elapsed(a['utc'],b['utc'])
    d=dict(protocol_sha256=check_protocol(),started_utc=a['utc'],ended_utc=b['utc'],pilot_elapsed_utc_seconds=seconds,
      pilot_monotonic_seconds=b['monotonic']-a['monotonic'] if a['pid']==b['pid'] else None,
      total_runs=60,workers=3,projected_hours=seconds*20/3600,threshold_hours=3,proceed=seconds*20<=10800,
      formula='actual pilot batch UTC elapsed * (60/3)',pilot_jobs=pilots)
    js(OUT/'TIMING_GATE.json',d)
    assert d['proceed'],'STOP: timing projection >3h; TEST remains locked'
    print('TIMING PASS: projected',round(d['projected_hours'],3),'hours',flush=True)
    return d

def run():
    check_protocol()
    lockpath=OUT/'QUEUE_LOCK'
    if not lockpath.exists():text(lockpath,'Immutable inode; ownership is the live OS flock.\n')
    lock=lockpath.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'ANALYSIS_COMPLETE.json').exists():
        verify_record(OUT/'ANALYSIS_COMPLETE.json');print('Already complete:',OUT/'REPORT_STEP22D.md');return
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        assert read(OUT/'DATA_GATE.json')['passed'],'Data gate absent/failed'
        assert all(r['unchanged'] for r in io.check_old_sources()),'Source hashes changed'
        q.event('queue_start',pid=os.getpid(),protocol_sha256=check_protocol())
        plan=[spec(t,w,h,s) for t in ['T0','T3'] for s in SEEDS for h in [20,10,5] for w in ['W1','W2']]
        if not (OUT/'GRID_PLAN.json').exists():js(OUT/'GRID_PLAN.json',dict(protocol_sha256=check_protocol(),jobs=plan))
        assert read(OUT/'GRID_PLAN.json')['jobs']==plan
        if not (OUT/'ALL_TRAINING_FINISHED.json').exists():
            wait_ac()
            pilots=[spec('T0','W1',20,0),spec('T0','W2',20,0),spec('T3','W2',20,0)]
            if not (OUT/'TIMING_PILOT_START.json').exists():js(OUT/'TIMING_PILOT_START.json',dict(utc=utc(),monotonic=time.perf_counter(),pid=os.getpid(),jobs=pilots,protocol_sha256=check_protocol()))
            q.queue(pilots)
            if not (OUT/'TIMING_PILOT_END.json').exists():js(OUT/'TIMING_PILOT_END.json',dict(utc=utc(),monotonic=time.perf_counter(),pid=os.getpid(),protocol_sha256=check_protocol()))
            timing(pilots)
            q.queue(plan)
            records=[]
            for j in plan:
                p,r=completion(j);assert r is not None
                if r['status']=='COMPLETE':
                    verify_record(p.parent/'ADAPTER_COMPLETE.json')
                    assert r['lr']==j['learning_rate'] and r['delta']==j['delta']
                records.append(dict(job=j,record=rel(p),status=r['status']))
            js(OUT/'ALL_TRAINING_FINISHED.json',dict(protocol_sha256=check_protocol(),utc=utc(),runs=records))
            q.event('all_training_finished',runs=60)
        barrier()
        q.event('test_evaluation_start')
        for j in plan:
            folder=OUT/'evaluation'/job_id(j)
            done=sorted(folder.glob('attempt_*/COMPLETE.json'))
            if done:
                assert len(done)==1;verify_record(done[0]);verify_record(done[0].parent/'ADAPTER_COMPLETE.json');continue
            attempt=target(folder/('attempt_'+stamp()));attempt.mkdir(parents=True)
            with target(attempt/'worker.log').open('x') as log:
                p=subprocess.run([sys.executable,'-B',str(q.WORKER),'evaluate',json.dumps(j),str(attempt)],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
            assert p.returncode==0,'Evaluation failed: '+str(attempt)
            verify_record(attempt/'COMPLETE.json');verify_record(attempt/'ADAPTER_COMPLETE.json')
        q.helper('analyze_d.py')
        verify_record(OUT/'ANALYSIS_COMPLETE.json')
        q.event('queue_complete');print('COMPLETE:',OUT/'REPORT_STEP22D.md',flush=True)
        subprocess.run(['/usr/bin/afplay','/System/Library/Sounds/Glass.aiff'],check=False)
    except BaseException as e:
        for r in q.ACTIVE.values():
            if r['process'].poll() is None:r['process'].terminate()
        for r in q.ACTIVE.values():r['process'].wait();r['handle'].close()
        q.ACTIVE.clear()
        body='# BLOCKING\n\n'+repr(e)+'\n\nProtocol SHA: `'+check_protocol()+'`.\n\nAll existing artifacts retained. TEST remains locked unless all60 training records and both gates passed.\n\n```text\n'+traceback.format_exc()+'\n```\n'
        p=OUT/('STOP_REPORT_'+stamp()+'.md');text(p,body)
        if not (OUT/'REPORT_STEP22D.md').exists():text(OUT/'REPORT_STEP22D.md',body)
        print('STOP:',p,flush=True);raise
    finally:
        awake.terminate();awake.wait();lock.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','status']);a=p.parse_args()
    status() if a.command=='status' else run()
