"""Step22E/22B scheduler imported;45 W3 parameter-group variants, user launched."""
import sys,types,ast,copy,subprocess,fcntl,time,threading,traceback
sys.dont_write_bytecode=True
import io_f as io
from io_f import *
sys.path.insert(0,str(io.E));e=io.module('_22f_scheduler_e',io.E/'run_e_v5.py');q=e.q
q.__dict__.update(io.exports());q.WORKER=OUT/'worker_f.py';q.completion=io.latest_terminal
for name in ['successful','import_queue','recover_partials','wait_ac']:
    fn=getattr(e,name);globals()[name]=types.FunctionType(fn.__code__,globals(),name,fn.__defaults__,fn.__closure__)
QUEUE_BINDING=import_queue()
# Preserve original snapshot statements; extend only the CSV columns for this job schema.
source=io.ROOT/'revision_work/outputs/step22d/session_v3/run_d_v4.py'
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
stmt=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fields' for t in n.targets))
stmt.value.elts += [ast.Constant('variant'),ast.Constant('multiplier')]
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),globals());q.status_snapshot=snapshot

def live_workers():
    lines=subprocess.run(['ps','-Ao','pid=,command='],capture_output=True,text=True,check=True).stdout.splitlines()
    return [s for s in lines if str(OUT/'worker_f.py') in s and (' train ' in s or ' evaluate ' in s)]
def status():
    rows=[]
    for p in (OUT/'fits').glob('*/attempt_*/COMPLETE.json'):
        try:rows.append(read(p))
        except json.JSONDecodeError:pass
    good={job_id(r['job']) for r in rows if r['status']=='COMPLETE'}
    print(utc(),'Step22F:',len(good),'/45 trained;',sum(r['status']!='COMPLETE' for r in rows),'failed/interrupted attempts;',len(list((OUT/'evaluation').glob('*/attempt_*/COMPLETE.json'))),'/45 TEST evaluated',flush=True)
    for s in live_workers():print(s,flush=True)
    for n in ['DATA_GATE.json','TIMING_GATE.json','ALL_TRAINING_FINISHED.json','ANALYSIS_COMPLETE.json','VALIDATION_COMPLETE.json']:print(n,'present' if (OUT/n).exists() else 'pending')
    for p in sorted(OUT.glob('STOP_REPORT_*.md'))[-1:]:print('STOP:',p)
    if (OUT/'REPORT_STEP22F_FINAL.md').exists():print('REPORT:',OUT/'REPORT_STEP22F_FINAL.md')
def heartbeat(stop):
    while not stop.wait(60):
        try:
            good=sum(read(p)['status']=='COMPLETE' for p in (OUT/'fits').glob('*/attempt_*/COMPLETE.json'))
            js(OUT/'health'/(stamp()+'.json'),dict(utc=utc(),trained=good,workers=[dict(pid=pid,job=r['job'],started_utc=r['started']) for pid,r in list(q.ACTIVE.items())],test_evaluated=len(list((OUT/'evaluation').glob('*/attempt_*/COMPLETE.json'))),load_average=list(os.getloadavg())))
            print('HEARTBEAT',utc(),good,'/45 trained;',len(q.ACTIVE),'active;',len(list((OUT/'evaluation').glob('*/attempt_*/COMPLETE.json'))),'/45 TEST',flush=True)
        except (json.JSONDecodeError,RuntimeError):pass

def timing(pilots):
    if (OUT/'TIMING_GATE.json').exists():
        a=read(OUT/'TIMING_GATE.json');assert a['proceed'],'STOP: projected>9h';return
    assert all(successful(j) for j in pilots)
    a=read(OUT/'TIMING_PILOT_START.json');b=read(OUT/'TIMING_PILOT_END.json');sec=elapsed(a['utc'],b['utc'])
    r=dict(protocol_sha256=check_protocol(),started_utc=a['utc'],ended_utc=b['utc'],pilot_seconds=sec,total_runs=45,workers=3,projected_hours=15*sec/3600,threshold_hours=9,proceed=15*sec<=32400,pilots=pilots,formula='batch elapsed x15, including attempts',monotonic_seconds=b['monotonic']-a['monotonic'] if a['pid']==b['pid'] else None)
    js(OUT/'TIMING_GATE.json',r);assert r['proceed'],'STOP: projected>9h; TEST locked'
    print('TIMING PASS:',round(r['projected_hours'],3),'hours',flush=True)
def execute():
    check_protocol();lockpath=OUT/'QUEUE_LOCK'
    if not lockpath.exists():text(lockpath,'Immutable inode; live ownership via OS flock.\n')
    lock=lockpath.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'VALIDATION_COMPLETE.json').exists():verify_record(OUT/'VALIDATION_COMPLETE.json');print('Already finished.');lock.close();return
    awake=subprocess.Popen(['/usr/bin/caffeinate','-dimsu','-w',str(os.getpid())]);stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(stop,),daemon=True);thread.start()
    try:
        q.event('queue_start',pid=os.getpid(),caffeinate_command=['/usr/bin/caffeinate','-dimsu','-w',str(os.getpid())],caffeinate_pid=awake.pid,scheduling_binding_sha256=QUEUE_BINDING)
        q.helper('preflight_f.py');assert read(OUT/'DATA_GATE.json')['passed']
        plan=[spec(v,h,s) for s in SEEDS for h in [20,10,5] for v in VARIANTS]
        if not (OUT/'GRID_PLAN.json').exists():js(OUT/'GRID_PLAN.json',dict(protocol_sha256=check_protocol(),jobs=plan))
        assert read(OUT/'GRID_PLAN.json')['jobs']==plan
        if not (OUT/'ALL_TRAINING_FINISHED.json').exists():
            recover_partials(plan);wait_ac()
            pilots=[spec('Static-R10',20,42),spec('Static-R100',20,0),spec('Adaptive-R100',20,42)]
            if not (OUT/'TIMING_PILOT_START.json').exists():js(OUT/'TIMING_PILOT_START.json',dict(utc=utc(),pid=os.getpid(),monotonic=time.perf_counter(),jobs=pilots,protocol_sha256=check_protocol()))
            q.queue(pilots)
            if not (OUT/'TIMING_PILOT_END.json').exists():js(OUT/'TIMING_PILOT_END.json',dict(utc=utc(),pid=os.getpid(),monotonic=time.perf_counter(),protocol_sha256=check_protocol()))
            timing(pilots);q.queue(plan)
            records=[]
            for j in plan:
                cp,r=io.completion(j);assert r['status']=='COMPLETE' and r['lr']==1e-4 and r['delta']==1.
                group=read(cp.parent/'OPTIMIZER_GROUPS.json');assert group['job']==j and group['exhaustive'] and group['disjoint']
                records.append(dict(job=j,record=rel(cp),status=r['status']))
            js(OUT/'ALL_TRAINING_FINISHED.json',dict(protocol_sha256=check_protocol(),utc=utc(),runs=records));q.event('all_training_finished',runs=45)
        barrier();q.event('test_evaluation_start')
        for j in plan:
            parent=OUT/'evaluation'/job_id(j);done=list(parent.glob('attempt_*/COMPLETE.json'))
            if done:assert len(done)==1;verify_record(done[0]);continue
            folder=target(parent/('attempt_'+stamp()));folder.mkdir(parents=True)
            with target(folder/'worker.log').open('x') as log:
                r=subprocess.run([sys.executable,'-B',str(q.WORKER),'evaluate',json.dumps(j),str(folder)],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
            assert r.returncode==0,'TEST error: '+str(folder)
            verify_record(folder/'COMPLETE.json');print('TEST',job_id(j),'COMPLETE',flush=True)
        q.helper('analyze_f_v2.py');verify_record(OUT/'ANALYSIS_COMPLETE.json')
        q.helper('validate_f.py');verify_record(OUT/'VALIDATION_COMPLETE.json')
        q.event('queue_complete');print('COMPLETE:',OUT/'REPORT_STEP22F_FINAL.md',OUT/'REPORT_STEP22F_VALIDATION.md',flush=True)
        subprocess.run(['/usr/bin/afplay','/System/Library/Sounds/Glass.aiff'],check=False)
    except BaseException as error:
        for r in q.ACTIVE.values():
            if r['process'].poll() is None:r['process'].terminate()
        for r in q.ACTIVE.values():r['process'].wait();r['handle'].close()
        q.ACTIVE.clear();p=OUT/('STOP_REPORT_'+stamp()+'.md');text(p,'# BLOCKING\n\n'+repr(error)+'\n\nAttempts preserved; no gate bypass or change of seed/configuration.\n\n```text\n'+traceback.format_exc()+'\n```\n');print('STOP:',p,flush=True);raise
    finally:stop.set();thread.join(timeout=2);awake.terminate();awake.wait();lock.close()
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('command',choices=['run','status']);r=a.parse_args();status() if r.command=='status' else execute()
