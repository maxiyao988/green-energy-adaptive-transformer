"""Imported Step22D scheduler, with registered identical-config retry policy."""
import sys
sys.dont_write_bytecode=True
import io_e as io
from io_e import *
sys.path.insert(0,str(io.D))
d=io.module('_22e_scheduler_d',io.D/'run_d_v4.py')
q=d.q
q.__dict__.update(io.exports());q.WORKER=OUT/'worker_e.py'
q.completion=io.latest_terminal
import ast,copy,types,subprocess,fcntl,time,threading,traceback

def successful(job):return io.completion(job)[0] is not None

def import_queue():
    # Preserve the original three-slot scheduling loop. Only pending/retry bookkeeping changes.
    tree=ast.parse((OLD/'run22b_v3.py').read_text())
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='queue'))
    first=fn.body[0];assert isinstance(first,ast.Assign) and first.targets[0].id=='pending'
    first.value=ast.parse('[job for job in jobs if not successful(job)]',mode='eval').body
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.iter)=='list(ACTIVE.items())']
    assert len(loops)==1
    loops[0].body+=ast.parse("""
if data['status'] != 'COMPLETE':
    if failed_attempts(item['job']) >= 2:
        raise RuntimeError('STOP: same run failed twice: '+job_id(item['job']))
    event('identical_retry',job=item['job'],previous_record=rel(path),same_seed_and_configuration=True)
    pending.insert(0,item['job'])
""").body
    node=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    q.__dict__.update(successful=successful,failed_attempts=io.failed_attempts)
    exec(compile(node,str(OLD/'run22b_v3.py'),'exec'),q.__dict__)
    return io.hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
QUEUE_BINDING=import_queue()
snapshot=types.FunctionType(d.snapshot.__code__,globals(),'snapshot',d.snapshot.__defaults__,d.snapshot.__closure__)
q.status_snapshot=snapshot
wait_ac=types.FunctionType(d.wait_ac.__code__,globals(),'wait_ac',d.wait_ac.__defaults__,d.wait_ac.__closure__)

def live_workers():
    lines=subprocess.run(['ps','-Ao','pid=,command='],capture_output=True,text=True,check=True).stdout.splitlines()
    return [l for l in lines if str(OUT/'worker_e.py') in l and (' train ' in l or ' evaluate ' in l)]

def status():
    rows=[]
    for p in (OUT/'fits').glob('*/attempt_*/COMPLETE.json'):
        try:rows.append((p,read(p)))
        except io.json.JSONDecodeError:pass
    good={job_id(r['job']) for _,r in rows if r['status']=='COMPLETE'}
    failed=[r for _,r in rows if r['status']!='COMPLETE']
    ev=list((OUT/'evaluation').glob('*/attempt_*/COMPLETE.json'))
    print(utc(),'Step22E:',len(good),'/60 trained;',len(failed),'failed/interrupted attempts;',len(ev),'/60 TEST evaluated',flush=True)
    for line in live_workers():print(line,flush=True)
    for n in ['DATA_GATE.json','SIGMA_IDENTITY_GATE.json','TIMING_GATE.json','ALL_TRAINING_FINISHED.json','ANALYSIS_COMPLETE.json','VALIDATION_COMPLETE.json']:
        print(n,'present' if (OUT/n).exists() else 'pending',flush=True)
    for p in sorted(OUT.glob('STOP_REPORT_*.md'))[-1:]:print('STOP:',p,flush=True)
    if (OUT/'REPORT_STEP22E_FINAL.md').exists():print('Report:',OUT/'REPORT_STEP22E_FINAL.md',flush=True)

def heartbeat(stop):
    while not stop.wait(60):
        try:
            good=sum(read(p)['status']=='COMPLETE' for p in (OUT/'fits').glob('*/attempt_*/COMPLETE.json'))
            active=[dict(pid=pid,job=r['job'],started_utc=r['started']) for pid,r in list(q.ACTIVE.items())]
            js(OUT/'health'/(stamp()+'.json'),dict(utc=utc(),trained=good,workers=active,all_training_finished=(OUT/'ALL_TRAINING_FINISHED.json').exists(),test_evaluated=len(list((OUT/'evaluation').glob('*/attempt_*/COMPLETE.json'))),load_average=list(os.getloadavg())))
            print('HEARTBEAT',utc(),good,'/60 trained;',len(active),'active;',len(list((OUT/'evaluation').glob('*/attempt_*/COMPLETE.json'))),'/60 TEST',flush=True)
        except (io.json.JSONDecodeError,RuntimeError):pass  # an exclusive completion writer can still be flushing

def recover_partials(plan):
    # A former controller may have died while its independent worker was finishing.
    while live_workers():
        print('Waiting for existing orphan workers; replacements will not overlap them.',flush=True);time.sleep(5)
    for j in plan:
        for folder in sorted((OUT/'fits'/job_id(j)).glob('attempt_*')):
            if not (folder/'COMPLETE.json').exists():
                paths=[p for p in folder.iterdir() if p.is_file()]
                js(folder/'COMPLETE.json',dict(protocol_sha256=check_protocol(),job=j,status='INTERRUPTED',
                  error='Previous controller/worker exited without a terminal record; identical retry only',epochs=None,best_epoch=None,
                  started_utc=None,ended_utc=utc(),artifacts=[artifact(p) for p in paths]))
                q.event('interrupted_attempt_retained',job=j,folder=rel(folder))
        assert io.failed_attempts(j)<2,'STOP: same run failed twice: '+job_id(j)
        cp,r=io.completion(j)
        if cp is not None and not (cp.parent/'ADAPTER_COMPLETE.json').exists():
            p=cp.parent/'POST_INTERRUPT_ADAPTER_VERIFICATION.json'
            if not p.exists():js(p,dict(utc=utc(),protocol_sha256=check_protocol(),job=j,verified_model_record=artifact(cp),missing_ancillary_adapter=True,model_will_not_be_retrained=True))

def timing(pilots):
    if (OUT/'TIMING_GATE.json').exists():
        a=read(OUT/'TIMING_GATE.json');assert a['proceed'],'STOP: timing projection >9h';return a
    assert all(successful(j) for j in pilots)
    a=read(OUT/'TIMING_PILOT_START.json');b=read(OUT/'TIMING_PILOT_END.json');secs=elapsed(a['utc'],b['utc'])
    result=dict(protocol_sha256=check_protocol(),started_utc=a['utc'],ended_utc=b['utc'],pilot_seconds=secs,projected_hours=20*secs/3600,
      threshold_hours=9,proceed=20*secs<=32400,total_runs=60,workers=3,pilots=pilots,
      monotonic_seconds=b['monotonic']-a['monotonic'] if a['pid']==b['pid'] else None,formula='first three-cell batch elapsed x20; attempts included')
    js(OUT/'TIMING_GATE.json',result);assert result['proceed'],'STOP: timing projection >9h; TEST locked'
    print('TIMING PASS: projected',round(result['projected_hours'],3),'hours',flush=True)
    return result

def execute():
    check_protocol()
    lockpath=OUT/'QUEUE_LOCK'
    if not lockpath.exists():text(lockpath,'Immutable inode; ownership is the live OS flock.\n')
    lock=lockpath.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'VALIDATION_COMPLETE.json').exists():verify_record(OUT/'VALIDATION_COMPLETE.json');print('Already finished.');return
    awake=subprocess.Popen(['/usr/bin/caffeinate','-dimsu','-w',str(os.getpid())])
    stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(stop,),daemon=True);thread.start()
    try:
        q.event('queue_start',pid=os.getpid(),caffeinate_command=['/usr/bin/caffeinate','-dimsu','-w',str(os.getpid())],caffeinate_pid=awake.pid,scheduling_binding_sha256=QUEUE_BINDING)
        assert read(OUT/'DATA_GATE.json')['passed'] and read(OUT/'SIGMA_IDENTITY_GATE.json')['passed']
        plan=[spec(m,w,h,s) for s in SEEDS for h in [20,10,5] for w in ['W1','W2'] for m in ['static','adaptive']]
        if not (OUT/'GRID_PLAN.json').exists():js(OUT/'GRID_PLAN.json',dict(protocol_sha256=check_protocol(),jobs=plan))
        assert read(OUT/'GRID_PLAN.json')['jobs']==plan
        if not (OUT/'ALL_TRAINING_FINISHED.json').exists():
            recover_partials(plan);wait_ac()
            pilots=[spec('static','W1',20,42),spec('adaptive','W2',20,42),spec('static','W2',20,0)]
            if not (OUT/'TIMING_PILOT_START.json').exists():js(OUT/'TIMING_PILOT_START.json',dict(utc=utc(),pid=os.getpid(),monotonic=time.perf_counter(),jobs=pilots,protocol_sha256=check_protocol()))
            q.queue(pilots)
            if not (OUT/'TIMING_PILOT_END.json').exists():js(OUT/'TIMING_PILOT_END.json',dict(utc=utc(),pid=os.getpid(),monotonic=time.perf_counter(),protocol_sha256=check_protocol()))
            timing(pilots);q.queue(plan)
            records=[]
            for j in plan:
                cp,r=io.completion(j);assert cp is not None and r['status']=='COMPLETE'
                assert r['lr']==j['learning_rate'] and r['delta']==j['delta']
                records.append(dict(job=j,record=rel(cp),status=r['status']))
            js(OUT/'ALL_TRAINING_FINISHED.json',dict(protocol_sha256=check_protocol(),utc=utc(),runs=records));q.event('all_training_finished',runs=60)
        barrier();q.event('test_evaluation_start')
        for j in plan:
            parent=OUT/'evaluation'/job_id(j);done=sorted(parent.glob('attempt_*/COMPLETE.json'))
            if done:
                assert len(done)==1;verify_record(done[0]);continue
            folder=target(parent/('attempt_'+stamp()));folder.mkdir(parents=True)
            with target(folder/'worker.log').open('x') as log:
                r=subprocess.run([sys.executable,'-B',str(q.WORKER),'evaluate',json.dumps(j),str(folder)],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
            assert r.returncode==0,'TEST evaluation error: '+str(folder)
            verify_record(folder/'COMPLETE.json');print('TEST',job_id(j),'COMPLETE',flush=True)
        q.helper('analyze_e.py');verify_record(OUT/'ANALYSIS_COMPLETE.json')
        q.helper('validate_e.py');verify_record(OUT/'VALIDATION_COMPLETE.json')
        q.event('queue_complete');print('COMPLETE:',OUT/'REPORT_STEP22E_FINAL.md',OUT/'REPORT_STEP22E_VALIDATION.md',flush=True)
        subprocess.run(['/usr/bin/afplay','/System/Library/Sounds/Glass.aiff'],check=False)
    except BaseException as error:
        for item in q.ACTIVE.values():
            if item['process'].poll() is None:item['process'].terminate()
        for item in q.ACTIVE.values():item['process'].wait();item['handle'].close()
        q.ACTIVE.clear()
        p=OUT/('STOP_REPORT_'+stamp()+'.md')
        text(p,'# BLOCKING\n\n'+repr(error)+'\n\nExisting attempts preserved. No configuration/seed change or gate bypass. TEST remains locked until the registered all-training barrier.\n\n```text\n'+traceback.format_exc()+'\n```\n')
        print('STOP:',p,flush=True);raise
    finally:
        stop.set();thread.join(timeout=2);awake.terminate();awake.wait();lock.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','status']);a=p.parse_args()
    status() if a.command=='status' else execute()
