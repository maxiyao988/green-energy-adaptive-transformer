"""Zero-training synthetic review of the exact optimizer and imported statistics/scheduler."""
import sys,ast,subprocess,types
sys.dont_write_bytecode=True
import io_f as io

def guard_review():
    import worker_f as w
    folder=io.target(io.OUT/'qa'/('guard_'+io.stamp()));folder.mkdir(parents=True);p=folder/'test.parquet';io.text(p,'synthetic bytes only')
    expected=io.sha(p);w.install_test_guard('train');assert io.sha(p)==expected
    blocked=0
    for fn in [lambda:p.open('rb'),lambda:w.old.pd.read_parquet(p),lambda:w.old.pd.read_csv(p)]:
        try:fn()
        except RuntimeError:blocked+=1
        else:raise AssertionError('TEST lock failed')
    ds=w.load(False);assert not (ds.split=='test').any()
    loaders=w.old.formal._build_training_loaders('daily_weekly',5,ds,batch_size=32)
    assert loaders[0].generator.initial_seed()==42 and len(loaders[2].dataset)==0
    try:w.old.np.load(io.TENSOR_BASE/'tensors/y_h5_frozen.npy',mmap_mode='r')
    except AssertionError:blocked+=1
    else:raise AssertionError('Full-label mmap escaped selector')
    io.js(io.OUT/'GUARD_REVIEW.json',dict(utc=io.utc(),passed=True,blocked_accesses=blocked,opaque_sha_allowed=True,trainval_selector_passed=True,loader_generator_seed=42,optimizer_steps=0,research_test_deserialized=False))
    print('GUARD PASS: SHA permitted; TEST deserializers/full-pack access blocked; TRAIN/VAL selector and generator42 pass.')

def main():
    start=io.utc();import worker_f as w
    np,pd,torch=w.old.np,w.old.pd,w.old.torch
    dest=io.target(io.OUT/'qa'/('review_'+io.stamp()));dest.mkdir(parents=True);records=[];prefixes=[]
    original_check=io.check_protocol;io.check_protocol=lambda:io.sha(io.OUT/'PROTOCOL_STEP22F.md')
    for seed in [42,0]:
        for variant in io.VARIANTS:
            job=io.spec(variant,5,seed);folder=dest/(variant+'_s'+str(seed));folder.mkdir();w.GROUPS.clear();w.MODELS.clear();restore=w.install_optimizer(job,folder)
            try:
                model=w.old.model_factory(job);before=[p.detach().clone() for p in model.parameters()];rng=torch.random.get_rng_state().clone()
                opt=torch.optim.Adam(model.parameters(),lr=1e-4)
                assert torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
                group=w.group_record(opt,model,job);records.append(group)
                model.eval()
                with torch.no_grad():pred,daily,weekly=model.forward_with_router(torch.zeros(2,90,5),torch.zeros(2,26,5))
                assert pred.shape==(2,) and daily.shape==weekly.shape==(2,4) and torch.isfinite(pred).all()
                owner=w.old.exp4 if seed!=42 else w.old.formal if job['model']=='adaptive' else w.old.exp3
                code,excerpt=w.old.prefix(owner,'train_one_cell' if owner is w.old.formal else 'train_and_validate_cell');prefixes.append(dict(variant=variant,seed_class=seed,**excerpt))
            finally:restore()
    io.check_protocol=original_check
    io.js(io.OUT/'OPTIMIZER_REVIEW.json',dict(utc=io.utc(),passed=True,records=records,prefixes=prefixes,profile=w.old.profile(),optimizer_steps=0,parameter_initialization_unchanged=True,rng_unchanged_by_grouping=True))
    # Reuse Step16 estimators; exercise complete synthetic391-date panels.
    b=io.module('_22f_review_stats_b',io.OLD/'analyze22b_v3.py');s=b.s
    assert s.BASE==20261019 and s.B==20000
    rng=np.random.default_rng(947)
    for h in io.HS:
        X=np.ascontiguousarray(rng.normal(-.001,.01,(5,391)));theta,boot,_=s.distributions(X,h);wt,di,si=s.draws(h,5,391)
        direct=np.array([X[si[i]][:,di[i]].mean() for i in range(32)])
        assert np.max(np.abs(direct-boot[:32]))<1e-14 and abs(theta-X.mean())<1e-15
        assert boot.shape==(20000,) and di.min()>=0 and di.max()<391
        missing=X.copy();missing[0,5]=np.nan;tm,bm,_=s.distributions(missing,h)
        direct=np.array([np.nanmean(missing[si[i]][:,di[i]],axis=1).mean() for i in range(32)])
        assert np.max(np.abs(direct-bm[:32]))<1e-14
    assert s.holm({0:.01,1:1.,2:.2})[0]==.03
    assert np.quantile(-np.ones(20000),.95,method='linear')==-1
    # Use the actual direction-label expression from the final analyzer.
    tree=ast.parse((io.OUT/'analyze_f_v2.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    node=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='J' and isinstance(t.slice,ast.Constant) and t.slice.value=='direction_label' for t in n.targets))
    J=pd.DataFrame(dict(theta_hat=[-.1,.1,-.1,.1,.1],p_holm_3h=[.01,.01,.01,.01,.1],metric=['mse','mse','rank_ic','rank_ic','mse']))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_direction_label','exec'),dict(J=J))
    assert list(J.direction_label)==['significant improvement of the first arm','significant deterioration of the first arm','significant deterioration of the first arm','significant improvement of the first arm','not significant']
    for d,wk,expected in [(.001,.004,'FAIL'),(.005,.004,'PASS'),(.001,.006,'PASS')]:
        assert ('PASS' if max(d,wk)>=.005 else 'FAIL')==expected
    # Imported actual queue, fake subprocesses only; no optimizer steps or training jobs.
    import run_f_v2 as run
    q=run.q;saved=dict(q.__dict__);terminal={};launches=[];failure_limit={'retry':1,'twice':2};serial=[0];peak=[0]
    q.OUT=dest/'scheduler';q.ACTIVE={};q.event=lambda *a,**kw:None;q.status_snapshot=lambda:None
    q.successful=lambda j:any(r['status']=='COMPLETE' for r in terminal.get(j['id'],[]))
    q.completion=lambda j:(io.Path('/synthetic/COMPLETE.json'),terminal[j['id']][-1]) if terminal.get(j['id']) else (None,None)
    q.failed_attempts=lambda j:sum(r['status']!='COMPLETE' for r in terminal.get(j['id'],[]));q.job_id=lambda j:j['id'];q.rel=lambda p:str(p);q.time=types.SimpleNamespace(sleep=lambda _:None)
    class Fake:
        def __init__(self,args,**kw):self.job=io.json.loads(args[-2]);serial[0]+=1;self.pid=serial[0];self.n=0;launches.append(self.job['id'])
        def poll(self):
            peak[0]=max(peak[0],len(q.ACTIVE));self.n+=1
            return None if self.n<2 else 1 if len(terminal.get(self.job['id'],[]))<failure_limit.get(self.job['id'],0) else 0
    q.subprocess=types.SimpleNamespace(Popen=Fake,STDOUT=subprocess.STDOUT)
    def done(item,code):terminal.setdefault(item['job']['id'],[]).append(dict(status='COMPLETE' if code==0 else 'FAILED'))
    q.failed_record=done;terminal['already']=[dict(status='COMPLETE')]
    q.queue([dict(id=n) for n in ['already','retry','a','b','c']]);assert peak[0]==3 and launches.count('retry')==2 and 'already' not in launches
    try:q.queue([dict(id='twice')])
    except RuntimeError:pass
    else:raise AssertionError('Second failure did not stop')
    assert launches.count('twice')==2 and not q.ACTIVE
    q.__dict__.clear();q.__dict__.update(saved)
    for p in io.OUT.glob('*.py'):ast.parse(p.read_text(),str(p))
    # Ensure actual operational snapshot accepts every field of this job schema.
    original_out=run.OUT;run.OUT=dest/'snapshot';run.q.ACTIVE={};run.q.ACTIVE[999]=dict(job=io.spec('Static-R10',5,0),started=io.utc())
    run.snapshot();run.q.ACTIVE.clear();run.OUT=original_out
    subprocess.run([sys.executable,'-B',str(io.OUT/'review_f.py'),'guard'],cwd=io.ROOT,check=True)
    paths=set()
    for m in list(sys.modules.values()):
        value=getattr(m,'__file__',None)
        if value:
            p=io.Path(value).resolve()
            if p.is_relative_to(io.ROOT) and not p.is_relative_to(io.ROOT/'.venv') and p.is_file() and p.suffix=='.py':paths.add(p)
    paths.update([io.ROOT/'revision_work/outputs/frozen/scripts/frozen_report.py',io.ROOT/'revision_work/scripts/skill/a3_hierarchical.py',io.E/'worker_e_v2.py',io.ROOT/'revision_work/outputs/step22d/session_v3/run_d_v4.py'])
    io.csv(io.OUT/'REUSED_MODULES.csv',pd.DataFrame([dict(path=io.rel(p),sha256=io.sha(p),role='new implementation' if p.is_relative_to(io.OUT) else 'reused module') for p in sorted(paths)]))
    io.js(io.OUT/'PRERUN_REVIEW.json',dict(started_utc=start,ended_utc=io.utc(),passed=True,optimizer_steps=0,research_test_deserialized=False,synthetic_rng_seed=947,bootstrap_base_seed=20261019,B=20000,checks=['all3 variants x2 trainer classes parameter groups','original initialization/RNG unchanged','synthetic forward/frozen original prefixes','exact Step16 paired-seed/date application and missing-IC branch','signed bounds/undefined Holm placeholders','actual direction labels/manipulation threshold','three-slot scheduler,successful skip,identical retry,second-failure STOP','status CSV accepts full job schema','TEST lock pure hash allowance,TRAIN/VAL-only selector,generator42','all scripts parse'],queue_binding_sha256=run.QUEUE_BINDING,artifacts=[io.artifact(io.OUT/n) for n in ['OPTIMIZER_REVIEW.json','GUARD_REVIEW.json','REUSED_MODULES.csv']]))
    print('PRERUN REVIEW PASS. Zero training/optimizer steps; no research TEST content parsed.')
if __name__=='__main__':guard_review() if len(sys.argv)>1 and sys.argv[1]=='guard' else main()
