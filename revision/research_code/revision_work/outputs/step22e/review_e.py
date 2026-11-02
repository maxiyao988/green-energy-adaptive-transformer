"""No-training review: imported factories, synthetic estimator/scheduler tests, TEST lock."""
import sys
sys.dont_write_bytecode=True
import io_e as io
import ast,subprocess,types,importlib.metadata

def guard_review():
    import worker_e_v2 as w
    dest=io.target(io.OUT/'qa'/('guard_'+io.stamp()));dest.mkdir(parents=True)
    sentinel=dest/'test.parquet';io.text(sentinel,'synthetic-only bytes; not research TEST data')
    expected=io.sha(sentinel);w.install_test_guard('train')
    assert io.sha(sentinel)==expected
    blocked=0
    for fn in [lambda:sentinel.open('rb'),lambda:w.old.pd.read_parquet(sentinel),lambda:w.old.pd.read_csv(sentinel)]:
        try:fn()
        except RuntimeError:blocked+=1
        else:raise AssertionError('TEST lock allowed a deserializer')
    ds=w.load('W1',False);assert not (ds.split=='test').any()
    try:w.old.np.load(io.B2/'tensors/W1/y_h5.npy',mmap_mode='r',allow_pickle=False)
    except AssertionError:blocked+=1
    else:raise AssertionError('Full-pack numpy read escaped selected TRAIN/VAL path')
    io.js(io.OUT/'GUARD_REVIEW.json',dict(utc=io.utc(),passed=True,opaque_sha_allowed=True,blocked_paths=blocked,trainval_rows=len(ds.split),research_test_deserialized=False,optimizer_steps=0))
    print('TEST lock PASS: pure SHA permitted; direct/pandas/full-label deserialization blocked; TRAIN/VAL selector works.')

def main():
    started=io.utc()
    import worker_e_v2 as w
    import numpy as np
    dest=io.target(io.OUT/'qa'/('synthetic_'+io.stamp()));dest.mkdir(parents=True)
    models=[];prefixes=[]
    for seed in [42,0]:
        for model in ['static','adaptive']:
            j=io.spec(model,'W1',5,seed);w.configure(j);m=w.old.model_factory(j);m.eval()
            count=w.old.formal.count_trainable_parameters(m);assert count=={'static':107785,'adaptive':116297}[model]
            with w.old.torch.no_grad():
                p,d,a=m.forward_with_router(w.old.torch.zeros(2,90,5),w.old.torch.zeros(2,26,5))
            assert p.shape==(2,) and d.shape==a.shape==(2,4) and w.old.torch.isfinite(p).all()
            owner=w.old.exp4 if seed!=42 else w.old.formal if model=='adaptive' else w.old.exp3
            name='train_one_cell' if owner is w.old.formal else 'train_and_validate_cell'
            code,excerpt=w.old.prefix(owner,name);prefixes.append(dict(model=model,seed_class=seed,**excerpt))
            models.append(dict(model=model,seed_class=seed,parameters=count,repr=str(m),factory_module=owner.__file__))
    io.js(io.OUT/'FACTORY_REVIEW.json',dict(utc=io.utc(),passed=True,profile=w.old.profile(),models=models,prefixes=prefixes,bindings=w.d.BINDINGS,optimizer_steps=0,input='synthetic zeros only'))
    from reuse_d import statistics
    G=statistics(dest);s=G['s'];rng=np.random.default_rng(947)
    for h in io.HS:
        for window in ['W1','W2','W3']:
            di,wt=G['draws'](window,h)
            assert di.shape==(20000,391) and wt.shape==(20000,5) and di.min()>=0 and di.max()<391
            a=np.arange(5)[:,None]+np.arange(391)[None,:]/1000
            actual=G['joint_boot'](a,window,h)
            direct=np.array([(wt[i]@a[:,di[i]]).mean() for i in range(32)])
            assert np.allclose(actual[:32],direct,rtol=0,atol=1e-14)
        assert not np.array_equal(G['draws']('W1',h)[0],G['draws']('W2',h)[0])
    series={window:rng.normal(.02,.1,391) for window in ['W1','W2','W3']}
    r=G['series_result'](series,['W1','W2'],10,10,tag='synthetic_two_sided')
    se=np.sqrt(sum(s.nw_t(series[window],10)[1]**2 for window in ['W1','W2']))/2
    assert abs(r['nw_se']-se)<1e-15 and abs(r['p_nw']-2*s.stats.norm.sf(abs(r['nw_t'])))<1e-15
    r=G['series_result'](series,['W1','W2'],5,4,one_sided=True,tag='synthetic_one_sided')
    assert abs(r['p_nw']-s.stats.norm.sf(r['nw_t']))<1e-15
    bad={window:a.copy() for window,a in series.items()};bad['W1'][0]=np.nan
    r=G['series_result'](bad,['W1','W2'],5,5,tag='synthetic_undefined');assert not r['full_support'] and np.isnan(r['p_nw'])
    y=rng.normal(0,.1,6647);y[:10]=0
    G['all_labels']={('W1',5):dict(y=y,m=0,gtm_mse=np.mean(y*y),zero_mse=np.mean(y*y))}
    r,loss,ic=G['per_seed'](y,'W1',5);assert r['n_valid_da']==6637 and r['da']==1 and r['mse']==0
    assert s.holm({0:.01,1:1.,2:.1})[0]==.03
    assert np.quantile(np.full(20000,-.02),.95,method='linear')<0
    # Exercise the actual imported queue on fake processes only: no subprocess training.
    import run_e_v2 as scheduler
    q=scheduler.q;saved=dict(q.__dict__);terminal={};launches=[];maxactive=[0];serial=[0];fail_limit={}
    q.OUT=dest/'fake_scheduler';q.ACTIVE={};q.event=lambda *a,**kw:None;q.status_snapshot=lambda:None
    q.successful=lambda j:any(r['status']=='COMPLETE' for r in terminal.get(j['id'],[]))
    q.completion=lambda j:(io.Path('/synthetic/COMPLETE.json'),terminal[j['id']][-1]) if terminal.get(j['id']) else (None,None)
    q.failed_attempts=lambda j:sum(r['status']!='COMPLETE' for r in terminal.get(j['id'],[]))
    q.job_id=lambda j:j['id'];q.rel=lambda p:str(p);q.time=types.SimpleNamespace(sleep=lambda _:None)
    class Fake:
        def __init__(self,args,**kw):
            self.j=io.json.loads(args[-2]);serial[0]+=1;self.pid=serial[0];self.polls=0;launches.append(self.j['id'])
        def poll(self):
            maxactive[0]=max(maxactive[0],len(q.ACTIVE));self.polls+=1
            if self.polls<2:return None
            n=len(terminal.get(self.j['id'],[]));return 1 if n<fail_limit.get(self.j['id'],0) else 0
    q.subprocess=types.SimpleNamespace(Popen=Fake)
    def completed(item,code):terminal.setdefault(item['job']['id'],[]).append(dict(status='COMPLETE' if code==0 else 'FAILED'))
    q.failed_record=completed
    terminal['already']=[dict(status='COMPLETE')];fail_limit['retry']=1
    q.queue([dict(id=n) for n in ['already','retry','b','c','d']])
    assert maxactive[0]==3 and launches.count('retry')==2 and 'already' not in launches and all(terminal[n][-1]['status']=='COMPLETE' for n in ['retry','b','c','d'])
    fail_limit['twice']=2
    try:q.queue([dict(id='twice')])
    except RuntimeError:pass
    else:raise AssertionError('Second failure did not stop queue')
    assert launches.count('twice')==2 and not q.ACTIVE
    q.__dict__.clear();q.__dict__.update(saved)
    for p in io.OUT.glob('*.py'):ast.parse(p.read_text(),str(p))
    # Transitive repository module provenance, plus registered AST-only source modules.
    paths=set()
    for m in list(sys.modules.values()):
        file=getattr(m,'__file__',None)
        if file:
            p=io.Path(file).resolve()
            if p.is_relative_to(io.ROOT) and not p.is_relative_to(io.ROOT/'.venv') and p.suffix=='.py' and p.is_file():paths.add(p)
    paths.update([io.ROOT/'revision_work/scripts/skill/a2_skill.py',io.B2/'scripts/b2_data.py',io.B2/'scripts/b2_worker.py'])
    io.csv(io.OUT/'REUSED_MODULES.csv',w.old.pd.DataFrame([dict(path=io.rel(p),sha256=io.sha(p),role='new implementation' if p.is_relative_to(io.OUT) else 'reused module') for p in sorted(paths)]))
    subprocess.run([sys.executable,'-B',str(io.OUT/'review_e.py'),'guard'],cwd=io.ROOT,check=True)
    io.js(io.OUT/'PRERUN_REVIEW.json',dict(started_utc=started,ended_utc=io.utc(),passed=True,optimizer_steps=0,research_test_deserialized=False,
       synthetic_rng_seed=947,bootstrap_base=20261022,B=20000,checks=['factory parameter counts and synthetic forward','original trainer-prefix bindings','independent windows and direct paired-seed resampling','NW variance combination and tails','undefined cells retained','DA zero exclusion','Holm placeholder','negative bounds retained','three-worker refill, successful skip, identical retry, stop on second failure','pure SHA TEST-lock exception and TRAIN/VAL-only selector','all scripts parse'],queue_binding_sha256=scheduler.QUEUE_BINDING,module_count=len(paths),artifacts=[io.artifact(io.OUT/n) for n in ['FACTORY_REVIEW.json','GUARD_REVIEW.json','REUSED_MODULES.csv']]))
    print('PRE-RUN REVIEW PASS; no training started; no research TEST values read.',flush=True)
if __name__=='__main__':guard_review() if len(sys.argv)>1 and sys.argv[1]=='guard' else main()
