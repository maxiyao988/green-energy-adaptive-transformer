"""Pre-registration QA: actual imported paths, synthetic metrics; zero optimizer steps."""
import sys,ast,subprocess,types
sys.dont_write_bytecode=True
import io_g as io

def guard_review():
    import worker_g as w
    folder=io.target(io.OUT/'qa'/('guard_'+io.stamp()));folder.mkdir(parents=True);p=folder/'test.parquet';io.text(p,'synthetic bytes only')
    expected=io.sha(p);w.install_test_guard('train');assert io.sha(p)==expected
    blocked=0
    for fn in [lambda:p.open('rb'),lambda:w.old.pd.read_parquet(p),lambda:w.old.pd.read_csv(p)]:
        try:fn()
        except RuntimeError:blocked+=1
        else:raise AssertionError('TEST lock failed')
    ds=w.dataset(False);assert not (ds.split=='test').any()
    loaders=w.old.formal._build_training_loaders('daily_weekly',5,ds,batch_size=32)
    assert loaders[0].generator.initial_seed()==42 and len(loaders[2].dataset)==0
    try:w.old.np.load(io.TENSOR_BASE/'tensors/y_h5_frozen.npy',mmap_mode='r')
    except AssertionError:blocked+=1
    else:raise AssertionError('Full labels escaped safe selector')
    io.js(io.OUT/'GUARD_REVIEW.json',dict(utc=io.utc(),passed=True,blocked_accesses=blocked,opaque_sha_allowed=True,trainval_selector_passed=True,loader_generator_seed=42,optimizer_steps=0,research_test_deserialized=False))
    print('GUARD PASS',flush=True)

def main():
    start=io.utc();import worker_g as w
    np,pd,torch=w.old.np,w.old.pd,w.old.torch
    dest=io.target(io.OUT/'qa'/('review_'+io.stamp()));dest.mkdir(parents=True)
    records=[];prefixes=[]
    for seed in [42,0]:
        for name in io.VARIANTS:
            job=io.spec(name,5,seed);model=w.old.model_factory(job).eval()
            with torch.no_grad():
                if name=='lstm':pred=model(torch.zeros(2,90,5),torch.zeros(2,26,5))
                else:
                    pred,daily,weekly=model.forward_with_router(torch.zeros(2,90,5),torch.zeros(2,26,5));assert daily.shape==weekly.shape==(2,4)
            assert pred.shape==(2,) and torch.isfinite(pred).all()
            owner=w.old.lstm if name=='lstm' else w.old.exp4 if seed!=42 else w.old.formal if name=='adaptive' else w.old.exp3
            _,excerpt=w.old.prefix(owner,'train_and_eval_frequency' if name=='lstm' else 'train_one_cell' if owner is w.old.formal else 'train_and_validate_cell')
            records.append(dict(model=name,seed=seed,parameters=sum(p.numel() for p in model.parameters())));prefixes.append(excerpt)
    # TRAIN/VAL-only target arithmetic, no research TEST labels/predictions.
    ds=w.dataset(False);info=w.INFO[id(ds)]
    sigma=io.read(io.OUT/'SIGMAS_CS.json')['values']
    for h in io.HS:
        y=info['cs64'][h];assert float(np.std(y[ds.split=='train'],ddof=1,dtype=np.float64))==sigma[str(h)]
        for d in np.unique(ds.anchor_date):assert abs(y[ds.anchor_date==d].mean())<1e-14
        assert np.array_equal(ds.y[h],y.astype(np.float32))
    # Exercise the actual post-inference suffix using entirely synthetic 391x17 data.
    dates=np.repeat(pd.date_range('2001-01-01',periods=391).to_numpy(),17);tick=np.tile(np.array([f'T{i:02}' for i in range(17)]),391)
    rng=np.random.default_rng(947);raw=rng.normal(0,.1,len(tick)).astype(np.float32)
    fake=w.old.common.PanelDataset(tick,dates,np.full(len(tick),'test'),None,None,{h:raw.copy() for h in io.HS});fake=w.center(fake)
    frame=pd.DataFrame(dict(ticker=fake.ticker,anchor_date=fake.anchor_date,split='test',y_true=fake.y[5],y_pred=fake.y[5]*.4))
    folder=dest/'synthetic_evaluation';folder.mkdir()
    original_check=io.check_protocol;original_old=w.old.check_protocol
    io.check_protocol=lambda:io.sha(io.OUT/'PROTOCOL_STEP22G.md');w.old.check_protocol=io.check_protocol
    try:w.saved_evaluation(io.spec('lstm',5,0),folder,fake,frame,'synthetic-only-no-research-TEST')
    finally:io.check_protocol=original_check;w.old.check_protocol=original_old
    mt=pd.read_csv(folder/'metrics.csv');f=pd.read_csv(folder/'predictions_losses_weights.csv',float_precision='round_trip')
    assert mt.global_train_mean.iloc[0]==0 and np.array_equal(f.y_true,w.INFO[id(fake)]['cs64'][5])
    assert np.array_equal(f.y_realized.to_numpy(np.float32),raw)
    assert abs(mt.mse.iloc[0]-np.mean((f.y_pred-f.y_true)**2))<1e-14
    # Exact Step16 RNG and missing-IC estimator, synthetic only.
    b=io.module('_22g_qa_statistics',io.OLD/'analyze22b_v3.py');s=b.s
    assert s.BASE==20261019 and s.B==20000
    for h in io.HS:
        X=np.ascontiguousarray(rng.normal(-.001,.01,(5,391)));theta,boot,_=s.distributions(X,h);wt,di,si=s.draws(h,5,391)
        assert np.max(np.abs(np.array([X[si[i]][:,di[i]].mean() for i in range(24)])-boot[:24]))<1e-14
        assert abs(theta-X.mean())<1e-15 and boot.shape==(20000,)
        X[0,5]=np.nan;_,boot,_=s.distributions(X,h)
        assert np.max(np.abs(np.array([np.nanmean(X[si[i]][:,di[i]],axis=1).mean() for i in range(24)])-boot[:24]))<1e-14
    assert s.holm({0:.01,1:1.,2:.2})[0]==.03
    assert np.quantile(-np.ones(20000),.95,method='linear')==-1
    tree=ast.parse((io.OUT/'analyze_g_v2.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    node=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='J' and isinstance(t.slice,ast.Constant) and t.slice.value=='direction_label' for t in n.targets))
    J=pd.DataFrame(dict(theta_hat=[-.1,.1,-.1,.1,.1],p_holm_3h=[.01,.01,.01,.01,.1],metric=['mse','mse','rank_ic','rank_ic','mse']))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-direction-label','exec'),dict(J=J))
    assert list(J.direction_label)==['significant improvement of the first arm','significant deterioration of the first arm','significant deterioration of the first arm','significant improvement of the first arm','not significant']
    # Reused original portfolio functions on synthetic returns: 200% gross/full L1 costs.
    engine=io.module('_22g_review_engine',io.ROOT/'revision_work/scripts/portfolio/engine.py')
    returns=np.zeros((12,17));weights=np.zeros(17);weights[:5]=.2;weights[-5:]=-.2
    days,gross,turn,period=engine.simulate(returns,{},[1,6],[weights.copy(),weights.copy()],11)
    net,pnet=engine.apply_costs(days,gross,[1,6],turn,period,.001)
    assert np.isclose(turn[0],2) and np.isclose(net[0],-.002) and np.isclose(weights.sum(),0)
    # Three-worker scheduler, skip, retry, second-failure STOP; fake processes only.
    import run_g_v2 as run
    q=run.q;saved=dict(q.__dict__);terminal={};launches=[];limit={'retry':1,'twice':2};serial=[0];peak=[0]
    q.OUT=dest/'scheduler';q.ACTIVE={};q.event=lambda *a,**kw:None;q.status_snapshot=lambda:None
    q.successful=lambda j:any(r['status']=='COMPLETE' for r in terminal.get(j['id'],[]))
    q.completion=lambda j:(io.Path('/synthetic/COMPLETE.json'),terminal[j['id']][-1]) if terminal.get(j['id']) else (None,None)
    q.failed_attempts=lambda j:sum(r['status']!='COMPLETE' for r in terminal.get(j['id'],[]));q.job_id=lambda j:j['id'];q.rel=lambda p:str(p);q.time=types.SimpleNamespace(sleep=lambda _:None)
    class Fake:
        def __init__(self,args,**kw):self.job=io.json.loads(args[-2]);serial[0]+=1;self.pid=serial[0];self.n=0;launches.append(self.job['id'])
        def poll(self):
            peak[0]=max(peak[0],len(q.ACTIVE));self.n+=1
            return None if self.n<2 else 1 if len(terminal.get(self.job['id'],[]))<limit.get(self.job['id'],0) else 0
    q.subprocess=types.SimpleNamespace(Popen=Fake,STDOUT=subprocess.STDOUT)
    q.failed_record=lambda item,code:terminal.setdefault(item['job']['id'],[]).append(dict(status='COMPLETE' if code==0 else 'FAILED'))
    terminal['already']=[dict(status='COMPLETE')];q.queue([dict(id=n) for n in ['already','retry','a','b','c']])
    assert peak[0]==3 and launches.count('retry')==2 and 'already' not in launches
    try:q.queue([dict(id='twice')])
    except RuntimeError:pass
    else:raise AssertionError('Second failure must STOP')
    assert launches.count('twice')==2 and not q.ACTIVE
    q.__dict__.clear();q.__dict__.update(saved)
    original_out=run.OUT;run.OUT=dest/'snapshot';run.q.ACTIVE={999:dict(job=io.spec('static',5,0),started=io.utc())};run.snapshot();run.q.ACTIVE.clear();run.OUT=original_out
    for p in io.OUT.glob('*.py'):ast.parse(p.read_text(),str(p))
    subprocess.run([sys.executable,'-B',str(io.OUT/'review_g.py'),'guard'],cwd=io.ROOT,check=True)
    paths=set()
    for m in list(sys.modules.values()):
        value=getattr(m,'__file__',None)
        if value:
            p=io.Path(value).resolve()
            if p.is_relative_to(io.ROOT) and not p.is_relative_to(io.ROOT/'.venv') and p.is_file() and p.suffix=='.py':paths.add(p)
    paths.update([io.ROOT/'revision_work/outputs/frozen/scripts/frozen_report.py',io.ROOT/'revision_work/scripts/skill/a3_hierarchical.py',io.E/'worker_e_v2.py',io.ROOT/'revision_work/outputs/step22d/session_v3/run_d_v4.py'])
    io.csv(io.OUT/'REUSED_MODULES.csv',pd.DataFrame([dict(path=io.rel(p),sha256=io.sha(p)) for p in sorted(paths)]))
    io.js(io.OUT/'PRERUN_REVIEW.json',dict(started_utc=start,ended_utc=io.utc(),passed=True,optimizer_steps=0,research_test_deserialized=False,model_parameter_counts=records,original_prefixes=prefixes,bindings=w.BINDINGS,profile=w.old.profile(),synthetic_qa_seed=947,checks=['TRAIN-only float64 sigma/demeaning/cast','all model factories and original training prefixes','synthetic actual evaluation suffix: zero reference/float64 CS labels/original returns','Step16 joint bootstrap and missing-IC branch','Holm placeholders/signed bounds/direction labels','actual portfolio full-gross L1 costs','three-slot queue,skip,retry,second-failure STOP,snapshot schema','TEST deserializer guard/hash exception/safe TRAINVAL selector/generator42','all syntax'],artifacts=[io.artifact(io.OUT/'REUSED_MODULES.csv'),io.artifact(io.OUT/'GUARD_REVIEW.json')]))
    print('PRERUN REVIEW PASS; zero optimizer steps, no research TEST content.',flush=True)
if __name__=='__main__':guard_review() if len(sys.argv)>1 and sys.argv[1]=='guard' else main()
