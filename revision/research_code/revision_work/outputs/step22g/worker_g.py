"""Step22B2 T3 prefixes, changing only cross-sectional target/sigma and metric benchmark."""
import os,sys,ast,copy
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import io_g as io
sys.path.insert(0,str(io.B2RUN));sys.path.insert(0,str(io.OLD));sys.path.insert(0,str(io.ROOT))
b2=io.module('_22g_worker_b2',io.B2RUN/'worker_b2_v2.py');old=b2.old;old.__dict__.update(io.exports())
load=old.dataset
INFO={}
def center(raw):
    np,pd=old.np,old.pd;dates,inverse=np.unique(raw.anchor_date,return_inverse=True);count=np.bincount(inverse);assert (count==17).all()
    cs64={};cs32={}
    for h in io.HS:
        y=raw.y[h].astype(np.float64);out=np.empty_like(y)
        for code in range(len(dates)):
            m=inverse==code;assert len(set(raw.split[m]))==1;out[m]=y[m]-np.mean(y[m],dtype=np.float64)
        cs64[h]=out;cs32[h]=out.astype(np.float32)
    ds=old.common.PanelDataset(raw.ticker,raw.anchor_date,raw.split,raw.X_daily,raw.X_weekly,cs32);INFO[id(ds)]=dict(raw=raw,cs64=cs64)
    return ds

def dataset(include_test=False):return center(load(include_test))
old.dataset=dataset

def test_targets(frame,ds,job):
    item=INFO[id(ds)];mask=ds.split=='test'
    f=old.pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=old.pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),cs64=item['cs64'][job['h']][mask],realized=item['raw'].y[job['h']][mask]))
    f=f.sort_values(['anchor_date','ticker']).reset_index(drop=True);assert f[['ticker','anchor_date']].equals(frame[['ticker','anchor_date']])
    return f
old.cs_test_targets=test_targets

def bind(name):
    # Begin with the already-imported Step22B2 T3 binding, then bind its sigma expression only.
    node=copy.deepcopy(next(n for n in ast.parse((io.OLD/'worker22b_v3.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
    node=copy.deepcopy(node)
    class Setting(ast.NodeTransformer):
        def visit_Compare(self,n):
            self.generic_visit(n)
            if len(n.ops)==1 and isinstance(n.ops[0],ast.Eq) and len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value in ['T1','T2'] and ast.unparse(n.left) in ["job['setting']",'setting']:
                v=n.comparators[0].value;n.ops=[ast.In()];n.comparators=[ast.List(elts=[ast.Constant(v),ast.Constant('T3')],ctx=ast.Load())]
            return n
    node=Setting().visit(node);changed=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='delta':n.value=ast.parse("float(job['delta'])",mode='eval').body;changed.append('delta')
        if name=='evaluate' and isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='m':n.value=ast.Constant(0.);changed.append('zero_reference')
    if name=='evaluate':
        position=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and [getattr(v,'id','') for v in t.elts]==['y','p'] for t in n.targets))
        node.body[position:position]=ast.parse("aligned_cs = cs_test_targets(frame, ds, job)\nframe['y_true'] = aligned_cs.cs64.to_numpy(float)\nframe['y_realized'] = aligned_cs.realized.to_numpy(float)").body;changed.append('float64_evaluation_target')
    assert sorted(changed)==(['delta'] if name=='train' else ['delta','float64_evaluation_target','zero_reference'])
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));exec(compile(module,str(io.OLD/'worker22b_v3.py'),'exec'),old.__dict__)
    return dict(function=name,base_b2_binding=b2.BINDINGS,changes=changed,bound_ast_sha256=io.hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest())
BINDINGS=[bind('train'),bind('evaluate')]
# Original Step22E TEST guard with its byte-hash exception and TRAIN/VAL-only selector enforcement.
node=next(n for n in ast.parse((io.E/'worker_e_v2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='install_test_guard')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(io.E/'worker_e_v2.py'),'exec'),globals())

def saved_evaluation(job,folder,ds,frame,source):
    """Ridge shares the exact post-inference Step22B2 evaluation suffix."""
    tree=ast.parse((io.OLD/'worker22b_v3.py').read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate'))
    pos=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='anchor_date' for t in n.targets))
    # Reuse the already CS-bound evaluation function AST from bind(), via the same transformations.
    suffix=node.body[pos:]
    # Replace sigma and the reference and insert aligned float64 labels, identically to bind().
    for n in ast.walk(ast.Module(body=suffix,type_ignores=[])):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id=='delta':n.value=ast.parse("float(job['delta'])",mode='eval').body
            if n.targets[0].id=='m':n.value=ast.Constant(0.)
    idx=next(i for i,n in enumerate(suffix) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and [getattr(v,'id','') for v in t.elts]==['y','p'] for t in n.targets))
    suffix[idx:idx]=ast.parse("aligned_cs = cs_test_targets(frame, ds, job)\nframe['y_true'] = aligned_cs.cs64.to_numpy(float)\nframe['y_realized'] = aligned_cs.realized.to_numpy(float)").body
    old.TELE=io.target(folder/'telemetry.jsonl').open('x');old.telemetry('test_evaluation_start',True)
    env=dict(old.__dict__,job=job,folder=folder,ds=ds,frame=frame,source=source,status='COMPLETE',setting=job['setting'],start=io.utc(),clock=old.time.perf_counter())
    exec(compile(ast.fix_missing_locations(ast.Module(body=suffix,type_ignores=[])),str(io.OLD/'worker22b_v3.py'),'exec'),env)

def main():
    action,encoded,where=sys.argv[1:];job=io.json.loads(encoded);assert job==io.spec(job['model'],job['h'],job['seed']);io.check_protocol()
    folder=io.target(io.Path(where));folder.mkdir(parents=True,exist_ok=True)
    io.js(folder/'TARGET_BINDING.json',dict(protocol_sha256=io.check_protocol(),job=job,bindings=BINDINGS,target='float64 per-date cross-sectional demeaning;float32 training;float64 test;zero reference'))
    io.js(folder/'POWER_START.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
    guard=install_test_guard(action);io.js(folder/'TEST_LOCK.json',dict(utc=io.utc(),action=action,**guard))
    try:
        (old.train if action=='train' else old.evaluate)(job,folder)
        io.js(folder/'POWER_END.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
        io.js(folder/'ADAPTER_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),job=job,utc=io.utc(),artifacts=[io.artifact(folder/n) for n in ['COMPLETE.json','TARGET_BINDING.json','POWER_START.json','POWER_END.json','TEST_LOCK.json']]))
    except BaseException:io.text(folder/('UNHANDLED_'+io.stamp()+'.txt'),old.traceback.format_exc());raise
if __name__=='__main__':main()
