"""Formal LSTM prefix imported from Step22B; only window and lr/delta bindings differ."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import io_d as io
sys.path.insert(0,str(io.ROOT));sys.path.insert(0,str(io.OLD))
import worker22b_v3 as old
import ast,copy
from window_data import load
old.__dict__.update(io.exports())

def bind(name):
    tree=ast.parse((io.OLD/'worker22b_v3.py').read_text())
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    changed=[]
    class Bind(ast.NodeTransformer):
        def visit_Assign(self,n):
            self.generic_visit(n)
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['lr','delta']:
                key='learning_rate' if n.targets[0].id=='lr' else 'delta'
                n.value=ast.parse(f"float(job['{key}'])",mode='eval').body
                changed.append(key)
            return n
        def visit_If(self,n):
            self.generic_visit(n)
            if name=='evaluate' and ast.unparse(n.test)=="setting == 'T0'":
                # These T0 fits are NEW W1/W2 cells, never the archived W3 T0 branch.
                n.test=ast.Constant(False);changed.append('new_window_T0_checkpoint')
            return n
    fn=Bind().visit(fn)
    assert sorted(changed)==(['delta','learning_rate'] if name=='train' else ['delta','new_window_T0_checkpoint'])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(module,str(io.OLD/'worker22b_v3.py'),'exec'),old.__dict__)
    return dict(function=name,bindings=changed,original_sha256=io.sha(io.OLD/'worker22b_v3.py'),
                bound_ast_sha256=io.hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest())
BINDINGS=[bind('train'),bind('evaluate')]

if __name__=='__main__':
    action,encoded,destination=sys.argv[1:]
    job=io.json.loads(encoded);assert job['model']=='lstm' and job['window'] in ['W1','W2'] and job['setting'] in ['T0','T3']
    io.check_protocol()
    assert job==io.spec(job['setting'],job['window'],job['h'],job['seed'])
    old.COUNTS=io.counts(job['window'])
    old.dataset=lambda include_test=False:load(job['window'],include_test)
    folder=io.target(io.Path(destination));folder.mkdir(parents=True,exist_ok=True)
    io.js(folder/'WINDOW_BINDING.json',dict(job=job,changes=BINDINGS,protocol_sha256=io.check_protocol(),
       counts=old.COUNTS,source_tensor_manifest=io.rel(io.B2/'tensor_manifest.json')))
    io.js(folder/'POWER_START.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
    def guard(event,args):
        if action=='train' and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=io.Path(os.fsdecode(args[0])).absolute().resolve()
            if path.suffix=='.parquet' and (path.is_relative_to(io.ROOT/'revision_work/outputs/analysis16') or path.name=='test.parquet'):
                raise RuntimeError('TEST prediction/label Parquet locked during training: '+str(path))
    sys.addaudithook(guard)
    try:
        if action=='train':old.train(job,folder)
        elif action=='evaluate':old.evaluate(job,folder)
        else:raise ValueError(action)
        io.js(folder/'POWER_END.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
        io.js(folder/'ADAPTER_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),job=job,utc=io.utc(),
           artifacts=[io.artifact(folder/n) for n in ['COMPLETE.json','WINDOW_BINDING.json','POWER_START.json','POWER_END.json']]))
    except Exception:
        io.text(folder/('UNHANDLED_'+io.stamp()+'.txt'),old.traceback.format_exc());raise
