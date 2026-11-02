"""Step22D training/inference by import; only registered job/window bindings differ."""
import sys,os
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import io_e as io
sys.path.insert(0,str(io.D));sys.path.insert(0,str(io.ROOT))
d=io.module('_22e_worker_d',io.D/'worker_d.py')
old=d.old
old.__dict__.update(io.exports())
from reuse_d import window_loader
load=window_loader()

def configure(job):
    assert job==io.spec(job['model'],job['window'],job['h'],job['seed'])
    ct=io.counts(job['window']);old.COUNTS=ct
    # Same process-local split-count binding as the existing B2 rolling worker.
    old.exp4.N_TRAIN_SAMPLES=ct['train'];old.exp3.N_TRAIN_SAMPLES=ct['train']
    old.dataset=lambda include_test=False:load(job['window'],include_test)

def install_test_guard(action):
    # Import the corrected Step22D guard exactly, including its permitted pure SHA channel.
    import ast
    tree=ast.parse((io.D/'worker_d.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='guard')
    env=dict(io=io,os=os,sys=sys,action=action)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(io.D/'worker_d.py'),'exec'),env)
    sys.addaudithook(env['guard'])

def main():
    action,encoded,where=sys.argv[1:];job=io.json.loads(encoded)
    assert action in ['train','evaluate'];io.check_protocol();configure(job)
    folder=io.target(io.Path(where));folder.mkdir(parents=True,exist_ok=True)
    io.js(folder/'WINDOW_BINDING.json',dict(utc=io.utc(),job=job,counts=old.COUNTS,
       reused_worker=io.artifact(io.D/'worker_d.py'),bindings=d.BINDINGS,
       rolling_count_binding_source=io.artifact(io.B2/'scripts/b2_worker.py'),protocol_sha256=io.check_protocol()))
    io.js(folder/'POWER_START.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
    install_test_guard(action)
    try:
        (old.train if action=='train' else old.evaluate)(job,folder)
        io.js(folder/'POWER_END.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
        io.js(folder/'ADAPTER_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),job=job,utc=io.utc(),
          artifacts=[io.artifact(folder/n) for n in ['COMPLETE.json','WINDOW_BINDING.json','POWER_START.json','POWER_END.json']]))
    except BaseException:
        io.text(folder/('UNHANDLED_'+io.stamp()+'.txt'),old.traceback.format_exc());raise

if __name__=='__main__':main()
