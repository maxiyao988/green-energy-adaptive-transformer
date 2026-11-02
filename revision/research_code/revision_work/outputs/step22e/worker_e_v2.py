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
    # Preserve the original writing boundary and TEST lock, adding the explicit SHA exception.
    # Python audit hooks alone do not intercept all C++ Arrow opens, so guard pandas as well.
    old.STAGE='evaluation' if action=='evaluate' else 'training'
    tmp=io.target(io.OUT/'tmp'/str(os.getpid()));tmp.mkdir(parents=True,exist_ok=True)
    old.tempfile.tempdir=str(tmp)
    sources={str((io.ROOT/r[k]).resolve()) for r in io.source_rows() for k in ['source','canonical']}
    def opaque_hash():
        f=sys._getframe(1)
        while f is not None:
            if f.f_code is io.sha.__code__:return True
            f=f.f_back
        return False
    def locked(path):
        p=io.Path(path).resolve();name=p.name.lower()
        return (str(p) in sources or p.is_relative_to(io.ROOT/'revision_work/outputs/analysis16') and p.suffix=='.parquet'
            or name in ['test.parquet','test.csv','predictions_losses_weights.csv','metrics.csv','b_forecast_skill.csv']
            or p.is_relative_to(io.ROOT/'revision_work/outputs/step21'))
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=io.Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and not p.is_relative_to(io.OUT):raise RuntimeError('READ-ONLY source protection: '+str(p))
        if action=='train' and not writing and locked(p) and not opaque_hash():raise RuntimeError('TEST locked during training: '+str(p))
    sys.addaudithook(audit)
    if action=='train':
        for name in ['read_csv','read_parquet']:
            original=getattr(old.pd,name)
            def wrapped(path,*a,_original=original,**kw):
                if isinstance(path,(str,bytes,os.PathLike)) and locked(os.fsdecode(path)):
                    raise RuntimeError('TEST deserializer locked during training: '+str(path))
                return _original(path,*a,**kw)
            setattr(old.pd,name,wrapped)
        original_load=old.np.load
        selected_code=next(c for c in load.__code__.co_consts if isinstance(c,type(load.__code__)) and c.co_name=='selected')
        def np_load(path,*a,**kw):
            if isinstance(path,(str,bytes,os.PathLike)):
                p=io.Path(os.fsdecode(path)).resolve()
                if p.is_relative_to(io.B2/'tensors') and p.suffix=='.npy':
                    assert sys._getframe(1).f_code is selected_code and kw.get('mmap_mode')=='r','Only the unchanged TRAIN/VAL selector may map the full frozen pack'
            return original_load(path,*a,**kw)
        old.np.load=np_load
    # The unchanged trainer invokes install_guard again; keep a single audited guard.
    old.install_guard=lambda evaluation=False:None
    return dict(pure_sha_allowed=True,pandas_guarded=True,numpy_trainval_selector_only=action=='train',source_writes_forbidden=True)

def main():
    action,encoded,where=sys.argv[1:];job=io.json.loads(encoded)
    assert action in ['train','evaluate'];io.check_protocol();configure(job)
    folder=io.target(io.Path(where));folder.mkdir(parents=True,exist_ok=True)
    io.js(folder/'WINDOW_BINDING.json',dict(utc=io.utc(),job=job,counts=old.COUNTS,
       reused_worker=io.artifact(io.D/'worker_d.py'),bindings=d.BINDINGS,
       rolling_count_binding_source=io.artifact(io.B2/'scripts/b2_worker.py'),protocol_sha256=io.check_protocol()))
    io.js(folder/'POWER_START.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
    guard=install_test_guard(action)
    io.js(folder/'TEST_LOCK.json',dict(utc=io.utc(),action=action,**guard))
    try:
        (old.train if action=='train' else old.evaluate)(job,folder)
        io.js(folder/'POWER_END.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
        io.js(folder/'ADAPTER_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),job=job,utc=io.utc(),
          artifacts=[io.artifact(folder/n) for n in ['COMPLETE.json','WINDOW_BINDING.json','POWER_START.json','POWER_END.json']]))
    except BaseException:
        io.text(folder/('UNHANDLED_'+io.stamp()+'.txt'),old.traceback.format_exc());raise

if __name__=='__main__':main()
