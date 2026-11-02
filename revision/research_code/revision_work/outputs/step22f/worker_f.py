"""Original Step22B trainers/inference; Adam parameter groups are the sole numerical change."""
import sys,os,ast,types
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import io_f as io
sys.path.insert(0,str(io.OLD));sys.path.insert(0,str(io.ROOT))
old=io.module('_22f_worker_b',io.OLD/'worker22b_v3.py');old.__dict__.update(io.exports())
load=old.dataset
node=next(n for n in ast.parse((io.E/'worker_e_v2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='install_test_guard')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(io.E/'worker_e_v2.py'),'exec'),globals())
GROUPS=[];MODELS=[]
BASE_ADAM=old.torch.optim.Adam

def group_optimizer(model,params,job,original=BASE_ADAM,**kwargs):
    values=list(params);named=list(model.named_parameters());lookup={id(p):n for n,p in named}
    assert all(isinstance(v,old.torch.nn.Parameter) for v in values)
    assert len(values)==len(named)==len({id(v) for v in values}) and set(lookup)=={id(v) for v in values}
    accelerated=lambda name:name.endswith('.aggregator.scale_logits') if job['model']=='static' else '.aggregator.router.' in name
    fast=[v for v in values if accelerated(lookup[id(v)])];base=[v for v in values if not accelerated(lookup[id(v)])]
    assert sum(v.numel() for v in fast)==(8 if job['model']=='static' else 8520)
    assert sum(v.numel() for v in base)==107777
    assert {lookup[id(v)].split('.')[0] for v in fast}=={'daily_branch','weekly_branch'}
    assert kwargs.get('lr',1e-3)==1e-4
    opt=original([dict(params=base,lr=1e-4,group_role='base'),dict(params=fast,lr=1e-4*job['multiplier'],group_role='accelerated')],**kwargs)
    return opt,group_record(opt,model,job)

def group_record(opt,model,job):
    named={id(p):n for n,p in model.named_parameters()};rows=[]
    for group in opt.param_groups:
        names=[named[id(p)] for p in group['params']]
        rows.append(dict(options={k:v for k,v in group.items() if k!='params'},parameter_names=names,parameter_ids=[id(p) for p in group['params']],parameter_count=sum(p.numel() for p in group['params'])))
    keys=set(rows[0]['options'])-{'lr','group_role'}
    assert keys==set(rows[1]['options'])-{'lr','group_role'}
    assert all(rows[0]['options'][k]==rows[1]['options'][k] for k in keys)
    assert rows[0]['options']['lr']==1e-4 and rows[1]['options']['lr']==1e-4*job['multiplier']
    return dict(utc=io.utc(),job=job,optimizer_class=type(opt).__module__+'.'+type(opt).__qualname__,groups=rows,defaults=opt.defaults,
      exhaustive=True,disjoint=True,same_betas_epsilon_weight_decay=True,all_parameters_trainable=all(p.requires_grad for p in model.parameters()))

def install_optimizer(job,folder):
    def capture(factory):
        def make(*a,**kw):
            model=factory(*a,**kw);MODELS.append(model);return model
        return make
    factories=[(old.formal,'_make_model'),(old.exp3,'_make_experiment3_model'),(old.exp4,'_make_experiment4_model')]
    saved=[(owner,name,getattr(owner,name)) for owner,name in factories]
    for owner,name,fn in saved:setattr(owner,name,capture(fn))
    def adam(params,*a,**kw):
        assert not a and MODELS and not GROUPS
        opt,record=group_optimizer(MODELS[-1],params,job,**kw);GROUPS.append(opt)
        io.js(folder/'OPTIMIZER_GROUPS.json',dict(protocol_sha256=io.check_protocol(),**record))
        return opt
    old.torch.optim.Adam=adam
    def restore():
        old.torch.optim.Adam=BASE_ADAM
        for owner,name,fn in saved:setattr(owner,name,fn)
    return restore

def main():
    action,encoded,where=sys.argv[1:];job=io.json.loads(encoded);assert job==io.spec(job['variant'],job['h'],job['seed'])
    assert action in ['train','evaluate'];io.check_protocol()
    folder=io.target(io.Path(where));folder.mkdir(parents=True,exist_ok=True)
    io.js(folder/'POWER_START.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
    guard=install_test_guard(action);io.js(folder/'TEST_LOCK.json',dict(utc=io.utc(),action=action,**guard))
    restore=install_optimizer(job,folder) if action=='train' else lambda:None
    try:
        (old.train if action=='train' else old.evaluate)(job,folder)
        if action=='train' and GROUPS:io.js(folder/'OPTIMIZER_GROUPS_END.json',dict(protocol_sha256=io.check_protocol(),**group_record(GROUPS[0],MODELS[-1],job)))
        io.js(folder/'POWER_END.json',dict(utc=io.utc(),snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
        names=['COMPLETE.json','POWER_START.json','POWER_END.json','TEST_LOCK.json']+(['OPTIMIZER_GROUPS.json','OPTIMIZER_GROUPS_END.json'] if action=='train' and GROUPS else [])
        io.js(folder/'ADAPTER_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),job=job,utc=io.utc(),artifacts=[io.artifact(folder/n) for n in names]))
    except BaseException:
        io.text(folder/('UNHANDLED_'+io.stamp()+'.txt'),old.traceback.format_exc());raise
    finally:restore()
if __name__=='__main__':main()
