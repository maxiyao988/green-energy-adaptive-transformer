"""Exercise final analyzer decision code with synthetic data, never real TEST data."""
import sys,ast,types
sys.dont_write_bytecode=True
import io_e as io
for k in io.ENV_KEYS[:5]:io.os.environ[k]='1'
import numpy as np,pandas as pd
from reuse_d import statistics

def main():
    start=io.utc();dest=io.target(io.OUT/'qa'/('final_'+io.stamp()));dest.mkdir(parents=True)
    G=statistics(dest);s=G['s'];tree=ast.parse((io.OUT/'analyze_e_v3.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    funcs={n.name:n for n in main.body if isinstance(n,ast.FunctionDef)}
    components={}
    for setting in ['T0','T3']:
        for model in ['static','adaptive']:
            for w in ['W1','W2']:
                for h in io.HS:
                    mse=np.full((5,391),.04 if setting=='T0' else .03 if model=='static' else .02)
                    mse+=np.arange(5)[:,None]*.001+np.arange(391)[None,:]*.000001
                    components[setting,model,w,h]=dict(mse=mse,rank_ic=.2-mse)
    env=dict(G,io=io,components=components,np=np,pd=pd,s=s,strict_mean=G['strict_mean'],joint_boot=G['joint_boot'],replicates=G['replicates'],binary=G['binary'],families=[])
    exec(compile(ast.Module(body=[funcs['contrast'],funcs['holm']],type_ignores=[]),str(io.OUT/'analyze_e_v3.py'),'exec'),env)
    for h in io.HS:
        r=env['contrast']('E3','T3-minus-T0','static',h,'mse',('T3','static'),('T0','static'),('T0','static'))
        assert np.isclose(r['estimate'],-.01) and np.isclose(r['estimate_pct'],100*r['estimate']/r['denominator_mean_mse'])
        for metric in ['mse','rank_ic']:
            r=env['contrast']('E4','static-minus-adaptive','static-vs-adaptive',h,metric,('T3','static'),('T3','adaptive'),('T3','adaptive'))
            assert np.isclose(r['adaptive_improvement_upper95'],.01)
            assert np.isnan(r['p_nw']) and np.isnan(r['p_nw_holm'])
    f=pd.DataFrame(dict(model=['static']*3,h=io.HS,p_mbb=[.001,np.nan,.2]));env['holm'](f,list(f.index),'synthetic','mbb')
    assert f.loc[0,'p_mbb_holm']==.003 and f.loc[1,'p_mbb_holm']==1
    a=[n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id in ['E3','E4'] and isinstance(t.slice,ast.Constant) and t.slice.value=='label' for t in n.targets)]
    assert len(a)==2
    e3=pd.DataFrame(dict(estimate=[-.1,.1,-.1,.1,0.],p_mbb_holm=[.01,.01,.1,.1,.01]))
    e4=pd.DataFrame(dict(estimate=[-.1,.1,-.1,.1,-.1,.1],p_mbb_holm=[.01,.01,.01,.01,.1,.1],metric=['mse','mse','rank_ic','rank_ic','mse','rank_ic']))
    context=dict(E3=e3,E4=e4);exec(compile(ast.Module(body=a,type_ignores=[]),str(io.OUT/'analyze_e_v3.py'),'exec'),context)
    assert list(e3.label)==['significant improvement','significant deterioration','not significant','not significant','not significant']
    assert list(e4.label)==['favours Static','favours Adaptive','favours Adaptive','favours Static','not significant','not significant']
    # Same immutable profile/protocol gate/source inventory used by final entrypoint.
    import run_e_v5 as run
    assert str(run.q.WORKER).endswith('/worker_e_v2.py')
    assert "q.helper('analyze_e_v3.py')" in (io.OUT/'run_e_v5.py').read_text() and "q.helper('validate_e_v2.py')" in (io.OUT/'run_e_v5.py').read_text()
    for p in io.OUT.glob('*.py'):ast.parse(p.read_text(),str(p))
    assert io.read(io.OUT/'PRERUN_REVIEW.json')['passed'] and io.read(io.OUT/'ARCHIVED_SOURCE_GATE.json')['n_cells']==210
    io.js(io.OUT/'FINAL_CODE_REVIEW.json',dict(started_utc=start,ended_utc=io.utc(),passed=True,optimizer_steps=0,research_test_deserialized=False,
      checks=['final analyzer contrast source on synthetic seed/date matrices','fixed MSE denominators','signed improvement bounds','undefined Holm placeholder','actual E3/E4 label expressions and all sign cases','final scheduler helper/worker bindings','all versions parse','archived210 metadata/hash inventory'],
      final_entrypoint=io.artifact(io.OUT/'run_e_v5.py'),analyzer=io.artifact(io.OUT/'analyze_e_v3.py'),validator=io.artifact(io.OUT/'validate_e_v2.py')))
    print('FINAL CODE REVIEW PASS; no training, no research TEST read.')
if __name__=='__main__':main()
