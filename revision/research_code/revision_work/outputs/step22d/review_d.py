"""Pre-registration QA on synthetic arrays only; no research TEST files opened."""
import sys
sys.dont_write_bytecode=True
from io_d import *
for k in ENV_KEYS[:5]:os.environ[k]='1'
sys.path.insert(0,str(OUT));sys.path.insert(0,str(ROOT/'revision_work/scripts/analysis16'))
import ast,functools
import numpy as np
from statistics16_v3 import s

def main():
    started=utc();source=OUT/'analyze_d_v2.py';tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    names=['draws','date_boot','joint_boot','series_result','per_seed']
    nodes=[n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name in names]
    captured={}
    env=dict(np=np,s=s,B=20000,BASE=20261021,SEEDS=SEEDS,lru_cache=functools.lru_cache,
      binary=lambda name,**a:captured.update({name:a}),replicates=lambda name,v:captured.update({name:v}) or name)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    rng=np.random.default_rng(947)
    for h in HS:
        for w in ['W1','W2','W3']:
            di,wt=env['draws'](w,h)
            assert di.shape==(20000,391) and wt.shape==(20000,5)
            assert di.min()>=0 and di.max()<391 and np.allclose(wt.sum(1),1,rtol=0,atol=1e-15)
            a=np.arange(5)[:,None]+np.arange(391)[None,:]/1000
            actual=env['joint_boot'](a,w,h)
            manual=np.array([(wt[i]@a[:,di[i]]).mean() for i in range(32)])
            assert np.max(np.abs(actual[:32]-manual))<1e-14
    series={w:rng.normal(.02,.1,391) for w in ['W1','W2','W3']}
    r=env['series_result'](series,['W1','W2','W3'],10,10,tag='synthetic_nw')
    nw=[s.nw_t(series[w],10) for w in series]
    expected=np.sqrt(sum(a[1]**2 for a in nw))/3
    assert abs(r['nw_se']-expected)<1e-15 and abs(r['mean']-np.mean([a[0] for a in nw]))<1e-15
    assert abs(r['p_nw']-2*s.stats.norm.sf(abs(r['nw_t'])))<1e-15
    r=env['series_result'](series,['W1','W2'],5,4,one_sided=True,tag='synthetic_cw')
    assert abs(r['p_nw']-s.stats.norm.sf(r['nw_t']))<1e-15
    r=env['series_result'](series,['W1'],5,5,test=False,tag='synthetic_descriptive')
    assert np.isnan(r['p_nw']) and np.isnan(r['p_mbb'])
    bad={w:v.copy() for w,v in series.items()};bad['W1'][0]=np.nan
    r=env['series_result'](bad,['W1','W2'],5,5,tag='synthetic_missing')
    assert not r['full_support'] and np.isnan(r['p_nw']) and np.isnan(r['p_mbb'])
    y=rng.normal(0,.1,6647);y[:10]=0
    env['all_labels']={('W1',5):dict(y=y,m=0,gtm_mse=np.mean(y*y),zero_mse=np.mean(y*y))}
    r,loss,ic=env['per_seed'](y,'W1',5)
    assert r['n_valid_da']==6637 and r['da']==1 and r['mse']==0
    # Family sizes / conservative placeholders and signed bounds, without any TEST data.
    q=s.holm({0:.01,1:1.,2:.1});assert q[0]==.03 and q[1]==1.
    assert np.quantile(np.full(20000,-.02),.95,method='linear')<0
    for path in OUT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    rows=check_old_sources();assert all(r['unchanged'] for r in rows)
    js(OUT/'PRERUN_REVIEW.json',dict(started_utc=started,ended_utc=utc(),passed=True,synthetic_only=True,
       research_test_files_opened=False,bootstrap_settings_tested='B20000; h5/10/20; independently stratified W1/W2/W3',
       synthetic_rng_seed=947,checks=['paired joint application equals direct resampling','stratified NW variance and one-/two-sided p',
       'Ridge descriptive p excluded','undefined series preserved','DA zero exclusion','Holm placeholder','negative improvement bounds retained','all scripts parse','Step22B source hashes unchanged'],
       original_source_checks=len(rows),analysis_source=artifact(source)))
    print('Synthetic statistical QA passed; research TEST files not opened.')
if __name__=='__main__':main()
