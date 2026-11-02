"""Independent formulas versus isolated production functions, SciPy/statsmodels and stored inputs."""
from review24_common import *
import independent24 as ind
import ast,math,warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
checks=[];differences=[]
def check(name,actual,expected,tol=1e-10,scope='synthetic'):
 a=np.asarray(actual,float);b=np.asarray(expected,float);equal_nan=np.array_equal(np.isnan(a),np.isnan(b));delta=float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else 0.;passed=bool(a.shape==b.shape and equal_nan and delta<=tol)
 checks.append(dict(check=name,scope=scope,passed=passed,max_abs_difference=delta,tolerance=tol,numeric_nan_support_equal=equal_nan))
 if not passed:differences.append(checks[-1])
# The reference module imports no project code. This isolated namespace contains only the production functions under test.
p='revision_work/scripts/skill/skill_common.py';tree=ast.parse(read(p));names=['nw_lrv','nw_t','rank_rows','rank_ic_rows','mbb_index','holm'];ns=dict(np=np,pd=pd,math=math,stats=stats)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),p,'exec'),ns)
rng=np.random.default_rng(20261022)
y=rng.normal(size=(391,17));pred=rng.normal(size=(391,17));pred[:3]=np.array([np.ones(17),np.arange(17)//4,np.arange(17)])[None,:,:] if False else pred[:3]
pred[0]=1;pred[1]=np.arange(17)//4;pred[2]=np.arange(17)
with warnings.catch_warnings():
 warnings.simplefilter('ignore');ref=np.array([stats.spearmanr(a,b).statistic for a,b in zip(pred,y)])
check('Rank IC ties and constant predictions: independent vs scipy',ind.rank_ic(pred,y),ref)
check('Rank IC production vs independent, ordinary scales',ns['rank_ic_rows'](pred,y),ind.rank_ic(pred,y))
# Record the absolute-std guard as a domain restriction, not a hidden perfect-correlation pass.
tiny=np.arange(17,dtype=float)*1e-14
check('Rank IC positive rescaling invariance at tiny scale',ns['rank_ic_rows'](tiny[None,:],np.arange(17)[None,:]),ind.rank_ic(tiny,np.arange(17)))
x=rng.normal(size=391);x[1:]+=.5*x[:-1];x+=.04
for lag in [0,4,5,9,10,19,20]:
 r=ind.nw(x,lag);q=ns['nw_t'](x,lag);fit=sm.OLS(x,np.ones((len(x),1))).fit();se=math.sqrt(cov_hac(fit,nlags=lag,use_correction=False)[0,0]);check('NW production vs independent lag'+str(lag),q[:3],[r['mean'],r['se'],r['t']]);check('NW independent SE vs statsmodels lag'+str(lag),r['se'],se)
for pv in [[.04,.01,.03,.2],[1,0,.5,.5],[.02,.02,.02]]:
 z=ind.holm(pv);from statsmodels.stats.multitest import multipletests
 check('Holm versus statsmodels '+str(pv),z,multipletests(pv,method='holm')[1]);check('Holm production '+str(pv),list(ns['holm(dict(enumerate(pv)))'].values()) if False else np.array([ns['holm'](dict(enumerate(pv)))[i] for i in range(len(pv))]),z)
# Independent block fill has the same prescribed overlap/start support/truncation as formal MBB.
for T,L,B in [(17,5,100),(391,5,500),(391,10,500),(391,20,500)]:
 a=ind.mbb(T,L,B,np.random.default_rng(721));b=ns['mbb_index'](T,L,B,np.random.default_rng(721));check(f'MBB index construction T{T} L{L}',a,b,0)
 check(f'MBB no out-of-range T{T} L{L}',np.array([a.min()>=0,a.max()<T]).astype(int),[1,1],0)
# Synthetic common-date/paired-seed contrast: a constant shift stays constant in every replicate.
X=rng.normal(size=(5,391));si=rng.integers(0,5,size=(1000,5));di=ind.mbb(391,10,1000,rng);shift=ind.joint_from_draws((X+3)-X,si,di);check('Joint common blocks + paired seeds constant-shift invariant',shift,np.full(1000,3.))
# Pooled NW uses squared n_w/n_total weights, not concatenation HAC across artificial boundaries.
series=[rng.normal(size=391)+.01,rng.normal(size=391)+.04,rng.normal(size=391)-.02];r=ind.pooled_nw(series,10);covs=[cov_hac(sm.OLS(v,np.ones((391,1))).fit(),nlags=10,use_correction=False)[0,0] for v in series];check('Pooled NW independent-window variance combination',r['se'],np.sqrt(sum(covs)/9))
# Production CW source assignment vs independent algebra.
tr=ast.parse(read('revision_work/scripts/skill/a2_skill.py'));fn=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='cell_metrics');node=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='f' for t in n.targets));env=dict(np=np,y=y.ravel(),p=pred.ravel(),b=np.full(y.size,.02));exec(compile(ast.Module(body=[node],type_ignores=[]),'CW_assignment','exec'),env)
f,dates,rr=ind.cw(y.ravel(),pred.ravel(),.02,y.shape,5);check('CW algebra production assignment vs independent 2e*d',env['f'],f)
# Reproduce selected stored bootstrap replicates from their saved inputs and draws (no new random test).
BROOT=ROOT/'revision_work/outputs/step22b2';j=js(BROOT/'ANALYSIS_COMPLETE.json');dest=ROOT/j['analysis_directory'] if 'analysis_directory'in j else next((ROOT/a['path']).parent for a in j['artifacts'] if a['path'].endswith('/joint_bootstrap_tests.csv'))
for h in [5,10,20]:
 draw=dest/f'draws/h{h}.npz'
 if not draw.exists():continue
 z=np.load(draw,allow_pickle=False);read(draw);print('draw fields',h,z.files,flush=True)
# Frozen22A bootstrap inputs are saved per seed/date, with draw files and replicate arrays.
FRO=ROOT/'revision_work/outputs/frozen';dt=np.load(FRO/'bootstrap/paired_distributions.npz',allow_pickle=False);read(FRO/'bootstrap/paired_distributions.npz')
from pathlib import Path
long=df('revision_work/outputs/frozen/paired_seed_date_differences.csv')
for h in [5,10,20]:
 dr=FRO/f'bootstrap/draws_h{h}.npz';z=np.load(dr);read(dr)
 q=long[(long.input_set=='C')&(long.backbone=='S1')&(long.setting=='head_frozen')&(long.horizon==h)&(long.contrast=='adaptive_minus_static')&(long.metric=='mse')]
 panel=q.pivot(index='seed',columns='anchor_date',values='paired_difference').sort_index().sort_index(axis=1).to_numpy();boot=ind.joint_from_draws(panel,z['seed_indices'],z['date_indices']);name=f'C_S1_head_frozen_h{h}_adaptive_minus_static_mse';check('Stored Step15 joint replicate array '+str(h),boot,dt[name],scope='stored-input replicate replay')
# Stored pooled Rank-IC means/SE/p values independently recomputed from stored window series.
D=ROOT/'revision_work/outputs/step22d/session_v3';j=js(D/'ANALYSIS_COMPLETE.json');dest=ROOT/j['analysis_directory'];q=df(dest/'part1_rank_ic_inference.csv')
for row in q[q.window.eq('W1+W2+W3')].itertuples():
 p=dest/f'inputs/part1_{row.input_set}_{row.model}_h{row.h}_ic.npz';z=np.load(p);read(p);r=ind.pooled_nw([z[w] for w in ['W1','W2','W3']],row.h);check(f'Pooled IC mean/SE {row.input_set} {row.model} h{row.h}',[row.mean,row.nw_se],[r['mean'],r['se']],scope='stored-input formula replay')
 if row.model!='ridge':check(f'Pooled IC NW two-sided p {row.input_set} {row.model} h{row.h}',row.p_nw,r['p_two'],scope='stored-input formula replay')
write('PRIORITY1_STATISTICS_CHECKS.csv',pd.DataFrame(checks));write('PRIORITY1_STATISTICS_DISCREPANCIES.csv',pd.DataFrame(differences));source_manifest('SOURCES_STATISTICS.csv')
jwrite('REFERENCE_ENVIRONMENT.json',dict(rng_seed=20261022,numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,statsmodels=__import__('statsmodels').__version__,reference_imports_project=False,production_function_harness='AST-isolated pure functions only',hypothesis_test_scope='Replay of existing formulas/specifications; no novel result family'))
print('STATISTICS',len(checks),'checks;',len(differences),'discrepancies',differences)
