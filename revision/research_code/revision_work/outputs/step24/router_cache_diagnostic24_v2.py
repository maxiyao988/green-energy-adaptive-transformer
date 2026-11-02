"""Reproduce H diagnostics from the actual cached router forward, not the full-model reference weights."""
from review24_common import *
import ast
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import torch
from torch import nn
torch.set_num_threads(12);torch.set_num_interop_threads(12)
p='scripts/python/panel_pathformer_main_v2.py';tree=ast.parse(read(p));cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScaleAggregator');ns=dict(torch=torch,nn=nn,D_MODEL=32,ROUTER_HIDDEN=32,SCALE_MODES=['single','fixed','static','adaptive']);exec(compile(ast.Module(body=[cl],type_ignores=[]),p,'exec'),ns)
bindings=js('revision_work/outputs/step22a/BINDINGS.json');diag=df('revision_work/outputs/analysis16/H_router_diagnostics.csv');checks=[]
for b in [b for b in bindings if b['backbone']=='S2']:
 ident=b['id'];read(b['checkpoint']);state=torch.load(ROOT/b['checkpoint'],weights_only=True,map_location='cpu');p=ROOT/b['cache']['test']['experts'];read(p);a=np.load(p);keys=pq(ROOT/b['cache']['test']['keys'])
 for branch,part,patch in [('daily',a[:,:4],[5,10,20,30]),('weekly',a[:,4:],[2,4,8,13])]:
  agg=ns['ScaleAggregator']('adaptive',patch);prefix=branch+'_branch.aggregator.';agg.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);agg.eval();weights=[]
  with torch.no_grad():
   for start in range(0,len(a),32):
    x=torch.from_numpy(part[start:start+32].copy());_,w=agg(list(x.unbind(1)));weights.append(w.numpy())
  w=np.concatenate(weights).astype(np.float64);sd=float(w.std(0).mean());between=float(pd.DataFrame(w).groupby(keys.anchor_date.to_numpy()).transform('mean').var().sum());total=float(pd.DataFrame(w).var().sum());ratio=between/total;z=diag[diag.id.eq(ident)&diag.branch.eq(branch)].iloc[0]
  for name,actual,expected in [('test_mean_sd_across_samples',sd,z.test_mean_sd_across_samples),('share_variance_between_dates',ratio,z.share_variance_between_dates)]:checks.append(dict(check='H actual cache router '+name,id=ident,branch=branch,independent_value=actual,stored_value=expected,max_abs_difference=abs(actual-expected),passed=abs(actual-expected)<=1e-10))
write('H_CACHE_ROUTER_DIAGNOSTIC_CHECKS_V2.csv',pd.DataFrame(checks));source_manifest('SOURCES_H_CACHE_DIAGNOSTICS_V2.csv');print('H CACHE',len(checks),'failed',sum(not c['passed'] for c in checks));print([c for c in checks if not c['passed']][:8])
