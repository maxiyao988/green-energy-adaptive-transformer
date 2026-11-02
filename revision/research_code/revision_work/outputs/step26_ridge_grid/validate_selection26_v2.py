"""Independent VAL calculation: direct Ridge loop, no original selection wrapper."""
import phase_b26 as b
from sklearn.linear_model import Ridge
import json,csv,time,traceback
np,pd=b.np,b.pd
OUT,ROOT=b.OUT,b.ROOT

def main():
 lock=json.loads((OUT/'SELECTION_LOCK_SHA256.json').read_text());assert b.sha(OUT/'SELECTION_LOCK.csv')==lock['sha256'];assert lock['n_cells']==33
 # Same TEST protection and TRAIN/VAL-only data selector as fitting; independent estimator/selection arithmetic.
 src=b.ast.parse(b.Path(b.__file__).read_text());node=next(n for n in src.body if isinstance(n,b.ast.FunctionDef) and n.name=='install_guard');node=b.copy.deepcopy(node)
 for n in b.ast.walk(node):
  if isinstance(n,b.ast.Constant) and n.value=='tmp_phase_b':n.value='tmp_independent'
 exec(compile(b.ast.fix_missing_locations(b.ast.Module(body=[node],type_ignores=[])),b.__file__,'exec'),b.__dict__);inv=b.install_guard();S={r['cell_id']:r for r in csv.DictReader((OUT/'SELECTION_LOCK.csv').open())};result=[];current=None
 for r in inv:
  key=r['window'],r['input_set'],r['target']
  if key!=current:ds=b.dataset(*key);stats=b.common.fit_train_only_normalizer(ds);current=key
  X=b.common.flatten_features(r['config'],b.common.apply_normalizer(ds,stats,r['config']));tr=ds.split=='train';va=ds.split=='val';y=ds.y[int(r['h'])];s=S[r['cell_id']];expected=json.loads(s['val_mse_by_alpha']);best=None;bestloss=float('inf');maxerr=0.;values=[]
  for q in expected:
   model=Ridge(alpha=q['alpha']);model.fit(X[tr],y[tr]);p=model.predict(X[va]);loss=float(np.average(np.square(p-y[va])));values.append(dict(alpha=q['alpha'],val_mse=loss));maxerr=max(maxerr,abs(loss-q['val_mse']))
   if loss<bestloss:best,bestloss=q['alpha'],loss
  z=dict(cell_id=r['cell_id'],recorded_alpha=float(s['selected_alpha']),independent_alpha=best,alpha_equal=best==float(s['selected_alpha']),max_val_mse_discrepancy=maxerr,independent_val_mse=bestloss,all_trials=values)
  result.append(z);b.js('independent/'+r['cell_id']+'.json',z) if False else None
  print('INDEPENDENT_VAL',r['cell_id'],z['alpha_equal'],maxerr,flush=True)
 b.js('INDEPENDENT_VAL_SELECTION.json',dict(cells=result,all_pass=all(z['alpha_equal'] and z['max_val_mse_discrepancy']<1e-12 for z in result),TEST_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:b.js('INDEPENDENT_VAL_BLOCKING.json',dict(error=traceback.format_exc()));raise
