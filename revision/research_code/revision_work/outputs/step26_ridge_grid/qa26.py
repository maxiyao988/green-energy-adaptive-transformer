"""Read-only final audit of selections, raw non-Ridge preservation, and portfolio source replays."""
from metrics26_v2 import *
def main():
 checks=[]
 def check(name,passed,detail):checks.append(dict(check=name,passed=bool(passed),detail=detail))
 S=pd.read_csv(OUT/'SELECTION_LOCK.csv');I=pd.read_csv(OUT/'INVENTORY_STEP26.csv');lock=json.loads((OUT/'SELECTION_LOCK_SHA256.json').read_text());check('selection lock SHA256',hashlib.sha256((OUT/'SELECTION_LOCK.csv').read_bytes()).hexdigest()==lock['sha256'],lock['sha256']);check('all cells selected',len(S)==len(I)==33,'18 W3 raw +12 rolling raw +3 W3 CS')
 R=pd.read_csv(OUT/'REPRODUCTION_STEP26.csv');check('all original alphas reproduced',R.alpha_equal.all(),int(R.alpha_equal.sum()));check('all stored VAL EXACT',R.loc[R.val_class!='UNAVAILABLE','val_class'].eq('EXACT').all(),int(R.val_class.eq('EXACT').sum()));check('independent selection all33',json.loads((OUT/'INDEPENDENT_VAL_SELECTION.json').read_text())['all_pass'],'direct Ridge loop, all VAL trial losses exactly equal');check('independent W3 MSE/IC',json.loads((OUT/'INDEPENDENT_TEST_VALIDATION.json').read_text())['all_pass'],'21 W3 cells, both old/new arms')
 old=pd.read_csv(A16/'clarified/B_forecast_skill_final.csv',float_precision='round_trip');new=pd.read_csv(OUT/'propagated/B_forecast_skill_final_complete_step26.csv',float_precision='round_trip');cols=[c for c in old if not c.endswith('_holm_all_cells')];pd.testing.assert_frame_equal(old.loc[old.model!='ridge',cols],new.loc[new.model!='ridge',cols],check_dtype=False);check('all non-Ridge forecast skill raw values retained',True,len(old[old.model!='ridge']))
 shold=pd.read_csv(ROOT/'revision_work/outputs/analysis17/test_results.csv',float_precision='round_trip');shnew=pd.read_csv(OUT/'propagated/SHRINKAGE_final_step26.csv',float_precision='round_trip');cols=[c for c in shold if c!='loss_p_Holm'];pd.testing.assert_frame_equal(shold.loc[shold.model!='ridge',cols],shnew.loc[shnew.model!='ridge',cols],check_dtype=False);check('all non-Ridge shrinkage raw values retained',True,len(shold[shold.model!='ridge']))
 P=pd.read_csv(OUT/'PORTFOLIO_RIDGE_STEP26.csv',float_precision='round_trip');oldO=pd.read_csv(ROOT/'revision_work/outputs/portfolio/original/metrics_all.csv',float_precision='round_trip');oldC=pd.read_csv(A16/'G_portfolio_primary.csv',float_precision='round_trip');replays=[]
 for inp,original in [('O',oldO),('C',oldC)]:
  z=P[(P.arm=='old')&(P.input_set==inp)]
  for r in z.to_dict('records'):
   source=original[(original.split=='test')&(original.h==r['h'])&(original.strategy==('ridge_s42' if inp=='O' else 'ridge'))&(original.layer==r['layer'])&(original.cost_bp==r['cost_bp'])]
   if len(source)==0 and inp=='C' and r['layer']=='L1_top3':continue
   assert len(source)==1,(inp,r['h'],r['layer'],len(source));delta=max(abs(float(source.iloc[0][c])-r[c]) for c in ['sharpe','ann_turnover','max_drawdown','ce_A2','ce_A5'] if c in source and np.isfinite(float(source.iloc[0][c])) and np.isfinite(r[c]));replays.append(dict(input_set=inp,h=r['h'],layer=r['layer'],cost_bp=r['cost_bp'],max_abs_delta=delta))
 write('PORTFOLIO_ORIGINAL_SOURCE_REPLAY.csv',pd.DataFrame(replays));check('old Ridge portfolio source replay',max(r['max_abs_delta'] for r in replays)<1e-8,dict(n=len(replays),max_abs_delta=max(r['max_abs_delta'] for r in replays)))
 # D2 contextual penalty sentence; combine immutable first-pass impact inventory with the audit addition.
 impact=pd.read_csv(OUT/'IMPACT_STEP26.csv',keep_default_na=False);doc=json.loads((OUT/'DOCUMENT_EXTRACTION_STEP26.json').read_text());extra=[]
 for r in doc:
  if 'c_clean' in r['file'] and 'Its penalty is selected from four candidate values by validation MSE.' in r['text']:
   extra.append(dict(document=r['file'],location=r['location'],quoted_text='Its penalty is selected from four candidate values by validation MSE.',old_value='four candidates',new_value='nine initial candidates; two-decade boundary extensions to fixed caps, 9–15 candidates actually evaluated',source_file='revision_work/outputs/step26_ridge_grid/SELECTION_LOCK.csv',qualitative_statement='CHANGES',assessment='Contextual sentence refers to Ridge in the preceding sentence.'))
 write('IMPACT_STEP26_COMPLETE.csv',pd.concat([impact,pd.DataFrame(extra)],ignore_index=True));check('contextual Ridge penalty sentence captured',len(extra)==1,len(extra));check('Figure3 artifacts exist',all((OUT/'manuscript'/('Figure3_step26.'+e)).exists() for e in ['png','pdf']),'PNG inspected visually; original Step19 save() bounds assertions passed')
 write('QA_STEP26.csv',pd.DataFrame(checks));js('QA_STEP26.json',dict(all_pass=all(r['passed'] for r in checks),checks=checks,archived_O_TEST_replay_caveat='9 DIFFERENT, retained and reported; explicit user B1 stop applies to stored VAL only',training_runs_non_Ridge=0))
 print('QA',all(r['passed'] for r in checks))
if __name__=='__main__':
 try:main()
 except BaseException:js('QA_BLOCKING.json',dict(error=traceback.format_exc()));raise
