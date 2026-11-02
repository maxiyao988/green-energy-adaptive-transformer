from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
B2=ROOT/'revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041'

def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def rel(p):return str(p.relative_to(ROOT))
def write(name,data):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
 reg=[x for x in rows(ROOT/'revision_work/outputs/analysis16/registry.csv') if x['model']=='ridge'];inventory=[]
 old={ (x['frequency_setting'],int(x['horizon'].rstrip('d'))):x for x in rows(ROOT/'dataset/audit/main_v2/ridge/panel_ridge_summary_metrics.csv') }
 for x in reg:
  src=ROOT/x['source'];w=x['window'];inp=x['input_set'];h=int(x['h']);cfg=x['config'];parent=src.parent
  if w=='W3' and inp=='O':
   s=old[cfg,h];a=float(s['best_alpha']);alpha_source='dataset/audit/main_v2/ridge/panel_ridge_summary_metrics.csv';train=val=fit='';hist='';vm=float(s['val_mse_at_best_alpha'])
  else:
   summary=rows(parent/'trainer_summary.csv')[0];a=float(summary['best_alpha']);vm=float(summary['val_mse_at_best_alpha']);alpha_source=rel(parent/'trainer_summary.csv');train=rel(parent/'train.parquet');val=rel(parent/'val.parquet');fit=rel(parent/'best_model.pkl');hist=rel(parent/'history.csv')
  inventory.append(dict(cell_id=x['id'],window=w,input_set=inp,config=cfg,h=h,target='raw',old_alpha=a,old_boundary='UPPER' if a==100 else 'LOWER' if a==.1 else 'INTERIOR',alpha_source=alpha_source,old_val_mse=vm,train=train,val=val,fitted_object=fit,history=hist,test_source=x['source'],canonical=x['canonical'],tensor_folder=rel(B2/'tensors'/w),consumers='Step5/Table4/S3;Step16 MSE decomposition;Step21/S13;Step17/S12;Step14cB2/Table10/S8/S9;Step16 pooledIC/S15;Step8 portfolio/S10/S11/480 alpha count;Steps18/19/Figure3' if w=='W3' else 'Step14cB2/Table10/S8/S9;Step16 pooledIC/S15;Step17/S12'))
 for h in [5,10,20]:
  dirs=list((ROOT/'revision_work/outputs/step22g/ridge'/f'h{h}').glob('attempt_*'));assert len(dirs)==1;p=dirs[0];trials=rows(p/'validation_trials.csv');a=float(next(x for x in trials if x['selected']=='True')['alpha']);vm=float(next(x for x in trials if x['selected']=='True')['val_mse'])
  inventory.append(dict(cell_id=f'ridge_CS_daily_weekly_{h}_W3',window='W3',input_set='C',config='daily_weekly',h=h,target='cross_sectional',old_alpha=a,old_boundary='UPPER' if a==100 else 'LOWER' if a==.1 else 'INTERIOR',alpha_source=rel(p/'validation_trials.csv'),old_val_mse=vm,train=rel(p/'train.csv'),val=rel(p/'val.csv'),fitted_object=rel(p/'best_model.pkl'),history=rel(p/'validation_trials.csv'),test_source=rel(p/'predictions_losses_weights.csv'),canonical='',tensor_folder='revision_work/outputs/causal_features/tensors',consumers='Step22G/Supplementary Table S18;CS descriptive portfolio'))
 order={'W3':0,'W1':1,'W2':2};cfgorder={'daily_only':0,'weekly_only':1,'daily_weekly':2}
 inventory.sort(key=lambda x:(order[x['window']],x['target']!='raw',x['input_set'],cfgorder[x['config']],x['h']))
 # Required protocol order O then C.
 inventory.sort(key=lambda x:(order[x['window']],x['target']!='raw',{'O':0,'C':1}[x['input_set']],cfgorder[x['config']],x['h']))
 write('INVENTORY_STEP26.csv',inventory)
 consumers=[('forecast_skill_joint_Holm','revision_work/scripts/skill/a2_skill.py','Table 4;Table S3;joint Holm can change adjusted p of non-Ridge only'),('MSE_decomposition','revision_work/scripts/analysis16/statistics16_v3.py','decomposition;Figure3 source mapping to verify'),('directional_accuracy','revision_work/outputs/step21','Table S13'),('shrinkage','revision_work/scripts/analysis17/compute17_v2.py','Table S12;include formerly unavailable W3 O'),('rolling_stock_regime_subperiod','revision_work/scripts/analysis16/statistics16_v3.py','Table10 note;Tables S8,S9;stock/volatility/subperiod'),('pooled_three_window_IC','revision_work/scripts/analysis16/statistics16_v3.py','Table S15;Holm family'),('portfolio','revision_work/scripts/portfolio/engine.py','Tables S10,S11;480 price-level alpha count;Sharpe/turnover/MDD/alpha/CE'),('manuscript','revision_work/outputs/manuscript18/build_step18_v4.py','Table4;Figure3;affected tables'),('manuscript19','revision_work/outputs/manuscript19/prepare_step19_v2.py','affected tables/figures'),('CS_Ridge','revision_work/outputs/step22g/analyze_g_v3.py','Table S18;descriptive portfolios')]
 write('CONSUMERS_STEP26.csv',[dict(analysis=a,source=s,reported_locations=l,status='pending source-level propagation audit') for a,s,l in consumers])
 with (OUT/'INPUT_PATH_CORRECTION.json').open('x') as f:json.dump(dict(requested='revision_work/inputs/step26',actual='revision_work/step26',reason='User asked to recheck; both uniquely matching documents found in revision_work/step26; use read-only',files=[dict(path=rel(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (ROOT/'revision_work/step26').glob('*.docx')]),f,indent=2)
 print('inventory',len(inventory),'W3 raw upper',sum(x['window']=='W3' and x['target']=='raw' and x['old_alpha']==100 for x in inventory))
if __name__=='__main__':main()
