"""Exact Table7/Step16 source extraction; signs/units only, no new tests."""
from sources23b import *
tables=js(OUT/'MANUSCRIPT_TABLES.json');rows=next(t['rows'] for t in tables if t['table']==7)[2:]
rates=table('a_rates');assert rates[rates.variant=='static'].lr.eq(1e-4).all()
sel=table('a_summary');base=table('a_base')
for r in sel[sel.variant=='static'].itertuples():
 q=base[(base.input_set==r.input_set)&(base.backbone==r.backbone)&(base.h==r.h)&(base.variant=='static')];assert len(q)==1 and abs(q.iloc[0].mse_mean-r.mse_mean)<1e-15
checks=[];wide=[];label=''
for ri,row in enumerate(rows,3):
 if row[0]:label=row[0]
 bb=label[:2];contrast=label.split(': ')[1];h=int(row[1])
 for input_set,offset in [('O',2),('C',5)]:
  if contrast=='Static − Fixed':
   q=select('base_tests',input_set=input_set,backbone=bb,setting='head_frozen',horizon=h,metric='mse',contrast='static_minus_fixed');assert len(q)==1;r=q.iloc[0];delta=r.theta;wins=select('base_wins',input_set=input_set,backbone=bb,setting='head_frozen',horizon=h,metric='mse',contrast='static_minus_fixed').left_wins.sum();p=r.p_holm_over_h;source=ALIASES['base_tests'];index=q.index[0];cols=['theta','seed_wins.csv:left_wins sum','p_holm_over_h'];mapping='Static selects1e-4 in every cell;selected/base MSE identity verified;reuse base-rate tests'
  else:
   reverse=contrast=='Adaptive − Static';co='static-minus-adaptive' if reverse else 'adaptive-minus-constant_gate';q=select('a_tests',input_set=input_set,backbone=bb,h=h,metric='mse',contrast=co);assert len(q)==1;r=q.iloc[0];delta=(-1 if reverse else 1)*r.theta_hat;wins=r.seed_losses if reverse else r.seed_wins;p=r.p_holm_3h;source=ALIASES['a_tests'];index=q.index[0];cols=['-theta_hat' if reverse else 'theta_hat','seed_losses' if reverse else 'seed_wins','p_holm_3h'];mapping='Reverse sign and swap stored wins/losses only;two-sided p unchanged' if reverse else 'Stored first-minus-second contrast'
  source=str(path(source).relative_to(ROOT));values=[delta*1e4,int(wins),p];expected=[float(row[offset].replace('−','-')),int(row[offset+1].split('/')[0]),float(row[offset+2])]
  for j,(actual,ex) in enumerate(zip(values,expected)):
   actual_round=round(actual,2) if j!=1 else int(actual)
   checks.append(dict(table='Manuscript Table7',manuscript_row=ri,manuscript_column=offset+j+1,backbone=bb,contrast=contrast,input_set=input_set,h=h,field=['delta_MSE_x1e4','seed_wins_first_lower_MSE','Holm_p_3h'][j],manuscript_value=row[offset+j],source_unrounded=actual,source_display=actual_round,source_file=source,source_csv_line=int(index)+2,source_column=cols[j],mapping=mapping,status='MATCH' if actual_round==ex else 'MISMATCH'))
  wide.append(dict(backbone=bb,contrast=contrast,input_set=input_set,h=h,delta_mse=delta,delta_mse_x1e4=delta*1e4,seed_wins=int(wins),p_holm_3h=p,source_file=source,source_csv_line=int(index)+2,mapping=mapping))
csvwrite('TABLE7_CROSSCHECK.csv',pd.DataFrame(checks));csvwrite('TABLE7_VERIFIED_VALUES.csv',pd.DataFrame(wide))
# Every stored neural contrast,including the conditional date-only references.
f=table('E').copy();f['source_file']='revision_work/outputs/analysis16/E_paired_tests.csv';f['source_csv_line']=f.index+2;f['setting']='T0';f['numerical_provenance']='Step16 registry;O benchmarks uniform12/12 B5FIX/B5';csvwrite('T0_NEURAL_BENCHMARK_CONTRASTS.csv',f)
# Reverse the stored LSTM-minus-Adaptive joint contrasts. Check seed ties using stored metrics.
B=table('B');out=[]
for h in [5,10,20]:
 for metric,bc in [('squared_loss','mse'),('rank_ic','mean_rank_ic')]:
  q=select('E',input_set='C',window='W3',contrast='lstm−adaptive',h=h,metric=metric,method='joint_seed_date');assert len(q)==1;r=q.iloc[0]
  seed_metrics=[]
  for model,group in [('lstm','neural'),('adaptive','core')]:
   v=select('B',input_set='C',window='W3',model=model,group=group,config='daily_weekly',h=h);v=v[v.seed.isin([0,1,21,42,3407])].sort_values('seed');assert len(v)==5;seed_metrics.append(v[bc].to_numpy(float))
  diff=seed_metrics[1]-seed_metrics[0];ties=int((diff==0).sum());assert ties==0
  wins=int((diff<0 if metric=='squared_loss' else diff>0).sum());assert wins==5-r.seed_wins
  out.append(dict(setting='T0',input_set='C',contrast='adaptive-minus-lstm',h=h,metric='mse' if metric=='squared_loss' else metric,estimate=-r.theta_hat,ci95_lower=-r.ci_upper_95,ci95_upper=-r.ci_lower_95,seed_wins=wins,seed_losses=int(r.seed_wins),seed_ties=ties,raw_p=r.centered_p_two_sided,p_holm_3h=r.p_holm_3h,source_file='revision_work/outputs/analysis16/E_paired_tests.csv',source_csv_line=int(q.index[0])+2,source_contrast=r.contrast,mapping='Sign/CI reversal;stored two-sided raw/Holm p copied unchanged;zero ties checked from stored per-seed metrics'))
csvwrite('T0_ADAPTIVE_MINUS_LSTM_STORED.csv',pd.DataFrame(out))
checkpoint_sources('SOURCES_PARTS13.csv')
print('Table7:',pd.DataFrame(checks).status.value_counts().to_dict(),';neural contrast rows:',len(f),';reversed Adaptive-LSTM cells:',len(out),';DA3 cells remain NOT STORED.')
