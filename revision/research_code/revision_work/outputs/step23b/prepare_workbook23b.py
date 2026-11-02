import sys,json,csv
from pathlib import Path
sys.dont_write_bytecode=True
D=Path(__file__).parent
import pandas as pd
p=json.loads((D.parent/'step23/payload23_v2.json').read_text());b=next(b for b in p['blocks'] if b['key']=='t0_not_stored')
f=pd.read_csv(D/'T0_ADAPTIVE_MINUS_LSTM_STORED.csv',float_precision='round_trip');e=pd.read_csv(D/'T0_NEURAL_BENCHMARK_CONTRASTS.csv',float_precision='round_trip')
patches=[]
for n,old in enumerate(b['rows']):
 h,metric=old[3:5]
 if metric=='da':continue
 r=f[(f.h==h)&(f.metric==metric)].iloc[0]
 row=['T0','C','adaptive-minus-lstm',h,metric,float(r.estimate),'See K:L (raw 95% interval)',int(r.seed_wins),float(r.p_holm_3h),str(r.mapping),float(r.ci95_lower),float(r.ci95_upper),r.source_file,int(r.source_csv_line)]
 patches.append({'row':b['first_data_row']+n,'values':row})
payload=dict(base=str(D.parent/'step23/Supplementary_Tables_S1-S18.xlsx'),patches=patches,append_row=584,columns=e.columns.tolist(),rows=e.astype(object).where(e.notna(),None).values.tolist(),raw_source='revision_work/outputs/analysis16/E_paired_tests.csv')
with (D/'WORKBOOK_PATCH.json').open('x') as h:json.dump(payload,h,allow_nan=False,indent=2)
