"""Add stored T0 paired contrasts and explicit missing-coverage rows; no tests."""
import json,hashlib,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.append(str(Path.cwd()/'revision_work/outputs/_deps_pyarrow15'))
import pandas as pd
D=Path('revision_work/outputs/step23');P=json.loads((D/'payload23.json').read_text());L=json.loads((D/'locations23.json').read_text());extra=[]
def col(c):
 s=''
 while c:c,k=divmod(c-1,26);s=chr(65+k)+s
 return s
def clean(v):return None if pd.isna(v) else v.item() if hasattr(v,'item') else v
def block(key,title,f,source):
 n=P['row_counts']['14'];b=dict(sheet='Table S14',sheet_number=14,key=key,title=title,source=source,columns=list(f),rows=[[clean(v) for v in r] for r in f.itertuples(index=False,name=None)],title_row=n,source_row=n+1,header_row=n+2,first_data_row=n+3)
 for i in range(len(f)):
  for j,c in enumerate(f):L[f'{key}|{i}|{c}']=f"'Table S14'!{col(j+1)}{n+3+i}"
 P['blocks'].append(b);P['row_counts']['14']+=len(f)+5
p=Path('revision_work/outputs/analysis16/A_paired_tests.csv');f=pd.read_csv(p,float_precision='round_trip');f=f[(f.input_set=='C')&(f.window=='W3')&(f.contrast=='static−adaptive')&(f.method=='joint_seed_date')].copy();assert len(f)==6
f['setting']='T0';f['evidence']='Original Step16 paired inference;unchanged correction';f['source_file']=str(p);f['source_csv_line']=f.index+2;block('t0_sa_stored','C. Stored T0 Static-minus-Adaptive MSE/Rank IC (Step16)',f,str(p));extra.append(p)
p=Path('revision_work/outputs/step21/paired_da_tests.csv');f=pd.read_csv(p,float_precision='round_trip');f=f[(f.input_set=='C')&(f.scope=='table4')&(f.contrast=='static-minus-adaptive')].copy();assert len(f)==3
f['training_setting']='T0';f['evidence']='Step21 paired DA inference;unchanged correction';f['source_file']=str(p);f['source_csv_line']=f.index+2;block('t0_da_stored','C. Stored T0 Static-minus-Adaptive DA (Step21)',f,str(p));extra.append(p)
f=pd.DataFrame([dict(setting='T0',input_set='C',contrast='adaptive-minus-lstm',h=h,metric=metric,estimate='NOT STORED',interval='NOT STORED',seed_wins='NOT STORED',p_holm='NOT STORED',evidence='NOT STORED;no new tests computed') for h in [5,10,20] for metric in ['mse','rank_ic','da']]);block('t0_not_stored','C. T0 Adaptive-minus-LSTM paired-test coverage: NOT STORED',f,'Step22B/B2 and original Step16 stored paired-test inventories')
P['notes']['14']=P['notes']['14'].replace('T0 paired tests not stored in Step22B; see the explicit coverage note.','T0 Static-minus-Adaptive MSE/Rank IC copied from Step16 and DA from Step21;T0 Adaptive-minus-LSTM paired tests NOT STORED,no new tests computed.')
for name,obj in [('payload23_v2.json',P),('locations23_v2.json',L)]:
 with (D/name).open('x') as h:json.dump(obj,h,ensure_ascii=False,allow_nan=False)
rows=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in extra]
with (D/'SOURCES_APPEND.csv').open('x',newline='') as h:pd.DataFrame(rows).to_csv(h,index=False)
print('Copied 9 stored T0 paired rows;explicitly retained9 NOT STORED T0 cells.')
