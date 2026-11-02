"""Read-only extraction to a typed payload; no training, evaluation or bootstrap."""
import sys,os,json,hashlib,csv,math,zipfile,re,datetime
from pathlib import Path
from xml.etree import ElementTree as ET
sys.dont_write_bytecode=True
sys.path.append(str(Path.cwd()/'revision_work/outputs/_deps_pyarrow15'))
import pandas as pd
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step23';R=ROOT/'revision_work/outputs'
sources={};computed=[];catalog={};blocks=[];locations={};titles={
1:'Corporate-action records, issuer filings and handling rules',2:'Mean realized returns by partition',3:'Financial benchmarks',4:'Single-resolution configurations',5:'Rank IC contrasts between aggregation mechanisms',6:'Frozen-representation results',7:'Scale-removal interventions and Rank IC intervention results',8:'Rolling-window results including Ridge',9:'Results by stock, volatility regime and subperiod',10:'Portfolio metrics',11:'Cost and risk-aversion sensitivity',12:'Shrinkage toward the training mean',13:'Directional accuracy',14:'Alternative training settings',15:'Pooled three-window analysis',16:'Held-out-window evaluation of the alternative training setting',17:'Static and Adaptive with larger aggregation learning rates',18:'Cross-sectional target'}
def readpath(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;b=p.read_bytes();sources[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));return p
def js(p):return json.loads(readpath(p).read_text())
def frame(p):return pd.read_csv(readpath(p),float_precision='round_trip')
def dump(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False)
def csvout(p,df):
 with Path(p).open('x',newline='') as f:df.to_csv(f,index=False)
def safe(v):
 if pd.isna(v):return None
 if hasattr(v,'item'):v=v.item()
 return v if not isinstance(v,float) or math.isfinite(v) else None
def col(c):
 s=''
 while c:c,k=divmod(c-1,26);s=chr(65+k)+s
 return s
def xlsx_cells(p):
 z=zipfile.ZipFile(readpath(p));ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 ss=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
 sheets=ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets',ns);out={}
 for i,s in enumerate(sheets,1):
  cells={};tree=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
  for c in tree.findall('.//m:c',ns):
   v=c.find('m:v',ns)
   if v is None:continue
   text=v.text;t=c.attrib.get('t')
   cells[c.attrib['r']]=ss[int(text)] if t=='s' else text if t in ['str','inlineStr','e'] else bool(int(text)) if t=='b' else float(text) if any(a in text.lower() for a in ['.','e']) else int(text)
  out[s.attrib['name']]=cells
 return out
def locate_steps():
 D={'frozen':R/'frozen','a':R/'step22a','21':R/'step21','17':R/'analysis17'}
 for k,n in [('b','step22b'),('b2','step22b2'),('d','step22d/session_v3'),('e','step22e'),('f','step22f'),('g','step22g')]:D[k]=ROOT/js(R/n/'ANALYSIS_COMPLETE.json')['analysis_directory']
 D['c']=next((R/'step22c/runs').glob('run_*'))
 return D
D=locate_steps();hashes={}
for n in ['step22a','step22b','step22b2','step22d/session_v3','step22e','step22f','step22g']:
 p=R/n/'RNG_MANIFEST.json';m=js(p);hashes[n]=m.get('protocol_sha256','NOT RECORDED')
 for q in (R/n).glob('PROTOCOL*.md'):readpath(q)
for p in [R/'frozen/REPORT_STEP15.md',R/'analysis17/REPORT_STEP17.md',R/'step21/REPORT_STEP21_FINAL.md',R/'step22a/REPORT_STEP22A_FINAL.md',R/'step22b/REPORT_STEP22B.md',R/'step22b2/REPORT_STEP22B2.md',R/'step22c/REPORT_STEP22C.md',R/'step22c/METHODS_STEP22C.md',R/'step22d/session_v3/REPORT_STEP22D.md',R/'step22e/REPORT_STEP22E_FINAL.md',R/'step22f/REPORT_STEP22F_FINAL.md',R/'step22g/REPORT_STEP22G_FINAL.md',R/'step22g/REPORT_STEP22G_VALIDATION.md']:
 if p.exists():readpath(p)
# This input was a local text file.
readpath('<pasted_manuscript_text.txt>')
base=R/'step21/supplementary_s13/Supplementary_Tables_with_S13.xlsx';old=xlsx_cells(base)
assert set(old)=={'README'}|{f'S{i}' for i in range(1,14)}
inventory=[]
for p in [base,R/'manuscript19/Supplementary_Tables.xlsx']:
 readpath(p);st=p.stat();inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=st.st_size,mtime_utc=datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),qualifies_S1_S13=p==base))
csvout(OUT/'WORKBOOK_INVENTORY.csv',pd.DataFrame(inventory))
original_inventory=[]
for i in range(1,14):
 cells=old[f'S{i}'];src=[v for a,v in cells.items() if re.fullmatch('B[0-9]+',a) and isinstance(v,str) and v.startswith('revision_work/')]
 for v in src:
  if (ROOT/v).exists():readpath(v)
 original_inventory.append(dict(sheet=f'S{i}',old_title=cells.get('A2'),new_title=f'Table S{i}. {titles[i]}',source_files='; '.join(src)))
csvout(OUT/'ORIGINAL_SHEETS.csv',pd.DataFrame(original_inventory))
notes={6:'Step22A FINAL selected learning rates, original Step15/16 base rate 1e-4, Step21 DA (zero realized returns excluded), head-retrained sensitivity, Step22C bounds. Fixed is untrained. Every trial/cell retained. Five-seed SD ddof=1.',14:'C inputs,W3; T0 lr1e-4/Huber delta1; T1 lr3e-5/delta1; T2 lr1e-4/delta sigma_h; T3 lr3e-5/delta sigma_h. Step22B/B2/C. No setting selection. T0 paired tests not stored in Step22B; see the explicit coverage note.',15:'Step22D Part1; W3 remains primary. Pooled three-window analysis supplementary. NW primary; stratified MBB robustness; Holm families copied. Per-window/pooled references remain window-specific.',16:'Step22D Part2 and Step22E. C inputs. W1/W2 confirmatory; W3 primary manuscript evaluation. T3 lr3e-5/delta window TRAIN SD. E1 SECONDARY,T0 descriptive; E2/E3/E4 CONFIRMATORY. E3/E4 NW N/A; use joint MBB.',17:'EXPLORATORY. Step22F C/W3,larger scale-logit/router learning rates. Failed manipulation affects interpretation only,never p-values/families. All48 attempts,including3 interrupted attempts,retained.',18:'EXPLORATORY. y_cs(i,t)=y(i,t)-mean_j y(j,t),same retained17 stocks/date. TRAIN sigma uses float64,ddof1;model targets float32 unscaled;TEST metrics use float64 y_cs. T3-CS lr3e-5/Huber delta sigma_cs. Rank IC against realized returns;MSE/R2 against zero on y_cs;DA excludes exact-zero y_cs. Ridge deterministic,n=1,SD N/A. Long-short gross200%,net0,10bp per full-book L1 turnover;excludes borrowing costs and cash interest;descriptive only,no portfolio tests.'}
rownext={i:5 for i in [6,14,15,16,17,18]}
def add(num,key,title,path,df=None,evidence=None,**metadata):
 original=frame(path);catalog[key]=original.copy();df=original.copy() if df is None else df.copy()
 assert len(df)==len(original),'Do not drop source rows '+key
 for k,v in metadata.items():df[k]=v
 if evidence is not None and 'evidence' not in df:df['evidence']=evidence
 df['source_file']=str(Path(path).relative_to(ROOT));df['source_csv_line']=range(2,len(df)+2)
 b=dict(sheet=f'Table S{num}',sheet_number=num,key=key,title=title,source=str(Path(path).relative_to(ROOT)),columns=list(df),rows=[[safe(v) for v in a] for a in df.itertuples(index=False,name=None)],title_row=rownext[num],source_row=rownext[num]+1,header_row=rownext[num]+2,first_data_row=rownext[num]+3)
 for n in range(len(df)):
  for j,c in enumerate(df):locations[key,n,c]=f"'Table S{num}'!{col(j+1)}{b['first_data_row']+n}"
 rownext[num]+=len(df)+5;blocks.append(b);return original

# S6, selected and base-rate frozen-representation results.
t=frame(D['a']/'tuning_trials.csv');rates=frame(D['a']/'selected_learning_rates.csv');t['selected_flag']=[bool(((rates.input_set==r.input_set)&(rates.backbone==r.backbone)&(rates.h==r.h)&(rates.variant==r.variant)&(rates.lr==r.lr)).any()) for r in t.itertuples()]
computed.append(dict(item='S6.A selected_flag',operation='Exact key+learning-rate match to stored selected_learning_rates;no metric recomputation',n_values=len(t),sources='tuning_trials.csv;selected_learning_rates.csv'))
add(6,'a_trials','A. Every learning-rate tuning trial',D['a']/'tuning_trials.csv',t,evidence='validation tuning;TEST not used',learning_rate_condition='all preregistered trial rates')
add(6,'a_rates','A. Selected learning rates',D['a']/'selected_learning_rates.csv',evidence='validation selection')
add(6,'a_summary','B. Selected-rate five-seed metrics',D['a']/'five_seed_summary.csv',evidence='descriptive',learning_rate_condition='selected',head_retrained=False)
add(6,'a_tests','C. All72 selected-rate contrasts',D['a']/'joint_bootstrap_tests.csv',learning_rate_condition='selected',head_retrained=False)
add(6,'a_base','D. Base-rate five-seed metrics (Step21 DA)',D['a']/'S6_STEP16_REFERENCE.csv',evidence='descriptive',learning_rate_condition='base 1e-4',head_retrained=False)
add(6,'base_tests','D/E. Base-rate head-frozen/head-retrained contrasts',D['frozen']/'paired_seed_date_bootstrap.csv',learning_rate_condition='base 1e-4',da_definition='Original Step15 direction accuracy may include zeros; see Step21 DA block')
add(6,'base_wins','D/E. Every base-rate seed outcome',D['frozen']/'seed_wins.csv',learning_rate_condition='base 1e-4')
add(6,'base_summary','E. All original base-rate settings (including head-retrained)',D['frozen']/'five_seed_summary.csv',evidence='descriptive',learning_rate_condition='base 1e-4',direction_accuracy_definition='Original Step15 definition;Step21 zero-excluded DA below')
add(6,'da_sensitivity','E. Sensitivity: experts frozen,head retrained (S1 only);Step21 DA',D['21']/'frozen_sensitivity_da_summary.csv',evidence='descriptive',learning_rate_condition='base 1e-4',head_retrained=True)
add(6,'da_tests','D/E. Zero-excluded base-rate DA contrasts (Step21)',D['21']/'paired_da_tests.csv',learning_rate_condition='base 1e-4;nonfrozen rows retain their original scope')
add(6,'a_router','F. Routing-weight movement and observation-invariant seed counts',D['a']/'router_diagnostics_summary.csv',evidence='descriptive')
add(6,'a_router_seed','F. Per-seed routing-weight movement',D['a']/'router_diagnostics_per_seed.csv',evidence='descriptive')
add(6,'c_frozen','G. Frozen tuned effect-size bounds (Step22C group2)',D['c']/'table_2_frozen_tuned.csv',evidence='descriptive bounds')
add(6,'a_tuned_base','G. Stored selected-rate/base-rate comparison',D['a']/'tuned_vs_Step16.csv',evidence='descriptive')

# S14 robustness settings and S15 pooled windows.
for key,title,name in [('b2_summary','A. Five-seed metrics,T0/T1/T2/T3','five_seed_summary.csv'),('b2_ensemble','B. Equal-weight ensembles','ensemble_metrics.csv'),('b2_tests','C. T3 joint paired contrasts','T3_joint_bootstrap_tests.csv'),('b2_bounds','D. T3 effect-size bounds','T3_effect_size_bounds.csv'),('b2_best','E. Best-epoch distribution','best_epoch_distribution.csv'),('b2_loss','E. Per-seed loss changes','learning_changes_per_seed.csv'),('b2_lossmean','E. Mean loss changes','learning_changes_summary.csv'),('b2_router','E. Routing-weight movement','router_diagnostics_summary.csv')]:add(14,key,title,D['b2']/name,evidence='descriptive' if key not in ['b2_tests'] else None)
add(14,'b_tests','C. T1/T2 joint paired contrasts',D['b']/'joint_bootstrap_tests.csv')
add(14,'c_end','D. T0 end-to-end bounds,O and C',D['c']/'table_1_end_to_end.csv',evidence='descriptive bounds')
add(14,'c_settings','D. T1/T2 bounds,C',D['c']/'table_3_training_settings.csv',evidence='descriptive bounds')
for key,title,name in [('d_summary','A. Per-window and pooled metrics','part1_metric_summary.csv'),('d_ic','B. Rank IC inference,NW and MBB','part1_rank_ic_inference.csv'),('d_contrast','C. Per-window and pooled Static-minus-Adaptive contrasts','part1_static_adaptive_contrasts.csv')]:add(15,key,title,D['d']/name)

# S16: E contains all three models and both settings (including the D LSTM source rows).
for key,title,name in [('e_summary','A. Five-seed metrics,all models/settings/windows','five_seed_summary.csv'),('e_ensemble','B. Equal-weight ensembles (both stored R2 references)','ensemble_metrics.csv'),('e_cw','C. CONFIRMATORY E2,CW','E2_CONFIRMATORY_CW.csv'),('e_setting','C. CONFIRMATORY E3,T3-minus-T0','E3_CONFIRMATORY_setting.csv'),('e_route','C. CONFIRMATORY E4,Static-minus-Adaptive','E4_CONFIRMATORY_routing.csv'),('e_ic','D. E1 SECONDARY T3 IC and descriptive T0','E1_SECONDARY_rank_ic.csv'),('e_3w','D. Three-window T3 Rank IC,descriptive','three_window_rank_ic_descriptive.csv'),('e_best','E. Best epochs per seed','best_epochs_per_seed.csv'),('e_bestdist','E. Best-epoch distribution','best_epoch_distribution.csv'),('e_logit','E. Static-logit/weight movement','static_logit_summary.csv'),('e_logitseed','E. Per-seed Static logits','static_logits_per_seed.csv'),('e_attempts','E. Every Step22E training attempt','all_training_attempts.csv')]:add(16,key,title,D['e']/name)
add(16,'d_confirm','C. CONFIRMATORY Step22D LSTM CW and Rank IC',D['d']/'confirmatory_tests.csv')
add(16,'d_attempts','E. Every Step22D LSTM run;window TRAIN sigma_h in delta',D['d']/'all_training_runs.csv')
add(16,'d_ref','E. Stored window-specific references',D['d']/'window_reference_values.csv',evidence='descriptive')

for key,title,name in [('f_summary','A. EXPLORATORY five-seed metrics','five_seed_summary.csv'),('f_ensemble','A. EXPLORATORY ensembles','ensemble_metrics.csv'),('f_move','B. EXPLORATORY branch-wise manipulation check','manipulation_summary.csv'),('f_gate','B. EXPLORATORY manipulation PASS/FAIL','manipulation_gate.csv'),('f_tests','C. EXPLORATORY all PRIMARY/SECONDARY contrasts','joint_bootstrap_tests.csv'),('f_attempts','D. EXPLORATORY every attempt,including interruptions','all_training_attempts.csv'),('f_router','B. EXPLORATORY per-seed weights/logits','router_diagnostics_per_seed.csv')]:add(17,key,title,D['f']/name,evidence='EXPLORATORY')

# S18 addendum: copy the Step22G FINAL source values and every attempt.
sig=frame(R/'step22g/sigma_cs.csv');sig['target_definition']='y_cs(i,t)=y(i,t)-mean_j y(j,t);17 retained stocks/date;unscaled'
add(18,'g_sigma','A. EXPLORATORY target definition and TRAIN sigma_cs',R/'step22g/sigma_cs.csv',sig,evidence='EXPLORATORY')
for key,title,name in [('g_summary','B. EXPLORATORY five-seed metrics;Ridge n=1','five_seed_summary.csv'),('g_ensemble','B. EXPLORATORY equal-weight ensembles','ensemble_metrics.csv'),('g_ic','C. EXPLORATORY PRIMARY/BENCHMARK Rank IC tests','IC_tests.csv'),('g_tests','D. EXPLORATORY SECONDARY-A/B/C contrasts and bounds','joint_bootstrap_tests.csv'),('g_rank','E. EXPLORATORY T0/T3/CS Rank IC','rank_ic_T0_T3_CS.csv'),('g_router','E. EXPLORATORY routing-weight movement','router_summary.csv'),('g_routerseed','E. EXPLORATORY per-seed routing movement','router_per_seed.csv'),('g_portfolio','F. EXPLORATORY portfolios,descriptive only;no borrowing costs/cash interest','portfolio_metrics.csv'),('g_attempts','G. EXPLORATORY every training attempt','all_training_attempts.csv')]:add(18,key,title,D['g']/name,evidence='EXPLORATORY')

# S13 retains every original cell; appended metadata explicitly identifies base-rate/head setting.
s13meta=[]
for header,first,key,source in [(83,84,'da_primary','frozen_primary_da_summary.csv'),(138,139,'da_sens','frozen_sensitivity_da_summary.csv'),(169,170,'da_pair','paired_da_tests.csv')]:
 f=frame(D['21']/source);catalog[key]=f
 for n,r in f.iterrows():
  frozen=r['scope']=='frozen';s13meta.append(dict(row=first+n,condition='base 1e-4' if frozen else 'formal end-to-end',head='retrained' if frozen and r.setting=='head_retrained' else 'frozen' if frozen else 'not applicable'))
catalog['da_table4']=frame(D['21']/'table4_da_summary.csv');catalog['da_references']=frame(D['21']/'references.csv')
for key,first,mapping in [('da_table4',23,{'input_set':1,'model':4,'h':5,'da_mean':7,'da_sd':8}),('da_primary',84,{'input_set':1,'model':4,'h':5,'da_mean':7,'da_sd':8}),('da_sens',139,{'input_set':1,'model':4,'h':5,'da_mean':7,'da_sd':8}),('da_pair',170,{'estimate_pp':6,'ci_lower_95_pp':7,'ci_upper_95_pp':8,'centered_p_two_sided':9,'p_holm_3h':10}),('da_references',13,{'positive_return_share':4,'always_up_da':3,'global_train_mean_da':2})]:
 for n in range(len(catalog[key])):
  for c,j in mapping.items():locations[key,n,c]=f"'Table S13'!{col(j)}{first+n}"
for i in range(1,14):
 if i==6:continue
 for addr,v in old[f'S{i}'].items():
  if isinstance(v,str) and v.startswith('revision_work/') and (ROOT/v).exists():readpath(v)

payload=dict(base=str(base),titles=titles,notes=notes,hashes=hashes,blocks=blocks,old_inventory=original_inventory,s13meta=s13meta,row_counts=rownext,table7_status='PENDING: user approved deferral of full manuscript Table7;manuscript source not supplied',crosscheck_claim_precision='Apply manuscript rounding to source values;retain exact underlying numbers')
dump(OUT/'payload23.json',payload)
dump(OUT/'catalog23.json',{k:dict(columns=list(f),rows=[[safe(v) for v in a] for a in f.itertuples(index=False,name=None)]) for k,f in catalog.items()})
dump(OUT/'locations23.json',{f'{k}|{i}|{c}':v for (k,i,c),v in locations.items()})
csvout(OUT/'SOURCES_PREPARE.csv',pd.DataFrame(list(sources.values())))
csvout(OUT/'COMPUTED_PREPARE.csv',pd.DataFrame(computed))
print('Prepared',len(blocks),'complete source blocks;',len(sources),'sources;no training/evaluation/bootstrap.')
