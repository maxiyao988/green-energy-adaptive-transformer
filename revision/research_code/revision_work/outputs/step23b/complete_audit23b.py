"""Final descriptive audit addenda; preserve all preceding draft files."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
D=Path(__file__).parent
s=(D/'audit_bounds23b_final.py').read_text();exec(compile(s.split("for x0 in EX['candidates']:")[0],str(D/'audit_bounds23b_final.py'),'exec'),globals())
f=pd.read_csv(D/'AUDIT_BOUNDS_STEP23B_FINAL.csv',float_precision='round_trip').where(lambda v:v.notna(),None)
AUD=f.to_dict('records');COVER=pd.read_csv(D/'BOUND_COVERAGE_STEP23B_FINAL.csv').to_dict('records')
recovered=set()
for x in EX['candidates']:
 ss=x['sentence'];new=False
 if 'Ridge gains most' in ss or 'Daily+Weekly 20-day out-of-sample R²' in ss:
  recovered.add(x['candidate_id']);q=table('B');h,config=(10,'daily_only') if 'Ridge gains most' in ss else (20,'daily_weekly');q=q[q.model.eq('ridge')&q.config.eq(config)&q.h.eq(h)]
  explicit(x,'Ridge stored R2 change O→C',q[['input_set','r2os_vs_global_train_mean','mean_rank_ic']].to_dict('records'),ALIASES['B']+' [ridge; '+config+'; h='+str(h)+']',scope='Two input sets, same configuration/horizon. Descriptive stored values only.')
 if 'Test period 12 November' in ss:
  recovered.add(x['candidate_id']);q=table('revision_audit/split_counts.csv');q=q[q.split.eq('test')];explicit(x,'Exact TEST date interval and10bp portfolio setting',q.to_dict('records'),'revision_audit/split_counts.csv:4; '+ALIASES['G']+' [split=test,cost_bp=10]')
 if 'BLNK between 2015 and 2018' in ss:
  # No count is recomputed from the snapshot in an extraction-only audit.
  pass
# Remove only generic unresolved placeholders for newly resolved candidates.
AUD=[a for a in AUD if not(a['candidate_id'] in recovered and a['classification']=='NOT_VERIFIABLE')]
for a in AUD:
 if a['claim']=='Scope of .33% limit':a['proposed_fix']='With price-level inputs and Fixed-trained representations, Static and Fixed differ by at most 0.34%; the .33% bound is rounding-only even in OS1 and does not cover C/S2.'
 # Floating-point definition data are copied at stored precision, not falsely treated as fitted model residuals.
 if a['claim']=='Learned per-seed cross-sectional R2 ≤.14%':
  a['scope']='The .14% description originated in the older O/seed42 benchmark scope (Step5). Current text does not restrict the scope; Step16 learned O/C per-seed cells reach .3840123340564161%. If the old scope is intended, name it explicitly.'
 if a['claim']=='Dividend-payer median original/new adjusted-price factor':
  a['proposed_fix']='Across all seven vendor cash-dividend payers the median snapshot factor spans1.000–1.027; JKS is1.000. If JKS is excluded, explicitly name the remaining six securities.' if a['classification']=='RANGE_MISMATCH' else a['proposed_fix']
# Audit actual unrounded NW and MBB methods alongside every statistical bound already mapped.
# Add the explicit Section4.1 O SWiM significance values that are central to Part3.
for x in EX['paragraphs']:
 if 'SWiM-style' not in x['text'] or '0.020' not in x['text'] or '0.030' not in x['text']:continue
 xx=dict(candidate_id=('M' if x['document']=='manuscript' else 'R')+str(x['paragraph']).zfill(4)+'.PART3',document=x['document'],location=x['location'],paragraph=x['paragraph'],sentence=x['text'])
 for c,v in [('swim−static',.020),('swim−adaptive',.030)]:add(xx,'O SWiM h20 MSE joint Holm p '+c,V('E','p_holm_3h',input_set='O',h=20,metric='squared_loss',method='joint_seed_date',contrast=c),'eq',v,3)
# Preserve an exhaustive candidate disposition inventory. Bibliographic ranges are excluded explicitly.
for c in COVER:
 if c['candidate_id'] in recovered:c['reason']='Recovered by exact stored-source addendum.'
FINAL=OUT/'final';FINAL.mkdir(exist_ok=False)
for name,rows in [('AUDIT_BOUNDS_STEP23B.csv',AUD),('BOUND_COVERAGE_STEP23B.csv',COVER)]:
 with (FINAL/name).open('x',newline='') as h:pd.DataFrame(rows).to_csv(h,index=False)
S=pd.concat([pd.read_csv(D/'SOURCES_BOUNDS_FINAL.csv'),pd.DataFrame(SOURCES.values())],ignore_index=True).drop_duplicates('path')
with (FINAL/'SOURCES_BOUNDS.csv').open('x',newline='') as h:S.to_csv(h,index=False)
print(pd.Series([a['classification'] for a in AUD]).value_counts().to_dict());print('Unverifiable',[(a['candidate_id'],a['claim']) for a in AUD if a['classification']=='NOT_VERIFIABLE'])
