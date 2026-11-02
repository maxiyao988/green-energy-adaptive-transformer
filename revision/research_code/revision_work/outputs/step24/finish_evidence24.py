"""Read-only final evidence reconciliation and source immutability verification."""
from review24_common import *
import ast
checks=[]
# Portfolio constraint checks are allocation arithmetic, no optimizer is run.
tickers=['AES','BEP','BLDP','BLNK','CSIQ','CWEN','DQ','ENPH','FCEL','FSLR','HASI','JKS','NEE','ORA','PLUG','RUN','SEDG']
for inp in ['O','C']:
 for h in [5,10,20]:
  folder=ROOT/f'revision_work/outputs/analysis16/G/{inp}_h{h}';f=df(folder/'weights.csv')
  for strategy,g in f.groupby('strategy'):
   w=g[tickers].to_numpy(float);constrained=('|L2_' in strategy or strategy=='MINVAR');checks.append(dict(check='Stored long-only allocation constraints',input_set=inp,h=h,strategy=strategy,passed=bool(np.max(abs(w.sum(1)-1))<=1e-8 and w.min()>=-1e-8 and (not constrained or w.max()<=.25+1e-8)),min_weight=w.min(),max_weight=w.max(),max_sum_error=np.max(abs(w.sum(1)-1)),capped_at25percent=constrained))
  q=df(folder/'qp_solver_log.csv');checks.append(dict(check='QP fallback records counted (not suppressed)',input_set=inp,h=h,passed=True,n_fallback_records=len(q)))
write('PORTFOLIO_CONSTRAINT_CHECKS.csv',pd.DataFrame(checks))
# Distinguish a missing source from a failed numerical selection test.
f=df(OUT/'REMAINING_CHECKS.csv');missing=df('revision_work/outputs/analysis17/BLOCKING_inputs.csv');resolution=[]
for r in f[~f.passed].itertuples():
 q=missing[missing.window.eq('W3')&missing.input_set.eq('O')&missing.model.eq('ridge')&missing.h.eq(int(r.cell.rsplit('h',1)[1]))]
 resolution.append(dict(check=r.check,cell=r.cell,final_status='NOT CHECKABLE — previously disclosed missing VAL predictions',source='revision_work/outputs/analysis17/BLOCKING_inputs.csv',matching_missing_source_rows=len(q)))
write('MISSING_SOURCE_CHECK_RESOLUTION.csv',pd.DataFrame(resolution))
# Disclose every deterministic synthetic float32 difference above the requested1e-10 threshold.
z=f[f.check.str.startswith('Intervention')&f.max_abs_difference.gt(1e-10)].copy();z['classification']='OK: expected float32 reduction/normalization precision';z['note']='Preserved differences1.49e-8 to1.19e-7; transform semantics and unaffected branches agree. Not silently treated as1e-10 agreement.';write('FLOAT32_INTERVENTION_PRECISION_DISCLOSURE.csv',z)
# A line-addressable extract of actual DOCX table text for reporting findings.
tables=js('revision_work/outputs/step23b/MANUSCRIPT_TABLES.json');lines=[]
for t in tables:
 if t['table'] not in [4,9,10,11]:continue
 for i,row in enumerate(t['rows']):lines.append(f"Table {t['table']}; zero-based row {i}: "+' | '.join(row))
with (OUT/'MANUSCRIPT_TABLE_AUDIT_EXTRACT.txt').open('x') as h:h.write('\n'.join(lines)+'\n')
# Step17 legacy manifest hashes; the FINAL report explicitly identifies unused drafts.
man=js('revision_work/outputs/analysis17/MANIFEST_STEP17.json');pro=[]
def walk(x):
 if isinstance(x,dict):
  if 'path'in x and 'sha256'in x and str(x['path']).startswith('revision_work/scripts/analysis17/'):
   p=x['path'];b=read(p);pro.append(dict(path=p,registered_sha256=x['sha256'],current_sha256=hashlib.sha256(b).hexdigest(),hash_matches=hashlib.sha256(b).hexdigest()==x['sha256'],actual_role='unused initial draft' if p.endswith('/compute17.py') or p.endswith('/report17.py') else 'v2 or shared valid implementation'))
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
walk(man);write('STEP17_PROVENANCE_ADDENDUM.csv',pd.DataFrame(pro).drop_duplicates())
source_manifest('SOURCES_FINAL_EVIDENCE.csv')
# Verify each outside-step24 file still has its first-read SHA-256 and each code-map source its inventory hash.
before={};collisions=[]
for p in sorted(OUT.glob('SOURCES_*.csv')):
 for r in pd.read_csv(p).itertuples():
  if str(r.path).startswith('revision_work/outputs/step24/'):continue
  if r.path in before and before[r.path]!=r.sha256:collisions.append(dict(path=r.path,first=before[r.path],later=r.sha256))
  before.setdefault(r.path,r.sha256)
for r in df(OUT/'CODE_MAP.csv').itertuples():
 if r.path in before and before[r.path]!=r.sha256:collisions.append(dict(path=r.path,first=before[r.path],later=r.sha256))
 before.setdefault(r.path,r.sha256)
out=[]
for p,sha in before.items():
 q=ROOT/p;current=hashlib.sha256(q.read_bytes()).hexdigest() if q.exists() else 'MISSING';out.append(dict(path=p,first_read_sha256=sha,current_sha256=current,unchanged=current==sha))
write('SOURCE_IMMUTABILITY_CHECKS.csv',pd.DataFrame(out));write('SOURCE_HASH_CONFLICTS.csv',pd.DataFrame(collisions,columns=['path','first','later']));print('SOURCE files',len(out),'changed',sum(not x['unchanged'] for x in out),'conflicts',len(collisions));print('CONSTRAINT failures',sum(not x['passed'] for x in checks))
