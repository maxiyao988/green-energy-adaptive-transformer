"""Exclusively create final stratified-replay addendum; preserve the initial completed report."""
from review24_common import *
checks=df(OUT/'STRATIFIED_REPLAY_CHECKS.csv');assert checks.passed.all()
summary=df(OUT/'CHECK_SUMMARY_STEP24.csv');summary=pd.concat([summary,pd.DataFrame([dict(evidence='STRATIFIED_REPLAY_CHECKS.csv',checks=len(checks),passed=len(checks),failed=0,not_checkable=0)])],ignore_index=True);write('CHECK_SUMMARY_STEP24_FINAL.csv',summary)
ledger=df(OUT/'CHECKLIST_STEP24.csv');extra=pd.DataFrame([dict(evidence='STRATIFIED_REPLAY_CHECKS.csv',evidence_row=int(i)+2,check=r['check'],status='PASS',details=r.to_json()) for i,r in checks.iterrows()]);write('CHECKLIST_STEP24_FINAL.csv',pd.concat([ledger,extra],ignore_index=True))
first={};conflicts=[]
for p in sorted(OUT.glob('SOURCES_*.csv')):
 for r in pd.read_csv(p).itertuples():
  if str(r.path).startswith('revision_work/outputs/step24/'):continue
  if r.path in first and first[r.path]!=r.sha256:conflicts.append(r.path)
  first.setdefault(r.path,r.sha256)
for r in df(OUT/'CODE_MAP.csv').itertuples():first.setdefault(r.path,r.sha256)
result=[]
for p,sha in first.items():
 now=hashlib.sha256((ROOT/p).read_bytes()).hexdigest();result.append(dict(path=p,first_read_sha256=sha,final_sha256=now,unchanged=sha==now))
assert not conflicts and all(x['unchanged'] for x in result);write('SOURCE_IMMUTABILITY_CHECKS_FINAL.csv',pd.DataFrame(result));tot=summary[['checks','passed','failed','not_checkable']].sum()
txt=f'''# Step24 final validation addendum

## BLOCKING

No CRITICAL finding. The classification in `REPORT_STEP24.md` is unchanged:3 MAJOR and3 MINOR findings, with no checked conclusion changed. No fix or training was performed.

After the initial report was exclusively created, a final326-assertion independent replay was added for the pooled/stratified Step22D calculations. All326 pass at1e−10. `stratified_replay24.py` imports the independent reference only and replays the exact stored inputs, draw matrices and20,000-replicate arrays:

- All9 within-window date-block matrices and paired-seed draw matrices match their recorded RNG streams; blocks remain within their own windows.
- All72 component/pooled Rank-IC replicate vectors and percentile intervals match; tested NW/MBB values agree.
- All48 component/pooled Static−Adaptive MSE/IC joint replicate vectors, estimates, intervals, p values and signed effect-size bounds match.
- All6 held-out LSTM confirmatory Rank-IC/CW replicate vectors and NW/MBB quantities match.
- Separate NW/MBB Holm families for PRIMARY, SECONDARY and held-out confirmation match.

No hypothesis, method, model, seed, source forecast or reported result was selected or changed. This supplement is a correctness replay of existing tests.

Final aggregate: **{int(tot.checks):,} assertions; {int(tot.passed):,} PASS; {int(tot.failed)} FAIL; {int(tot.not_checkable)} NOT CHECKABLE**. FAIL rows are the already documented findings, not326 new failures. The three NOT CHECKABLE rows remain the previously disclosed O/W3 Ridge missing-VAL selections.

Final source immutability check: **{len(result):,} source files,0 hash changes,0 conflicting first-read hashes**. The earlier3071-file count in `REPORT_STEP24.md` described the completed initial review; this larger final set includes the supplemental saved bootstrap sources.

Use `CHECK_SUMMARY_STEP24_FINAL.csv`, `CHECKLIST_STEP24_FINAL.csv`, `SOURCE_IMMUTABILITY_CHECKS_FINAL.csv` and `ARTIFACT_INDEX_STEP24_FINAL.csv` for the final full evidence set. The initial report, ledgers and artifact index are preserved exactly as created. Findings and per-experiment assessments remain in `REPORT_STEP24.md`, `FINDINGS_STEP24.csv` and `EXPERIMENT_STATUS_STEP24.csv`.
'''
with (OUT/'REPORT_STEP24_VALIDATION_ADDENDUM.md').open('x') as f:f.write(txt)
entries=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file():entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
write('ARTIFACT_INDEX_STEP24_FINAL.csv',pd.DataFrame(entries));print('FINAL',tot.to_dict(),'sources',len(result),'changes0')
