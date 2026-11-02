"""Package the authoritative report and strict-audit tables without overwrites."""
import sys,json,re,hashlib,csv,datetime
from pathlib import Path
sys.dont_write_bytecode=True
D=Path(__file__).parent;ROOT=Path.cwd();sys.path.insert(0,str(D));from sources23b import *
A=pd.read_csv(D/'final/AUDIT_BOUNDS_STEP23B.csv',float_precision='round_trip').to_dict('records');C=pd.read_csv(D/'final/BOUND_COVERAGE_STEP23B.csv').to_dict('records');EX=json.loads((D/'EXTRACTED_TEXT.json').read_text());REL=D/'release';REL.mkdir(exist_ok=False)
# Original snapshot: copy the zero-volume records, without rebuilding any weekly bars or model inputs.
k='dataset/finance/panel_raw/BLNK_daily.csv';f=table(k);q=f[f.datetime.ge('2015-08-05')&f.datetime.le('2026-07-07')&f.volume.eq(0)];wk='dataset/finance/panel_raw/BLNK_weekly.csv';w=table(wk);w=w[w.datetime.ge('2015-08-05')&w.datetime.le('2026-07-07')]
r=next(a for a in A if a['candidate_id']=='M0041.5');r.update(claim='BLNK original-snapshot zero-volume observations in the common interval',source_values=json.dumps(dict(daily_records=q.to_dict('records'),n_daily=len(q),weekly_zero_rows=int(w.volume.eq(0).sum()),weekly_interval_first=w.datetime.min(),weekly_interval_last=w.datetime.max())),source_file_cell='; '.join(k+':'+str(i+2)+' [datetime,volume]' for i in q.index)+'; '+wk+' [volume; common interval]',classification='STRICT_OK' if len(q)==8 and int(w.volume.eq(0).sum())==0 else 'WRONG',proposed_fix='',scope='Read-only row extraction; no weekly series rebuilt, no model evaluation or hypothesis test.')
# Stable, unambiguous DOCX locations: section + original nonempty paragraph number; reviewer + comment.
loc={};sec='Front matter';reviewer='Summary';comment='Summary'
for p in EX['paragraphs']:
 t=p['text'];n=p['paragraph'];doc=p['document']
 if doc=='manuscript':
  if not p['in_table'] and n>=10 and re.match(r'^\d+(?:\.\d+){0,2}\.?\s+[A-Za-z]',t) and len(t)<200:sec=t
  loc[(doc,n)]=sec+f'; manuscript P{n:04d}'
 else:
  m=re.search(r'Response to the Comments of Reviewer\s*#(\d+)',t,re.I)
  if m:reviewer='Reviewer #'+m[1];comment='Introduction'
  m=re.fullmatch(r'Comment\s*#(\d+)',t.strip(),re.I)
  if m:comment='Comment #'+m[1]
  loc[(doc,n)]=reviewer+' / '+comment+f'; response P{n:04d}'
for a in A:a['location']=loc[(a['document'],a['paragraph'])]
for a in C:
 doc='manuscript' if a['candidate_id'].startswith('M') else 'response';n=int(re.search(r'\d+',a['candidate_id'])[0]);a['location']=loc[(doc,n)]
# Clarify the scope of claims rather than silently treating rounded source summaries as full-precision estimates.
for a in A:
 if 'mean_train_huber_pct_change' in str(a['source_file_cell']) or 'mean_val_huber_pct_change' in str(a['source_file_cell']):a['scope']=str(a.get('scope',''))+' Step19 CSV is stored at six significant digits; those stored values, not reconstructed extra precision, are quoted. Threshold tests use higher-precision Step22B2 summaries when available.'

def writecsv(path,rows):
 with path.open('x',newline='') as h:pd.DataFrame(rows).to_csv(h,index=False)
writecsv(REL/'AUDIT_BOUNDS_STEP23B.csv',A);writecsv(REL/'BOUND_COVERAGE_STEP23B.csv',C)
S=pd.concat([pd.read_csv(D/'final/SOURCES_BOUNDS.csv'),pd.read_csv(D/'SOURCES_PARTS13.csv'),pd.read_csv(D/'SOURCE_DOCUMENTS.csv'),pd.DataFrame(SOURCES.values())],ignore_index=True).drop_duplicates('path')
checks=[]
for r in S.to_dict('records'):
 p=Path(r['path']);p=p if p.is_absolute() else ROOT/p;yes=hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];checks.append(dict(path=r['path'],unchanged=yes));assert yes,r
writecsv(REL/'SOURCES.csv',S.to_dict('records'));writecsv(REL/'SOURCE_HASH_VERIFICATION.csv',checks)
counts=pd.Series([a['classification'] for a in A]).value_counts().to_dict();writecsv(REL/'AUDIT_COUNTS.csv',[dict(classification=k,n=v) for k,v in counts.items()])
T=pd.read_csv(D/'TABLE7_CROSSCHECK.csv');assert len(T)==108 and T.status.eq('MATCH').all()
WB=json.loads((D/'WORKBOOK_COMPLETE.json').read_text());assert hashlib.sha256(Path(WB['workbook']).read_bytes()).hexdigest()==WB['sha256']
# No original workbook numeric data modified; artifact export/reimport validated all retained cells.
F=pd.read_csv(D/'T0_NEURAL_BENCHMARK_CONTRASTS.csv',float_precision='round_trip');c=F[(F.input_set=='C')&(F.h==5)&(F.metric=='squared_loss')&(F.method=='joint_seed_date')]
o=F[(F.input_set=='O')&(F.h==20)&(F.metric=='squared_loss')&(F.method=='joint_seed_date')&(F.contrast.isin(['swim−static','swim−adaptive']))]
assert len(c)==6 and c.seed_wins.between(4,5).all() and (c.theta_hat<0).all() and (c.p_holm_3h>=.067).all()
assert len(o)==2

def esc(v):return str(v).replace('|','\\|').replace('\n',' ')
def compact(a):
 if pd.notna(a.get('source_min')) and pd.notna(a.get('source_max')):return f"[{a['source_min']:.17g}, {a['source_max']:.17g}]"
 v=str(a.get('source_values',''));return v if len(v)<350 else v[:350]+' … (full values in CSV)'
def shortref(v):
 chunks=str(v).split('; ');return '; '.join(chunks[:3])+(' ; +'+str(len(chunks)-3)+' locators in CSV' if len(chunks)>3 else '')
def mdtable(rows,columns):return '| '+' | '.join(h for h,k in columns)+' |\n|'+ '|'.join('---' for h,k in columns)+'|\n'+''.join('| '+' | '.join(esc(r.get(k,'')) for h,k in columns)+' |\n' for r in rows)
intro='# BLOCKING\n\nThe documents require numerical wording corrections listed below. No source document was edited. Eight claim occurrences are NOT_VERIFIABLE from the located stored records: four power-requirement statements (two unique wordings), three assertions about Sharpe sampling error, and one fitted-residual attribution. NOT_VERIFIABLE is kept separate from WRONG because a missing/insufficient source cannot establish falsity. No new inference was computed.\n\n'
intro+='The authoritative machine-readable audit is [release/AUDIT_BOUNDS_STEP23B.csv](release/AUDIT_BOUNDS_STEP23B.csv). Earlier root-level/v5/FINAL CSVs and scripts are preserved development drafts; this report and the release/ files supersede them.\n\n'
intro+='## Definitions and coverage\n\nSTRICT_OK compares inequalities directly with stored unrounded values. ROUNDING_ONLY means the inequality fails before rounding but passes at the claim precision; proposed bounds are rounded outward. RANGE_OK requires the displayed endpoints to equal the rounded minimum and maximum; descending signed ranges are RANGE_MISMATCH under the requested min/max convention. A displayed equality (e.g. p=.015) is checked at its displayed precision and labelled STRICT_OK when it matches; it is not an unrounded inequality. Unnamed NW/MBB p claims include both methods. Claims explicitly naming NW are checked against NW, with the MBB disagreement recorded where stated. Approximate/descriptive claims are labelled with their exact scope in the CSV.\n\n'
intro+='Both DOCX files were extracted, including Word math text, without modification: '+str(len(EX['paragraphs']))+' nonempty paragraphs; '+str(len(C))+' candidate sentence occurrences. Coverage: '+str(pd.Series([r['disposition'] for r in C]).value_counts().to_dict())+'. Every exclusion is listed in release/BOUND_COVERAGE_STEP23B.csv (e.g. bibliographic page/date ranges, scientific-notation units and nonnumeric keyword hits); no exclusion is a statistical pass.\n\n'
intro+='Only stored-value lookup, descriptive division/means/counts, sign/interval reversal and rounding were used. No training, predictions re-evaluation, bootstrap rerun, p-value computation, correction or new test. CSV values retain their source precision; six-significant-digit source summaries are explicitly identified, never expanded into invented precision. Full raw vectors and exact file/row/column locators are in the CSV.\n\n'
intro+='## Counts\n\n'+mdtable([dict(classification=k,n=v) for k,v in counts.items()],[('Class','classification'),('Atomic claim occurrences','n')])+'\nCounts include repeated passages in the response letter; they are not counts of unique scientific findings.\n\n'
# Full detailed audit: every atomic claim, source vectors in CSV and compact exact endpoints here.
rows=[dict(**a,raw=compact(a),ref=shortref(a['source_file_cell'])) for a in A]
audit=intro+'## All bound/range claims\n\n'+mdtable(rows,[('ID','candidate_id'),('Location','location'),('Sentence','sentence'),('Atomic claim','claim'),('Unrounded source / extrema','raw'),('Source file/cell','ref'),('Class','classification'),('Proposed fix','proposed_fix')])
with (D/'AUDIT_BOUNDS_STEP23B.md').open('x') as h:h.write(audit)
issues=[a for a in rows if a['classification'] not in ['STRICT_OK','RANGE_OK']]
report=intro+'## Part 1 — Table7\n\n108 numeric cells (36 contrast/input/backbone/horizon rows × ΔMSE, seed wins and Holm p): **108 MATCH, 0 MISMATCH**. Mismatch list: none.\n\n[TABLE7_CROSSCHECK.csv](TABLE7_CROSSCHECK.csv) gives every DOCX row/column, stated value, stored value, direction mapping and CSV locator. ΔMSE is first minus second, scaled×10⁴. Adaptive-minus-Static is the exact sign/interval reversal of stored Static-minus-Adaptive; seed wins are swapped using verified absence of ties. All twelve Static selected rates are1e−4, and their MSE means coincide with the Step22A base reference, so Static-minus-Fixed uses the stored Step15 base-rate tests and seed-win records. No Holm p was recalculated.\n\n'
report+='## Part 2 — Strict audit\n\n'+str(len(A))+' atomic claim occurrences: '+str(counts)+'. The full sentence/source audit is [AUDIT_BOUNDS_STEP23B.md](AUDIT_BOUNDS_STEP23B.md), with full vectors in the release CSV. The list below contains every failed or unverifiable occurrence, including repeats; valid ranges (RANGE_OK) are listed immediately afterwards to satisfy the full non-STRICT_OK inventory.\n\n'
report+=mdtable(issues,[('ID','candidate_id'),('Location','location'),('Claim','claim'),('Raw source / extrema','raw'),('Class','classification'),('Proposed correction','proposed_fix')])
report+='\n### Valid ranges (non-STRICT_OK by class name)\n\n'+mdtable([a for a in rows if a['classification']=='RANGE_OK'],[('ID','candidate_id'),('Location','location'),('Range claim','claim'),('Unrounded extrema','raw'),('Class','classification')])
report+='\n## Part 3 — T0 neural-benchmark contrasts\n\nStored source: `revision_work/outputs/analysis16/E_paired_tests.csv` (144rows, joint_seed_date and date_only, both inputs, both metrics). No statement in the requested Section4.1 neural comparison lacks a stored source.\n\n'
report+=mdtable(c.to_dict('records'),[('C h5 contrast','contrast'),('ΔMSE','theta_hat'),('First-arm wins/5','seed_wins'),('Stored Holm p','p_holm_3h')])
report+='\nAll six estimates are negative, every neural model wins in4 or5 seeds, and the minimum joint Holm p is '+format(float(c.p_holm_3h.min()),'.17g')+' ≥.067 strictly. The statement is supported.\n\n'
report+=mdtable(o.to_dict('records'),[('O h20 contrast','contrast'),('ΔMSE','theta_hat'),('Stored Holm p','p_holm_3h')])
report+='\nThe two positive O h20 SWiM MSE differences have stored Holm p '+', '.join(f"{r.contrast}: {r.p_holm_3h:.17g}" for r in o.itertuples())+', which display as.020 and.030.\n\n'
report+='The new workbook adds all144 source rows in S14 blockK, rows584–730, and restores six C T0 Adaptive-minus-LSTM MSE/RankIC cells in S14 blockC (rows571–579). Source LSTM-minus-Adaptive estimates/intervals are reversed, seed wins/losses swapped and stored p retained. All three DA contrasts remain explicitly NOT STORED. The wider stored table includes O as well. README records the source and the now-complete Table7 audit.\n\n'
report+='Workbook: [Supplementary_Tables_S1-S18_v2.xlsx](Supplementary_Tables_S1-S18_v2.xlsx). FileSHA-256: `'+WB['sha256']+'`. All19 sheets retained; '+str(WB['original_cells_preserved'])+' original populated cells outside the authorized patch were checked, '+str(WB['checks'])+' validation checks passed. '+str(WB['utc_repairs'])+' UTC compatibility repairs were needed; all original UTC text is preserved. Source workbook SHA-256 rechecked unchanged. Native table/drawing/chart counts preserved; changed S14/README views rendered and visually reviewed.\n\n'
report+='## Files and source preservation\n\n- `AUDIT_BOUNDS_STEP23B.md`; `release/AUDIT_BOUNDS_STEP23B.csv`; `release/BOUND_COVERAGE_STEP23B.csv`; `release/AUDIT_COUNTS.csv`.\n- `TABLE7_CROSSCHECK.csv`, `TABLE7_VERIFIED_VALUES.csv`.\n- `T0_NEURAL_BENCHMARK_CONTRASTS.csv`, `T0_ADAPTIVE_MINUS_LSTM_STORED.csv`.\n- `release/SOURCES.csv` and `release/SOURCE_HASH_VERIFICATION.csv`: '+str(len(S))+' read-only sources, all hashes unchanged. Input DOCX hashes also recorded in SOURCE_DOCUMENTS.csv.\n- `WORKBOOK_COMPLETE.json`, `WORKBOOK_VALIDATION_v2.csv`: independent final-workbook integrity checks.\n\nAll authoring used @oai/artifact-tool; final ZIP validation is read-only except for its initial exclusive output creation. Existing documents, code, data and workbooks remain untouched. The inherited source workbook is preserved.\n'
with (D/'REPORT_STEP23B.md').open('x') as h:h.write(report)
with (D/'DRAFTS_README.md').open('x') as h:h.write('The authoritative outputs are REPORT_STEP23B.md, AUDIT_BOUNDS_STEP23B.md, release/*.csv, and Supplementary_Tables_S1-S18_v2.xlsx. Root-level audit CSVs, final/ audit CSVs, private exports and earlier builder scripts are preserved development drafts, superseded by release/. Existing-file read-only rules prevented overwriting or deleting drafts.\n')
with (D/'COMPLETE_STEP23B.json').open('x') as h:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),table7=dict(cells=108,match=108,mismatch=0),audit_atomic_occurrences=len(A),audit_counts=counts,candidate_occurrences=len(C),audit_csv=str((REL/'AUDIT_BOUNDS_STEP23B.csv').relative_to(ROOT)),workbook=WB['workbook'],workbook_sha256=WB['sha256'],source_files_unchanged=len(S),new_tests=False,new_training=False,unavailable_DA_tests=3),h,indent=2)
print('RELEASE COMPLETE',counts,'sources',len(S))
