"""Rename-only XLSX metadata compatibility shim and independent cell/source audit."""
import sys,json,hashlib,re,zipfile,io,csv,shutil,datetime
from pathlib import Path
from xml.etree import ElementTree as E
sys.dont_write_bytecode=True
sys.path.append(str(Path.cwd()/'revision_work/outputs/_deps_pyarrow15'))
import pandas as pd
ROOT=Path.cwd();D=ROOT/'revision_work/outputs/step23';P=json.loads((D/'payload23_v2.json').read_text());NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse(p):
 z=zipfile.ZipFile(p);strings=[''.join(x.itertext()) for x in E.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
 wb=E.fromstring(z.read('xl/workbook.xml'));relations=E.fromstring(z.read('xl/_rels/workbook.xml.rels'));targets={r.attrib['Id']:r.attrib['Target'] for r in relations};tables={};formulas=0
 for s in wb.find('m:sheets',NS):
  name=s.attrib['name'];rid=s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'];path=targets[rid];path=path.lstrip('/') if path.startswith('/') else 'xl/'+path
  t=E.fromstring(z.read(path));cells={}
  for c in t.findall('.//m:c',NS):
   formulas+=len(c.findall('m:f',NS));v=c.find('m:v',NS);typ=c.attrib.get('t')
   if typ=='inlineStr':cells[c.attrib['r']]=''.join(c.find('m:is',NS).itertext());continue
   if v is None:continue
   text=v.text or ''
   cells[c.attrib['r']]=strings[int(text)] if typ=='s' else text if typ in ['str','e','d'] else bool(int(text)) if typ=='b' else float(text) if '.' in text.lower() or 'e' in text.lower() else int(text)
  tables[name]=cells
 return tables,formulas,z,wb
def col(n):
 s=''
 while n:n,k=divmod(n-1,26);s=chr(k+65)+s
 return s
export=Path(json.loads((D/'EXPORT_ARTIFACT_FINAL.json').read_text())['path']);z=zipfile.ZipFile(export)
wb=E.fromstring(z.read('xl/workbook.xml'))
for s in wb.find('m:sheets',NS):
 if re.fullmatch('S[0-9]+',s.attrib['name']):s.attrib['name']='Table '+s.attrib['name']
for t in wb.findall('.//m:definedName',NS):
 if t.text:
  for i in range(13,0,-1):t.text=re.sub(r"(?<![A-Za-z0-9_])(?:'S"+str(i)+r"'|S"+str(i)+r")!", "'Table S"+str(i)+"'!",t.text)
target=D/'Supplementary_Tables_S1-S18.xlsx'
memory=io.BytesIO()
with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as zz:
 for x in z.infolist():zz.writestr(x,E.tostring(wb,encoding='utf-8',xml_declaration=True) if x.filename=='xl/workbook.xml' else z.read(x.filename))
source=Path(P['base']);old,_,_,_=parse(source);new,nf,_,_=parse(io.BytesIO(memory.getvalue()));checks=[]
def check(name,condition,detail=''):
 checks.append(dict(check=name,passed=bool(condition),detail=detail))
 if not condition:raise AssertionError((name,detail))
def same(a,b):
 if a is None:return b is None or b==''
 if isinstance(a,bool):return isinstance(b,bool) and a==b
 if isinstance(a,(float,int)):return isinstance(b,(float,int)) and not isinstance(b,bool) and (float(a)==float(b) or abs(float(a)-float(b))<=abs(float(a))*2e-16)
 return a==b
check('Exactly README and Table S1 through S18',list(new)==['README']+[f'Table S{i}' for i in range(1,19)])
check('No formulas',nf==0,str(nf))
for i in range(1,19):check('Exact A1 title S'+str(i),new[f'Table S{i}'].get('A1')==f'Table S{i}. {P["titles"][str(i)]}')
unchanged=0
for i in range(1,14):
 if i==6:continue
 for a,v in old[f'S{i}'].items():
  if a in ['A1','A2'] or i!=13 and a=='A4' or i==13 and a in ['A9','A19','A80','A135','A166']:continue
  check('Original cell retained '+f'Table S{i}!{a}',same(v,new[f'Table S{i}'].get(a)),str(v)[:160]);unchanged+=1
copied=0
for b in P['blocks']:
 for n,row in enumerate(b['rows']):
  for j,value in enumerate(row):
   cell=f'{col(j+1)}{b["first_data_row"]+n}';actual=new[b['sheet']].get(cell)
   check('Source value copied '+b['key']+' '+cell,same(value,actual),f'expected={value};actual={actual}');copied+=1
for r in P['s13meta']:
 check('S13 rate metadata '+str(r['row']),new['Table S13'].get('AD'+str(r['row']))==r['condition'])
 check('S13 head metadata '+str(r['row']),new['Table S13'].get('AE'+str(r['row']))==r['head'])
check('README title-list S18',new['README'].get('B22')=='S18 Cross-sectional target')
for b in P['blocks']:
 if b['sheet_number']==18:check('Every S18 block labelled EXPLORATORY', 'EXPLORATORY' in b['title'])
C=pd.read_csv(D/'CROSSCHECK_STEP23_v2.csv');check('179 manuscript claims retained',len(C)==179)
for r in C.itertuples():
 for ref in str(r.sheet_cells).split('; '):
  m=re.fullmatch(r"'([^']+)'!([A-Z]+[0-9]+)",ref)
  check('Cross-check cell resolves '+ref,bool(m) and m[2] in new[m[1]])
check('All S18 benchmark and contrast rows included',sum(len(b['rows']) for b in P['blocks'] if b['key']=='g_ic')==9 and sum(len(b['rows']) for b in P['blocks'] if b['key']=='g_tests')==18)
S=pd.concat([pd.read_csv(D/'SOURCES_PREPARE.csv'),pd.read_csv(D/'SOURCES_APPEND.csv')],ignore_index=True).drop_duplicates('path')
for r in S.itertuples():check('Source unchanged '+r.path,sha(ROOT/r.path if not Path(r.path).is_absolute() else Path(r.path))==r.sha256)
with target.open('xb') as h:h.write(memory.getvalue())
with (D/'SOURCES.csv').open('x',newline='') as h:S.to_csv(h,index=False)
V=pd.DataFrame(checks)
with (D/'VALIDATION_CHECKS.csv').open('x',newline='') as h:V.to_csv(h,index=False)
with (D/'CROSSCHECK_STEP23_FINAL.csv').open('x',newline='') as h:C.to_csv(h,index=False)
text=(D/'CROSSCHECK_STEP23_v2.md').read_text()+'\n\nScope clarifications: aggregation variants include same-patch Fixed. The negative5-day individual-run R2 statement is checked for Static/Adaptive. The pooledR2 range has2 decimals on its lower endpoint and1 on its upper endpoint. BENCHMARK p>=.24 holds for the primary NW method;the MBB robustness minimum is .214489 (display .21),so applying that claim to MBB is a MISMATCH.\n'
with (D/'CROSSCHECK_STEP23_FINAL.md').open('x') as h:h.write(text)
CV=pd.concat([pd.read_csv(D/'COMPUTED_PREPARE.csv'),pd.read_csv(D/'COMPUTED_CROSSCHECK_v2.csv')],ignore_index=True)
with (D/'COMPUTED_VALUES.csv').open('x',newline='') as h:CV.to_csv(h,index=False)
changes=pd.DataFrame([dict(sheet='Table '+r['sheet'],old_cell='A2',old_title=r['old_title'],new_cell='A1',new_title=r['new_title']) for r in P['old_inventory']])
with (D/'TITLE_CHANGES.csv').open('x',newline='') as h:changes.to_csv(h,index=False)
rows=[]
for name,cells in new.items():
 rows.append(dict(sheet=name,last_populated_row=max(int(re.search(r'[0-9]+',a)[0]) for a in cells),populated_cells=len(cells),data_rows_in_new_blocks=sum(len(b['rows']) for b in P['blocks'] if b['sheet']==name)))
with (D/'SHEET_INVENTORY.csv').open('x',newline='') as h:pd.DataFrame(rows).to_csv(h,index=False)
report='# BLOCKING\n\n- Full manuscript Table7 cross-check remains pending,as explicitly authorized by the user;the manuscript table was not provided.\n- T0 Adaptive-minus-LSTM joint paired MSE/RankIC/DA tests are NOT STORED in the named sources. All9 cells are explicitly retained as NOT STORED in S14;no new tests were computed. Stored T0 Static-minus-Adaptive MSE/RankIC and DA were copied from Step16/21.\n- Two rounding/claim discrepancies are listed below. The Step22G BENCHMARK discrepancy concerns MBB only;the primary NW version of the claim matches. No manuscript was modified.\n\n'
report+='Workbook: ['+target.name+']('+str(target)+').\n\nContains README and all18 tables,including the Step22G addendum. S18 A1 is exactly `Table S18. Cross-sectional target`. Every S18 block is EXPLORATORY. Long-short portfolios exclude borrowing costs and cash interest;descriptive only. All45 Step22G attempts and all48 Step22F attempts,including3 interrupted attempts,are retained.\n\n'
report+='## Cross-check\n\n179 claims: '+str(C.status.value_counts().to_dict())+'. Authoritative files: [CROSSCHECK_STEP23_FINAL.md]('+str(D/'CROSSCHECK_STEP23_FINAL.md')+') and CSV. Earlier draft cross-check files are preserved and superseded by FINAL.\n\n'
for r in C[C.status=='MISMATCH'].itertuples():report+='- '+r.item_id+': manuscript '+r.manuscript_value+';stored '+r.workbook_value+';rounded '+r.rounded_value+';cells '+r.sheet_cells+'.\n'
report+='\nStep22G SECONDARY-C minimum Holm p=.217189→.22 at two decimals;the rounded inequality p≥.22 does not assert the unrounded value is≥.22. BENCHMARK NW minimum≈.241596,MBB minimum≈.214489.\n\n'
report+='## Provenance and preservation\n\n'+str(len(S))+' source files are indexed with SHA-256 in [SOURCES.csv]('+str(D/'SOURCES.csv')+');all hashes rechecked unchanged. Original workbook selected uniquely from the S1–S13 candidate inventory. Every original S1–S13 data cell retained except the requested S6 rebuild and S13 metadata/title edits. '+str(unchanged)+' unchanged original cells and '+str(copied)+' new-block cells checked against source values. No formulas,training,new hypothesis tests,bootstrap reruns,model evaluation or manuscript edits.\n\n'
report+='All cell authoring/import/export used @oai/artifact-tool. Its documented worksheet API did not expose a rename operation;the final ZIP shim changes worksheet-name/defined-name metadata only,after Artifact export. No cells or styles are authored by that shim. The source workbook remains unchanged. Final export is created with exclusive xb.\n\n'
report+='## Sheet inventory\n\n|Sheet|Last populated row|Populated cells|Data rows in rebuilt/appended blocks|\n|---|---:|---:|---:|\n'
for r in rows:report+='|'+str(r['sheet'])+'|'+str(r['last_populated_row'])+'|'+str(r['populated_cells'])+'|'+str(r['data_rows_in_new_blocks'])+'|\n'
report+='\n## Title changes and computed values\n\n[TITLE_CHANGES.csv]('+str(D/'TITLE_CHANGES.csv')+') records every original-title change. [COMPUTED_VALUES.csv]('+str(D/'COMPUTED_VALUES.csv')+') records selected-flag metadata joins and descriptive arithmetic used only to cross-check manuscript claims. Every inference value is copied from stored outputs. R2 against zero is already stored for S16,so no prediction-derived R2 computation was needed. Every source and block retains its source-file/CSV-line columns;README lists source files and protocol hashes.\n\nValidation: '+str(len(checks))+' checks PASS. Visual previews are retained for every sheet.\n'
with (D/'REPORT_STEP23.md').open('x') as h:h.write(report)
with (D/'COMPLETE_STEP23.json').open('x') as h:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workbook=str(target.relative_to(ROOT)),workbook_sha256=sha(target),n_sheets=len(new),n_source_files=len(S),n_validation_checks=len(checks),original_cells_retained=unchanged,new_source_cells_checked=copied,crosscheck_counts=C.status.value_counts().to_dict(),protocol_tests_recomputed=False,table7_pending_user_authorized=True),h,indent=2)
print('FINAL workbook',target,'checks',len(checks),'sources',len(S),'crosschecks',C.status.value_counts().to_dict())
