"""Independent OOXML cell preservation check, plus exact UTC-text export compatibility."""
import sys,zipfile,io,json,re,hashlib
from pathlib import Path
from xml.etree import ElementTree as E
sys.dont_write_bytecode=True
D=Path(__file__).parent;P=json.loads((D/'WORKBOOK_PATCH.json').read_text());NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def col(n):
 s=''
 while n:n,k=divmod(n-1,26);s=chr(k+65)+s
 return s
def parse(f):
 z=zipfile.ZipFile(f);ss=[''.join(x.itertext()) for x in E.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else [];w=E.fromstring(z.read('xl/workbook.xml'));rels={r.attrib['Id']:r.attrib['Target'] for r in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))};data={};paths={}
 for sh in w.find('m:sheets',NS):
  n=sh.attrib['name'];p=rels[sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']];p=p.lstrip('/') if p.startswith('/') else 'xl/'+p;paths[n]=p;t=E.fromstring(z.read(p));cells={}
  for c in t.findall('.//m:c',NS):
   v=c.find('m:v',NS);typ=c.attrib.get('t');a=c.attrib['r']
   if typ=='inlineStr':cells[a]=''.join(c.find('m:is',NS).itertext());continue
   if v is None:continue
   st=v.text or '';cells[a]=ss[int(st)] if typ=='s' else st if typ in ['str','e','d'] else bool(int(st)) if typ=='b' else float(st) if '.' in st.lower() or 'e' in st.lower() else int(st)
  data[n]=cells
 return z,data,paths
source=Path(P['base']);src_hash=hashlib.sha256(source.read_bytes()).hexdigest();oldz,old,oldpaths=parse(source);exp=D/'private_export_v3/artifact.bin';z,new,paths=parse(exp);trees={};repairs=[]
for sh,cells in old.items():
 for a,v in cells.items():
  if not isinstance(v,str) or not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',v):continue
  if new[sh].get(a)==v:continue
  p=paths[sh]
  if p not in trees:trees[p]=E.fromstring(z.read(p))
  c=trees[p].find(".//m:c[@r='"+a+"']",NS);assert c is not None;c.attrib['t']='str';cv=c.find('m:v',NS)
  if cv is None:cv=E.SubElement(c,'{'+NS['m']+'}v')
  cv.text=v;repairs.append(dict(sheet=sh,cell=a,source_value=v,export_value=str(new[sh].get(a)),reason='Artifact coerces ISO UTC text to millisecond dates; restore original text only'))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as zz:
 for info in z.infolist():zz.writestr(info,E.tostring(trees[info.filename],encoding='utf-8',xml_declaration=True) if info.filename in trees else z.read(info.filename))
_,out,_=parse(io.BytesIO(buf.getvalue()));allowed={'README':{'A2','F18','A160','B160','A161','B161'},'Table S14':{'A2','A568','A569'}|{col(j)+str(r['row']) for r in P['patches'] for j in range(1,15)}|{col(j)+'570' for j in range(11,15)}}
checks=[]
def check(name,yes,detail=''):
 checks.append(dict(check=name,passed=bool(yes),detail=detail))
 if not yes:raise AssertionError((name,detail))
def same(a,b):
 if a is None:return b in [None,'']
 if isinstance(a,bool):return type(b)==bool and a==b
 if isinstance(a,(int,float)):return isinstance(b,(int,float)) and not isinstance(b,bool) and (a==b or abs(a-b)<=abs(a)*2e-16)
 return a==b
check('19 sheet names/order preserved',list(out)==list(old))
n=0
for sh,cells in old.items():
 for a,v in cells.items():
  if a in allowed.get(sh,set()):continue
  check('Original cell '+sh+'!'+a,same(v,out[sh].get(a)),f'{v!r} -> {out[sh].get(a)!r}');n+=1
for r in P['patches']:
 row=r['values'][:];row[6]='See K:L';row[9]='Sign/CI reversed'
 for j,v in enumerate(row,1):check('Restored T0 cell '+col(j)+str(r['row']),same(v,out['Table S14'].get(col(j)+str(r['row']))))
for i,row in enumerate(P['rows']):
 for j,v in enumerate(row,1):check('New E row '+str(i+2)+' col '+str(j),same(v,out['Table S14'].get(col(j)+str(P['append_row']+3+i))))
for r in [573,576,579]:check('DA remains NOT STORED '+str(r),out['Table S14'].get('F'+str(r))=='NOT STORED')
check('No stale Table7 pending wording','pending' not in out['README']['A2'].lower())
for x in ['tables','drawings','charts']:
 a=[p for p in oldz.namelist() if re.match('xl/'+x+'/[^/]+\\.xml$',p)];b=[p for p in z.namelist() if re.match('xl/'+x+'/[^/]+\\.xml$',p)];check('Native '+x+' count preserved',len(a)==len(b),str((len(a),len(b))))
check('Source workbook unchanged',hashlib.sha256(source.read_bytes()).hexdigest()==src_hash)
target=D/'Supplementary_Tables_S1-S18_v2.xlsx'
with target.open('xb') as f:f.write(buf.getvalue())
import csv
for name,rows in [('WORKBOOK_VALIDATION.csv',checks),('EXPORT_UTC_COMPATIBILITY.csv',repairs)]:
 with (D/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (D/'WORKBOOK_COMPLETE.json').open('x') as f:json.dump(dict(workbook=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_sha256=src_hash,source_workbook=str(source),checks=len(checks),original_cells_preserved=n,utc_repairs=len(repairs),restored_T0_cells=6,not_stored_DA_cells=3,appended_rows=len(P['rows'])),f,indent=2)
print('Workbook verified',len(checks),'checks;',n,'original cells;',len(repairs),'UTC strings restored')
