"""Retrieve block-length labels from structured rows; no statistics or model evaluation."""
import sys,zipfile,xml.etree.ElementTree as E,csv,re,posixpath
from pathlib import Path
sys.dont_write_bytecode=True
P=Path('revision_work/outputs/step23b/Supplementary_Tables_S1-S18_v2.xlsx')
if not P.exists():P=Path('revision_work/outputs/step23/Supplementary_Tables_S1-S18.xlsx')
N={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};rows=[]
with zipfile.ZipFile(P) as z:
 ss=[''.join(n.itertext()) for n in E.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml'in z.namelist() else []
 rel={n.attrib['Id']:n.attrib['Target'] for n in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
 for sh in E.fromstring(z.read('xl/workbook.xml')).find('s:sheets',N):
  t=rel[sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']];t=t.lstrip('/') if t.startswith('/') else posixpath.normpath('xl/'+t);header={};block=''
  for row in E.fromstring(z.read(t)).findall('.//s:sheetData/s:row',N):
   v={}
   for c in row:
    a=c.find('s:is',N);b=c.find('s:v',N);tx=''.join(a.itertext()) if a is not None else (b.text if b is not None else '')
    if c.attrib.get('t')=='s':tx=ss[int(tx)]
    if tx:v[re.sub(r'\d','',c.attrib['r'])]=str(tx)
   if len(v)==1 and re.match(r'^[A-Z]\.\s',next(iter(v.values()))):block=next(iter(v.values()));header={}
   low={k:tx.strip().lower().replace('_',' ') for k,tx in v.items()}
   if any(tx in ['l','block length'] for tx in low.values()) and any(tx in ['h','horizon'] for tx in low.values()):header=low;continue
   if not header:continue
   d={header.get(k,k):tx for k,tx in v.items()}
   h=d.get('h',d.get('horizon',''));L=d.get('l',d.get('block length',''))
   if not(re.fullmatch(r'\d+d?',h) and re.fullmatch(r'\d+',L)):continue
   rows.append(dict(sheet=sh.attrib['name'],row=row.attrib['r'],block=block,h_as_stored=h,L_as_stored=L,is_label_L_equals_2h=int(L)==2*int(h.rstrip('d')),input_set=d.get('input set',''),contrast=d.get('contrast',''),metric=d.get('metric',''),method=d.get('method',''),row_values=str(d)))
with Path('revision_work/outputs/step25/WORKBOOK_STRUCTURED_BLOCK_LABEL_ROWS.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['sheet','row','block','h_as_stored','L_as_stored','is_label_L_equals_2h']);w.writeheader();w.writerows(rows)
print('Retrieved structured rows with h/L labels:',len(rows),'L=2h label matches:',sum(r['is_label_L_equals_2h'] for r in rows))
print([r for r in rows if r['is_label_L_equals_2h']][:5])
