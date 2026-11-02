"""Read-only OOXML extraction with stable body paragraph/table/cell locators."""
from pathlib import Path
import zipfile,xml.etree.ElementTree as ET,json,csv,re,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
records=[];candidates=[]
for path in sorted((ROOT/'revision_work/inputs/step26').glob('*.docx')):
 with zipfile.ZipFile(path) as z:
  for part in ['word/document.xml','word/footnotes.xml','word/endnotes.xml']:
   if part not in z.namelist():continue
   node=ET.fromstring(z.read(part));body=node.find('w:body',NS) if part.endswith('document.xml') else node
   pi=ti=0;section='';table_caption=''
   for child in body:
    kind=child.tag.rsplit('}',1)[-1]
    if kind=='p':
     pi+=1;text=''.join(child.itertext()) if False else ''.join(n.text or '' for n in child.findall('.//w:t',NS));style=child.find('w:pPr/w:pStyle',NS);style=style.attrib.get('{'+NS['w']+'}val','') if style is not None else ''
     if style.lower().startswith('heading'):section=text
     if re.match(r'^(Table|Supplementary Table|Figure)\s',text):table_caption=text
     row=dict(file=str(path.relative_to(ROOT)),part=part,location=f'paragraph {pi}',section=section,text=text,style=style,has_drawing=child.find('.//w:drawing',NS) is not None);records.append(row)
     if re.search(r'ridge|480|7 alphas|holm|figure 3',text,re.I):candidates.append(row)
    elif kind=='tbl':
     ti+=1;rr=[]
     for ri,tr in enumerate(child.findall('w:tr',NS),1):
      for ci,tc in enumerate(tr.findall('w:tc',NS),1):
       text='\n'.join(''.join(n.text or '' for n in p.findall('.//w:t',NS)) for p in tc.findall('w:p',NS));row=dict(file=str(path.relative_to(ROOT)),part=part,location=f'table {ti} row {ri} cell {ci}',section=section,table_caption=table_caption,text=text,style='table',has_drawing=tc.find('.//w:drawing',NS) is not None);records.append(row);rr.append(row)
     if any(re.search(r'ridge',x['text'],re.I) for x in rr):candidates.extend(rr)
with (OUT/'DOCUMENT_EXTRACTION_STEP26.json').open('x') as f:json.dump(records,f,indent=2,ensure_ascii=False)
with (OUT/'RIDGE_DOCUMENT_CANDIDATES_STEP26.json').open('x') as f:json.dump(candidates,f,indent=2,ensure_ascii=False)
print('records',len(records),'candidates',len(candidates))
for r in candidates:
 if r['style']!='table':print(Path(r['file']).name,r['location'],r['text'])
