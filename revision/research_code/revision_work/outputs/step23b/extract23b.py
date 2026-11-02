"""Extract DOCX text/OMML and enumerate candidate numeric-bound sentences, read-only."""
import sys,re,json,hashlib,csv,zipfile
from pathlib import Path
from xml.etree import ElementTree as E
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step23b';OUT.mkdir(parents=True,exist_ok=True)
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
sources=[];paragraphs=[];candidates=[]
def txt(p):
 parts=[]
 for n in p.iter():
  if n.tag in ['{'+NS['w']+'}t','{'+NS['m']+'}t']:parts.append(n.text or '')
  elif n.tag=='{'+NS['w']+'}tab':parts.append('\t')
  elif n.tag=='{'+NS['w']+'}br':parts.append(' ')
 return ''.join(parts)
for key,name in [('manuscript','c_clean_revised_manuscript_without_highlighting.docx'),('response','a_point-by-point_response_to_the_reviewers_comments.docx')]:
 p=ROOT/'revision_work/inputs/step23b'/name;sources.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
 z=zipfile.ZipFile(p);tree=E.fromstring(z.read('word/document.xml'));section='Front matter';n=0;table=0;lines=[]
 for child in tree.find('w:body',NS):
  if child.tag=='{'+NS['w']+'}tbl':table+=1
  for para in ([child] if child.tag=='{'+NS['w']+'}p' else child.findall('.//w:p',NS) if child.tag=='{'+NS['w']+'}tbl' else []):
   t=txt(para).strip()
   if not t:continue
   n+=1;style=para.find('w:pPr/w:pStyle',NS);style=style.attrib.get('{'+NS['w']+'}val','') if style is not None else ''
   if (re.match(r'^(?:[1-9](?:\.[0-9]){0,2}\.?\s+[A-Za-z]|Reviewer\s*[#:]?\s*\d|Comment\s*[#:]?\s*\d|Response\s*\d)',t,re.I) and len(t)<240) or style.lower().startswith('heading'):section=t
   loc=f'{key} paragraph {n}; {section}'+(f';table {table}' if child.tag=='{'+NS['w']+'}tbl' else '')
   rec=dict(document=key,paragraph=n,location=loc,style=style,in_table=child.tag=='{'+NS['w']+'}tbl',text=t);paragraphs.append(rec);lines.append(f'P{n:04d} [{section}] {t}')
   sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z“\[])',t)
   for si,s in enumerate(sentences,1):
    if not re.search(r'\d',s):continue
    patterns=[]
    if re.search(r'(?:[<>≤≥]|\\(?:leq|geq|le|ge))\s*[−\-+]?\s*\d|\bp\s*[<>≤≥]',s,re.I):patterns.append('inequality')
    if re.search(r'\b(?:at most|at least|below|above|up to|less than|more than|no more than|within)\b',s,re.I):patterns.append('verbal bound')
    if re.search(r'[−\-+]?\d+(?:\.\d+)?\s*(?:%|pp)?\s*[–—−-]\s*[+−\-]?\d+(?:\.\d+)?|\bbetween\s+\d|\bfrom\s+[−\-+]?\d+.*?\bto\s+[−\-+]?\d|\d+(?:\.\d+)?\s*to\s*[−\-+]?\d',s,re.I):patterns.append('range')
    if '≈' in s or re.search(r'\bapproximately\b',s,re.I):patterns.append('approximation')
    if patterns:candidates.append(dict(candidate_id=f'{key[0].upper()}{n:04d}.{si}',**rec,sentence=s,patterns=';'.join(patterns)))
 with (OUT/(key+'_text.txt')).open('x') as h:h.write('\n'.join(lines))
for name,rows in [('SOURCE_DOCUMENTS.csv',sources),('PARAGRAPHS.csv',paragraphs),('BOUND_CANDIDATES.csv',candidates)]:
 with (OUT/name).open('x',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'EXTRACTED_TEXT.json').open('x') as h:json.dump(dict(paragraphs=paragraphs,candidates=candidates),h,ensure_ascii=False)
print('Paragraphs:',len(paragraphs),'candidate sentences:',len(candidates))
for r in candidates:print(r['candidate_id'],r['location'][:90],r['sentence'])
