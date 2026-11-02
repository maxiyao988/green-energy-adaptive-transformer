"""Read XLSX XML only, no Excel engine, formula evaluation, or statistics."""
import sys,zipfile,xml.etree.ElementTree as E,re,csv,json,hashlib,posixpath
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step25';p=ROOT/'revision_work/outputs/step23b/Supplementary_Tables_S1-S18_v2.xlsx';fallback=False
if not p.exists():p=ROOT/'revision_work/outputs/step23/Supplementary_Tables_S1-S18.xlsx';fallback=True
before=hashlib.sha256(p.read_bytes()).hexdigest();NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};RID='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
terms=['HAC','Bartlett','Diebold','DM','2h','L=2h','L = 2h','block length','sensitivity'];hits=[];coverage=[];contexts=[]
nw=re.compile(r'newey[\s–−-]*west|\bNW\b|\bnw_|_nw\b|\bnw\s+(?:se|t|p|lag)',re.I)
with zipfile.ZipFile(p) as z:
 ss=[]
 if 'xl/sharedStrings.xml' in z.namelist():ss=[''.join(t.text or '' for t in n.iter('{'+NS['s']+'}t')) for n in E.fromstring(z.read('xl/sharedStrings.xml'))]
 style=E.fromstring(z.read('xl/styles.xml'));fonts=style.find('s:fonts',NS);xf=style.find('s:cellXfs',NS);bold={i:fonts[int(n.attrib.get('fontId',0))].find('s:b',NS) is not None for i,n in enumerate(xf)}
 rel={n.attrib['Id']:n.attrib['Target'] for n in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
 for sh in E.fromstring(z.read('xl/workbook.xml')).find('s:sheets',NS):
  name=sh.attrib['name'];target=rel[sh.attrib[RID]];target=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target);sheet=E.fromstring(z.read(target));dim=sheet.find('s:dimension',NS);rowmaps=[];stored=0;nonempty=0;count=0
  for row in sheet.findall('.//s:sheetData/s:row',NS):
   values={};sty={}
   for cell in row.findall('s:c',NS):
    stored+=1;coord=cell.attrib['r'];v=cell.find('s:v',NS);inline=cell.find('s:is',NS);f=cell.find('s:f',NS);t=cell.attrib.get('t');value=''
    if inline is not None:value=''.join(x.text or '' for x in inline.iter('{'+NS['s']+'}t'))
    elif v is not None and v.text is not None:value=ss[int(v.text)] if t=='s' else v.text
    if f is not None:value='FORMULA: '+(f.text or '')+'; CACHED: '+value
    if value!='':values[coord]=value;sty[coord]=int(cell.attrib.get('s',0));nonempty+=1
   rowmaps.append((row.attrib['r'],values,sty))
  header={};block='';source='';sheet_hits=[]
  for rn,values,sty in rowmaps:
   vals=list(values.items());isheader=len(vals)>=2 and sum(bold.get(sty[c],False) for c,v in vals)>=len(vals)/2
   if len(vals)==1:
    tx=vals[0][1]
    if tx.startswith('Source:'):source=tx
    elif re.match(r'^[A-Z](?:\.|\s*[—–-])\s',tx) or tx.startswith('Table S'):block=tx;header={};source=''
   if isheader:header={re.sub(r'\d','',c):v for c,v in vals}
   label=[]
   for c,v in vals:
    h=header.get(re.sub(r'\d','',c),'').lower().replace('_',' ')
    if h in ['input set','input','model','mechanism','window','contrast','metric','h','horizon','seed','setting','backbone','method','specification','scope','branch','group','section','regime','config','configuration','frequency','frequency setting']:label.append(f'{h}={v}')
   rowlabel='; '.join(label) if label and not isheader else ' | '.join(v[:180] for c,v in vals[:3])
   if any(v.lower().startswith('source:') for c,v in vals):source=next(v for c,v in vals if v.lower().startswith('source:'))
   for coord,value in vals:
    matched=[t for t in terms if t.casefold() in value.casefold()];nwm=nw.findall(value)
    if not matched and not nwm:continue
    if nwm:matched+=['Newey–West / NW (additional search)']
    termsre=[re.search(re.escape(t),value,re.I) for t in terms if t.casefold() in value.casefold()];start=min([m.start() for m in termsre if m]+[0 if nwm else len(value)]);snippet=value[max(0,start-140):start+500]
    hit=dict(sheet=name,cell=coord,row_label=rowlabel,column_header=header.get(re.sub(r'\d','',coord),'(title/note or no identified local header)'),matched_terms='; '.join(matched),block=block,source_cell_text=source,cell_text=value,snippet=snippet,row_context=json.dumps(values,ensure_ascii=False),cell_is_header=isheader,standalone_DM=bool(re.search(r'\bDM\b',value,re.I)),sheet_state=sh.attrib.get('state','visible'))
    hits.append(hit);sheet_hits.append(hit);count+=1
  coverage.append(dict(sheet=name,state=sh.attrib.get('state','visible'),dimension=dim.attrib.get('ref','') if dim is not None else '',stored_cells_scanned=stored,nonempty_cells_scanned=nonempty,hits=count,xml_part=target))
  for rn,values,sty in rowmaps:
   if any(h['cell'] in values for h in sheet_hits) or (name in ['Table S5','Table S14'] and int(rn)<8):contexts.append(dict(sheet=name,row=rn,cells=values))
after=hashlib.sha256(p.read_bytes()).hexdigest();assert before==after
def save(name,rows):
 with (OUT/name).open('x',newline='') as f:
  fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
save('WORKBOOK_CELL_HITS_RAW.csv',hits);save('WORKBOOK_SHEET_COVERAGE.csv',coverage)
with (OUT/'WORKBOOK_HIT_ROW_CONTEXT.json').open('x') as f:json.dump(contexts,f,ensure_ascii=False,indent=2)
with (OUT/'WORKBOOK_SEARCH_MANIFEST.json').open('x') as f:json.dump(dict(workbook=str(p.relative_to(ROOT)),fallback=fallback,sha256_before=before,sha256_after=after,terms=terms,additional_search=nw.pattern,matching='case-insensitive literal substrings, including DM substrings; standalone_DM recorded separately; every stored XML cell in every sheet including hidden sheets; formulas and cached values searched without evaluating',blank_cells='All stored blank/style cells visited; unstored blank cells contain no searchable text',sheet_count=len(coverage),hit_count=len(hits),cells_scanned=sum(r['stored_cells_scanned'] for r in coverage)),f,ensure_ascii=False,indent=2)
print('Workbook',p,'sheets',len(coverage),'hits',len(hits),'cells',sum(r['stored_cells_scanned'] for r in coverage))
from collections import Counter
print('Term counts',Counter(t for h in hits for t in h['matched_terms'].split('; ')))
for h in hits:
 if any(t in h['matched_terms'] for t in ['HAC','Bartlett','Diebold','2h','block length']) or h['standalone_DM']:print({k:h[k] for k in ['sheet','cell','column_header','block','snippet']})
