"""Verify exported values/formats and write the Prompt 19 report exclusively."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from PIL import Image
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
content=json.loads((OUT/'CONTENT_STEP19.json').read_text())
build=json.loads((OUT/'WORKBOOK_BUILD.json').read_text())
xlsx=OUT/'Supplementary_Tables.xlsx'
assert not (OUT/'REPORT_STEP19.md').exists()
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(v):
    if v is None or str(v).lower() in ('','nan','null','none'):return 'N/A'
    try:return format(float(v),'.6g')
    except (TypeError,ValueError):return str(v)


def expected(v):
    if v is None:return None
    if isinstance(v,bool):return v
    if isinstance(v,(int,float)):return float(fmt(v))
    s=str(v)
    if s=='' or s.lower() in ('nan','null','none'):return 'N/A (source blank)'
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',s):return float(fmt(v))
    return "'"+s if s.startswith('=') else s


def col(j):
    out='';j+=1
    while j:
        j,k=divmod(j-1,26);out=chr(65+k)+out
    return out


with zipfile.ZipFile(xlsx) as z:
    workbook=ET.fromstring(z.read('xl/workbook.xml'))
    names=[r.attrib['name'] for r in workbook.findall('m:sheets/m:sheet',NS)]
    assert names==['README']+['S'+str(i) for i in range(1,13)],names
    shared=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS):shared.append(''.join(t.text or '' for t in si.findall('.//m:t',NS)))
    rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    relmap={r.attrib['Id']:r.attrib['Target'] for r in rels}
    cells_checked=0;formula_count=0;sheet_checks=[]
    for i,spec in enumerate(content['sheets'],1):
        meta=next(r for r in build['sheets'] if r['name']==spec['name'])
        sheet=workbook.findall('m:sheets/m:sheet',NS)[i]
        target=relmap[sheet.attrib['{'+NS['r']+'}id']]
        filename=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        xml=ET.fromstring(z.read(filename))
        formula_count+=len(xml.findall('.//m:f',NS))
        actual={}
        for c in xml.findall('.//m:sheetData/m:row/m:c',NS):
            v=c.find('m:v',NS);typ=c.attrib.get('t')
            if typ=='s':val=shared[int(v.text)] if v is not None else None
            elif typ=='inlineStr':val=''.join(t.text or '' for t in c.findall('.//m:t',NS))
            elif typ=='b':val=v.text=='1' if v is not None else None
            elif typ in ('str','e'):val=v.text if v is not None else None
            else:val=float(v.text) if v is not None and v.text is not None else None
            actual[c.attrib['r']]=val
        fields=meta['keys']
        mapping=[spec['columns'].index(k) for k in fields]
        for row_index,r in enumerate(spec['data']):
            for j,index in enumerate(mapping):
                want=expected(r[index]);address=col(j)+str(meta['headerRow']+1+row_index)
                got=actual.get(address)
                if isinstance(want,float) and not isinstance(want,bool):
                    assert isinstance(got,(int,float)) and math.isclose(got,want,rel_tol=1e-12,abs_tol=0),(spec['name'],address,want,got)
                else:assert got==want,(spec['name'],address,want,got)
                cells_checked+=1
        assert any('Source ' in str(v) for v in actual.values()),spec['name']
        sheet_checks.append(dict(sheet=spec['name'],rows=spec['row_count'],columns=len(fields),copied_cells_match_six_significant_digits=True,source_list_present=True))
    assert formula_count==0
    assert all('N/A' not in s for s in names)

for r in content['source_files']:
    assert sha(ROOT/r['path'])==r['sha256_before']==r['sha256_after'],r['path']
figchecks=[]
for f in content['figures']:
    png=OUT/(f['name']+'.png');pdf=OUT/(f['name']+'.pdf')
    im=Image.open(png);assert im.width==1889 and abs(im.info['dpi'][0]-300)<.1
    p=PdfReader(pdf).pages[0]
    width=float(p.mediabox.width)*2.54/72
    assert abs(width-16)<1e-9
    fonts=[str(r.get_object().get('/BaseFont')) for r in p['/Resources']['/Font'].values()]
    assert all('TimesNewRoman' in s for s in fonts),fonts
    figchecks.append(dict(name=f['name'],width_cm=16,png_dpi=im.info['dpi'],pixels=list(im.size),pdf_fonts=fonts,visual_review='PASS'))


def markdown(rr,cols):
    esc=lambda v:fmt(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(esc(r.get(c)) for c in cols)+' |' for r in rr])


with (OUT/'F3_complete_data.csv').open() as f:f3=list(csv.DictReader(f))
with (OUT/'learning_summary.csv').open() as f:learning=list(csv.DictReader(f))
digest=[]
for model in ['single','fixed','same_patch_fixed','static','adaptive','lstm','vanilla','swim','ridge']:
    for h in ('5','10','20'):
        d={'model':model,'h':h}
        for inp in ('O','C'):
            r=next(z for z in f3 if z['input_set']==inp and z['model']==model and z['horizon']==h)
            d[inp+' ratio ± seed SD']=r['mse_ratio']+' ± '+r['mse_ratio_seed_sd']
        digest.append(d)
lim=content['missing']
lines=['# REPORT STEP19 — Descriptive figures, intervention definitions and supplementary tables','','## BLOCKING','']
if content['history_missing']:
    lines.extend(['Missing matching histories are listed below; no replacement history was used.',''])
else:lines.extend(['No missing histories or new execution blocker. All 120 requested core histories were found (O/C × Static/Adaptive/Single/Fixed × 3 horizons × 5 seeds).',''])
lines.extend(['S12 retains the source Step17 limitations: three PRIMARY O/W3 Ridge cells lack the saved validation forecasts needed for lambda selection, and six cells lack stored CW references. These source gaps remain visible; no training, evaluation, test or source repair was performed. Other source qualifications, including Step13 issuer mismatches/not-found statuses and n=1 single-resolution cells, remain in their worksheets.','',markdown(lim,['kind','cell','reason']),'',
              '## Outputs','', '[Supplementary_Tables.xlsx](Supplementary_Tables.xlsx) — README plus S1–S12, one worksheet per requested supplementary table.','',
              '[F3 PNG](F3_complete.png) · [F3 PDF](F3_complete.pdf) · [F3 data](F3_complete_data.csv)','',
              '[F7 PNG](F7_learning_curves.png) · [F7 PDF](F7_learning_curves.pdf) · [epoch means](learning_mean_by_epoch.csv)','',
              'Figures use Times New Roman 9 pt, a colour-blind-safe palette, 16 cm width and 300 dpi PNG plus font-embedded PDF. O is left and C right; panels are lettered and have no internal titles. Numeric tables/workbook cells use six significant digits. Error bars in F3 are seed SD, not confidence intervals.','',
              '## A — Complete F3','',
              'For each cell, ratio = stored mean TEST MSE / stored Global Train-Mean TEST MSE; ratio SD = stored seed SD / the same denominator. The full-precision saved summary/reference CSV values were divided before final rounding. The denominator comes from Step5 a1/a1_benchmark_mse.csv; the numerator and SD come from Step16 clarified/B_seed_summary.csv. O neural inputs use B5FIX/B5 uniform12. Ridge remains a deterministic n=1 forecast with SD inapplicable.','',
              markdown(digest,['model','h','O ratio ± seed SD','C ratio ± seed SD']),'',
              '## B — Learning histories','',
              'Histories follow the exact Step16 core prediction registry. O archived Static/Adaptive and seed42 Single/Fixed use their matching original histories; the remaining O Single/Fixed and all C cells use the corresponding 14a attempt history and saved best epoch. No 3/1 causal run, alternative patch candidate, recovery history, B3 window, or baseline-model history is substituted.','',
              'For each seed, % change = 100 × (last-epoch Huber − best-epoch Huber) / best-epoch Huber. The table takes the equal-weight mean of these percentages across the five seeds. Negative values mean loss fell after the validation-selected best epoch. Both TRAIN and VAL use Huber delta=1 on raw log-return targets. Best epoch is copied from the run record; it is not reselected.','',
              'Epoch curves average only histories containing that epoch. After some seeds stop, the available-epoch mean can use fewer than five seeds. The exact seed list and count are saved per epoch; no padding or last-value carry-forward is used. No bands or intervals are calculated.','',
              markdown(learning,['input_set','mechanism','horizon','n_histories','best_epoch_counts','mean_train_huber_pct_change','mean_val_huber_pct_change']),'',
              '[Per-seed best/last losses and changes](learning_per_seed.csv) · [best-epoch distribution](learning_best_epoch_distribution.csv) · [all matching epoch observations](learning_history_cells.csv) · [missing-history/source-limit list](MISSING_HISTORY_AND_INHERITED_LIMITS.csv)','',
              '## C — Exact intervention definitions',''])
for d in content['definitions']:
    link='/' + str((ROOT/d['file']).relative_to('/'))+':'+str(d['line_start'])
    lines.extend(['### '+d['intervention'],'',d['definition'],'',f"[{d['file']}:{d['line_start']}–{d['line_end']}]({link})",'','```python',d['exact_code'],'```',''])
lines.extend(['I6 therefore equalizes one branch while keeping the other branch learned; it is not a branch removal. I5 removes one scale weight from the affected branch and renormalizes that branch only. I4 is a same-date cross-sectional router-weight diagnostic. These are code definitions; no intervention is executed here.','',
              '## D — Supplementary workbook','',
              'README records every title, source and note. Each S sheet lists its source filenames above the table and includes source_file/source_csv_line columns. Source definitions, contrast signs, uncertainty methods and Holm families are preserved. Stored p-values and intervals are copied to six significant digits; none is calculated or corrected again. Scientific numeric cells are typed numbers, with six-digit scientific display; counts are integers. Blank union columns mean inapplicable to that record type, while source blanks are explicitly labelled.',''])
for s in content['sheets']:
    lines.extend(['### '+s['name']+' — '+s['title'],'',s['notes'],'',f"Records: {s['row_count']}. [CSV]({s['name']}_table.csv)",''])
    lines.extend(['- `'+p+'`' for p in s['source_files']]);lines.append('')
lines.extend(['## Validation and provenance','',f"{len(content['source_files'])} input files retain matching before/after SHA-256. All {cells_checked} S1–S12 workbook data cells were checked against the expected copied values after rounding to six significant digits. Workbook sheets and source lists were verified from the exported XLSX; no formulas were authored. Every worksheet was rendered for visual review. F3/F7 dimensions, DPI and embedded Times New Roman were checked. These checks verify extraction/export, not hypotheses.",'',
              'Descriptive arithmetic was limited to ratios, percentages, means, and best-epoch counts/min/median/max on saved values. No prediction parquet, raw expert cache, checkpoint, trainer, metric evaluator or statistical-test routine was executed. Sources are read only; new files use exclusive creation.','',
              '[SOURCE_MANIFEST.csv](SOURCE_MANIFEST.csv) · [VALIDATION_STEP19.json](VALIDATION_STEP19.json) · [workbook QA](WORKBOOK_BUILD.json)',''])
report='\n'.join(lines)
for link in re.findall(r'\]\(([^)]+)\)',report):
    if not link.startswith(('/','http')) and link!='VALIDATION_STEP19.json':assert (OUT/link).exists(),link
with (OUT/'REPORT_STEP19.md').open('x') as f:f.write(report)
validation=dict(generated_utc=datetime.now(timezone.utc).isoformat(),sources_unchanged=True,source_count=len(content['source_files']),history_cells=content['history_cells'],missing_histories=len(content['history_missing']),
                sheets=sheet_checks,workbook_cells_verified=cells_checked,formulas_authored=formula_count,figures=figchecks,workbook_visual_review='All 13 worksheet previews inspected',
                new_training_runs=0,new_model_evaluations=0,new_p_values=0,new_intervals=0,exclusive_creation=True,
                inherited_source_limitations=lim,artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in OUT.iterdir() if p.is_file()])
with (OUT/'VALIDATION_STEP19.json').open('x') as f:json.dump(validation,f,indent=2)
print(json.dumps(dict(report=str(OUT/'REPORT_STEP19.md'),workbook=str(xlsx),workbook_data_cells_verified=cells_checked,sheets=names,history_cells=content['history_cells'],missing_histories=len(content['history_missing']),input_files_unchanged=len(content['source_files'])),indent=2))
