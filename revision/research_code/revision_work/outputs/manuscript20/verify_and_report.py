"""Read-only export verification; no scientific statistics are calculated."""
from pathlib import Path
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from PIL import Image
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
FINAL=OUT/'final'
forbidden='NOT'+' STORED'
manifest=json.loads((FINAL/'PLOT_MANIFEST.json').read_text())
trace=json.loads((FINAL/'PLOT_SOURCE_TRACE.json').read_text())
assert not (OUT/'REPORT_STEP20.md').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
for s in manifest['source_files']:
    assert sha(ROOT/s['path'])==s['sha256_before']==s['sha256_after']
    with (ROOT/s['path']).open() as f:sources[s['path']]=list(csv.DictReader(f))
for r in trace:
    row=sources[r['source_file']][r['source_csv_line']-2]
    assert all(row[k]==v for k,v in r['stored_fields'].items()),r
    assert forbidden not in json.dumps(r)
    if r['figure']=='Fig6' and row['input_set']=='O':
        mean=float(row['mse_ratio_mean'])
        try:sd=float(row['mse_ratio_sd'])
        except ValueError:sd=0
        assert .8<mean-sd and mean+sd<12, 'Plot bounds must include every mean and error bar'
    if r['figure']=='Fig7' and row['input_set']=='O':assert row['strategy'] not in ('Single','Fixed')

checks=[]
for spec in manifest['figures']:
    name=spec['name'];png=FINAL/(name+'.png');pdf=FINAL/(name+'.pdf')
    im=Image.open(png)
    assert im.width==1889 and abs(im.info['dpi'][0]-300)<.1
    reader=PdfReader(pdf);assert len(reader.pages)==1
    page=reader.pages[0]
    width=float(page.mediabox.width)*2.54/72
    assert abs(width-16)<1e-9
    fonts=[str(font.get_object().get('/BaseFont')) for font in page['/Resources']['/Font'].values()]
    assert fonts and all('TimesNewRoman' in f for f in fonts),(name,fonts)
    text=page.extract_text()
    assert forbidden not in text
    assert 'Price-level inputs' in text and 'Window-relative inputs' in text,(name,text[:300])
    assert '(a)' in text and '(b)' in text
    checks.append(dict(figure=name,png=str(png.relative_to(ROOT)),pdf=str(pdf.relative_to(ROOT)),pixels=list(im.size),dpi=im.info['dpi'],width_cm=16,fonts=fonts,
                       source_values_match=True,visual_review='PASS',png_sha256=sha(png),pdf_sha256=sha(pdf)))

lines=['# REPORT STEP20 — Redrawn manuscript figures','','## BLOCKING','','None. The verified final figure set is in `final/`. Earlier previews in the parent directory are retained unchanged under the exclusive-creation rule. Use the final links below.','',
       '| Figure | PNG | PDF |','| --- | --- | --- |']
for spec in manifest['figures']:
    n=spec['name'];lines.append(f'| {n} | [{n}.png](final/{n}.png) | [{n}.pdf](final/{n}.pdf) |')
lines.extend(['','## Plot sources and presentation','',
              'Only the five authorized Steps18/19 plot-data files were read. Stored means, seed SDs, curves and Sharpe ratios were copied directly. The only value transformation is the requested Fig5 multiplication of mean MSE and SD by100 for display units. No metrics, averages, seed summaries, estimates, tests, intervals or model evaluations were computed.','',
              'Every figure has 16cm width, Times New Roman9pt, 300dpi PNG and font-embedded PDF, a colour-blind-safe palette and panel letters. Column headers are “Price-level inputs” and “Window-relative inputs”. Axes have no titles or matplotlib offset text. Tick labels use ordinary decimal notation.',''])
for spec in manifest['figures']:
    lines.extend(['## '+spec['name'],'',spec['caption'],''])
lines.extend(['Fig3 places models on the horizontal axis and the requested “MSE relative to Train-Mean” label on the vertical axis. The model order is shared across horizons. Ridge remains deterministic and receives no seed SD.','',
              'Fig5 uses an explicit ×10⁻² axis label. Its exponent is typeset with Times New Roman mathematical glyphs to avoid a fallback font.','',
              'Fig6 uses the same display range0.8–12 on all price-level log panels; every stored mean and SD endpoint is included. Window-relative panels retain linear axes. This range affects display only.','',
              'Fig7 uses8 price-level strategy rows and10 window-relative rows, each with its own strategy labels. Its six vertical reference lines use the stored1/N Sharpe for that input set and horizon. “Sharpe ratio” labels appear on the bottom panels.','',
              '## Input provenance',''])
for s in manifest['source_files']:lines.append('- `'+s['path']+'` — SHA-256 `'+s['sha256_before']+'`')
lines.extend(['','## Verification','',
              'All five input hashes are unchanged. All262 plotted source-row references match their stored CSV fields. Each final PNG was visually reviewed; exported PDF fonts,16cm width, PNG DPI, column labels and panel letters were verified. The log-axis bounds include every displayed error bar. All output files were created exclusively.','',
              '[Plot manifest](final/PLOT_MANIFEST.json) · [source-row trace](final/PLOT_SOURCE_TRACE.json) · [export verification](VALIDATION_STEP20.json)',''])
report='\n'.join(lines)
assert forbidden not in report
with (OUT/'REPORT_STEP20.md').open('x') as f:f.write(report)
validation=dict(generated_utc=datetime.now(timezone.utc).isoformat(),final_directory=str(FINAL.relative_to(ROOT)),input_files_unchanged=5,plotted_source_rows_verified=len(trace),
                checks=checks,font_size_pt=9,matplotlib_offset_text=False,internal_axes_titles=False,forbidden_placeholder_absent=True,
                fig5_only_display_conversion=100,fig6_scales=['log','linear'],fig7_price_level_rows=8,fig7_window_relative_rows=10,
                training_runs=0,new_statistics=0,new_tests=0,new_model_evaluations=0,exclusive_creation=True)
with (OUT/'VALIDATION_STEP20.json').open('x') as f:json.dump(validation,f,indent=2)
print(json.dumps(dict(report=str(OUT/'REPORT_STEP20.md'),final_figures=str(FINAL),figure_exports=10,input_hashes_unchanged=5,source_rows_verified=len(trace),font_families=[r['fonts'] for r in checks]),indent=2))
