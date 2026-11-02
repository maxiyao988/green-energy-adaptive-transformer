"""Append artifact-tool-authored S13 without rewriting the original worksheet parts."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SPEC = json.loads((HERE/'S13_CONTENT.json').read_text())
QA = json.loads((HERE/'export_attempt2/S13_INSPECTION.json').read_text())
BASE = ROOT/SPEC['source_workbook']
DEST = HERE/'Supplementary_Tables_with_S13.xlsx'
PART = HERE/'export_attempt2/S13_sheet_package.bin'
S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
P = 'http://schemas.openxmlformats.org/package/2006/relationships'
C = 'http://schemas.openxmlformats.org/package/2006/content-types'
N = {'s': S}
E.register_namespace('r', R)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tag(name):
    return '{'+S+'}'+name


def xml(root, namespace):
    E.register_namespace('', namespace)
    return E.tostring(root, encoding='utf-8', xml_declaration=True)


def col(n):
    s = ''
    while n:
        n, r = divmod(n-1, 26)
        s = chr(65+r)+s
    return s


assert sha(BASE) == SPEC['source_workbook_sha256']
for section in SPEC['sections']:
    assert sha(ROOT/section['file']) == section['sha256']
assert not DEST.exists()

with zipfile.ZipFile(BASE) as original, zipfile.ZipFile(PART) as authored:
    base_parts = {name: original.read(name) for name in original.namelist()}
    styles = E.fromstring(base_parts['xl/styles.xml'])
    new_styles = E.fromstring(authored.read('xl/styles.xml'))
    preserved_styles = copy.deepcopy(styles)
    maps = {}

    # Add new style records after every original record so old style IDs stay unchanged.
    numfmts = styles.find('s:numFmts', N)
    new_numfmts = new_styles.find('s:numFmts', N)
    custom_map = {}
    if new_numfmts is not None and len(new_numfmts):
        if numfmts is None:
            numfmts = E.Element(tag('numFmts'), {'count':'0'})
            styles.insert(0, numfmts)
        bycode = {v.attrib['formatCode']: int(v.attrib['numFmtId']) for v in numfmts}
        next_id = max([163]+[int(v.attrib['numFmtId']) for v in numfmts])+1
        for value in new_numfmts:
            code = value.attrib['formatCode']
            if code not in bycode:
                item = copy.deepcopy(value)
                item.set('numFmtId', str(next_id))
                numfmts.append(item)
                bycode[code] = next_id
                next_id += 1
            custom_map[int(value.attrib['numFmtId'])] = bycode[code]
        numfmts.set('count', str(len(numfmts)))

    for collection in ['fonts', 'fills', 'borders']:
        dest = styles.find('s:'+collection, N)
        src = new_styles.find('s:'+collection, N)
        assert dest is not None and src is not None
        mapping = {}
        for i, item in enumerate(src):
            mapping[i] = len(dest)
            dest.append(copy.deepcopy(item))
        dest.set('count', str(len(dest)))
        maps[collection] = mapping

    for collection in ['cellStyleXfs', 'cellXfs']:
        dest = styles.find('s:'+collection, N)
        src = new_styles.find('s:'+collection, N)
        assert dest is not None and src is not None
        mapping = {}
        for i, source in enumerate(src):
            item = copy.deepcopy(source)
            for attr, group in [('fontId','fonts'), ('fillId','fills'), ('borderId','borders')]:
                if attr in item.attrib:
                    item.set(attr, str(maps[group][int(item.attrib[attr])]))
            fmt = int(item.get('numFmtId', '0'))
            if fmt in custom_map:
                item.set('numFmtId', str(custom_map[fmt]))
            if collection == 'cellXfs' and 'xfId' in item.attrib:
                item.set('xfId', str(maps['cellStyleXfs'][int(item.attrib['xfId'])]))
            mapping[i] = len(dest)
            dest.append(item)
        dest.set('count', str(len(dest)))
        maps[collection] = mapping

    # Confirm that every original indexed style and named-style record is untouched.
    for source_collection in preserved_styles:
        name = source_collection.tag.rsplit('}',1)[1]
        target_collection = styles.find('s:'+name, N)
        assert target_collection is not None
        for old, new in zip(source_collection, target_collection):
            assert E.tostring(old) == E.tostring(new), name
        assert len(target_collection) >= len(source_collection)

    sheet = E.fromstring(authored.read('xl/worksheets/sheet1.xml'))
    assert 'xl/worksheets/_rels/sheet1.xml.rels' not in authored.namelist(), 'Unexpected S13 objects'
    strings = []
    if 'xl/sharedStrings.xml' in authored.namelist():
        strings = list(E.fromstring(authored.read('xl/sharedStrings.xml')))
    for cell in sheet.findall('.//s:c', N):
        if 's' in cell.attrib:
            cell.set('s', str(maps['cellXfs'][int(cell.attrib['s'])]))
        if cell.get('t') == 's':
            value = cell.find('s:v', N)
            old = strings[int(value.text)]
            cell.remove(value)
            cell.set('t', 'inlineStr')
            inline = E.SubElement(cell, tag('is'))
            for child in old:
                inline.append(copy.deepcopy(child))
    cells = {c.attrib['r']: c for c in sheet.findall('.//s:c', N)}
    for item in QA['numericLexemes']:
        address = col(item['col'])+str(item['row'])
        cell = cells[address]
        assert cell.get('t', 'n') in ['n', '']
        value = cell.find('s:v', N)
        assert value is not None
        assert float(value.text) == float(item['lexeme'])
        # Preserve the CSV's original decimal text, without rounding its stored value.
        value.text = item['lexeme']
    assert not sheet.findall('.//s:f', N)
    for view in sheet.findall('s:sheetViews/s:sheetView', N):
        view.set('tabSelected', '0')

    workbook = E.fromstring(base_parts['xl/workbook.xml'])
    sheets = workbook.find('s:sheets', N)
    names = [v.attrib['name'] for v in sheets]
    assert names == ['README']+['S'+str(i) for i in range(1,13)]
    new_id = max(int(v.attrib['sheetId']) for v in sheets)+1
    name = f'xl/worksheets/sheet{new_id}.xml'
    assert name not in base_parts
    rels = E.fromstring(base_parts['xl/_rels/workbook.xml.rels'])
    rel_id = 'rId'+str(max(int(v.attrib['Id'][3:]) for v in rels)+1)
    E.SubElement(sheets, tag('sheet'), {'name':'S13', 'sheetId':str(new_id), '{'+R+'}id':rel_id})
    E.SubElement(rels, '{'+P+'}Relationship', {'Id':rel_id, 'Type':R+'/worksheet',
                                                  'Target':f'worksheets/sheet{new_id}.xml'})
    types = E.fromstring(base_parts['[Content_Types].xml'])
    E.SubElement(types, '{'+C+'}Override', {'PartName':'/'+name,
        'ContentType':'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'})
    updates = {'xl/styles.xml':xml(styles,S), name:xml(sheet,S),
               'xl/workbook.xml':xml(workbook,S),
               'xl/_rels/workbook.xml.rels':xml(rels,P), '[Content_Types].xml':xml(types,C)}
    with zipfile.ZipFile(DEST, 'x', compression=zipfile.ZIP_DEFLATED) as final:
        for info in original.infolist():
            final.writestr(info, updates.get(info.filename, base_parts[info.filename]))
        final.writestr(name, updates[name])

# Re-open the delivered package and check every new source value and requested format.
with zipfile.ZipFile(DEST) as final:
    differences = [p for p, data in base_parts.items() if final.read(p) != data]
    assert set(differences) == {'xl/styles.xml','xl/workbook.xml',
                                'xl/_rels/workbook.xml.rels','[Content_Types].xml'}
    assert all(final.read(f'xl/worksheets/sheet{i}.xml') == base_parts[f'xl/worksheets/sheet{i}.xml']
               for i in range(1,14))
    delivered = E.fromstring(final.read(name))
    c = {v.attrib['r']:v for v in delivered.findall('.//s:c',N)}
    sf = E.fromstring(final.read('xl/styles.xml'))
    xfs = sf.find('s:cellXfs',N)
    custom = {int(v.attrib['numFmtId']):v.attrib['formatCode'] for v in sf.find('s:numFmts',N)}
    builtins = {0:'General',1:'0',2:'0.00',9:'0%',10:'0.00%',11:'0.00E+00'}
    def content(cell):
        if cell.get('t') == 'inlineStr':
            return ''.join(cell.find('s:is',N).itertext())
        if cell.get('t') == 'b':
            return 'True' if cell.find('s:v',N).text == '1' else 'False'
        value = cell.find('s:v',N)
        return '' if value is None else value.text
    def number_format(cell):
        fmt = int(xfs[int(cell.attrib.get('s','0'))].attrib.get('numFmtId','0'))
        return custom.get(fmt,builtins.get(fmt,str(fmt)))
    assert content(c['A2']) == SPEC['title']
    assert content(c['A4']) == SPEC['note']
    checked = 0
    for section, block in zip(SPEC['sections'], QA['blocks']):
        assert content(c['A'+str(block['titleRow'])]) == section['title']
        for i, row in enumerate(section['data']):
            for j, value in enumerate(row):
                address = col(j+1)+str(block['dataRow']+i)
                actual = content(c[address]) if address in c else ''
                assert actual == value, (section['file'], address, actual, value)
                key = section['columns'][j]
                if value and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',value):
                    f = number_format(c[address])
                    if key in ['da_mean','da_sd','da_min','da_max','global_train_mean_da','always_up_da',
                               'positive_return_share','first_da_mean','first_da_sd','second_da_mean','second_da_sd']:
                        assert f == '0.0%', (key, f)
                    if key.endswith('_pp'):
                        assert f == '0.00', (key,f)
                    if key in ['centered_p_two_sided','p_holm_3h']:
                        assert f == '0.000', (key,f)
                checked += 1
    assert len(QA['blocks']) == 5 and sum(b['rows'] for b in QA['blocks']) == 171

assert sha(BASE) == SPEC['source_workbook_sha256']
for section in SPEC['sections']:
    assert sha(ROOT/section['file']) == section['sha256']
validation = dict(status='PASS',output=str(DEST.relative_to(ROOT)),sha256=sha(DEST),
    existing_workbook_unchanged=True,existing_csv_sources_unchanged=True,
    original_worksheet_parts_byte_identical=13,original_indexed_styles_preserved=True,
    new_sheet='S13',source_blocks=5,source_rows_copied=171,source_values_checked=checked,
    numeric_values_copied_without_rounding=True,requested_number_formats_verified=True,
    no_formulas_or_statistics_computed=True,exclusive_creation=True,
    output_sheet_count=14,source_workbook=SPEC['source_workbook'],
    source_workbook_sha256=SPEC['source_workbook_sha256'],
    sections=[dict(title=b['title'],source=b['file'],rows=b['rows'],header_row=b['headerRow'],
                   omitted_duplicate_fraction_columns=b['omitted_duplicate_fraction_columns']) for b in QA['blocks']])
with (HERE/'S13_BUILD_VALIDATION.json').open('x') as f:
    json.dump(validation,f,indent=2,ensure_ascii=False)
    f.write('\n')
print(json.dumps(validation,indent=2,ensure_ascii=False))
