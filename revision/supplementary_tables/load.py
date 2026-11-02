"""Load raw supplementary workbook sections as DataFrames."""
import openpyxl, pandas as pd
import os
RAW = os.environ.get('SUPP_FULL_RECORDS', '../../results/supplementary/Supplementary_Tables_S1-S18_full_records.xlsx')

def sections(sheet, wb=None):
    wb = wb or openpyxl.load_workbook(RAW, read_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    out, name, hdr, data = [], None, None, []
    def flush():
        if hdr is not None:
            cols = [c if c is not None else f'_c{i}' for i, c in enumerate(hdr)]
            df = pd.DataFrame([r[:len(cols)] for r in data], columns=cols)
            df = df.dropna(how='all')
            out.append((name, df))
    for r in rows:
        nn = [c for c in (r or ()) if c is not None]
        if not nn:
            continue
        if len(nn) <= 2 and hdr is None:
            name = ' | '.join(str(c) for c in nn) if name is None or not str(nn[0]).startswith('Source') else name
            if str(nn[0]).startswith(('A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'K.', 'D/E')):
                name = str(nn[0])
            continue
        if len(nn) <= 2 and hdr is not None and isinstance(nn[0], str) and (nn[0][:2] in ('A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'K.') or nn[0].startswith('D/E')):
            flush(); hdr, data = None, []; name = nn[0]; continue
        if hdr is None:
            hdr = r; data = []; continue
        if len(nn) <= 2 and isinstance(nn[0], str) and len(nn[0]) > 60:
            continue
        data.append(r)
    flush()
    return out
