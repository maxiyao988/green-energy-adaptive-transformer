"""Build the clean Supplementary Tables S1-S18 workbook (no audit fields).

Reads the raw audit workbook through load.sections, keeps the result columns, relabels codes and
writes one formatted sheet per table plus a Contents sheet. Numerical values are written unchanged;
only the display format is set per column.
usage: python3 build_clean.py OUT.xlsx   (reads the full-records workbook; override its path with SUPP_FULL_RECORDS)
"""
import math
import sys

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from load import sections

OUT = sys.argv[1]
from load import RAW as RAW_PATH
RAW = openpyxl.load_workbook(RAW_PATH, read_only=True)
REPO = 'https://github.com/maxiyao988/green-energy-adaptive-transformer'
S = {i: [df for _, df in sections('Table S%d' % i, RAW)] for i in range(1, 19)}

DASH = '–'  # en dash for "not applicable"
MINUS = '−'

# ----------------------------------------------------------------------------- labels
INPUT = {'O': 'Price-level', 'C': 'Window-relative'}
MODEL = {'zero': 'Zero Predictor', 'global_train_mean': 'Global Train-Mean', 'train_mean': 'Global Train-Mean',
         'gtm': 'Global Train-Mean', 'ticker_train_mean': 'Ticker Train-Mean',
         'expanding_pooled_mean': 'Expanding mean (pooled)', 'expanding_ticker_mean': 'Expanding mean (ticker)',
         'ols_mom6_1': 'Momentum 6' + DASH + '1 (calibrated)', 'ols_rev1m': 'Short-term reversal (calibrated)',
         'mom6_1_raw': 'Momentum 6' + DASH + '1 (raw signal)', 'rev1m_raw': 'Short-term reversal (raw signal)',
         'ridge': 'Ridge', 'lstm': 'LSTM', 'vanilla': 'Vanilla Transformer', 'swim': 'SWiM-style Transformer',
         'single': 'Single', 'fixed': 'Fixed', 'same_patch_fixed': 'Same-patch Fixed', 'static': 'Static',
         'adaptive': 'Adaptive', 'constant_gate': 'Constant gate', 'static-vs-adaptive': 'Static vs Adaptive'}
ORDER = ['zero', 'global_train_mean', 'train_mean', 'gtm', 'ticker_train_mean', 'expanding_pooled_mean',
         'expanding_ticker_mean', 'ols_mom6_1', 'ols_rev1m', 'mom6_1_raw', 'rev1m_raw', 'ridge', 'lstm', 'vanilla',
         'swim', 'single', 'fixed', 'same_patch_fixed', 'static', 'adaptive', 'constant_gate']
METRIC = {'mse': 'MSE', 'squared_loss': 'MSE', 'rank_ic': 'Rank IC', 'da': 'DA', 'direction_accuracy': 'DA',
          'directional_accuracy': 'DA'}
METHOD = {'joint_seed_date': 'Seed ' + '×' + ' date', 'date_only': 'Date only', 'Newey_West': 'Newey' + DASH + 'West'}
BACKBONE = {'S1': 'Fixed-trained (S1)', 'S2': 'Adaptive-trained (S2)'}
SETTING = {'head_frozen': 'Head frozen', 'head_retrained': 'Head retrained', 'tuned_head_frozen': 'Head frozen'}
CONFIG = {'daily_only': 'Daily only', 'weekly_only': 'Weekly only', 'daily_weekly': 'Daily+Weekly'}
LRCOND = {'selected': 'Selected rate', 'base 1e-4': 'Base rate 10' + '⁻⁴', 'tuned': 'Selected rate',
          'Step16_lr1e-4': 'Base rate 10' + '⁻⁴'}
INTERV = {'I0_learned': 'I0 Learned weights', 'I1_equal': 'I1 Equal weights',
          'I2_global_train_mean': 'I2 Training-mean weights (global)',
          'I3_ticker_train_mean': 'I3 Training-mean weights (ticker)',
          'I4_date_mean': 'I4 Same-date mean weights', 'I6_equal_daily_only': 'I6 Equal weights, Daily branch only',
          'I6_equal_weekly_only': 'I6 Equal weights, Weekly branch only'}


def lab_model(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x)
    return MODEL.get(s, MODEL.get(s.lower(), s))


def lab_interv(x):
    s = str(x)
    if s in INTERV:
        return INTERV[s]
    if s.startswith('I5_drop_'):
        br, sc = s[8:].rstrip('0123456789'), s[8:].lstrip('abcdefghijklmnopqrstuvwxyz')
        return 'I5 Remove %s scale %s' % (br.capitalize(), sc)
    return lab_model(s) if s.lower() in MODEL else s


def ikey(s):
    s = str(s).split('\u2212')[0]
    import re
    m = re.match(r'I(\d)_?([a-z_]*?)(\d*)$', s)
    return (int(m.group(1)), m.group(2), int(m.group(3)) if m.group(3) else 0) if m else (9, s, 0)


def lab_contrast(x):
    s = str(x)
    for sep in ('-minus-', '_minus_', '−', ' minus '):
        if sep in s:
            a, b = s.split(sep, 1)
            f = lab_interv if a.startswith('I') and '_' in a or a == 'I0' else lab_model
            g = lab_interv if b.startswith('I') and ('_' in b or b == 'I0') else lab_model
            return '%s %s %s' % (f(a), MINUS, g(b) if b != 'I0' else 'I0')
    return lab_model(s)


def fam(x):
    s = str(x)
    base, _, suf = s.partition('-')
    return base.capitalize() + (' ' + suf if suf else '')


def yes(x):
    if x is True or str(x) == 'True':
        return 'Yes'
    if x is False or str(x) == 'False':
        return 'No'
    return x


def okey(m):
    m = str(m).lower()
    return ORDER.index(m) if m in ORDER else 99


def sort(df, *cols):
    """Sort with price-level first and models in the manuscript order."""
    keys = []
    d = df.copy()
    for c in cols:
        k = '_k_' + c
        if c in ('input set', 'input_set', 'Input set'):
            d[k] = d[c].map({'O': 0, 'C': 1}).fillna(2)
        elif c in ('model', 'variant', 'Model', 'first model'):
            d[k] = d[c].map(okey)
        else:
            d[k] = pd.to_numeric(d[c], errors='coerce') if pd.to_numeric(d[c], errors='coerce').notna().all() else d[c].astype(str)
        keys.append(k)
    return d.sort_values(keys, kind='stable').drop(columns=keys)


# ----------------------------------------------------------------------------- formats
FMT = {'int': '0', 'mse': '0.000000', 'd6': '0.000000', 'ic': '0.0000', 'p': '0.0000', 'pctf': '0.00%',
       'pct': '0.00', 'pp': '0.00', 't': '0.00', 'r': '0.0000', 'sh': '0.000', 'f4': '0.0000', 'f3': '0.000',
       'lr': '0E+00', 'w': '0.0000', 'g': 'General', 'ep': '0.0', 'thou': '#,##0'}
NA_PREFIX = ('N/A', 'NOT STORED', 'NOT CHECKED')
import re
# source text lost spaces between words and numbers (e.g. 'to3 decimals', 'June1'); restore them
GLUE = re.compile(r'\b(to|at|of|per|vendor|issuer|represents|expands|January|February|March|April|May|June|July|August|'
                  r'September|October|November|December)(\d)')


def clean(v, fmt):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if math.isnan(v):
            return None
        v = float(v)
        if fmt in ('int', 'thou') and v.is_integer():
            return int(v)
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return 'Yes' if v else 'No'
    if isinstance(v, str):
        if not v.startswith('http'):
            v = GLUE.sub(lambda m: m.group(1) + ' ' + m.group(2), v)
        if v.startswith(NA_PREFIX):
            return DASH
        if v in ('True', 'False'):
            return yes(v)
        if fmt not in ('g', None) and fmt != 'txt':
            try:
                f = float(v)
                return int(f) if fmt in ('int', 'thou') and f.is_integer() else f
            except ValueError:
                return v
    return v


# ----------------------------------------------------------------------------- styling
FONT = Font(name='Arial', size=9)
BOLD = Font(name='Arial', size=9, bold=True)
TITLE = Font(name='Arial', size=12, bold=True)
NOTE = Font(name='Arial', size=9, italic=False, color='404040')
PANEL = Font(name='Arial', size=10, bold=True, color='1F3864')
HFILL = PatternFill('solid', fgColor='E7EBF2')
THIN = Side(style='thin', color='808080')
MED = Side(style='medium', color='404040')
WRAP_CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)
LEFT = Alignment(horizontal='left', vertical='top')
RIGHT = Alignment(horizontal='right', vertical='top')
CENTER = Alignment(horizontal='center', vertical='top')


class Sheet:
    def __init__(self, wb, num, title, notes):
        self.ws = wb.create_sheet('S%d' % num)
        self.num, self.title = num, title
        self.widths = {}
        self.panels = []
        ws = self.ws
        ws.sheet_view.showGridLines = False
        ws.cell(1, 1, 'Table S%d. %s' % (num, title)).font = TITLE
        r = 2
        for n in notes:
            c = ws.cell(r, 1, n)
            c.font = NOTE
            r += 1
        self.r = r + 1
        self.max_cols = 1

    def panel(self, title, df, spec):
        ws, r = self.ws, self.r
        missing = [src for _, src, _ in spec if not callable(src) and src not in df.columns]
        assert not missing, (self.num, title, missing)
        ws.cell(r, 1, title).font = PANEL
        self.panels.append((title, len(df)))
        r += 1
        hr = r
        for j, (h, src, fmt) in enumerate(spec, 1):
            c = ws.cell(r, j, h)
            c.font, c.fill, c.alignment = BOLD, HFILL, WRAP_CENTER
            c.border = Border(top=MED, bottom=THIN)
            self._w(j, max(len(w) for w in h.split()) + 2 if h else 4, header=len(h))
        ws.row_dimensions[r].height = 12 * max(1, max(math.ceil(len(h) / max(self.widths.get(j, 10), 8)) for j, (h, _, _) in enumerate(spec, 1)))
        r += 1
        first = r
        for _, row in df.iterrows():
            lines = 1
            for j, (h, src, fmt) in enumerate(spec, 1):
                v = src(row) if callable(src) else row.get(src)
                v = clean(v, fmt)
                if v is None:
                    v = DASH
                c = ws.cell(r, j, v)
                c.font = FONT
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    c.number_format = FMT.get(fmt, 'General')
                    c.alignment = RIGHT
                    self._w(j, 11 if fmt not in ('int', 'thou') else max(5, len(str(v)) + 2))
                elif v == DASH:
                    c.alignment = CENTER
                else:
                    c.alignment = LEFT
                    if v is not None:
                        self._w(j, min(len(str(v)) + 2, 48))
                        if len(str(v)) > 50:
                            c.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
                            lines = max(lines, math.ceil(len(str(v)) / 52))
            if lines > 1:
                ws.row_dimensions[r].height = 12 * lines
            r += 1
        for j in range(1, len(spec) + 1):
            ws.cell(r - 1, j).border = Border(bottom=MED)
        self.max_cols = max(self.max_cols, len(spec))
        self.r = r + 2
        return hr, first, r - 1

    def _w(self, j, w, header=None):
        self.widths[j] = max(self.widths.get(j, 6), min(w, 48))

    def finish(self):
        ws = self.ws
        for j, w in self.widths.items():
            ws.column_dimensions[get_column_letter(j)].width = max(w, 7)
        ws.freeze_panes = 'A2'


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ============================================================================= workbook
wb = openpyxl.Workbook()
contents = wb.active
contents.title = 'Contents'
sheets = []

COMMON_NOTE = ('Models are trained with five seeds (0, 1, 21, 42, 3407); ' 'SD is the sample standard deviation across seeds. '
               'MSE is in squared h-day log returns. R' + '²' + ' is out-of-sample R' + '²' + ' in percent. '
               + DASH + ' = not applicable or not defined.')

# ----------------------------------------------------------------------------- S1
d = S[1][0]
sh = Sheet(wb, 1, 'Corporate-action records, issuer sources, and treatment rules', [
    'Vendor (Yahoo Finance) corporate-action records for the 17 stocks, checked against issuer SEC filings. '
    'Rules R1' + DASH + 'R5 (Panel D) state how each event type affects inputs, labels and portfolio returns.',
    'Cash dividends are checked for the largest vendor dividend of each stock; the other cash records are listed in Panel E.'])
STATUS = {'MATCH': 'Match', 'MISMATCH': 'Mismatch', 'FOUND': 'Found'}


def status(row):
    s = str(row['verification status'])
    terms = str(row.get('issuer terms', ''))
    if s == 'MISMATCH' and 'rounded' in terms:
        return 'Match after vendor rounding to three decimals'
    if s.startswith('NOT APPLICABLE'):
        return 'No vendor cash record'
    return STATUS.get(s, s)


def cap(x):
    return x if not isinstance(x, str) or x.startswith('N/A') else x[:1].upper() + x[1:]


x = d[d['record type'] == 'issuer_verified_event']
sh.panel('A. Split-type events verified against issuer filings', sort(x, 'ticker', 'event date'), [
    ('Ticker', 'ticker', 'txt'), ('Vendor date', 'event date', 'txt'),
    ('Vendor event', lambda r: cap(r['vendor event']), 'txt'), ('Vendor ratio', 'vendor value', 'f3'),
    ('Issuer event', lambda r: cap(r['issuer event']), 'txt'), ('Issuer terms', 'issuer terms', 'txt'),
    ('Filing', 'issuer filing form', 'txt'), ('Filing date', 'issuer filing date', 'txt'),
    ('Effective date', 'issuer effective date', 'txt'),
    ('First post-event trading date', 'issuer first post event trade date', 'txt'),
    ('Date check', lambda r: STATUS.get(r['date check'], r['date check']), 'txt'),
    ('Ratio check', lambda r: STATUS.get(r['ratio check'], r['ratio check']), 'txt'),
    ('Result', status, 'txt'), ('Rule', 'handling rule id', 'txt'), ('Filing URL', 'issuer filing url', 'txt')])
x = d[d['record type'] == 'security_identity']
sh.panel('B. Security-identity events (name, ticker and share-class changes)', sort(x, 'ticker', 'event date'), [
    ('Ticker', 'ticker', 'txt'), ('Date', 'event date', 'txt'), ('Event', lambda r: cap(r['issuer event']), 'txt'),
    ('Details', 'issuer terms', 'txt'), ('Filing', 'issuer filing form', 'txt'),
    ('Filing date', 'issuer filing date', 'txt'), ('Result', status, 'txt'), ('Rule', 'handling rule id', 'txt'),
    ('Filing URL', 'issuer filing url', 'txt')])
x = d[d['record type'] == 'cash_dividend_spotcheck_summary']
sh.panel('C. Largest cash dividend per stock checked against issuer filings', sort(x, 'ticker'), [
    ('Ticker', 'ticker', 'txt'), ('Ex-date', 'event date', 'txt'), ('Vendor amount (USD)', 'vendor value', 'f3'),
    ('Issuer terms', lambda r: None if str(r['issuer terms']).startswith('No cash event') else r['issuer terms'], 'txt'), ('Filing', 'issuer filing form', 'txt'),
    ('Filing date', 'issuer filing date', 'txt'), ('Vendor cash records', 'cash dividend count', 'int'),
    ('First cash record', 'cash first date', 'txt'), ('Last cash record', 'cash last date', 'txt'),
    ('Result', status, 'txt'), ('Rule', 'handling rule id', 'txt'), ('Filing URL', 'issuer filing url', 'txt')])
x = d[d['record type'] == 'handling_rule']
sh.panel('D. Handling rules', x, [
    ('Rule', 'handling rule id', 'txt'), ('Event type', lambda r: cap(r['handling rule type']), 'txt'),
    ('Vendor treatment', 'vendor treatment', 'txt'), ('Price-level inputs', 'original level features', 'txt'),
    ('Window-relative inputs', 'causal window relative features', 'txt'), ('Labels', 'labels', 'txt'),
    ('Portfolio returns and covariances', 'portfolio returns covariances', 'txt')])
x = d[d['record type'] == 'vendor_action']
sh.panel('E. All vendor corporate-action records', sort(x, 'ticker', 'event date'), [
    ('Ticker', 'ticker', 'txt'), ('Date', 'event date', 'txt'), ('Event', lambda r: cap(r['vendor event']), 'txt'),
    ('Value', 'vendor value', 'g'), ('Rule', 'handling rule id', 'txt')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S2
d = S[2][0].copy()
d['share'] = d['gtm squared bias'].astype(float) / d['gtm test mse'].astype(float)
sh = Sheet(wb, 2, 'Mean realized returns by data partition', [
    'Mean h-day log return pooled over the 17 stocks and all anchor dates of each partition (primary window W3), in percent. '
    'The Global Train-Mean forecast equals the training mean.',
    'Squared-bias share = (test mean ' + MINUS + ' Global Train-Mean)' + '²' + ' / Global Train-Mean test MSE.'])
sh.panel('Mean realized returns and the Global Train-Mean', d, [
    ('Horizon (days)', 'horizon', 'int'), ('Training mean', 'train mean', 'pctf'),
    ('Validation mean', 'val mean', 'pctf'), ('Test mean', 'test mean', 'pctf'),
    ('Global Train-Mean test MSE', 'gtm test mse', 'mse'), ('Squared bias', 'gtm squared bias', 'g'),
    ('Squared-bias share of test MSE', 'share', 'pctf')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S3
d = S[3][0]
sh = Sheet(wb, 3, 'Historical-mean, momentum, and reversal benchmarks', [
    'Test-period (W3) results of the unconditional references, historical-mean variants, return-based signals and Ridge. '
    'R' + '²' + ' and the one-sided Clark' + DASH + 'West p-value are relative to the Global Train-Mean (unadjusted).',
    'Raw momentum and reversal signals are ranking signals, not return forecasts, so MSE is not defined; '
    'the calibrated versions map each signal into a return forecast with a pooled regression estimated on the training set.',
    'Rank IC t-statistics are Newey' + DASH + 'West with lag h ' + MINUS + ' 1 in Panel A and lag h in Panel B. '
    'Ridge is deterministic (one fit per cell).'])
x = d[d['input set'].astype(str).str.startswith('O source')]
spec3 = [('Horizon (days)', 'horizon', 'int'), ('Test MSE', 'test mse', 'mse'),
         ('R' + '²' + ' vs Global Train-Mean', 'r2 os vs train mean', 'pctf'),
         ('Clark' + DASH + 'West p', 'cw p nw vs train mean', 'p'), ('Mean Rank IC', 'mean rank ic', 'ic'),
         ('Rank IC t (NW)', 'rank ic nw t', 't')]
sh.panel('A. Unconditional references and financial benchmarks (identical for both input specifications)',
         sort(x, 'model', 'horizon'), [('Model', lambda r: lab_model(r['model']), 'txt')] + spec3)
x = d[~d['input set'].astype(str).str.startswith('O source')]
sh.panel('B. Ridge by input configuration', sort(x, 'input set', 'configuration', 'horizon'), [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'),
    ('Configuration', lambda r: CONFIG[r['configuration']], 'txt')] + spec3)
sheets.append(sh)

# ----------------------------------------------------------------------------- S4
d = S[4][0]
sh = Sheet(wb, 4, 'Forecast performance under single-resolution input configurations', [
    'Daily-only and Weekly-only versions of the models, primary window W3. One run per cell (seed 42) unless stated; '
    'the price-level Weekly-only Adaptive model was trained with all five seeds (Panel B).',
    'Clark' + DASH + 'West tests are one-sided against the Global Train-Mean (Newey' + DASH + 'West); Rank IC p-values from a moving block bootstrap. '
    'Holm adjustment is across all cells of the table.'])
sh.panel('A. Results by run', sort(d, 'input set', 'model', 'config', 'h', 'seed'), [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'),
    ('Configuration', lambda r: CONFIG[r['config']], 'txt'), ('Horizon (days)', 'h', 'int'), ('Seed', 'seed', 'int'),
    ('Test MSE', 'mse', 'mse'), ('R' + '²' + ' vs Global Train-Mean', 'r2os vs global train mean', 'pctf'),
    ('R' + '²' + ' vs zero', 'r2os vs zero', 'pctf'), ('Clark' + DASH + 'West p', 'cw p nw one sided', 'p'),
    ('Clark' + DASH + 'West p, Holm', 'cw p nw one sided holm all cells', 'p'), ('Mean Rank IC', 'mean rank ic', 'ic'),
    ('Rank IC t (NW)', 'rank ic nw t', 't'), ('Rank IC p (MBB)', 'rank ic p mbb', 'p'),
    ('Rank IC p (MBB), Holm', 'rank ic p mbb holm all cells', 'p')])
x = sort(d[d['n seeds'].astype(float) == 5].drop_duplicates(['input set', 'model', 'config', 'h']), 'h')
sh.panel('B. Five-seed summary, price-level Weekly-only Adaptive', x, [
    ('Horizon (days)', 'h', 'int'), ('Test MSE, mean', 'mse mean', 'mse'), ('Test MSE, SD', 'mse sd', 'mse'),
    ('Mean Rank IC, mean', 'rank ic mean', 'ic'), ('Mean Rank IC, SD', 'rank ic sd', 'ic'),
    ('Seeds beating Global Train-Mean (CW > 0)', 'cw seed wins', 'int'),
    ('Seeds with positive Rank IC', 'rank ic seed wins', 'int')])
sheets.append(sh)

# ----------------------------------------------------------------------------- shared contrast spec


def contrast_cols(est='theta hat', lo='ci lower 95', hi='ci upper 95', p='centered p two sided', holm='p holm 3h',
                  fmt_by_metric=True, metric_col='metric'):
    def f(col):
        def g(r):
            return r[col]
        return g
    fm = 'd6'
    return [('Estimate', est, fm), ('95% CI, lower', lo, fm), ('95% CI, upper', hi, fm),
            ('p (two-sided)', p, 'p'), ('Holm p (3 horizons)', holm, 'p')]


# ----------------------------------------------------------------------------- S5
d = S[5][0]
sh = Sheet(wb, 5, 'Rank IC comparisons between aggregation variants', [
    'Paired differences in mean Rank IC (first ' + MINUS + ' second model), Daily+Weekly configuration, primary window W3. '
    'Seed ' + '×' + ' date: hierarchical bootstrap over seeds and dates; date only: dates resampled with the five runs fixed. '
    'Moving blocks of length h, B = 20,000; two-sided centered p-values; Holm adjustment across the three horizons.',
    'Seed wins = number of seeds (of 5) in which the first model has the higher Rank IC.'])
sh.panel('Rank IC contrasts', sort(d, 'input set', 'contrast', 'h', 'method'), [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Inference', lambda r: METHOD[r['method']], 'txt'),
    ('Mean Rank IC, first', 'first mean', 'ic'), ('Mean Rank IC, second', 'second mean', 'ic'),
    ('Difference', 'theta hat', 'ic'), ('95% CI, lower', 'ci lower 95', 'ic'), ('95% CI, upper', 'ci upper 95', 'ic'),
    ('p (two-sided)', 'centered p two sided', 'p'), ('Holm p (3 horizons)', 'p holm 3h', 'p'),
    ('Seed wins', 'seed wins', 'int')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S6
s6 = S[6]
sh = Sheet(wb, 6, 'Frozen-representation comparisons and aggregation learning-rate analyses', [
    'Shared multi-scale backbones (Fixed-trained S1, Adaptive-trained S2) with input projections, experts and prediction head frozen; '
    'only the aggregation component is trained. Fixed has no trainable aggregation parameters (equal weights); Static has 8, '
    'Adaptive and the constant gate 8,520. Primary window W3, Daily+Weekly.',
    'Selected rate: learning rate chosen from {10' + '⁻⁴' + ', 10' + '⁻³' + ', 10' + '⁻²'
    + '} by validation MSE with seed 0, then used for all five seeds (main text, Table 7). Base rate: 10' + '⁻⁴' + ' (sensitivity).',
    'Contrasts are first ' + MINUS + ' second arm; seed ' + '×' + ' date bootstrap (B = 20,000, block length h), '
    'two-sided centered p-values, Holm adjustment across the three horizons. DA excludes realized returns equal to zero.',
    'Effect-size bounds: one-sided 95% upper bound on the improvement of Adaptive over the second arm, in percent of the '
    'second arm' + "'" + 's mean test MSE (negative values: the bound itself favours the second arm).'])
base = [('Input specification', lambda r: INPUT[r['input set']], 'txt'),
        ('Backbone', lambda r: BACKBONE[r['backbone']], 'txt')]
x = s6[1]
sh.panel('A. Selected learning rates', sort(x, 'input set', 'backbone', 'h', 'variant'), base + [
    ('Horizon (days)', 'h', 'int'), ('Aggregator', lambda r: lab_model(r['variant']), 'txt'),
    ('Selected learning rate', 'lr', 'lr'), ('Validation MSE (seed 0)', 'validation mse', 'mse')])
x = s6[2]
met = [('Test MSE, mean', 'MSE mean (log-return squared)', 'mse'), ('Test MSE, SD', 'MSE seed SD (log-return squared)', 'mse'),
       ('Mean Rank IC, mean', 'rank ic mean', 'ic'), ('Mean Rank IC, SD', 'rank ic sd', 'ic'),
       ('DA, mean', 'da mean', 'pctf'), ('DA, SD', 'da sd', 'pctf')]
sh.panel('B. Five-seed metrics at the selected learning rate', sort(x, 'input set', 'backbone', 'h', 'variant'), base + [
    ('Horizon (days)', 'h', 'int'), ('Aggregator', lambda r: lab_model(r['variant']), 'txt')] + met)
x = s6[3]


def cfmt(r):
    return 'ic' if r['metric'] == 'rank_ic' else 'd6'


def mval(col, kind='d6'):
    """Value formatted by metric: Rank IC and DA with 4 decimals, MSE with 6."""
    return col


sh.panel('C. Paired contrasts at the selected learning rate', sort(x, 'input set', 'backbone', 'contrast', 'metric', 'h'), base + [
    ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'), ('Metric', lambda r: METRIC[r['metric']], 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Difference', 'theta hat', 'd6'), ('95% CI, lower', 'ci lower 95', 'd6'),
    ('95% CI, upper', 'ci upper 95', 'd6'), ('p (two-sided)', 'centered p two sided', 'p'),
    ('Holm p (3 horizons)', 'p holm 3h', 'p'), ('Seed wins, first arm', 'seed wins', 'int'),
    ('Seed losses, first arm', 'seed losses', 'int')])
x = s6[12]
sh.panel('D. Effect-size bounds at the selected learning rate', sort(x, 'input set', 'backbone', 'contrast', 'h'), base + [
    ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Second-arm test MSE', 'second arm mean test mse', 'mse'),
    ('MSE difference (% of second arm)', 'mse difference pct second mse', 'pct'),
    ('95% CI, lower (%)', 'mse ci95 lower pct second mse', 'pct'), ('95% CI, upper (%)', 'mse ci95 upper pct second mse', 'pct'),
    ('Upper bound on Adaptive MSE gain (%)', 'adaptive mse improvement upper95 pct', 'pct'),
    ('Rank IC difference', 'rank ic difference', 'ic'), ('Rank IC 95% CI, lower', 'rank ic ci95 lower', 'ic'),
    ('Rank IC 95% CI, upper', 'rank ic ci95 upper', 'ic'),
    ('Upper bound on Adaptive Rank IC gain', 'adaptive rank ic improvement upper95', 'ic')])
# E: base-rate metrics, zero-excluded DA merged from the Step21 blocks
e = s6[7].rename(columns={'horizon': 'h'})
da = pd.concat([s6[4].assign(setting='head_frozen')[['input set', 'backbone', 'setting', 'h', 'variant', 'da mean', 'da sd']],
                s6[8].rename(columns={'model': 'variant'})[['input set', 'backbone', 'setting', 'h', 'variant', 'da mean', 'da sd']]])
for c in ('h',):
    e[c] = e[c].astype(int)
    da[c] = da[c].astype(int)
e = e.merge(da, on=['input set', 'backbone', 'setting', 'h', 'variant'], how='left', validate='1:1')
assert e['da mean'].notna().all() and len(e) == 72
sh.panel('E. Five-seed metrics at the base learning rate (head frozen; head retrained as sensitivity, S1 only)',
         sort(e, 'setting', 'input set', 'backbone', 'h', 'variant'), [
    ('Head', lambda r: SETTING[r['setting']], 'txt')] + base + [
    ('Horizon (days)', 'h', 'int'), ('Aggregator', lambda r: lab_model(r['variant']), 'txt'),
    ('Trainable parameters', 'trainable parameters', 'thou'), ('Best epoch, mean', 'best epoch mean', 'ep')] + met)
x = s6[5]
x = x[x['metric'] != 'direction_accuracy']
sh.panel('F. Paired MSE and Rank IC contrasts at the base learning rate', sort(x, 'setting', 'input set', 'backbone', 'contrast', 'metric', 'horizon'), [
    ('Head', lambda r: SETTING[r['setting']], 'txt')] + base + [
    ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'), ('Metric', lambda r: METRIC[r['metric']], 'txt'),
    ('Horizon (days)', 'horizon', 'int'), ('Difference', 'theta', 'd6'), ('95% CI, lower', 'ci lo', 'd6'),
    ('95% CI, upper', 'ci hi', 'd6'), ('p (two-sided)', 'p centered', 'p'), ('Holm p (3 horizons)', 'p holm over h', 'p')])
x = s6[9]
x = x[x['contrast'].notna() & x['backbone'].notna()]
sh.panel('G. Paired DA contrasts at the base learning rate (zero returns excluded)', sort(x, 'setting', 'input set', 'backbone', 'contrast', 'h'), [
    ('Head', lambda r: SETTING[r['setting']], 'txt')] + base + [
    ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('DA, first arm', 'first da mean', 'pctf'), ('DA, second arm', 'second da mean', 'pctf'),
    ('Difference (pp)', 'estimate pp', 'pp'), ('95% CI, lower (pp)', 'ci lower 95 pp', 'pp'),
    ('95% CI, upper (pp)', 'ci upper 95 pp', 'pp'), ('p (two-sided)', 'centered p two sided', 'p'),
    ('Holm p (3 horizons)', 'p holm 3h', 'p'), ('Seed wins, first arm', 'seed wins', 'int'),
    ('Seed losses, first arm', 'seed losses', 'int')])
x = s6[10]
sh.panel('H. Movement of the aggregation weights (five-seed means)', sort(x, 'regime', 'input set', 'backbone', 'h', 'variant', 'branch'), [
    ('Learning rate', lambda r: LRCOND[r['regime']], 'txt')] + base + [
    ('Horizon (days)', 'h', 'int'), ('Aggregator', lambda r: lab_model(r['variant']), 'txt'),
    ('Branch', lambda r: str(r['branch']).capitalize(), 'txt'),
    ('Mean |weight ' + MINUS + ' 0.25|, mean', 'mean abs deviation mean', 'w'),
    ('Mean |weight ' + MINUS + ' 0.25|, SD', 'mean abs deviation seed sd', 'w'),
    ('Within-model SD of weights, mean', 'sample sd mean', 'w'), ('Within-model SD of weights, SD', 'sample sd seed sd', 'w'),
    ('Between-date share of variance', 'between date share mean', 'f3'),
    ('Seeds with observation-invariant weights', 'n constant seeds', 'int')])
x = s6[0]
sh.panel('I. All learning-rate tuning trials (seed 0)', sort(x, 'input set', 'backbone', 'h', 'variant', 'lr'), base + [
    ('Horizon (days)', 'h', 'int'), ('Aggregator', lambda r: lab_model(r['variant']), 'txt'),
    ('Learning rate', 'lr', 'lr'), ('Validation MSE', 'validation mse', 'mse'), ('Best epoch', 'best epoch', 'int'),
    ('Epochs trained', 'epochs', 'int'), ('Selected', lambda r: yes(r['selected flag']), 'txt')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S7
s7 = S[7][0]
sh = Sheet(wb, 7, 'Within-model routing interventions, including scale removal and changes in MSE and Rank IC', [
    'Inference-time interventions on the trained Adaptive models (Daily+Weekly, primary window W3, test period): the learned routing '
    'weights (I0) are replaced by equal weights (I1), mean training weights (I2 global, I3 per ticker), same-date cross-sectional mean '
    'weights (I4), removal of one scale with renormalization (I5) or equal weights in one branch only (I6).',
    'Differences are intervention ' + MINUS + ' I0. Seeds improved = seeds (of 5) in which the intervention lowers MSE or raises Rank IC. '
    'Seed ' + '×' + ' date and date-only moving block bootstrap (block length h, B = 20,000), two-sided centered p-values, '
    'Holm adjustment across the three horizons.'])
x = s7[(s7['record type'] == 'five_seed_summary') & (s7.split == 'test')].copy()
x['_i'] = x['intervention'].map(ikey)
x = x.sort_values(['_i']).pipe(sort, 'input set', 'h')
sh.panel('A. Five-seed metrics by intervention', x, [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Intervention', lambda r: lab_interv(r['intervention']), 'txt'),
    ('Test MSE, mean', 'mse mean', 'mse'), ('Test MSE, SD', 'mse sd', 'mse'),
    ('MSE difference, mean', 'delta mse mean', 'd6'), ('MSE difference, SD', 'delta mse sd', 'd6'),
    ('Seeds improved, MSE', 'mse seed wins', 'int'),
    ('Mean Rank IC, mean', 'rank ic mean', 'ic'), ('Mean Rank IC, SD', 'rank ic sd', 'ic'),
    ('Rank IC difference, mean', 'delta rank ic mean', 'ic'), ('Rank IC difference, SD', 'delta rank ic sd', 'ic'),
    ('Seeds improved, Rank IC', 'rank ic seed wins', 'int')])
inf = s7[(s7['record type'] == 'stored_inference') & (s7.method.isin(['joint_seed_date', 'date_only']))]
key = ['input set', 'contrast', 'h', 'metric']
j = inf[inf.method == 'joint_seed_date'].set_index(key)
dd = inf[inf.method == 'date_only'].set_index(key)
assert len(j) == len(dd) == 168
w = j[['theta hat', 'ci lower 95', 'ci upper 95', 'centered p two sided', 'p holm 3h']].join(
    dd[['ci lower 95', 'ci upper 95', 'centered p two sided', 'p holm 3h']], rsuffix=' do').reset_index()
assert np.allclose(j['theta hat'].astype(float).values, dd.loc[j.index, 'theta hat'].astype(float).values)
w['_i'] = w['contrast'].map(ikey)
w = w.sort_values('_i').pipe(sort, 'input set', 'metric', 'h')
w['metric'] = w['metric'].map({'squared_loss': 'mse', 'rank_ic': 'rank_ic'})
sh.panel('B. Paired inference by intervention', w, [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Metric', lambda r: METRIC[r['metric']], 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'),
    ('Difference', 'theta hat', 'd6'),
    ('Seed ' + '×' + ' date: 95% CI, lower', 'ci lower 95', 'd6'), ('Seed ' + '×' + ' date: 95% CI, upper', 'ci upper 95', 'd6'),
    ('Seed ' + '×' + ' date: p', 'centered p two sided', 'p'), ('Seed ' + '×' + ' date: Holm p', 'p holm 3h', 'p'),
    ('Date only: 95% CI, lower', 'ci lower 95 do', 'd6'), ('Date only: 95% CI, upper', 'ci upper 95 do', 'd6'),
    ('Date only: p', 'centered p two sided do', 'p'), ('Date only: Holm p', 'p holm 3h do', 'p')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S8
s8 = S[8][0].copy()
s8['hh'] = s8['horizon'].fillna(s8['h'])
sh = Sheet(wb, 8, 'Chronological-window forecast results, including Ridge', [
    'Three non-overlapping test windows (W1, W2, W3 = primary), Daily+Weekly. MSE ratio = test MSE / test MSE of the window' + "'"
    + 's Global Train-Mean (five-seed mean and SD; Ridge deterministic).',
    'Clark' + DASH + 'West tests (one-sided) of the five per-seed forecasts against each reference, pooled over seeds: Newey' + DASH
    + 'West t on the seed-averaged statistic, and seed ' + '×' + ' date and date-only moving block bootstrap '
    '(block length h, B = 20,000). Holm adjustment across the three horizons.'])
x = s8[s8['record type'] == 'five_seed_rolling_metrics']
sh.panel('A. Test MSE relative to the Global Train-Mean', sort(x, 'input set', 'window', 'model', 'hh'), [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Window', 'window', 'txt'),
    ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'hh', 'int'), ('Seeds', 'n seeds', 'int'),
    ('MSE ratio, mean', 'mse ratio mean', 'r'), ('MSE ratio, SD', 'mse ratio sd', 'r')])
p = s8[s8['record type'] == 'pooled_stored_Clark_West']
k = ['input set', 'window', 'model', 'hh', 'reference']
nw, jt, do = (p[p.method == m].set_index(k) for m in ('Newey_West', 'joint_seed_date', 'date_only'))
cw = nw[['seed CW mean', 'seed wins', 'cw nw t', 'p value', 'p holm 3h']].join(
    jt[['p value', 'p holm 3h']], rsuffix=' j').join(do[['p value', 'p holm 3h']], rsuffix=' d').reset_index()
assert len(cw) == 72
REF = {'global_train_mean': 'Global Train-Mean', 'zero': 'Zero Predictor'}
sh.panel('B. Clark' + DASH + 'West tests pooled over seeds', sort(cw, 'input set', 'window', 'model', 'reference', 'hh'), [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Window', 'window', 'txt'),
    ('Model', lambda r: lab_model(r['model']), 'txt'), ('Reference', lambda r: REF[r['reference']], 'txt'),
    ('Horizon (days)', 'hh', 'int'), ('CW statistic, mean over seeds', 'seed CW mean', 'g'),
    ('Seeds with CW > 0', 'seed wins', 'int'), ('NW t', 'cw nw t', 't'), ('NW p', 'p value', 'p'), ('NW Holm p', 'p holm 3h', 'p'),
    ('Seed ' + '×' + ' date p', 'p value j', 'p'), ('Seed ' + '×' + ' date Holm p', 'p holm 3h j', 'p'),
    ('Date-only p', 'p value d', 'p'), ('Date-only Holm p', 'p holm 3h d', 'p')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S9
s9 = S[9][0].copy()
SUBG = {'first_half': 'First half', 'second_half': 'Second half', 'high': 'High', 'mid': 'Medium', 'low': 'Low'}
GT = {'stock': 'Stock', 'volatility': 'Volatility regime', 'subperiod': 'Subperiod'}
s9['_g'] = s9['subgroup type'].map({'stock': 0, 'volatility': 1, 'subperiod': 2})
s9['_s'] = s9['subgroup'].map({'low': 0, 'mid': 1, 'high': 2, 'first_half': 0, 'second_half': 1}).fillna(0)
s9 = s9.sort_values(['_g', '_s', 'subgroup'], kind='stable').pipe(sort, 'input set', 'window', 'model', 'h')
s9 = s9.sort_values(['_g'], kind='stable')
sh = Sheet(wb, 9, 'Performance by stock, volatility regime, and test subperiod', [
    'Five-seed means and SDs within each subgroup (Ridge and references deterministic). Relative MSE = model MSE / Global Train-Mean MSE '
    'in the same subgroup. Rank IC is computed only on dates with at least five stocks in the subgroup, so it is not defined for single stocks.',
    'Volatility regimes are terciles of trailing realized volatility; subperiods are the two halves of each test window. '
    'Forecast bias = mean (forecast ' + MINUS + ' realized return), pooled over seeds.'])
sh.panel('Subgroup results', s9, [
    ('Grouping', lambda r: GT[r['subgroup type']], 'txt'), ('Subgroup', lambda r: SUBG.get(r['subgroup'], r['subgroup']), 'txt'),
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Window', 'window', 'txt'),
    ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'), ('Observations', 'n rows', 'thou'),
    ('Test MSE, mean', 'mse mean', 'mse'), ('Test MSE, SD', 'mse sd', 'mse'),
    ('Relative MSE, mean', 'relative mse mean', 'r'), ('Relative MSE, SD', 'relative mse sd', 'r'),
    ('Mean Rank IC, mean', 'rank ic mean', 'ic'), ('Mean Rank IC, SD', 'rank ic sd', 'ic'),
    ('Forecast bias', 'mean forecast bias pooled seeds', 'd6'), ('Forecast bias, seed SD', 'mean forecast bias seed sd', 'd6')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S10 / S11
BENCH = {'EW': '1/N', 'MINVAR': 'Minimum variance', 'RISKPARITY': 'Risk parity'}


def strat(s):
    s = str(s)
    if s in BENCH:
        return BENCH[s], 'Forecast-free', None
    if s == 'gtm':
        return 'Global Train-Mean', 'Deterministic', None
    if s in ('ridge', 'ridge_s42'):
        return 'Ridge', 'Deterministic', None
    m, k = s.rsplit('_', 1)
    if k == 'ens':
        return lab_model(m), 'Five-seed ensemble', None
    return lab_model(m), 'Single run', int(k[1:])


def layer(l):
    if l.startswith('L2_g'):
        return 'Mean' + DASH + 'variance', int(l[4:])
    if l == 'L1_top5':
        return 'Top-5 ranking', None
    if l == 'L1_top3':
        return 'Top-3 ranking', None
    return 'Forecast-free benchmark', None


def prep_port(d):
    d = d.copy()
    key = ['input set', 'strategy', 'h', 'layer', 'cost bp'] + (['gamma'] if 'gamma' in d.columns else [])
    vals = [c for c in ('ann geo return', 'ann vol', 'sharpe') if c in d.columns]
    dup = d[d.duplicated(key, keep=False)]
    assert (dup.groupby(key)[vals].nunique() == 1).all().all(), 'conflicting duplicate portfolio rows'
    d = d.drop_duplicates(key, keep='first')
    t = d['strategy'].map(strat)
    d['Model'] = t.map(lambda z: z[0])
    d['Forecast'] = t.map(lambda z: z[1])
    d['Seed'] = t.map(lambda z: z[2])
    lt = d['layer'].map(layer)
    d['Portfolio'] = lt.map(lambda z: z[0])
    d['gam'] = lt.map(lambda z: z[1])
    d['_m'] = d['strategy'].map(lambda s: -1 if s in BENCH else okey(str(s).rsplit('_', 1)[0] if s not in ('gtm', 'ridge', 'ridge_s42') else ('gtm' if s == 'gtm' else 'ridge')))
    d['_b'] = d['strategy'].map(lambda s: ['EW', 'MINVAR', 'RISKPARITY'].index(s) if s in BENCH else 0)
    d['_f'] = d['Forecast'].map({'Forecast-free': 0, 'Deterministic': 1, 'Five-seed ensemble': 2, 'Single run': 3})
    return d


p10 = prep_port(S[10][0])
pcols = [('Ann. return', 'ann geo return', 'f4'), ('Ann. volatility', 'ann vol', 'f4'), ('Sharpe ratio', 'sharpe', 'sh'),
         ('Max. drawdown', 'max drawdown', 'f4'), ('Turnover per rebalance', 'avg turnover ex initial', 'f4'),
         ('Ann. turnover', 'ann turnover', 'f3'), ('Hit rate', 'hit rate periods', 'pctf'),
         ('Cumulative return', 'cumulative return', 'f4')]
id_cols = [('Input specification', lambda r: INPUT.get(r['input set'], r['input set']), 'txt'),
           ('Horizon (days)', 'h', 'int')]


def port_sort(d, *extra):
    return d.sort_values(['_f', '_m', '_b'] + list(extra), kind='stable').pipe(sort, 'input set', 'h')


sh = Sheet(wb, 10, 'Portfolio returns, volatility, Sharpe ratios, drawdowns, turnover, and Top-5 allocation results', [
    'Long-only portfolios of the 17 stocks, test period 12 November 2024 to 5 June 2026, rebalanced every h trading days, '
    'net of 10 bp per unit of turnover. Mean' + DASH + 'variance: ' + 'γ' + ' = 5, 25% weight cap, Ledoit' + DASH
    + 'Wolf covariance; top-5: equal weights in the five stocks with the highest forecasts.',
    'Returns, volatility and drawdown are annualized decimal fractions; turnover per rebalance excludes the initial allocation. '
    'Panel C summarizes the single-run portfolios of Panel B across seeds (mean and SD of the five values). '
    'Price-level Ridge was evaluated with its original portfolio grid, which includes a top-3 portfolio.'])
x = p10[(p10['cost bp'] == 10) & (p10['layer'].isin(['L2_g5', 'allocation benchmark'])) & (p10['Forecast'] != 'Single run')]
x = x[~((x['layer'] == 'allocation benchmark') & (x['input set'] == 'O'))]  # benchmarks identical for both specifications
assert (S[10][0].query("layer=='allocation benchmark' and `cost bp`==10").groupby(['strategy', 'h'])['sharpe'].nunique() == 1).all()
sh.panel('A. Mean' + DASH + 'variance portfolios (' + 'γ' + ' = 5) of deterministic and ensemble forecasts, and forecast-free benchmarks',
         port_sort(x), [
    ('Input specification', lambda r: 'Both' if r['layer'] == 'allocation benchmark' else INPUT[r['input set']], 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Model', 'Model', 'txt'), ('Forecast', 'Forecast', 'txt')] + pcols)
x = p10[(p10['cost bp'] == 10) & (p10['layer'] == 'L2_g5') & (p10['Forecast'] == 'Single run')]
sh.panel('B. Mean' + DASH + 'variance portfolios (' + 'γ' + ' = 5) of single-run forecasts', port_sort(x, 'Seed'), id_cols + [
    ('Model', 'Model', 'txt'), ('Seed', 'Seed', 'int')] + pcols)
agg = x.groupby(['input set', 'h', 'Model', '_m'])[['sharpe', 'max drawdown', 'ann geo return', 'ann vol']].agg(['mean', 'std'])
agg.columns = [a + ' ' + b for a, b in agg.columns]
agg = agg.reset_index().sort_values('_m', kind='stable').pipe(sort, 'input set', 'h')
sh.panel('C. Single-run mean' + DASH + 'variance portfolios (' + 'γ' + ' = 5): mean and SD across the five seeds', agg, id_cols + [
    ('Model', 'Model', 'txt'), ('Sharpe ratio, mean', 'sharpe mean', 'sh'), ('Sharpe ratio, SD', 'sharpe std', 'sh'),
    ('Max. drawdown, mean', 'max drawdown mean', 'f4'), ('Max. drawdown, SD', 'max drawdown std', 'f4'),
    ('Ann. return, mean', 'ann geo return mean', 'f4'), ('Ann. return, SD', 'ann geo return std', 'f4'),
    ('Ann. volatility, mean', 'ann vol mean', 'f4'), ('Ann. volatility, SD', 'ann vol std', 'f4')])
x = p10[(p10['cost bp'] == 10) & (p10['layer'].isin(['L1_top5', 'L1_top3']))]
sh.panel('D. Ranking portfolios', port_sort(x, 'Seed').sort_values(['Portfolio'], kind='stable', ascending=False), id_cols + [
    ('Portfolio', 'Portfolio', 'txt'), ('Model', 'Model', 'txt'), ('Forecast', 'Forecast', 'txt'), ('Seed', 'Seed', 'int')] + pcols)
sheets.append(sh)

p11 = prep_port(S[11][0])
p11 = p11[p11['Forecast'] != 'Single run']
p11 = p11[~((p11['layer'] == 'allocation benchmark') & (p11['input set'] == 'O'))]
sh = Sheet(wb, 11, 'Transaction-cost and risk-aversion sensitivity analyses', [
    'Deterministic and five-seed ensemble forecasts and forecast-free benchmarks under proportional costs of 0, 5, 10 and 20 bp per unit '
    'of turnover and risk aversion ' + 'γ' + ' ' + '∈' + ' {1, 2, 5, 10, 20} (main specification ' + 'γ' + ' = 5, 10 bp). '
    'Single-run portfolios at 10 bp are in Table S10.',
    'Certainty-equivalent (CE) return = annualized mean return ' + MINUS + ' (A/2) ' + '×' + ' annualized variance, for A = 2 and 5. '
    'Benchmarks do not depend on the input specification.'])
p11['_l'] = p11['layer'].map(lambda l: (0, 0) if l == 'allocation benchmark' else ((1, int(l[4:])) if l.startswith('L2') else (2, 0)))
x = p11.sort_values(['_l', '_f', '_m', '_b', 'cost bp'], kind='stable').pipe(sort, 'input set', 'h')
sh.panel('Annualized performance by cost level and risk aversion', x, [
    ('Input specification', lambda r: 'Both' if r['layer'] == 'allocation benchmark' else INPUT[r['input set']], 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Portfolio', 'Portfolio', 'txt'), ('γ', 'gam', 'int'), ('Model', 'Model', 'txt'),
    ('Forecast', 'Forecast', 'txt'), ('Cost (bp)', 'cost bp', 'int'), ('Ann. return', 'ann geo return', 'f4'),
    ('Ann. volatility', 'ann vol', 'f4'), ('Sharpe ratio', 'sharpe', 'sh'), ('Turnover per rebalance', 'avg turnover ex initial', 'f4'),
    ('CE return, A = 2', 'ce A2', 'f4'), ('CE return, A = 5', 'ce A5', 'f4')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S12
s12 = S[12][0].copy()
s12['_r'] = s12['regime'].map({'PRIMARY': 0, 'SECONDARY': 1})
s12 = s12.sort_values(['_r'], kind='stable').pipe(sort, 'window', 'input set', 'model', 'h', 'seed')
s12 = s12.sort_values(['_r'], kind='stable')
sh = Sheet(wb, 12, 'Forecast shrinkage toward the training mean', [
    'Combined forecast = ' + 'λ' + ' ' + '×' + ' model forecast + (1 ' + MINUS + ' ' + 'λ' + ') ' + '×'
    + ' Global Train-Mean, for every single-run forecast cell (414 cells). ' + 'λ' + ' is selected on the validation period '
    '(Panel A) or fixed at 0.5 (Panel B); ' + 'λ' + ' = 0 reproduces the Global Train-Mean.',
    'Loss difference = test MSE of the combined forecast ' + MINUS + ' test MSE of the Global Train-Mean, with a two-sided date-level '
    'moving block bootstrap (block length h, B = 20,000) conditional on the fitted runs; Holm adjustment across all cells. '
    'With ' + '\u03bb' + ' = 0 the combined forecast equals the Global Train-Mean and no test is performed. '
    'Rank IC is unchanged by shrinkage. Ridge is deterministic.'])
for reg, ttl in (('PRIMARY', 'A. Validation-selected shrinkage weight'), ('SECONDARY', 'B. Shrinkage weight fixed at 0.5')):
    x = s12[s12['regime'] == reg]
    sh.panel(ttl, x, [
        ('Window', 'window', 'txt'), ('Input specification', lambda r: INPUT[r['input set']], 'txt'),
        ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'),
        ('Forecast', lambda r: {'ensemble': 'Five-seed ensemble', 'deterministic': 'Deterministic'}.get(str(r['member']), 'Seed %s' % r['member']), 'txt'),
        ('λ', 'lambda used', 'f3'), ('Validation MSE', 'val mse at used lambda', 'mse'), ('Test MSE', 'test mse', 'mse'),
        ('R' + '²' + ' vs Global Train-Mean', 'r2os vs global train mean', 'pctf'), ('R' + '²' + ' vs zero', 'r2os vs zero', 'pctf'),
        ('Loss difference', 'loss difference', 'd6'), ('95% CI, lower', 'loss ci lower 95', 'd6'),
        ('95% CI, upper', 'loss ci upper 95', 'd6'), ('p (two-sided)', 'loss p raw', 'p'), ('Holm p', 'loss p Holm', 'p')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S13
s13 = S[13]
sh = Sheet(wb, 13, 'Directional accuracy and corresponding comparisons', [
    'Directional accuracy (DA) = share of stock' + DASH + 'date pairs whose forecast has the sign of the realized h-day return; realized returns '
    'exactly equal to zero are excluded. Primary window W3, Daily+Weekly. Five-seed mean, SD, minimum and maximum of per-seed DA.',
    'Paired tests (Panel D): seed ' + '×' + ' date bootstrap (B = 20,000, block length h), two-sided, Holm adjustment across horizons; '
    'differences in percentage points (first ' + MINUS + ' second arm).'])
sh.panel('A. References', s13[0], [
    ('Horizon (days)', 'Horizon', 'int'), ('Positive returns', 'Positive returns', 'thou'), ('Negative returns', 'Negative returns', 'thou'),
    ('Zero returns excluded', 'Zero returns excluded', 'int'), ('DA of Global Train-Mean (= always up)', 'Train-Mean DA (%)', 'pctf')])
dacols = [('DA, mean', 'DA mean (%)', 'pctf'), ('DA, SD', 'DA SD (%)', 'pctf'), ('DA, minimum', 'DA minimum (%)', 'pctf'),
          ('DA, maximum', 'DA maximum (%)', 'pctf')]
sh.panel('B. Benchmarks and aggregation variants (Table 4 models)', sort(s13[1], 'Input set', 'Model', 'Horizon'), [
    ('Input specification', lambda r: INPUT[r['Input set']], 'txt'), ('Model', lambda r: lab_model(r['Model']), 'txt'),
    ('Horizon (days)', 'Horizon', 'int'), ('Seeds', 'Seeds', 'int')] + dacols)
x = pd.concat([s13[2], s13[3]])
sh.panel('C. Frozen representations, base learning rate (head retrained: sensitivity, S1 only)',
         sort(x, 'Setting', 'Input set', 'Backbone', 'Horizon', 'Model'), [
    ('Head', lambda r: SETTING[r['Setting']], 'txt'), ('Input specification', lambda r: INPUT[r['Input set']], 'txt'),
    ('Backbone', lambda r: BACKBONE[r['Backbone']], 'txt'), ('Horizon (days)', 'Horizon', 'int'),
    ('Aggregator', lambda r: lab_model(r['Model']), 'txt')] + dacols)
x = s13[4]
sh.panel('D. Paired DA tests', sort(x, 'Scope', 'Setting', 'Input set', 'Backbone', 'Contrast', 'Horizon'), [
    ('Comparison', lambda r: 'End-to-end models' if r['Scope'] == 'table4' else 'Frozen, ' + SETTING.get(r['Setting'], '').lower(), 'txt'),
    ('Input specification', lambda r: INPUT[r['Input set']], 'txt'),
    ('Backbone', lambda r: BACKBONE.get(r['Backbone'], DASH), 'txt'), ('Contrast', lambda r: lab_contrast(r['Contrast']), 'txt'),
    ('Horizon (days)', 'Horizon', 'int'), ('DA, first arm', 'First arm DA mean (%)', 'pctf'),
    ('DA, second arm', 'Second arm DA mean (%)', 'pctf'), ('Difference (pp)', 'Difference (pp)', 'pp'),
    ('95% CI, lower (pp)', '95% CI lower (pp)', 'pp'), ('95% CI, upper (pp)', '95% CI upper (pp)', 'pp'),
    ('p (two-sided)', 'Raw p (two-sided)', 'p'), ('Holm p (3 horizons)', 'Holm p (3 horizons)', 'p'),
    ('Seed wins, first arm', 'Seed wins', 'int'), ('Seed losses, first arm', 'Seed losses', 'int')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S14
s14 = S[14]
SETN = {'T0': 'T0 (10' + '⁻⁴' + ', ' + 'δ' + ' = 1)', 'T1': 'T1 (3' + '×' + '10' + '⁻⁵' + ', ' + 'δ' + ' = 1)',
        'T2': 'T2 (10' + '⁻⁴' + ', ' + 'δ' + ' = ' + 'σ' + 'ₕ' + ')',
        'T3': 'T3 (3' + '×' + '10' + '⁻⁵' + ', ' + 'δ' + ' = ' + 'σ' + 'ₕ' + ')'}
sh = Sheet(wb, 14, 'Alternative training settings', [
    'Window-relative inputs, primary window W3. Training settings combine the learning rate and the Huber threshold ' + 'δ'
    + ': T0 (baseline, main results), T1, T2 and T3 (' + 'σ' + 'ₕ = training-set SD of the h-day return). No setting is selected on test data.',
    'Contrasts: first ' + MINUS + ' second arm, seed ' + '×' + ' date bootstrap (B = 20,000, block length h), two-sided, Holm across horizons. '
    'Bounds: one-sided 95% upper bound on the MSE improvement of Adaptive, in percent of its mean test MSE. '
    'Loss changes: change in training and validation loss from the selected epoch to the last epoch, in percent.'])
x = s14[0]
x = x.assign(_m=x.model.map(okey)).sort_values(['setting', '_m', 'h'])
sh.panel('A. Five-seed metrics', x, [
    ('Setting', lambda r: SETN[r['setting']], 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Test MSE, mean', 'MSE mean (log-return squared)', 'mse'), ('Test MSE, SD', 'MSE seed SD (log-return squared)', 'mse'),
    ('R' + '²' + ' vs Global Train-Mean, mean', 'r2os mean', 'pctf'), ('R' + '²' + ', SD', 'r2os sd', 'pctf'),
    ('Mean Rank IC, mean', 'rank ic mean', 'ic'), ('Mean Rank IC, SD', 'rank ic sd', 'ic'), ('DA, mean', 'da mean', 'pctf'),
    ('DA, SD', 'da sd', 'pctf')])
x = s14[1]
x = x.assign(_m=x.model.map(okey)).sort_values(['setting', '_m', 'h'])
sh.panel('B. Five-seed ensemble forecasts', x, [
    ('Setting', lambda r: SETN[r['setting']], 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Test MSE', 'mse', 'mse'), ('R' + '²' + ' vs Global Train-Mean', 'r2os', 'pctf'), ('Mean Rank IC', 'mean rank ic', 'ic'),
    ('DA', 'da', 'pctf')])
# C: Static - Adaptive and Adaptive - LSTM contrasts across settings
t0 = s14[11].assign(contrast='static-minus-adaptive', metric=s14[11]['metric'].map({'squared_loss': 'mse', 'rank_ic': 'rank_ic'}))
t0 = t0[['setting', 'contrast', 'metric', 'h', 'theta hat', 'ci lower 95', 'ci upper 95', 'centered p two sided', 'p holm 3h', 'seed wins']]
t0['seed losses'] = 5 - t0['seed wins'].astype(int)
t0da = s14[12].assign(setting='T0', metric='da')[['setting', 'contrast', 'metric', 'h', 'theta hat', 'ci lower 95', 'ci upper 95',
                                                     'centered p two sided', 'p holm 3h', 'seed wins', 'seed losses']]
cc = ['setting', 'contrast', 'metric', 'h', 'theta hat', 'ci lower 95', 'ci upper 95', 'centered p two sided', 'p holm 3h', 'seed wins', 'seed losses']
t123 = pd.concat([s14[8][cc], s14[2][cc]])
# T0 Adaptive - LSTM (MSE, Rank IC) from the stored contrast (LSTM - Adaptive reversed); DA not stored
al = s14[13]
al = al[al['metric'] != 'da'].copy()
al = al.rename(columns={'estimate': 'theta hat', 'CI lower (95%, raw)': 'ci lower 95', 'CI upper (95%, raw)': 'ci upper 95', 'p holm': 'p holm 3h'})
al['centered p two sided'] = None
al['seed losses'] = None
al = al[cc]
allc = pd.concat([t0, t0da, al, t123])
allc['_c'] = allc['contrast'].map({'static-minus-adaptive': 0, 'adaptive-minus-lstm': 1})
allc['_k'] = allc['metric'].map({'mse': 0, 'rank_ic': 1, 'da': 2})
allc = allc.sort_values(['_c', 'setting', '_k', 'h'])
allc = allc[allc['theta hat'].notna()]
sh.panel('C. Paired contrasts by training setting', allc, [
    ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'), ('Setting', lambda r: SETN[r['setting']], 'txt'),
    ('Metric', lambda r: METRIC[r['metric']], 'txt'), ('Horizon (days)', 'h', 'int'), ('Difference', 'theta hat', 'd6'),
    ('95% CI, lower', 'ci lower 95', 'd6'), ('95% CI, upper', 'ci upper 95', 'd6'), ('p (two-sided)', 'centered p two sided', 'p'),
    ('Holm p (3 horizons)', 'p holm 3h', 'p'), ('Seed wins, first arm', 'seed wins', 'int'), ('Seed losses, first arm', 'seed losses', 'int')])
bnd = []
for i in (9, 10):
    b = s14[i].rename(columns={'adaptive mse improvement upper95 pct': 'ub', 'mse difference pct second mse': 'est',
                               'mse ci95 lower pct second mse': 'lo', 'mse ci95 upper pct second mse': 'hi'})
    bnd.append(b[['input set', 'setting', 'contrast', 'h', 'second arm mean test mse', 'est', 'lo', 'hi', 'ub']])
b = s14[3][s14[3]['metric'] == 'mse'].rename(columns={'adaptive mse improvement upper95 pct second mse': 'ub', 'estimate pct second mse': 'est',
                                                      'ci95 lower pct second mse': 'lo', 'ci95 upper pct second mse': 'hi'})
b = b.assign(**{'input set': 'C'})
bnd.append(b[['input set', 'setting', 'contrast', 'h', 'second arm mean test mse', 'est', 'lo', 'hi', 'ub']])
bnd = pd.concat(bnd)
bnd['_c'] = bnd['contrast'].map({'static-minus-adaptive': 0, 'adaptive-minus-lstm': 1})
bnd = bnd.sort_values(['_c', 'setting', 'h'], kind='stable').pipe(sort, 'input set')
sh.panel('D. Effect-size bounds for the MSE contrasts', bnd, [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Setting', lambda r: SETN[r['setting']], 'txt'),
    ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Second-arm test MSE', 'second arm mean test mse', 'mse'), ('MSE difference (% of second arm)', 'est', 'pct'),
    ('95% CI, lower (%)', 'lo', 'pct'), ('95% CI, upper (%)', 'hi', 'pct'), ('Upper bound on Adaptive MSE gain (%)', 'ub', 'pct')])
x = s14[14].rename(columns=lambda c: c.replace('_', ' '))
key = ['input set', 'contrast', 'metric', 'h']
j = x[x.method == 'joint_seed_date'].set_index(key)
dd = x[x.method == 'date_only'].set_index(key)
w = j[['first mean', 'second mean', 'theta hat', 'ci lower 95', 'ci upper 95', 'centered p two sided', 'p holm 3h', 'seed wins']].join(
    dd[['centered p two sided', 'p holm 3h']], rsuffix=' do').reset_index()
assert len(w) == 72
w['metric'] = w['metric'].map({'squared_loss': 'mse', 'rank_ic': 'rank_ic'})
w = w.sort_values(['contrast', 'metric', 'h'], kind='stable').pipe(sort, 'input set')
sh.panel('E. Neural-benchmark contrasts under T0 (both input specifications)', w, [
    ('Input specification', lambda r: INPUT[r['input set']], 'txt'), ('Contrast', lambda r: lab_contrast(r['contrast']), 'txt'),
    ('Metric', lambda r: METRIC[r['metric']], 'txt'), ('Horizon (days)', 'h', 'int'), ('First arm, mean', 'first mean', 'd6'),
    ('Second arm, mean', 'second mean', 'd6'), ('Difference', 'theta hat', 'd6'),
    ('Seed ' + '×' + ' date: 95% CI, lower', 'ci lower 95', 'd6'), ('Seed ' + '×' + ' date: 95% CI, upper', 'ci upper 95', 'd6'),
    ('Seed ' + '×' + ' date: p', 'centered p two sided', 'p'), ('Seed ' + '×' + ' date: Holm p', 'p holm 3h', 'p'),
    ('Date only: p', 'centered p two sided do', 'p'), ('Date only: Holm p', 'p holm 3h do', 'p'), ('Seed wins, first arm', 'seed wins', 'int')])
x = s14[4]
x = x.assign(_m=x.model.map(okey)).sort_values(['setting', '_m', 'h', 'best epoch'])
x = x.groupby(['setting', 'model', '_m', 'h'], sort=False).apply(
    lambda g: '; '.join('epoch %d: %d' % (int(a), int(b)) for a, b in zip(g['best epoch'], g['n seeds']))).rename('dist').reset_index()
lc = s14[6]
x = x.merge(lc[['setting', 'model', 'h', 'mean train pct change', 'mean val pct change']], on=['setting', 'model', 'h'], how='left')
sh.panel('F. Selected (best) epoch and loss changes after it', x, [
    ('Setting', lambda r: SETN[r['setting']], 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Best epoch: number of seeds', 'dist', 'txt'), ('Training loss change (%)', 'mean train pct change', 'pct'),
    ('Validation loss change (%)', 'mean val pct change', 'pct')])
x = s14[7]
x = x.assign(_m=x.model.map(okey)).sort_values(['setting', '_m', 'h', 'branch'])
sh.panel('G. Movement of the aggregation weights (five-seed means)', x, [
    ('Setting', lambda r: SETN[r['setting']], 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Branch', lambda r: str(r['branch']).capitalize(), 'txt'), ('Mean |weight ' + MINUS + ' 0.25|', 'mean abs deviation', 'w'),
    ('Within-model SD of weights', 'mean sample sd', 'w'), ('Between-date share of variance', 'between date share', 'f3'),
    ('Seeds with observation-invariant weights', 'n constant seeds', 'int')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S15
s15 = S[15]
WORD = {'W1+W2+W3': 'Pooled W1' + DASH + 'W3'}
sh = Sheet(wb, 15, 'Pooled three-window analyses', [
    'Baseline setting T0, Daily+Weekly. Per-window results and the three test windows pooled (1,173 test dates). R' + '²'
    + ' is relative to each window' + "'" + 's own Global Train-Mean or zero forecast.',
    'Rank IC: mean of the seed-averaged daily Rank IC, Newey' + DASH + 'West (lag h) primary and stratified moving block bootstrap (MBB) as robustness; '
    'Holm adjustment is applied to the pooled tests only. Contrasts: Static ' + MINUS + ' Adaptive, stratified seed ' + '×'
    + ' date bootstrap with seeds paired within windows; bounds in percent of the Adaptive mean test MSE.'])
x = s15[0]
x = x.assign(_w=x.window.map(lambda w: 9 if '+' in w else int(w[1]))).sort_values('_w', kind='stable').pipe(sort, 'input set', 'model', 'h')
x = x.sort_values('_w', kind='stable')
sh.panel('A. Metrics by window and pooled', x, [
    ('Window', lambda r: WORD.get(r['window'], r['window']), 'txt'), ('Input specification', lambda r: INPUT[r['input set']], 'txt'),
    ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'), ('Seeds', 'n seeds', 'int'),
    ('Test MSE, mean', 'MSE mean (log-return squared)', 'mse'), ('Test MSE, SD', 'MSE seed SD (log-return squared)', 'mse'),
    ('R' + '²' + ' vs Global Train-Mean, mean', 'r2os train mean mean', 'pctf'), ('R' + '²' + ' vs Global Train-Mean, SD', 'r2os train mean sd', 'pctf'),
    ('R' + '²' + ' vs zero, mean', 'r2os zero mean', 'pctf'), ('R' + '²' + ' vs zero, SD', 'r2os zero sd', 'pctf'),
    ('Mean Rank IC, mean', 'mean rank ic mean', 'ic'), ('Mean Rank IC, SD', 'mean rank ic sd', 'ic'),
    ('DA, mean', lambda r: None if r['model'] == 'zero' else r['da mean'], 'pctf'), ('DA, SD', 'da sd', 'pctf')])
x = s15[1]
x = x.assign(_w=x.window.map(lambda w: 9 if '+' in w else int(w[1]))).sort_values('_w', kind='stable').pipe(sort, 'input set', 'model', 'h')
x = x.sort_values('_w', kind='stable')
sh.panel('B. Rank IC inference', x, [
    ('Window', lambda r: WORD.get(r['window'], r['window']), 'txt'), ('Input specification', lambda r: INPUT[r['input set']], 'txt'),
    ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Role', lambda r: {'PRIMARY': 'Primary', 'SECONDARY': 'Secondary', 'component_reference': 'Per window',
                        'descriptive': 'Descriptive'}[r['family']], 'txt'), ('Mean Rank IC', 'mean', 'ic'),
    ('NW t', 'nw t', 't'), ('NW p (two-sided)', 'p nw', 'p'), ('NW Holm p', 'p nw holm', 'p'), ('MBB p', 'p mbb', 'p'),
    ('MBB Holm p', 'p mbb holm', 'p'), ('95% CI, lower', 'ci95 lower', 'ic'), ('95% CI, upper', 'ci95 upper', 'ic')])
x = s15[2]
x = x.assign(_w=x.window.map(lambda w: 9 if '+' in w else int(w[1])), _k=x.metric.map({'mse': 0, 'rank_ic': 1}))
x = x.sort_values(['_w', '_k'], kind='stable').pipe(sort, 'input set', 'h').sort_values(['_w', '_k'], kind='stable')
sh.panel('C. Static ' + MINUS + ' Adaptive contrasts', x, [
    ('Window', lambda r: WORD.get(r['window'], r['window']), 'txt'), ('Input specification', lambda r: INPUT[r['input set']], 'txt'),
    ('Metric', lambda r: METRIC[r['metric']], 'txt'), ('Horizon (days)', 'h', 'int'), ('Difference', 'estimate', 'd6'),
    ('95% CI, lower', 'ci95 lower', 'd6'), ('95% CI, upper', 'ci95 upper', 'd6'), ('p (two-sided)', 'p mbb', 'p'),
    ('Holm p', 'p mbb holm', 'p'), ('Difference (% of Adaptive MSE)', 'estimate pct second mse', 'pct'),
    ('95% CI, lower (%)', 'ci95 lower pct second mse', 'pct'), ('95% CI, upper (%)', 'ci95 upper pct second mse', 'pct'),
    ('Upper bound on Adaptive MSE gain (%)', 'adaptive mse improvement upper95 pct', 'pct'),
    ('Upper bound on Adaptive gain (metric units)', 'adaptive improvement upper95', 'd6'),
    ('Seed wins, Static', 'seed wins static', 'int'), ('Seed losses, Static', 'seed losses static', 'int')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S16
s16 = S[16]
sh = Sheet(wb, 16, 'Retrospective evaluation of T3 in W1 and W2, including ensemble accuracy, mean run-level Rank IC, and comparisons with T0', [
    'T3 (learning rate 3' + '×' + '10' + '⁻⁵' + ', Huber ' + 'δ' + ' = each window' + "'" + 's training-set SD of the h-day return) was selected on '
    'the primary window W3 and is evaluated in W1 and W2; window-relative inputs, Daily+Weekly. T0 is the baseline. '
    'Pooled tests use W1 and W2 together (782 test dates).',
    'Clark' + DASH + 'West tests of the five-seed ensemble forecasts are one-sided against each window' + "'" + 's Global Train-Mean '
    '(Newey' + DASH + 'West, lag h ' + MINUS + ' 1; MBB as robustness). Rank IC tests are two-sided (Newey' + DASH + 'West, lag h). '
    'Contrasts use the joint seed ' + '×' + ' date bootstrap. Holm adjustment within each family (see Section 3).'])
x = s16[0]
x = x.assign(_m=x.model.map(okey)).sort_values(['setting', 'window', '_m', 'h'])
sh.panel('A. Five-seed metrics', x, [
    ('Setting', 'setting', 'txt'), ('Window', 'window', 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Test MSE, mean', 'MSE mean (log-return squared)', 'mse'),
    ('Test MSE, SD', 'MSE seed SD (log-return squared)', 'mse'),
    ('R' + '²' + ' vs Global Train-Mean, mean', 'r2os train mean mean', 'pctf'), ('R' + '²' + ' vs Global Train-Mean, SD', 'r2os train mean sd', 'pctf'),
    ('R' + '²' + ' vs zero, mean', 'r2os zero mean', 'pctf'), ('R' + '²' + ' vs zero, SD', 'r2os zero sd', 'pctf'),
    ('Mean Rank IC, mean', 'mean rank ic mean', 'ic'), ('Mean Rank IC, SD', 'mean rank ic sd', 'ic'),
    ('DA, mean', 'da mean', 'pctf'), ('DA, SD', 'da sd', 'pctf')])
x = s16[1]
x = x.assign(_m=x.model.map(okey)).sort_values(['setting', 'window', '_m', 'h'])
sh.panel('B. Five-seed ensemble forecasts', x, [
    ('Setting', 'setting', 'txt'), ('Window', 'window', 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Test MSE', 'mse', 'mse'), ('R' + '²' + ' vs Global Train-Mean', 'r2os train mean', 'pctf'),
    ('R' + '²' + ' vs zero', 'r2os zero', 'pctf'), ('Mean Rank IC', 'mean rank ic', 'ic'), ('DA', 'da', 'pctf')])
# C: CW and Rank IC tests
e2 = s16[2].assign(test='Clark' + DASH + 'West vs Global Train-Mean (ensemble)', setting='T3', fam='Confirmatory (E2)')
e1 = s16[5].assign(test='Mean Rank IC', fam=s16[5]['role'].map({'SECONDARY': 'Secondary (E1)', 'T0 descriptive; no flag': 'Descriptive'}))
lst = s16[12].assign(test=s16[12]['test'].map({'seed_mean_rank_ic': 'Mean Rank IC',
                                               'ensemble_cw_vs_train_mean': 'Clark' + DASH + 'West vs Global Train-Mean (ensemble)'}),
                     fam='Confirmatory (LSTM)')
cc = ['fam', 'test', 'setting', 'model', 'h', 'mean', 'nw t', 'p nw', 'p nw holm', 'p mbb', 'p mbb holm', 'ci95 lower', 'ci95 upper']
x = pd.concat([e2[cc], lst[cc], e1[cc]])
x['_t'] = x['test'].map(lambda t: 0 if t.startswith('Clark') else 1)
x = x.assign(_m=x.model.map(okey)).sort_values(['_t', 'fam', 'setting', '_m', 'h'], kind='stable')
x = x.sort_values(['_t'], kind='stable')
sh.panel('C. Clark' + DASH + 'West and Rank IC tests, W1 and W2 pooled', x, [
    ('Test', 'test', 'txt'), ('Family', 'fam', 'txt'), ('Setting', 'setting', 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Mean statistic', 'mean', 'd6'), ('NW t', 'nw t', 't'), ('NW p', 'p nw', 'p'),
    ('NW Holm p', 'p nw holm', 'p'), ('MBB p', 'p mbb', 'p'), ('MBB Holm p', 'p mbb holm', 'p'),
    ('95% CI, lower', 'ci95 lower', 'd6'), ('95% CI, upper', 'ci95 upper', 'd6')])
e3 = s16[3].assign(fam='Confirmatory (E3)', mdl=s16[3]['model'].map(lab_model))
e4 = s16[4].assign(fam='Confirmatory (E4)', mdl='Static vs Adaptive', **{'adaptive mse improvement upper95 pct': s16[4]['adaptive mse improvement upper95 pct']})
cc = ['fam', 'contrast', 'mdl', 'metric', 'h', 'estimate', 'ci95 lower', 'ci95 upper', 'p mbb', 'p mbb holm', 'estimate pct',
      'ci95 lower pct', 'ci95 upper pct']
x = pd.concat([e3[cc], e4[cc + ['adaptive mse improvement upper95 pct']]])
x['_k'] = x['metric'].map({'mse': 0, 'rank_ic': 1})
x = x.sort_values(['fam', 'mdl', '_k', 'h'], kind='stable')
sh.panel('D. Paired contrasts, W1 and W2 pooled', x, [
    ('Family', 'fam', 'txt'), ('Contrast', lambda r: lab_contrast(r['contrast']) if 'T3' not in str(r['contrast']) else 'T3 ' + MINUS + ' T0', 'txt'),
    ('Model', 'mdl', 'txt'), ('Metric', lambda r: METRIC[r['metric']], 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Difference', 'estimate', 'd6'), ('95% CI, lower', 'ci95 lower', 'd6'), ('95% CI, upper', 'ci95 upper', 'd6'),
    ('p (two-sided)', 'p mbb', 'p'), ('Holm p', 'p mbb holm', 'p'), ('Difference (% of reference MSE)', 'estimate pct', 'pct'),
    ('95% CI, lower (%)', 'ci95 lower pct', 'pct'), ('95% CI, upper (%)', 'ci95 upper pct', 'pct'),
    ('Upper bound on Adaptive MSE gain (%)', 'adaptive mse improvement upper95 pct', 'pct')])
x = s16[6]
sh.panel('E. Mean Rank IC under T3 over all three windows (descriptive; includes the selection window W3)', x, [
    ('Model', lambda r: lab_model(r['model']), 'txt'), ('Horizon (days)', 'h', 'int'), ('Test dates', 'n dates', 'thou'),
    ('Mean Rank IC', 'mean rank ic', 'ic'), ('NW SE', 'nw se', 'ic'), ('95% CI, lower', 'nw ci95 lower', 'ic'),
    ('95% CI, upper', 'nw ci95 upper', 'ic')])
x = s16[8]
x = x.assign(_m=x.model.map(okey)).sort_values(['setting', 'window', '_m', 'h'])
sh.panel('F. Selected (best) epoch across seeds', x, [
    ('Setting', 'setting', 'txt'), ('Window', 'window', 'txt'), ('Model', lambda r: lab_model(r['model']), 'txt'),
    ('Horizon (days)', 'h', 'int'), ('Minimum', 'best epoch min', 'int'), ('Median', 'best epoch median', 'ep'),
    ('Maximum', 'best epoch max', 'int'), ('Mean', 'best epoch mean', 'ep'), ('SD', 'best epoch sd', 'f3')])
x = s16[9].sort_values(['setting', 'window', 'h', 'branch'])
sh.panel('G. Movement of the Static scale logits and weights (five-seed means)', x, [
    ('Setting', 'setting', 'txt'), ('Window', 'window', 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Branch', lambda r: str(r['branch']).capitalize(), 'txt'), ('Mean |centered logit|, mean', 'centered logit mad mean', 'w'),
    ('Mean |centered logit|, SD', 'centered logit mad sd', 'w'), ('Mean |weight ' + MINUS + ' 0.25|, mean', 'weight mad mean', 'w'),
    ('Mean |weight ' + MINUS + ' 0.25|, SD', 'weight mad sd', 'w')])
x = s16[14]
sh.panel('H. Window-specific references', x, [
    ('Window', 'window', 'txt'), ('Horizon (days)', 'h', 'int'), ('Training observations', 'n train', 'thou'),
    ('Test observations', 'n test', 'thou'), ('First test date', 'first test date', 'txt'), ('Last test date', 'last test date', 'txt'),
    ('Training mean', 'train mean', 'd6'), ('Global Train-Mean test MSE', 'train mean test mse', 'mse'), ('Zero forecast test MSE', 'zero test mse', 'mse')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S17
s17 = S[17]
ARM_ORDER = ['Fixed', 'Static', 'Adaptive', 'Static-R10', 'Static-R100', 'Adaptive-R100']
sh = Sheet(wb, 17, 'Sensitivity to increased aggregation learning rates', [
    'Exploratory. Window-relative inputs, primary window W3, T0 otherwise. R10/R100: learning rate of the aggregation parameters '
    '(Static scale logits, Adaptive router) 10 or 100 times the base rate; all other parameters unchanged.',
    'Manipulation check: the Static weights are required to move by a five-seed mean |weight ' + MINUS + ' 0.25| of at least 0.005 in both '
    'branches; a failed check limits interpretation but does not change the tests. Contrasts: seed ' + '×' + ' date bootstrap '
    '(B = 20,000, block length h), two-sided, Holm across horizons. Primary family: Static-R10 and Static-R100 against Fixed '
    '(equal weights); secondary family: the remaining contrasts.'])
x = s17[0].assign(_a=s17[0]['arm'].map(ARM_ORDER.index)).sort_values(['_a', 'h'])
sh.panel('A. Five-seed metrics', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Test MSE, mean', 'MSE mean (log-return squared)', 'mse'),
    ('Test MSE, SD', 'MSE seed SD (log-return squared)', 'mse'), ('R' + '²' + ' vs Global Train-Mean, mean', 'r2os mean', 'pctf'),
    ('R' + '²' + ', SD', 'r2os sd', 'pctf'), ('Mean Rank IC, mean', 'rank ic mean', 'ic'), ('Mean Rank IC, SD', 'rank ic sd', 'ic'),
    ('DA, mean', 'da mean', 'pctf'), ('DA, SD', 'da sd', 'pctf')])
x = s17[1].assign(_a=s17[1]['arm'].map(ARM_ORDER.index)).sort_values(['_a', 'h'])
sh.panel('B. Five-seed ensemble forecasts', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Test MSE', 'mse', 'mse'),
    ('R' + '²' + ' vs Global Train-Mean', 'r2os', 'pctf'), ('Mean Rank IC', 'mean rank ic', 'ic'), ('DA', 'da', 'pctf')])
x = s17[2].assign(_a=s17[2]['arm'].map(ARM_ORDER.index)).sort_values(['_a', 'h', 'branch'])
sh.panel('C. Movement of the aggregation weights (five-seed means)', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Branch', lambda r: str(r['branch']).capitalize(), 'txt'),
    ('Mean |weight ' + MINUS + ' 0.25|, mean', 'weight mad mean', 'w'), ('Mean |weight ' + MINUS + ' 0.25|, SD', 'weight mad sd', 'w'),
    ('Within-model SD of weights, mean', 'within model sd mean', 'w'), ('Mean |centered logit|, mean', 'centered logit mad mean', 'w'),
    ('Seeds with |weight ' + MINUS + ' 0.25| ' + '≥' + ' 0.005', 'n seeds mad ge 005', 'int'),
    ('Weight 1', 'weight 1 mean', 'w'), ('Weight 2', 'weight 2 mean', 'w'), ('Weight 3', 'weight 3 mean', 'w'), ('Weight 4', 'weight 4 mean', 'w')])
x = s17[3].assign(_a=s17[3]['arm'].map(ARM_ORDER.index)).sort_values(['_a', 'h'])
sh.panel('D. Manipulation check', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Daily: mean |weight ' + MINUS + ' 0.25|', 'daily five seed mean mad', 'w'),
    ('Weekly: mean |weight ' + MINUS + ' 0.25|', 'weekly five seed mean mad', 'w'), ('Threshold', 'threshold', 'f3'),
    ('Result', lambda r: {'PASS': 'Passed', 'FAIL': 'Failed'}.get(r['status'], r['status']), 'txt')])
x = s17[4]
x = x.assign(_k=x.metric.map({'mse': 0, 'rank_ic': 1, 'da': 2})).sort_values(['family', 'contrast', '_k', 'h'], kind='stable')
sh.panel('E. Paired contrasts', x, [
    ('Family', lambda r: fam(r['family']), 'txt'), ('Contrast', lambda r: r['first'] + ' ' + MINUS + ' ' + r['second'], 'txt'),
    ('Metric', lambda r: METRIC[r['metric']], 'txt'), ('Horizon (days)', 'h', 'int'), ('Difference', 'theta hat', 'd6'),
    ('95% CI, lower', 'ci lower 95', 'd6'), ('95% CI, upper', 'ci upper 95', 'd6'), ('p (two-sided)', 'centered p two sided', 'p'),
    ('Holm p (3 horizons)', 'p holm 3h', 'p'), ('Seed wins, first arm', 'seed wins', 'int'), ('Seed losses, first arm', 'seed losses', 'int'),
    ('Difference (% of reference MSE)', 'estimate pct', 'pct'),
    ('Upper bound on MSE gain of first arm (%)', 'mse improvement upper95 pct', 'pct'), ('Manipulation check', lambda r: {'PASS': 'Passed', 'FAIL': 'Failed'}.get(r['manipulation status'], r['manipulation status']), 'txt')])
sheets.append(sh)

# ----------------------------------------------------------------------------- S18
s18 = S[18]
ARM18 = ['Static-T0', 'Adaptive-T0', 'LSTM-T0', 'Static-T3', 'Adaptive-T3', 'LSTM-T3', 'Ridge-CS', 'Static-CS', 'Adaptive-CS', 'LSTM-CS']


def aord(a):
    return ARM18.index(a) if a in ARM18 else 99


sh = Sheet(wb, 18, 'Cross-sectionally demeaned target analysis', [
    'Exploratory. Target y_cs = h-day log return minus its same-date mean over the 17 stocks; window-relative inputs, primary window W3. '
    'CS arms are trained on y_cs with learning rate 3' + '×' + '10' + '⁻⁵' + ' and Huber ' + 'δ' + ' = training SD of y_cs; '
    'Ridge-CS is deterministic. T0/T3 arms are the models trained on the raw return (Table S14).',
    'MSE and R' + '²' + ' are measured on y_cs against the zero forecast; Rank IC is against realized returns; DA excludes y_cs = 0. '
    'Portfolios are descriptive (10 bp per unit of turnover; the long' + DASH + 'short portfolio excludes borrowing costs and cash interest).',
    'Rank IC tests: primary family Static-CS and Adaptive-CS, benchmark LSTM-CS; Holm adjustment within family across horizons. '
    'Contrast families: A, CS target versus raw-return target (T3); B, Static-CS ' + MINUS + ' Adaptive-CS; C, CS models versus LSTM-CS.'])
sh.panel('A. Training-set dispersion of the target', s18[0], [
    ('Horizon (days)', 'h', 'int'), ('Training observations', 'n train', 'thou'), ('Training SD of y_cs', 'sigma cs', 'd6')])
x = s18[1].assign(_a=s18[1]['arm'].map(aord)).sort_values(['_a', 'h'])
sh.panel('B. Five-seed metrics', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Seeds', 'n seeds', 'int'), ('Test MSE, mean', 'MSE mean (log-return squared)', 'mse'),
    ('Test MSE, SD', 'MSE seed SD (log-return squared)', 'mse'), ('R' + '²' + ' vs zero, mean', 'r2os cs zero mean', 'pctf'),
    ('R' + '²' + ' vs zero, SD', 'r2os cs zero sd', 'pctf'), ('Mean Rank IC, mean', 'mean rank ic mean', 'ic'),
    ('Mean Rank IC, SD', 'mean rank ic sd', 'ic'), ('DA, mean', 'da mean', 'pctf'), ('DA, SD', 'da sd', 'pctf')])
x = s18[2].assign(_a=s18[2]['arm'].map(aord)).sort_values(['_a', 'h'])
sh.panel('C. Five-seed ensemble forecasts', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Test MSE', 'mse', 'mse'), ('R' + '²' + ' vs zero', 'r2os cs zero', 'pctf'),
    ('Mean Rank IC', 'mean rank ic', 'ic'), ('DA', 'da', 'pctf')])
x = s18[3].assign(_a=s18[3]['arm'].map(aord)).sort_values(['_a', 'h'])
sh.panel('D. Mean Rank IC tests (two-sided)', x, [
    ('Family', lambda r: fam(r['family']), 'txt'), ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'),
    ('Mean Rank IC', 'mean rank ic', 'ic'), ('NW t', 'nw t', 't'), ('NW p', 'p nw', 'p'), ('NW Holm p', 'p nw holm', 'p'),
    ('MBB p', 'p mbb', 'p'), ('MBB Holm p', 'p mbb holm', 'p'), ('95% CI, lower (NW)', 'nw ci95 lower', 'ic'),
    ('95% CI, upper (NW)', 'nw ci95 upper', 'ic')])
x = s18[4]
x = x.assign(_k=x.metric.map({'mse': 0, 'rank_ic': 1, 'da': 2})).sort_values(['family', 'contrast', '_k', 'h'], kind='stable')
sh.panel('E. Paired contrasts', x, [
    ('Family', lambda r: fam(r['family']), 'txt'), ('Contrast', lambda r: r['first'] + ' ' + MINUS + ' ' + r['second'], 'txt'),
    ('Metric', lambda r: METRIC[r['metric']], 'txt'), ('Horizon (days)', 'h', 'int'), ('Difference', 'theta hat', 'd6'),
    ('95% CI, lower', 'ci lower 95', 'd6'), ('95% CI, upper', 'ci upper 95', 'd6'), ('p (two-sided)', 'centered p two sided', 'p'),
    ('Holm p (3 horizons)', 'p holm 3h', 'p'), ('Seed wins, first arm', 'seed wins', 'int'), ('Seed losses, first arm', 'seed losses', 'int'),
    ('Difference (% of reference MSE)', 'estimate pct', 'pct'), ('Upper bound on Adaptive MSE gain (%)', 'adaptive mse improvement upper95 pct', 'pct')])
x = s18[5].assign(_a=s18[5]['arm'].map(aord)).sort_values(['_a', 'h'])
sh.panel('F. Mean Rank IC by training target and setting', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Mean Rank IC, mean', 'rank ic mean', 'ic'), ('Mean Rank IC, SD', 'rank ic sd', 'ic')])
x = s18[6].assign(_a=s18[6]['arm'].map(aord)).sort_values(['_a', 'h', 'branch'])
sh.panel('G. Movement of the aggregation weights (five-seed means)', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Branch', lambda r: str(r['branch']).capitalize(), 'txt'),
    ('Mean |weight ' + MINUS + ' 0.25|', 'weight mad', 'w'), ('Within-model SD of weights', 'within model sd', 'w'),
    ('Mean |centered logit|', 'centered logit mad', 'w'), ('Seeds with observation-invariant weights', 'n observation invariant', 'int')])
x = s18[8].assign(_a=s18[8]['arm'].map(aord)).sort_values(['_a', 'h', 'portfolio'])
PORT = {'top5_long_only': 'Top-5 long only', 'top5_minus_bottom5': 'Top-5 minus bottom-5'}
sh.panel('H. Ranking portfolios (descriptive)', x, [
    ('Arm', 'arm', 'txt'), ('Horizon (days)', 'h', 'int'), ('Portfolio', lambda r: PORT[r['portfolio']], 'txt'),
    ('Ann. return', 'ann geo return', 'f4'), ('Ann. volatility', 'ann vol', 'f4'), ('Sharpe ratio', 'sharpe', 'sh'),
    ('Max. drawdown', 'max drawdown', 'f4'), ('Turnover per rebalance', 'avg turnover ex initial', 'f4'),
    ('CE return, A = 2', 'ce A2', 'f4'), ('CE return, A = 5', 'ce A5', 'f4')])
sheets.append(sh)

# ----------------------------------------------------------------------------- contents
for s in sheets:
    s.finish()
ws = contents
ws.sheet_view.showGridLines = False
ws.cell(1, 1, 'Supplementary Tables S1' + DASH + 'S18').font = TITLE
ws.cell(2, 1, 'An Adaptive Multi-Scale Transformer for Green-Energy Stock Return Forecasting (manuscript computers-4613491)').font = NOTE
r = 4
for t in ['General notes',
          'Panel of 17 green-energy stocks; forecast horizons h = 5, 10 and 20 trading days. Primary test window W3 (12 November 2024 to '
          '5 June 2026); earlier windows W1 and W2 are defined in Table 3 of the manuscript.',
          'Input specifications: price-level inputs (standardized with training-period statistics) and window-relative inputs '
          '(prices relative to the forecast-date close). Model names follow Tables 2 and 4.',
          COMMON_NOTE,
          'Inference: Clark' + DASH + 'West (CW) adjusted test for nested forecasts; Newey' + DASH + 'West (NW) standard errors; moving block bootstrap (MBB); '
          'seed ' + '×' + ' date = hierarchical bootstrap that resamples training seeds and blocks of test dates. Holm = Holm' + DASH + 'Bonferroni adjustment; '
          'adjusted p-values are capped at 1.',
          'DA = directional accuracy; IC = information coefficient (Spearman rank correlation between forecasts and realized returns across stocks on each date).',
          'Dataset construction scripts, model implementation and analysis code: ' + REPO]:
    c = ws.cell(r, 1, t)
    c.font = BOLD if t == 'General notes' else FONT
    c.alignment = Alignment(wrap_text=True, vertical='top')
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
    ws.row_dimensions[r].height = 14 if t == 'General notes' else 14 * max(1, math.ceil(len(t) / 150))
    r += 1
r += 1
for j, h in enumerate(['Sheet', 'Table', 'Panels'], 1):
    c = ws.cell(r, j, h)
    c.font, c.fill, c.alignment, c.border = BOLD, HFILL, WRAP_CENTER, Border(top=MED, bottom=THIN)
r += 1
for s in sheets:
    c = ws.cell(r, 1, 'S%d' % s.num)
    c.hyperlink = "#'S%d'!A1" % s.num
    c.font = Font(name='Arial', size=9, color='1F4E79', underline='single')
    c.alignment = LEFT
    ws.cell(r, 2, 'Table S%d. %s' % (s.num, s.title)).font = FONT
    ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical='top')
    pl = '; '.join(t.split(' (')[0] if len(t) > 90 else t for t, _ in s.panels) if len(s.panels) > 1 else ''
    c = ws.cell(r, 3, pl)
    c.font = FONT
    c.alignment = Alignment(wrap_text=True, vertical='top')
    ws.row_dimensions[r].height = 12 * max(1, math.ceil(len(pl) / 95), math.ceil(len(ws.cell(r, 2).value) / 60))
    r += 1
for j in range(1, 4):
    ws.cell(r - 1, j).border = Border(bottom=MED)
ws.column_dimensions['A'].width = 8
ws.column_dimensions['B'].width = 62
ws.column_dimensions['C'].width = 100
ws.column_dimensions['D'].width = 10
wb.properties.title = 'Supplementary Tables S1' + DASH + 'S18'
wb.properties.creator = 'Authors'
wb.save(OUT)
print('saved', OUT, [(s.num, sum(n for _, n in s.panels)) for s in sheets])
