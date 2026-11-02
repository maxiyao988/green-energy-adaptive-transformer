"""Figure 5: inference-time router interventions (replaces former Table 8).

Data: Supplementary Table S7 (stored seed x date bootstrap inference, squared loss, I1-I4 and I6 vs I0).
Point = five-seed mean paired squared-loss difference, bar = 95% seed x date bootstrap interval, small dots =
per-seed differences; all expressed as % of the unmodified model's five-seed mean test MSE.
usage: python3 make_fig5_interventions.py OUT_STEM SUPP_XLSX
"""
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl
from matplotlib import font_manager
from matplotlib.lines import Line2D

for f in glob.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf'):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 8.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'xtick.major.size': 2.5, 'pdf.fonttype': 42})

out_stem, xlsx = sys.argv[1], sys.argv[2]
rows = list(openpyxl.load_workbook(xlsx, read_only=True)['Table S7'].iter_rows(values_only=True))
hi = [i for i, r in enumerate(rows) if r and 'record type' in r][0]
hdr = rows[hi]
recs = [dict(zip(hdr, r)) for r in rows[hi + 1:] if r and r[0]]

base = {(d['input set'], d['h']): d['mse mean'] for d in recs
        if d['record type'] == 'five_seed_summary' and d['intervention'] == 'I0_learned' and d['split'] == 'test'}
INTS = [('I1_equal', 'I1: Equal weights'), ('I2_global_train_mean', 'I2: Training average'),
        ('I3_ticker_train_mean', 'I3: Per-stock average'), ('I4_date_mean', 'I4: Same-date mean'),
        ('I6_equal_daily_only', 'I6a: Equal, Daily branch'),
        ('I6_equal_weekly_only', 'I6b: Equal, Weekly branch')]
joint, seeds = {}, {}
for d in recs:
    if d['record type'] != 'stored_inference' or d['metric'] != 'squared_loss' or not d['contrast']:
        continue
    key = d['contrast'].split('−')[0]
    k = (d['input set'], key, d['h'])
    if d['method'] == 'joint_seed_date':
        joint[k] = d
    elif d['method'] == 'per_seed':
        seeds.setdefault(k, []).append(d['theta hat'])

SPECS = [('O', 'Price-level'), ('C', 'Window-relative')]
H = [5, 10, 20]
fig, axes = plt.subplots(2, 3, figsize=(6.7, 4.3), sharey=True)
letters = 'abcdef'
ypos = list(range(len(INTS)))[::-1]
table = []
for r, (spec, name) in enumerate(SPECS):
    lim = 0
    for c, h in enumerate(H):
        ax = axes[r, c]
        b = base[(spec, h)]
        ax.axvline(0, color='#6b6b6b', lw=0.8, ls=(0, (4, 2.5)), zorder=1)
        for y, (key, lab) in zip(ypos, INTS):
            d = joint[(spec, key, h)]
            mu, lo, up = (100 * d[x] / b for x in ('theta hat', 'ci lower 95', 'ci upper 95'))
            sd = [100 * v / b for v in seeds[(spec, key, h)]]
            assert len(sd) == 5
            sig = d['p holm 3h'] < 0.05
            col = '#D55E00' if sig else '#2b2b2b'
            ax.scatter(sd, [y] * 5, s=7, color='#9a9a9a', alpha=0.7, lw=0, zorder=2)
            ax.plot([lo, up], [y, y], color=col, lw=1.1, zorder=3)
            ax.plot(mu, y, 'o', color=col, ms=3.6, zorder=4)
            lim = max(lim, abs(lo), abs(up), *[abs(v) for v in sd])
            table.append((spec, key, h, round(1e4 * d['theta hat'], 2), d['seed wins'], round(d['p holm 3h'], 2)))
        ax.set_title('(%s)  %s, h = %d' % (letters[3 * r + c], name, h), loc='left', fontsize=8.5, pad=4)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.grid(axis='x', color='#e3e3e3', lw=0.5, zorder=0)
    for ax in axes[r]:
        ax.set_xlim(-lim * 1.08, lim * 1.08)
axes[0, 0].set_yticks(ypos)
axes[0, 0].set_yticklabels([lab for _, lab in INTS])
handles = [Line2D([], [], marker='o', color='#2b2b2b', ms=3.6, lw=1.1, label='Five-seed mean, 95% seed×date interval'),
           Line2D([], [], marker='o', color='#9a9a9a', ms=3, ls='none', label='Individual seeds')]
fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.62, 1.0))
fig.subplots_adjust(left=0.205, right=0.985, top=0.875, bottom=0.11, wspace=0.12, hspace=0.42)
fig.text(0.61, 0.012, 'Change in test MSE relative to the unmodified model (%; positive = intervention worse)',
         ha='center', va='bottom', fontsize=8.5)
fig.savefig(out_stem + '.png', dpi=600)
fig.savefig(out_stem + '.pdf')
for t in table:
    print(t)
