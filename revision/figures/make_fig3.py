"""Figure 3 (revised layout): horizontal dot plot, 2 input specifications x 3 horizons.

Values are the Step26 five-seed means and between-seed SDs (Figure3_data_step26.csv), already expressed
as ratios to the Global Train-Mean test MSE. Model names and order follow Table 4.
usage: python3 make_fig3_v2.py OUT_STEM DATA_CSV   -> OUT_STEM.png (600 dpi) and OUT_STEM.pdf
"""
import csv
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

for f in glob.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf'):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 8.5, 'mathtext.fontset': 'stix',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.0,
                     'xtick.major.size': 2.5, 'pdf.fonttype': 42})

out_stem, data_csv = sys.argv[1], sys.argv[2]
rows = {(r['input_set'], r['model'], int(r['horizon'])): r for r in csv.DictReader(open(data_csv))}
assert len(rows) == 54

# Table 4 order: benchmarks first, then the aggregation variants
MODELS = [('ridge', 'Ridge'), ('lstm', 'LSTM'), ('vanilla', 'Vanilla Transformer'),
          ('swim', 'SWiM-style Transformer'),
          ('single', 'Single'), ('fixed', 'Fixed'), ('same_patch_fixed', 'Same-patch Fixed'),
          ('static', 'Static'), ('adaptive', 'Adaptive')]
N_BENCH = 4
GAP = 0.7
ypos = [-(i + (GAP if i >= N_BENCH else 0)) for i in range(len(MODELS))]
COL = {'static': '#0072B2', 'adaptive': '#D55E00', 'ridge': '#009E73'}
DARK = '#2b2b2b'

SPECS = [('O', 'Price-level inputs', (0.985, 1.62), [1.0, 1.2, 1.4, 1.6]),
         ('C', 'Window-relative inputs', (0.994, 1.064), [1.00, 1.02, 1.04, 1.06])]
H = [5, 10, 20]

fig, axes = plt.subplots(2, 3, figsize=(6.7, 4.75), sharey=True)
letters = 'abcdef'
for r, (spec, spec_name, xlim, xt) in enumerate(SPECS):
    for c, h in enumerate(H):
        ax = axes[r, c]
        # light band behind the aggregation variants
        ax.axhspan(ypos[-1] - 0.5, ypos[N_BENCH] + 0.5, color='#f1f1f1', zorder=0, lw=0)
        ax.axvline(1.0, ls=(0, (4, 2.5)), color='#6b6b6b', lw=0.8, zorder=1)
        for k, (key, _) in enumerate(MODELS):
            row = rows[(spec, key, h)]
            mu = float(row['mse_ratio'])
            color = COL.get(key, DARK)
            if key == 'ridge':
                assert row['mse_ratio_seed_sd'].startswith('N/A')
                ax.plot(mu, ypos[k], marker='^', color=color, ms=4.6, ls='none', zorder=4)
            else:
                sd = float(row['mse_ratio_seed_sd'])
                ax.errorbar(mu, ypos[k], xerr=sd, fmt='o', color=color, ms=3.6, capsize=2.2,
                            elinewidth=1.0, capthick=1.0, zorder=4)
        ax.set_xlim(*xlim)
        ax.set_xticks(xt)
        ax.set_xticklabels(['%.1f' % v if spec == 'O' else '%.2f' % v for v in xt])
        for v in xt:
            if abs(v - 1.0) > 1e-9:
                ax.axvline(v, color='#dedede', lw=0.5, zorder=0)
        ax.set_ylim(ypos[-1] - 0.6, ypos[0] + 0.6)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.set_title('(%s)  %s, h = %d' % (letters[3 * r + c], spec_name.split(' inputs')[0], h),
                     loc='left', fontsize=8.5, pad=4)

ax0 = axes[0, 0]
ax0.set_yticks(ypos)
ax0.set_yticklabels([m for _, m in MODELS])
for ax in axes[:, 0]:
    for lab, (key, _) in zip(ax.get_yticklabels(), MODELS):
        if key in COL:
            lab.set_color(COL[key])

# group labels in the left margin
for ax in axes[:, 0]:
    for (lo, hi), name in [((ypos[N_BENCH - 1], ypos[0]), 'Benchmarks'),
                           ((ypos[-1], ypos[N_BENCH]), 'Aggregation variants')]:
        mid = (lo + hi) / 2
        ax.annotate(name, xy=(-0.95, mid), xycoords=('axes fraction', 'data'), rotation=90, ha='center',
                    va='center', fontsize=7.5, style='italic', color='#555555', annotation_clip=False)
        ax.annotate('', xy=(-0.88, lo - 0.35), xytext=(-0.88, hi + 0.35),
                    xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.6), annotation_clip=False)

handles = [Line2D([], [], marker='o', color=DARK, ms=3.6, lw=1.0, label='Five-seed mean ± 1 SD'),
           Line2D([], [], marker='^', color=COL['ridge'], ms=4.6, ls='none', label='Ridge (deterministic)'),
           Line2D([], [], ls=(0, (4, 2.5)), color='#6b6b6b', lw=0.8, label='Global Train-Mean')]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.62, 1.0), handlelength=2.2, columnspacing=1.8)
fig.subplots_adjust(left=0.235, right=0.985, top=0.885, bottom=0.095, wspace=0.12, hspace=0.36)
fig.text(0.61, 0.012, 'Test MSE relative to the Global Train-Mean', ha='center', va='bottom', fontsize=8.5)
fig.savefig(out_stem + '.png', dpi=600)
fig.savefig(out_stem + '.pdf')
print('saved', out_stem)
