"""Figure 4 (revised): 20-day learning curves of Static and Adaptive, Daily+Weekly, T0.

Data: Step19 learning_history_cells.csv (per seed, per epoch train/validation Huber loss).
Thin lines: individual seeds; thick lines: five-seed means over the epochs reached by all five runs.
usage: python3 make_fig4_v2.py OUT_STEM CSV
"""
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

for f in glob.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf'):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 8.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'pdf.fonttype': 42})
out_stem, csv = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
d = d[(d.horizon == 20) & d.mechanism.isin(['static', 'adaptive'])]
COL = {'static': '#0072B2', 'adaptive': '#D55E00'}
fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
for ax, (spec, name, let) in zip(axes, [('O', 'Price-level inputs', 'a'), ('C', 'Window-relative inputs', 'b')]):
    s = d[d.input_set == spec]
    assert s.seed.nunique() == 5
    for mech in ('static', 'adaptive'):
        m = s[s.mechanism == mech]
        for _, g in m.groupby('seed'):
            g = g.sort_values('epoch')
            ax.plot(g.epoch, g.train_huber, color=COL[mech], lw=0.5, alpha=0.3)
            ax.plot(g.epoch, g.val_huber, color=COL[mech], lw=0.5, alpha=0.3, ls=(0, (3, 1.5)))
        agg = m.groupby('epoch').agg(tr=('train_huber', 'mean'), va=('val_huber', 'mean'), n=('seed', 'nunique'))
        agg = agg[agg.n == 5]
        ax.plot(agg.index, agg.tr, color=COL[mech], lw=1.6)
        ax.plot(agg.index, agg.va, color=COL[mech], lw=1.6, ls=(0, (3, 1.5)))
    ax.axvline(1, color='#7a7a7a', lw=0.7, ls=':')
    ax.set_title('(%s)  %s' % (let, name), loc='left', fontsize=8.5, pad=4)
    ax.set_xlabel('Epoch')
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13])
    ax.set_xlim(0.6, s.epoch.max() + 0.4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(color='#e6e6e6', lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel('Huber loss')
handles = [Line2D([], [], color=COL['static'], lw=1.6, label='Static'),
           Line2D([], [], color=COL['adaptive'], lw=1.6, label='Adaptive'),
           Line2D([], [], color='#444444', lw=1.2, label='Training'),
           Line2D([], [], color='#444444', lw=1.2, ls=(0, (3, 1.5)), label='Validation'),
           Line2D([], [], color='#444444', lw=0.5, alpha=0.5, label='Individual seeds')]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.53, 1.0),
           handlelength=2.0, columnspacing=1.4)
fig.subplots_adjust(left=0.09, right=0.99, top=0.83, bottom=0.15, wspace=0.22)
fig.savefig(out_stem + '.png', dpi=600)
fig.savefig(out_stem + '.pdf')
print('saved')
