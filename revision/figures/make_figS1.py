"""Figure S1 (redrawn): router diagnostics with price-level inputs, values from Tables S19 and S20.
usage: python3 make_figS1.py OUT_STEM -> OUT_STEM.png (600 dpi) and OUT_STEM.pdf
"""
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in glob.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf'):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 8.5, 'mathtext.fontset': 'stix',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42})

# (horizon, branch): sample TV, within-ticker TV, within-date TV, effective scales, mean max weight  [Table S19]
S19 = {(5, 'D'): (0.043434, 0.011443, 0.043107, 3.874252, 0.327476),
       (5, 'W'): (0.044880, 0.010754, 0.044600, 3.859643, 0.342920),
       (10, 'D'): (0.044964, 0.011975, 0.044729, 3.869684, 0.327516),
       (10, 'W'): (0.045741, 0.011448, 0.045386, 3.860532, 0.340791),
       (20, 'D'): (0.047926, 0.013467, 0.047493, 3.836305, 0.339682),
       (20, 'W'): (0.058269, 0.015133, 0.057794, 3.798510, 0.359753)}
# cross-seed pairwise TV  [Table S20]
S20 = {(5, 'D'): 0.151025, (5, 'W'): 0.157668, (10, 'D'): 0.152287, (10, 'W'): 0.151440,
       (20, 'D'): 0.171295, (20, 'W'): 0.175991}
H = [5, 10, 20]
KEYS = [(h, b) for h in H for b in 'DW']
BR = {'D': 'Daily', 'W': 'Weekly'}
C_D, C_W = '#0072B2', '#D55E00'
GREYS = ['#2b2b2b', '#8c8c8c', '#c9c9c9', '#009E73']

out = sys.argv[1]
fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.45), gridspec_kw={'width_ratios': [1.55, 1, 1]})

# (a) four variation measures, grouped by horizon-branch
ax = axes[0]
meas = [('Sample TV', [S19[k][0] for k in KEYS]), ('Within-ticker TV', [S19[k][1] for k in KEYS]),
        ('Within-date TV', [S19[k][2] for k in KEYS]), ('Cross-seed TV', [S20[k] for k in KEYS])]
x = np.arange(len(KEYS))
w = 0.2
for i, (name, v) in enumerate(meas):
    ax.bar(x + (i - 1.5) * w, v, w, color=GREYS[i], label=name, lw=0, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(['%d%s' % (h, b) for h, b in KEYS])
ax.set_xlabel('Horizon (days) and branch (D = Daily, W = Weekly)')
ax.set_ylabel('Total variation distance')
ax.set_ylim(0, 0.245)
ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
ax.yaxis.grid(True, color='#e3e3e3', lw=0.5, zorder=0)
ax.legend(frameon=False, fontsize=7.3, loc='upper left', ncol=2, handlelength=1.0, columnspacing=0.9,
          borderaxespad=0.2)
ax.set_title('(a) Routing variation', loc='left', fontsize=8.5, pad=4)

# (b) effective number of scales
ax = axes[1]
for b, col, mk in (('D', C_D, 'o'), ('W', C_W, 's')):
    ax.plot(H, [S19[(h, b)][3] for h in H], marker=mk, color=col, ms=3.6, lw=1.0, label=BR[b], zorder=3)
ax.axhline(4.0, ls=(0, (4, 2.5)), color='#6b6b6b', lw=0.8)
ax.text(20.6, 3.993, 'maximum (4)', ha='right', va='top', fontsize=7.3, color='#555555')
ax.set_ylim(3.70, 4.02)
ax.set_ylabel('Effective number of scales')
ax.set_title('(b) Effective number of scales', loc='left', fontsize=8.5, pad=4)

# (c) mean maximum weight
ax = axes[2]
for b, col, mk in (('D', C_D, 'o'), ('W', C_W, 's')):
    ax.plot(H, [S19[(h, b)][4] for h in H], marker=mk, color=col, ms=3.6, lw=1.0, label=BR[b], zorder=3)
ax.axhline(1.0, ls=(0, (4, 2.5)), color='#6b6b6b', lw=0.8)
ax.text(20.6, 0.975, 'one scale only (1.0)', ha='right', va='top', fontsize=7.3, color='#555555')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Mean maximum weight')
ax.set_title('(c) Mean maximum weight', loc='left', fontsize=8.5, pad=4)
ax.legend(frameon=False, fontsize=7.3, loc='center right')

for ax in axes[1:]:
    ax.set_xticks(H)
    ax.set_xlim(3.5, 21.5)
    ax.set_xlabel('Horizon (days)')
    ax.yaxis.grid(True, color='#e3e3e3', lw=0.5, zorder=0)
for ax in axes:
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.2, wspace=0.42)
fig.savefig(out + '.png', dpi=600)
fig.savefig(out + '.pdf')
print('saved', out)
