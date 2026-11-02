"""Prompt 20: plot only the five authorized stored data files.

No metrics, summaries, models, tests or intervals are computed.  Multiplication
by 100 in Fig5 is solely the requested axis-unit conversion, including its SD.
Every final artifact uses exclusive creation.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import os

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'final'
OUT.mkdir(exist_ok=False)
assert ROOT.name=='pathformer'
assert not any((OUT/f'Fig{i}.png').exists() or (OUT/f'Fig{i}.pdf').exists() for i in range(3,8))
os.environ['MPLCONFIGDIR']=str(OUT/('matplotlib_cache_'+Path(__file__).stem))
os.environ['OMP_NUM_THREADS']='1'
SOURCES={
 'f3':'revision_work/outputs/manuscript19/F3_complete_data.csv',
 'curves':'revision_work/outputs/manuscript19/learning_mean_by_epoch.csv',
 'matched':'revision_work/outputs/manuscript18/F4_stored_plot_data.csv',
 'rolling':'revision_work/outputs/manuscript18/F5_stored_plot_data.csv',
 'portfolios':'revision_work/outputs/manuscript18/F6_stored_plot_data.csv',
}
FORBIDDEN='NOT'+' STORED'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(key):
    s=SOURCES[key]
    return [dict(r,_source=s,_line=i) for i,r in enumerate(csv.DictReader(io.StringIO((ROOT/s).read_text())),2)]


def one(rr,**kw):
    hits=[r for r in rr if all(str(r[k])==str(v) for k,v in kw.items())]
    assert len(hits)==1,(kw,len(hits))
    return hits[0]


def value(r,key):
    x=float(r[key]);assert math.isfinite(x),(r,key)
    return x


def optional(r,key):
    try:return value(r,key)
    except (ValueError,TypeError):return None


before={s:sha(ROOT/s) for s in SOURCES.values()}
data={k:read(k) for k in SOURCES}
trace=[]


def record(fig,r,fields,**extra):
    trace.append(dict(figure=fig,source_file=r['_source'],source_csv_line=r['_line'],stored_fields={k:r[k] for k in fields},**extra))


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.ticker import MaxNLocator, FuncFormatter, LogLocator, NullFormatter

font=font_manager.findfont('Times New Roman',fallback_to_default=False)
ft=FT2Font(font)
assert ft.get_char_index(ord('×')) and ft.get_char_index(ord('−')), 'Required Times glyph unavailable'
plt.rcParams.update({
 'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
 'axes.titlesize':9,'figure.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'axes.formatter.useoffset':False,'lines.linewidth':1.1,
 'legend.frameon':False,'axes.unicode_minus':True,
 'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
 'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
 'mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman',
 'mathtext.cal':'Times New Roman','mathtext.default':'regular','mathtext.fallback':None,
})
WIDTH=16/2.54
COLORS={'static':'#0072B2','adaptive':'#D55E00','ridge':'#009E73','zero':'#CC79A7'}
HEADERS=['Price-level inputs','Window-relative inputs']
planned=[]


def decimal(x,pos=None):
    if abs(x)<1e-12:return '0'
    return format(x,'.4f').rstrip('0').rstrip('.')


def plain(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(decimal))
    ax.yaxis.set_major_formatter(FuncFormatter(decimal))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_axisbelow(True)


def headers(fig,axs,y):
    for col,label in enumerate(HEADERS):
        ax=axs[col] if len(axs.shape)==1 else axs[0,col]
        pos=ax.get_position()
        fig.text((pos.x0+pos.x1)/2,y,label,ha='center',va='center')


def panels(height,left=.135,right=.98,wspace=.34,bottom=.08,top=.88,hspace=.46):
    fig,axs=plt.subplots(3,2,figsize=(WIDTH,height/2.54))
    fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)
    headers(fig,axs,.945)
    for i,ax in enumerate(axs.flat):
        plain(ax)
        ax.text(0,1.055,'('+chr(97+i)+')  h = '+str((5,10,20)[i//2]),transform=ax.transAxes,ha='left',va='bottom')
    return fig,axs


def check_and_hold(fig,name,caption,details):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        assert not ax.get_title(),name
        assert not ax.xaxis.get_offset_text().get_visible()
        assert not ax.yaxis.get_offset_text().get_visible()
        xt=[t for t in ax.get_xticklabels() if min(ax.get_xlim())<=t.get_position()[0]<=max(ax.get_xlim())]
        yt=[t for t in ax.get_yticklabels() if min(ax.get_ylim())<=t.get_position()[1]<=max(ax.get_ylim())]
        texts=ax.texts+xt+yt+[ax.xaxis.label,ax.yaxis.label]
        for t in texts:
            assert FORBIDDEN not in t.get_text(),(name,t.get_text())
            if not t.get_visible() or not t.get_text():continue
            b=t.get_window_extent(renderer)
            assert b.x0>=-2 and b.y0>=-2 and b.x1<=fig.bbox.width+2 and b.y1<=fig.bbox.height+2,(name,t.get_text(),b.bounds,fig.bbox.bounds)
    for t in fig.texts:
        assert FORBIDDEN not in t.get_text()
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,t.get_text(),b.bounds)
    for leg in fig.legends:
        b=leg.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,'legend',b.bounds)
    planned.append(dict(name=name,figure=fig,caption=caption,details=details))


# Fig3: models on x, requested relative-MSE label on y.
MODELS=['single','fixed','same_patch_fixed','static','adaptive','lstm','vanilla','swim','ridge']
MODEL_LABELS=['Single','Fixed','Same-patch','Static','Adaptive','LSTM','Vanilla','SWiM','Ridge']
fig,axs=panels(21,left=.15,bottom=.14,top=.90,hspace=.26,wspace=.32)
for i,ax in enumerate(axs.flat):
    h=(5,10,20)[i//2];inp=('O','C')[i%2]
    ax.axhline(1,color='#666666',linestyle='--',linewidth=.8)
    for j,model in enumerate(MODELS):
        r=one(data['f3'],input_set=inp,horizon=h,model=model)
        y=value(r,'mse_ratio');sd=optional(r,'mse_ratio_seed_sd')
        ax.errorbar(j,y,yerr=sd,fmt='^' if model=='ridge' else 'o',color=COLORS.get(model,'#444444'),markersize=3,capsize=2,elinewidth=1)
        record('Fig3',r,['input_set','horizon','model','n_seeds','mse_ratio']+(['mse_ratio_seed_sd'] if sd is not None else []))
    ax.set_xticks(range(9),MODEL_LABELS if i//2==2 else ['']*9)
    if i//2==2:
        for t in ax.get_xticklabels():t.set_rotation(65);t.set_ha('right');t.set_rotation_mode('anchor')
    ax.set_xlim(-.6,8.6)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if i%2==0:ax.set_ylabel('MSE relative to Train-Mean')
    ax.grid(axis='y',color='#dddddd',linewidth=.4)
check_and_hold(fig,'Fig3','Mean TEST MSE relative to Global Train-Mean, with ±1 seed SD for the eight five-seed models. Ridge is deterministic (triangle), so it has no seed SD. Rows are horizons 5, 10 and 20; dashed horizontal reference is 1. Ratios and SDs are copied directly from Step19.',dict(reference=1,all_models=MODEL_LABELS,error_bars='stored seed SD',ylabel='MSE relative to Train-Mean'))

# Fig4: copy epoch means without re-averaging; mark the actual first epoch.
fig,axs=plt.subplots(1,2,figsize=(WIDTH,9.5/2.54))
fig.subplots_adjust(left=.13,right=.98,bottom=.20,top=.74,wspace=.29)
headers(fig,axs,.82)
epoch_counts=[]
for i,(inp,ax) in enumerate(zip(('O','C'),axs)):
    plain(ax)
    ax.text(0,1.055,'('+chr(97+i)+')',transform=ax.transAxes,ha='left',va='bottom')
    ax.axvline(1,color='#888888',linestyle=':',linewidth=.7)
    for mechanism in ('static','adaptive'):
        rr=sorted([r for r in data['curves'] if r['input_set']==inp and r['mechanism']==mechanism and r['horizon']=='20'],key=lambda r:int(r['epoch']))
        assert rr and int(rr[0]['epoch'])==1
        x=[value(r,'epoch') for r in rr]
        for channel,style,label in [('mean_train_huber','-','train'),('mean_val_huber','--','val')]:
            y=[value(r,channel) for r in rr]
            ax.plot(x,y,color=COLORS[mechanism],linestyle=style,label=mechanism.capitalize()+' '+label)
            ax.plot(x[0],y[0],marker='o',markersize=2.7,color=COLORS[mechanism],linestyle='None')
        for r in rr:
            record('Fig4',r,['input_set','mechanism','horizon','epoch','mean_train_huber','mean_val_huber','n_seeds_at_epoch','contributing_seeds'])
            epoch_counts.append(dict(input_set=inp,mechanism=mechanism,epoch=r['epoch'],n_seeds_at_epoch=r['n_seeds_at_epoch']))
    ax.set_xticks([v for v in (1,4,7,10,13) if v<=max(ax.get_xlim())])
    ax.set_xlabel('Epoch');ax.yaxis.set_major_locator(MaxNLocator(4));ax.grid(color='#dddddd',linewidth=.4)
    if i==0:ax.set_ylabel('Huber loss')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.995),ncol=2,columnspacing=1.5,handlelength=2)
check_and_hold(fig,'Fig4','Stored mean train/validation Huber loss by epoch for Static and Adaptive, horizon 20. Solid=train; dashed=val. Epoch 1 has an explicit tick, dotted vertical guide and initial-point markers. The saved means are used unchanged: window-relative Static epochs 12–13 each contain one seed; all other displayed means contain five. No extrapolation or new averaging is used.',dict(horizon=20,marked_epoch=1,epoch_counts=epoch_counts,xlabel='Epoch'))

# Fig5: the only arithmetic is the requested common display-unit multiplier.
fig,axs=panels(18.5)
cfgs=['daily_only','weekly_only','daily_weekly']
for i,ax in enumerate(axs.flat):
    h=(5,10,20)[i//2];inp=('O','C')[i%2]
    for mechanism,offset in [('static',-.075),('adaptive',.075)]:
        rr=[one(data['matched'],input_set=inp,horizon=h,configuration=c,mechanism=mechanism) for c in cfgs]
        ax.errorbar([j+offset for j in range(3)],[100*value(r,'test_mse_mean') for r in rr],yerr=[100*value(r,'test_mse_sd') for r in rr],fmt='o-',markersize=3,capsize=2,color=COLORS[mechanism],label=mechanism.capitalize())
        for r in rr:record('Fig5',r,['input_set','horizon','configuration','mechanism','test_mse_mean','test_mse_sd'],display_unit_multiplier=100)
    ax.set_xticks(range(3),['Daily','Weekly','D+W']);ax.set_xlim(-.35,2.35)
    ax.yaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='y',color='#dddddd',linewidth=.4)
    if i%2==0:ax.set_ylabel(r'Test MSE ($\times 10^{-2}$)')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.60,.995),ncol=2,columnspacing=1.5,handlelength=2)
check_and_hold(fig,'Fig5','Matched-coverage Static/Adaptive TEST MSE mean ± seed SD from the Step18 Fig4 data. Both values and SD are multiplied by 100 solely for the labelled ×10⁻² display units. Rows are horizons 5, 10 and 20. Daily and Weekly share the B3 completed-week coverage; its window-relative reference convention remains unchanged.',dict(display_unit_multiplier=100,ylabel='Test MSE (×10⁻²)',error_bars='stored seed SD times 100'))

# Fig6: only the coordinate scale changes; all means/SDs are stored values.
fig,axs=panels(18.5)
for i,ax in enumerate(axs.flat):
    h=(5,10,20)[i//2];inp=('O','C')[i%2]
    if inp=='O':
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.yaxis.set_major_formatter(FuncFormatter(decimal));ax.yaxis.set_minor_formatter(NullFormatter())
    else:ax.yaxis.set_major_locator(MaxNLocator(4))
    for model,offset,marker in [('static',-.12,'o'),('adaptive',-.04,'s'),('ridge',.04,'^'),('zero',.12,'D')]:
        rr=[one(data['rolling'],input_set=inp,horizon=h,window=w,model=model) for w in ('W1','W2','W3')]
        xs=[j+offset for j in range(3)];ys=[value(r,'mse_ratio_mean') for r in rr]
        ax.plot(xs,ys,marker=marker,markersize=3,color=COLORS[model],label=model.capitalize())
        for x,y,r in zip(xs,ys,rr):
            sd=optional(r,'mse_ratio_sd')
            if sd is not None:
                if inp=='O':assert y-sd>0
                ax.errorbar(x,y,yerr=sd,color=COLORS[model],capsize=2)
            record('Fig6',r,['input_set','horizon','window','model','n_seeds','mse_ratio_mean']+(['mse_ratio_sd'] if sd is not None else []))
    ax.axhline(1,color='#666666',linestyle='--',linewidth=.8)
    if inp=='O':ax.set_ylim(.8,12)
    ax.set_xticks(range(3),['W1','W2','W3']);ax.set_xlim(-.35,2.35)
    ax.grid(axis='y',which='major',color='#dddddd',linewidth=.4)
    ax.xaxis.get_offset_text().set_visible(False);ax.yaxis.get_offset_text().set_visible(False)
    if i%2==0:ax.set_ylabel('MSE relative to Train-Mean')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.60,.995),ncol=4,columnspacing=.9,handlelength=1.3)
check_and_hold(fig,'Fig6','Stored rolling-window MSE relative to each window/horizon Global Train-Mean. Price-level column uses a logarithmic y-axis; window-relative column uses a linear y-axis. Static/Adaptive show stored seed SD; deterministic Ridge/Zero have no seed SD. Dashed reference is 1. Rows are horizons 5, 10 and 20.',dict(y_scales=['log','linear'],price_level_display_limits=[.8,12],ylabel='MSE relative to Train-Mean',reference=1))

# Fig7: separate category axes after explicitly omitting two price-level rows.
fig,axs=panels(25,left=.22,right=.985,wspace=.82,top=.90,bottom=.075,hspace=.42)
all_names=['Single','Fixed','Static','Adaptive','LSTM','Vanilla','SWiM','1/N','Minimum variance','Risk parity']
references={'1/N','Minimum variance','Risk parity'}
for i,ax in enumerate(axs.flat):
    h=(5,10,20)[i//2];inp=('O','C')[i%2]
    names=all_names[2:] if inp=='O' else all_names
    assert len(names)==(8 if inp=='O' else 10)
    ew=one(data['portfolios'],input_set=inp,horizon=h,strategy='1/N')
    baseline=value(ew,'sharpe')
    ax.axvline(baseline,color='#777777',linestyle='--',linewidth=.8)
    for y,name in enumerate(names):
        r=one(data['portfolios'],input_set=inp,horizon=h,strategy=name)
        assert r['gamma']=='5' and r['cost_bp']=='10'
        ax.plot(value(r,'sharpe'),y,'D' if name in references else 'o',color='#666666' if name in references else '#0072B2',markersize=3.6)
        record('Fig7',r,['input_set','horizon','strategy','gamma','cost_bp','sharpe'])
    ax.set_yticks(range(len(names)),names);ax.set_ylim(len(names)-.6,-.6)
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.grid(axis='x',color='#dddddd',linewidth=.4)
    if i//2==2:ax.set_xlabel('Sharpe ratio')
check_and_hold(fig,'Fig7','Stored seed-ensemble mean–variance portfolio Sharpe, gamma 5 and cost 10 bp, beside 1/N, minimum variance and risk parity. Price-level Single/Fixed rows are omitted. Each panel has a dashed vertical reference at its own stored 1/N Sharpe. Blue circles are forecast ensembles; grey diamonds are allocation benchmarks. Rows are horizons 5, 10 and 20.',dict(omitted_price_level_strategies=['Single','Fixed'],reference='stored 1/N Sharpe per input/horizon',gamma=5,cost_bp=10,xlabel_bottom='Sharpe ratio',category_counts=[8,10]))

# Export only after every figure renders and passes layout/style checks in memory.
for p in planned:
    for ext in ('png','pdf'):
        with (OUT/(p['name']+'.'+ext)).open('xb') as f:p['figure'].savefig(f,format=ext,dpi=300)
    plt.close(p['figure'])
    del p['figure']
after={s:sha(ROOT/s) for s in SOURCES.values()}
assert before==after
payload=dict(source_files=[dict(path=s,sha256_before=before[s],sha256_after=after[s]) for s in before],figures=planned,
             figure_width_cm=16,font='Times New Roman',font_path=font,font_size_pt=9,png_dpi=300,
             numerical_values='Copied only from the five authorized plot-data CSV files; Fig5 uses the specified x100 display conversion',
             training_runs=0,new_statistics=0,new_tests=0,new_model_evaluations=0,exclusive_creation=True)
assert FORBIDDEN not in json.dumps(payload)
assert FORBIDDEN not in json.dumps(trace)
with (OUT/'PLOT_MANIFEST.json').open('x') as f:json.dump(payload,f,indent=2)
with (OUT/'PLOT_SOURCE_TRACE.json').open('x') as f:json.dump(trace,f,indent=2)
print(json.dumps(dict(figures=[p['name'] for p in planned],exports=10,sources_unchanged=len(before),plotted_source_rows=len(trace)),indent=2))
