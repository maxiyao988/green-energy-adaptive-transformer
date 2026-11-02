"""Task C: original history aggregation and frozen-label diagnostics; no training."""
import os,sys
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'outputs/matplotlib_cache')
from step3_common import *
from scripts.python import panel_pathformer_main_v2 as formal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def histories():
 selected=pd.read_csv(E3/'experiment3_single_selection.csv')
 rows=[];runs=[]
 sources=[
  ('multiscale_transformer',E3/'experiment3_training_history.csv',E3/'experiment3_training_summary.csv','experiment3'),
  ('multiscale_transformer',E4/'experiment4_training_history.csv',E4/'experiment4_training_summary.csv','experiment4'),
  ('multiscale_transformer',MAIN/'panel_pathformer_training_history.csv',MAIN/'panel_pathformer_summary_metrics.csv','main_v2'),
 ]
 for model,folder,name in [('lstm','lstm','lstm'),('vanilla_transformer','vanilla','vanilla'),('swim','swim_cpu','swim')]:
  base=ROOT/'dataset/audit/main_v2'/folder
  sources.append((model,base/f'panel_{name}_training_history.csv',base/f'panel_{name}_summary_metrics.csv','main_v2'))
 for model,hp,sp,family in sources:
  hist=pd.read_csv(hp);summary=pd.read_csv(sp)
  for _,s in summary.iterrows():
   h=horizon(s.horizon);seed=int(s.get('seed',42));freq=s.get('frequency_setting','daily_weekly');mechanism=s.get('mechanism','adaptive' if model=='multiscale_transformer' else 'baseline')
   mask=hist.horizon.map(horizon).eq(h)
   if 'seed' in hist:mask &= hist.seed.eq(seed)
   if 'frequency_setting' in hist:mask &= hist.frequency_setting.eq(freq)
   if 'mechanism' in hist:mask &= hist.mechanism.eq(mechanism)
   dp=s.get('daily_patch',float('nan'));wp=s.get('weekly_patch',float('nan'))
   if family=='experiment3' and mechanism=='single':mask &= hist.daily_patch.eq(dp)&hist.weekly_patch.eq(wp)
   cell=hist[mask].sort_values('epoch')
   if len(cell)!=int(s.epochs_trained) or cell.epoch.tolist()!=list(range(1,int(s.epochs_trained)+1)):raise ValueError(f'Invalid history continuity: {hp}/{s.to_dict()}')
   chosen=True
   if mechanism=='single':chosen=bool(((selected.horizon.map(horizon)==h)&(selected.selected_daily_patch==dp)&(selected.selected_weekly_patch==wp)).any())
   cellid=f'{model}/{mechanism}/{freq}/h{h}/s{seed}'+(f'/d{int(dp)}/w{int(wp)}' if mechanism=='single' else '')
   runs.append(dict(cell_id=cellid,model=model,mechanism=mechanism,input_config=freq,horizon=h,seed=seed,best_epoch=int(s.best_epoch),stop_epoch=int(s.epochs_trained),selected_for_formal_comparison=chosen,daily_patch=int(dp) if pd.notna(dp) else None,weekly_patch=int(wp) if pd.notna(wp) else None,source_history=rel(hp),source_summary=rel(sp)))
   for index,r in cell.iterrows():
    rows.append(dict(model=model,mechanism=mechanism,horizon=h,seed=seed,epoch=int(r.epoch),train_loss=float(r.train_loss),val_loss=float(r.val_loss),loss_type='Huber',huber_delta=1.0,label_scale='raw adjusted-price log return',best_epoch=int(s.best_epoch),input_config=freq,cell_id=cellid,selected_for_formal_comparison=chosen,daily_patch=int(dp) if pd.notna(dp) else None,weekly_patch=int(wp) if pd.notna(wp) else None,source_history=rel(hp),source_line=int(index)+2))
 curves=pd.DataFrame(rows);run=pd.DataFrame(runs)
 if run.cell_id.duplicated().any() or curves.duplicated(['cell_id','epoch']).any():raise ValueError('Duplicate history run')
 return curves,run

def main():
 environment=threads(1);curves,runs=histories()
 write_csv(OUT/'learning_curves.csv',curves)
 write_csv(OUT/'best_epoch_by_run.csv',runs)
 dist=[]
 for (model,mechanism,freq,h),g in runs[runs.selected_for_formal_comparison].groupby(['model','mechanism','input_config','horizon']):
  counts=g.best_epoch.value_counts().sort_index()
  dist.append(dict(model=model,mechanism=mechanism,input_config=freq,horizon=int(h),n_runs=len(g),n_best_epoch_1=int(g.best_epoch.eq(1).sum()),best_epoch_min=int(g.best_epoch.min()),best_epoch_median=float(g.best_epoch.median()),best_epoch_max=int(g.best_epoch.max()),best_epoch_counts=json.dumps({str(k):int(v) for k,v in counts.items()})))
 write_csv(OUT/'best_epoch_distribution.csv',pd.DataFrame(dist))
 ds=formal.load_frozen_panel();formal._formal_check_dataset_contract(ds);refs=[];means=[]
 for h in [5,10,20]:
  trainmean=float(np.mean(ds.y[h][ds.split=='train']))
  for split in ['train','val','test']:
   mask=ds.split==split;y=ds.y[h][mask]
   for predictor in ['zero','global_train_mean']:
    pred=np.zeros_like(y) if predictor=='zero' else np.full_like(y,trainmean)
    huber=float(torch.nn.HuberLoss(delta=1.0)(torch.from_numpy(pred),torch.from_numpy(y)).item())
    refs.append(dict(horizon=h,split=split,predictor=predictor,n_rows=len(y),pooled_training_mean=trainmean,constant_prediction=0.0 if predictor=='zero' else trainmean,mse=float(np.mean((pred-y)**2)),huber_loss=huber,huber_delta=1.0,label_scale='raw adjusted-price log return'))
   for ticker in ['POOLED']+sorted(np.unique(ds.ticker).tolist()):
    mask2=mask if ticker=='POOLED' else mask&(ds.ticker==ticker)
    mean=float(np.mean(ds.y[h][mask2]))
    means.append(dict(horizon=h,ticker=ticker,split=split,n_rows=int(mask2.sum()),mean_realized_return=mean,global_train_mean_prediction=trainmean,mean_minus_global_train_mean=mean-trainmean))
 refs=pd.DataFrame(refs);means=pd.DataFrame(means)
 write_csv(OUT/'baseline_loss_references.csv',refs);write_csv(OUT/'validation_naive_references.csv',refs[refs.split=='val'])
 write_csv(OUT/'mean_shift_by_split.csv',means)
 wide=means.pivot(index=['horizon','ticker','global_train_mean_prediction'],columns='split',values='mean_realized_return').reset_index().rename(columns={'train':'train_mean_return','val':'validation_mean_return','test':'test_mean_return'})
 write_csv(OUT/'mean_shift_table.csv',wide)
 # Show the requested primary Daily+Weekly mechanism comparison on a common Huber scale.
 fig,axes=plt.subplots(1,3,figsize=(16.8,5.3),layout='constrained')
 colors={'static':'#1f77b4','adaptive':'#d85a23'}
 for ax,h in zip(axes,[5,10,20]):
  sub=curves[(curves.horizon==h)&(curves.input_config=='daily_weekly')&curves.mechanism.isin(['static','adaptive'])]
  for (mechanism,seed),g in sub.groupby(['mechanism','seed']):
   g=g.sort_values('epoch');color=colors[mechanism]
   ax.plot(g.epoch,g.val_loss,color=color,linewidth=0.95,alpha=0.72)
   ax.plot(g.epoch,g.train_loss,color=color,linewidth=0.65,alpha=0.32,linestyle=':')
  for predictor,style,color in [('zero','--','#222222'),('global_train_mean','-.','#777777')]:
   value=refs[(refs.horizon==h)&(refs.split=='val')&(refs.predictor==predictor)].iloc[0].huber_loss
   ax.axhline(value,color=color,linestyle=style,linewidth=1.6,zorder=10)
  ax.set(title=f'h = {h} trading days',xlabel='Epoch',ylabel='Huber loss (delta = 1, raw return scale)')
  ax.grid(alpha=0.18);ax.set_xlim(left=1)
 handles=[Line2D([],[],color=colors['static'],label='Static validation (5 seeds)'),Line2D([],[],color=colors['adaptive'],label='Adaptive validation (5 seeds)'),Line2D([],[],color='#777777',linestyle=':',label='Training loss (thin dotted)'),Line2D([],[],color='#222222',linestyle='--',label='Zero validation reference'),Line2D([],[],color='#777777',linestyle='-.',label='Global train-mean validation reference')]
 fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
 fig.suptitle('Daily + Weekly learning curves — original formal V2 histories',fontsize=14)
 figure=target(OUT/'fig_learning_curves.png');fig.savefig(figure,dpi=190);plt.close(fig)
 write_json(OUT/'task_c_metadata.json',dict(environment=environment,completed_utc=utc(),n_history_rows=len(curves),n_training_runs=len(runs),n_selected_runs=int(runs.selected_for_formal_comparison.sum()),note='All canonical formal V2 histories and all 48 Single candidates retained; primary plot and report compare selected Daily+Weekly runs. Recovered trajectories are separately preserved under task_a, not double-counted here.',loss='Huber(delta=1.0), raw adjusted-price log returns; figure reference lines are validation Huber, not validation MSE.'))
 print('COMPLETE Task C','history rows',len(curves),'runs',len(runs),'selected runs',int(runs.selected_for_formal_comparison.sum()),flush=True)
if __name__=='__main__':main()
