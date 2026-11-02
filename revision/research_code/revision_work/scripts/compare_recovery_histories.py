"""Compare original and rerun histories on their shared epochs, without training."""
from step3_common import *
def main():
 old=pd.read_csv(MAIN/'panel_pathformer_training_history.csv');rows=[]
 for h in [5,10,20]:
  new=pd.read_csv(OUT/'task_a'/f'adaptive_dw_h{h}_s42_history.csv')
  formal=old[(old.frequency_setting=='daily_weekly')&(old.horizon==h)]
  merged=new.merge(formal[['epoch','train_loss','val_loss']],on='epoch',suffixes=('_new','_formal'))
  for loss in ['train_loss','val_loss']:merged[loss+'_difference']=merged[loss+'_new']-merged[loss+'_formal']
  rows.append(merged)
 write_csv(OUT/'task_a_history_overlap_comparison.csv',pd.concat(rows,ignore_index=True))
if __name__=='__main__':main()
