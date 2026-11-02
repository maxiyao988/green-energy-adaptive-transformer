"""No-training check of every imported loader path and mask/label transport."""
from b3_io import *
from b3_data import base,formal,load_panel,contract,fit_bound_stats,apply_bound_stats
import torch,numpy as np
from contextlib import ExitStack

def main():
 rows=[]
 def bind(stack,obj,key,value):
  old=getattr(obj,key);setattr(obj,key,value);stack.callback(setattr,obj,key,old)
 for key in ['O','C_anchor','C_end']:
  ds=load_panel(key);contract(ds)
  with ExitStack() as stack:
   for owner in [formal,base.exp3]:
    bind(stack,owner,'fit_train_only_normalizer',fit_bound_stats);bind(stack,owner,'apply_normalizer',apply_bound_stats)
   paths=[]
   for frequency in ['daily_only','weekly_only','daily_weekly']:paths.append((f'main_v2:{frequency}',formal._build_training_loaders(frequency,5,ds),frequency))
   paths.append(('Experiment3:dual-input loop',base.exp3._build_train_val_loaders_only(5,ds),'daily_weekly'))
   paths.append(('Experiment4:dual-input loop',base.exp4._train_val_loaders_only(5,ds),'daily_weekly'))
   for path,loaders,frequency in paths:
    assert loaders[0].generator.initial_seed()==42
    for split,loader in zip(['train','val','test'],loaders):
     assert len(loader.dataset)==COUNTS[split];tensors=loader.dataset.tensors;mask=ds.split==split
     expected=apply_bound_stats(ds,fit_bound_stats(ds),frequency);expected=list(expected) if isinstance(expected,tuple) else [expected]
     assert len(tensors)==len(expected)+1
     for actual,want in zip(tensors[:-1],expected):assert np.array_equal(actual.numpy(),want[mask])
     assert np.array_equal(tensors[-1].numpy(),ds.y[5][mask]);assert all(torch.isfinite(t).all() for t in tensors)
    rows.append(dict(tensor_key=key,loader_path=path,generator_seed=42,all_keys_labels_and_normalized_tensors_exact=True,train_rows=30634,val_rows=6290,test_loader_returned=len(loaders)==3,no_training=True))
  print('LOADER_CANDIDATE_PASS',key,flush=True)
 csvout(EXP/'imported_loader_checks.csv',rows);jsonout(EXP/'LOADER_CHECKS.json',dict(utc=utc(),passed=True,n_checks=len(rows),all_generators_42=True,no_training=True))
if __name__=='__main__':main()
