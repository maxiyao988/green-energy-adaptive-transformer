"""Exact B2 tensor binding; TRAIN/VAL slicing prevents pre-barrier TEST access."""
from io_d import *

def load(window, include_test=False):
    import numpy as np
    import pandas as pd
    from scripts.python import panel_common as common
    folder=B2/'tensors'/window
    if include_test:
        barrier()
        # Compile the existing loader definition verbatim, without import-time thread setters.
        import ast
        source=B2/'scripts/b2_data.py'
        tree=ast.parse(source.read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_panel']
        assert len(nodes)==1
        env=dict(EXP=B2,np=np,pd=pd,common=common)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
        ds=env['load_panel'](window,'C')
        assert {k:int((ds.split==k).sum()) for k in COUNTS}==counts(window)
        return ds
    meta=pd.read_csv(folder/'sample_index.csv')
    keep=meta.split.isin(['train','val']).to_numpy()
    keys=meta.loc[keep].reset_index(drop=True)
    def selected(path):
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        assert a.dtype==np.float32
        return a[keep].copy()
    ds=common.PanelDataset(keys.ticker.to_numpy(),pd.to_datetime(keys.anchor_date).to_numpy(),keys.split.to_numpy(),
       selected(folder/'C/X_daily_raw.npy'),selected(folder/'C/X_weekly_raw.npy'),
       {h:selected(folder/f'y_h{h}.npy') for h in HS})
    assert not (ds.split=='test').any()
    assert {sp:int((ds.split==sp).sum()) for sp in ['train','val']}=={sp:counts(window)[sp] for sp in ['train','val']}
    z=common.apply_normalizer(ds,common.fit_train_only_normalizer(ds),'daily_weekly')
    for branch,a in zip(['daily','weekly'],z):
        assert np.array_equal(a,selected(folder/f'C/X_{branch}_standardized.npy'))
    return ds
