"""Validation-only reproducibility decision; executed by the user-run scheduler."""
import sys
sys.dont_write_bytecode = True
from io22b_v3 import *
sys.path.append(str(ROOT / 'revision_work/outputs/_deps_pyarrow15'))
import numpy as np
import pandas as pd

def main():
    check_protocol()
    job = spec('T0', 'static', 5, 0, True)
    path, result = completion(job)
    assert result is not None
    row = next(r for r in source_rows() if (r['model'], r['h'], r['seed']) == ('static', 5, 0))
    if result['status'] != 'COMPLETE':
        js(OUT / 'REPRODUCIBILITY_GATE.json', dict(passed=False, error=result.get('error'), protocol_sha256=check_protocol(), utc=utc()))
        return
    new = pd.read_csv(path.parent / 'history.csv')
    old = pd.read_csv(ROOT / row['history'])
    same_epochs = new.epoch.tolist() == old.epoch.tolist() and bool(np.isfinite(new.val_loss).all() and np.isfinite(old.val_loss).all())
    compared = new[['epoch', 'val_loss']].merge(old[['epoch', 'val_loss']], on='epoch', how='outer', suffixes=('_new', '_Step16'), indicator=True)
    compared['abs_difference'] = (compared.val_loss_new - compared.val_loss_Step16).abs()
    csv(OUT / 'gate_history_comparison.csv', compared)
    nh = float(compared.abs_difference.max()) if same_epochs else None
    a = pd.read_csv(path.parent / 'val.csv')
    b = pd.read_parquet(ROOT / row['val'])
    for d in [a, b]:
        d.anchor_date = pd.to_datetime(d.anchor_date).dt.strftime('%Y-%m-%d')
    a = a.sort_values(['ticker', 'anchor_date']).reset_index(drop=True)
    b = b.sort_values(['ticker', 'anchor_date']).reset_index(drop=True)
    same_keys = len(a) == len(b) == COUNTS['val'] and a[['ticker', 'anchor_date']].equals(b[['ticker', 'anchor_date']])
    same_labels = same_keys and np.array_equal(a.y_true.to_numpy(np.float32), b.y_true.to_numpy(np.float32))
    dp = float(np.max(np.abs(a.y_pred.to_numpy(float) - b.y_pred.to_numpy(float)))) if same_keys else None
    passed = bool(same_epochs and same_labels and nh is not None and dp is not None and np.isfinite(nh) and np.isfinite(dp) and nh <= 1e-6 and dp <= 1e-6)
    js(OUT / 'REPRODUCIBILITY_GATE.json', dict(protocol_sha256=check_protocol(), utc=utc(), passed=passed,
       threshold=1e-6, matching_epoch_sequence=same_epochs, keys_equal=same_keys, labels_equal=same_labels,
       max_abs_validation_loss_difference=nh, max_abs_validation_prediction_difference=dp,
       gate_completion=artifact(path), source_history=artifact(ROOT / row['history']), source_val=artifact(ROOT / row['val']),
       test_read=False))
    print('Validation reproducibility:', 'PASS' if passed else 'FAIL', 'history', nh, 'predictions', dp)

if __name__ == '__main__':
    main()
