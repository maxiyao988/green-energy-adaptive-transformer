"""Read-only summary of the first required test while the separately authorized queue runs."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revision_work/outputs/repro_threads'
def main():
 r=json.loads((OUT/'runs/adaptive_dw_h10_s42_t12.json').read_text());prior=json.loads((ROOT/'revision_work/outputs/task_a/recovery_h10.json').read_text());samples=[json.loads(line) for line in (OUT/'telemetry/adaptive_dw_h10_s42_t12.jsonl').read_text().splitlines()]
 lines=['# Step3B h10 first result\n',f"**{r['classification']}**: max |Δpred| = {r['max_abs_delta_pred']:.16g}; ΔMSE(new−formal) = {r['test_mse_difference']:.16g}; test MSE = {r['replayed_test_mse']:.16g}.\n",f"Router max |Δ| = {r['router_max_abs_delta']:.16g}. Best/stop = {r['best_epoch']}/{r['stop_epoch']}, matching formal {r['formal_best_epoch']}/{r['formal_stop_epoch']}.\n",f"All11 history epochs are within1e-12; strict numeric equality is {r['history_exact_equal']}. Max train/val loss differences = {r['history_max_train_abs_delta']:.16g}/{r['history_max_val_abs_delta']:.16g}. The tiny archive-value differences are retained, not rounded to zero.\n",f"12/12 result above versus the prior3/1 result: {prior['classification']}, max |Δpred|={prior['max_abs_delta_pred']:.12g}. Restoring the requested12/12 profile is sufficient for h10; intra-op versus inter-op has not been isolated and historical runtime metadata remain missing.\n",'OMP_NUM_THREADS unset; OMP_DYNAMIC=FALSE; MKL_DYNAMIC=FALSE. All recorded main-thread counts stayed12/12. Resident apps were authorized. Background load is used for timing only.\n','| UTC | Event | Intra | Inter | Load1/5/15 |','| --- | --- | --- | --- | --- |']
 for e in samples:lines.append(f"| {e['utc']} | {e['event']} | {e['torch_num_threads']} | {e['torch_num_interop_threads']} | {e['load_average_1_5_15']} |")
 lines+=['\nTrain/val/test checkpoint-replay Parquets have30,634/6,290/6,647 rows, with Daily/Weekly router weights. Files: `preds/adaptive_dw_h10_s42_t12_{train,val,test}.parquet`.\n','The Stage3b+3c queue started only after this worker exited. h5 andh20 will run sequentially at12/12 after that queue completes. The final REPORT_STEP3B.md will include those results.\n']
 with (OUT/'H10_FIRST_RESULT.md').open('x') as f:f.write('\n'.join(lines))
 print('h10 first-result summary saved')
if __name__=='__main__':main()
