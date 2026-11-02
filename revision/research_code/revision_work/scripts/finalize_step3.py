"""Combine completed Step 3 evidence, verify protected inputs and write the final report."""
import os,sys,time
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1'
from step3_common import *

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|',' / ').replace('\n',' ') for x in row)+' |' for row in rows])+'\n'
def link(path,line=None):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 return f'[{p.name}{":"+str(line) if line else ""}]({p}{":"+str(line) if line else ""})'
def src(name,line):return link(ROOT/'scripts/python'/f'{name}.py',line)
def artifact(name):return link(OUT/name)
def label(row):return row['mechanism'].capitalize() if row['model']=='multiscale_transformer' else {'lstm':'LSTM','vanilla_transformer':'Vanilla','swim':'SWiM'}[row['model']]

def main():
 threads(1)
 a=[]
 for h in [5,10,20]:
  p=OUT/'task_a'/f'recovery_h{h}.json'
  if p.is_file():
   d=json.loads(p.read_text());d['utc_elapsed_wall_seconds']=(datetime.fromisoformat(d['completed_utc'])-datetime.fromisoformat(d['started_utc'])).total_seconds();a.append(d)
  else:a.append(dict(horizon=h,classification='NOT FOUND'))
 write_csv(OUT/'task_a_comparison.csv',pd.DataFrame([{k:v for k,v in d.items() if k not in ['environment']} for d in a]))
 replay=pd.concat([pd.read_csv(OUT/f'task_b_replay_{stage}.csv') for stage in ['existing','recovered'] if (OUT/f'task_b_replay_{stage}.csv').is_file()],ignore_index=True)
 files=pd.concat([pd.read_csv(OUT/f'task_b_files_{stage}.csv') for stage in ['existing','recovered'] if (OUT/f'task_b_files_{stage}.csv').is_file()],ignore_index=True)
 if replay.duplicated(['mechanism','horizon','seed']).any():raise ValueError('Duplicate replay cells')
 if files.duplicated(['mechanism','horizon','seed','split']).any():raise ValueError('Duplicate prediction artifacts')
 write_csv(OUT/'checkpoint_replay.csv',replay.sort_values(['mechanism','horizon','seed']))
 write_csv(OUT/'prediction_file_manifest.csv',files.sort_values(['mechanism','horizon','seed','split']))
 with target(OUT/'prediction_files.txt').open('x') as f:f.write('\n'.join(files.path.sort_values())+'\n')
 refs=pd.read_csv(OUT/'baseline_loss_references.csv');skills=[]
 for _,r in files.iterrows():
  d=pd.read_parquet(ROOT/r.path);n=COUNTS[r.split]
  if len(d)!=n or d['split'].nunique()!=1 or d['split'].iloc[0]!=r.split:raise ValueError('Saved split/count mismatch')
  y=torch.from_numpy(d.y_true.to_numpy(np.float32));p=torch.from_numpy(d.y_pred.to_numpy(np.float32))
  actual_mse=mse(d);huber=float(torch.nn.HuberLoss(delta=1.0)(p,y))
  reference=refs[(refs.horizon==r.horizon)&(refs.split==r.split)].set_index('predictor')
  skills.append(dict(mechanism=r.mechanism,horizon=int(r.horizon),seed=int(r.seed),split=r.split,n_rows=n,mse=actual_mse,huber_loss=huber,zero_mse=float(reference.loc['zero','mse']),train_mean_mse=float(reference.loc['global_train_mean','mse']),mse_skill_vs_zero=1-actual_mse/reference.loc['zero','mse'],mse_skill_vs_train_mean=1-actual_mse/reference.loc['global_train_mean','mse'],huber_skill_vs_zero=1-huber/reference.loc['zero','huber_loss'],huber_skill_vs_train_mean=1-huber/reference.loc['global_train_mean','huber_loss']))
 skills=pd.DataFrame(skills);write_csv(OUT/'forecast_skill.csv',skills)
 # Protect all files that existed before Step 3, including prior audit/download outputs.
 baseline=json.loads((OUT/'protected_inputs_before.json').read_text());changed=[]
 for item in baseline:
  path=ROOT/item['path']
  if not path.is_file():changed.append(dict(path=item['path'],issue='missing'));continue
  s=path.stat()
  if s.st_size!=item['size'] or s.st_mtime_ns!=item['mtime_ns']:changed.append(dict(path=item['path'],issue='size/mtime changed'))
  elif item.get('sha256') and sha(path)!=item['sha256']:changed.append(dict(path=item['path'],issue='sha256 changed'))
 integrity=dict(passed=not changed,n_protected_files=len(baseline),n_cryptographic_hashes=sum('sha256' in i for i in baseline),changed=changed,completed_utc=utc())
 write_json(OUT/'protected_inputs_verification.json',integrity)
 blocking=[]
 for d in a:
  if d['classification'] not in ['EXACT','NEAR']:blocking.append(f"Adaptive h={d['horizon']} recovery is {d['classification']}; excluded from Task B.")
 expected_cells=33+sum(d['classification'] in ['EXACT','NEAR'] for d in a)
 if len(replay)!=expected_cells:blocking.append(f'Only {len(replay)}/{expected_cells} eligible Daily+Weekly cells have completed replay.')
 bad=replay[replay.max_abs_delta_pred>=1e-5]
 if len(bad):blocking.append(f'{len(bad)} existing/recovered checkpoint replays differ by >=1e-5; see replay table.')
 if changed:blocking.append(f'{len(changed)} pre-existing files changed during execution; inspect the integrity report.')
 curves=pd.read_csv(OUT/'learning_curves.csv');runs=pd.read_csv(OUT/'best_epoch_by_run.csv');means=pd.read_csv(OUT/'mean_shift_table.csv');dist=pd.read_csv(OUT/'best_epoch_distribution.csv')
 process=json.loads((OUT/'task_a_processes.json').read_text()) if (OUT/'task_a_processes.json').is_file() else {}
 parts=[]
 def add(text):parts.append(text.strip()+'\n')
 add('# BLOCKING\n\n'+('\n'.join('- '+s for s in blocking) if blocking else 'None for the requested Step 3 coverage: recovery, checkpoint replay, required split counts and protected-input checks passed.'))
 add(f'''# Step 3 — Recovery, checkpoint replay and forecast diagnostics

Completed: {utc()} (UTC). All new source is under `revision_work/scripts/`; all new outputs, logs and isolated dependencies are under `revision_work/outputs/`. The frozen adjusted-price V2 panel was used throughout; `data_raw_revision/` was not used for training or inference.

Protected-input verification: **{'PASS' if integrity['passed'] else 'FAIL'}** for {integrity['n_protected_files']} pre-existing files, with SHA-256 checks on {integrity['n_cryptographic_hashes']} formal source/data/artifact files; details in {artifact('protected_inputs_verification.json')}. Existing checkpoints and formal results were not replaced. Script entrypoints: {link(ROOT/'revision_work/scripts/launch_recovery.py')}, {link(ROOT/'revision_work/scripts/recover_adaptive42.py')}, {link(ROOT/'revision_work/scripts/replay_checkpoints.py')}, {link(ROOT/'revision_work/scripts/learning_diagnostics.py')}.

## Task A — Adaptive seed-42 checkpoint recovery

Each worker calls the original `train_one_cell('daily_weekly', h, 'adaptive')` directly ({src('panel_pathformer_main_v2',1611)}). The original model factory is delegated to unchanged through an in-process observer that only retains a model reference and logs train/validation transitions; the original trainer, architecture and source files are not reimplemented or edited. After the original function restores its best state and evaluates test predictions, that restored model's state dictionary is saved. The observer adds no random draws or tensor transformations ({link(ROOT/'revision_work/scripts/recover_adaptive42.py',20)}).

The original path sets seed 42, builds a Train DataLoader with its explicit generator seeded 42, uses CPU, Adam lr=1e-4, batch size 32, Huber delta=1, gradient clip 1, max 100 epochs and patience 10; best epoch selection uses validation Huber with improvement threshold 1e-12 ({src('panel_pathformer_main_v2',67)}, {src('panel_pathformer_main_v2',1371)}, {src('panel_pathformer_main_v2',1632)}, {src('panel_pathformer_main_v2',1713)}). Exactly three horizons were launched concurrently as separate processes, each with `torch.set_num_threads(3)`, `torch.set_num_interop_threads(1)` and `OMP_NUM_THREADS=3`.

Classification is based only on test prediction maximum absolute difference: EXACT <1e-7, NEAR <1e-5, otherwise DIFFERENT. Epoch equality and router differences are reported independently. CSV reference predictions are compared in float64, while MSE follows the original float32 metric arithmetic ({src('panel_common',208)}).
''')
 add(table(['h','Class','max abs Δpred','Δtest MSE (new−formal)','Best epoch new/formal','Stop epoch new/formal','Best / stop equal','Router max abs Δ','End-to-end seconds¹','Training-loop seconds'],[[d['horizon'],d['classification'],f"{d.get('max_abs_delta_pred',float('nan')):.10g}",f"{d.get('test_mse_difference',float('nan')):.10g}",f"{d.get('best_epoch','NOT FOUND')}/{d.get('formal_best_epoch','NOT FOUND')}",f"{d.get('stop_epoch','NOT FOUND')}/{d.get('formal_stop_epoch','NOT FOUND')}",f"{d.get('best_epoch_equal','NOT FOUND')} / {d.get('stop_epoch_equal','NOT FOUND')}",f"{d.get('router_max_abs_delta',float('nan')):.10g}",f"{d.get('utc_elapsed_wall_seconds',float('nan')):.3f}",f"{d.get('formal_training_loop_seconds',float('nan')):.3f}"] for d in a]))
 add(f'''¹End-to-end seconds use UTC start/end timestamps and include data preparation, training, evaluation and saving before metadata completion. Per-process monotonic elapsed seconds are also retained in {artifact('task_a_comparison.csv')}; the trainer's own timing starts after loader/model setup and ends after best-state restoration ({src('panel_pathformer_main_v2',1647)}, {src('panel_pathformer_main_v2',1731)}). Parallel-launch timing: {artifact('task_a_processes.json')} (monotonic elapsed {process.get('parallel_wall_clock_seconds','NOT FOUND')} seconds). UTC intervals and the trainer's time.time-based duration include a roughly 1,973-second temporal gap not present in the worker monotonic timers; its cause was not established. Both clocks are retained rather than silently subtracting the gap. These timings are not an isolated performance benchmark.

**All three requested reruns were DIFFERENT and are retained only as new rerun artifacts; none qualifies as a recovered formal checkpoint.** Histories differ from the first epoch; overlap comparisons are in {artifact('task_a_history_overlap_comparison.csv')}. Exact historical training runtime settings are NOT FOUND. A forward-only synthetic-input comparison using the imported formal model at 3 versus 12 threads produced identical initialization, RNG state and first training-mode/eval-mode outputs ({artifact('thread_rng_diagnostic.json')}); thus this test does not support attributing the mismatch solely to the forward pass or dropout thread count. Differences in accumulated numerical training updates or unarchived historical execution details remain possible, but the root cause is **UNVERIFIED**. No additional recovery training or selection of a different seed was performed.

Each horizon has `checkpoints/adaptive_dw_h{{h}}_s42.pt`, plus `task_a/adaptive_dw_h{{h}}_s42_{{summary,predictions,router,history,rank_ic}}.csv` and `task_a/recovery_h{{h}}.json`. The checkpoint hashes, original/recovered epoch equality, history differences and exact thread settings are retained in those JSON files. Recovery does not silently promote a DIFFERENT result into the formal seed-42 family.

## Task B — Checkpoint replay and all-split predictions

Imported factories are Experiment 3 for selected Single/Fixed/seed-42 Static and Experiment 4 for the remaining Static/Adaptive cells. All state dictionaries are hash-verified and loaded strictly in eval mode on CPU. Inference uses the original Experiment 3 `_batched_inference` helper ({src('panel_pathformer_experiment3',579)}) or Experiment 4 helper ({src('panel_pathformer_experiment4_multiseed',860)}). Original normalization and split tensors come from the formal loader builder ({src('panel_pathformer_main_v2',1324)}). For TRAIN inference only, the same tensor dataset is wrapped in an ordered loader so ticker/date keys align with output rows; no stochastic training loader order is used for exported prediction metadata.

All columns begin with `ticker, anchor_date, split, y_true, y_pred`. Static/Adaptive additionally contain {', '.join(WEIGHTS)}; Static weights are checked constant across samples. Every Parquet file is read back and compared with its in-memory frame before it is accepted. Reference test keys and float32 labels must match exactly.
''')
 add(table(['Mechanism','h','Seed','Rows test','max abs Δpred','ΔMSE (new−formal)','Replay class'],[[r.mechanism,int(r.horizon),int(r.seed),int(r.n_rows),f'{r.max_abs_delta_pred:.10g}',f'{r.test_mse_difference:.10g}',r.classification] for _,r in replay.sort_values(['mechanism','horizon','seed']).iterrows()]))
 add(f'''Full replay results and checkpoint/source paths: {artifact('checkpoint_replay.csv')}. Predictions: **{len(replay)} cells × 3 splits = {len(files)} Parquet files**, with required counts 30,634 train / 6,290 val / 6,647 test per cell. Complete file list with counts, columns and SHA-256 hashes: {artifact('prediction_file_manifest.csv')}; plain list: {artifact('prediction_files.txt')}. Filenames follow `preds/{{mechanism}}_dw_h{{h}}_s{{seed}}_{{train,val,test}}.parquet`. Counts by mechanism:
''')
 add(table(['Mechanism','Cells','Files','Total train rows','Total val rows','Total test rows'],[[mech,len(g)//3,len(g),int(g[g.split=='train'].n_rows.sum()),int(g[g.split=='val'].n_rows.sum()),int(g[g.split=='test'].n_rows.sum())] for mech,g in files.groupby('mechanism')]))
 add('''An initial replay attempt failed because PyArrow 26 requires NumPy >=2, while formal V2 uses NumPy 1.26.4. The original environment was retained; PyArrow 15.0.2 was installed under the new output directory instead. The failed log and empty first-file attempt are preserved under `logs/replay_existing.log` and `failed_attempts/`; they are excluded from the prediction manifest. Completed replay outputs use PyArrow 15.0.2 and serialization now completes before exclusive file creation. This resolved issue does not indicate a checkpoint mismatch.''')
 add(f'''## Task C — Learning curves, reference forecasts and mean shift

{artifact('learning_curves.csv')} contains **{len(curves)} rows across {len(runs)} original formal training runs**: 48 Single patch candidates + 3 Fixed + 15 Static + 21 Adaptive (15 dual-resolution plus 6 single-resolution) + 9 each LSTM, Vanilla and SWiM. Each row includes the requested model/mechanism/horizon/seed/epoch/loss/best-epoch fields, plus input configuration, candidate selection flag, patch metadata, cell ID and source file/line. A total of {int(runs.selected_for_formal_comparison.sum())} runs remain after excluding unselected Single candidates. Historical prototypes and pre-temporal-fix histories are excluded. Recovered Task A histories are preserved separately rather than double-counted as original runs.

All plotted/history losses are **Huber(delta=1) on raw adjusted-price log returns**, not MSE. Reference lines use validation Huber on the same scale. The figure shows all five Static and Adaptive Daily+Weekly seeds as thin lines, training losses dotted and validation losses solid. Benchmarks and all other input configurations remain in the learning-curve CSV.

![Learning curves](fig_learning_curves.png)

### Validation Zero / Global Train-Mean references

The Global Train-Mean constant is the pooled training-label mean for that horizon, reused for every ticker and all later splits; it is not a ticker-specific or validation-fitted mean ({src('panel_ridge_validation',85)}). Baselines were computed directly from the frozen labels in memory with the same Huber implementation.
''')
 val=refs[refs.split=='val']
 add(table(['h','Predictor','Training mean','Validation MSE','Validation Huber'],[[int(r.horizon),r.predictor,f'{r.pooled_training_mean:.10f}',f'{r.mse:.10f}',f'{r.huber_loss:.10f}'] for _,r in val.iterrows()]))
 add(f'''Full references across all three splits: {artifact('baseline_loss_references.csv')}; requested validation-only table: {artifact('validation_naive_references.csv')}. Zero has lower validation MSE and Huber than the positive pooled train-mean predictor at all three horizons.

### Best-epoch distribution (selected Daily+Weekly runs)
''')
 primary=dist[dist.input_config=='daily_weekly']
 add(table(['Model/mechanism','h','Runs','Best epoch=1','Best min/median/max','Epoch counts'],[[label(r),int(r.horizon),int(r.n_runs),int(r.n_best_epoch_1),f'{int(r.best_epoch_min)}/{r.best_epoch_median:g}/{int(r.best_epoch_max)}',r.best_epoch_counts] for _,r in primary.iterrows()]))
 add(f'All configurations and per-run stop epochs: {artifact("best_epoch_distribution.csv")}, {artifact("best_epoch_by_run.csv")}.')
 add('### Replayed validation forecast skill\n\nMSE skill = 1 − model MSE / reference MSE; positive means lower error than that reference. These diagnostics do not refit or retune any model.')
 skilltable=[]
 for (mech,h),g in skills[skills.split=='val'].groupby(['mechanism','horizon']):
  skilltable.append([mech,int(h),len(g),f'{g.mse.mean():.10f}',f'{g.mse_skill_vs_zero.mean():.4%}',f'{g.mse_skill_vs_train_mean.mean():.4%}',int(g.mse_skill_vs_zero.gt(0).sum()),int(g.mse_skill_vs_train_mean.gt(0).sum())])
 add(table(['Mechanism','h','Seeds','Mean validation MSE','Mean skill vs Zero','Mean skill vs Train-Mean','Wins vs Zero','Wins vs Train-Mean'],skilltable))
 add(f'Adaptive skill diagnostics use only the four replayable original seeds per horizon; the learning-curve figure still includes the original saved seed-42 history. The differing new seed-42 reruns are not mixed into either formal family. Per-cell train/validation/test MSE, Huber and skill: {artifact("forecast_skill.csv")}. These are forecast-error comparisons, not tests of economic utility or profitability.')
 add('### Mean-shift table\n\nAll values are mean realized raw h-session log returns. The last column is the same pooled training mean used by Global Train-Mean, shown alongside every ticker. A ticker’s own train mean is descriptive; it was not used as that baseline’s forecast.')
 add(table(['h','Ticker','Train mean','Validation mean','Test mean','Global Train-Mean forecast'],[[int(r.horizon),r.ticker,f'{r.train_mean_return:.9f}',f'{r.validation_mean_return:.9f}',f'{r.test_mean_return:.9f}',f'{r.global_train_mean_prediction:.9f}'] for _,r in means.sort_values(['horizon','ticker']).iterrows()]))
 add(f'Long form with per-split counts and differences from the baseline: {artifact("mean_shift_by_split.csv")}; wide form: {artifact("mean_shift_table.csv")}. Pooled training means are positive, validation means negative, and test means positive at all three horizons. This describes a distribution shift; it does not establish its cause.')
 add('''## Reproduction commands

Run once in a fresh output namespace; writers refuse to overwrite completed outputs:

```sh
.venv/bin/python -B revision_work/scripts/launch_recovery.py
OMP_NUM_THREADS=3 .venv/bin/python -B revision_work/scripts/replay_checkpoints.py --stage existing
OMP_NUM_THREADS=3 .venv/bin/python -B revision_work/scripts/replay_checkpoints.py --stage recovered
OMP_NUM_THREADS=1 .venv/bin/python -B revision_work/scripts/learning_diagnostics.py
OMP_NUM_THREADS=1 .venv/bin/python -B revision_work/scripts/finalize_step3.py
```

The existing-checkpoint replay and Task C may run while recovery is in progress; the recovered replay and finalizer require completed recovery classifications. All Python calls should use `-B`/`PYTHONDONTWRITEBYTECODE=1` to avoid bytecode writes beside original source files.''')
 with target(OUT/'REPORT_STEP3.md').open('x') as f:f.write('\n'.join(parts))
 # Provenance manifest is created after the report, excluding mutable caches and itself.
 import importlib.metadata as md
 import pyarrow
 manifest=dict(completed_utc=utc(),packages={name:md.version(name) for name in ['torch','numpy','pandas','scipy','matplotlib']},pyarrow=pyarrow.__version__,blocking=blocking,integrity=integrity,files=[])
 for p in sorted((ROOT/'revision_work/scripts').glob('*.py')):manifest['files'].append(dict(path=rel(p),sha256=sha(p)))
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and not any(part.startswith('_deps') or part in ['matplotlib_cache','_tmp'] for part in p.relative_to(OUT).parts):manifest['files'].append(dict(path=rel(p),sha256=sha(p)))
 write_json(OUT/'STEP3_MANIFEST.json',manifest)
 print('FINALIZED',utc(),'A classifications',[(d['horizon'],d['classification']) for d in a],'B cells',len(replay),'files',len(files),'blocking',blocking,flush=True)
if __name__=='__main__':main()
