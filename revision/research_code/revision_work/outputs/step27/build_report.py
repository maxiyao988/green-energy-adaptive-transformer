"""Render the completed Step27 diagnostics without rerunning inference; exclusive outputs."""
import csv,json,hashlib,os,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def fmt(x):return format(x,'.6g') if isinstance(x,float) else str(x)
def textout(name,s):
    with (OUT/name).open('x') as f:f.write(s)
def table(rows,cols=None):
    cols=cols or list(rows[0])
    return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(fmt(r[c]).replace('|',' / ') for c in cols)+' |' for r in rows)+'\n'
def excerpt(p,a,b):
    lines=(ROOT/p).read_text().splitlines()
    return f'`{p}:{a}–{b}`\n```python\n'+'\n'.join(f'{i}: {lines[i-1]}' for i in range(a,b+1))+'\n```\n'
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)) and isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            p=Path(os.fsdecode(path)).resolve()
            if not p.is_relative_to(OUT) or not flags&os.O_EXCL:raise RuntimeError('READ-ONLY/EXCLUSIVE WRITE GUARD: '+str(p))
sys.addaudithook(guard)
d=json.loads((OUT/'calculation_state.json').read_text());A=d['A'];AY=d['AY'];C=d['C'];epoch1=d['epoch1'];manifest=d['manifest']
# Verify every input again before quoting it in the final report.
for r in manifest:
    p=Path(r['source_file']);p=p if p.is_absolute() else ROOT/p
    assert sha(p)==r['sha256'],p
main='scripts/python/panel_pathformer_main_v2.py';e3='scripts/python/panel_pathformer_experiment3.py';e4='scripts/python/panel_pathformer_experiment4_multiseed.py';worker='revision_work/scripts/causal/fullrun14a_worker_v2.py'
B='''# Loss logging evidence

1. **Dropout exists in both Static and Adaptive.** Both use the dual-frequency class with eight experts (four per branch); their scale aggregator differs. `DROPOUT = 0.1` (`scripts/python/panel_pathformer_main_v2.py:78`). Every expert has intra-patch attention dropout 0.1 (`:227–232`), inter-patch attention dropout 0.1 (`:236–241`), and `ffn[2] = nn.Dropout(0.1)` (`:243–248`). Expert positional dropout is 0 (`nn.Dropout` at `:120`, argument at `:234`); branch positional dropout is also 0 (`:433`). Each model therefore has 16 attention dropout arguments at 0.1, eight explicit FFN dropout modules at 0.1, and ten positional dropout modules at 0.0. No dropout is present in the aggregator or fusion/head. Parameter propagation: expert default `:215`, branch default `:422`, expert calls `:448/:459`, dual-frequency default `:540`, branch calls `:552/:563`. Single-frequency-only defaults/calls `:490/:511` are outside these runs. Both Static and Adaptive factories delegate this architecture: Experiment3 `:224–246`, Experiment4 `:411–414`, MAIN `:589–609`.

2. **train_huber is computed in model.train() mode during updates.** It averages losses from successive batch parameter states across the epoch, with dropout active. It is not a separate full training-set pass after the epoch. The forward/loss occurs before each optimizer step. Evidence: Experiment3 `:645–662`, Experiment4 `:922–941`, MAIN `:1650–1688`.

3. **val_huber is computed after the training epoch in model.eval() under torch.no_grad().** It uses fixed end-of-epoch parameters and disabled dropout. Evidence: Experiment3 `:664–675`, Experiment4 `:943–956`, MAIN `:1690–1709`.

4. **Both are sample-weighted means of per-batch means.** The accumulation is `loss.item() * y_batch.shape[0]`, divided by the total dataset length. Evidence: Experiment3 `:662/:675/:677–678`, Experiment4 `:941/:956/:958–959`, MAIN `:1688/:1709/:1711–1712`. Batch size is 32 (`MAIN:68`); retained train/val/test counts give final batch sizes 10/18/23. Loaders do not set drop_last=True (Experiment3 `:574–575`, MAIN `:1373–1375`). Thus the smaller final batch is not given equal weight to a full batch.

5. **Both use the same Huber delta=1, reduction="mean".** `MAIN:70` fixes delta; constructors are Experiment3 `:635`, Experiment4 `:912`, MAIN `:1641`. Reduction is omitted, so the installed `nn.HuberLoss` default is `"mean"`; the same criterion object evaluates both partitions. Each sample has one scalar target. The installed PyTorch HuberLoss source location and SHA-256 are in SOURCE_MANIFEST.csv.

**History binding:** O/Static/42 → Experiment3; O/Adaptive/42 → MAIN; O/Static and O/Adaptive other seeds → Experiment4. C uses the 14a adapter, which selects the same owner by mechanism/seed (`fullrun14a_worker_v2.py:148–151/:240–250`) and aliases `train_loss/val_loss` to `train_huber/val_huber` when writing history.csv (`:274–278`). The adapter delegates the original Huber constructor and observes validation MSE without replacing loss reduction or mode (`:190–210/:227–229`). The stored completion dependency hashes verify the corresponding training source files. Source histories and exact CSV rows are in epoch1_logged_losses.csv and C_eval_mode_losses.csv. Full-precision source history floats are used for all comparisons; Step19 exports six significant digits (`prepare_step19_v2.py:274–277`).
'''
excerpts='\n## Numbered code excerpts\n\n'
for p,a,b in [(main,65,78),(main,227,248),(e3,635,678),(e4,912,959),(main,1684,1712),(worker,240,250),(worker,271,278)]:excerpts+=excerpt(p,a,b)+'\n'
textout('B_loss_logging.md',B+excerpts)
A20={r['partition']:r for r in A if r['horizon']==20}
agg=[]
for inp in ['O','C']:
 for mech in ['static','adaptive']:
    g=[r for r in epoch1 if r['input_set']==inp and r['mechanism']==mech]
    agg.append(dict(input_set=inp,mechanism=mech,n_seeds=len(g),epoch1_train_mean=sum(r['train_huber'] for r in g)/len(g),epoch1_val_mean=sum(r['val_huber'] for r in g)/len(g)))
gap=A20['train']['huber_train_mean']-A20['val']['huber_train_mean']
train_reduction=[r['logged_train_huber_best']-r['eval_train_huber'] for r in C if r['checkpoint_status']!='NOT STORED']
report='''# BLOCKING

No failed sanity check or missing required sample/history input. All nine numerical checks PASS at six significant digits. All 20 cells have a stored selected checkpoint available for evaluation, with one provenance qualification: **O/Adaptive/seed42 uses a previously stored EXACT recovery**, rather than an original-run serialized state. The recovery record reports exactly matching formal train/validation histories, best epoch and test MSE, and maximum test-prediction difference 9.97466e-17. The original MAIN trainer retains best_state in memory and does not serialize it. This cell is explicitly marked STORED_RECOVERED_EXACT; original parameter-by-parameter identity is not provable without an original serialized state. No model was trained or recovered in Step27.

The separate magnitudes attributable to dropout and within-epoch weight updates cannot be identified here. Part C measures their combined logging difference at the selected checkpoint; no train-mode counterfactual inference is performed. C/Static/seed21 selects epoch 3; the other 19 cells select epoch 1.

# STEP27 — validation below training loss

Read-only diagnostic. All new files are exclusively created within revision_work/outputs/step27/. A runtime write guard rejects writes elsewhere and writes without O_EXCL. No optimizer or training entrypoint is called. Backward is disabled during inference; every forward is under eval + no_grad; state tensors are compared unchanged. No downloads, label rebuilding, split rebuilding, training or fine-tuning. Every substantive input is rehashed after analysis and before report assembly.

## Formal panel and numerical conventions

The preserved `revision_work/outputs/causal_features/tensors/sample_index.csv:2–43572` fixes all retained keys, their order, and post-purge partitions. Its frozen labels are bitwise verified against original formal per-ticker `y_5d.npy/y_10d.npy/y_20d.npy` using stored `meta.csv` anchor dates. The formal contract passes: 17 stocks, 2563 retained dates, 43571 rows per horizon. Train: 30634 samples/1802 dates, 2016-01-29–2023-03-27. Val: 6290/370, 2023-04-26–2024-10-14. Test: 6647/391, 2024-11-12–2026-06-05. Purging the final 20 train/val anchors is documented in `scripts/python/panel_common.py:78–88`; the unchanged contract is checked at `panel_pathformer_main_v2.py:613–715`. `revision_work/scripts/causal/build_and_audit.py:148–149` saved these original labels and retained keys; `causal/core.py:45–47` binds them to formal C runs. The earlier all-stock dataset/audit/panel_sample_index.csv is not used.

Reported statistics use six significant digits. Calculations promote the exact stored float32 label values to float64, with no intermediate decimal rounding. SD is pooled population SD (ddof=0). Huber delta=1 is 0.5e² for |e|≤1 and |e|−0.5 otherwise, averaged over individual samples. The Global Train-Mean is fitted once on the full training partition per horizon, then held fixed in val/test/year subsets. Year subsets do not refit the mean. Model forwards retain the formal float32 inputs and weights; residual arithmetic and loss reductions use float64. Source history floats are read at full precision, not reconstructed from rounded Step19 exports. calculation_state.json is the machine-precision intermediate audit record used to build this report without rounding before comparisons; the report and result CSVs have six significant digits.

SOURCE_MANIFEST.csv supplies every substantive input file, text line range (or binary location), purpose, SHA-256 before/after, and unchanged check. Binary arrays/checkpoints have no text line numbers: sample_provenance.csv maps every retained sample to its frozen-index CSV line, frozen-label element, original meta CSV line, and original array row. For code, precise lines are also cited below. Repository paths are relative to the repository root; runtime library paths are absolute.

## Required sanity checks

'''+table(d['checks'])+'''
## Part A — partition dispersion

`mse_train_mean` and `huber_train_mean` evaluate the Global Train-Mean forecast; `mse_zero` and `huber_zero` evaluate zero. The CSV files contain the same numbers.

'''+table(A)+'''
### h=20 by calendar year within each partition

'''+table(AY)
r2020=next(r for r in AY if r['partition']=='train' and r['year']==2020);r2021=next(r for r in AY if r['partition']=='train' and r['year']==2021)
report+=f'''
Training 2020 has SD {fmt(r2020['sd_y_ddof0'])} and mean(y²) {fmt(r2020['mean_y_squared'])}. It contains {fmt(d['train2020_n']/A20['train']['n']*100)}% of training samples and contributes {fmt(d['train2020_y2_share']*100)}% of training summed y². By comparison, 2021 SD is {fmt(r2021['sd_y_ddof0'])}, below the overall training SD {fmt(A20['train']['sd_y_ddof0'])}. These data support a large 2020 contribution; they do not support describing both 2020 and 2021 as unusually dispersed.

## Part B — loss logging

'''+B.replace('# Loss logging evidence\n','')+'''
Numbered code excerpts are in B_loss_logging.md. Code verifies the mode/timing asymmetry but does not quantify a dropout-only effect.

## Part C — selected-checkpoint evaluation

All three full partitions are evaluated at each cell's own validation-selected checkpoint in saved formal key order, batch size 32. The original train-only per-ticker/frequency/feature normalization is imported unchanged (`panel_common.py:134–165`), and C normalization is additionally bitwise checked against saved standardized arrays. Strict state_dict loading, no gradients and unchanged model tensors PASS for all 20 cells. The absolute difference between eval validation Huber and logged best-epoch validation Huber is below 1e-7 in every cell, accounting for float32 forward and reduction precision. Paths, hashes, checkpoint provenance and exact history CSV rows are in C_eval_mode_losses.csv and SOURCE_MANIFEST.csv.

'''+table(C,['input_set','mechanism','seed','best_epoch','checkpoint_status','logged_train_huber_best','logged_val_huber_best','eval_train_huber','eval_val_huber','eval_test_huber','eval_minus_logged_train','eval_minus_logged_val'])+'''
### Epoch-1 values referenced by Figure 4

Five-seed means computed from unrounded source histories:

'''+table(agg)+'''
Each individual seed:

'''+table(epoch1,['input_set','mechanism','seed','train_huber','val_huber','train_minus_val','source_file','source_line'])
report+=f'''
The prompt's approximate ranges correspond to five-seed means. Actual per-seed epoch-1 training losses range {fmt(min(r['train_huber'] for r in epoch1))}–{fmt(max(r['train_huber'] for r in epoch1))}; validation losses range {fmt(min(r['val_huber'] for r in epoch1))}–{fmt(max(r['val_huber'] for r in epoch1))}.

## Conclusion

**(A) directly explains the principal scale of the gap; (B) is a verified additional logging contribution.** At h=20, the same Global Train-Mean forecast has Huber {fmt(A20['train']['huber_train_mean'])} on train versus {fmt(A20['val']['huber_train_mean'])} on validation, a gap of {fmt(gap)}, with corresponding SD {fmt(A20['train']['sd_y_ddof0'])} versus {fmt(A20['val']['sd_y_ddof0'])}. This already reproduces the scale of epoch-1 logged five-seed train means ({fmt(min(r['epoch1_train_mean'] for r in agg))}–{fmt(max(r['epoch1_train_mean'] for r in agg))}) versus validation means ({fmt(min(r['epoch1_val_mean'] for r in agg))}–{fmt(max(r['epoch1_val_mean'] for r in agg))}). Training logs average dropout-active batches as weights change, while validation uses the fixed end-of-epoch model without dropout. At the selected checkpoints, eval training loss is lower than logged training loss in {sum(v>0 for v in train_reduction)}/20 cells, by {fmt(min(train_reduction))}–{fmt(max(train_reduction))}, confirming a combined mode/timing contribution without isolating dropout. Validation remains below training loss when both are evaluated in eval mode in {sum(r['eval_val_huber']<r['eval_train_huber'] for r in C)}/20 cells. The lower validation target dispersion therefore explains the persistent gap, while logging adds a smaller observed contribution; no dropout-only causal magnitude is claimed.

## Input SHA-256 and locations

'''+table(manifest,['source_file','source_location','sha256','unchanged'])+'''
## Output SHA-256

All already-completed outputs are hashed below. REPORT_STEP27.md itself is hashed after closure in the detached SHA256SUMS.txt. Embedding a file's final SHA-256 within itself would change that digest; the detached checksum inventory therefore does not include its own checksum. The report/build scripts and machine-precision audit intermediate are included among outputs. No file is overwritten to insert hashes.

'''
report+=table([dict(output_file=p.name,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()])
textout('REPORT_STEP27.md',report)
textout('SHA256SUMS.txt',''.join(f'{sha(p)}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file()))
print('REPORT COMPLETE',OUT/'REPORT_STEP27.md')
