# Step22E v2 — immutable protocol

This protocol was written after Step22D was completed. Step22D evaluated the same W1/W2 test windows with the LSTM (T0, T3) and with Static/Adaptive (T0). No Static or Adaptive T3 forecasts for W1 or W2 existed when this protocol was written.

All existing files, including Step22D, remain read-only. New files are exclusively created under revision_work/outputs/step22e/. Record this document's SHA-256 in RNG_MANIFEST.json before training; never edit it afterwards. If the specification cannot be implemented, stop and ask. The user will launch the terminal command; preparation, import checks and synthetic QA do not launch any training. No TEST predictions/label contents or performance metrics are read during preparation, training or retries. Opaque streaming SHA-256 is permitted explicitly. W3 remains the primary manuscript evaluation; these analyses are supplementary.

## 1. Scope, data and numerical provenance

Train60 new cells: Static/Adaptive × W1/W2 × h5/10/20 × seeds0,1,21,42,3407; C window-relative inputs, Daily+Weekly, T3 only. Reuse the exact Step22D/B2 tensor packs, split boundaries, labels, weekly availability, expanding TRAIN and TRAIN-only normalization. W1 TRAIN16660 rows; W2 TRAIN23647; VAL6290 and TEST6647 in each window. Each TEST has391 dates and17 stocks; W1+W2 has782 disjoint dates. W3 is excluded from every confirmatory/secondary family below.

T3 learning rate3e-5; Huber delta is the unrounded float64 TRAIN-target sample SD(ddof1) already saved in Step22D session_v3/SIGMAS.json. Read and assert identity with Step22D's full-precision table and root copy; do not estimate a different SD. Exact values:

| Window | h5 | h10 | h20 |
| --- | --- | --- | --- |
| W1 | 0.09283665233929897 | 0.1323544958863391 | 0.18240982323839283 |
| W2 | 0.10130533354640803 | 0.14392570252994666 | 0.20182610403267889 |

T0 comparators are the60 existing C Static/Adaptive W1/W2 cells in the Step16 registry used by Step22D Part1. Verify their existing key/float32-label evidence and tensor dependencies as Step22D did. Never retrain T0. W3 Static/Adaptive T0 uses Step22D sources; W3 T3 uses Step22B2. LSTM T0/T3 uses the Step22D prediction-file registry, including its W3 sources. Do not substitute seeds or specifications. No earlier planned T3 Static/Adaptive W1/W2 grid was found during read-only discovery.

## 2. Hyperparameters — T0 versus T3

Only learning rate and Huber delta differ. All model/optimizer/loader behavior is imported from the same formal trainer path as each existing T0 cell. No model or training loop is rewritten.

| Hyperparameter / rule | T0 (reused) | T3 (new) |
| --- | --- | --- |
| Model and configuration | Static or Adaptive; Daily+Weekly | identical |
| Seed42 trainer | Static: Experiment3 train_and_validate_cell; Adaptive: main_v2 train_one_cell | identical |
| Other seeds | Experiment4 train_and_validate_cell | identical |
| Inputs / targets | C; Daily90×5, Weekly26×5; raw h-day log-return float32 labels | identical |
| Normalization | ticker×resolution×channel, TRAIN entries only, ddof0; window-specific | identical |
| Width / attention / FFN | d_model32;4 heads; FFN64; ReLU | identical |
| Scale experts |4 per branch; Daily5/10/20/30; Weekly2/4/8/13; intra/inter attention with residuals, LayerNorm and occupancy pooling | identical |
| Positional encoding | sinusoidal base10000; capacity1024; positional dropout0 | identical |
| Model dropout |0.1 in expert attention/FFN | identical |
| Input projection | biased Linear5→32 per branch | identical |
| Static aggregation |4 zero-initialized trainable logits per branch; softmax; initially0.25 | identical |
| Adaptive router | per branch Linear128→32, ReLU, Linear32→4, softmax; no extra router dropout | identical |
| Prediction head | concatenate branches64→Linear64→ReLU→Linear1; no added dropout | identical |
| Parameters | Static107785; Adaptive116297; all trainable | identical |
| Initialization | unchanged formal factories/PyTorch2.2 defaults; exact source hashes and model structure recorded; no custom reinitialization | identical |
| Optimizer | Adam; betas0.9/0.999; eps1e-8; weight_decay0; amsgradFalse; maximizeFalse; foreach/fusedNone; capturable/differentiableFalse | identical |
| Learning rate |1e-4 |3e-5 |
| Scheduler | none | none |
| Training loss | mean Huber, raw target; delta1 | mean Huber, raw target; delta=stored sigma_h,w |
| Batch size |32 |32 |
| Train / VAL loaders | shuffleTrue/False; generator42; num_workers0; pin_memoryFalse; drop_lastFalse | identical |
| Run initialization RNG | Python/NumPy/Torch seeded with run seed | identical |
| Gradient clipping | L2 max_norm1.0, original finite-loss/gradient checks | identical |
| Maximum epochs / patience |100 /10 | identical |
| Checkpoint criterion | sample-weighted validation training-Huber loss; improvement>1e-12; restore lowest validation-loss state | identical, using T3 Huber delta |
| Device / dtype | CPU; float32 training/model inputs | identical |
| Numerical profile | Torch intra12/inter12; OMP_NUM_THREADS unset; OMP_DYNAMIC=FALSE; MKL_DYNAMIC=FALSE before Torch import | identical |
| Frozen split-contract binding | original B2 window-dependent row counts, Exp4 N_TRAIN_SAMPLES rebound in-process; formal loader generator remains42 | identical |

Other constructor/default attributes are unchanged because the original factories and training prefixes are imported. Archive factory repr, parameter counts, imported-module hashes and exact prefix/binding hashes. No new patch, architecture, optimizer or seed search. This lists the settings that govern every training decision; T0's own historical validation checkpoint remains its comparator.

## 3. Gates, attempts and execution

Data gate before training: opaque SHA-256 of the stored W1/W2 tensor/index/label bytes must match Step22D's16 recorded hashes and the original tensor manifest. Confirm prior T0 run dependencies and existing key/float32-label checks. Import Step22D's TRAIN/VAL-only data binding; its normalizer must bitwise reproduce the frozen C standardized TRAIN/VAL tensors. Copy sigma values exactly and assert equality; do not deserialize TEST. Every TEST lock explicitly permits the original pure SHA function and blocks prediction/label deserialization until the all-training barrier. Record gate timestamps and all source hashes.

First timing batch, three workers: Static/W1/h20/seed42 (Exp3), Adaptive/W2/h20/seed42 (main_v2), Static/W2/h20/seed0 (Exp4). These are three of the60, kept and reused. Projected total = actual first-batch UTC elapsed×20; STOP if>9hours. Include attempts/retry time in this batch elapsed. Record monotonic time too. No later outcome changes this rule.

Remaining queue: seeds0,1,21,42,3407; h20,10,5; W1/W2; interleave Static/Adaptive. Three training slots refill as processes finish; skip verified successful pilot cells. CPU12/12 and dynamic threads off before model work. Launch the queue under caffeinate -dimsu, record its full command and PID, maintain AC power (wait for AC before launching fits). Save start/end UTC, wall time, epochs/best epoch, profile/env/load/top-CPU telemetry at start/every5minutes/end, power snapshots, checkpoint, TRAIN/VAL predictions, history with train/VAL Huber and VAL MSE, Static logits and per-sample routing weights.

Interruption policy: keep a terminal successful COMPLETE record after artifact verification; never rewrite it. A run without one restarts from scratch with the identical seed/configuration, preserving every old attempt. Each failed or interrupted attempt is logged. One failure permits one identical retry; the same run's second failed/interrupted attempt stops the whole queue with TEST locked. A crash after a verified model COMPLETE but before an ancillary adapter/power sidecar does not invalidate the successful model record: retain it and document missing ancillary metadata without fabricating it. Do not retry an already successful cell. A checksum/source mismatch stops rather than overwriting or repairing frozen inputs. Live orphan workers must finish or exit before a restarted controller launches replacements; never exceed3 workers. A sleep that merely suspends still-live processes is not a new attempt. Immutable attempt folders, event/status journals and an OS lock make restart auditable.

Only after all60 cells have verified terminal successful records, write ALL_TRAINING_FINISHED.json. Never launch training after that barrier. All new TEST inference, old TEST prediction reads, comparisons and statistics occur after it. Save TRAIN/VAL/TEST rows with ticker, anchor_date, split, y_true, y_pred, horizon/seed/window provenance; router weights where applicable, Static logits, per-date/per-stock losses and summaries. Verify split counts, uniqueness, keys and float32 labels. Inference uses eval mode and the unchanged formal batched helper. Failed/undefined results are retained; a repeated training failure or failed timing/data gate stops before TEST. Statistical undefined cells retain p=1 in their declared Holm family (raw p=N/A), never drop a family member.

## 4. Shared statistics and RNG

Reuse Step22D's imported/extracted function definitions for per-seed metrics, date-only MBB, paired seed×date bootstrap application, NW SE combination, percentile intervals, and Step5/16 Holm/Rank-IC/CW code. Nested definitions may be imported by compiling their unchanged AST definitions into the new output namespace; record source hashes and binding changes. Do not rewrite models or estimators.

Rank IC is per-date Spearman across17 stocks for each seed, then a fixed arithmetic mean over all5 seeds per date. It is not IC of ensemble forecasts. A constant or unsupported cross-section is undefined. Preserve all dates; no fewer-seed substitution. An incomplete test series is N/A with conservative p=1 in its family. W1/W2 have391 dates each,782 pooled. NW variance of pooled means=sum a_w² SE_w², a_w=1/2, under the Step22D independence assumption; do not claim that assumption is verified. NW Rank-IC lag=h; CW lag=h−1.

Bootstrap B20000, L=h; overlapping noncircular blocks, ceil(391/h) starts then truncate391. Draw independently within each window, never across a window boundary. Pair arms by seed ID within a window, independently resample five seed units in each window. Common date blocks across models, inputs and tests for a given window/horizon. Root entropy20261022: date SeedSequence([20261022,h,wcode,0]); paired-seed SeedSequence([20261022,h,wcode,1]), wcode1/2/3. Save date/seed indices, weights, input matrices and every20000-replicate array. Seed order0,1,21,42,3407. Record package versions and streams in RNG_MANIFEST.json.

MSE = mean stock/date squared loss for each seed; five-seed summaries are mean±sample SD(ddof1), not ensemble MSE. Ensemble = equal-weight arithmetic mean of the5 forecasts. DA excludes exactly-zero realized returns; prediction zero versus nonzero is incorrect. Pooled DA uses total correct/total eligible pairs. R2_OS=1−total model squared error/total benchmark squared error using each window's own float64 TRAIN mean; do not average window R². No R² tests. All pooled windows are equally weighted by their equal391 dates/6647 rows.

## 5. E2, E3, E4, E1 — complete fixed families

E2 CONFIRMATORY CW: Static-T3 and Adaptive-T3 equal-weight5-seed ensemble versus own-window TRAIN mean, pooled W1+W2;2 models×3h=6 tests. CW f=(y−m)²−[(y−forecast)²−(m−forecast)²], mean stocks then dates. One-sided NW and stratified centered date-MBB p as Step22D; Holm separately over all6 per method. NW PRIMARY; MBB robustness. Flag only positive CW mean and NW Holm p<.05. E2 flagged interpretation: "In held-out windows, the T3 ensemble forecasts beat the historical-mean benchmark." Otherwise "not confirmed in held-out windows". An ensemble identical to the benchmark is undefined/p=1 placeholder.

E3 CONFIRMATORY setting effect: T3−T0 individual-seed MSE, pooled W1+W2, paired by seed within window. Joint seed×date bootstrap, two-sided centered p; Holm across3h separately for Static and Adaptive. Report raw MSE difference,95% percentile interval and100×difference/(T0's fixed five-seed pooled mean MSE), with interval endpoints scaled by the same denominator. User clarification: the joint MBB is the decision method for this family; NW is **N/A**, not an additional test. Labels: "significant improvement" only if difference<0 AND MBB Holm p<.05, triggering "the MSE improvement from T3 replicates in held-out windows"; "significant deterioration" if difference>0 AND MBB Holm p<.05, interpreted as "T3 increased MSE in held-out windows; the W3 improvement did not replicate"; otherwise "not significant". It remains a two-sided test, with unchanged p-values.

E4 CONFIRMATORY routing: Static-T3−Adaptive-T3 for MSE and Rank IC, W1+W2. Same joint bootstrap; two-sided p; Holm across3h separately within metric. User clarification: joint MBB is the decision method and NW is **N/A**. Direction-specific labels: MSE difference<0 favours Static,>0 favours Adaptive; Rank-IC difference>0 favours Static,<0 favours Adaptive, only if MBB Holm p<.05; otherwise "not significant". Report whichever outcome occurs. Also report Step22C bounds: Adaptive improvement=D for MSE and−D for IC; one-sided95% upper bound=q.95 of that signed bootstrap distribution; retain negatives. MSE difference/CI/bound as percent use Adaptive's fixed five-seed pooled mean MSE (the second arm), not a random denominator. Bounds are pointwise, not simultaneous. No new bound tests.

E1 SECONDARY in every table: Static-T3 and Adaptive-T3 pooled W1+W2 Rank IC,6 tests, two-sided NW/MBB with separate6-test Holm families. NW primary, MBB robustness. Positive flag only mean>0 and NW Holm p<.05. E1 is secondary evidence only, never confirmation, because Step22D already revealed the T0 ICs. Next to E1 report the identical method's mean/SE/CI/raw p for Static-T0/Adaptive-T0, labelled **descriptive, no flag**; they are not added to E1's Holm family and have no new adjusted p or significance label. Raw component p here are references only. Do not promote them to confirmation.

NW-primary flag rules apply to E1/E2 only; they do not create unregistered NW tests for E3/E4. Final E3/E4 tables show raw joint-MBB p, joint-MBB Holm p, NW raw/Holm=N/A, and the direction-specific labels above. Explicitly list any NW/MBB adjusted rejection disagreement for E1/E2, including negative means as well as positive flags. MBB alone never sets an E1/E2 flag.

## 6. Descriptive outputs, no extra tests/flags

For Static, Adaptive and LSTM; T0/T3; W1,W2,W3 and pooled W1+W2 / W1+W2+W3: per-seed metrics and five-seed mean±SD TEST MSE, Rank IC, DA, R² versus own-window TRAIN mean; equal-weight ensemble metrics; best epochs. Retain all cells and source provenance. W3 T3 sigma uses the original rounded Step22B2 constants; W1/W2 use Step22D's unrounded values, as specified. Do not retrain W3 or LSTM.

Static diagnostic: report both the invariant centered-logit mean absolute magnitude (logits minus their branch mean; equal logits imply uniform aggregation) and mean absolute softmax-weight deviation from0.25, by seed/branch and summaries. This distinguishes logits from probabilities without changing model behavior. Pool windows descriptively (all have391 test dates); report per-window best epochs and their distribution, rather than invent a pooled training epoch. Where a required archived history/logit is absent, report NOT STORED and do not substitute another run. Other descriptive metrics have no new flags/tests.

For three-window pooled Static-T3/Adaptive-T3 Rank IC: mean and NW normal95% interval only (mean±1.959963984540054×pooled NW SE), labelled "includes the selection window W3". No p, Holm, flag or MBB interval is used as evidence for this descriptive line. Window-independent NW variance combination as above, weights1/3.

## 7. Reporting and independent validation

REPORT_STEP22E_FINAL.md: BLOCKING first; protocol hash; gates/actual timing/caffeinate invocation/restarts; exact sigma table; all descriptive tables; E2,E3,E4,E1 with raw/adjusted p and explicit N/A NW for E3/E4; NW/MBB disagreements; every cell and training attempt(status, epochs, best epoch, elapsed/error); file/source list and saved replicate arrays. E1 SECONDARY in every table; no post-TEST selection or re-specification. W3 remains primary, all new analysis supplementary.

After report generation, run an independent validation pass and write REPORT_STEP22E_VALIDATION.md: verify immutable source/protocol hashes, sigma identity, counts/keys/float32 labels, same60 planned successful configurations with preserved failed attempts, thread/env profiles, actual TEST read order, family sizes/directions/NW-vs-MBB rule, and stored-replicate reproduction of intervals/bounds. Validation computes no new hypothesis tests. Play completion sound only after the complete pipeline and validation. A failed gate or twice-failed cell produces a STOP report with TEST locked; no silent setting change, seed drop, overwrite or forced bypass.
