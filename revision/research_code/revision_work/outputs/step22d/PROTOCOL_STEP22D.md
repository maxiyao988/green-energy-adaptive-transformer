# Step22D pre-registered pooled rolling-window analysis and LSTM confirmation

Immutable after its SHA-256 is recorded in RNG_MANIFEST.json, before any Step22D training or TEST-array/prediction processing. All existing files are read-only. All new files are exclusively created under revision_work/outputs/step22d/. If the specification cannot be implemented, STOP and ask; never replace a setting or select cells after outcomes are seen. W3 remains the manuscript's primary evaluation; every pooled result here is supplementary.

Repository discovery consulted existing design/reports and CSV schema examples. No new Step22D statistic, model inference, raw TEST tensor/label contents or canonical TEST prediction arrays was processed before this protocol. The clarified read order below governs all execution.

## 1. Scope and immutable sources

Part1: existing five-seed W1/W2/W3 Static and Adaptive Daily+Weekly forecasts, Ridge, Zero, and each window/horizon's Global Train-Mean; O and C; h5/10/20. Use Step16 registry/canonical sources underlying the original-window/Table10 comparisons. W1/W2 predictions retain all seeds0,1,21,42,3407 from B2 plus its supplement. W3 is Step16's original window, not a retraining or a new setting. Ridge retains its one stored predictor; Zero/GTM are the same Step16 reference definitions, not five artificial seeds.

Use the existing B2 W1/W2 tensor packs and partitions, without rebuilding or moving boundaries. TRAIN rows16660(W1),23647(W2),30634(W3); VAL6290 and TEST6647 in each window. Each TEST has391 dates/17 stocks; W1/W2/W3 TEST dates are disjoint,1173 pooled. TRAIN expands from the common first anchor; weekly availability, labels,20-anchor purges, Daily90/Weekly26 and per-ticker/resolution/channel TRAIN-only ddof0 feature normalization stay unchanged.

Part2:60 new C Daily+Weekly LSTM fits: settings T0/T3 × windows W1/W2 × horizons5/10/20 × five seeds. T0 lr1e-4/Huber delta1; T3 lr3e-5/Huber delta sigma_h,w. Sigma is the **sample TRAIN-target SD (ddof1)** in float64 from each window's frozen float32 labels, computed without rounding. Record TRAIN n, mean, population SD and sample SD; only sample SD sets T3 delta. This matches the sample-SD convention behind the supplied rounded W3 constants; W3 T3 is reused as produced with its original rounded constants, not retuned.

LSTM class, initialization, dropout, Adam, batch32, clip1.0, max100 epochs, patience10, validation-Huber minimum checkpoint/restoration and global run-seed shuffled-loader convention(generator=None) exactly follow the formal LSTM/Step22B path. Input data and window-specific normalization match the rolling Static/Adaptive pipeline. Import the existing trainers, data-loader binding and inference block; do not reimplement the model or loop. CPU12/12; OMP_NUM_THREADS unset; OMP_DYNAMIC=FALSE/MKL_DYNAMIC=FALSE before Torch import; set Torch intra/inter12 immediately after import. Three training worker slots, refilled as fits finish. Keep AC power and caffeinate -i; if on battery, prepare but automatically wait for AC before launch, without further approval.

## 2. Clarified read order and gates

Before training, verify W1/W2 partition/key/label identities by the existing manifests and SHA-256 records. Opaque streaming hashes of frozen tensor/label files are permitted; do not deserialize or index TEST label values, open TEST predictions or compute metrics. Confirm the same label/index hashes occur in the original rolling-run dependencies and existing key/float32-label checks. TRAIN/VAL-only row slices may be materialized to verify input equality, normalizers and sigma.

User-authorized fallback only if a required file lacks a stored hash: compare that file's keys and float32 labels against its existing source, outputting **only PASS/FAIL and maximum absolute difference**, with start/end timestamps; no predictions or metrics. If manifests/hashes are available, do not use fallback. Any partition/key/float32 mismatch stops training. Record all gate evidence.

Part1 and **all TEST evaluation/metrics** are deferred until all60 Part2 fits have verified terminal records and ALL_TRAINING_FINISHED.json is written. Never start training after this barrier. No pre-training TEST prediction replay is needed. All post-training comparisons include failed/degenerate cells as N/A rather than dropping seeds.

Timing pilot: T0/W1/h20/seed0, T0/W2/h20/seed0, T3/W2/h20/seed0, concurrently. These are three of the60 fits, reused. D is actual UTC elapsed from first pilot launch until all three processes exit and artifacts are verified; total projection=20D for60 fits/3 workers. If projection>3h, STOP and report without TEST. A failed pilot makes the projection undefined and stops. Decisions depend only on time. Record UTC and monotonic elapsed separately.

Remaining order T0 then T3; seeds0,1,21,42,3407; horizons20,10,5; W1 then W2; skip completed pilots. Each fit is a fresh process. Save checkpoint, per-epoch train/validation Huber and val MSE, TRAIN/VAL predictions, threads/env/load/top-CPU telemetry at start/every5 minutes/end and power status. Immutable journals/status snapshots and exclusive attempt folders support restart; preserve partial attempts and retry only those without a terminal record; retain recorded failures.

## 3. Part1 pooled definitions

All three windows carry their391 TEST dates. Static/Adaptive date-level Rank IC = arithmetic mean of **the five per-seed date IC values**, not the IC of averaged forecasts. Ridge IC as is. Zero/GTM IC is undefined, not zero. Record all finite-support counts; do not substitute fewer seeds. If a requested five-seed/day series is not fully defined, preserve dates, report N/A and conservative multiplicity placeholder1; do not silently reduce the calendar or select seeds.

MSE is the mean stock/date squared loss, averaged over five seeds for Static/Adaptive; it is not their ensemble MSE. Pooled R2_OS =1−sum(model squared errors)/sum(reference squared errors), using each window's own TRAIN mean or zero, pooling errors rather than averaging window R2. Step16's float64 mean of frozen TRAIN labels defines GTM. Report every statistic per window and pooled W1+W2+W3, with denominators and counts.

(a) Mean IC and window NW SE(lag h). Let a_w=391/1173. Pooled NW variance=sum_w a_w^2 SE_w^2, assuming independence of window means as specified. Pooled mean=sum_w a_w mean_w; two-sided normal p=2*sf(abs(mean/SE)). Stratified noncircular moving-block bootstrap B20000,L=h draws dates independently within each window, never crossing boundaries; pool each replicate with the same weights. Report percentile95% interval and centered two-sided MBB p. Ridge gets mean/SE/descriptive interval only, **no p or significance flag**. Per-window Static/Adaptive raw component p-values/intervals are diagnostic references, not pooled decision families; no per-window significance flag or extra Holm family.

(b) Pooled and per-window R2 against own-window GTM and Zero are descriptive only. No R2 tests.

(c) Static−Adaptive MSE/Rank-IC contrast: independently within each window resample five seeds as **paired arm units**, plus that window's date blocks. Combine window replicate means using a_w. Keep arm/date pairing and all dates. Report first−second estimate,95% percentile interval, centered two-sided p and Holm across h within input×metric. Also report per-window component differences, intervals and raw component p for reference, with no per-window Holm/significance decision. Adaptive improvement =D for MSE and−D for Rank IC; q0.95 of this signed distribution is its one-sided95% upper bound. For MSE express D/interval/bound as100×value/(second arm's fixed five-seed mean MSE for the same pooled/window scope); negative bounds retained. These are Step22C-style **pointwise**, not simultaneous bounds.

## 4. Families and primary method (user clarification)

Newey–West is the **PRIMARY method** for IC; stratified MBB is a robustness check. Both IC p-values are two-sided. Adjust NW and MBB separately, never mix their p-values:

- PRIMARY: pooled Static/Adaptive IC, C, h5/10/20:6 tests per method, Holm within6.
- SECONDARY: same O:6 tests per method, Holm within6.
- CONFIRMATORY Rank IC: LSTM T3 pooled W1+W2:3 tests per method, Holm within3.

An IC is flagged significantly positive **only** if mean>0 and **NW Holm p<.05**. Report adjacent MBB Holm p and explicitly state every NW/MBB flag disagreement. MBB alone never sets the flag. Undefined tests stay in their complete family using conservative p=1; raw p remains N/A. No Ridge/Zero/GTM IC test. Part1 pooled Static−Adaptive contrasts have four MBB Holm families(input×metric), each3 horizons. Per-window diagnostics do not set a claim.

## 5. Part2 confirmatory and descriptive analysis

W1+W2 only for confirmation; W3 is excluded because T3/LSTM were singled out there. Confirmation IC series is **the average of the five per-seed date IC series**, exactly Part1's definition. Ensemble-forecast date IC is descriptive only, with no test.

CW: equal-weight five-seed T3 ensemble forecast in each window versus its own GTM. Form f=(y−m)^2−[(y−forecast)^2−(m−forecast)^2], average stocks per date, then pool W1/W2. Report the manuscript's one-sided NW and stratified centered MBB p-values; NW lag h−1 for CW, as the existing Step5 implementation, and within-window L=h for MBB. Combine window NW mean variances with weights1/2, assuming independence. Apply Holm separately across3 horizons to NW and MBB. NW is primary, MBB robustness; positive CW mean and NW Holm p<.05 sets the primary CW flag. An ensemble identical to GTM makes CW undefined, with conservative family placeholder1. No CW test on individual seeds, T0, W3 or any other model.

Confirmation IC: T3 W1+W2 seed-averaged date IC, Part1 NW lag h and date-stratified MBB, two-sided, separate3-test Holm families and NW-only positive flag as above. No new per-window LSTM IC tests.

Descriptive: T0/T3 LSTM per seed, five-seed mean±sample SD and equal-weight five-seed ensembles, per W1/W2/W3, pooled W1+W2 and pooled W1+W2+W3. Metrics MSE, mean Rank IC, DA(excluding y exactly0, prediction0 against nonzero is wrong), R2_OS against own-window GTM and Zero. Pooled DA uses total correct/total eligible pairs, not the average window DA. W3 reuses Step22B T0 and Step22B2 T3 stored predictions without inference or retraining. Missing seeds make five-seed/ensemble statistics N/A. Report every run/cell regardless of outcome.

## 6. RNG and implementation

All new stratified draws fixed at base20261021. For each h and wcode1/2/3: date stream SeedSequence([20261021,h,wcode,0]); paired-seed stream SeedSequence([20261021,h,wcode,1]). Both independent across windows; dates are common to all models/inputs/tests at a horizon/window. B20000, L=h, overlapping noncircular blocks, ceil(391/h) blocks then truncate391, using existing mbb_index. Pair seed identities within each resampled window; seed draws between windows are independent. The same date streams serve Part1 date-only IC, Part1 seed×date contrasts and W1/W2 confirmation. Save indices, seed weights, replicate arrays, inputs, method labels and hashes. Reuse existing NW, Holm, Rank-IC, CW-formula and bootstrap-application code by import/AST where needed; no replacement model or training loop.

## 7. Report and completion

REPORT_STEP22D.md starts with BLOCKING, then protocol hash, clarified methods/read order, gates/timing, all per-window/pooled tables, PRIMARY/SECONDARY/CONFIRMATORY adjusted p-values and NW/MBB disagreements, contrast bounds, per-run failures and provenance. No seed/horizon/specification is selected or dropped after TEST. Preserve complete attempts, checkpoints, predictions/losses/date and stock summaries, all reference source hashes and final validation. Write completion record only after automatic evaluation, both analysis parts and report. Play completion sound. Training may start automatically once gates and AC pass; no manual code command or further approval is required.
