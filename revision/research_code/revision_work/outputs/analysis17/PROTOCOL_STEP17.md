# PROTOCOL STEP17 — pre-registered before TEST metric computation

Created UTC: 2026-10-19T09:08:55.304968+00:00

## Scope and frozen source rule

No training, model fitting/refitting, checkpoint inference, patch/seed search or pipeline changes. Reuse the same Daily+Weekly TEST forecasts as Prompt16: O/C Single, Fixed, Static, Adaptive, LSTM, Vanilla, SWiM and Ridge in W3; O/C Static, Adaptive and Ridge in W1/W2; h=5/10/20. Neural/mechanism seeds are0,1,21,42,3407 plus an equal-weight ensemble; Ridge is one deterministic predictor. O W3 neural seed42 is B5FIX and other seeds are B5; C neural seed42 is14c-day and other seeds B5. O Static/Adaptive and O Single/Fixed seed42 retain the verified archived formal forecasts. C mechanisms are14a12/12. Rolling windows reuse B2/B2 supplement unchanged.

The grid applies to the ensemble, not separately to individual seeds. Ridge has one deterministic forecast rather than an invented five-seed ensemble. Planned base forecast cells per regime: W3=258 and W1/W2=78 each, total414. Planned λ-selection groups=84; two regimes yield828 displayed TEST-result rows, including blocked/missing rows.

## Pre-registered combination and selection

For each window/horizon, m is the float64 mean of that window's frozen TRAIN labels, imported through the Step16/Step5 reference-forecast path. f_lambda=m+lambda*(f-m). Endpoints are evaluated exactly: lambda0 returns m; lambda1 returns the original f. Equal weighting of the five forecasts uses the imported Step7B ensemble function with sorted seeds0,1,21,42,3407.

PRIMARY selects one lambda for every model×input×horizon×window from0,0.1,...,1 by the smallest pooled VALIDATION MSE of the five-seed ensemble (Ridge: its single forecast). Evaluate all11 candidates in float64 and ascending order; exact minimum ties go to smaller lambda. Validation-only selection is persisted and hash-bound before computing ANY TEST forecast-skill metric. Apply the same selected lambda unchanged to the TEST ensemble and all five seed forecasts. Per-seed validation curves are saved descriptively but never choose lambda. No reselection after TEST.

SECONDARY uses exactly lambda0.5, with no tuning. Both regimes keep every requested cell, including lambda0 and blocked PRIMARY cells. The selection group and validation curve are shared with individual-seed rows; Ridge is not replicated into five seeds.

## Missing inputs, before TEST metrics

Initial repository search found no saved VAL predictions or fitted checkpoint for O W3 archived Ridge. The formal Ridge exporter saves TEST predictions only. If complete VAL forecasts from this exact existing fit are not found during preflight, PRIMARY for O Ridge W3 h5/10/20 remains unavailable: lambda and metrics are NA, with explicit BLOCKING. Do not reconstruct VAL by refitting, substitute another window/input/model, or use TEST to choose lambda. SECONDARY lambda0.5 is still evaluable from frozen TEST forecasts. Missing VAL curves are explicitly NA, rather than omitted.

Other sources must have a matching saved VAL forecast and a paired TEST forecast reproducing the Prompt16 canonical TEST predictions EXACTLY (max absolute delta<1e-7). Verify TRAIN/VAL/TEST labels, keys, split row counts and finite values. Never replace a failed source silently. All TEST scores use Prompt16 canonical values; the pair match only verifies that VAL comes from the same numerical source/qualified checkpoint path.

## Metrics, invariance and tests

Report selected/used lambda, all11 validation MSEs, TEST MSE, R2_OS vs Global Train-Mean and Zero, mean date-level cross-sectional RankIC and per-date comparison to the uncombined forecast. Positive lambda preserves ordering algebraically; check finite-sample RankIC equality using the imported rank function. Lambda0 is a constant forecast: RankIC is undefined, not zero and not an invariance PASS.

Reuse Step5 cell_metrics for scores and CW, Step5 rank_ic_rows, mbb_index and holm, Step5 a3_hierarchical.series_stats for two-sided centered MBB p/percentile95% CI, and Step7B ens_series for ensemble and paired date-loss aggregation. No new implementation of these formulas.

CW versus Train-Mean uses unchanged one-sided Step5 Newey-West (lag h-1) and MBB. Algebraically CW_f(lambda)=lambda*CW_f(1); for lambda>0 evaluate the canonical original forecast's CW p once and carry it unchanged, checking the scale identity numerically. This avoids apparent p differences caused solely by floating-point threshold ties. At lambda0, CW is undefined; raw p=N/A. No IC p-value or new model-vs-model test is added.

The loss comparison is first-minus-second: squared error(f_lambda) minus squared error(m), average over17 stocks within each date, then dates. Negative favors the combined forecast. Use a two-sided centered moving-block-bootstrap p with+1 correction and percentile95% CI, L=h, B=20000; noncircular overlapping blocks with ceil(T/L) blocks, truncate391 dates. Tests condition on the saved forecasts/seed ensemble and selected validation lambda; no new joint seed/bootstrap uncertainty is introduced. Lambda0 has identically zero loss difference, interval[0,0] and p=1, and remains included.

## Multiplicity frozen before TEST

Literal single-family interpretation: within PRIMARY, combine all three requested p-values for all414 planned cells (CW-NW, CW-MBB, two-sided loss-MBB) into ONE Holm family of1242 entries. SECONDARY has its own one family of1242 entries. No splitting by input/model/horizon/window/seed/ensemble or method. Undefined CW at lambda0 and unavailable PRIMARY cells retain conservative p=1 placeholders in the adjustment; their original raw p remains NA. They cannot create a rejection and remain in the family count. Report adjusted p for each method and count per method, plus any-test cell counts without double counting cells. The loss-MBB count is the pre-specified forecast-accuracy significance count. Repeated invariant CW hypotheses are retained conservatively as requested cell-wise entries.

An optional clarification question was raised on this interpretation. If the user explicitly settles an alternative BEFORE TEST metrics, record an exclusive dated protocol amendment and freeze it before the metric gate. Otherwise the conservative single-family definition above governs; no post-TEST relaxation will be presented as this pre-registration.

## RNG and arithmetic

Bootstrap entropy for each window/horizon: SeedSequence([20261019,17,window_number,h]), with window_number1/2/3. Common date indices pair forecasts/regimes/inputs within that window/horizon. B=20000, L=h; RNG draws generated only after validation selection is frozen. Persist entropy, draw order, actual index files and checksums. Statistics run with BLAS/thread pools fixed to1 for deterministic arithmetic, with no torch import/inference. All predictions/labels/metrics are float64 from the saved values; validate labels by exact float32 equality to the formal frozen panels. Exact rank checks use the same stored-value arithmetic; any numerical mismatch is reported, not relabelled invariant.

## Verification, reporting and retention

Write source inventory and SHA256, preflight checks, TRAIN means, complete VAL curves, selected lambdas and a pre-TEST gate with protocol/selection hashes. Only then compute TEST metrics. Report all828 requested regime/cell rows. Save combined TEST forecasts, date losses/IC checks, method raw/adjusted p and a full test ledger. Count all displayed p entries, undefined placeholders and valid tests separately. No subgroup tests, portfolio work, new IC tests or extra shrinkage grids. Digest contains O/C side-by-side W3 model×horizon rows and W1/W2/W3 Static/Adaptive/Ridge rows, with ensemble/Ridge metrics and linked per-seed mean/SD (ddof1), without averaging individual metrics into an ensemble.

Existing files are read only. New code is only under revision_work/scripts/analysis17; outputs only revision_work/outputs/analysis17. Exclusively create every new file; hash-verified resume skips completed phases. Never overwrite a protocol, selection, existing output or original source. BLOCKING comes first in REPORT_STEP17.md. Use descriptive wording limited to the recorded tests; no data-vintage, historical identity or leakage claims.
