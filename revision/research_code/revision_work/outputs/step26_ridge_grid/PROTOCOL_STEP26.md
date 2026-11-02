# PROTOCOL_STEP26 — immutable preregistration

Registered before data loading, numerical computation, fitting, or outcome inspection. Repository inspection before registration was limited to paths, source code, prior protocol, and git status. All existing files are read-only; every new file is created exclusively under revision_work/outputs/step26_ridge_grid/. No existing output will be overwritten, moved, or deleted. Python bytecode disabled; caches/temp directories must be inside this directory and new files exclusive.

The full binding request below specifies the analysis and all gates. Inventory every reported Ridge fit from source/registries, including all 18 W3 raw-target cells, all rolling W1/W2 reported cells, and any CS-target fits. Deterministic order: window W3,W1,W2; raw target before CS; input O,C; configuration daily_only,weekly_only,daily_weekly; h5,10,20, limited only by existence of a reported source cell. No selective reruns or dropped cells. Preserve all failures.

Read-only inventory and missing-input checks may proceed while TEST is locked. Missing manuscript inputs block Phase D but do not by themselves prevent independent A/B work. Do not substitute another manuscript silently. A numerical reproduction failure, runtime projection over two hours, or inability to implement the original data/environment/TEST isolation halts dependent computation and is reported as BLOCKING. If stopped, do not create a selection lock, unlock TEST, fabricate unavailable values, or regenerate downstream artifacts.

TEST lock follows the earlier Step22D/22G opaque-hash/deserializer distinction: hashing bytes is permitted for preservation, but no TEST arrays, labels, forecasts, performance or portfolio records may be deserialized before all Phase B selections have been written and their lock hash recorded. Mixed TRAIN/VAL/TEST tensor containers must use the prior TRAIN/VAL-only selector, not load_frozen_panel or an unguarded loader. Enforce by explicit loader guard, source inspection and read-event log. Inventory may inspect schema/path/code/config and selected-alpha metadata only. All-cell selection barrier, not per-cell unlock. Original run_one accesses TEST and must be reused by an AST-derived selection-only wrapper preserving normalizer, features, Ridge defaults, dtype, original strict ascending-grid comparison and float arithmetic; record code provenance. Validate wrapper against stored TRAIN/VAL and original-grid selections before extended selection. Cells with no stored VAL defer prediction comparison to Phase C.

Use Step14c-B2's Ridge environment and BLAS thread controls after confirming from source; record actual Python, NumPy, SciPy, sklearn, threadpool and environment settings before fit. No fitting seeds. Time first three complete reproduction+extended-selection cells and project across the entire inventory, including observed data setup; if >7200s STOP. No alternative solver, feature transform, precision or cached eigendecomposition to accelerate. Original grid [.1,1,10,100]; extended initial grid [1e-2,1e-1,1,10,100,1e3,1e4,1e5,1e6]; symmetric two-decade boundary extensions capped at 1e-6 and 1e12 (clip next candidates at caps). Compare stored predictions aligned by exact stock/date/horizon keys; EXACT iff max absolute difference<1e-7, NEAR iff<1e-5, else DIFFERENT. Any DIFFERENT stored VAL cell stops. Missing artifacts are explicitly reported, never treated as passing evidence.

Independent validation uses a separate code path rebuilding VAL MSE selections for every completed cell; after unlock independently recompute W3 MSE and per-date Spearman Rank IC. Reuse original inference/bootstrap/portfolio/Holm code/settings, replacing only Ridge forecasts. Preserve all non-Ridge point estimates and raw p values; only joint-family Holm correction can change other-model adjusted p values. No non-Ridge training/evaluation.

Snapshot all pre-existing regular-file paths and metadata, with streaming SHA-256 of opaque bytes, before analytic work. At conclusion rehash all pre-existing files and report all discrepancies; never claim preservation beyond verified scope. Opaque hashes do not release TEST. Record runtime and all created-file hashes in a manifest (manifest cannot self-hash; report/manifest mutual-reference limitation disclosed). Report deviations explicitly; protocol never edited.

## Binding user request (verbatim)

Step 26 — Ridge benchmark: extended penalty grid (pre-registered)

CONTEXT
Repo root: <project_root>
Ridge is fitted by scripts/python/panel_baseline_ridge.py (run_one; alpha grid {0.1, 1, 10, 100}; sklearn Ridge
with its default solver; fitted on TRAIN; penalty selected by minimum pooled VAL MSE with strict comparison in grid
order). The revised manuscript states that alpha = 100, the largest candidate, is selected in 16 of 18 primary-window
cells. A selection at the upper grid boundary means the grid is truncated. This step re-selects the Ridge penalty on an
extended grid and propagates the result to every reported Ridge quantity. No other model is retrained or changed.

RULES (same as previous steps)
1. All existing files are read-only. Write only into a new directory revision_work/outputs/step26_ridge_grid/ with
   exclusive file creation. Never overwrite, move or delete anything.
2. Write PROTOCOL_STEP26.md before any computation and record its SHA-256. Do not edit it afterwards. Any deviation is
   reported as a deviation in the report, never applied silently.
3. TEST lock: no TEST labels and no TEST predictions are read until every penalty selection (Phase B) is complete and
   written to SELECTION_LOCK.csv with its SHA-256. Use the same TEST-lock mechanism as Steps 22D/22G.
4. Use the same environment controls as the earlier Ridge fits (Step14c-B2); record numpy/scipy/sklearn versions and
   BLAS thread settings. Ridge is deterministic; there are no seeds.
5. Report every cell. No cell may be dropped, re-run selectively or chosen after TEST results are seen.

PHASE A — Inventory (read-only)
A1. List every Ridge fit used in any reported result, with its source files and the downstream analyses that use it.
    At least:
    - W3 (primary window), price-level (O) and window-relative (C) inputs, Daily-only / Weekly-only / Daily+Weekly,
      h in {5, 10, 20}: 18 cells.
    - Rolling windows W1 and W2 (Step14c-B2), O and C: the Ridge cells used there.
    - Any Ridge fit with the cross-sectionally demeaned target (Supplementary Table S18), if one exists.
    - Downstream consumers: forecast-skill cells and the joint Holm family across cells (Step 5; Table 4, Table S3),
      MSE decomposition, directional accuracy (Step 21; Table S13), shrinkage toward the training mean (Step 17;
      Table S12), rolling-window results (Step14c-B2; Table 10 note, Tables S8 and S9), pooled three-window Rank IC
      (Table S15; the source of "Ridge reaches a pooled Rank IC of 0.042–0.087"), portfolio strategies built on Ridge
      (Step 8; Tables S10 and S11, and the count of |t| > 1.96 among the 480 price-level strategy–layer–horizon
      alphas), and manuscript tables/figures (Steps 18/19, Table 4, Figure 3).
A2. For each cell, record the currently selected alpha and whether it lies at a grid boundary.
A3. Step 17 found that the archived W3 price-level Ridge (MAIN V2) has TEST predictions only, without VAL predictions or
    a fitted object. Document its selected alpha from code or logs if possible, and state it if not.

PHASE B — Pre-registered selection (no TEST access)
B1. Reproduction gate. Refit every cell from A1 through the existing code path with the original grid
    {0.1, 1, 10, 100}. Compare the selected alpha and the TRAIN/VAL predictions with the stored artifacts:
    EXACT (max|Δ| < 1e-7), NEAR (< 1e-5) or DIFFERENT. Cells without stored VAL predictions (archived W3 O) are compared
    on TEST predictions only after the TEST lock is released (Phase C), and the result is reported. If any cell with
    stored VAL predictions is DIFFERENT, stop and report it under BLOCKING.
B2. Extended grid: alpha in {1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6} (a superset of the original grid).
    Selection exactly as in the original code: models fitted on TRAIN, minimum pooled VAL MSE, strict comparison in
    ascending grid order (exact ties go to the smaller alpha). Features, standardization, solver and all other settings
    unchanged. If run_one hard-codes the grid, write a wrapper in the step26 directory that reuses its feature
    construction and fitting code with the grid as a parameter, and show in B1 that the wrapper reproduces the
    original selections.
B3. Boundary rule (pre-registered). If the selected alpha equals the largest grid value, add the next two decades
    (1e7, 1e8) and re-select; repeat in steps of two decades up to a hard cap of 1e12. Apply the symmetric rule at the
    lower end down to 1e-6. Log every extension. A cell still at the cap is reported as BOUNDARY_AT_CAP; no other
    choice is made.
B4. Write SELECTION_LOCK.csv (window, input spec, input configuration, h, target, VAL MSE at every grid value,
    selected alpha, number of extensions, interior flag) and its SHA-256. Only then release the TEST lock.
B5. Runtime gate: time the first three cells and project the total; stop with a BLOCKING note if the projection exceeds
    2 hours.

PHASE C — TEST evaluation and propagation
C1. Evaluate the selected models on TEST with the existing metric functions: pooled MSE, out-of-sample R² against the
    Global Train-Mean and against the Zero Predictor, Clark–West (Newey–West lag h−1 and moving block bootstrap), mean
    Rank IC (Newey–West lag h and moving block bootstrap), directional accuracy and the MSE decomposition.
C2. Recompute every downstream analysis from A1 that uses Ridge, with the original code and settings, replacing only
    the Ridge forecasts:
    - the joint Holm family of forecast-skill cells (state whether any Holm decision changes, for Ridge or any other
      model);
    - shrinkage toward the training mean, now including the W3 O Ridge cells that were N/A;
    - rolling-window ratios to each window's Global Train-Mean, and the stock / volatility-regime / subperiod results;
    - pooled three-window Rank IC with its Holm family;
    - Ridge portfolio strategies (Sharpe ratio, turnover, maximum drawdown, alpha t-statistics, certainty-equivalent
      returns where reported) and the count of |t| > 1.96 among the 480 price-level alphas.
C3. Do not retrain, re-evaluate or modify any non-Ridge model.

PHASE D — Manuscript impact
D1. Read the manuscript and response letter in revision_work/inputs/step26/ (read-only). Extract every sentence, table
    cell and figure that reports a Ridge quantity or a quantity that depends on Ridge. Examples:
    "alpha = 100 in 16 of 18 cells"; the Ridge rows of Table 4; Ridge out-of-sample R² −8.4% → −1.2%; Rank IC 0.068;
    Daily+Weekly 20-day R² −31% → −5%; pooled Rank IC 0.042–0.087; "Ridge on price levels deteriorates there (4.7–10.3
    times)"; PLUG relative MSE 26–57; "Ridge on price levels ... produces some of the highest Sharpe ratios ... about
    56 times per year"; "only 7 alphas have |t| > 1.96"; Holm-adjusted values from joint families.
D2. Write IMPACT_STEP26.csv: location (section / table / paragraph), quoted text, old value, new value, source file,
    and whether the qualitative statement still holds (HOLDS / CHANGES / NO_LONGER_TRUE).
D3. Regenerate the affected tables and Figure 3 with the Step 18/19 code as new files with a _step26 suffix. Do not
    overwrite the existing ones.

DELIVERABLES
1. REPORT_STEP26_RIDGE_GRID.md, beginning with a BLOCKING section ("None" if nothing blocks), followed by:
   protocol SHA-256 and gates; the Phase A inventory; the B1 reproduction table; the selection table (old alpha →
   new alpha, interior flag, extension log); old versus new TEST metrics for every cell; old versus new values for
   every recomputed downstream result; a summary of IMPACT_STEP26.csv; runtime; every new file with its SHA-256; and a
   hash check confirming that all pre-existing files are unchanged.
2. VALIDATION_STEP26.md: an independent recomputation of the VAL-based selection for all cells and of the W3 TEST MSE
   and Rank IC for all Ridge cells, with any discrepancy listed.
3. A plain-language summary at the end of the report (5–8 sentences): how many cells changed alpha, whether any
   selection remains at a boundary, and which manuscript statements change.