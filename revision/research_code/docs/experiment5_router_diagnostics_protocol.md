# Experiment 5 — Adaptive Router Diagnostic / Failure Analysis Protocol

Status: **PROTOCOL ONLY — MUST BE STATIC-AUDITED AND SHA256-FROZEN BEFORE EXECUTION**

Date: 2026-08-26

This document defines the pre-computation scientific protocol for Experiment
5. It contains **no Experiment 5 result**. It exists to freeze the
diagnostic design so a later, separately implemented runner can be
statically audited against it before execution.

---

## 1. Scientific context

Experiment 3, Experiment 4 descriptive multi-seed robustness, and
Experiment 4 serial-dependence-aware inference are already finalized and
frozen.

The formal Experiment 4 result is:

> Across the five pre-specified random seeds, globally learned static
> multi-scale aggregation achieves lower out-of-sample squared forecast
> loss than sample-dependent adaptive routing at all three horizons under
> the pre-specified primary Moving Block Bootstrap.

Therefore Experiment 5 is **not** designed to establish Adaptive
superiority. Experiment 5 is a diagnostic / failure-analysis study. Its
purpose is to determine what the sample-dependent Adaptive router
actually learned, and whether its additional routing flexibility appears
stable, substantial, and descriptively useful.

---

## 2. Primary scientific questions

Experiment 5 must answer exactly four primary diagnostic questions.

**Q1. Is Adaptive routing materially sample-dependent?**

Distinguish:

a. overall sample variation;
b. within-ticker temporal variation;
c. within-date cross-sectional variation.

Do not equate cross-sectional ticker heterogeneity with dynamic temporal
adaptation.

**Q2. How concentrated or diffuse are Adaptive router allocations?**

Measure: entropy; normalized entropy; effective number of scales; maximum
router weight; dominant-scale usage.

**Q3. How stable are Adaptive routing decisions across the five
pre-specified random seeds?**

Measure the same ticker/date/horizon/frequency observation across seeds
`0, 1, 21, 42, 3407`.

**Q4. Is a larger Adaptive departure from the same-seed Static scale
allocation descriptively associated with better or worse relative squared
forecast loss?**

This is descriptive mechanism analysis only. No new hypothesis test is
introduced.

---

## 3. Scope

- Mechanisms: Adaptive and Static only. Adaptive is the mechanism under
  diagnostic study; Static is used only as the already-established
  comparison/reference allocation.
- Frequency configuration: `daily_weekly` only. Analyze the two router
  branches (`daily`, `weekly`) separately. Do **not** construct a global
  cross-frequency router.
- Horizons: `5d`, `10d`, `20d`.
- Seeds: `0, 1, 21, 42, 3407`.
- Universe: exactly the frozen Dataset Contract V2 17-stock panel.
- TEST dates: exactly the frozen 391 TEST anchor dates.
- Samples per horizon/seed/mechanism: `6647`.

---

## 4. Scale order

Canonical Daily patch order: `[5, 10, 20, 30]`.

Canonical Weekly patch order: `[2, 4, 8, 13]`.

All vector operations must preserve these exact orders. Do **not** reorder
scales by observed weight magnitude.

---

## 5. No new model execution

Experiment 5 must use **only** already-persisted frozen TEST evidence.
There is: no training; no checkpoint loading; no forward pass; no fresh
TEST prediction; no re-evaluation; no hyperparameter tuning; no
architecture modification; no model selection. Experiment 5 is strictly a
read-only analysis of frozen evidence.

---

## 6. Frozen inputs

The future Experiment 5 runner must hash-verify all required frozen
inputs before any diagnostic computation.

**A. New-seed Experiment 4 scale weights**

- `dataset/audit/experiment4_multiseed/experiment4_scale_weights.csv`
- SHA256: `ba1fd9280341dcf68d65ffe7c52675a69d77f1c214e5458cbb17e6ffc6fdb887`
- Contains new seeds `0, 1, 21, 3407` for Adaptive and Static.

**B. New-seed Experiment 4 predictions**

- `dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv`
- SHA256: `6b5f5ea94f1bb5f25ebd5ed09d7e4e7eb609945f97f953e7fb95116e2e866e8a`

**C. Adaptive seed42 router weights**

- `dataset/audit/main_v2/pathformer/panel_pathformer_router_weights.csv`
- SHA256: `bcf06150f21d35d9ec10132d248d088cc758c9782771efdf9b68849531c9814e`
- Use only: `frequency_setting == daily_weekly`, `seed == 42`, numeric
  `horizon` in `{5, 10, 20}`. Map `5 -> "5d"`, `10 -> "10d"`, `20 -> "20d"`.

**D. Adaptive seed42 predictions**

- `dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv`
- SHA256: `a05955a561da1b8880f1bb0878a6576852c087cc45707546d446ce96b0fabdaf`
- Apply the same seed42 / `daily_weekly` / horizon filtering.

**E. Static seed42 scale weights**

- `dataset/audit/experiment3_pathformer_ablation/experiment3_scale_weights.csv`
- SHA256: `219e908f0c3f75bd9b38d7fe466242a15a64d8c41b1f235ae75c91df5f6eb248`
- Use only: `mechanism == static`, `seed == 42`, `horizon` in
  `{"5d", "10d", "20d"}`.

**F. Static seed42 predictions**

- `dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv`
- SHA256: `95d1b06cc3ce0209e99833c29a1c3672a6d7a1a9d46d32f16ef1e48fa9846515`

**G. Frozen Experiment 4 inference context** (gate files only; Experiment 5
must not reinterpret or overwrite Experiment 4 inference)

- `dataset/audit/experiment4_inference/experiment4_inference_manifest.csv`
  SHA256: `93801b778f7ef1d60794b2777ffa5b983cb1bc03a2ef425df5551dbe7358f89e`
- `dataset/audit/experiment4_inference/experiment4_inference_summary.txt`
  SHA256: `3f3e2f67097ba2859374832ca06386e03e6a0620776a9919559a1d3ab0c0c35b`

**H. Frozen Experiment 4 paired seed deltas**

- `dataset/audit/experiment4_multiseed/experiment4_seed_pair_deltas.csv`
- SHA256: `2f2a66689ac6abc8371565c2d8567c5ffa08259e5cb4879dd1fb306645c7f103`
- Purpose: authoritative 15-cell descriptive Adaptive-minus-Static MSE
  bridge. The future runner must hash-verify this file before any
  diagnostic computation.

Also freeze the exact five-seed descriptive mean `delta_mse` audit
constants (used only for the loss bridge audit in Section 18; these are
audit constants, not newly estimated Experiment 5 results):

- `5d`: `0.0003479534760117531`
- `10d`: `0.0005263369530439377`
- `20d`: `0.002097966521978378`

---

## 7. Canonical weight table

Normalize frozen scale-weight evidence into an internal table keyed by
`mechanism`, `horizon`, `seed`, `ticker`, `anchor_date`, with fields:

```
daily_w_5, daily_w_10, daily_w_20, daily_w_30
weekly_w_2, weekly_w_4, weekly_w_8, weekly_w_13
```

Require exactly `2 mechanisms x 3 horizons x 5 seeds = 30 cells`. Each
cell: `6647` rows. Total canonical mechanism-weight rows: `199410`.

Require for every cell:

- 17 unique tickers;
- 391 unique dates;
- exactly 17 observations per date;
- no duplicate ticker/anchor_date keys.

Require identical ticker/date keys between Adaptive and Static for each
horizon/seed. Require identical ordered TEST keys across all five seeds
within horizon.

`anchor_date` must be parsed as datetime first and serialized later as
`YYYY-MM-DD`.

---

## 8. Router weight validity

For every Adaptive and Static sample:

```
w_D = [daily_w_5, daily_w_10, daily_w_20, daily_w_30]
w_W = [weekly_w_2, weekly_w_4, weekly_w_8, weekly_w_13]
```

Require: all four weights finite; every weight `>= 0`; every weight
`<= 1`; absolute difference between the vector sum and `1` `<= 1e-6`. Do
**not** renormalize invalid weights. Any violation: **FAIL CLOSED**.

---

## 9. Static weight audit

Static is expected to be sample-independent within each `horizon x seed x
frequency`. For each of the four scale dimensions:

```
max(weight) - min(weight) <= 1e-10
```

Otherwise: **FAIL CLOSED**.

After this audit passes, define the canonical Static reference vector `s`
for that `horizon x seed x frequency` cell **exactly** as the
component-wise arithmetic mean of all 6647 persisted Static sample weight
vectors in that cell, computed in `float64`. Do **not**: select the first
row; use the median; renormalize; or round the vector.

After constructing `s`, require its component sum to remain within
`1e-6` of `1` and all components finite and within `[0, 1]`. Any
violation: **FAIL CLOSED**.

Do **not** require Adaptive weights to vary — a nearly or exactly constant
Adaptive router is a scientifically valid diagnostic result and must not
cause failure.

---

## 10. Vector distance

Freeze the Total Variation (TV) distance between two four-scale weight
vectors:

```
TV(a, b) = 0.5 * sum_k |a_k - b_k|
```

Range: `[0, 1]`. Use TV distance for all router-distance diagnostics. Do
**not** substitute Euclidean distance, cosine distance, Jensen-Shannon
divergence, or KL divergence.

---

## 11. Entropy / concentration

For each Adaptive sample weight vector `w`:

```
H(w) = - sum_{k: w_k > 0} w_k * ln(w_k)     (natural log; 0*ln(0) = 0)

H_norm(w) = H(w) / ln(4)                     range [0, 1]

N_eff(w) = exp(H(w))                         range [1, 4]

max_weight(w) = max_k w_k

dominant_scale = argmax_k w_k
```

Tie-breaking: use the **first** scale in canonical scale order — an exact
tie is resolved toward the smaller patch size. This rule is mechanical
only and must not be given economic meaning.

---

## 12. Q1 — Sample dependence

For every `horizon x seed x frequency`, using Adaptive weights only. Let
the 6647 sample vectors be `w_i`. Define the cell mean vector:

```
w_bar = (1/N) * sum_i w_i
```

### 12.1 Overall sample-dependence index

```
sample_dependence_index = (1/N) * sum_i TV(w_i, w_bar)
```

Captures overall sample-level variation around the cell mean router
allocation.

### 12.2 Within-ticker temporal variation

For each ticker `j`, compute that ticker's mean vector across its TEST
observations, `w_bar_j`. Then:

```
temporal_tv_j = mean over observations i in ticker j of TV(w_i, w_bar_j)

within_ticker_temporal_tv = equal-weight mean of temporal_tv_j across exactly 17 tickers
```

Every ticker receives equal weight. This metric isolates temporal router
movement within a ticker.

### 12.3 Within-date cross-sectional variation

For each TEST date `t`, compute the equal-weight mean router vector
across exactly 17 tickers, `w_bar_t`. Then:

```
crosssectional_tv_t = mean across the 17 tickers on date t of TV(w_it, w_bar_t)

within_date_crosssectional_tv = equal-weight mean across exactly 391 dates
```

Captures cross-sectional differentiation at a given date.

Do **not** interpret `sample_dependence_index` alone as proof of temporal
adaptation. The temporal and cross-sectional components must be reported
separately.

---

## 13. Q2 — Router concentration

For each `horizon x seed x frequency`, summarize Adaptive sample-level
`normalized_entropy`, `effective_scales`, `max_weight`. For each metric,
report: mean; sample SD (`ddof=1`); median; p10; p90 — using
`np.quantile(..., method="linear")` for p10/median/p90.

Also report, for each of the four scales: mean router weight; sample SD
of router weight (`ddof=1`).

---

## 14. Dominant scale usage

For each `horizon x seed x frequency x patch_size`, report `count` and
`share`, where `share = count / 6647`. Require the four shares to sum to
`1` within `1e-12`. Do **not** selectively report only the dominant
scale — all four scale shares must be persisted.

---

## 15. Q3 — Cross-seed routing stability

Use Adaptive only. For every `horizon x ticker x anchor_date x
frequency`, there must be exactly five Adaptive weight vectors
corresponding to seeds `0, 1, 21, 42, 3407`. There are `C(5,2) = 10` seed
pairs.

Compute all 10 pairwise TV distances. Define:

```
mean_pairwise_seed_tv = mean of the 10 pairwise TV distances
max_pairwise_seed_tv = maximum of the 10 pairwise TV distances
```

For each sample/frequency, also compute the dominant patch for each seed.
Define:

```
dominant_agreement_share = (largest number of seeds selecting the same dominant patch) / 5

dominant_unanimous = 1 if dominant_agreement_share == 1.0 else 0
```

Possible `dominant_agreement_share` values include `0.4, 0.6, 0.8, 1.0`.

For each `horizon x frequency`, summarize:

- `mean_pairwise_seed_tv`: mean, sample SD, median, p90;
- `max_pairwise_seed_tv`: mean, median, p90;
- `dominant_agreement_share`: mean, median;
- `dominant_unanimous_share`: mean of `dominant_unanimous`.

No inferential test.

---

## 16. Q4 — Adaptive vs Static distance

For every `horizon x seed x frequency`, let `s` be the verified
sample-independent Static scale vector. For every Adaptive sample `i`:

```
tv_to_static_i = TV(w_i, s)

tv_cell_mean_to_static = TV(w_bar, s)   (w_bar = Adaptive cell mean vector)
```

For sample-level `tv_to_static`, report: mean, sample SD, median, p10,
p90.

The distinction must be retained:

- `tv_cell_mean_to_static`: systematic difference between Adaptive's
  average allocation and Static;
- variation around the Adaptive mean: sample-dependent routing movement;
- `tv_to_static`: combined observed departure from Static.

Do not collapse these concepts into one statement.

---

## 17. Prediction / loss bridge

Build canonical Adaptive and Static prediction tables using the same
source rules as Experiment 4 inference. For each `horizon x seed`,
require exactly 6647 paired rows, sorted by `ticker`, `anchor_date`, with
exact key equality. Cast persisted `y_true`/`y_pred` to `np.float32`, and
use the same float32 loss semantics frozen in Experiment 4 inference:

```
adaptive_error32 = (y_true32 - adaptive_pred32) ** 2
static_error32 = (y_true32 - static_pred32) ** 2
loss_diff32 = adaptive_error32 - static_error32
loss_diff = float64(loss_diff32)
```

Positive: Adaptive has higher squared forecast loss. Negative: Adaptive
has lower squared forecast loss. Require finite. Do **not** reconstruct
elementwise loss differences from summary MSE values.

The Adaptive and Static prediction is one joint Daily+Weekly model
output. Therefore, within a fixed `horizon x seed x ticker x anchor_date`,
the same sample-level `loss_diff` is attached to **both** router
diagnostic rows (`frequency == daily` and `frequency == weekly`). The
router-distance variable differs by branch, but the forecast-loss outcome
is the same joint D+W outcome. This duplication is intentional and
deterministic. Do **not** compute a frequency-specific forecast loss,
because no separate D/W prediction exists for this D+W Adaptive-vs-Static
contrast.

Interpretation guardrail: a branch-specific router-distance/loss
correlation is a descriptive association between that branch's routing
departure and the JOINT D+W forecast loss. It does **not** identify the
causal or separable performance contribution of that branch.

---

## 18. Loss bridge audit

Require exactly 15 bridge rows in the frozen reference table
(`experiment4_seed_pair_deltas.csv`, Section 6.H) and exactly one matching
row for every `horizon x seed`.

For each `horizon x seed`, the mean of the 6647 float64-cast elementwise
`loss_diff` values must match its one corresponding `delta_mse` row in
`experiment4_seed_pair_deltas.csv` within `rtol=1e-8, atol=1e-8`.

Then compute the equal-weight arithmetic mean across the five seed-level
reconstructed deltas for each horizon, and require agreement with the
exact frozen constants from Section 6.H (`5d`, `10d`, `20d`) within
`rtol=1e-8, atol=1e-8`.

Any mismatch: **FAIL CLOSED**.

---

## 19. Q4 — Loss-association diagnostic

This is explicitly **DESCRIPTIVE**. No p-values. No confidence intervals.
No bootstrap. No HAC. No regression significance test.

For each `horizon x seed x frequency`, pair `x_i = tv_to_static_i`,
`y_i = loss_diff_i`. Compute a deterministic Spearman rank correlation,
implemented manually:

1. Rank `x` using average ranks for ties.
2. Rank `y` using average ranks for ties.
3. Compute the ordinary Pearson correlation of the two rank arrays using
   the following frozen formula, on `float64` rank arrays `rx`, `ry`:

```
rx_centered = rx - mean(rx)
ry_centered = ry - mean(ry)

denominator = sqrt(sum(rx_centered ** 2) * sum(ry_centered ** 2))

if denominator == 0:
    rho = NaN
    correlation_defined = False
else:
    rho = sum(rx_centered * ry_centered) / denominator
    correlation_defined = True
```

The exact zero-variance condition is `denominator == 0`; no epsilon
threshold may be substituted. Require defined `rho` finite and within
`[-1, 1]` up to ordinary floating-point roundoff.

Do not use: `scipy.stats.spearmanr`; p-values; `np.corrcoef` as an
alternative definition; or arbitrary zero-variance epsilon thresholds.

If either rank array has zero variance, the
correlation is undefined: return `NaN` and `correlation_defined = False`.
Do **not** fail closed, because a constant router-distance series is
itself a valid diagnostic result.

Interpretation of sign:

- `rho > 0`: larger Adaptive departure from Static is descriptively
  associated with higher Adaptive-minus-Static loss, i.e. relatively
  worse Adaptive forecast loss.
- `rho < 0`: larger Adaptive departure from Static is descriptively
  associated with lower Adaptive-minus-Static loss, i.e. relatively
  better Adaptive forecast loss.
- `rho` near 0: no clear monotonic descriptive association.

Do **not** attach significance language to `rho`.

---

## 20. Date-level association

For each `horizon x seed x frequency x anchor_date`, take the
equal-weight average across exactly 17 tickers of `tv_to_static`
(`mean_tv_to_static`) and `loss_diff` (`mean_loss_diff`). Then compute the
Spearman correlation across exactly 391 dates using the same manual
average-rank definition.

Persist `sample_level_spearman_rho` and `date_level_spearman_rho`, and
separate `_defined` flags. This distinction prevents the diagnostic from
relying only on pooled cross-sectional observations.

---

## 21. Across-seed association summary

For every `horizon x frequency`, summarize the five seed-level
`sample_level_spearman_rho` values and, separately, the five
`date_level_spearman_rho` values.

Sign counts use only defined `rho` values, with the exact frozen
definitions (no "near-zero" threshold may be introduced for counting):

```
positive: rho > 0.0
negative: rho < 0.0
zero:     rho == 0.0
```

Frozen undefined-correlation summary rule, applied separately to
sample-level and date-level rho for each `horizon x frequency`:

- If `n_defined == 0`: `mean = NaN`, `sd = NaN`, `median = NaN`,
  `n_positive = 0`, `n_negative = 0`, `n_zero = 0`.
- If `n_defined == 1`: `mean` = the one defined rho, `median` = the one
  defined rho, `sd = NaN`.
- If `n_defined >= 2`: `mean` = arithmetic mean of defined rho, `median`
  = median of defined rho, `sd` = sample SD with `ddof=1`.

Do not perform t-tests, sign tests, bootstrap, Fisher-z inference, or Holm
correction. This section is descriptive only.

---

## 22. No event / regime discovery

Experiment 5 must **not**:

- label dates as market regimes;
- inspect news/events to explain individual router movements;
- cherry-pick specific ticker/date examples;
- search for macro variables that correlate with router weights;
- claim router weights identify economic regimes;
- claim router weights identify investment opportunities.

Any future regime analysis would require a separate protocol frozen
*before* examining such relationships.

---

## 23. No performance-claim reversal

Experiment 5 cannot override Experiment 4 inference. Even if some
diagnostic subset appears favorable to Adaptive, do **not** claim
Adaptive superiority based on a post-hoc subset. The formal performance
conclusion remains the frozen Experiment 4 result. Experiment 5 may
explain or characterize mechanism behavior only.

---

## 24. Output namespace

The future runner writes **only** to:

```
dataset/audit/experiment5_router_diagnostics/
```

No existing Experiment 3 / 4 / MAIN V2 evidence may be modified.

Final planned outputs:

1. `experiment5_router_sample_metrics.csv`
2. `experiment5_router_cell_summary.csv`
3. `experiment5_dominant_scale_usage.csv`
4. `experiment5_cross_seed_stability_by_sample.csv`
5. `experiment5_cross_seed_stability_summary.csv`
6. `experiment5_loss_association_by_seed.csv`
7. `experiment5_loss_association_summary.csv`
8. `experiment5_router_diagnostics_summary.txt`
9. `experiment5_router_diagnostics_manifest.csv`

---

## 25. Output 1 — sample metrics

`experiment5_router_sample_metrics.csv` — Adaptive only. One row per
`horizon, seed, ticker, anchor_date, frequency`.

Expected rows: `3 x 5 x 6647 x 2 = 199410`.

Exact columns:

```
horizon, seed, ticker, anchor_date, frequency,
patch_1, patch_2, patch_3, patch_4,
weight_1, weight_2, weight_3, weight_4,
entropy, normalized_entropy, effective_scales, max_weight, dominant_patch,
tv_to_cell_mean, tv_to_static, loss_diff
```

Patch columns must be: Daily rows `5, 10, 20, 30`; Weekly rows
`2, 4, 8, 13`.

Ordering: horizon `5d, 10d, 20d`; then seed `0, 1, 21, 42, 3407`; then
ticker ascending; then `anchor_date` ascending; then frequency `daily,
weekly`.

---

## 26. Output 2 — cell summary

`experiment5_router_cell_summary.csv`

Expected rows: `3 horizons x 5 seeds x 2 frequencies = 30`.

Exact columns:

```
horizon, seed, frequency, n_samples, n_tickers, n_dates,
mean_weight_1, mean_weight_2, mean_weight_3, mean_weight_4,
sd_weight_1, sd_weight_2, sd_weight_3, sd_weight_4,
sample_dependence_index, within_ticker_temporal_tv, within_date_crosssectional_tv,
normalized_entropy_mean, normalized_entropy_sd, normalized_entropy_median,
normalized_entropy_p10, normalized_entropy_p90,
effective_scales_mean, effective_scales_sd, effective_scales_median,
effective_scales_p10, effective_scales_p90,
max_weight_mean, max_weight_sd, max_weight_median, max_weight_p10, max_weight_p90,
static_weight_1, static_weight_2, static_weight_3, static_weight_4,
tv_cell_mean_to_static,
tv_to_static_mean, tv_to_static_sd, tv_to_static_median, tv_to_static_p10, tv_to_static_p90
```

No additional diagnostic columns.

Canonical ordering: horizon, then seed, then frequency `daily` then
`weekly`.

---

## 27. Output 3 — dominant scale

`experiment5_dominant_scale_usage.csv`

Expected rows: `3 x 5 x 2 x 4 = 120`.

Columns: `horizon, seed, frequency, patch_size, dominant_count,
dominant_share`.

Persist all four scales even if `count == 0`.

---

## 28. Output 4 — cross-seed sample stability

`experiment5_cross_seed_stability_by_sample.csv` — Adaptive only. One row
per `horizon, ticker, anchor_date, frequency`.

Expected rows: `3 x 6647 x 2 = 39882`.

Columns: `horizon, ticker, anchor_date, frequency, n_seeds,
mean_pairwise_seed_tv, max_pairwise_seed_tv, dominant_agreement_share,
dominant_unanimous`.

`n_seeds` must equal `5` everywhere.

---

## 29. Output 5 — cross-seed summary

`experiment5_cross_seed_stability_summary.csv`

Expected rows: `3 horizons x 2 frequencies = 6`.

Exact columns: `horizon, frequency, n_samples,
mean_pairwise_seed_tv_mean, mean_pairwise_seed_tv_sd,
mean_pairwise_seed_tv_median, mean_pairwise_seed_tv_p90,
max_pairwise_seed_tv_mean, max_pairwise_seed_tv_median, max_pairwise_seed_tv_p90,
dominant_agreement_share_mean, dominant_agreement_share_median,
dominant_unanimous_share`.

No additional diagnostic columns.

---

## 30. Output 6 — loss association by seed

`experiment5_loss_association_by_seed.csv`

Expected rows: `3 x 5 x 2 = 30`.

Columns: `horizon, seed, frequency, n_samples, n_dates,
sample_level_spearman_rho, sample_level_correlation_defined,
date_level_spearman_rho, date_level_correlation_defined`.

---

## 31. Output 7 — loss association summary

`experiment5_loss_association_summary.csv`

Expected rows: `3 x 2 = 6`.

Exact columns:

```
horizon, frequency,
sample_level_rho_mean, sample_level_rho_sd, sample_level_rho_median,
sample_level_n_defined, sample_level_n_positive, sample_level_n_negative, sample_level_n_zero,
date_level_rho_mean, date_level_rho_sd, date_level_rho_median,
date_level_n_defined, date_level_n_positive, date_level_n_negative, date_level_n_zero
```

Use only defined rho
values for mean/SD/median, per the frozen undefined-correlation summary
rule in Section 21. No additional diagnostic columns.

---

## 32. Output 8 — human-readable summary

`experiment5_router_diagnostics_summary.txt`

Must report, without overclaiming, for each horizon and frequency:

- Adaptive average scale allocation;
- overall `sample_dependence_index`;
- within-ticker temporal TV;
- within-date cross-sectional TV;
- normalized entropy;
- effective number of scales;
- max-weight concentration;
- dominant-scale shares;
- Adaptive mean-vector TV from Static;
- mean sample TV from Static;
- cross-seed mean pairwise TV;
- dominant-scale seed agreement;
- sample-level distance/loss Spearman;
- date-level distance/loss Spearman.

The text may use: "more/less variable", "more/less concentrated",
"more/less seed-stable", "descriptively associated".

It must **not** use: "significant", "causes", "proves", "economic
regime", "alpha", "profitable", "optimal routing" — unless a future
separately frozen protocol establishes those claims.

### Human-summary across-seed aggregation (frozen)

The human-readable summary is organized by `horizon x frequency` and must
**not** select individual seeds for narrative emphasis.

For seed-specific cell-summary metrics, report the equal-weight
arithmetic mean across the five pre-specified seeds `0, 1, 21, 42, 3407`.
This applies to: Adaptive mean scale allocation; `sample_dependence_index`;
`within_ticker_temporal_tv`; `within_date_crosssectional_tv`;
`normalized_entropy_mean`; `effective_scales_mean`; `max_weight_mean`;
`tv_cell_mean_to_static`; `tv_to_static_mean`.

For dominant-scale usage: for each patch, report the equal-weight
arithmetic mean of the five seed-specific `dominant_share` values.

For cross-seed stability: report the corresponding `horizon x frequency`
row directly from `experiment5_cross_seed_stability_summary.csv`.

For loss association: report `sample_level_rho_mean` and
`sample_level_n_defined`, and `date_level_rho_mean` and
`date_level_n_defined`, from `experiment5_loss_association_summary.csv`.

The summary may mention seed dispersion only using pre-specified
persisted summary fields. Do not cherry-pick individual seeds, tickers,
dates, or patches for narrative emphasis.

---

## 33. Interpretation framework

Do **not** freeze arbitrary numerical thresholds such as "entropy below X
means collapsed" or "TV above X means unstable". Interpret continuous
evidence comparatively. Permitted diagnostic patterns include:

**A.** Low `sample_dependence_index` + low temporal TV + Adaptive mean
close to Static -> Adaptive behaves approximately like a static
allocation.

**B.** Meaningful sample variation + large cross-seed TV / low
dominant-scale agreement -> sample-dependent routing exists but is not
stable across stochastic training realizations.

**C.** Meaningful sample variation + high seed stability + positive
loss-association rho -> router deviations are structured/stable but
larger deviations are descriptively associated with worse relative
squared loss.

**D.** Meaningful sample variation + high seed stability + negative
loss-association rho -> some descriptive evidence that larger routing
departures may coincide with lower relative loss, but this does **not**
overturn the frozen formal Experiment 4 aggregate result.

These are interpretive templates, not mechanical classifications.

---

## 34. Manifest

`experiment5_router_diagnostics_manifest.csv` — exactly one row. Record
at minimum:

```
run_family, temporal_status, dataset_contract, diagnostic_scope,
mechanisms, frequency_configuration, frequencies, horizons, seeds,
n_tickers, n_test_dates, n_samples_per_cell,
daily_patch_order, weekly_patch_order,
distance_metric, entropy_log_base, entropy_normalization,
effective_scale_definition, dominant_tie_rule,
sample_dependence_definition, temporal_variation_definition,
crosssectional_variation_definition, cross_seed_definition,
loss_association_definition, spearman_tie_rule,
no_training = true, no_checkpoint_loading = true,
no_fresh_test_predictions = true, no_hypothesis_testing = true,
no_regime_analysis = true, no_subset_model_selection = true,
protocol_path, protocol_sha256,
<all frozen input paths and SHA256 values>,
<output row counts>,
<SHA256 of outputs 1-8>
```

Do not include the manifest's own SHA inside itself.

---

## 35. Fail-closed governance

The future runner must:

- hash-verify every frozen input;
- verify the protocol SHA;
- fail if any final Experiment 5 output already exists;
- validate exact schemas / row counts / ordering;
- use temporary files;
- validate serialized temp outputs;
- hash temp outputs;
- build the manifest;
- atomically commit;
- roll back only outputs committed by the current run if commit fails;
- never modify frozen upstream evidence.

---

## 36. Protocol workflow

1. Create this protocol only.
2. Independent static audit.
3. SHA256-freeze the protocol.
4. Implement the Experiment 5 runner.
5. Static-audit the runner.
6. SHA256-freeze the runner.
7. User manually executes exactly once.
8. Independent evidence audit.
9. Freeze Experiment 5 outputs.
10. Only then write the final router interpretation.

No diagnostic result may be generated before step 3.

---

## 37. Final scientific guardrail

Experiment 5 is mechanism characterization after a negative
Adaptive-vs-Static performance result. It is not a search for a favorable
subgroup or a narrative that rescues Adaptive.

The experiment should help distinguish among: approximately static
Adaptive behavior; genuinely sample-dependent but unstable routing;
stable but unhelpful routing; or a more nuanced combination.

Whatever the diagnostic outcome, the frozen Experiment 4 performance
inference remains authoritative.

---

## 38. No unregistered diagnostics

After this protocol is SHA256-frozen, the Experiment 5 runner may compute
**only** the diagnostic quantities explicitly defined in this protocol.
No additional distance metric, concentration metric, correlation,
subgroup, threshold, aggregation, event analysis, regime analysis, or
output column may be added based on observed Experiment 5 results.

Any scientifically useful additional diagnostic discovered later requires
a separate protocol amendment frozen *before* computing that additional
quantity.

Implementation-only validation/provenance fields in the one-row manifest
are allowed, but no new scientific diagnostic output is allowed.

