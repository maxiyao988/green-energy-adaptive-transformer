# Experiment 4 — Serial-Dependence-Aware Inference Protocol

Status: **PROTOCOL ONLY — FROZEN BEFORE EXECUTION**

This document defines the statistical-inference protocol for the next stage
of Experiment 4 (Adaptive vs Static multi-seed robustness). The descriptive
Experiment 4 evidence (five-seed Adaptive-vs-Static robustness) is already
**FINALIZED / FROZEN**. This protocol is written *before* any statistical
inference result is generated, and it contains **no inference result**.

This document is documentation only. It does not train, evaluate, run
inference, or compute any statistic. It exists to freeze the inferential
design so that a later, separately implemented inference runner can be
statically audited against it before execution.

---

## 1. Current frozen Experiment 4 status (context, not new evidence)

- Primary contrast: Adaptive vs Static
- Frequency: `daily_weekly` only
- Horizons: `5d`, `10d`, `20d`
- Seeds: `0, 1, 21, 42, 3407`
  - Seeds `0, 1, 21, 3407`: new Experiment 4 evidence.
  - Seed `42`: frozen external reuse — Adaptive from MAIN V2, Static from
    Experiment 3. Seed 42 was **NOT retrained** and **NOT re-evaluated**.
- TEST panel: 17 tickers, 391 TEST anchor dates, 6647 ticker-date
  observations per model/horizon/seed.
- Primary descriptive metric: pooled TEST MSE.
- Frozen sign convention:

  ```
  delta_mse = Adaptive MSE - Static MSE
  ```

  - `delta_mse < 0` => Adaptive lower loss
  - `delta_mse > 0` => Static lower loss

### 1.1 Frozen descriptive point estimates (evidence only, not inferential)

These values are descriptive evidence only and are **not** inferential
results. They are recorded here so that the later inference runner can
audit its own point-estimate reconstruction against them.

Mean five-seed `delta_mse`:

- 5d: `0.0003479534760117531`
- 10d: `0.0005263369530439377`
- 20d: `0.002097966521978378`

The later inference construction must reproduce these point estimates up
to the frozen numerical audit tolerance defined in Section 3.

---

## 2. Primary loss differential

For each horizon `h`, seed `s`, ticker `i`, TEST anchor date `t`, define:

```
d[s,i,t,h]
=
(y_true[i,t,h] - y_pred_adaptive[s,i,t,h])^2
-
(y_true[i,t,h] - y_pred_static[s,i,t,h])^2
```

- Positive: Static has lower squared error.
- Negative: Adaptive has lower squared error.

The comparison is strictly paired on identical ticker/date observations.

---

## 3. Aggregation hierarchy

For each seed/date/horizon:

```
d_seed_date[s,t,h] = equal-weight mean across the 17 tickers of d[s,i,t,h]
```

Require exactly 17 tickers on every date.

Then, for each date/horizon:

```
d_date[t,h] = equal-weight mean across the five pre-specified seeds of d_seed_date[s,t,h]
```

This produces exactly **391 date-level observations per horizon**.

The primary estimand is:

```
theta_hat[h] = mean over the 391 TEST dates of d_date[t,h]
```

Because the panel is balanced and the seed weights are equal,
`theta_hat[h]` must reproduce the frozen five-seed mean `delta_mse` (Section
1.1) up to strict numerical tolerance:

- `rtol = 1e-8`
- `atol = 1e-8`

If the point-estimate reconstruction fails, inference must **FAIL CLOSED**.

---

## 4. Inferential scope

The formal statistical inference is conditional on the five pre-specified
model seeds `0, 1, 21, 42, 3407`.

The time-series bootstrap (Section 6) quantifies sampling uncertainty
across the frozen 391-date TEST period while preserving temporal
dependence. It does **not** estimate uncertainty over a hypothetical
infinite population of random model initializations.

Seed sensitivity is separately documented by the already-frozen descriptive
Experiment 4 cross-seed dispersion (see the CURRENT STATUS section of
`project_progress.md`), not by this inference protocol.

---

## 5. Numeric semantics

For persisted predictions:

1. Cast `y_true` to `np.float32`.
2. Cast `y_pred` to `np.float32`.
3. Construct elementwise squared errors from these float32 values.
4. Cast the resulting loss differential to `float64` for cross-sectional,
   seed, date, bootstrap, and HAC aggregation.

Do not infer loss differentials from summary metrics. Do not alter
persisted predictions.

### 5.1 Anchor-date canonicalization

Freeze:

- parse `anchor_date` with pandas datetime parsing;
- reject `NaT`;
- canonical representation for keys/output: `YYYY-MM-DD`;
- sort chronologically ascending;
- require exactly 391 unique ordered TEST dates per horizon.

Do not use lexical string ordering before datetime parsing.

---

## 6. Primary inference method — Moving Block Bootstrap (MBB)

Use a fixed-length **moving block bootstrap (MBB)** on the chronological
391-observation `d_date[t,h]` series, separately for each horizon. This is
a date-block bootstrap.

For a block length `L`, define all overlapping blocks:

```
B_j = (d_j, ..., d_{j+L-1})   for j = 1, ..., T-L+1
```

where `T = 391`.

For each bootstrap replicate:

1. Let `K = ceil(T / L)`.
2. Sample `K` blocks independently with replacement from the full set of
   overlapping blocks.
3. Concatenate the sampled blocks.
4. Truncate the concatenated series to exactly `T = 391` observations.
5. Compute the mean of the resampled 391-observation series.

Do **NOT** independently resample individual dates. Do **NOT**
independently resample tickers. Do **NOT** independently resample seeds.

### Block-length unit convention

MBB block length `L` is measured in consecutive ordered TEST anchor-date
**observations** in the 391-date series. It is **not** calendar-day
length.

Thus `L=5` means 5 consecutive TEST anchor dates, `L=10` means 10
consecutive TEST anchor dates, etc.

### 6.1 Primary block lengths

Freeze the primary block length to the forecast horizon:

- `5d`  -> `L = 5`
- `10d` -> `L = 10`
- `20d` -> `L = 20`

Rationale: the forward-return targets overlap mechanically across adjacent
forecast origins, and the horizon-matched block spans the corresponding
target overlap scale.

These are the **PRIMARY** block lengths. Do not choose block length from
inference results.

### 6.2 Pre-specified block-length sensitivity

Freeze exactly one sensitivity specification:

```
L = 2 x horizon
```

therefore:

- `5d`  -> `L = 10`
- `10d` -> `L = 20`
- `20d` -> `L = 40`

The sensitivity analysis is mandatory and must always be reported. The
primary conclusion remains based on `L = h`.

If the primary and sensitivity specifications produce different
significance/classification conclusions, report:

> "the inference is sensitive to the pre-specified block-length choice."

Do **NOT** select whichever block length gives the more favorable result.

### 6.3 Bootstrap parameters

Freeze:

- Bootstrap replications: `B = 20,000`
- Base RNG seed: `20260826`
- Confidence level: `95%`
- CI type: two-sided percentile bootstrap confidence interval

Use independent deterministic RNG streams for every `(horizon,
block_length)` specification:

```python
np.random.SeedSequence([20260826, horizon_integer, block_length])
# then:
np.random.default_rng(...)
```

where `horizon_integer` is `5`, `10`, or `20`. This prevents code-loop
ordering from changing bootstrap results.

### Bootstrap dispersion / quantile convention

Freeze:

```python
bootstrap_se = np.std(bootstrap_means, ddof=1)

np.quantile(bootstrap_means, [0.025, 0.975], method="linear")
```

Do not allow an alternative interpolation/quantile method.

### 6.4 Bootstrap output statistics

For every horizon x block-length specification, report:

- `horizon`
- `block_length`
- `specification`: `PRIMARY` or `SENSITIVITY`
- `T = 391`
- `B = 20000`
- point estimate `theta_hat`
- bootstrap mean
- bootstrap standard error
- 2.5% percentile
- 97.5% percentile
- two-sided 95% percentile CI
- two-sided centered-bootstrap p-value

Centered-bootstrap p-value:

```
p = (1 + count(abs(theta_bootstrap - theta_hat) >= abs(theta_hat))) / (B + 1)
```

---

## 7. Primary null and decision rule

For each horizon:

- `H0`: `theta[h] = 0`
- `H1`: `theta[h] != 0`

The primary decision rule uses the **PRIMARY `L = h`** 95% MBB percentile
CI.

- If `CI_lower > 0`, report:

  > "Static has statistically lower expected squared forecast loss than
  > Adaptive under the primary date-block bootstrap specification."

- If `CI_upper < 0`, report:

  > "Adaptive has statistically lower expected squared forecast loss than
  > Static under the primary date-block bootstrap specification."

- If zero lies inside the CI, report:

  > "No statistically distinguishable difference in expected squared
  > forecast loss is established under the primary date-block bootstrap
  > specification."

Avoid the phrase "definitively superior".

### 7.1 Supplementary primary-horizon multiplicity transparency

The PRIMARY horizon-level decision rule above is unchanged: it is the
PRIMARY `L=h` two-sided 95% percentile MBB CI, evaluated independently per
horizon.

Additionally, report Holm-adjusted centered-bootstrap p-values (Section 6.4)
across the three PRIMARY-specification horizon tests, using the exact Holm
algorithm frozen in Section 9.1, as a **SUPPLEMENTARY** multiplicity-
transparency diagnostic.

This does **NOT** replace the primary percentile-CI rule and does **NOT**
create an omnibus test. Do not adjust the SENSITIVITY block-length
p-values for this primary multiplicity table.

### 7.2 MBB `classification` field (frozen)

Freeze the exact mechanical semantics of the `classification` field in
`experiment4_block_bootstrap_summary.csv`, applied identically to both
`PRIMARY` and `SENSITIVITY` rows, for every horizon x block specification:

```
if ci_lower_95 > 0:
    classification = "MBB_STATIC_LOWER_LOSS"
elif ci_upper_95 < 0:
    classification = "MBB_ADAPTIVE_LOWER_LOSS"
else:
    classification = "MBB_NO_DISTINGUISHABLE_DIFFERENCE"
```

The `else` case includes: zero strictly inside the interval;
`ci_lower_95 == 0`; `ci_upper_95 == 0`.

Classification is based **only** on the percentile CI endpoints.
`centered_p_value` and Holm-adjusted p-values do **NOT** determine this
classification.

- For `PRIMARY` rows, this classification implements the formal primary
  decision rule already frozen in Section 7.
- For `SENSITIVITY` rows, this classification is diagnostic only and does
  **NOT** replace or override the PRIMARY result.
- The formal Experiment 4 inference conclusion remains based on the
  `PRIMARY L=h` rows.
- If `PRIMARY` and `SENSITIVITY` classifications differ for a horizon,
  retain the already-frozen statement (Section 6.2) that the inference is
  sensitive to the pre-specified block-length choice.

The natural-language reporting statements above in Section 7 are
unchanged.

---

## 8. Secondary HAC analysis

Include a secondary HAC / Newey-West-style mean-loss-differential analysis
on the **same** 391-date `d_date[t,h]` series. This is **SECONDARY**. It
does not override the MBB result.

Freeze HAC maximum lag:

```
q = horizon - 1
```

therefore:

- `5d`  -> `q = 4`
- `10d` -> `q = 9`
- `20d` -> `q = 19`

Use Bartlett weights. Implement the long-run variance explicitly:

```
gamma_l = (1/T) * sum_{t=l+1}^{T} (d_t - mean_d) * (d_{t-l} - mean_d)

LRV = gamma_0 + 2 * sum_{l=1}^{q} (1 - l/(q+1)) * gamma_l

SE(mean) = sqrt(LRV / T)

z = theta_hat / SE(mean)
```

Two-sided p-value, using Python's standard-library `math` module (no new
SciPy/statsmodels dependency):

```python
p_raw = math.erfc(abs(z) / math.sqrt(2.0))
```

This is the asymptotic standard Normal two-sided p-value.

If `LRV` is non-finite or `<= 0`: **FAIL CLOSED** for the HAC calculation.

### 8.1 HAC classification_raw (frozen)

Freeze the significance threshold `alpha = 0.05`.

`classification_raw` is based **only** on the raw HAC p-value and
`theta_hat`:

```
if p_value_raw < 0.05 and theta_hat > 0:
    classification_raw = "SECONDARY_HAC_STATIC_LOWER_LOSS"
elif p_value_raw < 0.05 and theta_hat < 0:
    classification_raw = "SECONDARY_HAC_ADAPTIVE_LOWER_LOSS"
else:
    classification_raw = "SECONDARY_HAC_NO_DISTINGUISHABLE_DIFFERENCE"
```

If `theta_hat == 0` exactly, use
`"SECONDARY_HAC_NO_DISTINGUISHABLE_DIFFERENCE"`.

Clarify:

- `p_value_holm` (Section 9.1) is supplementary multiplicity transparency
  only;
- do **NOT** create a second Holm-based HAC decision rule;
- HAC remains **SECONDARY** and never overrides the PRIMARY MBB
  conclusion (Section 7).

Call this a "HAC mean-loss-differential test" or "DM-style HAC secondary
diagnostic". Do **NOT** present it as a literal classical single-series
Diebold-Mariano setup without qualification, because the loss differential
has first been aggregated across the stock panel and five fixed seeds.

---

## 9. Multiple horizons

The three horizons are pre-specified separate estimands. Primary MBB
inference is reported horizon-by-horizon.

Do **NOT** make an omnibus statement such as "at least one horizon is
significant" without a separate pre-registered omnibus test.

For secondary HAC transparency, report:

- raw two-sided HAC p-value;
- Holm-adjusted HAC p-value across the three horizons.

The Holm adjustment is secondary only. Do not retroactively choose a
horizon based on significance.

### 9.1 Exact Holm algorithm (frozen)

Freeze the Holm adjustment across exactly the 3 horizons (used both for
the secondary HAC p-values here and for the supplementary primary-MBB
transparency table in Section 7.1):

1. Sort raw p-values ascending: `p_(1) <= p_(2) <= p_(3)`.
2. For ordered rank `j = 1, ..., m` with `m = 3`:
   `candidate_j = (m - j + 1) * p_(j)`.
3. Enforce monotonicity using the cumulative maximum over the ordered
   candidates (each adjusted value must be >= the previous one in rank
   order).
4. Cap adjusted values at `1.0`.
5. Map the adjusted p-values back to their original horizons.

---

## 10. Source subsetting, bridge audits, and pairability

### 10.1 Exact frozen source subsetting

Freeze the canonical extraction rules used to build the 30 model cells.

New Experiment 4 predictions:

```
mechanism in {"adaptive", "static"}
seed in {0, 1, 21, 3407}
horizon in {"5d", "10d", "20d"}
```

Frozen Adaptive seed42 (MAIN V2):

```
frequency_setting == "daily_weekly"
seed == 42
numeric horizon in {5, 10, 20}
```

mapped as:

```
5  -> "5d"
10 -> "10d"
20 -> "20d"
```

Frozen Static seed42 (Experiment 3):

```
mechanism == "static"
seed == 42
horizon in {"5d", "10d", "20d"}
```

After normalization, require exactly `2 mechanisms x 5 seeds x 3 horizons =
30 cells`.

### 10.2 Cell-level bridge audit

Before constructing the formal inference series, the later runner must
reconstruct the descriptive evidence from persisted predictions.

For all 30 cells:

1. Cast `y_true`/`y_pred` to `np.float32`.
2. Reconstruct pooled MSE from persisted predictions.
3. Compare against frozen `experiment4_combined_cell_metrics.csv` using
   `rtol=1e-8, atol=1e-8`.

Then reconstruct all 15 `delta_mse = Adaptive pooled MSE - Static pooled
MSE` values and compare against frozen `experiment4_seed_pair_deltas.csv`
using `rtol=1e-8, atol=1e-8`.

Only after these audits pass may the loss-differential time series
(Section 2 onward) be built. The existing final audit that `theta_hat`
reproduces the frozen five-seed mean `delta_mse` (Section 3) is retained
unchanged and applies after this bridge audit.

Any mismatch: **FAIL CLOSED**.

### 10.3 Pairability / fail-closed audits

Before inference, require for every horizon x seed:

- Adaptive predictions: 6647 rows.
- Static predictions: 6647 rows.

After canonical sorting by `ticker`, `anchor_date`, require identical keys.

Cast `y_true` to `np.float32`. Require `y_true` identity:

1. First: `np.array_equal`.
2. Otherwise:

   ```python
   np.allclose(
       ...,
       rtol=0.0,
       atol=np.finfo(np.float32).eps,
       equal_nan=False,
   )
   ```

Require: 17 tickers, 391 dates, 17 observations on every date.

Require the same 391 ordered anchor dates across all five seeds and both
mechanisms within a horizon.

No missing loss differential is permitted. Any violation: **FAIL CLOSED**.

---

## 11. Frozen input evidence / hash binding

The later inference runner must hash-verify all relevant frozen inputs. At
minimum, bind to:

- Experiment 4 new TEST predictions:
  `dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv`
  SHA256: `6b5f5ea94f1bb5f25ebd5ed09d7e4e7eb609945f97f953e7fb95116e2e866e8a`

- Frozen Adaptive seed42 predictions:
  `dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv`
  SHA256: `a05955a561da1b8880f1bb0878a6576852c087cc45707546d446ce96b0fabdaf`

- Frozen Static seed42 predictions:
  `dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv`
  SHA256: `95d1b06cc3ce0209e99833c29a1c3672a6d7a1a9d46d32f16ef1e48fa9846515`

- Frozen Experiment 4 combined cell metrics:
  `dataset/audit/experiment4_multiseed/experiment4_combined_cell_metrics.csv`
  SHA256: `89cfa52907d065cfc6f94ed662f17c07cb18a3a9c19d8ce961ec15a1959d5a41`

- Frozen Experiment 4 paired deltas:
  `dataset/audit/experiment4_multiseed/experiment4_seed_pair_deltas.csv`
  SHA256: `2f2a66689ac6abc8371565c2d8567c5ffa08259e5cb4879dd1fb306645c7f103`

- Frozen robustness summary:
  `dataset/audit/experiment4_multiseed/experiment4_robustness_summary.csv`
  SHA256: `9571b4424ad4261e8e973bcdc528718c26a6744e1d55e5d6323f69fb57d063a3`

- Experiment 4 descriptive finalizer:
  `scripts/python/panel_finalize_pathformer_experiment4_multiseed.py`
  SHA256: `d9ff7b456186848ada227fba1cc75a43508d65659f28ce523e024fe5744e7c39`

Do **NOT** modify any of these inputs.

### 11.1 Protocol hash binding

After this protocol is statically audited and SHA256-frozen, the later
inference runner **MUST** embed and verify the exact frozen SHA256 of:

`docs/experiment4_inference_protocol.md`

The exact hash is intentionally not present yet because this document has
not yet received its final SHA freeze. The runner must fail closed if the
protocol hash differs.

---

## 12. Inference namespace

The later inference implementation must use a **new, isolated namespace**:

```
dataset/audit/experiment4_inference/
```

It must **never** write into:

- `dataset/audit/experiment4_multiseed/`
- `dataset/audit/experiment3_pathformer_ablation/`
- `dataset/audit/main_v2/`

All existing evidence remains read-only.

---

## 13. Planned inference outputs

Freeze the intended output family now. The later inference runner should
write:

1. `experiment4_loss_diff_by_seed_date.csv`
   - Expected rows: `3 horizons x 5 seeds x 391 dates = 5865`
   - Fields: `horizon`, `seed`, `anchor_date`, `mean_loss_diff`, `n_tickers`

2. `experiment4_loss_diff_by_date.csv`
   - Expected rows: `3 x 391 = 1173`
   - Fields: `horizon`, `anchor_date`, `mean_loss_diff`, `n_seeds`

3. `experiment4_block_bootstrap_replicates.csv`
   - Expected rows: `3 horizons x 2 block specifications x 20000 replicates = 120000`
   - Fields: `horizon`, `block_length`, `specification`, `replicate`,
     `bootstrap_mean_loss_diff`
   - Do **NOT** save full bootstrap date-index matrices.

4. `experiment4_block_bootstrap_summary.csv`
   - Expected rows: `6` (3 horizons x `PRIMARY`/`SENSITIVITY`)
   - Exact fields: `horizon`, `block_length`, `specification`, `n_dates`,
     `n_bootstrap`, `theta_hat`, `bootstrap_mean`, `bootstrap_se`,
     `ci_lower_95`, `ci_upper_95`, `centered_p_value`,
     `centered_p_value_holm_primary`, `classification`.
     For `SENSITIVITY` rows, `centered_p_value_holm_primary` must be
     NaN / empty, because Holm-primary adjustment (Section 7.1) applies
     only to the three `PRIMARY` rows. `classification` semantics are
     frozen in Section 7.2.

5. `experiment4_hac_summary.csv`
   - Expected rows: `3`
   - Exact fields: `horizon`, `n_dates`, `max_lag`, `theta_hat`,
     `long_run_variance`, `se_mean`, `z_stat`, `p_value_raw`,
     `p_value_holm`, `classification_raw`.

6. `experiment4_inference_summary.txt`

7. `experiment4_inference_manifest.csv`

The later runner must fail closed if **any** final inference output
already exists. Use transactional / temporary-file writing so a partial
failure does not leave an incomplete inference namespace.

### 13.1 Deterministic output order / replicate index (frozen)

Canonical horizon order: `5d`, `10d`, `20d`. Do **not** use alphabetical
sorting for horizon, because that would place `10d`/`20d` before `5d`.

Canonical seed order: `0`, `1`, `21`, `42`, `3407`.

Canonical specification order: `PRIMARY`, `SENSITIVITY`.

Bootstrap replicate indexing: `0, 1, ..., 19999` — exactly 20,000
replicate IDs per horizon/specification.

Canonical `anchor_date` order: chronological ascending after datetime
parsing (Section 5.1).

Freeze final CSV row ordering:

- `experiment4_loss_diff_by_seed_date.csv`: horizon order, then seed
  order, then `anchor_date` ascending.
- `experiment4_loss_diff_by_date.csv`: horizon order, then `anchor_date`
  ascending.
- `experiment4_block_bootstrap_replicates.csv`: horizon order, then
  specification order (`PRIMARY` -> `SENSITIVITY`), then replicate
  ascending (`0` -> `19999`).
- `experiment4_block_bootstrap_summary.csv`: horizon order, then
  specification order (`PRIMARY` -> `SENSITIVITY`).
- `experiment4_hac_summary.csv`: horizon order.

---

## 14. Interpretation governance

- MBB is **PRIMARY**.
- HAC is **SECONDARY**.
- If MBB and HAC disagree: report the disagreement. Do not select the more
  favorable method.
- If PRIMARY and SENSITIVITY block lengths disagree: explicitly label the
  inference block-length sensitive.

Formal inference applies only to expected squared forecast loss. It does
**NOT** establish:

- profitability;
- economic utility;
- causality;
- universal architecture superiority;
- universal failure of Adaptive routing;
- router economic regime discovery.

Adaptive has greater parameter capacity than Static; retain this caveat.

The result is conditional on:

- the frozen Dataset V2;
- the frozen 17-stock panel;
- the frozen TEST window;
- the five pre-specified seeds;
- the frozen model/training contracts.

---

## 15. Router gate

Router / scale-weight analysis remains gated. Do **NOT** perform router
interpretation as part of this inference protocol or the later inference
runner. Only after the inference stage is finalized may router analysis
begin.

If Adaptive still lacks robust evidence of superiority after inference,
router analysis must be framed primarily as diagnostic / failure analysis,
not economic validation.

---

## 16. Workflow after this protocol

1. Create this protocol only.
2. Static-audit this protocol.
3. Freeze its SHA256.
4. Only then implement the inference runner.
5. Static-audit the runner.
6. User manually runs inference.
7. Independently audit inference evidence.
8. Finalize/freeze inference.
9. Only then move to router diagnostics.

---

## 17. Preservation notes

This protocol document does not modify, and must never be used to modify:

- `project_progress.md`
- `docs/experiment4_multiseed_protocol.md`
- `scripts/python/panel_pathformer_experiment4_multiseed.py`
- `scripts/python/panel_finalize_pathformer_experiment4_multiseed.py`
- Experiment 4 raw/final evidence
- Experiment 3 evidence
- MAIN V2 evidence
- Dataset V2 evidence

No inference result is calculated in this document. No command was
executed to produce this document.
