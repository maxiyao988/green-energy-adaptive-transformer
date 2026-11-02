"""Assemble findings and final check ledger; never modifies any existing artifact."""
from review24_common import *
frozen=df(OUT/'FROZEN_OPTIMIZER_MEMBERSHIP.csv');struct=[]
for r in frozen[frozen.variant.ne('fixed')].itertuples():struct.append(dict(check='Literal optimizer contains only trainable aggregator parameters (source-derived)',passed=r.frozen_parameters_present_in_optimizer==0,backbone=r.backbone_id,variant=r.variant,frozen_parameters_in_argument=r.frozen_parameters_present_in_optimizer,classification='MINOR',finding='m2'))
for p,line in [('revision_work/scripts/analysis17/compute17.py',61),('revision_work/scripts/analysis17/report17.py',53)]:struct.append(dict(check='Located legacy draft parses',passed=False,path=p,line=line,classification='MINOR (unused, explicitly superseded)',finding='m3'))
write('ADDITIONAL_STRUCTURAL_CHECKS.csv',pd.DataFrame(struct))
findings=[
 dict(id='M1',classification='MAJOR',experiment='E10',title='GRS factor-covariance normalization differs from exact finite-sample normalization',location='revision_work/scripts/portfolio/engine.py:277-280',minimal_reproduction='priority2_portfolio.py: N=1 synthetic alpha GRS must equal ordinary OLS intercept t squared; production does not.',numerical_impact='Synthetic F discrepancy0.004909467330412143; Step16 maximum absolute p difference1.5371499807070865e-5; original Step8 O maximum4.980201452897681e-6; no .05 rejection changes in18 stored TEST GRS groups.',affected='Step8/16 GRS; manuscript4.5/Table12 notes; Supplementary S10/S11 and any GRS extraction',evidence='GRS_NORMALIZATION_IMPACT.csv; ORIGINAL_O_GRS_NORMALIZATION_IMPACT.csv; PRIORITY2_PORTFOLIO_CHECKS.csv'),
 dict(id='M2',classification='MAJOR',experiment='E4',title='Table4 Zero h20 last printed digit differs from direct rounding',location='revision_work/outputs/step24/MANUSCRIPT_TABLE_AUDIT_EXTRACT.txt:4 (original DOCX Table4 Zero row, h20 MSE)',minimal_reproduction="Mean(TEST y_true**2)=0.03670345225686257; format(100*MSE,'.4f')='3.6703', printed3.6704.",numerical_impact='One display unit0.0001 in the table scale, equivalent to1e-6 MSE; printed central value exceeds raw MSE by5.4774313743e-7. No ordering or inference changes.',affected='Manuscript Table4; numerical formatting only; raw Step16 labels/results agree.',evidence='LITERAL_TABLES_4_9_10_11.csv; INDEPENDENT_TABLE4_SUMMARY.csv'),
 dict(id='M3',classification='MAJOR',experiment='E4/E8',title='Table9 three O date-MBB Holm p values do not round directly to the printed values',location='revision_work/outputs/step24/MANUSCRIPT_TABLE_AUDIT_EXTRACT.txt:28-30 (original DOCX Table9 O date-MBB Holm column)',minimal_reproduction="Stored Holm p=3/20001=0.00014999250037498125; format(p,'.4f')='0.0001', printed0.0002.",numerical_impact='Three point-value display mismatches; underlying p and all significance decisions unchanged. Actual manuscript editing/rounding path not stored.',affected='Manuscript Table9 only; original Step5/Step16 computations remain consistent.',evidence='LITERAL_TABLES_4_9_10_11.csv; INDEPENDENT_TABLE9.csv'),
 dict(id='m1',classification='MINOR',experiment='E4',title='Rank IC has an undocumented absolute standard-deviation threshold',location='revision_work/scripts/skill/skill_common.py:204-206',minimal_reproduction='pred=np.arange(17)*1e-14, y=np.arange(17): production NaN; independent/SciPy Spearman1.',numerical_impact='Synthetic positive-rescaling invariance failure; no numerical mismatch across the738 current Step16 skill cells.',affected='RankIC helper domain/documentation; potential very-small-scale forecasts; no affected current table identified.',evidence='PRIORITY1_STATISTICS_DISCREPANCIES.csv; INDEPENDENT_STEP16_METRICS_738.csv'),
 dict(id='m2',classification='MINOR',experiment='E6',title='Frozen parameters remain in the formal Adam parameter argument',location='scripts/python/panel_pathformer_main_v2.py:1640; revision_work/outputs/frozen/scripts/frozen_worker_v3.py:121-137',minimal_reproduction='Formal Adam(model.parameters()) contains107777 frozen parameters plus8 Static or8520 gate parameters; requires_grad=False on frozen parameters.',numerical_impact='Literal optimizer-only-aggregator condition fails; None gradients are skipped by Adam, frozen fingerprints agree, eval-mode/cache checks pass. No numerical effect established.',affected='Step15/22A optimizer-membership wording; frozen primary experiments remain supported by tested invariants.',evidence='FROZEN_OPTIMIZER_MEMBERSHIP.csv; ADDITIONAL_STRUCTURAL_CHECKS.csv; PRIORITY1_FROZEN_CHECKS.csv'),
 dict(id='m3',classification='MINOR',experiment='E4/E12',title='Two retained unused Step17 drafts are syntactically invalid',location='revision_work/scripts/analysis17/compute17.py:61; revision_work/scripts/analysis17/report17.py:53',minimal_reproduction='ast.parse(path.read_text()) raises SyntaxError; valid *_v2.py files are the explicitly documented execution versions.',numerical_impact='No reported numerical effect; draft files are identified as unused in report17_v2.py:62; no execution of invalid versions assumed.',affected='Repository navigation/reproducibility clarity, not Step17 results.',evidence='CODE_MAP.csv; STEP17_PROVENANCE_ADDENDUM.csv')]
write('FINDINGS_STEP24.csv',pd.DataFrame(findings))
files=['PRIORITY1_STATISTICS_CHECKS.csv','PRIORITY1_DATA_CHECKS.csv','PRIORITY1_HEADLINE_CHECKS.csv','PRIORITY1_FROZEN_CHECKS.csv','TRAINING_HISTORY_CHECKS.csv','PRIORITY2_PORTFOLIO_CHECKS.csv','LITERAL_TABLES_4_9_10_11.csv','TABLE11_ADDITIONAL_NUMERIC_CHECKS.csv','REMAINING_CHECKS.csv','DIAGNOSTICS_MODELS_SETTINGS_CHECKS.csv','H_CACHE_ROUTER_DIAGNOSTIC_CHECKS_V2.csv','ENSEMBLE_REPLAY_CHECKS_V3.csv','ALPHA_HAC_CHECKS.csv','PORTFOLIO_CONSTRAINT_CHECKS.csv','ADDITIONAL_STRUCTURAL_CHECKS.csv']
ledger=[];summary=[]
for name in files:
 f=df(OUT/name)
 if name=='DIAGNOSTICS_MODELS_SETTINGS_CHECKS.csv':f=f[~f.check.str.startswith('H ')]
 passes=f.passed.astype(bool);unknown=(f['cell'].isin(['W3_O_ridge_h5','W3_O_ridge_h10','W3_O_ridge_h20'])&f.check.eq('Shrinkage VAL minimum and smaller-lambda tie rule')) if name=='REMAINING_CHECKS.csv' else pd.Series(False,index=f.index)
 for idx,row in f.iterrows():ledger.append(dict(evidence=name,evidence_row=int(idx)+2,check=str(row.get('check',f'Table{row.get("table")} row{row.get("row")} col{row.get("column")}')),status='NOT CHECKABLE' if unknown.loc[idx] else ('PASS' if passes.loc[idx] else 'FAIL'),details=row.to_json()))
 summary.append(dict(evidence=name,checks=len(f),passed=int(passes.sum()),failed=int((~passes&~unknown).sum()),not_checkable=int(unknown.sum())))
write('CHECKLIST_STEP24.csv',pd.DataFrame(ledger));write('CHECK_SUMMARY_STEP24.csv',pd.DataFrame(summary))
hashes=df(OUT/'SOURCE_IMMUTABILITY_CHECKS.csv');assert hashes.unchanged.all();assert not len(df(OUT/'SOURCE_HASH_CONFLICTS.csv'))
src=OUT/'SOURCE_HASH_CONFLICTS.csv';totals=pd.DataFrame(summary)[['checks','passed','failed','not_checkable']].sum().to_dict()
E=[
 ('E1','OK within checked snapshot/timing scope','836 data checks; raw calendar/labels, TRAIN-only scaling, future perturbation; known ex-post corporate actions are not recast as point-in-time.','PRIORITY1_DATA_CHECKS.csv; RECOMPUTED_SPLIT_BOUNDARIES.csv'),
 ('E2','OK','Formal Pathformer counts and patch sets; same-patch forward check; LSTM44673, Vanilla/SWiM142977; Ridge/signal/unconditional code inspected.','PRIORITY1_FROZEN_CHECKS.csv; DIAGNOSTICS_MODELS_SETTINGS_CHECKS.csv; CODE_MAP.csv'),
 ('E3','OK for recorded evidence','598 histories/terminal records; best epoch/patience/LR; actual Huber deltas; R10/R100 saved Adam groups; eval/restoration code.','TRAINING_HISTORY_CHECKS.csv; ADAM_GROUPS_REVIEWED.csv; FORMAL_TRAINING_LINE_EVIDENCE.csv'),
 ('E4','MAJOR M2/M3; MINOR m1/m3','738-cell independent skill replay and full Holm family agree. Four literal printed cells differ; tiny-scale IC guard is a domain limitation.','PRIORITY1_HEADLINE_CHECKS.csv; HOLM_738_FAMILY_AUDIT.csv; LITERAL_TABLES_4_9_10_11.csv'),
 ('E5','OK within sampled tensor checks','B3 completed26-week coverage, no unfinished-week bars, strict C endpoint reference, masked Daily127; multi-scale paths inventoried.','PRIORITY1_DATA_CHECKS.csv; FEATURE_PERTURBATION_EXAMPLES.csv'),
 ('E6','MINOR m2','Frozen/eval and constant-input checks pass; four-backbone fresh/cache samples and16 selected-aggregator replays pass; optimizer membership literal requirement differs.','PRIORITY1_FROZEN_CHECKS.csv; FROZEN_OPTIMIZER_MEMBERSHIP.csv; REMAINING_CHECKS.csv'),
 ('E7','OK with float32 precision disclosed','All30 TEST I0 cells reproduce formal forecasts; I1–I6 weight transformations, branch isolation, I4 groups and I5 renormalization verified.','REMAINING_CHECKS.csv; FLOAT32_INTERVENTION_PRECISION_DISCLOSURE.csv'),
 ('E8','OK; M3 is display only','Independent date/common-block/paired-seed checks; full20k saved draw replay;33 full/LOO ensembles and24 composition metric/spec replicate sets match actual Step7B source files.','PRIORITY1_STATISTICS_CHECKS.csv; ENSEMBLE_REPLAY_CHECKS_V3.csv; INDEPENDENT_TABLE9.csv'),
 ('E9','OK within checked scope','Raw W1/W2/W3 split reconstruction, disjoint TEST windows, Table10 replay and153 TRAIN volatility cutpoints pass; subgroup formulas inspected.','RECOMPUTED_SPLIT_BOUNDARIES.csv; PRIORITY1_HEADLINE_CHECKS.csv; REMAINING_CHECKS.csv'),
 ('E10','MAJOR M1','Return/cost/drift/initial turnover/next-close replay and risk-parity/constraints pass; exact finite-sample GRS normalization differs without observed rejection change.','PRIORITY2_PORTFOLIO_CHECKS.csv; ALPHA_HAC_CHECKS.csv; GRS_NORMALIZATION_IMPACT.csv'),
 ('E11','OK after matching inference dtype/source','Entropy/effective scales/TV,10 cross-seed pairs/modal agreement, sample/date Spearman and120 H cached-router diagnostics agree.','DIAGNOSTICS_MODELS_SETTINGS_CHECKS.csv; H_CACHE_ROUTER_DIAGNOSTIC_CHECKS_V2.csv'),
 ('E12','OK within checked robustness scope; inherited findings above','Immutable protocol hashes, effective T settings, TRAIN sigmas,36 learning-rate choices,84 bound replicate vectors, pooled NW/held-out CW and signed portfolios checked. No new training.','TRAINING_HISTORY_CHECKS.csv; TRAIN_ONLY_SIGMA_CHECKS.csv; REMAINING_CHECKS.csv; TABLE11_ADDITIONAL_NUMERIC_CHECKS.csv')]
write('EXPERIMENT_STATUS_STEP24.csv',pd.DataFrame(E,columns=['experiment','status','assessment','evidence']))
refs=[dict(title='statsmodels HAC covariance official documentation',url='https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html',use='Bartlett HAC reference; use_correction=False matched to recorded convention'),dict(title='SciPy Spearman official documentation',url='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html',use='Average-tie rank correlation reference'),dict(title='CRAN GRS.test source',url='https://raw.githubusercontent.com/cran/GRS.test/master/R/GRS.test.R',use='Shows the same unbiased-factor normalization variant as the repository; not treated as proof of exact N=1 identity'),dict(title='Testing and Ranking of Asset Pricing Models Using the GRS Statistic',url='https://doi.org/10.3390/jrfm17040168',use='Primary research discusses covariance-normalization ambiguity; publisher full-page access returned429. Main finding instead rests on reproduced N=1 OLS identity and explicit algebra.')]
write('EXTERNAL_REFERENCES.csv',pd.DataFrame(refs))
jwrite('RNG_AUDIT_MANIFEST.json',dict(review_seed=20261022,synthetic_block_equivalence_seed=721,table9_original_O_bootstrap_seed=20260826,table9_C_and_Step16_seed=20261019,ensemble_composition_seed=20261017,bootstrap_B=20000,draw_order='Copied from the stored protocol/manifests; independent block construction; saved draw matrices reused when present',new_training_runs=0,new_hypothesis_families=0,reference_module='independent24.py (imports no project modules)',source_file_count=len(hashes),source_hash_changes=0))
header='''# REPORT STEP24 — Independent correctness review

## BLOCKING

**No CRITICAL finding was identified in the checked scope.** Three MAJOR findings change numerical values or their printed representation without changing the checked conclusions; three MINOR findings have no demonstrated effect on current reported numbers. No fix, training run, checkpoint update, manuscript edit or replacement of a reported result was performed.

This is a code/data correctness review of the stored evidence, not a new experiment or a claim that every historical execution is fully recoverable. The MAJOR label follows the requested rule even for last-digit table discrepancies.

## Inventory and method

`CODE_MAP.csv` contains284 experiment-to-module rows,199 distinct paths, functions/classes with line ranges, SHA-256, linked reports and output locations. All E1–E12 have located code; no UNLOCATED entry. Registered source matches are distinguished from legacy/version candidates. `STEP17_PROVENANCE_ADDENDUM.csv` resolves the unused Step17 drafts against their stored manifest and the final v2 report.

The numerical reference is `independent24.py`, which imports no experiment code. Production pure functions were AST-isolated for comparison, and formal model classes were reused only for construction/inference checks. No backward pass or optimizer step ran. SciPy and statsmodels supply additional references. Missing review dependencies were installed only under `step24/deps`, without changing the repository virtual environment.

All retained review artifacts are under `step24` and were exclusively created. An initial empty `/tmp/step24_inventory_source.txt` was inadvertently created and immediately removed after checking that it was empty; no pre-existing content was affected. All3,071 inventoried/read source files passed the end-of-review SHA-256 check, with zero changes or conflicting first-read hashes (`SOURCE_IMMUTABILITY_CHECKS.csv`). This is a hash check of reviewed sources, not an assertion about every unexamined file on the disk.

## Per-experiment assessment

| Experiment | Status | Evidence and scope |
|---|---|---|
'''
body=header+'\n'.join('| '+e+' | '+status+' | '+assessment+' Evidence: '+ev+' |' for e,status,assessment,ev in E)+'\n\n'
body+='''## MAJOR findings

### M1 — GRS factor covariance normalization (E10)

Location: `revision_work/scripts/portfolio/engine.py:277–280`, function `grs`. Residual covariance is divided by T−2, with a prefactor that compensates for it, but the factor variance uses `ddof=1`. Relative to the exact finite-sample form with factor covariance divided by T, the extra factor-variance normalization remains.

An independent minimal check uses one test asset. The exact GRS F statistic must then equal the ordinary OLS alpha t statistic squared. In the saved synthetic example (seed20261022,T=60), the independent GRS and OLS t² agree within1e−10; production GRS differs by **0.004909467330412143**. This is a finite-sample normalization issue, not a floating-point rounding discrepancy. It is also a variant found in the [CRAN GRS.test source](https://raw.githubusercontent.com/cran/GRS.test/master/R/GRS.test.R); the code review does not assume a formula is exact merely because another package uses it.

For K=1, the reference computes S=E′E/T and V_f=Σ(f−f̄)²/T, then F=((T−N−1)/N)·α′S⁻¹α/(1+f̄²/V_f), with F(N,T−N−1) reference degrees of freedom. The N=1 OLS identity provides a direct independent reproducibility check.

Stored-result impact: all12 Step16 TEST GRS groups and all6 original Step8 O TEST groups were recomputed using the same assets and hypotheses. Maximum absolute p change is **1.5371499807070865e−5** for Step16 and **4.980201452897681e−6** for the Step8 O reference. **No p<.05 decision changes** in these18 groups. Tables/sections affected are portfolio GRS entries and their extractions (Step8/16, manuscript4.5/Table12 notes, Supplementary S10/S11); returns, Sharpe ratios and individual HAC alpha statistics are unaffected by this formula.

Minimal reproduction: the synthetic N=1 section of `priority2_portfolio.py`; independent formula in `independent24.py:grs`. Full numerical impact is in `GRS_NORMALIZATION_IMPACT.csv` and `ORIGINAL_O_GRS_NORMALIZATION_IMPACT.csv`. No pipeline correction was made.

### M2 — Table4 Zero h=20 printed last digit (E4)

Location: original manuscript DOCX Table4, Zero Predictor row,20-day MSE; line-addressable read-only extract `MANUSCRIPT_TABLE_AUDIT_EXTRACT.txt:4`. Independently averaging the stored float32 TEST labels in float64 gives MSE **0.03670345225686257**. Therefore `format(100*MSE,'.4f')` is **3.6703**; the manuscript prints **3.6704**.

Impact: one last digit, equivalent to1e−6 in displayed unscaled MSE units; the printed central value exceeds the raw value by approximately5.47743e−7. No model ordering or statistical conclusion changes. Raw labels and model metric files agree; this is a manuscript-number issue. No responsible automated DOCX formatting path was stored, so its editing cause is unverified. Evidence: `LITERAL_TABLES_4_9_10_11.csv`, Table4/row3/column3; reproduction in `literal_tables24_v3.py`.

### M3 — Table9 O date-MBB Holm p display (E4/E8)

Location: original manuscript DOCX Table9, O h=5/10/20 date-level Holm p column; extract `MANUSCRIPT_TABLE_AUDIT_EXTRACT.txt:28–30`. The stored and independently replayed p values are **0.00014999250037498125** at all three horizons. Direct four-decimal formatting gives **0.0001**, whereas all three printed cells show **0.0002**. Equivalently, `format(3/20001,'.4f') == '0.0001'`.

Impact: three printed point-value mismatches of0.0001 at that display precision. The underlying bootstrap p values, joint-bootstrap results, intervals and significance decisions agree. Intermediate rounding could produce this display, but the actual document editing history is not stored and is not inferred. Evidence/reproduction: `INDEPENDENT_TABLE9.csv`, `LITERAL_TABLES_4_9_10_11.csv`, `literal_tables24_v3.py`. No source or manuscript was edited.

## MINOR findings

### m1 — Absolute scale guard in Spearman helper (E4)

`revision_work/scripts/skill/skill_common.py:204–206` requires forecast and realized-return standard deviations to be at least1e−12, in addition to nonconstant ranks. Thus a positively rescaled but nonconstant forecast can become undefined: `pred=np.arange(17)*1e-14`, `y=np.arange(17)` yields production NaN versus independent/SciPy Spearman1. The helper docstring says only constant inputs are undefined. This is an undocumented domain restriction, not ordinary Spearman scale invariance.

No current738-cell skill value differed in the independent replay. It is classified MINOR because no numerical effect on current reported cells was found. The synthetic failure remains explicitly recorded in `PRIORITY1_STATISTICS_DISCREPANCIES.csv`; NaN-support mismatch must not be read as a zero numerical error merely because a finite maximum cannot be formed.

### m2 — Optimizer argument includes frozen parameters (E6)

`panel_pathformer_main_v2.py:1640` constructs `Adam(model.parameters(),...)`. The Step15/22A binding freezes parameters at `frozen_worker_v3.py:121–137` but reuses that constructor. Consequently the formal argument includes107,777 frozen parameters alongside8 Static logits or8,520 gate parameters. It does not literally contain only aggregator parameters, as requested by the audit checklist.

Frozen parameters have `requires_grad=False`, Adam skips parameters with no gradient, stored frozen fingerprints agree, and tested frozen modules stay in eval mode. The102 frozen checks pass. No update or numerical effect on a frozen parameter was found. `FROZEN_OPTIMIZER_MEMBERSHIP.csv` is source/construction-derived membership evidence; unsaved historical optimizer state is not claimed to have been inspected. `ADDITIONAL_STRUCTURAL_CHECKS.csv` records the literal condition as FAIL for12 sampled trained variants. Fixed is an untrained reference and is excluded from those membership failures.

### m3 — Retained unused invalid Step17 drafts (E4/E12)

`revision_work/scripts/analysis17/compute17.py:61` and `report17.py:53` fail AST parsing. The valid `compute17_v2.py` and `report17_v2.py` are explicitly identified as the executed versions in `report17_v2.py:62`; the invalid drafts are explicitly called unused. Hashes match the Step17 manifest. No result impact was found. Evidence: `CODE_MAP.csv`, `STEP17_PROVENANCE_ADDENDUM.csv`; minimal reproduction is read-only `ast.parse(path.read_text())`. These files were not repaired.

## Check results

The final ledger distinguishes failed assertions, known missing inputs and corrected review-harness assumptions. Each row points to the evidence CSV and its row number. Counts are assertions, not independent experiments or new hypothesis tests.

'''
body+=f"Final numerical/source-structure ledger: **{totals['checks']:,} checks; {totals['passed']:,} PASS; {totals['failed']:,} FAIL; {totals['not_checkable']} NOT CHECKABLE**. The3071 source-hash checks are additional and all pass.\n\n"
body+='| Evidence CSV | Checks | PASS | FAIL | Not checkable |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {r['evidence']} | {r['checks']} | {r['passed']} | {r['failed']} | {r['not_checkable']} |" for r in summary)+'\n\n'
body+='''The FAIL rows resolve to the six findings above: GRS normalization, four printed table cells, the synthetic IC guard,12 literal optimizer-membership conditions and2 unused syntax-invalid drafts. Three missing-validation Ridge selections are NOT CHECKABLE, not failed model selections.

### Priority1 detail

- **Timing, labels and splits:** raw intersection2603 eligible anchors; retained TRAIN counts980/1391/1802 for W1/W2/W3, VAL370 and TEST391 each;20-anchor purges and disjoint TEST ranges. Pooled TRAIN rows16660/23647/30634, VAL6290, TEST6647. Raw h-day log labels, target dates and split boundary checks agree. Per ticker/resolution/channel TRAIN pooled-entry standardization uses ddof0 and matches. Sampling covers27 ticker/window/partition anchors (AES,CWEN,FCEL), both O/C branches, with after-anchor price and volume perturbations. Features at those anchors stay unchanged. These are sample perturbation checks plus full-array/calendar/label/statistic checks, not exhaustive fresh feature reconstruction at every anchor.
- **Statistics:** ties/constant cases, NW lags0/4/5/9/10/19/20 against statsmodels, CW algebra and one-sided p, Holm, MBB overlapping noncircular blocks, common blocks/paired seeds, pooled variance weights squared, saved Step15 replicates and pooled Step22D NW quantities. The738 finite-cell Holm families match without dropping undefined members; all738 relevant raw p values are finite. Pooled R² is computed from sums of squared errors against each window's own reference.
- **Headline tables:** all numeric cells of Tables4/9/10/11 were compared. The291 literal cells yield287 matches and4 last-digit mismatches (M2/M3). Independent full-precision source checks include6198 assertions for738 skill cells, Table4 seed summaries, Table10 rolling contrasts/Holm and Table11 ensemble R²;45 additional Table11 CW/IC checks pass. Table9 O uses its recorded Step5 RNG20260826, C uses Step16 RNG20261019, rather than substituting a new random stream.
- **Frozen controls:** four representative backbones OS1h5,CS1h20,OS2h20,CS2h5 at seed0;32 TRAIN/VAL/TEST examples per backbone were forwarded freshly through all experts and compared with stored caches. Sixteen selected variant checkpoint replays agree. Frozen module eval/gradient flags, sample-invariant constant gates and parameter counts agree. Literal optimizer membership differs as m2. Full fresh recaching of all60 backbones was not attempted.
- **Training records:**598 stored histories/terminal records were read. Recorded chosen epochs achieve the validation training-loss minimum within the archived1e−12 improvement rule; patience10/max-epoch handling agrees. Actual per-epoch Huber delta/LR checks, seven immutable protocol hashes, TRAIN-only rolling/raw and cross-sectional SDs, and every saved R10/R100 Adam start/end group agree. Restoration/eval and loader seed42 versus benchmark global-RNG paths were inspected. No training reproducibility rerun was performed.

### Priority2 and3 detail

- Matched coverage uses Daily maximum127 tokens, valid122–127 sessions, the26 completed weeks, no current unfinished-week bars and the authorized B3-C completed-week endpoint reference. Formal Pathformer counts30401/107777/107785/116297 and same-patch107777 agree. Neural model constructors give44673/142977/142977.
- All30 I0 TEST prediction cells match the formal Adaptive forecasts at the prescribed1e−7 replay criterion, with identical keys. I4 uses contemporaneous date-group weights, I6 changes only one branch, I5 zeros one scale and renormalizes the others. Synthetic float32 means/simplex reductions differ from alternative reductions by1.4901161e−8 to1.1920929e−7; **these exceed1e−10 and are explicitly listed** in `FLOAT32_INTERVENTION_PRECISION_DISCLOSURE.csv`. They are expected float32 arithmetic effects, not evidence of incorrect intervention semantics. Exact one-hot weight removal is not separately stress-tested for a zero renormalization denominator; the inspected/current weights are not that case.
- All153 volatility cutpoint rows are reconstructed from trailing60-session simple-return SD(ddof1), evaluated only on the relevant TRAIN anchors. By-stock/tercile/subperiod filtering and minimum-five-stocks IC rules were inspected; every subgroup summary cell was not independently recomputed.
- Step16 portfolio daily gross/net10bp returns, drifted L1 turnover, initial build, annual metrics and CE were replayed; Step22G's24 signed-book/long-only portfolios agree. Covariances use252 daily returns through decision close, multiplied by h, with future perturbations unchanged; sampled stored risk-parity weights equalize risk contributions. All long-only sums/bounds and capped MV/minvar weights pass. Solver fallback logs are counted and not suppressed. Global QP optimality is not independently proven for every allocation.
- Execution is decision close t → execution close t+1 → ending close t+1+h. It intentionally differs from the forecast label interval t→t+h and is documented by the original engine. `PORTFOLIO_REALIZED_DATE_RANGES.csv` separates forecast and realized holding dates; the latter can end after the final forecast anchor. No claim of same-holding-period label/portfolio equality is made.
- Individual alpha OLS/HAC formulas match statsmodels at lag5 and the other checked lags, without finite-sample covariance correction. All273 original-O TEST/10bp alpha rows replay within the10-significant-digit source precision; maximum |Δt|=1.0338283563e−9, explicitly above1e−10, with no |t|>1.96 decision changes. These quantization differences are retained in `ORIGINAL_O_ALPHA_REFERENCE.csv`, not silently accepted as exact.
- DA was independently recomputed for all606 Step21 rows, including S1 head-retrained sensitivity, using sign equality and excluding exact-zero realized returns. Counts and values agree. Earlier historical DA definitions were not silently substituted for Step21.
- Entropy, effective scales, sample/within-ticker/within-date TV, all10 cross-seed pairwise distances, modal dominant agreement, and sample/date Spearman match stored O diagnostic inputs. All120 H cached-router SD/between-date-share checks pass using the formal inference helper's float64 weight output convention. Float32 alternatives are disclosed, not presented as formula errors.
- Step7B full/leave-one-out ensemble metrics and all24 composition-bootstrap replicate sets were replayed from the actual files in `run_info.json`, with seed multisets changing the ensemble forecast before evaluating loss/IC. All20,000 values per set and the reported intervals/p agree at1e−10. This is separate from averaging seed-level losses in the paired-model bootstrap.
- All36 Step22A learning-rate choices minimize seed0 validation MSE over the three rates. Shrinkage selection, full primary/secondary Holm families and conservative lambda0 placeholders agree wherever VAL inputs exist. All84 Step22C replicate vectors reproduce the stored percentile intervals, fixed second-arm MSE denominators and signed95th-percentile improvement bounds, retaining negative bounds. Window-specific pooled NW and held-out CW are independently checked; saved-draw/bootstrap reconstruction checks retain the pre-specified within-window stratification.

## Review-harness reconciliation

Initial review drafts and their outputs are retained, not overwritten. They are not additional experiment failures:

1. The inventory initially encountered unused invalid Step17 drafts; `inventory24_v2.py` records them explicitly instead of aborting.
2. The first statistics review lacked statsmodels. The dependency was installed under step24 only; `priority1_statistics_v2.py` is the completed run. `training_records24.py` contained a review-only syntax typo; its v2 is the completed record audit.
3. Early literal-table scripts selected duplicate B3/core rows or assumed unconditional references belonged to the738 registry. `literal_tables24_v3.py` uses the correct core groups and stored labels. No project code was changed.
4. Three Step17 O/W3 Ridge selections have `missing_validation` status. The first review check treated NaN selector values as a comparison failure; `MISSING_SOURCE_CHECK_RESOLUTION.csv` identifies the existing `BLOCKING_inputs.csv` evidence and preserves them as NOT CHECKABLE.
5. Initial H comparisons used full-model reference weights or float32 reductions. The production helper returns float64 weights. `router_cache_diagnostic24_v2.py` reuses the actual cached expert tensors, original aggregator class,32-example batches and12/12 inference; all120 checks match within1e−10. Earlier discrepancy CSVs are retained but superseded for these checks.
6. Initial ensemble references used Step16 canonical copies or a rounded stored point estimate. Tiny source quantization matters for exact rank ties/discrete bootstrap thresholds. `ensemble_replay24_v3.py` loads the actual original Step7B manifest sources and computes its unrounded point estimate; all105 final assertions pass. Earlier differences are not attributed to the production bootstrap.

The final reconciled ledger is `CHECKLIST_STEP24.csv`; superseded raw outputs remain available for transparency.

## What remains unchecked or unverifiable

- True point-in-time vendor files, historical security identity and delisted-security coverage cannot be established from the available frozen snapshot. Ex-post-adjusted O levels and snapshot/vender-revision limitations remain as previously documented. This review makes no point-in-time or blanket leakage-free claim.
- Step17 O/W3 Ridge has no saved VAL forecast/checkpoint for three horizons. Its validation selection cannot be reconstructed under the no-training rule; the existing missing-input disclosure is preserved.
- Historical CPU/device/runtime behavior is supported only where stored telemetry/source hashes exist. An unsaved historical optimizer state, an unlogged environment setting, or absence of all outside TEST access cannot be proved retrospectively. Frozen optimizer membership was derived from the executed constructor/binding; actual stored group records were available and inspected for Step22F R10/R100.
- Best-state restoration was checked in code, histories and sampled checkpoint replay, not by retraining all cells. Fresh expert/cache forward checks cover four backbones, not all60. Reported frozen/head-retrained DA rows were checked; no fresh optimizer step was used to re-prove head-retrained training behavior.
- Portfolio allocations were checked for arithmetic, constraints, recorded fallback behavior and sampled covariance/risk-contribution consistency. Every SLSQP solution was not independently re-optimized. GRS normalization impact is for18 stored TEST groups; no separate new hypothesis family was introduced.
- Full article/response prose, all bounds/ranges, workbook formatting and every supplementary cell were not re-audited here. The existing Step23b strict-claim audit remains a separate, unresolved-to-document-edit source; this report does not imply those earlier proposed wording corrections were applied. The current literal check covers all numeric cells in Tables4/9/10/11.
- No general proof covers arbitrary future extreme inputs, floating-point underflow, exact one-hot I5 removal, or every platform/version. Precision exceptions and the synthetic Spearman domain failure are expressly recorded.

## Deliverables and references

Main files: `REPORT_STEP24.md`, `CODE_MAP.csv`, `EXPERIMENT_STATUS_STEP24.csv`, `FINDINGS_STEP24.csv`, `CHECKLIST_STEP24.csv`, `CHECK_SUMMARY_STEP24.csv`, `SOURCE_IMMUTABILITY_CHECKS.csv`, `RNG_AUDIT_MANIFEST.json`. All independent scripts, intermediate review drafts, numerical evidence and dependency versions remain in this directory. `ARTIFACT_INDEX_STEP24.csv` supplies paths, sizes and SHA-256 values of final retained artifacts (excluding itself).

References are linked in `EXTERNAL_REFERENCES.csv`. HAC checks use the [official statsmodels HAC implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html); rank checks use [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html). GRS normalization ambiguities are also discussed in [primary research](https://doi.org/10.3390/jrfm17040168); the exact finding here rests on the independently reproduced finite-sample algebra and N=1 OLS identity.
'''
with (OUT/'REPORT_STEP24.md').open('x') as h:h.write(body)
index=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file():index.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='isolated audit dependency' if '/deps/' in str(p) else 'review artifact'))
write('ARTIFACT_INDEX_STEP24.csv',pd.DataFrame(index));print('FINAL totals',totals,'source hashes',len(hashes),'MAJOR3 MINOR3 CRITICAL0');print('Report',OUT/'REPORT_STEP24.md')
