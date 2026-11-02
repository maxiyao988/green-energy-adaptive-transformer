# Step17 protocol amendment01 — explicit user clarification BEFORE TEST metrics

UTC: 2026-10-19T09:13:00.136809+00:00

This amendment overrides the multiplicity, CW evaluation and lambda0 test conventions of PROTOCOL_STEP17.md. PRETEST_GATE.json confirms that only source checks and validation selection have been performed. The saved validation curves/selected lambdas are unchanged.

- PRIMARY family: one two-sided loss-MBB p per ALL414 planned PRIMARY cells, one Holm family. SECONDARY: the same414-cell family for fixed lambda0.5. These are the ONLY Step17 multiplicity families. Do not split by model, input, horizon, window, seed/ensemble. Unavailable PRIMARY cells remain with a p=1 correction placeholder and raw pNA.
- CW is NOT recomputed and NOT re-corrected. For lambda>0 copy Step16's raw and Holm p for both NW and MBB and label "unchanged by shrinkage; corrected in Step16". At lambda0 show CW NA. Each copied reference retains its original Step16 family/source; it is not a Step17 rejection.
- Lambda0 equals Train-Mean. The loss-MBB test is undefined: raw pNA, inferential intervalNA, correction placeholder1; retain the cell and explicitly count lambda0 selections. Its deterministic squared-loss difference is0, not evidence of superiority.
- Copy W3 individual forecasts from clarified/B_forecast_skill_final.csv and existing C/neural ensemble forecasts from clarified/B_ensemble_skill.csv. For rolling individual cells copy C_rolling_CW.csv vsGlobal Train-Mean. For Static/Adaptive ensembles not covered by the ensemble table, copy the existing seed-pooled Step16 CW NW/date-only results from C_rolling_pooled_CW.csv: CW is linear in forecast deviation, so this is the same ensemble CW hypothesis, already evaluated/corrected in Step16. Copy the original three-horizon Holm fields unchanged, with this source/family explicit. No fresh CW calculation is used to establish the equivalence.
- No Step16 CW ensemble reference has been found for O W3 Single/Fixed ensembles. If not found in the designated Step16 outputs, retain CW fields as NOT FOUND; report this as a CW-reference BLOCKING gap rather than compute a new CW test or invent adjusted p. All available Step17 loss tests remain evaluable.
- Missing O Ridge W3 VAL remains a PRIMARY-only BLOCKING gap; fixed secondary remains evaluable. No refitting or source substitution.

All other pre-registered forecast/source/grid/tie/arithmetic/RNG/metric/loss-MBB/retention rules stay fixed. No TEST forecast-skill metric has been computed before this amendment. No new training, IC tests, CW tests or extra comparisons.
