# Step22C methods and read-only handoff

This is a code-only delivery: no Step22C bootstrap reconstruction, bound calculation, new model evaluation or training is executed during preparation. The user launches the registered script. All old artifacts are read-only; new artifacts are exclusively created under this directory.

## Sources and scope

1. End-to-end original-window Static−Adaptive, O and C, h5/10/20: Step16 A_paired_tests.csv, joint_seed_date rows only. These are the uniform-profile primary results used in the Step18 Static−Adaptive table (the Table9 comparison). Do not substitute the older Step5/12A distributions: their RNG streams and, for earlier C, numerical profiles differ.
2. Frozen primary: the Step22A **selected/tuned** head-frozen results, O/C × S1/S2 × h5/10/20, Adaptive−constant_gate and Static−Adaptive. No Step15 untuned or head-retrained distributions are substituted.
3. Training-setting robustness: Step22B T1/T2, C, h5/10/20, Static−Adaptive and Adaptive−LSTM. No T0 or optional Fixed comparisons are added.

There are42 contrast/context/horizon cells, each with MSE and Rank IC:84 metric rows. One wide table per numbered group contains6,24,12 rows, respectively. All cells are retained.

The three target source directories have no saved full replicate distributions for these comparisons. Step16 has RNG index archives and saved intervals; Step22A/B have the seed×date paired differences and saved intervals. Therefore reconstruct the **same** joint distributions only, using the registered RNG manifests and stored indices. The older original/frozen bootstrap archives belong to different analyses and are not substitutes. Store all reconstructed20000-replicate arrays in the new Step22C run directory. Do not compute new hypotheses, p-values, Holm adjustments or date-only analyses.

For Step16 only, reconstruct the original seed×date matrix from its canonical saved predictions using its existing ensemble-loss/Rank-IC helper, with the same date-major keys and sorted seed order. This is reproduction of an existing bootstrap input, not model inference. For Step22A/B read their stored paired_seed_date_differences.csv matrices without recomputing prediction metrics. Keep all391 dates, five seeds ordered0,1,21,42,3407 and paired-valid missing-IC handling.

Import the Step16 RNG/draw code and the Step15 missing-IC application AST. Before application, verify generated seed/date indices equal the saved Step16 indices, B=20000, L=h, base20261019 and SeedSequence([20261019,h,h,1]). The original contiguous float64 matrix layout is retained. The existing hypothesis-test and multiplicity functions are never called. Use only the joint application branch; do not compute unrequested date-only distributions.

## Intervals, scale and improvement direction

Every difference D is the named **first arm minus second arm**. Report estimate and95% percentile interval [q0.025(D*),q0.975(D*)] in raw MSE or Rank IC units. Verify the reconstructed point estimate and both interval endpoints against their stored values with absolute tolerance1e-12 before publishing any effect-size tables. Save all replay errors. A failed reconstruction stops publication and creates a new STOP_REPORT.

For **every MSE percentage, including the improvement bound**, divide by the fixed, stored five-seed mean TEST MSE of the **second arm**, and multiply by100. Do not bootstrap this denominator, use an ensemble's MSE, change to the first arm, or express Rank IC as a percentage. Thus Static−Adaptive uses Adaptive's mean MSE; Adaptive−constant_gate uses constant_gate's mean MSE; Adaptive−LSTM uses LSTM's mean MSE. The numeric denominator and both arm names accompany every row. These are percentage units relative to the specified second-arm MSE, not an unlabelled conventional reduction relative to some other baseline.

Adaptive's signed improvement I (positive is better for Adaptive):

| Contrast | MSE improvement I | Rank IC improvement I |
| --- | --- | --- |
| Static−Adaptive | D | −D |
| Adaptive−constant_gate | −D | D |
| Adaptive−LSTM | −D | D |

The **one-sided95% upper bound** is U=q0.95(I*), using linear percentile interpolation, not the97.5th percentile. For negative orientation this is −q0.05(D*), not −q0.95(D*). Store this signed upper bound unchanged; U<=0 means this pointwise bound allows no positive improvement. Do not silently clip negative bounds to zero. For MSE also report100U/second-arm-mean-MSE; for Rank IC report U in Rank IC units.

Bounds are **pointwise** by cell, without multiplicity adjustment or simultaneous coverage. Each group's one-line digest reports the maximum upper bound across all its contexts/horizons, separately for MSE percentage and Rank IC, and expressly states that this is a summary of pointwise bounds, not a simultaneous95% statement. No equivalence, practical-equivalence or absence-of-effect claim is made.

## Outputs after the user runs the script

- REPORT_STEP22C.md, with BLOCKING first, one table and one-line digest per numbered group, formulas, denominators and provenance.
- tables1_end_to_end.csv,2_frozen_tuned.csv,3_training_settings.csv; all_effect_bounds.csv (84 long-form rows).
- group_bound_digest.csv, bootstrap_interval_replay.csv, source_inventory.csv, provenance.json and source_checks.csv.
- bootstrap/<cell-id>.npy: every reconstructed20000-replicate joint distribution in named first-minus-second orientation.
- paired_inputs/<cell-id>.npz: seed×date input matrix, seed order and dates used for each reconstruction.
- COMPLETE_STEP22C.json, with the method/code/input/output fingerprints. It is written only after success.

Each execution writes into a new runs/run_<timestamp>/ directory and never overwrites partial attempts. A successful re-invocation verifies the completion record and returns without new calculations. There is no training entrypoint. The script uses one numerical thread, has a filesystem write guard, and prints progress. A STOP_REPORT is diagnostic, not permission to change a source or tolerance.
