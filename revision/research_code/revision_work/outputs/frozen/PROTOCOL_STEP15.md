# Step15 fixed protocol and artifact map

The approved pool has60 backbones:43 from14a,2 Exp3 O Fixed seed42 h10/20,12 Exp4 O Adaptive seeds0/1/21/3407,3 EXACT Step3B O Adaptive seed42 recoveries. Every external source must replay formal TEST at max absolute prediction difference <1e-7. Any failed replay stops the process; no backbone is retrained.

One CPU numerical worker uses12 intra-op and12 inter-op threads; OMP_NUM_THREADS is absent and OMP_DYNAMIC/MKL_DYNAMIC are FALSE. The supervisor and caffeinate contain no numerical/training worker. No14c job is launched by this task.

Primary:60 backbones x4 variants=240 cells. S1 secondary:30 backbones x3 trained variants=90 further cells. Fixed(a) for each secondary group explicitly references the primary Fixed baseline, adding30 displayed baseline rows but no run. Total330 unique cells,270 trained aggregators and/or heads,990 unique prediction Parquets.

Static has8 trainable logits. Constant-input and Adaptive both have8520 parameters and paired initial router hashes. The secondary head adds4225 parameters and starts from the same backbone head. TRAIN batch generator42, model/router seed=backbone seed, Adam1e-4, batch32, Huber1, gradient clip1, max100, patience10, validation improvement1e-12, restored validation-best state. The unchanged formal train_one_cell implements the training loop.

Constant-input vectors use the TRAIN-only mean of each backbone expert cache: float64 reduction, then float32 storage. Neither validation nor test is used. All inputs/projections/experts are fixed, eval, and never called while fitting an aggregator. The primary head is frozen/eval; the secondary head is jointly fine-tuned. Expert cache shape is(N,8,32); cache_expert_order.csv fixes the expert axis. Sample rows match the corresponding backbone_{train,val,test}.parquet exactly.

Before seeing comparisons, the paired-bootstrap families are fixed: input x backbone x setting x contrast x metric, Holm over the3 horizons within each family. Contrasts are Adaptive-Static, Adaptive-Constant-input, Static-Fixed; metrics MSE, mean Rank IC, direction accuracy. Stock means precede paired-seed and shared-calendar-date resampling. B20000,L=h, non-circular overlapping moving blocks, seed20261019 with SeedSequence([base,h,h,1]). Exact draws and all contrasts are retained. MSE paired losses use float64 accumulation; formal per-seed pooled MSE retains its original float32 arithmetic.

Every file is created exclusively. Completed caches/cells resume only after artifact SHA256 verification; interrupted attempts remain intact and retries use new attempt directories. Status snapshots are immutable files under sessions/<id>/status/. Final report and every table are written only after completion.
