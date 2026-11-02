Prompt 22F — End-to-end Static with trainable scale logits (pre-registered)

Repo: <project_root>. Same rules as Step22E: existing files (including all earlier step outputs) are read-only; all new outputs use exclusive creation under revision_work/outputs/step22f/; CPU only, 12/12 threads, dynamic threads off, 3 parallel workers; seeds {0, 1, 21, 42, 3407}. Reuse the existing Step16/Step22B training, evaluation and bootstrap code (import, do not rewrite); record the SHA-256 of every reused module.

STEP 0 — Protocol
Write PROTOCOL_STEP22F.md containing everything below, record its SHA-256 in RNG_MANIFEST.json, and do not change it afterwards. If anything cannot be implemented as written, STOP and ask me.
Include this provenance statement verbatim: "The scientific design of this protocol (hypothesis, variants, contrasts and families) was fixed on 2026-10-21 before Step22E was run; this version adds only operational rules. No Static-R10, Static-R100 or Adaptive-R100 runs existed when it was written."

1. Hypothesis (pre-specified)
End-to-end, the Static scale weights barely move from equal weighting (mean |softmax(logit) − 0.25| ≈ 0.0003–0.0007 in Step22B/22E), so Static ≈ Fixed. In the frozen-representation comparisons, learned global weights improved slightly on equal weighting in all seeds. H: giving the scale logits a larger learning rate lets Static improve on Fixed end-to-end.

2. Variants
Everything else = the Step16 T0 setting (base learning rate 1e-4, Huber δ = 1); C inputs; Daily+Weekly; W3; h = 5, 10, 20; five seeds.
- Static-R10: Static, with the scale-logit learning rate = 10 × base (1e-3), in both branches; all other parameters at the base rate.
- Static-R100: scale-logit learning rate = 100 × base (1e-2).
- Adaptive-R100: Adaptive, with the router (all router parameters, both branches) learning rate = 100 × base (1e-2); all other parameters at the base rate (fairness control).
Implement via Adam parameter groups only. The new groups keep the base group's Adam betas, epsilon and weight decay; gradient clipping, scheduler, early stopping and VAL-based checkpoint selection are unchanged. No other change. 3 variants × 3 horizons × 5 seeds = 45 runs.
Reference arms (reuse, do not retrain): Fixed, Static and Adaptive, C inputs, T0, W3, five seeds, from Step16/Step22B. If a complete five-seed C-input T0 set for any of these three on W3 does not exist, STOP and ask me.

3. Gates and run policy
- Data gate (before training): verify the W3 partitions, keys and labels by opaque SHA-256 of the stored bytes against the hashes recorded in earlier steps, and confirm that the TRAIN/VAL tensors match those of the reference arms. No TEST deserialization. The TEST lock must explicitly permit this byte-level hashing; everything else that touches TEST stays blocked.
- Timing gate: after the first batch of 3 runs (one per worker) completes, projected total = batch elapsed time × 15. If > 9 h, STOP and report.
- Interruption policy: runs with a terminal COMPLETE record are kept; any run without one is restarted from scratch with the identical configuration and seed; every attempt is logged. If the same run fails twice, STOP and report. No TEST data are read during any restart.
- Launch the long-running job under `caffeinate -dimsu`, and record this in the run log.
- TEST is read only after all 45 runs have terminal records.

4. Evaluation
- Descriptive: five-seed mean ± SD test MSE, Rank IC, DA (excluding exactly-zero realized returns) and out-of-sample R² vs the Global Train-Mean, plus five-seed ensemble metrics, for the three new variants next to Fixed, Static and Adaptive (T0). Best epochs per run.
- Manipulation check (descriptive): for Static-R10 and Static-R100, the learned weights per branch and horizon, the centered-logit MAD and the weight MAD (mean |softmax − 0.25|), and the within-model SD; for Adaptive-R100, the router-weight deviation from 0.25 and its within-model SD; all next to the T0 Static/Adaptive values.
- Inference: the joint seed×date bootstrap with the Step16 settings (B = 20,000, L = h; seeds paired by seed ID), two-sided. NW is not used in this step. Contrasts (first minus second):
  PRIMARY: Static-R10 − Fixed and Static-R100 − Fixed, for MSE and Rank IC;
  SECONDARY: Static-R100 − Adaptive-R100, Adaptive-R100 − Adaptive, Static-R100 − Static, for MSE and Rank IC.
  Holm across the 3 horizons within each contrast × metric. Report seed wins (k/5) for every row.
- Step22C-style effect bounds for the PRIMARY contrasts: the 95% percentile interval, MSE expressed as % of the second arm's five-seed mean, and the one-sided 95% upper bound on the improvement of the first arm.
- Direction-specific labels (p-values unchanged): MSE difference < 0 or Rank IC difference > 0 with Holm p < .05 → "significant improvement of the first arm"; the opposite sign with Holm p < .05 → "significant deterioration of the first arm"; otherwise "not significant".

5. Interpretation rules (fixed before TEST is read)
- If the Static weight MAD under R10 and R100 is below 0.005 in both branches (i.e., less than about 10 × the T0 level), report that the manipulation did not move the weights and draw no conclusion about H.
- A PRIMARY "significant improvement" → "EXPLORATORY: with a larger logit learning rate, learned global scale weights improved on equal weighting end-to-end (W3 only, no out-of-window confirmation)".
- No PRIMARY improvement (and the manipulation check passed) → "learned global weights did not improve on equal weighting end-to-end even when the weights were allowed to move".
- Any positive finding is labelled EXPLORATORY in every table and in the report.

6. Reporting
Report every run and every cell, including undefined ones (kept in their Holm family with p = 1). Do not select, drop or re-specify anything after TEST is read. W3 T0 remains the primary evaluation in the manuscript; these results are supplementary.
Write REPORT_STEP22F_FINAL.md: BLOCKING section first; protocol hash; gates, timing and any restarts; descriptive tables; manipulation check; PRIMARY and SECONDARY contrasts with raw p, Holm p, labels, seed wins and bounds; every training attempt (status, epochs, best epoch, elapsed time); the file list. Then run an independent validation pass and write REPORT_STEP22F_VALIDATION.md, as in Step22E.

Operational addendum fixed before any Step22F training or TEST parsing
===================================================================
The scientific-design provenance statement above is the user's supplied provenance. This operational document is physically written after Step22E completed; it does not imply that a file/hash existed before Step22E. All Step22F training remains unstarted. The user will launch the reviewed code in Terminal. No research TEST forecasts or label contents are parsed during code preparation.

User clarification — manipulation gate and interpretation (binding)
- Evaluate each Static variant (R10 and R100) and horizon separately, using the five-seed mean weight MAD per branch.
- A variant × horizon FAILS if BOTH Daily and Weekly five-seed mean MADs are <0.005; PASSES if at least one branch is ≥0.005. If either branch's required five-seed statistic is undefined, the gate is UNDEFINED, not a pass.
- Failed variant × horizon PRIMARY rows are labelled "manipulation failed — no conclusion about H". All raw/adjusted p-values, effect bounds, direction labels and family membership remain reported and unchanged.
- H is judged only from passing variant × horizon cells. If all six fail, there is no overall conclusion about H. Undefined gates also supply no conclusion. Report every seed's MAD and the number of seeds with MAD≥0.005 per variant×horizon×branch.
- The significance label describes the test direction; a separate interpretation column applies manipulation status. Every significant improvement is labelled EXPLORATORY in every table and report. No positive result becomes confirmatory.

Trainer and optimizer binding
- Reuse Step22B worker22b_v3.py train(), evaluate(), dataset(), inference(), prefix(), factories and telemetry by import. Static seed42 uses Experiment3; Adaptive seed42 uses main_v2; other seeds use Experiment4. Model initialization and loader conventions, generator42, C tensors, CPU, dropout0.1, batch32, clip L2 norm1, Huber delta1, base Adam1e-4, max100, patience10 and improvement1e-12 stay unchanged.
- Only replace Adam construction with two exhaustive disjoint parameter groups. In original named-parameter order, base group contains all remaining trainable parameters; accelerated group contains exactly the two 4-element .aggregator.scale_logits vectors for Static (8 scalars) or all parameters under .aggregator.router. in both branches for Adaptive (8,520 scalars). Both base groups have107,777 parameters; total counts Static107,785 and Adaptive116,297. No freezing or reinitialization. Record parameter names/IDs/counts and all actual Adam defaults/group settings before and after training.
- Group LR is1e-3 for Static-R10;1e-2 for Static-R100/Adaptive-R100; remaining parameters1e-4. All group betas(.9,.999), eps1e-8, weight_decay0, amsgradFalse and remaining defaults equal the base optimizer's. No scheduler is added. Preserve a single global gradient clip over the entire model.
- The code must demonstrate exhaustive group membership and equal defaults with synthetic checks and zero optimizer steps before training. This preparation is not an extra scientific run.

Data and TEST access
- Verify all8 frozen W3 causal tensor/index/target files against previously stored dependency hashes: sample_index.csv,3 frozen y arrays,2 causal raw tensors and2 causal standardized tensors. Verify reference identities against Step16 registry/checks and original COMPLETE/dependency records; require45 C W3 T0 reference cells (Fixed/Static/Adaptive, all5 seeds, all3h), all12/12.
- Opaque streaming SHA is permitted for all frozen TEST-bearing files. TRAIN/VAL-only memory mapping uses the unchanged Step22B selector, with bitwise re-normalization checks. Compare TRAIN/VAL keys/float32 labels and standardized inputs to the reference binding without parsing TEST. No research TEST predictions, label values, metrics or routing diagnostics are read before all45 successful terminal model records.
- First batch: Static-R10/h20/seed42, Static-R100/h20/seed0, Adaptive-R100/h20/seed42; keep these3 in the45. Timing includes retry elapsed; projection=first-batch UTC elapsed×15; stop if>9h. Save monotonic time too.
- Remaining order: seeds0,1,21,42,3407;h20,10,5;interleave the3 variants. Three independent worker processes refill as slots free, eachCPU intra12/inter12, OMP unset and dynamic FALSE before Torch import. Queue uses caffeinate -dimsu, waits for AC, logs invocation/PID, heartbeat60s, thread/env/load/top-CPU telemetry at start/every5min/end.
- Resume from verified successful COMPLETE only; never retrain successful cells. Preserve incomplete attempt, mark interrupted and restart identical config/seed from scratch; one retry allowed, second failure/interruption stops. Wait for still-live orphan workers before replacement; one controller OS lock. No training after ALL_TRAINING_FINISHED.json. A verified model COMPLETE is retained if the controller crashed before optional power/adapter end metadata; document missing sidecars, do not invent them.
- After training barrier: infer TRAIN/VAL already saved and TEST with the original eval-mode batched helper; strict checkpoint load. Reuse stored reference TEST forecasts, never retrain/re-infer references. Save complete per-stock/date/seed forecasts, squared/Huber losses, routing weights, logits, histories, checkpoint and source/profile metadata. Counts30634/6290/6647, exact keys/float32 labels. Degenerate/undefined cells retained; no seed or date selection after TEST.

Metric/bootstrap definitions, no NW tests
- Use Step22B metrics and Step16 statistics16_v3 imported joint distribution, centered two-sided p, percentile interval and Holm code. DA exactly excludes y_true==0; no sign-zero substitution. Five-seed means±sample SD(ddof1); equal-weight forecast ensemble also reported descriptively. R² uses the fixed float64 mean of W3 TRAIN labels as Global Train-Mean, following Steps22B/22E.
- B20000,L=h,noncircular blocks on391 TEST dates; seed ID pairing. Use the existing Step16 stream: BASE20261019, SeedSequence([20261019,h,h,1]), five paired seed draws first then date blocks, verbatim imported code. Bootstrap matrix seed order42,0,1,21,3407 (record this separately from queue order). Common draws across contrasts/models/metrics at each h. Save seed/date indices, weights, input matrices and all20,000 replicates. Reuse the Step16 missing-IC branch; preserve the391-date calendar and report missing paired dates per seed. A seed with no paired-defined dates or nonfinite distribution makes the cell undefined, rawp=N/A,Holm placeholder1. Never drop a family member.
- Five contrasts×2 metrics×3h=30 test rows. PRIMARY=Static-R10−Fixed and Static-R100−Fixed. SECONDARY=Static-R100−Adaptive-R100,Adaptive-R100−Adaptive,Static-R100−Static. Holm separately across3h for each contrast×metric (10 families). No correction across PRIMARY/SECONDARY contrasts, no NW or new tests.
- For first-minus-second D, improvement of first arm is -D for MSE and D for IC. PRIMARY bound=q.95 of signed improvement, retaining negative values; percentile95% difference CI; MSE difference/CI/bound scaled by100 divided by Fixed's fixed five-seed mean TEST MSE, not a bootstrap denominator. Pointwise bounds, not simultaneous. SECONDARY bounds not requested and not added.
- Seed wins mean per-seed MSE difference<0 or per-seed paired-date mean IC difference>0; report wins/losses/ties and defined seed counts, never choose seeds by wins.

Descriptive manipulation details
- Per branch logit/weight values for each Static seed, and means±SD by variant/h/branch; centered-logit MAD=mean|logits−branch mean(logits)|; weight MAD=mean over observations/scales |softmax−.25|. These are separate quantities.
- Within-model SD is the mean across4 scales of the population SD over that model's6647 TEST observations, with the prior Step22B constant-to1e-7 convention yielding0; report max range and constancy too. Adaptive uses per-observation router weights. Copy T0 Static/Adaptive logits/weights from their exact stored sources. No checkpoint/model evaluation is added for T0.
- Interpretation is per passing variant×h. Any PRIMARY significant improvement is EXPLORATORY; a failed manipulation never supports H or its negation. For passing cells with no PRIMARY improvement, use the preregistered no-improvement wording, scoped only to those cells. Report metric-specific directions, even if MSE and IC disagree, without specification selection.

Delivery/validation
One user-launched run command executes fresh hash gate, timing pilots,45 fits,TEST inference,analysis,report,independent validation and completion sound. Existing inputs never written; every new file created x/xb under step22f only. Record every imported/reused Python source and AST-only statistical/trainer source SHA, package versions, seeds and profile in RNG_MANIFEST.json; protocol never edited. No user confirmation needed to proceed once launched, except a STOP condition requires reporting rather than bypass.
REPORT_STEP22F_FINAL.md begins BLOCKING; includes protocol hash, gates,timing,restarts,all6 arms,ensemble/epoch/manipulation tables,all30 inference rows with raw/Holm p,direction,EXPLORATORY tags,seed wins and PRIMARY bounds; attempt/file/source inventory. Independent REPORT_STEP22F_VALIDATION.md checks source/protocol hashes,groups/defaults,profiles/counts/keys/float32 labels,TEST timing,30 rows/10 families,undefined p1,unchanged tests under manipulation gate,direction/bound signs,stored replicate interval/bound reproduction. No additional hypothesis tests.
