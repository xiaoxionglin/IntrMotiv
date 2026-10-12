# C05 worker hit timing under longer PPO credit

**Status, 12 October 2026:** Both 2M-frame seed-99 qualifications passed. All ten 100M-frame production jobs 8344868–8344877 exited `0:0`; each run directory has a named 100,007,936-frame checkpoint. The canonical 90–100M online collector completed for all five paired seeds. Its results below are interim training readouts; checkpoint-content, spatial-snapshot, place-field, downstream-layer, trajectory, graph, and executed-control audits remain open.

## Question and contrast

Does removing temporal modulation from the worker's reward at a commanded DG-detector hit matter when delayed rewards receive longer PPO credit? Hold the C05 encoder's temporal event credit, architecture, manager, environment, optimizer, and all other scientific arguments fixed. Change the worker bonus from the original temporal distance signal to the existing C05 mean-matched constant of 59.1543529604 distance units. Both arms retain the base hit reward of 1 and distance-bonus coefficient of 0.1; the constant removes timing information from the bonus, not the hit reward itself.

Use $\gamma=0.999$ and GAE $\lambda=0.99$, the established longer-credit regimen. The original same-source C05 timing factorial uses $\gamma=0.99$ and the runtime's implicit $\lambda=0.95$. Its existing five-seed worker contrast is a useful historical comparison, but a difference between the old and new contrasts would reflect the **joint credit-horizon change**, not $\gamma$ alone. The new within-regimen temporal-minus-constant contrast is cleanly matched by seed. The constant was calibrated from C05's 2M temporal-bonus reference, not borrowed from C15; [the original calibration and source record](cosyne27_staged_push_pull_submission_20261010.md) retain its provenance.

| Gate | Exact StudySpec | Runs and status |
| --- | --- | --- |
| 2M qualification | [Qualification StudySpec](../../hpc_runs/studies/cosyne27_c05_long_credit_worker_qualification_20261012.study.json), SHA-256 `32f83fbc1a99e16acca5b82a45f621100bfec558e558758be48355ce6f3665e8` | Jobs 8343877/8343878 completed with exit zero; [submitted manifest](results/cosyne27_high_discount_hit_timing_20261012/jobs.tsv), [exact audit](results/cosyne27_high_discount_hit_timing_20261012/qualification_submitted_audit.json), and [hashed runtime check](results/cosyne27_high_discount_hit_timing_20261012/qualification_runtime_check.json). |
| Five-seed production | [Production StudySpec](../../hpc_runs/studies/cosyne27_c05_long_credit_worker_s5_20261012.study.json), SHA-256 `448161895fe97f0d289e809e15f17152886a8445a86a913b8efb31ad34c264c5` | Jobs 8344868–8344877 across seeds 8, 99, 123, 17, 41; [print-only audit](results/cosyne27_high_discount_hit_timing_20261012/production_print_only_audit.json), [submitted manifest](results/cosyne27_high_discount_hit_timing_20261012/production_submitted_jobs.tsv), and [submitted audit](results/cosyne27_high_discount_hit_timing_20261012/production_submitted_audit.json). |

Both specifications use schema `intrmotiv/study/v1` and workflow `1.14.1`. The isolated NEMO2 source is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_cosyne27_c05_long_credit_worker_20261012/`, copied from the validated C05 COSYNE release. The canonical workflow suite passed 38/38 there. Every rendered command passed the actual `parse_dmlab_args` and asserted $\gamma$, $\lambda$, encoder mode, worker mode, hit constant, and frame target. The seed-99 production arms differ from their earlier temporal-encoder C05 matches only in $\gamma$, explicit $\lambda$, and run/tracking identity. The submitted two-job and ten-job audits report exact command matching and workspace-only paths. The active allocation had 1.3 TB available and passed a write/fsync check. The ten production jobs were absent from queue and output directories before their single submission, then all entered `RUNNING` at the first scheduler check.

## Qualification evidence and analysis gate

Both qualifications saved checkpoints at 2,064,384 environment frames. Each logged 63 finite policy-loss, value-loss, advantage-spread, DG-density, decoder-loss, and hit-control observations. Each wrote both 1M and 2M online-spatial snapshots; the 2M arrays have finite DG activity, 16/16 active DG units, and 16 finite spatial-information scores. Decoder loss was finite and nonzero, with 165,760 decoder parameters reported. The [runtime check](results/cosyne27_high_discount_hit_timing_20261012/qualification_runtime_check.json) binds the source event and snapshot files by SHA-256.

At seed 99 over this short qualification, temporal versus constant bonus had 1,114 versus 843 logged correct-hit observations; correct-hit-count-weighted reward magnitudes were 1.6172 versus 1.5915, and weighted elapsed decisions were 10.96 versus 12.97. The constant magnitude matches the calibrated fixed reward, and the temporal term was engaged. These are repeated learner-batch observations from one seed, so their differences do **not** estimate the five-seed effect or executed goal control. Both Slurm jobs exited zero.

## Interim five-seed online window

The [canonical 90–100M collection](results/cosyne27_high_discount_hit_timing_20261012/online_90_100m/analysis_manifest.json) matches the submitted StudySpec SHA-256 `448161895fe97f0d289e809e15f17152886a8445a86a913b8efb31ad34c264c5`, schema `intrmotiv/study/v1`, and workflow `1.14.1`. Its [ten run rows](results/cosyne27_high_discount_hit_timing_20261012/online_90_100m/per_run.csv) all reach 100,040,704 frames. The selected learner scalars have 305 finite observations per run within the common 90–100M window; online coverage has 520–541 finite observations. These scalar means are *per logged observation*, not count-weighted trial latency or total exposure.

| Training seed | Temporal minus constant correct detector-hit count per log point | Temporal minus constant online coverage AUC |
| --- | ---: | ---: |
| 8 | −2.16 | −6.90 |
| 17 | +16.65 | −22.44 |
| 41 | +56.49 | −33.18 |
| 99 | −25.77 | +22.31 |
| 123 | +57.15 | −21.44 |
| Paired mean | +20.47 | −12.33 |

The [canonical paired table](results/cosyne27_high_discount_hit_timing_20261012/online_90_100m/paired_contrasts.csv) and [summary](results/cosyne27_high_discount_hit_timing_20261012/online_90_100m/paired_contrast_summary.csv) retain all 16 declared metrics. The temporal bonus has a higher logged correct-hit count in 3/5 seed pairs, while online coverage AUC is lower in 4/5. All ten policy-loss, value-loss, advantage-spread, DG-density, and DG-usage-entropy window means are finite, and the logged DG silent fraction is zero in every run. This is mixed online evidence: neither a consistent speed gain nor executed goal control follows. Count-weighted hit latency and reward magnitude require aligning each scalar to its contemporaneous correct-hit count; arithmetic means of the logged latency or magnitude are insufficient.

For completed 100M runs, report five paired seed differences for 90–100M correct-hit exposure and count-weighted latency/reward, online coverage, policy-specific and common-observation DG/CA3/decoder information, trajectories, and directed graph support. Keep stored prospective detector-hit ratios separate from executed commands. Any frozen matched-command evaluation needs bounded source shards and independently mapped physical arrival; the original monolithic evaluator timed out and the current schema only certifies DG-detector hits. Report 64/128/256/900-decision censoring and failures separately. Do not infer better goal control from online hits, field quality, or stored graph ratios alone.

## Reusable lesson

The original COSYNE StudySpec omitted GAE $\lambda$, and the verified runtime default is 0.95. A high-$\gamma$ study should declare $\lambda$ and describe the joint effective credit horizon before comparing it with that original cohort. Reusing the versioned C05 source and its own hit calibration made the reward contrast precise without a new training implementation.
