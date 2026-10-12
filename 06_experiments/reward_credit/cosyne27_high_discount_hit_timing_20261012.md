# C05 worker hit timing under longer PPO credit

**Status, 12 October 2026:** Two 2M-frame seed-99 qualifications are running on NEMO2: temporal hit bonus job 8343877 and mean-matched constant hit bonus job 8343878. The ten 100M-frame production commands for five paired seeds passed print-only review but have **not** been submitted. No high-discount outcome is available yet.

## Question and contrast

Does removing temporal modulation from the worker's reward at a commanded DG-detector hit matter when delayed rewards receive longer PPO credit? Hold the C05 encoder's temporal event credit, architecture, manager, environment, optimizer, and all other scientific arguments fixed. Change the worker bonus from the original temporal distance signal to the existing C05 mean-matched constant of 59.1543529604 distance units. Both arms retain the base hit reward of 1 and distance-bonus coefficient of 0.1; the constant removes timing information from the bonus, not the hit reward itself.

Use $\gamma=0.999$ and GAE $\lambda=0.99$, the established longer-credit regimen. The original same-source C05 timing factorial uses $\gamma=0.99$ and the runtime's implicit $\lambda=0.95$. Its existing five-seed worker contrast is a useful historical comparison, but a difference between the old and new contrasts would reflect the **joint credit-horizon change**, not $\gamma$ alone. The new within-regimen temporal-minus-constant contrast is cleanly matched by seed. The constant was calibrated from C05's 2M temporal-bonus reference, not borrowed from C15; [the original calibration and source record](cosyne27_staged_push_pull_submission_20261010.md) retain its provenance.

| Gate | Exact StudySpec | Runs and status |
| --- | --- | --- |
| 2M qualification | [Qualification StudySpec](../../hpc_runs/studies/cosyne27_c05_long_credit_worker_qualification_20261012.study.json), SHA-256 `32f83fbc1a99e16acca5b82a45f621100bfec558e558758be48355ce6f3665e8` | Two seed-99 arms submitted as 8343877/8343878; [submitted manifest](results/cosyne27_high_discount_hit_timing_20261012/jobs.tsv) and [exact audit](results/cosyne27_high_discount_hit_timing_20261012/qualification_submitted_audit.json). |
| Five-seed production | [Production StudySpec](../../hpc_runs/studies/cosyne27_c05_long_credit_worker_s5_20261012.study.json), SHA-256 `448161895fe97f0d289e809e15f17152886a8445a86a913b8efb31ad34c264c5` | Ten commands across seeds 8, 99, 123, 17, 41; [print-only audit](results/cosyne27_high_discount_hit_timing_20261012/production_print_only_audit.json); no jobs submitted. |

Both specifications use schema `intrmotiv/study/v1` and workflow `1.14.1`. The isolated NEMO2 source is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_cosyne27_c05_long_credit_worker_20261012/`, copied from the validated C05 COSYNE release. The canonical workflow suite passed 38/38 there. Every rendered command passed the actual `parse_dmlab_args` and asserted $\gamma$, $\lambda$, encoder mode, worker mode, hit constant, and frame target. The seed-99 production arms differ from their earlier temporal-encoder C05 matches only in $\gamma$, explicit $\lambda$, and run/tracking identity. The submitted two-job and generated ten-job audits report exact command matching and workspace-only paths. The active allocation had 1.3 TB available and passed a write/fsync check. Both qualification jobs entered `RUNNING` at the first scheduler check.

## Release and analysis gate

Before submitting production, require exit zero for both qualifications, checkpoints beyond 2M frames, finite policy/value/advantage statistics, active DG/decoder output, and the two declared online-spatial snapshots. Confirm the reward perturbation is engaged: hit-weighted temporal and constant magnitudes, correct-hit counts, and conditional elapsed decisions. A failed qualification requires diagnosis and a new immutable study identity if scientific commands change.

For completed 100M runs, report five paired seed differences for 90–100M correct-hit exposure and count-weighted latency/reward, online coverage, policy-specific and common-observation DG/CA3/decoder information, trajectories, and directed graph support. Keep stored prospective detector-hit ratios separate from executed commands. Any frozen matched-command evaluation needs bounded source shards and independently mapped physical arrival; the original monolithic evaluator timed out and the current schema only certifies DG-detector hits. Report 64/128/256/900-decision censoring and failures separately. Do not infer better goal control from online hits, field quality, or stored graph ratios alone.

## Reusable lesson

The original COSYNE StudySpec omitted GAE $\lambda$, and the verified runtime default is 0.95. A high-$\gamma$ study should declare $\lambda$ and describe the joint effective credit horizon before comparing it with that original cohort. Reusing the versioned C05 source and its own hit calibration made the reward contrast precise without a new training implementation.
