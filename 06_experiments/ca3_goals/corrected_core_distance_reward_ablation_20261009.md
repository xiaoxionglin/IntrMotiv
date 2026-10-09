# Corrected-core encoder and worker distance-reward ablation

**Status, 9 October 2026:** the six 100M C05/C15 hit-only first-wave runs were submitted as jobs **8317710–8317715** and passed the canonical submitted-job audit. Both short compute-node qualifications finished with finite losses, commanded hits paid exactly 1, and their final checkpoints passed exact learner reload. Archived checkpoint restoration and historical field-map replay found no material mismatch beyond observed same-source stochastic variation; the September source is still unavailable, so historical comparisons retain residual source uncertainty. **The 45-run encoder wave remains held until the complete first-wave result is shown.** No recurring monitor is active.

## Question and matrix

This program separates three questions in the original corrected-core C01, C05, and C15 families: whether the worker's temporal hit bonus helps, whether DG benefits from temporal rather than generic event credit, and whether PPO gradients into DG help when interval credit is absent. It does not use the recent episode-long variants.

The [original 100M corrected-core analysis](../corrected_core_reevaluation_20260901.md) reported mean terminal coverage AUC of 38.0 (C01), 42.0 (C05), and 78.6 (C15) across the three seeds. C15's target-hit lift was 0.898 and action sensitivity 0.0019 despite its broad graph; C05 had lift 1.091 and sensitivity 0.0256. These are descriptive online quantities and explain why this program makes matched-start command behavior, rather than coverage or graph confidence, its control endpoint.

| Release | Families | Encoder interval credit | PPO-to-DG | Worker reward | New 100M runs |
| --- | --- | --- | --- | --- | ---: |
| First | C05, C15 | Original temporal | STOP | Hit reward 1, distance bonus 0 | 6 |
| Held | C05, C15 | Constant, none with STOP, none with JOINT | As named | Original temporal bonus or zero bonus | 36 |
| Held | C01 | Constant, none with STOP, none with JOINT | As named | Original flat worker reward | 9 |

The nine completed September C01/C05/C15 runs supply the historical temporal-credit references. The [first-wave StudySpec](../../hpc_runs/studies/corrected_core_hit_only_first_20261009.study.json) has schema `intrmotiv/study/v1`, workflow `1.14.1`, SHA-256 `8a2b432c06f4fd1f81fe3bd39212b7e01ed444b6cdd89c9672fb68b7f61cc5cc`. The [held second-wave StudySpec](../../hpc_runs/studies/corrected_core_encoder_worker_later_20261009.study.json) has the same schema and workflow, 45 runs, and SHA-256 `37733408996f10975eb055ea93226885e15bdb36b39e6ee5722e8afad354a853`. The [two-run qualification StudySpec](../../hpc_runs/studies/corrected_core_hit_only_qualification_20261009.study.json) is separate from the 51 production runs, SHA-256 `9a1db8d51e0fa107ff1a6f5f21593d418bd4c2a623f5bb978646ac7c2a2dc930`.

Both production waves use seeds 8, 99, and 123, 100M frames, and checkpoint/snapshot ages 5M, 25M, 50M, 75M, and 100M. Family-specific archived goal timing, deadlines, architecture, DG maintenance losses, $\gamma=0.99$, and GAE $\lambda=0.95$ remain fixed. The first wave changes `hrl_distance_bonus_coeff=0.1` to `0.0`, keeping `hrl_worker_reward_mode=hit_distance`. The second wave changes only its named encoder credit, PPO gradient route, or worker bonus factor. Constant credit applies to the same qualifying dominant DG onsets and arrival rows; `none+JOINT` retains batch recruitment and multi-activation maintenance losses.

The fixed distance-unit constants come from the event-count-weighted mean feedback in the archived first 5M frames, divided by `reward_scale=0.1`. [Calibration evidence](results/corrected_core_hit_only_20261009/encoder_constant_calibration.json) records 227,355 qualifying C01 events, 190,111 C05 events, and 182,864 C15 events, giving 3.00821620774, 3.18483411306, and 3.48152723267 respectively. Every family exceeded the predeclared 100-event threshold, so the 35.5 midpoint fallback was unnecessary. These constants were fixed before later-wave results and do not change with seed, worker reward, or training age.

The temporal worker bonus versus zero-bonus contrast changes both timing information and added reward magnitude. A magnitude-matched constant-hit bonus is a future, separate experiment if a timing-only claim becomes necessary.

The canonical telemetry contracts include the five checkpoint ages for field maps, trajectories, directed-graph analysis, and frozen command probes at 75M and 100M. A synthetic checkpoint inventory generated 18 field-map rows, 10 trajectory rows, and 12 intervention rows for the first wave. The held wave generated 129, 75, and 72 respectively, with intervention rows restricted to C05/C15; C01 has no commanded-goal probe. No telemetry jobs are submitted before their checkpoints exist.

## Provenance and release checks

The archived [job manifest](results/corrected_core_hit_only_20261009/historical_jobs.tsv), [saved configurations](results/corrected_core_hit_only_20261009/historical_configs.json), and [reference manifest with config and checkpoint hashes](results/corrected_core_hit_only_20261009/historical_provenance.json) pin all nine historical cells. The new isolated workspace source is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_corrected_core_distance_ablation_20261009`, copied from base commit `cdf9ffe9026dc483e9c31dae7a1365af67c28642` plus the [reward-ablation patch](../../hpc_runs/patches/push_pull_ablation_20261007.patch) and a [restore-only learning-rate patch](../../hpc_runs/patches/preserve_loaded_optimizer_lr_20261009.patch). Its [source release manifest](results/corrected_core_hit_only_20261009/source_release.json) pins 750 Python files under a reproducible sorted-path/file-hash manifest SHA-256 `e47b1e4d67ee58d20ad44a87f786eedf9372b085c48eb514b97ea55cf683a2f0`. The September jobs predate the earliest preserved model-source revision, so even a passing compatibility audit cannot turn historical comparisons into exact within-source causal estimates.

The archived and current parsers shared 239 saved configuration keys in each family. Of these, 234 matched; the five unequal keys were command line, Git identity, and heartbeat bookkeeping. All 51 production commands and both qualification commands parsed in the isolated source with `[128,128]` decoder layers, the intended $\gamma$ and GAE $\lambda$, DG maintenance enabled, and the declared STOP or JOINT route. All six first-wave commands match their archived same-seed scientific arguments after excluding the intended hit-bonus change, explicit compatibility/default flags, output paths, and tracking/checkpoint metadata. Focused reward and study tests passed, 42/42; another 20 PPO-gradient boundary and DG maintenance tests passed. A direct reward-path probe paid exactly 1 on a commanded hit and 0 on non-hits at coefficient zero; the separate exploration and encoder reward streams stayed on their original paths.

The qualification print-only manifest at `train_dir/_slurm/corrected_core_hit_only_qualification_20261009/20261009T011026Z/jobs.tsv` and submitted manifest at timestamp `20261009T011048Z` both passed the canonical audit. Qualification jobs 8317584–8317585 requested 40 CPUs, 80 GB and four hours each, with all outputs and caches in the active workspace; both completed at 2,064,384 frames with exit code zero. After correcting an analysis-only TensorBoard tag, the [qualification analysis StudySpec](../../hpc_runs/studies/corrected_core_hit_only_qualification_analysis_20261009.study.json), SHA-256 `bad3940c80ab656ba97ed34f9501c7480aaf621f670465c9e52658d93d5b99ed`, collected their online data without changing the immutable training StudySpec. The final first-wave [print-only manifest](results/corrected_core_hit_only_20261009/first_wave_print_only_jobs.tsv) and [submitted manifest](results/corrected_core_hit_only_20261009/first_wave_submitted_jobs.tsv) passed six-command, job-ID, and workspace audits with a 60-hour CPU limit under StudySpec SHA-256 `8a2b432c06f4fd1f81fe3bd39212b7e01ed444b6cdd89c9672fb68b7f61cc5cc`. The remote submission manifest is at `train_dir/_slurm/corrected_core_hit_only_first_20261009/20261009T022058Z/jobs.tsv`; the root prefix is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/`. Jobs 8317710–8317712 are C05 seeds 8/99/123; 8317713–8317715 are C15 seeds 8/99/123. All six were running immediately after submission.

The first historical reload attempt failed before model construction because the archived config lacks a later parser key. A compatibility sidecar filled newly introduced parser defaults and overlaid every saved archived value. The strict full-state test then found newer model keys. The key audit found no archived-only model keys; the newer model adds graph-anchor, calibration, prospective-diagnostic, and DG feature-statistics buffers. The archived graph-anchor mode is `off` and DG BatchNorm mode is `legacy_batch`, so the added anchor and feature-centering states should be inactive; recruitment row counts are derived from the archived committed-row state for C15. The [C05 certificate](results/corrected_core_hit_only_20261009/historical_shared_c05.json) and [C15 certificate](results/corrected_core_hit_only_20261009/historical_shared_c15.json) show exact equality of every archived model tensor, optimizer entry, and training counter at 100,040,704 frames, with 42 newer buffers explicitly excluded from full-state equality. This establishes compatible archived-state restoration; the replay comparison below supplies bounded behavioral evidence. It does not remove the residual cross-source uncertainty of historical controls.

The first historical replay accidentally used the training `fixedlength` environment: the current evaluator preserves the saved config's environment, whereas the September evaluator overrode it with `openfield_map2_fixed_loc3_noreward`. Jobs 8317604–8317605 were cancelled when the mismatch was found. New immutable replay sidecars changed only that evaluation environment; jobs 8317627–8317628 confirmed the same nonfixed level as September logs 7975099_33 and 7975099_68. Repeat current-source jobs 8317660–8317661 then measured stochastic replay variation. All four completed with exit code zero. The [three-way map comparison](results/corrected_core_hit_only_20261009/reference_replay_variation.json) includes raw NPZ hashes and the following median per-unit map correlations:

| Family | September versus current | Current versus current repeat | DG active-fraction mean absolute change, old/current versus repeat |
| --- | ---: | ---: | ---: |
| C05 | 0.613 | 0.567 | 0.0056 versus 0.0033 |
| C15 | 0.381 | 0.397 | 0.0043 versus 0.0018 |

Visited-bin Jaccard was 0.785 versus 0.819 for C05 and 0.950 versus 0.965 for C15. Spatial-information mean absolute changes were 0.020 versus 0.022 for C05 and 0.017 versus 0.010 for C15. The old-to-current map differences are of the same broad scale as repeat variation, with somewhat larger C15 activity and information differences. The replay gate found no material behavioral mismatch, so the six first-wave runs were released. This is a bounded compatibility check, not proof that the missing September source is identical.

The [online qualification summaries](results/corrected_core_hit_only_20261009/qualification_health.json) cover 1.0–2.064M frames. They establish reward-path and numerical health, not learned control.

| Measure | C05 | C15 |
| --- | ---: | ---: |
| Mean correct command outcomes per reported interval | 26.3 | 7.7 |
| Mean correct hit reward | 1.0 | 1.0 |
| Mean active goal fraction | 0.999 | 0.652 |
| Mean DG silent-unit fraction | 0 | 0 |
| Mean policy loss | −0.00052 | −0.00009 |
| Mean value loss | 0.301 | 0.523 |
| Mean coverage AUC | 69.9 | 62.3 |

All selected policy/value/advantage and DG statistics were finite. C15 had fewer selected goals and hits at this young age; the qualification is an exposure check, not a performance comparison. A strict C15 reload first failed because its saved optimizer held a temporary effective rate of 0.000175 after an invalid-sample minibatch, while learner initialization restored the nominal `curr_lr=0.0002`. The [restore-only patch](../../hpc_runs/patches/preserve_loaded_optimizer_lr_20261009.patch) leaves fresh initialization unchanged and preserves the checkpoint's exact optimizer state during reload; the next training update already reapplies its effective rate. The [fresh-initialization certificate](results/corrected_core_hit_only_20261009/fresh_init_check.json) and [C05](results/corrected_core_hit_only_20261009/qualification_reload_c05_exact.json)/[C15](results/corrected_core_hit_only_20261009/qualification_reload_c15_exact.json) exact-reload certificates passed on compute nodes.

## Analysis and decision rule

At 5M, check goal exposure, hit counts and latencies, DG activity, finite policy/value losses, advantage behavior, and checkpoint integrity. At matched ages, report each seed's effect with uncertainty and failures/censoring. Online hit counts alone cannot establish goal control. If hits are too scarce, classify the bonus test as inconclusive.

### Place fields — pending

Run the canonical manifest-driven 10k-decision evaluation at declared checkpoints, plus terminal seeds 8 and 123. Report active and silent DG units, thresholded and pre-threshold maps, amplitude-weighted spatial information, active-only cosine, distinct peaks, mono-field structure, and visited-bin denominators. Replay a common observation panel so policy visitation does not masquerade as field change.

### Trajectories — pending

Use segmented, reset-aware paths and occupancy at the same ages. Report coverage and visited support by seed; checkpoint rollouts are policy driven, not a fixed-trajectory drift test.

### Directed graphs — pending

Report attempted and unattempted ordered edges, outcomes, reliability, and connectivity at shared ages. Keep stored confidence separate from executed command success. C01 has no controllable graph; document its exploration behavior without inventing zero-valued graph evidence.

### Frozen command control — pending

The StudySpecs declare the canonical landmark matched-command intervention at 75M and 100M for all three seeds, selecting C05/C15 and excluding flat C01. It uses frozen policies and graphs, identical starts, alternative commands, and observation windows of 64, 128, 256, and 900 decisions. The September controls have no archived matched-command action traces, so evaluate their checkpoints under the same current-source probe and record actual checkpoint frames, especially near 75M. Report executed-command detector hits, action changes, times to hit, failures, censoring, eligible source/goal support, and physical destinations only when independent field maps justify them. Judge the worker bonus from matched-start command effects and three paired seed differences, not retrospective label shuffles or online hits. C01 is the flat exploration reference and has no commanded-hit intervention.

After the full first-wave 100M result and these four analyses are shown, the later-wave release can be reviewed. No automatic second-wave submission is authorized.

## Reusable lesson

The archived launch manifest plus saved config were needed to recover the true C05/C15 timing and deadlines. The canonical StudySpecs and parser audit prevented a newer calibration configuration from silently replacing them. Historical config replay needs a compatibility layer for later parser defaults, an evaluation-level check against old logs, and a within-source noise reference before interpreting map differences. Keep future historical controls in an intact source snapshot when possible.
