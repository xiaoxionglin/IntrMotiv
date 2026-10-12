# Fresh COSYNE comparison completion

**Status, 12 October 2026:** All 15 production runs common to every proposed scope passed local and NEMO2 StudySpec validation, exact print-only launcher generation, both 2M qualification gates, real runtime argument parsing, and submitted-manifest audits. C01 seeds 99, 123, and 41, jobs 8335153, 8335154, and 8335156, completed with Slurm exit `0:0` and 100,007,936-frame checkpoints. The other 12 jobs were running with advancing learner samples at the 14:52 Berlin check. Their 100M outcomes remain pending. Additional cells below are an inventory only and have not been submitted.

## Initial 15 runs

| Comparison | StudySpec | Five-seed runs | Source |
| --- | --- | ---: | --- |
| C15 constant worker hit bonus × temporal/constant encoder credit | [C15 production](../../hpc_runs/studies/cosyne27_c15_constant_hit_s5_20261012.study.json), SHA-256 `3a6eaacac55bd97d1bc6f9a6678d9f6592e6372672630805338220a7022bafab` | 10 | Isolated C15 PPO recovery source; patch SHA-256 `797dd997ef5d9e83839449fdce8978bfc5b365ee98d47a5d86aecbcea49c93b7` |
| C01 no encoder event credit, PPO stopped at DG | [C01 production](../../hpc_runs/studies/cosyne27_c01_no_credit_s5_20261012.study.json), SHA-256 `ab244148366ec9f52cc7c351650b42a624fb1f4fb8e7d66e428faa752630d837` | 5 | Original isolated COSYNE source |

All rows use seeds 8, 99, 123, 17, and 41, the original environment, 100M-frame horizon, architecture and optimization settings, and the existing online spatial and evaluation protocol. The only C01 scientific argument change from its temporal arm is `--encoder_interval_credit_mode=none`; `--ppo_dg_gradient=stop` is retained. The only C15 scientific argument additions relative to its original temporal-hit arm are `--hrl_distance_bonus_mode=constant` and `--hrl_distance_bonus_constant=59.0162404172`. The C15 encoder constant remains 5.58606080982 distance units. New names, tracking identity, W&B directories, and workspace output roots prevent collisions with existing attempts.

The C15 hit constant comes from C15's own 2M temporal-credit/temporal-hit reference job 8324844, not C05's calibration. In the first 2M frames, 46 logged blocks contain 308 correct hits. Weighting each block's `intrmotiv/hrl/control/correct_reward_magnitude_mean` by `intrmotiv/hrl/control/correct_count` gives a mean reward of 1.5901624041718323. With hit reward 1, distance coefficient 0.1, and reward scale 0.1, the matching distance constant is 59.016240417183226. The [machine-readable calibration](results/cosyne27_staged_push_pull_20261010/c15_worker_calibration_20261012.json) pins the event-file hash and calculation. C05's constant, 59.1543529604, is not used.

The isolated C15 source is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_cosyne27_c15_ppo_recovery_20261011/`; the C01 source is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_cosyne27_push_pull_20261010/`. Both have canonical workflow 1.14.1. The prior [C15 recovery record](cosyne27_staged_push_pull_submission_20261010.md#completion-and-c15-recovery) documents the learner patch and its 61 focused source tests. Newly staged files are only declarative StudySpecs and thin adapters; the focused canonical workflow suite passed 42/42 in **each** staged source. All generated train, W&B, cache, telemetry, and Slurm paths are in `/work/classic/fr_xl1014-corridor-geometry/`, whose available capacity was 1.3 TB at preparation.

## Qualification and release

| Gate | StudySpec SHA-256 | Job IDs | Required evidence |
| --- | --- | --- | --- |
| C01 no-credit 2M | [StudySpec](../../hpc_runs/studies/cosyne27_c01_no_credit_qualification_20261012.study.json), `ccdff39b4c586273f2bcd073ae6dec4ff1fa93e0639699211af51b0f23e63bcc` | 8335147 | Passed: exit zero, 2.064M-frame checkpoint, 2.032M learner steps, finite losses/activity, two workspace online-spatial snapshots |
| C15 constant-hit temporal/constant encoder 2M | [StudySpec](../../hpc_runs/studies/cosyne27_c15_constant_hit_qualification_20261012.study.json), `3152029dd13c86a68cb2e0cb51f38e3b0b92a3f13d613eb64a46908e76e415f4` | 8335148–8335149 | Passed: both exit zero, 2.064M-frame checkpoints, 2.032M learner steps, finite losses/activity, two workspace snapshots each; 354/294 correct hits with respective weighted reward 1.590162415/1.590162427 |

Both archived qualification manifests, [C01](results/cosyne27_staged_push_pull_20261010/c01_no_credit_qualification_20261012_jobs.tsv) and [C15](results/cosyne27_staged_push_pull_20261010/c15_constant_hit_qualification_20261012_jobs.tsv), passed `audit-submission --submitted`. Production print-only and [C01](results/cosyne27_staged_push_pull_20261010/c01_no_credit_s5_20261012_jobs.tsv) and [C15](results/cosyne27_staged_push_pull_20261010/c15_constant_hit_s5_20261012_jobs.tsv) submitted manifests have 5 and 10 exact commands, 60-hour CPU limits, 40 CPUs, 80 GB, and `workspace_paths_valid=true`. Every command passed the real `parse_dmlab_args` in its staged source; both submitted manifests passed `audit-submission --submitted`. All 15 jobs entered `RUNNING` at release. Jobs 8335153, 8335154, and 8335156 then completed with exit `0:0` and 100,007,936-frame milestones; each advanced beyond 12.5 million learner samples. The [verified outcome record](../cosyne_2027/verified_production_outcomes.tsv) drives its catalog status. The other 12 were advancing in actor frames and learner samples at the 12 October 14:52 Berlin check. Require exit zero and a checkpoint at or beyond 100M before scientific analysis. The two pre-existing C15 matched-constant encoder replacements, jobs 8329014 and 8329015, completed with exit `0:0` and each saved a 100,007,936-frame milestone; the [recovery record](cosyne27_staged_push_pull_submission_20261010.md#completion-and-c15-recovery) contains the details. Their failed originals, 8327183 and 8327184, are audit attempts, not replicates.

## Remaining fresh factorial cells, pending scope choice

The comparable fresh matrix is three encoder settings (temporal credit, mean-matched constant credit, no credit with PPO-to-DG stopped) crossed with three worker settings (temporal hit bonus, mean-matched constant hit bonus, zero hit bonus) for C05 and C15. C01 has no worker-hit factor. Older no-credit and zero-bonus runs differ in source or constant calibration and do not fill these fresh cells. Each absent cell would require five new seeds.

| Family | Encoder credit | Worker hit bonus | Additional runs |
| --- | --- | --- | ---: |
| C05 | Temporal | Zero | 5 |
| C05 | Mean-matched constant | Zero | 5 |
| C05 | None; PPO-to-DG stopped | Temporal | 5 |
| C05 | None; PPO-to-DG stopped | Mean-matched constant | 5 |
| C05 | None; PPO-to-DG stopped | Zero | 5 |
| C15 | Temporal | Zero | 5 |
| C15 | Mean-matched constant | Zero | 5 |
| C15 | None; PPO-to-DG stopped | Temporal | 5 |
| C15 | None; PPO-to-DG stopped | Mean-matched constant | 5 |
| C15 | None; PPO-to-DG stopped | Zero | 5 |
| **Total beyond initial 15** | **10 absent cells** | | **50** |

C01 has no further absent cell in this three-level encoder matrix; its five no-credit runs are part of the initial 15.

The full completion would therefore be 65 new production runs: the agreed 15 plus 50 pending the user's scope choice. A narrower 40-run completion could add C05's five cells; a 45-run completion could additionally add one C15 no-credit cell. Such prioritization is a design choice, not a submitted plan. A separate PPO-to-DG joint-gradient branch would expand the matrix further and is outside this 65-run count.

## Workflow lesson

The existing COSYNE StudySpecs made scientific argument parity straightforward: validated run expansion showed only the intended credit or worker-bonus arguments changing. NEMO2's `sacct` state alone is insufficient when the learner can stall, so qualification also needs frame and checkpoint evidence. The existing canonical launcher and submission audit provided exact workspace and command checks; no new launcher code was needed. Keep C15's hit calibration as a source-family-specific evidence artifact before any later factorial extension.
