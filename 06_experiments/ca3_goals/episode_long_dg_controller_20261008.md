# Episode-long DG goals: matched controller study

**Status, 8 October 2026:** all four corrected short qualifications, their exact checkpoint reloads, and a frozen place-field smoke passed. The 12 declared 150M-frame production jobs, 8306923–8306934, have been submitted and were running at the first scheduler check. The bounded matched-command evaluator smoke was still running. No production outcome or horizon effect is claimed yet.

## Question and design

Does a 64-decision fallback prevent the controller from learning to reach useful DG goals? The intervention changes option persistence: a commanded goal remains selected until that field is hit or the physical episode ends. Intervening visits to other fields do not terminate it. The environment's fixed-length episode ends after 900 policy decisions (7,200 engine frames at frameskip 8). CA3 length $L=64$ is unchanged.

The [production StudySpec](../../hpc_runs/studies/episode_long_dg_controller_20261008.study.json) declares 12 runs, seeds 8, 99 and 123, 150M frames per run, and checkpoint and online-spatial milestones at 5M, 25M, 50M, 75M, 100M and 150M. Each arm has 16 DG channels, the same fixed visual trunk, CA3 size, reward rule, map instruction coefficient 9, action set, and direct frontier manager. Recruitment remains monitor-only with zero replacements.

| Arm | DG and eligible goals | Worker conditioning |
| --- | --- | --- |
| ORACLE_FILM | Four fixed Gaussian fields in channels 0–3; twelve learned context channels; goals 0–3 | FiLM |
| ORACLE_LEG9 | Same fields and goal IDs | Legacy decoder; goal one-hot multiplied by 9 only at its decoder input |
| LEARNED4_FILM | Sixteen learned channels; goals 0–3 | FiLM |
| C15_FILM | Sixteen learned channels; goals 0–15 | FiLM |

The fixed-field centers and $\sigma=20$ match the [finite-horizon four-field screen](four_prescribed_dg_controller_20261007.md). Its four events are accessible yet sparse in policy rollouts, so arrival and failure denominators must be reported by field and seed. For ORACLE, the learned projection's first four pre-threshold rows are unused; only post-replacement DG activity represents the fixed fields.

## Comparisons and interpretation

At 75M, compare ORACLE_FILM and LEARNED4_FILM with their same-seed finite-horizon counterparts from the six-run screen. The rendered argument audit found only the new episode-long mode, planned training length and milestone/tracking metadata as substantive differences. The default mode is `deadline`, and focused tests preserve its old behavior. This establishes a matched 75M horizon comparison after source and checkpoint parity are rechecked at evaluation time. There is no finite-horizon 150M control.

At 150M, compare the four new arms within the common episode-long protocol and describe their longer training course. ORACLE_FILM versus LEARNED4_FILM tests fixed-field replacement under the same four-goal vocabulary. ORACLE_LEG9 versus ORACLE_FILM tests the combined conditioning-and-gain package; without a legacy-gain-1 arm it cannot isolate gain 9. C15_FILM asks how the controller behaves with all 16 learned goals and is not a representation-matched control.

The later [C05 FiLM supplement](episode_long_c05_supplement_20261008.md) adds three separately fingerprinted, seed-matched episode-long runs for a C05–C15 controller-family contrast at 75M and 150M. It does not alter this submitted twelve-run StudySpec or its jobs.

## Outcome protocol

The primary control measure is a **physical arrival** at a target field from an identical start under its own command versus alternative commands. The frozen policy and graph remain unchanged within each matched panel. Measure arrivals by 64, 128, 256 decisions and episode end at 900 decisions; report failures, episode censoring, time to first hit, command-induced initial action change, target exposure, per-field sample size, trajectories and graph outcomes. A 900-decision trial that ends at the physical boundary without a hit is a failure, not a goal timeout. Hit-only option success is secondary because the controller no longer imposes an option deadline.

The canonical manifest-driven 10k-decision checkpoint evaluator supplies field maps and trajectories at the declared ages, with terminal replicas for seeds 8 and 123. Report active-only map cosine, silent units, spatial information, peak diversity, mono-field structure, pre-threshold maps and occupancy support together. Verify the four narrow fixed fields directly against position-aligned post-replacement samples because the standard coarse grid cannot certify their support. Analyze online goal exposure and graph development separately from frozen-policy arrival evidence.

## Implementation and release gates

Source branch `codex/episode-long-dg-20261008` commit `45e0e0d4` is pushed in `SF_hipposlam`; the isolated NEMO2 runtime is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_episode_long_dg_20261008`. The production StudySpec uses workflow `1.14.1`, schema `intrmotiv/study/v1`, SHA-256 `261fd73bab3689dc6c178e105795ae1f727160d70ff3bf26c5f6abe05052100b`. Its two short qualification StudySpecs preserve their distinct submission fingerprints: [four-arm original](../../hpc_runs/studies/episode_long_dg_controller_qualification_20261008.study.json) and [legacy correction](../../hpc_runs/studies/episode_long_dg_legacy9_qualification_20261008.study.json).

The code adds a direct episode-long option mode and a legacy decoder wrapper that scales only the command slice. The goal ID, hit signal, graph, DG activity and reward are unscaled. Default gain 1 leaves the legacy decoder untouched. The local and synchronized NEMO2 focused suites each passed 108 tests, including former-deadline crossing, wrong-field visits, hit completion, default behavior, goal masking, legacy decoder slice scaling and packed replay. A content-only synchronization check found no difference in changed files between the pushed source and isolated runtime.

Four-arm seed-99 qualification jobs 8306901–8306904 were submitted from the audited [original qualification spec](../../hpc_runs/studies/episode_long_dg_controller_qualification_20261008.study.json), 524,288 frames each. Job 8306902 stopped after 10 seconds at a deterministic config check: `goal-adapter reset requires hrl_goal_conditioning=target_id_film`. This flag is irrelevant in the monitor-only, zero-replacement experiment, but it must be false for the legacy decoder. The corrected production spec declares that flag explicitly in every arm: true for the three FiLM arms, false for legacy. All four production configurations pass the real argument parser. Corrected legacy qualification job 8306906 uses the separate one-arm spec with SHA-256 `f7589dc946686837f5479409ce82883efad359e8d0cab0de89052df4cec2a9e9`; its exact Slurm submission audit passed.

All four corrected short jobs exited 0 and reached their 524,288-frame milestone checkpoint. Their final online learner windows were:

| Arm | Train loss | Active target fraction | Target-hit numerator |
| --- | ---: | ---: | ---: |
| ORACLE_FILM | 0.199 | 0.307 | 0 |
| ORACLE_LEG9 | −0.028 | 0.236 | 0 |
| LEARNED4_FILM | 0.914 | 0.435 | 5 |
| C15_FILM | 0.445 | 0.847 | 17 |

All listed losses were finite. These are unmatched short windows; zero oracle hits do not test long-horizon control or represent a production outcome. Exact learner reload jobs 8306917–8306920 each exited 0 and wrote a certificate binding the copied checkpoint SHA-256 to restored model, optimizer and counters. Standard manifest-driven frozen place-field smoke job 8306921 loaded the corrected legacy checkpoint, completed 500 decisions, wrote the 78-array NPZ and exited 0; rate maps were finite on the 79 occupied grid cells. That short path sampled none of the four narrow fixed fields. A bounded matched-command smoke job, 8306922, remained active at production submission and is an evaluator check, not a performance result.

## Production release and next gates

The canonical print-only launcher and submission audit both matched the final 12-cell StudySpec. All twelve independent scripts request 60 hours on the CPU partition, whose maximum is four days; every bulk path resolves under `/work/classic/fr_xl1014-corridor-geometry`. The submitted Slurm manifest is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_episode_long_dg_controller_20261008/20261008T000131Z/jobs.tsv`. It passed `audit-submission --submitted` with unique job IDs 8306923–8306934. All twelve were RUNNING at the first scheduler check. The separate pre-existing 75M finite-horizon runs were not changed or resubmitted.

At 5M, verify all twelve checkpoints, four prescribed-field encounters, twelve active learned-context channels, goal exposure, finite losses and checkpoint integrity. Stop a failed arm rather than changing it in flight. At 75M, perform the planned paired horizon analysis with an updated source/checkpoint parity audit. At 150M, compare the four new arms and finish frozen place-field, trajectory, graph and matched-command analyses. The `episode-long-dg-study-follow-up` thread heartbeat checks progress every 30 minutes and stays quiet when there is no actionable change.

## Reusable lesson

Parsing every rendered arm through the actual training argument validator before a compute-node launch would have caught the FiLM-only flag in the legacy arm. Keep submitted qualification fingerprints immutable when correcting a failed cell; use a separately declared correction and connect both records to the final production spec. The 900-decision physical episode boundary is the appropriate episode-end evaluation horizon for this level; the separate long-episode variant must not be substituted when interpreting this study.
