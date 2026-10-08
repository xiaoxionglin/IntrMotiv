# Odor and CA3 goal quality: corrected 40-run launch

**Status, 8 October 2026:** all 40 replacement runs have their declared 5M/10M/25M/50M/75M checkpoints. The full 40-run coverage histories, all 70 declared 10k-decision place-field rollouts, and all 200 online spatial/graph snapshots have been analyzed. See the [place-field, trajectory, and graph report](odor_goal_quality_v2_spatial_analysis_20261007.md). All 40 ordinary target-hit and 40 first-distinct evaluations completed; the [command analysis](odor_goal_quality_v2_command_interventions_20261007.md) records their stopping-rule, retrospective-label, and variable-deadline limitations. The first full-panel exact-start qualification timed out after eight hours; its bounded four-source replacement passed, and the remaining exact-start jobs await final plan audit.

## Why the first launch was replaced

The original [study](odor_goal_quality_40_run_plan_20261006.md) passed its first runtime gate, but follow-up review found four implementation problems. Its CA3 event update checked the acceptance flag of the action after the event-producing action. Candidate-subset normalization changed the magnitude and ordering of C15 frontier scores. Candidate and odor draws could not be reconstructed from a fixed stored rollout. The manager's actual candidate sets were absent from training telemetry. All 40 original production jobs, 8290663–8290715, were cancelled from their audited submitted manifest; their partial trajectories are excluded from the replacement study.

## Corrected contract

- The learner pairs a new exclusive DG event with the preceding detached CA3 state. The update does not use the action identity. It processes each accepted rollout once and updates the corresponding prototype and quality rows for its qualifying events. The fixed update rate remains $0.01$ from start to finish in every condition, including ALL16 and RANDOM.
- Candidate selection masks the **unchanged globally computed C15 frontier score**. RANDOM chooses up to $K$ eligible IDs uniformly; HEBB uses the fixed quality score with uniform exact ties. All 16 DG identities remain eligible over time. A key from actor seed, persistent manager-choice count, and detached CA3 context reproduces candidate sets on a fixed stored rollout without consuming PyTorch's global random stream.
- Odor noise is keyed by environment seed and observation index. OFF is zero; ON is four [fixed Gaussian spatial fields](figures/odor_clean_fields_v2_20261007.png) with independent observation noise of standard deviation $0.15$ of the clean peak, without clipping. The field centers and width are identical across ON runs; the observed values vary with position and keyed noise. The gain of 42.68481611601036 is shared across cells and comes from the separate [fixed 10k-decision calibration](odor_gain_calibration_20261007.json). Only DG receives odor.
- Actor outputs store the behavior-time candidate mask and choice flag. The learner logs candidate count, distinct IDs, per-ID exposure, empty pools, and successive-choice turnover from accepted rollouts. PPO replay uses its stored behavior goal condition and cannot update the CA3 quality buffers.

The fixed-rollout tests establish exact restoration of CA3 buffers and candidate draws. A process restart can begin a different DMLab episode because the live environment state is not part of the Sample Factory checkpoint; it therefore does not promise a bitwise-identical future training trajectory.

**Rollout-boundary interpretation and decision, 7 October 2026.** The accepted-only rule needs the learner's validity flag for the transition between the preceding CA3 state and the observed DG event. For the first event observed in a rollout, that flag belongs to the previous rollout, so v2 conservatively excludes the event. This is a missing acceptance-link record, not a need to identify which action produced the observation or to assign causal credit to that action. At most one transition per 64-decision rollout can be excluded this way. The current logs do not count excluded events, so their actual fraction is unknown; an approximately $1/64$ event fraction would require events to be spread evenly across rollout positions. In the four first-tranche cells at about 4.4M frames, every DG identity already had accepted CA3 events: the lowest per-goal counts were 369 in ON/HEBB8, 900 in ON/RANDOM4, and more than 5,000 in each OFF cell. Thus the present evidence shows relative frequency imbalance but no goal with zero support. Learning from every observed transition would remove this bookkeeping omission, but would change the declared accepted-only rule. The user chose to document the limitation and let the unchanged 40-run batch continue.

## Source, study, and qualification

The runtime source is isolated on branch `codex/odor-ca3-goal-quality-v2-20261007`, commit `73730b115dcc373767efffdeb77add2968242bed`, pushed and remote hash verified, and synchronized to `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_odor_ca3_v2_20261007/`. The corrected [workflow-1.14 StudySpec](../../hpc_runs/studies/odor_ca3_goal_quality_v2_20261007.study.json) has SHA-256 `c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de`, 40 unique cells, seeds 8/99/123/2026, a 75M-frame horizon, and milestones at 5/10/25/50/75M. Its full print-only manifest at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_odor_ca3_goal_quality_v2_20261007/20261006T232811Z/jobs.tsv` passed the canonical command and workspace-path audit. Local and synchronized NEMO2 focused suites each passed 63 tests.

Qualification jobs 8290765–8290768 covered OFF/RANDOM4, OFF/HEBB8, ON/HEBB4, and ON/RANDOM8. All completed with exit code zero and reached scalar step 540,672, beyond the 524,288-frame target. Every cell supported all 16 DG goals and had no nonfinite scalar samples. OFF odor norms were zero; ON norms were 27.97 and 28.75 at the final step. Candidate-set means were exactly 4 or 8 as configured, with no empty candidate sets. The final update exposed 16 distinct candidates in both RANDOM cells, 9 in OFF/HEBB8, and 5 in ON/HEBB4. These are health and exposure checks, not performance comparisons.

## Production release

The four first-tranche cells were OFF/ALL16, OFF/HEBB4, ON/RANDOM4, and ON/HEBB8, all seed 8. Their print-only manifest at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_odor_ca3_goal_quality_v2_20261007/20261006T233434Z/jobs.tsv` matched the corresponding commands in the audited 40-cell manifest exactly. Their submitted manifest is the sibling `20261006T233522Z/jobs.tsv`, with Slurm IDs 8290776–8290779.

The early health check covered scalar steps 557,056–589,824. All four learner frame counters advanced, and no recorded scalar was nonfinite. Accepted CA3 quality support covered 14/16 goals in OFF/ALL16 and 16/16 in the other three cells. Their latest candidate exposure covered 16, 11, 16, and 10 distinct goals, respectively; there were no empty candidate pools at that update. OFF odor norms were zero; ON norms were 22.21 and 20.04. Frontier score means were finite, between 1.99 and 2.25. No performance threshold selected cells.

The remaining 36-cell print-only manifest at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_odor_ca3_goal_quality_v2_20261007/20261006T233901Z/jobs.tsv` exactly partitioned the 40 audited commands with the first four. Its submitted sibling is `20261006T234219Z/jobs.tsv`. The combined submitted manifest is at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_odor_ca3_goal_quality_v2_20261007/40_cells_submitted.tsv`, with its canonical audit beside it as `40_cells_submitted_audit.json`. These contain 40 unique submitted Slurm IDs, 8290776–8290827. The audit confirms exact command matches, workspace-only paths, workflow `1.14.0`, and unchanged canonical study SHA-256 `c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de`. All 40 jobs were `RUNNING` at the immediate scheduler check.

## Available milestone evaluation

Before the 75M field and intervention checkpoints existed, the [thin partial renderer](render_odor_v2_partial.py) projected only the available 5M/10M/25M trajectory targets from the **unchanged** StudySpec, using its run expansion, the established checkpoint selector, and canonical manifest writer. It produced 30 rows: ten conditions at each of three targets, all seed 99. Actual checkpoint frames were 5,013,504, 10,027,008, and 25,001,984, at most 27,008 frames from target. The partial provenance records the original study SHA-256. The manifest and provenance are at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/odor_ca3_goal_quality_v2_pf_5_10_25m_20261007/`.

The first render exposed a workflow discovery collision: each launcher run also has an online-spatial artifact directory with the exact run name. Workflow 1.14.1 excludes directories below `analysis/` from run discovery, while still rejecting duplicate actual runs. The original StudySpec and training commands did not change. All 38 canonical tests passed locally and in the synchronized NEMO2 source; the runtime branch commit `b085cefa3605f0483413f670835d83036454fe5b` was pushed and its remote hash verified.

The print-only preflight command used ordinary Slurm job 8291118 for ON/HEBB8 seed 99 at 25M. It exited `0:0`, loaded the checkpoint, and produced a readable place-field NPZ with 257 observations from a 256-decision probe. The 30 production commands were then print-only reviewed for exact manifest rows, 10k decisions, and workspace paths. Their ordinary jobs, 8291121–8291150, all completed with exit code `0:0` and wrote 30 readable 10k-decision NPZs. The submitted record is `production_10k/submission_manifest_20261007T055409Z.tsv` beneath the same analysis root; it matches all 30 reviewed command lines exactly. The documented postprocessors wrote `production_10k/summary/place_field_summary.csv` and `derived_place_field_metrics.csv`, each with 30 data rows, plus `per_unit_place_field_metrics.csv` with 480 unit rows. These are 5M/10M/25M, seed-99 results; no scientific comparison has yet been reported.

All 40 runs now have each declared 5M/10M/25M/50M/75M milestone. The canonical full `render-telemetry` manifest is at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/odor_ca3_goal_quality_v2_full_20261007/`: 70 place-field rows, 50 seed-99 trajectory rows, and 40 terminal intervention rows. Its first 30 place-field rows match the completed partial manifest exactly by label and checkpoint. The remaining 40 rows cover seed 99 at 50M/75M and seeds 8/123 at 75M. Their print-only plan passed exact-row, 10k-decision, ordinary-job, and workspace-path checks; the `remaining_10k/submission_manifest_20261007T113848Z.tsv` commands match the reviewed plan exactly. All 40 remaining field jobs completed; their NPZs and the first 30 were joined by exact manifest label and postprocessed into the [complete spatial report](odor_goal_quality_v2_spatial_analysis_20261007.md). The 40 target-hit intervention commands likewise passed print-only review for 75M checkpoints, 100k-decision caps, canonical runner, and workspace paths; `interventions/submission_manifest_20261007T114025Z.tsv` matches the reviewed plan exactly. All 40 completed successfully.

## Interim analysis: complete training histories

The canonical `collect-online` workflow loaded all 40 replacement runs over a fixed 70–75M-frame terminal window and exported the selected event histories with workflow 1.14.1 and the unchanged study SHA-256. Its workspace output is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/odor_ca3_goal_quality_v2_online_20261007/`. The [coverage analysis script](analyze_odor_v2_coverage.py) consumes the exported external coverage-AUC tag named in the StudySpec, retains repeated event steps, averages observations within each 5M-frame bin for each run, then pairs the four declared seeds. The early window is 0–10M, the terminal window is 70–75M, and the full-course value equally weights the fifteen 5M bins. This metric is the external *episode coverage AUC* logged during training; a high initial value is not a high area under the training learning curve. The [figure](results/odor_v2_coverage_20261007/coverage_learning_curves.png), [per-run windows](results/odor_v2_coverage_20261007/coverage_windows.csv), [all paired contrasts](results/odor_v2_coverage_20261007/coverage_contrasts.csv), [canonical terminal table](results/odor_v2_coverage_20261007/canonical_terminal_per_run.csv), and [collector provenance](results/odor_v2_coverage_20261007/canonical_online_manifest.json) are saved here; the large raw histories remain in the workspace.

| Odor | Goal set | Early 0–10M | Terminal 70–75M |
| --- | --- | ---: | ---: |
| OFF | ALL16 | 79.3 | 51.6 |
| OFF | RANDOM4 | 72.4 | 42.0 |
| OFF | HEBB4 | 63.5 | 35.2 |
| OFF | RANDOM8 | 72.8 | 38.6 |
| OFF | HEBB8 | 62.9 | 42.5 |
| ON | ALL16 | 72.1 | 44.0 |
| ON | RANDOM4 | 75.0 | 34.6 |
| ON | HEBB4 | 76.8 | 24.6 |
| ON | RANDOM8 | 76.4 | 40.8 |
| ON | HEBB8 | 74.6 | 38.7 |

Values are means across the four paired seeds. Every condition declined from the early window to the terminal window. The weakest terminal mean was ON/HEBB4 at 24.6; OFF/ALL16 was highest at 51.6. The condition trajectories and seed variation are visible in the figure and per-run CSV.

| Paired contrast (first minus second) | Early difference | Terminal difference, 95% interval |
| --- | ---: | ---: |
| HEBB4–RANDOM4; OFF | -8.9 | -6.8 [-17.3, +3.7] |
| HEBB8–RANDOM8; OFF | -10.0 | +3.9 [-13.4, +21.2] |
| HEBB4–RANDOM4; ON | +1.8 | -9.9 [-26.4, +6.5] |
| HEBB8–RANDOM8; ON | -1.7 | -2.1 [-11.4, +7.3] |
| RANDOM4–ALL16; OFF | -6.9 | -9.6 [-39.0, +19.9] |
| RANDOM8–ALL16; OFF | -6.4 | -12.9 [-36.2, +10.4] |
| RANDOM4–ALL16; ON | +2.9 | -9.5 [-26.0, +7.0] |
| RANDOM8–ALL16; ON | +4.3 | -3.2 [-14.2, +7.7] |
| RANDOM4–RANDOM8; OFF | -0.4 | +3.4 [-22.3, +29.0] |
| HEBB4–HEBB8; OFF | +0.7 | -7.3 [-33.5, +18.9] |
| RANDOM4–RANDOM8; ON | -1.4 | -6.2 [-20.3, +7.8] |
| HEBB4–HEBB8; ON | +2.2 | -14.1 [-28.1, -0.1] |
| ON–OFF; ALL16 | -7.2 | -7.5 [-21.6, +6.6] |
| ON–OFF; RANDOM4 | +2.6 | -7.4 [-41.4, +26.6] |
| ON–OFF; HEBB4 | +13.3 | -10.6 [-30.9, +9.7] |
| ON–OFF; RANDOM8 | +3.5 | +2.2 [-15.3, +19.6] |
| ON–OFF; HEBB8 | +11.8 | -3.8 [-21.3, +13.8] |

The intervals use a paired $t$ calculation across four seeds and are unadjusted for 17 comparisons. Most contain zero. The ON HEBB4–HEBB8 terminal interval barely excludes zero before adjustment, so it is a lead for inspection rather than a firm selector effect. Taken together, these curves do not show a consistent coverage advantage for the fixed Hebbian selector or odor. This conclusion concerns coverage; it does not establish whether target commands causally control navigation.

At 70–75M, every run reports support for all 16 CA3 quality rows, zero DG silent fraction in the online metric, and zero behavior-replay mismatch. OFF odor norms are zero; ON condition means are 24.8–28.9 after the shared DG gain. Across the 40 runs, the online target-active fraction averages 2.215%, compared with 2.218% for a shuffled target; the per-run difference averages -0.003 percentage points. The categorical action-probability total variation under alternate goal IDs averages 0.0013. The target-active diagnostic is a per-step DG activation measure, not an alternative-command trial outcome. Its near-equality with the shuffled control raises a weak-conditioning concern that the matched-start interventions must resolve.

## Complete place-field, trajectory, and graph analysis

The [complete spatial analysis](odor_goal_quality_v2_spatial_analysis_20261007.md) reports all 70 independent 10k-decision field rollouts and all 200 retained online spatial/graph snapshots. No evaluated DG unit was silent. At 75M, active peak bins averaged 15.4 of 16, but only 5.2% of eligible unit maps met the established mono-field criterion; many maps contain multiple hotspots. The seed-99 checkpoint trajectories and map-stability tables are descriptive because the learned policy and occupancy differ between checkpoints. Across four paired seeds under both odor settings, HEBB4's reliable graph had fewer edges and lower ordered-pair reachability than RANDOM4, despite more prospective attempts. The [field curves](results/odor_v2_spatial_20261007/place_field_checkpoint_trajectory.png), [graph curves](results/odor_v2_spatial_20261007/graph_reachability.png), selected field/trajectory/graph atlases, tables, and provenance are linked from that report. These observational graph measures do not establish causal command success.

## Command interventions and remaining analysis

The [command report](odor_goal_quality_v2_command_interventions_20261007.md) covers 40 completed target-hit and 40 completed first-distinct jobs, each with 48,000 trials. **Eventual requested-goal hit within the deadline is the primary command outcome**; other DG events may occur first. HEBB4 has higher ordinary target-hit rates than RANDOM4 under both odor settings, but its graph-derived mean deadline is 7.9 decisions longer OFF and 9.4 longer ON. The comparison with context-matched shuffled *labels* is biased by target-hit stopping and does not execute another command. In the separately fingerprinted first-distinct protocol, commanded identity was the first new DG outcome in 6.75% of trials, versus 6.57% for the retrospective shuffled label; this is a secondary diagnostic and cannot negate later goal hits. A third [evaluation-only exact-start StudySpec](../../hpc_runs/studies/odor_ca3_goal_quality_v2_20261008_exact_start_analysis.study.json) preserves all 40 production run definitions; its single full-panel qualification timed out after eight hours without a summary. The bounded four-source replacement completed successfully with exact starts and 180 paired goal comparisons. The remaining source shards and cells await final plan audit and submission. The primary causal endpoint is paired eventual goal-hit lift under *executed* alternate commands from identical starts.

## Reusable lesson

Unit tests for a stateful controller must verify the actor's buffer timing, not just a locally chosen tensor convention. Keep the C15 score as one shared function when comparing candidate sets, and store behavior-time candidate evidence because commanded targets alone cannot recover exposure. The canonical StudySpec, print-only manifest, and submitted-manifest audit remained the source of truth for the restart; no run matrix was reconstructed from names. For repeated learning-curve windows, `collect-online --export-histories --loader-backend process` read each run's TensorBoard events once, then the lightweight exported histories supplied the 5M bins and paired windows; this avoided another full event scan. Concurrent field jobs overwrote their shared raw `summary.csv`, but their separate NPZs were complete and the documented postprocessors recovered the full 30-row table. Future field sweeps should trust the manifest plus NPZ inventory and aggregate only after the jobs finish.
