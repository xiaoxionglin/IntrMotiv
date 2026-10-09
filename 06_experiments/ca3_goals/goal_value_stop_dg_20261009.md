# Goal-hit-terminal worker value study

**Status, 9 October 2026:** all eighteen production jobs 8316687–8316704 completed 100M frames. The canonical 95–100M online collection, all ninety declared spatial snapshots, and all eighteen 100M frozen 10k-decision evaluations are complete. No recurring monitor is active.

## Question and intervention

The continuing-episode goal-conditioned critic currently learns the discounted return from the current command through later commands until the physical episode ends. Thus a fast hit can have a lower value than a slower hit when the first location leads to less valuable subsequent goals. This study tests whether stopping the **worker value target and GAE** at the commanded goal hit improves goal-specific control while the environment, manager, graph, and recurrent state continue normally.

For a transition at decision $t$, the continuing control uses $d_t^{\mathrm{value}}=d_t^{\mathrm{environment}}$. The new arm uses

$$
d_t^{\mathrm{value}}=d_t^{\mathrm{environment}}\lor h_t^{\mathrm{commanded\ goal}},
$$

and supplies this mask only to the worker's GAE. The hit reward remains on the terminal transition. At a 64-decision rollout cutoff without a hit, GAE still bootstraps the value of the same command. The physical `dones` tensor, episode reset, goal recognition and reassignment, graph updates, and observation stream are unchanged. The new target is the expected discounted **current-option** return to hit or physical termination; the next command begins its own value segment.

Because these arms pay worker reward on a commanded hit, the intended goal-only value is approximately the discounted hit reward times the probability of hitting before physical termination, with hit latency included in the discount. An unresolved goal contributes no later-command reward to its value target. This is a critic and advantage-target intervention, not a physical episode-ending change. It tests a different learning objective for the worker; it does not make the graph or manager estimate only immediate goal value.

## Matched matrix

The [production StudySpec](../../hpc_runs/studies/goal_value_stop_dg_20261009.study.json) declares 18 new runs: prescribed DG, C15, and C05, each with nearest-DG and source-command hit-bonus clocks at seeds 8, 99, and 123. Every run uses 100M frames, $\gamma=0.999$, GAE $\lambda=0.99$, orthogonal FiLM, episode-long goals, 900-decision physical episodes, the existing hit reward and encoder objective, and the same checkpoint/snapshot ages of 5M, 25M, 50M, 75M, and 100M.

| Family | Reward clock | Continuing-value match | Goal-hit-terminal new arm |
| --- | --- | --- | --- |
| Prescribed: four fixed Gaussian goals plus 12 learned context units | Nearest DG | `long_credit_dg_20261008`: `oracle_film` | `oracle_nearest` |
| Prescribed | Source command | `source_distance_hit_dg_20261009`: `oracle_source` | `oracle_source` |
| C15: 16 learned goals, frontier manager | Nearest DG | `long_credit_dg_20261008`: `c15_film` | `c15_nearest` |
| C15 | Source command | `source_distance_hit_dg_20261009`: `c15_source` | `c15_source` |
| C05: 16 learned goals, visit-direct manager and its declared DG regularizers | Nearest DG | `source_distance_hit_dg_20261009`: `c05_nearest` | `c05_nearest` |
| C05 | Source command | `source_distance_hit_dg_20261009`: `c05_source` | `c05_source` |

Within every row, paired seeds and checkpoint ages isolate the value-target boundary. Within each family, the two new arms compare reward clocks under the same goal-hit-terminal value target. The crossed comparisons assess whether the reward-clock effect depends on the value boundary. Family comparisons remain descriptive because representation, manager, and DG regularizers differ.

## 100M result

The [95–100M per-run online table](results/goal_value_stop_dg_20261009/online_95_100m/per_run.csv) and complete [five-age spatial table](results/goal_value_stop_dg_20261009/spatial_through_100m/per_snapshot.csv) carry the immutable production StudySpec SHA-256 `0159bd2e8944a7f1c283822a23cdf0e720336beaf54607b3e60aa1d92069cf2f`. The [cross-study decision record](distance_reward_ablation_decision_20261009.md) gives the paired seed contrasts, field and trajectory analyses, graphs, and the distinction between online and frozen control evidence. Stopping the worker value target at the goal hit has not produced a replicated command-selective benefit in these 100M online results. In C15 it reduces logged hit rates for both reward clocks in all three paired seeds. The completed [100M frozen field results](results/goal_value_stop_dg_20261009/frozen_100m/derived_place_field_metrics.csv) show no replicated field-structure benefit. Executed-command tests remain necessary before making a positive control claim.

## Code and release checks

The isolated runtime at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_goal_value_stop_20261009` was copied from the final source-command study. The [delta patch](../../hpc_runs/patches/goal_value_stop_delta_20261009.patch), SHA-256 `ca73083d0d9909c19cd43cadeca6e911a23be718c170edac753a0459f8ef2983`, applies cleanly to that parent and adds `--hrl_value_termination=episode|goal_hit`, default `episode`. It changes only the worker GAE done mask when `goal_hit` is selected; V-trace and double-value modes are rejected for that setting. The parent source-command patch is SHA-256 `3317d1a262517fd451143eb1aaa9e13bae6062c03f34fc7adcb22a99844decd3`.

Focused reward/controller/summary tests passed **54/54** on NEMO2. A synthetic GAE test verifies that a goal hit stops later-command reward from propagating backward, a non-hit rollout cutoff still bootstraps, the next command's own segment is retained, and physical `dones` are not mutated. The real training parser accepted all six arms in production and qualification, with $\gamma=0.999$, $\lambda=0.99$, `with_vtrace=False`, `double_value=False`, and episode-long target expiration. A rendered-argument audit found no unplanned scientific difference from each row's continuing-value match: the new value flag is the only behavioral change, and explicit `nearest_dg` equals that baseline's default.

Production StudySpec schema `intrmotiv/study/v1`, workflow `1.14.1`, canonical SHA-256 `0159bd2e8944a7f1c283822a23cdf0e720336beaf54607b3e60aa1d92069cf2f`. The [qualification StudySpec](../../hpc_runs/studies/goal_value_stop_dg_qualification_20261009.study.json) declares six 1,048,576-frame seed-99 runs, canonical SHA-256 `b381343cf015efbd716095f6a0ce136f86f546b187b79eba85870e8e3475934e`. Its submitted audit matched all six commands and workspace-only paths in `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/goal_value_stop_dg_qualification_20261009/20261008T235015Z/jobs.tsv`. The 18 production scripts passed print-only audit with workspace-only paths and a 60-hour CPU limit.

All six qualification jobs completed exit-zero at the milestone, each with 34 finite policy/value/advantage logging points. Every run had a complete 100k-observation online spatial snapshot with all 16 DG units active. The fixed prescribed rows were each encountered; their counts ranged from 29 to 198 per 100k observations across the two prescribed arms. All six saved checkpoints passed exact model, optimizer and counter reload. Six ordinary frozen 500-decision place-field jobs completed exit-zero with finite occupancy and DG active fractions. The [qualification evidence](results/goal_value_stop_dg_20261009/qualification/) contains hit-weighted online summaries, field tables, frozen smoke summary and reload certificates.

Logged correct-hit counts were 5 and 2 in prescribed nearest/source, 89 and 146 in C15 nearest/source, and 202 and 236 in C05 nearest/source. These short, single-seed counts check reward exposure; they are **not** evidence of value-stop control benefit. Prescribed outcome remains inconclusive at this age. The [production submission audit](results/goal_value_stop_dg_20261009/production_submission_audit.json) matched all 18 exact commands and workspace paths under the immutable [jobs manifest](results/goal_value_stop_dg_20261009/production_jobs.tsv), submitted from `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/goal_value_stop_dg_20261009/20261009T000443Z/`. The jobs started under the 60-hour CPU limit. Do not tune arms during the main study.

## Analysis and decision rule

At paired ages, report hit counts and latency, goal exposure, coverage, value loss and advantage behavior. Complete 10k frozen place-field maps, segmented trajectories and occupancy, directed graph support and outcomes, and frozen matched-command interventions. For prescribed DG, the primary outcome is commanded-field **physical arrival lift** over alternative commands from identical starts at 64, 128, 256, and 900 decisions, with time to arrival, failures, and episode censoring. For C15/C05, report detector hits and counterfactual action changes, and name physical destinations only when their frozen field maps support that interpretation. Do not treat the episode-long graph's censored prospective-success fraction as control evidence.

An option-terminal benefit requires replicated paired-seed improvement in command-specific frozen control at 100M, supported by sufficient goal exposure. Earlier online hit counts or a lower value loss alone do not establish it. If prescribed-goal hits remain sparse or seed effects disagree, leave the mechanism unresolved. Report the three paired effects and uncertainty for each reward clock, then test the value-boundary by reward-clock interaction within each family.
