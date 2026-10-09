# Source-DG temporal hit-bonus study

**Status, 9 October 2026:** the source-command-time intervention passed compute-node qualification, exact checkpoint reload and frozen field smoke. Twelve 100M production jobs, 8316657–8316668, were submitted with exact StudySpec audit. Results beyond the short qualification are pending. No recurring monitor is active.

## Scientific question

The existing episode-long worker hit bonus uses the nearest preceding DG event, which may be an intervening wrong field or learned context unit rather than the source stored when the goal was assigned. This study tests whether making **only the worker hit-bonus distance** refer to elapsed time since that source command was assigned improves command-specific control. It retains the continuing-episode critic, physical episodes, and the separate DG encoder objective. It does not test option-terminal value learning.

For a commanded hit, the unchanged base and coefficient give

$$
r_{\mathrm{hit}}=1+0.01\max(71-d,0).
$$

The original $d$ is the age of the nearest preceding eligible DG event. In the source condition, $d$ is the controller's exact elapsed decision count from assigning the goal and stored source to hitting that goal, clipped to $71$ for the bonus. A missing source sets $d=71$ and pays the base reward alone. Intervening activations of **any** DG row, including the stored source row, cannot refresh this bonus. If the stored source was selected from an older CA3 trace rather than an active DG event at assignment, this measure starts at command assignment; it does not recover that earlier event's timestamp.

The source-clock bonus pays $1.70$ at a one-decision hit, $1.07$ at a 64-decision hit, and $1.00$ from decision $71$ onward. It therefore rewards shorter paths directly only inside that window. Discounting still favors earlier hits beyond the window if other rewards are equal; the continuing-episode critic can also assign value to what happens after the hit. This batch tests the clock correction, not whether that continuing value should be terminated at option completion.

## Matrix and comparisons

The [production StudySpec](../../hpc_runs/studies/source_distance_hit_dg_20261009.study.json) declares four arms, seeds 8, 99 and 123, 100M frames each, and milestones at 5M, 25M, 50M, 75M and 100M. All arms use orthogonal FiLM, $\gamma=0.999$, GAE $\lambda=0.99$, 64-decision PPO rollouts, episode-long commands, and 900-decision physical episodes. They use workspace-only paths and independent CPU jobs.

| Arm | Goal representation and manager | Worker bonus distance | Comparison |
| --- | --- | --- | --- |
| ORACLE_SOURCE | Four fixed Gaussian goals plus 12 learned context units; frontier-direct | Elapsed time from source-command assignment | Existing longer-credit ORACLE at paired seed and age |
| C15_SOURCE | Sixteen learned DG goals; frontier-direct | Elapsed time from source-command assignment | Existing longer-credit C15 at paired seed and age |
| C05_SOURCE | Sixteen learned DG goals; visit-direct, global punishment 0.01, row repulsion 1 | Elapsed time from source-command assignment | New C05_NEAREST at paired seed and age |
| C05_NEAREST | Same C05 architecture and longer-credit settings | Original nearest DG | New C05_SOURCE at paired seed and age |

C05_SOURCE versus C05_NEAREST isolates the distance rule within C05. The existing orthogonal C05 runs use $\gamma=0.99$ and default GAE $\lambda=0.95$; comparing them to either new C05 arm would combine reward-distance and credit-setting effects. Prescribed/C15 source arms pair with the six already running longer-credit controls; those baseline jobs remain unchanged. Cross-family C05–C15 comparisons jointly vary the manager and DG regularizers and are descriptive.

## Encoder reward and value scope

The DG encoder is trained separately from PPO. With `encoder_reward_require_local_predecessor=True` and `encoder_reward_recipient=arrival`, a dominant DG arrival receives positive credit only when the nearest earlier CA3 trace corresponds to a verified dominant DG event inside the same accepted 64-decision rollout. If that nearest trace fails verification, the code does not search for a farther eligible predecessor. The assigned credit is proportional to the verified lag. The predecessor need not be the manager's option source, and the arrival need not be the commanded goal. The encoder loss encourages activity of the **arrival row** for larger verified lags, whereas the worker bonus is larger for shorter lags; these are separate objectives with opposite local distance preferences. The always-enabled batch-usage objective also recruits unused learned rows. In ORACLE, the four Gaussian values replace projection outputs and cannot be changed by this encoder gradient; the 12 learned context rows can. The patch leaves this entire encoder pathway unchanged, so a worker-control result cannot be attributed to a deliberate encoder-reward change.

The goal-conditioned critic still predicts continuing-episode return, including rewards after a hit and new command. An option-terminal critic might better estimate the current goal's hit value, but changing both the bonus and critic now would obscure the source-distance test. Physical episodes are not ended at goal hits.

## Implementation and release gates

The versioned [runtime patch](../../hpc_runs/patches/source_distance_hit_bonus_20261009.patch) adds `--hrl_hit_bonus_distance=nearest_dg|source_dg`, defaulting to historical `nearest_dg`. It reads the completed old command's stored source and elapsed time before the hit-driven target reset. Hit recognition, base reward, graph, manager, DG activity, encoder credit, FiLM, and value-target termination remain unchanged. Three new per-hit summaries record source-command elapsed time, nearest-DG age, and the fraction where those distances differ, with correct-hit count as denominator context.

The production patch SHA-256 is `3317d1a262517fd451143eb1aaa9e13bae6062c03f34fc7adcb22a99844decd3`. It was applied only to `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_source_distance_dg_20261009`, copied from the existing orthogonal-FiLM runtime. The running source at `source_orthogonal_film_dg_20261008` was not edited. Focused NEMO2 reward and summary tests passed **53/53**. Unit coverage verifies intervening DG events cannot change source-command elapsed time, absent sources give no bonus, and non-hit transitions remain zero. The real training parser accepted all four arm configurations in both production and qualification at $\gamma=0.999$, $\lambda=0.99$, episode expiration and hit-distance reward. Rendered argument parity found no scientific difference from the existing longer-credit prescribed/C15 runs except the new source-distance flag. C05_NEAREST is the longer-credit C05 control under the unchanged distance rule.

Production StudySpec schema `intrmotiv/study/v1`, workflow `1.14.1`, canonical SHA-256 `9fd08a86aebf3c4e809b8d4b9ebf505708797b858b1982f099d152289d39400d`. The [final qualification StudySpec](../../hpc_runs/studies/source_distance_hit_dg_qualification_v3_20261009.study.json) declares one 1,048,576-frame run per arm, canonical SHA-256 `fc5cdb14064461d376bce4b3a7ab847f83301825692cf67fca492912231a3278`. The first four qualification jobs 8316224–8316227 were cancelled at zero frames because a code review found that simultaneous source activation could receive a zero-lag bonus. Their [submitted StudySpec](../../hpc_runs/studies/source_distance_hit_dg_qualification_20261009.study.json) and [runtime patch](../../hpc_runs/patches/source_distance_hit_bonus_v1_20261009.patch) remain preserved.

The second qualification, jobs 8316633–8316636, completed 1,048,576 frames per arm with finite losses, exact checkpoint reload in all four arms, 100k-observation online spatial snapshots, and four successful 500-decision frozen field jobs. It used a [preserved trace-age patch](../../hpc_runs/patches/source_distance_hit_bonus_v2_20261009.patch) and [immutable v2 StudySpec](../../hpc_runs/studies/source_distance_hit_dg_qualification_v2_20261009.study.json). Review of its telemetry exposed the same-source refresh flaw: at hits, source-trace age averaged much less than completed-option elapsed time in C15/C05. Those runs are diagnostic only and are not the final intervention. Their actual run root is `train_dir/source_distance_hit_dg_qualification_v2_20261009`; the submitted StudySpec's descriptive `output_root` mistakenly names `train_dir/source_distance_hit_dg_20261009_qualification_v2`. Both are in the workspace, but the mismatch is recorded in [infra.md](../../infra.md). The final v3 qualifier corrects this metadata. Its submitted audit matched four exact commands and workspace paths, and jobs 8316645–8316648 use manifest `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/source_distance_hit_dg_qualification_v3_20261009/20261008T232901Z/jobs.tsv`.

The v3 qualifier uses its preserved [runtime patch](../../hpc_runs/patches/source_distance_hit_bonus_v3_20261009.patch), SHA-256 `d199a690733d9c3b06ee3461c1fc74f8b0bc3a3c358183a1d5457373ebaa6bcf`. A subsequent code review found that its summary attribute still had the old `source_dg_distance` name while the tag map requested `source_command_distance`; the **production** patch changes only that summary attribute and the CLI help text. The reward computation and all training settings are identical. The 12 production scripts passed a fresh print-only audit against this final patch with workspace-only paths and a 60-hour CPU limit.

Jobs 8316645–8316648 completed 1,048,576 frames each with 34 finite policy/value/advantage logging points per arm, four exact checkpoint reloads, four complete 100k-observation online spatial snapshots, and four ordinary 500-decision frozen field smoke jobs. All 16 DG rows were active in each online snapshot. In ORACLE, the four fixed rows were observed 58, 19, 30 and 117 times per 100k observations; all 12 learned context rows were active. Prescribed-goal hit exposure was just two logged correct hits at this short age, so its reward/control outcome remains **inconclusive**, not a failed qualification. The [v3 qualification evidence](results/source_distance_hit_dg_20261009/qualification_v3/) preserves hit-weighted summaries, spatial tables, smoke summary and reload certificates.

Because the v3 summary producer used the old source-distance metric name, the immutable qualification StudySpec requests one unavailable diagnostic tag. The [online analysis-only spec](results/source_distance_hit_dg_20261009/qualification_v3/online_analysis_only.study.json) omits that tag and keeps the submitted run matrix unchanged; the production producer and tag map were cross-checked and use the same new name. The new `intrmotiv/hrl/control/correct_source_command_distance_mean` tag was then verified in a running C05-source production event file, with 72 scalar points. This telemetry-only repair does not affect the qualified reward rule. The [production submission audit](results/source_distance_hit_dg_20261009/production_v4_submission_audit.json) matched all 12 exact commands and workspace paths under canonical StudySpec SHA-256 `9fd08a86aebf3c4e809b8d4b9ebf505708797b858b1982f099d152289d39400d`, with a 60-hour CPU limit. The immutable [jobs manifest](results/source_distance_hit_dg_20261009/production_jobs.tsv) was submitted from `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/source_distance_hit_dg_20261009/20261008T234417Z/`.

The separately requested [goal-hit-terminal value study](goal_value_stop_dg_20261009.md) adds a matched value-target factor while preserving both reward clocks. It will use these continuing-value runs and the already running longer-credit prescribed/C15 runs as its paired controls.

The v2 qualification made the correction necessary. Its logged correct-hit batches give these hit-weighted diagnostic means; these are short, single-seed training observations, not control outcomes:

| v2 arm | Logged hits | Completed-option decisions | Stored-source CA3 trace age | Nearest-DG age |
| --- | ---: | ---: | ---: | ---: |
| ORACLE_SOURCE | 6 | 143.0 | 64.2 | 24.5 |
| C15_SOURCE | 126 | 176.2 | 41.4 | 15.6 |
| C05_SOURCE | 148 | 176.8 | 55.2 | 15.0 |
| C05_NEAREST | 166 | 148.1 | 47.5 | 11.2 |

Thus a stored-source **trace age** is often much younger than the age of the command, because that DG row may activate again. The final patch uses completed-option elapsed time, which cannot be refreshed by DG activity. The v2 arm hit counts and magnitudes must not be compared as if they tested the final intervention.

Before production, require each arm to reach its short milestone with finite policy/value losses, actual goal/source exposure, hit-reward timing where hits occur, an intact online spatial snapshot, and exact checkpoint reload. A rare-hit arm remains inconclusive for the reward hypothesis rather than a failed training qualification. Run ordinary manifest-driven frozen field smoke jobs on the saved checkpoints, then review the 12-run production print-only audit before submission.

## Planned analysis and decision rule

At paired ages, compare correct-hit counts and latencies, source versus nearest DG distance at hits, changed-bonus fraction, value/advantage behavior, command exposure and coverage. Complete the standard 10k frozen place-field maps, segmented trajectories, directed graphs and frozen matched-command interventions. For prescribed DG, the primary measure is commanded-field **physical arrival lift** versus alternative commands from identical starts at 64, 128, 256 and 900 decisions, including failures and episode censoring. For C15/C05, keep detector hits separate from physical destinations unless frozen maps justify the latter. Do not interpret the episode-long graph's censored prospective-success fraction as control evidence.

A source-only advantage requires replicated paired-seed improvement in command-specific arrival or detector control, supported by exposure and frozen-policy behavior; online hit counts alone are insufficient. Report seed effects and uncertainty. A null result with too few hits, scarce source traces, or inconsistent seed effects leaves the mechanism unresolved.
