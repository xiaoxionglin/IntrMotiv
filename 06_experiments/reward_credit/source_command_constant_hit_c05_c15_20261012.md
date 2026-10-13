# Source-command timing versus constant hit reward: C05 and C15

**Status, 13 October 2026:** Both seed-99 2M qualifications and both 500-decision frozen field smokes passed. The original six production jobs were externally cancelled before their first checkpoint, so their partial directories are excluded. Fresh production jobs now cover five paired seeds: ten constant-hit controls and four new source-clock arms for seeds 17 and 41. All 14 learner frame counters advanced after submission. No 100M result is available for the new runs.

## Question and matched intervention

Do the C05 and C15 learned-DG agents gain useful spatial representation or commanded-goal control from the *timing dependence* of the source-command hit bonus? The completed [source-clock study](source_distance_hit_dg_20261009.md) provides seeds 8, 99, and 123 at 100M for each arm. Four additional source-clock runs at seeds 17 and 41 are training for a five-seed paired comparison. Its hit magnitude is

$$
r_{\mathrm{source}}(d)=1+0.01\max(71-d,0),
$$

where $d$ is the elapsed decision count from assigning the stored source and command to hitting the commanded target. The new control uses a fixed positive magnitude at each correct hit. This keeps hit occurrence, the episode-long controller, $γ=0.999$, GAE $λ=0.99$, 900-decision physical episodes, architecture, manager, DG regularizers, and training duration matched within C05 and within C15. The source runtime's `hit` reward mode returns the configured constant; `hit_distance` adds the temporal term. Rendered argument comparison found only these two scientific changes, plus output logging paths and group names. C05 and C15 are separate architecture comparisons; comparing them directly would mix manager and regularization effects.

## Constant magnitude calibration

The constants are the count-weighted mean *total hit magnitude* of each source-clock arm in the mature 95–100M window, across seeds 8, 99, and 123:

$$
\bar r_a=\frac{\sum_s n_{a,s}\bar r_{a,s}}{\sum_s n_{a,s}}.
$$

Here $n_{a,s}$ counts logged learner-batch hit observations, not independent physical arrival trials. [Calibration inputs and formula](results/source_command_constant_c05_c15_20261012/calibration.json) retain the source table SHA-256 and per-seed counts.

| Arm | Count-weighted source hits | Fixed total hit magnitude |
| --- | ---: | ---: |
| C05 | 33,344 | 1.51073508574529 |
| C15 | 6,689 | 1.39379281430238 |

This matches late-stage mean reward exposure, not each hit or the early-training mean. A difference can therefore reflect timing contingency and different earlier reward magnitude. Compare hit counts and realized reward per hit alongside outcomes; avoid calling the contrast a pure timing effect at every age.

## Reproducible studies and release gate

The [qualification StudySpec](../../hpc_runs/studies/source_command_c05_c15_constant_hit_qualification_20261012.study.json) uses schema `intrmotiv/study/v1`, workflow `1.14.1`, SHA-256 `975606b177ef0093cd898da0f00fd2224cd8f6c770f73317552b11eadb088464`, and two 2M-frame seed-99 jobs. Its submitted launcher manifest is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/source_command_c05_c15_constant_hit_qualification_20261012/20261012T154703Z/jobs.tsv`; job IDs are 8350027 (C05) and 8350029 (C15). The canonical audit matched both commands, their IDs, and workspace paths. The isolated runtime source is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_source_command_constant_c05_c15_20261012`, copied from the completed source-clock study with no training-code edits. The canonical workflow tests passed 38/38 and focused controller tests passed 46/46 on NEMO2.

The [original three-seed production StudySpec](../../hpc_runs/studies/source_command_c05_c15_constant_hit_s3_20261012.study.json) declared six 100M controls with SHA-256 `6e1d6461544dc982e9c38a4d3c1a8164cc82fdc67402003a6a6d4117d1d093e7`. Its jobs 8350083–8350088 advanced only to 65,536–98,304 frames before simultaneous external cancellation, with no checkpoint or learner exception. The new run names and batch roots avoid their partial output directories. The [replacement constant-hit StudySpec](../../hpc_runs/studies/source_command_c05_c15_constant_hit_s5_recovery_20261013.study.json) has ten 100M runs at seeds 8, 99, 123, 17, and 41, SHA-256 `bc7f4a8b9ca2d6c93b00dbdb7e31fde8912b273fa275666e101a4e7374100c04`. The [source-clock extension StudySpec](../../hpc_runs/studies/source_command_c05_c15_source_clock_s17s41_20261013.study.json) has four 100M runs at seeds 17 and 41, SHA-256 `e537b9a1ec418ebf002cb7ac91be3acecd46d0c3d8cb78f76d1164bb76e87a1e`. Both use schema `intrmotiv/study/v1`, workflow `1.14.1`, and the 5M/25M/50M/75M/100M milestones. Rendered scientific arguments match their respective parent studies; only run identity and workspace logging paths changed.

Both qualification arms reached 2M with finite [policy and value losses](results/source_command_constant_c05_c15_20261012/qualification_online_1_2m/per_run.csv) and complete 100,000-observation [spatial snapshots](results/source_command_constant_c05_c15_20261012/qualification_spatial_2m/per_snapshot.csv) at 1M and 2M. Their 1M [C05](results/source_command_constant_c05_c15_20261012/c05_qualification_1m_reload.json) and [C15](results/source_command_constant_c05_c15_20261012/c15_qualification_1m_reload.json) milestone checkpoints have exact model/buffer, optimizer, and counter reload certificates from ordinary compute-node jobs 8350073 and 8350074. The count-weighted 1–2M reward histories contain 1,057 C05 and 311 C15 logged correct-hit observations; every positive magnitude equals the arm's constant to TensorBoard precision. These are repeated learner-batch observations rather than independent successful traversals. At 2M, the amplitude-weighted active-unit spatial scores are 0.04378 for C05 and 0.04262 for C15, with no silent units. The [qualification telemetry manifest](results/source_command_constant_c05_c15_20261012/qualification_telemetry/analysis_manifest.tsv) selected the 2M checkpoints for ordinary 500-decision frozen field smoke jobs 8350081 and 8350082. Both completed with exit code zero, loaded the declared checkpoints, and produced `place_fields.npz` files. The source-clock comparison is paired by seed and checkpoint age once all five seed pairs complete; its immutable StudySpec SHA-256 is `9fd08a86aebf3c4e809b8d4b9ebf505708797b858b1982f099d152289d39400d`.

## Five-seed production release

Both new studies passed local and synchronized NEMO2 validation, `render-runs`, launcher print-only review, and `audit-submission --submitted`. The canonical workflow tests passed 38/38 on the runtime source. The active workspace had 1.3 TB available at release. Each job requests 40 CPU cores, 80 GB, and 60 hours. New training output, W&B data, cache, Slurm logs, and temporary files resolve under `/work/classic/fr_xl1014-corridor-geometry`.

| Arm | Seeds | New job IDs | State at release check |
| --- | --- | --- | --- |
| C05 constant hit | 8, 99, 123, 17, 41 | 8369217–8369221 | All running; each advanced to at least 65,536 frames |
| C15 constant hit | 8, 99, 123, 17, 41 | 8369222–8369226 | All running; each advanced to at least 32,768 frames |
| C15 source clock | 17, 41 | 8369227–8369228 | Both running; each advanced to at least 32,768 frames |
| C05 source clock | 17, 41 | 8369229–8369230 | Both running; each advanced to at least 32,768 frames |

The [ten-run submitted manifest](results/source_command_constant_c05_c15_20261012/submitted_constant_s5_jobs.tsv) and [four-run submitted manifest](results/source_command_constant_c05_c15_20261012/submitted_source_s17s41_jobs.tsv) match the validated StudySpecs. Their run roots were absent before submission. This establishes release and initial learner progress, not the eventual training health or reward effect. The constant magnitudes remain fixed from the original three-seed 95–100M source-clock calibration; the two new source-clock seeds extend the paired outcome sample without recalibrating the intervention after seeing new outcomes.

## Preliminary seed-99 spatial readout at about 1M

The [constant-arm 1M spatial summary](results/source_command_constant_c05_c15_20261012/qualification_spatial_1m/per_snapshot.csv) and [source-clock qualification summary](results/source_distance_hit_dg_20261009/qualification_v3/spatial_per_snapshot.csv) give the following amplitude-weighted active-unit spatial scores:

| Architecture | Source clock at 1,048,576 frames | Constant hit at 1,015,808 frames | Constant minus source |
| --- | ---: | ---: | ---: |
| C05, seed 99 | 0.01158 | 0.00816 | -0.00342 |
| C15, seed 99 | 0.01513 | 0.01603 | +0.00090 |

All four snapshots have 100,000 valid observations and no silent units. This is a single-seed, slightly unmatched-age qualification comparison; it shows no common direction across C05 and C15 and cannot answer the 100M question.

## Analysis when new checkpoints are ready

The primary spatial comparison is paired-seed active-unit spatial information at identical ages. The canonical score is amplitude-weighted, not normalized bits per activation, so report unit activity and coverage alongside it. Report silent fraction, active-only map cosine, peak diversity, pre-threshold and thresholded fields, and occupancy support. The completed source-clock 100M [online-spatial table](results/source_distance_hit_dg_20261009/spatial_through_100m/per_snapshot.csv) provides C05 source scores 0.0758, 0.1909, and 0.1172 for seeds 8, 99, and 123; C15 source is 0.1350, 0.0775, and 0.1020. All six source snapshots have silent fraction 0. These are reference values, not constant-arm differences.

Complete the standard frozen 10k-decision place fields, segmented trajectories, directed graphs, and matched-start executed-command interventions for the ten new constant arms and the four source-clock extension arms. Compare source and constant at paired seeds and ages; keep detector hits separate from physical arrivals and state failures and censoring. Online coverage, hit rates, and spatial snapshots alone are interim evidence. The final question is whether any spatial-information difference aligns with replicated commanded-goal behavior, rather than whether one reward produces more hits in its own training rollouts.
