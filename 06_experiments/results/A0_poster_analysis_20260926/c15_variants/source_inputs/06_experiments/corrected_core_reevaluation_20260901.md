# Corrected-Core Historical Design Re-evaluation

**Outcome status, 2026-09-02:** all 48 training runs, final-10M behavioral
analysis, and standard spatial telemetry are complete. The focused 77-probe
candidate sweep completed as Slurm array `7975099` (77/77 tasks, exit `0:0`),
and its manifest-driven summaries, trajectories, and stability analyses are
available under the allocated NEMO2 workspace.

## Purpose

This batch re-evaluates the meaningful prior IntrMotiv designs after repairing
core DG/CA3 learning semantics. Earlier results are useful historical evidence,
but they do not isolate architecture quality from the old activation and reward
logic.

The repaired common core now:

- preserves the fixed ImageNet-pretrained ResNet-18 layer-2 weights;
- assigns feedback to the dominant current DG onset;
- applies the multi-activation penalty to simultaneous non-dominant onsets;
- treats an empty CA3 history as unknown rather than DG unit 0;
- recruits units unused in valid learner data with a pre-threshold softplus loss;
- restores unit-norm DG projection rows after optimizer steps;
- uses the configured graph half-life and the most recently active DG fallback;
- rejects the invalid decoder auxiliary and `distance_metric` PBT configurations.

The remaining actor/learner BatchNorm-state mismatch is intentionally deferred.
New behavior-event and learner-activity metrics make that boundary visible, but
the two rates have different event definitions and are not a direct mismatch
estimate.

## Shared Configuration

- W&B project: `SF_IntrMotiv_CorrectedCoreReevaluation`
- production group: `intrmotiv_corrected_core_reevaluation_20260901_production`
- seeds: `8`, `99`, `123`
- 100M environment frames per job
- fixed 900-decision no-reward open-field episodes
- fixed ImageNet-pretrained `layer2_resnet18`
- `F=16`, `R=8`, `L=64`, threshold `2.43`
- `encourage`, learner batch recruitment, and corrected multi-activation loss
- one policy, no PBT, 32 workers x 2 environments, CPU execution
- 40 CPUs, 80 GB, 30-hour Slurm request
- all bulk outputs under `/work/classic/fr_xl1014-train`

## Cells

| Cell | Condition                                                            |
| ---- | -------------------------------------------------------------------- |
| C01  | Corrected flat temporal-distance anchor                              |
| C02  | Direct global HRL, delayed target, short deadline                    |
| C03  | Direct global HRL, immediate target, short deadline                  |
| C04  | C03 with iterative encoder/decoder updates                           |
| C05  | C03 with global punishment `0.01` and row repulsion `1.0`            |
| C06  | C05 with CA3 temporal exclusion `1.0`                                |
| C07  | C03 with orthogonal recruitment                                      |
| C08  | C03 with CA3 temporal exclusion and orthogonal recruitment           |
| C09  | C03 with empirical PPO-HER, horizon 16, coefficient `0.5`            |
| C10  | C03 with empirical PPO-HER, horizon 64, coefficient `0.5`            |
| C11  | C02 with empirical PPO-HER, horizon 64, coefficient `0.5`            |
| C12  | Delayed direct HRL, CA3 exclusion and recruitment, long deadline     |
| C13  | C12 with manager exploration probability `0.10` and timeout recovery |
| C14  | Passive topology with visit-based direct frontier selection          |
| C15  | Passive topology with UCB direct frontier selection                  |
| C16  | Passive topology with UCB frontier selection and waypoint planning   |

Each cell has all three seeds, producing 48 jobs. The flat cell is a single
matched anchor rather than a second broad flat sweep.

## Planned Contrasts

1. C02 vs C03: delayed versus immediate target timing.
2. C03 vs C04: simultaneous versus iterative updates.
3. C03 vs C05-C08: structural anti-collapse mechanisms.
4. C03/C02 vs C09-C11: empirical PPO-HER horizon and timing.
5. C12 vs C13: manager exploration and timeout recovery.
6. C14 vs C15: visit novelty versus UCB frontier scoring.
7. C15 vs C16: direct final-target conditioning versus explicit waypoints.
8. HRL cells vs C01: goal-conditioned behavior versus the corrected flat
   temporal-distance anchor.

Primary evidence is matched-seed coverage AUC and unique cells together with DG
representation health, worker reward density, option success, target-hit lift,
and topology mechanism metrics. A coverage difference without mechanism health
is not enough to support an architectural conclusion.

## Verification Record

The full IntrMotiv test suite passed: `121 passed`. All 48 generated commands
also passed the real DMLab argument parser.

Four 2M-frame, seed-99 preflights completed across multiple physical episodes:

| Cell | Slurm job | FPS | Key mechanism checks |
| --- | ---: | ---: | --- |
| C01 | 7967828 | 1786 | corrected flat reward and DG diagnostics nonzero |
| C03 | 7967829 | 1760 | active targets 86%, hit rate 0.67%, option success 15.1% |
| C10 | 7967830 | 1743 | about 31 accepted HER segments per rollout |
| C16 | 7967831 | 1655 | passive updates, validations, routes, and waypoints all nonzero |

C16 used genuine multi-hop navigation on about 1.07% of transitions; its
waypoint-step hit rate averaged 1.60%, and selected routes reached about 3.5
hops at maximum. This is sparse, but it establishes that the full mechanism is
executed before the 100M-frame comparison.

The short preflights do not establish performance. In particular, mean
target-hit lift remained below one in the direct and HER checks. The production
batch is intended to test whether target-conditioned control emerges with
sufficient training, not to assume that it already has.

Preflight manifest:
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_corrected_core_reevaluation_20260901_preflight/20260901T173032Z`

Print-only manifest audited before submission:
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_corrected_core_reevaluation_20260901/20260901T203845Z`

Submitted production manifest:
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_corrected_core_reevaluation_20260901/20260901T204112Z`

The submitted job IDs are `7969086` through `7969126` and `7969130`
through `7969136` (48 jobs total). All 48 entered `RUNNING` state after
submission. The pre-existing empirical-HER jobs were not interrupted.

## Final Analysis

### Completion and analysis window

All 48 jobs reached 100,040,704 environment frames and produced terminal
checkpoints. The quantitative table below uses, for each run, the mean of every
available scalar sample in its final 10M frames. Each cell is therefore a
three-seed mean +/- sample SD over seeds 8, 99, and 123. Coverage metrics come
from matched 900-decision, no-reward physical episodes; learner diagnostics
come from valid PPO transitions and use their metric-specific denominators.

The table is descriptive rather than a powered significance analysis. Planned
contrasts were also calculated as within-seed differences, and the main claims
below report whether their direction was consistent across all three seeds.

| Cell |      Coverage AUC |       Unique cells |        DG density | Target-hit lift | Action sensitivity |  Option success |     Known edges |
| ---- | ----------------: | -----------------: | ----------------: | --------------: | -----------------: | --------------: | --------------: |
| C01  |      38.0 +/- 5.6 |       47.1 +/- 5.6 | 0.0263 +/- 0.0011 |              -- |                 -- |              -- |              -- |
| C02  |      45.1 +/- 8.0 |       63.6 +/- 5.1 | 0.0255 +/- 0.0029 | 0.959 +/- 0.070 |  0.0303 +/- 0.0217 | 0.277 +/- 0.042 | 0.180 +/- 0.046 |
| C03  |      23.2 +/- 8.7 |      27.9 +/- 12.6 | 0.0272 +/- 0.0031 | 1.003 +/- 0.023 |  0.0125 +/- 0.0075 | 0.453 +/- 0.209 | 0.147 +/- 0.004 |
| C04  |      39.6 +/- 6.6 |      53.6 +/- 12.0 | 0.0265 +/- 0.0008 | 1.033 +/- 0.111 |  0.0204 +/- 0.0141 | 0.193 +/- 0.060 | 0.153 +/- 0.013 |
| C05  |      42.0 +/- 2.0 |       53.3 +/- 5.8 | 0.0249 +/- 0.0006 | 1.091 +/- 0.051 |  0.0256 +/- 0.0191 | 0.317 +/- 0.081 | 0.169 +/- 0.006 |
| C06  |      37.8 +/- 8.4 |      49.1 +/- 11.7 | 0.0214 +/- 0.0015 | 1.147 +/- 0.112 |  0.0124 +/- 0.0023 | 0.196 +/- 0.048 | 0.175 +/- 0.014 |
| C07  |      36.0 +/- 5.3 |       45.9 +/- 8.1 | 0.0255 +/- 0.0014 | 1.120 +/- 0.132 |  0.0224 +/- 0.0043 | 0.313 +/- 0.063 | 0.178 +/- 0.006 |
| C08  |     47.5 +/- 27.3 |      64.1 +/- 44.4 | 0.0225 +/- 0.0010 | 0.984 +/- 0.032 |  0.0105 +/- 0.0079 | 0.222 +/- 0.114 | 0.175 +/- 0.031 |
| C09  |      36.7 +/- 4.1 |       48.7 +/- 8.1 | 0.0245 +/- 0.0015 | 1.036 +/- 0.112 |  0.0232 +/- 0.0090 | 0.251 +/- 0.103 | 0.167 +/- 0.023 |
| C10  |      38.2 +/- 9.9 |      46.7 +/- 14.2 | 0.0228 +/- 0.0020 | 0.979 +/- 0.035 |  0.0200 +/- 0.0101 | 0.244 +/- 0.042 | 0.152 +/- 0.031 |
| C11  |      30.7 +/- 7.6 |      38.7 +/- 11.5 | 0.0264 +/- 0.0035 | 1.032 +/- 0.048 |  0.0251 +/- 0.0094 | 0.326 +/- 0.015 | 0.160 +/- 0.018 |
| C12  |     48.1 +/- 11.4 |      67.4 +/- 23.3 | 0.0259 +/- 0.0010 | 0.918 +/- 0.012 |  0.0750 +/- 0.0591 | 0.759 +/- 0.129 | 0.168 +/- 0.024 |
| C13  |     38.5 +/- 11.3 |      48.5 +/- 15.3 | 0.0238 +/- 0.0024 | 0.959 +/- 0.018 |  0.0073 +/- 0.0025 | 0.640 +/- 0.154 | 0.107 +/- 0.031 |
| C14  |     48.1 +/- 10.1 |      67.9 +/- 16.8 | 0.0269 +/- 0.0011 | 0.928 +/- 0.024 |  0.0030 +/- 0.0006 | 0.399 +/- 0.049 | 0.957 +/- 0.020 |
| C15  | **78.6 +/- 13.8** | **128.1 +/- 23.8** | 0.0279 +/- 0.0017 | 0.898 +/- 0.005 |  0.0019 +/- 0.0006 | 0.388 +/- 0.008 | 0.992 +/- 0.001 |
| C16  |      41.6 +/- 2.4 |       51.4 +/- 5.7 | 0.0245 +/- 0.0003 | 1.000 +/- 0.034 |  0.0085 +/- 0.0025 | 0.428 +/- 0.023 | 0.866 +/- 0.085 |

Target-hit lift is the current target's DG activation rate divided by the
activation rate for a one-position-shifted target in the same minibatch. One is
the shuffled-target reference. Action sensitivity measures the change in
action probabilities under alternative targets; it measures conditioning, not
whether the change is useful.

### Main results

1. **The repaired core avoids online DG population collapse.** Mean terminal
   DG density lies between 0.021 and 0.028 in every cell. The mean minibatch
   silent-unit fraction is zero in 15 cells and 0.00095 in C06; usage entropy
   ranges from 0.960 to 0.978. This establishes active population use but not
   spatial selectivity, which requires the offline maps below.

2. **C15 gives the strongest and most replicated external coverage.** Relative
   to C14, UCB frontier selection raises coverage AUC by 30.4 and unique cells
   by 60.2 on average. Both differences are positive in every paired seed.
   C15 reaches 78.6 AUC and 128.1 unique cells, compared with 38.0 and 47.1 for
   the corrected flat anchor.

3. **C15's coverage is not evidence for target-directed worker control.** Its
   target-hit lift is 0.898 in the mean and below one in every seed; action
   sensitivity is only 0.0019. Its passive and deliberate graphs become nearly
   saturated (0.986 passive and 0.992 controllable known-edge fractions), yet
   final-frontier reach events remain extremely rare. The frontier mechanism
   changes exploration, but the worker is not demonstrably following landmark
   commands.

4. **Explicit waypoint planning is not the source of the C15 benefit.** C16
   uses longer routes (mean selected hop count 1.51 versus 1.33 for C15) and
   raises target-hit lift to approximately the shuffled reference. It loses
   37.0 coverage-AUC points and 76.6 unique cells relative to C15, with the
   decrease present in all three seeds. More planning structure therefore does
   not repair the weak local controller.

5. **Delayed direct conditioning is behaviorally preferable to immediate
   conditioning in this batch.** C02 exceeds C03 by 21.9 coverage-AUC points
   and 35.8 unique cells, consistently across all seeds. Immediate conditioning
   has more raw hits and completed-option successes but lower action
   sensitivity. Those extra hits cannot be interpreted as better intentional
   control because its mean lift is only 1.003.

6. **The cleanest immediate structural candidate is C05, not the most complex
   combination.** Global punishment plus row repulsion raises coverage over C03
   in all seeds, raises mean lift from 1.003 to 1.091, and raises sensitivity in
   two seeds. Adding CA3 exclusion in C06 raises lift further but lowers raw
   hit rate, option success, and action sensitivity. C08's high mean coverage is
   driven by seed 99 and has very large variance.

7. **Long deadlines improve completion statistics without establishing target
   control.** C12 has the highest action sensitivity (0.075) and option success
   (0.759), but its lift is below one in all seeds. The policy reacts to the
   target input in two seeds without becoming more likely to activate the
   commanded DG unit. Long deadlines make accidental completion more likely
   and reduce timeout frequency; they do not by themselves prove navigation.

8. **C13 recovery is counterproductive for target control.** Relative to C12,
   it lowers raw target-hit rate, action sensitivity, option success, and known
   edge fraction in all three seeds. Mean coverage also falls, although seed 99
   moves in the opposite direction. Only about 45% of its late transitions have
   a normal DG target because timeout recovery spends substantial time in the
   reserved exploration mode.

9. **Iterative updates and empirical HER do not provide a clean positive
   result.** C04 raises coverage relative to C03 in all seeds but lowers hit
   rate and option success in all seeds. C09-C11 show mixed target-control
   effects. After these runs, an audit found that empirical HER may place its
   terminal reward on a later recurrence of a hindsight DG target rather than
   the first occurrence. The HER cells therefore test the implemented biased
   auxiliary, not a clean first-hit HER algorithm, and cannot support a general
   claim about HER.

### Offline DG representation telemetry

The focused standard sweep contains 77 stochastic 10,000-decision probes: five
seed-99 checkpoints and terminal checkpoints for seeds 8 and 123 in C01-C08 and
C14-C16. This covers the flat/direct controls, every unmeasured structural
candidate, the high-variance C08 condition, and the complete topology contrast.
C09-C11 are excluded because their HER implementation lacks first-achievement
termination; C12-C13 already have standard telemetry. The manifest has unique
labels, verified checkpoints, and workspace-only paths. Raw arrays, complete
plots, logs, and temporary data remain under:

`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/corrected_core_candidates_20260902_place_fields/`

The already-completed C12/C13 subset is reported in
[[target_control_her_provisional_place_field_telemetry_20260902|Target-Control HER: Provisional Place-Field Telemetry]].
It shows all 16 units active at terminal, with C13 producing higher spatial
information and lower map redundancy than C12 despite its worse target control.
This is direct evidence that non-collapsed, differentiated landmark activity is
not sufficient for a worker to obey landmark targets.

The C12/C13 terminal maps are split into four-unit pages so each DG unit is
readable at report width. Each page uses one thresholded,
occupancy-corrected activity scale across its four units; gray cells were not
visited in that rollout. Brightness is therefore comparable *within* a page,
not between pages or conditions.

![C12 seed-99 terminal DG activity, units 0--3](assets/target_control_her_place_fields_20260902/readable_c12_s99_rate_units_0.png)

![C12 seed-99 terminal DG activity, units 4--7](assets/target_control_her_place_fields_20260902/readable_c12_s99_rate_units_4.png)

![C12 seed-99 terminal DG activity, units 8--11](assets/target_control_her_place_fields_20260902/readable_c12_s99_rate_units_8.png)

![C12 seed-99 terminal DG activity, units 12--15](assets/target_control_her_place_fields_20260902/readable_c12_s99_rate_units_12.png)

![C13 seed-99 terminal DG activity, units 0--3](assets/target_control_her_place_fields_20260902/readable_c13_s99_rate_units_0.png)

![C13 seed-99 terminal DG activity, units 4--7](assets/target_control_her_place_fields_20260902/readable_c13_s99_rate_units_4.png)

![C13 seed-99 terminal DG activity, units 8--11](assets/target_control_her_place_fields_20260902/readable_c13_s99_rate_units_8.png)

![C13 seed-99 terminal DG activity, units 12--15](assets/target_control_her_place_fields_20260902/readable_c13_s99_rate_units_12.png)

The short-path panels also show that the policies can repeatedly circle or
remain nearly stationary even while the DG population is active:

![Corrected C12 and C13 short policy paths](assets/target_control_her_place_fields_20260902/corrected_c12_vs_c13_s99_trajectory_chunks.png)

The candidate sweep is now complete: all 77 stochastic 10,000-decision probes
produced raw arrays, and no Slurm error log or traceback was found. The table
below is terminal mean +/- sample SD over seeds 8, 99, and 123. Map cosine and
peak metrics are computed on active thresholded maps; continuous-logit metrics
gave the same qualitative C05 result.

| Condition | Visited cells | SI (bits) | Active-map cosine | Unique peak bins | Peak separation (grid bins) | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C03 immediate control | 165.7 +/- 35.7 | **0.209 +/- 0.049** | **0.053 +/- 0.029** | **15.7 +/- 0.6** | 10.39 +/- 1.55 | Strongest spatial differentiation among the immediate-control comparison. |
| C05 punish + row repulsion | 232.7 +/- 42.1 | 0.110 +/- 0.045 | 0.127 +/- 0.065 | 14.3 +/- 0.6 | 10.93 +/- 1.27 | Broadly sampled and non-silent, but lower-information and more redundant than C03. |
| C06 C05 + CA3 exclusion | 244.0 +/- 36.6 | 0.150 +/- 0.018 | 0.086 +/- 0.019 | 14.3 +/- 1.5 | **11.88 +/- 1.26** | Healthier spatial representation than C05, alongside its higher hit lift, but not a clean behavioral win. |
| C15 UCB-direct topology | **277.0 +/- 30.8** | 0.135 +/- 0.023 | 0.116 +/- 0.043 | **16.0 +/- 0.0** | 10.75 +/- 0.65 | The coverage winner remains spatially distributed; this still does not demonstrate target following. |

Peak separation is in 19 x 19 grid bins (one bin is 100 DMLab position
units). All listed conditions had 16 active and zero silent units at terminal.
Thus C05's online activity is not a silent-population failure, but its lower
spatial information and higher map overlap mean the behavioral C05 advantage
cannot be attributed to a uniquely clean landmark code.

### C05: why is `T_ctrl` short?

C05's apparent goal-reaching advantage remains real but modest: final-window
target-hit lift is 1.091 +/- 0.051, action sensitivity is 0.0256 +/- 0.0191,
and option success is 0.317 +/- 0.081. Successful arrivals take 4.34 policy
decisions on average, while known-edge median `T_ctrl` is only 3.91, 4.15, and
5.16 decisions in seeds 8, 99, and 123. With frame skip 8, that is roughly
35--41 repeated simulator frames, not four primitive motor updates.

The spatial telemetry argues against interpreting those short times as
efficient navigation between compact, distant landmarks:

1. **Not a tiny whole-policy region.** The three terminal probes visit 184,
   257, and 257 of 361 grid cells. C05 therefore explores a substantial
   fraction of the map during the 10k-decision evaluator; a global physical
   confinement explanation is not supported. This does not prove that the
   *successful option events* themselves span the same area.
2. **Substantial multi-location aliasing.** At half of each unit's positive
   peak, 12/16, 15/16, and 14/16 C05 units have at least two 4-connected high
   rate-map components (means 2.44, 5.94, and 4.12 components per unit).
   This deliberately simple contour diagnostic is sensitive to sparse sampling
   islands, but the thresholded and continuous maps visibly show several
   disconnected high-response patches rather than one compact field per unit.
3. **The graph funnels into weak spatial targets.** The confidence-qualified
   hub destinations are DG 4 (seed 8, SI 0.013), DG 2 and 14 (seed 99, 0.061
   and 0.040), and DG 13 (seed 123, 0.128). They are not the most spatially
   informative units in their probes. Together with only 3--5 destination
   nodes and 33--42 known directed edges per seed, this fits easy/recurrent
   target activations better than uniform local control.
4. **Short `T_ctrl` does not scale with a field-separation proxy.** For each
   stored known edge, we measured the distance between its source and target
   units' terminal map peaks. Median distances are 14.04, 10.55, and 6.68 bins
   (about 1404, 1055, and 668 DMLab units) for seeds 8, 99, and 123, yet the
   Pearson correlations of edge `T_ctrl` with this distance are 0.028, 0.146,
   and 0.007 (rank correlations -0.136, 0.050, and 0.003). There is no
   positive distance-time relation suggesting slower physical travel to farther
   landmark targets.

The last test is only a diagnostic, not a physical-navigation measurement: a
unit's global peak is ambiguous when its map has multiple components, and the
checkpoint does not store valid landmark poses for C05. The telemetry cannot
show that actions are effective at reaching a prescribed far target because it
records ordinary stochastic policy rollouts, not a forced current-position /
target-unit intervention. The appropriate next test is a target-conditioned
probe that starts from controlled pose bins, commands each DG unit, records
first-hit location and path length, and compares those outcomes to matched
shuffled targets. On the present evidence, the conservative explanation is a
sparse destination-funnel with spatial aliasing, not established fast
long-range control.

The C05 thresholded maps use the same four-unit, shared-within-page format and
occupancy mask. They show which locations receive post-threshold DG activity;
they are not a fixed-trajectory test and cannot establish causal field drift.

![C05 seed-99 terminal DG activity, units 0--3](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_rate_units_0.png)

![C05 seed-99 terminal DG activity, units 4--7](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_rate_units_4.png)

![C05 seed-99 terminal DG activity, units 8--11](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_rate_units_8.png)

![C05 seed-99 terminal DG activity, units 12--15](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_rate_units_12.png)

The continuous-logit pages retain subthreshold structure. Their red--blue
scale is symmetric around zero within each page, so red and blue mean positive
and negative occupancy-corrected logits respectively; its magnitude is not
comparable across pages.

![C05 seed-99 terminal pre-threshold logits, units 0--3](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_prethreshold_units_0.png)

![C05 seed-99 terminal pre-threshold logits, units 4--7](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_prethreshold_units_4.png)

![C05 seed-99 terminal pre-threshold logits, units 8--11](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_prethreshold_units_8.png)

![C05 seed-99 terminal pre-threshold logits, units 12--15](assets/corrected_core_candidates_place_fields_20260902/readable_c05_s99_prethreshold_units_12.png)

The following four large-format panels show the corresponding terminal policy
trajectory for seed 99. Each is a separate 300-decision window so text and
paths remain readable at report width. Purple-to-yellow denotes early-to-late
within the window; green circles and red crosses mark continuous-segment starts
and ends. Resets and large position jumps are not connected.

![C05 seed-99 trajectory, decisions 0--299](assets/corrected_core_candidates_place_fields_20260902/c05_s99_trajectory_chunk_0.png)

![C05 seed-99 trajectory, decisions 3,233--3,532](assets/corrected_core_candidates_place_fields_20260902/c05_s99_trajectory_chunk_1.png)

![C05 seed-99 trajectory, decisions 6,467--6,766](assets/corrected_core_candidates_place_fields_20260902/c05_s99_trajectory_chunk_2.png)

![C05 seed-99 trajectory, decisions 9,701--10,000](assets/corrected_core_candidates_place_fields_20260902/c05_s99_trajectory_chunk_3.png)

### Original C15: distributed peaks, fragmented fields, uneven exploration

**Added 2026-09-08 from the already completed probes; no new training or
rollout.** This is original corrected-core **`c15_topology_ucb_direct_o1`**,
not a later DGP, CPD, or SAT condition with the same C15 label. All seven
manifest rows resolve to existing artifacts. These are **10,000-decision
stochastic policy probes** (10,001 recorded samples including the initial
sample), not the later 100k-decision gallery or online 100k-sample snapshots.
The three terminal checkpoints are all at **100,040,704 training frames**.

The maps refine the earlier “coverage winner” interpretation: **C15 reaches
widely separated parts of the environment, but does not learn a population of
compact, single-location fields.** Its behavior can spend long intervals in a
small region even while cumulative coverage is high. This supports uneven,
sometimes locally repetitive exploration; it does not establish that the
representation is useful for intentional local control.

| Terminal seed | Visited cells / 361 | Samples in top 10 cells | Active / silent units | SI (bits) | Active / pre-threshold cosine | Threshold / pre-threshold peak bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 249 (69.0%) | 59.6% | 16 / 0 | 0.134 | 0.098 / −0.009 | 16 / 15 |
| 99 | 310 (85.9%) | 24.0% | 16 / 0 | 0.112 | 0.165 / −0.013 | 16 / 16 |
| 123 | 272 (75.3%) | 60.3% | 16 / 0 | 0.158 | 0.085 / −0.016 | 16 / 15 |

These are individual policy probes, not independent per-unit replicates.
SI is mean spatial information across the active units. The top-ten statistic
uses the ten highest-occupancy bins, which need not form one contiguous region.
The occupancy and paths below locate that concentration: seeds 8 and 123
heavily revisit the lower-left boundary/corner; seed 123 has repeated local
loops in both middle windows. Seed 99 samples the interior and perimeter more
widely. Coverage counts include all visited bins, without a dwell-time minimum;
361 is the rectangular grid size, not a verified count of navigable cells.

**Peak diversity is real, but insufficient.** Thresholded mean pairwise peak
separation is 10.30, 11.49, and 10.47 bins for seeds 8, 99, and 123;
continuous-logit separation is 9.77, 10.08, and 9.68 bins. One bin is 100
DMLab position units. The peak plots show local neighbors within a globally
distributed population: in every seed the nearest-neighbor distance has
median 2.24 bins and minimum 1 bin. Distinct peaks therefore do not imply
uniform tiling or absence of local clustering. Low continuous-map cosine and
15–16 continuous peak bins show that between-unit differentiation is not
created solely by the hard threshold. Signed logit cosine is not a direct
measure of compactness.

**Field fragmentation persists after the canonical smoothing check.** On the
raw maps, 16/16, 15/16, and 13/16 units have multiple 4-connected regions at
half their own peak. Those raw counts are sensitive to sparse sampling.
The canonical offline multilevel analysis instead smooths activity sums and
occupancy separately with the binomial kernel, divides them, masks unvisited
bins, and uses 8-connected components at 30%, 50%, and 70% of each unit's peak.
A unit is eligible with at least 20 active in-bounds observations across three
bins; a single-field classification requires at least 80% of superlevel mass
in the dominant component at every threshold.

| Seed | Eligible units | Single-field units / eligible | Mean components at 30% / 50% / 70% |
| --- | ---: | ---: | ---: |
| 8 | 16 | 1 / 16 | 6.50 / 4.25 / 2.50 |
| 99 | 15 | 0 / 15 | 8.80 / 5.67 / 2.93 |
| 123 | 15 | 0 / 15 | 10.47 / 5.20 / 2.93 |

Thus only one of 46 eligible terminal unit maps passes this criterion; the
two ineligible units remain active, rather than silent. The visual maps show
multiple separated response patches and broad responses as well as isolated
hotspots. For example, seed-8 DG 15 has a broad interior response, while
seed-99 DG 7 has several separated hotspots that also appear in its continuous
logit map. Occupancy is sparse away from favored paths (median visited-bin
sample counts 8, 16, and 9), so single-bin peaks and component counts should
not be treated as established stable biological place fields. These legacy
NPZs have no `pose_alignment` metadata and may retain the historical
one-decision pose offset; they have not been silently replaced by the later
alignment-corrected evaluator.

#### Seed-99 checkpoint evolution

| Actual training frames | Visited cells | SI (bits) | Active-map cosine | Unique peaks | Single-field / eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6,258,688 | 308 | 0.094 | 0.164 | 14 | 0 / 16 |
| 26,443,776 | 318 | 0.105 | 0.191 | 16 | 0 / 15 |
| 50,102,272 | 317 | 0.122 | 0.175 | 14 | 1 / 16 |
| 73,728,000 | 308 | 0.106 | 0.161 | 15 | 0 / 15 |
| 100,040,704 | 310 | 0.112 | 0.165 | 16 | 0 / 15 |

The five probes stay broadly sampled with differentiated peaks, without a
monotonic transition to compact fields. They follow different policy-driven
paths, so this is a checkpoint summary rather than a fixed-trajectory drift
test. Original C15's earlier external-exploration advantage remains valid;
these maps do not explain it causally. Target-command interventions and
chance-corrected target arrivals are still needed to demonstrate useful local
control.

#### Figure reading and display scales

Every terminal seed below includes **all 16 DG units**, four per page, followed
by all 16 continuous pre-threshold maps, occupancy/peak context, and four fixed
300-sample trajectory windows. Unit IDs are within-run identities, not matched
cells across independently trained seeds. Gray means unvisited; no map
interpolation or display smoothing is applied. Arrays are stored in x/y order
and transposed for Cartesian display. Bin 0 is centered at position 150;
bin 18 at 1,950, on each axis.

To reveal field shape despite large amplitude differences, each thresholded
map is divided by its own positive peak; each continuous map is divided by its
own maximum absolute visited-bin value. The numeric **scale** printed above
each unit is that divisor in the original activity/logit units. Color intensity
therefore compares relative spatial shape, not absolute activation between
units or seeds; multiply by the displayed scale to recover the map value.
This normalization changes neither peak positions nor the reported metrics,
which use the original arrays. Occupancy uses a logarithmic sample-count scale.
Trajectory windows are selected at evenly spaced start indices, break at
recorded episode boundaries, and mark the window start in orange.

#### C15 terminal seed 8

![C15 seed 8: Occupancy and unit peak locations](assets/corrected_core_c15_place_fields_20260908/c15_s8_context.png)

![C15 seed 8: Four trajectory windows](assets/corrected_core_c15_place_fields_20260908/c15_s8_paths.png)

**Thresholded DG activity, seed 8.** Unit-specific normalization as defined above.

![C15 seed 8: Thresholded DG activity, units 0–3](assets/corrected_core_c15_place_fields_20260908/c15_s8_rate_00.png)

![C15 seed 8: Thresholded DG activity, units 4–7](assets/corrected_core_c15_place_fields_20260908/c15_s8_rate_04.png)

![C15 seed 8: Thresholded DG activity, units 8–11](assets/corrected_core_c15_place_fields_20260908/c15_s8_rate_08.png)

![C15 seed 8: Thresholded DG activity, units 12–15](assets/corrected_core_c15_place_fields_20260908/c15_s8_rate_12.png)

**Continuous pre-threshold DG logits, seed 8.** Unit-specific normalization as defined above.

![C15 seed 8: Continuous pre-threshold DG logits, units 0–3](assets/corrected_core_c15_place_fields_20260908/c15_s8_prethreshold_00.png)

![C15 seed 8: Continuous pre-threshold DG logits, units 4–7](assets/corrected_core_c15_place_fields_20260908/c15_s8_prethreshold_04.png)

![C15 seed 8: Continuous pre-threshold DG logits, units 8–11](assets/corrected_core_c15_place_fields_20260908/c15_s8_prethreshold_08.png)

![C15 seed 8: Continuous pre-threshold DG logits, units 12–15](assets/corrected_core_c15_place_fields_20260908/c15_s8_prethreshold_12.png)

#### C15 terminal seed 99

![C15 seed 99: Occupancy and unit peak locations](assets/corrected_core_c15_place_fields_20260908/c15_s99_context.png)

![C15 seed 99: Four trajectory windows](assets/corrected_core_c15_place_fields_20260908/c15_s99_paths.png)

**Thresholded DG activity, seed 99.** Unit-specific normalization as defined above.

![C15 seed 99: Thresholded DG activity, units 0–3](assets/corrected_core_c15_place_fields_20260908/c15_s99_rate_00.png)

![C15 seed 99: Thresholded DG activity, units 4–7](assets/corrected_core_c15_place_fields_20260908/c15_s99_rate_04.png)

![C15 seed 99: Thresholded DG activity, units 8–11](assets/corrected_core_c15_place_fields_20260908/c15_s99_rate_08.png)

![C15 seed 99: Thresholded DG activity, units 12–15](assets/corrected_core_c15_place_fields_20260908/c15_s99_rate_12.png)

**Continuous pre-threshold DG logits, seed 99.** Unit-specific normalization as defined above.

![C15 seed 99: Continuous pre-threshold DG logits, units 0–3](assets/corrected_core_c15_place_fields_20260908/c15_s99_prethreshold_00.png)

![C15 seed 99: Continuous pre-threshold DG logits, units 4–7](assets/corrected_core_c15_place_fields_20260908/c15_s99_prethreshold_04.png)

![C15 seed 99: Continuous pre-threshold DG logits, units 8–11](assets/corrected_core_c15_place_fields_20260908/c15_s99_prethreshold_08.png)

![C15 seed 99: Continuous pre-threshold DG logits, units 12–15](assets/corrected_core_c15_place_fields_20260908/c15_s99_prethreshold_12.png)

#### C15 terminal seed 123

![C15 seed 123: Occupancy and unit peak locations](assets/corrected_core_c15_place_fields_20260908/c15_s123_context.png)

![C15 seed 123: Four trajectory windows](assets/corrected_core_c15_place_fields_20260908/c15_s123_paths.png)

**Thresholded DG activity, seed 123.** Unit-specific normalization as defined above.

![C15 seed 123: Thresholded DG activity, units 0–3](assets/corrected_core_c15_place_fields_20260908/c15_s123_rate_00.png)

![C15 seed 123: Thresholded DG activity, units 4–7](assets/corrected_core_c15_place_fields_20260908/c15_s123_rate_04.png)

![C15 seed 123: Thresholded DG activity, units 8–11](assets/corrected_core_c15_place_fields_20260908/c15_s123_rate_08.png)

![C15 seed 123: Thresholded DG activity, units 12–15](assets/corrected_core_c15_place_fields_20260908/c15_s123_rate_12.png)

**Continuous pre-threshold DG logits, seed 123.** Unit-specific normalization as defined above.

![C15 seed 123: Continuous pre-threshold DG logits, units 0–3](assets/corrected_core_c15_place_fields_20260908/c15_s123_prethreshold_00.png)

![C15 seed 123: Continuous pre-threshold DG logits, units 4–7](assets/corrected_core_c15_place_fields_20260908/c15_s123_prethreshold_04.png)

![C15 seed 123: Continuous pre-threshold DG logits, units 8–11](assets/corrected_core_c15_place_fields_20260908/c15_s123_prethreshold_08.png)

![C15 seed 123: Continuous pre-threshold DG logits, units 12–15](assets/corrected_core_c15_place_fields_20260908/c15_s123_prethreshold_12.png)

#### Reproduction and reusable lesson

The thin adapter `06_experiments/render_c15_completed_place_fields.py` reads
the existing `analysis_manifest.tsv`, selects the exact condition, resolves
artifacts through the canonical evaluator, and calls its `derive_row` and
`per_unit_rows`. Run it from the NEMO2 runtime checkout with `PYTHONPATH=.` and
the documented SFgit interpreter, passing the telemetry root and a workspace
output directory. Only array analysis and rendering run on the login node;
there is no DMLab rollout. Outputs remain in `c15_report/` under the original
telemetry root. The local figure folder includes SVG exports, `c15_metrics.csv`,
`c15_per_unit.csv`, and `provenance.json` with manifest/artifact SHA-256 values.
This is recovery of a historical manifest predating StudySpec; no schema,
workflow version, or study fingerprint has been invented for it.

Reuse lesson: completed manifest rows and NPZs were authoritative; a missing
report gallery did not mean missing telemetry. Reusing the canonical analysis
avoided a new sweep. A shared absolute color scale initially hid most units
behind a few high-amplitude hotspots, so the final shape gallery explicitly
labels each unit's normalization. Verify array axes against evaluator binning,
inspect occupancy and full maps before interpreting peak diversity, and keep
raw NPZs remote. Scalable fonts and all figure pages were checked at report
viewing size; the initial long context-panel title was shortened after QA.

### Valid conclusions and limitations

The corrected batch supports three provisional conclusions:

- corrected sparse DG/CA3 activity can coexist with substantial external
  exploration in the absence of environmental reward;
- frontier scoring can markedly increase coverage, but the effect is not
  mediated by demonstrated target-conditioned landmark control;
- adding manager, recovery, HER, or waypoint machinery does not monotonically
  improve the system, so complexity is not a substitute for a causal mechanism
  test.

It does **not** yet establish compact canonical place fields across all cells,
efficient landmark-conditioned navigation, or a complete HRL/SLAM solution.
The actor/learner DG BatchNorm semantic mismatch remains deferred, CA3 activity
is amplitude-accumulating while event rewards use occupancy, and sparse event
gradients remain normalized by all valid transitions. These unresolved core
boundaries should be addressed before the next paper-defining ablation.

### Reproducibility artifacts

- Analyzer:
  `06_experiments/analyze_corrected_core_reevaluation.py`
- Per-run late-window table:
  `06_experiments/results/corrected_core_reevaluation_20260902/per_run_terminal_10m.csv`
- Three-seed cell aggregate:
  `06_experiments/results/corrected_core_reevaluation_20260902/cell_terminal_10m.csv`
- Seed-paired planned contrasts:
  `06_experiments/results/corrected_core_reevaluation_20260902/paired_contrasts_terminal_10m.csv`
- Place-field manifest builder:
  `06_experiments/build_corrected_core_place_field_manifest.py`
- Candidate terminal and per-probe spatial metrics:
  `06_experiments/assets/corrected_core_candidates_place_fields_20260902/terminal_three_seed_aggregate.csv`
  and `derived_place_field_metrics.csv`
