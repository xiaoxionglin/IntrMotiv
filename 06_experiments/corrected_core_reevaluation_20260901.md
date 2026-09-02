# Corrected-Core Historical Design Re-evaluation

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

| Cell | Condition |
| --- | --- |
| C01 | Corrected flat temporal-distance anchor |
| C02 | Direct global HRL, delayed target, short deadline |
| C03 | Direct global HRL, immediate target, short deadline |
| C04 | C03 with iterative encoder/decoder updates |
| C05 | C03 with global punishment `0.01` and row repulsion `1.0` |
| C06 | C05 with CA3 temporal exclusion `1.0` |
| C07 | C03 with orthogonal recruitment |
| C08 | C03 with CA3 temporal exclusion and orthogonal recruitment |
| C09 | C03 with empirical PPO-HER, horizon 16, coefficient `0.5` |
| C10 | C03 with empirical PPO-HER, horizon 64, coefficient `0.5` |
| C11 | C02 with empirical PPO-HER, horizon 64, coefficient `0.5` |
| C12 | Delayed direct HRL, CA3 exclusion and recruitment, long deadline |
| C13 | C12 with manager exploration probability `0.10` and timeout recovery |
| C14 | Passive topology with visit-based direct frontier selection |
| C15 | Passive topology with UCB direct frontier selection |
| C16 | Passive topology with UCB frontier selection and waypoint planning |

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

| Cell | Coverage AUC | Unique cells | DG density | Target-hit lift | Action sensitivity | Option success | Known edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| C01 | 38.0 +/- 5.6 | 47.1 +/- 5.6 | 0.0263 +/- 0.0011 | -- | -- | -- | -- |
| C02 | 45.1 +/- 8.0 | 63.6 +/- 5.1 | 0.0255 +/- 0.0029 | 0.959 +/- 0.070 | 0.0303 +/- 0.0217 | 0.277 +/- 0.042 | 0.180 +/- 0.046 |
| C03 | 23.2 +/- 8.7 | 27.9 +/- 12.6 | 0.0272 +/- 0.0031 | 1.003 +/- 0.023 | 0.0125 +/- 0.0075 | 0.453 +/- 0.209 | 0.147 +/- 0.004 |
| C04 | 39.6 +/- 6.6 | 53.6 +/- 12.0 | 0.0265 +/- 0.0008 | 1.033 +/- 0.111 | 0.0204 +/- 0.0141 | 0.193 +/- 0.060 | 0.153 +/- 0.013 |
| C05 | 42.0 +/- 2.0 | 53.3 +/- 5.8 | 0.0249 +/- 0.0006 | 1.091 +/- 0.051 | 0.0256 +/- 0.0191 | 0.317 +/- 0.081 | 0.169 +/- 0.006 |
| C06 | 37.8 +/- 8.4 | 49.1 +/- 11.7 | 0.0214 +/- 0.0015 | 1.147 +/- 0.112 | 0.0124 +/- 0.0023 | 0.196 +/- 0.048 | 0.175 +/- 0.014 |
| C07 | 36.0 +/- 5.3 | 45.9 +/- 8.1 | 0.0255 +/- 0.0014 | 1.120 +/- 0.132 | 0.0224 +/- 0.0043 | 0.313 +/- 0.063 | 0.178 +/- 0.006 |
| C08 | 47.5 +/- 27.3 | 64.1 +/- 44.4 | 0.0225 +/- 0.0010 | 0.984 +/- 0.032 | 0.0105 +/- 0.0079 | 0.222 +/- 0.114 | 0.175 +/- 0.031 |
| C09 | 36.7 +/- 4.1 | 48.7 +/- 8.1 | 0.0245 +/- 0.0015 | 1.036 +/- 0.112 | 0.0232 +/- 0.0090 | 0.251 +/- 0.103 | 0.167 +/- 0.023 |
| C10 | 38.2 +/- 9.9 | 46.7 +/- 14.2 | 0.0228 +/- 0.0020 | 0.979 +/- 0.035 | 0.0200 +/- 0.0101 | 0.244 +/- 0.042 | 0.152 +/- 0.031 |
| C11 | 30.7 +/- 7.6 | 38.7 +/- 11.5 | 0.0264 +/- 0.0035 | 1.032 +/- 0.048 | 0.0251 +/- 0.0094 | 0.326 +/- 0.015 | 0.160 +/- 0.018 |
| C12 | 48.1 +/- 11.4 | 67.4 +/- 23.3 | 0.0259 +/- 0.0010 | 0.918 +/- 0.012 | 0.0750 +/- 0.0591 | 0.759 +/- 0.129 | 0.168 +/- 0.024 |
| C13 | 38.5 +/- 11.3 | 48.5 +/- 15.3 | 0.0238 +/- 0.0024 | 0.959 +/- 0.018 | 0.0073 +/- 0.0025 | 0.640 +/- 0.154 | 0.107 +/- 0.031 |
| C14 | 48.1 +/- 10.1 | 67.9 +/- 16.8 | 0.0269 +/- 0.0011 | 0.928 +/- 0.024 | 0.0030 +/- 0.0006 | 0.399 +/- 0.049 | 0.957 +/- 0.020 |
| C15 | **78.6 +/- 13.8** | **128.1 +/- 23.8** | 0.0279 +/- 0.0017 | 0.898 +/- 0.005 | 0.0019 +/- 0.0006 | 0.388 +/- 0.008 | 0.992 +/- 0.001 |
| C16 | 41.6 +/- 2.4 | 51.4 +/- 5.7 | 0.0245 +/- 0.0003 | 1.000 +/- 0.034 | 0.0085 +/- 0.0025 | 0.428 +/- 0.023 | 0.866 +/- 0.085 |

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

The complete standard sweep contains 112 stochastic 10,000-decision probes:
five seed-99 checkpoints and terminal checkpoints for seeds 8 and 123 in every
cell. The manifest has unique labels, verified checkpoints, and workspace-only
paths. Raw arrays, complete plots, logs, and temporary data remain under:

`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/corrected_core_reevaluation_20260902_place_fields/`

The already-completed C12/C13 subset is reported in
[[target_control_her_provisional_place_field_telemetry_20260902|Target-Control HER: Provisional Place-Field Telemetry]].
It shows all 16 units active at terminal, with C13 producing higher spatial
information and lower map redundancy than C12 despite its worse target control.
This is direct evidence that non-collapsed, differentiated landmark activity is
not sufficient for a worker to obey landmark targets.

The full 16-cell terminal aggregate, seed-99 field trajectories, and selected
figures will be inserted here after the queued Slurm sweep and standard
postprocessing complete.

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
