# Directional and Predictive Recruitment Batch

**Status:** Implementation plan. Updated 2026-09-04 after retiring the
incident-connectivity (`INC`) treatment.

## Question

Can a minimal replacement rule remove landmarks that fail as controllable
sources, or landmarks whose goal-directed outcomes reveal contextual aliasing,
without adding a new representation model?

The contextual-CA3 landmark architectures remain deferred to
`contextual_landmark_state_design.md`. This batch changes only recruitment and
goal decoding.

## Matrix

- Bases: corrected-core C05, C13-like, and C15.
- Recruitment: monitor-only (`MON`), directional controllability (`DIR`), and
  batch-local predictive inconsistency (`PRED`).
- Goal conditioning: restored legacy decoder (`LEG`) and target-ID FiLM
  (`FILM`). The target-trace adapter is excluded.
- Seeds: 8, 99, and 123.
- Total: `3 x 3 x 2 x 3 = 54` fresh 75M-step runs.

Use immediate behavior targets in every cell. Use
`ceil(1.2 * Tctrl) + 2` for learned deadlines and 64 decisions for unknown
edges. Preserve the current policy/exploration branching; it is not a factor.

Proposed identities:

- study: `directional_predictive_recruitment_20260904`;
- batch: `intrmotiv_directional_predictive_recruitment_20260904`;
- W&B project: `SF_IntrMotiv_DirectionalPredictiveRecruitment`;
- run: `DPR_{base}_{recruitment}_{goal}_S{seed}`.

Use StudySpec schema `intrmotiv/study/v1` and workflow `1.2.0`. Preserve its
SHA-256 through print-only review, submission audit, online analysis, and
telemetry manifests.

Validated StudySpec SHA-256:
`72b0ac2d04ad7a297a674f96d4f32c85d48dcf89a9fc62abb7243adb22ea53aa`.

## Common recruitment mechanics

- Evaluate replacement only at the existing silent endpoint `L=64`.
- Allow at most one replacement per accepted rollout.
- Require birth maturity: `birth_support <= 0.25`.
- Reset the replaced row's passive and controllability graph evidence,
  including attempt evidence. The row must mature and qualify again before
  another replacement.
- Retain the 5k-option global half-life for control confidence, attempts, and
  birth support.

A directed controllability edge `j -> k` is reliable exactly when:

```text
Tctrl[j,k] > 0
and edge_confidence[j,k] >= 0.5
and (edge_confidence[j,k] + 1) / (control_attempts[j,k] + 2) >= 0.5
```

Only completed intentional goal attempts update `control_attempts`. Free
exploration is excluded.

## MON

Compute and log both DIR and PRED diagnostics but set maximum replacements per
rollout to zero.

## DIR

A node is fully tested only when every off-diagonal outgoing pair retains at
least 0.5 decayed attempt mass:

```text
fully_tested(j) = AND over k != j: control_attempts[j,k] >= 0.5

bad_source(j) = mature(j)
                and fully_tested(j)
                and reliable_out_degree(j) == 0
```

There is no incoming-edge requirement and no operational distinction between
an isolated node and a sink. If any of the 15 alternatives is untested, the
node is protected. When several bad sources qualify, choose the lowest DG
index rather than adding another score.

DIR alone also removes mutual-close duplicates. With `D=4`:

```text
duplicate(j,k) = reliable(j -> k)
                 and reliable(k -> j)
                 and Tctrl[j,k] <= 4
                 and Tctrl[k,j] <= 4
```

Choose the duplicate loser by lower reliable outgoing degree, then lower total
reliable outgoing confidence, then higher DG index. Prioritize a bad-source
victim over a duplicate loser. MON and PRED only report duplicate diagnostics.

## PRED

PRED uses only the current complete accepted rollout batch. It adds no learned
predictor, recurrent feedback, checkpoint buffer, or persistent statistics.

For each completed goal-directed option whose start and completion both occur
in the accepted batch, extract:

- source landmark `j`;
- commanded goal `g`;
- predecessor context `c`, the most recent distinct exclusive landmark in the
  first `R` CA3 positions at option start. Exclude the current source row
  before testing predecessor exclusivity because its trace can persist across
  several CA3 positions without making the distinct predecessor ambiguous;
- outcome `y`, success or target timeout.

Exclude free-exploration options, invalid transitions, options that began
before the batch, and starts without a distinct predecessor context. For every
observed `(j,g,c)` group:

```text
reliability(j,g,c) = (successes + 1) / (attempts + 2)
```

Source `j` is PRED-eligible when one `(j,g)` is observed under at least two
predecessor contexts, each with at least two completed attempts, and at least
one context has reliability `>= 0.5` while another is `< 0.5`.

Apply birth maturity after this classification. Rank candidates by largest
cross-context reliability gap, then total supporting attempts, then lowest DG
index. Hold the result only until this rollout's single replacement decision
and then discard it. PRED does not perform duplicate replacement.

## Goal conditioning and bases

- `LEG` concatenates the replayed target one-hot to the ordinary decoder
  input.
- `FILM` uses the target one-hot only to select a zero-initialized 128-unit
  FiLM scale/shift row. It uses neither a target trace nor a learned target
  embedding. The all-zero target produces exact identity modulation.
- Both teacher-force the immediate behavior target during PPO replay.

Base definitions:

- C05: direct visit manager, global punishment `0.01`, row repulsion `1.0`, no
  temporal exclusion, and no manager exploration.
- C13-like: direct visit manager, temporal exclusion `1.0`, 10% manager
  exploration plus existing timeout recovery, and no G/R terms.
- C15: UCB-direct topology manager, no temporal exclusion or G/R terms, and no
  action integration or geometry. Direct frontier selection may command any
  observed landmark; reliable-route reachability is required only by waypoint
  or common-manager planning.

Keep the frozen ImageNet ResNet-18 layer-2 trunk, trainable DG projection and
BatchNorm, `F=16`, `R=8`, `L=64`, and corrected-core optimization settings.
Do not add CA3 feedback, path scatter, dense shaping, HER, or the common
edge-probe manager.

## Telemetry and analysis

Add online metrics for:

- fully tested, untested zero-outdegree, and bad-source nodes;
- off-diagonal attempt-coverage fraction and outgoing coverage by node;
- reliable incoming/outgoing degree and confidence;
- duplicate and PRED eligibility, PRED context groups and reliability gaps;
- bad-source, duplicate, predictive, first, and repeat assignments;
- graph concentration, reciprocal edges, SCC size, reachable-pair fraction,
  target-hit lift, target action sensitivity, and FiLM modulation norms.

Extend the standard place-field NPZ with backward-compatible optional arrays:
control confidence, attempts and `Tctrl`; passive confidence and elapsed;
birth support; and per-row assignment counts. Join these arrays with spatial
information, activity, and four-connected field components in analysis.

StudySpec contrasts are evaluated within base:

- `DIR - MON`, `PRED - MON`, and `DIR - PRED`, separately for LEG and FILM;
- `FILM - LEG` separately for MON, DIR, and PRED;
- the DIR x goal and PRED x goal interactions.

Run standard 10k place-field telemetry at 5M, 25M, 50M, and 75M for seed 99
and at 75M for seeds 8 and 123: 108 ordinary jobs. Run the frozen terminal
target-intervention evaluator for all 54 runs.

## Verification and launch

Add unit tests for:

- all 15 alternatives being required for `fully_tested`;
- decayed attempt mass below 0.5 protecting a node;
- incoming edges not affecting bad-source classification;
- unreliable outgoing edges not protecting and any reliable outgoing edge
  protecting a node;
- free exploration not contributing attempt evidence;
- mutual-close duplicate replacement occurring only in DIR;
- PRED using only one accepted batch and retaining no persistent state;
- maturity, invalidation, repeat recruitment, replay conditioning, FiLM
  identity initialization, and checkpoint compatibility.

Run a fresh seed-99 5M preflight for all 18 factor cells. Require finite
losses, exact replay conditions, zero MON replacements, correct intended
manager-mode activity, nonzero FiLM gradients on goal samples, and no DIR
replacement before complete outgoing-pair coverage.

After StudySpec validation, print-only review, workspace-path audit, and a
passing preflight, submit the 54 fresh production runs. Collect synchronized
25M, 50M, and 75M summaries. Stop and inspect if at least two seeds of a cell
have silent fraction above 0.25 or repeat assignments exceed first
assignments.

The recruitment claim is centered on C05. DIR must exercise bad-source or
duplicate replacement in at least two seeds; PRED must exercise predictive
replacement in at least two seeds. A treatment must reduce graph concentration
and multi-component fields relative to MON without increasing mean silent
units above one or repeats above first assignments. Claim improved control
only when terminal commanded-target success is at least 25% above matched
shuffled targets and action sensitivity improves.

DIR deliberately reads the controllability graph and remains an open-field
heuristic rather than a transferable recruitment rule. PRED is the
architecture-neutral test of contextual insufficiency. Effects are reported
within base and are not pooled across C05, C13-like, and C15.
