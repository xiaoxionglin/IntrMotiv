# Directional Sink Recruitment and Goal Conditioning Batch

**Status:** Implementation plan. Updated 2026-09-04 to include the restored
legacy goal decoder and the new target-ID FiLM decoder.

## Question

Can strict replacement of mature control-graph sinks reduce landmark aliasing
and destination funnels, and does the answer depend on how the worker receives
its commanded target?

The contextual-CA3 landmark proposal is deferred to
`contextual_landmark_state_design.md` and is not part of this batch.

## Matrix

- Bases: corrected-core C05, current C13-like, and C15.
- Recruitment treatments: monitor-only (`MON`), existing incident rule (`INC`),
  and directional sink-aware rule (`DIR`).
- Goal conditioning: restored previous decoder (`LEG`) and target-ID FiLM
  (`FILM`). The target-trace adapter is excluded.
- Seeds: 8, 99, and 123.
- Total: 3 x 3 x 2 x 3 = 54 fresh 75M-step runs.

Use immediate behavior targets in every cell because FiLM requires immediate
timing and the comparison must differ only in the goal decoder. All cells use
`ceil(1.2 * Tctrl) + 2` for learned deadlines and 64 decisions for unknown
edges.

Proposed identities:

- study: `directional_sink_goal_conditioning_20260904`;
- batch: `intrmotiv_directional_sink_goal_conditioning_20260904`;
- W&B project: `SF_IntrMotiv_DirectionalSinkGoalConditioning`;
- run: `DSG_{base}_{recruitment}_{goal}_S{seed}`.

Use StudySpec schema `intrmotiv/study/v1` and workflow `1.2.0`. Preserve its
SHA-256 through print-only review, submission audit, online analysis, and
telemetry manifests.

## Directional recruitment

Add backward-compatible graph-source and victim-rule settings. Existing runs
retain `auto` graph source and the current `incident` rule; this study fixes
the source to the policy controllability graph.

A supported recruitment edge has confidence above `0.25` and positive
`Tctrl`. With `D=4`:

```text
strict_sink(j) = no supported j -> k edge
                 and at least two distinct supported i -> j edges
                 with Tctrl[i,j] <= D
```

Keep isolated-node eligibility. Keep the mutual-close duplicate definition,
but in the directional rule choose the lower-outgoing-support pair member;
ties choose the higher DG index. Apply birth maturity after structural
classification. Victim priority is isolated, strict sink, then duplicate.

For multiple sinks, choose highest fast incoming degree, then highest fast
incoming confidence, then lowest index. For multiple duplicate losers, choose
lowest outgoing support, then lowest index. Do not add a low-outdegree ratio,
connectivity score, or attempt threshold.

The silent endpoint at exactly `L=64` remains the only replacement proposal.
Reassignment and invalidation remain unchanged. Repeat replacement is allowed
only after the reset birth support has matured again. Passive and control
half-lives are fixed at 5k.

Treatments are:

- `MON`: directional classifier active, maximum replacements per rollout zero;
- `INC`: current incident classifier, maximum one replacement per rollout;
- `DIR`: directional classifier, maximum one replacement per rollout.

## Goal conditioning

- `LEG` uses the restored ordinary decoder with the replayed target one-hot
  concatenated to the core state.
- `FILM` removes the target one-hot from the state stream and uses it only to
  select a target-specific, zero-initialized scale/shift row over a shared
  128-unit state decoder. It does not extract a target CA3 trace.
- Both use the same immediate behavior target stored by the actor and
  teacher-forced during PPO replay.
- Log decoder parameter count, target-valid fraction, action sensitivity,
  value span, and per-target FiLM modulation norms. An all-zero target must
  produce exactly identity modulation.

## Base definitions and fixed settings

- C05: global punishment `0.01`, row repulsion `1.0`, no temporal margin,
  direct visit manager, no manager exploration.
- C13-like: current revised temporal margin `1.0`, direct visit manager, 10%
  exploration plus timeout recovery. It is not an exact historical C13 because
  timing is immediate and the temporal-margin implementation changed.
- C15: no temporal margin or G/R terms, UCB-direct topology manager, no action
  integration or geometry. Timing is standardized to immediate.

Keep the frozen pretrained ResNet-18 layer-2 trunk, trainable DG projection and
BatchNorm, `F=16`, `R=8`, all other corrected-core optimization settings, and
the shared exploration head unchanged. Do not add CA3 feedback, path scatter,
dense shaping, HER, or the common edge-exploration manager.

## Telemetry and analysis

Add online outgoing/incoming support, strict-sink and sink-eligible counts,
fast-incoming degree, sink assignments, zero-outdegree attempted versus
untested nodes, top-three incoming-confidence share, SCC/reachable-pair
measures, and per-row assignment counts.

Add optional graph buffers to the standard place-field NPZ without changing
existing fields: control confidence/attempts/`Tctrl`, passive
confidence/elapsed, birth support, and row assignment counts. Join these per
unit with spatial information, activity, and four-connected components above
half the unit's positive peak.

StudySpec contrasts are evaluated within base:

- `DIR - MON`, `INC - MON`, and `DIR - INC`, separately for LEG and FILM;
- `FILM - LEG` separately for every recruitment treatment;
- `(DIR - INC)_FILM - (DIR - INC)_LEG` interaction.

Run standard 10k place-field telemetry at 5M, 25M, 50M, and 75M for seed 99
and at 75M for seeds 8 and 123: 108 ordinary jobs. Run the frozen terminal
target-intervention evaluator for all 54 runs.

## Verification and launch

Add unit tests for strict zero-outdegree sink classification, two-source fast
incoming evidence, outgoing protection, maturity/repeat behavior, victim
priority, outgoing-support duplicate selection, graph-source selection,
monitor-only behavior, invalidation, checkpoint compatibility, and optional
NPZ fields.

Add goal tests proving restored LEG decoder selection, exact initial FiLM
identity modulation, target-row-specific gradients, no-target behavior,
teacher-forced immediate replay, action/value head gradients, and checkpoint
round trips.

Run a fresh seed-99 5M preflight for all 18 factor cells. Require finite losses,
control-graph activity, exactly zero MON replacements, exact behavior-target
replay, nonzero FiLM modulation gradients on goal-directed samples, and no
strict-sink classification when a supported outgoing edge exists.

After the 54-run print-only review and workspace-path audit, submit all runs as
fresh training. Collect synchronized 25M, 50M, and 75M summaries. Stop and
inspect if at least two seeds of a cell have silent fraction above `0.25` or
repeat assignments exceed first assignments.

The recruitment claim is centered on C05 and requires DIR to exercise sink
replacement in at least two seeds, reduce strict sinks and top-three incoming
share relative to both controls, reduce multi-component fields, keep mean
silent units at or below one, and keep repeats no greater than first
assignments. Claim improved control only when terminal commanded-target success
is at least 25% above matched shuffled targets and action sensitivity improves.

Because eligibility deliberately reads the control graph, this remains an
open-field experiment rather than a transferable recruitment rule. C13-like
and C15 provide stress tests; effects are not pooled across bases.
