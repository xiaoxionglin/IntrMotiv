# Saturday Batch

Date: 2026-09-05

## Purpose

Re-establish a healthy C15 landmark baseline and test the three unresolved
scientific mechanisms without repeating the confounds of the preceding
batches:

1. whether temporal encoder credit belongs to the arrival or source landmark;
2. whether open-gate DIR or PRED retirement improves an already healthy
   landmark vocabulary;
3. whether target-ID FiLM improves causal target control over the restored
   legacy decoder.

Normalization is fixed infrastructure, not a factor. Use the cleaned
single-forward `legacy_batch` behavior from SF-XXL. Keep the corrected
DG/controller gradient boundary and replacement-generation transaction.

The seed-99 5M normalization audit selected this baseline decisively. In its
terminal 4--5M window, `legacy_batch` retained all 16 active landmarks,
`0.994` usage entropy, a 15-node reliable SCC, and `0.938` reachable-pair
fraction. The post-step and input-centered alternatives instead produced
SCCs of 1 and 3 and reachable-pair fractions of `0.073` and `0.208`.
Normalization is therefore deliberately removed from the Saturday matrix.

## Run count

The scientific matrix contains 36 fresh 75M-step runs:

| Factor | Levels |
|---|---|
| Encoder credit | `ARR`, `SRC` |
| Retirement | `MON`, `DIRO`, `PREDO` |
| Goal conditioning | `LEG`, `FILM` |
| Seeds | `8`, `99`, `123` |

Before production, run three 1M engineering jobs that force exactly one
replacement: DIRO-FILM, PREDO-FILM, and DIRO-LEG. The complete Saturday launch
therefore uses 39 training jobs. Offline telemetry and intervention jobs are
separate evaluation work and are not counted as training runs.

## Fixed algorithm

- Base: C15, frozen ImageNet ResNet-18 layer-2 trunk, current DG projection,
  `Hippo_R=8`, `Hippo_L=64`, and current frontier-direct manager.
- DG normalization: `legacy_batch` with one learner DG forward and one
  BatchNorm update per encoder-active minibatch.
- Controller boundary: PPO cannot update DG through CA3; bypass and goal inputs
  retain their controller gradients.
- Targets: immediate behavior targets, teacher-forced during replay.
- Deadline: `ceil(1.2 * Tctrl) + 2` through the existing ratio and fixed-margin
  settings.
- Graph: current directed reliable controllability definition and 5k-option
  confidence/attempt half-life.
- No dense shaping, HER, geometry, action embedding, learned predictor,
  normalization experiment, or new encoder objective.

## Factors

### Encoder credit

`ARR` and `SRC` use identical matchable dominant-onset events and identical
total reward mass. Only the credited row and timestep differ. Unresolved and
cross-rollout predecessor events are excluded from both conditions. Existing
batch-use and multi-activation losses remain encoder-only and destination
based.

### Retirement

- `MON` computes diagnostics but never replaces.
- `DIRO` uses the existing fully-tested zero-reliable-outdegree and
  mutual-close-duplicate rules with the endpoint gate open.
- `PREDO` uses persistent decayed predecessor-context evidence with the
  endpoint gate open.

Silent-gate variants are omitted. Previous batches showed that endpoint
silence measures coverage rather than victim quality and frequently prevents
the intended manipulation.

Every replacement is one atomic transaction: DG row and optimizer state,
graph/attempt/PRED/active-option invalidation, FiLM row and Adam-state reset
when present, and representation-generation increment. Old-generation
experience is discarded only as complete 64-step rollouts. If fewer than one
normal fresh minibatch remains, the entire learner update is deferred before
graph updates, reward construction, GAE, or normalization.

### Goal conditioning

- `LEG`: replayed target one-hot concatenated to the ordinary decoder input.
- `FILM`: zero-initialized target-ID FiLM scale/shift with no target trace or
  learned target embedding.

FiLM is retained as a factor because prior online action sensitivity favored
it, but causal target advantage remained unproven.

## Why this matrix is identifiable

The complete crossing estimates ARR versus SRC within MON before interpreting
retirement, while still measuring credit-by-retirement interactions. MON is a
matched control for every credit and goal cell. DIRO and PREDO are compared
both against MON and against one another. LEG and FILM are matched within every
credit-retirement cell. C05/C13 bases and silent endpoint gates are excluded so
representation family and trigger opportunity cannot absorb these effects.

## Preflight gate

The three forced-replacement jobs must each:

- complete exactly one replacement after ordinary learning has begun;
- publish the new generation and reject old experience as whole rollouts;
- defer any undersized mixed-generation update;
- resume with a normal-size fresh batch and finite PPO/encoder losses;
- show zero behavior target/mode mismatch and zero publication-generation
  mismatch;
- preserve non-victim DG, graph, and goal-adapter rows;
- reset exactly one FiLM row in the FiLM jobs and none in the LEG job.

Production is blocked if this manipulation check fails. The forced mechanism
must be guarded as engineering-only and unavailable in ordinary studies.

## Analysis

Collect synchronized online summaries at 25M, 50M, and 75M. Retain separate
numerators, denominators, and event counts for commanded and shuffled target
hits; never interpret an unsupported lift ratio. Treat logit sensitivity as a
diagnostic only and add action-probability total variation as the readable
online sensitivity measure.

Primary representation outcomes:

- spatial information and active-only map cosine;
- mono-field fraction and components at 30%, 50%, and 70% of peak;
- dominant-component mass, silent units, usage entropy, and top-three activity
  share;
- graph-confidence correlation with field spread.

Primary control/graph outcomes:

- source/context-matched commanded versus shuffled target success;
- counterfactual action-probability sensitivity;
- option success, time-to-hit, and empirical reliability of routed edges;
- reliable outgoing-node coverage, SCC size, reachable-pair fraction,
  reciprocal edges, and top-three incoming-confidence share.

Retirement outcomes include eligible opportunities, residual passes,
replacement conversion, repeat replacements, generation drops/deferred
updates, FiLM resets, and post-replacement recovery time.

Run standard 10k-decision place-field telemetry at 5M, 25M, 50M, and 75M for
seed 99 and terminal telemetry for seeds 8 and 123. Run the frozen
manifest-driven target intervention at 75M for every scientific run.

## Interpretation gates

First establish the common MON baseline. A cell is not scientifically usable
if it has representation collapse, non-finite learning, replay mismatch, or an
unsupported target comparison.

- Prefer `SRC` only if it improves spatial identity in at least two seeds and
  the three-seed mean while preserving at least 90% of ARR coverage and causal
  target advantage.
- Claim useful retirement only if DIRO or PREDO improves spatial identity over
  its matched MON cell in at least two seeds without repeat replacements
  exceeding first replacements, reliable reachability falling below 80% of
  MON, or intervention advantage declining.
- Claim useful goal conditioning only from probability-space sensitivity and
  commanded-versus-matched-shuffled intervention results, not raw logit change.
- A graph is useful only if routed edges retain at least 50% empirical success,
  at least 12/16 nodes have reliable outgoing connectivity, the reliable SCC
  reaches at least 8/16 nodes in two seeds, and the top-three incoming share is
  at most 60%.

If MON itself fails, do not tune retirement. If MON has healthy landmarks but
no target advantage, the next problem is the controller objective/interface.
If MON passes and neither retirement treatment improves it, retain MON and
stop treating replacement as necessary.

## Study identity

- Schema: `intrmotiv/study/v1`
- Workflow: `1.4.1`
- Study: `saturday_batch_20260905`
- Batch: `intrmotiv_saturday_batch_20260905`
- Project: `SF_IntrMotiv_SaturdayBatch`
- Expected scientific runs: `36`
- Total training jobs including forced preflight: `39`

Preserve the validated StudySpec SHA-256 through print-only review,
submission, online analysis, spatial telemetry, and intervention artifacts.

Validated production StudySpec SHA-256:
`dcbce502053cfd14ec5ce12c88d2fa06c42f4a97367896954eccd0fe0f8831cb`.

The separate three-job engineering preflight is
`saturday_replacement_preflight_20260905`, SHA-256
`2eaf8545fa8ffb019421cec01f2a9be5c5b155fe4e2b279ddc1113916754675b`.
