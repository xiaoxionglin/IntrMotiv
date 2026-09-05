# Landmark Contract Recovery Plan

## Purpose

Recover a stable, domain-general landmark process before running another
recruitment study. The latest batch confounded three mechanisms:

1. normalization of the DG landmark scores;
2. temporal credit to DG versus PPO credit to the controller;
3. discrete landmark replacement and replay invalidation.

These must be correct independently. Recruitment is maintenance of an already
meaningful landmark vocabulary; it must not be used to repair a broken encoder
or to manufacture a dense graph.

## Evidence from the latest batch

- Both ARR–DIR jobs failed immediately after their first replacement. The next
  batch rejected `81.7%` or `91.0%` of samples by global generation and then
  produced NaN PPO advantages.
- The larger scientific regression also occurs in MON, which has no
  replacements and zero stale-generation rejection. Sample removal therefore
  explains the crashes but not the common representation collapse.
- In matched seed-99 MON cells, mean DG activity fell from about `2.8%` per row
  to `1.1–1.3%`, usage entropy fell from about `0.99` to `0.82`, spatial
  information fell from about `0.109` to `0.009–0.011`, and the largest
  reliable SCC fell from `15–16` nodes to about `1–2`.
- Terminal commanded-target hit rates remain equal to shuffled-target hit
  rates in both the old and new batches. The previous dense graph was not a
  controllability result; restoring edge count alone is not the objective.

## Core invariants

### Landmark-state invariant

At every published policy version, DG projection weights and the normalization
statistics used with those weights form one atomic representation state.
Actors and learners given the same observation and representation state must
produce the same DG scores and active rows.

### Credit invariant

Keep the explicit push–pull contract only:

\[
r_{enc}=\beta d,\qquad r_{dec}=\beta(E-d).
\]

One differentiable DG/CA3 forward supplies encoder objectives. PPO receives a
CA3 stop-gradient. No PPO-conditioned DG loss, gradient reversal, dense
shaping, or second full encoder forward is introduced.

### Replacement invariant

A replacement is a discontinuous representation-version change. Experience
from the old version is discarded at complete recurrent-rollout boundaries;
it is never mixed into a partially valid PPO minibatch.

### Evaluation invariant

Progress is ordered:

1. stable and sufficiently expressed landmark identities;
2. target-conditioned action and target-hit advantage;
3. reliable directed graph and planning;
4. retirement quality.

Later stages cannot compensate for failure at an earlier stage.

## Implementation corrections

### Mechanism tests completed

`06_experiments/test_landmark_contract_hypotheses.py` provides four small,
deterministic tests. They confirm that:

- a singleton valid advantage produces the same NaN sample standard deviation
  and PyTorch warning seen in both failed DIR jobs;
- element-wise generation filtering can leave only six complete 64-step
  sequences and must not be treated as a normal PPO batch;
- for anisotropic, non-zero-mean frozen visual features, a modest DG row
  rotation paired with pre-update statistics changes more than `0.2%` of all
  threshold decisions at `2.43`, which is a large fraction of sparse events;
- recalibrating statistics on the updated projection restores the defining
  zero-mean/unit-variance invariant, and the measured activity-rate drop
  predicts a greater than fourfold loss of event-pair opportunities.
- even when batch-stat and fixed-stat normalization have exactly identical
  forward values, their encoder gradients are almost orthogonal. On a
  deterministic non-centered feature test, fixed-stat gradients align with
  the common feature mean and with one another, while differentiable batch
  moments remove most of that common mode. Thus post-step statistic alignment
  is necessary but may not be sufficient.

These are mechanism tests, not deployed-runtime proof. The production modules
must implement the same invariants and pass their existing source-tree tests
plus the forced-replacement Slurm check below.

### 1. Post-step atomic normalization

Add a new DG normalization mode, `running_poststep_atomic`.

For an encoder-active minibatch:

1. Run the single differentiable DG/CA3 forward with the currently published
   weights and frozen running statistics.
2. Cache the detached visual projection inputs used by valid sequences.
3. Backpropagate the selected encoder/controller losses once and step the
   optimizer.
4. Renormalize DG weight rows.
5. With the cached inputs and updated DG weights, recompute only the cheap
   linear DG logits under `no_grad`.
6. Initialize or update running mean/variance once from those post-update
   logits.
7. Publish the updated weights and statistics together under the policy lock.

Actors never update statistics. Decoder-only phases update neither DG weights
nor DG statistics. The post-step logit calculation is not a second visual,
CA3, policy, or differentiable encoder forward.

Checkpoint a shared representation-state generation for both weights and
statistics. Log their generations separately and require equality whenever a
policy is published.

Do not change the DG intercept in this diagnostic. If a correctly calibrated
`2.43` threshold supplies too few landmark events, event rate becomes an
explicit later architectural choice rather than being recovered through a
BatchNorm mismatch.

### 1a. Make the normalization gradient explicit

Legacy training-mode BatchNorm did more than normalize the forward scores. Its
differentiable batch mean and variance centered and coupled the DG gradients.
The new running-stat forward removes that Jacobian. With positive temporal
credit and non-centered ResNet ReLU features, fixed-stat gradients can pull
many DG rows toward the same common feature direction. This hidden change is a
plausible cause of the observed loss of usage entropy.

The final algorithm must not depend accidentally on a learner-only BatchNorm
Jacobian. Test an explicit, actor/learner-identical alternative in which the
frozen visual projection input is centered by a checkpointed running feature
mean that is independent of DG weights, DG rows remain unit-normalized, and
any projected scale statistics are published atomically after the DG update.
This removes the common feature direction without making one sample's
landmark identity depend on the other samples in a learner minibatch.

Before training, run one fixed-rollout gradient audit on real frozen ResNet
features. Compare legacy differentiable batch moments, current fixed projected
moments, post-step-aligned fixed moments, and input-centered fixed moments.
Record forward equality, gradient cosine, alignment with the feature mean,
pairwise DG-row gradient cosine, and the one-step change in activation masks.

### 2. Generation barrier after replacement

Replace per-timestep global-generation masking with a rollout boundary:

1. Apply replacement after the current accepted batch has finished training.
2. Atomically update the DG row, optimizer row, graph/PRED state, FiLM row,
   normalization row, and representation generation.
3. Force policy publication to actors.
4. Reject every complete incoming rollout whose stored generation is not the
   current generation. Do not compute graph updates, bootstrap values,
   internal rewards, GAE, DG statistics, or gradients from it.
5. Resume optimization only after enough complete current-generation
   recurrent sequences are available for a normal minibatch.

Log dropped rollouts, dropped decisions, generations observed, and decisions
until learning resumes. Guard every advantage and branch-loss reduction
against fewer than two valid values, but treat that guard as an invariant
violation rather than a normal training path.

Per-landmark selective invalidation is deferred. It could save unaffected
experience but requires more semantic bookkeeping; replacement is rare enough
that a full rollout barrier is the minimal reliable mechanism.

### 3. Credit diagnostics

Retain scheduled/applied credit telemetry, but separate:

- behavior row still active under the identical published representation;
- behavior row changed because of ordinary policy lag;
- rollout rejected by a representation-generation barrier.

Report both cumulative match and the most recent 1M-step match. Do not conflate
discarded encoder-credit events with discarded PPO samples.

## Stage A: causal normalization preflight

### Scope decision

Normalization is infrastructure rather than a primary research axis. Use the
cleaned single-forward `legacy_batch` implementation as the production
baseline. Retain the explicit DG/controller gradient boundary, source-credit
alignment, whole-rollout generation barrier, and atomic replacement reset, but
do not add a custom BatchNorm backward, surrogate gradient, or further
normalization mechanism.

`running_poststep_atomic` and `input_centered_atomic` remain diagnostic cells
in the already-running 5M comparison. Promote neither unless it shows a large,
unambiguous advantage over legacy behavior; a merely cleaner normalization
contract is not sufficient. The actor/learner BatchNorm-mode difference is a
documented engineering limitation of the baseline and is outside the current
research focus.

### Execution status (2026-09-05)

The generation barrier and both atomic normalization modes are implemented in
the authoritative NEMO2 checkout. After a runtime-only scope error was exposed
and fixed, the complete IntrMotiv suite passes `220/220`; the workflow suite
passes `26/26`.

A 4,096-decision audit using real frozen-ResNet features from the stable C15
ARR-MON checkpoint confirmed the proposed mechanism. Fixed projected-moment
gradients had mean absolute cosine `0.951` with the feature mean and pairwise
DG-row gradient cosine `0.901`. Legacy differentiable BatchNorm reduced these
to `0.044` and `0.018`; explicit running feature-mean subtraction reduced them
to `0.035` and `-0.043` while reproducing the legacy forward activation mask
exactly on the audited rollout. Post-step moment alignment alone therefore
fixes publication consistency but does not remove the common-mode gradient.

The first submitted 5M launch (`7989323`–`7989325`) was stopped deliberately:
the first learner update exposed a `NameError` in post-step diagnostic logging.
No result from that launch is scientifically usable. The fix now has a focused
regression test and is included in the `220/220` passing suite.

A fresh 200k-step runtime smoke then completed successfully in jobs `7989334`
–`7989336`. Both atomic modes had finite training and zero behavior replay,
stale-generation, and publication-generation mismatch. POST retained the
common-mode pathology (credited-row replay match `0.696`, normalized-logit
mean absolute value `1.149`), while CENTER reached replay match `0.951` and a
normalized-logit mean absolute value of `0.101`. The only non-finite logged
value was the unrelated planning diagnostic `hop_count_mean`: multiplying an
infinite no-route sentinel by a zero route mask does not mask it. No loss,
activation, gradient, or parameter was implicated.

The uncontaminated 5M launch uses
`hpc_runs/studies/landmark_normalization_contract_preflight_v2.study.json`,
SHA-256
`21dbbd4e7503dd95a9ab734024c059378ec60d544e810300bb71256cfab93b3c`.
It passed print-only review and submission audit; NEMO2 jobs `7989346`
(legacy), `7989347` (post-step atomic), and `7989348` (input-centered atomic)
were running at the time of this update.

After the fixed-rollout gradient audit, create a three-run, seed-99, 5M-step
C15-FiLM ARR-MON study. Both ARR and SRC suffered the common regression, so
one credit condition suffices to select the normalization mechanism.

| Factor | Levels |
|---|---|
| Encoder credit | `ARR` |
| DG normalization | `legacy_batch`, `running_poststep_atomic`, `input_centered_atomic` |

Suggested identifiers:

- Study: `landmark_normalization_contract_preflight_20260906`
- Batch: `intrmotiv_landmark_normalization_contract_preflight_20260906`
- Project: `SF_IntrMotiv_LandmarkNormalizationContractPreflight`
- Schema/workflow: `intrmotiv/study/v1`, `1.4.1`
- Expected runs: `3`

Run all three fresh under the same corrected code. Use the completed current
`running_consistent` ARR-MON run as a diagnosed reference, not as a formal
StudySpec cell. Disable replacement (`MON`) so normalization is the only
representation-state manipulation.

Required telemetry:

- per-row raw-logit and normalized-logit mean/variance;
- per-row activation probability, usage entropy, silent fraction, and top-three
  activity share;
- weights generation, statistics generation, and publication mismatch;
- actor/learner DG output agreement on an identical stored input;
- scheduled/applied credit and recent-window replay match;
- spatial information, active-only map cosine, peak diversity, and field
  components from the 5M compact snapshot;
- target action sensitivity and matched target/shuffled hit numerators, without
  requiring control to be learned by 5M.

Engineering gates:

- finite losses and zero behavior target/mode mismatch;
- one differentiable DG forward and exactly one post-step calibration in the
  new mode;
- zero actor-side or decoder-only normalization updates;
- weights/statistics generation equality at every publication;
- exact actor/learner DG equality for identical weights, statistics, and input
  in both running-stat modes; `legacy_batch` is retained as the deliberately
  mismatched diagnostic reference;
- correct ARR branch selection and identical scheduled event/reward mass
  between normalization levels before activity intersection.

Selection gates for either atomic candidate:

- all 16 rows active in the 100k-sample snapshot, normalized usage entropy at
  least `0.90`, and top-three activity share at most `0.50`;
- recent 1M credited-row replay match at least `0.95`;
- active-unit mean spatial information at least half of the matched
  `legacy_batch` value, preventing another order-of-magnitude regression;
- graph and control metrics are reported but are not Stage-A selection gates.

Every cell must pass the numerical and routing checks. At least one atomic
candidate must pass the selection gates without relying on learner-only batch
semantics; otherwise stop and revise the landmark score rather than selecting
legacy behavior as the final algorithm.

The purpose of `legacy_batch` is causal localization, not selection as the
final algorithm. If it alone recovers the old operating regime, the old
BatchNorm Jacobian was functioning as an undeclared representation-learning
mechanism. Make the necessary centering/competition explicit rather than
restoring actor/learner mismatch.

## Stage B: forced replacement preflight

After Stage A selects a normalization contract, run an engineering-only forced
replacement test before any DIR/PRED study:

- trigger exactly one known-row replacement at an accepted batch boundary;
- verify the complete replacement transaction and one FiLM-row reset;
- verify all old-generation rollouts are dropped whole;
- verify no optimizer/statistics/graph update occurs while waiting;
- verify learning resumes on a full current-generation batch with finite PPO
  and encoder losses.

Use unit tests plus one ordinary 1M-step ARR job. Credit recipient does not
change the generation barrier, so a second SRC job would not test another
mechanism. This is a runtime manipulation check, not a scientific factor.

## Stage C: representation and control study

Only after Stages A and B pass, run C15-FiLM MON with `{ARR,SRC}` and seeds
`{8,99,123}` to 25M. No recruitment factor is included.

The representation gate requires improved landmark identity without relying
on graph density:

- no silent or dominant-row collapse;
- spatial information and component concentration do not regress from the
  preceding stable C15 reference;
- peak diversity remains broad across DG identities;
- results agree in at least two seeds and in the three-seed mean.

The control gate is reported separately:

- target action sensitivity;
- commanded versus source/context-matched shuffled target hits;
- option success and time-to-hit;
- reliable-edge empirical success.

If representation passes but target lift remains one, the next problem is the
controller objective/state interface—not recruitment. If representation fails,
change the landmark formulation before touching the manager.

Continue a configuration to 75M and multiple retirement conditions only if it
passes the 25M representation gate and shows a target-conditioned signal above
its matched shuffled control. Otherwise use the 25M result to redesign the
controller interface rather than spending a full production batch.

## Retirement policy

Do not run another DIR/PRED matrix until the common MON contract passes.
Thereafter, reintroduce one retirement rule at a time. A retirement treatment
must improve representation identity while preserving target-conditioned
control; edge count or SCC size alone is insufficient.

DIR remains an environment-specific controllability heuristic. PRED remains
the more domain-general diagnostic because it tests whether one landmark
identity has inconsistent consequences across predecessor contexts without
using physical geometry. Neither rule should influence the encoder loss in the
contract-recovery stages.

## Big-picture interpretation

The transferable object is not a place field or a dense transition graph. It
is a stable event identity whose consequences become predictable when paired
with a commanded subgoal. The graph is evidence accumulated over those stable
identities; retirement only edits the vocabulary when evidence shows an
identity is unhelpful. Keeping these roles separate prevents environment-
specific patches from becoming the algorithm.
