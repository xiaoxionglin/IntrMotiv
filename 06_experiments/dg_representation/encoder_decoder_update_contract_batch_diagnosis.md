# Encoder–decoder update-contract batch diagnosis

Date: 2026-09-05  
Study: `encoder_decoder_update_contract_preflight_20260905`  
Study SHA-256: `1cc2a8debf8cfc49a9dc5d17b0b270180f3b3aaa47edbde08e536b10363cc291`

## Outcome

The new one-forward update-contract batch is not a valid production result.
Eight of ten seed-99 jobs reached 75M steps. Both ARR–DIR jobs failed after
their first landmark replacement. More importantly, matched MON controls show
that the new common learning contract severely degraded DG representation and
graph formation relative to the preceding workflow-1.4.1 source-credit batch.

The batch still establishes three useful facts:

1. The ordinary one-forward execution is numerically stable in the absence of
   an unlucky post-replacement batch, and terminal credited-row replay match is
   approximately 97–98% in most surviving cells.
2. Global generation rejection is unsafe after replacement because it can
   leave too few valid trajectories for PPO advantage normalization.
3. The new common contract—most plausibly the `running_consistent`
   normalization semantics—reduces DG activity and landmark diversity enough
   to destroy the controllability graph. Source-centered credit does not
   compensate for this.

## Runtime audit

| Cell | Result | Steps | Replacements |
|---|---:|---:|---:|
| ARR MON | completed | 75M | 0 |
| ARR DIR-silent | failed | 8.78M | 1 |
| ARR DIR-open | failed | 10.13M | 1 |
| ARR PRED-silent | completed | 75M | 0 |
| ARR PRED-open | completed | 75M | 3 |
| SRC MON | completed | 75M | 0 |
| SRC DIR-silent | completed | 75M | 0 |
| SRC DIR-open | completed | 75M | 0 |
| SRC PRED-silent | completed | 75M | 8 |
| SRC PRED-open | completed | 75M | 2 |

The dedicated full-history audit passed 7/10 cells. Pooled credited-row replay
match was `0.9308`, below the prespecified `0.95` gate. The terminal 70–75M
window was better: approximately `0.97–0.98` in seven survivors, with ARR
PRED-silent remaining at `0.890`.

The two DIR failures have the same exact sequence:

- the first replacement increments the global representation generation and
  resets one FiLM row;
- the next recorded batch rejects `81.71%` (DIR-silent) or `90.97%`
  (DIR-open) of samples as stale-generation experience;
- `torch.std_mean` receives an effectively empty or singleton valid selection;
- PPO losses become NaN and actor inference fails while sampling from the NaN
  action distribution.

Thus the replacement transaction itself completed correctly; the bug is that
the learner proceeds with PPO normalization after global generation filtering
has removed nearly the entire batch. PRED replacements happened to survive
their post-replacement batches, but the same failure remains possible there.

## Matched comparison with the preceding batch

The clean comparison uses seed-99 C15-FiLM MON cells, for which goal-adapter
reset and retirement never execute. The preceding study is
`source_credit_retirement_20260904` with SHA-256
`aa34bb2ef868df37dbafc38e7c4b8dc5c9cc684e5f8dba5e05a586d068ad8a0d`.

### Terminal online metrics, 70–75M

| Credit | Batch | DG density | Usage entropy | Option success | Action sensitivity | Largest SCC | Reachable pairs | Top-3 incoming share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARR | previous | 0.0282 | 0.9916 | 0.4088 | 0.0146 | 15.99 | 0.999 | 0.558 |
| ARR | new | 0.0125 | 0.8274 | 0.0610 | 0.0008 | 1.86 | 0.120 | 0.947 |
| SRC | previous | 0.0285 | 0.9940 | 0.4623 | 0.0190 | 15.67 | 0.979 | 0.564 |
| SRC | new | 0.0112 | 0.8237 | 0.0386 | 0.0005 | 1.01 | 0.048 | 0.997 |

Commanded and source/context-shuffled target-hit rates remain essentially
equal in both studies. The old graphs therefore were not genuinely
target-controllable, but the new contract additionally destroys most event
reachability and target-conditioned action variation.

### 75M online-spatial snapshots

| Credit | Batch | Mean spatial information | Distinct peak bins | Reliable edges | Largest SCC | Reachable pairs |
|---|---|---:|---:|---:|---:|---:|
| ARR | previous | 0.108 | 13 | 70 | 16 | 1.000 |
| ARR | new | 0.011 | 11 | 27 | 1 | 0.167 |
| SRC | previous | 0.109 | 16 | 78 | 15 | 0.938 |
| SRC | new | 0.009 | 11 | 17 | 1 | 0.088 |

Mean summed DG activity across rows fell from about `0.44–0.45` to
`0.19–0.20`, while the top-three share of that activity rose from
`0.23–0.24` to `0.44–0.47`. This explains the sparse, funnel-like graphs.
The apparent increase in mono-field fraction in some new cells is not evidence
of improved landmarks: it accompanies roughly tenfold lower spatial
information and a collapsed graph.

Physical coverage AUC increased in the new MON cells. This is not navigation
success: target-hit lift is `1.002` for ARR and `1.008` for SRC, and action
sensitivity is nearly zero. The policy is covering space without responding
meaningfully to the commanded target.

## Factor interpretation

### ARR versus SRC

At 5–8M, SRC gives more balanced landmark use than ARR: usage entropy is
roughly `0.87–0.90` versus `0.61–0.71`, and the top-three DG activity share is
roughly `0.37–0.44` versus `0.61–0.71`. This benefit disappears in the matched
75M MON comparison. SRC does not improve terminal spatial information,
controllability, target-hit lift, or action sensitivity.

### DIR

DIR cannot be evaluated scientifically. Both ARR cells crash after their first
replacement, while neither SRC cell ever replaces a node. The silent/open
comparison therefore lacks a manipulation check.

### PRED

PRED replacement counts are strongly path-dependent and do not show a clean
open-gate effect: ARR has 0 silent versus 3 open replacements, whereas SRC has
8 silent versus 2 open. No PRED condition establishes target control. SRC
PRED-silent retains the strongest surviving graph, but terminal target lift is
only `1.008`; ARR PRED-silent's high mono-field score co-occurs with 23% silent
units, zero option success, and a dead graph, so it represents collapse rather
than useful localization.

## Likely causes

The matched MON configurations differ in the new common update implementation
and in the required `--dg_batchnorm_semantics=running_consistent`; goal-adapter
reset is inactive in MON. Because ARR and SRC degrade similarly, the credit
recipient is not the primary cause.

The leading hypothesis is normalization semantics. The new DG operates at
less than half the previous activity density and with substantially lower
usage entropy. Exact replay match reaches approximately 98% late in training,
so discarded replay credit alone is too small to explain the terminal
collapse. The next hypothesis is a difference in the single-forward encoder
loss/gradient routing that was previously masked by the second forward or
gradient-manipulation path. These mechanisms are not separated by the current
study.

A deterministic gradient test identifies a specific part of that second
hypothesis. Training-mode BatchNorm and fixed running-stat normalization can
produce identical normalized forward values but very different DG gradients.
On non-centered synthetic features, fixed-stat positive-credit gradients were
strongly aligned with the common feature mean and with one another, whereas
differentiating through batch moments removed most of that common mode. The
two flattened gradients were nearly orthogonal. The old learner-only
BatchNorm Jacobian was therefore an unplanned centering/competition mechanism;
removing it may contribute directly to DG-row convergence even if statistics
are made perfectly current.

More specifically, the new running statistics are updated during the
encoder-active forward, after which the optimizer changes and renormalizes the
DG rows. The synchronized actor therefore receives post-update projection
weights paired with statistics estimated for the pre-update weights. The next
learner forward also begins from that lagged pair. A hard threshold of `2.43`
amplifies small mean/variance errors in the distribution tail. This can make a
slightly under-active row receive fewer temporal-credit events and become even
less active, producing the observed loss of usage entropy.

The old seed-99 MON snapshots had mean per-row activity of approximately
`0.028`; the new snapshots have `0.011–0.013`. For independent rows this
changes the probability of at least one landmark event from roughly 37% to
17–19%. Directed transition opportunities depend on pairs of events and can
therefore fall superlinearly, consistent with the observed reduction from
`70–78` reliable edges to `17–27`.

This event-rate argument explains the between-batch scale change but is not a
complete graph model. Across the eight surviving new cells, total activity has
only weak correlation with reliable-edge count; retirement history and which
identities fire remain important. Across old and new snapshots together,
activity correlates strongly with reachability largely because the batches
occupy two distinct regimes. Event scarcity is therefore a mediator supported
by the data, not proof that increasing density would restore control.

Sample removal is a separate issue. Global stale-generation filtering is the
direct cause of the two crashes, but it cannot explain the MON regression,
where no generation changes or stale-generation rejections occur. Replay
activity intersection removes only encoder-credit events, not MON PPO
samples. It discards a substantial fraction early in training and may amplify
path dependence, but only about 2% of scheduled credit is rejected in most
terminal windows. It is therefore a plausible secondary contributor rather
than the primary explanation for the persistent common collapse.

## Required next checks

1. Fix post-replacement handling before any recruitment rerun: skip an update
   when the valid sequence count is insufficient, force policy
   synchronization, and discard/flush complete old-generation rollouts rather
   than normalizing PPO over fragmented, almost-empty minibatches. Also guard
   all advantage and branch-loss reductions against zero or one valid sample.
2. Run a fixed-rollout gradient audit on real frozen ResNet features, comparing
   differentiable batch moments, fixed projected moments, and explicitly
   input-centered fixed moments.
3. Test a post-update-statistics implementation that caches detached visual
   features, recomputes only the DG linear logits after the encoder optimizer
   step, updates running statistics once, and publishes weights and statistics
   atomically. This preserves one differentiable DG forward and avoids the old
   second full encoder pass.
4. Run the three-cell ARR-MON normalization preflight defined in
   `05_plans/landmark_contract_recovery_plan.md`: `legacy_batch`,
   `running_poststep_atomic`, and `input_centered_atomic`.
5. Do not tune DIR or PRED eligibility from this batch. Their outcomes are
   downstream of the common representation regression, and DIR is additionally
   censored by the post-replacement crash.

## Artifact notes

Authoritative training and raw telemetry remain under the NEMO2 workspace.
The standard spatial collector required an analysis-only StudySpec copy whose
online targets were changed from the offline rollout schedule
`5/15/30/50/75M` to the actually generated compact-snapshot schedule
`5/25/50/75M`. This workaround has a different fingerprint and is not a
replacement for either validated study. The target-list mismatch should be
fixed in the workflow contract before final archival analysis.
