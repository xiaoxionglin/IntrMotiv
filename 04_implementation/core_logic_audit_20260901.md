# IntrMotiv Core Logic Audit, 2026-09-01

## Scope

This audit traces the active `DistanceLearnerReward`, `batchnorm_relu` DG,
`BypassSS` CA3 core, fixed/global HRL graph, iterative update, and the loss
flags used by prior IntrMotiv batches. Source was read from:

`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/sf_working_directories/IntrMotiv/`

The dominant-event correction made immediately before this audit is treated as
the new source behavior. Already-running and historical jobs retain the old
behavior loaded when their processes started.

## Repair Status, 2026-09-01

The following IntrMotiv-local repairs were implemented after the audit:

| Finding | Active source status |
|---|---|
| ImageNet trunk overwritten by policy initialization | Fixed. Marked frozen trunk modules are skipped by the IntrMotiv actor initializer. Fresh runs retain torchvision weights. Old checkpoints remain load-compatible and retain their historical frozen random trunk. |
| Actor/learner DG BatchNorm mismatch | Deferred. The recommended design is one running-stat update per accepted rollout, followed by running-stat normalization in both behavior and learner policy evaluation. |
| Batch loss has zero gradient for silent rows | Fixed. Recruitment now uses a temperature-controlled pre-threshold softplus, valid-only usage classification, and normalization by the number of unused rows. |
| DG row normalization precedes SGD | Fixed. Unit-row projection runs after `optimizer.step()` while the policy lock is held. |
| Policy-buffer graph uses local half-life 1 | Fixed. Temporary selection state now uses the configured `hrl_fast_weight_half_life_options`. |
| Fallback source uses accumulated amplitude | Fixed. It selects the unit with the youngest occupied CA3 slot and returns unknown (`-1`) for an empty trace. |
| CA3 amplitude accumulation | Documented and retained. A saturated/nonaccumulating variant remains a future controlled comparison. |
| Event-gradient dilution | Deferred. Event-count normalization is the direct alternative, but changes loss scale and variance and needs a matched ablation. |
| Wrong-sign legacy decoder auxiliary | Disabled with a startup error and a repair-required source comment. |
| `distance_metric` used by PBT | Fixed minimally. It is no longer exported to PBT policy statistics, and selecting it as `pbt_target_objective` fails at startup. Ordinary diagnostic logging is unchanged. |

The complete IntrMotiv test suite passed after these repairs: 115 tests.

## Findings

### Critical, historical: the fixed `layer2_resnet18` was not ImageNet-pretrained

`ResNet18Layer2` loads ImageNet weights and sets `requires_grad=False` in
`custom_encoder.py:545-590`. Actor construction subsequently executes
`self.apply(self.initialize_weights)` in
`sample_factory/model/actor_critic.py:136-158`. Sample Factory defaults to
`policy_initialization=orthogonal`, and its initializer rewrites every exact
`nn.Conv2d` and `nn.Linear` regardless of `requires_grad`.

Therefore the trunk was first loaded from ImageNet, then orthogonally
reinitialized, then frozen. A representative historical checkpoint confirms
this for its stem:

| Quantity | Value |
|---|---:|
| checkpoint convolution norm | 8.0000 |
| ImageNet convolution norm | 12.5794 |
| checkpoint/ImageNet cosine | 0.00379 |
| mean squared difference | 0.02354 |

This invalidates the intended architectural description for all affected
`layer2_resnet18` runs. Matched same-seed comparisons can still compare HRL
changes under the same random fixed feature map, but conclusions do not apply
to the intended pretrained representation. Across-seed variance also includes
a different random frozen visual trunk.

Jannek's `pretrained_resnet` constructor appears exposed to the same actor-wide
initializer, and no experiment override of `policy_initialization` was found.
That path should be confirmed against one of Jannek's actual checkpoints before
rewriting conclusions about his report.

### Critical: actor and learner use different DG BatchNorm semantics

Inference workers call `actor_critic.eval()`, so DG BatchNorm uses running
statistics. The learner explicitly calls `actor_critic.train()`, so replay uses
minibatch statistics. At threshold 2.0-2.43, a modest normalization difference
can change whether a DG event exists at all.

The running statistics are also updated more than once per accepted learner
batch:

1. the bootstrap forward on the final observation;
2. the ordinary policy/value replay forward for each minibatch;
3. the second head-only encoder forward for each minibatch.

This means PPO can reconstruct a different DG/CA3 sequence from the behavior
sequence even before an optimizer update. Iterative decoder phases clear
encoder gradients but still mutate DG BatchNorm running statistics, so they do
not actually freeze the complete DG representation.

### High, repaired: batch unused-unit loss could not recruit a silent unit

The always-enabled batch loss identifies a DG row absent from incoming CA3 and
then minimizes

$$
L_{batch}=-\mathbb E_t\sum_j U_j a_{tj},
$$

where `a` is the post-threshold ReLU output. A truly silent unit has
`a=0` below threshold and therefore zero ReLU derivative. It receives no
gradient and cannot be revived by this loss.

Additional issues are:

- invalid/stale transitions participate in deciding whether a row is unused;
- all unused rows receive the same broad minibatch objective;
- there is no coefficient, so its scale grows with the number of marked rows;
- it is averaged over transitions, rather than normalized by useful events.

This loss was enabled in nearly every baseline, persistence, structural,
manager, and goal-condition batch. It did not provide the anti-collapse
mechanism attributed to it and may reinforce rows only after they happen to
cross threshold.

### High, repaired: DG rows were normalized before, not after, `optimizer.step()`

`custom_learner.py:2738-2744` renormalizes DG linear rows to norm one and then
calls `optimizer.step()` at `2762-2763`. The synchronized actor therefore sees
the post-update, non-unit rows. The next encoder update normalizes them again.

This violates the rotation-only interpretation of DG learning. Training-mode
BatchNorm partly hides row-scale changes, while actor eval mode uses stale
running statistics and does not necessarily hide them.

### High, repaired: global-graph actors halved graph confidence at target selection

`update_option_state_from_policy_graph()` expands the global graph and calls
`update_hrl_state(... persistent_fast_weights=True,
fast_weight_half_life_options=1.0)`. On every option reset, the temporary
expanded visits and edge confidence are multiplied by `0.5` before target
selection and deadline calculation.

The temporary graph is discarded afterward, so this is not persistent
forgetting. It instead doubles the effective confidence and minimum-visit
thresholds seen during selection. With configured edge threshold `0.5`, a
global edge below confidence `1.0` is treated as infeasible at reset. This can
reduce route availability and increase bootstrap timeouts.

### High, repaired: CA3 fallback source favored accumulated amplitude, not recency

When no DG is currently active, `source_from_trace()` selects

$$
\arg\max_j\sum_k |S_{j,k}|.
$$

Because CA3 adds repeated DG activity into the register, this favors a broad or
persistently active unit. It is not the most recent DG landmark. A wrong option
source contaminates `T_ctrl`, deadlines, and controllability statistics.

### High, historical: the removed `2R` mask reinforced established activity

Prior processes used

$$
[p_t[j]=0][\#S_{t,j}\ge2R]
$$

for the main encoder update. This selected sustained or repeated same-unit
activity, while distance reward was defined for a new onset relative to other
DG units. It could suppress fresh field formation, reinforce broad/multi-field
rows, and concentrate learning on already active units.

The active source now stores one dominant behavior onset and uses only that row
for the distance-scaled update. Historical results remain affected.

### Medium: CA3 amplitudes accumulate while distances use only occupancy

`BypassSS` shifts the previous CA3 state and adds the current DG output into
the first `R` slots. Sustained activation therefore accumulates amplitudes
rather than producing a saturated binary pulse. Progression and temporal
distance ignore magnitude, but the policy decoder consumes the full
amplitude-valued CA3 state.

Consequences include scale growth for broad fields, a decoder bias toward
persistent units, and disagreement between the information used for temporal
reward and the information supplied to the policy.

### Medium: sparse event gradients are diluted by all transitions

The reward-weighted encoder loss is averaged over all valid transitions, not
over dominant DG events. If useful events occupy 0.5% of transitions, their
gradient is diluted by roughly 200 relative to an event-normalized objective.
Its scale also changes with `L`, threshold, behavior, and collapse state. This
can make the intended learning rule vanish precisely when DG events become
sparse.

### Medium: iteration-2 auxiliary losses did not implement their apparent metrics

Iteration 2 enabled population usage, density, collision, and multi-activation
losses. In that historical code:

- multi-activation used the erroneous `2R` mask rather than simultaneous
  onset candidates;
- density targeted mean post-ReLU amplitude, not activation probability;
- collision penalized summed amplitude above one, not the number of active
  units;
- population terms included invalid transitions;
- post-ReLU usage/density terms could not revive a below-threshold row.

The shadow CA3 predictor detached CA3, DG, and target labels before prediction
and did not enter actor/manager decisions. It trained only its own predictor
network, so it is unlikely to have caused DG collapse or improved navigation.

### Medium, disabled: optional decoder event loss has the wrong optimization sign

`_extra_decoder_loss()` returns a positive clipped probability-ratio term and
adds it to the minimized decoder loss. If enabled, this suppresses rather than
reinforces actions associated with DG events. Main IntrMotiv batches explicitly
used `extra_decoder_loss=False`, so this is dormant rather than an explanation
for their failures.

### Medium: distance metrics are not place-field or exploration metrics

`distance_metric` averages all `F^2` pairwise CA3 progression differences,
including inactive sentinel values. It can increase when a small active subset
is far from many inactive rows. `distance_metric_masked` zeros inactive pairs
but still divides by `F^2`, so sparsity changes its denominator implicitly.

These metrics are useful diagnostics only with activity statistics. The active
source rejects `distance_metric` as a PBT target and does not route it into PBT
policy statistics. Historical best-checkpoint configurations may still use it;
that remaining use should be removed when those experiment manifests are
revised.

### Diagnostic: logged DG activity is learner-mode activity

`dg_density`, multi-activation, and silent fraction are computed from the
learner's training-mode head output and do not apply the valid mask. They need
not match eval-mode actor DG events that generated the rollout. This limits
what historical W&B curves can establish about behavior-time landmark density.

## Interpretation Of Prior Batches

The most defensible conclusions are relative comparisons among matched runs
with the same seed and feature initialization. Claims that depend on the
absolute biological or representational quality of a fixed ImageNet layer-2
code are not valid yet.

The strongest likely contributors to prior failure are:

1. random rather than pretrained frozen visual features;
2. actor/learner DG BatchNorm mismatch;
3. the historical `2R` event-mask mismatch;
4. ineffective post-ReLU silent-unit recruitment;
5. local half-decay of the policy graph during target selection.

Orthogonal recruitment, global punishment, row repulsion, temporal exclusion,
frontier planning, and empirical HER were layered on top of these issues. Poor
results from those batches do not cleanly reject the added mechanisms.

## Recommended Fix Order

1. Preserve and test ImageNet weights after actor construction. Add a checksum
   or exact state comparison against torchvision before the first rollout.
2. Make DG normalization behavior-identical during sampling and replay. Update
   running statistics exactly once from accepted rollout observations, outside
   PPO replay and iterative decoder phases.
3. Move row normalization after `optimizer.step()` and test synchronized row
   norms.
4. Replace batch post-ReLU recruitment with an explicit pre-threshold or
   structural rule, with a coefficient and valid-only mask.
5. Remove temporary `half_life=1` graph decay from policy-buffer selection and
   use recency, not accumulated CA3 amplitude, for fallback source.
6. Decide whether CA3 should be binary/saturated or amplitude-valued and make
   reward, graph, and worker consume the same semantics.
7. Normalize event-conditioned losses by accepted event count and log both
   actor-eval and learner-train DG activity during transition.
8. Only then rerun a minimal matched baseline versus direct HRL comparison
   before adding more auxiliary mechanisms.

## Required Regression Tests

- ImageNet trunk equality before rollout and after checkpoint round-trip;
- actor-eval versus learner-replay DG output and CA3 sequence equality;
- one BatchNorm-stat update per accepted rollout and none in decoder-only phase;
- exact post-optimizer DG row norms;
- nonzero gradient for a below-threshold unused row under the replacement
  recruitment rule;
- no local graph half-decay during policy-buffer target selection;
- fallback source selects the most recent valid exclusive DG event;
- event-loss signs verified with one synthetic optimizer step;
- distance metrics tested under controlled active/silent populations.
