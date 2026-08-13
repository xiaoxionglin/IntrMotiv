# IntrMotiv Loss Recommendations From Jannek's Report

Date: 2026-06-05

This note interprets the evidence from Jannek's report at the algorithm level: which intrinsic losses appear important, which are redundant, what should be kept, and what should be folded into a cleaner internal DG-CA3 objective.

## Main Evidence From The Report

The report gives three important empirical signals.

First, punishment-style encoder shaping produced the sparsest DG activity and the broadest behavioral exploration. The report states that punished agents explored a larger fraction of the arena per episode, and later argues that this broader behavioral sampling may help produce better neural representations across the map.

Second, punishment also made DG activity more landmark-like, but with a cost. It sharpened selectivity by suppressing weak activations, but also produced noisy and fragmented activation maps: individual sequences could activate in multiple disconnected locations.

Third, the batch loss seems especially important. The report says the batch loss significantly improved participation ratio and place-field coverage, likely because it caused more DG sequences to become active. It also suggests that when more sequences are recruited, each sequence can cover less physical space, allowing better tuning.

The report gives weaker evidence for the other auxiliary losses. Multi-activation and unused-sequence losses are intuitively reasonable, but the report does not show the same clear performance improvement for them. Intercept manipulation also did not strongly explain the main outcomes.

## Which Losses Seem Most Important

### 1. Sparse/Punishment Pressure

This seems important and should be kept.

Algorithmic role:

- suppresses diffuse DG activity;
- forces only strong visual features to become landmarks;
- reduces overlap between DG units;
- encourages the policy to move rather than obtain reward from excessive local activation.

Evidence:

- punishment produced sparse DG activity;
- punishment was associated with broader exploration;
- suppression of DG activity is named in the report conclusion as a key factor.

Risk:

- too much punishment can under-use DG units;
- landmarks can become fragmented or noisy;
- if activity becomes too sparse, the policy may lose useful internal feedback.

Recommendation:

Keep punishment-like shaping as the default activity-control regime, but pair it with a resource-use term so the model does not solve the objective by turning DG off.

### 2. Batch Loss / Unused Resource Pressure

This seems highly important and should be kept.

Algorithmic role:

- prevents collapse onto a small number of DG units;
- encourages the model to use the available sequence reservoir;
- improves coverage by distributing representational load;
- makes each landmark field smaller and more specific because more landmarks are available.

Evidence:

- report says batch loss significantly improved participation ratio and place-field coverage;
- report suggests more active sequences allowed each sequence to cover less physical space;
- conclusion emphasizes both suppressed activity and encouragement to utilize all sequences.

Risk:

- if too strong, it may force units to activate even when observations do not support clean landmarks;
- could create artificial allocation pressure unrelated to behavior.

Recommendation:

Keep batch/resource-utilization pressure. This is probably the most important auxiliary loss to preserve.

## Which Losses Seem More Redundant

### 1. Unused-Sequence Loss

This is partially redundant with batch loss.

Both unused-sequence loss and batch loss try to recruit inactive DG units. The difference is local versus minibatch/global pressure:

- unused-sequence loss rewards activation of a sequence that was previously inactive;
- batch loss rewards sequences that were not used over a minibatch.

The batch loss has stronger evidence and directly targets population utilization. The unused-sequence loss may still help, but it risks rewarding activation for its own sake.

Recommendation:

Do not keep unused-sequence loss as a separate hand-tuned auxiliary loss unless ablations show it adds something beyond batch loss. Fold its intended function into a broader resource-utilization term.

### 2. Multi-Activation Loss

This is conceptually valid but overlaps with sparsity/punishment.

Multi-activation loss discourages several DG sequences from activating at the same time. That is important because simultaneous activations make landmarks less distinct and waste sequence capacity. However, general sparse punishment already pushes in the same direction.

The difference is that sparse punishment controls total activity, while multi-activation loss specifically penalizes collisions at one time step.

Recommendation:

Keep the idea, but make it a collision penalty inside the DG-CA3 internal objective rather than a separate encoder auxiliary loss. It should be small and targeted:

```text
penalize simultaneous new activations, not ordinary sparse continuation activity
```

### 3. Intercept/Sparsity Tuning

This should not be treated as an algorithmic loss.

The report suggests changing activation intercept affected activity level but did not clearly explain or improve the key representation/behavior outcomes.

Recommendation:

Treat intercept as a hyperparameter for DG activation threshold, not as the main mechanism. The algorithm should not rely on delicate intercept tuning to get coverage.

## What To Keep

For the next clean baseline, keep only three intrinsic pressures:

1. Transition-distance reward.

   This remains the core DG-CA3 intrinsic motivation signal. It links behavior to sequence timing and makes exploration part of internal map formation.

2. Sparse/punishment pressure.

   This keeps DG landmarks selective and prevents diffuse activation.

3. Population/resource utilization pressure.

   This prevents collapse onto a few DG units and encourages coverage of the available sequence reservoir.

A small collision penalty can be included, but it should be considered part of the sparse allocation rule, not a separate major objective.

## What To Remove Or Demote

Demote these from independent losses:

- unused-sequence loss;
- multi-activation loss as a separate encoder loss;
- extra decoder loss if it duplicates the same collision/redundancy penalty;
- intercept sweeps as a substitute for a stable objective.

These should become ablations, not default components.

## What Should Become An Internal Loss

The current design has several auxiliary losses that patch different failure modes. A cleaner algorithm should express them as terms of one internal DG-CA3 objective.

The internal objective should evaluate the quality of a DG-CA3 transition:

```text
internal_objective =
    transition_distance_term
  + population_usage_term
  - collision_term
  - overactivity_term
```

Where:

- `transition_distance_term` rewards useful temporal spacing between newly activated sequence and prior sequence activity.
- `population_usage_term` rewards balanced use of DG/CA3 sequence resources over a rollout or minibatch.
- `collision_term` penalizes multiple new sequence activations at the same time step.
- `overactivity_term` penalizes excessive DG activity or diffuse activation.

This turns separate losses into one internal map-quality signal.

## Encoder Versus Decoder Assignment

The same internal components should not affect encoder and decoder identically.

For the encoder:

```text
teach DG to create sparse, reusable, spatially selective landmarks
```

The encoder should receive the population usage, collision, and overactivity terms most directly. These are representation-quality terms.

For the decoder:

```text
teach the policy to move through the environment in ways that reveal useful landmark transitions
```

The decoder should mainly receive the transition-distance reward and perhaps a weak collision penalty. It should not be strongly rewarded just for activating unused DG units, otherwise it may learn behavior that triggers arbitrary internal resource allocation rather than broad exploration.

## Recommended Default Objective

Recommended baseline:

```text
encoder:
    punish-style reward method
    + balanced population usage
    + targeted collision penalty
    + mild overactivity penalty

decoder:
    delayed transition-distance intrinsic reward
    + optional weak collision penalty
```

In current code terms, this means:

- keep `encoder_reward_method=punish` as the primary baseline;
- keep a batch/resource-utilization term;
- fold `encoder_batch_loss`, `encoder_unused_sequence_loss`, and `encoder_multi_activation_loss` into a single internal DG allocation objective;
- keep multi-activation as a targeted collision term, not a broad extra loss;
- remove or disable unused-sequence as a separate default once the population usage term exists.

## Practical Next Step

The next implementation should add a structured internal loss output from the DG-CA3 interface:

```python
DGCA3InternalLosses(
    transition_reward,
    population_usage_bonus,
    collision_penalty,
    overactivity_penalty,
    encoder_internal_loss,
    decoder_internal_reward,
)
```

Then the learner can use named internal signals instead of several separate auxiliary switches. This would make the algorithm more coherent: one intrinsic motivation system with interpretable components, rather than a base reward plus several hand-added corrections.

## Bottom Line

The evidence supports keeping two ideas strongly:

1. sparse punishment of DG activity;
2. batch-level/resource-utilization pressure.

The transition-distance reward remains the core intrinsic motivation signal, but by itself it is not enough to guarantee full map coverage.

The most redundant parts are unused-sequence loss and multi-activation loss as separate auxiliary losses. Their intended effects are valid, but they should be absorbed into a cleaner internal DG-CA3 allocation objective: use all sequence resources, avoid simultaneous collisions, and keep DG landmarks sparse.
