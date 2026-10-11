# COSYNE 2027 abstract: prose draft for discussion

**Status:** Revised narrative draft, 12 October 2026. This version emphasizes the proposed mechanism and take-home message; the two-page layout is still pending. The [bullet outline](cosyne%202027%20abstract%20-%20evidence-linked%20draft.md) and [question reports](../06_experiments/cosyne_2027/README.md) remain the evidence index. The attached 2026 COSYNE submission is a style reference, not evidence for the 2027 results.

## Abstract

How can spatial representations emerge from egocentric experience without a map, position labels, or external reward? We propose that a hippocampal-inspired circuit can turn the timing of encountered sensory events into a learning signal. In our agent, sparse dentate gyrus (DG) responses mark salient visual experiences, while CA3-like sequences carry their recent history. Temporal separation between events creates two complementary pressures: a “push” that favors distinct DG landmarks and a “pull” that motivates the policy to move between them. Spatial structure can thus arise through the agent’s interaction with the world rather than being imposed as a training target.

In a continuous visual arena, the agent develops spatially informative DG activity during exploration. The critical test is whether event timing matters beyond simply rewarding DG activity. In the flat exploration agent, replacing temporally weighted DG feedback with constant feedback of the same average magnitude produces responses that are more redundant and less evenly distributed across space. The advantage of temporal feedback persists when both networks are evaluated on the same observations, showing a difference in the learned representation beyond differences in the paths sampled during evaluation.

The agent’s behavior also affects how spatial information is expressed. With network weights frozen, its learned policy elicits more spatially informative downstream responses than random actions do. Even after learning, the experiences selected by the agent shape the spatial signals available to a downstream reader.

Together, these findings offer a route from egocentric sensation to spatial coding: intrinsic hippocampal-like dynamics organize sensory events in time, while self-directed behavior determines how that organization is sampled and read out. They suggest that spatial representations can emerge from interactions between memory, learning, and action without direct spatial instruction.

## Comments to resolve before the two-page draft

1. **Learning-course claim.** The increase in DG spatial score is from policy-sampled online data. A multi-seed common-observation checkpoint series is still needed to say that the *fixed-input representation* improves throughout training. Current wording keeps that distinction.
2. **Behavioral causality.** The own-policy versus random-action intervention freezes weights. It supports a change in *expression*, not a claim that behavior caused the learned weights to form. The abstract keeps this distinction while making the result a headline.
3. **Worker and control payoff.** The “pull” describes the policy's intrinsic objective. Fresh C05 temporal versus constant hit-bonus coverage is mixed, and matched-start command outcomes remain open. The abstract therefore does not claim that this objective improves exploration or goal following; decide whether the mixed outcome belongs in the two-page body.
4. **Venue and historical style.** The attached 2026 example uses a 296-word summary followed by figures. Confirm that this 2027 draft should retain a similar prose tone and whether the current submission instructions still allow a roughly 300-word opening summary.

## Evidence and next work

- [Model and online emergence](../06_experiments/cosyne_2027/01_model_and_spatial_emergence.md)
- [Frozen-policy behavior intervention](../06_experiments/cosyne_2027/02_behavior_expression.md)
- [Encoder temporal-credit comparison](../06_experiments/cosyne_2027/03_encoder_credit.md)
- [Worker bonus and control limits](../06_experiments/cosyne_2027/04_worker_bonus_and_scope.md)

For the later two-page layout, first reserve space for the model schematic, one spatial-field example, the five-seed paired encoder result, and the frozen-policy intervention. Keep the worker result conditional on completed trajectory and command analysis.
