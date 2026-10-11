# COSYNE 2027 abstract: prose draft for discussion

**Status:** First narrative draft, 12 October 2026. The abstract text below is intentionally separate from the two-page layout. The [bullet outline](cosyne%202027%20abstract%20-%20evidence-linked%20draft.md) and [question reports](../06_experiments/cosyne_2027/README.md) remain the evidence index. The attached 2026 COSYNE submission is a style reference, not evidence for the 2027 results.

## Abstract

How can a spatial code arise from egocentric vision when neither location labels nor external rewards instruct it? We study an agent that explores a visually rendered two-dimensional arena using a fixed visual trunk, a trainable sparse dentate gyrus (DG)-like landmark layer, and CA3-like sequence memory. Internal landmark events provide temporal feedback to the DG layer and intrinsic reward to the policy. This setup lets us ask whether the timing of internally generated events contributes information beyond the presence of positive feedback.

During reward-free exploration, DG activity becomes spatially informative in online measurements across flat and goal-conditioned agents. Because these measurements follow each agent's changing policy, we separately evaluate trained networks on identical observations. In the flat agent, temporal event credit yields higher DG spatial information, lower similarity between active unit maps, and broader spatial coverage of eligible response peaks than event-matched, mean-matched constant credit. All three differences favor temporal credit in five paired training seeds. Earlier component-removal controls further show that positive event credit helps sustain usable DG responses, while the new comparison isolates the contribution of temporal weighting.

Behavior also changes how a learned representation is expressed. Holding network weights fixed and replacing each agent's policy with uniform random actions reduces spatial information per activation in the first decoder layer in all three training seeds tested for each of three architectures. Thus, the action policy shapes the spatial signals available to a downstream reader even after learning. Goal-conditioned variants show that stronger spatial coding need not imply broader exploration or reliable response to a commanded landmark: changing the timing of the worker's internal hit bonus has mixed effects on coverage. Together, these results identify two separable contributions to emergent spatial coding: temporal structure in internally generated learning feedback and behavior-dependent expression of the resulting representation.

## Comments to resolve before the two-page draft

1. **Learning-course claim.** The increase in DG spatial score is from policy-sampled online data. A multi-seed common-observation checkpoint series is still needed to say that the *fixed-input representation* improves throughout training. Current wording keeps that distinction.
2. **Behavioral causality.** The own-policy versus random-action intervention freezes weights. It supports a change in *expression*, not a claim that behavior caused the learned weights to form. Confirm whether that narrower claim matches the intended story.
3. **Worker and control payoff.** Fresh C05 temporal versus constant hit-bonus coverage is mixed. Matched-start command outcomes and physical destinations remain open, so this draft does not claim improved goal following. Decide whether the mixed worker result belongs in a 300-word abstract or only in the two-page body.
4. **Venue and historical style.** The attached 2026 example uses a 296-word summary followed by figures. Confirm that this 2027 draft should retain a similar prose tone and whether the current submission instructions still allow a roughly 300-word opening summary.

## Evidence and next work

- [Model and online emergence](../06_experiments/cosyne_2027/01_model_and_spatial_emergence.md)
- [Frozen-policy behavior intervention](../06_experiments/cosyne_2027/02_behavior_expression.md)
- [Encoder temporal-credit comparison](../06_experiments/cosyne_2027/03_encoder_credit.md)
- [Worker bonus and control limits](../06_experiments/cosyne_2027/04_worker_bonus_and_scope.md)

For the later two-page layout, first reserve space for the model schematic, one spatial-field example, the five-seed paired encoder result, and the frozen-policy intervention. Keep the worker result conditional on completed trajectory and command analysis.
