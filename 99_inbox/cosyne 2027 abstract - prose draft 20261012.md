# COSYNE 2027 abstract: prose draft for discussion

**Status:** Revised narrative draft, 12 October 2026. This version emphasizes the proposed mechanism and take-home message; the two-page layout is still pending. The [bullet outline](cosyne%202027%20abstract%20-%20evidence-linked%20draft.md) and [question reports](../06_experiments/cosyne_2027/README.md) remain the evidence index. The attached 2026 COSYNE submission is a style reference, not evidence for the 2027 results.

## Abstract

How can spatial representations emerge from egocentric experience without a map, position labels, or external reward? We propose that a hippocampal-inspired circuit can turn the timing of encountered sensory events into a learning signal. In our agent, sparse dentate gyrus (DG) responses mark salient visual experiences, while CA3-like sequences carry their recent history. Temporal separation between events creates two complementary pressures: a “push” that favors distinct DG landmarks and a “pull” that rewards the policy for reaching internally defined targets. Spatial structure can thus arise through the agent’s interaction with the world rather than being imposed as a training target.

In a continuous visual arena, the agent develops spatially informative DG activity during exploration. Replacing temporal DG feedback with constant feedback makes DG responses less spatially informative, more redundant, and concentrated in fewer parts of the arena. This supports a role for temporal pattern separation in organizing the code. We also test the policy-side “pull”: replacing its temporally weighted target-hit bonus with a constant bonus has mixed effects on the representation and exploration. Its effect can be largely substituted by the actor-critic temporal discount factor to encourage faster traversal. 

The agent's interaction with the world further shapes the representation because of the recurrent memory. With weights frozen, the learned policy evokes more spatially informative downstream responses than random actions, showing that behavior changes how the code is expressed. This exemplifies the intertwinement of policy and representational learning.

Together, these results show how temporal memory and self-directed action can produce spatial organization without direct spatial instruction. 

~~They also separate two claims: forming a spatial code and using it to explore.~~
comment: do we even have space for this?

## Comments to resolve before the two-page draft

1. **Learning-course claim.** The increase in DG spatial score is from policy-sampled online data. A multi-seed common-observation checkpoint series is still needed to say that the *fixed-input representation* improves throughout training. Current wording keeps that distinction.
2. **Behavioral causality.** The own-policy versus random-action intervention freezes weights. It supports a policy effect on *expression* of the learned code. A training-policy intervention would be needed to claim that the policy *refines the DG weights during learning*. The abstract presents refinement as a possible route, not a measured outcome.
3. **Pull ablation and control payoff.** The author requested that the pull ablation appear beside the DG ablation in the result paragraph. The fresh C05 [temporal-versus-constant target-hit comparison](../06_experiments/cosyne_2027/04_worker_bonus_and_scope.md) is the direct timing ablation of the “pull.” Its five-seed [coverage figure](../06_experiments/results/cosyne_2027_compact_figures/04_worker_coverage_values_and_delta.svg) has mixed effects in both encoder arms, so it cannot currently establish that temporal pull is necessary for exploration. Earlier bonus-versus-no-bonus controls also changed reward magnitude and crossed a source boundary. The newly submitted C15 constant-hit arm remains pending in the [run index](../06_experiments/cosyne_2027/README.md). Matched-start command outcomes and physical destinations remain open, so the abstract does not claim improved goal following.

   The fresh decoder-1 [behavior-expression figure](https://github.com/xiaoxionglin/IntrMotiv/blob/6412596/06_experiments/results/cosyne_2027_compact_figures/02_behavior_expression_fresh.png) and [seed table](https://github.com/xiaoxionglin/IntrMotiv/blob/6412596/06_experiments/results/cosyne_2027_behavior_secondary/all_training_seeds.csv) show a **lead**, not yet a clear representation benefit from temporal pull. At fixed temporal/constant DG credit, temporal worker credit raises the mean *own-policy minus random-policy* decoder-1 information contrast by +0.055/+0.187 bits per activation, but only 3/5 and 4/5 paired seeds favor it. In the underlying [evaluation table](https://github.com/xiaoxionglin/IntrMotiv/blob/6412596/06_experiments/results/cosyne_2027_behavior_secondary/fresh_pair_analysis/paired_evaluation_repeats.csv), the trained policy's absolute decoder-1 information changes by just +0.011/+0.042 bits per activation, with 3/5 positive seeds in each DG-credit arm. Every C05 factor cell favors its own policy over random actions, including constant pull. Keep this within-cell behavior effect distinct from the temporal-versus-constant worker contrast; the fresh figure comes from branch `codex/cosyne-behavior-fresh-seeds` at commit `6412596` and is not yet on `main`.
4. **Venue and historical style.** The attached 2026 example uses a 296-word summary followed by figures. Confirm that this 2027 draft should retain a similar prose tone and whether the current submission instructions still allow a roughly 300-word opening summary.

## Evidence and next work

- [Model and online emergence](../06_experiments/cosyne_2027/01_model_and_spatial_emergence.md)
- [Frozen-policy behavior intervention](../06_experiments/cosyne_2027/02_behavior_expression.md)
- [Encoder temporal-credit comparison](../06_experiments/cosyne_2027/03_encoder_credit.md)
- [Worker bonus and control limits](../06_experiments/cosyne_2027/04_worker_bonus_and_scope.md)

For the later two-page layout, first reserve space for the model schematic, one spatial-field example, the five-seed paired encoder result, and the frozen-policy intervention. Keep the worker result conditional on completed trajectory and command analysis.
