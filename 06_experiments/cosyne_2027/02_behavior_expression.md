# 3. Does behavior change expression of a learned code?

[Outline](../../99_inbox/cosyne%202027%20abstract%20-%20evidence-linked%20draft.md) · [Question index](README.md) · [Detailed frozen-policy analysis](online_and_behavior_analysis_20261011.md#other-own-policy-versus-random-policy-readouts)

![Compact own-policy minus uniform-random decoder-1 contrasts](../results/cosyne_2027_compact_figures/02_behavior_expression.png)

[Editable SVG](../results/cosyne_2027_compact_figures/02_behavior_expression.svg) · [per-training-seed table](../results/cosyne_2027_behavior_secondary/paired_training_seeds.csv) · [original 54-probe intervention](../dg_representation/behavior_field_expression_20261007.md#results-own-policy-versus-uniform-random) · [evaluation-to-training-run index](frozen_policy_evaluations.csv)

**Claim supported:** With frozen weights, the own policy gives higher decoder-1 spatial information per activation than uniform random actions in all three training seeds of C01, C05, and C15. Mean own-minus-random differences are +0.557, +0.324, and +0.344 bits per activation. Panel **a** shows training-seed differences; the two evaluation repeats were averaged within each training seed. The independent unit of replication is the training seed ($n=3$ per architecture).

**Secondary representation readout:** Split-half decoder-1 map correlation is higher under own policy in all nine training seeds; mean differences are +0.257, +0.187, and +0.136 for C01/C05/C15. Panel **b** shows this result. The map-reliability metric uses each policy's own occupied bins, so repeated or concentrated visitation can improve it. A common-support or occupancy-matched reliability analysis is needed before interpreting it as a gain in code stability. The [all-layer diagnostic](../results/cosyne_2027_behavior_secondary/figures/own_minus_random_map_reliability.svg) and [source tables](../results/cosyne_2027_behavior_secondary/paired_evaluation_repeats.csv) retain DG, CA3, and decoder 2.

**Causal scope:** The intervention changes executed actions with the model frozen. It demonstrates behavior-dependent *expression* of the learned code, not that the behavior caused the weights to develop. Decoder-1 information per activation is the clean headline; reliability is a candidate corroboration with the occupancy caveat.

**Plot contract:** Two 1.9-inch panels, Arial 10 pt. Colored dots are training-seed differences and black segments are means; zero is no own/random difference. Regenerate from [cached inputs](../results/cosyne_2027_compact_figures/manifest.json) with [the compact renderer](render_compact_figures.py).
