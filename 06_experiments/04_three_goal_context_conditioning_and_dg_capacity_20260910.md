# 04 — Three-goal HippoSLAM: context conditioning and DG capacity

Read-only W&B API audit, 10 September 2026. Project: [HippoSLAM_R3_2](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/HippoSLAM_R3_2). Thirty December 2025 runs cross 8/16/32 DG units, instruction scales 1/9, and five seeds. Each run contains eight PBT policies; those policies are not independent seeds. All runs are marked crashed but retain substantial training metrics. No crash diagnosis or matched-history analysis was performed.

## Observed terminal summaries

Values first average the eight policy summaries within each run, then the five seeds. These are final logged windows at unequal frame counts, not aligned evaluations or best-policy scores.

| DG units | Scale 1 weighted score | Scale 9 weighted score | Scale 1 raw score | Scale 9 raw score |
|---|---:|---:|---:|---:|
| 8 | 5.93 | 5.64 | 8.20 | 7.91 |
| 16 | 7.37 | 7.50 | 9.49 | 9.54 |
| 32 | 8.37 | 8.23 | 9.91 | 9.83 |

Capacity improves terminal scores; increasing instruction amplitude does not consistently improve them. The 32-unit populations reached about 92–168M frames per policy, versus 163–290M for 8 units. Increasing DG units also increases memory dimension and decoder input size, so this is not an isolated neuron-count intervention.

## Architecture and evidence boundary

Configurations specify numerical instructions, depth, pretrained visual encoder, batchnorm_relu DG, and BypassSS. Retained local default custom_encoder.py concatenates the scaled three-way instruction with visual features before DG and also includes it in the depth/instruction bypass. Schematically:

$$z_t=[\operatorname{BN}(W_xx_t+lpha W_ce(c_t))-	heta]_+,\qquad \pi(a_t\mid C_t,d_t,lpha e(c_t)).$$

This supplies direct context dependence in DG, unlike the recent CPD GOAL prediction head. However, historical W&B git_hash is unknown; the retained local environment openfield_map2_fixed_loc3.lua currently hardcodes context 3 with random selection commented out. It cannot authenticate historical three-goal randomization. The randomized task description comes from the user. This sweep has no pathway-removal conditions, so necessity of both context inputs remains the user's historical observation, not an ablation established by this project.

## Interpretation and next discriminating checks

Context can separate observations with different behavioral meanings, while direct decoder context avoids reconstructing the current instruction from sparse memory. More capacity may accommodate context-specific responses, preserve sparse events, or improve the larger downstream representation; the scores alone do not distinguish these explanations. A three-context partition into separate cell sets is a hypothesis, not an observed result.

First recover archived task source and context-pathway ablations. Then evaluate the same observations and histories under each context, recording continuous and thresholded DG responses, conditional fields, unit sharing, and per-goal behavior. Counterfactual context changes should be distinguished from changes in policy-induced occupancy. For a future capacity contrast, hold downstream width fixed; if testing whether extra active units explain the effect, explicitly control population activity separately. Fixed-reference reward in W_REF could permit a goal-conditioned live DG without letting that DG redefine success; graph identities should remain tied to the stable reference in that experiment.

Process lesson: inspect W&B resolved configs and per-policy metric namespaces rather than run names or best curves. Average within PBT populations before across seeds, and recover exact archived source before treating a modified local environment as provenance. API summaries were sufficient for this bounded audit; repeat collection and training were unnecessary. Retained numerical evidence: [terminal summaries](results/hipposlam_r3_2_20260910/terminal_summary.json).
