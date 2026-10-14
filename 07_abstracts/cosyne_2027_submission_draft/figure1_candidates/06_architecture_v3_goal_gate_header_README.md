# Figure 1 v3 candidate: goal gate above the training curves

[SVG](06_architecture_v3_goal_gate_header.svg) · [Print PDF](06_architecture_v3_goal_gate_header.pdf) · [PNG preview](06_architecture_v3_goal_gate_header.png) · [Source manifest](06_architecture_v3_goal_gate_header_manifest.json) · [Previous candidate](05_architecture_v3_relative_reward_README.md)

This separate 7.5 × 1.9-inch candidate keeps the original vector architecture at left. The side-gate schematic occupies the top of panel B, with NG/UG/FG stacked vertically to its right. The DG spatial-information and relative intrinsic-reward courses sit together below them. The three C15 DG examples become panel C. The schematic uses the editable [side-gate source](goal_policy_icon_options/simple/05_side_gate.svg); the source artwork and previous Figure 1 candidates remain available.

The schematic shows the goal-conditioned cases: the selected goal $g$ conditions policy $\pi$, while $j=g$ gates the pull signal from a transition $\Delta_{i\to j}$. In NG there is no selected goal, the policy is not goal-conditioned, and every DG event is eligible for pull. UG selects a goal uniformly; FG selects a frontier goal. Both goal-conditioned cases require activation of the selected goal DG for pull. The blue $\Delta\to\pi$ path is continuous, with the comparator entering as a side gate.

Panel B shows each architecture's five-seed mean and SEM over training. Spatial information is in bits per step. Intrinsic reward is plotted on a linear axis after dividing each condition by the arithmetic mean of its own unsmoothed 99-point training course; one fold is the dotted reference. Each seed curve is smoothed with a Gaussian kernel of 1.5 million-frame standard deviation before the plotted mean and SEM are computed. The reward curves thus show within-architecture changes, not comparable absolute reward magnitudes. The normalization factors and all input hashes are in the manifest.

Panel C shows occupancy-corrected response maps from C15 temporal seed 123 at 100,007,936 frames, measured during a 10,001-decision policy probe that visited 307 of 361 spatial bins. The selected DG units are 4, 1, and 11; their peak cells are distinct. Each map uses its own visited-bin min–max color scale, and pale cells were unvisited. The three maps are illustrative examples rather than an estimate across seeds; the manifest records unit scores and strict-monofield status.

## Suggested caption for B and C

**B,** NG uses an unconditioned policy and allows every DG event to give pull. UG selects a goal uniformly; FG selects a frontier goal. In UG and FG, the selected goal conditions the policy and only activation of that goal DG passes the pull gate ($j=g$). Curves show five-seed mean ± SEM over training for DG spatial information (bits per step) and intrinsic reward divided by each architecture's own unsmoothed training-course mean. Seed curves were Gaussian-smoothed across training (1.5-million-frame standard deviation) before aggregation; the dotted reward line marks one fold. **C,** Example occupancy-corrected DG response maps from a C15 seed-123 terminal policy probe (10,001 decisions; 307/361 bins visited). Each map has its own visited-bin min–max color scale; pale bins were unvisited.

## Regeneration

From the repository root:

```bash
MPLCONFIGDIR=/tmp/cosyne_figure1_gate_header_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python 07_abstracts/cosyne_2027_submission_draft/figure1_candidates/render_architecture_v3_course_candidate.py --goal-gate-header
```

The selected submission figure and TeX files are unchanged.
