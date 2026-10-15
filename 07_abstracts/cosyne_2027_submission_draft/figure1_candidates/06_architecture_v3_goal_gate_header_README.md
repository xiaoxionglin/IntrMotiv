# Figure 1 v3: author-edited goal-gate layout

[SVG](06_architecture_v3_goal_gate_header.svg) · [Print PDF](06_architecture_v3_goal_gate_header.pdf) · [PNG preview](06_architecture_v3_goal_gate_header.png) · [Source manifest](06_architecture_v3_goal_gate_header_manifest.json) · [Previous candidate](05_architecture_v3_relative_reward_README.md)

The current working color selection for this layout is [H: orange push, violet pull](../figure_palette_candidates/priority_rgb_orange_violet_figure1.pdf). Use its [editable SVG](../figure_palette_candidates/priority_rgb_orange_violet_figure1.svg) for color-consistent next revisions; this author-edited base remains the geometry source for comparing palettes.

The SVG is the author-edited geometry base for Figure 1 revisions. Its PDF and PNG are direct exports from that SVG. This 7.5 × 1.9-inch layout keeps the original vector architecture at left. Its heading reads Policy, and its three temporal-separation symbols read TS, matching Figure 2's vocabulary. The side-gate schematic occupies the top of panel B, with NG/UG/FG stacked vertically to its right. The DG spatial-information and relative intrinsic-reward courses sit together below them. The three C15 DG examples have only a C panel letter; their description is in the caption. The schematic began with the editable [side-gate source](goal_policy_icon_options/simple/05_side_gate.svg), with TS and Pol. substituted during composition; the original source artwork and previous Figure 1 candidates remain available.

The schematic shows the goal-conditioned cases: the selected goal $g$ conditions the Policy module (Pol. in the small icon), while $j=g$ gates the pull signal associated with the TS-marked DG event. TS denotes CA3 temporal separation, not a literal pairwise timestamp subtraction. In NG there is no selected goal, the policy is not goal-conditioned, and every DG event is eligible for pull. UG selects a goal uniformly; FG selects a frontier goal. Both goal-conditioned cases require activation of the selected goal DG for pull. The blue TS-to-Pol. path is continuous, with the comparator entering as a side gate.

Panel B shows each architecture's five-seed mean and SEM over training. Spatial information is in bits per step. The author-edited reward axis reads “Intrinsic Reward”; its values remain divided by each condition's arithmetic mean over its own unsmoothed 99-point training course. One fold is the dotted reference. Each seed curve is smoothed with a Gaussian kernel of 1.5 million-frame standard deviation before the plotted mean and SEM are computed. The reward curves thus show within-architecture changes, not comparable absolute reward magnitudes. The normalization factors and all input hashes are in the manifest.

Panel C shows occupancy-corrected response maps from C15 temporal seed 123 at 100,007,936 frames, measured during a 10,001-decision policy probe that visited 307 of 361 spatial bins. The selected DG units are 4, 1, and 11; their peak cells are distinct. Each map uses its own visited-bin min–max color scale, and pale cells were unvisited. The three maps are illustrative examples rather than an estimate across seeds; the manifest records unit scores and strict-monofield status. Their display has been flipped vertically relative to the archived atlas tiles so that increasing world $y$ points upward, matching the maze view and Figure 2 trajectories; the recorded peak-cell coordinates remain in the original $[x,y]$ data convention.

## Suggested caption for B and C

**B,** TS denotes CA3 temporal separation and Pol. denotes policy. NG uses an unconditioned policy and allows every DG event to give pull. UG selects a goal uniformly; FG selects a frontier goal. In UG and FG, the selected goal conditions the policy and only activation of that goal DG passes the pull gate ($j=g$). Curves show five-seed mean ± SEM over training for DG spatial information (bits per step) and intrinsic reward divided by each architecture's own unsmoothed training-course mean. Seed curves were Gaussian-smoothed across training (1.5-million-frame standard deviation) before aggregation; the dotted reward line marks one fold. **C,** Example occupancy-corrected DG response maps from a C15 seed-123 terminal policy probe (10,001 decisions; 307/361 bins visited). Each map has its own visited-bin min–max color scale; pale bins were unvisited.

## Editing and export

Edit [this SVG](06_architecture_v3_goal_gate_header.svg) for further layout changes. To refresh the derived formats from the repository root:

```bash
inkscape 07_abstracts/cosyne_2027_submission_draft/figure1_candidates/06_architecture_v3_goal_gate_header.svg --export-type=pdf --export-filename=07_abstracts/cosyne_2027_submission_draft/figure1_candidates/06_architecture_v3_goal_gate_header.pdf
inkscape 07_abstracts/cosyne_2027_submission_draft/figure1_candidates/06_architecture_v3_goal_gate_header.svg --export-width=2250 --export-filename=07_abstracts/cosyne_2027_submission_draft/figure1_candidates/06_architecture_v3_goal_gate_header.png
```

The Python renderer retains the pre-edit composition logic for reference. Use `--goal-gate-header --output-stem NEW_NAME` for a separate generated alternative; it refuses to overwrite an existing goal-gate SVG. The selected submission figure and TeX files are unchanged.
