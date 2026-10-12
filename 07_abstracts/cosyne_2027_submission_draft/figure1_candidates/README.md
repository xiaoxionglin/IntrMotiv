# Experimental Figure 1: spatial results at right

[Latest architecture-priority candidate PDF](03_architecture_priority_compact_results.pdf) · [SVG](03_architecture_priority_compact_results.svg) · [PNG preview](03_architecture_priority_compact_results.png)

[Earlier original-art candidate PDF](02_original_architecture_spatial_right.pdf) · [SVG](02_original_architecture_spatial_right.svg) · [PNG preview](02_original_architecture_spatial_right.png)

[Earlier simplified candidate PDF](01_architecture_spatial_right.pdf) · [SVG](01_architecture_spatial_right.svg) · [PNG preview](01_architecture_spatial_right.png)

All three layouts are **separate candidates** at 7.5 × 3.12 inches. The submitted architecture figure, Figure 2, `main.tex`, and the compact Overleaf ZIP are unchanged. Candidate 3 gives the author's original architecture artwork about 5.8 inches of width, compresses the spatial-emergence plot into a narrow top-right strip, and places all four archived DG maps in a tiny two-by-two grid beneath it. Candidate 2 uses the same art at about 4.5 inches wide with a wider plot and two larger maps. Candidate 1 retains a simplified-box comparison.

The original art includes the egocentric scene, ResNet-18 wedge, DG sparsifier, CA3 sequence, readout, maze, feedback arrows, and temporal-separation inset. The adapter crops away the large formula rows and masks the observation-update formula above the diagram. Isolated symbols for the threshold, matrix links, and event interval remain in the artwork; no equation is displayed. Three prose lines beneath it state which modules are fixed or learned and identify the push and pull paths. In candidate 3, the diagram's explanatory word labels are redrawn at 10 pt in their original locations; its other added text is 11–12 pt. The original small mathematical and matrix symbols remain part of the artwork. The art is embedded at 600 dpi; plot lines, uncertainty bands, axes, and added text are vector in the PDF and SVG.

## Suggested caption

**Architecture and emergent spatial activity. A,** Egocentric RGB input passes through an ImageNet-pretrained ResNet-18 visual trunk fixed through layer 2. A trainable projection with batch normalization (BN) and thresholded rectification produces sparse dentate gyrus (DG) events. A fixed CA3-like memory retains their sequence. Its flattened state and egocentric depth feed a learned, two-layer readout multilayer perceptron (MLP) with 128 units per layer, which estimates value and selects among five actions. Temporal DG credit strengthens eligible event responses more after longer intervals between events (red push). Policy learning favors shorter transitions between internally detected events, or faster internally detected target hits in goal-conditioned variants (blue pull). Neither learning signal uses position labels, a map, physical distance, or a spatial reward.

**B,** Online amplitude-weighted DG spatial-information score in bits per step over active units, computed on each policy's latest 10,000 observations and shown across millions of training environment frames. Each curve is the mean of five training seeds; the shaded band is plus or minus one standard error of that mean. C01 explores directly, C05 uses a DG target, and C15 adds frontier-based target choice. All three displayed groups use temporal DG credit; C05 and C15 also use temporal worker hit credit. The policy and network both change during training, so these curves describe emergence along the sampled trajectories and do not isolate a learning-signal intervention.

**C,** Four illustrative, occupancy-corrected DG response maps from a separate archived C15 seed-99 model at 100,040,704 training frames, measured during one 10,001-decision policy evaluation. This probe visited 310 of 361 spatial bins (85.9%), compared with 220 of 361 (60.9%) in the previous C01 seed-8 example. The four displayed units are the top four by spatial-information score in that archived unit table: DG 7, 11, 1, and 0 score 0.217, 0.185, 0.176, and 0.167 bits, respectively. Candidate 2 shows only DG 7 and 0. Each tile uses its **own** response-amplitude color range, starting at zero, so colors cannot be compared as absolute amplitudes between units; white cells were unvisited. The maps contain multiple response hotspots and are illustrative spatial responses, not evidence that any unit has a single compact field. This archived model is separate from the networks in the current common-replay temporal-credit comparison and is not a representative estimate across training seeds.

## Source and regeneration

Panel A of candidates 2 and 3 crops the original [architecture v2 SVG](../figures/architecture_v2.svg) with exact export, crop, and formula-mask coordinates recorded in [candidate 3's manifest](03_architecture_priority_compact_results_manifest.json). Panel B reads the archived [online arm summary](../../../06_experiments/results/cosyne_2027_online_wandb/arm_summaries.csv). Panel C crops four tiles from the [C15 seed-99 per-unit-scale SVG](../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/c15_seed99/place_fields_individual_scale.svg); the [probe table](../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/matched_terminal_per_run.csv) verifies occupancy and evaluation length, and the [unit table](../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/place_field_individual_scales.csv) verifies field identities, spatial-information scores, and color ranges. The [candidate 1 manifest](manifest.json) and [candidate 2 manifest](02_original_architecture_spatial_right_manifest.json) record their input hashes. The field tiles are raster crops in all candidates.

From this directory, rebuild with:

```bash
MPLCONFIGDIR=/tmp/cosyne_figure1_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_architecture_with_fields.py
MPLCONFIGDIR=/tmp/cosyne_figure1_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_original_architecture_candidate.py
MPLCONFIGDIR=/tmp/cosyne_figure1_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_architecture_priority_candidate.py
```

This layout would require moving the Figure 2A spatial-emergence panel and the Figure 2C field examples out of Figure 2 if selected for the submission, then updating both captions and figure references. The present candidates do not make those changes.
