# Experimental Figure 1: spatial results at right

[PDF](01_architecture_spatial_right.pdf) · [SVG](01_architecture_spatial_right.svg) · [PNG preview](01_architecture_spatial_right.png)

This is a **separate candidate** at 7.5 × 3.12 inches. The submitted architecture figure, Figure 2, `main.tex`, and the compact Overleaf ZIP are unchanged. The left schematic is simplified so that its text remains at least 12 pt; the equations from the current architecture do not appear in this design. The spatial-emergence course and two example place fields occupy a narrow column on the right.

## Suggested caption

**Architecture and emergent spatial activity. A,** Egocentric RGB input passes through an ImageNet-pretrained ResNet-18 visual trunk fixed through layer 2. A trainable projection with batch normalization (BN) and thresholded rectification produces sparse dentate gyrus (DG) events. A fixed CA3-like memory retains their sequence. Its flattened state and egocentric depth feed a learned, two-layer readout multilayer perceptron (MLP) with 128 units per layer, which estimates value and selects among five actions. Temporal DG credit strengthens eligible event responses more after longer intervals between events (red push). Policy learning favors shorter transitions between internally detected events, or faster internally detected target hits in goal-conditioned variants (blue pull). Neither learning signal uses position labels, a map, physical distance, or a spatial reward.

**B,** Online amplitude-weighted DG spatial-information score in bits per step over active units, computed on each policy's latest 10,000 observations and shown across millions of training environment frames. Each curve is the mean of five training seeds; the shaded band is plus or minus one standard error of that mean. C01 explores directly, C05 uses a DG target, and C15 adds frontier-based target choice. All three displayed groups use temporal DG credit; C05 and C15 also use temporal worker hit credit. The policy and network both change during training, so these curves describe emergence along the sampled trajectories and do not isolate a learning-signal intervention.

**C,** Two illustrative, occupancy-corrected DG response maps from a separate archived C15 seed-99 model at 100,040,704 training frames, measured during one 10,001-decision policy evaluation. This probe visited 310 of 361 spatial bins (85.9%), compared with 220 of 361 (60.9%) in the previous C01 seed-8 example. DG units 7 and 0 have spatial-information scores of 0.217 and 0.167 bits in the source analysis. Each tile uses its **own** response-amplitude color range, starting at zero, so colors cannot be compared as absolute amplitudes between the two units; white cells were unvisited. The maps contain more than one response hotspot and are illustrative spatial responses, not evidence that either unit has a single compact field. This archived model is separate from the networks in the current common-replay temporal-credit comparison and is not a representative estimate across training seeds.

## Source and regeneration

Panel B reads the archived [online arm summary](../../../06_experiments/results/cosyne_2027_online_wandb/arm_summaries.csv). Panel C crops two tiles from the [C15 seed-99 per-unit-scale SVG](../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/c15_seed99/place_fields_individual_scale.svg); the [probe table](../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/matched_terminal_per_run.csv) verifies occupancy and evaluation length, and the [unit table](../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/place_field_individual_scales.csv) verifies field identities, spatial-information scores, and color ranges. [manifest.json](manifest.json) records the source hashes. The field tiles are raster crops; the schematic, curve, uncertainty bands, axes, and labels remain vector in the PDF and SVG.

From this directory, rebuild with:

```bash
MPLCONFIGDIR=/tmp/cosyne_figure1_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_architecture_with_fields.py
```

This layout would require moving the Figure 2A spatial-emergence panel and the two Figure 2C field tiles out of Figure 2 if selected for the submission, then updating both captions and figure references. The present candidate does not make those changes.
