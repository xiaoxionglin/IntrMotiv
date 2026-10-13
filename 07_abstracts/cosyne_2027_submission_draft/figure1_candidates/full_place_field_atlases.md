# Complete COSYNE DG place-field atlases available in this checkout

These are the **unfiltered source atlases** behind the Figure 1 selection sheets. Each PNG displays every DG unit, **0–15**, in a four-by-four grid. The row and column position is the DG index in row-major order: 0–3 at the top, 4–7 in the second row, 8–11 in the third, and 12–15 at the bottom. No units were excluded from these assets.

The six locally saved atlases contain 96 displayed unit maps from six terminal network probes. They are examples from selected seeds, **not** the full 40-model terminal evaluation collection. The analysis [source record](../../../06_experiments/cosyne_2027/checkpoint_fields_20261011.md) documents all 40 terminal checkpoint evaluations and their remote NPZs; those raw per-model maps have not all been copied into this checkout.

## Architecture notes

The three agents share the fixed ImageNet-pretrained ResNet-18 trunk through layer 2, a trainable DG projection with batch normalization, fixed CA3-like sequence memory, and learned readout. **C01** explores directly. **C05** navigates toward an internally detected DG target. **C15** adds frontier-based selection of DG targets. See the [original architecture artwork](../figures/architecture_v2.svg) and [submission architecture description](../main.tex) for the common modules and learning signals.

## Full atlas assets

Select a map by **architecture, condition, seed, and DG unit**, for example `C15 / temporal DG / seed 123 / DG 4`. Every PNG is a 1,200-by-1,206-pixel source atlas of one 10,001-decision frozen-policy probe near 100 million training frames. Gray means an unvisited spatial bin. Each atlas has one within-run response color scale; color intensity across different atlases is not directly comparable.

| Architecture | DG credit; worker credit | Seed | Visited bins | All 16 maps | Per-unit metrics |
| --- | --- | ---: | ---: | --- | --- |
| C01 | Temporal DG; no target worker | 99 | 226/361 (63%) | [Full atlas](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c01_temporal_s3/summary/place_fields_COS27_C01_TEMPORAL_S99__00_COS27_C01_TEMPORAL_S99__COS27_C01_TEMPORAL__s99__f100007936.png) | [CSV](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c01_temporal_s3/summary/per_unit_place_field_metrics.csv) |
| C01 | Constant DG; no target worker | 99 | 307/361 (85%) | [Full atlas](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c01_constant_s3/summary/place_fields_COS27_C01_CONSTANT_S99__00_COS27_C01_CONSTANT_S99__COS27_C01_CONSTANT__s99__f100007936.png) | [CSV](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c01_constant_s3/summary/per_unit_place_field_metrics.csv) |
| C05 | Constant DG; temporal worker | 99 | 248/361 (69%) | [Full atlas](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c05_pushpull_s3/summary/place_fields_COS27_C05_CONSTANT_TEMPORAL_S99__00_COS27_C05_CONSTANT_TEMPORAL_S99__COS27_C05_CONSTANT_TEMPORAL__s99__f100007936.png) | [CSV](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c05_pushpull_s3/summary/per_unit_place_field_metrics.csv) |
| C05 | Temporal DG; temporal worker | 99 | 190/361 (53%) | [Full atlas](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c05_pushpull_s3/summary/place_fields_COS27_C05_TEMPORAL_TEMPORAL_S99__00_COS27_C05_TEMPORAL_TEMPORAL_S99__COS27_C05_TEMPORAL_TEMPORAL__s99__f100007936.png) | [CSV](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c05_pushpull_s3/summary/per_unit_place_field_metrics.csv) |
| C15 | Temporal DG; frontier target selection | 123 | 307/361 (85%) | [Full atlas](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c15_available/summary/place_fields_COS27_C15_TEMPORAL_S123__00_COS27_C15_TEMPORAL_S123__COS27_C15_TEMPORAL__s123__f100007936.png) | [CSV](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c15_available/summary/per_unit_place_field_metrics.csv) |
| C15 | Constant DG; frontier target selection | 123 | 239/361 (66%) | [Full atlas](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c15_available/summary/place_fields_COS27_C15_CONSTANT_S123__00_COS27_C15_CONSTANT_S123__COS27_C15_CONSTANT__s123__f100007936.png) | [CSV](../../../06_experiments/results/cosyne_2027_checkpoint_fields/c15_available/summary/per_unit_place_field_metrics.csv) |

The C05 and C15 CSVs contain rows for additional probes in their batches. Filter the CSV by the condition and seed named in the atlas row. The [screened eight-per-architecture sheets](place_field_options/eight_per_architecture/README.md) remain available as a visual shortlist, but the six full atlases above are the complete local visual assets for these example networks.
