# DG spatial-information icon candidates

[Comparison preview](comparison.png) · [editable comparison](comparison.svg) · [comparison PDF](comparison.pdf) · [builder](build_spatial_information_icons.py) · [manifest](manifest.json)

Four separate **10 × 20 mm** vector icons are available for visual comparison. The comparison sheet shows each icon enlarged and at its actual page size. None has been added to Figure 2 or the Overleaf package.

Each icon depicts **one DG unit** in the selected Figure 2 palette (the repeated red circle identifies that unit). The upper example has higher raw spatial information and the arrow points up, as in the existing metric icons. Both examples assume equal spatial occupancy and equal total DG activity. The drawings are schematic: they encode no measured values, architecture ranking, or intervention effect.

| Candidate | Editable SVG | Print PDF | Encoding |
| --- | --- | --- | --- |
| Rate map | [SVG](01_rate_map.svg) | [PDF](01_rate_map.pdf) | Four strong adjacent spatial bins above versus sixteen weak bins below. Color intensity sums equally across the two maps. This most closely resembles an occupancy-corrected DG rate map. |
| DG events | [SVG](02_dg_events.svg) | [PDF](02_dg_events.pdf) | Nine activations of the same unit cluster in one spatial bin above versus one activation in each of nine bins below. The identical 3 × 3 grids mark corresponding positions; equal occupancy is an explicit assumption. |
| Position profile | [SVG](03_position_profile.svg) | [PDF](03_position_profile.pdf) | Four spatial-bin response bars: one strong bin above versus uniform responses below. The bar heights sum to the same value in both examples. |
| Occupancy + DG | [SVG](04_occupancy_and_dg_map.svg) | [PDF](04_occupancy_and_dg_map.pdf) | Two map cards per row match the overlap and reliability icon layout. The gray `occ` card repeats the same uniform occupancy; the red `DG` card shows one unit's localized field above and diffuse response below. The lower diffuse opacity approximates the upper field's mean, illustrating equal total activity. The cards are different quantities, not two DG units or two time points. |

For the displayed bits-per-step statistic, a schematic unit rate $r_b$ at position $b$ contributes to $J=\sum_b p_b r_b\log_2(r_b/\bar r)$, where $p_b$ is occupancy and $\bar r=\sum_b p_b r_b$. With occupancy and total activity held equal, a more location-specific response has higher $J$. The candidate icons convey this within-unit relation. The selected Figure 2 box plots are **differences between conditions**; choosing an icon would not assert the direction of any observed difference.

These assets use the neutral frame and upward-arrow style of [the existing Figure 2 metric icons](../../Figure2_metric_icons_README.md), plus the selected figure's DG-red identity from [the palette manifest](../../figure_palette_candidates/manifest.json). The occupancy + DG option also reuses their two-card geometry and soft field. Rebuild with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_spatial_information_icons.py` from this folder. The builder reuses the main icon exporter and records the reference SVG and palette hashes in `manifest.json`.
