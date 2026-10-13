# Eight screened DG examples per architecture

These are **selection sheets** for choosing Figure 1 examples. Each tile has a unique ID printed beside its DG unit number. Send back IDs such as `C15-1`, `C15-2`, `C05-1`, and `C01-1`; the ID identifies the exact architecture, training condition, seed, and unit through the [selection table](selected_units.csv). No submission figure has been changed from this gallery.

## Architecture and asset index

All three architectures share the fixed ImageNet-pretrained ResNet-18 trunk through layer 2, a learned DG projection with batch normalization, a fixed CA3-like sequence memory, and a learned action/value readout. The architecture names describe how the policy explores or selects internally detected DG goals:

| Architecture | Navigation design | Eight-map assets | Selection IDs |
| --- | --- | --- | --- |
| **C01** | Direct exploration; no DG target-selection worker. | [PNG](c01_eight_screened_fields.png) · [PDF](c01_eight_screened_fields.pdf) | `C01-1`–`C01-8` |
| **C05** | Uses an internally detected DG event as a navigation target. | [PNG](c05_eight_screened_fields.png) · [PDF](c05_eight_screened_fields.pdf) | `C05-1`–`C05-8` |
| **C15** | Adds frontier-based selection of DG targets. | [PNG](c15_eight_screened_fields.png) · [PDF](c15_eight_screened_fields.pdf) | `C15-1`–`C15-8` |

The [original architecture artwork](../../../figures/architecture_v2.svg) and [Figure 1 design notes](../../README.md) give the shared module diagram and longer explanation.

## Exact unit index

Read each sheet left to right: IDs 1–4 are the top row and IDs 5–8 are the bottom row. Each row comes from one terminal network and one policy-driven probe. The listed DG numbers are in the same left-to-right order as the IDs.

| IDs | Condition in that row | Seed | DG units in ID order | Visited bins |
| --- | --- | ---: | --- | ---: |
| `C01-1`–`C01-4` | Temporal DG credit | 99 | 6, 15, 5, 0 | 226/361 (63%) |
| `C01-5`–`C01-8` | Constant DG credit | 99 | 2, 10, 7, 15 | 307/361 (85%) |
| `C05-1`–`C05-4` | Constant DG / temporal worker | 99 | 5, 12, 7, 6 | 248/361 (69%) |
| `C05-5`–`C05-8` | Temporal DG / temporal worker | 99 | 11, 7, 3, 5 | 190/361 (53%) |
| `C15-1`–`C15-4` | Temporal DG credit | 123 | 4, 3, 7, 11 | 307/361 (85%) |
| `C15-5`–`C15-8` | Constant DG credit | 123 | 0, 11, 3, 14 | 239/361 (66%) |

DG credit changes how event responses are reinforced. The C05 row labels also specify the policy worker's target-hit credit; both displayed C05 rows use temporal worker credit. The C15 comparison varies DG credit while its frontier-target policy remains the same architecture. Condition, seed, and probe coverage must travel with any selected map.

**Screening.** I inspected all 16 maps in each of the six saved atlases and excluded the most visibly diffuse responses, including C01 constant DG 4, 11, and 14; C05 constant/temporal DG 4 and 15; and C15 temporal DG 0, 1, 12, and 14. Selection also considered the saved spatial-information and mono-field scores and avoided nearly silent units. This is a qualitative screen, not a new statistical field-size threshold. Some retained candidates are weak, border-weighted, or have multiple hotspots. C01 in particular does **not** supply eight convincing compact place fields; the sheet makes its less compelling alternatives visible rather than silently replacing them with broad high-score maps.

**Reading each tile.** `SI` is the evaluator's amplitude-weighted occupancy-corrected spatial-information score for that unit, reported in bits; it is not normalized bits per activation. `mono` is the minimum fraction of above-threshold response mass in the largest connected component across 30%, 50%, and 70% of peak rate. `strict` requires an eligible map and a mono score of at least 0.8. A high mono score does not by itself guarantee a compact field. `active` is the fraction of the 10,001 probe decisions with a DG event. Gray bins were not visited. Color is scaled jointly within each run, so response strength can be compared within a row, but each row has its own scale.

All networks were evaluated near 100 million training frames using one 10,001-decision stochastic frozen-policy probe. Each map is occupancy-corrected on a 19-by-19 spatial grid. The different policies visited different parts of the maze. The sheets offer examples to inspect, not an estimate of how often a given architecture develops a compact field. Preserve condition, seed, DG index, probe coverage, activity fraction, and scale convention when choosing examples for the submission.

[Selected-unit values](selected_units.csv) and the [source hash manifest](manifest.json) make the sheets auditable. The original atlas PNGs, metric tables, submitted figures, and Overleaf ZIP are unchanged. Regenerate from the existing saved artifacts with:

```bash
MPLCONFIGDIR=/tmp/cosyne_eight_fields_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python ../../render_eight_field_gallery.py
```
