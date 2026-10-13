# Eight screened DG examples per architecture

These are **selection sheets**, not replacements for the submitted Figure 1. Each sheet shows eight different DG units from two terminal networks of the same architecture. The rows are separate trained runs and separate frozen-policy probes; comparisons of response amplitude or map location across rows are not controlled.

| Architecture | Full-size sheet | Source conditions | Visited bins |
| --- | --- | --- | --- |
| C01, direct exploration | [PNG](c01_eight_screened_fields.png) · [PDF](c01_eight_screened_fields.pdf) | temporal DG credit; constant DG credit, seed 99 | 226/361; 307/361 |
| C05, DG-target navigation | [PNG](c05_eight_screened_fields.png) · [PDF](c05_eight_screened_fields.pdf) | constant DG / temporal worker; temporal DG / temporal worker, seed 99 | 248/361; 190/361 |
| C15, frontier-selected DG targets | [PNG](c15_eight_screened_fields.png) · [PDF](c15_eight_screened_fields.pdf) | temporal DG credit; constant DG credit, seed 123 | 307/361; 239/361 |

**Screening.** I inspected all 16 maps in each of the six saved atlases and excluded the most visibly diffuse responses, including C01 constant DG 4, 11, and 14; C05 constant/temporal DG 4 and 15; and C15 temporal DG 0, 1, 12, and 14. Selection also considered the saved spatial-information and mono-field scores and avoided nearly silent units. This is a qualitative screen, not a new statistical field-size threshold. Some retained candidates are weak, border-weighted, or have multiple hotspots. C01 in particular does **not** supply eight convincing compact place fields; the sheet makes its less compelling alternatives visible rather than silently replacing them with broad high-score maps.

**Reading each tile.** `SI` is the evaluator's amplitude-weighted occupancy-corrected spatial-information score for that unit, reported in bits; it is not normalized bits per activation. `mono` is the minimum fraction of above-threshold response mass in the largest connected component across 30%, 50%, and 70% of peak rate. `strict` requires an eligible map and a mono score of at least 0.8. A high mono score does not by itself guarantee a compact field. `active` is the fraction of the 10,001 probe decisions with a DG event. Gray bins were not visited. Color is scaled jointly within each run, so response strength can be compared within a row, but each row has its own scale.

All networks were evaluated near 100 million training frames using one 10,001-decision stochastic frozen-policy probe. Each map is occupancy-corrected on a 19-by-19 spatial grid. The different policies visited different parts of the maze. The sheets offer examples to inspect, not an estimate of how often a given architecture develops a compact field. Preserve condition, seed, DG index, probe coverage, activity fraction, and scale convention when choosing examples for the submission.

[Selected-unit values](selected_units.csv) and the [source hash manifest](manifest.json) make the sheets auditable. The original atlas PNGs, metric tables, submitted figures, and Overleaf ZIP are unchanged. Regenerate from the existing saved artifacts with:

```bash
MPLCONFIGDIR=/tmp/cosyne_eight_fields_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python ../../render_eight_field_gallery.py
```
