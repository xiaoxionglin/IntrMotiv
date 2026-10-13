# Five four-field options for Figure 1

[View all five in one scrollable overview](all_five_overview.png). Each option below also has a full-size PNG and PDF; the current submitted figures and Overleaf ZIP are unchanged.

| Option | Four DG units | Visited bins | Strict mono-fields | Visual tradeoff |
| --- | --- | ---: | ---: | --- |
| [1. COSYNE C15 temporal, seed 123](01_cosyne_c15_temporal_s123.png) ([PDF](01_cosyne_c15_temporal_s123.pdf)) | 4, 3, 15, 12 | 307/361 (85.0%) | 2 | Best probe coverage and two strict scores; those two units are active in only 1.7% and 1.5% of observations. |
| [2. COSYNE C05 constant/temporal, seed 99](02_cosyne_c05_const_temporal_s99.png) ([PDF](02_cosyne_c05_const_temporal_s99.pdf)) | 5, 4, 15, 12 | 248/361 (68.7%) | 1 | Highest displayed spatial score (0.373) with one strict unit; lower coverage than option 1. |
| [3. Earlier C15, seed 8](03_archived_c15_s8.png) ([PDF](03_archived_c15_s8.pdf)) | 15, 10, 7, 14 | 249/361 (69.0%) | 1 | Two broad high-score maps; the strict unit is very sparse (0.2% active). |
| [4. Earlier C05, seed 123](04_archived_c05_s123.png) ([PDF](04_archived_c05_s123.pdf)) | 6, 1, 2, 3 | 257/361 (71.2%) | 1 | One higher-activity strict unit; DG 1 has a high spatial score but only 0.2% activity. |
| [5. Earlier C01, seed 99](05_archived_c01_s99.png) ([PDF](05_archived_c01_s99.pdf)) | 1, 2, 12, 13 | 264/361 (73.1%) | 0 | Highest four-unit mean spatial score (0.279), without a strict mono-field. |

**Reading the labels.** `SI` is the canonical amplitude-weighted occupancy-corrected spatial score, in bits per unit under the archived evaluator; it is sensitive to activity amplitude and should not be read as normalized bits per activation. `mono` is the minimum dominant-component mass across 30%, 50%, and 70% peak thresholds. A `strict` label requires an eligible unit and at least 0.8 mass at every threshold. A high mono score measures connected-field dominance; it does not ensure a small, clean receptive field. `active` is the fraction of probe decisions with a DG event. The score and active fraction together expose sparse units that might otherwise look unusually selective.

Each panel comes from one 10,001-decision stochastic frozen-policy probe at approximately 100 million training frames. Current COSYNE options 1 and 2 use a common color scale **within each run** and mark unvisited bins gray. Earlier matched-probe options 3–5 use independent zero-based color scales **for each unit** and mark unvisited bins white. These are illustrative networks, not estimates across seeds. Their different study lineages and display scales prevent using this gallery as a controlled comparison of learning interventions or response amplitude.

For the existing Figure 1, **option 2 offers the clearest numerical compromise** among spatial score, strict mono-field classification, and nontrivial activity in this shortlist, while option 1 has substantially better occupancy. The visual maps should be judged alongside those numbers. None of the five establishes that all four displayed DG units are compact single fields. Keep the selected option's condition, seed, probe coverage, unit identities, scale convention, and activity caveat in the submission caption.

The [unit-level selection table](selected_units.csv) and [hash manifest](manifest.json) record every plotted value and source. The current options use the saved COSYNE per-unit metric tables, probe summaries, and 16-unit atlas PNGs. The earlier options use the archived unit tables, probe table, and per-unit-scale SVGs. To regenerate all panels from these saved artifacts, run:

```bash
MPLCONFIGDIR=/tmp/cosyne_field_options_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python ../render_place_field_options.py
```

The renderer reads each SVG's own axes bounds before cropping. The archived SVG canvases have slightly different dimensions; a fixed pixel crop can include tick marks or omit edge bins. Source art and metric tables are never edited.
