# Cross-run representation, controllability, and exploration

This survey contains **323 distinct run names** from all locally retained canonical spatial summary tables and the available poster/corrected-core frozen probes. It uses existing saved data throughout. The [point table](../../../data/poster_missing_analyses_20260926/cross_run_scatter/all_run_metrics.csv) records each run, condition, seed, exact observation/checkpoint age, protocol, capacity, source, and plotted metric. The [source audit](../../../data/poster_missing_analyses_20260926/cross_run_scatter/source_audit.csv) and [analysis metadata](../../../data/poster_missing_analyses_20260926/cross_run_scatter/analysis_metadata.json) preserve fingerprints and resolved StudySpec metadata.

## Scope and sample unit

| Data stratum | Run rows | Representation input | Control/exploration input |
| --- | ---: | --- | --- |
| Latest available online spatial window | 283 | Saved 100k-observation policy-driven training window | Graph buffers/counters at that snapshot; occupancy from the same window |
| Latest archived frozen-policy probe | 84 | Saved 10k-decision frozen policy evaluation | Stored graph where available; frozen-policy occupancy |
| Common-history representation joined to frozen outcomes | 38 | Existing common observation/action-history panel within a comparison family | Exact-age frozen policy evaluation and stored graph |

The 283 online rows comprise 81 CPD, 24 DGP, 27 DG-capacity, 24 CPU cadence/controller, 24 CA3-state, 18 Navigation8, 30 source-credit, 16 persistent-control, 12 five-cue transfer, and 27 corridor-layout runs. The 84 frozen rows include all 39 original poster candidates, 33 corrected-core runs, and 12 terminal CPD runs. A run can appear in multiple protocols; the 38 common-history rows reuse frozen outcomes and add a separate representation view.

For each run/policy/protocol, the latest **available saved** age is selected without inspecting its score. Repeated exports of the same run/policy/age are deduplicated, checked for metric agreement, and coalesced from the export with richer metadata. The [970-snapshot inventory](../../../data/poster_missing_analyses_20260926/cross_run_scatter/online_snapshot_inventory.csv) retains earlier observations for learning dynamics. Online ages range from 25,001,984 to 300,007,424 frames; archived frozen ages range from 75,022,336 to 300,023,808. These broad survey plots do not define matched-checkpoint condition contrasts. The matched poster comparisons retain their audited checkpoints in the [batch report](../batch_summary.md).

## Figures to inspect

Colors identify experiment families. Circles, triangles, and squares identify DG capacities 16, 32, and 64. Each point is one run-level measurement; the caption gives the number with both plotted metrics. Missing values are excluded only from the affected panel. All sheets have editable SVG and PNG copies; [individual panels](panels/) are also exported for poster selection with the same scales and 12 pt source fonts.

| Figure | Question |
| --- | --- |
| [Online representation / graph / exploration](online_latest_saved_window.svg) | How do spatial score and absolute peak-bin diversity relate to graph reachability and visited area across the 256 legacy-layout online runs? |
| [Frozen representation / graph / exploration](frozen_latest_archived_probe.svg) | Does the association also appear in archived fixed-policy evaluations? Graph data exist for 50 of 84 rows. |
| [Common-history representation with frozen outcomes](common_replay_with_frozen_outcomes.svg) | What changes when representation is measured on a common replay panel rather than each policy's own visitation? |
| [Online prospective and grounded control](online_prospective_control.svg) | How do representation metrics relate to measured command-event counters and spatial grounding? |
| [Within-family prospective control](online_within_family_control.svg) | Are pooled trends also present within CPD, DGP, CPU, DG-capacity, and other families? |
| [Raw versus capacity-normalized peak diversity](peak_capacity_contrast.svg) | How much does DG capacity change the apparent peak-diversity/graph relationship? |
| [Frozen normalized diversity and mono-field fraction](frozen_latest_archived_probe_normalized.svg) | How do capacity-normalized peak counts and strict mono-fields relate to reachability and exploration? |
| [Online exploration detail](online_exploration_detail.svg) | Inspect the near-ceiling online coverage values with an expanded vertical scale; every point is retained. |
| [Online control / exploration](online_latest_saved_window_control_exploration.svg), [frozen control / exploration](frozen_latest_archived_probe_control_exploration.svg) | Do graph/control measures accompany broader visitation within each protocol? |
| [Common-history representation / matched command benefit](common_replay_command_control.svg) | For the 21 available exact-age command interventions, does representation relate to executed-minus-shuffled success? |
| [Corridor-layout scatter](corridor_layouts.svg) | Color distinguishes wall-removal probabilities 0%, 35%, and 75%; shape distinguishes SAT arrival F16, DGP hit F16, and Waypoint HER F64. |
| [Wall removal / representation](corridor_probability_representation.svg), [wall removal / control and exploration](corridor_probability_control_exploration.svg) | Compare probabilities within each architecture, retaining all three layout seeds and showing their mean. |

## Metric definitions

- **Spatial score:** The existing mean active-unit spatial-information statistic, computed from occupancy-corrected rate maps. Its canonical formula is $I=\sum_b p_b r_b\log_2(r_b/\bar r)$, with $\bar r=\sum_b p_b r_b$. It is amplitude weighted; the plots label it as a spatial score rather than implying information per activation.
- **Distinct active DG peak bins:** Count of different 19×19 grid bins containing the maximum of at least one active DG map. This includes broad and multi-field units. The capacity-normalized variant divides by all DG units; its denominator is recovered from validated StudySpec arguments or the same run's frozen summary. All selected rows have known capacity.
- **Mono-field fraction:** Canonical mono-field units divided by **all** DG units, available for 60 frozen rows. Online `mono_field_unit_fraction` uses eligible units as its denominator and is retained separately in the CSV. It is not substituted into the all-DG plot.
- **Stored-graph reachability:** Fraction of ordered DG-node pairs connected by the canonical thresholded stored graph. It describes the checkpoint's graph evidence. Frozen archives without graph buffers remain missing.
- **Prospective success:** Recorded successful DG target events divided by attempted prospective transitions at the online snapshot. These buffers aggregate historical command evidence, whereas spatial fields use the recent 100k-observation window. Zero-attempt cases remain missing. This event criterion differs from physical target-region arrival.
- **Grounded controllability:** The canonical prospective-success fraction multiplied by the fraction of reliable edges whose endpoints pass the spatial field criterion. Its dependence on field eligibility is part of its definition.
- **Exploration:** Visited coarse-grid cells divided by all grid cells during the saved window/probe. It is an occupancy-coverage proxy, with different sample budgets across protocols; it is not coverage AUC or a common-policy test. Many online legacy-layout values cluster at the observed maximum of 0.881.
- **Matched command benefit:** Executed command success minus success for a matched shuffled command, from the existing heldout intervention protocol at the exact same model age.

## Main quantitative observations

The [association table](../../../data/poster_missing_analyses_20260926/cross_run_scatter/descriptive_associations.csv) gives descriptive Spearman correlations, sample counts, protocol, family, and capacity strata. No pooled regression, significance test, or causal claim is imposed on these heterogeneous studies.

1. **The pooled online spatial-score/control trend largely separates families.** Spatial score versus prospective success has $\rho=0.769$ across 234 legacy-layout runs with graph counters. Within CPD it is $\rho=0.094$ across 81 runs, and within DGP it is $\rho=-0.001$ across 24. The family-specific panels are necessary when interpreting the pooled cloud.
2. **Capacity changes the peak-diversity interpretation.** Absolute unique peak bins versus graph reachability has $\rho=-0.422$ online and $\rho=-0.368$ in frozen probes. Dividing by all DG units changes these to $\rho=0.510$ and $\rho=0.556$, respectively. The common-history view shows the same reversal, $-0.485$ to $0.761$. These contrasting associations expose capacity/architecture structure in the survey; they do not establish that increasing diversity improves control.
3. **Exploration is only weakly linked to these representation summaries.** Spatial score versus online occupancy coverage has $\rho=-0.285$ across 256 runs, but common-history spatial score versus frozen coverage is essentially zero, $\rho=-0.0003$ across 38. Stored-graph reachability versus frozen coverage is $\rho=0.053$ across 50 probes. Online coverage is heavily saturated, so inspect the expanded coverage view as well as the full 0–1 scale.
4. **The heldout command test does not show a simple monotonic spatial-score benefit.** Common-history spatial score versus executed-minus-shuffled success has $\rho=-0.186$ across 21 available run rows. The sample combines intervention families, so the plot is a screening view rather than a universal law.

For poster selection, the strongest broad figure candidates are **raw versus normalized peak diversity**, **pooled versus within-family prospective control**, and **common-history representation versus frozen exploration**. Together they show which associations survive a change in measurement and grouping.

## Corridor layouts: wall-removal probability

All **27 runs** share the latest saved age **100,007,936 frames**, with 100k-observation spatial windows. There are three wall-removal probabilities (0%, 35%, 75%), three architectures, and three layout seeds (1001–1003), all with training seed 99. Architecture and probability are verified against the validated corridor StudySpec; they are not inferred from run names. The point table retains `openness` and the corridor-specific [run table](../../../data/poster_missing_analyses_20260926/cross_run_scatter/corridor_probability_per_run.csv) also names it explicitly as `wall_removal_probability`.

The scatter uses probability color and architecture shape. The probability-comparison figures show faint traces for equal layout seeds across probabilities and bold architecture means. The small horizontal offsets only separate overlapping architectures. These traces do not assert nested wall sets, and three layout seeds are not three independent training seeds. The [summary table](../../../data/poster_missing_analyses_20260926/cross_run_scatter/corridor_probability_summary.csv) retains means, sample standard deviations across layouts, minima, maxima, and counts; [75%-minus-0% differences](../../../data/poster_missing_analyses_20260926/cross_run_scatter/corridor_probability_paired_differences.csv) retain each architecture/layout seed pair. No inferential error bars or significance tests are used.

Mean visited-grid fraction across the three layout seeds:

| Architecture | 0% wall removal | 35% wall removal | 75% wall removal |
| --- | ---: | ---: | ---: |
| SAT arrival F16 | 43.0% | 68.9% | 89.6% |
| DGP hit F16 | 41.7% | 69.3% | 75.2% |
| Waypoint HER F64 | 55.1% | 73.1% | 89.6% |

Coverage is divided by **all coarse-grid cells**, including inaccessible geometry. Its rise therefore combines changes in accessible area and visitation; it cannot alone establish improved exploration of the available floor. Meanwhile, prospective target-event success decreases from 0% to 75% wall removal: SAT 56.6% to 48.0%, DGP 60.4% to 54.1%, and Waypoint 47.0% to 39.4%. Stored graph reachability stays at 100% for DGP, rises from 95.8% to 100% for SAT, and remains low for Waypoint (1.05% to 0.50%). Grounded controllability is zero in 26 runs; only SAT, probability 75%, layout seed 1001 is nonzero (0.0703). These are distinct measures and should appear together when assessing geometry sensitivity.

The reusable lesson is to expose the manipulated geometry factor directly in the figures and preserve architecture/layout strata. A single family color hid the probability contrast in the original corridor scatter.

## Reuse and reproducibility

The adapter [render_all_run_spatial_scatter.py](../../../render_all_run_spatial_scatter.py) reuses the canonical StudySpec loader, previously calculated spatial/graph metrics, and the shared poster plotting style. [Pinned input tables and relevant studies](../../../data/poster_missing_analyses_20260926/cross_run_scatter/input_sources/) make the plot set reproducible without cluster access or unpublished neighboring work. To replot, use the adapter's `--source-bundle` option with that folder; source filenames and hashes remain attached to the point table. The earlier historical adapter now also records canonical all-active peak-bin counts, avoiding confusion with its mono-field-only peak counts.

What worked was identifying exports by their standardized metric contract and preserving protocol-specific point rows. The main interpretation trap was pooling capacities and study families: raw and normalized peak relationships point in opposite directions. Future surveys should start from the pinned point table, verify duplicate agreement, and inspect family/capacity strata before selecting a poster trend.
