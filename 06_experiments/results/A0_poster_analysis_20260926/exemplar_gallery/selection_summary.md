# Poster exemplar and architecture gallery

This gallery is a choice set, not a final poster selection. The [complete run index](run_index.md) links **all 39 candidate runs** to seven per-run panels: DG fields, mono-field peak locations, frozen-policy trajectory and occupancy, occupancy flow, checkpoint-stored graph, full-range DG → CA3 → decoder-1 kernels, and radial kernel profiles. The [numeric inventory](exemplar_inventory.csv) lists the exact checkpoint, unit counts, spatial information, active-only map cosine, peak-bin diversity, coverage, flow, graph, and long-range kernel values for every run. The main [batch report](../batch_summary.md) supplies methods, common-history replay provenance, command interventions, paired reward curves, heldout transfer tests, and limitations.

The [historical extension](historical_extension/report.md) adds **42 run-protocol analyses** across corrected-core C04/C05/C15, CPD C15, and Navigation8, with [all individual panels](historical_extension/run_index.md) and separate [CPD frozen](historical_extension/aggregate_cpd_frozen.svg), [CPD online](historical_extension/aggregate_cpd_online.svg), [Navigation8](historical_extension/aggregate_n8_online.svg), and [corrected-core](historical_extension/aggregate_ccr_frozen.svg) aggregates. Frozen corrected-core and CPD probes use the latest exact shared checkpoints, 100,040,704 and 75,038,720 frames, respectively. CPD and Navigation8 online windows end at 75,005,952 and are labeled developmental; Navigation8 has no exact terminal checkpoint shared across the selected seeds. The [CPD](historical_extension/kernel_long_cpd.svg) and [corrected-core](historical_extension/kernel_long_ccr.svg) common-history kernel summaries include all three layers over the full arena. The earlier [candidate screen](historical_extension_screen.md) records initial selection rather than the final checkpoint audit.

## Scope and checkpoint rule

The gallery contains 36 current runs: three seeds per arm of Saturday ARR/SRC, CPU2048 Direct/Waypoint HER, DGP HIT/FIRST, DGC Direct/Waypoint candidates, D50 source/random transfer, and D51 source/random transfer. It also contains three mature single-run exemplars: CPU2048 Direct DDQN, CPU2048 Waypoint DDQN, and full-system PPO. All endpoint pairs use the **latest exact checkpoint shared by their two conditions and all three seeds**. DGC Waypoint has no common three-seed checkpoint, so its three latest individual checkpoints are shown only as candidates; the Direct/Waypoint DGC figures are age-unmatched. The mature single runs do not enter three-seed architecture means. Earlier transfer and development checkpoints are in the batch report solely to show learning dynamics.

| Family | Exact current frames per seed |
| --- | --- |
| Saturday ARR/SRC | 75,038,720 for seeds 8, 99, 123 |
| CPU2048 Direct/Waypoint DDQN+HER, cadence 2048 | 300,007,424 for seeds 8, 99, 123 |
| DGP HIT/FIRST | 75,038,720 for seeds 8, 99, 123 |
| DGC Direct-Worker F16 | 300,023,808 for seeds 8, 99, 123 |
| DGC Waypoint F64 | seed 8: 197,509,120; seed 99: 158,203,904; seed 123: 135,004,160 |
| D50 source/random transfer | 75,022,336 for seeds 42, 1234, 9999 |
| D51 source/random transfer | 75,038,720 for seeds 42, 1234, 9999 |
| Mature Direct DDQN / Waypoint DDQN / full-system PPO | 300,007,424 / 300,007,424 / 192,266,240; one seed each |

The [refreshed checkpoint inventory](../../../data/poster_missing_analyses_20260926/checkpoint_inventory_current.tsv) and [full-range replay manifests](../../../data/poster_missing_analyses_20260926/fullrange_plan/) provide exact source paths. The 54 full-range replay rows also include 15 developmental snapshots; the 39-run gallery focuses on current candidates and the three mature singles.

## Aggregate figures

These SVGs are editable; matching PNGs sit beside them. Times New Roman is resolved from an installed scalable font, and the source artwork uses 12 pt text and approximately 1.3 pt lines. The small multiples are sized for readable A4 review and can be enlarged for A0.

| Figure | What to inspect |
| --- | --- |
| [Architecture overview](architecture_overview.svg) | Three seed dots and condition mean bars for active DG map cosine, distinct peak bins, coarse-cell coverage, and stored-graph reachability. DGC Waypoint ages differ. |
| [Field quality overview](field_quality_overview.svg) | Active-unit spatial information and fraction of active DG units with a defined field peak. |
| [Mono-field overview](mono_field_overview.svg) | Per-seed dots and condition means for mono-field units divided by **all** DG units and distinct mono-field peak bins. |
| [Representation versus graph scatter](representation_graph_scatter.svg) | Every seed, with family-specific panels; use to see whether a single mean hides seed variation. These are associations, not a mediation analysis. |
| [All-available-run scatter survey](../cross_run_scatter/report.md): [online](../cross_run_scatter/online_latest_saved_window.svg), [frozen](../cross_run_scatter/frozen_latest_archived_probe.svg), [common history](../cross_run_scatter/common_replay_with_frozen_outcomes.svg) | 323 distinct run names, including the full local CPD/DGP matrices and architecture follow-ups. Spatial score and peak-bin diversity are contrasted with graph/control and exploration; protocols and DG capacities remain explicit. |
| [Raw versus normalized peak diversity](../cross_run_scatter/peak_capacity_contrast.svg), [within-family control](../cross_run_scatter/online_within_family_control.svg) | Candidate poster controls for pooled associations: capacity normalization reverses the raw peak-count/graph ordering, and within-family correlations are much weaker than the pooled spatial-score/control association. |
| [Corridor probability scatter](../cross_run_scatter/corridor_layouts.svg), [representation](../cross_run_scatter/corridor_probability_representation.svg), [control / exploration](../cross_run_scatter/corridor_probability_control_exploration.svg) | All 27 corridor runs at a shared saved age, showing 0%, 35%, and 75% wall removal across three architectures and three layout seeds. Total-grid visitation rises while prospective event success falls; accessibility affects the coverage denominator. |
| [Control diagnostics](control_diagnostics.svg) | Executed-minus-shuffled command success and complete ordered-pair coverage. DGC Waypoint's separate exact-start probe has a different protocol and is described in the batch report. |
| Peak distributions: [Saturday](peak_map_saturday.svg), [CPU2048](peak_map_cpu2048.svg), [DGP](peak_map_dgp.svg), [DGC](peak_map_dgc_candidates.svg), [D50](peak_map_d50_transfer.svg), [D51](peak_map_d51_transfer.svg) | Each arm pools three frozen-policy seed maps on a 19×19 arena grid. Color is share of **defined** active-unit peaks within that arm, with a common color scale within each paired family. The annotation reports defined peaks over all active units. |
| Mono-field peaks by run: [Saturday](mono_peak_saturday.svg), [CPU2048](mono_peak_cpu2048.svg), [DGP](mono_peak_dgp.svg), [DGC](mono_peak_dgc_candidates.svg), [D50](mono_peak_d50_transfer.svg), [D51](mono_peak_d51_transfer.svg) | Six seed-by-arm contact sheets show only canonical mono-field unit peaks over each run's own frozen-policy visitation mask. Each panel states mono-field units / all DG units and distinct peak bins; dot area increases for coincident peaks. The [run index](run_index.md) links the larger occupancy-plus-peak panel for every one of the 39 runs, including the three mature singles. |
| Long-range layer profiles: [Saturday](kernel_long_saturday.svg), [CPU2048](kernel_long_cpu2048.svg), [DGP](kernel_long_dgp.svg), [DGC](kernel_long_dgc_candidates.svg), [D50](kernel_long_d50_transfer.svg), [D51](kernel_long_d51_transfer.svg) | DG, CA3, and decoder-1 population correlation at supported 13–18-bin offsets, three seeds per arm. Matched seeds are connected except in age-unmatched DGC. Full 37×37 kernels are linked per run. |
| [Paired source/random reward curves](../poster_candidates/reward_curves_paired_0_75m.svg), [75M heldout success](../poster_candidates/five_cue_75m_heldout.svg) | Transfer performance should accompany representation panels; reward AUC and exact matched-reset outcomes are in the batch report. |

## Mono-field peak analysis for every run

This follows the updated C15 context-panel idea: put the DG peak coordinates on the same 19×19 arena grid as frozen-policy occupancy, while displaying only units that pass the **canonical mono-field rule**. A unit must be field-eligible and have one dominant 8-connected superlevel component containing at least 80% of superlevel mass at **30%, 50%, and 70%** of its peak. The numerator below counts such units with a valid peak bin; the denominator is **all 16 or 64 DG units**, including silent and ineligible units. The older `mono_field_fraction` in evaluator summaries instead divides by *eligible* units. Exact unit-level flags and coordinates are in [the saved peak table](../../../data/poster_missing_analyses_20260926/frozen_peak_positions_with_mature_direct.csv); the new [count table](mono_field_peak_counts.csv) gives all 39 denominators, eligible counts, mono-field counts, fractions, and distinct mono-field peak bins.

| Condition | Seed-specific mono-field / all DG; distinct peak bins in parentheses |
| --- | --- |
| Saturday ARR | S8 0/16 (0); S99 1/16 (1); S123 3/16 (2) |
| Saturday SRC | S8 0/16 (0); S99 0/16 (0); S123 0/16 (0) |
| CPU2048 Direct HER | S8 1/16 (1); S99 0/16 (0); S123 0/16 (0) |
| CPU2048 Waypoint HER | S8 3/64 (3); S99 2/64 (2); S123 1/64 (1) |
| DGP HIT | S8 1/16 (1); S99 0/16 (0); S123 3/16 (3) |
| DGP FIRST | S8 2/16 (1); S99 5/16 (3); S123 2/16 (2) |
| DGC Direct candidate | S8 6/16 (3); S99 3/16 (2); S123 0/16 (0) |
| DGC Waypoint candidate | S8 12/64 (8); S99 30/64 (9); S123 15/64 (11) |
| D50 SOURCE_DG | S42 14/64 (14); S1234 8/64 (8); S9999 8/64 (7) |
| D50 RAND_DG | S42 9/64 (9); S1234 10/64 (9); S9999 13/64 (12) |
| D51 SOURCE_DG | S42 1/64 (1); S1234 0/64 (0); S9999 1/64 (1) |
| D51 RAND_DG | S42 7/64 (6); S1234 4/64 (4); S9999 9/64 (5) |
| Mature Direct DDQN | S99 0/16 (0) |
| Mature Waypoint DDQN | S99 1/64 (1) |
| Mature full-system PPO | S123 9/64 (4) |

The new subset changes how the all-peak distributions should be read. Saturday ARR has 0–3 mono-field units per seed and SRC has none; a spread of **all** DG peaks there does not imply many compact individual fields. DGP FIRST has more mono-field units than HIT (mean 3.0 versus 1.3 of 16) despite its much sparser stored graph. D50 source and random have similar three-seed mono counts (10.0 versus 10.7 of 64), whereas D51 source has only 0–1 of 64 and random has 4–9 of 64. DGC Waypoint seed 99 places 30 qualifying unit peaks into just nine bins, with several overlapping near one corner. Its ages differ from Direct, and both capacity and policy visitation differ, so the panel is a candidate illustration rather than a matched-age effect. Runs with zero qualifying units are explicitly shown as empty, not omitted.

The added exact-checkpoint CPD frozen probes yield a stronger 16-unit historical example: [ADD CA3 DIR seed 8](historical_extension/per_run/CPD_C15_ADD_CA3_DIR_S8_75038720/mono_field_peaks.svg) has **7/16** mono-field units, but only three distinct peak bins. [GATE ACT DIR GOAL seed 99](historical_extension/per_run/CPD_C15_GATE_ACT_DIR_GOAL_S99_75038720/mono_field_peaks.svg) has **4/16 in four bins** and is the clearer distributed example. Three-seed totals are 10/48 and 7/48, respectively; each condition also has one zero-mono seed. The [terminal CPD aggregate](historical_extension/aggregate_cpd_frozen.svg) and [per-run inventory](historical_extension/historical_exemplar_inventory.csv) should accompany any exemplar panel.

## Quantitative guide to the paired families

Values below are arithmetic means of three seeds at the exact current checkpoints, from [the run inventory](exemplar_inventory.csv). `Peaks` counts distinct arena bins containing a DG-unit field peak, so it differs from the number of eligible units in the peak-map annotation. `Long` is the all-history mean population correlation for supported 13–18-bin spatial offsets in DG / CA3 / decoder-1; lower values indicate more population change across that displacement, but do not alone establish better control. Graph reachability is the fraction of ordered node pairs connected by the canonical thresholded **stored** graph.

| Family and arm | Active DG | Eligible peak fraction | Active SI (bits) | Active map cosine | Distinct peak bins | Visited cells | Flow coherence | Graph reach | Long DG / CA3 / Dec-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Saturday ARR | 16.0 | 0.979 | 0.176 | 0.081 | 15.7 | 0.757 | 0.453 | 0.958 | 0.138 / 0.273 / 0.870 |
| Saturday SRC | 16.0 | 1.000 | 0.112 | 0.178 | 15.0 | 0.802 | 0.439 | 0.958 | 0.073 / 0.185 / 0.883 |
| CPU2048 Direct HER | 16.0 | 1.000 | 0.099 | 0.158 | 15.3 | 0.705 | 0.337 | 0.938 | 0.042 / 0.060 / 0.936 |
| CPU2048 Waypoint HER | 43.7 | 0.952 | 0.085 | 0.138 | 33.3 | 0.693 | 0.248 | 0.010 | 0.145 / 0.168 / 0.934 |
| DGP HIT | 16.0 | 1.000 | 0.164 | 0.078 | 15.3 | 0.603 | 0.492 | 0.961 | 0.136 / 0.251 / 0.750 |
| DGP FIRST | 16.0 | 0.979 | 0.165 | 0.083 | 14.3 | 0.573 | 0.383 | 0.049 | 0.190 / 0.334 / 0.778 |
| DGC Direct candidate | 16.0 | 1.000 | 0.132 | 0.132 | 12.7 | 0.503 | 0.324 | 0.958 | 0.370 / 0.303 / 0.716 |
| DGC Waypoint candidate | 63.3 | 0.968 | 0.144 | 0.090 | 34.7 | 0.474 | 0.681 | 0.089 | 0.081 / 0.096 / 0.530 |
| D50 SOURCE_DG | 55.0 | 0.557 | 0.043 | 0.045 | 44.3 | 0.453 | 0.189 | 0.053 | 0.163 / 0.164 / 0.471 |
| D50 RAND_DG | 63.0 | 0.974 | 0.009 | 0.073 | 50.7 | 0.805 | 0.539 | 0.103 | 0.029 / 0.049 / 0.700 |
| D51 SOURCE_DG | 37.0 | 0.383 | 0.043 | 0.050 | 28.3 | 0.602 | 0.290 | 0.014 | 0.664 / 0.663 / 0.824 |
| D51 RAND_DG | 63.7 | 0.922 | 0.009 | 0.065 | 52.0 | 0.829 | 0.512 | 0.081 | 0.038 / 0.051 / 0.644 |

The strongest graph contrast is DGP HIT versus FIRST: reachability is 96.1% versus 4.9%, with nearly equal frozen-policy DG map cosine. CPU2048 Waypoint spreads defined peaks over more bins, yet its stored graph reachability is only 1.0% versus Direct's 93.8%. Saturday ARR has lower frozen-policy field overlap than SRC, while both have high stored reachability. These differences are descriptive; the command figure shows the bounded intervention coverage and should accompany a graph-control claim.

In five-cue transfer, source DG has higher active-unit spatial information than calibrated random DG at both D50 and D51, but fewer units with eligible peaks and lower policy coverage. At D50 the source/random long-range ordering in DG and CA3 reverses in decoder-1; at D51 the source's higher long-range correlation persists through decoder-1. This is a population-similarity difference, not evidence by itself that one arm has better place coding. The [paired reward curves](../poster_candidates/reward_curves_paired_0_75m.svg) and [heldout results](../poster_candidates/five_cue_75m_heldout.svg) favor random at the 75M endpoint overall; D50 seed-level outcomes are mixed.

## Suggested panels to inspect first

These are entry points into the **complete** [39-run index](run_index.md), not a restricted set of analyzed runs. The selected units in each field sheet are the four highest active-unit SI values for visual browsing; all active units contribute to the aggregate metrics. A high-SI field can still follow a wall or sparse boundary sampling.

| Story or visual | First pair of run panels | Key statistic and caution |
| --- | --- | --- |
| Saturday field overlap | [ARR seed 99 fields](per_run/SAT_C15_ARR_MON_FILM_S99_75038720/top_four_fields.svg) and [SRC seed 99 fields](per_run/SAT_C15_SRC_MON_FILM_S99_75038720/top_four_fields.svg); also [ARR seed 123 fields](per_run/SAT_C15_ARR_MON_FILM_S123_75038720/top_four_fields.svg) | ARR/SRC mean frozen map cosine 0.081/0.178; seed 123 ARR contains visually strong boundary hotspots, so label the actual unit map rather than calling every peak compact. |
| CPU2048 capacity and graph | [Direct seed 8 fields](per_run/CPU2048_DIRECT_F16_DDQN_HER_S8_300007424/top_four_fields.svg), [Waypoint seed 8 fields](per_run/CPU2048_WAYPOINT_DECODER_F64_DDQN_HER_S8_300007424/top_four_fields.svg), and their [Direct](per_run/CPU2048_DIRECT_F16_DDQN_HER_S8_300007424/stored_graph.png) / [Waypoint](per_run/CPU2048_WAYPOINT_DECODER_F64_DDQN_HER_S8_300007424/stored_graph.png) graphs | Seed 8 has 16/40 distinct peak bins and 1.00/0.02 reachability, Direct/Waypoint. The conditions also differ in manager and goal interface. Seed 99 reverses the family mean map-cosine ordering, so seed 8 better illustrates that particular trend. |
| DGP graph formation | [HIT seed 99 graph](per_run/DGP_C15_HIT_JOINT_LEG_S99_75038720/stored_graph.png) and [FIRST seed 99 graph](per_run/DGP_C15_FIRST_JOINT_LEG_S99_75038720/stored_graph.png), plus the [control diagnostic](control_diagnostics.svg) | Seed 99 graph reachability is 0.883/0.071. Stored confidence is not a fresh prospective transition test. |
| DGC candidate survey | [Direct seed 99 trajectory](per_run/DGC_DIRECT_WORKER_F16_S99_300023808/occupancy_trajectory.png), [Waypoint seed 99 trajectory](per_run/DGC_WAYPOINT_DG_F64_S99_158203904/occupancy_trajectory.png), [all peaks](peak_map_dgc_candidates.svg), and [mono-field peaks](mono_peak_dgc_candidates.svg) | Waypoint seed 99 has 30 mono-field units but nine distinct mono peak bins. Direct is 300M frames while Waypoint seed 99 is 158M; show ages and avoid a matched-age claim. |
| D50 decoder reversal | [Source seed 42 kernels](per_run/CR5C_D50_W_SOURCE_DG_S42_75022336/layerwise_kernels.svg), [random seed 42 kernels](per_run/CR5C_D50_W_RAND_DG_S42_75022336/layerwise_kernels.svg), and [three-seed profile](kernel_long_d50_transfer.svg) | Long DG/CA3 correlation is higher for source, but decoder-1 is lower: 0.471 versus 0.700 across seeds. The matched heldout source deficit is mixed by seed. |
| D51 transfer behavior and persistence | [Source seed 42 trajectory](per_run/CR5C_D51_W_SOURCE_DG_S42_75038720/occupancy_trajectory.png), [random seed 42 trajectory](per_run/CR5C_D51_W_RAND_DG_S42_75038720/occupancy_trajectory.png), [three-seed kernels](kernel_long_d51_transfer.svg), [all peaks](peak_map_d51_transfer.svg), and [mono-field peaks](mono_peak_d51_transfer.svg) | Source/random coverage is 0.602/0.829 and graph reachability 0.014/0.081 across seeds; long-range correlation difference remains through decoder-1. Source seed 1234 has a 1.62-bit top DG unit that forms a boundary stripe, not a compact isolated field, and 0/64 strict mono-fields. |
| Mature model reference | [Direct DDQN](per_run/CPU2048_DIRECT_F16_DDQN_S99_300007424/layerwise_kernels.svg), [Waypoint DDQN](per_run/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_300007424/layerwise_kernels.svg), [full-system PPO](per_run/FSCS_WAYPOINT_DECODER_F64_PPO_S123_192266240/layerwise_kernels.svg) | One seed per architecture; use as examples, not architecture averages. The historical Direct graph image comes from its existing online graph-outcomes archive, and its graph-reachability metric is absent from the harmonized inventory. |

## Interpretation and provenance

Frozen-policy rollouts supply place fields, trajectory/occupancy, flow, and checkpoint-stored graph structure. Representation comparisons and layer kernels use a **common replayed observation/action-history panel** within each matched family so different visitation does not drive those contrasts. A policy trajectory should therefore not be presented as the path that produced the common-history kernel. Each full-range kernel spans spatial offsets $-18$ through $+18$ bins on both axes; the identity cell is masked, and an offset enters the 13–18-bin summary only with at least ten eligible cell pairs. The supported 19–25-bin diagonal band is much sparser. Equal-cue long-range transfer estimates are not claimed for all five cues because cue 2 has no supported 13–18-bin offsets.

The peak map is a density of **eligible unit peak positions**, not a place-field coverage map. Source DG in D50 has 93 eligible peaks among 165 active units across seeds; D51 has 43 among 111. Random DG has 184/189 in D50 and 176/191 in D51. Unvisited bins are not evidence of absent fields. Graph matrices display stored edge confidence per attempt; the command probes test control separately and cover a limited set of ordered pairs. The stored graph cannot establish prospective success for every pair.

This pass reused the saved frozen NPZ archives and full-range kernel arrays; it launched no new DMLab evaluation. The local peak-position extractor is [extract_poster_peak_positions.py](../../../extract_poster_peak_positions.py) and the reusable figure/inventory adapter is [prepare_poster_exemplar_gallery.py](../../../prepare_poster_exemplar_gallery.py). Compact top-four field maps and the 1,584-unit peak-position table are under [poster data](../../../data/poster_missing_analyses_20260926/); bulk raw archives remain in the active NEMO2 workspace. Reusing the original frozen images avoided an incompatible local rendering function signature; future regeneration should use the evaluator version that wrote each archive or adapt the renderer's graph arguments explicitly.
