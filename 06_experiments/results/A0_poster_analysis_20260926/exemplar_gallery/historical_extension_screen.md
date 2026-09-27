# Historical mono-field candidate screen

This is the first selection pass for extending the 39-run poster gallery. The [ranked condition table](../../../data/poster_missing_analyses_20260926/historical_extension/historical_candidate_ranking.csv) and [102 latest locally saved run rows](../../../data/poster_missing_analyses_20260926/historical_extension/historical_candidate_runs.csv) were read from canonical historical spatial summaries. The [screening adapter](../../../screen_poster_historical_candidates.py) keeps only each condition–seed's latest locally recorded snapshot. Exact checkpoint-file availability must be refreshed before new matched comparisons or kernel replay.

The screen uses the historical evaluator's **mono-field fraction among eligible units**. It does not silently convert that fraction to mono-field units divided by all DG units. The existing [39-run frozen-policy gallery](mono_field_peak_counts.csv) reports the latter from raw unit flags. Online 100k training-window snapshots and frozen-policy 10k-decision probes also differ in visitation. Compare condition patterns within a protocol first; avoid treating all percentages below as one pooled ranking.

| Candidate family | Latest saved three-seed mono-field fraction | Initial interpretation |
| --- | --- | --- |
| Navigation8 `N8_W_REF_STOP` | 25.0%, 6.25%, 12.5%; mean 14.6% | All three seeds qualify; a useful replicated historical comparator. |
| CPD C15 `CPD_C15_BASE` | 25.0%, 6.25%, 6.25%; mean 12.5% | All three seeds qualify; a useful C15-family baseline. |
| Navigation8 `N8_W_REF_JOINT` | 12.5%, 12.5%, 6.25%; mean 10.4% | All three seeds qualify; contrasts with STOP. |
| Navigation8 `N8_SCR_ARR_DIRS` | 50.0%, 0%, 6.25%; mean 18.8% | Strong seed-8 illustration, not a consistent three-seed effect. |
| CPD C15 `CPD_C15_GATE_ACT_DIR_GOAL` | Mean 18.8%; range 0–43.8% | Strong individual seed with high between-seed variation. |
| CPD C15 `CPD_C15_GATE_ACT_DIR` | Mean 12.5%; range 0–37.5% | Strong individual seed with high between-seed variation. |
| CPD C15 `CPD_C15_ADD_CA3_DIR` | Mean 12.5%; range 0–31.25% | Strong individual seed with high between-seed variation. |
| Corrected-core C15 `CCR_C15_TOPOLOGY_UCB_DIRECT_O1` | 6.25%, 0%, 0% at 100.04M | Historical architecture anchor rather than a high-mono exemplar; a seed-99 50.10M value of 6.25% is developmental only. |
| Corrected-core C05 | Mono-field fraction unavailable in the lightweight local table | The completed 100M sweep has saved evaluator NPZs in the historical NEMO2 workspace. Its earlier report flags 12/16, 15/16, and 14/16 units with multiple raw half-peak regions; that is a different diagnostic and cannot substitute for the strict mono-field count. |

The Navigation8 [saved per-unit table](../../../data/navigation8_algorithm_screen_interim_20260916/online_spatial/per_unit.csv) confirms that all 16 units are eligible in the selected 75M runs. STOP has 4/16, 1/16, and 2/16 mono-field units in seeds 8/99/123, at four, one, and two distinct mono-peak bins. JOINT has 2/16, 2/16, and 1/16, at two, two, and one bins. SCR's seed-8 spike is 8/16 units but only **five** distinct bins; seeds 99/123 have 0/16 and 1/16. Thus the large SCR percentage is an individual example with spatial peak clustering, rather than a replicated representation advantage.

The saved 75M online-window summaries also supply context for these candidates. Three-seed means are below; spatial information is the canonical amplitude-weighted score, stored-graph reachability is a checkpoint buffer diagnostic, and prospective success is conditional on attempted transitions. Navigation8 reference baselines have no graph values in this table.

| Historical condition | Spatial score | Active map cosine | All-unit peak bins | Visited fraction | Graph reachable pairs | Prospective success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPD C15 BASE | 0.142 | 0.170 | 14.3 | 0.877 | 0.035 | 0.589 |
| CPD C15 GATE ACT DIR GOAL | 0.379 | 0.256 | 12.3 | 0.870 | 0.197 | 0.695 |
| CPD C15 GATE ACT DIR | 0.324 | 0.296 | 12.0 | 0.881 | 0.121 | 0.678 |
| CPD C15 ADD CA3 DIR | 0.162 | 0.276 | 11.7 | 0.877 | 0.103 | 0.667 |
| Navigation8 SCR ARR DIRS | 0.111 | 0.245 | 12.0 | 0.857 | 0.939 | 0.504 |
| Navigation8 Waypoint STOP | 0.073 | 0.262 | 14.7 | 0.878 | — | — |
| Navigation8 Waypoint JOINT | 0.057 | 0.213 | 15.7 | 0.876 | — | — |

The intended full extension includes each selected terminal run's canonical DG fields and mono-only peak map, policy trajectory/occupancy and flow, stored graph, command diagnostics where saved, and DG → CA3 → decoder-1 full-range kernels on shared histories where checkpoints and histories exist. The local evaluator manifest lists **100,040,704 frames** for all six C05/C15 terminal runs, pending direct checkpoint-file verification. Earlier checkpoints will be labeled only as learning dynamics. The selected high-mono historical families should be treated as separate protocol groups, with their own exact checkpoint audits.
