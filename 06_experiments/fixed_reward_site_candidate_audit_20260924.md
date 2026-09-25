# Fixed-reward site candidate audit, 24 September 2026

## Decision

Do not fix a physical reward coordinate from the four shortlisted source runs yet. A DG target need **not** have one isolated field to be useful: a reward region can coincide with one of several response areas if commands reliably bring the agent into that region. Ranking **individual targets by spatial information plus at least one tested incoming edge** puts DG-capacity unit 50 and full-system PPO unit 51 at the front. Both have a prominent upper-left response, but their compact regions do not yet define one verified shared reward point. The existing graph counters establish DG-event arrival, but do not record the physical location of each successful commanded arrival. A peak coordinate alone is therefore not evidence that commanding a target reaches that site.

This audit addresses a deliberately source-aligned reward site: a physical location near a DG field with strong incoming arrival statistics. It does not assess transfer to an arbitrary reward location.

## Evidence and selection rule

The four runs were shortlisted from W&B summaries before inspecting the spatial arrays here. The two DDQN runs have the best combination of action-probability TV and positive hit-count difference; the PPO and earlier DG-capacity runs have larger hit-count differences but very small TV. Their summary records and the nearest inspected spatial snapshots are **not at identical ages**. Do not treat the edge counts as simultaneous with the W&B hit differences.

| W&B run | Summary age | Commanded / shuffled hits | Action-probability TV | Inspected spatial age |
| --- | ---: | ---: | ---: | ---: |
| [Cadence-64 waypoint F64 DDQN, seed 99](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_CPU2048/runs/00_CPU2048D64_WAYPOINT_DECODER_F64_DDQN_S99_20260915_122147_555688) | 113.7M | 180/1,845 vs 100/1,772 | 0.1205 | 75M |
| [Cadence-2048 waypoint F64 DDQN, seed 99](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_CPU2048/runs/00_CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_20260915_122147_556515) | 273.3M | 161/1,794 vs 112/1,688 | 0.1108 | 300M |
| [DG-capacity waypoint F64, seed 123](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_DGCapacityGoalConditioning/runs/00_DGC_WAYPOINT_DG_F64_S123_20260910_203752_596622) | 135.0M | 376/1,937 vs 240/1,881 | 0.0009 | 75M |
| [Full-system waypoint F64 PPO, seed 123](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_FullSystemController/runs/00_FSCS_WAYPOINT_DECODER_F64_PPO_S123_20260912_194733_015728) | 192.2M | 289/1,905 vs 118/1,833 | 0.0014 | 150M |

For a transparent edge screen, require at least 20 cumulative prospective attempts, at least 80% observed prospective success, and membership in the checkpoint's cached reliable directed graph. These are reporting thresholds, not an independent success guarantee. `field_mono` is reported as context, **not a pass/fail requirement**. The relevant field question is whether the intended reward region captures a useful fraction of actual target activations and, more decisively, of commanded arrivals. Graph counters are cumulative, while field maps describe a retained policy-driven 100k-sample window. Neither is a frozen-policy matched-command trial.

| Source run and snapshot | Reliable edges | Single-field units, context only | Screened reliable edges | Distinct screened target units / peak bins |
| --- | ---: | ---: | ---: | ---: |
| Cadence-64 waypoint F64 DDQN, seed 99, 75M | 12 | 2 | 0 | 0 / 0 |
| Cadence-2048 waypoint F64 DDQN, seed 99, 300M | 45 | 3 | 1 | 1 / 1 |
| DG-capacity waypoint F64, seed 123, 75M | 43 | 7 | 7 | 5 / 2 |
| Full-system waypoint F64 PPO, seed 123, 150M | 41 | 4 | 15 | 13 / 4 |

The cadence-64 run also has no screened edge at 5M or 25M. The cadence-2048 run has one screened reliable edge at 25M and 75M. The DG-capacity run has no screened edge at 5M or 25M. The PPO run has 19 screened reliable edges at 75M. Cached `graph_spatial_endpoint_valid_fraction` is zero at every checkpoint in the table because that metric requires single-field endpoints; this does **not** rule out a usable broader reward region.

## Individual graph and field checks

**Cadence-64 DDQN, seed 99.** At 75M, its most-tested reliable edge 31→37 has 12,304/23,143 prospective successes (53.2%). The reverse 37→31 has 12,160/16,055 (75.7%). Neither reaches the 80% screen. Unit 31's diagnostic peak moves from (1250, 150) at 25M to (1950, 550) at 75M; at 75M its map has multiple components at each stored threshold. The latest positive W&B hit difference at about 113.7M has no matching online spatial snapshot; 75M is the latest saved one found.

**Cadence-2048 DDQN, seed 99.** At 300M, the sole screened reliable pair is 49→19, with 635/753 prospective successes (84.3%) and current graph posterior 82.9%. Target 19 peaks at (750, 750), but only 4.7% of its thresholded activations in the retained behavior window lie within 150 world units of that peak. The largest-count edges are weaker: 15→41 has 146,008/214,771 (68.0%), and 41→15 has 139,375/214,006 (65.1%). Unit 15's peak shifts from (550, 350) at 75M to (750, 1050) at 300M; unit 41 remains in the upper region but shifts between peak bins. The positive W&B hit difference was recorded around 273.3M, between the saved 150M and 300M spatial checkpoints.

**DG-capacity waypoint, seed 123.** At 75M, seven screened reliable edges reach five target units, but four of those targets share the exact diagnostic peak bin (950, 1950). For example, 13→7 has 48,058/51,437 prospective successes (93.4%), and 7→30 has 47,370/58,983 (80.3%). The graph supports testing whether this is one broad physical boundary region, rather than selecting five distinct reward destinations from the five target IDs.

**Full-system PPO, seed 123.** At 150M, 15 screened reliable edges reach 13 target units, but eight targets share peak bin (850, 250). The highest-count edges show 98–99% prospective success, such as 61→53 at 516,742/521,303. At 75M, 19 screened reliable edges exist. These counts make a common-event or common-sink explanation plausible; they do not establish command-specific physical arrival.

## Physical concentration around diagnostic peaks

The saved snapshots also contain 100,000 behavior-time poses and thresholded DG activities. For each target below, we counted the fraction of its active observations within a 150-world-unit radius of its diagnostic peak. This is an **observational field check**, not a commanded-arrival measure. A broad field may still be useful if an explicitly chosen larger reward region captures commanded arrivals; the 150-unit radius is only a common scale for comparing peaks.

| Run / target | Peak at inspected snapshot | Fraction of target activations within radius 150 | Fraction of all observations there | Reading |
| --- | --- | ---: | ---: | --- |
| Cadence-2048 DDQN 49→19 | (750, 750) | 4.7% | 1.2% | Enriched near the peak, but most activations are elsewhere. |
| DG-capacity targets 7, 30, 13, 49 | (950, 1950) | 3.6–3.9% | 2.1% | These targets are active over a broad boundary area; the point peak captures little of their activity. |
| DG-capacity target 50 | (750, 650) | 3.3% | 0.3% | Stronger enrichment, still low absolute concentration. |
| PPO targets 4, 53, 61, 62, 32, 18, 0 | (850, 250) | about 0.1% | 2.0% | The repeated diagnostic peak is especially misleading as a proposed reward point. |

At 150M, approximately 61% of active observations for PPO targets 4, 53, 61, and 32 occur along the west boundary (`x < 200`), versus 43% of all retained observations. For DG-capacity targets 7, 30, 13, and 49, roughly 63–70% of active observations occur along the top boundary (`y > 1900`). This supports inspecting broad physical regions and boundary occupancy, rather than using a single peak bin as a reward coordinate. It still does not reveal where *commanded* successes physically ended.

As a less strict region screen, scan each visited 3-by-3 spatial-bin neighborhood (at most 300 by 300 world units) and record the neighborhood capturing the most thresholded activations for a target. This is an *optimistic in-sample maximum*, not a reward-placement result:

| Run and target group | Best neighborhood center | Target-activation share | Overall occupancy share | Reading |
| --- | --- | ---: | ---: | --- |
| PPO 150M, targets 4, 53, 61, 32 | (150, 450) | 58.2–58.7% | 20.8% | A shared west-boundary region is a plausible first intervention target. |
| DG-capacity 75M, targets 7, 30, 13, 49 | (250, 1850) | 68.2–73.0% | 51.8% | Strong raw concentration, but the behavior policy itself spends over half the window there. |
| DG-capacity 75M, target 50 | (350, 1850) | 64.8% | 30.2% | Better enrichment than the four-target cluster; warrants a separate command probe. |
| Cadence-2048 DDQN 300M, target 19 | (850, 250) | 8.2% | 6.3% | Weak compact-region support despite the 49→19 DG-event edge. |

The PPO neighborhood is clipped by the western boundary and covers fewer than nine full bins; its actual traversable footprint and legal reward placement need verification. Its strong activation enrichment is useful for prioritization, but the nearly identical responses of several target IDs may represent a common sink rather than target-specific navigation. The cadence-64 DDQN run has no edge passing the prospective-success screen at its latest 75M spatial checkpoint, so it has no comparable region candidate from this rule.

## Target-first shortlist: spatial information with incoming edges

The first screen above emphasized high-success edges and exposed common sinks. The intended reward-site question instead starts with a spatially informative DG unit and asks whether **at least one** incoming edge has useful support. The table below uses the saved 100k-sample spatial-information metric, which is amplitude-weighted; compare its rank within a run and inspect the maps rather than treating differences between architectures as normalized information gains. Incoming-edge success is the cumulative prospective DG-event ratio, not physical reward-region arrival.

| Run / checkpoint / target           | Spatial information | Best tested incoming reliable edge                  | Target activity in its best 3-by-3-bin neighborhood | Overall occupancy there | Reading                                                                    |
| ----------------------------------- | ------------------: | --------------------------------------------------- | --------------------------------------------------: | ----------------------: | -------------------------------------------------------------------------- |
| DG-capacity 75M, unit **50**        |               0.159 | **25→50: 18,821/21,070 (89.3%)**, posterior 0.899   |                              64.8% near (350, 1850) |                   30.2% | Best balance of incoming event success and a concentrated response.        |
| PPO 150M, unit **51**               |               0.282 | **55→51: 1,692/2,886 (58.6%)**, posterior 0.618     |                              69.2% near (150, 1850) |                   16.5% | More selective region; weaker incoming success.                            |
| PPO 150M, unit **42**               |               0.336 | **22→42: 123,164/126,144 (97.6%)**, posterior 0.953 |                               58.4% near (150, 550) |                   25.9% | Strong edge, but many PPO targets share the western response.              |
| DG-capacity 75M, unit **9**         |               0.218 | **21→9: 376/603 (62.4%)**, posterior 0.565          |                              70.3% near (350, 1850) |                   30.2% | More informative than unit 50; incoming edge is much weaker.               |
| Cadence-2048 DDQN 300M, unit **56** |               0.139 | **58→56: 82,978/110,757 (74.9%)**, posterior 0.629  |                               14.4% near (350, 450) |                    4.3% | Retains the high-TV source; response is distributed across multiple areas. |
| Cadence-64 DDQN 75M, unit **37**    |               0.131 | **31→37: 12,304/23,143 (53.2%)**, posterior 0.632   |                              15.4% near (1850, 250) |                    3.8% | High within-run information, but weak incoming success.                    |

Selected maps, each normalized **within its own unit** and with unvisited cells gray: [DG-capacity units 50 and 9](assets/fixed_reward_site_candidate_audit_20260924/dgc_s123_75m_selected_fields.png), [PPO units 51 and 42](assets/fixed_reward_site_candidate_audit_20260924/ppo_s123_150m_selected_fields.png), [cadence-2048 DDQN units 56 and 58](assets/fixed_reward_site_candidate_audit_20260924/ddqn_s99_300m_selected_fields.png), and [cadence-64 DDQN units 37 and 31](assets/fixed_reward_site_candidate_audit_20260924/ddqn_d64_s99_75m_selected_fields.png). Visual inspection confirms multiple responses in every displayed unit. PPO 51 is concentrated in the upper-left with weaker secondary patches; DG-capacity 50 has strong upper-left and lower-left patches. DDQN 56/58 have broader distributed fields. No figure shows a physical command intervention.

Units 50 and 51 are late-emerging at the available milestones: unit 50 has no tested reliable incoming edge at 5M or 25M and reaches its quoted information/edge values at 75M; unit 51 has no tested reliable incoming edge through 75M and reaches its quoted values at 150M. Thus the corresponding checkpoint must be frozen for transfer. A later W&B hit advantage does not prove that the same target-region relationship persisted.

The two upper-left 3-by-3-bin neighborhoods overlap only coarsely. When evaluated as a **150-unit-radius disk**, the region centered at (250, 1850) captures 38.1% of DG-capacity unit-50 activations and 69.2% of PPO unit-51 activations, but it has not been checked for traversable reward placement or command-conditioned arrival. The nominal best centers (350, 1850) and (150, 1850) should not be averaged into a final reward coordinate.

**Literal peak placement needs special care.** The cached `field_dominant_peak_xy` for DG-capacity unit 50 is (750, 650), but only 3.3% of its observed activations are within radius 150 there. Its highest *smoothed-rate* bin is instead (450, 1950); a radius-150 disk there contains 51.1% of activations. PPO unit 51 has its highest smoothed-rate bin and cached peak at (350, 1950), with 69.2% of activations inside radius 150. Those two smoothed peaks are 100 world units apart, yet their actual active-pose clusters differ: the median active pose inside a radius-150 disk around (350, 1950) is about (384, 1917) for unit 50 and (220, 1973) for unit 51. The exact reward trigger point and radius must therefore be checked against physical traversability and matched-command endpoint distributions before treating the two runs as sharing one destination. If separate site-aligned tasks are used, each needs its own matched scratch control.

## What would qualify a reward site

Before fixing the reward coordinate, use the [manifest-driven place-field evaluator](../04_implementation/reusable_place_field_telemetry.md) at an immutable source checkpoint near the chosen W&B interval. Inspect pre-threshold and thresholded maps, occupancy, and the physical traversability of candidate **regions**. Then run the established frozen-policy target-control intervention from multiple exclusive source events, comparing commanded and shuffled targets on the **same start states**. Record physical arrival positions, and choose a reward region only if commanding a target reaches it more often than shuffled commands from held-out starts. A target may have other fields; what matters is sufficient, repeatable arrival probability at the chosen region. A small number of successful edges suffices; all-pairs graph coverage is irrelevant to this decision.

The current snapshots cannot satisfy that site-selection gate because they do not tie each successful command to a physical arrival region. Reward placement at a diagnostic peak such as (850, 250) or (950, 1950) would be especially premature given the low fraction of observed target activations near those points. **First test DG-capacity unit 50 and PPO unit 51 near their upper-left response areas**, keeping the exact physical regions separate until geometry and matched-command outcomes justify a shared site. Neither is a selected reward coordinate yet.

## Provenance and reusable lesson

- Source arrays are existing `intrmotiv/online-spatial/v1` NPZ snapshots under `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/online_spatial/`, read in place without changing the historical workspace. The four run directories are `controller_cpu_cadence64_20260914/CPU2048D64_WAYPOINT_DECODER_F64_DDQN_S99`, `controller_cpu_f64_ddqn_20260914/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99`, `intrmotiv_dg_capacity_goal_conditioning_20260910/DGC_WAYPOINT_DG_F64_S123`, and `intrmotiv_full_system_controller_stored_production_20260912/FSCS_WAYPOINT_DECODER_F64_PPO_S123`.
- The NPZ's cached graph adjacency, prospective attempt/success matrices, `field_mono`, field peak coordinates, and cached grounded-edge diagnostic are the authoritative values for this audit. The [CPU2048 25M atlas](data/cpu2048_analysis_20260917/atlas.md) and [analysis](cpu2048_analysis_20260917.md) corroborate the early failure of graph-to-field grounding.
- Reusable shortcut: inspect per-edge counts and target field validity at the same saved checkpoint before rendering a large atlas or choosing a physical reward site. W&B summaries shortlist runs; they cannot identify a reward coordinate. Preserve checkpoint age when joining W&B and spatial evidence.
