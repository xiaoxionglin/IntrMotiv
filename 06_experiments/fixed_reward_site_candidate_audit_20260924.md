# Fixed-reward site candidate audit, 24 September 2026

## Decision

Do not fix a physical reward coordinate from the four shortlisted source runs yet. A DG target need **not** have one isolated field to be useful: a reward region can coincide with one of several response areas if commands reliably bring the agent into that region. A relaxed region screen identifies a promising **west-boundary area in the PPO run** and a weaker **upper-left boundary area in the DG-capacity run**. The existing graph counters establish DG-event arrival, but do not record the physical location of each successful commanded arrival. The strongest hit-count arms also route many DG targets to a small number of diagnostic peak bins. A peak coordinate alone is therefore not evidence that commanding a target reaches that site.

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

## What would qualify a reward site

Before fixing the reward coordinate, use the [manifest-driven place-field evaluator](../04_implementation/reusable_place_field_telemetry.md) at an immutable source checkpoint near the chosen W&B interval. Inspect pre-threshold and thresholded maps, occupancy, and the physical traversability of candidate **regions**. Then run the established frozen-policy target-control intervention from multiple exclusive source events, comparing commanded and shuffled targets on the **same start states**. Record physical arrival positions, and choose a reward region only if commanding a target reaches it more often than shuffled commands from held-out starts. A target may have other fields; what matters is sufficient, repeatable arrival probability at the chosen region. A small number of successful edges suffices; all-pairs graph coverage is irrelevant to this decision.

The current snapshots cannot satisfy that site-selection gate because they do not tie each successful command to a physical arrival region. Reward placement at a diagnostic peak such as (850, 250) or (950, 1950) would be especially premature given the low fraction of observed target activations near those points. The **PPO west-boundary neighborhood around (150, 450)** is the first provisional region to test, followed by DG-capacity target 50 around (350, 1850). Neither is a selected reward coordinate yet.

## Provenance and reusable lesson

- Source arrays are existing `intrmotiv/online-spatial/v1` NPZ snapshots under `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/online_spatial/`, read in place without changing the historical workspace. The four run directories are `controller_cpu_cadence64_20260914/CPU2048D64_WAYPOINT_DECODER_F64_DDQN_S99`, `controller_cpu_f64_ddqn_20260914/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99`, `intrmotiv_dg_capacity_goal_conditioning_20260910/DGC_WAYPOINT_DG_F64_S123`, and `intrmotiv_full_system_controller_stored_production_20260912/FSCS_WAYPOINT_DECODER_F64_PPO_S123`.
- The NPZ's cached graph adjacency, prospective attempt/success matrices, `field_mono`, field peak coordinates, and cached grounded-edge diagnostic are the authoritative values for this audit. The [CPU2048 25M atlas](data/cpu2048_analysis_20260917/atlas.md) and [analysis](cpu2048_analysis_20260917.md) corroborate the early failure of graph-to-field grounding.
- Reusable shortcut: inspect per-edge counts and target field validity at the same saved checkpoint before rendering a large atlas or choosing a physical reward site. W&B summaries shortlist runs; they cannot identify a reward coordinate. Preserve checkpoint age when joining W&B and spatial evidence.
