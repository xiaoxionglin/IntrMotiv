# A0 poster evidence screen — 26 September 2026

**Current matched analyses:** [Key IntrMotiv poster batch summary](batch_summary.md).
The evidence screen below preserves earlier 25M/50M comparisons and mature
single-run examples. Its older checkpoints are developmental context; use the
linked batch summary for the latest checkpoint shared by every required run.

**Kernel range update:** The layerwise figures preserved below show the original
central 13×13 displacement window. The [full-range Direct](poster_candidates/full_range/CPU2048_DIRECT_F16_DDQN_S99_300007424/layerwise_kernels.svg)
and [full-range Waypoint](poster_candidates/full_range/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_300007424/layerwise_kernels.svg)
figures cover all 37×37 offsets on the same newer common replay panel. Their
[radial profiles](poster_candidates/full_range/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_300007424/layerwise_radial.svg)
extend through diagonal distance 25 bins. See the [updated batch analysis](batch_summary.md#full-range-spatial-kernels)
for three-seed comparisons and spatial-support limits; the older numerical
table in this screen describes its original panel and six-bin reference.

This screen follows the [A0 analysis plan at repository commit `0a19eec`](https://github.com/xiaoxionglin/IntrMotiv/blob/0a19eec/05_plans/A0_poster_analysis_plan_20260926.md). The named plan is absent from this local checkout, so its committed revision was read with `git show`. Figures below are editable SVG with text retained as SVG text. The report separates seed-paired aggregate comparisons, policy-driven mature exemplars, and a common-history layerwise replay.

## Mature 300M exemplar panels

The two selected runs are **single-seed mature exemplars**, both at the same saved 300,007,424-frame checkpoint. Their original online-spatial archives were copied from the historical NEMO2 analysis directory into [Direct F16 raw snapshot](raw_snapshots/direct_f16_ddqn_s99_300m.npz) and [Waypoint F64 raw snapshot](raw_snapshots/waypoint_f64_ddqn_s99_300m.npz); the [source-path and SHA-256 manifest](snapshot_sources.tsv) records the exact originals. They each contain 100,000 policy-driven training-window observations, pose and segment boundaries, DG activity, and prospective graph counts. The [rendering adapter](render_mature_exemplars.py) calls the established `hpc_runs.intrmotiv_study.spatial` place-field, trajectory, segment, and graph recipes; it adds occupancy flow and a DG displacement kernel, and exports SVG. The [numeric metrics](mature_exemplar_metrics.csv) are derived with the canonical spatial metric function.

| Panel | Direct F16 DDQN S99, 300M | Waypoint F64 DDQN S99, 300M |
| --- | --- | --- |
| DG place fields | [16-unit atlas](poster_candidates/direct_f16_ddqn_s99_300m/place_fields_page01.svg) | [Pages 1](poster_candidates/waypoint_f64_ddqn_s99_300m/place_fields_page01.svg), [2](poster_candidates/waypoint_f64_ddqn_s99_300m/place_fields_page02.svg), [3](poster_candidates/waypoint_f64_ddqn_s99_300m/place_fields_page03.svg), [4](poster_candidates/waypoint_f64_ddqn_s99_300m/place_fields_page04.svg) |
| Occupancy and trajectory | [Overview](poster_candidates/direct_f16_ddqn_s99_300m/trajectory.svg), [four segments](poster_candidates/direct_f16_ddqn_s99_300m/segments.svg) | [Overview](poster_candidates/waypoint_f64_ddqn_s99_300m/trajectory.svg), [four segments](poster_candidates/waypoint_f64_ddqn_s99_300m/segments.svg) |
| Occupancy flow | [Direct flow](poster_candidates/direct_f16_ddqn_s99_300m/flow.svg) | [Waypoint flow](poster_candidates/waypoint_f64_ddqn_s99_300m/flow.svg) |
| Directed graph outcomes and reliable mask | [Direct outcomes](poster_candidates/direct_f16_ddqn_s99_300m/graph_outcomes.svg), [edge mask](poster_candidates/direct_f16_ddqn_s99_300m/graph_reliable_mask.svg) | [Waypoint outcomes](poster_candidates/waypoint_f64_ddqn_s99_300m/graph_outcomes.svg), [edge mask](poster_candidates/waypoint_f64_ddqn_s99_300m/graph_reliable_mask.svg) |
| DG displacement kernel, online visits | [Direct kernel](poster_candidates/direct_f16_ddqn_s99_300m/dg_kernel.svg) | [Waypoint kernel](poster_candidates/waypoint_f64_ddqn_s99_300m/dg_kernel.svg) |
| DG→CA3→decoder-1, common history | [Direct kernels](poster_candidates/direct_f16_ddqn_s99_300m/layerwise_kernels.svg), [radial profile](poster_candidates/direct_f16_ddqn_s99_300m/layerwise_radial.svg) | [Waypoint kernels](poster_candidates/waypoint_f64_ddqn_s99_300m/layerwise_kernels.svg), [radial profile](poster_candidates/waypoint_f64_ddqn_s99_300m/layerwise_radial.svg) |

The graph matrices show prospective hits divided by attempts for every attempted source–target pair; gray means unattempted. They are not command interventions. Flow uses only within-segment transitions shorter than the snapshot's jump threshold, with arrows shown for bins having at least 20 transitions; direction and length encode the mean direction and coherence. The DG-only kernels use occupied bins with at least five observations and correlate population vectors at each displacement. Those panels are **policy-driven online snapshots**. The layerwise kernels instead replay the **same 10,001-observation complete history** for each checkpoint, with episode boundaries preserved; source and measured-layer metadata are in the [Direct replay JSON](kernels/direct_f16_ddqn_s99_300m.json) and [Waypoint replay JSON](kernels/waypoint_f64_ddqn_s99_300m.json), and the [raw kernel arrays](kernels/) support replotting. Frozen-policy panels are reported separately below.

| Online snapshot metric | Direct F16 S99 | Waypoint F64 S99 |
| --- | ---: | ---: |
| Active DG units | 16/16 | 64/64 |
| Active-only map cosine | 0.516 | 0.303 |
| Mono-field fraction among eligible units | 0.0% | 4.7% |
| Distinct peak bins | 15 | 35 |
| Visited 19×19 cells | 88.1% | 88.1% |
| Mean local flow coherence, bins with ≥20 transitions | 0.143 | 0.224 |
| Reliable directed edges | 66 | 45 |
| Reliable edge density | 27.5% | 1.1% |
| Largest strongly connected component | 14/16 nodes | 4/64 nodes |
| Reachable ordered node pairs | 88.3% | 2.6% |
| Prospective transition success | 41.4% | 52.1% |
| DG kernel near minus far correlation | 0.226 | 0.146 |

The mature Waypoint example has more distributed DG peaks and lower map overlap, while the Direct example has far greater graph reachability. Waypoint's higher prospective success is conditional on its attempted transitions and does not contradict its sparse connected graph. Both trajectories cover the same fraction of coarse cells in these online windows; that alone does not establish equal exploration quality or frozen-policy coverage. A spatially grounded graph drawing would have zero qualified mono-field nodes for Direct and only three for Waypoint in these online snapshots. Forcing all other graph nodes onto a single peak coordinate would misrepresent ambiguous/multi-field units, so the complete all-node matrices are the honest graph view.

### Frozen-policy fields, trajectories, and flow

The [two-row frozen evaluation manifest](mature_frozen_manifest.tsv) was reviewed print-only before two ordinary NEMO2 jobs were submitted. Both finished using the same 300M checkpoints, 10,001 observations, and six episodes. Their first reset pose was identical. The canonical `place_fields.py` evaluator wrote the original [Direct frozen field archive](frozen_raw/CPU2048_DIRECT_F16_DDQN_S99__00_CPU2048_DIRECT_F16_DDQN_S99__CPU2048_DIRECT_F16_DDQN_S99_300007424/place_fields.npz), [Direct pose rows](frozen_raw/CPU2048_DIRECT_F16_DDQN_S99__00_CPU2048_DIRECT_F16_DDQN_S99__CPU2048_DIRECT_F16_DDQN_S99_300007424/pose.csv), [Waypoint frozen field archive](frozen_raw/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99__00_CPU2048_WAYPOINT_DECODER_F64_DDQN_S99__CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_300007424/place_fields.npz), and [Waypoint pose rows](frozen_raw/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99__00_CPU2048_WAYPOINT_DECODER_F64_DDQN_S99__CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_300007424/pose.csv). These are local copies of the job outputs under `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/A0_poster_20260926/frozen/raw/`.

| Frozen panel | Direct F16 DDQN S99 | Waypoint F64 DDQN S99 |
| --- | --- | --- |
| DG place fields | [16-unit atlas](poster_candidates/direct_f16_ddqn_s99_300m/frozen_fields_page01.svg) | [Pages 1](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_fields_page01.svg), [2](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_fields_page02.svg), [3](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_fields_page03.svg), [4](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_fields_page04.svg) |
| Occupancy and complete episode trajectories | [Direct](poster_candidates/direct_f16_ddqn_s99_300m/frozen_trajectory.svg) | [Waypoint](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_trajectory.svg) |
| Occupancy flow | [Direct](poster_candidates/direct_f16_ddqn_s99_300m/frozen_flow.svg) | [Waypoint](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_flow.svg) |

| Frozen metric | Direct | Waypoint |
| --- | ---: | ---: |
| Coarse-cell coverage | 66.5% | 70.4% |
| Active DG units | 16/16 | 36/64 |
| Active-only map cosine | 0.209 | 0.104 |
| Mono-field fraction among eligible units | 0.0% | 3.0% |
| Mean active-unit spatial information | 0.096 bits | 0.104 bits |
| Mean local flow coherence, bins with ≥5 transitions | 0.395 | 0.467 |

The frozen Waypoint policy covers slightly more of the coarse arena and shows a coherent peripheral route in this rollout, but 28 of its 64 DG units are silent. Its lower map cosine is calculated **among active units only**; the historical summarizer includes silent vectors in its raw cosine mean, which would understate overlap. These are single-seed descriptive differences, not an architecture-level coverage effect. The [frozen metrics CSV](frozen_exemplar_metrics.csv) records exact denominators. For the poster, use one readable subset of fields plus the two trajectory/flow panels; keep the complete atlases in supplementary material.

The [compact eight-field SVG](poster_candidates/frozen_top_information_fields.svg) implements that subset: the four active units with highest spatial information from each fixed checkpoint. Its selection rule is explicit and should appear in any caption; the full atlases above show the population, including silent Waypoint units.

### Where spatial contrast changes across layers

The common-history kernel uses population-vector correlation at spatial offsets on the same replayed sensory/action sequence. Each replay produced aligned DG, CA3, and first decoder hidden-layer arrays, with 16/1,136/128 features in Direct and 64/4,544/128 in Waypoint. The decoder hook resolved to `decoder.state_layer.1` in both runs. The common panel had 293 spatial bins with at least five observations; the one-bin horizontal offsets had 226–232 eligible pairs and the six-bin offsets 163–168. All 169 kernel offsets were finite for each layer. The table shows mean correlation at one-bin cardinal offsets, at six-bin cardinal offsets, and their difference:

| Exemplar and layer | Near | Far | Near − far |
| --- | ---: | ---: | ---: |
| Direct DG | 0.248 | 0.146 | 0.102 |
| Direct CA3 | 0.509 | 0.263 | 0.246 |
| Direct decoder-1 | 0.986 | 0.979 | 0.007 |
| Waypoint DG | 0.376 | 0.248 | 0.128 |
| Waypoint CA3 | 0.606 | 0.368 | 0.238 |
| Waypoint decoder-1 | 0.979 | 0.960 | 0.019 |

**Poster candidate:** a compact three-layer kernel panel with the radial profile. In these two exemplars, CA3 retains a stronger near-versus-far contrast than DG, while decoder-1 population vectors are highly similar across locations for the replayed input context. This is a kernel-shape observation, not proof that location is undecodable from decoder-1 or that every goal context behaves this way. The current replay does not balance target goals or cues, so a goal-conditioned decoder conclusion needs that control.

## Poster choice

| Candidate | Use on poster? | Supported statement | Boundary |
| --- | --- | --- | --- |
| [CPU2048 representation versus graph SVG](poster_candidates/cpu2048_direct_waypoint_25m.svg) | **Yes, central result** | At the six-run matched 25M checkpoint, Waypoint F64 has lower map overlap and more mono-field units, while Direct F16 has much higher graph reachability. | Bundled architecture comparison; graph reachability is passive and does not establish command control. |
| [Frozen-DG transfer SVG](poster_candidates/frozen_dg_transfer_50m.svg) | **Yes, with qualification** | At 50M, source DG has lower map overlap in both D50 and D51; saved 40–50M reward means show no consistent source advantage. | D50/D51 are separate source families. The saved scalar table stops at 50M even though D51 has 75M spatial snapshots. This is not the requested 75M behavioral endpoint. |
| Saturday ARR versus SRC | Hold | Existing [poster synthesis](../../../05_plans/poster_results_synthesis_20260922.md) discusses the mechanism. | No local canonical seed-paired Saturday scalar/spatial table was found for a refreshed latest-common contrast. |
| DGP HIT versus FIRST | Hold for a diagnostic inset | [Historical control report](../../06_high_option_success_goal_sets_and_controls_20260914.md) records high HIT completion and near-unit target/shuffle activation ratio. | Separate trained policies and scalar windows do not provide matched-start command interventions. |
| Mature 300M exemplars | **Yes, as descriptive examples** | Frozen fields, complete trajectories, flow, online graph structure, and common-history kernels are now available for two saved runs. | Single seed per architecture; graph counters are passive; the decoder replay uses one input context. |

**Suggested main-poster set:** lead with the [eight-field mature comparison](poster_candidates/frozen_top_information_fields.svg), show the [Direct](poster_candidates/direct_f16_ddqn_s99_300m/frozen_trajectory.svg) and [Waypoint](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_trajectory.svg) frozen trajectories with one [Waypoint flow panel](poster_candidates/waypoint_f64_ddqn_s99_300m/frozen_flow.svg), then use the [full-range common-history layerwise kernels](poster_candidates/full_range/CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_300007424/layerwise_kernels.svg) to motivate the DG→CA3→decoder question. Put the [matched three-seed CPU2048 plot](poster_candidates/cpu2048_direct_waypoint_25m.svg) beside those exemplars so the single-seed pictures do not carry the architecture claim alone. The [all-node graph matrices](poster_candidates/direct_f16_ddqn_s99_300m/graph_outcomes.svg) and complete atlases are suitable for a QR-linked supplement; a small graph-reachability number can remain on the main poster. Use the transfer result only if space allows, with its 50M scalar-window caveat.

## Paired results available now

### CPU2048 DDQN+HER: representation and graph separate

The [original all-snapshot table](../../data/cpu2048_analysis_20260917/all_snapshots.csv) contains all three seeds per arm at 25M. The existing [CPU2048 report](../../cpu2048_analysis_20260917.md) documents the evaluator. Means over seeds 8, 99, and 123:

| Arm | Active-only map cosine ↓ | Mono-field fraction ↑ | Reliable edges | Reachable ordered pairs ↑ |
| --- | ---: | ---: | ---: | ---: |
| Direct F16 | 0.364 | 0.0% | 61.0 | 83.6% |
| Waypoint F64 | 0.227 | 9.5% | 16.3 | 0.6% |

The SVG connects matched seeds. Node capacity differs, so reliable edge count is descriptive; reachable-pair fraction helps normalize the graph comparison. Neither measure demonstrates goal-specific behavior. The [existing atlas](../../data/cpu2048_analysis_20260917/atlas.md) supplies the original place-field and graph panels. The local all-snapshot table does not contain a complete 75M HER comparison. Later checkpoint paths in the [prepared inventory](../../data/poster_missing_analyses_20260926/checkpoint_inventory.tsv) still require remote spatial evaluation before they can supersede 25M.

### Frozen source versus random DG: better maps, no demonstrated reward benefit

The [original online table](../../data/cued_reward5_frozen_dg_interim_20260926/online/per_run.csv) contains all six source/random seed pairs for the 40–50M reward window. The [original spatial table](../../data/cued_reward5_frozen_dg_interim_20260926/spatial/per_snapshot.csv) contains all pairs at 50M. The [full interim report](../../cued_reward5_frozen_dg_interim_analysis_20260926.md) gives the study design and qualification limits.

| Site family | Source minus random reward mean, 40–50M | Source map cosine, 50M | Random map cosine, 50M | Source graph reachability, 50M | Random graph reachability, 50M |
| --- | ---: | ---: | ---: | ---: | ---: |
| D50 | −0.000086 | 0.097 | 0.212 | 5.7% | 8.3% |
| D51 | +0.000005 | 0.100 | 0.201 | 3.1% | 13.6% |

The reward differences are means of three paired seed differences, sourced from the [canonical contrast table](../../data/cued_reward5_frozen_dg_interim_20260926/online/paired_contrast_summary.csv). D50 runs reached about 55–61M in that table; D51 runs reached 75M, but the exported reward window still ends at 50M. D51 has complete 75M spatial rows; D50 does not. A 75M transfer claim and learning-curve AUC require a fresh scalar-history export. The map measurements are policy-driven online snapshots, not common-history replay, so they may also reflect visitation differences.

## Inventory and remaining work

The prepared [analysis manifest](../../data/poster_missing_analyses_20260926/analysis_manifest.tsv) and [checkpoint inventory](../../data/poster_missing_analyses_20260926/checkpoint_inventory.tsv) list candidate remote checkpoints, including 300M CPU2048 and 75M transfer controls. The two 300M exemplars were verified directly on NEMO2; the broader candidate inventory was not refreshed cluster-wide. The [25 September screening](../../poster_candidate_screening_20260925.md) documents other evaluated matched panels and missing interventions.

To finish the remaining plan, use the existing `hpc_runs/intrmotiv_study` collector and `sf_working_directories/IntrMotiv/evaluation` manifest evaluator on NEMO2, after the standard print-only review. First export complete scalar histories for ARR/SRC, DGP HIT/FIRST, and D50/D51 through 75M where reached. Then evaluate the latest checkpoint common to each condition × seed set. A matched-start command intervention and goal/cue-balanced decoder kernel are still needed before claiming causal node control or general decoder loss of spatial information. Do not present passive graph counters as matched-command evidence.

## Reproducibility and workflow lesson

The paired SVGs are regenerated by [the summary renderer](render_transfer_summary.py) from canonical CSVs. [Mature online panels](render_mature_exemplars.py), [frozen panels](render_frozen_exemplars.py), and [layerwise kernels](render_layer_kernels.py) have separate small SVG renderers from the linked NPZ/CSV evidence. The [layerwise collector](../../collect_poster_population_kernels.py) uses the established evaluator checkpoint loader and complete-history replay; the frozen rollout uses the canonical manifest worker. All four Slurm jobs completed with exit code zero: frozen jobs 8218878/8218879 and layerwise jobs 8218863/8218864. What worked was using existing 300M online snapshots for immediate inspectable panels, then running only two frozen and two layerwise jobs. What was slow was checkpoint loading and 10k-step CPU replay. Next time, start with a refreshed canonical inventory and exported histories, then render directly from their manifests. The existing [infrastructure tracker](../../../infra.md) records the reusable active-workspace input/path problem; a compact availability table that distinguishes checkpoint existence, online NPZ, scalar histories, common panels, and frozen evaluation would further reduce repeated discovery.
