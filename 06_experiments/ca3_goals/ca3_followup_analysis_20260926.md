# CA3 state-goal follow-up: matched results and checkpoint record

**Status (26 September 2026):** The G500 75M and CPU 75M panels contain all
four conditions and three seeds. Later endpoint inventories have unequal ages;
only the fixed-anchor arms have a complete three-seed 300M comparison. The
September 24 CPU 25M interim report is incorporated below as an early spatial
record. The two hardware releases are analyzed separately.

## Study design and reading order

| Factor | Compared levels | Held fixed |
| --- | --- | --- |
| Anchor maintenance | FIXED first confirmed anchor; EMA representative refinement | Predictive CA3 readout and continuous state goal |
| Candidate admission | DOM strongest DG event; UNIQUE exactly one contextual match | F64 DG, H32 prediction horizon, contextual hit rule |
| Controller | No arm difference | Stored-state DDQN+HER and waypoint/context graph |
| Declared contrasts | EMA, UNIQUE, interaction, and UNIQUE versus DOM within each anchor mode | Paired seeds at a common frame and within one hardware release |

The worker receives a 16-dimensional predictive readout of raw CA3 history.
Its predictor uses actions and a 32-decision horizon; goals are continuous
readout states. Prediction and variance/covariance losses train the readout.
Each active graph slot stores a confirmed CA3 anchor. FIXED retains its first
anchor. EMA can replace it with a more representative occurrence of the same
recognized identity while retaining the slot and graph edges. DOM admits the
strongest raw DG event; UNIQUE admits an event only when exactly one selectable
contextual anchor matches. The four cells are
`CTX_FIXED_DOM_H32`, `CTX_FIXED_UNIQUE_H32`, `CTX_EMA_DOM_H32`, and
`CTX_EMA_UNIQUE_H32`, each with seeds 8, 99, and 123. The
[study definition](../../hpc_runs/studies/ca3_state_goal_followup_20260922_production.study.json)
and [scientific plan](../../05_plans/ca3_state_goal_followup_20260922.md)
specify the declared contrasts.

Start with [matched-frame results](#matched-frame-main-result) for the balanced
comparison, then [mechanism diagnostics](#mechanism-diagnostics). The
[latest-checkpoint inventory](#latest-available-checkpoints-and-spatial-observations)
describes run status rather than treatment effects. The
[early CPU record](#early-cpu-spatial-record-25m-and-restricted-75m) preserves
the matched 25M map analysis and the seed-99 atlas.

Analysis date: 26 September 2026. Latest available checkpoint and spatial-snapshot results come first, with their respective training ages shown separately. CPU and G500 releases are analyzed separately. The subsequent matched-frame analysis compares all four conditions and seeds 8, 99, and 123 at 75M. Its G500 scalar comparison averages the declared 70M–75M window; cumulative counters are sampled at its endpoint. These are descriptive analyses, with no asymptotic significance claims.

## Latest available checkpoints and spatial observations

The latest endpoints describe the state reached by each run. They are unequal
in training age and must not be used as a factorial treatment-effect estimate.
Checkpoint age is the highest-frame retained archive read for this analysis;
spatial age is the latest complete online snapshot found by the canonical
collector. Edges, prospective success, and grounded control below belong to that
spatial snapshot, not to a new rollout of the retained checkpoint. CPU jobs may
continue beyond this captured inventory.

### G500: all twelve runs

| Anchor / recognition | Seed | Saved model (M frames) | Spatial snapshot (M frames) | Reliable edges | Prospective success | Grounded control |
|---|---:|---:|---:|---:|---:|---:|
| Fixed / dominant | 8 | 300.057 | 300.007 | 27 | 29.30% | 0 |
| Fixed / dominant | 99 | 300.040 | 300.007 | 8 | 14.39% | 0 |
| Fixed / dominant | 123 | 300.040 | 300.007 | 7 | 15.09% | 0 |
| Fixed / unique | 8 | 300.057 | 300.007 | 0 | 13.31% | 0 |
| Fixed / unique | 99 | 300.057 | 300.007 | 0 | 15.09% | 0 |
| Fixed / unique | 123 | 300.057 | 300.007 | 0 | 30.05% | 0 |
| EMA / dominant | 8 | 300.057 | 300.007 | 25 | 43.94% | 0 |
| EMA / dominant | 99 | 300.040 | 300.007 | 35 | 37.56% | 0 |
| EMA / dominant | 123 | 130.105 | 75.006 | 30 | 39.59% | 0 |
| EMA / unique | 8 | 126.484 | 150.012 | 0 | 12.55% | 0 |
| EMA / unique | 99 | 121.242 | 150.012 | 1 | 12.95% | 0 |
| EMA / unique | 123 | 120.177 | 150.012 | 2 | 13.01% | 0 |

The completed fixed/UNIQUE runs all end with zero reliable edges. The two
completed EMA/DOM runs have 25 and 35 edges and higher prospective success than
the three completed fixed/DOM runs, but this incomplete endpoint comparison
cannot isolate EMA's effect. Every latest G500 spatial snapshot has zero
grounded controllability. Thus a larger internal graph and higher prospective
success have not established a graph with spatially grounded control under the
current diagnostic.

The three interrupted EMA/UNIQUE runs have later telemetry than retained valid
model saves. Their 150M snapshots document observed behavior but do not provide
150M restart states. Conversely, EMA/DOM seed 123 has a 130M model but only a
75M spatial snapshot. This limits endpoint interpretation and revival targets;
neither discrepancy should be silently filled by interpolation.

Direct reads of all twelve latest retained G500 model checkpoints confirm that
every one stores 64 active goals and a ready calibrator. The saved collision
diagnostics remain high at these later ages:

| Anchor / recognition | Seeds, in order | Saved anchor collision fractions | Accepted refinements |
|---|---|---|---|
| Fixed / dominant | 8, 99, 123 | 100%, 99.95%, 100% | 0, 0, 0 |
| Fixed / unique | 8, 99, 123 | 100%, 100%, 100% | 0, 0, 0 |
| EMA / dominant | 8, 99, 123 | 51.84%, 37.15%, 33.78% | 996, 704, 409 |
| EMA / unique | 8, 99, 123 | 61.56%, 44.30%, 45.14% | 141, 252, 238 |

These are checkpointed values from the most recent calibration, not a fresh
recalibration under the exact saved weights. They establish that the high
collision signal persists beyond the 75M comparison. In particular, completing
300M training did not separate the fixed anchors' predictive signatures. EMA
has performed real refinements, but its anchors still collide frequently.
[Exact checkpoint paths, model counters, and frame counts](../data/ca3_followup_20260926/g500_latest_checkpoint_diagnostics.json)
are saved separately from the online snapshot tables. Checkpoints were loaded
read-only on CPU with memory mapping; no training or policy rollout was run.

### CPU: all twelve runs

| Anchor / recognition | Seed | Saved model (M frames) | Spatial snapshot (M frames) | Reliable edges | Prospective success | Grounded control |
|---|---:|---:|---:|---:|---:|---:|
| Fixed / dominant | 8 | 166.527 | 150.012 | 24 | 22.80% | 0 |
| Fixed / dominant | 99 | 150.012 | 150.012 | 1 | 14.70% | 0 |
| Fixed / dominant | 123 | 150.012 | 150.012 | 19 | 20.68% | 0 |
| Fixed / unique | 8 | 149.848 | 75.006 | 7 | 44.19% | 0 |
| Fixed / unique | 99 | 150.012 | 150.012 | 0 | 3.06% | 0 |
| Fixed / unique | 123 | 75.006 | 75.006 | 2 | 17.12% | 0 |
| EMA / dominant | 8 | 150.012 | 150.012 | 31 | 42.92% | 0.0138 |
| EMA / dominant | 99 | 150.012 | 150.012 | 40 | 36.18% | 0 |
| EMA / dominant | 123 | 150.012 | 150.012 | 16 | 37.57% | 0 |
| EMA / unique | 8 | 75.006 | 75.006 | 2 | 10.18% | 0 |
| EMA / unique | 99 | 150.012 | 150.012 | 1 | 9.40% | 0 |
| EMA / unique | 123 | 150.012 | 150.012 | 1 | 9.09% | 0 |

The CPU endpoints show the same broad graph-sparsity pattern. One run,
EMA/DOM seed 8 at 150M, has nonzero grounded controllability (0.0138); the other
eleven latest snapshots are zero. This isolated positive result deserves
targeted inspection rather than either being discarded or treated as replicated
control. Its mono-field fraction is 14.3%, compared with 3.1% and 15.6% in the
other EMA/DOM seeds, so mono-field fraction alone does not explain the result.

All twelve CPU model checkpoints were also read directly. Every checkpoint
stores 64 active goals and a ready calibrator. Saved anchor collisions are
98.71–100% for fixed/DOM and 100% for fixed/UNIQUE. EMA/DOM stores 42.61%,
42.71%, and 36.71% for seeds 8, 99, and 123, with 230, 664, and 486 accepted
refinements. EMA/UNIQUE stores 53.08%, 37.80%, and 46.43%, with 99, 207, and
231 refinements. These are saved calibration diagnostics, as for G500. The
nonzero-control CPU seed therefore still has substantial signature overlap.
[CPU checkpoint diagnostics](../data/ca3_followup_20260926/cpu_latest_checkpoint_diagnostics.json)
record exact loaded paths and model ages. Active training rotated one file
during extraction; its latest complete replacement was selected, without
reloading already completed rows. The captured model ages need not equal the
newest live training state after this report was written.

Across both releases, the latest saved observations support persistent UNIQUE
graph sparsity and mostly absent grounded control. They provide a current
inventory and identify the exceptional CPU seed. The matched-frame analyses
below establish which patterns survive equal training-age comparisons.

## Matched-frame main result

EMA improves G500 coverage AUC in each paired seed, but does not establish improved grounded control. UNIQUE recognition strongly suppresses accepted events and graph formation in both hardware releases. Predictive signature collisions provide a concrete explanation: fixed-anchor runs report a collision fraction of 1.0, and EMA runs about 0.49–0.50. Maintaining 64 active addresses has not produced 64 distinguishable predictive goal identities.

## G500 results at matched training age

Values are three-seed means. Coverage AUC is in its logged units; option success is a fraction shown as a percentage. Reliable edges are from the 75M snapshot, while grounded control is the mean scalar in the 70M–75M window.

| Anchor / recognition | Coverage AUC | Option success | Active goals | Reliable edges | Grounded control, scalar |
|---|---:|---:|---:|---:|---:|
| Fixed / dominant | 57.53 | 6.59% | 64 | 20.33 | 0.003586 |
| Fixed / unique | 57.55 | 6.60% | 64 | 0.33 | 0 |
| EMA / dominant | 64.41 | 14.65% | 64 | 30.33 | 0.001287 |
| EMA / unique | 64.25 | 3.05% | 64 | 2.00 | 0 |

All four cells have zero grounded controllability in their exact 75M spatial snapshots. This is compatible with the small nonzero scalar-window averages: these summarize different windows and should not be substituted for each other. Neither metric constitutes a matched-command causal intervention.

The paired EMA coverage effects, averaged over candidate rules, are +15.47, +4.11, and +0.78 for seeds 8, 99, and 123 (mean +6.79). Its grounded-control effects are negative or zero. UNIQUE coverage effects are +0.86, −4.16, and +3.08 (mean −0.07), giving no consistent coverage benefit. UNIQUE reduces option success in every seed when averaged over anchor modes. Within EMA, the reductions are 11.45, 10.94, and 12.43 percentage points. The coverage interaction changes sign across seeds, so the current evidence does not establish a stable interaction.

All five declared contrasts and all declared metrics are retained in [paired contrasts](../data/ca3_followup_20260926/paired_contrasts.csv); [per-run values](../data/ca3_followup_20260926/per_run.csv) allow direct seed inspection.

## Mechanism diagnostics

| Anchor / recognition | Accepted events | Zero matches | Multiple matches | Anchor collision fraction | Background above threshold | Anchor refinements |
|---|---:|---:|---:|---:|---:|---:|
| Fixed / dominant | 66,833 | 171,518 | 0 | 1.000 | 0.315 | 0 |
| Fixed / unique | 6,790 | 137,080 | 89,155 | 1.000 | 0.498 | 0 |
| EMA / dominant | 130,130 | 112,674 | 0 | 0.486 | 0.692 | 154 |
| EMA / unique | 12,897 | 32,158 | 196,953 | 0.503 | 0.704 | 133 |

The source computes collision fraction over distinct pairs of selectable anchors: normalized action-probe signature cosine exceeding the current recognition threshold. It is a representation diagnostic, not a privileged spatial false-confirmation rate. Background pairs are also diagnostic-only; their high acceptance is evidence of limited separation, not a measured physical error rate. The fixed-arm collision value and the EMA multiple-match counts merit attention before introducing more permissive activation rules.

Every run is calibration-ready, with all 64 goals active. Latent standard deviation means range from 1.09 to 1.27 and variance penalties are small, so simple latent variance collapse is not supported. This does not rule out loss of discrimination in the predictor's normalized signatures. State-shuffle loss deltas are positive (cell means 0.00231–0.00381); action-shuffle deltas are also positive but small (0.00079–0.00216), compared with prediction losses around 0.53–0.82. The predictor uses these inputs to a measurable but limited extent under this diagnostic.

Contextual HER positive rates are 0.96–1.48%. EMA records millions of refinement comparisons but only roughly 133–154 accepted refinements per run. Replay mismatch and stale-generation rejection counts are zero in the matched window. UNIQUE does rescue some events, but its accepted-event counts remain roughly tenfold smaller than DOM. Counters are reported as logged; these are not unique physical visits.

## CPU replication at 75M

| Anchor / recognition | Reliable edges | Prospective attempts | Prospective success | Mono-field fraction | Grounded control |
|---|---:|---:|---:|---:|---:|
| Fixed / dominant | 19.33 | 17,522 | 21.47% | 5.03% | 0 |
| Fixed / unique | 3.00 | 568 | 21.55% | 4.69% | 0 |
| EMA / dominant | 24.67 | 75,211 | 34.87% | 7.29% | 0 |
| EMA / unique | 1.67 | 1,085 | 10.45% | 14.58% | 0 |

The graph suppression replicates across CPU and GPU. Map-quality effects do not replicate uniformly: for example, EMA/UNIQUE has a higher mono-field fraction than EMA/DOM on CPU, but a lower one on G500. Do not pool these releases or claim that cleaner DG fields explain the recognition effect. These are policy-conditioned online windows, not frozen-policy spatial alias evaluations.

## Completion and storage caveat

User-directed status update: the four unfinished G500 runs were stopped on
September 26. Their verified process groups contained 144 processes; all exited
on SIGTERM, with no survivors or SIGKILL escalation. These are EMA/DOM seed 123
and EMA/UNIQUE seeds 8, 99, and 123. Revival is deferred in the root `infra.md`
todo entry. CPU jobs were not stopped by this instruction.

The G500 queue records eight completed runs at 300M. Its remaining four statuses are stale: the queue state was last written on September 25 at 15:00, scratch was full, and subsequent trainer logs show logging failures and blocked queues. Later saved snapshots exist, so the stale 121M–131M frame counters are not final progress measurements. Eight 300M and eleven 150M snapshots were found, but a complete terminal factorial is unavailable. CPU has twelve snapshots at 75M and nine at 150M, with no 300M snapshots found by this collection. No claim of completed production is made.

The requested cleanup retained eight checkpoint files per run in each production root, including the earliest available checkpoint and latest archive-complete rolling restart. Frame-grid selection minimized squared distance to eight uniformly spaced targets, subject to those retention requirements. Existing gaps prevent exact spacing; several G500 runs had already rotated away most early checkpoints. CPU selection also excludes redundant copies at the same frame. Archive checks verify ZIP end structure, not full tensor reload or replay equality.

- G500: 246 to 96 files; 150 deleted; 579.53 GiB of file data removed. Filesystem free space subsequently measured 575 GiB.
- CPU: 209 to 96 files; 113 deleted; 393.26 GiB of file data removed.
- [G500 deletion/retention manifest](../data/ca3_followup_20260926/g500_checkpoint_pruning.json) and [CPU manifest](../data/ca3_followup_20260926/cpu_checkpoint_pruning.json) preserve exact paths, frames, sizes, and modification times. Deletions are permanent; retained files and spatial snapshots were not rewritten.

This is a one-time retention cleanup. It does not change future checkpoint cadence. The four unfinished G500 jobs were subsequently stopped at the user's request. Future CPU saves may increase their count again. Any restart should first validate the retained full checkpoint through the existing reload workflow.

## Later saved results: persistent graph sparsity

At 150M, seeds 8 and 99 form a complete matched factorial. EMA/DOM seed 123 is
missing, so the additional third-seed observations are excluded from this table.

| Anchor / recognition | Seeds | Reliable edges | Prospective success | Grounded control |
|---|---|---:|---:|---:|
| Fixed / dominant | 8, 99 | 19.0 | 22.55% | 0 |
| Fixed / unique | 8, 99 | 0.5 | 13.68% | 0 |
| EMA / dominant | 8, 99 | 25.0 | 41.23% | 0 |
| EMA / unique | 8, 99 | 0.5 | 12.75% | 0 |

The 300M data support a complete three-seed comparison of the two fixed-anchor
arms. Dominant has 27, 8, and 7 reliable edges for seeds 8, 99, and 123; UNIQUE
has zero in every seed. Mean accumulated prospective attempts are 93,403 versus
621, roughly a 150-fold difference. Mean per-run prospective success is similar
(19.59% versus 19.48%), showing why a success fraction alone is misleading when
opportunity counts differ so sharply. Both arms have zero snapshot grounded
controllability. EMA/DOM has two completed seeds, averaging 30 reliable edges,
but no EMA/UNIQUE run has a 300M snapshot; no terminal factorial is claimed.

Across the complete G500 5M, 25M, and 75M panels, EMA/DOM reliable edges grow
10.3 → 27.3 → 30.3, while EMA/UNIQUE remains 1.3 → 1.3 → 2.0. Fixed/DOM grows
0.3 → 11.7 → 20.3, whereas fixed/UNIQUE is 0 → 3.0 → 0.3. Combined with the
later matched comparisons, this favors persistent recognition-related graph
sparsity over a simple training-delay explanation. It does not prove which
representation change would solve the problem.

The practical interpretation is to keep three questions separate: EMA can help
exploration coverage; UNIQUE can reject ambiguous identities; and neither result
establishes dependable command-to-destination control. Improving coverage alone
would not resolve the current architectural limitation. The existing evidence
points toward investigating predictive-signature discrimination and recognized
event opportunities before adding further recognition thresholds.

## Early CPU spatial record: 25M and restricted 75M

The original September 24 NEMO2 CPU collection had all twelve 5M and 25M
online spatial snapshots. It had ten 75M rows; fixed-anchor seed 8 was absent.
The balanced 25M comparison and its atlas remain useful for development over
time. The restricted 75M table below is retained for exact provenance; the
later full three-seed CPU 75M analysis above is the current matched comparison.
These online windows are policy-conditioned and are separate from the G500
scalar history and checkpoint diagnostics.

### Matched 25M online spatial results

Values are three-seed means at the declared 25M snapshot. `Mono` is the fraction of eligible DG units with one field. `Cosine` is active-only map overlap. `Reliable edges` and `reachable pairs` describe the internal graph; neither by itself proves command-caused travel.

| Anchor / candidate | Cosine | Mono | Distinct DG peak bins | Reliable edges | Reachable pairs | Grounded control |
|---|---:|---:|---:|---:|---:|---:|
| Fixed / dominant | 0.186 | 0.047 | 42.3 | 23.0 | 0.007 | 0.000 |
| Fixed / unique | 0.209 | 0.102 | 45.0 | 0.3 | 0.000 | 0.000 |
| EMA / dominant | 0.276 | 0.052 | 40.0 | 28.0 | 0.008 | 0.000 |
| EMA / unique | 0.280 | 0.188 | 38.0 | 1.0 | 0.000 | 0.000 |

Within fixed anchors, unique contextual admission raises mono-field fraction in all three paired seeds but removes 36, 17, and 15 reliable edges relative to dominant admission. Within EMA anchors, it removes 32, 21, and 28 edges; mono-field fraction improves in two seeds and is effectively unchanged in one. This is a strong early graph-density association with the candidate rule. Because the graph construction depends on accepted events, the result is consistent with frequent `UNIQUE` abstention, but the table alone does not measure abstention.

At the per-run level, unique arms have only 0–2 reliable edges, whereas dominant arms have 15–36. The graph gap is consistent across both anchor modes and all three paired seeds; the mono-field advantage is less consistent.

Fixed anchors have lower mean map overlap than EMA in both candidate-rule strata, but the fixed-minus-EMA difference is not consistent in all three seeds for the dominant rule. This 25M map table alone cannot credit EMA with a representation effect; later refinement and recognition diagnostics are reported above. All four cells have zero mean grounded controllability at 25M.

The candidate-rule graph gap grows across the first two complete checkpoints. Under fixed anchors, dominant/unique mean reliable edges are 1.3/0 at 5M and 23.0/0.3 at 25M. Under EMA, they are 16.0/4.3 at 5M and 28.0/1.0 at 25M. Thus the unique rule is already graph-sparse early, and the dominant arms add edges while unique arms do not. At the September 24 collection cutoff, the full three-seed 75M factorial was incomplete. The later full CPU 75M collection above supersedes that availability limit.

### Seed-99 place-field and trajectory atlas at 25M

These canonical `segmented-atlas/v1` panels show the retained *online training windows* for fields and trajectories at the matched 25M target; directed matrices show graph counters stored at that checkpoint, not counts limited to the window. They are not frozen-policy rollouts. Each of the 64 DG units is normalized to its own peak (common 0–1 color scale), and gray cells were not visited. The selected seed-99 windows have 64/64 active units, but most do not meet the stricter mono-field criterion. The shared visitation pattern permits visual comparison of maps within a seed; it does not prove that either anchor rule improved control.

#### Fixed anchor: dominant versus unique candidate

![Fixed anchor and dominant candidate, DG units 0–15](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page01.png)

![Fixed anchor and unique contextual candidate, DG units 0–15](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page01.png)

![Fixed anchor and dominant candidate: occupancy and segmented trajectories](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_trajectory.png)

![Fixed anchor and unique candidate: occupancy and segmented trajectories](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_trajectory.png)

![Fixed anchor and dominant candidate: prospective outcome matrix](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_graph.png)

![Fixed anchor and unique candidate: prospective outcome matrix](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_graph.png)

Matrix rows are source DG units, columns target units, and color is prospective hits / attempts; gray means unattempted. This matrix includes attempts that did *not* become reliable graph edges. Fixed-anchor [dominant segment examples](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_segments.png) · [unique segment examples](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_segments.png).

The accumulated graph buffers corroborate sparse prospective evidence, not just a strict reliable-edge threshold: across seeds, fixed/dominant records 2,758–4,404 attempts versus 17–82 for fixed/unique; EMA/dominant records 4,136–9,911 versus 543–710 for EMA/unique. These counts are not per-decision attempt rates, and they do not by themselves identify which recognition gate suppressed opportunities.

#### EMA anchor: dominant versus unique candidate

![EMA anchor and dominant candidate, DG units 0–15](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page01.png)

![EMA anchor and unique contextual candidate, DG units 0–15](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page01.png)

Full 64-unit sheets and occupancy/segmented trajectories:

| Anchor / candidate | DG units 16–31 | 32–47 | 48–63 | Occupancy and trajectory |
|---|---|---|---|---|
| Fixed / dominant | [page 2](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page02.png) | [page 3](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page03.png) | [page 4](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page04.png) | [trajectory](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_trajectory.png) |
| Fixed / unique | [page 2](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page02.png) | [page 3](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page03.png) | [page 4](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page04.png) | [trajectory](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_FIXED_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_trajectory.png) |
| EMA / dominant | [page 2](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page02.png) | [page 3](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page03.png) | [page 4](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page04.png) | [trajectory](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_trajectory.png) |
| EMA / unique | [page 2](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page02.png) | [page 3](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page03.png) | [page 4](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_place_fields_page04.png) | [trajectory](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_trajectory.png) |

EMA extras: dominant [segment examples](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_segments.png) and [graph matrix](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_DOM_H32_DDQN_HER_S99/target_000025000000_policy_00_graph.png); unique [segments](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_segments.png) and [graph](../results/recent_architecture_batches_20260924/followup/CA3FU_CTX_EMA_UNIQUE_H32_DDQN_HER_S99/target_000025000000_policy_00_graph.png).

Trajectory colors mark independent retained segments, not time or speed; starts are circles, ends crosses. These seed-99 pictures complement, but cannot replace, the paired three-seed graph and mono-field comparisons above.

### Restricted September 24 snapshot: 75M factorial (seeds 99 and 123)

The September 24 canonical 75M snapshot table contained a complete 2×2 factorial for seeds 99
and 123. Cell summaries below use only those matched two seeds; each cell has
100,000 retained behavior samples per run. The prospective counts are pooled
over the two seed runs, while map, movement, graph-size, and reachability
columns are means of per-run summaries. These are descriptive two-seed results,
not confirmatory estimates. The source collector manifest records 34 of 60
expected snapshots overall and `include_details: false`; the 75M numeric table
is therefore reusable, while the 75M per-unit/per-field/edge details and atlas
were not produced in this bundle. The existing visual atlas remains the
complete matched 25M atlas.

| Anchor / candidate | N | Map cosine | Mono-field | Distinct peak bins | Visited grid | Stationary | Path efficiency | Reliable edges | Reachable pairs | Prospective hits / attempts | Grounded control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed / dominant | 2 | 0.187 | 6.8% | 47.5 | 88.1% | 13.7% | 0.512 | 19.0 | 0.632% | 7,727 / 36,475 (21.2%) | 0 |
| Fixed / unique | 2 | 0.217 | 7.0% | 50.0 | 88.1% | 10.6% | 0.388 | 1.0 | 0.025% | 25 / 291 (8.6%) | 0 |
| EMA / dominant | 2 | 0.361 | 2.3% | 38.0 | 88.1% | 18.1% | 0.476 | 25.5 | 0.744% | 45,146 / 138,355 (32.6%) | 0 |
| EMA / unique | 2 | 0.276 | 18.0% | 39.0 | 88.1% | 23.4% | 0.493 | 1.5 | 0.037% | 225 / 2,134 (10.5%) | 0 |

Averaged over the two anchor rules within each seed, UNIQUE minus DOM has a
mean mono-field increase of 7.9 percentage points and a mean cosine reduction
of 0.028, alongside 21 fewer reliable edges per run, 0.66 percentage points
less reachable-pair fraction, and a 16.2-point reduction in the mean per-run
prospective success fraction. The raw attempt count also contracts sharply
under UNIQUE. These paired descriptive contrasts are consistent across seeds
for edge count and reachable-pair fraction; the sample contains only two seeds.
All four cells remain at zero grounded controllability. High visited-grid
coverage (88.1% in every row) does not distinguish command control.

The two additional EMA seed-8 observations allow a separate three-seed EMA
subset: EMA/DOM averages 24.7 reliable edges and 34.9% prospective success,
whereas EMA/UNIQUE averages 1.7 edges and 10.5% success. Do not combine these
three-seed EMA means with the two-seed full-factorial contrasts.

[Canonical ten-row 75M snapshot subset](../data/recent_architecture_batches_20260924/ca3_state_goal_75m_restricted/per_snapshot.csv) · [two-seed cell summaries](../data/recent_architecture_batches_20260924/ca3_state_goal_75m_restricted/restricted_factorial_cell_summary.csv) · [seed-paired contrasts](../data/recent_architecture_batches_20260924/ca3_state_goal_75m_restricted/paired_seed_contrasts.csv) · [source canonical analysis manifest](../data/recent_architecture_batches_20260924/ca3_state_goal_75m_restricted/source_analysis_manifest.json).

The early collection used StudySpec schema `intrmotiv/study/v1`, workflow
`1.11.0`, and SHA-256
`68a0911fbc4719cc20eca4cf4145ffe0405b570d2ac7bf5a49a84643ab9f8437`.
Its [study definition](../../hpc_runs/studies/ca3_state_goal_followup_20260922_production.study.json),
[restricted 75M tables](../data/recent_architecture_batches_20260924/ca3_state_goal_75m_restricted),
and the retained figure links preserve the original snapshot provenance.
The original CPU scalar scan could not reconstruct balanced early histories
because the discovered fixed/dominant seed-99 event file ended at 5,865,472
steps. No early scalar contrast is inferred from those spatial rows.

## Provenance and next interpretation

Canonical workflow collectors produced the tables from the exact pinned G500 1.11 study and the compatible NEMO2 analysis checkout. Source, schema, study hashes, and windows are recorded in [G500 online manifest](../data/ca3_followup_20260926/g500_online_manifest.json), [G500 spatial manifest](../data/ca3_followup_20260926/g500_spatial_manifest.json), and [CPU spatial manifest](../data/ca3_followup_20260926/cpu_spatial_manifest.json). Full selected scalar histories remain on G500 at `/home/lin/ca3_analysis_20260926/online75m/histories`; only compact tables are copied here. CPU and G500 per-snapshot tables are in the same local data directory.

The strongest supported scientific result is recognition ambiguity coupled with severe UNIQUE graph sparsity. EMA's coverage benefit is encouraging but variable in magnitude and unsupported by improved grounded control. The next discriminating evaluation is the existing matched-command intervention plus spatial alias diagnostics on retained compatible checkpoints, with exact frame availability reported. No new training or evaluation jobs were launched in this analysis.

Reusable lesson: verify log and queue timestamps before trusting running status; inspect checkpoint byte sizes before projecting storage; reuse canonical scalar exports once rather than rescan events; and distinguish latent variance from predictive-signature separability. Initial spatial collection was blocked by the full scratch filesystem, and proceeded after the authorized cleanup. Unrelated local edits and merge conflicts were left intact.

The latest-model supplement successfully loaded 24 checkpoints. Memory mapping
limits tensor allocation but does not avoid parsing large replay metadata; CPU
reads took several minutes. Reuse the extracted diagnostics for further report
edits, and resolve rotating checkpoint paths immediately before reading them.
