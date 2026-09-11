# DDQN dashboard consistency audit — 2026-09-12

The native SF migration did not preserve the earlier DG/exploration/control telemetry contract. Configuration flags alone falsely suggested coverage: the native learner never invoked `TrainingSpatialTelemetry`, and its runner did not register `write_intrmotiv_summaries`. Completed DDQN runs have no behavior-time spatial snapshots. Their missing history cannot be recreated from counters or model checkpoints.

## Where the old dashboards and data are

The earlier DG-capacity experiments use the W&B project [SF_IntrMotiv_DGCapacityGoalConditioning](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_DGCapacityGoalConditioning). The [F16 worker parent](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_DGCapacityGoalConditioning/runs/00_DGC_DIRECT_WORKER_F16_S99_20260910_202756_343003) retains its original metrics. Native SF DDQN runs were logged to the separate `IntrMotiv` project. [Historical dashboard inventory](data/intrmotiv_ddqn_metric_consistency_20260912/old_dashboards.json) preserves the seed-99 run URLs and the F16 parent’s cloud summary.

On NEMO2, under `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/`:

- Old TensorBoard/config/checkpoints: `intrmotiv_dg_capacity_goal_conditioning_20260910/`.
- Old behavior snapshots: `analysis/online_spatial/intrmotiv_dg_capacity_goal_conditioning_20260910/` (78 NPZs at audit).
- DDQN v1 logs: `intrmotiv_ddqn_her_frozen_pilot_20260911/`.
- DDQN v2 standalone logs: `intrmotiv_ddqn_her_v2_20260911/`.
- Native SF DDQN logs: `intrmotiv_ddqn_sf_production_20260912/`.

DDQN run directories retain `metrics.jsonl`, TensorBoard event files and `runtime_gate.json`. [Exact inventory](data/intrmotiv_ddqn_metric_consistency_20260912/inventory.json) contains every discovered path, event file, tag and terminal counter. No old data were deleted, relabeled or uploaded as new measurements.

## Architecture and metric comparability

| Component | Earlier DG/HRL batches | Current DDQN/HER implementation |
|---|---|---|
| Representation | Fixed layer-2 ResNet; DG learning depends on study cell | Entire transferred encoder/DG fixed |
| Exploration | Intrinsic reward and manager strategy depend on cell | Epsilon-greedy actions and random eligible commands from goals 1/4/11 |
| Control | PPO worker and, in relevant cells, persistent graph/manager | Recurrent DDQN first-arrival controller with 64-decision deadline; optional future HER |
| DG activity and spatial maps | Original IntrMotiv telemetry | Applicable, but collector/router were omitted |
| Environment coverage | Existing reward-shaping wrapper | Wrapper retained; SF batched list-info path discarded episode extras |
| Graph/recruitment diagnostics | Available when those modules are active | No active graph or recruitment learner; unavailable, not zero |
| Control evidence | Option/graph diagnostics and independent interventions | Per-goal commanded arrivals, attempts, TD support; independent intervention evaluation still required |

The repaired dashboard cannot imply that adaptive representation learning or graph planning has been implemented. A low TD loss or a restored DG density trace does not qualify commanded control or diverse place fields.

Compared with v2 standalone, native SF retains the v2 worker, replay/HER objective, source checkpoint, registry and nominal learning budget. Standalone finished at 5,000,064 frames and 19,265 updates; native stopped at exactly 5,000,000 and 19,264. This one-update endpoint difference follows complete standalone collection batches versus the native exact stop. It is not a changed update ratio. All six native jobs completed with exit 0:0 and terminal runtime gates.

## Repair

- Reuse `TrainingSpatialTelemetry`, its latest-100k snapshot ring and latest-10k scalar window, and the existing spatial metric calculations and NPZ contract.
- Reconstruct exact thresholded DG activity from cached frozen preactivations using the source projection's activation and intercept. Exclusive goal-recognition events are not substituted for multi-unit DG activity. Pose remains outside policy feature inputs.
- Register the existing IntrMotiv summary router, preserving `intrmotiv/dg/*`, `intrmotiv/online/place_field/*` and `intrmotiv/online/trajectory/*` aliases.
- Add an opt-in SF list-info report adapter forwarding the existing environment's episode extras. Deployed SF already removes and forwards periodic stats; the adapter does not duplicate those reports. No coverage formula, environment reward or SF source file is replaced.
- Omit graph headline zeros emitted by the generic spatial coordinator when no graph exists. Never pass the inherited parent's stale graph as the child's learned graph.
- Restore omitted DDQN update-budget scalars and effective TD positions per decision. Native epsilon is labeled against the shared inference-decision clock, which can lead consumed decisions under asynchronous collection; ingestion time is not renamed as collection time.
- Remove inherited unused extra policy-output declarations and the parent's stale command-line metadata; retain the independent W&B identity fix and declare both `intrmotiv` and `ddqn` metric namespaces.

The completed production checkout and artifacts remain unchanged. The repair is staged in `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ddqn_telemetry_20260912` and the vault's canonical off-policy package.

## Qualification

The existing 75 focused tests plus two exploration-report regressions passed on NEMO2. An additional real-runtime test checks exact activity reconstruction, unchanged cached input data and absence of fabricated graph zeros. Two 1M-frame compute-node preflights exercise DDQN/HER, W&B and 500k/1M snapshots.

Study: `hpc_runs/studies/intrmotiv_ddqn_telemetry_preflight.study.json`; schema `intrmotiv/study/v1`; workflow 1.8.0; SHA-256 `4c979bc928bcf335859d1ee7ac0047f84d3d286b96b10878eb4bd88b46fce416`. Print review `20260911T232200Z`; audited submission `20260911T232301Z`; jobs 8057244/8057245. Both use ordinary Slurm jobs with workspace-only outputs. Completed runtime/cloud qualification is recorded below.

## Completed qualification

Both Slurm jobs completed with exit 0:0 (6:26 and 7:11). The 78 focused tests passed on NEMO2 across the suite and added reconstruction test; locally 77 passed and the full-runtime reconstruction test was skipped because that runtime module is absent. Both runs reached exactly 1,000,000 frames and 3,648 updates; frozen-reference, exact TD-position, target-copy and initialization gates passed. The four snapshots each contain 100,000 aligned pose/DG samples and pass the existing spatial contract. Local and deployed adapter hashes match.

| Run | W&B | Frames | Updates | Snapshots |
|---|---|---:|---:|---:|
| DQTEL_DDQN_HER_S99 | [Verified dashboard](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQTEL_DDQN_HER_S99_20260912_012350_646486) | 1,000,000 | 3,648 | 2 |
| DQTEL_DDQN_S99 | [Verified dashboard](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQTEL_DDQN_S99_20260912_012349_605919) | 1,000,000 | 3,648 | 2 |

Verified cloud families: `intrmotiv/dg/*`, `intrmotiv/online/place_field/*`, `intrmotiv/online/trajectory/*`, and `policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc` (plus coverage entropy, cells and pose summaries). DDQN goal-control counters remain under `ddqn/goal_{1,4,11}/*`; they are not aliases for graph reliability, target-hit lift or independent commanded evaluation. The environment resets before its configured 10k-step periodic exploration window, so episode coverage is the applicable exploration series for these runs.

Evidence: [cloud verification](data/intrmotiv_ddqn_metric_consistency_20260912/preflight_verification.json), [runtime audit](data/intrmotiv_ddqn_metric_consistency_20260912/runtime_audit.json), [spatial validation](data/intrmotiv_ddqn_metric_consistency_20260912/spatial_validation.json), [submission audit](data/intrmotiv_ddqn_metric_consistency_20260912/submission_audit.json). Learner-active throughput was 3,949 FPS (DDQN) and 3,300 FPS (HER); this is a qualification measurement, not a paired estimate of telemetry overhead.

## Reusable lessons

Treat metric availability as a runtime contract, not a copied configuration flag. Native framework execution does not automatically invoke application-specific learner hooks. Preflight promotion must check representative DG, spatial, exploration and control tags plus a real NPZ artifact. Mark absent architecture components as unavailable. For locating old dashboards, inspect config/W&B metadata and cached analysis first; a full historical TensorBoard scan was unnecessarily slow and was stopped. Use canonical run discovery and preserve original histories; checkpoint probes are new controlled measurements, never retrospective behavior logs.
