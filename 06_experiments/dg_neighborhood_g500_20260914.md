# DG neighborhood experiment and G500 resource qualification

Status: implementation staged and focused tests passed; resource qualification
is running. Scientific preflights and production have **not** been launched.
This is the activation-anchored revision agreed in the task, not the earlier
fixed physical-center oracle proposal.

Read-only five-minute heartbeat `monitor-g500-resource-probes` is active for
the two current fixed-frame probes. Automatic approval review rejected a broader
automation that would implement changes and launch future runs, citing broad
future authority without renewed confirmation. The narrower monitoring action
was approved. No unattended implementation or production-launch automation is
enabled; do not reinterpret the read-only heartbeat as authorization to mutate.

## Scientific contract

Four online HIT/STOP/target-ID-FiLM conditions: existing DG objective,
neighborhood supervision, neighborhood plus temporal repulsion, and neighborhood
plus physical repulsion. Sixteen units; paired seeds 99, 8, 123. Frozen ImageNet
layer-2 ResNet, legacy non-affine DG BN, threshold 2.43, sparse-only CA3 writes.
Controller PPO does not update DG. Training trajectories may diverge; shared
held-out observation trajectories are required for representation comparison.

For each unit and each valid uninterrupted episode segment within a rollout,
the strongest positive activation is the detached anchor (earliest tie).
Positive samples lie inside a heading-aligned ellipse with semiaxes 150/100
and yaw difference at most 15 degrees. Negatives lie outside the 300/200 outer
ellipse or have yaw difference at least 30 degrees. Ignore the boundary and the
anchor itself. Positive and negative softplus terms use margin 1, negative weight
1, and separate count normalization. Average over anchored segments per unit,
then all 16 units. An entirely unanchored unit contributes zero; silent neighboring
responses still receive gradients from that unit's active anchor.

Temporal repulsion uses consecutive observations without crossing resets.
Physical repulsion samples one distinct neighbor within 100 world units using
a KD-tree and weights it by $100/\max(d,10)$. It ignores heading. Both use
coefficient 0.1 and the off-diagonal sparse-activity product sum. Physical pair
sampling is reproducible from training seed and saved learner update counter.
New arms bypass the old DG objective and auxiliary losses completely.

## Source and tests

- Parent runtime commit: `d94155f9be0436828ee9a744b57097db07022344`.
- Isolated G500 source:
  `/scratch/lin/IntrMotiv/src/SF_hipposlam_dg_neighborhood_20260914`.
- Desktop staging tree: `/tmp/intrmotiv-neighborhood-runtime`.
- Durable overlay, patch, and hashes:
  [source snapshot](../hpc_runs/source_snapshots/dg_neighborhood_20260914.sha256.json),
  [patch](../hpc_runs/source_snapshots/dg_neighborhood_20260914.patch),
  [overlay archive](../hpc_runs/source_snapshots/dg_neighborhood_20260914.tar.gz).
- Existing G500 canonical workflow remains 1.8.1. The overlay does not replace
  it with the older package in the original source archive.
- Desktop focused regression suite: **38 passed, 1 CUDA test skipped**.
- G500 neighborhood and CPU/CUDA device suite: **9 passed**.
- Resource-counter tests: **3 passed**.

The implementation adds `dg_neighborhood.py`, opt-in objective/pairing flags,
raw pose preservation through learner preparation, and scalar diagnostics. It
reuses the same DG forward's pre-threshold logits. No changes were made to the
desktop runtime checkout's unrelated in-progress depth work.

## Resource evidence and current probes

Live inspection found 96 physical cores / 192 logical CPUs, 377 GiB RAM with
329 GiB available, two RTX PRO 6000 Blackwell GPUs with about 87 GiB free each,
and 1.3 TiB scratch available. Other users have resident GPU processes; memory
and utilization are sampled, not assumed to be dedicated to this experiment.

The initial two concurrent GPU probes (8 and 16 environment workers, 2 envs per
worker, batch 2048) consumed only about 3.4% host CPU at the 95th percentile,
left at least 308 GiB available RAM, and left more than 81 GiB free per GPU.
GPU utilization was typically below 8%. Early completed-update counters did
not show a clear benefit from 16 workers. These interrupted probes are diagnostic
only, not qualified throughput measurements: their shutdown and frame reporting
windows are incomplete.

Current fixed-frame probes, started after both earlier process trees exited:

| Directory | GPU | Workers | Envs/worker | Batch | Frame target |
|---|---:|---:|---:|---:|---:|
| `w32b2048_fixed` | 0 | 32 | 2 | 2048 | 131072 |
| `w8b1024_fixed` | 1 | 8 | 2 | 1024 | 131072 |

All profiling outputs are under
`/scratch/lin/IntrMotiv/train_dir/resource_profile_20260914/`. Each directory
contains exact commands and parent StudySpec provenance, process identities,
five-second host resource samples, logs, and a terminal summary.

W&B project:
[SF_IntrMotiv_DGNeighborhood](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_DGNeighborhood).
Profiling group: `g500_resource_profile_20260914`. Authentication was verified
from G500; the existing desktop credential is supplied through encrypted SSH
stdin and inherited environment, never written to the workstation or command
arguments. Do not print process environments or authentication material.

Use [launch_profile.py](../hpc_runs/hosts/g500/launch_profile.py) from the desktop;
without `--execute` it prints the exact configuration. Use a fresh output name
for every retry. The normal stopping condition is completed environment frames;
`--seconds` is a separate external safety limit. Profile return code 0, no
traceback, online W&B, completed target frames, and measurable post-startup
throughput are required before ranking a configuration.

For a compact update, run on G500:

```bash
/scratch/lin/IntrMotiv/envs/SF_git/bin/python \
  /scratch/lin/IntrMotiv/tools/g500/summarize_profiles.py \
  /scratch/lin/IntrMotiv/train_dir/resource_profile_20260914
```

The [saved compact summary](data/dg_neighborhood_20260914/resource_profiles.json)
is a snapshot, not a live status source.

## Selection and remaining qualification

1. Complete fixed-frame worker/batch comparisons. Include a qualified 8-worker,
   batch-2048 reference; interrupted diagnostics cannot establish the optimum.
   Test batch 4096 only with enough frames for at least four reported updates
   (262144 frames). Keep the smaller worker allocation when throughput differs
   by less than 10%.
2. Prefer batch 2048 to preserve historical BN sampling unless an alternative
   provides at least 20% repeatable throughput improvement and passes the same
   numerical/BN/update gates. Fix batch geometry across scientific arms;
   never resize it during an active run.
3. Compare matched configurations at 2 and 4 simultaneous runs, then 8 and up
   to 12 only if aggregate throughput continues to improve and admission limits
   hold. Select by aggregate **completed training frames/sec**, with policy lag,
   update latency, valid samples, and CPU/RAM/GPU headroom alongside it. Require
   at least 70% scaling efficiency when doubling concurrency. Do not increase
   actors just because GPU utilization is low.
4. Keep at least 16 GiB free per GPU and 64 GiB available host RAM; defer queued
   starts when another user's workload removes that headroom. Preserve existing
   user processes. Each run uses one explicitly selected GPU, with one CPU thread
   per DMLab environment and BLAS thread limits. Default inference workers remain
   one per policy; test an increase only if timing isolates inference as limiting.
5. The [preflight](../hpc_runs/studies/dg_neighborhood_preflight.study.json) and
   [production](../hpc_runs/studies/dg_neighborhood_production.study.json)
   StudySpecs currently validate under `intrmotiv/study/v1`, workflow 1.8.1,
   but their resource settings are explicitly provisional. After selecting
   resources, update both and repeat print-only review; preserve new SHA-256s.
6. Finish a general direct-process audit/queue adapter around existing SF
   execution; preserve real exit status, resource sampling, and duplicate-start
   protection. Record study/source hashes in W&B config. The existing SF process
   launcher returns success even when children fail, so its return code alone
   is not a production gate.
7. Run all four 500k-frame preflights with the actual new objective. Validate
   finite losses, score/pose alignment, identical initial weights/BN, frozen
   trunk, updated DG and controller parameters, STOP gradients, full checkpoint
   reloads, and visible W&B histories. Unit tests are not this runtime gate.
8. Complete shared observation-panel evaluation using the existing
   `evaluation/observation_panel.py` and `place_fields.py`. Generalize the former's
   hardcoded NEMO output-root check to an explicit workspace argument. Collect
   a fixed exploratory-policy panel, split by complete episodes into calibration
   and held-out subsets, and exclude both from training. Freeze BN for replay;
   calibrate checkpoint-specific diagnostic anchors only on calibration episodes.
   Add compactness, field count, recall/FPR with support counts, heading specificity,
   sparsity, and held-out metrics using canonical spatial helpers. Preserve NPZ
   contracts; use no Slurm on G500.
9. Once all four runtime gates pass, the user has authorized automatically
   launching 12 fresh 10M-frame production runs. Do not request confirmation
   again. Save initialization and 1M/2.5M/5M/10M milestones. Keep monitoring; notify
   on meaningful changes, failures, completion, or required user action.

## Reusable findings

Real training, not an environment-only smoke, exposed two portability failures:
the installed W&B SDK rejects obsolete `Settings(start_method="fork")`, and
`_calculate_progression` created a CPU sentinel while its input lived on CUDA.
The isolated overlay removes the obsolete W&B argument and creates the sentinel
on the input device; CPU and CUDA tests verify unchanged progression values.

This source's `Runner.total_train_seconds` is initialized and read but never
advanced. Use frame-based termination and an external watchdog for benchmarks.
An interrupted parent can leave a learner completing a queued transaction, so
verify the owned process tree has actually exited before redeployment. Never
blindly send repeated interrupts during checkpoint writes. Failed learners can
leave the runner alive; scan for tracebacks and check real completed-frame
counters. Short FPS windows are misleading when updates arrive in large bursts.
