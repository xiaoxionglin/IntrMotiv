# DG neighborhood experiment and G500 resource qualification

Status: the four corrected scientific preflights (revision r4) are running
with online W&B. The shared held-out panel and replay smoke have passed.
The unattended transition process is waiting for preflight completion and will
launch production only after exact checkpoint/evaluation gates pass. Production
has **not yet** started at this update.
This is the activation-anchored revision agreed in the task, not the earlier
fixed physical-center oracle proposal.

The user renewed unattended authority on September 14: “when you are done
evaluating, start the production, decide without asking me questions.” The
existing five-minute heartbeat `monitor-g500-resource-probes` has consequently
been updated to **G500 DG production gate**, with implementation, qualification,
and production launch authorized. Automatic review approved this update.

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

## Resource evidence and completed probes

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

Completed fixed-frame probes, started after both earlier process trees exited:

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

## Selected resources and unattended execution

Eight workers, two environments per worker, batch 2048, recurrence/rollout 64,
one BLAS thread, one selected GPU per run. All four 500k preflights run together,
two on each GPU. A 32-worker profile did not improve throughput. Completed
normal-exit profiles measured 325.65 frames/s for 8 workers/batch1024 and
280.12 frames/s for 32 workers/batch2048; these are confounded comparisons,
so they do not establish a batch-size optimum. Keep historical batch2048.

Production uses two or four slots. The transition compares aggregate completed-
update preflight throughput to the conservative two-GPU profile reference
560.24 frames/s. Four slots require at least 1.4 times that reference and the
memory reserves below; otherwise use two. This is a practical resource decision
using mixed-objective evidence, not a global optimum. Further 8/12-way profiling
is deferred to avoid delaying the first interpretable experiment.

The canonical direct queue keeps at least 64 GiB host RAM and 16 GiB free per
GPU, plus admission allowance for each new run. It waits when other workloads
consume headroom. It owns only its children and does not stop other users' jobs.
Scientific batch geometry never changes mid-run.

### Authoritative live artifacts on G500

All paths below are under `/scratch/lin/IntrMotiv`:

- Preflights: `train_dir/intrmotiv_dg_neighborhood_preflight_20260914_r4/`.
  `direct_execution/state.json` contains exact process identity, progress,
  real exit status and W&B links. `manifest.json` binds commands and hashes.
  Queue PID at launch: **1239255**.
- Transition: `train_dir/analysis/dg_neighborhood_transition_20260914_r4/`.
  `status.json` is the stage indicator; `evaluations/qualification.json`
  is the fail-closed gate; `resource_decision.json` records slot selection;
  `production_review.json` is the print-only manifest before launch.
  Log: `logs/dg-production-transition-r4.log`. Transition PID: **1246395**.
- Production: `train_dir/intrmotiv_dg_neighborhood_production_20260914/`.
  Its own `direct_execution/` appears only after qualification.
- Panel: `train_dir/analysis/dg_neighborhood_shared_panel_20260914/`.
  **20,700 decisions / 23 whole episodes**; first 9,900 calibration,
  remaining 10,800 held out. Seed 314159, random action held for 8 decisions.
  Panel SHA: `8307d87cbb943d20986bdba2bb2db982bebcb6659fbad0f1d73567f460add2da`.
- Replay smoke: `train_dir/analysis/dg_neighborhood_evaluation_smoke_20260914/`.
  Log: `logs/dg-evaluation-smoke.log`; exact reload/paired-init checks:
  `logs/dg-reload-smoke.log`.

Do not launch a second transition or production queue while these owned
processes are active. The five-minute heartbeat monitors them and handles
failures; it should report only meaningful changes. A completed queue does not
need resubmission. Failed output directories are retained as evidence; recovery
must use an explicit new revision rather than implicit resume.

### Correctness and evaluation gates

The four initial model state dictionaries match exactly. Actual learner logs
show finite neighborhood losses, positive/negative supervision support, zero
pairs for self-only, and nonzero pairs/penalties for both repulsion arms.
The real model/Adam reload smoke passed. Queue tests cover successful child
completion and failure blocking a queued sibling (4 canonical tests total),
plus 3 profile-counter tests. Two remote metric tests cover silent-unit undefined
values and circular heading boundaries; full frozen-panel replay also passed.

At 500k, require normal exits, actual terminal frame counts, online W&B,
finite model/optimizer tensors, unchanged frozen visual trunk, learned DG and
controller weights, saved source/study/manifest provenance, valid shared-panel milestone
NPZs, exact model/optimizer reloads and shared held-out evaluation. Scientific
metric quality is reported, not used to silently drop an experimental arm.

Evaluation reuses `observation_panel.record_observation`, the existing
`place_fields.load_policy_env` checkpoint loader and canonical
`calculate_place_field_details`. The small workstation adapter writes the
existing panel arrays under scratch rather than modifying the active source's
legacy NEMO-only `save_panel` guard. Frozen visual features retain the existing
fixed instruction embedding. DG replay keeps BN in eval mode and checks that
its buffers do not change and activations equal `relu(z - 2.43)` exactly.

Per-unit summaries include occupancy-corrected spatial RMS radius, disconnected
fields at canonical 0.3/0.5/0.7 peak thresholds, eligibility, recall/FPR and support
counts, active fraction, heading resultant and amplitude-weighted spatial score.
Population summaries include sparsity, silent units, active-only map cosine and
peak diversity. Calibration anchors are never selected from held-out episodes.
Undefined metrics are null. The raw activity/pre-threshold/pose arrays and maps
are retained for both splits. Online learning trajectories are not the fixed-
trajectory comparison.

### Startup failures retained for diagnosis

- Original preflight: inherited `dg_recruitment_reset_goal_adapter=True`
  conflicted with recruitment disabled; failed before training.
- r2: the new provenance assignment treated argparse Namespace as a dictionary;
  corrected to `setattr`, before any training updates.
- r3: inherited default telemetry interval 25M exceeded the 500k maximum;
  explicit compatible intervals are now 250k preflight / 1M production.
- r4: the corrected matrix is executing real learner updates with online W&B.

Production remains the authorized four arms × seeds 99/8/123 × 10M frames,
with initial and 1M/2.5M/5M/10M milestones. It starts automatically after these
gates, with no further confirmation. The same panel is reused through the canonical evaluation manifest: declared
1M/2.5M/5M/10M checkpoints for seed 99 and terminal checkpoints for seeds 8/123.
Complete per-unit summaries are uploaded as W&B evaluation artifacts.

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

## Current provenance

- Schema: `intrmotiv/study/v1`; workflow: `1.8.1`.
- Preflight StudySpec SHA: `04969ce9f0a4db548d57bde58ae38ec4a3ab245936426b0ff6b9479a1396f8d2`.
- Production StudySpec SHA: `a0748f739149bade77db00c46e5dc2cbcab7dc6ba0d0ec2ebe2cb7306921cb5b`.
- Running source SHA: `cfaaba2f7d3f116520837bb6965e8a1d5682629f1ddd5229fa8d4d716ed840cc`.

The waiting transition was revised twice before qualification began, first to
reuse canonical milestone manifests and publish production W&B artifacts, then
to match the established telemetry `batch/run/policy` directory layout. These
restarts did not interrupt the four training processes. Only r4 is active.

### Online map warmup and preflight evaluation

The retained online map window is 100,000 observations × 8 frames/observation,
so it cannot fill during a 500k-frame preflight. Absence of full online map
NPZs before 800k is expected. Scalar diagnostics use the smaller 10k window
and are present. Transition r4 evaluates both the saved 262,144-frame milestone
and terminal checkpoint on the identical shared panel and validates their map
NPZs; it retains all model, loss, reload and provenance gates. Production must
produce online snapshots once the configured window fills, before its first
1M target. This corrects the gate's warmup assumption without changing training,
shortening the map window, or discarding the required checkpoint evaluations.

Four additional transition-helper tests verify throughput-based two/four-slot
selection, low-RAM fallback, and nested optimizer-finiteness checks.
