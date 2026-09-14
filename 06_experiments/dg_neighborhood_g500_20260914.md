# DG neighborhood experiment and G500 resource qualification

Status: **original production stopped at the user's request on September 15;
a controlled throughput search is running before a fresh production restart**.
The original four parents exited cleanly after SIGINT; pending runs were cancelled,
and all partial outputs remain available. Do not resume the old supervisor.

## September 15 throughput search and restart

The user explicitly requested stopping current runs, searching near 32 workers
and 8 environments/worker, then launching production with the best configuration.
The earlier 8-versus-32-worker probes used different batch sizes and cannot
establish an optimal worker count. Current measured CPU use was only 5.34% of
192 logical CPUs (96 physical cores), with about 297 GiB RAM available.

- Candidates (user revised): **32×8×1 epoch, 48×8×1, 32×16×1,
  32×8×2, 48×8×2**, sequential candidate groups. The brief 8×2 control
  was stopped and superseded; all four of its training parents exited.
- Each group: four concurrent runs (GPU slots 0,1,0,1), one per scientific arm,
  seed99; 262,144-frame target per run; 1,800-second safety deadline.
- Fixed batch2048, two minibatches, rollout/recurrence64; online W&B.
  Epoch count is an explicit user-authorized learning factor, not just a hardware
  setting. Selection maximizes environment throughput; nominal optimizer sample
  throughput is also recorded and does not imply better learning.
- Selection: greatest summed completed-frame FPS after warmup among normally
  completed candidates with at least 64 GiB RAM and 16 GiB GPU headroom.
- Winner receives a fresh 500k four-arm scientific qualification, then the same
  12×10M production matrix starts from original paired seeds in a new output root.
  Shared-panel checkpoint evaluation remains automatic after production.
- Search process: **1270880**; script
  `/scratch/lin/IntrMotiv/tools/g500/search_neighborhood_throughput.py`.
- Search status/results/reviews:
  `/scratch/lin/IntrMotiv/train_dir/analysis/dg_throughput_search_20260915_aggressive/`.
- Supervisor log: `/scratch/lin/IntrMotiv/logs/dg-throughput-search-20260915-aggressive.log`.
- Once selected, `selected_preflight.study.json`, `selected_production.study.json`,
  and `decision.json` in that directory are authoritative; mirror selected studies
  into local `hpc_runs/studies/` for the durable record.
- Production transition will be `production_transition/status.json` under the
  search directory. New production root is
  `/scratch/lin/IntrMotiv/train_dir/intrmotiv_dg_neighborhood_production_20260915_aggressive`.
- Five focused profiler/selection/study-preservation tests pass locally.
  The scientific runtime source fingerprint remains unchanged.

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
  Its `direct_execution/` is active and records four running/eight pending runs
  at launch; later waves follow canonical StudySpec order.
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
- Qualified and running source SHA: `cfaaba2f7d3f116520837bb6965e8a1d5682629f1ddd5229fa8d4d716ed840cc`.

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

## Passed preflight and production launch

- [Qualification report](data/dg_neighborhood_20260914/qualification.json): all
  four normal exits, exact paired initial states, frozen trunk, learned DG and
  decoder, finite losses/model/Adam state, exact reload and provenance passed.
- [Eight-checkpoint evaluation manifest](data/dg_neighborhood_20260914/preflight_evaluation_manifest.tsv)
  and [all per-unit summaries](data/dg_neighborhood_20260914/preflight_evaluations.json).
- [W&B qualification and held-out metrics](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_DGNeighborhood/runs/sjuei3if).
- [Resource decision](data/dg_neighborhood_20260914/resource_decision.json):
  1,769.38 aggregate frames/s; individual completed-update rates 438.86–444.88
  frames/s. Minimum available RAM 294.86 GiB; minimum free GPU memory 76.75 GiB.
  Selected four slots `[0,1,0,1]`, eight workers, two environments/worker, batch2048.
- [Production command manifest](data/dg_neighborhood_20260914/production_manifest.json);
  manifest SHA `302dd0e205990af82bcd8f500cd04f9ac483c40e27316a4d2ed137c8c235f66e`.
  The first wave is BASE seeds99/8/123 and SELF seed99; the remaining eight
  rows stay queued. No scientific condition or seed was removed.

### Early held-out representation results (seed 99 only)

| Objective | Active population | Silent units | Recall | False positives | Spatial RMS radius | Fields at 50% peak (eligible units) |
|---|---:|---:|---:|---:|---:|---:|
| Existing | 1.07% | 0 | 6.7% | 1.05% | 677.1 | 4.33 (12/16) |
| Neighborhood | 6.18% | 0 | 52.2% | 6.04% | 820.6 | 4.62 (16/16) |
| + temporal repulsion | 6.15% | 1 | 54.2% | 6.42% | 823.2 | 5.20 (15/16) |
| + physical repulsion | 4.86% | 0 | 54.9% | 4.77% | 817.4 | 5.07 (15/16) |

Recall/FPR and compactness are macro means over defined unit values. Field
counts average only canonical eligible units; support differs across arms.
Diagnostic anchors are selected separately for each checkpoint on calibration
episodes, so recall is local consistency around those anchors, not accuracy
on an identical set of externally fixed centers. RMS radius uses world units
and occupancy-corrected rates.

The new losses increased recall and population activity but also increased
false positives and spatial extent. Neither repulsion variant has yet shown
better compactness or fewer disconnected fields at this short checkpoint.
Physical repulsion reduced activity/FPR relative to neighborhood-only, while
remaining well above baseline. These are early single-seed observations; the
production comparison keeps every arm to test persistence and seed variability.

### Reusable execution lessons

The authoritative launch evidence is the saved manifest plus real child exit
status and completed-frame counters; the scientific evidence is the shared
feature-panel hash and per-unit checkpoint summaries. Checking full configs and
telemetry warmup assumptions earlier would have avoided startup retries and
gate revisions. Keep source immutable, separate host execution adapters from
scientific mechanisms, retain missing-denominator/eligibility information, and
reuse canonical checkpoint selection and map calculations. Four-run profiling
was sufficient for this first production allocation; broader concurrency tuning
can be a separate matched benchmark rather than changing an active study.

Production startup verification: all four first-wave runs reached 32,768
completed frames without tracebacks, with online W&B URLs. Available host RAM
was 295.96 GiB and GPU utilization was 14% on each GPU. The heartbeat is now
named **G500 DG production monitor** and follows the existing production queue.

Production milestone verification (18:58 UTC): the first four runs passed 1M
frames without tracebacks. All four 1M online snapshots passed canonical
`load_spatial_snapshot` validation at actual frame 1,015,808, each containing
100,000 observations and finite activity for 16 units. RAM remained about
295 GiB available and both GPUs remained at 14% utilization.

Production milestone verification (20:15 UTC): all four first-wave runs passed
2.5M frames; their 2.5M online snapshots passed canonical validation at actual
frame 2,523,136, each with 100,000 observations. SELF seed99 reached 3.05M;
the three baseline runs reached 2.59–2.72M. No tracebacks; eight runs remain
queued, RAM availability about297 GiB and GPU utilization14% each.
