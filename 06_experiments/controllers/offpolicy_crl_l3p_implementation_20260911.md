# Off-policy CRL+ and L3P+ implementation

Date: 2026-09-11

## Status

CRL+ and L3P+ are implemented as discrete-action, visual goal-conditioned
off-policy baselines. The corrected canonical preflight5 StudySpec is
`hpc_runs/studies/offpolicy_goal_baselines_preflight.study.json`, workflow
version `1.7.0`, SHA-256
`34fcd452196234686fb3e1320f384bb4d38fa0ecc6544513fbc5b9723c66b8f0`.

NEMO2 jobs `8052564` (CRL+) and `8052565` (L3P+) completed 100k frames with
exit code zero in 7:30 and 6:27 respectively. Both wrote final checkpoints,
TensorBoard histories, provenance, and finite learner metrics. These are
runtime/learning gates, not final performance runs.

The promoted three-seed, one-million-frame pilot is declared by
`hpc_runs/studies/offpolicy_goal_baselines_pilot.study.json`, workflow version
`1.7.0`, SHA-256
`aa3a7077e88657bfad711f86ceb872221e691ba3ecca0315d2cbc703a19d0f15`.
Its six audited jobs are `8052591`--`8052596` for seeds 8, 99, and 123.

The strengthened production implementation is declared by
`hpc_runs/studies/offpolicy_goal_baselines_parallel_production_v2.study.json`,
workflow version `1.7.0`, SHA-256
`8ecad0d54b17f9ee5f67fe0a65b9e2eda880decfe2c171058facc5f2d6e05ebe`.
Its six released L40S jobs are `8054886`--`8054891`.

## Why this is an adaptation

The official CRL code is Acme/JAX/Reverb with continuous SAC-style policies.
The official L3P code is PyTorch/MPI DDPG with coordinate goals, HER, a learned
goal autoencoder, variational clusters, and continuous control. Importing either
trainer would change the environment and action interface more than the
algorithm comparison.

The retained causal components are:

- CRL's action-labeled future-state contrast and goal-conditioned actor;
- the strongest published JaxGCRL loss choices: symmetric InfoNCE, L2 energy,
  and log-sum-exp stabilization;
- L3P's reachability-aligned latent landmarks, sparse directed graph, temporal
  abstraction, and graph search.

The visual adaptation stores stable float16 outputs from the exact frozen
ImageNet layer-2 trunk. Trainable state, goal, action, distance, and actor heads
operate after that trunk. This avoids repeatedly storing or re-encoding pixels
while preserving the requested frozen visual encoder.

## L3P+ changes

The coordinate decoder and Gaussian-mixture centroids in original L3P are not
well-defined visual goals under a fixed trunk. L3P+ therefore uses achieved
observation medoids selected by farthest-point sampling in a symmetric temporal
landmark space. Every node is actionable. A directed temporal-distance model
is supervised by future offsets and censored random negatives; only predicted-
reachable edges enter the sparse directed graph. Floyd-Warshall supplies the
next visual subgoal.

The initial adaptation had three substantive weak links. It selected medoids
in the generic contrastive goal embedding rather than a reachability-aligned
space, replanned on a fixed cadence rather than committing for predicted travel
time, and admitted every nearest-neighbor edge even when the distance model
predicted it was unreachable. The strengthened implementation adds an L3P-only
symmetric temporal landmark objective, clips graph edges at the trained
64-decision horizon, commits to a landmark for its predicted duration, and
excludes the immediately failed landmark on replanning. CRL does not evaluate
or optimize the L3P-only head.

Long 120-second DMLab episodes also delayed replay learning until timeout. Both
methods now stream ordered 512-decision trajectory segments into replay. This
preserves future-state ordering and the 64-decision relabeling horizon while
making the first replay data available around 66k aggregate frames with 32
collectors.

A local-reachability gate permits the direct current-to-goal edge only when
predicted distance is at most 16 decisions. Without this gate, the direct edge
always bypassed the landmark graph even though its nodes and edges were valid.

This is intended as a stronger compatible variant, not a claim of exact source
reproduction. The matched CRL+ cell isolates the effect of explicit landmark
planning because both cells share the controller and representation objective.

## Preflight history and reusable lessons

The first attempt (`8052500`, `8052501`) reached environment and encoder
construction but failed while serializing Gym's NumPy action count. JSON
provenance now converts NumPy scalars explicitly.

The second attempt (`8052508`, `8052509`) completed an episode and exposed the
environment's terminal-image compatibility behavior: DMLab returns a cached
image after termination, and the pixel wrapper can transpose that cached value
again. The trainer now treats the last valid feature and pose as the absorbing
terminal sample. Each correction used a new batch/output namespace, preventing
failed artifacts from being confused with valid results.

The preflight4 pair (`8052541`, `8052542`) completed cleanly but exposed the
planner bypass: L3P+ rebuilt 50-node graphs with 400 finite directed edges, yet
its landmark-subgoal fraction remained exactly zero. Preflight5 added the
local-reachability gate. At its last periodic record, L3P+ had rebuilt three
graphs and selected landmark subgoals for 83.6% of 972 planner queries.

The canonical collector then exposed two standalone-launcher integration
details. Sample Factory projected `00_RUN` into a `RUN_` directory, and the
standalone trainer initially wrote events to `events/` rather than
`.summary/0/`. Discovery now accepts the deterministic separator suffix, with
a regression test, and new training writes the canonical summary layout.
Compatibility symlinks were added only to the two completed preflight batches.

The authoritative checks were the generated `jobs.tsv`, canonical submission
audit, Slurm stderr, workspace `run_config.json`, checkpoints, TensorBoard
events, and live `metrics.jsonl`. Repository-only unit tests could not expose
the environment/runtime boundaries. Future off-policy integrations should run
a one-episode Slurm gate before larger matrices and validate JSON scalar types,
terminal observation shapes, planner use, run-directory discovery, and event
layout explicitly.

## 100k-frame result and decision gate

Over the final 50k-frame TensorBoard window for seed 99, CRL+ reached mean
coverage AUC 51.16, contrastive retrieval accuracy 0.128, policy entropy 1.45,
and option success 0.0309. L3P+ reached coverage AUC 49.53, retrieval accuracy
0.131, entropy 1.63, and option success 0.0259. Chance retrieval is 1/128, so
both critics learned nontrivial matching. There is no credible winner at one
seed and 100k frames; CRL+'s AUC lead is only 1.63, and L3P+ has only recently
begun using its graph.

The gate criteria are satisfied: terminal checkpoints exist, metrics are
finite, entropy has not collapsed, and the corrected planner is actively used.
The one-million-frame pilot therefore advances the same settings without new
tuning. Interpret its paired three-seed terminal windows before deciding on a
long production comparison with IntrMotiv.

## Production screen

The production screen is declared by
`hpc_runs/studies/offpolicy_goal_baselines_production.study.json`, workflow
version `1.7.0`, SHA-256
`d1dbe5065ecc820606b4490ed79366bd414c241412e1a779ea61016e5ca1cb6e`.
It runs CRL+ and graph-gated L3P+ for 10M frames at paired seeds 8, 99, and 123,
with checkpoints every 1M frames and synchronized analysis milestones at 1M,
2.5M, 5M, 7.5M, and 10M. The replay holds 200k frozen feature transitions;
all other learned-objective settings are unchanged from the passing gate.

Ten million frames is the production screening horizon rather than the usual
75M IntrMotiv horizon. Measured single-environment throughput projects 75M to
roughly four days per run, beyond the established 30-hour NEMO2 CPU envelope;
10M is expected to finish in approximately 12--13 hours and includes forty L3P
graph rebuild opportunities. Escalate the winner to the shared longer horizon
only if paired coverage and control evidence at 10M warrants the cost.

The original print-only matrix and submitted manifest passed the canonical
audit, but its single-environment jobs `8052615`--`8052620` were cancelled once
their roughly 200-frame/s design was judged incomparable with the approximately
3,267-frame/s IntrMotiv/APPO reference. The one-million-frame pilot jobs remain
valid scientific runs; the cancelled 10M matrix must not be resumed or treated
as production evidence.

## Parallel throughput implementation and gate

`hpc_runs/offpolicy_goal_baselines/train_parallel.py` replaces serial collection
with 32 shared-memory Gymnasium `AsyncVectorEnv` workers, batches the exact
frozen ImageNet ResNet-18 layer-2 trunk across observations, batches policy
action inference, and retains a single shared episode replay and learner. The
scientific update intensity is unchanged at one replay update per 16 decisions.
The vector worker implements same-step reset while preserving final terminal
information under both Gymnasium API variants used by the project.

The CPU gate (`8052658`, `8052659`) completed 500k frames with exit code zero.
Collection-only throughput reached about 2.4k frames/s, but sustained throughput
after learner activation was only 770.94 frames/s for CRL+ and 767.30 frames/s
for L3P+. This isolates the bottleneck: environment collection parallelizes,
whereas the frozen trunk and replay updates were still serialized through one
CPU learner.

The GPU path deliberately reuses the existing `dmlab_pack` CUDA environment,
rather than rebuilding PyTorch or DMLab wheels. Compatibility fixes were kept
small: Python 3.8-safe annotations, an idempotent source-controlled Lua-level
overlay, conda activation without nounset, and preloading that environment's
`libstdc++`. RTX preflight7 (`8052759`, `8052760`) proved all 32 custom DMLab
workers initialize, then correctly failed before training because the RTX PRO
6000 Blackwell GPU requires `sm_120` kernels absent from the installed CUDA
wheel. L40S preflight8 jobs `8052765` and `8052766` used a compatible Ada GPU
and completed 500,096 frames with exit code zero in 5:21 and 5:22. Its
StudySpec fingerprint is
`8f578bd09b14e14ef84ff04bf2e7b73f0fc13ea3006ea9b8923c68171f062bf0`.

The original launch decision required both CRL+ and L3P+ to reach 500k frames
with finite metrics, nonzero replay updates, active L3P graph use, and learner-
active throughput of at least 2,500 frames/s each (76.5% of the reference). The
staged production StudySpec is
`hpc_runs/studies/offpolicy_goal_baselines_parallel_production.study.json`,
workflow `1.7.0`, fingerprint
`85feeb53089aef1f4f68a0d31c1a6a2ba571c0a295d3af258dc24194bd9e91df`.
Its six 10M-frame L40S runs passed local validation, the 35-test focused suite,
NEMO2 validation, print-only review, and canonical submission audit. It was not
submitted because the throughput gate failed. Between the first learner-active
and last recorded TensorBoard points, CRL+ processed 216,192 frames in 150.231 s
(1,439 frames/s) and L3P+ processed 206,336 frames in 150.242 s (1,373 frames/s).
Their final recorded cumulative rates were 2,291 and 2,239 frames/s, also below
the 2,500-frame/s threshold. Both learners made thousands of finite updates;
L3P+ rebuilt four graphs, retained 400 finite edges, and used a landmark for
88.0% of 2,787 planner queries at its last periodic record. Correctness passed,
but throughput did not, so no production jobs were submitted.

The reusable lesson is to benchmark collection-only and learner-active phases
separately. Off-policy replay makes collectors easy to parallelize, but it does
not automatically parallelize frozen visual inference or gradient updates; a
single CPU learner simply moves the bottleneck. Reusing a GPU environment also
requires matching GPU compute capability, not merely observing that CUDA is
available.

## One-million-frame pilot result

All six pilot jobs `8052591`--`8052596` completed with exit code zero. The
canonical latest-common analysis used steps 893,364--993,364 and preserved
StudySpec fingerprint
`aa3a7077e88657bfad711f86ceb872221e691ba3ecca0315d2cbc703a19d0f15`.
The analysis artifacts are under
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/offpolicy_goal_baselines_pilot_20260911_final/`.

Across seeds 8, 99, and 123, CRL+ achieved coverage AUC $57.32 \pm 5.16$ and
100.46 $\pm$ 9.39 unique cells; L3P+ achieved $55.45 \pm 4.18$ and
99.69 $\pm$ 6.83. The paired L3P-minus-CRL coverage difference was -1.87 AUC
and -0.77 unique cells. Option success was essentially identical (0.03874 for
CRL+, 0.03862 for L3P+), as was entropy (1.0368 versus 1.0316). L3P+ had modestly
higher contrastive retrieval accuracy, 0.3754 versus 0.3518, but this did not
translate into higher coverage.

L3P+'s graph was operational rather than bypassed: every seed ended with nine
rebuilds and 400 finite directed edges, and landmark-subgoal fractions were
85.8%, 86.2%, and 86.6%. The matched pilot therefore provides no evidence that
explicit landmark planning improves exploration over CRL+ at 1M frames. Given
the failed throughput gate and small, inconsistent seed-wise coverage effects,
neither method showed an empirical advantage that independently warranted the
staged 10M matrix. The throughput measurements and pilot result remain valid;
the later production decision was an explicit choice to obtain the longer-
horizon comparison after strengthening the algorithmic implementation.

## Strengthened gate and production v2 launch

The strengthened code passed 38 local and synchronized NEMO2 tests. A fresh
L40S correctness pair (`8054884`, `8054885`) was audited but remained pending
because all partition GPUs were allocated, then was canceled without running.
The correctness-only CPU fallback StudySpec has SHA-256
`effabeee3649a3d0feba17f1f1d31797215768074453bf9690dbcb9742da9164`.
Jobs `8054893` (CRL+) and `8054894` (L3P+) completed 250,112 frames with exit
code `0:0` in 5:51 and 7:10. Both wrote 125k, 250k, and terminal checkpoints.

The gate exercised the causal weak links rather than merely importing the
modules. Replay contained 16,384 transitions by the first records at 65,792--
72,576 frames, well before full-episode timeout. CRL reported zero landmark
loss and no gradient path through the L3P-only head. L3P's landmark loss fell
from 2.51 to 0.50 by its last periodic record. At that record its retrieval
accuracy was 0.273 (chance is 0.0078), graph pair reachability was 0.710, 400
filtered directed edges remained, 93.4% of 1,795 planner queries selected a
landmark, mean commitment was 29.6 decisions, and 540 immediate repeats had
been avoided. Pair reachability reached 1.0 at the preceding rebuild and later
varied as the learned geometry changed; this is a useful production diagnostic,
not a fixed invariant.

Production v2 uses the same 32-collector L40S configuration whose measured
throughput was accepted, the exact frozen ImageNet ResNet-18 layer-2 trunk, no
pose policy input, paired seeds 8/99/123, and 10M frames per cell. Its print-only
and submitted manifests passed the canonical audit. The scheduler ignored an
attempted `SBATCH_DEPENDENCY` environment variable, so jobs `8054886`--
`8054891` were explicitly held before allocation, then released only after the
CPU correctness gate passed. They are now eligible in the L40S queue; a
`Priority` pending reason reflects partition saturation rather than a hold or
dependency.

Reusable launch lesson: do not assume a launcher-propagated `SBATCH_DEPENDENCY`
environment variable is honored. Verify `Dependency=` in `scontrol show job`
immediately. For a manually gated batch, explicit `scontrol hold`/`release`
provided the authoritative scheduler state, while the StudySpec audit remained
authoritative for the scientific command matrix.
