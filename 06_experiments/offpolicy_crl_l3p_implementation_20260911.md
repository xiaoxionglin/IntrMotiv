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
observation medoids selected by farthest-point sampling in learned goal space.
Every node is actionable. A directed temporal-distance model is supervised by
future offsets and censored random negatives; its predicted costs form a
sparse nearest-neighbor graph. Floyd-Warshall supplies the next visual subgoal.

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

The print-only matrix and submitted manifest both passed the canonical audit.
NEMO2 jobs `8052615`--`8052620` were submitted with a 30-hour limit, and all
six entered the running state.
