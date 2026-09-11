# Off-policy CRL+ and L3P+ implementation

Date: 2026-09-11

## Status

CRL+ and L3P+ are implemented as discrete-action, visual goal-conditioned
off-policy baselines. The canonical preflight3 StudySpec is
`hpc_runs/studies/offpolicy_goal_baselines_preflight.study.json`, workflow
version `1.7.0`, SHA-256
`9906edd19692afe8540a85d1899559d07e67e251d176ad738c51b9705cbedf0d`.

NEMO2 jobs `8052513` (CRL+) and `8052514` (L3P+) passed environment creation,
the frozen encoder contract, a complete 1,800-step episode, future-goal replay,
and the first learner updates. They are 100k-frame runtime gates, not final
performance runs.

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

The authoritative checks were the generated `jobs.tsv`, canonical submission
audit, Slurm stderr, workspace `run_config.json`, and live `metrics.jsonl`.
Repository-only unit tests could not expose either runtime boundary. Future
off-policy environment integrations should run a one-episode Slurm gate before
larger matrix work and should validate JSON scalar types and terminal
observation shapes explicitly.

## Decision gate

Do not launch multi-seed production solely because losses are finite. First
require both preflights to finish with checkpoints and event files, confirm L3P
rebuilds and uses its graph, inspect policy entropy for collapse, and compare
coverage against random-action behavior over the same 100k frames. Only then
freeze a production StudySpec and repeat print-only review.
