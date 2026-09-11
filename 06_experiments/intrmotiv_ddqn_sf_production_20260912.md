# Native SF DDQN/HER production batch — 2026-09-12

Submitted six 5M-frame runs using the qualified native asynchronous Sample Factory backend: DDQN and DDQN+HER, each with seeds 8, 99 and 123. This continues the frozen-reference local-control diagnostic; adaptive DG, graph/manager learning and independent commanded-control qualification are not inferred from throughput results.

## Preflight and logging qualification

The final release preflight was re-audited before submission. Both arms passed: 720 optimizer updates, 184,320 valid TD positions, seven target copies, 32 excluded invalid finals and unchanged frozen representations. Native SF achieved 3,326.88 learner-active FPS versus 1,693.51 for the batched standalone runner. See [integration report](intrmotiv_ddqn_sample_factory_integration_20260911.md).

A logging issue was caught before promotion: copying the parent configuration retained its `wandb_unique_id`, group and tags. Fresh child configurations now clear these fields before calling SF's existing W&B initializer. The parent configuration remains unchanged, and SF generates a separate run identity for each child. No learner, replay, representation or SF core code changed for this repair. All 75 focused tests passed locally and on NEMO2, including a regression test for inherited W&B identity.

Compute-node smoke job **8057193** completed successfully in 26 seconds using the actual SF multiprocess runtime and W&B integration. The W&B API confirmed its finished state and uploaded `ddqn/*` metrics and `train/env_steps`: 4,096 frames, 14 updates and 896 TD positions. [Cloud verification](data/intrmotiv_ddqn_sf_production_20260912/wandb_smoke_verification.json) records the run URL and uploaded values. The production jobs enable W&B in the `IntrMotiv` project.

## Canonical study and submission

- Study: `hpc_runs/studies/intrmotiv_ddqn_sf_production.study.json`.
- Schema: `intrmotiv/study/v1`; workflow: `1.8.0`.
- SHA-256: `4eb4cf1fcf4a1bf1d73ebb647c6cb2f43188e5109e12708eb859d3861f4527af`.
- Print-only review: `20260911T224003Z` (UTC).
- Submission: `20260911T224131Z` (UTC), 2026-09-12 00:41 Berlin.
- Six ordinary Slurm jobs, each requesting 40 CPUs, 80 GB, no GPUs, two-hour cap.
- Total requested training: 30M frames. No earlier jobs were cancelled or modified.

The parent remains `DGC_DIRECT_WORKER_F16_S99` at 25,001,984 source frames. Child counters start at zero. Each run uses 32 environments, 16 rollout workers, two splits, 32-step rollouts, SF batches of 1,024 transitions, eight inference threads and one learner thread. Replay/HER semantics are unchanged: registry 1/4/11, first-arrival horizon 64, 16,384 accepted-decision warmup, one update per 64 accepted decisions, 256 TD positions per update and target copies every 100 updates. Native SF preserves update debt and physical stream order.

The source checkout is `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ddqn_sf_20260912`. Training data are under `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_ddqn_sf_production_20260912/`. Submission records and logs are under the same training root at `_slurm/intrmotiv_ddqn_sf_production_20260912/20260911T224131Z/`. Caches, temporary files, TensorBoard/W&B data and checkpoints resolve into the allocated workspace. The [submission audit](data/intrmotiv_ddqn_sf_production_20260912/submission_audit.json) verifies all six commands and workspace paths; the [jobs manifest](data/intrmotiv_ddqn_sf_production_20260912/jobs.tsv) is authoritative for run/job mapping.

## Initial training verification

All six runs passed warmup and are performing optimizer updates. Each has a distinct running W&B identity, with nonzero frames and updates independently confirmed through the cloud API. Cloud summaries trail local metrics because uploads are periodic. This is a startup snapshot, not a completed scientific evaluation.

| Run | Slurm job | Frames | Updates | W&B |
|---|---:|---:|---:|---|
| DQSFPROD_DDQN_S8 | 8057197 | 638,848 | 2,238 | [Live run](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQSFPROD_DDQN_S8_20260912_004204_506161) |
| DQSFPROD_DDQN_S99 | 8057198 | 356,224 | 1,135 | [Live run](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQSFPROD_DDQN_S99_20260912_004221_165965) |
| DQSFPROD_DDQN_S123 | 8057199 | 511,872 | 1,742 | [Live run](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQSFPROD_DDQN_S123_20260912_004222_072840) |
| DQSFPROD_DDQN_HER_S8 | 8057200 | 606,080 | 2,110 | [Live run](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQSFPROD_DDQN_HER_S8_20260912_004220_462382) |
| DQSFPROD_DDQN_HER_S99 | 8057201 | 610,176 | 2,126 | [Live run](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQSFPROD_DDQN_HER_S99_20260912_004220_328488) |
| DQSFPROD_DDQN_HER_S123 | 8057202 | 458,624 | 1,534 | [Live run](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/IntrMotiv/runs/00_DQSFPROD_DDQN_HER_S123_20260912_004221_027203) |

[Startup evidence](data/intrmotiv_ddqn_sf_production_20260912/startup_verification.json).

## Reusable lessons

Before enabling logging on transferred models, distinguish inherited scientific configuration from run identity. Use the framework's initializer to generate new identities; do not reuse parent W&B IDs. Verify an actual compute-node upload through the W&B API, rather than treating SDK installation or local log creation as success. Reuse the canonical runtime and submission audits for promotion. Keep the scientific evaluation requirement separate from execution qualification, and never silently change learning budgets to obtain higher throughput.
