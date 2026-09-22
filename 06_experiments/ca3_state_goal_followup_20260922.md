# CA3 state-goal follow-up release

## Release state

The corrected four-cell CA3 state-goal factorial is in a fresh post-HER-batching release qualification on both CPU and NVIDIA L40S nodes. Production remains mechanically gated: qualification performance does not select cells, but correctness, exact reload, required telemetry, and usable sustained throughput must pass before either twelve-run production copy is submitted.

The scientific cells are fixed/EMA anchors crossed with dominant/unique-context candidate recognition. Each StudySpec supplies one flat, seed-independent `config.wandb_tags` value per cell.

## Source snapshots

- CPU and GPU release qualification: `70534498e1e0ed582989e8af77b3ec8fd2b4281e`.
- Published runtime branch: `codex/ca3-state-goal-followup-20260922` in `xiaoxionglin/SF_hipposlam`.
- CPU checkout: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ca3_state_goal_followup_release_70534498_cpu_20260922`.
- GPU checkout: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ca3_state_goal_followup_release_70534498_gpu_20260922`.

The earlier jobs `8144552`–`8144555`, `8145179`–`8145182`, and `8145197`–`8145200` remain diagnostic evidence only. The latter two waves predate batched contextual HER comparison and cannot authorize production.

## CPU qualification

- StudySpec: `hpc_runs/studies/ca3_state_goal_followup_20260922_preflight_v3.study.json` in the pinned runtime branch.
- Study SHA-256: `9f10d6f26b4d4aba2ecb0e6f58c7bb35264d5e1ffc6b14d82f5ec1aed14cc323`.
- Jobs: `8147624`–`8147627`.
- Allocation: 40 CPUs, 128 GiB, CPU partition, 30 hours.
- Submission state: all four jobs entered `RUNNING`; the sustained post-HER result is pending.

The release batches both online contextual recognition and every contextual HER start/goal action-probe comparison while preserving exact zero/one/multiple-match semantics, candidate/rejection counters, RNG selection, and exact-goal behavior.

## GPU qualification

- StudySpec: `hpc_runs/studies/ca3_state_goal_followup_20260922_gpu_preflight_v2.study.json` in the pinned runtime branch.
- Study SHA-256: `a75a47607a87b2653ea2f641d9ab88b358d10c5d8632e13e47b5f4acb76d5f5f`.
- Jobs: `8147628`–`8147631`.
- Allocation: one L40S GPU, 40 CPUs, 128 GiB, 30 hours.
- Sample Factory geometry: 32 workers, 8 environments per worker, 8 worker splits, one epoch, batch size 2,048, two minibatches, rollout/recurrence 64.
- State at launch: all four jobs are queued for scheduler priority.

The 32-by-8 geometry comes from historical high-throughput NEMO GPU runs and is requalified here on the exact current model. The approximately 19,347 aggregate FPS measurement came from a separate four-run G500 workstation test after the GPU-core synchronization fix, not from NEMO L40S. Its two-epoch setting is not copied because that changes optimizer exposure. The release branch contains the same GPU-core fixes, but production depends on the fresh L40S measurement.

## Production definitions

- [CPU production](../hpc_runs/studies/ca3_state_goal_followup_20260922_production.study.json): twelve fresh 300M-frame runs, 40 CPUs, 128 GiB, CPU partition, 96-hour waves.
- [GPU production](../hpc_runs/studies/ca3_state_goal_followup_20260922_gpu_production.study.json): twelve fresh 300M-frame runs, one L40S GPU, 40 CPUs, 128 GiB, 48-hour waves.
- GPU production Study SHA-256: `b371dff60f4a1722a9eb506d2cc6798a89243cbda43f93eabdb373155ba5f766`.

CPU and GPU copies use separate study IDs, run prefixes, output roots, W&B groups, Slurm manifests, and pinned source checkouts. Qualification checkpoints are never continued into production.

## Remaining gate

For each device, all four jobs must reach 2M frames, restore exact model/optimizer/replay/calibration/anchor state, pass the canonical follow-up audit, exercise the required contextual paths, emit finite declared telemetry, and complete privileged offline diagnostics without feeding coordinates into training. After a device-specific gate passes, its twelve production jobs are submitted immediately and audited.

## Reusable lesson

The authoritative throughput comparison is an identical fresh StudySpec wave after warm-up. Short W&B FPS windows and results from another GPU host are insufficient. Reuse sampler geometry independently from scientific optimizer settings: environments per worker and worker splits are execution geometry, whereas epochs alter learning exposure. The canonical launcher also defaults to print-only unless `--submit` is passed explicitly; audits must require nonempty job IDs and `submitted_complete=true`, not merely valid generated scripts.
