# CA3 state-goal follow-up release

## Release state

The corrected four-cell CA3 state-goal factorial is in fresh release qualification on both CPU and NVIDIA L40S nodes. Production remains mechanically gated: qualification performance does not select cells, but correctness, exact reload, required telemetry, and usable throughput must pass before either twelve-run production copy is submitted.

The scientific cells are fixed/EMA anchors crossed with dominant/unique-context candidate recognition. Each StudySpec supplies one flat, seed-independent `config.wandb_tags` value per cell.

## Source snapshots

- Optimized CPU qualification: `36451e7a2ae6f5a382c1f904990719d5b40ac87e`.
- GPU qualification and production definitions: `774942699e3f62ab63d2d17ecab6507d3caddfc6`.
- Published runtime branch: `codex/ca3-state-goal-followup-20260922` in `xiaoxionglin/SF_hipposlam`.
- CPU checkout: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ca3_state_goal_followup_optimized_20260922`.
- GPU checkout: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ca3_state_goal_followup_gpu_20260922`.

The earlier jobs `8144552`–`8144555` remain diagnostic evidence only. They used the unoptimized serial contextual-recognition path and cannot authorize production.

## CPU qualification

- StudySpec: [optimized CPU preflight](../hpc_runs/studies/ca3_state_goal_followup_20260922_preflight_v2.study.json).
- Study SHA-256: `3bb4b95e5b0b58070fe541c480b5a8e09fd863711d3a6376792d7bcabc63e3de`.
- Jobs: `8145179`–`8145182`.
- Allocation: 40 CPUs, 128 GiB, CPU partition, 30 hours.
- Initial sustained result: all four cells reached roughly 676–769 five-minute FPS around 150k–180k frames. Unique-context recognition now matches dominant recognition instead of the earlier 110–170 FPS collapse.

The runtime fix batches current-state signatures, selectable-anchor signatures, and similarity evaluation while preserving exact zero/one/multiple-match semantics and telemetry counters.

## GPU qualification

- StudySpec: [L40S preflight](../hpc_runs/studies/ca3_state_goal_followup_20260922_gpu_preflight.study.json).
- Study SHA-256: `f9c5605ca98af63539144738db3566ec8b11503fbfbe08be82a66fac6248f22d`.
- Jobs: `8145197`–`8145200`.
- Allocation: one L40S GPU, 40 CPUs, 128 GiB, 30 hours.
- State at launch: queued for scheduler priority.

Print-only review exposed an auditability gap: GPU resources were added only to the eventual `sbatch` command, not rendered into the saved script. The shared launcher/template now renders `#SBATCH --gres=gpu:1`, keeps the actual submission request, omits `gpu:0` for CPU submissions, and has a focused regression test. This did not change the scientific configuration.

## Production definitions

- [CPU production](../hpc_runs/studies/ca3_state_goal_followup_20260922_production.study.json): twelve fresh 300M-frame runs, 40 CPUs, 128 GiB, CPU partition, 96-hour waves.
- [GPU production](../hpc_runs/studies/ca3_state_goal_followup_20260922_gpu_production.study.json): twelve fresh 300M-frame runs, one L40S GPU, 40 CPUs, 128 GiB, 48-hour waves.
- GPU production Study SHA-256: `92312b7153c5e09e955a204d91b66cacd417ee99053be6c5313774d8d97d4cc5`.

CPU and GPU copies use separate study IDs, run prefixes, output roots, W&B groups, Slurm manifests, and pinned source checkouts. Qualification checkpoints are never continued into production.

## Remaining gate

For each device, all four jobs must reach 2M frames, restore exact model/optimizer/replay/calibration/anchor state, pass the canonical follow-up audit, exercise the required contextual paths, emit finite declared telemetry, and complete privileged offline diagnostics without feeding coordinates into training. After a device-specific gate passes, its twelve production jobs are submitted immediately and audited.

## Reusable lesson

The authoritative throughput comparison is an identical fresh StudySpec wave after warm-up. The serial unique-context path looked like an environmental slowdown but was isolated by comparing cells on the same CPU profile. GPU resource intent must also be visible in saved print-only artifacts; relying on an unrecorded launcher-time option weakens pre-submission review even when Slurm ultimately receives the request.
