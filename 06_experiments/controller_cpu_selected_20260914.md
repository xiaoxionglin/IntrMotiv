# Selected CPU comparison — 14 September 2026

Selection uses the current production W&B group only, ranking mean action
probability TV over each run's latest 5M frames. Unequal horizons and selection
on observed performance mean this is exploratory, not an unbiased algorithm
comparison. Historical preflights are excluded because their implementation and
horizon differ. The selected runs are direct F16 DDQN seed 99 (0.036651) and
DDQN+HER seed 8 (0.036271). Raw rankings are archived in
`data/intrmotiv_full_system_controller_20260912/stored_state/cpu_selection_ranking_20260914.json`.

Fresh reruns retain the fixed ImageNet trunk and original DG/graph/telemetry
objectives. New W&B project: `SF_IntrMotiv_CPU2048`, group
`intrmotiv_cpu_selected_20260914`. Existing runs are untouched. No learner code
changes: device=cpu, fresh SF batch=2048, replay cadence=2048 accepted decisions,
DDQN main=2048, HER main=1024 plus up to 1024 auxiliary. Separate mean losses
and coefficient 1 remain. Thus HER total is capped at 1:1 with eligibility
failures unfilled, and main exposure is lower than plain DDQN. Target refresh
is every 3 updates (6144 accepted decisions). This retains the already-declared
RR1 efficiency comparison, rather than isolating hardware alone.

Canonical one-run specs preserve the actual selected seeds without inventing
an additional cross-product:

- `hpc_runs/studies/controller_cpu_selected_ddqn.study.json`, SHA
  `892b70083a745187bc13ec017b449babdffdabf0234e3c1ac87755c000fd4142`.
- `hpc_runs/studies/controller_cpu_selected_her.study.json`, SHA
  `607d6de7c44f0673fa51fa1ef737a02b6d804aa36d9e7b97c2841583b76acdd9`.

Both schema `intrmotiv/study/v1`, workflow 1.8.1. Runtime clone:
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_cpu_selected_20260914`.
CPU partition, zero GPUs, 40 CPUs, 128GiB RAM, 96h allocation (verified partition
maximum). Horizon 300M frames; checkpoint resumes may still be necessary.
All bulk output remains under `/work/classic/fr_xl1014-train`.

CPU throughput must be measured, not assumed equal to PPO or past CPU studies.
The existing launcher already supports CPU jobs; reuse it rather than adding
another execution backend. The existing stored replay path batches the decoder
and bypasses the reconstruction path's CPU microbatch limit. Monitoring stays
paused unless explicitly requested otherwise.
