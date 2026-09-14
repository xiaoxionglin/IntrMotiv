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

## User-added F64 configurations

The user additionally requested the high-initial-sensitivity F64 candidates.
The stored-online preflights actually completed approximately 2M frames, with
terminal TV 0.071597 DDQN and 0.090323 HER; failed production entries have no
TV metrics. Add both waypoint decoder F64 modes at seed 99, fresh, using the
same CPU/RR1 contract. These are selected preflight candidates rather than the
top two current-production sustained scores. All four use the same new project.

Additional StudySpec hashes (schema v1, workflow 1.8.1):

- `controller_cpu_f64_ddqn.study.json`:
  `ae3b21126799528c5b8476e4c37891c32efbc78b1690e883f6f17bfeb6f3da0e`.
- `controller_cpu_f64_her.study.json`:
  `3ea18be8bace7b5eadd0a3cd388a5a9e8b4f697a5afc4dcc2a756a7bc8ba1b4a`.

39 focused runtime/workflow tests passed remotely before submission. Each
one-run canonical manifest receives print-only review and a submitted audit.

Submitted successfully with canonical submitted audits: 8061402 direct DDQN
seed 99; 8061403 direct HER seed 8; 8061404 waypoint DDQN seed 99; 8061405
waypoint HER seed 99. Initial queue check reports CPU partition pending,
`ReqNodeNotAvail, Reserved for maintenance`. Submission is verified; training
and W&B startup are not yet verified. Do not claim measured CPU throughput.

## All seeds and 64-decision cadence extension

User requested all three seeds and `controller_decisions_per_update=64` for
comparison. Add missing seeds to the four existing configurations (8 jobs),
and add cadence64 for all architectures/modes/seeds (12 jobs). The four queued
original jobs are preserved; the complete comparison has 24 jobs.

Batch budgets remain fixed: DDQN 2048 main; HER 1024 main plus up to 1024
auxiliary. Consequently cadence64 is total replay ratio 32 (HER at most 32),
not the old 256-example replay ratio4 arm. This interpretation was explicitly
communicated before submission. Target refresh is 96 updates at cadence64
versus 3 at cadence2048: both 6144 accepted decisions. Fresh SF batch and DG
schedule are unchanged. This arm may be much slower; it is a requested
replay-intensity comparison, not an efficiency promise.

New canonical specs are `controller_cpu_selected_ddqn_seeds`,
`controller_cpu_selected_her_seeds`, `controller_cpu_f64_ddqn_seeds`,
`controller_cpu_f64_her_seeds`, and `controller_cpu_cadence64` under
`hpc_runs/studies/`. Validation fingerprints (schema v1/workflow1.8.1) are in
`data/intrmotiv_full_system_controller_20260912/stored_state/cpu_expansion_validation_20260914.json`.
Group in W&B by `Hippo_n_feature`, `controller_her`, and
`controller_decisions_per_update`, aggregating `seed`. All use project
`SF_IntrMotiv_CPU2048`. Existing learner code and existing jobs are unchanged.

Reusable lesson: changing decision cadence with a fixed minibatch changes replay
ratio. Record both numbers and align target refresh in physical decisions so
that a cadence comparison does not also change target age by a factor of 32.

All 20 additional jobs submitted and passed `audit-submission --submitted`:
8061409–8061416 fill missing seeds at cadence2048; 8061417–8061428 cover all
12 cadence64 cells. Combined with original jobs8061402–8061405 there are 24
CPU jobs. Each StudySpec retains its canonical workspace jobs.tsv and audit
JSON under workspace analysis. No duplicate original-seed jobs were submitted.
