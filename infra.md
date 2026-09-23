# Infrastructure Improvements

Track concrete infrastructure findings here so future threads can reuse
evidence and coordinate improvements. Consult this file before related work;
keep implementation guidance in the canonical workflow documents linked below.

## Open Improvements

### Study factors were not available as a flat W&B grouping key

- **Evidence:** The CA3 predictive active-goal StudySpec declared one
  seven-level `architecture` factor, but Sample Factory received only the
  factor's scientific arguments. W&B therefore required nested grouping over
  six configuration fields, while the canonical seed-independent condition
  existed only in the StudySpec and run name.
- **Impact:** Dashboard grouping was cumbersome and could obscure the intended
  seven-arm comparison, especially when several scientific parameters jointly
  define one architecture.
- **Improvement/status:** Workflow 1.11.0 emits metadata-only `study_id`,
  `study_condition`, and `study_base` configuration fields. New dashboards can
  group once by `study_condition` and pair by `seed`. It also emits exactly one
  `wandb_tags` value equal to the condition; Sample Factory duplicates this as
  `config.wandb_tags`, providing the requested flat-list grouping shortcut.
  The behavior is default for studies declaring 1.11 or later and configurable with
  `training.emit_tracking_identity`. Existing submitted specs remain unchanged
  to preserve their hashes and audited commands.
- **Acceptance:** Schema and expansion tests prove the versioned default and
  explicit override, identical condition identity across seeds, rejection of
  non-boolean configuration, and no command changes for existing studies. The
  runtime parser accepts the three fields. Local focused tests pass; synchronize
  and rerun them in an isolated NEMO2 checkout before using workflow 1.11 for a
  cluster launch.
- **Lesson:** Analysis identity should travel with the launched run rather than
  be reconstructed from scientific parameters or parsed back out of names.

### Fixed geometry lacked accessible-area telemetry and release-scoped assets

- **Evidence:** Historical coverage divides by decisions but not accessible floor
  area. Online and offline maps use different axis conventions. Native corridor
  qualification also exposed Q3Map buffer overflow with full-hash filenames and
  non-identical RGB after same-instance resets despite identical spawn poses.
- **Impact:** Wall-density comparisons could reward reduced floor area, transpose
  wall masks, collide in native assets, or invalidate causal reset comparisons.
- **Improvement:** Shared geometry archive/verification and normalized episode
  coverage, explicit axis adapters, short native map names with full hash checks,
  isolated runfiles, and fresh-engine prefix replay with exact state checks. See the
  [canonical workflow](04_implementation/standardized_study_workflow.md).
- **Status:** Implemented locally; native nine-map qualification passes. Source
  staged remotely. The old `train` allocation reported zero free bytes; a new
  `corridor-geometry` allocation reports 4.6T available and passed write/fsync.
  Capacity readings must be checked on the target workspace, not interpreted
  as filesystem-wide exhaustion. The IntrMotiv launcher and project policy now
  use this allocation by default; the CA3 predictive active-goal release copied
  its required Python runtime there and passed zero-leakage print-only audits.
  A first CA3 qualification wave nevertheless reached its terminal snapshot
  with an omitted online-spatial override and inherited the old full workspace.
  The canonical submission audit now requires explicit spatial output and
  workspace roots whenever online spatial telemetry is enabled, validates both
  against the StudySpec workspace, and has a regression test for this leak. The
  subsequent offline gate found the ordinary-job telemetry submitter still had
  a fixed old-workspace guard; it now accepts and propagates an explicit
  workspace root to the worker. A bounded contextual row also caught worker-only
  spatial diagnostics being scoped under contextual output; the evaluator now
  computes them only when worker activity exists. Run checkpoint-heavy release
  audits as compute jobs rather than loading multi-gigabyte checkpoints on the
  login node.
  Native qualification passed (8109095). First training attempts exposed the
  previously documented long-TMPDIR shared-memory failure; stopped outputs are
  archived, and the template now uses short paths plus a fail-fast socket probe.
  The Python evaluators now honor the worker's workspace-root override. Early
  real-model gates also exposed unused-seed cache interference and missing PPO
  terminal pose telemetry; evaluation bypasses training seed selection and the
  shared pose reader uses the certified terminal binding. All nine 2M runs,
  nine real learner reloads, three frozen evaluator gates and all 18 spatial
  snapshots passed. All three standard field jobs and bounded matched-command probes also passed;
  the 27-run production batch completed 100M frames per run. Full production
  evaluation is launched. The command worker was corrected to honor the isolated
  source, terminal binding and short workspace TMPDIR before that launch. Full
  intervention production then exposed a native lifecycle failure: 36 rows
  aborted with buffer-overflow exit 134 after repeatedly reconstructing DMLab.
  A one-engine seeded-reset repair was rejected by the exact-state guard on all
  36 replacement rows (8127269--8127304) and bounded qualification 8127266:
  same-seed resets were not observation-identical. The authoritative evaluator
  was restored. The remaining fix is bounded fresh-engine process shards with
  an audited merge; exact matching must remain mandatory.
  Evidence is tracked in the
  [corridor record](06_experiments/corridor_geometry_20260919.md).
- **Acceptance:** Nine maps connected and nested, common spawns, exact replay,
  zero reward and 120-second timeout, no geometry/pose model input, compatible
  telemetry tests, nine 2M preflights and checkpoint-bound reload/evaluator gates.
- **Lesson:** Compile actual Lua maps before scheduling training; cached map
  summaries alone cannot expose engine limits or reset-history dependence.
  Bind reload proof to immutable checkpoint hashes; archive shortened spatial
  schedules before production reuses run identities. Apply short TMPDIRs to
  evaluation workers as well as training. Matched-command qualification also
  needs an explicit source-count bound: fresh-engine construction can dominate
  despite an 8k-decision limit. Use a declared one-source/five-repeat qualification
  panel; leave production intervention budgets unchanged. Make the exact replay
  comparison authoritative. Neither many native reconstructions in one process
  nor same-process seeded resets are safe; isolate bounded fresh-engine work in
  short-lived processes.


### Atlas figure recipes diverged between batches — standardized and verified

- **Evidence:** CPU2048's private renderer replaced Navigation8's colored
  trajectory segments, markers, and headings with a low-opacity blue overlay.
  The user requested the previous style and a common atlas standard.
- **Impact:** Batch comparisons differed in readability and encoding; duplicated
  plotting code let field scaling and silent-unit display diverge too.
- **Improvement/status:** Shared workflow 1.9.0 recipes with style identifier
  `segmented-atlas/v1`; CPU2048 adapter delegates all four atlas figure types.
  38 tests pass locally and in the isolated NEMO2 analysis copy. All 24 entries
  were regenerated after the user renewed OTP authentication. The shared
  training checkout remains unchanged.
- **Acceptance:** Remote tests pass; all 24 CPU2048 entries regenerated with
  unchanged run/checkpoint selection; representative F16/F64 renders visually
  checked; provenance records the new figure style. Numerical CSV hashes are
  unchanged; the existing figure URLs are preserved.
- **Lesson:** Reuse the renderer, not just the metric calculation. Keep the
  [figure contract in the canonical guide](04_implementation/standardized_study_workflow.md)
  authoritative, and use batched artists to retain clear segmented trajectories.
  Stage both canonical test StudySpecs, not only the main reference; a missing
  secondary fixture caused one avoidable failed verification run.

### Spatial collector rejected declared late milestones — fixed in isolated analysis

- **Evidence:** CPU2048 collection on September 17 rejected a saved 150M NPZ;
  `expected_spatial_targets` filtered declared targets through a historical
  5M–100M list. This also made completeness checks ignore later expected targets.
- **Impact:** Valid long-running studies failed collection; a successful early
  snapshot collection did not establish completeness at the declared horizon.
- **Improvement/status:** Workflow 1.8.2 respects declared positive unique targets.
  33 desktop tests and three NEMO2 regressions pass. Deployed only in the
  CPU2048 workspace analysis copy; shared runtime deployment remains pending.
- **Acceptance:** 150M/300M declarations survive discovery; invalid/duplicate
  targets fail; original default behavior stays available when no targets exist.
- **Lesson:** Check target declarations and collector validation before inferring
  a stopped run from available analysis artifacts. See the
  [canonical guide](04_implementation/standardized_study_workflow.md).

### NEMO source consolidation — complete; live-release retirement pending

- **Evidence:** September 15 inventory found the canonical `SF_hipposlam` plus
  36 alternatives, many with uncommitted changes or no Git metadata. Three
  alternatives support 44 running/queued jobs; the remaining 33 are retirement
  candidates at inventory. There are 373 distinct differing source-file hashes.
- **Impact:** Fixes and qualified controller modules were missing from the main
  checkout, while repeated copies made source selection and cleanup ambiguous.
- **Improvement:** Integrate the latest qualified controller with newer shared
  Git history, canonical workflow and baseline modules; preserve superseded
  prototypes in a complete workspace archive. Use one main development checkout
  and retain separate releases only while jobs require their source.
- **Status:** All 37 original checkouts archived and checksum-verified under
  `/work/classic/fr_xl1014-train/IntrMotiv/source_retirement_20260915/`.
  Published branch `codex/nemo-consolidation-20260915` passes pinned hooks,
  583 desktop tests, and 573 NEMO tests (10 CUDA skips). All 33 inactive folders
  retired after a fresh checksum comparison and job checks. The three live
  release folders remain; their source hashes were verified unchanged. Shared
  publication-worktree metadata is locked against pruning. Desktop depth work
  is preserved on `codex/local-depth-preserved-20260915`; both canonical
  checkouts use the consolidated branch.
- **Published result:** [GitHub branch](https://github.com/xiaoxionglin/SF_hipposlam/tree/codex/nemo-consolidation-20260915),
  final documentation revision `c002faff2c6832f9b0ce63bc6401f95e9cb4718d`.
  The [canonical procedure](https://github.com/xiaoxionglin/SF_hipposlam/blob/codex/nemo-consolidation-20260915/docs/intrmotiv_source_consolidation.md)
  records lineage selection, archive contents, verification, and retirement.
- **Remaining work:** Retire `controller_cpu_selected_20260914`,
  `controller_rr1_20260913`, and `controller_stored_production_release_20260912`
  only after their running/queued jobs finish and a fresh dependency audit.
  Their existence is currently required, rather than redundant.
- **September 22 workflow-skew finding:** The consolidated training checkout
  still carries workflow 1.8.1 while the vault's canonical analysis package is
  1.10.1. New CA3 readout StudySpecs therefore live with the canonical package,
  explicitly declare 1.10.1, and validate there; copying a study into the
  training checkout is not a safe substitute for synchronizing the whole
  versioned package. Before NEMO2 submission, synchronize
  `hpc_runs/intrmotiv_study/`, rerun its focused tests, validate both study
  hashes, and perform the ordinary print-only/audit review.
- **September 22 CA3 batch release finding:** The first valid CA3 StudySpecs
  still inherited unreachable preflight telemetry targets, an old W&B group,
  and analysis grouping that pooled all seven architecture cells. The runtime
  also computed readout losses without routing them to canonical scalar tags.
  The release now requires generated-config parsing plus a seven-arm 2M runtime
  audit covering exact reload, scalar presence/finite values, calibration-path
  execution, and manifest-driven alias telemetry before any 21-run production
  submission. Acceptance is an audited artifact bound to the deployed commit
  and StudySpec hashes; relative qualification performance never prunes cells.
- **Acceptance criteria:** Matching published/canonical commit; focused tests
  pass on NEMO; inactive folders removed only after a fresh job/dependency check
  and checksum comparison; live source folders and all training data preserved.
- **Reusable lesson:** File hashes are authoritative for dirty snapshots;
  commits alone are insufficient. Compare ASTs to identify formatting-only
  differences, merge against the organized parent, and reuse the pinned hook
  environment at `/tmp/intrmotiv-precommit-env` on this desktop. Preserve shared
  Git worktree metadata while any retained copies refer to it. Neither shell
  had GitHub push credentials; the connected app published the tree, whose Git
  tree SHA was verified identical to the tested source before updating refs.

### G500 training qualification and resource-aware direct execution — in progress

- **Evidence:** Actual GPU training rejected legacy W&B `start_method` settings,
  then hit a CPU/CUDA sentinel mismatch in reward progression. The runtime's
  wall-clock limit reads a counter that is never advanced, and the existing
  process launcher returns success even when a child fails. See the
  [G500 DG qualification record](06_experiments/dg_neighborhood_g500_20260914.md).
- **Impact:** Environment smokes and apparent idle GPUs do not establish
  training correctness, useful concurrency, or reliable termination.
- **Improvement:** Isolated W&B/device fixes; StudySpec-derived fixed-frame
  profiler with five-second resource samples, real frame counters, process
  identities and failure detection. Reuse the
  [canonical study workflow](04_implementation/standardized_study_workflow.md)
  for the scientific matrix. `intrmotiv_study.direct` now preserves real child
  failure, resource admission, duplicate-start exclusion and manifest provenance.
- **Status:** 38 desktop regression tests passed (one CUDA skip), 9 G500
  CPU/CUDA tests passed, 4 direct-queue and 3 profiler tests passed, and 2
  metric tests passed. All four scientific preflights passed shared-panel and exact model/optimizer
  gates; 12 production runs launched with four GPU slots. Six remote evaluation/
  transition tests pass. Monitoring and production checkpoint evaluation remain active.
- **Acceptance criteria:** Qualified worker/batch/concurrency measurements;
  accurate child failure propagation; no duplicate starts or source mutation
  during active jobs; online W&B; matched four-arm preflights pass before the
  already-authorized production queue starts.
- **September 15 revision:** User stopped the initial production to qualify larger
  sampler geometry. Extend the existing profiler with environments/worker and
  paired four-arm selection rather than a duplicate benchmark. Five focused
  tests pass. The earlier worker comparison also changed batch size and was
  confounded; the new search fixes all learning settings and measures concurrent
  throughput plus resource peaks before a fresh scientific gate and production.
- **Measured bottleneck (September 15):** Completed `w32_e8_ep1` SELF-arm
  profile reports 739.09 s training, of which 730.51 s is
  `bptt_forward_core`; visual head 1.66 s, encoder losses 3.86 s, optimizer
  update 0.98 s. Inference forward is 706.75/709.15 s of policy handling.
  Core code iterates sequence steps and invokes HRL state updates. Profile
  that path before adding more samplers; these wall timers do not yet isolate
  individual CUDA kernels or synchronization. Source remains unchanged.
- **GPU-core resolution (September 15):** GitHub core commit `26827509` reduced
  G500 batch-32 manager latency from 547.50 to 6.30 ms/step. Its new CUDA
  profiler test exposed two hidden scalar reads from diagnostic slice clearing;
  local commit `48e6512f` replaces those assignments with `zero_()`. The combined
  remote suite passes 29 tests. Four-arm end-to-end aggregate throughput rose
  from 1,126 to about 19,347 frames/s under 32×8, splits 8 and two epochs.
  Very fast probes can finish before a fixed warmup window; derive fallback
  throughput only from distinct completed-batch counters over at least 20 s.
- **Mature-replay resolution (September 23):** The CA3 state-goal batch exposed
  two additional event-density-dependent synchronization paths that short
  probes missed. First, matched encoder credit read CUDA scalars once per DG
  event; vectorized interval validation and reward scatter reduced that phase
  from 4--7 s to roughly 0.02--0.04 s. Second, stored main replay checked the
  contextual selectable mask and anchor generation separately for every
  commanded row. Subphase telemetry isolated example construction at 4.86 s
  of a 5.20 s main update at 1.25M frames. A transaction-level snapshot of the
  two 64-element authority vectors reduced example construction to 0.03--0.06
  s and main replay to 0.30--0.43 s without changing eligibility semantics.
  The matched four-cell gate reached 17.75k aggregate FPS in a fully aligned
  mature 60-second window, 9.8% above the prior 16.16k reference; the sum of
  mature per-run medians was 16.11k, within 0.3% of the reference. The reusable rule is to run
  throughput gates beyond activation and replay maturity, emit subphase timing,
  and treat every Python boolean or integer conversion from a CUDA tensor in a
  per-event or per-row loop as a synchronization defect. Preserve authority by
  snapshotting small immutable vectors once per transaction; do not weaken the
  active-mask or generation checks.
- **Release-gate lesson (September 23):** Exact restore of a 4 GB checkpoint
  with 200k replay rows stored as many Python objects remained CPU-bound for
  about 24 minutes even though training throughput was healthy. Run independent
  reload certificates concurrently and start them as soon as each qualification
  cell finishes. A future checkpoint-format revision should store replay in
  columnar tensor blocks while preserving the exact-restore contract. Direct
  evaluator invocations must also pin the qualified source root in `PYTHONPATH`;
  otherwise executing `evaluation/place_fields.py` by path can import an older
  installed IntrMotiv package and reconstruct the wrong model. Evaluator workers
  need existing, isolated `TMPDIR` directories because DMLab does not create the
  configured parent itself. Release audits must also declare whether certified
  terminal-successor transport is part of the study contract: the state-goal
  follow-up intentionally rejects contextual terminal HER without raw successor
  CA3, while older controller studies retain the strict transport requirement.
- **Core device fix (September 15):** Local runtime now batches the non-probing
  topological manager and shared landmark bookkeeping on the state device,
  removing per-stream scalar reads from the active study's path. 106 focused
  tests pass; 10 CUDA tests are skipped on the desktop. Reference comparisons
  cover 192 configurations and nine edge-probing cases. See the
  [implementation, fixtures, and timing record](04_implementation/topological_manager_device_20260915.md).
  G500 source transfer was blocked by automatic approval review pending explicit
  export authorization; CUDA timing and end-to-end qualification remain open.
  Acceptance: device/oracle tests pass on G500 and a fresh isolated benchmark
  measures improvement before qualifying the next production source snapshot.
- **Lesson:** Select by completed-update throughput and full process lifecycle,
  not GPU memory alone or short-window FPS. Reuse compact resource summaries
  instead of loading checkpoints repeatedly for monitoring. Validate complete
  scientific configs, including recruitment-dependent flags and telemetry
  interval/max consistency and full-window warmup, before launch. A 100k-
  observation map window needs 800k frames at frameskip8; shorter preflights
  should use shared-panel checkpoint maps. G500 lacks `rg`; use `grep` there.

### Portable runtime bootstrap — verified; patcher cleanup proposed

- **Evidence:** G500 setup found a nonportable absolute DMLab wheel path in the
  runtime `requirements.txt`, plus a conda-only guard in the DMLab asset patcher.
  The desktop environment also reports torch/torchvision versions that are not
  an official matched pair. See the [setup record](04_implementation/g500_environment.md).
- **Impact:** Blind environment copying or requirements installation is not a
  reliable way to prepare a second GPU host.
- **Proposed improvement:** Reuse `setup.py`, the existing DMLab wheel and asset
  patcher, and a matched GPU package pair; record the verified environment and
  make the asset patcher accept an explicitly selected Python environment.
- **Acceptance criteria:** `pip check`, GPU/ResNet inference, custom DMLab
  reset/step, and focused runtime tests pass in an isolated target environment.
- **Status:** G500 environment complete after explicit source-transfer approval.
  All listed acceptance checks passed, including 68 focused tests. The existing
  patcher was reused with a command-scoped `CONDA_PREFIX`; removing its conda-only
  guard remains a proposed runtime-source cleanup.
- **Outcome:** A fresh matched CUDA environment and user-local Ubuntu SDL2 library
  worked. `ldd` isolated the only missing native dependency. Reuse the recorded
  activation helper, resolved package manifest and smoke script next time.

### Bounded progress and recovery for canonical online collection — open

- **Evidence:** On September 16, the Navigation8 screen's canonical
  `collect-spatial --require-complete --include-details` completed for all 54
  cached snapshots promptly. In contrast, its separate `collect-online
  --latest-common` process remained active without output after the initial
  connection returned; its destination directory remained empty. This repeats
  the long single-CPU collection behavior recorded in the standardized workflow
  after the September 11 DG-capacity scan.
- **Impact:** A complete spatial interim analysis can be delivered, but aligned
  TensorBoard metrics cannot be safely treated as available or repeatedly
  retried. Future batch reports waste remote I/O and leave ambiguous background
  collectors.
- **Proposed improvement:** Add bounded progress records and a resumable
  per-run cache with event-file provenance to the canonical collector, retaining
  exact latest-common semantics. Reuse the documented process backend only after
  its NEMO2 benchmark validates the progress/recovery contract.
- **Partial implementation (September 21):** Workflow 1.10.1 exports all selected
  scalar histories and source file metadata during the existing scan, and accepts
  an execution-only process-backend override. The corridor batch has 6.8 GB of
  events; the 27-run process collection completed in 10m22s with per-run
  progress and exports.
  Desktop and NEMO2 focused suites each pass 38 tests, including exact export
  contents and unchanged StudySpec settings. Plotting reuses CSVs instead of
  rescanning events. Automatic cache validation/reuse after interruption remains
  open; these exports are not yet a resumable loader cache. See the
  [canonical guide](04_implementation/standardized_study_workflow.md).
- **Acceptance criteria:** A 18-run latest-common scan writes an observable
  per-run progress record, terminates or emits a recoverable partial state under
  a declared time bound, and a resumed collection does not reread already
  validated event histories.

### Slurm GPU resources must be visible in print-only artifacts — completed locally, 2026-09-22

- **Evidence:** The CA3 follow-up L40S print-only scripts omitted a GPU directive even
  though the launcher added `--gres=gpu:1` only when it later invoked `sbatch`.
- **Impact:** The saved script and ordinary print-only review could not prove that a
  GPU was requested; a submission audit checked commands and workspace paths but not
  this scheduler-side resource.
- **Improvement:** The shared launcher now exposes a conditional `GPU_DIRECTIVE`, the
  canonical NEMO2 template renders it, and CPU submission no longer emits
  `--gres=gpu:0`. A focused launcher test covers the rendered directive.
- **Status and outcome:** Nineteen focused launcher, StudySpec, and CA3 runtime tests
  pass remotely. The regenerated L40S scripts contain `#SBATCH --gres=gpu:1`, and
  jobs `8145197`–`8145200` were submitted with the same explicit CLI request.
- **Acceptance criteria:** Print-only and submitted artifacts agree on GPU count;
  CPU scripts omit GPU resources; the StudySpec submission audit passes.

### Canonical launcher submission mode must be explicit — documented, 2026-09-22

- **Evidence:** Calling `launch_nemo2.sh` without a mode generated valid scripts but
  submitted no jobs because the wrapper intentionally defaults to `--print-only`.
  The resulting audit had empty job IDs, `generated` status, and
  `submitted_complete=false`.
- **Impact:** A release operator can mistake a successful render/audit for a live
  submission, especially when the output directory is named `submitted`.
- **Improvement:** Always call the wrapper with `--submit` for the real launch and
  preserve print-only and submitted artifacts in distinct directories. Treat
  nonempty job IDs, `submitted` row status, and `submitted_complete=true` as the
  launch gate.
- **Status and outcome:** The CA3 follow-up release retained the mistaken generated
  directory as evidence, then submitted into `submitted_real`; CPU jobs
  `8147624`–`8147627` and GPU jobs `8147628`–`8147631` passed submission audit.
- **Acceptance criteria:** Future release checklists and automation prompts name the
  explicit mode and verify all three submission indicators.

For each finding, record:

- **Title and status:** proposed, in progress, blocked, or completed.
- **Evidence:** observed failure, duplication, bottleneck, or readability issue,
  with relevant file links or commands.
- **Impact:** how it affects users, future threads, reliability, or efficiency.
- **Proposed improvement:** the smallest reusable change, including existing
  modules or packages considered.
- **Acceptance criteria:** observable checks that demonstrate improvement.
- **Outcome:** changes made, verification results, and reusable lessons.

## Canonical References

- [Agent guidelines](AGENTS.md)
- [Standardized study workflow](04_implementation/standardized_study_workflow.md)
- [Reusable place-field telemetry](04_implementation/reusable_place_field_telemetry.md)

## Completed Improvements

### Explicit depth preprocessing contract — completed locally, 2026-09-14

- **Evidence:** `sample_factory/utils/normalize.py` applies fixed observation
  scaling even with `normalize_input=False`. The initial hard-coded inverse
  transform missed this and also changed historical configuration behavior.
- **Impact:** Depth units and old-policy inputs could change silently.
- **Improvement:** Shared `DepthEncoder` now defaults to legacy pass-through,
  with a single inverse switch and fixed gain; inverse mode restores fixed scaling
  and rejects running normalization of depth. See the
  [architecture reference](04_implementation/current_hrl_architecture_summary.md#optional-inverse-depth-response-local-runtime-2026-09-14)
  for parameters and usage.
- **Acceptance criteria and outcome:** 17 focused tests passed, covering old
  configs, exact legacy sampling, unchanged checkpoint state, CLI toggles,
  fixed preprocessing, saved mode/gain compatibility, explicit-switch precedence,
  invalid saved gains, normalization rejection, and norms.
  Implemented only in `/home/xiaoxiong/SFgit/SF_hipposlam`.
- **Reusable lesson:** Trace the full preprocessing path before changing an
  observation transform; test with the real normalizer. Reference-distance
  scaling is a heuristic until actual feature norms are measured. Search source
  file types first to avoid large saved experiment artifacts, and reuse the
  shared encoder rather than duplicating transforms across consumers. Expose
  only the experimental choice needed for new runs; keep arbitrary scaling
  constants fixed and migration support confined to saved-config loading.
- **Authoritative check:** Run
  `python -m pytest -q -p no:cacheprovider sf_working_directories/IntrMotiv/tests/test_depth_encoder.py`
  from the runtime checkout using the `SF_git` Python environment.
# Direct queue replacement must stop child process groups

- **Evidence:** Interrupting the DG throughput/search supervisor left four already
  launched 10M production trainers orphaned under PID 1. They retained roughly the
  full four-run memory footprint and held available RAM near 62 GiB, so the new
  direct queue correctly refused admission at its 72 GiB threshold. The same failure
  mode recurred in the CA3 state-goal timing-profile batch: a crashed supervisor left
  35 trainer and worker processes under PID 1, including an 11 GiB learner. Selecting
  processes by their exact pinned source working directory stopped all 35 cleanly
  with SIGTERM and released about 4.4 GiB of swap without touching production.
- **Impact:** Replacing an active preflight-to-production workflow can appear complete
  while its owned Sample Factory runs continue consuming workstation resources.
- **Proposed improvement:** Persist every launched child PID and creation time at the
  transition-supervisor level and install SIGINT/SIGTERM handlers that forward the
  signal to each verified child process group before exiting.
- **Status:** Open and recurring. The affected process groups were verified by exact
  pinned source working directory and stopped manually; cleanup manifests were saved
  with the active batch artifacts. Supervisor-level child-process cleanup is still
  required.
- **Acceptance criteria:** Interrupting a transition supervisor leaves no matching
  trainer, environment worker, or W&B child alive, while unrelated user processes
  remain untouched; an integration test covers the signal path.

# Direct queues should continue after isolated run failures

- **Evidence:** One 100M DG run hit a transient 99.90%-invalid PPO batch at 1.02M
  frames. The original queue left the other three runs healthy but marked all eight
  pending cells `blocked_by_failure`, wasting an available slot.
- **Impact:** A single seed-specific failure can stall an otherwise independent
  workstation study for hours.
- **Improvement:** The direct queue now records the failed cell, keeps healthy runs,
  and continues admitting pending cells. It raises after the queue drains so the
  failure remains visible. Admission reserves 55 GiB of host RAM per 32x16 run,
  derived from the completed G500 sweep, rather than the old 8 GiB estimate.
- **Status:** Implemented locally with focused queue tests; the recovery runtime is
  isolated from the source used by active runs.
- **Acceptance criteria:** A synthetic failed run is followed by a completed run;
  the final queue still reports failure, and four-run admission retains at least
  64 GiB of measured host headroom.
