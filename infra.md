# Infrastructure Improvements

Track concrete infrastructure findings here so future threads can reuse
evidence and coordinate improvements. Consult this file before related work;
keep implementation guidance in the canonical workflow documents linked below.

## Open Improvements

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
  direct queue correctly refused admission at its 72 GiB threshold.
- **Impact:** Replacing an active preflight-to-production workflow can appear complete
  while its owned Sample Factory runs continue consuming workstation resources.
- **Proposed improvement:** Persist every launched child PID and creation time at the
  transition-supervisor level and install SIGINT/SIGTERM handlers that forward the
  signal to each verified child process group before exiting.
- **Status:** Open. The affected process groups were verified by batch output path and
  stopped manually; the replacement queue then admitted four runs.
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
