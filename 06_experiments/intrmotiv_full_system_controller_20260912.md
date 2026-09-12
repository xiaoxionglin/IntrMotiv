# Full-system IntrMotiv controller integration — 12 September 2026

**Current status: repaired revision-3 preflights are submitted; continue autonomously until production is running.**
Jobs **8057267–8057272** use
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_terminal_20260912` and the
isolated native terminal binding under workspace `runtime/controller_terminal_binding_v1`.
The 18-run production guard is still active. The thread heartbeat
`qualify-and-launch-intrmotiv-production` must continue qualification/repairs and
launch, and pause only after production is verified running.

Revision-3 canonical batch: `intrmotiv_full_system_controller_preflight_20260912_r3`;
study SHA-256 `d12812ca420f6da68e7fa0d8f1c3e63e369329a49e4896dca801f3f7d97347b9`.
Its submission manifest is workspace `train_dir/_slurm/` followed by that batch
name and `/submission/jobs.tsv`. Older revision-2 jobs were stopped and preserved.
The later sections below retain their historical evidence; use this status and
the latest canonical StudySpec for current operations.

Native terminal gate 8057262 passed, full PPO episode parity 8057263 passed with
identical 1,800-decision observation/reward/termination hash, and end-to-end
wrapper gate 8057266 passed with a certified final RGBD observation. All 342
system tests and 32 workflow/auditor tests passed remotely. The engine and assets
are unchanged; only an explicit terminal reader was added to an isolated Python
binding. Neither the shared installation nor unrelated jobs were modified.

## Scope and provenance

The intervention is controller learning only. All three historical parents are
included: `DGC_DIRECT_WORKER_F16_S99`, `DGC_WAYPOINT_DG_F64_S8`, and
`DGC_WAYPOINT_DG_F64_S99`, at 25,001,984 frames. These are a historical
compatibility shortlist, not a new performance ranking.

The source is the actual NEMO2 checkout at
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam`, revision
`636be3a5862529bfb510b3e52e2f41f8cae9e78d`, including its tracked modifications
and untracked goal-write implementation. The isolated implementation checkout is
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_20260912`, branch
`codex/full-system-controller`. This is not a revert to public master.

[Evidence](data/intrmotiv_full_system_controller_20260912/) contains complete
resolved parent configs, verified checkpoint hashes, source-file hashes, per-parent
preservation manifests, the implementation patch, and test results. Current
source provenance is recorded; exact launch-time source equivalence remains a
separate provenance check. Configuration replication and checkpoint continuation
must remain distinct initialization modes.

## Objective decision

The user approved **a separate auxiliary HER head**, sharing original worker
features with the acting main DDQN head. Main Q-values select actions and estimate
the parent's continuing return. Auxiliary Q-values never select actions or
replace the main Bellman target. Shared-feature gradients can still interact;
this is an explicit experimental factor, not a claim of complete independence.

Source authority: `DistanceLearnerReward._calculate_internal_reward` and
`_prepare_batch` in the original `custom_learner.py`. The selected parents use
`hit_distance`, `gamma=0.99`, and `value_bootstrap=False`.

| Boundary/event | Main return | Auxiliary HER |
|---|---|---|
| Physical termination | Stop bootstrap | Stop bootstrap |
| Physical truncation for these parents | Stop bootstrap (`value_bootstrap=False`) | Stop bootstrap |
| Manager retargeting | Continue with the actual successor command/context | Does not substitute a new manager objective |
| Next waypoint | Continue with actual successor waypoint/context | Virtual worker task remains separately specified |
| Option timeout | Continue according to original physical return | Can end the explicitly bounded virtual task |
| Target/other-landmark arrival | Preserve original reward components and continuation | Virtual task's achievement/boundary must be recomputed consistently |
| SF rollout/replay chunk ending | Not a physical terminal; require valid continuation | Not a manufactured terminal |
| Missing successor observation | Reject required bootstrap example; never use reset observation | Same validity requirement |

The main target is $r_t+\gamma(1-d_t)Q^-(s_{t+1},\arg\max_aQ(s_{t+1},a))$.
There is no 0–1 Q clamp, three-goal vocabulary, or replacement first-arrival reward.
The auxiliary head includes a remaining-budget scalar and state-budget interactions
because its proposed objective is bounded. Its horizon construction, loss weighting,
and equal-TD-budget study configuration are **not yet qualified**; no fixed
64-decision task has been introduced into the acting controller.

## Implemented and checked

- Opt-in original actor/controller extension: main action-value head and optional
  auxiliary head over the actual decoder. PPO and shadow acting paths retain
  their original parameter initialization and action calculation.
- STOP/JOINT controller boundary reused; goal-write uses the actual worker view.
- Separate Bellman boundary functions; native SF categorical action distribution
  and sampling reused. Zero-epsilon/masked-action entropy stays finite.
- Original DG auxiliary loss block extracted into one shared method without
  copying its formulas. Fresh optimization scheduling is not replaced.
- Existing no-extra-encoder-loss telemetry bug fixed: disabled loss components
  now have zero telemetry instead of undefined local variables.
- Whole-model snapshots include DG, write modulation, normalization, and graph
  buffers. Differentiable replay reaches online parameters through an isolated
  model with cloned snapshot buffers. Version mismatch is rejected; replay does
  not advance the parent's RNG or update its live caches/statistics/graph.
- Reconstruction through original preprocessing, encoder, packed core, and decoder
  for the selected finite simple-core families. Complete worker histories are
  rebuilt separately under online/target parameters. A packed call retains
  gradients through earlier goal writes; repeated single-step calls would detach
  those histories in the original core.
- Per-config delta validation rejects unclassified changes, including freezing DG,
  reducing its vocabulary, changing manager families, or misspelled new options.

## Native integration and qualification

Native SF transport now carries stream/episode/decision/publication identity,
action-time conditions, physical frame counts, separate termination/truncation,
and certified terminal observations. Ordered ingress owns observations before SF
buffer reuse and joins physical successors across rollout boundaries. Replay
stores frozen visual features after exact equivalence checks, retaining depth
and instructions. Uncertified terminal observations are rejected.

Online and target snapshots reconstruct recognition, original reward components
and event labels together. Main DDQN follows the actual successor command;
auxiliary HER reconstructs a separate fixed-goal history and budget. Structural
or event incompatibility is counted. HER positions add gradients and never replace
main positions or write virtual evidence into the real graph.

The original learner owns optimization. Fresh DG updates and graph processing
precede replay, which does not run DG auxiliary losses or update normalization or
graph buffers. A separate published model prevents actors seeing intermediate
fresh/controller updates. Publication includes buffers and version; inference
rebuilds finite history while preserving the actual manager state. Checkpoints
include replay, online/target parameters, optimizer, RNGs and counters. Restart
opens new physical sessions and discards unfinished joins.

- **342 tests pass locally and on NEMO2**, including transport ordering, certified
  terminal wrappers, reward/HER parity, actor history, safe checkpoint loading,
  finite-history values/gradients and matched controller clocks.
- **32 canonical workflow/auditor tests pass locally and remotely.**
- Actual native learner update and complete checkpoint-resume audits pass for all
  three parent configurations. These use synthetic observations, not DMLab.
- PPO full learner audit matches all updated model/buffer and optimizer tensors
  bitwise against the saved original learner for all three configurations.
- Full 256-position replay benchmark preserves loss and Q statistics while smaller
  internal batches reduce one measured waypoint update from about 32 to 10 seconds
  on the desktop CPU. This is not yet a cluster-throughput guarantee. Internal
  batching does not change positions per optimizer update or loss scaling.

## Six preflights

The canonical [StudySpec](../hpc_runs/studies/full_system_controller_preflight.study.json)
is the source of truth: schema `intrmotiv/study/v1`, workflow **1.8.0**, SHA-256
`3a40649a9a79b3c5de3edf9c587722524ef3a11779c591891c0a607fda458fd3`.
All six start fresh with the fixed ImageNet trunk, seed 99, original fresh DG
schedule, W&B, exploration/control dashboards and online spatial telemetry.
Checkpoint/snapshot targets are 1M and 2M frames for qualification.

| Architecture | PPO | DDQN | DDQN + auxiliary HER |
|---|---|---|---|
| Direct F16 | 8057255 | 8057256 | 8057257 |
| Waypoint / goal-write F64 | 8057258 | 8057259 | 8057260 |

Submission artifacts are under workspace
`train_dir/_slurm/intrmotiv_full_system_controller_preflight_20260912_r2/submission/`.
The [submitted audit](data/intrmotiv_full_system_controller_20260912/r2/controller_preflight_submitted_r2.json)
confirms six submitted rows, matching canonical commands and workspace paths.
The template uses the submitting checkout and a short workspace TMPDIR. The
first print-only review caught its old hardcoded original-checkout path before
submission. No training outputs or caches are directed to the home filesystem.

First-attempt jobs 8057249–8057254 were stopped and preserved. DDQN correctly
rejected missing initial history because SF's default startup decorrelation walks
occur before inference. Revision 2 uses SF's existing
`decorrelate_envs_on_one_worker=False` in **all six arms** so policy history starts
at physical reset. Worker delays remain enabled. This explicit startup difference
is classified in the parent-config delta audit; model/fresh-learner preservation
tests do not claim runtime identity for that startup walk.

Remaining gates: real DMLab terminal provenance, active-episode publication
rebuilds, main update debt/counts, positive auxiliary HER learning, fresh DG and
frozen-trunk changes, W&B dashboard delivery, full 2M completion and runtime
checkpoint reload. The controller runtime auditor extends the existing DG
preflight auditor. Its numerical checks do not substitute for a live restart or
W&B API check. Only after all gates pass may the 18-run, 300M-frame production
StudySpec be rendered, reviewed and submitted.

## Reusable experience

Use the actual runtime snapshot and per-parent config manifests first. Local Python
imports initially resolved the older editable checkout because the snapshot omitted
its parent package `__init__.py` files; restoring those package markers corrected
source resolution. The actual-model replay audit also caught integer instruction
channels being converted to floats by direct normalization; replay now reuses SF
`prepare_and_normalize_obs` for the original type/device restoration. One initial suite failure was a missing copied Lua environment
patch, not a controller regression; retrieving the original patch made the complete
suite runnable. Check imported module paths before trusting a local preservation test.

The authoritative checks are the full original IntrMotiv test directory, actual-model
parity against the saved pre-change actor source, exact checkpoint/source hashes,
and generated config deltas. Reuse these artifacts rather than re-inventorying the
historical batch scripts. Every subsequent claim should distinguish module tests,
full learner parity, environment preflight, and production qualification.

For subsequent replay optimization, profile a full 256-position update, including
backward. The dominant desktop cost was tensor copies/zero fills during backward,
not environment collection. Batched history slices caused repeated large gradient
scatters; a single indexed gather and smaller internal batches improved the same
objective. Preserve gradient/value parity and total TD counts when changing this.
Keep checkpoint replay payloads as tensors/plain containers so the established
weights-only place-field evaluator can load them. Never deploy into the running
original checkout; verify patch preimages and audit the rendered template's `cd`.

At the first corrected live telemetry check, all six runs were collecting without
startup exceptions. Each had 113 DG/spatial metric tags. At 65,536 frames, each
DDQN arm recorded eight fresh DG steps, four fresh graph batches, over 100 actor
history rebuilds and zero actor-memory version failures. Replay warm-up was still
completing (about 16,330 accepted decisions); these observations do not yet
qualify TD learning or HER overhead. All six W&B runs connected successfully.

## Interim production-gate check

The follow-up production request remains gated: all six jobs were still running
below 2M frames. At the live check, PPO reached approximately 1.08M direct / 0.67M
waypoint frames; DDQN arms were around 0.10–0.18M. Main controller clocks had zero
update debt and exactly 256 TD positions per completed update; target refreshes
were exercised and actor-memory version failures remained zero. The W&B API
confirmed delivery of controller counters and 113–114 DG/spatial summary tags.

Both HER arms still had zero auxiliary positions. Saved direct-arm replay
contexts had zero active-landmark IDs and zero positive manager budgets, so no
eligible virtual option was available in those early snapshots. DG projection
weights changed while all frozen-trunk tensors matched initialization. Real
terminal observations and positive HER learning remain unqualified, as does a
live process restart. No production jobs were submitted; preflights continue.

The interim auditor initially used bare `weights_only=True`, which rejected a
PPO checkpoint's original NumPy scalar metadata. The established evaluator
already handles that metadata safely. The auditor now reuses
`place_fields.load_checkpoint_dict`; its focused regression test passes. This
was an audit-loader mismatch, not evidence of a corrupted training checkpoint.
See [interim gate evidence](data/intrmotiv_full_system_controller_20260912/r2/interim_gate_check.json).

## Autonomous continuation and terminal blocker

The user explicitly required autonomous continuation until production is running.
A 15-minute thread heartbeat `qualify-and-launch-intrmotiv-production` is active
to continue qualification/repairs and launch only after all original gates pass.
It must be paused once production is verified running.

Independent compute-node terminal gate **8057261** failed: the actual environment
ended after 1,800 decisions with `controller_final_observation_valid=False`.
The native Python binding rejects `observations()` once the engine reports
termination. Keep the replay rejection in place. A minimal explicit native
terminal-observation API is being investigated; preserve default PPO behavior and
the original 120-second episode boundary. Do not substitute the preceding image,
shorten episodes, or declare this gate passed based on the wrapper's mocked tests.
Existing native source is `/home/fr/fr_xl1014/deepmindlab/lab` with locally modified
BUILD/WORKSPACE. Do not modify the installed shared engine used by running jobs.

## Checkpointed restart qualification in progress

A private replay optimization removes a redundant large output pack/unpack while
retaining SF's packed input and the original core. Full 256-position desktop
update time decreased from about 10 s to 3.7 s with identical loss/Q statistics;
value/gradient parity tests cover both cores, and full original PPO learner parity
passes for all parents. The full suite with three new tests has 345 cases.

Current action: a deliberate SIGINT was sent to batch processes 8057267–8057272
at about 05:01 local time. Workers ignore SIGINT and finish their transactions;
`LearnerWorker.on_stop` saves before exiting. Exit code 2 here is the requested
interruption, not a new runtime defect. **Do not restart experiment directories
until all six original jobs have stopped and each has a complete checkpoint.**

The optimized inactive source is
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_padded_20260912`.
Its hash-checked patch is [padded replay patch](data/intrmotiv_full_system_controller_20260912/r3/padded_replay.patch).
Remote tests and canonical `restart_review` are being run. After they pass,
archive/hardlink each latest immutable checkpoint as a restart baseline, then
submit the **same R3 StudySpec and experiment directories** using the new source
and Slurm workdir `_slurm/intrmotiv_full_system_controller_preflight_20260912_r3/restart_submission`.
This resumes the already fresh-initialized preflights, rather than restarting
scientific learning. Verify new physical replay session, restored main/target/DG
clocks, nondecreasing checkpoint counters, actor-history correctness, and final
2M completion. Retain original submission artifacts and record replacement job
IDs. This live restart is part of the user's mandatory qualification gate.

All six controlled stops completed by 05:12 local. The optimized checkout passed
345 runtime tests and 32 canonical/auditor tests on NEMO2. The restart print-only
audit passed with unchanged R3 study SHA and workspace-only output paths.
Before submission, latest checkpoints are hardlinked and safely loaded under
`train_dir/analysis/restart_baselines/intrmotiv_full_system_controller_preflight_20260912_r3/`.
Direct DDQN and DDQN+HER both stopped at 131,072 frames, 255 main updates and
65,280 main TD positions, with 32,704 accepted decisions and zero update debt.

Restart submitted successfully: **8057273–8057278**, in canonical order direct
PPO/DDQN/DDQN+HER, waypoint PPO/DDQN/DDQN+HER. New authoritative jobs manifest:
`_slurm/intrmotiv_full_system_controller_preflight_20260912_r3/restart_submission/jobs.tsv`.
The submitted audit passed exact commands, all six IDs and workspace paths.
Baseline counters are in [restart baselines](data/intrmotiv_full_system_controller_20260912/r3/restart_baselines.json).
These are resumed preflights, not the 18-run production batch. Production remains
gated on complete 2M-frame runtime and live restart evidence.

All six replacements loaded their exact pre-stop checkpoint paths and entered
RUNNING. W&B retained the original R3 run IDs. The first comparable resumed
Direct F16 updates match: 318 completed main updates, 81,408 main positions,
zero debt and zero actor-memory failures in both DDQN arms. HER remains zero at
this early checkpoint. Independent compute-node job **8057281** tests exact
model/optimizer/target/RNG restore from the immutable baselines using SF's real
environment-info extraction and learner initialization. The final runtime auditor
now accepts `--restart-baselines <baselines.json>`; focused tests pass locally
and remotely and require a fresh physical session and preserved counters.
