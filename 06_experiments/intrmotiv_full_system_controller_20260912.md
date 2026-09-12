# Full-system IntrMotiv controller integration — 12 September 2026

**Current status: waypoint HER replay consistency failure under investigation; production is blocked.**
Job 8057338 failed its selected replay eligibility recheck at about 557,056 frames.
The other three DDQN preflights remain running. A bounded checkpoint reproduction
is testing batch-dependent recognition/event labels; no compatibility checks have
been relaxed and active training source remains immutable.
Both PPO jobs **8057292/8057295** completed their 2M horizons and pass the parent
runtime audit. DDQN jobs **8057335–8057338** resumed from complete immutable
checkpoints with 128 GB host memory each. Exact GPU reload job **8057339 passed** all four new checkpoints
with exit 0; both original PPO certificates are retained. The authoritative mixed
manifest is `train_dir/_slurm/intrmotiv_full_system_controller_preflight_20260912_r4/reachability_submission/jobs.tsv`.

Active DDQN source:
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_reachability_20260912`,
local mirror `/tmp/intrmotiv_reachability_benchmark`. Do not modify active source.
All **373 runtime tests** pass locally and remotely. GPU gate **8057330** proved
bitwise-identical model, optimizer and main/HER RNG states after three full
updates, with a **1.94× speedup** over the previously qualified screened source.
The prior screening improvement itself passed GPU gate 8057324 with a 2.23×
speedup. Canonical print-only and submitted audits pass for the new source.

Fifth-restart baselines are under
`analysis/restart_baselines/intrmotiv_full_system_controller_preflight_20260912_r4_reachability/`.
All four prior jobs 8057325–8057328 completed deliberate SIGINT shutdowns and
saved complete state before resubmission. Their final frame counts were
704,512 / 720,896 / 475,136 / 507,904 in manifest DDQN order. Main update counts
were 2,491 / 2,555 / 1,595 / 1,723 with exactly 256 main positions per update;
HER separately added 128,821 and 18,983 positions. Fresh DG counts match the
parent clock. New sessions must advance from 4 to 5 and cumulative incomplete
tail discards from 192 to 256. W&B resumes the original four run IDs.

The canonical offline place-field probe **8057320 passed** and produced valid
DG, worker and pre-threshold maps across six physical episodes. Continue until
all four DDQN runs finish 2M and the full restart/runtime audit passes; only
then release the guard, render/audit the 18-run 300M StudySpec and launch it.
Use 16 CPU cores, 128 GB host memory and one L40S GPU per production run.

R4 batch `intrmotiv_full_system_controller_preflight_20260912_r4`, Study SHA-256
`2890df1aa1152fd94e12ee30b09ada873f474b0e11eb4f9400e6033866aafdae`.
GPU Torch 2.9.1 / torchvision 0.24.1 / NumPy 1.26.4 are isolated in workspace
`runtime/torch_cuda_2_9_1`; the native terminal binding is also isolated there.
Jobs use one L40S GPU and 16 CPU cores (first completed PPO had 40); all retain
32 SF workers × 2 environments and the original scientific parameters.

Exact GPU reload job **8057305 passed all six immutable R4 checkpoints**,
including PPO model/buffers/optimizer/counters, with checkpoint SHA-bound results.
Certificate: `analysis/restart_baselines/<R4 batch>/reload_certificate.json`.
A proposed audit upload that skipped PPO restart validation was rejected by
 automatic approval review and reverted. The stronger replacement has now been
uploaded and its remote tests pass: it retains all PPO validation and requires
checkpoint-bound exact reload evidence for an already-completed PPO preflight.
All original DG/telemetry/horizon and DDQN live-progress checks remain active.

The deployed telemetry repair is mirrored in `/tmp/intrmotiv_telemetry_benchmark`:
reuse the original goal-sensitivity diagnostics in DDQN and record goal-write
gradients after controller replay. All 357 tests pass and full CPU PPO learner
model/buffer/optimizer parity passes for all three parent references. This is
now included in jobs 8057308–8057311. At 262,144 frames, waypoint DDQN+HER had 113
unpaid updates because its current snapshot has no compatible replay. The
strict rejection rule remains intact; recovery is a required production gate.

No production study has been rendered and its guard remains active. Heartbeat
`qualify-and-launch-intrmotiv-production` must continue until the canonical
18-run production batch is verified running, then pause. Earlier sections below
retain the complete CPU R1–R3 and R4 repair evidence.

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

Exact restore gate **8057281 COMPLETED 0:0**: all six model/optimizer states and
all four DDQN target/clock/main+HER RNG/Python/NumPy/Torch RNG states match their
immutable baselines. Physical sessions increment and ingress starts empty.
See [exact restore results](data/intrmotiv_full_system_controller_20260912/r3/exact_restore_results.jsonl).

CPU replay remains the dominant runtime cost. A bounded GPU benchmark (8057282)
failed because the existing SFgit Torch 2.9.1 build is CPU-only, before training.
A matching Torch 2.9.1 / torchvision 0.24.1 CUDA 12.8 overlay is being installed
under workspace `runtime/torch_cuda_2_9_1`, with NumPy pinned to existing 1.26.4.
The installed shared SFgit environment and active CPU runs are unchanged. Only
consider GPU deployment after benchmark and preservation/runtime qualification.

GPU benchmark lesson: jobs 8057283/8057284 used an invalid synthetic CPU training
buffer with a GPU learner. SF's real Batcher already allocates training buffers
on `policy_device`; fixing the harness removed both device errors. The exploratory
graph-device patch was reverted and must not be promoted. Correct harness job
8057285 completed a full 256-position update (~4.78 s main) but found a real GPU
checkpoint issue: `map_location=cuda` moves saved CUDA RNG byte tensors, while
`set_rng_state_all` needs CPU ByteTensors. An isolated one-line `.cpu()` restore
fix and wider GPU-only benchmark batching are being tested in 8057286. Active
CPU source remains the padded checkout. No GPU production decision has been made.

Batched parent event/reward reconstruction passed **347 tests** including mixed
main/HER, terminal, rejection and gradient parity against scalar evaluation.
GPU benchmark **8057287** reduced main 256-position time from 4.36 s to **0.903 s**
(whole fresh/replay transaction 1.55 s), with identical loss and successful GPU
checkpoint reload. Candidate code is only in inactive
`SF_hipposlam_controller_cuda_20260912`; local prototype `/tmp/intrmotiv_cuda_benchmark`.
It currently uses experimental internal width 256 for all devices; before any
promotion retain CPU width 16, qualify GPU width explicitly, and run full remote
suite. No new production or GPU preflight has been submitted.

**New production blocker at 05:42:** direct DDQN at 376,832 frames had 645 main
updates and debt 569; direct HER at 442,368 had 638 main updates and debt 832.
HER has learned 21 auxiliary positions. Actor memory failures remain zero.
Rejections are predominantly `history_recognition_changed`; the implementation
abandons a main batch after only four times its size in candidates. This arbitrary
bound is not in the user's contract. Do not weaken recognition/event rejection or
clear debt to pass the gate. Slurm job **8057288** measures eligibility of 4,096
candidates from a real latest direct checkpoint using the isolated GPU candidate.
Use its evidence to replace the premature candidate cutoff with principled
sampling/accounting, and qualify the repair before production. Existing CPU
preflights 8057273–8057278 remain running on unchanged padded source.

R4 submission and canonical audit passed: **8057292–8057297** in canonical order.
Direct PPO 8057292 started with 40 CPUs; other jobs initially waited because free
L40S GPUs had fewer than 40 CPU cores available. A second canonical print-only
review with `--slurm_cpus_per_job=16` passed. Pending jobs 8057293–8057297 were
updated through `scontrol` to 16 CPUs/16 CPUsPerTask without altering training
commands, worker counts, model/data settings or GPU type. The actual reservation
record is workspace `analysis/controller_r4_resource_adjustment.log`. Use 16 CPU
cores per L40S GPU for the eventual canonical submission unless runtime evidence
requires a different reservation. Keep 32 SF workers and two environments each.

## R4 GPU checkpoint restart and publication qualification

GPU publication probe **8057300** measured the original client returning in 5 ms
while its queued GPU copy took 45 ms. The repaired client waited 40 ms and its
completion event was finished before the lock/version became visible. The fix
reuses SF `synchronize` inside its existing lock, only for DDQN/shadow; PPO's
path is unchanged. All **355 tests** pass locally and remotely.

The restart candidate is
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_publication_20260912`;
local `/tmp/intrmotiv_publication_benchmark`. Its extra patch is
[publication barrier](data/intrmotiv_full_system_controller_20260912/r4/publication_barrier.patch).
At about 06:16, controlled SIGINT was sent to DDQN jobs 8057293/94/96/97. They
finish transactions and save before restart. Direct PPO 8057292 completed 2M;
waypoint PPO 8057295 continues on the unchanged PPO path. Preserve both PPO jobs.
The canonical R4 `restart_review` passed with 1 L40S GPU, 16 CPUs, 80G and
unchanged scientific commands/Study SHA. After all four DDQN jobs stop and their
baselines load, render `restart_submission` print-only, preserve the original
PPO rows, mark only the four DDQN rows `pending_submission`, and use the existing
`resume_slurm_submission.py` utility. `controller_prepare_partial_restart.py`
in workspace analysis performs this artifact adaptation and rejects active old
DDQN jobs. Then run full submitted audit and verify GPU checkpoint restore.

## R4 sparse-candidate GPU memory and original control telemetry repair

The interim full audit detected CUDA OOM in direct/waypoint plain DDQN jobs
8057301/8057303; their learner processes failed while Slurm still showed RUNNING.
The lightweight status helper had matched only a fixed list of exception names,
missing `torch.OutOfMemoryError` in stderr. Use the full gate; the helper now
checks generic exception names and traceback headers in both log streams.
Sparse compatibility search retained an autograd graph for every candidate
batch containing even one accepted position. Selection now runs under no-grad,
then reconstructs only the selected 256 positions under the unchanged snapshot.
A changed eligibility result in this same-snapshot pass raises an explicit error.
Uniform selection, loss mean, optimizer steps, rejection rules and RNG streams
are unchanged. Tests require no gradient-enabled candidate search and exactly
one 256-position main budget; the complete runtime suite passes 357 tests.
Real-checkpoint GPU job 8057306 completed a full update (574→575, +256 TD positions)
in 5.75 seconds with peak allocated GPU memory 4,883,519,488 bytes.

The original goal-write gradient metric was sampled during fresh DG processing,
before controller replay, so it reported zero even when HER changed the weights.
A shared parent helper now records it after replay. The original goal-readout
sensitivity helper is also reused in DDQN instead of placeholder zeros. Both
changes preserve original metric names; CPU full PPO model/buffer/optimizer
parity passed for all three parent references. The isolated candidate source is
`SF_hipposlam_controller_telemetry_20260912`; no active source was overwritten.

## Production checkpoint storage qualification

The workspace filesystem has 4.6 TB total and 3.7 TB free at this check. A
53k-decision controller checkpoint is already 2.6 GB; an unbounded full-replay
archive every 30 minutes would exceed capacity during the 300M horizon.
Controller periodic milestones now use the existing SF `keep_checkpoints=8`
retention limit, **plus protected canonical frame checkpoints**. Every retained
checkpoint still contains full replay, online/target models, optimizer, RNG and
counters. Regular rolling/best/initial checkpoints and save times are unchanged.
The parent SF save and discovery methods are reused; only periodic milestone
retention is bounded. Canonical filenames are recorded before save and restored
from checkpoints so a later restart cannot unpin them. Two focused tests verify
pinning before save and retention of canonical plus latest periodic checkpoints.
This keeps the estimated production checkpoint footprint around 2.6 TB at
200k replay capacity across the 12 DDQN runs, instead of unbounded growth.
PPO checkpoint behavior is unchanged. Complete runtime suite: 359 passed locally
and remotely. All four DDQN jobs resumed only after canonical print-only review
and submission audit; their IDs are 8057308, 8057309, 8057310 and 8057311.

## Pending: align replay burn-in with actor reconstruction

Jobs 8057308–8057311 have no logged memory exceptions, but waypoint/HER still
has update debt (418 at 344,064 frames). The remaining prefix-wide recognition
veto conflicts with the actor contract: actor publication rebuilds changed
71-decision memory while retaining actual manager context. Replay must likewise
reconstruct its worker prefix under the declared snapshot and treat recorded
commands as exogenous; it must not counterfactually re-plan the real manager.
`_calculate_reward_components` uses reconstructed traces for temporal rewards,
and only current/successor manager contexts for transition outcome labels.
A candidate removes the historical-prefix veto, retaining structural-generation,
current-recognition, successor-event, and online/target compatibility checks.
This is being compared against the old filter on the same stalled real replay
before deployment. New tests cover changed burn-in with unchanged actual
manager context, plus continued rejection of changed current recognition.

Candidate (not active training source):
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_compatibility_20260912`,
local `/tmp/intrmotiv_compatibility_benchmark`; 363 tests pass locally.
It also fixes cumulative restart-tail rejection counts and makes offline
checkpoint loading use safe private mmap on CPU (exact values/aliasing tested),
avoiding eager replay loads during audits. The existing place-field shell
worker now resolves its own source checkout and the certified terminal binding,
instead of importing the unrelated original editable installation. Active
telemetry training source remains unchanged. The offline mmap loader preserves
`weights_only=True` and its exact NumPy allowlist.

## Replay recovery and maintenance safety evidence

Paired GPU probe 8057313 tested the same 4,096 examples from the waypoint/HER
360,448-frame checkpoint. The old prefix veto rejected all; reconstruction with
unchanged current/successor checks admitted 54 (including a commanded example).
GPU gate 8057314 then completed 31 full main updates through target refresh:
670→701 updates, 171,520→179,456 main positions, 123→350 HER positions, target
600→700. Goal-write norm rose from 0.0183 to 0.1674; main/auxiliary losses were
0.6727/0.1440 and peak allocation was 11,480,076,288 bytes. Runtime was 315.2s.

Automatic review initially rejected stopping the four active preflights as
unproven disruption. The stop was approved after concrete evidence: SF workers
ignore SIGINT (`learner_worker.py:38`), `LearnerWorker.on_stop` saves complete
state, the regular save uses atomic rename, exact GPU reload job 8057312 passed
all four states, and 8057314 proved recovery on a current real checkpoint.
Only these four preflights were signaled; no unrelated production job was touched.
A queued replay transaction can make this graceful stop take many minutes.

Long catch-up transactions now report actual completed work through SF's existing
heartbeat callback at its configured cadence. This avoids confusing a progressing
learner with a dead one without background heartbeats or disabling the watchdog.
Auxiliary Q mean/absolute maximum are logged separately. Private mmap reduced the
real checkpoint audit's peak RSS from 6.15GB to 2.47GB with equal model/counters;
load time stayed 50–55s, showing metadata parsing remains expensive.

Before production: use architecture and controller as separate analysis groups,
include HER-minus-DDQN contrasts within each architecture, and retain the parent's
5M/25M/75M/150M/300M checkpoints and intervention telemetry. L40S jobs have a 48h
maximum allocation; the 300M learning horizon remains a separate setting.

## Throughput evidence and screened candidate

GPU profile **8057321** restored the immutable 393,216-frame waypoint/HER state
and completed three main updates (1277→1280, +768 main positions, +120 HER).
Of 128.4 profiled seconds, history assembly accounted for 52.4s, manager advance
34.3s (28.2s in transitive reachability), and finite neural history reconstruction
14.1s. Most history assembly was discarded by the unchanged current-recognition
veto. This is evidence for screening that veto before reconstructing history.

The isolated candidate `SF_hipposlam_controller_screened_20260912` (local
`/tmp/intrmotiv_screened_benchmark`) adds that early screen using each snapshot's
unconditioned DG head and SF observation preparation. Accepted examples still
use the original complete history, event/reward reconstruction and gradients.
370 tests pass locally and remotely; paired GPU gate **8057322** compares 8,192
physical examples under online and target snapshots and then repeats the same
three-update profile. This candidate is not deployed to training yet.

Offline evaluator **8057320 completed with exit 0**. The legacy inclusive loop
recorded 10,001 decisions across six physical episodes. Occupied-cell maps and
activities are finite; absent peaks/unvisited cells retain their documented NaNs.
All 64 DG units were active, with 53 distinct peak bins, active-only map cosine
0.1289 and 41 pre-threshold peak bins across 277 occupied bins. The existing
spatial-information scalar is amplitude-weighted (mean 0.05212), despite legacy
CSV columns containing `bits`; it is not normalized bits per activation.
The probe used the 327,680-frame waypoint/HER checkpoint and qualifies the
telemetry path, not a training-result comparison or fixed-trajectory drift test.

## Qualified candidate-only screening

The first two exploratory GPU comparisons (8057322/8057323) found float32
roundoff when candidate histories were packed into smaller batches. Screening
is now used **only for no-gradient candidate search**; all main/HER gradient and
target-value evaluations retain the original full batch geometry. It does not
change the sampler, TD counts, losses, reward/event functions or target schedule.

GPU gate **8057324 passed**: 8,192 physical examples (16,384 online/target checks)
had identical rejection/event/boundary labels. Three complete updates from the
same immutable checkpoint produced **bitwise-identical model, optimizer, main
RNG, HER RNG and clock states**. Time fell from 126.02s to 56.58s (2.23×).
371 tests pass locally and remotely; the canonical print-only six-run manifest
and audit passed. Source is `SF_hipposlam_controller_screened_20260912`, local
`/tmp/intrmotiv_screened_benchmark`. Main and HER learning values are never
approximated by screening results. Preserve this exact optimizer-transaction
comparison for future replay batching optimizations.

## Exact graph reachability qualification

The remaining graph bottleneck recomputed a transitive closure for every
candidate edge. The isolated reachability candidate computes the closure once;
newly connected pairs equal old predecessors of the source crossed with old
successors of the destination, excluding already reachable pairs. Scores and
tie-breaking are unchanged. Exhaustive three-node graphs and larger cyclic and
acyclic graphs match an independent BFS oracle. All 373 runtime tests pass
locally and remotely; three parent PPO fresh updates retain bitwise model,
buffer and optimizer parity. GPU job 8057330 checks complete DDQN/HER update
parity and runtime before any deployment to active training.

GPU gate **8057330 passed** with exact model, optimizer, clock and both replay
RNG states after three complete updates. Existing screened search plus the
original graph helper took 55.00 seconds; the exact reachability helper took
28.42 seconds (**1.94×**). All 16,384 online/target eligibility checks preserved
rejection and event labels. A canonical print-only resume was rendered in
`reachability_submission`, and jobs 8057325–8057328 received graceful SIGINT.
The existing coordinator waits for complete shutdown before immutable baseline
capture, submission audits, replacement jobs and exact GPU reload checks.
