# Persistent intrinsic landmark control — implementation and launch record

## Scientific contract

This study separates three outcomes: raw DG field localization, efficient
commanded-landmark control, and coarse exploration. Physical pose remains
telemetry-only and is never a model input, reward input, or training target.

The graph-free goal conditions use a 16-way one-hot appended to the controller
input. The live DG detector receives no goal command. A command persists for a
900-decision environment episode; a unique dominant target onset pays once:

$$
r_{\mathrm{goal}}(\tau)=6.4\frac{901-\tau}{900},\quad 1\le\tau\le900.
$$

Wrong and ambiguous landmark activations do not pay or cancel the command.
Success is latched until the recurrent state is cleared at the environment
boundary. This preserves the reward scale while giving temporal discount and
the bounded multiplier the same preference for early arrival.

The separate-controller condition has 16 independent decoder, policy-output,
and value heads with shared sensory, DG, and CA3 state. Only the commanded head
receives gradient. The contextual condition reuses the causal eight-step
CA3/action-history gate. It is not described as path integration.

The frozen-reference STOP/JOINT pair imports only the projection weight and
BatchNorm state from the 75,038,720-frame
`SCR_C15_ARR_DIRS_S123` checkpoint. The reference is permanently evaluation
mode and non-trainable; the live DG starts from the same tensors but continues
intrinsic learning. This prevents PPO from moving the rewarded arrival
definition. It does not repair fragmentation already present in the parent.

## Production matrix

The batch has 36 main runs:

- 33 primary runs: 11 conditions × seeds 8, 99, and 123.
- 27 primary runs start from scratch; six are matched reference warm starts.
- Three exact original `c15_topology_ucb_direct_o1` continuations resume their
  own seed-matched 100,040,704-frame model and optimizer checkpoints.
- Every lineage targets 600M frames. The CPU partition permits four days; jobs
  request 84 hours rather than the launcher's 30-hour default.

Authoritative specifications:

- `hpc_runs/studies/persistent_intrinsic_control.study.json`
- `hpc_runs/studies/persistent_intrinsic_control_c15.study.json`

The primary preflight is derived from the production specification and selects
seed 99 once per condition. It changes only the frame target, spatial snapshot
targets/cadence, and W&B group. Goal success is not required at 2M frames; the gate
requires finite telemetry, active commands, correct replay, landmark learning
signals, and correct STOP/JOINT gradient routing.

## Evaluation contract

Online spatial snapshots are scheduled at 100M, 200M, 300M, 450M, and 600M.
Raw/pre-inhibition maps must be reported separately from inhibited behavior
maps. A fixed physical destination inferred from an independent evaluation
panel is evaluation-only. All 16 identities remain in the control denominator.

Primary targets are at least 12/16 localized mono-field units, 80% controlled
arrival at the fixed evaluation destination, and 90% navigable 4×4 coarse-cell
coverage. These are preregistered engineering targets, not guaranteed outcomes
or biological thresholds.

## Reusable workflow lessons

- Read the scheduler partition limit separately from launcher defaults. Here
  the launcher requested 30 hours while the partition allowed four days.
- Keep structurally incompatible continuation controls in a companion
  StudySpec. This preserves strict checkpoint loading without duplicating or
  contaminating the primary flat-model configuration.
- Derive one-seed preflights from the production StudySpec instead of copying a
  second condition list. This prevents preflight/production drift.
- A selected spatial endpoint is not a general warm-start result. Record its
  exact checkpoint and use it only in a matched mechanistic comparison.
- Do not stop valuable running jobs merely because new work is ready. Record
  recoverable checkpoints first and cancel only if observed resource contention
  prevents the validated new batch from starting.

## Deployment record

Source snapshot:
`hpc_runs/source_snapshots/persistent_intrinsic_control_20260908.tar.gz`
(SHA-256 `751ef505dd63e6b2cfac993ea22157bc7a17d454e18acea2f2fd150932d4c3b3`).

The final local focused runtime suite passed 47 tests; the StudySpec/workflow
suite passed 29 tests. The complete deployed NEMO2 runtime suite passed 260
tests. Deployed source hashes match the sealed local source tree.

The first live preflight exposed two launch blockers before production: its 2M
snapshot maximum violated the default 25M cadence contract, and the frozen-DG
wrapper rejected legacy checkpoints that lacked only newly introduced
bookkeeping buffers. The second preflight uses a valid 1M cadence and retains
strict loading of learned DG tensors while allowing the DG module's existing
legacy defaults for those buffers. The failed zero-frame jobs were stopped and
their output namespace was not reused.

The corrected 11-condition preflight (jobs `8002647`–`8002657`) completed at
2,031,616 frames with exit code zero for every condition. Its persisted
scientific audit passed: landmark onsets and encoder gradients were nonzero,
flat rewards were nonzero, goal commands were active, replay mismatch was zero,
STOP PPO-to-DG gradients were zero, and the JOINT gradient was positive. The
strict C15 resume smoke (`8002565`) independently advanced the seed-99 parent
from 100,040,704 to 102,039,552 frames and completed cleanly.

The first C15 production submission revealed that the historical adapter still
used the old 100M online-spatial ceiling. Those three just-started jobs were
stopped, and no output directory was reused. The corrected companion StudySpec
uses the clean `c15_r2` namespace and explicitly schedules 100M, 200M, 300M,
450M, and 600M snapshots. The logging path was also made safe when there is no
future snapshot. The authoritative remote telemetry/workflow regression suite
passed 15 tests after this correction.

Production submission completed as 36 running jobs:

- Primary workdir:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_persistent_intrinsic_control_20260908/20260908T021801Z`
  (jobs `8002719`, `8002721`, and `8002723`–`8002753`).
- Corrected C15 workdir:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_persistent_intrinsic_control_c15_r2_20260908/20260908T022742Z`
  (jobs `8002791`–`8002793`).
- Primary StudySpec SHA-256:
  `b03aeba21b2eaa6478c9a67f653b10cfab3cc4d0eaa615137f30e827480bb396`.
- C15 r2 StudySpec SHA-256:
  `75afa81ce9f82545a5adeba45f070aa2d5237befc9d66939d9d04f2028049844`.

Both submitted-manifest audits report complete exact matrices and valid
workspace paths. At the final health check all 36 jobs were running, the 33
scratch/warm-start runs had advanced frames without tracebacks, and all three
C15 jobs had advanced from 100,040,704 to 100,171,776 frames with their next
online snapshot correctly set to 200M. Lightweight authoritative records are
preserved under
`06_experiments/results/persistent_intrinsic_control_submission_20260908/`.

## Legacy-recruitment hotfix and recovery

On 8 September, C15 seed 123 failed at 148,930,560 frames and seed 99 later
failed with the same exception: legacy orthogonal DG recruitment replaced a
row while the policy-buffer controllability graph was enabled, after which the
learner unconditionally tried to invalidate PRED evidence. PRED evidence
buffers intentionally do not exist in legacy recruitment mode. This was a
mode-compatibility bug, not evidence of a scientific configuration failure.

The learner now always invalidates policy-graph state for a replaced DG row,
but invalidates predictive recruitment evidence only in graph/PRED recruitment
mode. Legacy replacement therefore records zero predictive invalidation mass;
graph/PRED mode retains its strict evidence-buffer requirement and reports the
removed mass. The existing forced-recruitment hook now also accepts legacy
orthogonal recruitment while retaining its open-gate, one-replacement, and
at-most-1M-step restrictions. No StudySpec, scientific argument, or public
configuration interface changed; workflow version remains 1.5.0.

NEMO2 validation passed 30 focused tests and the complete 263-test IntrMotiv
runtime suite. The forced legacy-replacement smoke job `8010593` completed with
exit code zero at 1,081,344 frames. Telemetry reported exactly one committed
replacement, and the run advanced without the missing-evidence exception.

The three original C15 jobs `8002791`–`8002793` were cancelled after recording
their latest checkpoints. The first recovery submission (`8010723`, `8010726`,
and `8010727`) was stopped after startup audit showed that its inherited
`--load_model_path` selected the 100,040,704-frame parent rather than each
run's latest production checkpoint. It ran for only about two minutes and did
not replace the intended recovery checkpoints. A checkpoint-pinned recovery
adapter was then print-reviewed and validated against the unchanged C15
StudySpec: every run identity, scientific option, output root, and 84-hour
resource request matches the original row; the sole intentional command
difference is the explicit latest-production `--load_model_path`.

The corrected recoveries are:

- Seed 8: job `8010779`, checkpoint 167,084,032; verified at 167,182,336.
- Seed 99: job `8010780`, checkpoint 160,301,056; verified at 160,464,896.
- Seed 123: job `8010781`, checkpoint 148,701,184; verified at 148,832,256.

All three startup logs name those exact checkpoints, resume their existing W&B
run IDs, and contain no traceback. The primary `W_REF_JOINT` seed-123 job
`8002753` did not require replacement: Slurm automatically requeued it from the
failed node onto `n3503`, where it loaded its own 54,231,040-frame checkpoint,
resumed the existing W&B identity, and reached 57,933,824 frames without a
traceback. Submitting a duplicate retry would have been unsafe, so the prepared
one-row retry adapter was retained but not launched. The other 32 primary jobs
were untouched; the final scheduler audit showed all 36 production jobs
running.

The complete post-hotfix source archive is
`hpc_runs/source_snapshots/persistent_intrinsic_control_hotfix_20260908.tar.gz`
(SHA-256
`a55749e0467713b2447f3f5ccc1c402b910aab5cc59fc028c7ec22584dbbe16c`).
The original archive remains unchanged. Deployed hashes for the key files are:

- `custom_learner.py`: `0f56770a591dfd8f830aa9212a49e6a79a5c1bd1c5dcae6b8164535c881ef54c`
- `train_hipposlam.py`: `7f5e60a028fe0f68ee24cf028b4e2c920f7f7393e91974a577979fb47bad3608`
- `test_recruitment_invalidation.py`: `54b6fc4435f99032fbc681fa4947e10d2065bc58cd03727a2ab199ca15918aa8`
- C15 recovery adapter: `9401f635d679854e3b516871b7a57b25e8652b964789f7d1036535294fc97924`

Reusable recovery lesson: a print-only StudySpec audit proves the intended
scientific command, but does not prove which checkpoint Sample Factory will
prefer in an existing output directory. A restart must pin the captured
checkpoint explicitly and gate continuation on the startup log's actual
`Loading state from checkpoint` line, recovered frame counter, W&B resume ID,
and absence of traceback. A forced failure-path preflight should likewise
trigger the event itself and verify its count, rather than merely exercise the
surrounding configuration.

## F_SIGNED early termination, 9 September 2026

At the user's request, jobs `8002730` (seed 8), `8002731` (seed 99), and
`8002732` (seed 123) were cancelled at 14:35:23 CEST. Slurm accounting confirms
CANCELLED for all three. The other 33 production jobs remained RUNNING.
Spatial information is a primary selection criterion: replicated near-zero
selectivity and recent-window silence made F_SIGNED unsuitable for further
investment in this batch. This is intentional early termination, not a completed
600M result; preserve it in comparisons rather than omitting the failed condition.

Latest checkpoints verified immediately before cancellation:

- Seed 8: `checkpoint_000018620_305070080.pth` (305,070,080 frames).
- Seed 99: `checkpoint_000014292_234160128.pth` (234,160,128 frames).
- Seed 123: `checkpoint_000014534_238125056.pth` (238,125,056 frames).

All checkpoints, logs and telemetry were retained in the original production
directories. The original StudySpec remains unchanged as the experimental
record. F_SIGNED is the flat, CA3-absence-gated decoder condition with signed
encoder feedback on dominant onsets:

$$
r_{\mathrm{enc}}=0.1(d-R),\qquad R=8.
$$

Here $d$ is the implementation's temporal transition distance, not physical
separation. Relative to F_GATE, the changed factor is
`--dg_ca3_temporal_exclusion_coeff=1.0` instead of zero. It has no commanded
goal and no DG reentry inhibition. The decoder retains distance scaling and
only pays when the arriving landmark is absent from CA3 memory.

Reusable procedure: confirm exact job identities and recoverable checkpoints,
cancel only the selected condition's jobs, check Slurm accounting, and retain
the early-stop decision alongside the unchanged scientific specification.
