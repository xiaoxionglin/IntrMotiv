# Persistent intrinsic landmark control — implementation and launch record

## Scientific contract

This study separates three outcomes: raw DG field localization, efficient
commanded-landmark control, and coarse exploration. Physical pose remains
telemetry-only and is never a model input, reward input, or training target.

The graph-free goal conditions use a 16-way one-hot appended to the controller
input. The live DG detector receives no goal command. A command persists for a
900-decision environment episode; a unique dominant target onset pays once:

\[
r_{goal}(\tau)=6.4\frac{901-\tau}{900},\quad 1\le\tau\le900.
\]

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
