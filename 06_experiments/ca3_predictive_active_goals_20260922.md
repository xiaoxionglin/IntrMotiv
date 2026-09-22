# CA3 predictive active-goal batch — 22 September 2026

## Status

Implementation, deployment, and release preparation are complete. The first
qualification wave (`8131848`--`8131854`) was invalidated and cancelled after
the 2M snapshot exposed an implicit online-spatial default that still targeted
the full, historical workspace. No checkpoint or result from that wave is
eligible for qualification or production continuation.

Commit `977ca1e0` makes both spatial roots explicit, extends the canonical
submission audit to require and validate them whenever online spatial telemetry
is enabled, and gives the fresh qualification a distinct `preflight_r2`
namespace. The corrected seven-job wave (`8131982`--`8131988`) was submitted on
22 September 2026 and passed the canonical submitted-manifest audit. Exact
checkpoint-reload jobs (`8131991`--`8131997`) are dependency-bound one-to-one to
those runs. Production remains blocked until runtime, exact-reload, scalar,
calibration-path, and offline telemetry gates all pass. Qualification
performance does not select or remove production cells. The superseded task
heartbeat is paused while this release is actively supervised.

The deployed implementation is commit `977ca1e0` on
`codex/ca3-predictive-active-goals-20260922`. Local release checks passed:
repository-wide pre-commit, the focused IntrMotiv and online-spatial tests, 38
workflow/batch/direct tests, parsing of all 28 generated run configurations,
and launcher print-only review for both matrices. The same workflow tests and
both submitted-manifest audits pass in the authoritative NEMO2 checkout.

## Scientific matrix

The production matrix contains the seven declared architectures at seeds 8, 99,
and 123, for 21 fresh 300M-frame runs. All arms use the stored DDQN+HER controller,
cadence 2048, F64 capacity, navigation-eight actions, and the fixed ImageNet trunk.
DG, controller, optimizer, predictor, graph, anchors, and active-goal state start
fresh.

The primary paired contrast is `CTX_FULL_H16 - ZGOAL_FIXED_H16`. Secondary
contrasts isolate the complete mechanism, action conditioning, H32 versus H16,
continuous fixed goals, compressed worker state, and the unused predictor control.
Coverage AUC and grounded controllability are co-primary behavioral outcomes.

## Qualification contract

The qualification StudySpec is a seven-run seed-99 matrix terminating at 2M
frames, with its only checkpoint and spatial target at 2M. It must establish:

- finite canonical CA3 readout, controller, active-set, and replay telemetry;
- at least 256 positive calibration records and ready finite thresholds in every
  fixed/contextual anchor arm;
- exercised registration and confirmation-attempt paths;
- exact checkpoint restoration of model/buffers, optimizer, counters, replay,
  anchors, generations, and active masks;
- offline contextual-alias artifacts for the three contextual-hit arms.

Activation success is reported but is not a correctness gate. Thresholds are not
relaxed to force activation.

## Privileged evaluation boundary

Training, activation, recognition, replacement, routing, rewards, and target
selection never receive coordinates. The frozen manifest-driven evaluator may use
pose only after rollout to count eight-connected spatial components of recognized
active-goal samples. It reports component count and the fraction outside the
largest sample-mass component. This is an offline alias diagnostic, not a
reconstruction of historical confirmation correctness.

## Canonical artifacts

- Schema: `intrmotiv/study/v1`
- Workflow: `1.10.1`
- Qualification: `hpc_runs/studies/ca3_predictive_active_goals_preflight.study.json`
- Production: `hpc_runs/studies/ca3_predictive_active_goals_production.study.json`
- Runtime audit: `hpc_runs/audit_ca3_predictive_active_goals_preflight.py`
- Active workspace: `/work/classic/fr_xl1014-corridor-geometry`
- Bulk outputs: `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/`
- Submitted qualification manifest:
  `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/ca3_predictive_active_goals_20260922_preflight_r2/977ca1e0_submitted/jobs.tsv`
- Qualification StudySpec SHA-256:
  `2f27ab21075d753efb16aefdc75256b8f6e9915987e69bb7d8819ea537a85b7b`
- Production StudySpec SHA-256:
  `74346c29bb7e065082e1cf6682943f0b0f366924c17fb11d5d55ac16da2467f5`

Record the final StudySpec hashes, deployed Git commit, print-only manifest,
submitted `jobs.tsv`, submission audit, exact-reload certificates, telemetry
manifest, and qualification audit here after each corresponding gate completes.

## Reusable execution lesson

Study validation alone did not make this batch launch-ready. The inherited
preflight declared unreachable production checkpoints, pooled architectures in
analysis, reused an old W&B group, and omitted the new scalar namespace. For a
new architectural batch, validate the runtime telemetry contract and generated
analysis cells before treating a syntactically valid StudySpec as releasable.
