# Cross-run defaults audit — 2026-09-26

The audited runtime is GitHub `master` revision `06790d4c` (code revision
`f8fffe04`), with workflow 1.14.0. The desktop checkout and canonical vault workflow copy reflect
these changes. An isolated NEMO2 checkout passed 68 focused tests; the NEMO2
checkout used by running jobs remains on its earlier revision. The 500-decision
landmark evaluator preflight completed separately as job `8218673`.

This audit distinguishes three layers: the Sample Factory parser, the IntrMotiv
runtime, and explicit StudySpec arguments. A parser default does not override a
saved experiment configuration or a StudySpec. It describes **new runs without
an explicit setting**. Historical studies and live jobs retain their recorded
configurations.

| Fix or contract | New-run default in audited source | Scope and evidence |
| --- | --- | --- |
| Limit retained milestones | **Yes for fresh IntrMotiv runs.** Eight frame targets are spaced across `train_for_env_steps`, including the planned horizon. The controller retains those eight lightweight evaluation artifacts; non-controller learners cap automatic milestones at eight. | Shared checkpoint schedule and retention tests. Explicit frame targets and positive time-based schedules remain available; saved configurations and older StudySpecs keep their historical behavior and fingerprints. |
| Keep replay out of evaluation checkpoints | **Yes, for the controller learner.** Milestone and best artifacts set `checkpoint_role=evaluation` and `replay_included=False`. A rolling `restart` checkpoint includes replay and can resume; evaluation artifacts are rejected for resume. | `controller_learner.py::_get_checkpoint_dict` and `_load_state`; checkpoint tests. The separate standalone DDQN backend has its own checkpoint contract and was not changed by this fix. |
| Inverse depth when depth is enabled | **Yes for fresh IntrMotiv runs and version 1.13 studies.** `depth_sensor_inverse=True` selects $10/\max(d,1)$ after undoing fixed observation scaling; `False` selects pass-through. | IntrMotiv parser, `DepthEncoder`, normalizer tests. `depth_sensor` itself remains disabled by default. Saved configs retain their old response. Studies declaring workflow 1.12 or earlier render `--depth_sensor_inverse=False` when depth is enabled and the switch is absent. Explicit choices win. |
| Keep bulk telemetry in an allocated workspace | **Yes when `train_dir` is workspace-backed.** The online-spatial path resolves under `train_dir/analysis/online_spatial`; the default level cache now resolves under `train_dir/runtime/dmlab_cache`. | Parser and telemetry tests. NEMO2 submissions must still set `train_dir` to an allocated workspace and audit explicit paths. Saved and explicitly configured cache paths remain unchanged. |
| Keep reward streams causally distinct | **Implemented, but deliberately condition-specific.** The explicit `advantage_reward_source=external` setting reaches PPO and the reward-goal manager in the single-value learner. | `custom_learner.py` and reward-transfer tests. Globally selecting external reward would disable intended intrinsic-reward experiments. |
| Use the current differentiable goal in packed replay | **Yes when fixed-task goal writing is selected.** Packed replay recomputes from the current mixture instead of using a detached stored goal. | `goal_conditioned_dg.py` and goal-write replay tests. The fixed-task mechanism remains opt-in. |
| Match online snapshot targets to the study contract | **Yes for new workflow 1.14 studies.** Validation requires declared targets when online spatial telemetry is enabled and rejects mismatched explicit or automatic runtime schedules. | Focused StudySpec tests. Historical immutable studies remain readable; the earlier frozen-DG mismatch still requires its analysis-only copy. |
| Use the correct offline place-field grid for each level | **Verified on NEMO2.** The evaluator derives grain and bounds from verified geometry, yielding 19-by-19 corridor and 9-by-9 landmark maps. | Job `8218673` completed with a 9-by-9 landmark NPZ, `[100, 1000]` bounds on each axis, matching raw/pre-threshold arrays, and a valid summary. The 12-row 10k plan passed print-only review; it has not been submitted. |

The time-milestone distinction matters: the shared parser defaults to no
time-based checkpoints, while older declarative studies explicitly override
it. Fresh IntrMotiv runs without that override use the eight-point frame
schedule; a deliberate positive time interval without frame targets retains
the historical cadence. Changing a published
StudySpec in place would change its SHA-256 and undermine comparison with
already submitted runs; create a new version for any changed schedule.

At this audit, 96 of 126 checked-in StudySpecs explicitly request 1,800-second
milestones; 76 also declare fixed frame targets. All 114 that enable depth lack
an explicit inverse switch. Workflow 1.13 therefore preserves their legacy
depth response while leaving their files and fingerprints untouched. Workflow
1.14 also renders an explicit empty frame-target setting for pre-1.14 studies
that omitted it. Only new studies inherit both new defaults when omitted.

For the next broad default audit, inspect the parser, saved-config loading,
expanded StudySpec commands, and the actual learner/evaluator path together.
The authoritative checks were `test_depth_encoder.py`,
`test_controller_checkpoint_retention.py`, the geometry-grid test, and the
runtime and workflow suites. Desktop: 536 passed, 15 skipped; NEMO2 focused:
68 passed. A new parser default could otherwise change old checkpoint behavior
on resume, so saved-config and declared-workflow migrations are essential.
