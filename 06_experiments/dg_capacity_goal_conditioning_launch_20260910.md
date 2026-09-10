# DG capacity and goal conditioning: implementation and launch

Status: implementation in progress; nine preflights running. Production has
not been submitted. The user authorized pursuing the plan through verified
production startup. Downstream transfer remains deferred.

## Declarative study

- Production: `hpc_runs/studies/dg_capacity_goal_conditioning.study.json`,
  27 runs, schema `intrmotiv/study/v1`, workflow 1.5.0.
- Current production SHA-256:
  `53197eede2cf4183a1546bf2b7320c8e58a593b558f1a9673973897680bb5ed7`.
- Preflight: `hpc_runs/studies/dg_capacity_goal_conditioning_preflight.study.json`,
  nine runs, same schema/workflow.
- Current preflight SHA-256:
  `7cb93f9e133d18f7582cdccb231a1d5746f8250b665cfb55f887420eff775e95`.

The matrix is direct worker-only, direct DG+worker, waypoint DG+worker, crossed
with 16/32/64 DG and seeds 8/99/123. All use repeat 4, navigation8, PPO STOP into
base DG, no replacements, and a 300M-frame production target.

## Implemented runtime

`GoalConditionedDGCore` keeps canonical DG/CA3 and graph evidence unchanged,
with a separate worker trace from goal-modulated preactivations. It reuses one
projection/BatchNorm evaluation. The policy tail reads the worker trace;
encoder objectives read the canonical prefix. A small generic learner hook
supplies action-aligned goals before recurrent packing. The stored behavior
descriptor remains the persistent state suffix.

The waypoint study explicitly enables existing passive discovery and deliberate
edge validation. The implementation existed, but parser validation formerly
allowed edge exploration only for `control_graph`; it now also permits
`frontier_waypoint`. Direct controls are unchanged. No timeout exclusion,
goal discrimination, compact embeddings, or balanced selection was added.

Runtime modifications were context-applied on top of the existing dirty NEMO2
checkout. Baseline and work snapshots are under `/tmp/dg_capacity_runtime/`
locally. Versioned patch/source snapshots are under `hpc_runs/source_snapshots/`
with the `dg_capacity_goal_conditioning_20260910` stem; refresh after further
changes. No unrelated source edits were reverted.

## Verification so far

- 289 IntrMotiv tests passed on NEMO2, including four new goal-memory tests.
- 29 canonical/study tests passed locally and on NEMO2.
- All 36 production/preflight commands parsed through the actual entry point.
- Synthetic full-model smoke passed with actual encoder, core, and decoder;
  new modulation gradient norm was 2.00045 in its forced-active test case.
- Tests cover identity initialization, unconditioned detector invariance,
  packed/plain replay, goal switches, gradients stopping at base inputs,
  checkpoint restoration, and all three state sizes.

The full regression run first exposed a minimal actor fixture without `cfg`;
the tail now detects the core capability directly. The suite passed after that
compatibility fix. Runtime learning and scientific preflight gates remain to
be checked.

## Preflight submission

Workspace train root:
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/`.

Print-only directory:
`_slurm/intrmotiv_dg_capacity_goal_conditioning_preflight_20260910/20260910T165529Z`.
Submitted directory:
`_slurm/intrmotiv_dg_capacity_goal_conditioning_preflight_20260910/20260910T165742Z`.
Both canonical submission audits passed. The submitted `jobs.tsv` is the
authoritative mapping; jobs 8048750–8048758 were all running at the initial
scheduler check, without startup tracebacks in stderr.

Preflights request two hours and target 2M frames with a one-hour training-time
ceiling. Checkpoint retention is 100 to preserve frame-zero evidence. Logs for
print, submit, and regression tests are in the workspace analysis directory
with the `dg_capacity_` prefix.

## Remaining gates

Verify real learning, finite metrics, nonzero new-modulation training, retained
checkpoints, independent detection, telemetry, and exercised waypoint
validation/routing. Finish compatible evaluation/diagnostic additions and
record their tests. Then review production scripts and all workspace paths,
submit, audit exact job membership, and verify every production job starts.

Reusable lesson: inspect the actual model payload and recurrent replay path
before adding goal input to memory. Overriding only decoder goals after replay
is insufficient once the goal changes memory writes. Preserve separate
grounding and worker features rather than allowing a command to create its
own achievement event.
