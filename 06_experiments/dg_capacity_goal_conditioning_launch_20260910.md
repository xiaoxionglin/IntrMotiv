# DG capacity and goal conditioning: implementation and launch

Status: implementation in progress; nine preflights running. Production has
not been submitted. The user authorized pursuing the plan through verified
production startup. Downstream transfer remains deferred.

## Declarative study

- Production: `hpc_runs/studies/dg_capacity_goal_conditioning.study.json`,
  27 runs, schema `intrmotiv/study/v1`, workflow 1.6.0.
- Current production SHA-256:
  `eeb8bafd190ceb15da1fa6066532cb5f44b20c742cc4c1b9fb56b698fe01f47e`.
- Preflight: `hpc_runs/studies/dg_capacity_goal_conditioning_preflight.study.json`,
  nine runs, same schema, declared workflow 1.5.0 (compatible with 1.6.0).
- Current preflight SHA-256:
  `dbbfc58291a8e76197f94f3e6208c2324f92234510055733796a440af5490069`.

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

- 294 IntrMotiv tests passed on NEMO2; subsequent summary-only evaluator
  additions passed both focused intervention tests.
- 30 canonical/study tests passed locally and on NEMO2, including the
  54-row two-checkpoint intervention inventory.
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

## Final preflight and evaluator evidence (in progress)

The first diagnostic submission (8048750–8048758) was cancelled after it exposed
missing frame-zero checkpoints. Its study definition is archived in its submitted
directory as `study.reviewed.json`; its outputs remain diagnostic only.

Final preflight jobs **8048800–8048808** use the separate
`intrmotiv_dg_capacity_goal_conditioning_preflight2_20260910` namespace.
Submitted metadata: `_slurm/intrmotiv_dg_capacity_goal_conditioning_preflight2_20260910/20260910T171148Z`.
All nine preserved true frame-zero checkpoints. The submitted matrix audit passed.
Progress audits show finite learning, unchanged frozen trunk tensors including
normalization buffers, trainable DG projections, nonzero modulation training,
and successful waypoint validation/routing in all three waypoint cells.
The complete 2M-frame gate remains pending; no production submission yet.

Real DMLab evaluation smoke **8049046 completed, exit 0**. Its workspace output
is `analysis/dg_capacity_preflight/evaluation_smoke_8049046/`. It verified exact
observation-panel replay, command-invariant canonical DG, goal-sensitive worker
DG, and alternate commands from identical physical/recurrent starts with frozen
policy and graph. Its tiny panel had two paired comparisons and zero arrival
lift; this is infrastructure verification, not evidence of learned control.
Subsequent summary additions report unsupported panels, coverage, timeout, and
initial action total variation explicitly; both intervention tests passed.

Permanent frame checkpoints now use the established `checkpoint_p0/milestones`
format at the first batch crossing each requested target. Initial checkpoints
use a separate `initial_` prefix outside rolling retention. The focused tests
verify crossing, no repeated saves, and resume behavior. This retention-only
change does not alter learning, and was added after the final preflights started.

Canonical workflow **1.6.0** extends intervention manifests to multiple targets
without changing existing row fields. Production uses 75M and 300M across all
27 runs. The compatible 1.5.0 preflight study is unchanged.

Final production print-only review:
`_slurm/intrmotiv_dg_capacity_goal_conditioning_20260910/20260910T173152Z`.
Its audit confirms 27 exact commands, final study SHA, and workspace-only output
paths. Reviewed scripts request **40 CPUs, 80G, 96 hours, partition genoa**.
The Milan preflight was substantially slower than Genoa preflights; production
uses the faster architecture while preserving scientific settings. A 96-hour
allocation is not a promise that 300M frames finish on every node. Any run that
reaches the allocation limit before 300M must resume its existing checkpoint;
it must not be counted as a completed 300M replicate.

At 19:35 CEST, job 8048800 was checkpoint-preservingly requeued from Milan to
Genoa using `scontrol requeuehold`, partition update, and release. Its config
explicitly sets `restart_behavior=resume`; the saved checkpoint was at 720,896
frames. The same job ID, training directory, seed, and scientific arguments are
retained. Pre-requeue logs and checkpoint SHA are in the submitted directory's
`requeue_8048800/`. This resource change is additional scheduler provenance;
the original generated script still records its original CPU partition.

Retained frame-zero comparisons also confirm that all 113 common state tensors
are exactly equal between worker-only and each DG-conditioned arm at every
capacity (six paired comparisons). The new modulator alone adds identity-zero
parameters. Evidence: `initial_pair_comparison.json` in the final-preflight
submission directory.

The final evaluator review corrected two outcome-bookkeeping cases: arrivals
after a target deadline are timeouts, and an observed on-time success is not
censored by a later episode termination. All four matched-intervention tests
pass, including concrete delayed-arrival and early-terminal environments.
These changes affect evaluation summaries only; training is unchanged.

Eight preflights passed every runtime gate by 20:02 CEST. The final waypoint-F64
preflight was requeued at 20:03 CEST from **1,572,864 frames**, preserving job
8048808 and its training directory, now in `genoa`. At its measured throughput,
the one-hour training ceiling would stop it below 2M. Requeueing before 1.6M
also leaves enough post-resume frames to refill the 100k-sample telemetry ring
before the 2M snapshot. Its pre-requeue logs and checkpoint SHA are archived in
`requeue_8048808/` beside `jobs.tsv`. The runner's elapsed-training timer resets
on resume; model/optimizer progress is restored. Scientific arguments remain
unchanged. Final completion/audit remains pending.

The final source archive now contains all **153 relevant runtime Python files**,
not only the 13 modified files; every archived file hash matches NEMO2. The
baseline-relative patch remains scoped to this implementation. The final
print-only directory also retains the runtime Git revision and tracked runtime
diff, preserving existing source changes without reverting them.

The legacy online goal-action sensitivity/TV diagnostic changes the worker FiLM
command while holding replayed memory fixed. In DG-conditioned arms it therefore
measures the **readout-only** response, not the total response through DG writes.
Retain its historical metric contract, label that limitation in comparisons,
and use forced-goal observation replay and matched-command interventions to
measure the complete DG-plus-worker pathway. No goal-discrimination loss is
introduced by these no-gradient diagnostic forwards.
