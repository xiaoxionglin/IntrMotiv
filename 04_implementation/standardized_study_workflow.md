# Standardized IntrMotiv Study Workflow

## Status

Current implementation: **1.8.1** (local; original shared NEMO2 checkout **1.7.1**); study schema:
**`intrmotiv/study/v1`**. Canonical code: `hpc_runs/intrmotiv_study/`.
Reference study: `hpc_runs/studies/graph_stabilized_recruitment.study.json`

The tested NEMO2 runtime copy is under
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/hpc_runs/intrmotiv_study/`, with
study files in the adjacent `hpc_runs/studies/` directory. The vault copy is
the versioned source of truth; synchronize the runtime copy and run its focused
tests whenever the implementation changes. Consult `LATEST.md` for the current
deployment status before cluster use.

This is the default workflow for new training batches, repeated online
analysis, and place-field telemetry. It preserves the existing Sample Factory
launcher and the established NEMO2 telemetry evaluator as execution backends.
The study specification is the shared source of truth above both backends.

## Why this exists

Previously, reusable launchers performed the expensive work, but each batch
still recreated its condition matrix, run-name parser, terminal-window loader,
aggregation, seed-paired contrasts, checkpoint manifest, and provenance
metadata. That duplicated code and allowed training, analysis, and telemetry
to disagree about condition identities.

The v1 workflow moves repeated mechanics into one package. A new study should
normally contribute:

1. one reviewed `*.study.json` file;
2. optionally, a thin Sample Factory experiment module;
3. only genuinely novel scientific diagnostics or plots.

## Codex-efficiency rule

The main optimization target is repeatable agent work, not cluster runtime. A
future task should not re-inventory historical launchers and analyzers before
ordinary batch work. It should:

1. read `hpc_runs/intrmotiv_study/LATEST.md` and the selected `*.study.json`;
2. run the canonical CLI commands for validation, submission audit, analysis,
   and telemetry planning;
3. inspect the generated manifests and only the exceptions they report;
4. reuse standardized CSV and NPZ outputs when writing the scientific report;
5. inspect or create new implementation code only when the study cannot be
   represented by the current versioned contract.

This keeps routine work mechanical and leaves Codex reasoning for scientific
design, novel diagnostics, failures, and genuine schema extensions. Do not
repeat repository-wide searches merely to rediscover commands already captured
by this workflow.

For controller migrations, qualify the actual runtime model before creating the
study matrix. Preserve tracked modifications and required untracked modules in
an isolated source copy; save complete parent configs and checkpoint/source
hashes. Check imported module paths: missing package markers can make a local
snapshot silently import an older editable installation. Use Sample Factory's
`prepare_and_normalize_obs` for replay so instruction/action channels retain
their required integer types. Distinguish module tests, model parity, full
learner preservation, and environment preflight in qualification records. The
current worked example and remaining gates are recorded in
`06_experiments/intrmotiv_full_system_controller_20260912.md`.

## Source-of-truth hierarchy

1. The study JSON defines bases, factors, seeds, run names, arguments, metrics,
   contrasts, and telemetry protocol.
2. `StudySpec` validates and expands it. Its SHA-256 fingerprint identifies the
   exact reviewed definition.
3. Sample Factory remains authoritative for generated training commands and
   Slurm submission records.
4. The manifest-driven place-field evaluator remains authoritative for raw
   rollout artifacts and spatial metrics.
5. Batch reports interpret generated outputs; they must not redefine the
   experimental matrix.

Never maintain a second handwritten list of the same runs. Historical scripts
may remain as thin compatibility adapters, as demonstrated by
`hpc_runs/graph_stabilized_recruitment_manifest.py`.

## Study schema

A study declares:

- `schema` and `workflow_version`;
- a stable `study_id`, description, and exact expected run count;
- complete base configurations or an explicitly marked historical
  `supplemental_args` matrix;
- ordered factor levels, including command arguments and human-readable labels;
- paired seeds;
- workspace-only training paths;
- TensorBoard metric names, analysis windows, grouping, and explicit linear
  contrasts;
- the place-field protocol, checkpoint targets, trajectory seed, terminal
  seeds, and manifest metadata.

Factor contrasts are written as explicit weighted cells. A term must select
exactly one row within each declared paired cell. The analyzer fails on an
ambiguous or missing cell instead of silently averaging over an omitted
factor.

`training.mode` has two values:

- `sample_factory`: the specification contains the complete command and can be
  converted directly into a `RunDescription`;
- `supplemental_args`: a historical or transitional specification whose base
  command bundles live elsewhere. The Sample Factory adapter deliberately
  refuses to submit it as a complete study.

New studies should use `sample_factory`.

If a reusable Python base factory cannot be represented as a complete argument
bundle, `build_run_description(STUDY, experiment_builder=...)` is the supported
extension point. The callback may construct the base command but must preserve
the study-generated names and order. Matrix expansion must not move back into
the adapter.

The machine-readable structural schema is
`hpc_runs/intrmotiv_study/study.schema.json`. Runtime validation remains
authoritative because it also checks Cartesian-product counts, rendered-name
uniqueness, template fields, and workspace containment.

## Standard lifecycle

### 1. Define and validate

Create the study under `hpc_runs/studies/`, then run:

```bash
python -m hpc_runs.intrmotiv_study validate \
  hpc_runs/studies/my_study.study.json

python -m hpc_runs.intrmotiv_study render-runs \
  hpc_runs/studies/my_study.study.json \
  --output /tmp/my_study_runs.json
```

Review the count, unique names, arguments, paths, workflow version, and study
fingerprint. Preserve the rendered plan with the experiment record or NEMO2
submission metadata.

### 2. Connect to Sample Factory

A new NEMO2 experiment module should be only a thin adapter:

```python
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).with_name("my_study.study.json")
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
```

Use the established launcher for print-only review and submission exactly as
described in `BATCH_SUBMISSION.md`. The workflow package does not call `sbatch`
for training and does not replace `jobs.tsv`, `submission.json`, or
`scancel.sh`.

If the fingerprint changes after print-only review, repeat the review before
submission.

Audit either the print-only or submitted launcher manifest against the same
study. Add `--submitted` after a real submission to require numeric job IDs and
`submitted` status for every row:

```bash
python -m hpc_runs.intrmotiv_study audit-submission \
  hpc_runs/studies/my_study.study.json WORKDIR/jobs.tsv --submitted \
  --output WORKDIR/study_audit.json
```

The audit requires the exact run matrix, study arguments in every generated
command, matching `--experiment` names, unique job IDs, and workspace-only
training and Slurm paths.

For a shortened training preflight, validate all coupled cadence constraints,
not only the shortened frame target. In particular, an online-spatial snapshot
maximum must be at least one snapshot interval and divisible by that interval;
override the interval together with the maximum when the production cadence is
longer than the smoke run. Rerun a failed preflight in a new batch/output
namespace so incomplete event files or configs cannot be mistaken for the
corrected run. A strict legacy-checkpoint smoke should instantiate the model and
advance frames; print-only argument validation cannot exercise state-dict
compatibility.

### 3. Collect and analyze online metrics

From the NEMO2 source checkout:

```bash
python -m hpc_runs.intrmotiv_study collect-online \
  hpc_runs/studies/my_study.study.json \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/MY_BATCH \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/MY_ANALYSIS
```

Use `--window-low` and `--window-high` together for a synchronized window.
Without them, each run uses its declared terminal window.
TensorBoard histories are loaded with a bounded thread pool; set
`analysis.max_workers` in the study when the default of four is inappropriate
for the filesystem or event volume.

Standard outputs are:

- `per_run.csv`;
- `condition_summary.csv`;
- `paired_contrasts.csv`;
- `paired_contrast_summary.csv`;
- `analysis_manifest.json`, including schema, version, and study fingerprint.

`analyze-csv` applies the same validation and statistics to an existing
standardized `per_run.csv`. It requires exactly one row for every declared run.

### Latest shared-step comparisons (1.7.0)

Optional `analysis.loader_backend: "process"` in locally staged 1.8.0 uses
spawned workers bounded by `analysis.max_workers`; the default is `"thread"`.
The CLI reports each finished run, and the manifest records the backend.
A September 11 full-batch threaded scan remained at one CPU after about 30
minutes and was stopped without results. Process loading addresses Python
thread contention, but full-batch speedup is unmeasured. Forty-three focused
tests pass locally, including real-event backend equivalence; deploy and test
on NEMO2 before use. Changing this study analysis option requires the usual
fingerprint preservation, print-only review, and submission audit.


Directory discovery in 1.7.1 recognizes `RUN_/00_RUN` as a launcher container
and one actual experiment when the container has no experiment payload. It
still rejects genuine duplicate runs. This fixes the DG-capacity batch's 27
false duplicate matches; 41 canonical/common-window/study tests passed locally
and on NEMO2.

Before a multi-gigabyte `--latest-common` scan for a new study, verify declared
tags against enough initial events to include episode and online-spatial
summaries (the first few batches are insufficient). In the DG-capacity study,
graph summaries use `intrmotiv/hrl/summary/…`, and the across-arm replay check
is `intrmotiv/hrl/behavior_replay_mismatch`; the memory-only replay tag is absent
in worker-only runs. Group by both base arm and capacity. Correcting these
analysis declarations changes the StudySpec fingerprint, so preserve the
launch fingerprint, regenerate print-only rows, and audit them against the
original submission to prove training commands are unchanged.

Failed strict scans currently wait for other threaded loads and discard their
histories. This caused expensive repeat reads during the September 11 interim
analysis. A future general improvement should validate tag coverage early and
cache selected histories with event-file provenance, while preserving exact
common-window semantics; do not recreate event readers in each report.

For repeated “check again at the latest steps” requests, use:

```bash
python -m hpc_runs.intrmotiv_study collect-online \
  hpc_runs/studies/fixed_reward_transfer_repeat8.study.json \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_fixed_reward_transfer_repeat8_20260910 \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/frt8_latest_comparison \
  --latest-common
```

The high endpoint is the minimum latest event step across all declared runs'
step and metric histories. The low endpoint is `max(0, high - terminal_width)`.
For this study the width is 10M environment frames. This is global environment
progress, not optimizer updates or wall time. The command rejects a simultaneous
explicit window, missing required histories, and empty/nonfinite window means.
It retains all scalar events in this mode (no reservoir sampling), loads each
event directory once, and retains only the declared histories for aggregation.
The resulting means remain event-sample means, as in the existing collector.

`per_run.csv` records each run's actual maximum and the exact common window;
`analysis_manifest.json` records collection mode, step tag, scalar sampling,
schema, workflow version and StudySpec SHA-256. Use a new output directory for
each snapshot that must be preserved. For condition-level interpretation, use
StudySpec condition/base identities and pair seeds against the declared control;
check the study's `analysis.group_by` before interpreting `condition_summary.csv`
(an empty list means an overall summary).

Reusable lesson: the previous transfer check read all 21 histories twice to
first discover endpoints and then collect a fixed window. This repeated I/O and
parsing took minutes and sampled long histories under the old scalar cap.
Prefer the single-pass common-window option; use recorded TensorBoard event
steps, not checkpoint filenames or W&B run ordering, as the alignment evidence.
A shared endpoint does not establish convergence or account for pretraining cost.

### 3a. Collect compact online spatial snapshots

On NEMO2, invoke the documented training/analysis interpreter directly:
`/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python`. The login default Python
may lack NumPy; `dmlab0` uses Python 3.8 and cannot import this workflow.
For late-training outlier audits, screen cached per-run metrics first, then
validate candidates with these longer snapshots. Keep latest-10k W&B scalars
separate from latest-100k snapshot metrics, and inspect selected maps: many
unique peak bins or a high single-field fraction can still describe several
units concentrated in one region. See the
[2026-09-08 outlier audit](../06_experiments/late_training_outliers_20260908.md).

Training stores scalar-only W&B monitoring over the latest 10k samples plus
compressed latest-100k behavior snapshots at 5M, 25M, 50M, 75M, and 100M
frames. In addition to aligned raw
behavior data, new v1 snapshots cache evaluator-compatible maps, multilevel
field components, complete available graph buffers, prospective edge outcomes,
and deterministic graph diagnostics. Analyze them without rendering the full
batch:

```bash
python -m hpc_runs.intrmotiv_study collect-spatial \
  hpc_runs/studies/my_study.study.json \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/online_spatial/MY_BATCH \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/MY_SPATIAL_ANALYSIS
```

The default is CSV-only and writes `per_snapshot.csv`,
`snapshot_inventory.csv`, `condition_summary.csv`, `seed_summary.csv`, and an
`analysis_manifest.json` containing workflow version, study SHA-256,
completeness, and exact paths. Add `--require-complete` for a final batch audit.

Figures require explicit exact run names; optional targets restrict them
further:

```bash
python -m hpc_runs.intrmotiv_study collect-spatial STUDY SNAPSHOT_ROOT OUTPUT_DIR \
  --plot-run GSR_C05_D4_H5K_S99 --plot-target 25000000
```

This renders paginated thresholded-DG contact sheets and a segmented
occupancy/trajectory panel under `OUTPUT_DIR/figures/`. It never uploads them.
The controlled manifest-driven checkpoint rollout remains the authoritative
scientific place-field and map-stability evaluation.

Add `--include-details` to write `per_unit.csv`, `per_field.csv`, and
`graph_edge.csv`. Cached maps and graph scores are validated against raw/map
and graph-buffer recomputation before those tables are accepted; no DMLab
rollout is run. Older v1 snapshots without optional cached details remain
loadable and their spatial details are recomputed from the retained raw arrays.

### 4. Generate place-field telemetry manifests

After the required checkpoints exist, run from the authoritative NEMO2
checkout so `build_place_field_sweep.py` is importable:

```bash
python -m hpc_runs.intrmotiv_study render-telemetry \
  hpc_runs/studies/my_study.study.json \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/MY_BATCH \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/MY_TELEMETRY
```

This discovers each declared run by exact name, reuses the authoritative
checkpoint selector, enforces workspace paths, and writes:

- `analysis_manifest.tsv`;
- `trajectory_manifest.tsv`;
- `intervention_manifest.tsv` when the study declares the compatible
  `target-control-intervention-v1` protocol;
- `study_manifest.json` with the study fingerprint.

Since 1.5.0, `telemetry.intervention.where` can select an exact subset using
RunSpec context (for example `{"cell_control": "goal"}`). Unknown fields and
empty selections fail. The selected checkpoint is included for every selected
seed, independently of the smaller ordinary place-field seed subset. Omitting
`where` preserves all-run intervention behavior. This avoids duplicate run lists
and ensures extra goal-probe seeds use the same checkpoint inventory.

Since 1.6.0, `telemetry.intervention.target_frames` may contain multiple sorted,
unique positive frame targets. The manifest requires exactly one row per
selected condition, seed, and target. Test manifest generation with a synthetic
checkpoint inventory before training: study expansion alone cannot demonstrate
that a new telemetry protocol is executable. The DG-capacity study exercises
all 27 runs at both 75M and 300M, producing 54 intervention rows.

Then use `evaluation/submit_place_field_sweep.py` for its required print-only
preflight and ordinary-job production submission. Postprocess with the existing
summarizer, manifest analyzer, trajectory plotter, and stability tool documented
in `reusable_place_field_telemetry.md`.

### Weights-only transfer runtime gates

Validate the actual source config against the destination's frame repeat and
ordered action interface before launching. Compatible tensor dimensions do not
establish compatible control semantics. For a timing-only rerun, compare every
expanded command against its parent, allowing only the declared timing and
namespace changes. `hpc_runs/audit_fixed_reward_transfer_preflight.py` derives
repeat from RunSpec and supports source-interface checks for the repeat-8
transfer study.

For a StudySpec that initializes selected modules from an older checkpoint,
hash the declared source checkpoints and run one ordinary-job preflight per
design cell before production. The preflight must exercise the actual launcher
and old checkpoint format, check the declared load scope, and compare terminal
tensors exactly for every frozen module. Freezing a normalization layer includes
its running statistics; setting only its parameters to `requires_grad=False` is
insufficient when the module remains in training mode.

Do not assume a frame-zero checkpoint exists. If the backend first saves after
training has begun, verify selective initialization and fresh excluded modules
with focused tests and explicit runtime load logs, then use terminal checkpoints
only for properties that must remain invariant. Record this limitation in the
study report rather than treating a tuned terminal tensor as evidence of its
initial value.

## Extending the standard

When a later task needs another repeated capability:

1. check this package and guide first;
2. add a general component here with focused tests;
3. make new fields optional when the old interpretation is unchanged;
4. bump the implementation minor version for a compatible feature or patch
   version for a fix;
5. update `LATEST.md`, `version.py`, this guide, and a real reference study;
6. retain existing artifact meanings and study-file validity.

A study authored against an older compatible 1.x implementation remains
loadable. A study requiring a newer implementation, or one with a different
major workflow version, is rejected rather than partially interpreted.

Study-specific visualization or causal analysis may remain outside the package,
but it should consume standardized CSV/NPZ outputs and must not reimplement
run discovery, checkpoint selection, path validation, or basic aggregation.

Create a separate workflow only when the study cannot reasonably preserve the
v1 run, analysis, or telemetry contracts. Record the concrete incompatibility
and either add a versioned adapter or introduce a new schema with migration
notes. “This batch is unusual” is not by itself sufficient reason to fork the
core.

## Current boundaries

### Future graph-planning runtime prerequisite (2026-09-11)

Before the next new batch using topological/waypoint planning, incorporate the
[validated graph-planning optimization](graph_planning_optimization_20260911.md)
into that batch's source checkout and rerun its focused tests. It is currently
staged, not deployed to the active DG-capacity batch. Preserve the source
revision/diff in the ordinary print-only review and measure end-to-end FPS in
the usual Slurm preflight; synthetic graph-function timings are not training
throughput measurements. This source-only optimization does not change the
StudySpec or workflow contract.

Cache derived graph quantities by their actual effective inputs: current
policy synchronization may bypass tensor mutation counters. Check exact
selection and recurrent-state equivalence against archived source, including
graph updates and checkpoint reloads. For FPS diagnosis, inspect 60-/300-second
rates and cumulative frames; 10-second rates can read zero between learner
counter updates.

Version 1.x standardizes definition, collection, analysis, and telemetry
planning. Slurm submission, monitoring, cancellation, and raw place-field
execution intentionally remain with the established launchers. Generic figure
recipes and automated report/index assembly are suitable next components, but
scientific conclusions should continue to require explicit review.

### Goal-conditioned memory deployment lessons (2026-09-10)

Once a command affects recurrent writes, force it before the core during both
learner replay and interventions. A decoder-only override tests a different
policy. Keep canonical detection separate, and verify alternate commands from
identical observations and recurrent starts. Reuse the observation-panel and
matched-command evaluators; the real DMLab smoke passed on an ordinary Slurm job.

Explicit frame-zero checkpoints and permanent frame milestones must be requested;
rolling and wall-clock saves do not guarantee the planned evaluation inventory.
The runtime supports `save_initial_checkpoint` and `checkpoint_frame_targets`.
Save each target at the first learner batch crossing it, preserving actual frame
counts in standard milestone filenames. Resume does not backfill earlier targets.

Check telemetry aliases before interpreting an absent audit signal: validation
success is `intrmotiv/hrl/validation/success_rate`, while waypoint routing remains
a raw `train/hrl_planning_waypoint_navigation_fraction` tag. Use initial/terminal
tensor comparisons to verify frozen visual trunks including normalization buffers.

For a time-limited preflight resume, check the telemetry ring as well as the
checkpoint. The current ring is not checkpointed. A retained-100k snapshot at
repeat 4 needs at least 400k fresh valid frames after resume; resuming too close
to a snapshot/terminal target may leave that artifact pending even when the
model reaches its frame target. Preserve old logs and checkpoint hashes when
requeueing, and inspect the actual runner's timer semantics rather than assuming
wall-clock progress is restored with optimizer/model state.

### Recurrent off-policy preflight lessons (2026-09-11)

Derive preflight length from aggregate decisions and all delayed cadences.
With 16,384 warmup decisions, one update per 64 decisions, and a target copy
every 1,000 updates, a 250k-frame/repeat-4 preflight cannot reach the first copy.
The DDQN diagnostic uses 500k frames and checks physical resets separately.

Keep multiprocessing TMPDIR paths short while staying inside the allocated
workspace. The first isolated DDQN jobs (8055861/8055862) failed at forkserver
startup with `AF_UNIX path too long`; `/work/classic/fr_xl1014-train/tmp/ddqn_JOBID`
leaves room for Python's generated socket suffix. Preserve the failed artifacts
and use a new run namespace after corrections.

When deploying canonical tests into an isolated runtime, include their reference
StudySpecs. A missing `graph_stabilized_recruitment.study.json` caused fixture
errors on the initial copy; the unchanged tests passed after copying fixtures.
Use the [DDQN batch record](../06_experiments/intrmotiv_ddqn_her_implementation_20260911.md)
for exact source, tests, manifests, and the distinction between frozen-reference
control and future adaptive-DG work.

The DDQN production startup records subsequently measured about 0.123–0.127 s
for prefix reconstruction plus batch preparation and 0.020–0.031 s for the
learner update. Future optimization should batch independent prefix histories,
verify full-history and target-network equivalence, and measure the same
components again. This is a measured infrastructure bottleneck, not evidence
that the scientific loss should change; do not patch active jobs mid-comparison.

### Full-system replay qualification (2026-09-12)

Before finite actor-history qualification, check SF's
`decorrelate_envs_on_one_worker`: its startup random-action walk happens before
inference sees any observations. The first full-system controller preflight
correctly failed with `Actor history missing at non-reset decision`. Use the
existing switch set to `False` consistently across comparison arms when the
contract requires the first policy decision at a physical reset; retain ordinary
worker startup delays. Record this explicit startup difference in StudySpec and
parent-delta classification. Do not weaken the missing-history check or silently
clear memory. Preserve failed artifacts and use a fresh batch namespace.

For isolated checkouts, inspect the rendered Slurm script, not only its command
arguments: the historical template hardcoded the original source directory.
`SLURM_SUBMIT_DIR` keeps execution in the reviewed checkout when the canonical
launcher is invoked from that root. Keep multiprocessing TMPDIR short and under
the allocated workspace. NEMO2 lacks `rg`; after that discovery use `grep` for
remote logs instead of repeatedly retrying ripgrep.

Profile full requested replay updates including backward before claiming batching
is faster. The full-system 256-position waypoint test spent most CPU time copying
and zeroing gradient tensors. One indexed history gather and internal 16-history
batches reduced measured cost without changing the 256-position mean loss or
optimizer clock. This is a desktop measurement; the 2M Slurm gate must establish
actual cluster throughput and HER overhead. Preserve plain/tensor replay
checkpoint serialization for the established weights-only place-field loader.

For controller runtime audits, call the established evaluator's
`place_fields.load_checkpoint_dict`. Bare PyTorch `weights_only=True` rejects
original PPO NumPy scalar metadata after best-performance reporting starts;
the existing loader already has the narrowly scoped allowlist. Check live
counters and required frame completion before repeatedly loading large replay
checkpoints. A zero HER count before the manager has any active landmarks or
positive option budgets is an unmet qualification gate, not by itself a replay
transport bug. Confirm both context labels and fresh DG parameter changes before
changing the scientific objective.

## Controller qualification lessons (2026-09-12)

Use both Slurm log streams and generic exception/traceback detection: a crashed
learner can leave the enclosing job RUNNING, and a fixed exception-name list
misses `torch.OutOfMemoryError`. Long replay searches must select compatible
examples under no-grad before constructing the fixed TD batch, otherwise one
accepted example can retain each large candidate batch's backward graph.
Reuse SF's existing heartbeat signal at completed-work boundaries for long
controller transactions; do not disable the watchdog or send unconditional
background heartbeats that could hide a blocked operation.

A checkpoint's learner-init equality certificate should be bound to its immutable
file SHA and include PPO too. Completed bounded PPO runs need exact reload proof,
not extra training past their horizon. SF's worker SIGINT handlers and
`LearnerWorker.on_stop` provide the controlled full-checkpoint shutdown path;
verify that path and exact reload evidence before maintenance. The worked record
is `06_experiments/intrmotiv_full_system_controller_20260912.md`.

The first real `render-telemetry` qualification exposed legacy hardcoded
checkpoint targets. Version 1.8.1 passes StudySpec targets into the existing
selector, preserving legacy defaults. Test the actual runtime bridge, not only
manifest construction from a fabricated inventory. In isolated source checkouts,
the evaluator shell must resolve its own source and put that root on PYTHONPATH.
Safe private mmap reduces CPU audit RSS for replay-bearing checkpoints, but
Python metadata parsing still dominated the measured load time (50–55 seconds);
avoid repeating full checkpoint audits when compact counters suffice.
