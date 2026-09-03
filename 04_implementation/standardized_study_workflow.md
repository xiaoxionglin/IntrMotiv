# Standardized IntrMotiv Study Workflow

## Status

Current implementation: **1.2.0**; study schema:
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

Then use `evaluation/submit_place_field_sweep.py` for its required print-only
preflight and ordinary-job production submission. Postprocess with the existing
summarizer, manifest analyzer, trajectory plotter, and stability tool documented
in `reusable_place_field_telemetry.md`.

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

Version 1.x standardizes definition, collection, analysis, and telemetry
planning. Slurm submission, monitoring, cancellation, and raw place-field
execution intentionally remain with the established launchers. Generic figure
recipes and automated report/index assembly are suitable next components, but
scientific conclusions should continue to require explicit review.
