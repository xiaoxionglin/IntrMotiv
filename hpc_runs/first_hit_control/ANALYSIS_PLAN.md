# Frozen-replay analysis plans

The canonical [analysis sidecar](../studies/fixed_goal_first_hit_c05_s99_20261127.analysis.json) defines the execution settings. Its validator loads the unchanged donor through `intrmotiv_study.load_study`, checks its canonical fingerprint, and selects one existing `RunSpec` by declared condition and seed.

This uses a separate versioned schema because the artifacts form a dependency graph: source assets → factual frozen replay → exposure report → copied readout → frozen fit diagnostics → matched evaluations sharing that readout. The standard `intrmotiv/study/v1` schema expands independent Cartesian Sample Factory training jobs. Inventing training `RunSpec` rows for dependent analysis stages would misstate their identities and provenance.

From the repository root:

```bash
python -m hpc_runs.first_hit_control.analysis_plan validate hpc_runs/studies/fixed_goal_first_hit_c05_s99_20261127.analysis.json
python -m hpc_runs.first_hit_control.analysis_plan render hpc_runs/studies/fixed_goal_first_hit_c05_s99_20261127.analysis.json
```

Both commands are read-only and print JSON. Rendering emits argument arrays, shell-quoted display strings, artifact dependencies, environment paths, computed budgets and the effective plan fingerprint. It never submits a job or imports a native environment. An explicit `--source-release /work/...` override changes the effective fingerprint while preserving the original plan fingerprint; regenerate the print-only review before execution.

The validator checks donor/config/checkpoint pins against the qualification protocol, complete disjoint source shards, the evaluator's discovery reserve, worst-case matched-block budgets, workspace paths, and agreement with fixed options supported by the frozen fit CLI. These static checks do not inspect remote file bytes, symlinks, available storage or completed runtime qualification. Those remain execution gates. Resource reservations may use qualification timing; scientific settings remain fixed before efficacy outcomes.

Use ordinary allocated Slurm jobs and the existing canonical runfiles builder. A launcher must preserve the rendered dependencies and argv, bind the immutable source release, set short workspace temporary paths, and retain submission/output receipts. No new scheduler backend is provided here. Exposure counts are descriptive; source-DG/action counts are not a certificate of support at every recurrent state.

`render-slurm` adds reviewable shell artifacts without submitting anything:

```bash
python -m hpc_runs.first_hit_control.analysis_plan render-slurm hpc_runs/studies/fixed_goal_first_hit_c05_s99_20261127.analysis.json --release-receipt 06_experiments/controllers/fixed_goal_first_hit_20261127/runtime/remote_release_r2_qualification.json --output-directory NEW_DIRECTORY
```

The bundle contains the effective protocol, rendered command JSON, source-release record, one sequential collection/support/fit/diagnostics script, four evaluation scripts, and a manifest mapping exact argv and SHA-256 values to every script. The evaluator jobs require `afterok` dependencies on the actual collection/fitting job ID. Create the manifest's remote launch/log directories and upload the bundle before submission; the renderer does neither. Review the completed integration qualification first. The staged runtime remains unchanged and does not need to contain this local renderer.

Scripts require a Slurm allocation, verify source-release and donor identities before native imports, check resolved workspace paths, use short workspace temporary paths, and refuse existing stage outputs. Walltime overrides may reduce the reviewed reservation without changing scientific argv. A partial run requires a separately reviewed recovery; these scripts do not overwrite or silently resume artifacts.

After all four evaluations complete, use the model-free analysis adapters:

```bash
python -m hpc_runs.first_hit_control.aggregate_evaluation --shards SOURCE_SHARD_0 SOURCE_SHARD_1 SOURCE_SHARD_2 SOURCE_SHARD_3 --output NEW_AGGREGATE_DIRECTORY
python -m hpc_runs.first_hit_control.trace_analysis --aggregate NEW_AGGREGATE_DIRECTORY/aggregate.json --output NEW_TRACE_DIRECTORY --field-audit FIELD_REUSE_AUDIT.json --field-npz EXACT_COMMON_PANEL/place_fields.npz
```

The aggregator verifies complete disjoint source coverage, immutable trial and
source bytes, matched prefixes and the full declared command/scoring product.
It retains missing and censored comparisons as identification bounds. Its
competing-target FIRST contrast is separate from ordinary own-stop arrival
lift, which can be positive for identical goal-blind action paths.

The trace adapter reads each issued own-command path once and reuses canonical
spatial binning. It checks detector/action/time alignment, preserves missing
rows and unobserved goal capacity, and keeps terminal successors unknown when
the evaluator did not observe them. Maps count observations (including known
endpoints), not decisions, and exclude zero-action initially satisfied queries.
Factual adjacent-winner counts retain silence and observed teleports; the
existing 250-unit jump convention applies only to contiguous physical-motion
descriptors. The archived half-peak field footprint is a secondary physical
descriptor with explicit unmapped positions, not a replacement goal detector.
Historical field arrays use `[x,y,unit]`; canonical output maps use
`[unit,y,x]`, with orientation checks in the focused tests. Keep new analysis
sources separate from the immutable running native release.
