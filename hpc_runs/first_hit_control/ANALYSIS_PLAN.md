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
