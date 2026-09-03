# Latest Standardized Workflow

- Implementation: `1.1.0`
- Study schema: `intrmotiv/study/v1`
- Canonical package: `hpc_runs/intrmotiv_study/`
- NEMO2 runtime copy: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/hpc_runs/intrmotiv_study/`
- Canonical guide: `04_implementation/standardized_study_workflow.md`
- Reference study: `hpc_runs/studies/graph_stabilized_recruitment.study.json`

## Deployment status

The vault source and NEMO2 runtime copy are synchronized at `1.1.0`. The NEMO2
copy passed 14 focused tests, validated the reference study, and audited all 36
real graph-stabilized submission rows, commands, job IDs, and workspace paths.
For a later version, synchronize the complete package and rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest hpc_runs.test_intrmotiv_study
python -m hpc_runs.intrmotiv_study validate \
  hpc_runs/studies/graph_stabilized_recruitment.study.json
```

The full-data collector benchmark was intentionally stopped once synchronization
and contract verification were complete. Runtime acceleration is secondary to
reducing repeated Codex discovery and code generation.

## Version policy

- Patch: compatible bug fix with no study-file changes.
- Minor: backward-compatible component or optional field.
- Major: incompatible schema or artifact contract. Introduce a new schema ID
  and migration notes; do not silently reinterpret existing studies.

Update this file, `version.py`, tests, and the guide together when the standard
changes, then synchronize and test the NEMO2 runtime copy.

## 1.1.0

- Added bounded parallel TensorBoard loading, configurable with
  `analysis.max_workers` and defaulting to four workers.
- Added `audit-submission` for exact matrix, command, job-ID, and workspace-path
  validation against real Sample Factory `jobs.tsv` files.
