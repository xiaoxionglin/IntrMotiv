# Latest Standardized Workflow

- Implementation: `1.0.0`
- Study schema: `intrmotiv/study/v1`
- Canonical package: `hpc_runs/intrmotiv_study/`
- NEMO2 runtime copy: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/hpc_runs/intrmotiv_study/`
- Canonical guide: `04_implementation/standardized_study_workflow.md`
- Reference study: `hpc_runs/studies/graph_stabilized_recruitment.study.json`

## Version policy

- Patch: compatible bug fix with no study-file changes.
- Minor: backward-compatible component or optional field.
- Major: incompatible schema or artifact contract. Introduce a new schema ID
  and migration notes; do not silently reinterpret existing studies.

Update this file, `version.py`, tests, and the guide together when the standard
changes, then synchronize and test the NEMO2 runtime copy.
