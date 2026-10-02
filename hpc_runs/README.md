# Run code and study definitions

Start with [LATEST](intrmotiv_study/LATEST.md) and the [standardized workflow](../04_implementation/standardized_study_workflow.md) before creating, validating, or submitting a study.

| Need | Location |
| --- | --- |
| Declarative experiment matrices and their hashes | [`studies/`](studies/) — find a study by factor or date with `rg --files hpc_runs/studies | rg '<term>'` from the repository root |
| Validation, rendering, online collection, and telemetry orchestration | [`intrmotiv_study/`](intrmotiv_study/README.md) |
| Sample Factory launch and NEMO2 submission | [Batch submission guide](BATCH_SUBMISSION.md) |
| Host-specific preflight and launch helpers | [`hosts/`](hosts/) |
| Shared off-policy and baseline support | [`intrmotiv_offpolicy/`](intrmotiv_offpolicy/README.md), [`offpolicy_goal_baselines/`](offpolicy_goal_baselines/README.md) |
| Historical run adapters and focused tests | Top-level `*.py`, `test_*.py`, and `*.sh` files in this folder |

Study-specific adapters at the top level are retained at their original paths because historical commands and reports refer to them. New repeated behavior belongs in `intrmotiv_study/`; a new study matrix belongs in `studies/`. Use the [experiment status map](../06_experiments/README.md) to find the result report before interpreting a study's outputs.
