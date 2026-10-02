# IntrMotiv repository

Start with the [research notes index](00_index/README.md) for the scientific idea, the [experiment status map](06_experiments/README.md) for results, or the [latest study workflow](hpc_runs/intrmotiv_study/LATEST.md) for reproducible runs.

## Find a file by purpose

| Looking for | Start here |
| --- | --- |
| Current findings and missing evidence | [Experiment status map](06_experiments/README.md) and [open analyses](06_experiments/open_analyses.md) |
| Theory, hypotheses, and source context | [Research notes index](00_index/README.md) |
| Model architecture and metric definitions | [Architecture reference](04_implementation/architecture/README.md) and [metric reference](04_implementation/IntrMotiv_metric_reference.md) |
| A proposed study or poster | [Plans index](05_plans/README.md) |
| A validated run matrix, launcher, or test | [Run code index](hpc_runs/README.md) |
| Experiment scripts, pinned inputs, and generated figures | [Experiment index](06_experiments/README.md); keep each study's inputs under `06_experiments/data/` and outputs under `06_experiments/results/` |
| Literature, abstract drafts, or a handoff bundle | [Literature](08_literature/README.md), [abstract drafts](07_abstracts/README.md), or `exports/` |
| Unresolved idea or infrastructure issue | `99_inbox/` or [infrastructure tracker](infra.md) |

The numbered folders hold research material; `hpc_runs/` holds runnable study definitions and analysis tools. The Sample Factory training implementation lives in the separate `SF_hipposlam` repository. See [the standardized workflow](04_implementation/standardized_study_workflow.md) before defining a study or launching training.

## Organization rules

- Give one study one result owner in `06_experiments/README.md`. Put dated snapshots and launch records next to it, with a forward link to the owner.
- Keep source data, generated figures, and pinned provenance with the study that produced them. Some byte-identical files are intentional pinned inputs for different reports; preserve their paths and hashes.
- Place new research notes in the subject folder, not the repository root. Add a link to the nearest index when the note becomes useful to other readers.
- Keep the root for this map, project instructions, and the infrastructure tracker. Ignore transient cache and Finder files.

The [organization note](00_index/repository_organization_20261002.md) records the cleanup and follow-up steps.
