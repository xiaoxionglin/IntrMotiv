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
| Experiment notes, scripts, inputs, and figures | [Experiment index](06_experiments/README.md) for topic folders and the shared or topic-local data and result bundles |
| Reusable IntrMotiv visuals for papers or slides | [COSYNE visual asset index](07_abstracts/cosyne_2027_submission_draft/REUSABLE_ASSETS.md) for individual plots, icons, maps, trajectories, and figure source tables |
| Literature, abstract drafts, or a handoff bundle | [Literature](08_literature/README.md), [abstract drafts](07_abstracts/README.md), or `exports/` |
| Unresolved idea or infrastructure issue | `99_inbox/` or [infrastructure tracker](infra.md) |

The numbered folders hold research material; `hpc_runs/` holds runnable study definitions and analysis tools. The Sample Factory training implementation lives in the separate `SF_hipposlam` repository. See [the standardized workflow](04_implementation/standardized_study_workflow.md) before defining a study or launching training.

## Folder map

Use the purpose links above to resume work; use this map when browsing or placing a file. The numbers group material by role, not by priority or freshness.

| Folder | What belongs here |
| --- | --- |
| [`00_index/`](00_index/README.md) | Scientific reading routes and organization decisions |
| [`01_project_context/`](01_project_context/) | Project background, source reports, and interface assessments |
| [`02_algorithm/`](02_algorithm/) | Core formulations and model hypotheses |
| [`03_transition_distance/`](03_transition_distance/) | Transition-distance theory and reward design |
| [`04_implementation/`](04_implementation/) | Architecture, metric definitions, and reusable workflow documentation |
| [`05_plans/`](05_plans/README.md) | Proposed studies and poster plans |
| [`06_experiments/`](06_experiments/README.md) | Result owners, experiment records, analysis scripts, inputs, and outputs |
| [`07_abstracts/`](07_abstracts/README.md) | Abstract drafts, submission packages, and their figure assets |
| [`08_literature/`](08_literature/README.md) | Prior-work notes and source papers |
| [`99_inbox/`](99_inbox/) | Unresolved ideas awaiting a subject home |
| [`hpc_runs/`](hpc_runs/README.md) | Declarative studies, launch helpers, and reusable analysis code |
| [`assets/`](assets/) | Shared visual material; study and submission assets also stay beside their owners |
| [`exports/`](exports/) | Portable handoff bundles; use the originating project notes and source files when making revisions |

For a study's current evidence, follow its owner in the [experiment status map](06_experiments/README.md#report-status-and-ownership). A dated filename records provenance; it does not establish which report is authoritative.

## Organization rules

- Give one study one result owner in `06_experiments/README.md`. Put dated snapshots and launch records next to it, with a forward link to the owner.
- Keep source data, generated figures, and pinned provenance with the study that produced them. Some byte-identical files are intentional pinned inputs for different reports; preserve their paths and hashes.
- Place new research notes in the subject folder, not the repository root. Add a link to the nearest index when the note becomes useful to other readers.
- Keep the root for this map, project instructions, and the infrastructure tracker. Ignore transient cache and Finder files.

The [organization note](00_index/repository_organization_20261002.md) records the cleanup and follow-up steps.
