# Experiment Markdown organization strategy

## Scope and navigation contract

The inventory on 2 October 2026 contained 79 Markdown files directly in
`06_experiments/` and 17 below `data/` or `results/`. The top-level files are
human-authored study notes, reports, audits, and syntheses. Nested Markdown is
part of pinned data or generated result bundles and retains its original path.
The [README](README.md) is the single ordered index for both sets.
Top-level note paths remain stable because Obsidian links, reports, and
historical commands refer to them. The README sections provide the topical
chunks without moving the entire archive or changing pinned source paths.

Use the **study line** as the organizing unit. For each line, show its current
result owner first, followed by the plan, implementation or launch record,
dated analyses, and telemetry in evidence order. Give each link a role label:
`RESULT`, `PLAN`, `RUN`, `TELEMETRY`, `AUDIT`, `SYNTHESIS`, or `ARTIFACT`.
The role describes the file's purpose, not the strength of its conclusion.
The report-owner table remains the source for current evidence boundaries;
the complete index makes historical files findable without promoting them to
current results.

## Merge rule

Merge notes when they report complementary measurements or dated snapshots of
the **same declared study** and can keep their protocols, denominators,
tables, figures, and provenance intact in one result owner. Place an
orientation paragraph at the top, retain the older section under a dated
heading, and update internal references. Do not merge a plan with an outcome,
different architectures or task protocols, or generated source snapshots whose
path and hash are part of a manifest.

This pass merges the following pairs:

| Result owner | Incorporated evidence |
| --- | --- |
| `dg_anti_collapse_results.md` | six-condition place-field follow-up |
| `dg_structural_and_manager_exploration_results.md` | selected 10k-decision place-field telemetry |
| `encourage_dg_regularizers_interim_analysis.md` | matched 50M candidate telemetry |
| `persistent_intrinsic_control_status_20260909.md` | detailed 8 September learning audit before the 9 September status |

The short unstructured second-iteration questions move into the existing HRL
iteration owner. Other neighboring documents keep separate roles or protocols.
In particular, a release record remains distinct from its result, a newer
poster snapshot does not overwrite a matched study result, and the pinned
`results/.../source_inputs/` copy is retained.
After consolidation, the catalogue covers 72 top-level study notes and all
17 nested Markdown files exactly once, excluding this strategy, the README,
and the open-analysis register.

## Maintenance

When adding a Markdown report, link it once in the README's relevant study
line, label its role, and point a superseded snapshot toward its result owner.
Before deleting an overlapping note, check all Markdown links, wiki links,
script literals, and generated manifests. Preserve the study schema, workflow
version, source hashes, and evaluation protocol during consolidation. After
editing, check local links, Markdown math delimiters, and the inventory count.
