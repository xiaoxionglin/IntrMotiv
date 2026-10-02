# Repository organization: 2 October 2026

## Problem and evidence

The repository had roughly 5,900 tracked files, including more than 5,000 under `06_experiments/`. Reports, scripts, data, and figures were difficult to distinguish by browsing the folder alone. Root-level theory notes, a launcher guide, a host repair script, empty Obsidian placeholders, and tracked `.DS_Store` files added noise. An exact-content scan found 251 duplicate groups (281 excess copies, about 32.7 MB). Many copies are pinned survey inputs or repeated report figures whose paths appear in manifests and analysis code.

## Plan and implemented changes

1. **Make a short front door.** The root [README](../README.md) routes by purpose. The [research index](README.md) handles science, the [experiment index](../06_experiments/README.md) owns results, and the [run code index](../hpc_runs/README.md) points to validated studies and launch guidance.
2. **Move clearly misplaced files.** The structured-memory hypothesis is in `02_algorithm/`, its prior-work survey in `08_literature/`, the source report in `01_project_context/`, the batch guide in `hpc_runs/`, and the local CUDA repair script in `hpc_runs/hosts/local/`. The unresolved CA3 note now has a descriptive filename in `99_inbox/`. Update known links after the moves.
3. **Remove non-content.** Delete the zero-byte dated note, empty Obsidian base/canvas placeholders, and tracked `.DS_Store` files; ignore future Finder metadata.
4. **Preserve reproducibility.** Keep the pinned duplicate inputs, rendered outputs, study definitions, run scripts, and original report paths. Broad relocations would invalidate source manifests and historical commands.

## Follow-up for future study work

When a study is next updated, link its report, StudySpec, pinned input folder, and output folder from the existing experiment owner. Consolidate a duplicate only when all references and hashes are audited and the study's replay contract remains valid. Do not reorganize `hpc_runs/studies/` or historical `06_experiments/` paths without a print-only study validation and a link check.
