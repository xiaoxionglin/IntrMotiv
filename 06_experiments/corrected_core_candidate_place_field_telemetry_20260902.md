# Corrected-Core Candidate Place-Field Telemetry

**Status:** production Slurm array `7975099` submitted; 77 tasks queued.

## Scientific question

The late-window training metrics identify candidate policies, but they cannot
show whether DG units define separated, localized landmarks. In particular,
C05's confidence-qualified controllability graph remains sparse enough that its
interpretation is ambiguous: the landmarks may be far apart, broad or aliased,
or the worker may simply fail to control them.

This sweep measures the representation side of that ambiguity. It cannot by
itself establish target control.

## Conditions

| Role | Cells | Reason |
|---|---|---|
| Anchors | C01-C03 | Flat, delayed-direct, and immediate-direct controls |
| Update candidate | C04 | Coverage gain from iterative updates |
| Structural candidates | C05-C08 | Stable C05, highest-lift C06, recruitment C07, and high-variance C08 |
| Topology contrast | C14-C16 | Visit, UCB-direct coverage winner, and waypoint comparator |

C09-C11 are excluded because the empirical HER implementation does not enforce
first achievement of the relabeled goal. C12-C13 are excluded because their
standard five-checkpoint/three-terminal-seed telemetry is already complete.

## Protocol

- 10,000 stochastic policy decisions per artifact;
- seed 99 near 5M, 25M, 50M, 75M, and 100M frames;
- seeds 8 and 123 at terminal;
- 19 x 19 spatial grid;
- thresholded and continuous pre-threshold DG maps;
- 77 unique labels and 55-row seed-99 trajectory manifest;
- checkpoints, output, DMLab cache, temporary data, and Slurm logs under
  `/work/classic/fr_xl1014-train`.

The 500-decision preflight completed under job `7974997` with thresholded and
pre-threshold arrays of shape `19 x 19 x 16` and no error trace.

Telemetry root:

`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/corrected_core_candidates_20260902_place_fields/`

## Required analysis

For every condition report:

- active and silent units;
- spatial information;
- active-only and pre-threshold map cosine;
- unique peak bins and peak entropy;
- mean pairwise peak distance in grid bins and DMLab position units;
- the distribution and minimum nearest-neighbor distance between active peaks;
- visited grid cells;
- seed-99 map stability to terminal.

For C05 specifically, distinguish three hypotheses:

1. **Separated landmarks, weak worker:** large peak separation and low map
   overlap, paired with low target control.
2. **Broad or redundant landmarks:** small peak separation or high map cosine,
   which makes nominal DG targets hard to distinguish.
3. **Multi-location aliasing:** separated peaks do not rescue a unit whose map
   contains several disconnected response regions; inspect per-unit maps and
   continuous logits before interpreting graph edges.

Pair these maps with the already-extracted final-10M target-hit lift, action
sensitivity, option success, and known-edge fraction. Do not infer
controllability from spatial separation alone.

## Submission record

- preflight: `7974997`, completed `0:0`;
- production: `7975099`, array `0-76%20`, 4 CPUs, 16 GB, 1 hour per task;
- production manifest rows: 77;
- seed-99 trajectory manifest rows: 55.

## Results

Pending production completion and standard manifest-driven postprocessing.
