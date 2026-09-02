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

### C05 lightweight checkpoint analysis

Before the spatial sweep completes, the final checkpoints and 10M-frame event
windows already constrain the interpretation:

- successful C05 target arrivals take 4.34 policy decisions on average across
  seeds (4.27, 4.06, and 4.69 for seeds 8, 99, and 123);
- confidence-qualified `T_ctrl` edges have a pooled median of 4.15 decisions;
  100/113 (88.5%) are at most 8 decisions and 108/113 (95.6%) are under 10;
- only 33, 42, and 38 of 240 possible directed off-diagonal edges are known;
- all 16 landmarks occur as sources, but only 5, 4, and 3 landmarks occur as
  confidence-qualified destinations;
- in every seed, one destination receives 14 incoming known edges. Only six
  directed edges per seed are reciprocal.

This is not evidence of uniformly effective local control. The graph is a
sparse, strongly destination-concentrated funnel: most of its successful edges
lead to a few easy DG targets. The concentration is compatible with either
genuinely easy local landmarks or broad/multi-location target fields. It makes
the latter an important hypothesis, especially for the hub units:

- seed 8: DG 4 (indegree 14);
- seed 99: DG 2 and DG 14 (both indegree 14);
- seed 123: DG 13 (indegree 14).

The map analysis should inspect those hub fields first, then compare their
connected-component counts, map area, peak sharpness, and pre-threshold logits
against low-indegree fields. It should also measure physical peak distances for
known edges and test whether `T_ctrl` time correlates with peak distance.

The checkpoint stores no valid landmark poses for C05 because action path
integration is disabled, so physical separation cannot be recovered from the
graph alone. A policy decision uses frame skip 8; therefore 4.34 decisions are
about 35 repeated simulator frames, not four primitive motor updates.

Artifacts:

- `06_experiments/analyze_c05_option_timing.py`;
- `06_experiments/analyze_c05_checkpoint_graph.py`;
- `06_experiments/results/corrected_core_reevaluation_20260902/c03_c05_option_timing_10m.csv`;
- `06_experiments/results/corrected_core_reevaluation_20260902/c05_terminal_checkpoint_graph.csv`.

### Spatial maps

Pending production completion and standard manifest-driven postprocessing. As
of the lightweight analysis, Slurm array `7975099` remains pending; no DMLab
rollout was run on the login node.
