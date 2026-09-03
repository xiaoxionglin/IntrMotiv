# Graph-Stabilized Orthogonal Recruitment

Date: 2026-09-03

## Status

The authoritative NEMO2 IntrMotiv checkout now implements graph-stabilized DG
recruitment behind `--dg_orthogonal_recruitment_mode=graph`. The default is
`legacy`, so historical commands retain their previous behavior. The 36-run
production batch was submitted after the preflights completed successfully.

Five seed-99, 2M-frame ordinary Slurm preflights ran as jobs `7976505` through
`7976509`. All five completed with exit code `0:0` in 16--20 minutes. They span
direct, recovery, topology, both half-lives, both redundancy thresholds, and
the flat passive fallback. Their outputs, caches, W&B data, and Slurm logs
resolve below `/work/classic/fr_xl1014-train`.

The 36 production jobs are `7976516` through `7976551`. At the post-submission
audit, all 36 unique IDs were recorded as submitted and were running. The
audited submission directory is
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_graph_stabilized_recruitment_20260903/20260903T011943Z`.

## Implemented rule

The original `L`-decision silent endpoint remains the proposal opportunity.
Graph mode changes only victim eligibility and choice:

1. A new field has birth support one and is protected while support is above
   `0.25`.
2. An edge exists when confidence is above `0.25` and elapsed time is positive.
3. A mature isolated vertex is eligible.
4. A mature vertex is also eligible if it is the lower-incident-confidence
   member of a bidirectionally supported pair with both elapsed times at most
   `D`; exact ties select the higher DG index.
5. Isolated vertices are preferred, then the lowest-support eligible vertex.
   If none is eligible, recruitment is skipped.

Policy-buffer HRL uses controllability confidence and `T_ctrl`. Flat and other
non-policy-buffer agents use a checkpointed passive graph. Passive edges come
from consecutive different exclusive DG events within `L` behavior decisions,
cross rollout boundaries, reset at physical episode boundaries, and never read
CA3. Representation-generation IDs reject evidence collected before a row was
reassigned.

Reassignment clears the affected row's optimizer and BatchNorm state, clears
incident passive and controllability edges, increments both representation
generations, restores birth support to one, and remains capped at one assignment
per accepted rollout. The old `recruitment_committed` buffer is retained as
ever-recruited telemetry but is no longer an eligibility gate in graph mode.

## Verification

- Focused implementation suite: 60 passed before the final evidence-preference
  test was added.
- Complete IntrMotiv suite: 146 passed, with 9 existing warnings.
- Graph-recruitment unit file after the additional policy-versus-passive test:
  12 passed.
- Real argument parsing succeeded for direct C05, topological C15, and flat
  passive-fallback configurations with the expected enlarged recurrent state.
- The audited production manifest contains 36 runs: corrected C05/C13/C15,
  `D` in `{4, 8}`, half-life in `{5k, 10k}`, and seeds `{8, 99, 123}`.

Completed original C05/C13/C15 runs remain contextual controls rather than
contemporaneous matched runs.

## Interim training telemetry

**Snapshot:** 2026-09-03 14:41 CEST. All 36 jobs were still `RUNNING` with no
recorded Slurm failure. Their event streams ranged from 71.3M to 98.4M frames,
so this is a mechanism audit, not the final factorial result. Each scalar below
is the mean over that run's latest 10M frames; each table cell is mean +/-
sample SD over seeds 8, 99, and 123. Cumulative recruitment counters are
sampled at the latest event rather than averaged.

| Backbone | `D` | Half-life | Mean progress (M) | Coverage AUC | Unique cells | Hit lift | Option success | Assignments | Repeat assignments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C05 | 4 | 5k | 86.2 | 48.1 +/- 13.7 | 69.8 +/- 29.2 | 0.995 | 0.247 | 0.7 | 0.0 |
| C05 | 4 | 10k | 81.2 | 48.7 +/- 17.3 | 70.8 +/- 31.5 | 1.118 | 0.113 | 1.0 | 0.0 |
| C05 | 8 | 5k | 87.0 | 44.4 +/- 14.2 | 59.6 +/- 19.1 | 0.989 | 0.239 | 2.7 | 1.0 |
| C05 | 8 | 10k | 96.5 | 43.0 +/- 24.6 | 61.3 +/- 41.8 | 1.073 | 0.395 | 5.0 | 2.0 |
| C13 | 4 | 5k | 94.6 | 45.0 +/- 14.7 | 63.1 +/- 25.9 | 0.961 | 0.559 | 0.0 | 0.0 |
| C13 | 4 | 10k | 96.1 | 40.3 +/- 16.9 | 52.4 +/- 25.9 | 0.902 | 0.569 | 0.0 | 0.0 |
| C13 | 8 | 5k | 96.1 | 32.6 +/- 3.6 | 44.0 +/- 6.3 | 0.921 | 0.427 | 0.0 | 0.0 |
| C13 | 8 | 10k | 95.6 | 32.8 +/- 6.2 | 40.5 +/- 9.7 | 0.902 | 0.635 | 0.0 | 0.0 |
| C15 | 4 | 5k | 95.9 | 54.5 +/- 8.2 | 81.8 +/- 15.5 | 0.909 | 0.393 | 0.0 | 0.0 |
| C15 | 4 | 10k | 81.0 | 59.2 +/- 7.0 | 90.5 +/- 12.1 | 0.921 | 0.414 | 0.0 | 0.0 |
| C15 | 8 | 5k | 75.2 | 61.0 +/- 15.1 | 92.3 +/- 35.3 | 0.931 | 0.414 | 0.0 | 0.0 |
| C15 | 8 | 10k | 82.8 | 55.9 +/- 9.0 | 87.5 +/- 15.7 | 0.866 | 0.404 | 0.0 | 0.0 |

### Mechanism reading

The graph state is healthy but the mutation rule is mostly inactive. Mean
connected fraction is 1.000 in 11/12 parameter cells and 0.999 in the other;
passive graph density is 0.79--0.96, with roughly 1,175--1,622 accepted passive
updates per rollout. Birth protection is already zero in every cell average
except the single actively mutating C05 run described below. Thus graph-clock
startup protection is not suppressing late recruitment.

Only 5/36 runs have made any assignment. All are C05: three runs made one or
two assignments, `C05 D8 H5k` seed 123 made eight with three repeats, and
`C05 D8 H10k` seed 8 made fifteen with six repeats. The other 31 runs,
including every C13 and C15 run, remain at zero. Late-window assignment rates
are zero everywhere except a mean 0.0022 redundant assignments per rollout in
`C05 D8 H5k` seed 123. This rules out population-wide runaway churn at the
snapshot, but it also shows that most of the sweep does not actually intervene
on the representation.

The `D=8` setting does change eligibility without necessarily causing
recruitment. For C15, `D=4` has zero redundant pairs and eligible vertices,
whereas the two `D=8` cells average 0.76--1.41 redundant pairs and 0.54--1.22
eligible vertices in the latest window. Nevertheless, all twelve C15 runs
have zero assignments because graph eligibility is only a victim filter; the
separate `L`-decision silent endpoint proposal remains necessary. This is the
central mechanistic result so far.

Online representation diagnostics do not indicate broad collapse: all cell
means have DG density 0.0246--0.0289 and usage entropy 0.968--0.978. Silent-unit
fraction is exactly zero in 35/36 run windows and 0.0010 in the remaining
`C05 D8 H5k` seed-123 run. These minibatch statistics cannot establish place-
field diversity, so they are not a substitute for the standard offline sweep.

One run warrants a final-window check: `C05 D8 H5k` seed 123 combines eight
assignments, three repeats, nonzero birth protection, and 10.27 passive stale
transitions per rollout in its latest window. The other 35 runs have zero
late-window stale-transition telemetry. A trajectory is needed before calling
this persistent stale evidence rather than a short post-reassignment episode.

### Scientific reading and pending analyses

The unsynchronized outcome columns do not justify a winner. Relative to the
historical terminal backbones, the current C05 and C13 ranges overlap their
old coverage, while no C15 cell yet matches the original C15 terminal mean of
78.6 AUC. These comparisons are contextual only because progress differs by
27.1M frames and the original controls are not contemporaneous.

A synchronized 60--70M analysis is specified in
`analyze_graph_stabilized_recruitment.py`; it reports within-seed paired main
effects for `D`, half-life, and their interaction. At the snapshot, the rerun
was blocked by an expired NEMO2 SSH authentication session. After the jobs
finish, the final analysis must use the common 90--100M window and terminal
checkpoints.

The offline spatial evaluation will use the reusable manifest-driven package,
not a batch-specific evaluator. At minimum it should compare the best terminal
C05, C13, and C15 graph cells with the original backbones using five seed-99
checkpoints plus terminal seeds 8 and 123. It must report active and silent
units, spatial information, active-only map cosine, peak diversity, continuous
pre-threshold maps, and seed-99 stability. The live online telemetry cannot
answer those spatial questions.

Lightweight snapshot tables are in
`results/graph_stabilized_recruitment_20260903/`; bulk event files and future
place-field artifacts remain in the allocated NEMO2 workspace.
