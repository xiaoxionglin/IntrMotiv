# Graph-planning optimization for future batches

Implemented and validated on 2026-09-11. **Staged for future deployment; not
applied to the active NEMO2 source or running DG-capacity batch.** The patch
passes `git apply --check` against the live runtime. Before the next new batch,
apply this patch in its intended source checkout, rerun the focused tests, and
include the resulting source revision/diff in the ordinary print-only review.
Use the normal Slurm preflight to measure end-to-end throughput. Do not silently
apply this optimization during the current comparison.

## Cause and exact change

The old edge-validation selector recomputed transitive closure twice for every
candidate edge, separately for each stream selecting a task. The manager also
recomputed all-pairs shortest paths at every recurrent step, even when its
policy graph was unchanged. The matched online throughput pattern implicated
this work, but no live-process profile established its fraction of total time.

For reflexive reachability $R$, inserting edge $s\to d$ adds exactly the
previously unreachable pairs $(i,j)$ with $R_{is}R_{dj}=1$. Repeated crossings of
the inserted edge can be removed from a path; the identity also holds for
directed cycles. Let $M=1-R$, with a zero diagonal. All insertion gains are
computed together as

$$G=R^\top M R^\top.$$

This replaces candidate-by-candidate closure with one closure and two matrix
multiplications per distinct reliable adjacency. Float64 arithmetic counts
these integer-valued products exactly at supported graph sizes. The selector
retains its candidate filters, normalization, score arithmetic, and row-major
tie breaking. Standalone `connectivity_gain` uses the equivalent outer-product
identity.

Shortest paths retain the original Floyd–Warshall implementation and tie
ordering, with cached results for identical effective costs and reliable-edge
masks. Cached path outputs are cloned to protect them from caller mutation.
Caches compare tensor contents, dtype, and device rather than relying on
mutation counters: `.data` copies can bypass those counters. They refresh after
graph updates, changed thresholds, node retirement, tensor replacement, and
checkpoint loads. They are ordinary per-graph Python attributes, not registered
buffers or checkpoint fields. No reward, goal, deadline, model-state shape,
StudySpec, schema, or workflow-version changes were made.

## Evidence

39 focused tests passed locally (2.09 seconds) and in an isolated NEMO2 source
directory (3.12 seconds): the existing 31 planner tests plus eight new
performance/correctness tests. New tests cover all 512 directed three-node
adjacencies, independent graph-traversal reference gains, sparse/dense/cyclic
DG16/32/64 graphs, selection/ties, unchanged-cache reuse, mutation invalidation,
and unchanged checkpoint keys.

The benchmark additionally compared complete manager outputs against the
archived original implementation for 24 steps × two streams at each of
DG16/32/64, including active and silent observations, expiry, changed edge
statistics, and node retirement. Outputs matched exactly.

Local synthetic single-thread CPU microbenchmark, PyTorch 2.7.1; medians of 12
calls after warm-up. Reliable-edge density 0.02, candidate density 0.10, all
sources eligible. These are graph-function timings, **not measured training FPS
or a predicted end-to-end speedup**.

| DG | Candidate edges | Old selection (ms) | New, cache rebuilt (ms) | New, cached (ms) |
|---:|---:|---:|---:|---:|
| 16 | 18 | 3.67 | 0.198 | 0.095 |
| 32 | 100 | 43.55 | 0.607 | 0.393 |
| 64 | 391 | 444.95 | 2.157 | 1.563 |

DG64 cached shortest paths fell from 1.818 ms to 0.038 ms. A cache miss still
runs the original path algorithm (1.913 ms including cache storage).

Raw results: [microbenchmark.json](../06_experiments/data/graph_planning_optimization_20260911/microbenchmark.json).

## Source and future deployment

- [Reviewable patch](../hpc_runs/source_snapshots/graph_planning_optimization_20260911.patch).
- [Scoped source archive](../hpc_runs/source_snapshots/graph_planning_optimization_20260911.tar.gz): modified planner, new tests, reusable benchmark.
- [SHA-256 provenance](../hpc_runs/source_snapshots/graph_planning_optimization_20260911.sha256.json).
- Baseline: `topological_frontier.py` from `dg_capacity_goal_conditioning_20260910.tar.gz`, identical to the live module read on September 11.
- NEMO2 isolated validation source and staged patch:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/graph_planning_optimization_20260911/`.

In the source checkout selected for the **next batch**, after preserving its
current source provenance:

```bash
git apply --check /path/to/graph_planning_optimization_20260911.patch
git apply /path/to/graph_planning_optimization_20260911.patch
python -m pytest -q \
  sf_working_directories/IntrMotiv/tests/test_graph_planning_performance.py \
  sf_working_directories/IntrMotiv/tests/test_topological_frontier.py
python -m sf_working_directories.IntrMotiv.analysis.benchmark_graph_planning \
  --baseline /path/to/archived/topological_frontier.py
```

Use `SF_git` locally and `SFgit` on NEMO2. Run substantial profiling and DMLab
preflights through Slurm, with output in the allocated workspace. If patch
review fails against newer source, reconcile changes rather than overwriting
the runtime with the archived complete module.

## Reusable lesson

Compare matched long-window FPS before blaming DG capacity or requesting more
cores. Benchmark the implicated functions against a frozen original and verify
decisions as well as scalar outputs. Reuse derived graph quantities while their
actual inputs are equal; version counters alone are insufficient for the
existing synchronization paths. Synthetic function speedups need a subsequent
ordinary-job preflight before making end-to-end throughput claims.

The local source snapshot lacks some package initializers and can be shadowed
by an installed checkout despite `PYTHONPATH`. The isolated validation tree
included package initializers, and test tracebacks confirmed the intended
source path. No temporary validation checkout is a second maintained source.
