# Device-resident topological manager

Implemented locally on 2026-09-15 in
`/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/IntrMotiv/dmlab/topological_frontier.py`.
**G500 deployment and CUDA timing remain pending.** Automatic approval review
rejected exporting the candidate source to G500 without explicit authorization.
Only an isolated copy of the existing remote source was created under
`/scratch/lin/IntrMotiv/analysis/core_gpu_validation_20260915/`; the candidate
was not transferred and live runtime source was not modified.

## Cause and scope

The [G500 qualification record](../06_experiments/dg_neighborhood_g500_20260914.md)
and `infra.md` identify the core as the dominant measured training cost.
The active study selects `frontier_direct`, policy-buffer graph memory, and
`hrl_edge_exploration=False`. Although its tensors already follow the model
device, the topological manager inspected individual CUDA values with Python
conditionals and `.item()` inside loops over environments. These operations
force host/device synchronization.

Landmark anchors, passive transition bookkeeping, stale-state reset, and deferred
validation bookkeeping now use fixed-shape tensor operations. With edge probing
disabled, batched option updates handle target selection, hits, wrong first
outcomes, timeouts, exploration, and geometry on the state device. CPU and GPU
use the same implementation. The optional edge-probing planner retains its
existing per-stream decision loop. State shapes, checkpoint fields, graph-update
ownership, configuration, and scientific objectives are unchanged.

The earlier [graph-planning optimization](graph_planning_optimization_20260911.md)
addresses a separate cost: derived graph quantities and edge-probe scoring.
Its content-comparison cache is not included here; tensor equality checks would
need separate CUDA synchronization review. Do not overwrite newer runtime files
with either archived full module when combining these changes.

## Validation

- 106 focused tests passed in the isolated local runtime; 10 CUDA cases skipped
  because desktop PyTorch reports CUDA unavailable.
- The 10 new CPU tests also passed from the actual updated local checkout.
- One-time reference comparison: 192 parameter combinations, 24 streams, seven
  nodes, five successive transitions each. Every output matched within
  `rtol=1e-5, atol=1e-5`. Nine additional edge-probing configurations passed.
- Persistent input/output fixtures cover nine representative configurations;
  they record graph buffers and all state/condition outputs from the original
  implementation. Tests check graph and input immutability as well as values.
- An ATen profiler regression rejects scalar extraction and dynamic index
  operations in the non-probing manager. CPU-only `one_hot` bounds checking is
  the sole allowed exception; CUDA has no exception.
- `git diff --check` passed. Existing encoder/parameter edits were preserved.

Single-thread CPU microbenchmark, DG16, 64 successive calls after four warmups:

| Streams | Original ms/update | Batched ms/update | Ratio |
|---:|---:|---:|---:|
| 32 | 2.150 | 1.862 | 1.15× |
| 256 | 11.597 | 1.765 | 6.57× |

These are synthetic manager timings, not GPU measurements or training FPS.
Use the runtime's `evaluation/benchmark_topological_manager.py --reference
/path/to/original/topological_frontier.py --device=cuda` for the pending GPU
comparison. It synchronizes CUDA around the measured region. Run
`python -m pytest -q sf_working_directories/IntrMotiv/tests/test_topological_manager_device.py`
in the chosen runtime to repeat the device/oracle checks.

## Review and reuse

- [Text patch](../hpc_runs/source_snapshots/topological_manager_device_20260915.patch).
- [Five-file source archive](../hpc_runs/source_snapshots/topological_manager_device_20260915.tar.gz),
  including the binary NPZ fixture omitted from the text patch.
- [File and archive hashes](../hpc_runs/source_snapshots/topological_manager_device_20260915.sha256.json).

The reliable diagnostic is the combination of a measured core bottleneck,
per-stream scalar reads in the actual selected manager, and reference parity
plus ATen profiling. Device placement alone does not establish device-resident
execution. Reuse the saved input/output fixtures instead of regenerating an
oracle from changed code. Check the existing source-snapshot registry early
for related optimizations. Desktop tests cannot establish CUDA throughput;
finish isolated GPU validation before attributing a training speedup to this
change or qualifying a new production source snapshot.
