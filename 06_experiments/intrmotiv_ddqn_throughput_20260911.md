# DDQN/HER throughput qualification — 2026-09-11

The opt-in `--learner-execution=batched` backend approximately doubled learner-active end-to-end throughput in a paired 250k-frame HER preflight. It preserves the v2 objective, replay/HER semantics and exact TD-position budget. The six ongoing v2 scientific runs retain their original source and execution path. This test qualifies execution, not control performance or adaptive representation learning.

## Measured result

| Metric | Reference | Batched |
|---|---:|---:|
| Job | 8056866 | 8056867 |
| Slurm elapsed including startup | 5m13s | 3m44s |
| Learner-active FPS, runtime audit | 1,025.05 | 2,004.60 |
| Frames | 250,112 | 250,112 |
| Optimizer updates | 720 | 720 |
| Valid TD positions | 184,320 | 184,320 |
| Target copies | 7 | 7 |
| Invalid final-observation exclusions | 32 | 32 |
| Frozen reference unchanged | yes | yes |

The audited throughput ratio is 1.956. This is one seed and one paired system test, not a general speed guarantee. Floating-point batching changes rounding and can eventually change trajectories; realized HER samples were 2,384 and 2,363. Do not interpret this as a paired scientific outcome comparison. Both runs retain exactly 256 TD positions per update and one update per 64 accepted decisions after warmup.

The later per-run windows show 1,041 versus 2,056 FPS, with collection accounting for 36% versus 74% of measured collection-plus-learning time. These windows differ slightly because logging is periodic; use the full learner-active audit for the headline. Mean sampled prefix/batch preparation fell from 70.80 to 3.84 ms and learner updates from 71.85 to 17.97 ms. These are scalar samples, not full profiler traces.

The separate numerical benchmark (job 8056856, 2m29s, 8 CPU/8 GB, no environment) used the actual production FiLM decoder and a trained v2 checkpoint with synthetic frozen-feature replay. At 8 Torch threads, combined batch preparation and learning improved from 260.93 to 33.69 ms for DDQN (7.74×), and 258.85 to 36.93 ms for HER (7.01×). At one thread gains were 4.45× and 4.69×. This excludes encoder, environment and replay sampling costs. Do not extrapolate these learner-only ratios to whole training, or reduce trainer threads solely from this result: the same process runs ResNet inference.

## Implementation

`batch.py` reconstructs independent physical prefixes together with zero-activity padding before each history. Identical frozen, goal-independent writes allow online and target prefix memories to be shared read-only. `worker.py` retains temporal CA3 updates but evaluates the row-independent decoder over flattened time and batch dimensions in one call per network. This removes Python calls and small matrix operations without shortening history, changing goals, dropping TD positions or changing optimizer cadence.

The default remains `reference`. Conversion qualifies the inspected `TargetFiLMDecoder`; batched mode rejects unqualified decoders and goal-dependent writes, and rejects training BatchNorm/dropout. This qualification must be revisited if that decoder implementation changes. The worker state-dict schema stays v2. The frozen ResNet/DG feature contract is unchanged; adaptive DG still requires storing trunk features and rebuilding current working representations, as documented in the package README.

Validation: 64 local tests passed across original, v2, throughput and canonical study suites. The original/v2/throughput 34 tests also passed on NEMO2. Actual-decoder parity was checked across 1/2/4/8 threads, DDQN/HER and three optimizer updates including a target copy; maximum parameter discrepancy was below 4.1e-7 and maximum loss discrepancy below 1e-9. End-to-end runtime audit passed for both runs. Independent commanded/spatial evaluation remains necessary for scientific claims.

## Sample Factory mechanisms and migration choices

1. **Batched tensor computation: implemented and measured.** The existing collector already batches 32 encoder inputs and uses shared image buffers. The immediate waste was per-prefix/per-time learner work.
2. **Overlap collection with learning: next small integration.** Issue `step_async(actions)`, learn from completed replay while environments advance, then `step_wait()` and append transitions. Preserve per-stream ordering, accepted-decision accounting, bounded outstanding work and checkpoint/final flushing. With collection now dominant, this is a better next target than further micro-optimizing the learner.
3. **Double buffering: moderate integration.** Sample Factory divides workers into environment groups so inference and simulation overlap. Our collector needs per-group memory, goals and counters before enabling this. See [double buffering](https://www.samplefactory.dev/07-advanced-topics/double-buffered/).
4. **Full SF actor/inference/learner integration: larger change.** Reuse its shared buffers, batched inference and backpressure, but retain our recurrent replay, physical-episode validity, deadline/HER and exact update-budget contracts. See [SF architecture](https://www.samplefactory.dev/06-architecture/overview/). GPU inference, affinity and thread tuning should be measured end-to-end after integration; none was established by this CPU test.

Policy age does not invalidate one-step DDQN targets. PPO-style likelihood-ratio corrections and policy-version rejection are not required here. Stale actors can delay improved exploration, but this is a data-distribution/efficiency issue. Representation and recurrent-state consistency are separate concerns; the current frozen, goal-independent writes make asynchronous collection especially straightforward.

## Existing upstream DQN

There is an upstream [DQN/IDQN PR #331](https://github.com/alex-petrenko/sample-factory/pull/331), open and unmerged when checked on 2026-09-11, head `4687deb3a1947a65c2d1bfb4c4631e934ba2bae5`. It is absent from our installed SF fork. It implements Double DQN, prioritized replay, epsilon-greedy collection and integration with SF workers. Its learner explicitly rejects RNN/recurrence because sequence replay is not implemented; no HER contract is supplied. Thus it provides useful infrastructure, but is not a drop-in replacement for our CA3/HER worker.

Two details from the inspected learner matter for migration. It deliberately does not filter policy lag, matching the off-policy correction above. Its update accounting subtracts the entire calculated debt before applying `dqn_max_updates_per_batch`, so enabling a cap discards excess updates. We must disable or repair that behavior to preserve our declared replay intensity. Review the branch at the pinned commit before adoption; no PR code was installed into active runs.

Our installed fork already has `register_learner_factory` and `create_learner` hooks. A native SF route should reuse these hooks and the PR's actor/inference plumbing while adapting sequence replay and exact update accounting. The difficult part is preserving those contracts, not implementing the basic DDQN loss again.

## Reproduction and provenance

Study: `hpc_runs/studies/intrmotiv_ddqn_throughput_preflight.study.json`; schema `intrmotiv/study/v1`, workflow `1.8.0`, SHA-256 `5ca30e197467022adec06c55d18e3ec6a694710b04ed9715eb70c942bd463229`. It varies only execution backend, using HER seed 99, 32 environments, 8 Torch threads and 250k requested frames. Each job requested 40 CPUs, 80 GB, no GPU, 30 minutes. Numerical benchmark requested 8 CPUs, 8 GB and 20 minutes.

Isolated runtime: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ddqn_throughput_20260911`. Data root: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/`. Submission manifests: `_slurm/intrmotiv_ddqn_throughput_preflight_20260911/20260911T203535Z/`; print-only review: `20260911T203256Z/`. Training outputs: `intrmotiv_ddqn_throughput_preflight_20260911/`. Numerical benchmark and profiles: `analysis/ddqn_throughput_20260911/`.

Lightweight evidence is under [data/intrmotiv_ddqn_throughput_20260911](data/intrmotiv_ddqn_throughput_20260911/): benchmark, runtime audit, stage profiles and upstream PR status. All bulk data, caches and logs stay in the allocated workspace.

Use `python -m hpc_runs.intrmotiv_offpolicy.audit_runtime STUDY BATCH_ROOT --output AUDIT.json` for budget/frozen-reference qualification and `python -m hpc_runs.intrmotiv_offpolicy.profile_runtime STUDY BATCH_ROOT --window 125000 --output PROFILE.json` for stage attribution. Both use canonical StudySpec discovery. The numerical benchmark uses `python -m hpc_runs.intrmotiv_offpolicy.benchmark --help` for exact checkpoint/output arguments.

## Reusable lessons

Profile cumulative collection/learning counters first; batch independent work before changing algorithms. Validate actual decoder shapes and numerical parity before DMLab. Local SF source can lag the cluster decoder: fixture tests alone cannot qualify production conversion, so run actual-checkpoint parity in the isolated cluster checkout. Separate kernel benchmarks from whole-loop timing and distinguish startup time from learner-active FPS. On NEMO2 use Python or `grep --include='*.py'` when `rg` is unavailable, and keep environment tests in Slurm. Verify available upstream implementations before treating framework migration as a from-scratch task.
