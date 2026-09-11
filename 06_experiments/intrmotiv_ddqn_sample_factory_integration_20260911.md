# Native Sample Factory DDQN/HER integration — 2026-09-11

The final paired test reached **3,327 learner-active FPS with native SF versus 1,694 FPS with the batched standalone runner (1.96×)**, at identical optimizer and TD-position budgets. All 97 Sample Factory Python files were verified unchanged.

The DDQN/HER diagnostic now has a native Sample Factory execution backend, selected through the existing training entry point with `--execution-backend=sample_factory`. It uses the installed fork's learner/model factories rather than requiring the unmerged upstream DQN PR. The scientific loss, recurrent replay, HER and exact TD-position budget are reused.

## Reuse and necessary adapters

| Responsibility | Existing implementation reused |
|---|---|
| Process lifecycle, failure handling, shutdown | SF `ParallelRunner`, worker event loops and heartbeats |
| Environment collection and double buffering | SF `RolloutWorker`, `BatchedVectorEnvRunner`, `BatchedVecEnv` |
| Central batched inference | SF `InferenceWorker` |
| Shared tensors and rollout transport | SF `BufferMgr`, `TensorDict`, existing output copying |
| Training batch assembly and backpressure | SF `Batcher` |
| Model publication and inference copies | SF `ParameterServer`, `ParameterClientAsync` |
| TensorBoard/W&B and Slurm submission | SF runner summaries/W&B and existing canonical launcher |
| DDQN, target copies and optimization | Existing `DoubleDQNLearner` and batched `learn_batch` |
| Recurrent histories, HER and exact TD counts | Existing `SequenceReplay`, `PositionBatcher` and contracts |
| Child checkpoints, source verification, evaluation | Existing conversion/save functions and manifest-driven evaluator |

The custom adapter contains only the integration contracts absent from ordinary SF/PPO: physical stream identity, commanded goals/budgets and CA3 state, cached frozen-feature packets, ordered recurrent replay ingestion, exact off-policy update debt, and the learner/model factories. No SF source file was edited. A small opt-in shape extension declares `ddqn_packet`; SF's existing inference/rollout/batcher copies transport it. Ordinary SF configurations retain their original buffer schema.

SF's slice merger can reorder trajectory buffers, so buffer index cannot identify physical temporal order. The adapter holds a bounded queue keyed by stream and serial decision, then feeds the existing replay in physical order. A nonterminal tail waits for the next actor feature packet, including across rollout boundaries. This avoids another encoder pass and preserves exact successor/current feature equality. Invalid DMLab finals remain excluded, never replaced with reset observations.

The actor carries only CA3 memory and goal/budget/option state in SF's recurrent-state buffer. Stream/episode/index metadata is not fed to Q except for the already-declared physical episode clock. Goal-independent frozen writes allow memory to remain valid while Q parameters update asynchronously. A separate shared decision counter drives the existing epsilon schedule and is not overwritten by SF parameter publication. The model factory explicitly seeds Torch; the installed fork's inference initializer does not do that itself.

## Learning and sample accounting

After warmup, the learner completes one optimizer update per 64 accepted decisions and exactly 256 valid TD positions per update. Debt is calculated from completed updates and never discarded by an update cap. Neither policy age nor PPO likelihood ratios determine replay eligibility. Target copies remain every 100 optimizer updates. SF rollout length and training-batch size determine when data arrives, not how many TD positions are trained.

At shutdown, consumed frames and unconsumed transport tails are recorded separately. In the initial native pair, each run consumed 62,500 decisions (250,000 frames), including 32 invalid finals, and completed 720 updates/184,320 TD positions. It had received 63,488 transition packets and retained 988 outside the requested training budget. The runtime gate verifies `received = emitted + pending`, accepted plus invalid equals consumed decisions, and zero update debt. This is bounded asynchronous collection overhead, not policy-lag filtering.

The backend remains the frozen-reference local-control diagnostic. It does not add adaptive DG learning or an exploration manager. Replay still contains frozen preactivations, so adaptive DG requires the separately documented trunk-feature/reconstruction contract. No claim of improved task success or sample efficiency is inferred from FPS alone.

## Qualification

74 focused tests passed locally and on NEMO2: existing replay/worker/v2/throughput contracts, native packet/action consistency, reset identity, reordered delivery, missing/invalid successors, duplicate/bound rejection, debt auditing, backend dispatch and canonical studies. Actual SF serial and multiprocess smoke runs both completed 4,096 frames with 14 optimizer updates, 896 TD positions, four target copies and zero update debt. The multiprocess test uses SF's real shared-memory and parameter-client path, not mocks of its runtime.

The first DMLab pair compared SF synchronous and asynchronous execution at 32 environments, 16 rollout workers, two splits, 32-step rollouts, 1,024 transitions per delivered SF batch, eight inference Torch threads and one learner Torch thread. Both jobs completed successfully and passed the runtime audit:

| Native SF mode | Job | Learner-active FPS | Slurm elapsed |
|---|---:|---:|---:|
| Synchronous | 8056976 | 1,871.72 | 4m12s |
| Asynchronous | 8056977 | 3,802.29 | 3m13s |

This is 2.03× learner-active throughput within that pair. Startup accounts for a material fraction of these short jobs; Slurm elapsed and learner-active FPS answer different questions. These initial timings precede the final explicit inference-seed initialization, reduced duplicate summary writes and frame-milestone handling; the release comparison below tests the final source.

## Final release comparison

| Metric | Batched standalone | Native SF asynchronous |
|---|---:|---:|
| Job | 8056993 | 8056994 |
| Learner-active FPS | 1,693.51 | 3,326.88 |
| Slurm elapsed including startup | 4m06s | 3m19s |
| Consumed frames | 250,112 | 250,000 |
| Optimizer updates | 720 | 720 |
| Valid TD positions | 184,320 | 184,320 |
| Target copies | 7 | 7 |
| Invalid final exclusions | 32 | 32 |
| Frozen reference unchanged | yes | yes |

The final runtime audit passed, including identical seed-99 worker initialization, frozen-reference verification, terminal/TD contracts, checkpoint presence, finite metrics and native transport/debt accounting. Native SF stops at a single consumed-decision boundary; the standalone vector loop rounds up by its 32-environment step, explaining the 112-frame difference. This does not change the number of optimizer updates or TD positions in this comparison.

The 1.96× ratio applies after learner warmup. Whole-job wall time improved by about 19% in this short test because environment/process startup dominates a substantial fraction. These are two completed paired preflights, not a broad hardware benchmark or evidence that control success improves per sample. Actual trajectories and requested-HER fallbacks differ: final realized HER fractions were 0.345 versus 0.315. No scientific equivalence is claimed from matching runtime budgets alone.

The final source is selected by `nemo2_ddqn_sf_release.sh`; its hashes match the local integration files. Submission `20260911T212913Z` passed canonical command/path audit. [Release runtime audit](data/intrmotiv_ddqn_sf_20260911/release_runtime_audit.json), [submission audit](data/intrmotiv_ddqn_sf_20260911/release_submission_audit.json), and [source hashes](data/intrmotiv_ddqn_sf_20260911/source_sha256.json) preserve the evidence.

## Qualified entry point and deployment

Keep the usual parent-manifest, parent-run, registry, replay, frame and learning-budget arguments. Add:

```text
--execution-backend=sample_factory
--learner-execution=batched
--sf-async=true
--num-envs=32
--sf-workers=16
--sf-splits=2
--sf-rollout=32
--sf-batch-size=1024
--torch-threads=8
--sf-learner-threads=1
```

The command remains `python -m hpc_runs.intrmotiv_offpolicy.train`. The standalone backend remains its default. Native TensorBoard summaries include the same `ddqn/*` and `train/env_steps` tags; `--with-wandb` enables SF's existing integration (`--wandb-project` selects the project). The preflight jobs did not enable W&B.

The release runtime is `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ddqn_sf_release_20260911`. Existing scientific runs and their source were not changed. Child worker/checkpoint schema remains v2; checkpoints still require replay refill and are not exact resumes. The native entry refuses an existing run directory. Multi-policy/PBT, write-conditioned collection, variable frames per decision and certified final-image transport are outside the qualified adapter and rejected where encountered. GPU execution has not been benchmarked.

Both studies use schema `intrmotiv/study/v1`, workflow `1.8.0`, ordinary independent Slurm jobs, and the existing canonical launcher. Each job requested 40 CPUs, 80 GB, zero GPUs, and a 30-minute cap. All bulk artifacts, logs and caches remain under `/work/classic/fr_xl1014-train`.

- Initial study: `intrmotiv_ddqn_sf_preflight_20260911`, SHA-256 `5d91a7f7d740d43e3c13053036404e63bfe4db0328727f1b6cb885f765db52dd`; print review `20260911T212015Z`, submission `20260911T212101Z`.
- Release study: `intrmotiv_ddqn_sf_release_preflight_20260911`, SHA-256 `d24f0b021f2d8887f5db30966967430bb73085efc484b784e703103251287ac4`; print review `20260911T212757Z`.

Lightweight evidence is stored in [data/intrmotiv_ddqn_sf_20260911](data/intrmotiv_ddqn_sf_20260911/). Submission artifacts live under the corresponding `_slurm/STUDY/TIMESTAMP/` directory in the workspace training root. Runtime audits reuse `python -m hpc_runs.intrmotiv_offpolicy.audit_runtime STUDY BATCH_ROOT --output AUDIT.json`.

## Reusable experience

Start with SF's actual extension hooks and preserve its worker/transport implementation. Use a deterministic toy environment to exercise the real runtime before paying DMLab startup costs. The desktop sandbox initially blocked `torch_shm_manager`; an ordinary permitted execution passed, so replacing shared memory would have been the wrong repair. This SF fork's resume configuration helper expects a Namespace on its fresh-run path; the adapter uses guarded fresh-run configuration instead. Production parent normalization remains inside the existing feature extractor, exactly once.

Keep source hashes because a copied cluster checkout may not have usable Git metadata. Compare completed-update budgets before quoting throughput. SF's built-in worker profiles and canonical runtime audits are authoritative; use them rather than a new profiler or handwritten run list. The next performance work should follow those profiles, not add a second collection framework or change replay intensity to inflate FPS.
