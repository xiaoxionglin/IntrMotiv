# Recurrent replay throughput repair — 12 September 2026

## Release requirement

The user requires completing implementation to obtain throughput comparable to
previous IntrMotiv batches, preserving the approved controller-learning contract.
The original 18-run production objective remains active, but the old slow pipeline
must not be released merely because its correctness preflights complete. Measure
both complete optimizer transactions and sustained real-run throughput. Do not
claim an isolated packing speedup is a comparable-throughput result.

Reference: `06_experiments/intrmotiv_full_system_controller_20260912.md`.
All existing preflights remain on their immutable sources while candidates are
qualified. Final audits 8057457/8057458 are still pending their dependencies.

## Existing mechanisms to reuse

Sample Factory already provides `TensorDict`, `init_tensor`, indexed trajectory
buffers, observation preparation, and recurrent packing. Its local checkout does
not provide an off-the-shelf persistent DDQN/HER replay implementation.

Established references:

- [SB3 HER replay source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/her/her_replay_buffer.html): batched virtual goals and rewards. Its default mixing replaces some real examples; retain our separately budgeted main and auxiliary objectives.
- [TorchRL recurrent DQN](https://docs.pytorch.org/rl/stable/tutorials/dqn_with_rnn.html): recurrent transport and replay learning; stored recurrent state is not automatically exact under changed representations.
- [TorchRL SliceSampler](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.replay_buffers.SliceSampler.html): trajectory-aware contiguous sampling. Do not change our uniform main-position sampling merely to copy a sequence sampler.

## Candidate 1: SF tensor storage and shared raw gathers

Local: `/tmp/intrmotiv_tensor_replay_20260912`.
NEMO2: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_tensor_replay_20260912`.

`controller_tensor_replay.py` uses SF's tensor allocation and recursive indexing.
A derived CPU tensor mirror of physical replay replaces repeated NumPy stacks;
physical rows retain ownership, sampling order and checkpoint schema. The mirror
is not serialized and is rebuilt lazily after restart. Ring-slot ownership checks
prevent terminal successors or evicted held rows from aliasing overwritten data.
Two raw gather entries can be shared by online and target evaluations, without
sharing learned normalization or representation outputs. HER future histories use
one batch gather. This costs an additional replay-sized CPU tensor mirror; measure
memory as well as speed before deployment.

Local runtime tests: 387 passed. Remote runtime plus controller-audit tests:
392 passed. The existing storage-policy test incorrectly assumed pytest's temporary
directory was outside the allocated workspace; it now explicitly checks a home
path and prevents file writes even if validation fails. Decoder test helper imports
were made package-qualified for importlib test collection. Neither fix changes
training behavior.

GPU jobs 8057459 (direct) and 8057460 (waypoint) compare an original decoder-only
reference against this candidate from the same immutable checkpoint. Ten plain
DDQN optimizer updates preserve exact models, optimizer, target, clocks and RNGs:

| Architecture | Original seconds | Tensor seconds | Speedup |
|---|---:|---:|---:|
| Direct F16 | 18.5893 | 17.2596 | 1.077× |
| Waypoint decoder F64 | 38.4593 | 27.3749 | 1.405× |

These gains are insufficient for the user's throughput requirement. Both jobs
also perform 101 HER updates across a target refresh; inspect their final results.
Reports: `train_dir/analysis/controller_tensor_replay_{direct,waypoint}_profile.json`.
Do not deploy this intermediate candidate as a finished performance fix.

## Candidate 2: reuse encoded inputs within a frozen snapshot

Local: `/tmp/intrmotiv_snapshot_replay_20260912`.
NEMO2 intended path:
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_snapshot_replay_20260912`.

`SnapshotHeadCache` lives for one fresh-DG/controller transaction. Under STOP,
DG does not change inside that controller block. Cache encoder outputs only when
the remaining encoder parameters are fixed too; JOINT and trainable text/other
encoder components retain full reconstruction. Within functional replay, caching
asserts that the encoder is frozen and that returned features require no gradient.

Online and target models have separate cache entries. Target refresh explicitly
invalidates its entry. A fresh DG transaction creates a completely new cache.
Only actual physical-row observation identities use stable slots; external
certified successors are encoded separately. Missing rows are encoded in the
same fixed 256-row geometry as the reference, then gathered in requested order.
No cached learned feature enters a checkpoint or actor publication.

Once encoded inputs are available, replay gathers only conditions and manager
contexts. Existing finite-memory reconstruction accepts the precomputed head
and preserves its core, decoder, rewards, sampling and optimizer geometry.
This removes redundant observation uploads and encoder execution across main,
eligibility and auxiliary evaluations while retaining the actual decoder updates.

Local runtime tests: 391 passed, including cache reuse, independent targets,
invalidation, rejection of trainable encoders, and exact main/HER values and
shared-worker gradients. Remote runtime plus audit tests: 396 passed. GPU jobs **8057462 (direct)**
and **8057463 (waypoint)** are running. Ten main updates are bitwise identical;
initial speedups are only 1.271× and 1.378×, so more work is required.
Expected report/job records in `train_dir/analysis/`:
`controller_snapshot_replay_tests.log`,
`controller_snapshot_replay_profile_jobs.json`, and
`controller_snapshot_replay_{direct,waypoint}_profile.json`.

## Measurement and workflow lessons

At approximately cluster log time 16:50, old-run 30-minute throughput was 72.8 FPS
direct DDQN, 36.4 direct HER, and 54.6 for both waypoint DDQN arms. Earlier
completed PPO whole-run averages were 3,031 FPS direct and 1,474 FPS waypoint.
Find a matched earlier HER-like baseline before claiming comparable throughput;
those numbers alone use different workloads and averaging windows.

Keep scientific acceptance checks and ordinary main-update counts unchanged.
Do not hide overhead by lowering replay updates, truncating histories, relaxing
recognition, reusing stale target features or dropping auxiliary work. Require
exact complete optimizer-transaction comparisons, including a target refresh,
then test restart and real ingestion before deploying a performance candidate.
Read a real GPU profile when a candidate's measured gain is insufficient.
Never edit a source checkout while its training or qualification job is running.
All bulk artifacts remain in the allocated workspace.

## Candidate 3 in development: skip computations unused by selection

Local source: `/tmp/intrmotiv_replay_work_20260912`; no remote copy yet.
Based on snapshot-cache candidate, with two exact work eliminations:

- Main eligibility search reconstructs canonical/event state but skips decoder,
  Q prediction, Bellman losses and reward calculations that search does not use.
  Selected main/HER updates retain the complete original calculations.
- HER future-goal selection uses canonical DG head activity directly: it equals
  trace slot zero. It no longer reconstructs preceding or future worker-memory
  states solely to extract those activities. Actual auxiliary TD evaluation still
  reconstructs its required memory under each snapshot.

391 existing runtime tests and four new eligibility/canonical-equivalence tests
pass locally. Remote staging and full GPU qualification remain to do.
A five-update cProfile job **8057464** measures the snapshot-cache candidate's
remaining waypoint HER cost; output `controller-replay-stage-profile-8057464.out`.
A separate CPU probe has been submitted from `controller_cpu_replay_profile.sh`
for the same waypoint checkpoint workload at eight Torch threads; record its job
ID and results before considering any device change. These are diagnostics,
not production changes.

## Authoritative earlier HER baseline and correction

The actual earlier recurrent HER implementation is documented in
`06_experiments/intrmotiv_ddqn_throughput_20260911.md`, with code in
`hpc_runs/intrmotiv_offpolicy/batch.py` and `worker.py`. It achieved **2,004.60
learner-active FPS** in Slurm job 8056867 versus 1,025.05 reference, on 40 allocated
CPUs/eight Torch threads, no GPU. Both performed 720 updates and 184,320 TD
positions over 250,112 frames. It used frozen DG, batched prefix rebuilding,
read-only prefix sharing when valid, and flattened time/batch decoder readout.
The earlier explanation that this was merely a fresh-batch HER update was
incorrect for these recurrent DDQN/HER runs; this correction was given to the user.

Its original numerical qualification allowed floating-point batching differences:
maximum parameter difference below 4.1e-7 and loss difference below 1e-9 over
three optimizer updates, with exact accounting. Do not confuse bitwise trajectory
identity with the user's algorithmic preservation requirement when batching
row-independent Q operations. Keep discrete reward/recognition/boundary checks
strict and measure numerical differences explicitly. Prefer reusing this proven
batching approach over proliferating new replay workflows.

Candidate 1's 101-update HER GPU checks completed with exact optimizer/target/RNG
parity across refresh: direct 380.28→344.21 seconds (1.105×), waypoint
433.47→368.25 seconds (1.177×). These are not adequate final throughput gains.

## Profile-driven removal of redundant work

The five-update waypoint profile, job **8057464**, measured 18.117 seconds and
13.61 million Python calls. Manager advancement alone consumed 5.51 seconds,
including 2,722 calls to `choose_task`; 220,312 `Tensor.item` calls consumed
1.63 seconds. Replay was computing new manager plans even though its worker
reward uses the recorded action-time command and reconstructed outcome pulses.
This was avoidable integration overhead, not a necessary cost of auxiliary HER.

Candidate 2's completed 101-update HER comparisons preserved exact optimizer,
target and RNG state: direct 371.824→209.945 seconds (1.771×); waypoint
388.430→248.043 seconds (1.566×). CPU diagnostic **8057466**, eight Torch threads,
was slower: ten HER updates 62.486→41.773 seconds. Moving this implementation to
CPU is not a remedy.

Candidate 3 now factors `option_outcome_masks` out of the real topological
manager and reuses it for replay event reconstruction. Replay retains the actual
successor context and overwrites only outcome fields used by reward and event
validation. It does not run graph routing or successor task selection. The fast
path covers the approved parents without motion filtering; other parents retain
the original reconstruction. Exploration/passive-event priority is preserved.
Four additional tests cross target vocabulary, modes, expiry, recognition and
passive ages, comparing against complete manager advancement without graph
mutation. HER future selection now transfers canonical activities once per
snapshot and uses vector masks, preserving future ordering and duplicate goals.
Current-label and already-achieved checks are batched as well.

Candidate 3 source: `/tmp/intrmotiv_replay_work_20260912`; remote immutable copy
`SF_hipposlam_controller_replay_work_20260912`. Local runtime suite: 399 tests
(including the four new manager tests); remote runtime plus audit tests passed
before comparison jobs **8057467 direct / 8057468 waypoint** were submitted.
These are optimizer comparisons, not production jobs.

Candidate 4, `/tmp/intrmotiv_vector_replay_20260912`, additionally skips complete
worker-history reconstruction for eligibility-only queries in supported parents.
It batches main/auxiliary Q readout and Bellman/Huber operations, following the
previous implementation's flattened readout pattern. Local runtime: 399 passed;
remote tests passed; GPU jobs **8057470 direct / 8057471 waypoint** compare against
the original implementation. Discrete labels, rejection counts, update clocks and
RNG state must remain exact; floating model/optimizer tensors are measured with
explicit `atol=1e-6, rtol=1e-5`, and all nonzero differences are recorded.
This is a numerical batching qualification, not an assertion of bitwise equality.

Candidate 5 is local only at `/tmp/intrmotiv_batched_context_20260912`. It batches
context overrides and current/previous/successor state assembly, and keeps
physical decision IDs on CPU for ordering validation. These IDs are metadata,
not GPU model inputs. This removes per-row scalar synchronization and tiny
context-write kernels. All 399 runtime tests pass; remote qualification remains.

Do not describe any of these intermediate improvements as comparable sustained
throughput until complete fresh-data/controller transactions have been measured.
