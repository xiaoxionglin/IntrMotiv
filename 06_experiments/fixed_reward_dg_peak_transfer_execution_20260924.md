# Fixed-reward transfer execution record (2026-09-24)

## Study and source checkpoints

The validated [42-run StudySpec](../hpc_runs/studies/fixed_reward_dg_peak_transfer_20260924.study.json) has schema `intrmotiv/study/v1`, workflow version `1.12.0`, and SHA-256 `82c499f5b9c93b3df18ea686f327c3b614f97bebf61993cd712c89f998260dae`. It crosses two sites, seven learning arms, and seeds 42, 1234, and 9999. All arms target 100M environment frames. Source checkpoints were changed from the pruned 75M/150M artifacts to the earliest surviving artifacts, as authorized by the user:

| Site | Source checkpoint | Source frames | SHA-256 | Reward center |
| --- | --- | ---: | --- | --- |
| DG-capacity, goal 50 | `checkpoint_000015950_130662400.pth` | 130,662,400 | `6951c12faecda21848bd73a3422ffe1928dd2dc07a515ef1b8b9720efc2cc3c0` | (350, 1950) |
| Full-system PPO, goal 51 | `checkpoint_000021868_179142656.pth` | 179,142,656 | `2cbc50fca8658f938775496956502d24a52ed8e437037f0bf7fe0e32574ae694` | (250, 1950) |

The checkpoint copies, level runfiles, training outputs, Slurm logs, W&B files, caches, and raw source revalidation data are under `/work/classic/fr_xl1014-fixed-reward-transfer/IntrMotiv/SF_hipposlam/`. The source checkout contains code only.

## Qualification evidence

- Ten-thousand-decision source field jobs 8193716 and 8193717 completed. Goal 50 had raw spatial information 0.218 bits and active fraction 0.022; its selected reward bin had 529 observations and mean activation 0.463. Goal 51 had raw spatial information 0.458 bits and active fraction 0.024; its selected reward bin was the raw peak, with 1,105 observations and mean activation 1.311. The selected goal-50 bin is a visited response area, although a nearby bin at (450, 1950) had a stronger raw mean. This should be retained in interpretation rather than described as an exact peak.
- Native engine preflight 8193725 verified legal entry into each reward cell, +10 external reward, immediate termination, spawn exclusion, and unchanged visual map content. The interface retains eight actions, repeat 4, and 120-second timeout.
- Four short CPU qualifications and two GPU PPO qualifications completed. The latter ran on RTX with CUDA PyTorch 2.9.1. The model-only transfer audit found identical source DG and worker tensors in fixed-options qualification checkpoints, including BatchNorm running statistics; flat full transfer changed DG projection weights under reward learning.
- A manager diagnostic found a placement error in the initial reward-statistics update. After moving the update to the active `DistanceLearnerReward` batch path, job 8193839 completed with 960 goal-update counts and nonzero goal values in its 131k-frame checkpoint. Forty focused NEMO2 tests passed after the correction.
- Twenty matched resets per source were probed for 64 worker decisions with nominated, wrong, and shuffled goal IDs. Neither site had a physical reward contact in that short horizon. These zero-shot results are option-horizon diagnostics, not an episode-level success estimate or a training gate. CSVs record DG activity, start and terminal positions, and reward contact separately.

## Corrected qualification and site checks

The corrected candidate uses external reward for single-value PPO and manager returns, uniform flat goal mixtures, and a separate `W_GRAPH` arm that copies the source graph while resetting reward-goal values. It preserves the original `W_FIXED` arm as DG and worker transfer with an empty graph. This adds six downstream runs, making the candidate 48 runs if both sites qualify. The corrected eight-run qualification StudySpec has SHA-256 `901db9af4e7903b6dde1bd5eae42a6a8f777aa042616795c9fe33fb5984a49cd`.

Jobs 8195485–8195492 completed successfully across both source architectures. In every arm, the logged PPO reward mean exactly matched the separately saved environment reward mean at every common TensorBoard step; each arm had at least five batches with positive external reward. Initial checkpoints showed zero reward-goal counts in both waypoint arms, source graph edge-confidence sums of 4040.88 and 3667.80 only in the new graph arm, and exactly uniform flat mixture logits in both scratch and transfer arms. The deterministic held-out evaluator completed a two-reset Slurm smoke test (8195505). A telemetry-only code correction now keeps worker reward separately so the `intrinsic` diagnostic no longer repeats the selected PPO reward.

Twelve matched reset seeds with a 512-decision sustained command reached the exact reward cell 5/12 versus 3/12 for nominated versus shuffled DG 50, and 2/12 versus 1/12 for DG 51. Entry within radius 200 was 6/12 versus 3/12 for DG 50 and 2/12 versus 1/12 for DG 51. These small differences are insufficient to qualify either site as reliably commanded, especially DG 51. Additional matched seeds are running before any production decision.

## Production submission status

The final print-only CPU and GPU shard manifests were combined and audited against all 42 StudySpec runs. The DG-capacity shard uses CPU, 40 cores, 80 GB, and 96 hours. The PPO shard uses one RTX or L40S GPU, 40 cores, 80 GB, and 48 hours. Early production logs show both CPU and GPU runs advancing frames and receiving external reward.

Slurm accepted 33 jobs before the user requested that the study stop on September 25: all 21 CPU jobs and 12 GPU jobs. The quota-aware GPU retry process was stopped, then all 33 accepted job IDs were cancelled. The other nine GPU jobs were never submitted. A subsequent queue check found no jobs from this study running or pending. Seven CPU jobs had encountered cluster `NODE_FAIL` during prolog before cancellation.

The cancellation followed a confirmed reward-path error: the active single-value learner saved external rewards for telemetry, then replaced the PPO reward with intrinsic worker reward before GAE. Its manager received returns from that intrinsic stream. Consequently, the partial production outputs cannot support fixed-reward transfer claims, regardless of observed environment contacts. Keep their artifacts only as diagnostics and do not resume these checkpoints.

The corrected [candidate StudySpec](../hpc_runs/studies/fixed_reward_dg_peak_transfer_corrected_20260925.study.json) keeps the two sites and downstream seeds, starts all flat goal mixtures uniformly, and adds a separate source-graph waypoint arm. Its active learner selects external reward before GAE; the held-out evaluator defaults to deterministic actions and seeds stochastic actions when requested. Short corrected-reward qualifications and longer matched-command physical approach probes were submitted on September 25. Production remains stopped pending their results, a held-out evaluator preflight, and fresh print-only review.
