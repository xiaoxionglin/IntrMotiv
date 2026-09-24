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

## Production submission status

The final print-only CPU and GPU shard manifests were combined and audited against all 42 StudySpec runs. The DG-capacity shard uses CPU, 40 cores, 80 GB, and 96 hours. The PPO shard uses one RTX or L40S GPU, 40 cores, 80 GB, and 48 hours. Early production logs show both CPU and GPU runs advancing frames and receiving external reward.

At submission, Slurm accepted all 21 CPU jobs and the first 8 GPU jobs. Its per-user submitted-job limit rejected the remaining 13 GPU jobs. A quota-aware process records accepted GPU job IDs in the original manifest and retries missing jobs every five minutes. It will create a combined submitted audit after the last job is accepted. Its status log is `train_dir/_slurm/fixed_reward_dg_peak_transfer_20260924/gpu_submission_retry.log` in the new workspace. Seven CPU jobs encountered cluster `NODE_FAIL` during prolog and were requeued under their original job IDs; verify that they actually resume before analyzing the batch.

The source-aligned sites and three seeds limit any positive conclusion to transfer on these two tasks. The primary analysis remains held-out physical reward success at 0, 5, 10, 20, 50, and 100M frames, with paired-seed area over 0–20M and 0–50M, start-region success, and time to reward. Online reward and DG-hit metrics are diagnostics. The training batch has been launched but cannot yet support a transfer conclusion.

The [held-out evaluator](../hpc_runs/fixed_reward_runtime_20260924.patch) is staged in the isolated NEMO2 source checkout and records physical reward contact, start region, time to reward, and DG activity on matched held-out reset seeds. It has passed syntax checking but still needs a short Slurm qualification before use on the 0, 5, 10, 20, 50, and 100M checkpoints. Do not infer held-out performance from training rewards.
