# Five-cue reward transfer batch — 2026-09-25

## Architecture factorization

| Factor | Levels / setting | Interpretation |
| --- | --- | --- |
| Downstream task | Five invisible reward locations, uniformly selected per episode and cued by stable number instruction | Broader multi-goal transfer than the earlier single fixed reward |
| Geometry / sensory map | Original openfield_map2 retained | Reward location changes without changing the base visual geometry |
| Source representation | Two surviving source checkpoints plus scratch/transfer variants in the eight declared arms | Source identity must be tracked separately from transfer operation |
| Goal signal | Number instruction specifies the external reward location | Instruction is downstream task context, not privileged coordinate input |
| Controller / transfer | Waypoint manager or flat goal mixture depending arm; external reward drives learning | Different arms reuse different amounts of source structure |
| Evaluation | 48-run, 8-arm × 3-seed production design, 75M frames | This is designed to test reusable multi-goal structure rather than one-site specialization |

Cross-report context: [[README|factorized experiment synthesis]].


## Task and study

The source `openfield_map2` geometry and visual assets are retained. Each episode uniformly selects one of five invisible $+10$ reward cells, ends on first entry, and presents its stable number instruction to the high-level waypoint manager or the flat arm's goal mixture. All five cells are excluded from spawning. The worker and DG source-map identity remains the original map-3 input.

| Instruction | Reward center $(x,y)$ | Origin                         |
| ----------- | --------------------- | ------------------------------ |
| 1           | $(350,1950)$          | DG-capacity target-50 site     |
| 2           | $(250,1950)$          | Full-system PPO target-51 site |
| 3           | $(1550,1250)$         | Additional reachable cell      |
| 4           | $(650,1450)$          | Additional reachable cell      |
| 5           | $(850,250)$           | Additional reachable cell      |

The validated StudySpec is `hpc_runs/studies/cued_reward5_transfer_20260925.study.json` in the isolated source checkout. Its schema is `intrmotiv/study/v1`, workflow version `1.12.0`, and SHA-256 is `9922ce6fed52d11a6d7cff0d860a03860b219444417279976d6792602a721d59`. It declares both surviving source checkpoints, eight arms, seeds 42/1234/9999, and 75M frames per run: 48 runs total. W&B project: `SF_IntrMotiv_CuedReward5Transfer`.

Source branch: `codex/cued-reward5-transfer-20260925` at NEMO2 commit `52a6b6d6`, checkout `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_fixed_reward_transfer_20260924`. All outputs and caches are in `/work/classic/fr_xl1014-fixed-reward-transfer`. The production submission directory is `/work/classic/fr_xl1014-fixed-reward-transfer/IntrMotiv/SF_hipposlam/train_dir/_slurm/cued_reward5_transfer_20260925/production_submitted/`; `jobs.tsv` is the live accepted-job manifest, `submit.log` records quota retries, and `scancel.sh` cancels all accepted jobs.

The first jobs started just before that commit was created from the same deployed files, so their automatically captured W&B Git hash may be the parent `e389a7ab`, while the committed implementation is `52a6b6d6`. Use the StudySpec fingerprint and submission manifest alongside the source commit for provenance.

## Qualification and launch

The combined print-only audit found all 48 commands matched the StudySpec and all output paths stayed in the workspace. Four 262,144-frame CPU/GPU qualifications covering waypoint full refinement and flat full transfer completed with exit code 0. The cue-aware CUDA input issue found in the first GPU attempt was fixed before the clean GPU qualification. At shared TensorBoard steps, the PPO reward mean equaled environment reward mean; positive reward batches occurred in both waypoint qualifications and the CPU flat qualification. Source graph and policy tensors loaded in the relevant arms.

The native engine preflight confirmed cue changes, map resets, and physical reward contact for instruction 3. It was stopped once production training and the four qualifications showed reward pickups, to free the Slurm slot; it did not certify physical contact for every instruction individually. The structural map test checked all five reward markers and spawn exclusion.

The prior fixed-location production jobs were cancelled before this launch. On September 25, Slurm accepted the new production jobs beginning at ID `8196560`; CPU and GPU jobs entered `RUNNING`, and W&B run files appeared in the new campaign. The per-user `QOSMaxSubmitJobPerUserLimit` was reached after the first 21 accepted jobs. The quota-aware submitter retried as slots opened and ultimately recorded all 48 job IDs in `jobs.tsv`; check that manifest for exact identities and live Slurm for current states.

## Matched frozen-DG controls

The follow-up StudySpec `hpc_runs/studies/cued_reward5_frozen_dg_controls_20260925.study.json` has schema `intrmotiv/study/v1`, workflow `1.12.0`, SHA-256 `e3d4f5a1d29240dcb415b9ee8a163f236b1b41c145f6c766f4f3056698a1a32a`, and 12 runs: two source architectures, seeds 42/1234/9999, and two waypoint arms. Both arms use a fresh worker, fresh reward manager, empty graph, identical five-cue task, and 75M frames. `W_RAND_DG` freezes freshly initialized random DG projection weights, calibrates BatchNorm moments on one unlabeled learner minibatch before the PPO update, then freezes the moments. `W_SOURCE_DG` loads only source DG projection and BatchNorm tensors, then freezes both. Their paired difference tests whether the pretrained DG initialization helps relative to a calibrated random representation; the original `W_FIXED` arm also transfers a worker and therefore is not its direct control.

The add-on source is isolated from running production at NEMO2 checkout `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_cued_reward5_frozen_dg_20260925`, commit `20fbcbe0`. The first 262,144-frame qualification showed a frozen random DG was completely silent with untouched default BatchNorm moments; those four jobs and their original StudySpec remain as failure evidence. The one-minibatch calibration revision passed 19 focused tests on NEMO2 and fresh 4-run and 12-run print-only audits. The revised qualification StudySpec SHA-256 is `bbb4ad077375835a3c810a5de8c1ff2df0a70482e87deb9f64b924e9cdffe7b0`; jobs `8205826`–`8205829` all completed successfully. Random-DG density was 0.0127 on CPU and 0.0111 on GPU, with active-transition fractions 0.528 and 0.489. Both random DGs had exactly one normalization update and unchanged weights; both source DGs kept weights and moments fixed. All four runs had positive external reward batches and exact PPO/external reward equality at shared steps.

The qualification gate report is in the production submission directory. It released all 12 audited production controls, Slurm IDs `8208173`–`8208186` with gaps in the cluster's global ID sequence. The submitted audit reports `submitted_complete=true`, 12 unique IDs, exact StudySpec commands, and valid workspace paths. At the launch check, all six GPU controls were running and the six CPU controls were pending scheduler priority. The paired 75M-frame outcome remains to be measured.

## Reusable lesson

For a cue-conditioned transfer campaign, preserve source DG/worker tensor shapes by carrying the task cue separately into the planner and flat mixture. Verify the actual PPO reward path with positive-reward batches. The print-only audit and quota-aware live manifest are authoritative for whether a large study is fully submitted; Slurm acceptance can be partial even when the study is valid.
