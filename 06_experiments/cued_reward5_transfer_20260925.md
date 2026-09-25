# Five-cue reward transfer batch — 2026-09-25

## Task and study

The source `openfield_map2` geometry and visual assets are retained. Each episode uniformly selects one of five invisible $+10$ reward cells, ends on first entry, and presents its stable number instruction to the high-level waypoint manager or the flat arm's goal mixture. All five cells are excluded from spawning. The worker and DG source-map identity remains the original map-3 input.

| Instruction | Reward center $(x,y)$ | Origin |
| --- | --- | --- |
| 1 | $(350,1950)$ | DG-capacity target-50 site |
| 2 | $(250,1950)$ | Full-system PPO target-51 site |
| 3 | $(1550,1250)$ | Additional reachable cell |
| 4 | $(650,1450)$ | Additional reachable cell |
| 5 | $(850,250)$ | Additional reachable cell |

The validated StudySpec is `hpc_runs/studies/cued_reward5_transfer_20260925.study.json` in the isolated source checkout. Its schema is `intrmotiv/study/v1`, workflow version `1.12.0`, and SHA-256 is `9922ce6fed52d11a6d7cff0d860a03860b219444417279976d6792602a721d59`. It declares both surviving source checkpoints, eight arms, seeds 42/1234/9999, and 75M frames per run: 48 runs total. W&B project: `SF_IntrMotiv_CuedReward5Transfer`.

Source branch: `codex/cued-reward5-transfer-20260925` at NEMO2 commit `52a6b6d6`, checkout `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_fixed_reward_transfer_20260924`. All outputs and caches are in `/work/classic/fr_xl1014-fixed-reward-transfer`. The production submission directory is `/work/classic/fr_xl1014-fixed-reward-transfer/IntrMotiv/SF_hipposlam/train_dir/_slurm/cued_reward5_transfer_20260925/production_submitted/`; `jobs.tsv` is the live accepted-job manifest, `submit.log` records quota retries, and `scancel.sh` cancels all accepted jobs.

The first jobs started just before that commit was created from the same deployed files, so their automatically captured W&B Git hash may be the parent `e389a7ab`, while the committed implementation is `52a6b6d6`. Use the StudySpec fingerprint and submission manifest alongside the source commit for provenance.

## Qualification and launch

The combined print-only audit found all 48 commands matched the StudySpec and all output paths stayed in the workspace. Four 262,144-frame CPU/GPU qualifications covering waypoint full refinement and flat full transfer completed with exit code 0. The cue-aware CUDA input issue found in the first GPU attempt was fixed before the clean GPU qualification. At shared TensorBoard steps, the PPO reward mean equaled environment reward mean; positive reward batches occurred in both waypoint qualifications and the CPU flat qualification. Source graph and policy tensors loaded in the relevant arms.

The native engine preflight confirmed cue changes, map resets, and physical reward contact for instruction 3. It was stopped once production training and the four qualifications showed reward pickups, to free the Slurm slot; it did not certify physical contact for every instruction individually. The structural map test checked all five reward markers and spawn exclusion.

The prior fixed-location production jobs were cancelled before this launch. On September 25, Slurm accepted the new production jobs beginning at ID `8196560`; CPU and GPU jobs entered `RUNNING`, and W&B run files appeared in the new campaign. The per-user `QOSMaxSubmitJobPerUserLimit` was reached after the first 21 accepted jobs. The quota-aware submitter continues to retry remaining StudySpec rows every 60 seconds and records each accepted ID in `jobs.tsv`. Check that manifest for the current count rather than treating 48 as already queued.

## Reusable lesson

For a cue-conditioned transfer campaign, preserve source DG/worker tensor shapes by carrying the task cue separately into the planner and flat mixture. Verify the actual PPO reward path with positive-reward batches. The print-only audit and quota-aware live manifest are authoritative for whether a large study is fully submitted; Slurm acceptance can be partial even when the study is valid.
