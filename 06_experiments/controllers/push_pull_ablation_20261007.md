# Push–pull temporal-distance ablation: execution record

Status: **intact-reference calibration running**, 7 October 2026. No four-arm ablation or production comparison has been submitted. Scientific design: [COSYNE ablation plan](../../05_plans/cosyne_push_pull_ablation_20261007.md).

## Project and source

- [W&B project](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_PushPullAblation) for all new runs: `SF_IntrMotiv_PushPullAblation`. The C15 job log confirms W&B synchronization to this project. Calibration groups are `intrmotiv_push_pull_c15_calibration_20261007` and `intrmotiv_push_pull_cpu2048_calibration_20261007`. The existing local IntrMotiv Git worktree remains in use.
- Isolated NEMO2 source snapshot: `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_push_pull_ablation_20261007/`, archived from SF_hipposlam commit `cdf9ffe9026dc483e9c31dae7a1365af67c28642`. All bulk outputs go to `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/`.
- Reproducible runtime delta: [push–pull patch](../../hpc_runs/patches/push_pull_ablation_20261007.patch), SHA-256 `1c280baf97d6a348c161333d8edd9359d6938d51db2dae17aababdac8aa6df56`. `git apply --check` passed against the clean base without modifying it. Source-file SHA-256 values: `custom_learner.py` `c80ead0d99bd1997caa956dc165831329918a7f681d1ae3cbd38ecb39e10997c`, `custom_params.py` `b1409de2d0eceb9266025962d6cd0bbb05dc5034647cac571c55107c84924346`.
- The patch adds `temporal`, `constant`, and `none` modes at the encoder-credit and target-hit bonus boundaries, including PPO empirical HER and the stored magnitude consumed by DDQN+HER. Temporal remains the default. C15's original unmatched encoder credit and CPU2048's matched-arrival credit retain their own event masks.

## Submitted calibration

Both StudySpecs use schema `intrmotiv/study/v1`, workflow `1.14.1`, seed 99, the intact temporal modes, and a 2M-frame horizon. They are calibration runs, not evidence of an ablation effect.

| Family | StudySpec and SHA-256 | Slurm job | Submitted manifest |
| --- | --- | ---: | --- |
| Original corrected-core C15 PPO | [C15 calibration](../../hpc_runs/studies/push_pull_c15_calibration_20261007.study.json), `2a923dc688580289ae0ae6eae742d8b2ebcadf82bd702f955ab84760810d3678` | `8293691` | `.../_slurm/intrmotiv_push_pull_c15_calibration_20261007/20261007T144705Z/jobs.tsv` |
| CPU2048 Direct F16 DDQN+HER | [CPU2048 calibration](../../hpc_runs/studies/push_pull_cpu2048_calibration_20261007.study.json), `ee2a9b1bea367db6a41045581f586eb10299c84151b304206edf99e6ade7df5f` | `8293692` | `.../_slurm/intrmotiv_push_pull_cpu2048_calibration_20261007/20261007T144723Z/jobs.tsv` |

The manifest prefixes in the table are under `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/`. Both print-only and submitted `audit-submission` checks passed with exactly one run, workspace paths valid, and commands matching the StudySpec. The real IntrMotiv parser accepted both commands, including historical C15 decoder layers `[128, 128]`, legacy BatchNorm, and the dedicated W&B project. The isolated source passed 49 focused learner/HRL tests; the synchronized workflow passed 42 focused tests. The older `dmlab_pack` Python lacks pytest and uses Python 3.8; the established SFgit Python 3.10 was used for the passing tests.

## Calibration rule and next gate

Use the completed intact-reference histories over the declared 0–2M interval. For original C15, estimate $c_{\rm enc}$ as the dominant-event-count-weighted mean of `intrmotiv/encoder/feedback_on_dominant_event_mean` divided by the reward scale. For CPU2048, weight `intrmotiv/encoder/credit/source_lag_mean` by `intrmotiv/encoder/credit/credited_events`. Estimate $c_{\rm hit}$ from the correct-outcome-count-weighted mean of `intrmotiv/hrl/control/correct_reward_magnitude_mean`, subtract the base hit reward, and divide by the distance-bonus coefficient times reward scale. Use only finite values and record denominators and variability. If either reference has fewer than 100 qualifying events, use the predeclared theoretical midpoint $35.5$ distance units for that coefficient and report that its mean scale was not empirically matched. Freeze one pair of constants per family before rendering any four-arm StudySpec.

The submitted StudySpecs contain provisional analysis metric aliases for encoder feedback and worker magnitude. The actual runtime tags are the paths in the preceding paragraph; the submitted StudySpecs must remain immutable. Use the raw scalar histories to compute the constants and validate the correct production metric tags before submission. This is an analysis metadata correction, not a change to the submitted training commands.

After calibration, create separate complete Cartesian $2\times2$ StudySpecs for C15 and CPU2048, with seeds 8, 99, and 123, all routed to the same W&B project. Run print-only review, real-parser validation, and four-arm correctness preflights before production. The full study remains open until matched online outcomes plus place-field, trajectory, and graph analyses are complete.
