# Odor and CA3 goal quality: terminal command interventions

**Status, 7 October 2026:** all 40 frozen 75M target-hit intervention jobs completed successfully and their 48,000 trials are analyzed. The separately required first-distinct-outcome protocol was absent from the original intervention setting. Its 40 evaluation-only jobs are now running under a separately fingerprinted StudySpec, using exactly the same training runs and checkpoints. This is an interim command analysis until those results arrive.

## Protocols and evidence

The ordinary target-hit evaluator starts only on an exclusive DG source event, balances five commanded trials over each of 240 off-diagonal directed source/target pairs, and stops a trial at the commanded target, a graph-derived deadline, or an episode boundary. It records every DG identity hit before stopping and computes a nearest-context, same-source shuffled alternative from another trial. Policy, DG, and graph buffers remain frozen. All 40 outputs contain 1,200 trials and all 240 complete pairs. Across 48,000 trials, 30,768 ended with the commanded target, 15,834 timed out, and 1,398 were censored at an episode boundary. Every summary carries the original workflow 1.14 StudySpec SHA-256 `c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de`.

The [per-run table](results/odor_v2_interventions_20261007/interventions_per_run.csv), [condition table](results/odor_v2_interventions_20261007/interventions_condition_summary.csv), [declared seed-paired contrasts](results/odor_v2_interventions_20261007/interventions_paired_contrasts_summary.csv), and [figure](results/odor_v2_interventions_20261007/interventions_success.png) are lightweight copies. [Analysis provenance](results/odor_v2_interventions_20261007/analysis_manifest.json) and per-run SHA-256 values bind them to the original JSON and trial files. The full trial records remain under `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/odor_ca3_goal_quality_v2_full_20261007/interventions/raw/`.

| Odor | Goal set | Commanded target hit | Matched shuffled target hit | Difference | Timeout | Episode censoring |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OFF | ALL16 | 62.8% | 40.9% | +21.9 pp | 34.2% | 3.0% |
| OFF | RANDOM4 | 57.1% | 37.2% | +19.9 pp | 39.8% | 3.1% |
| OFF | HEBB4 | 65.1% | 42.0% | +23.0 pp | 31.8% | 3.2% |
| OFF | RANDOM8 | 61.2% | 39.7% | +21.5 pp | 36.2% | 2.7% |
| OFF | HEBB8 | 67.4% | 43.9% | +23.5 pp | 29.7% | 2.9% |
| ON | ALL16 | 65.2% | 39.6% | +25.6 pp | 32.0% | 2.9% |
| ON | RANDOM4 | 56.6% | 39.0% | +17.5 pp | 40.5% | 2.9% |
| ON | HEBB4 | 68.8% | 41.8% | +27.0 pp | 28.2% | 3.0% |
| ON | RANDOM8 | 62.8% | 42.7% | +20.0 pp | 34.4% | 2.9% |
| ON | HEBB8 | 74.1% | 44.7% | +29.4 pp | 23.2% | 2.7% |

Percentages are means across the four paired learner seeds, with 1,200 trials per run. The figure shows seed standard errors. HEBB4's ordinary target-hit rate exceeds RANDOM4 by 8.0 percentage points OFF and 12.3 ON, even though [its 75M reliable graph reachability and external coverage are lower](odor_goal_quality_v2_spatial_analysis_20261007.md). These measures describe different properties; the target-hit comparison alone cannot establish a better controller.

## Why the first-distinct test is still necessary

The target-hit trial **ends when its commanded identity is reached**. Its shuffled target is checked only against DG events seen before that stop. A commanded target can therefore appear to beat the shuffled target because the trial was stopped in its favor, even if its command had little effect on actions. This is especially important here: the training-time alternative-goal action-distribution diagnostic was very small, while the target-hit protocol shows large apparent lifts. The target-hit CSV records a hit mask, not the order of wrong DG identities, so the first distinct outcome cannot be reconstructed afterward.

The [evaluation-only first-distinct StudySpec](../../hpc_runs/studies/odor_ca3_goal_quality_v2_20261007_first_distinct_analysis.study.json) adds only `terminate_on_first_distinct_exclusive_outcome: true`. Its expanded 40 training runs are byte-for-byte equal to the original run definitions; the training StudySpec and checkpoints remain unchanged. Its SHA-256 is `602ea435864c7a193d0563c365e406c93be91d80aceb3cfdbc8b5da8be1b8000`. The canonical renderer produced 40 intervention rows identical in condition, seed, label, checkpoint, and run directory to the original intervention manifest. The saved print-only plan passed a 40-command, ordinary-job, row-index, and workspace-path audit. Its submitted manifest is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/odor_ca3_goal_quality_v2_first_distinct_20261007/interventions/submission_manifest_20261007T175304Z.tsv`; all 40 commands match that reviewed plan exactly. No training or target-hit job was resubmitted.

Once those first-distinct jobs finish, report correct first outcomes, wrong first outcomes, timeouts, censoring, commanded-versus-matched-shuffled first-outcome fractions, and the declared paired contrasts. Relate those outcomes to the [place-field, trajectory, and graph analysis](odor_goal_quality_v2_spatial_analysis_20261007.md) and the [external coverage history](odor_goal_quality_40_run_v2_20261007.md), while keeping each protocol's stopping rule explicit.

## Reusable lesson

Check that a declared scientific outcome is actually identifiable from the saved trial schema **before** launching the evaluation matrix. A target-hit summary and a hit mask do not recover first-distinct event order. The efficient recovery here is a separately fingerprinted evaluation-only StudySpec with the one missing stopping-rule setting, exact checkpoint-manifest equality, and no change to the 40 trained agents.
