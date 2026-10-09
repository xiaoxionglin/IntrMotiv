# Open analyses and claim gates

This register describes gaps in the **checked-in analyses**; the original inventory
was made on 6 October 2026 and rows are updated as studies advance. It is not the live state of cluster jobs. A later checkpoint or a submitted
job does not close a gap until its matched result and provenance are reported.
Use the validated StudySpec and canonical collection workflow for repeated
work; [the experiment index](README.md) defines the report owner for each line.

| Priority | Study and current evidence | Missing analysis | Completion criterion |
| --- | --- | --- | --- |
| 1 | [Episode-long DG comparison](dg_goal_control/episode_long_dg_comparison_20261009.md): all twelve 150M and six older-credit 100M frozen field jobs completed and analyzed; 36 longer-credit reward-factor cells, online trajectories, directed graphs, and partial 75M exact-start panels are reported | Paired three-seed *physical* command-arrival panels for prescribed DG at 75M, 100M and 150M, with independent geometric distances, failures and episode censoring. The 75M source-0 qualification passed with eight exact starts and 72 executed rows; fifteen remaining source shards were submitted 10 October. | Validate all shard outputs and exact starts, complete paired fixed-field arrival at 64/128/256/900 decisions, and keep learned detector hits separate from physical destination claims. Do not treat 150M fields, online hit rates or graph confidence as a substitute. |
| 1 | [CA3 state-goal follow-up](ca3_goals/ca3_followup_analysis_20260926.md): balanced CPU/G500 75M factorials, persistent UNIQUE graph sparsity, and mostly zero grounded control | Frozen matched-command versus shuffled-command outcomes from identical starts, with predictive-signature alias diagnostics on compatible retained checkpoints | Report executed/shuffled lift, ordered-pair support, recognition collisions and event opportunities by arm and seed. Keep CPU and G500 separate. |
| 1 | [Five-cue transfer](cued_reward5_transfer_20260925.md): paired source/random controls have exact 75M heldout outcomes in the [endpoint analysis](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings) | Completed matched outcome analysis for the declared eight-arm, 48-run campaign, including scratch and transferred worker/graph arms | Audit all run/checkpoint identities; compare each declared contrast at a common age using paired seeds, per-cue heldout success, censored time, reward AUC, and source-component provenance. |
| 2 | [CA3 predictive active goals](ca3_goals/ca3_predictive_active_goals_interim_analysis_20260924.md): all 21 arms have a 75M online spatial window, but selected scalar histories were stale in the discovered event path | Recover validated matched scalar histories and later complete checkpoints; test frozen goal choice and contextual aliasing | One balanced seven-cell panel per claimed age with coverage, option outcomes, prediction/shuffle diagnostics, recognition events, frozen place fields, and matched-command outcomes. Do not pool partial 150M rows. |
| 2 | [DG capacity and goal conditioning](dg_representation/dg_capacity_goal_conditioning_interim_20260911.md): balanced 25M matrix and a later direct-only subset; [poster endpoint](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings) has age-unmatched F64 waypoint candidates | Equal-age waypoint-versus-direct comparison and broader command coverage | Use a genuinely shared checkpoint or label individual endpoints descriptive. Report map, graph, coverage, and matched-command outcomes with ordered-pair denominator and seed support. |
| 2 | [Navigation8](controllers/navigation8_algorithm_screen_interim_20260916.md): three-seed matched 75M online windows; selected seeds lack one exact shared saved terminal model | Complete frozen-policy and matched-command comparison at a verified common saved age, or document why it cannot be formed | Checkpoint inventory for every arm and seed, then a protocol-matched frozen comparison with explicit missing rows and no latest-per-run substitution. |

## How to close a row

Update the study's result report in place: status and shared checkpoint first,
factorized design and declared contrast next, then results, interpretation
limits, figures, and source manifests. Keep a restricted subset explicitly
labeled if the full matrix is unavailable. Link the completed report from the
[experiment index](README.md), then remove or revise the corresponding row
here. The canonical [study workflow](../04_implementation/standardized_study_workflow.md)
and [place-field telemetry guide](../04_implementation/reusable_place_field_telemetry.md)
define collection and submission contracts.
