# Easy landmark maze: rich SCR and DGP place fields at 100M

These are the existing seed-99 online snapshots at the 100M target (actual 100,007,936 frames), rendered with the canonical `segmented-atlas/v1` field renderer from verified workflow 1.14. Each snapshot retains 100,000 samples. No new rollout was run. All rich SCR and DGP seeds reached this target; rich Waypoint and all neutral runs did not, so these figures show within-rich development rather than a matched cue contrast.

| Family | All DG units |
| --- | --- |
| SCR | [Units 0–15](ELM_RICH_SCR_ARR_DIRS_S99/target_000100000000_policy_00_place_fields_page01.png) |
| DGP | [Units 0–15](ELM_RICH_DGP_HIT_JOINT_LEG_S99/target_000100000000_policy_00_place_fields_page01.png) |

Each unit is normalized by its own peak, so color shows within-unit field shape, not absolute response. Gray bins were unvisited and black bins are walls. Squares mark decal-adjacent floor cells; diamonds mark colored-wall-adjacent floor cells. The [production report](../../../environments_transfer/easy_landmark_maze_production_analysis_20261006.md) gives the all-unit cue-site match counts and the limits of interpreting proximity as visual recognition. Open the images at full size to inspect cue IDs.
