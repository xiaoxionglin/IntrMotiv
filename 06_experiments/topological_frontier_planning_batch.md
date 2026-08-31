# Topological Frontier Planning Batch

**Project:** `SF_IntrMotiv_TopologicalFrontierPlanning`  
**Group:** `intrmotiv_topological_frontier_planning_20260831`  
**Status:** implementation and preflight validation in progress  
**Source module:** `dmlab/experiments/topological_frontier_planning.py`

## Question

Can a geometry-free landmark graph explore efficiently by separating passive
local topology from deliberate controllability, scoring graph frontiers by
uncertainty, and conditioning the worker on explicit next-hop waypoints? Action
path integration is tested as a localization cue and local edge-length signal,
not as learned DG coordinates. One SE(2) pose graph is included only as a
matched metric control.

## Fixed Setup

- ImageNet-pretrained, frozen ResNet-18 through layer 2.
- `F=16`, `R=8`, `L=64`, DG threshold `2.43`.
- `encourage` encoder feedback and encoder batch loss enabled.
- Fixed-length no-reward DMLab, reduced five-action set, frameskip 8.
- Learner-owned policy graph, one policy, no PBT, simultaneous update.
- `hit_distance` deliberate waypoint reward and dense temporal-distance reward
  during bounded 64-decision frontier exploration.
- 32 workers x 2 environments, CPU, 100M frames per run.
- Slurm: 40 CPUs, 80 GB, 30 hours; all bulk outputs under
  `/work/classic/fr_xl1014-train`.

## Conditions

| Cell | Condition |
| ---: | --- |
| 1 | Flat `G001_R100 X0 O0` reference. |
| 2 | Existing direct-target `CTRL_X0_O1` HRL reference. |
| 3 | Passive graph plus UCB frontier, direct final target. |
| 4 | Frontier plus next-hop planning with elapsed-time edges and no action integration. |
| 5 | Full topology and action integration; motion excluded from worker; no scatter loss. |
| 6 | Cell 5 plus motion-conditioned worker. |
| 7 | Full topology, graph-only motion, scatter `0.01`. |
| 8 | Full topology, motion-conditioned worker, scatter `0.01`. |
| 9 | Cell 8 with scatter `0.005`. |
| 10 | Cell 8 with scatter `0.05`. |
| 11 | Cell 8 with scatter distance `4`. |
| 12 | Cell 8 with scatter distance `12`. |
| 13 | Cell 8 with UCB uncertainty weight `0.5`. |
| 14 | Cell 8 with UCB uncertainty weight `2.0`. |
| 15 | Cell 8 on `G001_R100 X1 O0`. |
| 16 | Cell 8 plus SE(2) pose-graph control. |

Each cell uses seeds `8`, `99`, and `123`, for 48 independent jobs.

## Mechanism Checks

The three 2M-frame preflights cover cell 4, cell 8, and cell 16. They must run
beyond multiple 900-decision physical episodes and establish:

1. nonzero passive updates and candidate edges;
2. no passive edge in shortest-path planning before deliberate validation;
3. RETURN and VALIDATE events with at least some deliberate successes;
4. nonzero route availability and explicit multi-hop first-hop selection;
5. nonzero waypoint intrinsic reward when targets are hit;
6. stable PPO replay after policy-graph and pose-buffer changes;
7. graph persistence across terminals without reset-spanning transitions;
8. action-path and scatter diagnostics for motion runs;
9. finite SE(2) stress and initialized poses in the control;
10. workspace-only training, cache, W&B, and Slurm output paths.

Production submission is gated on these checks and a 48-job print-only manifest
audit. The complete metric definitions are in
[[IntrMotiv_metric_reference|IntrMotiv Metric Reference]].

## Analysis Order

First reject runs with collapsed or ambiguous DG activity. Then inspect passive
topology, deliberate validation, validated route use, target-hit lift, and
finally coverage AUC, unique cells, entropy, and place-field telemetry. This
ordering separates successful behavioral exploration from operation of the
intended graph mechanism.
