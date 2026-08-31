# Frontier Manager Isolation: 2026-08-31

## Scope

This is the first, deliberately narrow experiment extracted from the deferred
topological-frontier HRL plan. It compares the current direct-target
controllable-graph policy to staged topological managers.

It includes RETURN/VALIDATE behavior and bounded local exploration in the
topological arms; the two waypoint rows additionally test next-hop routing.
It does not test action path integration, motion-conditioned worker input, the
DG scatter loss, or the SE(2) pose-graph control.

This is not a causal UCB-only ablation: a difference between `DIRECT_TARGET`
and either frontier condition can arise from UCB selection, passive-edge
validation, or the topological manager's 64-step dense-reward exploration
phase. The clean planning comparison is `PASSIVE_UCB_FRONTIER` versus
`PASSIVE_UCB_FRONTIER_WAYPOINT`; a future UCB-only comparison requires a
matched bounded-exploration schedule in the direct condition.

The source now contains that causal follow-up as
`frontier_manager_matched_control.py`. Its two five-seed rows both retain the
topological state machine, passive evidence, RETURN/VALIDATE, and the bounded
local exploration phase. `TOPOLOGY_VISIT_DIRECT` uses novelty-only visit rank;
`TOPOLOGY_UCB_DIRECT` adds UCB uncertainty and discovery yield. This follow-up
is defined but deliberately not submitted while the present 20 jobs run.

## Conditions

Each condition has one policy and seeds `8`, `23`, `57`, `99`, and `123`, for
twenty runs in total. Every run uses 100M environment frames, 32 workers x 2 environments,
40 CPUs, 80 GB, a 30-hour Slurm limit, frozen pretrained `layer2_resnet18`,
`F=16`, `R=8`, `L=64`, threshold `2.43`, `encourage`, batch loss, and
`hit_distance` worker reward.

| Condition | Manager behavior |
| --- | --- |
| `DIRECT_TARGET` | Existing least-visited direct target (`visit_direct`). |
| `PASSIVE_UCB_FRONTIER` | Passive local graph, UCB frontier selection, deliberate RETURN/VALIDATE, and direct worker conditioning. |
| `PASSIVE_UCB_FRONTIER_WAYPOINT` | The same UCB frontier manager, but worker conditioning follows a validated shortest-path waypoint (`frontier_waypoint`). |
| `PASSIVE_UCB05_FRONTIER_WAYPOINT` | Waypoint planning with a lower UCB uncertainty weight, `0.5`. |

The direct condition has no topological manager state. The frontier condition
has only the state needed to collect passive transitions and select frontiers;
neither condition provides path or geometric features to the worker.

## Run Record

- W&B project: `SF_IntrMotiv_FrontierManagerIsolation`
- W&B group: `intrmotiv_frontier_manager_isolation_20260831`
- Initial Slurm jobs: `7956653`-`7956658`
- Submission manifest: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_frontier_manager_isolation_20260831/20260831T110207Z`
- Extension Slurm jobs: `7956672`-`7956685`
- Extension manifest: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_frontier_manager_isolation_20260831_extension/20260831T111612Z`
- Source module: `sf_working_directories/IntrMotiv/dmlab/experiments/frontier_manager_isolation.py`
- Action-integration preflight: Slurm `7958757`, `2M` frames, action graph only
  (`frontier_waypoint`, no motion-policy input, no scatter loss). Its workspace
  manifest is under
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_topological_frontier_motion_preflight_20260831/20260831T163302Z`.

## Analysis Decision

Compare five-seed distributions and learning curves for coverage AUC, unique
cells, occupancy entropy, target hit lift, intrinsic reward density, DG
density/usage entropy, and policy throughput. For the frontier conditions also
check passive update rate, candidate-edge fraction, frontier selection rate,
frontier attempts/discoveries, validation success, and route availability.
Use the waypoint-versus-frontier-direct pair for the routing conclusion. A
frontier score that is nonzero without a coverage improvement is not sufficient
evidence that the manager improves exploration.

## Deferred Work

Before any action-enabled batch: inspect the submitted DMLab motion telemetry
preflight after the repeated-command transform change. Then resume the
original plan in this order: action path integration, same-DG scatter
regularization, then the SE(2) control. For a UCB-only claim, use the defined
topology-matched visit-rank control rather than `DIRECT_TARGET`.
