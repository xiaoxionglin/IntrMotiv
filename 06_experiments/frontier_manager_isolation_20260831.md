# Frontier Manager Isolation: 2026-08-31

## Scope

This is the first, deliberately narrow experiment extracted from the deferred
topological-frontier HRL plan. It measures only whether passive transition
evidence plus UCB frontier target selection improves exploration over the
current direct-target controllable-graph HRL policy.

It does not test waypoint planning, RETURN/VALIDATE behavior, action path
integration, motion-conditioned worker input, the DG scatter loss, or the
SE(2) pose-graph control. Those features remain deferred until this comparison
is analyzed.

## Conditions

Each condition has one policy and seeds `8`, `23`, `57`, `99`, and `123`, for
twenty runs in total. Every run uses 100M environment frames, 32 workers x 2 environments,
40 CPUs, 80 GB, a 30-hour Slurm limit, frozen pretrained `layer2_resnet18`,
`F=16`, `R=8`, `L=64`, threshold `2.43`, `encourage`, batch loss, and
`hit_distance` worker reward.

| Condition | Target selection |
| --- | --- |
| `DIRECT_TARGET` | Existing least-visited direct target (`visit_direct`). |
| `PASSIVE_UCB_FRONTIER` | Passive local transition graph plus UCB frontier target selection (`frontier_direct`), with direct worker conditioning. |
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

## Analysis Decision

Compare five-seed distributions and learning curves for coverage AUC, unique
cells, occupancy entropy, target hit lift, intrinsic reward density, DG
density/usage entropy, and policy throughput. For the frontier condition also
check passive update rate, candidate-edge fraction, frontier selection rate,
frontier attempts/discoveries, and route availability. A frontier score that
is nonzero without a coverage improvement is not sufficient evidence that the
manager improves exploration.

## Deferred Work

Only after this result is available, resume the original implementation plan
in this order: next-hop planning and deliberate validation, action path
integration, same-DG scatter regularization, then the SE(2) control.
