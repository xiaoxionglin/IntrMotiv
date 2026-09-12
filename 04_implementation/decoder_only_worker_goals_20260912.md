# Decoder-only worker goals — approved 12 September 2026

The user approved goal-independent worker memory with the subgoal supplied only
to the action/value decoder. This supersedes the waypoint goal-write requirement
in the earlier full-system DDQN + auxiliary HER contract.

| Component | Previous waypoint F64 | Revised waypoint F64 |
|---|---|---|
| DG goal input | `dg_goal_input=write` | `dg_goal_input=none` |
| Worker memory | Goal-modulated writes, separate worker trace | Existing goal-independent DG/CA3 trace and bypass |
| Subgoal conditioning | Memory writes and decoder | Existing `TargetFiLMDecoder` only |
| Manager | `frontier_waypoint`, edge exploration enabled | Same |
| DG capacity | 64 | 64 |
| Learning modes | PPO, DDQN, DDQN + auxiliary HER | Same; all three use the revised architecture |

Direct F16 already uses `dg_goal_input=none` and is unchanged. The intended
production matrix remains two architectures, three learning modes, and seeds
8, 99, 123 (18 fresh 300M-frame runs). The revised waypoint arm is explicitly
named `WAYPOINT_DECODER_F64`; results must not be pooled with goal-write F64 or
presented as a controller-only change relative to that historical parent.

Reuse the existing `SimpleSequenceWithBypassCore` and `TargetFiLMDecoder`.
No new memory or decoder implementation is required. The ImageNet trunk remains
fixed; DG learning, normalization, manager, graph learning, intrinsic reward
components, STOP routing and telemetry schedules remain as specified. The
intentional removal of goal-write parameters also removes their gradient/norm
telemetry; goal-decoder sensitivity and modulation telemetry remain applicable.

HER still has a separate auxiliary head and additional TD budget. It changes
the virtual subgoal, auxiliary reward, option budget and task-ending labels.
For a fixed representation snapshot, physical worker memory does not depend on
the virtual goal. A decoder relabel must match full virtual-history evaluation
in outputs and decoder gradients. Main DDQN still uses the actual successor
command across manager/waypoint switches; virtual outcomes never update the
real manager or graph.

In the original reconstruction mode, online and target DG/normalization snapshots
can differ and require their own physical-memory reconstruction. The user has
subsequently approved an explicit [stored-state trial](stored_state_replay_20260912.md):
it accepts behavior-time representation lag and shares stored worker inputs
between online and target decoders. That trial supersedes the reconstruction
requirement only for its labeled mode; it does not claim snapshot equivalence.
Its measured throughput and qualification are recorded in the linked contract.

Qualification uses the unchanged direct R5 evidence and three new fresh 2M
waypoint preflights in
`hpc_runs/studies/full_system_controller_decoder_preflight.study.json`.
Historical goal-write checkpoints stay as provenance, not initialization for the
new architecture. Require new waypoint PPO/DDQN/HER runtime and reload gates;
old goal-write qualification does not qualify the revised architecture.

Reusable lesson: explicitly separate goal-dependent policy conditioning from
goal-dependent memory writes. Decoder-only relabeling is valid only when the
representation and physical memory are invariant to the changed goal. Test that
invariance and relabeled decoder gradients with nonzero goal modulation, rather
than relying on identity-initialized weights that hide missing conditioning.
