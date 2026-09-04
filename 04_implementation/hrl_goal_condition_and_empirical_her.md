# HRL Goal Timing And Empirical PPO-HER

## Goal timing

The policy-buffer graph is learner-owned. The actor stores a sampled worker
goal so PPO can replay the exact condition even after the graph changes.

```text
delayed
state_t[g_t] + obs_t -> manager writes g_(t+1)
                       -> pi(a_t | obs_t, g_t), V(obs_t, g_t)

immediate
state_t[g_t] + obs_t -> manager writes g_(t+1)
                       -> pi(a_t | obs_t, g_(t+1)), V(obs_t, g_(t+1))
```

`delayed` is the compatibility default. `immediate` is restricted to the
direct policy-buffer manager. It appends a replay-only target ID to the RNN
state; the ID is terminal-persistent but never read by the live core. This
retains the target used by a final action even when Sample Factory resets the
physical recurrent state.

During replay, the learner replaces the target slice in the core output before
the shared decoder. The actor logits and critic value therefore use the same
stored target. Graph changes only affect future sampling.

Sample Factory keeps compact recurrent-state storage before it constructs the
action-aligned packed replay sequence. For that reason, the empirical labels
are built from the replayed DG activations after packing, never by treating raw
`rnn_states` storage as a `[stream, decision]` tensor. This preserves the
normal recurrent replay path and prevents labels from being misaligned with
actions.

## Empirical PPO-HER

This is a deliberately biased auxiliary, not Hindsight Policy Gradients.
For one exclusive future DG event per rollout stream, choose a same-episode
source segment of at most `H` decisions and keep the achieved DG target fixed
throughout it. The relabeled pseudo-return is terminal at the achieved event:

```text
r^H_t = 0                         for t < e
r^H_e = r_hit + c * max(0, E-d_e)
G^H_t = gamma^(e-t) r^H_e
```

The auxiliary re-evaluates policy and critic at this fixed target and adds a
clipped PPO-style loss using the logged behavior-goal log probability as its
denominator. Ordinary PPO, graph updates, and DG losses are unchanged. The
goal mismatch is logged rather than claimed to be importance-corrected.

## Batch

`SF_IntrMotiv_HRLGoalConditionHER` compares only direct HRL:

| Timing | Hindsight horizon | Seeds |
| --- | --- | --- |
| delayed, immediate | off, 16, 64 | 8, 99, 123 |

There are 18 jobs. No flat control is included. See the authoritative source
documentation in `sf_working_directories/IntrMotiv/HRL_ARCHITECTURE.md` and
`LOGGING.md` on NEMO2 for metric names and implementation details.

## Target-ID FiLM goal conditioning (2026-09-04)

The controller now has an optional goal-conditioning mode:

```text
--hrl_goal_conditioning=target_id_film
```

Let `g` be the replayed 16-dimensional target one-hot and let `s` contain every
other policy feature, including the full CA3 state. With hidden width 128, the
decoder computes:

```text
z = ReLU(W_s s + b_s)
[delta_gamma, beta] = g @ M,       M has shape [16, 256]
z_goal = (1 + delta_gamma) * z + beta
h = ReLU(W_2 z_goal + b_2)
```

`delta_gamma` and `beta` are 128-dimensional vectors. `M` is initialized to
zero, so FiLM begins as the identity transformation. The all-zero no-target
condition also produces exact identity modulation.

This mode deliberately does **not** concatenate the target one-hot into the
ordinary state stream, learn a separate target embedding, or condition on the
selected target's CA3 trace. The target enters the decoder only through the
direct one-hot matrix multiplication above. The full CA3 trace remains part of
the current state, not the goal specification. Target-relative geometry, when
enabled, remains an explicit additional condition.

The behavior target is still stored in rollout state and teacher-forced during
PPO replay, so learner-side graph changes cannot retrospectively alter the goal
used for an action. The compatibility modes `legacy` and `target_trace` remain
available. `target_id_film` requires immediate target timing, the controllable
policy-buffer graph, shared actor/critic weights, and the single-value path.

For a new batch, use at minimum:

```text
--hrl_goal_conditioning=target_id_film
--hrl_target_timing=immediate
```

NEMO2 validation: the focused controller/replay suite passed 16 tests and the
complete IntrMotiv suite passed 171 tests. No training jobs were launched or
modified as part of this implementation.
