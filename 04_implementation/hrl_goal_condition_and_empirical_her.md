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
