# DG-CA3 Gradient Timing Evaluation

Date: 2026-06-05

## Question

The DG-CA3 intrinsic reward is only known after the CA3 sequence state advances:

```text
CA3_state_t + DG_t -> CA3_state_{t+1} -> intrinsic_reward_{t+1}
```

The concern is how to update the encoder using the relevant current-time DG activity when the reward is only available after the next forward transition.

## Recommendation

Use a detached next-step reward as the baseline.

The encoder should receive gradients through the DG activation at the causal time step, while the intrinsic reward computed from the next CA3 state should be treated as a detached scalar teaching signal.

In other words:

```python
ca3_state_tplus1 = ca3_transition(ca3_state_t, dg_t)
reward_tplus1 = compute_intrinsic_reward(ca3_state_tplus1).detach()

dg_t_with_grad = encoder(obs_t)
encoder_loss = encoder_objective(dg_t_with_grad, reward_tplus1, activation_mask_tplus1)
```

Detached does not mean the encoder ignores the next step. It means the next-step reward is used as a delayed label or modulatory signal, not as a differentiable path through the reward calculator.

## Algorithmic Evaluation

Detached reward is the cleaner default for this system.

First, it matches actor-critic credit assignment. The action and DG activity at time `t` cause the transition, and the reward is observed after that transition. PPO already uses this delayed reward structure for the decoder. The encoder can use the same temporal assignment: update `DG_t` according to the intrinsic consequence observed at `t+1`.

Second, the intrinsic reward calculation is mostly discrete or piecewise-defined. It depends on thresholded DG activation, CA3 sequence progression, refractory behavior, activation masks, `argmax`-style progression recovery, and minimum-distance selection. These are not naturally smooth loss components. Forcing gradients through them would mostly require surrogate gradients or straight-through estimators, which would turn the reward metric into an engineered differentiable objective rather than a measured consequence of the DG-CA3 state.

Third, detaching the reward avoids a common failure mode: the encoder optimizing quirks of the reward calculator itself. If gradients flow through the reward computation, the encoder may learn to manipulate threshold boundaries, masks, or progression decoding in ways that improve the differentiable surrogate without improving the intended sequence representation.

Fourth, it preserves the useful separation between model parts. The decoder learns behavior from delayed intrinsic rewards through PPO. The DG projection learns sequence allocation from a local encoder loss weighted by delayed DG-CA3 feedback. This keeps the pull-push structure from the report easier to reason about.

## Biological Evaluation

Detached reward is also more biologically plausible.

The biological analogy is not that DG receives an exact gradient through CA3 dynamics and the intrinsic reward function. A more plausible mechanism is:

```text
DG activity occurs -> CA3 sequence evolves -> a later local/global feedback signal modulates plasticity
```

That feedback could be interpreted as a delayed modulatory or eligibility-like signal. The synaptic update is still tied to the earlier DG activity, but the feedback itself is not a differentiable computational graph.

This is closer to local plasticity with delayed modulation than to backpropagation through a hand-written reward metric. It also fits the model's use of sparse thresholded DG activation and fixed/semi-fixed CA3 sequence dynamics: those mechanisms look more like state transitions that generate a teaching signal than like differentiable layers whose internal reward should be optimized end-to-end.

Straight-through estimators can still be useful engineering tools for binary DG or CA3 gates. But they should be treated as surrogate-gradient approximations for training, not as the biological interpretation of the intrinsic reward.

## Comparison Of Options

### Detached Reward Label

This should be the default.

The reward is computed after the transition and detached. A fresh encoder forward pass provides `DG_t` with gradients, and the detached reward is aligned back to the DG activation that caused the next CA3 state.

Benefits:

- Stable and compatible with PPO-style delayed credit assignment.
- Avoids differentiating through discrete CA3 progression logic.
- Makes the reward a measured consequence rather than an exploitable differentiable target.
- More biologically plausible as delayed feedback.

Cost:

- Requires careful timestep alignment.
- The encoder does not receive gradient information about how small continuous changes to DG would alter the reward, only whether the observed activation was good or bad under the chosen encoder loss.

### Differentiable Reward Through CA3

This should be an ablation, not the baseline.

Here the CA3 transition and reward calculation remain in the graph, so the encoder receives gradients through `reward_tplus1`.

Potential benefit:

- Gives a direct optimization path if a smooth reward surrogate is carefully defined.

Risks:

- The true metric is not naturally smooth.
- Surrogate gradients may dominate the intended DG-CA3 behavior.
- The encoder may exploit threshold/mask artifacts.
- It is less biologically defensible.

### Hybrid Or Straight-Through Reward

This is a later experimental option.

Binary gates or refractory masks could use straight-through estimators while still detaching the final scalar reward. This may help train sparse activations, but it should be introduced only after the detached baseline is stable and tested.

## Interface Implication

The clean DG-CA3 interface should expose a transition object with explicit alignment:

```text
dg_t
ca3_state_before_t
ca3_state_after_t
progression_after_t
new_activation_mask_t
intrinsic_reward_tplus1_detached
encoder_reward_for_dg_t
decoder_reward_for_action_t
```

The important design rule is:

```text
reward is computed from CA3_state_after_t, but encoder gradients flow through DG_t
```

The current learner already approximates this with a two-pass pattern:

1. Compute intrinsic rewards from rollout CA3 states and the additional next-step state.
2. Run an encoder-only forward pass to obtain fresh `head_outputs` with gradients.
3. Multiply those current DG activations by `rewards_encoder`.

The refactor should keep this timing but make the indexing explicit. Instead of hidden slices like `[:, 2:]` and `[:, 1:-1]`, the DG-CA3 helper should return named aligned tensors:

```python
transition = dg_ca3.compute_transition_labels(
    rnn_states=batch_rnn_states,
    next_rnn_state=bootstrap_next_rnn_state,
)

decoder_reward_t = transition.decoder_reward_for_action_t
encoder_reward_t = transition.encoder_reward_for_dg_t.detach()
new_activation_t = transition.new_activation_mask_for_dg_t.detach()

dg_t_with_grad = encoder_forward_outputs.dg
encoder_loss = encoder_objective(dg_t_with_grad, encoder_reward_t, new_activation_t)
```

This directly answers the timing problem: the reward is delayed, but it is assigned to the current timestep as a detached label, just like reward in reinforcement learning is assigned to the transition that produced it.

## Practical Default

For the IntrMotiv branch, the default should be:

- Detached intrinsic reward.
- Encoder gradients through fresh current-step DG activations.
- Explicit transition-aligned tensors from a DG-CA3 helper.
- Differentiable reward-through-CA3 only as a later ablation.

This gives the cleanest algorithmic story and the strongest biological interpretation.
