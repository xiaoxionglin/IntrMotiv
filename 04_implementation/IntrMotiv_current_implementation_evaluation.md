# IntrMotiv Current Implementation Evaluation

Date: 2026-06-05

This note evaluates the current IntrMotiv implementation after the DG-CA3 interface refactor. The relevant code is mainly:

- `/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/IntrMotiv/dmlab/dg_ca3_interface.py`
- `/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/IntrMotiv/dmlab/custom_learner.py`
- `/home/xiaoxiong/SFgit/SF_hipposlam/tests/intrmotiv/test_dg_ca3_interface.py`

## Current Implementation

The current active design still follows the original Sample Factory learner structure. The DG-CA3 state is embedded in the actor-critic recurrent state, and the learner reads that state after rollout collection to construct intrinsic reward labels.

The new `DGCA3Interface` wraps the DG-CA3-specific pieces:

1. It extracts the sequence core from the recurrent state.
2. It converts sequence-core activity into a progression vector.
3. It computes distance matrices for distance-learning variants and logging.
4. It computes transition-aligned intrinsic labels from the recorded rollout states plus a bootstrap next recurrent state.

The active reward path is in `DistanceLearnerReward._prepare_batch()`. It does one additional forward call on the normalized final observation:

```python
additional_step = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)
```

That produces `additional_step["new_rnn_states"]`, which gives the DG-CA3 state for the step immediately after the stored rollout. `DistanceLearnerReward._calculate_internal_reward()` then calls:

```python
labels = self.dg_ca3.compute_transition_labels(
    buff["rnn_states"],
    additional_step["new_rnn_states"],
)
buff["rewards"] = labels.decoder_reward_for_action_t
buff["rewards_encoder"] = labels.encoder_reward_for_dg_t
```

This means the confusing temporal shift is no longer written directly in the learner as anonymous slices. The interface names the two labels:

- `decoder_reward_for_action_t`: the reward used by PPO/critic/policy for action-time credit assignment.
- `encoder_reward_for_dg_t`: the detached label used to train the encoder/DG projection for the DG state at the current time.

The tests in `tests/intrmotiv/test_dg_ca3_interface.py` verify that this interface preserves the previous timing exactly for all existing `encoder_reward_method` variants.

## Why It Works This Way

The implementation works because the DG-CA3 reward is not available from the pre-transition state alone. The event being rewarded is a change in the sequence model: whether a new DG/sequence row was activated and how that activation relates to prior sequence progression. That event is only visible after running the model into the next recurrent state.

So the learner must use states from a window that is one step longer than the action/reward training window:

- stored rollout recurrent states: DG states already produced during rollout
- one bootstrap recurrent state: the state after the last stored step
- internal reward tensor: computed over the extended state sequence
- decoder reward: shifted so the consequence observed at `t+1` trains the action at `t`
- encoder reward: shifted so the DG representation at `t` gets a label derived from the following transition

This is normal for reinforcement learning credit assignment. Environment reward is also observed after an action is taken; here the intrinsic reward is just produced by the internal DG-CA3 dynamics instead of the external environment.

The detached-label choice is also important. The intrinsic reward should act like a teaching or modulation signal, not like a differentiable objective that backpropagates through the next-step DG-CA3 transition that generated the label. In algorithmic terms, this avoids self-referential gradients where the model changes the label-generating mechanism to make its own reward easier. In biological terms, it is closer to a delayed modulatory or novelty signal: the event is observed after the transition, then assigned back to the representation/action that caused it.

## How This Theoretically Helps

The interface refactor helps mainly by making the temporal semantics explicit and local.

Previously, the learner contained the model-specific timing logic directly. The meaning of slices like `[:, 2:]` and `[:, 1:-1]` had to be inferred from the surrounding code. That made the implementation fragile: changing rollout length, bootstrap handling, or the DG-CA3 state layout could silently change the reward alignment.

The current interface improves this in several ways:

1. It creates a single place where DG-CA3 state layout is defined.

   `Hippo_R`, `Hippo_L`, `Hippo_n_feature`, `expanded_length`, and `core_output_size` now live together in `DGCA3Interface`. This reduces the chance that the learner and model disagree about the recurrent-state layout.

2. It separates model semantics from learner mechanics.

   The learner still handles Sample Factory details: validity masks, normalization, GAE, PPO losses, value bootstrapping, optimizer updates. The interface handles DG-CA3 details: progression, new activation detection, distance geometry, and reward-label alignment.

3. It gives the two reward targets names.

   `decoder_reward_for_action_t` and `encoder_reward_for_dg_t` are much easier to reason about than raw shifted tensors. This is especially useful because the two labels are intentionally not identical in timing.

4. It gives us a testable seam for future DG-CA3 changes.

   The tests can compare old behavior to new behavior without running the full Sample Factory learner. That makes it safer to later change the DG-CA3 interface, the reward rule, or the encoder objective.

Theoretically, this should make it easier to move toward a cleaner DG-CA3 model interface where the model exposes named state components and transition events directly. In that future version, the learner would not need to know that the first `Hippo_n_feature * (R + L - 1)` recurrent-state entries are the sequence core. It would simply ask the DG-CA3 component for transition labels or transition diagnostics.

## What Is Still Awkward

The current implementation is not yet a fully clean DG-CA3 interface. It is an intermediate refactor.

The main remaining issue is that the interface still receives raw recurrent-state tensors. It assumes the DG-CA3 sequence core occupies the leading block of `rnn_state`. That is better than duplicating the assumption throughout the learner, but it is still not a true model-level contract. A cleaner model would return something structured, for example:

```python
outputs.dg_ca3_state.sequence_core
outputs.dg_ca3_state.progression
outputs.dg_ca3_transition.new_activations
```

or expose a method on the model/core that computes these diagnostics directly.

The learner also still has to run the bootstrap forward pass. That is probably acceptable because Sample Factory already needs a final value bootstrap. But conceptually the DG-CA3 transition label construction is still split across two places:

- the learner obtains the next recurrent state;
- the interface interprets the state sequence and returns labels.

That split is practical, but not fully internal to the model.

Another awkward point is that the reward rule is implemented with Python dictionaries and loops over new activations. This is fine for correctness and readability now, but it may be slow if rollout batches become large. A vectorized implementation would be preferable once the rule is stable.

Finally, the interface preserves the original timing exactly, including its boundary behavior. It uses `torch.roll(..., dims=1)` to compare each time step to the previous one. For `t=0`, that compares against the final bootstrap state. This is mostly harmless because the returned training labels slice away the earliest time step, but it is semantically circular. A future cleanup should make the first-step boundary explicit instead of relying on later slicing.

## Redundant Parts

Several parts of the current code are redundant or legacy-like.

1. Learner wrapper methods now mostly forward to `DGCA3Interface`.

   `BaseDistanceRecorder._calculate_sequence_core()`, `_calculate_progression()`, and `_record_distance_matrix()` are compatibility wrappers. They help avoid editing every old learner subclass at once, but they are not conceptually necessary anymore.

2. Multiple learner subclasses repeat the same distance-matrix logic.

   `custom_learner.py` contains many older variants that call `_record_distance_matrix()` and then compute variants of intrinsic advantages or statistics. These are useful if they are still experimental ablations. If not, they add noise and make it harder to identify the active algorithm.

3. Some older intrinsic-reward logic remains outside the new interface.

   In particular, older classes still compute intrinsic rewards directly from distance matrices and progression. The active `DistanceLearnerReward` path uses `DGCA3Interface.compute_transition_labels()`, but the file still contains previous approaches.

4. The encoder auxiliary losses duplicate DG-CA3 state interpretation.

   `_extra_encoder_loss()`, `_encoder_loss()`, and `_extra_decoder_loss()` still call `_calculate_sequence_core()` and reconstruct masks such as `progression == 0` and `mask_active_now`. These masks should eventually be exposed by the interface as named diagnostics, so the learner does not repeatedly reconstruct DG-CA3 concepts.

5. `_register_forward_hooks()` contains unreachable code.

   In `DistanceLearnerReward._register_forward_hooks()`, the method immediately returns `super()._register_forward_hooks()`. Everything after that return is dead code.

6. Gradient-flipping helpers are likely obsolete or inactive in the current path.

   `make_grad_flip_hook()`, `flip_module_grads()`, and `_manipulate_gradients()` remain, but the active training path now performs separate decoder and encoder backward passes and manually clears DG projection gradients between them. The old gradient-flip design should be removed or clearly marked if it is no longer part of the intended algorithm.

7. Some summary variables can be undefined.

   In `DistanceLearnerReward._calculate_losses()`, `encoder_penalty_loss`, `encoder_reward_loss`, and `encoder_batch_loss` are added to `additional_stats` even when `self.cfg.extra_encoder_losses` is false. In the current defaults this may not trigger, but it is a real cleanup target.

8. The tests intentionally duplicate legacy reward logic.

   This is good for now because it proves behavior preservation. But once the current implementation becomes the source of truth, the tests should evolve from "match legacy slices" toward explicit behavioral examples.

## Recommended Cleanup Order

The best next steps are conservative:

1. Keep `DGCA3Interface.compute_transition_labels()` as the source of truth for active intrinsic reward timing.
2. Move repeated encoder/decoder masks into named interface outputs.
3. Remove or isolate inactive learner subclasses into a legacy/ablation file.
4. Delete unreachable hook code and inactive gradient-flip helpers if they are not part of the current algorithm.
5. Replace the `torch.roll` boundary behavior with explicit previous-state handling.
6. After the reward rule is stable, vectorize `compute_transition_labels()`.

The important point is that the current implementation works because it preserves the delayed-reward timing while making the alignment explicit. The biggest theoretical benefit is not simplification by itself; it is making DG-CA3 transition events into named, testable algorithmic objects instead of hidden learner-side tensor slicing.
