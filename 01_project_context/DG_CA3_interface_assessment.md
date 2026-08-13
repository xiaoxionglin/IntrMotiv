# IntrMotiv DG-CA3 Interface Assessment

Date: 2026-06-05

## Code Location

Code changes should happen in:

`/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/IntrMotiv`

Results, notes, and reports should stay in:

`/home/xiaoxiong/Desktop/Projects/IntrMotiv`

The `IntrMotiv` code folder was initialized as a copy of `sf_working_directories/jannek`.

## Current Structure

The current intrinsic motivation implementation is mostly concentrated in three model-facing areas:

- `custom_encoder.py`: visual frontend, instruction input, and DG projection.
- `custom_core.py`: CA3-like sequence propagation core.
- `custom_learner.py`: intrinsic reward computation, reward rewriting, and encoder-specific losses.

The reward is currently calculated inside the learner from stored rollout `rnn_states`. This works with Sample Factory's training loop, but it makes the DG-CA3 mechanism implicit: the encoder/core produce ordinary actor-critic tensors, and the learner reconstructs DG-CA3 state semantics afterward.

## Clean DG-CA3 Interface Feasibility

Moving closer to a clean DG-CA3 model interface is feasible with moderate effort.

The natural boundary is a small module that owns:

- DG activations from the encoder.
- CA3 sequence state/progression.
- New-activation masks.
- Internal reward or distance metrics.
- Encoder-local losses such as multi-activation, unused-sequence, and batch-use losses.

This module can expose a structured output such as:

```python
DGCA3Output(
    dg_activation,
    ca3_state,
    progression,
    new_activation_mask,
    intrinsic_reward,
    diagnostics,
)
```

The actor-critic can still consume the flattened CA3/bypass tensor for policy/value prediction, while the learner can consume the structured diagnostics directly instead of recomputing progression from raw `rnn_states`.

## Effort Estimate

Low effort:

- Keep learner-level reward injection.
- Factor `_calculate_sequence_core`, `_calculate_progression`, and `_calculate_internal_reward` into a helper module.
- Add tests around progression and reward alignment.

Moderate effort:

- Introduce a DG-CA3 module used by both core and learner.
- Return structured diagnostics from forward passes.
- Remove duplicated progression/reward logic from learner variants.

Higher effort:

- Redesign Sample Factory rollout storage to carry structured DG-CA3 diagnostics across sampling and training.
- This may be cleaner conceptually, but it increases CPU/GPU transfer and Sample Factory integration risk.

Recommended first step: moderate effort, but avoid changing rollout storage. Compute and expose structured DG-CA3 diagnostics during learner forward passes first.

## Simplification Targets

The current flags `use_internal`, `use_external`, `metric`, and `masked_distance_matrix` are only partially represented in the active `DistanceLearnerReward` path. They should either be wired explicitly into reward computation or removed from active presets to reduce ambiguity.

The active baseline should be:

- `distance_learning=True`
- `DistanceLearnerReward`
- `encoder_reward_method=punish`
- `core_name=BypassSS`
- `DG_name=batchnorm_relu`
- `encoder_conv_architecture=layer2_resnet18`

## GPU Note

The copied `IntrMotiv` learner has been adjusted so progression sentinel tensors and `internal_reward` are allocated on the same device as the tensors they operate with. This was a plausible GPU failure source in the Jannek version.

The specific risk was CPU tensors created inside intrinsic reward calculation while rollout/core tensors may be CUDA tensors.

## Visual Frontend

The default visual frontend in `IntrMotiv` is now `layer2_resnet18`, taken from `sf_xxl`.

For the older `pretrained_resnet` path, the canonical checkpoint fallback is:

`/home/xiaoxiong/try0120/train_dir/Random3_resnet_DG_relu_SS_RNN/checkpoint_p2/best_000020923_170811392_reward_87.534.pth`
