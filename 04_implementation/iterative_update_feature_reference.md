# Iterative Encoder-Decoder Update Feature Reference

Date: 2026-06-10

This document is the implementation-facing reference for iterative updates in IntrMotiv. It supersedes the older one-optimizer, DG-projection-only plan in `../05_plans/iterative_update_implementation_plan.md`.

## Purpose

Iterative update separates the update timescales of the encoder and decoder while keeping the transition-distance reward as the central intrinsic motivation signal.

The intended mechanism is:

$$
\text{stable DG landmarks}
\rightarrow
\text{decoder learns shorter landmark transitions}
\rightarrow
\text{better behavioral coverage}
\rightarrow
\text{encoder improves landmark allocation}
$$

The feature must not change transition-distance reward timing. The decoder still receives delayed reward aligned to the action that caused the next DG-CA3 transition, and the encoder still uses detached transition labels.

## Active Design

Iterative update is optional and off by default.

Default behavior:

- `iterative_update=False` preserves current simultaneous training.
- `iterative_first_phase=decoder` by default, but can be set to `encoder`.
- `iterative_phase_mode=fixed` by default.
- Plateau switching is supported as an experimental mode.

Parameter ownership:

- Encoder phase owns the whole `actor_critic.encoder`.
- Decoder phase owns all trainable actor-critic parameters outside `actor_critic.encoder`.
- Do not restrict encoder phase to `encoder.DG_projection`; the active design treats the whole encoder as the landmark-forming component.

Optimizer ownership:

- Use two optimizers in iterative mode.
- Decoder optimizer trains all non-encoder parameters.
- Encoder optimizer trains all encoder parameters.
- Decoder LR comes from Sample Factory `learning_rate` and current scheduler LR.
- Encoder LR is resolved as `iterative_encoder_lr`, else `DG_lr`, else `learning_rate`.

Keep `DistanceLearnerReward` as the only active implementation target. Do not implement this feature in `DoubleDistanceLearnerReward` unless double-value training becomes a separate explicit target.

## Config Contract

Add these config args:

| Argument                          | Default   | Meaning                                                                    |
| --------------------------------- | --------- | -------------------------------------------------------------------------- |
| `--iterative_update`              | `False`   | Enable phase-based encoder/decoder updates.                                |
| `--iterative_first_phase`         | `decoder` | First active phase: `decoder` or `encoder`.                                |
| `--iterative_phase_mode`          | `fixed`   | Switching mode: `fixed` or `plateau`.                                      |
| `--iterative_decoder_phase_steps` | `1000`    | Decoder phase length in learner optimizer steps for fixed mode.            |
| `--iterative_encoder_phase_steps` | `1000`    | Encoder phase length in learner optimizer steps for fixed mode.            |
| `--iterative_plateau_window`      | `100`     | Rolling window for plateau mode.                                           |
| `--iterative_plateau_min_delta`   | `1e-3`    | Minimum recent improvement before plateau switch.                          |
| `--iterative_min_phase_steps`     | `100`     | Minimum phase length before plateau switching is allowed.                  |
| `--iterative_max_phase_steps`     | `5000`    | Forced phase switch limit in plateau mode.                                 |
| `--iterative_decoder_lr`          | `None`    | Optional decoder LR override. Falls back to scheduler/current LR.          |
| `--iterative_encoder_lr`          | `None`    | Optional encoder LR override. Falls back to `DG_lr`, then `learning_rate`. |

Fixed phase lengths are learner optimizer steps, not environment frames. With `batch_size=B` and `num_epochs=E`, a rough fresh-sample scale is:

$$
\text{fresh samples}
\approx
\frac{\text{phase\_steps} \cdot B}{E}
$$

## Training Behavior

In simultaneous mode:

- Preserve the current `DistanceLearnerReward` behavior.
- Compute decoder loss and encoder loss.
- Backpropagate both according to current logic.
- Use existing optimizer/checkpoint behavior.

In decoder phase:

- Use the regular full forward path for policy/value/core outputs.
- Compute decoder loss from policy, value, exploration, KL, and enabled decoder auxiliary terms.
- Do not backpropagate encoder loss.
- Step only the decoder optimizer.
- Ensure encoder parameters do not receive updates.
- Skip the second encoder-only `head_only` forward if no encoder metrics require it.
- Do not apply DG projection row normalization after a decoder-only step.

In encoder phase:

- Compute the DG-CA3 transition labels exactly as in the current reward path.
- Use a fresh encoder/head forward for encoder loss so gradients flow through current encoder outputs.
- Do not backpropagate decoder loss.
- Step only the encoder optimizer.
- Ensure non-encoder parameters do not receive updates.
- Apply existing DG projection row normalization after the encoder step if the active DG projection has a `linear` module.

For both phases:

- Keep logging decoder loss, encoder loss, intrinsic reward, phase id, phase step, and active optimizer.
- Increment `train_step`, synchronize model weights, and update `policy_versions_tensor` after every active optimizer step.
- Keep delayed reward and detached encoder label semantics unchanged.

## Phase Controller

The phase controller is learner-owned state.

Required state:

- `phase`: `decoder` or `encoder`.
- `phase_step`: number of optimizer steps in the current phase.
- recent decoder scores for plateau mode.
- recent encoder scores for plateau mode.

Fixed mode:

- Switch `decoder -> encoder` after `iterative_decoder_phase_steps`.
- Switch `encoder -> decoder` after `iterative_encoder_phase_steps`.

Plateau mode:

- Decoder score: mean intrinsic decoder reward, maximize.
- Encoder score: `-encoder_loss`, maximize.
- Do not switch before `iterative_min_phase_steps`.
- Switch when rolling-window improvement is below `iterative_plateau_min_delta`.
- Always switch at `iterative_max_phase_steps`.

Plateau mode is experimental. Fixed mode remains the reproducible default.

## Checkpointing

Iterative mode changes optimizer state and phase state, so checkpoint handling must be explicit.

Save in iterative mode:

- model state;
- decoder optimizer state;
- encoder optimizer state;
- current phase controller state;
- current decoder LR and encoder LR.

Load in iterative mode:

- If dual optimizer states are present, restore both and restore phase state.
- If only the old single `optimizer` state is present, load model weights, initialize both optimizers fresh, set phase from `iterative_first_phase`, and log a warning.
- Non-iterative checkpoints must remain loadable.

Save/load in non-iterative mode should preserve current Sample Factory behavior.

## Diagnostics

Minimum summaries:

- `iterative_phase`: numeric or string-compatible phase id.
- `iterative_phase_step`.
- `iterative_decoder_active`.
- `iterative_encoder_active`.
- `iterative_decoder_lr`.
- `iterative_encoder_lr`.
- decoder intrinsic reward mean.
- encoder loss.

Recommended research diagnostics:

- transition time between repeated landmark pairs;
- landmark stability before/after encoder phases;
- route efficiency during decoder phases;
- DG unit usage;
- simultaneous new-activation collision rate.

These diagnostics should evaluate whether iterative training makes transition time more geodesic-like. Scalar intrinsic reward alone is not sufficient.

## Invariants

Do not violate these invariants:

- Transition-distance reward remains the primary intrinsic signal.
- No external coordinate reward is added.
- Reward timing remains delayed: `r_{t+1}` trains action/DG state at `t`.
- Encoder reward labels remain detached.
- `iterative_update=False` must preserve current training behavior.
- `DistanceLearnerReward` remains the active implementation target.
- `DoubleDistanceLearnerReward` remains legacy unless explicitly revived.

## Test Requirements

Phase controller tests:

- fixed mode switches at the configured decoder and encoder step counts;
- first phase follows `iterative_first_phase`;
- plateau mode does not switch before `iterative_min_phase_steps`;
- plateau mode switches when improvement is below threshold;
- plateau mode switches at `iterative_max_phase_steps`.

Parameter split tests:

- encoder and decoder parameter sets have no overlap;
- all trainable `actor_critic.encoder` params are in the encoder optimizer;
- all trainable non-encoder params are in the decoder optimizer.

Training-step tests:

- decoder phase changes non-encoder params and leaves encoder params unchanged;
- encoder phase changes encoder params and leaves non-encoder params unchanged;
- simultaneous mode remains unchanged when iterative update is disabled;
- decoder phase can skip the second `head_only` encoder forward;
- encoder phase does not backpropagate decoder loss.

Checkpoint tests:

- iterative checkpoint roundtrip restores model, both optimizers, LR values, phase, and phase step;
- old single-optimizer checkpoints load in iterative mode with fresh dual optimizers and a warning;
- non-iterative checkpoint behavior remains unchanged.

Regression tests:

- existing DG-CA3 timing tests still pass;
- compile/check modified IntrMotiv learner and parameter files.

## Non-Goals

- Do not redesign the transition-distance reward.
- Do not add coordinate-based coverage reward.
- Do not implement metric/geodesic diagnostics as training reward.
- Do not fold auxiliary losses into a new intrinsic objective in this feature.
- Do not refactor legacy learner classes beyond what is required to keep imports/tests passing.
# Implementation status

The first compatible implementation is now in the local SFgit checkout under
`sf_working_directories/IntrMotiv`. It is opt-in (`--iterative_update=True`) and
uses one Sample Factory optimizer plus a pure `train_step` schedule. This keeps
the baseline checkpoint/PBT schema unchanged. Decoder phases freeze the encoder;
encoder phases retain only the existing DG projection ownership. The more
ambitious two-optimizer/whole-encoder variant described below remains a future
extension and is intentionally not required for the first validation.
