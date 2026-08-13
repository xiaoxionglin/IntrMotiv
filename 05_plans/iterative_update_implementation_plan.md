# Iterative Encoder-Decoder Update Implementation Plan

Date: 2026-06-08

> Superseded note, 2026-06-10: use [[../04_implementation/iterative_update_feature_reference|Iterative Update Feature Reference]] as the active implementation contract. This older plan kept one optimizer and updated only `encoder.DG_projection`; the current active design uses two optimizers and treats the whole `actor_critic.encoder` as the encoder-side landmark component.

## Summary

Implement iterative updates in the active `DistanceLearnerReward` path, keeping simultaneous training as the default. Phase switching is by learner optimizer steps, with phase sizes chosen to approximate stable landmark-map updates without requiring metric-based semi-convergence in v1.

Default schedule, assuming `batch_size=1024`, `num_epochs=1`:

- initial encoder warmup: `128` train steps, about `131k` transitions
- decoder phase: `512` train steps, about `524k` transitions
- encoder phase: `128` train steps, about `131k` transitions
- cycle ratio: `4 decoder : 1 encoder`

Smoke schedule for tests:

- initial encoder warmup: `4` train steps
- decoder phase: `8-16` train steps
- encoder phase: `4-8` train steps

Ablation sweep:

| Schedule | Decoder steps | Encoder steps |
| --- | ---: | ---: |
| Short | 512 | 128 |
| Medium | 1024 | 256 |
| Long | 2048 | 512 |

If `num_epochs > 1`, optimizer steps include repeated passes over the same rollout data:

$$
\text{fresh samples}
\approx
\frac{\text{phase\_train\_steps} \cdot \text{batch\_size}}{\text{num\_epochs}}
$$

## Key Implementation Changes

- Add config args in `custom_params.py`: `--iterative_update`, `--iterative_initial_encoder_steps`, `--iterative_decoder_steps`, `--iterative_encoder_steps`, and `--iterative_start_phase`.
- Add a phase helper in `DistanceLearnerReward`: warmup encoder phase first, then alternate decoder and encoder phases from `self.train_step` so checkpoint resume needs no new state.
- Keep one optimizer and mask inactive gradients before `optimizer.step()`.
- Decoder phase: update decoder/policy/core/value path, clear encoder grads, and skip the second encoder-only forward.
- Encoder phase: update only `actor_critic.encoder.DG_projection`, clear all non-DG grads, and keep the encoder-loss path.
- Simultaneous mode: preserve current behavior when `--iterative_update=False`.
- Apply DG projection row normalization only in simultaneous or encoder phases.
- Add phase summary scalars: `iterative_phase`, `iterative_decoder_active`, and `iterative_encoder_active`.
- Fix current auxiliary-loss initialization so summary variables exist even when `extra_encoder_losses=False`.

## Learner Design Decision

Do not add another learner class and do not implement iterative updates in `DoubleDistanceLearnerReward`.

`DistanceLearnerReward` is the active transition-distance path and already uses `DGCA3Interface`. `DoubleDistanceLearnerReward` duplicates older reward and training-loop logic and should remain a legacy ablation until double-value training becomes an explicit target again.

The current two-forward design inside `DistanceLearnerReward` is correct but should become phase-aware:

- simultaneous mode keeps the full forward plus second `head_only` forward;
- decoder phase skips the `head_only` encoder forward;
- encoder phase avoids decoder backward and updates only DG projection.

## Test Plan

- Phase scheduler tests: warmup, cycle boundaries, disabled mode, decoder-first/encoder-first behavior.
- Gradient masking tests: decoder phase does not update encoder/DG params; encoder phase updates only `encoder.DG_projection`; simultaneous mode remains unchanged.
- Computation-path tests: decoder phase does not call the second encoder-only forward; encoder phase does not backprop decoder loss.
- Regression: existing `tests/intrmotiv/test_dg_ca3_interface.py` still passes.
- Smoke run: use tiny phase sizes and verify summaries show phase switching.

## Acceptance Criteria

- With `--iterative_update=False`, behavior is unchanged except the auxiliary-loss summary bug fix.
- With iterative mode enabled, phases derive deterministically from `train_step`.
- Decoder phase keeps the landmark map stable by preventing encoder parameter updates.
- Encoder phase updates only `encoder.DG_projection`.
- Checkpoint compatibility is preserved because the existing optimizer and checkpoint schema remain unchanged.
