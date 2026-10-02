# Matched layer-2 ResNet+LSTM baseline — 2026-10-02

## Status and purpose

The opt-in `--layer2_lstm_baseline={sparse,dense}` switch is implemented in
[`SF_hipposlam` commit `e8506c9b`](https://github.com/xiaoxionglin/SF_hipposlam/commit/e8506c9b69457f5385937eb0ea6bdf3994f75906).
It replaces the CA3 state with a standard 256-unit LSTM by default and tests
the effect of DG sparsification while preserving the same frozen visual trunk,
projection width, depth and cue transforms, and direct controller bypass.
`off` remains the default, preserving existing run and checkpoint behavior.

The switch has passed local model construction, forward/backward, replay, and
focused regression checks. **No DMLab environment rollout, optimizer training
run, or NEMO2 release qualification has been completed.** It is available in
the published source, but its scientific performance is unknown.

## What each setting means

| Setting | Projection after frozen ImageNet ResNet-18 layer 2 | Temporal state | Other inputs |
|---|---|---|---|
| `off` | Existing IntrMotiv configuration | Existing configured core, usually CA3/`BypassSS` | Existing behavior. |
| `sparse` | Same DG-width linear layer and BatchNorm, then thresholded ReLU at `DG_BN_intercept` | Sample Factory LSTM on projected features | Existing inverse-depth and map/task cues bypass the recurrence and reach the decoder on the current step. |
| `dense` | Same DG-width linear layer and BatchNorm, with signed normalized output and no threshold/ReLU | Same LSTM and hidden width | Identical bypass and observation transforms. |

The `dense` arm retains the matched linear bottleneck and BatchNorm to isolate
the sparsification step. It is **not** a raw 3,840-value ResNet-feature LSTM.
That direct-feature model would change input width, recurrent parameter count,
and compute at once, so it needs a separately labeled benchmark if desired.
The existing `encoder_name=Default` path remains a different, trainable IMPALA
ResNet with different cue/depth handling; do not substitute it for this pair.

The switch selects `layer2_resnet18`, `DG_name=batchnorm_relu`, an LSTM wrapper
that reuses `ModelCoreRNN`, and ordinary PPO. A requested positive `rnn_size`
is retained; legacy `rnn_size=0` resolves to 256 for the baseline rather than
the CA3 serialized-state width. PPO gradients reach both the LSTM and
projection. The wrapper recurs only over the projection, then appends the
depth/map/task bypass unchanged, including during packed-sequence replay.
Current CA3/HRL, intrinsic-reward, DG-distance learner, and DG-only online
spatial telemetry options are rejected or disabled as documented in the
[`dmlab` README](https://github.com/xiaoxionglin/SF_hipposlam/blob/e8506c9b69457f5385937eb0ea6bdf3994f75906/sf_working_directories/IntrMotiv/dmlab/README.md).

## How to compare it

For a sparse-versus-dense LSTM pair, vary only
`--layer2_lstm_baseline=sparse` versus `--layer2_lstm_baseline=dense` on the
same **externally rewarded** task. Hold `Hippo_n_feature`, LSTM width, frozen
trunk, RGB/depth/cue settings, reward, action set, frameskip, PPO settings,
seeds, frame budget, checkpoint ages, and evaluation protocol fixed. Set
`--depth_sensor=True --normalize_input=False` where the matched IntrMotiv
condition uses capped inverse depth. The default sparse threshold is 2; record
the explicit `DG_BN_intercept` in the study. Do not enable DG-specific online
spatial telemetry for this comparison.

For a CA3-versus-LSTM comparison, add a CA3 arm with the same external reward
and observation settings and state explicitly which DG training objective and
PPO gradient boundary it uses. The current IntrMotiv no-reward studies use
DG-derived internal reward and often extra representation objectives. An LSTM
baseline trained on external PPO alone cannot isolate memory architecture
against those studies. Report trainable parameter counts, throughput, and
evaluation outcomes as well as reward, since CA3 is fixed state while the LSTM
has learned weights. Define the conditions in a new declarative StudySpec;
do not inherit a `core_name=BypassSS` factor label and present it as the
resolved baseline architecture. Follow the
[`standardized study workflow`](../standardized_study_workflow.md) and its
print-only review before submission.

## Verification and remaining gates

The source checkout ran the new baseline tests plus core-logic, update-contract,
and depth-encoder regressions with the `SF_git` Python environment: **50
passed**. The tests construct both full actors, verify the requested pretrained
ResNet trunk is frozen, check equal projected/bypass widths and sparse versus
dense output, pass PPO gradients into the LSTM and projection, preserve direct
bypass values through packed replay, and confirm legacy defaults. A separate
local probe found sparse projected activity around 2.2% and dense signed output
without exact zeros for its sample batch; these are smoke measurements, not
dataset statistics.

Before scientific use, run one short **workspace-only** training and restart
smoke for each mode, verify reward is nonzero and checkpoints reload, then
qualify the paired StudySpec and evaluation on NEMO2. The source release needs
synchronization to the NEMO2 runtime checkout and focused tests there. Live
cluster status was not checked in this task because the previous read-only SSH
attempt received an authentication denial; the project access rule requires a
manual login before any automated retry.

## Reuse lesson

Trace encoder, core, actor-critic, and learner paths together when defining an
architecture switch. The old `Default` flags concealed a DG/CA3-derived LSTM
size, a stopped PPO gradient, and a missing direct depth/cue bypass. The new
switch isolates the change in one explicit mode and reuses the existing
projection and LSTM implementations; future study definitions should consume
that mode rather than reproduce its settings in ad hoc launch scripts.
