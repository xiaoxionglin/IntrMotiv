# ResNet+LSTM baseline feasibility — 2026-10-02

## Answer

The existing `Default` model switches can construct a trainable IMPALA-style
ResNet followed by a Sample Factory LSTM. They **cannot yet select a clean,
comparable baseline** for current IntrMotiv studies. Those studies use a fixed
ImageNet ResNet-18 through layer 2, a trainable DG projection, a CA3-style
state, and often a DG-conditioned controller. A fair no-DG ResNet+LSTM control
needs a small implementation change and a qualified run; it should not be
reported as available merely because `encoder_name=Default` and
`core_name=Default` exist.

This is a source and CPU construction audit of the local `SF_hipposlam` commit
`06790d4c59cc60118a8adb44cebce08671094f78`. It is **not** a training or
learning-performance result. Current NEMO2 deployment and job state could not
be verified: one read-only SSH check received an authentication denial, so no
further automated login was attempted.

## What the switches actually select

| Path | Visual input | Temporal core | Learner | Comparison issue |
|---|---|---|---|---|
| Current IntrMotiv studies | Frozen ImageNet `layer2_resnet18` plus trainable DG and current cue/depth processing | `BypassSS`/CA3; study-specific manager | IntrMotiv PPO or controller variant | Reference condition. |
| `encoder_name=Default`, `encoder_conv_architecture=resnet_impala`, `core_name=Default` | Trainable IMPALA ResNet over the observed image tensor; token-instruction encoder | Sample Factory LSTM if `rnn_type=lstm` | `DefaultLearner` if distance recording/learning are off | Different visual trunk, trainability, cue/depth treatment, and default PPO gradient boundary. |
| IntrMotiv encoder plus `core_name=Default` | Frozen `layer2_resnet18` but still projects to DG | LSTM | Configurable | DG remains in the input, so this is a memory ablation, not a no-DG visual baseline. |
| `Default` encoder plus `layer2_resnet18` | Construction fails in generic `make_img_encoder` | Not reached | Not reached | No supported flag-only route to the matched trunk. |

The generic `pretrained_resnet` name also maps to the IMPALA ResNet
configuration; it does not load the fixed ImageNet ResNet-18 used by IntrMotiv.
The current `HipposlamEncoder` handles number instructions as a three-way
one-hot cue, can route capped inverse depth through a ten-value bypass, and
freezes the ResNet trunk. `DmlabEncoder` instead uses a word-embedding/instruction
LSTM and feeds the observed image tensor into its generic CNN. With depth
enabled, these paths are not equivalent.

Two additional interactions block a valid LSTM control:

1. `maybe_overwrite_rnn_size` tests a literal `cli_args.rnn_size` attribute
   and replaces the requested width with the DG/CA3-derived size. A CPU parse
   requesting an LSTM width of 256 returned 3,533 in both `cfg.rnn_size` and
   `cfg.cli_args["rnn_size"]`.
2. The shared actor-critic's default `ppo_dg_gradient=stop` detaches its
   core's `core_output_size` prefix. That prefix is the **entire** output for
   `ModelCoreRNN`, so PPO cannot train the LSTM through that path. Explicit
   `ppo_dg_gradient=joint` avoids the detachment where its compatibility guard
   permits it, but an ordinary LSTM baseline should not depend on a DG-named
   flag to receive policy gradients. A local tensor-backward probe with a
   256-value core output confirmed zero input gradient in `stop` mode and a
   nonzero gradient in `joint` mode.

These are code-path findings. They do not establish that a particular cluster
run used this combination or that a learning curve failed for this reason.

## Minimal enablement and comparison contract

Reuse `ResNet18Layer2` and the existing cue and inverse-depth transforms in a
small no-DG encoder. Define explicitly whether the LSTM sees full frozen visual
features or a shared, non-DG bottleneck, and use that same input definition in
the matching IntrMotiv control when making a causal claim. Keep the generic
IMPALA ResNet path available as a separate whole-system benchmark. Use
`ModelCoreRNN` with an explicit hidden width and `DefaultLearner` for PPO;
apply the CA3 stop-gradient rule only to cores that expose a CA3 prefix. Disable
DG-specific online telemetry for the no-DG baseline. Reuse the standardized
study workflow rather than creating a second launcher or metric collector.

For a direct architecture comparison, select a task with the **same external
reward** for both systems, then match the environment/level seeds, actions,
frameskip, RGB/depth/cue observations, visual trunk and freeze policy, PPO
settings, frame budget, checkpoint ages, and evaluation rollouts. The existing
fixed-reward study family is a possible template, not a ready baseline matrix.
Report trainable parameters and throughput because CA3 state and an LSTM have
different compute and parameter costs. On no-extrinsic-reward exploration
tasks, a no-DG PPO agent has no equivalent DG-derived intrinsic reward; compare
coverage as a whole-system outcome and label the reward-objective difference
explicitly.

Before a production comparison, verify one CPU model construction for each
path, output and recurrent-state shapes, fixed-trunk hashes, nonzero PPO
gradients through the LSTM and decoder, exact observation transforms,
checkpoint/restart compatibility, and a short workspace-only training and
evaluation smoke. Keep legacy checkpoint/config behavior unchanged. Put the
paired matrix in `hpc_runs/studies/` and run the canonical print-only review.

## Current infrastructure priorities

1. **Baseline validity:** the configuration and gradient problems above
   prevent a clean architecture control; tracked in [infra.md](../../infra.md).
2. **Submission path safety:** workflow 1.14 fixes the fresh DMLab cache
   default, but one pre-submission audit of every writable bulk path, including
   temporary files, remains open.
3. **Repeatable analysis:** online scalar collection has per-run progress and
   exports, but interrupted scans still lack automatic provenance-checked
   cache reuse. Spatial summary exports still need a complete protocol and
   DG-capacity contract.
4. **Source and report freshness:** the vault records workflow 1.14 verified
   in an isolated NEMO2 source tree while its September 26 `LATEST.md` still
   described the active checkout as 1.10.1. That is a dated observation, not
   current cluster status. The result-owner/index freshness check also remains
   open, and the three previously live source-release folders require a fresh
   job audit before retirement.

## Source map and reuse lesson

Source paths below refer to
[`SF_hipposlam` at the audited commit](https://github.com/xiaoxionglin/SF_hipposlam/tree/06790d4c59cc60118a8adb44cebce08671094f78).
The switch factories and transforms are in
`sf_working_directories/IntrMotiv/dmlab/{custom_encoder.py,custom_core.py,dmlab_model.py,custom_learner.py}`;
the size override and gradient boundary are in
`{train_hipposlam.py,custom_actor_critic.py}` in the same directory. The generic
CNN factory is `sample_factory/model/encoder.py`. The workflow state is in
[`LATEST.md`](../../hpc_runs/intrmotiv_study/LATEST.md), and the authoritative
launch/analysis procedure is the
[`standardized_study_workflow.md`](../standardized_study_workflow.md).

The fastest useful check was to trace the encoder, core, actor-critic, and
learner factories together, then construct the requested CPU configuration.
Checking only the model names would have missed the recurrent-size override,
the detached gradient, and the cue/depth mismatch. The live-cluster check was
blocked by SSH authentication; reuse the checked-in source and dated workflow
records until a manual NEMO2 login is available.
