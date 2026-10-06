# Odor grounding and CA3 goal quality: 40-run plan

**Status, 6 October 2026:** design proposal. No StudySpec, new runtime feature, submission, or NEMO2 qualification is claimed here. The standardized workflow remains the launch authority once the runtime contracts below exist.

## Question and minimal design

Can spatially grounded sensory input and a small, context-predictable DG goal vocabulary improve exploration and *command-caused* navigation? Keep the C15-derived $F=16$, $R=8$, $L=64$ backbone, ARR encoder credit, FiLM target ID, `frontier_direct`, PPO-to-DG STOP, and existing intrinsic rewards. Do not add DG replacement, HER, waypoints, geometry control, a predictor network, or a sparsity sweep.

Run the complete $2\times5\times4=40$ Cartesian design for 75M environment frames per run. Use paired seeds **8, 99, 123, 2026**. The fourth seed is proposed here and must be fixed in the StudySpec before any run. Use one shared map and reward protocol across cells.

| Goal-set condition | Odor off | Odor on | Main question |
| --- | ---: | ---: | --- |
| ALL16 | 4 | 4 | Grounding with the original goal vocabulary |
| RANDOM4 | 4 | 4 | Effect of reducing the controller's choices to four |
| HEBB4 | 4 | 4 | Value of CA3-based selection at four goals |
| RANDOM8 | 4 | 4 | Effect of reducing the choices to eight |
| HEBB8 | 4 | 4 | Value of CA3-based selection at eight goals |

This compact balanced design tests odor, vocabulary size, and CA3 selection at two capacities. An exposure-selected set or visit-only frontier selector would answer further questions, but would displace replication or another primary contrast at 40 runs. Keep them as a later batch if the present results warrant it.

## Fixed sensory and associative mechanisms

Use four broad, spatially separated Gaussian odor channels with clean peak amplitude 1. All runs instantiate the same DG input shape; OFF supplies four zeros. For ON, add independent zero-mean Gaussian noise to each channel at each observation, with fixed standard deviation **0.15 of the clean channel peak** (between the requested 0.1 and 0.2), then apply the fixed odor gain: $x_{\mathrm{odor},k}=\lambda_o[o_k(x)+\epsilon_k]$, $\epsilon_k\sim\mathcal N(0,0.15^2)$. Do not clip the noisy values, which would bias weak odor regions upward. Use a dedicated reproducible noise stream, independent of action/goal sampling, and preserve its state on resume. Deliver odor only to DG, with no direct policy, decoder, manager, or CA3 bypass. Set one fixed odor gain using a separate preflight trajectory to make the **clean** mean odor-block norm comparable to the visual-block norm; log the resulting noisy norm too. Record the centers, width, noise scale, gain, map hash, and calibration artifact; never recalibrate by run or seed. This tests one noisy sensory cue, not a gain or noise sweep.

For a transition from decision $t$ to $t+1$ that first reaches exclusive DG event $j$, pair the detached, goal-independent CA3 state $c_t$ recorded **before** the action and event observation with $j$. Normalize $c_t$ to $\hat c_t$ and update only the observed row:

$$
M_j \leftarrow (1-\alpha)M_j+\alpha\hat c_t,\qquad \alpha=0.01.
$$

Apply this same event-triggered rule from the first frame through 75M in **every cell**, including ALL16 and RANDOM. No learning-rate schedule, optimizer, prediction loss, autograd path, or feedback into DG/CA3 is permitted. Non-event transitions leave $M$ unchanged. Save and restore $M$, event counts, and quality state in checkpoints. A fresh run and an exact resume must produce the same rows and counts on a fixed event stream.

Score each event *before* updating its row. Against supported alternative rows, use the own-versus-best-other cosine margin:

$$
q_t(j)=\cos(\hat c_t,M_j)-\max_{k\ne j}\cos(\hat c_t,M_k).
$$

Maintain an event-wise quality EMA $Q_j\leftarrow0.99Q_j+0.01q_t(j)$ when the margin is defined. A row needs at least 100 observed events before it can be selected for HEBB or RANDOM. Record both event support and the prequential margin; a high score without sufficient support is not evidence of predictability. These constants are fixed before production and are not tuned to the outcome.

At **10M frames**, make one goal-set selection. HEBB$K$ chooses the top $K$ supported $Q_j$ rows, with DG index as the deterministic tie-breaker. RANDOM$K$ chooses $K$ rows uniformly without replacement from that **same support-qualified pool**, using a separate documented random stream. If fewer than eight rows qualify in any run, the qualification has failed; investigate the event contract rather than silently filling from unsupported rows. Freeze the selected *goal IDs* through 75M, while $M$, $Q$, and support counts continue updating by the unchanged rule for diagnostics. All cells use ALL16 during the initial 10M, so the subset comparisons share the same early controller protocol. Odor can already change the DG representation during this period.

Within a restricted vocabulary, compute C15 novelty rank and uncertainty against eligible goals only; preserve the existing C15 frontier formula and other controller behavior. Check that this normalization does not introduce a K-dependent score-scale artifact. Keep the original 16-dimensional DG and target-ID decoder; mask ineligible targets only at manager selection, and continue logging all DG events. The RANDOM and HEBB pairs at each $K$ must share every implementation detail except the selection criterion.

## Outcomes and contrasts

The primary training outcome is external coverage AUC over **10–75M** frames, paired by seed. Retain the 0–10M trace as a calibration check, not part of the subset effect. The primary causal contrasts are:

1. HEBB4 minus RANDOM4 and HEBB8 minus RANDOM8, separately at each odor level: does CA3 selection improve the same-size vocabulary?
2. RANDOM4 and RANDOM8 versus ALL16 at each odor level: does a smaller vocabulary help without quality information?
3. ON minus OFF within each matched goal-set condition: does sensory grounding help, and does it alter the selection effect?
4. Four versus eight within RANDOM and within HEBB: is there a capacity tradeoff?

Report paired seed values and uncertainty intervals, rather than treating ten related contrasts as ten independent discoveries. With four seeds this is a mechanistic screen, not a precise effect-size estimate. Record selected IDs, event frequencies, support, $Q_j$, margin distributions, and the spatial coverage of selected versus excluded goals; explicitly test whether HEBB merely picks frequent events. A favorable HEBB result with no advantage over exposure-matched post hoc controls should motivate a frequency-selected production arm before claiming predictive quality.

Coverage alone is insufficient. At the final checkpoint, use the established matched-start, alternative-command intervention to measure first distinct DG outcome, target-specific success above shuffled command, and action-distribution change. Use macro averages over commanded goals, and preserve failed/time-out trials in denominators. Report ordinary option success separately; historical C15-family work shows that high aggregate target-hit rates can coexist with weak command specificity. The spatial panel must include silent units, active-only map cosine, peak diversity, spatial information, and pre-threshold maps. An odor benefit that only creates duplicate place fields is not the intended result.

Use the standard 10k-decision place-field rollout at 5/10/25/50/75M for seed 99 plus terminal rollouts for seeds 8 and 123. Seed 2026 needs at least the terminal command intervention and core scalar outcomes; a terminal field rollout is desirable if capacity permits. Do not infer fixed-trajectory drift from policy-driven checkpoint rollouts.

## Implementation and launch gates

1. Implement and test the shared odor input, CA3 event pairing, checkpointable associative rows, and manager candidate mask in the runtime. Verify pre-event timing, exclusion of goal ID and privileged pose from $c_t$, no gradients through $M$, row-local updates, exact resume (including the odor noise stream), OFF/ON observation shapes, and the noise mean/standard deviation. Use the current runtime's frozen ImageNet ResNet-18 trunk and trainable DG projection/BatchNorm contract.
2. Create one complete `sample_factory` StudySpec under `hpc_runs/studies/` with schema `intrmotiv/study/v1`, current workflow version, 40 expected runs, declared factors/seeds/metrics/contrasts/telemetry, and workspace-only outputs under `/work/classic/fr_xl1014-corridor-geometry`. Validate, render all 40 commands, review unique names and paths, and preserve the generated SHA-256. Repeat print-only review after every spec change.
3. Synchronize the new runtime/workflow components to an isolated NEMO2 source copy and rerun focused tests. Run a small Slurm preflight for OFF/ON and RANDOM/HEBB paths. Inspect actual odor norms and noise statistics, event rates, selection determinism, unchanged Hebbian step size, finite rewards, and manager score ranges; complete the print-only submission audit.
4. Submit a small first tranche of the declared 40 runs through Sample Factory and check support at their actual 10M boundaries before admitting the rest. If support is inadequate, revise the event definition or one predeclared support criterion, create a new StudySpec fingerprint, and restart affected runs; do not adjust the Hebbian rule mid-run. Use ordinary independent jobs for place-field manifests after their print-only review. Keep all bulk training, cache, W&B, log, and analysis paths in the allocated workspace.

## Reusable lesson

The standardized `StudySpec` and manifest evaluators already cover the matrix, paired contrasts, and spatial rollout orchestration. The missing work is a small runtime feature with explicit event timing and state persistence. Fix the associative rule once, use support-matched controls and a shared 10M readout, and reserve new controller or sparsity mechanisms for questions that remain after the 40-run screen.

**Sources:** [standardized workflow](../../04_implementation/standardized_study_workflow.md), [latest release](../../hpc_runs/intrmotiv_study/LATEST.md), [reusable field telemetry](../../04_implementation/reusable_place_field_telemetry.md), and [historical goal-control interpretation](../syntheses/06_high_option_success_goal_sets_and_controls_20260914.md). The supplied discussion motivated the three-factor design; this document fixes a smaller matrix and an unchanged associative learning rule.
