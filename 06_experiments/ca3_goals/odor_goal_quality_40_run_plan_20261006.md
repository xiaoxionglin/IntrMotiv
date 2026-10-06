# Odor grounding and CA3 goal quality: 40-run plan

**Status, 7 October 2026:** this first implementation was superseded after review. Its 40 production jobs, 8290663–8290715, were cancelled as a group from the audited submitted manifest. The replacement [v2 StudySpec](../../hpc_runs/studies/odor_ca3_goal_quality_v2_20261007.study.json), SHA-256 `c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de`, fixes accepted-event timing, preserves the original C15 frontier score, makes candidate and odor draws reproducible on a fixed stored rollout, and records actual candidate sets. The [replacement launch record](odor_goal_quality_40_run_v2_20261007.md) documents its passed gate and all 40 newly submitted jobs. No result from the cancelled jobs will be treated as a 75M outcome.

## Implementation and launch record

- Runtime source: isolated `codex/odor-ca3-goal-quality-20261006` branch, commit `d30d3dcb1e26834a57ceb460db32f9b92fe1ff73` pushed and remote hash verified. The corrected source was synchronized to `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_odor_ca3_20261006/`, where all 57 focused tests passed. Odor reaches only the DG projection; the fixed-rate CA3 prototypes and quality are learner-owned checkpoint buffers; replay teacher-forces its stored behavior goal.
- Calibration: [fixed 10k-decision artifact](odor_gain_calibration_20261006.json). The mean clean odor norm was 0.41927 and mean visual-block norm 17.89645, yielding one shared gain of 42.68481611601036. The local and NEMO2 map Lua SHA-256 matched. With noise, the qualification ON runs initially logged odor norms around 22–23; OFF logged zero.
- Production source of truth: [workflow-1.14 StudySpec](../../hpc_runs/studies/odor_ca3_goal_quality_20261006.study.json), SHA-256 `f4e1c5d7d3be4178c095432ad3481d4c09cddcc8c674db6119a39ff0f2823e91`. It validates as 40 unique cells. The repeated final print-only manifest at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_odor_ca3_goal_quality_20261006/20261006T172506Z/jobs.tsv` passed the canonical command and workspace-path audit. A separate first-tranche print-only manifest contains the four declared cells.
- Superseded qualification jobs 8288060–8288063 were cancelled. A local full-learner reproduction had exposed a `3843`-versus-`3847` projection width error in legacy recruitment telemetry. The corrected source appends odor through one shared DG input builder. Its local ON/HEBB4 full-learner check reached 135,168 frames against 131,072 with exit code zero. Tests cover event timing, row-local updates, invalid transitions, exact quality-buffer restoration, candidate eligibility and ties, replayed behavior condition, one learner update, odor statistics, and the shared DG input width/gain.
- Corrected NEMO2 qualification jobs 8290652–8290655 (OFF/RANDOM4, OFF/HEBB8, ON/HEBB4, ON/RANDOM8) each completed with exit code zero and 532,480 logged frames against a 524,288-frame target. At the final scalar step all 16 DG goals had accepted event support in every cell; CA3 scores and C15 frontier scores were finite. OFF odor norms were zero, while ON norms were 25.5 and 27.0. These short runs establish runtime health, not comparative performance.
- Production first tranche: 8290663–8290666 (OFF/ALL16, OFF/HEBB4, ON/RANDOM4, ON/HEBB8; seed 8) passed an early health gate around 557k–590k frames. CA3 quality had support on 14–16 goals per cell; 6, 12, 16, and 11 distinct goals, respectively, had been commanded cumulatively. All recorded scalars were finite. OFF odor norm was zero; ON norms were 21.1 and 23.5. The remaining 36 cells were then submitted without changing the StudySpec or runtime.
- Superseded production submission: first manifest `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_odor_ca3_goal_quality_20261006/20261006T223828Z/jobs.tsv`; remaining manifest in sibling directory `20261006T224625Z`; combined submitted manifest in the batch directory as `40_cells_submitted.tsv`. The canonical `audit-submission --submitted` result `40_cells_submitted_audit.json` verified the original 40 commands and workspace paths. Those jobs were cancelled before analysis after the follow-up code review found the four problems summarized above.

## Question and minimal design

Can spatially grounded sensory input and a small, context-predictable DG candidate set improve exploration and *command-caused* navigation? Keep the C15-derived $F=16$, $R=8$, $L=64$ backbone, ARR encoder credit, FiLM target ID, `frontier_direct`, PPO-to-DG STOP, and existing intrinsic rewards. Do not add DG replacement, HER, waypoints, geometry control, a predictor network, or a sparsity sweep.

Run the complete $2\times5\times4=40$ Cartesian design for 75M environment frames per run. Use paired seeds **8, 99, 123, 2026**. The fourth seed is proposed here and must be fixed in the StudySpec before any run. Use one shared map and reward protocol across cells.

| Goal-set condition | Odor off | Odor on | Main question |
| --- | ---: | ---: | --- |
| ALL16 | 4 | 4 | Grounding with the original goal vocabulary |
| RANDOM4 | 4 | 4 | Effect of reducing the controller's choices to four |
| HEBB4 | 4 | 4 | Value of CA3-based selection at four goals |
| RANDOM8 | 4 | 4 | Effect of reducing the choices to eight |
| HEBB8 | 4 | 4 | Value of CA3-based selection at eight goals |

This compact balanced design tests odor, the number of candidates offered at each manager choice, and CA3 selection at two capacities. Every DG goal remains available across choices; a permanently restricted goal vocabulary is a different experiment. An exposure-selected set or visit-only frontier selector would answer further questions, but would displace replication or another primary contrast at 40 runs. Keep them as a later batch if the present results warrant it.

## Fixed sensory and associative mechanisms

Use four broad, spatially separated Gaussian odor channels with clean peak amplitude 1. All runs instantiate the same DG input shape; OFF supplies four zeros. For ON, add independent zero-mean Gaussian noise to each channel at each observation, with fixed standard deviation **0.15 of the clean channel peak** (between the requested 0.1 and 0.2), then apply the fixed odor gain: $x_{\mathrm{odor},k}=\lambda_o[o_k(x)+\epsilon_k]$, $\epsilon_k\sim\mathcal N(0,0.15^2)$. Do not clip the noisy values, which would bias weak odor regions upward. Key the dedicated noise stream by environment seed and observation index, independently of action and goal sampling. This reproduces the noise on a fixed stored trajectory; Sample Factory still does not checkpoint a live DMLab episode, so a process restart need not continue the identical future trajectory. Deliver odor only to DG, with no direct policy, decoder, manager, or CA3 bypass. Set one fixed odor gain using a separate preflight trajectory to make the **clean** mean odor-block norm comparable to the visual-block norm; log the resulting noisy norm too. Record the centers, width, noise scale, gain, map hash, and calibration artifact; never recalibrate by run or seed. This tests one noisy sensory cue, not a gain or noise sweep.

For a transition from decision $t$ to $t+1$ that first reaches exclusive DG event $j$, pair the detached, goal-independent CA3 state $c_t$ recorded **before** the action and event observation with $j$. Normalize $c_t$ to $\hat c_t$ and update only the observed row:

$$
M_j \leftarrow (1-\alpha)M_j+\alpha\hat c_t,\qquad \alpha=0.01.
$$

Apply this same event-triggered rule from the first frame through 75M in **every cell**, including ALL16 and RANDOM. No learning-rate schedule, optimizer, prediction loss, autograd path, or feedback into DG/CA3 is permitted. Non-event transitions leave $M$ unchanged. Save and restore $M$, event counts, and quality state in checkpoints. A fresh run and an exact resume must produce the same rows and counts on a fixed event stream.

Score each event *before* updating its row. Once its own row and at least one alternative row have nonzero prototypes, use the own-versus-best-other cosine margin against nonzero alternatives:

$$
q_t(j)=\cos(\hat c_t,M_j)-\max_{k\ne j}\cos(\hat c_t,M_k).
$$

Maintain an event-wise quality EMA $Q_j\leftarrow0.99Q_j+0.01q_t(j)$ when the margin is defined, and count its valid observations as $n_j$. Initialize $Q_j=n_j=0$. The same update and the same fixed coefficients apply throughout training. For selection, shrink uncertain quality toward zero with $S_j=n_jQ_j/(n_j+100)$. This is one continuous evidence rule, not a minimum-support gate or a time-triggered switch. Record event support and prequential margins; a high score from a few events is not evidence of reliable predictability. Fix the shrinkage constant before production and do not tune it to outcomes.

At **every manager target choice from the start of training**, form a fresh candidate set from DG IDs currently selectable by C15's frontier logic. HEBB$K$ takes up to $K$ IDs with the highest current $S_j$ values; RANDOM$K$ samples up to $K$ uniformly without replacement. Use all selectable IDs when there are fewer than $K$, and use C15's exploration fallback only when none is selectable. Randomize exact HEBB score ties, so all-zero initial scores give a uniform candidate set. Hold the sampled set only for that target choice. The next choice reads the latest $S_j$ and samples again. There is no 10M handoff, minimum-support threshold, frozen goal set, or scheduled change to $M$, $Q$, or candidate selection. The replacement uses a deterministic key from the actor seed, manager-choice count, and detached CA3 context; restoring a fixed stored state reproduces its candidate set without depending on PyTorch's global random stream. A fresh DMLab episode after process restart can still follow a different trajectory.

Compute C15 novelty rank and uncertainty over the original full frontier pool, then restrict the current choice to the candidate set. This preserves C15's score scale across $K$. Keep the original 16-dimensional DG and target-ID decoder; mask other targets only for the current manager choice, and continue logging all DG events. The RANDOM and HEBB pairs at each $K$ must share every implementation detail except the candidate-selection criterion.

## Outcomes and contrasts

The primary training outcome is external coverage AUC over **0–75M** frames, paired by seed. Show early and late windows separately to see whether selection quality improves as evidence accumulates, without changing the rule between windows. The primary causal contrasts are:

1. HEBB4 minus RANDOM4 and HEBB8 minus RANDOM8, separately at each odor level: does CA3 selection improve the same-size candidate set?
2. RANDOM4 and RANDOM8 versus ALL16 at each odor level: does offering fewer candidates per choice help without quality information?
3. ON minus OFF within each matched goal-set condition: does sensory grounding help, and does it alter the selection effect?
4. Four versus eight within RANDOM and within HEBB: is there a capacity tradeoff?

Report paired seed values and uncertainty intervals, rather than treating ten related contrasts as ten independent discoveries. With four seeds this is a mechanistic screen, not a precise effect-size estimate. Record candidate and commanded-ID frequencies, set turnover, event frequencies, support, $Q_j$, margin distributions, and the spatial coverage of frequently versus rarely selected goals; explicitly test whether HEBB merely favors frequent events. A favorable HEBB result with no advantage over exposure-matched post hoc controls should motivate a frequency-selected production arm before claiming predictive quality.

Coverage alone is insufficient. At the final checkpoint, use the established matched-start, alternative-command intervention to measure first distinct DG outcome, target-specific success above shuffled command, and action-distribution change. Use macro averages over commanded goals, and preserve failed/time-out trials in denominators. Report ordinary option success separately; historical C15-family work shows that high aggregate target-hit rates can coexist with weak command specificity. The spatial panel must include silent units, active-only map cosine, peak diversity, spatial information, and pre-threshold maps. An odor benefit that only creates duplicate place fields is not the intended result.

Use the standard 10k-decision place-field rollout at 5/10/25/50/75M for seed 99 plus terminal rollouts for seeds 8 and 123. Seed 2026 needs at least the terminal command intervention and core scalar outcomes; a terminal field rollout is desirable if capacity permits. Do not infer fixed-trajectory drift from policy-driven checkpoint rollouts.

## Implementation and launch gates

1. Implement and test the shared odor input, CA3 event pairing, checkpointable associative rows, and per-choice manager candidate mask in the runtime. Verify pre-event timing, exclusion of goal ID and privileged pose from $c_t$, no gradients through $M$, row-local updates, exact CA3 module restoration on a fixed stored rollout, uniform HEBB tie behavior at initialization, OFF/ON observation shapes, and the noise mean/standard deviation. The environment and actor random-stream resume limitation is recorded above. Use the current runtime's frozen ImageNet ResNet-18 trunk and trainable DG projection/BatchNorm contract.
2. Create one complete `sample_factory` StudySpec under `hpc_runs/studies/` with schema `intrmotiv/study/v1`, current workflow version, 40 expected runs, declared factors/seeds/metrics/contrasts/telemetry, and workspace-only outputs under `/work/classic/fr_xl1014-corridor-geometry`. Validate, render all 40 commands, review unique names and paths, and preserve the generated SHA-256. Repeat print-only review after every spec change.
3. Synchronize the new runtime/workflow components to an isolated NEMO2 source copy and rerun focused tests. Run a small Slurm preflight for OFF/ON and RANDOM/HEBB paths. Inspect actual odor norms and noise statistics, event rates, selection determinism, unchanged Hebbian step size, finite rewards, and manager score ranges; complete the print-only submission audit.
4. Submit a small first tranche of the declared 40 runs through Sample Factory and check event support, candidate diversity, and finite scores before admitting the rest. If these fail, revise the event or scoring contract, create a new StudySpec fingerprint, and restart affected runs; do not alter a running rule. Use ordinary independent jobs for place-field manifests after their print-only review. Keep all bulk training, cache, W&B, log, and analysis paths in the allocated workspace.

## Reusable lesson

The standardized `StudySpec` and manifest evaluators already cover the matrix, paired contrasts, and spatial rollout orchestration. The missing work is a small runtime feature with explicit event timing and state persistence. Fix the associative and selection rules once, draw matched-size candidate sets from the start, and reserve new controller or sparsity mechanisms for questions that remain after the 40-run screen.

**Sources:** [standardized workflow](../../04_implementation/standardized_study_workflow.md), [latest release](../../hpc_runs/intrmotiv_study/LATEST.md), [reusable field telemetry](../../04_implementation/reusable_place_field_telemetry.md), and [historical goal-control interpretation](../syntheses/06_high_option_success_goal_sets_and_controls_20260914.md). The supplied discussion motivated the three-factor design; this document fixes a smaller matrix and an unchanged associative learning rule.
