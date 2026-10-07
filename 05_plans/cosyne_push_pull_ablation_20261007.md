# COSYNE temporal-distance push–pull ablation: C15 and CPU2048 Direct DDQN+HER

Status: calibration in progress, 7 October 2026. Two intact-reference 2M-frame jobs have been submitted under the dedicated W&B project `SF_IntrMotiv_PushPullAblation`; the four-arm ablation StudySpecs and production jobs await fixed-constant calibration and runtime preflight. The live record is [push–pull ablation execution](../06_experiments/controllers/push_pull_ablation_20261007.md).

## Decision and claim

Run the same two-factor ablation **within each of two established controller families**. Corrected-core C15 PPO is the exploration reference at 100M environment frames. CPU2048 Direct F16 DDQN+HER is the control-oriented reference at 300M frames. Both use a learned sparse DG and a target-conditioned worker, but they differ in action protocol, controller update, goal adapter, replay, and horizon. Estimate treatment effects within each family and paired seed; use agreement across families as a robustness observation, never pool their absolute scores. The primary factor comparison is **temporal versus non-temporal credit at the same event locations**; removing a term entirely is a separate necessity control.

The testable claim is narrow: *Does the actual temporal separation assigned to each DG event and target hit matter for exploration, DG differentiation, and commanded-target behavior?* In these HRL designs, the worker distance term is a **bonus conditional on a target hit**. Here “push” and “pull” are shorthand for the two **temporal-distance components**. The experiment does not remove all encoder learning or all goal-directed worker learning, and cannot establish that the complete DG–controller learning system is necessary.

The historical evidence motivating the two references is [corrected-core C15](../06_experiments/corrected_core_reevaluation_20260901.md) and the [exact CPU2048 endpoint analysis](../06_experiments/results/A0_poster_analysis_20260926/batch_summary.md). C15's replicated coverage advantage does not establish commanded control; its target-hit lift is below the shuffled reference in all three seeds. CPU2048 Direct DDQN+HER has a positive bounded executed-minus-shuffled command result at 300M, but fewer than 1% of ordered pairs are complete. Neither historical result is a no-push or no-pull control.

## Interventions

For a qualifying DG onset, write its temporal separation as $d_t$, the CA3 sentinel as $E=R+L-1$, reward scale as $\beta$, and the scheduled active-unit credit mask as $m_{tj}$. The two terms are schematically:

$$
L_{\mathrm{push}}=-\mathbb E_t\!\left[\beta d_t\sum_j m_{tj}a_{tj}\right],
\qquad
r_{\mathrm{worker},t}=\mathbf 1_{\mathrm{hit},t}
\left[r_{\mathrm{hit}}+\alpha\beta\max(E-d_t,0)\right].
$$

The actual learner uses its saved reward/credit alignment and valid-sample masks; this expression is for factor definition, not a replacement implementation. See the [loss catalogue](../04_implementation/architecture/losses.md) and [explicit update contract](explicit_dg_controller_update_contract.md).

For the non-temporal controls, replace $d_t$ by fixed $c_{\rm enc}$ **at the same kind of qualifying onset**, and replace $\max(E-d_t,0)$ by fixed $c_{\rm hit}$ **at a target hit**. Removing a term instead sets its encoder contribution to zero or its worker bonus to zero. In the latter case the worker still receives $\mathbf 1_{\mathrm{hit},t}r_{\mathrm{hit}}$.

| Factor | Temporal reference | Non-temporal control | Term removed, secondary control |
| --- | --- | --- | --- |
| Encoder push | Existing arrival-credited loss weighted by each event's $d_t$ | Credit the **same qualifying onset mask** with a fixed $c_{\rm enc}>0$ instead of $d_t$ | Set only the interval-credit loss to zero; retain DG maintenance and trainable BatchNorm/projection |
| Worker pull | Existing `hit_distance` target-hit reward with bonus proportional to $\max(E-d_t,0)$ | Give the **same target hits** a fixed positive bonus proportional to $c_{\rm hit}$ | Set only the distance bonus to zero; retain the target-hit reward and worker learning |

The non-temporal controls preserve *where* credit is assigned while erasing *which intervals* receive more credit. They ask whether temporal ordering matters beyond generic event reinforcement or a larger fixed hit reward. Term removal asks whether that extra reinforcement or bonus helps at all; temporal versus removed would confound temporal information with reward/gradient magnitude. In every version the DG maintenance losses and target-hit reward remain active. A frozen-DG or no-target-reward arm would answer a separate, more disruptive component-necessity question and is not part of this plan.

Implement explicit `temporal`, `constant`, and `none` modes for the **interval-credit component only**, defaulting to current temporal behavior. Apply the worker bonus mode consistently to real hits and HER relabeled terminal rewards; preserve its existing hit reward. A coefficient of zero already implements the worker-bonus `none` level, but a constant-bonus mode needs an explicit reward path. The current `encoder_grad_coeff=0` or `extra_encoder_losses=False` must **not** implement interval-credit removal: the former suppresses other DG learning terms, and the latter removes maintenance losses. Keep raw diagnostic credit separate from applied loss so the manipulation check is visible.

Choose one fixed $c_{\rm enc}$ and one fixed $c_{\rm hit}$ **per controller family** before the ablation runs, using a preregistered intact-reference calibration of qualifying events and target hits. Match the temporal arm's mean coefficient over that reference distribution, then freeze both constants for all seeds, arms, and training ages in the family. Do not recompute them from each ablated run: that would reintroduce a trajectory-dependent temporal signal. Record the reference checkpoint/rollout, denominator, estimate uncertainty, and realized reward/gradient scales; if target hits are too rare to calibrate $c_{\rm hit}$ reliably, use a declared fixed theoretical constant and label the scale mismatch.

The dedicated W&B project is `SF_IntrMotiv_PushPullAblation`, with distinct groups for calibration, C15, and CPU2048 Direct DDQN+HER. The existing IntrMotiv Git worktree is the source of the StudySpecs and report; the NEMO2 source snapshot under the active workspace keeps running jobs insulated from runtime edits.

## Factorial and provenance

Each family gets the primary $2\times2$ **temporal versus constant** matrix below with seeds 8, 99, and 123. Every arm starts from a fresh initialization with the same seed and baseline configuration within its family. The fresh temporal/temporal arm is the internal comparator; old checkpoints are external references only.

| Arm | Encoder credit | Worker bonus | Meaning |
| --- | --- | --- | --- |
| Temporal / temporal | $d_t$ | $\max(E-d_t,0)$ | Fresh intact baseline |
| Constant / temporal | $c_{\rm enc}$ | $\max(E-d_t,0)$ | Removes temporal weighting from DG credit |
| Temporal / constant | $d_t$ | $c_{\rm hit}$ | Removes temporal weighting from the hit bonus |
| Constant / constant | $c_{\rm enc}$ | $c_{\rm hit}$ | Removes both temporal weightings while retaining their generic signals |

Create **two complete Cartesian StudySpecs** under `hpc_runs/studies/`, one per family, after qualifying the new coefficient. The validated specs, rather than this prose, will own run names, factors, seeds, commands, metrics, contrasts, milestones, telemetry metadata, schema, workflow version, and generated SHA-256. Use the current [standardized study workflow](../04_implementation/standardized_study_workflow.md) and its [latest deployment record](../hpc_runs/intrmotiv_study/LATEST.md).

- **C15:** Reconstruct the exact corrected-core C15 training configuration from its saved per-run configs and source provenance, cross-checking the archived commands in `06_experiments/results/late_lift_audit_20260908/inventory.json`, then run fresh to the historical 100M target. Do not treat the 600M continuation StudySpec as the original fresh baseline: it loads the 100M checkpoint, and its historical output root is in the retired workspace. Preserve C15's reduced action set, repeat-8 protocol, **delayed** target timing, frontier manager, orthogonal recruitment, DG width 16, target interface, and original normalization/credit semantics unless a change is explicitly necessary and shared by all four arms.
- **CPU2048 Direct DDQN+HER:** Consolidate the original seed-8 and seed-99/123 [StudySpecs](../hpc_runs/studies/controller_cpu_selected_her_seeds.study.json) into one fresh three-seed factorized definition, retaining Direct F16, stored-state replay, one controller update per 2,048 accepted decisions, 1,024 main plus up to 1,024 HER positions, repeat-4 navigation action set, target-ID FiLM, and the 300M target. Do not substitute the earlier frozen-DG native DDQN pilot; it cannot ablate learned encoder push.
- Put all training, checkpoint, W&B, cache, Slurm, telemetry, and temporary outputs in `/work/classic/fr_xl1014-corridor-geometry`. Pin an isolated NEMO2 source release; synchronize the canonical workflow and run its focused tests there. Explicitly preserve historical depth-response, normalization, and update defaults when rendering with a newer workflow. Preserve source, baseline-config, checkpoint, and StudySpec hashes. Do not change the running checkout or historical submitted StudySpecs.

The primary production matrix is 24 fresh runs: 12 C15 and 12 CPU2048. At the historical horizons that is 4.8B environment frames in total. C15 repeat-8 and CPU2048 repeat-4 make equal frame counts unequal decision counts, so the families are separate replications of factor effects, not a joint performance leaderboard. Measure current CPU2048 throughput in the qualified source before reserving the full matrix; its original report did not establish a controlled long-run CPU throughput estimate. A complete three-level $3\times3$ matrix would require 54 runs across the two families; do not launch that matrix merely to answer the primary temporal-information question.

## Predeclared comparisons and measurements

For each outcome $Y$ and each family, code temporal as 1 and constant as 0. Report the three individual paired-seed contrasts, their mean, and the interaction:

$$
\Delta_{\mathrm{push}\mid\mathrm{pull}}=Y_{11}-Y_{01},\quad
\Delta_{\mathrm{pull}\mid\mathrm{push}}=Y_{11}-Y_{10},\quad
\Delta_{\mathrm{interaction}}=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
$$

Also show each temporal factor's effect when the other factor is constant. State the favorable direction for every measure before interpreting it; a lower map cosine is favorable only when active units, information, and peak diversity remain healthy. Three seeds support transparent paired descriptions, not a strong population-level significance claim.

1. **Manipulation and health:** actual applied interval-credit loss/gradient, scheduled versus applied DG credit, unused-unit and multi-onset losses, DG density/silence/usage, BatchNorm behavior, real-hit distance-bonus frequency and magnitude, HER realized fraction and relabeled reward, main/auxiliary TD support, and fresh DG/controller optimizer counts. Check that constant modes use the unchanged event masks but have no per-event dependence on $d_t$; verify fixed reference constants and report realized credit/reward scales. For any secondary `none` arm, verify that only its named term has zero gradient or bonus.
2. **Physical exploration:** same-protocol complete-episode coverage AUC, unique accessible cells, mobile-window 20/40-decision returns, and path/occupancy summaries at exact shared milestones. Use 5M, 25M, 50M, and 100M C15 targets; 5M, 25M, 75M, 150M, and 300M CPU2048 targets. Compare within family at the same checkpoint and terminal window; do not substitute each arm's latest available age.
3. **Representation:** canonical online and frozen 10k-decision DG maps, silent units, active-only cosine, spatial information, mono-field fraction with denominator, distinct active peaks, and pre-threshold maps. Replay a **predeclared common observation/action-history panel within each family** through every checkpoint so a field difference is not explained solely by different policy visitation. Preserve episode boundaries and inspect coverage of the panel itself.
4. **Commanded behavior:** use the manifest-driven exact-start target intervention with frozen policy and graph. Report executed-minus-matched-shuffled target-event success, initial-action distribution change, eligible sources, attempted and complete ordered pairs, wrong-first outcomes, timeouts, and physical endpoint/trajectory distributions. A DG target event is not automatically arrival at a unique physical destination; qualify the target fields before making that claim. Use the same intervention protocol and declared source/target coverage within each family; label restricted panels as restricted.

C15's main behavioral readout is coverage; CPU2048 Direct's main behavioral readout is command-caused outcome lift. Report **both readouts and the representation panel in both families**, including nulls and trade-offs. A positive interaction in a behavioral metric alone does not prove that reciprocal learning created stable place fields. If the worker bonus is almost never delivered, a null temporal-versus-constant contrast means the intended pressure was weakly exercised, not that it is generally unnecessary. If the constant/constant arm loses exploration, check DG activity and goal opportunity before interpreting failure as a specific interaction.

### Secondary term-removal contrast

If the primary result makes term necessity relevant to the COSYNE claim, add two predeclared single-factor controls **within each family**: `none/temporal` and `temporal/none`. Compare each with both `temporal/temporal` and its matching constant arm. These add 12 runs total at three seeds per family, bringing the possible total to 36 runs and 7.2B environment frames. They test whether generic event credit or a fixed hit bonus matters in addition to its temporal weighting. They do **not** form a full interaction test for term removal; add `none/none` only if that specific interaction becomes the claim. Keep the primary four-arm StudySpecs immutable and define the removal controls in separate versioned StudySpecs after the decision to run them.

## Release sequence

1. **Source qualification:** identify the exact source/config lineage for both baselines; add narrowly scoped interval-credit and worker-bonus modes with `temporal` as the default. Focused tests must show intact-baseline parity, constant-mode independence from each event's $d_t$ at identical masks, preservation of maintenance gradients and hit rewards, and correct real/HER reward parity. Test the `none` levels separately: zero interval-only DG gradient with maintained other DG gradients, and zero real/HER distance bonus. Audit that PPO-to-DG remains stopped and DDQN replay does not update DG.
2. **Print-only study review:** create/validate/render both complete StudySpecs with current workflow; inspect the entire factor expansion, action protocol, reward arguments, seeds, milestone targets, workspace paths, and fingerprints. Qualify the isolated source and workflow on NEMO2. Repeat print-only review after any study-file edit.
3. **Runtime preflight:** run all four arms of each family at seed 99 for a bounded 2M-frame correctness preflight, with a spatial window small enough to fill. Verify checkpoint reload, actual optimizer and reward manipulation checks, finite losses, source-module paths, and representative telemetry. This preflight is not a performance-based arm-selection stage. Measure CPU2048 full-buffer throughput and project wall time/resource use from it.
4. **Production and evaluation:** submit the 24 fresh runs through Sample Factory only after print-only and submission audits. Retain the standardized scalar/spatial artifacts. Run the manifest-driven place-field and matched-command evaluator as ordinary independent jobs after checkpoint inventory and print-only review. Analyze synchronized endpoints, then write one result owner with provenance, paired effects, individual seeds, figures, limitations, and the COSYNE claim boundary.

The validated StudySpecs and runtime gates make the proposed experiment reviewable before launch.

## Reusable lesson and infrastructure

The canonical reports and StudySpecs exposed the key distinction: a fixed event reward controls temporal **information**, whereas a zero term removes both information and generic credit. The worker bonus already has a zero coefficient; the constant modes need narrowly scoped new paths. Broad “disable encoder” switches would also disable DG maintenance and answer a different question. The historical CPU2048 throughput is not a controlled benchmark; verify the actual 2,048-decision cadence in the selected source rather than extrapolating an earlier replay implementation. Existing [infrastructure entries](../infra.md) already cover source/workflow versioning and retired-workspace paths; this plan found no separate recurring infrastructure defect to add.
