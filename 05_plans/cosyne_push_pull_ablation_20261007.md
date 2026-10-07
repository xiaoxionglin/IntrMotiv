# COSYNE push–pull ablation: C15 and CPU2048 Direct DDQN+HER

Status: experimental plan, 7 October 2026. No new StudySpec, source release, training job, or evaluation job has been created or submitted under this plan.

## Decision and claim

Run the same two-factor ablation **within each of two established controller families**. Corrected-core C15 PPO is the exploration reference at 100M environment frames. CPU2048 Direct F16 DDQN+HER is the control-oriented reference at 300M frames. Both use a learned sparse DG and a target-conditioned worker, but they differ in action protocol, controller update, goal adapter, replay, and horizon. Estimate treatment effects within each family and paired seed; use agreement across families as a robustness observation, never pool their absolute scores.

The testable claim is narrow: *Does temporal-distance pressure on the DG encoder, on the goal-conditioned worker, or their combination contribute to exploration, DG differentiation, and commanded-target behavior in these implementations?* In these HRL designs, the worker pull is a distance **bonus conditional on a target hit**. Its ablation leaves the target-hit reward intact. This is an ablation of the implemented reciprocal distance mechanism, not a test of removing all goal learning or of the original flat decoder reward by itself.

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

| Factor | On | Off | Preserve in both levels |
| --- | --- | --- | --- |
| Encoder push | Existing distance-weighted, arrival-credited DG loss | Multiply **only that interval-credit loss** by zero | DG projection and BatchNorm remain trainable; batch unused-unit recruitment, multi-onset penalty, threshold, CA3, normalization, and other baseline DG terms stay fixed |
| Worker pull | Existing `hit_distance` mode with distance-bonus coefficient $\alpha=0.1$ | Keep `hit_distance` mode and set $\alpha=0$ | Target-hit reward, hit/outcome semantics, manager, goal input, option termination, PPO or DDQN objective, and HER settings stay fixed |

The current `encoder_grad_coeff=0` or `extra_encoder_losses=False` must **not** implement push-off: the former suppresses other DG learning terms, and the latter removes maintenance losses. Introduce one default-on coefficient for the interval-credit component in the canonical learner, and verify its exact loss/gradient boundary in both PPO and the fresh-data DG step used by stored-state DDQN. The existing distance-bonus coefficient appears sufficient for pull-off, subject to a runtime test showing that zero removes the bonus in both real target-hit rewards and HER relabeled terminal rewards. Keep raw diagnostic credit separate from applied loss so the manipulation check is visible.

## Factorial and provenance

Each family gets the complete $2\times2$ matrix below with seeds 8, 99, and 123. Every arm starts from a fresh initialization with the same seed and baseline configuration within its family. The fresh push-on/pull-on arm is the internal comparator; old checkpoints are external references only.

| Arm | Push | Pull | Meaning |
| --- | ---: | ---: | --- |
| Both | 1 | 1 | Fresh intact baseline |
| Push only | 1 | 0 | Worker keeps target-hit learning without temporal-distance bonus |
| Pull only | 0 | 1 | DG keeps maintenance learning without interval credit |
| Neither | 0 | 0 | Target-hit learning and DG maintenance remain active |

Create **two complete Cartesian StudySpecs** under `hpc_runs/studies/`, one per family, after qualifying the new coefficient. The validated specs, rather than this prose, will own run names, factors, seeds, commands, metrics, contrasts, milestones, telemetry metadata, schema, workflow version, and generated SHA-256. Use the current [standardized study workflow](../04_implementation/standardized_study_workflow.md) and its [latest deployment record](../hpc_runs/intrmotiv_study/LATEST.md).

- **C15:** Reconstruct the exact corrected-core C15 training configuration from its saved per-run configs and source provenance, cross-checking the archived commands in `06_experiments/results/late_lift_audit_20260908/inventory.json`, then run fresh to the historical 100M target. Do not treat the 600M continuation StudySpec as the original fresh baseline: it loads the 100M checkpoint, and its historical output root is in the retired workspace. Preserve C15's reduced action set, repeat-8 protocol, **delayed** target timing, frontier manager, orthogonal recruitment, DG width 16, target interface, and original normalization/credit semantics unless a change is explicitly necessary and shared by all four arms.
- **CPU2048 Direct DDQN+HER:** Consolidate the original seed-8 and seed-99/123 [StudySpecs](../hpc_runs/studies/controller_cpu_selected_her_seeds.study.json) into one fresh three-seed factorized definition, retaining Direct F16, stored-state replay, one controller update per 2,048 accepted decisions, 1,024 main plus up to 1,024 HER positions, repeat-4 navigation action set, target-ID FiLM, and the 300M target. Do not substitute the earlier frozen-DG native DDQN pilot; it cannot ablate learned encoder push.
- Put all training, checkpoint, W&B, cache, Slurm, telemetry, and temporary outputs in `/work/classic/fr_xl1014-corridor-geometry`. Pin an isolated NEMO2 source release; synchronize the canonical workflow and run its focused tests there. Explicitly preserve historical depth-response, normalization, and update defaults when rendering with a newer workflow. Preserve source, baseline-config, checkpoint, and StudySpec hashes. Do not change the running checkout or historical submitted StudySpecs.

The complete production matrix is 24 fresh runs: 12 C15 and 12 CPU2048. At the historical horizons that is 4.8B environment frames in total. C15 repeat-8 and CPU2048 repeat-4 make equal frame counts unequal decision counts, so the families are separate replications of factor effects, not a joint performance leaderboard. Measure current CPU2048 throughput in the qualified source before reserving the full matrix; its original report did not establish a controlled long-run CPU throughput estimate.

## Predeclared comparisons and measurements

For each outcome $Y$ and each family, report the three individual paired-seed contrasts, their mean, and the interaction:

$$
\Delta_{\mathrm{push}\mid\mathrm{pull}}=Y_{11}-Y_{01},\quad
\Delta_{\mathrm{pull}\mid\mathrm{push}}=Y_{11}-Y_{10},\quad
\Delta_{\mathrm{interaction}}=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
$$

Also show push without pull and pull without push. State the favorable direction for every measure before interpreting it; a lower map cosine is favorable only when active units, information, and peak diversity remain healthy. Three seeds support transparent paired descriptions, not a strong population-level significance claim.

1. **Manipulation and health:** actual applied interval-credit loss/gradient, scheduled versus applied DG credit, unused-unit and multi-onset losses, DG density/silence/usage, BatchNorm behavior, real-hit distance-bonus frequency and magnitude, HER realized fraction and relabeled reward, main/auxiliary TD support, and fresh DG/controller optimizer counts. Check that pull-off has zero distance bonus while hit rewards remain, and push-off has zero interval gradient while maintenance gradients remain.
2. **Physical exploration:** same-protocol complete-episode coverage AUC, unique accessible cells, mobile-window 20/40-decision returns, and path/occupancy summaries at exact shared milestones. Use 5M, 25M, 50M, and 100M C15 targets; 5M, 25M, 75M, 150M, and 300M CPU2048 targets. Compare within family at the same checkpoint and terminal window; do not substitute each arm's latest available age.
3. **Representation:** canonical online and frozen 10k-decision DG maps, silent units, active-only cosine, spatial information, mono-field fraction with denominator, distinct active peaks, and pre-threshold maps. Replay a **predeclared common observation/action-history panel within each family** through every checkpoint so a field difference is not explained solely by different policy visitation. Preserve episode boundaries and inspect coverage of the panel itself.
4. **Commanded behavior:** use the manifest-driven exact-start target intervention with frozen policy and graph. Report executed-minus-matched-shuffled target-event success, initial-action distribution change, eligible sources, attempted and complete ordered pairs, wrong-first outcomes, timeouts, and physical endpoint/trajectory distributions. A DG target event is not automatically arrival at a unique physical destination; qualify the target fields before making that claim. Use the same intervention protocol and declared source/target coverage within each family; label restricted panels as restricted.

C15's main behavioral readout is coverage; CPU2048 Direct's main behavioral readout is command-caused outcome lift. Report **both readouts and the representation panel in both families**, including nulls and trade-offs. A positive interaction in a behavioral metric alone does not prove that reciprocal learning created stable place fields. If the pull bonus is almost never delivered, a null pull ablation means the intended pressure was weakly exercised, not that it is generally unnecessary. If neither arm retains exploration, check DG activity and goal opportunity before interpreting failure as a specific push–pull interaction.

## Release sequence

1. **Source qualification:** identify the exact source/config lineage for both baselines; add the single interval-credit coefficient with a default of one. Focused tests must show baseline parity at one, zero interval-only DG gradient at zero, preserved maintenance gradients, and zero real/HER distance bonus at $\alpha=0$ with unchanged target-hit reward. Audit that PPO-to-DG remains stopped and DDQN replay does not update DG.
2. **Print-only study review:** create/validate/render both complete StudySpecs with current workflow; inspect the entire factor expansion, action protocol, reward arguments, seeds, milestone targets, workspace paths, and fingerprints. Qualify the isolated source and workflow on NEMO2. Repeat print-only review after any study-file edit.
3. **Runtime preflight:** run all four arms of each family at seed 99 for a bounded 2M-frame correctness preflight, with a spatial window small enough to fill. Verify checkpoint reload, actual optimizer and reward manipulation checks, finite losses, source-module paths, and representative telemetry. This preflight is not a performance-based arm-selection stage. Measure CPU2048 full-buffer throughput and project wall time/resource use from it.
4. **Production and evaluation:** submit the 24 fresh runs through Sample Factory only after print-only and submission audits. Retain the standardized scalar/spatial artifacts. Run the manifest-driven place-field and matched-command evaluator as ordinary independent jobs after checkpoint inventory and print-only review. Analyze synchronized endpoints, then write one result owner with provenance, paired effects, individual seeds, figures, limitations, and the COSYNE claim boundary.

The validated StudySpecs and runtime gates make the proposed experiment reviewable before launch.

## Reusable lesson and infrastructure

The canonical reports and StudySpecs exposed the key shortcut: the worker bonus already has a coefficient, while encoder push needs an interval-only coefficient. Broad “disable encoder” switches would also disable DG maintenance and answer a different question. The historical CPU2048 throughput is not a controlled benchmark; verify the actual 2,048-decision cadence in the selected source rather than extrapolating an earlier replay implementation. Existing [infrastructure entries](../infra.md) already cover source/workflow versioning and retired-workspace paths; this plan found no separate recurring infrastructure defect to add.
