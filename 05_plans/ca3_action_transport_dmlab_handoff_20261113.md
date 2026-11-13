# Handoff: improve IntrMotiv with action-conditioned CA3 and test 80 DMLab runs

## User authorization and requested execution

On 13 November 2026 the user requested: “hand it to a separate chat, use this lesson to improve IntrMotiv design, put it to test, a batch of 80 runs, use astra extra high”.

Use **GPT-6-Astra (`gpt-6-astra`) with extra-high reasoning (`xhigh`)**. Own implementation, qualification, the scientifically matched 80-run study, audited submission, monitoring and complete analysis in the new chat. Continue beyond a design proposal. The user authorizes the study and ordinary reversible implementation, tests, commits, pushes and manifest-reviewed submissions; do not ask again for routine permission. A real authentication, allocation or correctness failure must still be reported honestly.

Workspace: `/home/xiaoxiong/Desktop/Projects/IntrMotiv`. Handoff scientific source commit: `9b208571114e138355d01738e9ec6541a0ec9c3f`, already pushed. Read this repository's `AGENTS.md`, applicable runtime instructions and accessible personal-profile notes. Preserve unrelated changes in every checkout. Keep this originating discussion separate from the execution chat.

## Scientific objective

Determine whether learned action-conditioned CA3 recurrence improves **real DMLab sensory/DG prediction, recursive counterfactual prediction and useful IntrMotiv control**, relative to existing memory and action-history controls. The immediate causal test is future sensory representation prediction, not privileged displacement regression. Do not call a prediction gain planning or reward improvement without the corresponding executed-control test.

Use the smallest principled extension. Preserve literal landmark identity and temporal age in the original scalar CA3 channel; transform only additional tag coordinates. Keep the intentional frozen ImageNet `layer2_resnet18` trunk. Do not redesign DG, worker, manager, rewards and graph simultaneously. In particular, do not replace the original pretrained goal-command interface with free task vectors or scratch machinery and describe that as transfer.

## Read these authoritative artifacts first

1. [Kinematics explanation and limits](../02_algorithm/ca3_action_transport_kinematics.md).
2. [Matched low-rank comparison, weights and inspection](../06_experiments/ca3_goals/parallel_action_register_probe_20261111/low_rank_recurrence/README.md).
3. [Parallel-register comparison and split controls](../06_experiments/ca3_goals/parallel_action_register_probe_20261111/README.md).
4. Runtime-independent prototype: `hpc_runs/ca3_action_transport.py`; factual register adapters: `exports/intrmotiv_prediction_starter/intrmotiv_transfer/state.py`.
5. Weight interpretation: `hpc_runs/inspect_ca3_action_transport.py`; focused tests in `hpc_runs/test_ca3_action_transport.py` and `hpc_runs/test_ca3_kinematic_inspection.py`.
6. [Current architecture reference](../04_implementation/architecture/README.md), [standardized study workflow](../04_implementation/standardized_study_workflow.md), `hpc_runs/intrmotiv_study/LATEST.md`, [telemetry contract](../04_implementation/reusable_place_field_telemetry.md), and existing `infra.md` entries.
7. Existing runtime readout/predictor and action integration, rather than new copies: `ca3_state_readout.py`, `custom_core.py`, `custom_learner.py`, `custom_encoder.py`, `topological_frontier.py`, and `wrappers/reward_shaping.py` under the IntrMotiv DMLab runtime.

The current local runtime mirror is `/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/IntrMotiv/`; NEMO2 runtime source is `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/`. Check actual branches, changes, source hashes and runtime behavior before editing or copying anything. Historical architecture snapshots are not current run defaults.

## What is established, and what is not

- The synthetic task used an isolated DG cue encoding initial heading, then noiseless forward/quarter-turn words. Targets were endpoint displacement, **not** next sensory input. Training words had 2–6 actions and held-out longer words 7–9; the key comparison matched landmark age at nine steps. It used one fixed split and three optimizer seeds.
- Shared rank-2 factors $T_a=I+B\operatorname{diag}(e_a)A$ did not improve longer-word endpoint accuracy. The distinct per-action-factor form $T_a=I+B_aA_a$ with rank 2 and a linear readout achieved 100% rounded endpoint accuracy, approximately $3.50\times10^{-6}$ RMSE. Do not attribute that success to shared factors.
- All three successful weight sets admit position/heading coordinates, with matrix errors below $3.46\times10^{-6}$. The readout-based coordinate change makes the forward upper blocks true by construction; forward lower blocks and turn matrices provide independent validation. The focused suite has 34 passing tests. These are fixed artifacts, not DMLab results.
- Shared rank 1 is order-blind during silent intervals because its action matrices commute. Per-action rank 1 can be noncommuting and has **not** been tested in the synthetic comparison. Do not generalize the shared-rank-one failure into a universal rank-one impossibility claim.
- Pose-like tags merge different routes with the same endpoint, heading and age. Original CA3 retains intervening DG event order within its window, but not every action word. History-dependent task events need appropriate retained state.
- The prototype's pooled tag update is exact only before trace expiry. Production must implement full shift/expiry semantics or a separately proven equivalent, including repeated/overlapping DG events. Do not carry the no-expiry shortcut into arbitrary DMLab histories.
- Our local five-action DMLab set includes strafing and simultaneous turn+forward. Commanded motion differs from realized motion near walls. Initial heading need not be determined by DG identity. These are transfer questions to test, not facts solved by the synthetic result.
- Vector-HaSH is an architectural comparison, not the implemented model. Modular grid states, scaffold attractor correction and heteroassociative sensory retrieval are absent. Do not add them all in the first intervention.

## Minimal mechanism and causal timing

Use $X$ for CA3 state, $u$ for DG input and $a$ for action. Preserve the scalar channel and update extra tag channels as

$$
X^{(0)}_{t+1}[i,k]=X^{(0)}_t[i,k-1]+u_{t+1}[i]\mathbf 1[k<R],
$$

$$
X^{(\mathrm{tag})}_{t+1}[i,k]
=T_{a_t}X^{(\mathrm{tag})}_t[i,k-1]
+u_{t+1}[i]\mathbf 1[k<R]c_0.
$$

Zero the shifted term at $k=0$. The same action rule is shared across landmark identities and ages; it must not mix landmark rows or move information between age slots beyond the existing shift. Per-action low-rank factors are the supported first candidate. Learn the maps from the chosen observation-prediction objective, not DMLab pose labels. Debug position is evaluation telemetry only.

For next-DG prediction, apply the executed action to the state and predict **before** injecting the target observation. Previous action attached to observation $t$ produced $u_t$; candidate action $a_t$ produces $u_{t+1}$. Keep actor/replay alignment, reset sentinels, episode/valid masks, behavior subgoal timing and graph-update contracts exact. The current detached-source auxiliary cannot train new recurrent parameters unless gradients are deliberately routed through replayed transitions. Document this gradient change explicitly.

Qualify initially with fixed DG identities and frozen BatchNorm statistics; separate any subsequent DG-plasticity claim. Sparse DG zeros demand state-only, persistence/zero/frequency and action-support controls, plus active-event metrics. Consider existing pre-threshold/visual features as a diagnostic if zeros conceal action information. Recursively imagined worker states must also predict every changing bypass component the worker consumes. Unexecuted actions are not factual labels merely because the model predicts them.

## Exactly 80 production runs

Build a canonical declarative StudySpec with exactly **80 production training cells including seeds**. Prefer eight scientifically distinct conditions with ten matched seeds over a large unreplicated architecture catalogue. Select actual seeds and run lengths from the current canonical study conventions and available resources; do not infer them from historical filenames. The execution chat owns the final factorization and may refine this starting contrast set after inspecting the real code:

1. Current IntrMotiv baseline, without the new auxiliary.
2. Current CA3 plus the existing action-conditioned prediction auxiliary, keeping actions in the head.
3. Parallel action register with a linear predictive readout.
4. Parallel action register with a small nonlinear predictive readout.
5. Per-action rank-1 tag recurrence with a linear predictive readout.
6. Per-action rank-2 tag recurrence with a linear predictive readout.
7. Per-action rank-2 tag recurrence with a small nonlinear predictive readout.
8. A command-path-integration control reusing existing code, with matched prediction supervision; if it cannot be made clean without changing graph/reward behavior, replace it with a clearly justified order/transport control and document the reason.

This list is a design starting point, not a validated submission manifest. Match data, available action information, supervision, worker/manager settings, observations, training budget and seed identities for each claimed contrast. Record parameter, state and compute differences rather than claiming full resource matching. Keep original-baseline versus auxiliary-effect contrasts distinct from memory-architecture contrasts. Keep only minimal necessary qualifications; label and account for them separately from the 80 production cells. Do not silently launch another production matrix.

At minimum evaluate free-space versus wall/contact transitions, long DG-silent gaps, repeated landmarks, loops/route merging, and held-out action compositions without crossing episode boundaries. Separate same-policy prediction, fixed-command intervention outcomes, and counterfactual rollouts. Use actual sensor prediction and physical success metrics appropriate to the chosen environment; endpoint geometry alone is not a sensory-world-model result.

## Execution requirements and completion

- Reuse the latest canonical `intrmotiv_study` package. Place the final study under `hpc_runs/studies/`, validate it, render print-only plans, audit submission and bind schema/workflow version/study SHA-256 before submission. Keep one manifest source of truth. Reconcile current LATEST and runtime release; the handoff observed workflow 1.15.2 but this must be checked again.
- Preserve pretrained fixed trunk intent and existing unrelated work. Extend canonical source modules, add mechanism/replay tests, then run local and remote focused qualification. No hardcoded ideal 90-degree turns, physical coordinates or externally supplied heading may leak into training inputs.
- All NEMO2 training data, checkpoints, caches, Slurm logs and temporary bulk output must resolve under `/work/classic/fr_xl1014-corridor-geometry`. The previous train allocation is read-only. Source/home storage must stay lightweight.
- Audit existing jobs and StudySpec fingerprints before submission. Submit each production manifest row once, with immutable provenance; do not requeue completed, timed-out or failed runs without an explicit audit.
- On NEMO2 authentication errors, stop automatic retries and request one manual `ssh nemo2`/OTP login. Continue independent local implementation and print-only review while remote access is blocked. Never claim jobs were submitted without Slurm receipts.
- Follow the analysis completion standard: place fields, trajectories, graphs and goal control, plus model-prediction and counterfactual diagnostics. Mark online-only summaries interim. Use the manifest-driven telemetry backend; no DMLab telemetry on the login node and no new telemetry arrays.
- Keep routine monitoring quiet; notify on real failures, meaningful completion or needed user action. Save a concise handoff/status note so later turns can continue without rediscovery.
- Finish with a coherent report separating demonstrated results, hypotheses and limitations. Update existing infra findings instead of duplicating them. Run appropriate focused tests and `git diff --check`; commit and push task changes and verify the remote. Preserve editable, legible, render-checked scientific figures.

The first progress update in the new chat should confirm Astra/xhigh, that this handoff and linked artifacts were read, and the concrete implementation/qualification work underway. Model selection or a queued prompt alone is not evidence that the 80 runs have started.

## Delivery receipt — 13 November 2026

- Receiving chat: [IntrMotiv: action-conditioned CA3 — 80-run DMLab study](codex://threads/01a1d53f-5f86-7a52-bd4d-d4b86df57260).
- Verified persisted turn metadata: `model=gpt-6-astra`, `effort=xhigh`. The receiving chat acknowledged this handoff and read the linked scientific and runtime contracts.
- Execution is underway in isolated project/runtime checkouts. The execution stream records implementation edits and a successful focused `hpc_runs/test_intrmotiv_study.py` run followed by `git diff --check`. This is local qualification progress, not a claim that the final model or production matrix has passed qualification.
- Exactly 80 production cells remain authorized. No production submission receipt exists at delivery. Remote qualification/submission awaits a confirmed manual `ssh nemo2` login with OTP, as required by the saved access failure and project policy; independent local work continues.
- Delivery used the supported persistent Codex CLI because the app tool inventory lacked a create-thread operation. App title/read operations confirmed the receiving chat. Its app status snapshot reported `notLoaded`/`interrupted` while its CLI writer was still executing; the active process and advancing execution stream supplied the progress evidence. A resume attempt was rejected as an active-writer conflict and created no duplicate execution. A follow-up was delivered using `codex queue` to the existing thread.
