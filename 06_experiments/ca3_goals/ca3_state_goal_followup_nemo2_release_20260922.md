# CA3 State-Goal Follow-Up — September 22, 2026

## Release identity

- Runtime branch: `codex/ca3-state-goal-followup-20260922`
- Runtime parent: `a28ffee4`
- Batch release commit: `2e8085316e703b158e5f63de613928fe5f2cdd81`
- Workflow: `intrmotiv/study/v1`, version `1.11.0`
- Qualification StudySpec SHA-256: `2a6bb9f6c3e2df76df260984c25acfcf55c3cf886bc6b812508bf8b893569998`
- Production StudySpec SHA-256: `68a0911fbc4719cc20eca4cf4145ffe0405b570d2ac7bf5a49a84643ab9f8437`
- W&B project: `SF_IntrMotiv_CA3StateGoals`
- Workspace root: `/work/classic/fr_xl1014-corridor-geometry`

The running September 22 architecture batch remains immutable. This follow-up uses a separate source branch, output namespace, W&B project, and checkpoint lineage.

## Implemented corrections

Contextual state-readout HER now treats the future raw CA3 snapshot as the semantic goal. A real DG event, ready calibration, and action-probe signature similarity above the recognition threshold define success; DG-slot equality is not required. The slot remains replay bookkeeping only. Target-ID HER retains exact historical ID semantics, inactive endpoints remain legal for HER, and uncertified terminal CA3 targets are rejected.

The predictive objective now includes variance and covariance regularization with coefficients $0.1$ and $0.01$. It computes those terms from detached valid current CA3 states, updates the readout but not upstream CA3, and preserves the readout learning-rate ratio. State-shuffle diagnostics now use deterministic within-horizon half-rotation.

The contextual graph adds non-destructive EMA-signature refinement and configurable exclusive, dominant, and unique-contextual recognition. EMA refinement preserves the active generation and graph evidence. Calibration checkpoints bounded positive and background diagnostic reservoirs and reports similarity quantiles, threshold exceedance, and active-anchor collision without treating background pairs as training negatives.

The prior archived graph-planning patch did not apply cleanly to the deployed runtime parent. Its semantic optimization was ported manually: validated paths and all-edge connectivity gains are cached while graph evidence, thresholds, active masks, and generations remain cache inputs. Focused tests cover output and invalidation behavior.

## Batch matrix

| Flat condition | Anchor mode | Candidate mode |
|---|---|---|
| `CTX_FIXED_DOM_H32` | fixed | dominant |
| `CTX_FIXED_UNIQUE_H32` | fixed | unique contextual |
| `CTX_EMA_DOM_H32` | EMA signature | dominant |
| `CTX_EMA_UNIQUE_H32` | EMA signature | unique contextual |

Every cell uses H32 action conditioning, state-readout goals, contextual graph hits, F64 waypoint control, stored DDQN+HER, and cadence 2048. `study_condition` and the single `config.wandb_tags` value are identical within a cell across seeds.

## Verification evidence

- Full IntrMotiv and workflow selection: 441 passed, 10 environment-dependent skips.
- Focused release selection: 62 passed.
- Black, isort, flake8, and `git diff --check` pass on the complete release diff.
- The repository-wide hook exposed a pre-existing `E402` baseline in `hpc_runs/intrmotiv_study/render_geometry.py`; no unrelated formatter changes were retained.
- Both StudySpecs validate and render exactly four seed-99 qualification rows and twelve production rows at seeds 8, 99, and 123.
- Render review confirmed the four exact flat condition tags and all output/cache/W&B paths under the active workspace.

## Launch state

The complete Git bundle SHA-256 is `3f38f875ad222e36f354264178570ee7f0ec4bd1d4e3f1ffd3d49b319d3286de`. It was verified on NEMO2 and cloned into the independent pinned source `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ca3_state_goal_followup_20260922`. Remote verification reproduced 441 passing tests with 10 environment-dependent skips and both StudySpec hashes.

Print-only submission audits pass for the four qualification rows and twelve production rows. The production render additionally verifies 40 CPUs, 128 GiB, and 96-hour limits for every row. The fresh 2M qualification wave is running as jobs `8144552`–`8144555`; its submitted audit records all four rows, the qualification StudySpec hash, and active-workspace paths. Initial health checks show all cells advancing beyond 32k frames without exceptions or real training NaNs. A thread heartbeat monitors the wave through exact reload, offline telemetry, the mechanical audit, and immediate fresh production submission if the gate passes.

Production is not selected by qualification performance. Zero activations or refinements is scientific output; missing calibration, contextual HER, candidate-mode, EMA-comparison, reload, scalar, or offline-diagnostic paths blocks release.

## Isolated task-general transfer

The transfer feature is intentionally absent from the batch release commit. It is implemented on branch `codex/ca3-task-general-transfer-20260922` at commit `b279cc94feda50920cb3cba4daa8814203f01bd8`, whose parent is the batch release.

`transfer_scope=task_general` transfers structurally compatible encoder/DG and normalization state, CA3 readout and predictor, contextual graph, worker decoder/FiLM, navigation policy, and controller Q. It deliberately leaves fixed-task bindings, return normalization, and critic/reward heads fresh. Loading only model tensors means optimizer, replay, progress, target cadence, publication, exploration counters, RNG, and transient episode state start fresh. Controller initialization constructs the target-Q snapshot from the transferred online model.

The loader requires the exact ordered navigation-eight action vectors and matching structural CA3/readout/graph configuration. Every source and destination model tensor is classified as transferred or deliberately reset; an unclassified new module fails the transfer. Seven focused transfer tests and the full 438-test IntrMotiv selection pass, with 10 environment-dependent skips.

## Reusable release lessons

Use an isolated source clone from the exact runtime parent, because an active batch checkout is not a development surface. When an archived optimization no longer applies, port its invariants and prove behavioral/cache equivalence instead of forcing the patch. Run diff-scoped hooks after the repository-wide baseline check so unrelated historical formatting does not enter a scientific release. Bind the final launch to the runtime commit, StudySpec hashes, rendered commands, submitted manifest, and checkpoint reload certificates.
