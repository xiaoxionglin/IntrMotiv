# Action-conditioned CA3: 80-run DMLab execution record

**Status, 13 November 2026: local implementation and native qualification passed;
0 production Slurm receipts. No DMLab result is established.**

This is the execution/result owner for the
[authorized handoff](../../../05_plans/ca3_action_transport_dmlab_handoff_20261113.md).
The originating handoff remains unchanged. GPT-6-Astra with `xhigh` reasoning
owns runtime implementation; the execution coordinator owns the canonical
study, audit, common-panel diagnostics and this record.

## Scientific comparison

The [StudySpec](../../../hpc_runs/studies/ca3_action_transport_80_20261113.study.json)
is the sole production matrix: eight conditions, each with ten matched
worker/environment/optimizer seeds, for exactly **80 production cells**.
The current candidate budget is 75M frames per cell, subject to remote compute
qualification before the immutable submission fingerprint is fixed.

| Condition | Memory | Predictive head | Role |
| --- | --- | --- | --- |
| BASE | Original scalar CA3 | None | Current C15 architecture under the common fixed-DG regime |
| SCALAR | Original scalar CA3, action in prediction head | Linear | Auxiliary-only control |
| PAR_LINEAR | Expiring parallel action register | Linear | Explicit ordered action-history control |
| PAR_MLP | Same parallel register | MLP 64 | Readout nonlinearity contrast |
| TAG_R1 | Per-action rank-1 tags | Linear | Rank contrast without assuming rank 1 is order-blind |
| TAG_R2 | Per-action rank-2 tags | Linear | Primary transport candidate |
| TAG_R2_MLP | Same rank-2 tags | MLP 64 | Readout nonlinearity contrast |
| SHARED_R1 | Shared rank-1 tags | Linear | Commuting action-order control |

The scalar auxiliary uses the same one-step DG/bypass supervision as the other
six auxiliary arms, reusing the existing active/zero loss kernel. This refines
the handoff's existing-auxiliary arm to avoid conflating memory with a different
prediction horizon or loss. Shared rank 1 replaces command path integration:
the existing command integrator also changes passive-graph admission, which
would confound this memory comparison.

Shared rank-1 commutation removes action order within DG-silent intervals.
Intervening DG injections can still preserve event timing; this is not a claim
that all histories become order-blind.

All arms retain the reduced five-action corridor task, frozen ImageNet
`layer2_resnet18`, F16/R8/L64 scalar CA3, target-ID FiLM commands, C15 manager
and graph rules, worker reward, optimizer settings and observation budget.
PPO sees the added memory features; only the sensory auxiliary trains the
transport maps. The maps transform tag coordinates, never DG rows or ages.
Full age-indexed storage implements overlap and expiry.

Production is conditional on **one common learned DG checkpoint**, frozen
together with its BatchNorm statistics. The ten seeds replicate worker,
environment and auxiliary fitting; they do not replicate DG pretraining.
The source candidate is the retained C15 TEMPORAL seed-99 100,007,936-frame
checkpoint listed in
[the canonical retained-checkpoint manifest](../../results/cosyne_2027_checkpoint_fields/available_terminal_manifest.tsv).
Its exact path is in the StudySpec. Remote hash, configuration, unchanged
projection/BN and nonzero sensory activity must be verified before release.
The workers are freshly trained under the same original goal-command
interface; this is **not a pretrained-worker transfer test**.

The main on-policy training arms share settings and seeds. Their trajectories
can diverge when their policies diverge, so these are not identical-data fits.
A common factual observation/action panel supplies matched evaluation data.
Pure mechanism-capacity claims additionally require matched-data fitting;
paired online reward alone cannot establish those claims.

## Causal and gradient contract

For state $X_t$, candidate action $a_t$ first shifts the scalar memory and
transports tags, with finite expiry. The predictor reads this advanced state
before the target observation $u_{t+1}$ is injected. Acting-state updates use
the previous action that produced the current observation, including the
reset sentinel. No physical position or supplied heading enters training.

The auxiliary replays within each learner sequence using detached numerical
boundary states and detached observed sensory targets. Unlike the previous
detached-source auxiliary, interior tag transitions remain differentiable.
Invalid gaps and episode termination break the causal chain. Truncated replay
does not credit actions before the stored boundary. PPO consumes detached
memory features and trains the worker rather than the action maps.

Each auxiliary predicts post-threshold DG plus the changing worker bypass.
The declared active-coordinate Smooth L1 coefficient is 1 and zero-coordinate
coefficient is 0.1; bypass Smooth L1 coefficient is 1. Recursive prediction
rectifies DG only at assimilation, then advances both DG memory and bypass.
Predictions for unexecuted actions are hypotheses until their outcomes are
observed in an executed matched-start intervention.

## Qualification and release gates

| Gate | Current evidence/status |
| --- | --- |
| Synthetic reference | 20 tests pass across action transport, kinematic inspection and parallel-register modules; the saved synthetic result remains distinct from DMLab |
| Canonical workflow, matrix and reuse controls | 51 tests plus 14 negative subtests pass; exactly 80 unique cells and ten pairs per contrast |
| Common-panel diagnostic | Six focused tests pass: episode/invalid boundaries, sparse active errors, train-only baseline, recursive windows, workspace/symlink checks and Python 3.10 provenance hashing |
| Runtime mechanism/core/learner | [Full suite log](qualification/runtime_tests.log): 576 passed, 15 skipped |
| Actual runtime parser for all eight arms | Passed for all 80-cell production conditions and all eight short qualification cells; inherited recruitment-only goal-adapter reset disabled |
| Local native DMLab smoke | 80 real decisions and one actual learner auxiliary update pass with cached ImageNet trunk and an explicitly once-calibrated random-DG fixture; this does not qualify the declared learned DG source |
| Source checkpoint and native sensory activity | Pending NEMO2 access |
| Compute-node short qualification and exact reload | Pending source release, runtime tests and Slurm preflight |
| Resource/storage/queue audit | Pending NEMO2 access; all declared write roots use `fr_xl1014-corridor-geometry` |
| Production launcher print-only audit | Pending remote qualification |
| Production submission | **0/80 receipts; none claimed submitted** |

The project worktree is `/tmp/intrmotiv-ca3-action-80-20261113`, branch
`codex/ca3-action-transport-80-20261113`, based on project commit `11c950a`.
The clean runtime mirror was `master` at `206ee833`; runtime changes are
isolated at `/tmp/intrmotiv-ca3-action-runtime`, branch
`codex/ca3-action-transport-20261113`. Preserve the original checkouts.

[Study validation](study_validation.json) binds candidate production fingerprint
`7ffacbbb1229821e835f505099cc46f4fe99a66d919fe26ba150d213be6aa63e`
to 80 actual Sample Factory command expansions. The separate
[qualification StudySpec](../../../hpc_runs/studies/ca3_action_transport_qualification_20261113.study.json)
contains eight 1,048,576-frame cells and has fingerprint
`4f908120eee94b572b7a6bc1674c2cc849512c26e8aa20276fb4129f1b2406f3`.
Neither has Slurm receipts. Adapter command expansion is not a remote launcher
print-only/submission audit. The final release must revalidate these drafts.

The [resource audit](qualification/condition_resources.json) records actual
parser/core dimensions. Baseline/scalar state has 1,252 floats per stream,
parallel state 1,607, and tag state 5,796. Parallel memory adds 355 worker
features; tag memory adds 64. Per-action rank-1/rank-2 maps have 40/80 trainable
parameters; shared rank 1 has 13. Linear prediction heads have 33,495 scalar,
43,790 parallel and 35,351 tag parameters. Thus these are matched scientific
settings, not equal state/parameter/compute budgets. Compute-node throughput
and peak memory remain to be measured.

The saved NEMO2 access note records a prior authentication failure. Per
`AGENTS.md`, automated SSH is paused until the user confirms a successful
manual `ssh nemo2`/OTP login. Local implementation continues meanwhile.
Do not submit unrelated old plans or resubmit jobs `8306935`/`8314863`.

Workflow 1.15.3 tightens explicit write-path audits without changing the v1
StudySpec schema. The remote shared checkout is documented as older; release
must synchronize the canonical package into an isolated source checkout and
rerun focused tests. Lexical audit does not replace remote symlink resolution,
capacity/write-fsync checks and inspection of job cache/temp environment.

The reconciled canonical/runtime package also retains graph-export and
explicit discovery options; their combined suite passes 51 tests plus 14
subtests. [File hashes](workflow_source_hashes.json) confirm the two local
package copies match. The native fixture exposed missing installed level Lua;
the existing `prepare_runfiles` staged repository assets in isolation, and
both StudySpecs now pin the release runfiles path explicitly.

The final [native certificate](qualification/native_auxiliary.json) and
[log](qualification/native_auxiliary.log) bind the script and StudySpec hashes.
All 80 factual source/target pairs are valid; all five actions have support,
transport and head gradients are nonzero, both update, and replay discrepancy
is zero. Actual-image DG outputs are invariant to other batch members in this
fixture, and all encoder weights/BN buffers remain bitwise unchanged. This
certifies mechanics, not a prediction gain, planning result or production run.
The reset seed actually used by the native engine is recorded; the wrapper's
ignored `reset(seed=...)` keyword is not treated as a seed certificate.

Before production, bind the final StudySpec SHA-256, runtime commit/source
hashes, source checkpoint/config hashes, parser certificate, native
qualification/reload artifacts and launcher `jobs.tsv`/`submission.json`.
Audit queue/history and existing receipts for these identities before the
single submission. Keep all qualifications in separately named namespaces;
they are not additional production cells. Changes before release require a
fresh print-only review. Do not change a submitted StudySpec.

## Analysis completion ledger

All components remain open until real training/checkpoint artifacts exist.
Online statistics alone are interim.

| Component | Required analysis and current missing input |
| --- | --- |
| Sensory prediction | Common factual panel; DG active/zero/event and bypass errors, horizons 1/2/4/8/16/32/64, state-only fitted baseline, zero/persistence/train-frequency controls, action support. No production panel or model yet. |
| Counterfactual prediction | Exact-start executed alternative commands with frozen models; recursive sensory predictions along each executed action sequence, shared-start certificate and failed/censored trials. Not supplied by a predicted branch alone. |
| Place fields | Manifest-driven 10k-decision evaluations, seed 99 at five ages and all ten terminal seeds; thresholded/pre-threshold maps, silence, active-only cosine, information and peak diversity. No checkpoints yet. |
| Trajectories | Occupancy and segmented paths, reset boundaries, matched checkpoint and terminal summaries. Common-panel and own-policy trajectories stay distinct. |
| Graphs | Complete directed attempted/unattempted edge support, reliability/connectivity and prospective outcomes. Graph confidence is separate from physical success. |
| Goal control | Frozen matched-start alternative target commands using the existing backend, physical arrivals versus detector hits, action change, time, failures/censoring and source/goal exposure. |

`evaluation.ca3_action_prediction` reuses the existing observation-panel
collector and model loader. Its CLI requires a Slurm job and explicit disjoint
training/test episode IDs; no forecast crosses a terminal or invalid gap.
Frequency and fitted affine state-only controls see training episodes only.
The ridge diagnostic is not claimed to match production parameter or compute
budgets. All arms must produce the same encoded-sensory hash on a common panel
before their prediction errors are compared.

Wall/contact, free-space, repeated-landmark and loop/route-merging strata must
be recorded or derived under a declared evaluation-only protocol. Missing
labels are reported as missing, not inferred from an arbitrary zero-displacement
threshold. Action words absent from the panel's calibration episodes do not
prove absence from the earlier online training history; a genuinely held-out
composition claim needs the corresponding training-support audit/refit.

## Reproduction and continuation

Use `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python` locally. From the project
worktree:

```bash
python -m pytest -q hpc_runs/test_intrmotiv_study.py hpc_runs/test_ca3_action_transport_study.py
python -m hpc_runs.intrmotiv_study validate hpc_runs/studies/ca3_action_transport_80_20261113.study.json
python -m hpc_runs.intrmotiv_study render-runs hpc_runs/studies/ca3_action_transport_80_20261113.study.json --output /tmp/ca3_action_transport_80_runs.json
```

Use the existing Sample Factory launcher, then canonical `audit-submission`;
after actual receipt use `--submitted`. Follow the
[standard workflow](../../../04_implementation/standardized_study_workflow.md)
and [telemetry contract](../../../04_implementation/reusable_place_field_telemetry.md).
Keep bulk panels, NPZs, checkpoints, logs, caches and figures in the allocated
workspace; copy selected lightweight evidence here.

The efficient approach was to inspect canonical StudySpecs and live runtime
contracts, reuse the existing loss/collector/launcher and isolate both source
trees. Two concrete infrastructure issues were found: lexical path escapes
and the panel writer's retired-allocation restriction. Their fixes and remaining
remote acceptance gates are tracked in the existing entries in
[infra.md](../../../infra.md). Native execution, not parser success or model
selection, remains authoritative for claiming the 80 runs started.
