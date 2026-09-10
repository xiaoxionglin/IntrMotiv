# DG capacity, goal conditioning, and waypoint planning

Status: agreed experimental matrix recorded on 10 September 2026. The user
subsequently authorized implementation through verified production startup.
See the [implementation and launch record](dg_capacity_goal_conditioning_launch_20260910.md).
This replaces the earlier conversational matrices.

## Objective and evidence

Learn spatially structured representations and command-dependent control
through intrinsic motivation. Be smart, if some implementation is for sure going to fail just by reasoning, patch it.

The [historical three-goal audit](04_three_goal_context_conditioning_and_dg_capacity_20260910.md)
motivates testing greater capacity, but does not establish that capacity alone
repairs IntrMotiv control. Increasing DG count also expands memory and decoder
inputs. Goal input to DG and waypoint execution are experimental hypotheses.

## Agreed matrix: 27 production runs

| Execution | Goal conditioning | DG sizes | Seeds |
|---|---|---|---|
| C15 direct frontier | Worker only | 16, 32, 64 | 8, 99, 123 |
| C15 direct frontier | DG + worker | 16, 32, 64 | 8, 99, 123 |
| Waypoint frontier | DG + worker | 16, 32, 64 | 8, 99, 123 |

Each run starts fresh and trains for **300M environment frames**, using
**frameskip 4 and the eight-action navigation interface**. Use the SCR-derived
navigation8 configuration, arrival encoder credit, and no discrete
recruitment/replacement. Retain the frozen ImageNet ResNet trunk, trainable
DG projection, existing normalization and optimizer settings, one policy,
and no PBT. Keep `Hippo_R=8`, `Hippo_L=64`; derive dependent state sizes.

The worker is the learned CA3 readout. Existing worker FiLM remains enabled
in every arm; there is no separate additional CA3-readout modulation factor.
Excluded: goal-discrimination auxiliary, waypoint worker-only arm, compact
goal embeddings/shared modulation, balanced-goal selection, and a new
timeout-exclusion rule. These mechanisms were discussed but are not part of
the final matrix.

## Conditioning and graph semantics

- Worker-only retains unconditioned DG and CA3 with the existing
  goal-conditioned decoder.
- DG + worker adds identity-initialized goal-dependent affine modulation of
  DG preactivations before thresholding and writing worker CA3 memory.
  The intrinsic manager's selected landmark ID conditions both this modulation
  and the existing worker FiLM. It is distinct from the environment instruction.
- Maintain an unconditioned DG branch and canonical memory for achievement,
  graph evidence, manager state, and representation losses. Conditioned worker
  activity cannot declare its own target achieved.
- Reuse a single visual projection/BatchNorm evaluation per observation.
  Preserve the baseline PPO stop-gradient into base DG; train the new
  modulation through the worker loss. Verify its gradient through memory.
- Select goals from grounded evidence, store the exact behavior goal, and
  replay that goal during learning. Reset canonical and worker episode memory
  without introducing reset-spanning graph transitions.

**Waypoint graph construction is an explicit additional mechanism.** Current
C15 disables edge exploration. The waypoint arm must use the existing passive
discovery and deliberate edge-validation machinery to obtain executable
routes. Its contrast with direct DG + worker therefore tests routing plus
graph construction, not routing alone. Do not silently enable validation in
the C15 controls. Finalize and review the exact waypoint settings in the
StudySpec before implementation/submission.

Preserve current C15 direct-selection semantics: target-specific frontier
scores rank observed landmarks, without requiring a validated route. Source
affects deadline and exploration behavior. At a silent-DG timeout, the stored
option source is reused and the same frontier can be selected again. Record
consecutive same-target timeouts and uninterrupted command duration.

The baseline unknown-travel-time deadline and exploration horizon remain
64 decisions. Known duration $T$ gives $\lceil1.2T\rceil+2$ decisions. At repeat
4, the fallback spans 256 simulation frames; this matches navigation8 but is
half the physical duration of the repeat-8 predecessor. No hidden horizon
rescaling is included.

## Evaluation and interpretation

Save checkpoints at 5M, 25M, 75M, 150M, and 300M frames. Reuse canonical online
spatial telemetry and manifest-driven place-field evaluation: five checkpoints
for seed 99 and terminal checkpoints for seeds 8 and 123.

Report coverage, silent units, mono-field counts/fractions, peak diversity,
active-only map cosine, spatial information, and pre-threshold maps. Separate
canonical landmark maps from conditioned worker activity; compare different
goals on matched observation histories.

At 75M and 300M, evaluate all seeds using frozen checkpoints and graph buffers.
Extend the existing intervention evaluator compatibly for actual command
interventions from reproducibly matched physical starts. A post-hoc shuffle
of target labels is not an executed command intervention. Use up to 16 observed
sources, four observed outgoing targets per source, and five repeated starts
per pair, with deterministic panel selection. Record unsupported trials and
panel coverage rather than assuming the panel represents every landmark.

Measure unconditioned target arrival, first distinct landmark, timeout,
duration, physical endpoint, and action sensitivity. For waypoint runs,
distinguish intermediate hits from final-frontier completion and report
validated route availability and validation/exploration time.

Goal dilution remains a measured risk, not a claimed solved problem. Log
per-goal attempts, commanded decisions, successes, timeouts, eligible-goal
counts, command entropy, and modulation-gradient exposure. Primary comparisons
use equal frame budgets; secondary control-versus-attempts-per-goal analyses
are observational. Keep selection unchanged within each declared mechanism.

Predeclare capacity contrasts within each row, DG-conditioning contrasts
within direct C15, and waypoint-plus-validation versus direct under DG +
worker conditioning. There is no waypoint worker-only condition, so this
matrix cannot estimate the full conditioning-by-routing interaction.

Require useful spatial coverage and command-dependent arrival together for a
promising interpretation. Report all seeds and uncertainty; reward, graph
density, or action sensitivity alone does not establish controllability.

## Implementation and launch gates

Use StudySpec schema `intrmotiv/study/v1`, workflow **1.5.0**, and the canonical
package under `hpc_runs/intrmotiv_study/`. Create the declarative study under
`hpc_runs/studies/` during implementation; it becomes the sole run inventory.
The study is now declared in `dg_capacity_goal_conditioning.study.json`, with
validated SHA-256 `53197eede2cf4183a1546bf2b7320c8e58a593b558f1a9673973897680bb5ed7`.
Record updated fingerprints after study changes and repeat print-only review.

Before launch, resolve the waypoint configuration and validate identity
initialization, detector invariance under command changes, single BatchNorm
updates, state sizing, gradient routing, actor/replay agreement, resets, and
checkpoint restoration. Run nine ordinary Slurm preflights, one per
size/mechanism at seed 99, to 2M frames, retaining initial checkpoints.
Require correct intrinsic-only reward routing, finite learning, working
telemetry, and exercised waypoint validation/routing before production.

Synchronize reusable runtime/evaluator changes to NEMO2 and pass focused tests
there. Use canonical validate, render-runs, print-only launcher review, and
audit-submission. All bulk output must resolve under
`/work/classic/fr_xl1014-train`. Existing training batches remain separate.

## Review boundary and reusable lessons

Do not launch downstream transfer automatically. Review pretraining with the
user first. The intended later transfer starts with trainable task-goal
embeddings reusing pretrained representations and policies, with matched
scratch controls. Verify archived task randomization/instruction semantics
before defining that downstream experiment.

Use resolved configs and the actual manager branch as authoritative evidence:
direct and waypoint eligibility/timeout rules differ, and worker FiLM already
is goal-conditioned CA3 readout. Keep tested mechanisms separate from proposed
ones; preserve explicit graph-construction and frame-repeat differences in
future comparisons.
