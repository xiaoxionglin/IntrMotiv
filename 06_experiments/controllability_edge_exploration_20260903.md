# Controllability and Edge-Exploration Batch

## Status

Implementation and the complete 48-run StudySpec are in place. Production is
gated on a 16-cell, seed-99, 1M-step preflight. The StudySpec is authored
against workflow `1.1.0`; the backward-compatible workflow implementation is
`1.2.0`, which adds intervention-manifest generation.

- Batch: `intrmotiv_controllability_edge_exploration_20260903`
- W&B project: `SF_IntrMotiv_ControllabilityEdgeExploration`
- Study SHA-256:
  `4b940a37e09bdabc7efd8bbedc21053194b5288543fb22d32d0c9bc1323e9734`
- Matrix: representation × exploration head × manager objective × geometry ×
  seed = `2 × 2 × 2 × 2 × 3 = 48`
- Training: fresh initialization, 75M environment steps

The 48 production jobs have not been submitted. A print-only production plan
was generated and audited against the StudySpec; it will be regenerated after
the runtime preflight passes.

## Implemented controller

The prior asymmetric HRL managers are replaced for this study by one common
manager with `FREE_EXPLORE`, `NAVIGATE`, and directed `EDGE_PROBE` behavior.
The worker receives the manager-selected target immediately. Its goal decoder
combines a learned 32-dimensional target embedding and a 32-dimensional
projection of the selected CA3 trace with the shared policy state before the
fixed `[128, 128]` decoder.

In the separate-policy cells, free exploration uses its own one-layer,
128-unit decoder, actor head, and value head. Visual and sequence features are
shared. Navigation and edge probes always use the goal worker. PPO replay
teacher-forces the behavior target, four geometry values, and manager mode
stored in recurrent rollout state.

The graph uses directed intentional success/attempt counts and `Tctrl`.
Routing requires positive `Tctrl`, confidence at least 0.5, and posterior
reliability `(successes + 1) / (attempts + 2)` at least 0.5. Hits update both
counts and mean `Tctrl`; timeouts update attempts only. Both counts retain the
5k-option half-life. Candidate probes require passive confidence at least two,
are never inferred from their reverse edge, and receive the specified
success-UCB plus normalized connectivity-gain score. Failed probes have a
64-decision per-stream cooldown.

Reliable-edge deadlines are `ceil(1.2 × Tctrl)`. Probe deadlines are
`ceil(1.2 × passive_time)` with a 64-decision bootstrap fallback, and free
exploration lasts 64 decisions. Geometry-on cells use the existing fixed
reduced-action SE(2) integration, six motion-policy inputs, four relative
geometry inputs, and three-nearest candidates within 32 integrated units.

## Evaluation contracts

Online telemetry includes mode occupancy, branch losses, probe lifecycle,
edge calibration, graph connectivity and funnel metrics, target control,
timing, path efficiency, and loop/straightness measures. Standard spatial
telemetry uses seed 99 at 5M, 25M, 50M, and 75M and seeds 8 and 123 at 75M.

The workflow now derives `intervention_manifest.tsv` from exactly the same
checkpoint inventory as place-field evaluation. At 75M, the intervention
runner freezes policy and graph updates, begins trials at exclusive source
events, balances all 15 alternative targets up to five attempts per ordered
pair within 100k decisions, records pose/path/outcome data, and computes both
source/context-matched shuffled controls and counterfactual action
sensitivity.

## Verification and preflight history

The authoritative NEMO2 suite passes 184 tests, including immediate replay,
branch-specific gradients, checkpoint round trips, directed graph updates,
reliability dropout and decay, cooldown, connectivity scoring, deadline rules,
and intervention output contracts.

The first submitted preflight stopped at frame zero with a useful integration
failure: the core required the new six-value behavior descriptor while Sample
Factory had allocated only its historical one-value target suffix. All 16 jobs
were cancelled and their artifacts were preserved under
`train_dir/_failed_preflights/`. The allocator and terminal-persistence size
now both include target id + four geometry values + mode id, and a regression
test asserts that exact state-size contract.

The corrected preflight submission is recorded under:

```text
/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/
intrmotiv_controllability_edge_exploration_20260903_preflight/
20260903T181401Z/
```

Production remains blocked until all 16 cells reach 1M with finite losses,
zero replay mismatch, free and goal activity, probe activity only in edge-aware
cells, and active goal/free branches. The branch-isolation unit test separately
verifies that a free sample cannot update the goal-only heads and a probe
cannot update the exploration-only heads.

## Canonical artifacts

- StudySpec: `hpc_runs/studies/controllability_edge_exploration.study.json`
- Production adapter: `hpc_runs/controllability_edge_exploration.py`
- Preflight adapter: `hpc_runs/controllability_edge_exploration_preflight.py`
- Plan auditor: `hpc_runs/audit_controllability_preflight.py`
- Runtime-gate analyzer: `hpc_runs/analyze_controllability_preflight.py`
- Standard telemetry guide:
  `04_implementation/reusable_place_field_telemetry.md`

