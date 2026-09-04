# Directional and Predictive Recruitment Batch

## Status

Implementation and the 54-run StudySpec are complete. Production is gated on
the 18-cell, seed-99, 5M-step preflight submitted on 2026-09-04.

- Study: `directional_predictive_recruitment_20260904`
- Batch: `intrmotiv_directional_predictive_recruitment_20260904`
- W&B project: `SF_IntrMotiv_DirectionalPredictiveRecruitment`
- Study SHA-256:
  `72b0ac2d04ad7a297a674f96d4f32c85d48dcf89a9fc62abb7243adb22ea53aa`
- Matrix: base x recruitment x goal decoder x seed = `3 x 3 x 2 x 3 = 54`
- Training: fresh initialization, 75M environment steps

## Implementation

The graph recruitment path now has backward-compatible `incident`, `monitor`,
`directional`, and `predictive` victim rules. The new DIR rule requires all 15
off-diagonal outgoing targets to retain at least 0.5 decayed attempt mass and
replaces a mature source only when its reliable outgoing degree is zero. DIR
alone also handles mutual reliable duplicates with both control times at most
four decisions.

PRED derives `(source, goal, predecessor context, outcome)` records only from
goal options whose start and completion occur in the same accepted rollout.
It uses the most recent distinct exclusive landmark in the first `R` CA3
positions and discards the derived statistics after the rollout's recruitment
decision. It adds no model parameters or checkpoint state.

Online telemetry covers attempt coverage, fully tested and zero-outdegree
nodes, DIR/PRED eligibility and assignments, PRED context evidence, and FiLM
parameter/modulation diagnostics. Standard place-field NPZs now optionally
carry frozen control, passive, maturity, and row-assignment buffers; the
analyzer writes a per-unit join with spatial information, activity, graph
degrees, and four-connected half-peak component counts.

## Verification

- Local workflow/study tests: 19 passed.
- NEMO2 focused recruitment, telemetry, replay, and FiLM tests: 34 passed.
- NEMO2 complete IntrMotiv suite: 178 passed.
- Print-only preflight: 18 unique commands, workspace paths valid.
- Submitted preflight jobs: `7980596` through `7980613`.
- Submission manifest:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_directional_predictive_recruitment_20260904_preflight/20260904T024250Z/jobs.tsv`

Early runtime checks show nonzero FiLM modulation after valid target samples
and no premature DIR assignments under partial outgoing-pair coverage. PRED
context-event telemetry remained zero through approximately 1.4M frames and
must become active by the terminal preflight analysis; otherwise production
will not be submitted.
