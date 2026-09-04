# Directional and Predictive Recruitment Batch

## Status

Implementation and the 54-run StudySpec are complete. Production is gated on
the corrected 18-cell, seed-99, 5M-step preflight submitted on 2026-09-04.

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
- NEMO2 complete IntrMotiv suite after runtime fixes: 179 passed.
- Corrected print-only preflight: 18 unique commands, workspace paths valid.
- Submitted fresh corrected preflight jobs: `7980651`, `7980652`, `7980653`,
  and `7980656` through `7980670`.
- Submission manifest:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_directional_predictive_recruitment_20260904_preflight_r2/20260904T052100Z/jobs.tsv`
- Fail-closed preflight-to-production gate: job `7980671`. It runs the
  terminal preflight analyzer, StudySpec validation, production print-only
  rendering and audit, and only then submits and audits the 54 production
  jobs. Any failed command prevents submission.

The first preflight (`7980596` through `7980613`) was canceled after live
telemetry exposed two integration defects. Direct C15 frontier selection had
incorrectly required a pre-existing reliable route, causing source-local free
exploration with no goal targets. PRED also counted the current source's
persistent CA3 trace when testing whether a distinct predecessor was unique.
The fixes let direct frontier control command any observed landmark while
retaining reachability checks for waypoint/common-manager planning, and remove
only the current source row from PRED predecessor competition. Regression
tests cover both cases. The canceled outputs are excluded from all gates and
production artifacts.

A first corrected resubmission (`7980629` through `7980646`) was also canceled
before certification because its unchanged output paths resumed the canceled
checkpoints. The final `preflight_r2` namespace has distinct run directories
and W&B group; its first summary began at zero environment steps.
