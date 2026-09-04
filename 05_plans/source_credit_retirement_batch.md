# Source-Credit and Retirement-Trigger Batch

## Frozen study contract

- Study: `source_credit_retirement_20260904`
- Batch: `intrmotiv_source_credit_retirement_20260904`
- Project: `SF_IntrMotiv_SourceCreditRetirement`
- Schema/workflow: `intrmotiv/study/v1`, `1.4.1`
- Validated StudySpec SHA-256: `aa34bb2ef868df37dbafc38e7c4b8dc5c9cc684e5f8dba5e05a586d068ad8a0d`
- Matrix: C15-FiLM × `{ARR,SRC}` × `{MON,DIRS,DIRO,PREDS,PREDO}` × seeds `{8,99,123}`
- Training: 30 fresh runs to 75M environment steps

The declarative source of truth is
[`hpc_runs/studies/source_credit_retirement.study.json`](../hpc_runs/studies/source_credit_retirement.study.json).

## Implemented mechanisms

ARR and SRC share an exact, behavior-time matched event set. A destination onset is retained only when its nearest non-simultaneous predecessor resolves to a valid dominant onset within the same 64-step actor rollout. ARR credits the destination row/time; SRC credits the predecessor row/time. Reward collisions are summed before flattening and the row/reward tensors are replayed unchanged. Destination masks still drive all auxiliary encoder losses.

The retirement endpoint gate is independently selectable as `silent` or `open`. Both paths compute the same victim and residual diagnostics; only the final activity gate differs. MON computes graph, DIR, PRED, and endpoint diagnostics while assigning no rows.

PRED evidence is checkpointed in decayed `[source, goal, predecessor]` success and attempt tensors. It is updated in completed-intentional-option order with a 5k-option half-life. Study cells require at least 4.0 decayed attempts in each inconsistent context. Reassignment clears every tensor slice in which the victim appears on any axis.

## Runtime gate

The ten seed-99 preflights train to 5M. Production is blocked unless all cells have finite losses, exact replay, active target/frontier/PRED paths, the selected encoder-credit loss only, valid retirement ordering, and at least 50% credited dominant onsets.

First preflight submission:

- Slurm jobs: `7984015`–`7984024`
- Dependent scientific gate: `7984025` (`afterok` on all ten preflights; failed
  the matchability manipulation check and therefore submitted no production
  jobs)
- Manifest: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_source_credit_retirement_20260904_preflight/20260904_submitted/jobs.tsv`
- State at 2026-09-04 18:50 CEST: all ten running, gate dependency pending,
  production not submitted. The gate performs a second StudySpec validation,
  production print-only review, and submission audit before launching any of
  the 30 production jobs.

The first gate found 49.7% pooled credited events: 4.1% were genuine boundary
drops, but 46.2% were labeled alignment failures. The dominant cause was a row
tie bug: simultaneous predecessor rows shared the same nearest CA3 age, while
the matcher selected the lowest DG index before checking the stored dominant
label. The repaired matcher preserves the nearest predecessor time and selects
the behavior-dominant row within that nearest-age tie. It does not search a
more distant predecessor or weaken the 50% gate. A fresh `_preflight_v2` is
required before production.

The short v2 live audit exposed a second zero-age case before completion, so
jobs `7984986`–`7984995` and gate `7984996` were cancelled after about seven
minutes. Continuously active DG rows are reinjected into CA3 slot zero even
when they are not new-onset candidates; these simultaneous rows must also be
excluded from the predecessor set. The final matcher therefore requires a
strictly positive CA3 age, then resolves the behavior-dominant row within the
nearest positive-age tie. This is covered by a separate regression test and is
being evaluated in a clean `_preflight_v3`.

The 50% source-storage manipulation check is evaluated over the pooled dominant
arrival events from all ten preflight cells, matching the study requirement.
Per-cell fractions remain in the report as diagnostics, but stochastic
per-cell deviations do not redefine the batch-wide storage criterion.

Final v3 execution:

- Preflight jobs: `7985000`–`7985009`
- Dependent production gate: `7985010`
- Focused source-credit tests: 10 passed
- Full IntrMotiv suite after both matcher fixes: 208 passed
- StudySpec SHA-256 remains
  `aa34bb2ef868df37dbafc38e7c4b8dc5c9cc684e5f8dba5e05a586d068ad8a0d`

## Evaluation

Online summaries are synchronized at 25M, 50M, and 75M. Place-field telemetry uses five seed-99 checkpoints (5M, 15M, 30M, 50M, 75M) and terminal checkpoints for seeds 8 and 123, for 70 ordinary Slurm jobs. All 30 terminal runs receive frozen target-intervention evaluation. Spatial primary outcomes are mono-field fraction, component counts at 30/50/70% peak, dominant-component mass, spatial information, active-only map cosine, and confidence–field-spread correlation.

Production acceptance follows the requested SRC-in-MON, open-gate manipulation, and useful PRED-open criteria. DIR is reported independently; no eligible DIR victim is a scientific null result rather than an implementation failure.
