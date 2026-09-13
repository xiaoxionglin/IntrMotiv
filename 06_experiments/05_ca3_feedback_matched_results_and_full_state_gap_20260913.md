# 05 — CA3 feedback comparison: matched results and missing full-state test

Prepared 13 September 2026 from saved September 8 artifacts; no fresh runtime audit. Historical CPD StudySpec schema intrmotiv/study/v1, declared workflow 1.4.1. No numerical interpolation: use matched cells and leave untested conditions missing.

All primary rows: CPD C15 ARR/FIRST/JOINT/FiLM, legacy BN, 16 DG units, no auxiliary predictor, no discrete recruitment, same 75M budget and seeds 8/99/123. DIRECT/BPTT refers to feedback-history gradient routing; PPO-to-DG remains JOINT. Spatial measurements use 100k behavior samples at 75M; coverage and FIRST lift use 65–75M. These are policy-driven snapshots, not fixed-observation comparisons. All 16 units are active in all 27 selected snapshots.

| Feedback | Mono-field units, seeds 8 / 99 / 123 | Mean map cosine ↓ | Mean distinct peaks /16 | Mean coverage AUC ↑ | Mean FIRST lift ↑ |
|---|---|---:|---:|---:|---:|
| No feedback | 4 / 1 / 1 | 0.170 | 14.3 | 40.0 | 0.945 |
| Recent CA3 gate / DIRECT | 0 / 0 / 0 | 0.304 | 13.0 | 31.3 | 0.774 |
| Recent CA3 gate / BPTT | 0 / 3 / 0 | 0.248 | 13.7 | 37.3 | 0.790 |
| Recent CA3 additive / DIRECT | 5 / 0 / 1 | 0.276 | 11.7 | 40.1 | 0.801 |
| Recent CA3 additive / BPTT | 0 / 0 / 0 | 0.295 | 13.7 | 32.4 | 0.847 |
| Recent CA3 + actions gate / DIRECT | 0 / 6 / 0 | 0.296 | 12.0 | 38.4 | 0.785 |
| Recent CA3 + actions gate / BPTT | 0 / 0 / 0 | 0.314 | 14.3 | 24.5 | 0.749 |
| Recent CA3 + actions additive / DIRECT | 0 / 0 / 1 | 0.286 | 15.3 | 34.2 | 0.833 |
| Recent CA3 + actions additive / BPTT | 0 / 0 / 0 | 0.292 | 13.7 | 32.6 | 0.816 |
| Full 71-position CA3 feedback | Not tested in this comparison | — | — | — | — |

Coverage AUC is an average cumulative visited-cell count, not a percentage. FIRST lift is the arithmetic mean of the saved per-run lift values, not a pooled ratio or causal intervention. Every row has mean lift below one. Map cosine measures overlap; peak count does not establish spatially distributed mono-fields.

## Interpretation

No feedback variant improves mean map cosine over the matched baseline. CA3 additive DIRECT ties baseline mean mono-field count (2/16), with worse overlap; CA3 gate BPTT has 1/16 mean versus baseline 2/16. The selected gate-BPTT S99 example improves mono count from 1 to 3, but this does not generalize across seeds. These results downgrade the earlier selected-seed framing. They support neither a reliable mono-field gain from learned recent feedback nor an extrapolated full-state benefit. They do not establish that full-state feedback would fail.

## Separate evidence: fixed re-entry inhibition

Persistent-control 55–75M mean coverage: F_GATE 73.6 versus F_INHIB 75.4; G_SHARED 90.9 versus G_INHIB 91.2. Later F_INHIB snapshots: S8/S99 zero mono-fields at 200M, S123 zero at 300M. These have different controllers, objectives and ages from CPD and must not be inserted as matched CPD rows. See [inhibition status](persistent_intrinsic_control_status_20260909.md) and [matched coverage audit](persistent_intrinsic_control_learning_audit_20260908.md).

Full-state feedback is a missing experimental factor, not an intermediate condition whose performance can be interpolated. Reuse the no-feedback/recent-feedback controls when comparing it, and retain per-seed outcomes rather than selected checkpoints.

Sources: [canonical 75M and 65–75M joined results](results/late_outliers_20260908/cpd_terminal_rankings.csv), [spatial snapshots](results/late_outliers_20260908/cpd_snapshots/per_snapshot.csv), [CPD design](../05_plans/ca3_feedback_predictive_dg_batch.md).

Workflow lesson: joining by explicit cell metadata exposed a matched baseline absent from the earlier outlier narrative. Reuse saved canonical tables, keep measurement windows explicit, and report seed-paired controls before cross-batch analogies. No training or study configuration was changed.
