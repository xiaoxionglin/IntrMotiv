# Fixed-reward transfer: latest shared-step comparison, 2026-09-11

All 21 repeat-8 production runs are compared over global environment frames **50,784,640–60,784,640**. The endpoint is the latest covered by every declared run and metric at collection time. Available per-run maxima span 60,784,640–97,615,872; individual latest endpoints would be an unequal-progress comparison.

Each score is the event-sample mean within this inclusive 10M-frame window, then averaged equally across seeds 42, 1234, and 9999. Higher length-weighted score is better.

| Condition | Mean | Difference from scratch | Seed 42 | Seed 1234 | Seed 9999 |
|---|---:|---:|---:|---:|---:|
| SCRATCH | 9.115 | +0.000 | 8.966 | 9.300 | 9.079 |
| SCR_DG_FROZEN | 6.103 | -3.012 | 6.669 | 4.612 | 7.028 |
| SCR_DG_TUNE | 8.547 | -0.568 | 9.201 | 7.578 | 8.862 |
| SCR_POLICY_TUNE | 7.342 | -1.773 | 9.159 | 3.670 | 9.196 |
| SAT_DG_FROZEN | 5.355 | -3.761 | 5.435 | 5.590 | 5.039 |
| SAT_DG_TUNE | 8.522 | -0.593 | 8.642 | 8.297 | 8.627 |
| SAT_POLICY_TUNE | 9.031 | -0.084 | 8.827 | 8.999 | 9.269 |

Scratch leads all six transfer condition means. Both frozen-DG conditions trail scratch in every paired seed. Tuned DG is closer than in the earlier 34–44M-frame comparison, but still below scratch on average. SAT policy transfer is nearly tied (−0.084); SCR policy transfer beats scratch in two seeds but seed 1234 remains poor (3.670). This is a descriptive three-seed interim comparison, not evidence of final convergence or a general impossibility of transfer.

## Reuse and verification

The canonical collector now supports `collect-online STUDY BATCH_ROOT OUTPUT_DIR --latest-common`. The [workflow guide](../04_implementation/standardized_study_workflow.md#latest-shared-step-comparisons-170) provides the full NEMO2 command. It uses StudySpec identities, reads each history once, keeps full scalar histories, and fails on missing histories or empty/nonfinite metric windows. The default and explicit-window modes are preserved. All 35 canonical/common-window/repeat-8 tests passed locally and on NEMO2.

Use [per_run.csv](data/frt8_latest_common_20260911/per_run.csv) for condition/seed comparisons and [analysis_manifest.json](data/frt8_latest_common_20260911/analysis_manifest.json) for provenance. Schema: `intrmotiv/study/v1`; implementation: `1.7.0`; study version: `1.5.0`; unchanged study SHA-256: `99910170169d07d10d584d0aa645b3993189daaa9dbe78ba28b9ad1e2c74b56d`.

## Reusable lesson

The previous procedure parsed event histories twice: once to discover endpoints and again to align the window. The single-pass option removes that repeated work and avoids reservoir sampling. The canonical NEMO2 TensorBoard data and StudySpec were authoritative; no W&B authentication was needed. The existing version-pinned test initially failed after the intentional version bump and was updated. Future checks should use this command directly and retain its small CSV/manifest rather than rebuilding run lists or comparing unmatched latest endpoints.
