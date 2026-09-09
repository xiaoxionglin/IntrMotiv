# Interim status, 9 September 2026

Checked approximately 14:24–14:30 CEST. All 36 W&B runs have fresh telemetry
and all 36 corresponding Slurm jobs are running. G_CONTEXT S99 has recovered
as job 8015660 on n3416. Primary runs are approximately 222–304M frames and
C15 continuations 329–365M of 600M planned. No jobs were modified in this audit.

## Candidate assessment

- F_GATE S123 is the best current partial landmark candidate: the saved 100k
  behavior samples give 6/16 mono-field units at 100M, 4/16 at 200M and 7/16 at
  300M, with all 16 eligible. At 300M its active map cosine is 0.094, spatial
  information 0.195 bits, 15 distinct peak bins, and zero silent units.
  Recent coverage AUC averages 47.5, below its earlier 55–75M value of about
  64.9. This is partial localization with a coverage trade-off, not success.
- C15 S99 is the strongest C15 movement candidate: snapshot path straightness
  is 0.672 at 200M and 0.688 at 300M, versus 0.081 and 0.115 for the other
  seeds at 300M. Recent coverage AUC is 79.7 versus 39.6 and 40.8. However,
  mono-fields decline from 3/16 to 2/16; straightness does not prove controlled
  destination arrival or absence of cyclic routes.
- G_SHARED has consistently broad exploration: recent coverage AUC is
  90.95–92.29 across seeds. All three 200M snapshots have zero mono-field units;
  seed 123 still has zero at 300M. Useful movement does not establish useful
  landmark goals.
- W_REF_JOINT now has recent coverage AUC 87.2–90.9, compared with W_REF_STOP
  71.1–87.2. At the matched 180–200M window JOINT is also higher in all three
  seed pairs. This is an exploration signal worth following, not evidence of
  better target accuracy. Its 200M mono-field counts remain very low.
- G_CONTEXT remains exploratory but should be downgraded as a localization
  candidate: all three 200M snapshots have zero mono-fields, and active map
  cosine rises from 0.35–0.42 at 100M to 0.38–0.50 at 200M. S8 has two silent
  units in the 200M snapshot. Higher spatial information can coexist with
  redundant, fragmented fields.
- F_INHIB S8/S99 retain reasonable exploration (recent AUC 69.1/66.8), but
  neither has mono-fields at 200M. S123 reaches 300M with zero mono-fields.
- F_SIGNED remains the weakest representation candidate. The 200M snapshots
  have only about 0.00049–0.00103 bits of spatial information. Current latest-10k
  W&B windows show 4–9 silent units; the older latest-100k snapshots did not
  show this silence, so it is a recent/window-dependent warning rather than
  proof of permanent unit death. Its occasional mono-field fractions must be
  interpreted with the eligible-unit denominator and near-zero selectivity.
- Baseline exploration stays restricted; separate controllers and longer
  discount still have no demonstrated advantage in physical goal control.

## Interpretation and next evidence

No run has demonstrated the combined target of separated mono-field landmarks,
efficient physical destination control, and broad coarse exploration. Prioritize
F_GATE S123 for raw/pre-threshold independent field inspection and C15 S99 for
trajectory and target-command interventions. Use G_SHARED and W_REF_JOINT as
goal-control comparisons. Keep F_SIGNED first in line for a design review.
The current evidence consists of policy-driven thresholded maps and online
behavior; it cannot establish fixed-panel stability or command-caused arrival.
Coverage AUC is average cumulative visited-cell count, not a coverage percentage.

## Evidence and workflow lesson

`results/persistent_intrinsic_control_submission_20260908/status_20260909.json`
preserves sampled W&B coverage means/counts and canonical spatial metrics.
W&B project: https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_PersistentIntrinsicControl

The existing local W&B login permits direct API access without NEMO2 OTP.
Use it first for summaries, freshness and bounded sampled histories; cluster
access is needed for Slurm truth and raw snapshot validation. Avoid repeating
full TensorBoard scans. The spatial CLI currently rejects 200M/300M study
targets because expected_spatial_targets filters them through historical
DEFAULT_TARGETS. This audit used the canonical calculations with only an
in-memory target-list override from the unchanged StudySpec; no source or
StudySpec was edited. A future workflow fix should validate actual declared
online snapshot targets and add a regression for studies above 100M.
