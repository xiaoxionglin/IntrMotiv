# Persistent intrinsic control: 8–9 September status and learning audit

The 9 September status is the newer snapshot; the detailed 8 September
audit follows below. Neither is a final controlled outcome.

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

`../results/persistent_intrinsic_control_submission_20260908/status_20260909.json`
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

## Earlier matched learning audit — 8 September 2026

This earlier audit uses a matched 55–75M window and records run health
and control limitations before the later 9 September status above. Its
source note was `persistent_intrinsic_control_learning_audit_20260908.md`.

Audit: 8 September 2026, 17:51–18:05 CEST. Primary runs are approximately
82–122M/600M frames; C15 continuations approximately 177–191M/600M. This is an
interim evaluation of existing telemetry, not a final controlled evaluation.

### Operational health

Slurm listed all 36 jobs as RUNNING, but only 35 had fresh progress.
`PIC_G_CONTEXT_S99`, job 8002740, stopped logging at 17:18:37 on n3113.
The node reports `MIXED+NOT_RESPONDING`; sstat reports communication failure.
The latest checkpoint is `checkpoint_000005502_90144768.pth`, saved at 17:18.
This is an infrastructure incident, not evidence against the context design.
The C15 hotfix recoveries and W_REF_JOINT S123 are advancing. No training jobs
were cancelled or changed during this audit.

### Comparable learning metrics

Coverage AUC is the average cumulative number of visited 100-unit cells within
an episode, not a percentage. The middle column uses the same 55–75M frame
window for every primary run. Latest values average each seed's latest 10M
frames and are therefore not aligned in training age.

| Condition | Mean coverage AUC, 55–75M | Mean latest coverage AUC |
| --- | ---: | ---: |
| F_BASE | 30.2 | 29.5 |
| F_GATE | 73.6 | 61.5 |
| F_INHIB | 75.4 | 69.6 |
| F_SIGNED | 90.0 | 89.4 |
| G_SHARED | 90.9 | 93.5 |
| G_INHIB | 91.2 | 91.8 |
| G_CONTEXT | 86.4 | 83.9 |
| G_SEPARATE | 83.8 | 84.5 |
| G_LONG_DISCOUNT | 90.9 | 89.4 |
| W_REF_STOP | 73.0 | 82.3 |
| W_REF_JOINT | 81.3 | 75.5 |

Reward gating improves coverage in all three paired seeds at 55–75M. This
benefit is not stable in every seed: F_GATE S123 declines from 66.4 to 46.9;
F_INHIB S123 declines from 63.8 to 46.5. F_BASE S8 falls from 31.3 at 25–50M
to 16.4 recently, with policy entropy approximately 0.0145 and only about
18 visited cells per episode. This is consistent with restricted repetitive
behavior, but event motifs/trajectories are required to establish cyclicity.

### DG and control findings

Sixteen primary runs had 100M snapshots. All 16 units were active in each
snapshot. Fourteen snapshots had zero units classified as mono-field;
F_GATE S123 had 6/16 and F_INHIB S99 had 1/16. None meets the 12/16 target.
These are thresholded behavior-window maps, not independent raw/pre-threshold
or fixed-observation-panel results. Aggregate occupancy spans many workers
and episodes and cannot establish 90% coarse coverage within an episode.

F_SIGNED is the strongest representation warning. Its seed-8 100M snapshot
has mean spatial information 0.000423 bits and 0/16 mono-field units. All three
seeds have latest online spatial information only 0.0016–0.0040 bits, versus
roughly 0.08–0.23 for other new designs. DG activity and encoder gradients
remain nonzero. High coverage therefore does not imply useful landmarks.
The implemented signed term is a margin on dominant onsets, not a blanket
penalty on all CA3-conflicting activity; the present evidence does not identify
the precise cause of the poor selectivity.

Goal policies have active commands, zero recorded replay mismatch, and about
0.0010–0.00113 rewarded decisions per decision. This statistic counts reward
pulses, not episode success, and barely changes from 25–50M. The warm-reference
variants have lower pulse rates, roughly 0.00050–0.00077. Their detector differs,
so comparing pulse rates across these groups is not an equal-difficulty test.
STOP PPO-to-DG gradients are zero; JOINT gradients are nonzero. There is no
consistent JOINT improvement in control or spatial metrics yet.

G_CONTEXT has higher online spatial information (about 0.18–0.21 bits) than
G_SHARED (about 0.11–0.12), but also denser DG activity. It warrants spatial
inspection, not a claim of superior fields. Separate controllers and longer
discount have not demonstrated improved navigation. C15 seed 99 currently
has coverage AUC 82.6, versus 45.0 and 48.0 for seeds 8 and 123; their later
training age and different lineage prevent a matched comparison here.

### Recommended decisions

Recover G_CONTEXT S99 as an infrastructure incident. Preserve the main
comparisons through a common checkpoint. F_SIGNED is the first candidate for
reallocation if its remaining 100M spatial snapshots confirm the replicated
low-information warning. Any revised objective should run under a new identity.
Do not spend replacement budget merely on more controller heads or longer
discount before validating the reward target.

The highest-value next test is matched goal commands from identical starts,
with independently measured raw DG localization, physical arrival endpoints,
latency and route efficiency. Add a mismatched-command control. These tests
distinguish intentional destination control from accidentally encountering a
broad/multi-field DG identity. Use the existing intervention/common-panel
workflow; no new rollout was launched here. A better localized fixed-reference
detector is a minimal follow-up only if an independent panel qualifies one;
none is established by this audit.

### Evidence and reusable process lesson

Per-seed windows: `../results/persistent_intrinsic_control_submission_20260908/learning_audit_20260908.json`.
Spatial summaries: `../results/persistent_intrinsic_control_submission_20260908/learning_audit_100m_spatial.csv`.
Study schema and workflow remain intrmotiv/study/v1 and 1.5.0; both unchanged
StudySpec hashes are in the JSON. Run discovery used the canonical package.

The canonical spatial collector completed without new environment rollouts.
The threaded full TensorBoard collector remained CPU-bound after about twelve
minutes; it was stopped after a bounded four-process diagnostic reader had
completed all 36 runs and the required additional time windows. Repeated full
event reads are a recurring infrastructure inefficiency. A future bounded
improvement should extract all requested windows/tags in one pass and compare
results with canonical means, including restart behavior. Also, scheduler
RUNNING must be paired with log/checkpoint freshness and node responsiveness.
No infrastructure agent was dispatched in this evaluation.
