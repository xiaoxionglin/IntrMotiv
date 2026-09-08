# Persistent intrinsic control: interim learning audit

Audit: 8 September 2026, 17:51–18:05 CEST. Primary runs are approximately
82–122M/600M frames; C15 continuations approximately 177–191M/600M. This is an
interim evaluation of existing telemetry, not a final controlled evaluation.

## Operational health

Slurm listed all 36 jobs as RUNNING, but only 35 had fresh progress.
`PIC_G_CONTEXT_S99`, job 8002740, stopped logging at 17:18:37 on n3113.
The node reports `MIXED+NOT_RESPONDING`; sstat reports communication failure.
The latest checkpoint is `checkpoint_000005502_90144768.pth`, saved at 17:18.
This is an infrastructure incident, not evidence against the context design.
The C15 hotfix recoveries and W_REF_JOINT S123 are advancing. No training jobs
were cancelled or changed during this audit.

## Comparable learning metrics

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

## DG and control findings

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

## Recommended decisions

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

## Evidence and reusable process lesson

Per-seed windows: `results/persistent_intrinsic_control_submission_20260908/learning_audit_20260908.json`.
Spatial summaries: `results/persistent_intrinsic_control_submission_20260908/learning_audit_100m_spatial.csv`.
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
