# DG-capacity run health — 13 September 2026

Scope: the original 27-run DG-capacity/goal-conditioning study, checked around
13:55 CEST. The newer controller studies are separate. No training was changed
or stopped.

## Operational health

All 27 original Slurm jobs are RUNNING. All 27 learner logs were fresh within
five seconds and every frame counter advanced across its bounded log tail.
Latest five-minute throughput was approximately 1,166–1,222 frames/s for direct
runs and 500–1,111 frames/s for waypoint runs. Direct runs were at 265–283M;
waypoint DG64 was at 99–133M. Intermittent ten-second zero FPS is a reporting
window effect and does not establish a stalled process. Extrinsic episode
reward is zero by design in this no-reward environment.

## Behavioral warning signs

Latest W&B summary values are individual logged windows, not matched trials or
pooled success estimates. Stationarity means translation of at most one DMLab
unit per valid decision; rotation may continue.

| Design | Seed | Frames | Stationary | Option success | Readout logit sensitivity |
|---|---:|---:|---:|---:|---:|
| Direct DG64, DG + worker conditioning | 8 | 273M | 94.2% | 0% | 0.0002 |
| Direct DG64, DG + worker conditioning | 99 | 277M | 92.1% | 0% | 0.0001 |
| Direct DG64, DG + worker conditioning | 123 | 265M | 8.9% | 0% | approximately zero |
| Direct DG64, worker conditioning | 8 | 278M | 60.9% | 0% | 0.0002 |
| Direct DG64, worker conditioning | 99 | 278M | 35.8% | 0% | approximately zero |
| Direct DG64, worker conditioning | 123 | 275M | 9.3% | 16.7% | 0.0147 |

The two DG-conditioned seeds 8/99 are the clearest current movement-collapse
candidates. Mean translation is only 0.77 and 1.07 units per decision. The
DG-conditioned seed 123 still moves, but its goal-control diagnostics are poor.
Worker-only seeds 8/99 also show control failure, with less extreme immobility.
Partial DG silence is present, but none of these is an entirely silent network.
Readout sensitivity changes only the decoder command with memory held fixed;
it does not measure the full goal-conditioned DG write path.

The canonical snapshot collection now contains 105 snapshots: all 27 at 5M,
25M and 75M, plus 24 at 150M. The three DG64 waypoint runs have not reached
150M. At 150M, DG-conditioned DG64 seeds 8/99 had 27.6%/64.9% stationary
samples, compared with 94.2%/92.1% in their latest shorter windows. This supports
worsening movement, although the windows differ in length and policy behavior.

DG64 waypoint remains mobile: latest stationary fractions are 1.8%, 1.7%,
and 6.7% for seeds 8, 99, and 123, with movement around 18 units/decision.
It should not be called dead because training is slower. Waypoint DG32 seeds
99/123 remain movement-constrained (35.7%/45.7% stationary), but report nonzero
option success; seed 123 is less stationary than its 75M snapshot (68.4%).
A low policy entropy by itself also does not imply failure: waypoint DG16
seed 123 has entropy about 0.009 while remaining highly mobile.

## Persistence in sampled history

W&B histories were sampled at up to 400 records per run; these are descriptive
samples, not exhaustive event counts or a statistically weighted estimate.
Among records after 240M frames, observed option-success means were:

- Direct DG64 with DG conditioning, seed 8: 0.15% across 23 records.
- Direct DG64 with DG conditioning, seed 99: 0% across 25 records.
- Direct DG64 with DG conditioning, seed 123: 0% across 9 records.
- Worker-only DG64, seeds 8/99: 0% across 31/22 records.
- Worker-only DG64, seed 123: 9.1% across 30 records.

Thus five of the six direct DG64 runs have sustained near-zero option success
in the sampled late-training records. This is stronger evidence of control
failure than the individual latest zeros. It does not show that all five have
stopped moving, or that no small useful skill survives in any saved checkpoint.

## Evidence and limitations

- [Latest run metrics](data/dgc_health_20260913/latest_health.csv).
- [Bounded learner log evidence](data/dgc_health_20260913/log_health.json).
- [Sampled W&B histories](data/dgc_health_20260913/dgc_history.jsonl).
- [Current W&B summaries](data/dgc_health_20260913/dgc_wandb_health.jsonl).
- [All snapshot metrics](data/dgc_health_20260913/per_snapshot.csv).

This is a health assessment, not a demonstration of downstream transfer or
causal goal control. Useful control over a small, meaningful landmark set is
sufficient; global pair coverage is not a failure criterion.

## Collection provenance and reusable lesson

The study schema is `intrmotiv/study/v1`, collector workflow `1.7.1`, and
study SHA-256 is
`d09a11d735806584c3bc68f91b24d19e9cf5349d3d900bf4a981cda2cd06c780`.
The [canonical manifest](data/dgc_health_20260913/analysis_manifest.json)
records the complete discovery provenance.

The shared NEMO2 canonical collector is 1.7.1. Its historical target whitelist
incorrectly rejects the study's declared 150M snapshots. The analysis adapter
changes target discovery in memory only to the exact
`--online_spatial_snapshot_targets` training declaration, then calls the
canonical collector with all its other validation and calculations intact.
No live source or study file was modified. The adapter and override record are
saved in the data directory. A general follow-up should make target validation
honor declared positive checkpoint targets and test 150M/300M snapshots.

For future health checks, start with bounded learner log tails and W&B summaries;
use sampled history for persistence and compact snapshots for spatial behavior.
Avoid repeating a multi-gigabyte TensorBoard scan for a status question. W&B
names include a timestamp suffix; match exact StudySpec names with that known
wrapper, rather than assuming the configured group equals the study ID.
