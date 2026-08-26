# DG Structural Diversity Batch

**Date:** 2026-08-26  
**W&B project:** `SF_IntrMotiv_DGStructuralDiversity`  
**Source batch:** `intrmotiv_dg_structural_diversity_20260826`

## Objective

Increase DG place-field diversity without changing the frozen ImageNet
`layer2_resnet18` visual trunk or adding a learned planning module. The batch
tests two lightweight additions to the best conditions from the preceding
encourage/regularizer iteration.

## Additions

### CA3 temporal exclusion

The incoming CA3 register supplies an exact replay-stable history mask. For
current DG unit `j`, another unit in slot `R-1` means a different DG was active
within the preceding `R` policy decisions:

```text
h[t,j] = 1[sum_{i != j} 1[CA3_in[t,i,R-1] > 0] > 0]
a[t,j] = relu(z[t,j] - theta)
L_ca3 = lambda_ca3 * mean_valid,t,j[h[t,j] * a[t,j]]
```

The history mask is detached. The loss is active-only: it does not suppress a
below-threshold unit. Terminal CA3 reset prevents cross-episode conflicts.

### Orthogonal DG recruitment

When a lone source pulse first reaches the CA3 tail after `L` policy decisions,
the learner tests the endpoint. A different intervening DG rejects the event;
same-source reactivation does not restart its timer. At a DG-silent endpoint,
the least-used row that has not already been recruited is replaced with the
component of the current fixed visual feature outside the span of all other DG
rows:

```text
r = x - projection_span(W_without_j)(x)
w_j <- r / ||r||
```

The learner applies at most one replacement after PPO per accepted rollout.
Each row is recruited at most once, so structural mutation is bounded by
`F=16`. Optimizer moments for that row are cleared and BatchNorm is calibrated
to put the assignment observation `0.5` above threshold. Recruitment state is
checkpointed and synchronized with the model; old checkpoints load with empty
state.

## Production Grid

The intended design is a fully replicated `2 x 2 x 2 x 2` factorial with
seeds 8, 99, and 123, for 48 jobs:

| Factor | Values |
| --- | --- |
| Architecture | flat encourage; fixed/global HRL with `hit_distance` |
| Existing regularizers | none; global punishment 0.01 + row repulsion 1.0 |
| CA3 exclusion | off; one coefficient selected by preflight |
| Orthogonal recruitment | off; on |

All conditions retain threshold 2.43, `F=16`, `R=8`, `L=64`, simultaneous
encoder/decoder updates, batch usage loss, no PBT, one policy, 32 workers x 2
environments, and 100M environment frames. The full factorial distinguishes
each new mechanism, their interaction, compatibility with the previous best
regularizers, and dependence on HRL.

## Preflight Record

First wave, 2M frames each: jobs `7869203`-`7869207`. It tests exclusion
coefficients 0.03, 0.1, and 0.3, recruitment alone, and 0.1 plus recruitment.
All jobs started and emitted the new TensorBoard/W&B metrics. At approximately
0.6M frames, recruitment runs had nonzero L-step candidates and had recruited
12-15 rows without runtime errors. The weighted exclusion terms were small
(`~0.001`-`0.018`) compared with total encoder-loss magnitudes (`~1`-`7`).

Second wave, 2M frames each: jobs `7869219`-`7869222`. It uses the corrected
same-source timer and tests exclusion coefficients 1.0 and 3.0, recruitment
alone, and 1.0 plus recruitment.

The production coefficient and Slurm job IDs will be recorded here after the
multi-episode preflights finish and their final diagnostics are checked.

## Primary Analysis

Online panels:

- coverage AUC and unique cells;
- DG density, multi-activation fraction, duty-cycle entropy, and silent units;
- CA3 conflict fraction/activity and weighted exclusion loss;
- recruitment candidates, silent endpoints, cumulative assignments, and tiny
  residual skips;
- for HRL, option success fraction, target hits, timeouts, known edges, and
  controllability time.

Checkpoint telemetry should use at least 10,000 policy decisions and report
peak-bin diversity, peak-bin entropy, pairwise peak distance, map cosine
redundancy, spatial information, and silent units. Because stochastic telemetry
is trajectory-conditioned, compare all three seeds and matched architecture /
regularizer controls rather than selecting a single favorable map.
