# Exact-cycle audit of the fixed stationary controllers

Every projected controller reaches every declared goal in this audit: 138/138 noninitial queries across the fork and sparse tasks. Its five earlier fork misses are delays of seven or eight decisions. The original unprojected controllers reach 134/138 queries; four enter exact cycles that prove perpetual nonarrival under those fixed deterministic policies. **All original six-step results and failed gates remain unchanged.** This is an explicitly post-hoc fixed-weight diagnostic, not a new fit, a selected larger deadline, or native-environment qualification.

## Complete results

Counts exclude initially satisfied commands. Maximum hit latency is over the hits in that row; cycles are counted separately and never assigned a finite latency. All twelve rows have zero non-goal physical terminations and zero cap-unknown outcomes.

| Controller | Task | Seed | Original H6 hits | Audit hits | Proven cycles | Maximum hit latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Unprojected | Fork | 8 | 39/42 | 40/42 | 2 | 7 |
| Unprojected | Fork | 99 | 34/42 | 42/42 | 0 | 32 |
| Unprojected | Fork | 123 | 41/42 | 42/42 | 0 | 7 |
| Unprojected | Sparse memory | 8 | 3/4 | 3/4 | 1 | 6 |
| Unprojected | Sparse memory | 99 | 3/4 | 3/4 | 1 | 1 |
| Unprojected | Sparse memory | 123 | 4/4 | 4/4 | 0 | 6 |
| Projected | Fork | 8 | 40/42 | 42/42 | 0 | 7 |
| Projected | Fork | 99 | 41/42 | 42/42 | 0 | 8 |
| Projected | Fork | 123 | 40/42 | 42/42 | 0 | 8 |
| Projected | Sparse memory | 8 | 4/4 | 4/4 | 0 | 1 |
| Projected | Sparse memory | 99 | 4/4 | 4/4 | 0 | 1 |
| Projected | Sparse memory | 123 | 4/4 | 4/4 | 0 | 1 |

The full denominator is 318 queries: 276 noninitial plus 42 initially satisfied fork commands, all of which stop at time zero. The complete result is 314 hits and four proven cycles. Every declared query resolves before the fixed 100,000-decision cap; no outcome remains unknown. These query counts are repeated evaluations of twelve weights, not independent training replications.

## What establishes nonarrival

For a fixed command $g$, the augmented state is $x_t=(s_t,M_t,g)$: exact physical node, complete CA3 register after the current observation, and command identity. The original MLP and raw action head are fixed and deterministic, the physical transition is deterministic, and the CA3 update is the original length-eight, repeat-one finite-alphabet shift register. Consequently $x_{t+1}=F_g(x_t)$. If $x_t=x_\mu$ before acting, the future repeats with period $t-\mu$. With no earlier goal hit, this recurrence proves that this trajectory never reaches the goal. Repeated physical nodes alone do not establish recurrence because CA3 age/history can change the next action.

The four unprojected cycle witnesses are:

| Task | Seed | Start → goal | Shortest physical distance | Cycle preperiod | Cycle period |
| --- | ---: | --- | ---: | ---: | ---: |
| Fork | 8 | 4 → 2 | 1 | 8 | 4 |
| Fork | 8 | 6 → 2 | 2 | 7 | 4 |
| Sparse memory | 8 | 2 → 5 | 1 | 7 | 1 |
| Sparse memory | 99 | 2 → 4 | 1 | 7 | 1 |

The sparse cycles occur after the original context cue has shifted out of memory. Each query record retains the entire state/action path, repeated physical node, exact repeated CA3 bytes with dtype/shape/hash, preperiod and period. No tolerance, discretization or approximate state match is used.

The five projected fork misses at H6 instead terminate in a hit: seed 8, $5\to6$ and $6\to5$, both at step seven; seed 99, $6\to5$, at step eight; seed 123, $2\to5$ at step eight and $4\to5$ at step seven. The unchanged six-step gate remains a failure, while this distinct audit establishes eventual arrival on those fixed starts. It does not establish shortest routing or successful control from arbitrary memory histories.

## Fixed protocol and integrity

The [protocol](../runtime/stationary_cycle_audit_protocol.json) was written at 2026-11-27 10:50:32 UTC, before implementation or actual checkpoint rollouts, and reviewed before execution. It prespecified all twelve weights and 318 queries, the exact recurrence criterion, and a 100,000-decision cap. A capped query would be **unknown**, never proven failure. Hits take precedence over simultaneous physical termination, recurrence and the cap. No remaining clock is passed to the model.

The [prepared receipt](prepared.json) archives every checkpoint, original trajectory table and relevant source before restoration. Every original H6 outcome and shared trajectory prefix matches the new audit. Every model tensor hash is unchanged, as are all archived/live sources and checkpoint/trajectory inputs. The saved per-run `queries.csv` and `summary.json` include all queries, physical shortest distances, event latencies, cycle witnesses, per-goal denominators and checksums.

- Protocol SHA-256: `c7d4664b2864276bd5496154efabf30cd695ad8b0f08d458024a9be3096ce301`.
- Prepared receipt SHA-256: `94fd961b949489bd698eeda5fa1af462012a3d1db0cb34a8f15405eb57aa9bf8`.
- [Results](results.json) SHA-256: `328a3e13a75beec6ec9647593511f9c411e205c143fdee6acd165b0983ddef30`.
- Eight synthetic tests cover history-sensitive recurrence, full-history cycles, hit precedence, physical termination, unresolved caps, deterministic-state assumptions and archive-before-restoration ordering.

The [audit implementation](../../../../hpc_runs/first_hit_control/stationary_cycle_audit.py) reuses the original environments, evaluation starts, CA3 update and readout classes. No training, new replay collection, grid extension or native rollout was run. The reusable distinction is between a fixed-deadline miss, a later observed hit, and an exact deterministic cycle; none should be substituted for the others. These synthetic finite-state findings do not certify learned visual representations, stochastic dynamics, unseen starts or DMLab performance.
