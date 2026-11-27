# Projected stationary targets: post-hoc fork and memory regressions

Projection recovers all four sparse-memory queries in every seed, but the seven-goal fork still misses queries in every seed. The declared **all-six every-pair gate fails**. This is the explicitly post-hoc regression specified in the [fixed protocol](../runtime/projected_regression_protocol.json), after the separate fresh-grid comparison. It is not independent validation, a hyperparameter selection exercise, or DMLab evidence.

The six new fits reuse the exact earlier 4096-transition replay files, original MLP/action-head classes, CA3 updates, initial-seed construction and sampled-batch sequence. The sole objective change from the [existing stationary references](../stationary_original_head/results.json) is clipping target-network action values to $[0,1]$ before multiplying by $\gamma=.99$. The deployed raw output, greedy actor, goal input and parameter count are unchanged. No old reference was retrained or overwritten.

## Complete outcomes

Arrival and shortest arrival count noninitial queries. Worst-goal arrival is over six starts on the fork or two on the sparse task. FIRST is the competing-target ordering contrast over 210 fork comparisons or four sparse comparisons. The fork also has seven analytically satisfied initial-goal cases, counted separately. Sparse evaluation uses the four declared cue-conditioned starts and has no initial-goal case.

| Task | Seed | Existing raw arrival | Projected arrival | Worst goal | Projected shortest | Projected FIRST |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fork | 8 | 39/42 | 40/42 | 5/6 | 34/42 | 0.333333 |
| Fork | 99 | 34/42 | 41/42 | 5/6 | 31/42 | 0.338095 |
| Fork | 123 | 41/42 | 40/42 | 4/6 | 37/42 | 0.319048 |
| Sparse memory | 8 | 3/4 | 4/4 | 2/2 | 4/4 | 1.000000 |
| Sparse memory | 99 | 3/4 | 4/4 | 2/2 | 4/4 | 1.000000 |
| Sparse memory | 123 | 4/4 | 4/4 | 2/2 | 4/4 | 1.000000 |

The unchanged evaluation guard is six decisions; it is not a network input. The five failed fork queries are seed 8: $5\to6$ and $6\to5$; seed 99: $6\to5$; seed 123: $2\to5$ and $4\to5$. Saved paths contain waits or detours and do not reach the requested goal before the guard. No longer rollout was run, so these misses establish bounded-evaluation failure, not perpetual nonreachability. Projection improves some outcomes and worsens seed 123 relative to its raw reference; it does not uniformly dominate it.

On the sparse task, removing CA3 history from the same frozen projected network yields 1/4, 0/4 and 0/4 arrivals, with FIRST zero in all seeds. The full model has FIRST one and shortest arrival on all twelve declared queries. This supports use of the recent context cue on those starts. The finite CA3 history can still forget older cues; the earlier seed-8 replay contains one expired-cue row, and these results do not establish control sufficiency at arbitrary histories.

## Fresh residuals and ranges

Each row below uses all usable factual replay action-goal labels with the **same final frozen model** on both sides of the projected operator. These are fresh residuals, not the recorded stale-target training loss. Every fit also saves the unprojected operator diagnostic for the same final model.

| Task | Seed | Raw prediction range | Projected target range | Fresh MSE | Maximum absolute residual |
| --- | --- | --- | --- | ---: | ---: |
| Fork | 8 | [0.920688, 1.008449] | [0.927811, 1] | 1.12626e-5 | 0.018998 |
| Fork | 99 | [0.923616, 1.015318] | [0.925346, 1] | 1.40781e-5 | 0.024357 |
| Fork | 123 | [0.913976, 1.014702] | [0.919573, 1] | 1.32021e-5 | 0.022464 |
| Sparse memory | 8 | [−0.002410, 1.027027] | [0, 1] | 2.05152e-6 | 0.104889 |
| Sparse memory | 99 | [−0.000009, 1.000020] | [0, 1] | 3.72073e-11 | 0.000325 |
| Sparse memory | 123 | [−0.000820, 1.000186] | [0, 1] | 1.59087e-8 | 0.003169 |

Targets are bounded while raw predictions remain unbounded. Maximum value errors on the declared fresh-start action queries are 0.02040, 0.01719 and 0.02009 on the fork, and 0.0002345, 0.00000154 and 0.00002252 on sparse starts. These finite-start calibration diagnostics and empirical replay residuals are not uniform guarantees over all represented histories. In particular, the fork errors exceed the sufficient shortest-action bound of approximately 0.004755 at distance six.

## Provenance and matching

- The [prepared receipt](prepared.json), including exact source snapshots and replay hashes, was written before all six fits. The protocol SHA-256 is `addba931cb6918beb03531b0682735f78eebd1016169bd72f327e9452e43b3f0` and prepared receipt SHA-256 is `b3b88d37f1d01b5571599565245fe6b9e89d3da9014034943255f674fc958895`.
- Actual new initial tensors, initial parameter hashes, sampled-row hashes and usable-label-mask hashes are retained. They agree with reconstruction from the unchanged historical classes, configuration, data and seeds. **Historical matching is reconstructed, not comparison with measured old initialization/batch hashes**; it assumes unchanged library RNG semantics.
- Each fork fit has 25,091 parameters and 3,840,000 usable goal labels. Each sparse fit has 21,635 parameters and 1,280,000 usable labels. Label counts exactly match each corresponding existing raw fit. Both tasks retain 2500 updates, batch 256, Adam learning rate $.003$, target refresh every 100 updates and $\gamma=.99$.
- All six saved final checkpoints exactly reproduce every evaluation trace. Evaluation leaves model hashes unchanged. Archived and live sources, replay bytes and historical result bytes remain unchanged. Four focused adapter tests passed before the six fits; all six together took 78.3 seconds on one numerical thread.
- No fit was repeated, extended or selected after outcomes. No native environment, new replay collection or additional hyperparameter was introduced.

The authoritative [results](results.json) have SHA-256 `b00c51fc487de68fd4ac03c267c23b235bfcb3b1569363221e73f0af11a392b0`. Each task/seed folder retains initial and final weights, all trajectory/control rows, per-goal metrics, competing-FIRST rows, calibration and fresh residuals. The thin [adapter](../../../../hpc_runs/first_hit_control/projected_regression.py) reuses the frozen trainer and projection helper.

The reusable lesson is to distinguish an improved objective target from an adequately fitted controller. Target projection addresses invalid bootstrap values without adding architecture, and it resolves the declared sparse-memory failures here; it does not by itself ensure sufficiently accurate action ordering on the fork. These regressions retain that limitation rather than treating mean improvement or the fresh-grid successes as a universal qualification.
