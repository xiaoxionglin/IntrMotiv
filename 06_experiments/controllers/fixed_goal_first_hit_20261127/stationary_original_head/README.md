# Original action head: stationary first-hit qualification

**The smaller candidate failed the fixed-budget every-pair gate.** Five of six runs missed a declared target. The existing modules support the interface, but these fits do not establish reliable control. No tuning, additional fits, native rollouts, or changes to the running DMLab study followed.

## Frozen experiment and results

The [prospective theory](../runtime/minimal_stationary_head_design.md) was archived before fitting. The actual `MlpDecoderJit` uses two 128-unit ReLU layers and `ActionParameterizationDefault`, with three raw outputs, no clock, and no auxiliary head. These are fresh toy weights; the [donor audit](../runtime/stationary_action_head_audit.json) separately certifies the native five-action head.

Each run reused its exact 4,096-row reward-free replay and sampled transition batches: 2,500 Adam updates, batch 256, learning rate .003, target refresh 100, and $\gamma=.99$. Evaluation held weights fixed and stopped on the requested goal, with an external six-decision limit.

| Task | Seed | Success | Shortest success | Competing FIRST | Maximum start-query Q error | Final sampled TD MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-seven fork | 8 | 39/42 | 32/42 | .2857 | .05226 | $1.34\times10^{-5}$ |
| All-seven fork | 99 | 34/42 | 31/42 | .3048 | .06095 | $1.29\times10^{-5}$ |
| All-seven fork | 123 | 41/42 | 39/42 | .3333 | .02450 | $1.50\times10^{-5}$ |
| Sparse history | 8 | 3/4 | 2/4 | .7500 | .23763 | $1.48\times10^{-7}$ |
| Sparse history | 99 | 3/4 | 3/4 | .7500 | .02071 | $1.18\times10^{-10}$ |
| Sparse history | 123 | 4/4 | 3/4 | 1.0000 | .09717 | $1.85\times10^{-11}$ |

Fork evaluation retains 42 nontrivial directed pairs and seven analytic time-zero stops; sparse evaluation has four fresh-cue queries. FIRST compares goal $g$ before competitor $c$ under both executed commands: 210 ordered comparisons per fork run and four per sparse run. Unavoidable corridor ordering can give zero contrast. Memory removal produced 0/4 sparse successes in each seed.

The saved finite-horizon baseline succeeded on every pair in all six runs. Its shortest-success counts were 21/42, 20/42, 20/42 on the fork and 0/4 on each sparse run; its FIRST means were .3333 and 1 respectively. Thus the stationary candidate often shortened successful routes while sacrificing complete goal reaching at this budget.

## Theory and comparison limits

Given a sufficient stationary Markov representation and identified action consequences, exact discounted absorbing backups converge; their greedy policy follows shortest deterministic routes. This does not establish nonlinear optimization, memory sufficiency, coverage, or representability. Under stochastic transitions the objective trades arrival probability against speed.

The sufficient uniform ranking-error threshold is $.005(.99)^{D-1}$: .004755 at distance six and .005 for the sparse one-step starts. All measured maxima exceed these thresholds. The bound is sufficient, not necessary; start-query errors are not uniform certificates over all CA3 histories.

This comparison also removes the earlier shared-decoder continuing loss and masks initially satisfied commands. Each fork run used 3.84 million of 4.48 million enumerated labels; each sparse run used all 1.28 million. Updates and batches match, but labels and computation do not. Parameters total 25,091 for the fork and 21,635 for sparse history, including the sole 387-parameter action head.

## Saved-path diagnostic

The [post-hoc audit](diagnostic/results.json) reconstructed CA3 along 14 saved failed paths; every recorded action reproduced. At the first strict wrong ranking, 12/14 exact CA3 inputs appeared in replay; 9/14 included both selected and shortest actions. Ten first errors selected a self-loop. These selected failures are not population error rates.

For sparse seed 99, the exact CA3 input occurred in 826 replay rows: 278 wait and 274 direct-success actions. Predicted wait/direct values were 1.010706/1.000007; exact values are .99/1. The recorded path waited six times despite final sampled training MSE $1.18\times10^{-10}$. Small sampled TD loss coexisted with wrong ranking; it is not a uniform residual bound. The audit records final-network one-step residuals separately from historical minibatch loss.

This observation does not isolate objective, capacity, optimization, or representation effects. Waiting overestimation also occurs without a neural clock.

## Provenance and reuse

[Pre-run manifest](pre_run_manifest.json) SHA-256: `7200ffd7206020d0602aaf0b3d817bfe0c53682f347ceae05ee17e411a629203`. [Results](results.json) SHA-256: `509a79b55315e700ff91b5acf917fe05da6fb7d8e1b95feea728500b04f6f25a`. Exact source snapshots, all six replay hashes, baseline traces, saved-weight reproduction, and frozen-state checks passed. Eight fitting contracts and three diagnostic contracts passed; diagnostic inputs and source bytes also remained unchanged.

Reuse replay, CA3, goal conditioning, event classification, and source snapshots; check action ranking and exact-history action support alongside regression loss. The optimizer is separate because the earlier loop unconditionally constructs two heads and trains their joint loss.
