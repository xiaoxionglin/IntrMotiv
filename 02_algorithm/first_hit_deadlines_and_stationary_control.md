# Reaching a landmark versus meeting a deadline

The [first-hit study](../06_experiments/controllers/fixed_goal_first_hit_20261127/README.md) distinguishes two requirements: eventually reaching a commanded DG event, and reaching it before a fixed deadline. This matters for minimality. A stationary shortest-path policy is sufficient in a fully observed deterministic system; adding a clock is not intrinsically necessary for that policy. The objective and approximation error can nevertheless make a deadline-conditioned learner behave differently.

These statements require a frozen representation sufficient for controlled transitions, a fixed goal detector, and adequate factual action support. Physical determinism alone is insufficient if different relevant physical states alias in the actor's input. The [initial theory](fixed_goal_first_hit_control.md) and [projected-target derivation](../06_experiments/controllers/fixed_goal_first_hit_20261127/runtime/projected_bootstrap_design.md) state the corresponding learning and support limits.

## A sufficient finite-deadline error margin

For a deterministic represented system and a noninitial goal query, the optimal probability of arriving within $h$ decisions is binary: $Q_h^*(x,a,g)\in\{0,1\}$. Suppose, on every executable query and legal action,

$$
|\widehat Q_h(x,a,g)-Q_h^*(x,a,g)|\le\epsilon<\frac12.
$$

If the goal is reachable within $h$, at least one action has value one. Its prediction is at least $1-\epsilon$, whereas each value-zero action predicts at most $\epsilon$. Greedy selection therefore chooses a viable action. That action either reaches the goal or leaves a successor from which the goal remains reachable within $h-1$. Induction with a **decrementing deadline** proves arrival by the initial deadline. At $h=1$, every viable action must hit immediately.

Ties among value-one actions may consume slack, but cannot prevent arrival under this uniform bound. They do not guarantee shortest paths. Resetting the deadline on every step breaks the induction. Average replay MSE does not establish the uniform hypothesis, and stochastic or aliased-state values need not have a binary gap.

## Stationary values have a different discrimination problem

With a unit first-arrival reward and discount $\gamma<1$, a shortest path of $D$ decisions has value $\gamma^{D-1}$. A path requiring at least one extra decision has value at most $\gamma^D$. A sufficient error bound for preserving shortest-action selection is therefore

$$
\epsilon<\frac12(1-\gamma)\gamma^{D-1}.
$$

For a uniform statement, use the largest relevant shortest distance. This condition is conservative and **not necessary for eventual arrival**: a longer successful route still reaches the requested landmark. Comparing it with the finite-deadline margin compares different prediction objectives. Finite-deadline values also require accuracy across the additional time coordinate.

The projected stationary operator preserves the exact discounted fixed point while keeping bootstrap targets in their analytically valid range. It does not ensure accurate raw network outputs, convergence of nonlinear regression, or a proper learned policy. A low stale-target loss is not a certificate for any of these properties.

For the finite toy systems, the complete physical state, fixed command and finite CA3 shift register form a deterministic finite system. Repeating that **complete** state before a goal hit proves perpetual nonarrival under fixed weights. Repeating a physical node while CA3 changes does not. The [fixed-weight cycle audit](../06_experiments/controllers/fixed_goal_first_hit_20261127/stationary_cycle_audit_r1/README.md) uses exact equality, preserves the six-step outcomes, and labels an exhausted computational cap unknown. It establishes reaching or cycling only for its enumerated starts and weights.

## When time affects the optimal policy

A stationary shortest-path policy succeeds for every feasible deadline in a deterministic system. At an infeasible deadline, every policy has success probability zero. Thus a clock-free policy can be deadline-optimal even though a clock-free value predictor cannot represent the changing success probabilities.

Stochastic dynamics provide a different case. At the same start and goal, suppose a fast action arrives in one decision with probability $p\in(0,1)$ and otherwise terminates unsuccessfully; a safe action arrives surely in two decisions. Fast uniquely wins at $h=1$, while safe uniquely wins at $h=2$. A policy receiving identical inputs without the remaining time cannot optimize both queries. Discounted stationary control instead makes a fixed probability–speed tradeoff.

The current finite-horizon head is affine in positive remaining time for a fixed state, goal and action. It cannot represent arbitrary first-hit curves. In a deterministic system with minimum required action count $d$, the integer-deadline curve has the ReLU representation

$$
\mathbf 1[h\ge d]=\operatorname{ReLU}(h-d+1)-\operatorname{ReLU}(h-d).
$$

Passing remaining time before a nonlinear decoder can remove that specific restriction, but computing different thresholds across all histories, goals and actions still requires shared capacity. This observation neither proves that an additional clock is needed for the user's reaching task nor makes the existing decoder uniformly accurate. The native saved-target audit also attributes only part of its observed residual to the per-query affine restriction; its targets are self-generated, not optimal-value ground truth.

The current minimal native follow-up therefore keeps the original network and tests projected stationary first-hit learning. Its [declaration](../hpc_runs/studies/stationary_projected_c05_s99_20261127.analysis.json) preserves all prior deadline failures and fixes the unchanged native 256-decision evaluation budget before candidate fitting.
