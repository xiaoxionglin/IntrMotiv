# Prospective value-range projection after the stationary-head failure

Design finalized: 2026-11-27 10:20:05 UTC. This note is written after observing the failed stationary-head protocol and its frozen-weight residual audit, but before any projected-target implementation tests, fits or policy trials. It records a **post-failure hypothesis**, not a result or a correction to the original preregistration. The immutable [stationary design](minimal_stationary_head_design.md), its failed runs, and the ongoing finite-horizon DMLab study remain unchanged.

**Recommendation:** a projection of bootstrap targets onto their analytically known value range is a principled, smaller next candidate. It adds no inference parameters, clock or action head. It removes propagation of impossible target values, but it does not by itself repair an inaccurate neural regressor, missing action support, or insufficient memory. Test it first against the unprojected stationary objective on a fresh, fixed GeoMem navigation task. Do not tune on the failed evaluation queries.

## What the observed failure establishes

The [final replay residual audit](../stationary_original_head/final_replay_residual/results.json), SHA-256 `8754a8a5d80955a02cd636d099f15bfd9cc15c5941f7fc0e02c1ab055997961a`, holds sparse-context seed 99 weights and all 4096 replay rows fixed. Its recorded last stale-target minibatch MSE is $1.1773\times10^{-10}$, whereas recomputing targets with the same final model yields factual all-goal MSE $1.4888\times10^{-6}$ and maximum absolute residual $0.0033112$. There are 938 predictions above one among 8192 factual goal-action predictions. A small loss against a lagged target therefore did not certify a fixed point or successful control.

At the failed query, the wait value is $1.010705829$. Waiting changes the CA3 cue age: the true recorded successor maximum is $1.024110436$, so the fresh unprojected target is $1.013869286$ and the signed residual is $-0.003163457$. **The physical-node self-loop is not a self-loop in represented CA3 state.** The shortcut calculation $Q(x,\mathrm{wait})-\gamma Q(x,\mathrm{wait})$ is invalid for this query. With the proposed projection, this particular successor maximum would instead give a wait target of $.99$. This is an arithmetic consequence of the proposal, not evidence that a newly fitted model will choose correctly.

## Exact projected operator

Keep the original raw action-head outputs as the value approximator. Define the scalar projection $C(v)=\min\{1,\max\{0,v\}\}$ and, for a current non-goal state,

$$
(T_C Q)(x,a,g)=\mathbb E\left[y_g(x')+\gamma(1-y_g(x'))(1-d_{\rm phys})C\left(\max_b Q(x',b,g)\right)\mid x,a\right].
$$

The assumptions are the same as in the original stationary theorem: frozen, controlled-Markov-sufficient representation, goal-independent physical dynamics, stationary transition/termination law, and the unchanged target detector. Initial already-satisfied commands stop outside the equation. Success takes precedence over physical failure. Missing terminal successors remain unusable labels; a buffer cut is not a failure.

The absorbing unit-arrival return lies in $[0,1]$, hence $Q^*\in[0,1]$ and $C(Q^*)=Q^*$. Projection is monotone and one-Lipschitz, and $C(\max_b Q_b)=\max_b C(Q_b)$. For bounded value functions,

$$
\|T_C Q-T_C R\|_\infty\le\gamma\|Q-R\|_\infty.
$$

Since $T_CQ^*=TQ^*=Q^*$, contraction proves that the projected operator has the **same unique fixed point**. Exact iterations satisfy

$$
\|T_C^kQ_0-Q^*\|_\infty\le\gamma^k\|Q_0-Q^*\|_\infty.
$$

In fact $T_C$ maps any bounded input into $[0,1]$ after one exact backup. A deterministic immediate-hit transition has target one; a non-hit continuing transition has target at most $\gamma$; a known terminal failure has target zero. The endpoints zero and one come from the reward/absorption definition, not benchmark tuning. For a different reward scale or continuing-reward objective, these bounds would require a new derivation.

The order matters: apply $C$ **before** multiplying by $\gamma$. Clipping $\gamma\max Q'$ to $[0,1]$ instead permits non-hit targets equal to one and loses the guaranteed $.01$ separation from immediate arrival at $\gamma=.99$.

For the existing Double-Q implementation, the corresponding sampled target is

$$
y+\gamma(1-y)(1-d)C\left(Q_{\bar\theta}(x',\arg\max_bQ_\theta(x',b,g),g)\right).
$$

The target is detached. This can be implemented by passing $\gamma\,C(Q_{\bar\theta})$ into the existing absorbing-label helper; clipping commutes with gathering the selected target action. Online action selection still uses the original raw outputs. When online and target functions agree, this is the exact projected operator above. With different networks, the action-selection discontinuity means the pair of online/target functions is **not** covered by that contraction theorem. Neither the theorem nor the clipping operation proves neural SGD convergence.

## Why it addresses the proposed mechanism, and what remains

An independent analytic witness illustrates slow correction without claiming that it is the observed CA3 process. Consider one represented non-goal state, an immediate-goal action whose value is already one, and a true self-loop action. With exact regression against each successive lagged target, its unprojected values obey $q_{k+1}=\gamma\max(1,q_k)$. While $q_k>1$, an optimistic initial value decays only as $\gamma^k q_0$. At $.99$, 25 ideal target generations retain about $.7778$ of that optimism's multiplicative scale. Training MSE against each fixed target can be zero throughout. With target projection, the first backup of any $q_k>1$ is instead $q_{k+1}=\gamma$.

The actual sparse example has changing histories and approximate function fitting, so this scalar recurrence is a mechanism witness, not a fitted explanation of its trajectory. The observed audit supports investigating range propagation; it does not establish that insufficient target generations are the sole cause. Projection also provides no direct cure for the sparse task's expired-cue aliasing.

If approximate projected iterations satisfy $Q_{k+1}=T_CQ_k+e_k$ with uniform $\|e_k\|_\infty\le\eta$, then

$$
\|Q_k-Q^*\|_\infty\le\gamma^k\|Q_0-Q^*\|_\infty+\frac{\eta(1-\gamma^k)}{1-\gamma}.
$$

This is a conditional bound on function error, not a statement about minibatch MSE. A uniform fresh projected Bellman residual $\delta_C=\|T_C\widehat Q-\widehat Q\|_\infty$ gives $\|\widehat Q-Q^*\|_\infty\le\delta_C/(1-\gamma)$. The original deterministic action-gap condition still applies: at maximum shortest distance $D$, a sufficient approximation error is less than $\tfrac12(1-\gamma)\gamma^{D-1}$. Clipping targets alone does not bound the raw fitted output, and raw predictions can still exceed one or rank a wait action first. Unsupported actions, unseen histories, stochastic sampling, shared-parameter interference and optimization error remain separate failure sources.

At a deterministically observed direct-hit versus non-hit decision, target projection supplies labels one versus at most $\gamma$. If the two fitted action predictions are each within $\epsilon<(1-\gamma)/2$ of their applicable labels, direct wins. This local statement requires those regression-error bounds and factual support; it does not follow merely from range-valid targets.

## Why not add a sigmoid or clip the actor instead?

Target projection keeps the exact deployed decoder, raw linear action-head algebra and greedy rule. It adds a bounded elementwise operation during target construction and its compute/storage should be recorded. The trained raw outputs are still unconstrained.

A sigmoid would change the value parameterization and its gradients. It bounds predictions, but exact zero and one require infinite logits and gradients can saturate. Although a strictly increasing sigmoid preserves the argmax of a fixed logit vector, training sigmoid values is a different intervention. It is not needed to enforce the known target range and should not be silently bundled with projection. Clipping raw values at action selection can create ties among overestimated actions and is also not the proposed repair. A categorical softmax is unsuitable because action values need not sum to one.

Thus this is a value-range constraint derived from the objective, rather than a sweep of clipping thresholds, discounts, budgets or spatial heuristics. The guarantee is about the Bellman operator; empirical usefulness still requires a matched test.

## Finite-horizon counterpart

For the existing $\gamma=1$ finite-deadline objective, keep $Q_0=0$ and replace the successor maximum by $C(\max_bQ_{h-1})$ in the same first-hit backup. Every exact $Q_h^*$ lies in $[0,1]$, so induction gives the same finite-horizon solution. The operator is nonexpansive, not a strict sup-norm contraction; uniqueness follows from the fixed $h=0$ boundary and the acyclic deadline recursion. With synchronous exact backups, all $H$ layers are determined after at most $H$ iterations from arbitrary bounded positive-deadline initialization.

This prevents propagation of impossible target values but does not remove the affine-clock approximation defect. Even bounded exact targets $[0,1,1,1]$ can have an affine least-squares fit ending at $1.2$. The current finite-horizon actor can therefore remain miscalibrated after target projection. Projection must be a separately declared future factor; do not alter the active DMLab study or silently relabel its results.

## Fresh fixed gate before revisiting failed queries

Proposed task: the canonical GeoMem `clamped_lattice(width=4, height=4)` in `/home/xiaoxiong/Desktop/Projects/geomem/geomem-env/src/geomem_env/environment.py`, inspected source SHA-256 `6354015e2960e9170c97f37851cce9124b6f49d97f2fde17301f84406b653015`. This constructor supplies sixteen nodes and four cardinal actions. Use the actual transition-only `Environment.step`; do not introduce `step_result` rewards. Boundary actions execute self-loops; retain those factual transitions and do not mask actions using the environment's privileged validity flag. Requested goals are external DG commands.

Use this separately trained fresh topology as a **design task**: it is neither policy transfer nor held-out-topology or unseen-goal generalization. All sixteen identities are relabeled during value learning. A distinct fixed synthetic DG one-hot observation at each node, the unchanged CA3 update with length eight and repeat one, original `[128,128]` decoder and original action-parameterization class provide a fully observed state-sufficiency positive control. These are synthetic observations, not learned visual fields. No coordinates, shortest distances, node IDs beyond that explicitly supplied DG observation, or transition-table values enter TD targets. Uniform reset to any node is an explicit simulator capability, not an ability claimed for native DMLab. This first gate isolates target range from memory ambiguity; it cannot qualify partial-observation control.

Before constructing data or fitting, freeze an executable manifest and source archive with:

| Item | Fixed proposal |
| --- | --- |
| Conditions | Unprojected stationary target versus the same target with $C$; no other model/optimizer difference |
| Seeds | 8, 99, 123; paired initialization, replay and minibatch rows between conditions |
| Reward-free replay | 4096 factual transitions per seed; uniformly random starts and four actions; reset collection segments every 32 decisions, with cuts labeled as truncations rather than physical failure |
| Learner | Original module classes, hidden widths `[128,128]`, all sixteen goals, current already-satisfied goals omitted identically in both conditions; only first-hit loss |
| Budget | $\gamma=.99$, 2500 Adam updates, batch 256, learning rate $.003$, target copy every 100 updates; no sweep, early stopping or chosen checkpoint |
| Evaluation | Frozen final weights; every 240 nontrivial start-goal pair from a declared fresh-current-observation CA3 prefix, plus sixteen separately counted initial-goal stops; external deadline sixteen, no clock supplied to the model |
| Success gate | At least $.9$ on every deterministic pair in every seed, which here requires every pair to succeed; additionally report per-goal/worst-goal success, shortest-arrival fraction and all 3360 ordered competing-target FIRST comparisons per seed/condition |
| Diagnostics | Node-action exposure, all-goal label counts, raw prediction/target ranges, fresh same-final-model residuals, independent deterministic shortest-path value error, saved-weight replay identity, compute and source/data hashes |

The task diameter is six, so the sixteen-decision evaluation budget permits every reachable pair. The expected shortest distances come from the public environment definition and are used only for evaluation diagnostics. Every query stays in the denominator; no selection of easier goals after viewing outcomes. The primary comparison is the paired projected-versus-unprojected candidate on this fresh task, with individual seed outcomes retained. If both pass, projection's practical benefit is not established by the pass alone; if projection fails, preserve that result without tuning this evaluation panel.

Any later rerun of the earlier fork/sparse tasks is an explicitly post-hoc regression check, cannot select this design, and cannot be recast as independent validation. This proposal does not authorize a new native rollout or modify any existing training module; implementation/fits require the next explicit task assignment.
