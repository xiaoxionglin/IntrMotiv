# Prospective audit: stationary first-hit values in the original action head

Finalized 2026-11-27 09:57:25 UTC, after the finite-horizon study was authorized and before any stationary-head training or evaluation. This is a mathematical/source audit and a proposed next comparison; it does not change the running finite-horizon study or report a new experimental result.

**Recommendation:** test this strictly smaller candidate next on the saved graph replay. The current C05 already supplies the goal identity to a two-layer MLP and outputs one raw logit per action. Reinterpreting those same outputs as discounted absorbing first-hit values requires zero additional inference parameters, recurrent state, or clock input. The change is to the training objective and acting rule. It is a conditional sufficiency claim, not a proof that the existing finite-width network can learn every navigation problem.

## Objective and conditional theorem

Let $x$ be the frozen physical representation, $g$ a fixed learned DG identity, and $y_g(x')$ its original successor detector. The command stops immediately if already satisfied. On transitions from a non-goal state, use

$$
(T_g Q)(x,a)=\mathbb E\left[y_g(x')+\gamma(1-y_g(x'))(1-d_{\rm phys})\max_b Q(x',b,g)\mid x,a\right],\qquad \gamma=0.99.
$$

Success on a terminal successor takes precedence over failure. The corresponding value is

$$
Q_\gamma^*(x,a,g)=\sup_\pi\mathbb E\left[\gamma^{\tau_g-1}\mathbf 1\{\tau_g<\infty\text{ and arrival precedes physical failure}\}\mid x,a,\pi\right],
$$

where the first arrival transition has $\tau_g\ge1$. There is no continuing reward after the hit and no substitution of the successor's manager-selected goal. Initial already-at-goal queries stop outside this equation.

Assume the represented process is controlled-Markov sufficient for these goal detectors and transitions, physical dynamics are goal-independent, and the transition/termination law is stationary. For bounded $Q$, $T_g$ is a $\gamma$-contraction in the sup norm: the detector and failure multipliers have magnitude at most one, and taking a maximum is nonexpansive. Hence there is a unique fixed point and exact value iteration converges to it. This does not establish convergence of nonlinear replay training. Tabular Q-learning additionally needs adequate repeated sampling and step-size assumptions; its established theorem does not cover arbitrary neural approximators. [Watkins and Dayan, 1992](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).

For a **finite deterministic** represented graph and any reachable goal, let $d(x,g)$ be the shortest number of decisions to its detector set. If taking $a$ first permits a shortest total route of $D_a$ decisions, then

$$
Q_\gamma^*(x,a,g)=\gamma^{D_a-1},
$$

with value zero for an action whose successor cannot reach the goal before failure. Because $0<\gamma<1$, every exact greedy action minimizes $D_a$, decreases $d$ by one, and reaches the goal in $d(x,g)$ decisions. Ties between shortest actions are harmless. A simple shortest path uses at most $|X|-1$ decisions. A prescribed deadline $H$ is met whenever $d(x,g)\le H$, although the learned function has no deadline argument.

This proves that a new clock/head/world model is unnecessary for deterministic graph reachability **if** the existing input is sufficient, the relevant action consequences are identified, the head can represent the values, and learning reaches them. It does not prove any of those premises from successful exploration alone. A hidden-state alias with incompatible required actions or an unobserved necessary action still defeats the claim.

## Approximation tolerance

If $\|\widehat Q-Q^*\|_\infty\le\epsilon$, greedy selection loses at most $2\epsilon$ of optimal one-step action value. The standard recursive argument then gives

$$
\|V^*-V^{\pi_{\widehat Q}}\|_\infty\le \min\left\{1,\frac{2\epsilon}{1-\gamma}\right\}.
$$

For a deterministic state at shortest distance $d\ge1$, every strictly longer feasible action has a value gap at least $(1-\gamma)\gamma^{d-1}$; an unreachable action has a larger gap. Therefore, across a set of reachable states with maximum distance $D$, the sufficient uniform condition

$$
\epsilon<\tfrac12(1-\gamma)\gamma^{D-1}
$$

preserves shortest-action selection everywhere in that set. This bound is conservative, not necessary. For $\gamma=.99$, its right side is approximately $0.004755$, $0.002655$, $0.0003854$, and $2.180\times10^{-7}$ at $D=6,64,256,1000$. Long paths can consequently be hard to distinguish even when their values appear numerically close.

For a **true uniform Bellman residual** $\delta=\|T\widehat Q-\widehat Q\|_\infty$, contraction gives $\epsilon\le\delta/(1-\gamma)$ and the sufficient ranking condition $\delta<\tfrac12(1-\gamma)^2\gamma^{D-1}$. A sampled training or held-out mean-squared residual is not this quantity and cannot certify that condition.

## What changes in stochastic environments

The criterion values both probability and speed. A safe route reaching the goal certainly at decision 100 has value $.99^{99}\approx.36973$. A risky action reaching it immediately with probability $.9$ and otherwise terminating has value $.9$. This objective selects the risky action although its eventual success probability is lower. It is generally neither eventual-hit-probability optimal nor finite-$H$-hit-probability optimal. For each fixed policy the discounted return tends to eventual success probability as $\gamma\uparrow1$; fixed $.99$ supplies no probability-first guarantee.

Physical time limits also need care. Collector cuts and the evaluator's external deadline are not absorbing failures. An environment timeout depending on unrepresented remaining episode time breaks the stationary Markov premise. It must be treated under an explicitly justified truncation/bootstrap contract, or its relevant time state must already be represented; silently adding a clock would abandon this candidate's strict interface claim. Missing terminal successors remain unknown, and reset observations must never supply their detector labels.

## Exact source reuse

The donor is `COS27_C05_TEMPORAL_TEMPORAL_S99`, checkpoint 100,007,936, checkpoint SHA-256 `f35464c7e058df3ac51b676d5560470aa3cddd3c57f02ed1c0012a5fbcfaed38`, config SHA-256 `a18da40ae1fbbd428e8c61c8cefcb79827fd66fa3d1f8646ba7201369df563fd`.

The actual r3 replay certificate in [replay metadata](qualification_r3_results/replay/metadata.json) records `MlpDecoderJit`, five actions, input width 1165, CA3 slice `[0,1136)`, physical bypass `[1136,1149)`, and the existing 16-goal one-hot slice `[1149,1165)`. The earlier local 1162-wide FiLM fixture is not this donor. The original source decoder uses `[128,128]` ReLU layers. Its source SHA agrees between the local inspected copy and the r3 certificate.

The inspected `sample_factory/model/action_parameterization.py` defines `ActionParameterizationDefault.distribution_linear = nn.Linear(core_out_size, num_action_outputs)`. Its forward method returns the raw linear outputs separately from the categorical distribution. `ActorCritic.forward_tail` applies the decoder and this parameterization and stores the raw outputs as `action_logits`; sampling is a separate step. For this discrete five-action donor the original final layer has $5(128+1)=645$ parameters. The existing deadline auxiliary has $5(2\cdot128+2)=1290$ final-layer parameters, plus its clock interface. These are readout-only counts, not a whole-model memory comparison.

Local inspected source root: `/home/xiaoxiong/SFgit/SF_hipposlam`. SHA-256:

| Source file | SHA-256 |
| --- | --- |
| `sample_factory/model/action_parameterization.py` | `523d54adc1ead65475d12657527bcb631dced6b4c963657b4af9239ff403967a` |
| `sample_factory/model/actor_critic.py` | `66b18192aa741da7f33b1032ff2e90bcf71aabf6c90e3ce7254ec5436b0b9e5b` |
| `sf_working_directories/IntrMotiv/dmlab/custom_decoder.py` | `6f4b3f3d14cf6cdf4cc669f64d0b4bd3ba70de70e87fb4b39dc8245d97da2991` |

The prospective implementation should copy the actual donor decoder and `actor.action_parameterization`, preserve all their dimensions and weights initially, and enable gradients on those copies only. Existing `collector.readout_snapshot` already saves `decoder_state_dict`, `action_head_state_dict`, their class names, and decoder configuration. Reconstruct the certified original class and strictly load the original `distribution_linear.weight` and `.bias`; verify module/code and state-dict hashes at that boundary. This audit inspected the local action-parameterization source; a prospective remote implementation must additionally bind that file's donor-source identity rather than infer it from the decoder hash.

For each factual replay row, replace only the certified goal one-hot slice in both stored current and successor physical inputs with the same $g$. Use the original positive-argmax winner detector on **core slot-zero DG** (ties follow source argmax), not every positive DG or the raw encoder output. Regress the raw output for the factual action against the absorbing target for all goals, using a frozen target copy as required by the declared optimizer. Do not softmax the regression values. Goal enumeration is independent of the sampled successor, avoiding outcome-selected relabeling; it does not create missing action coverage. Neither this candidate nor the earlier all-goal fits establish unseen-goal generalization merely by evaluating all trained IDs.

At execution, freeze the fitted copy, query the fixed requested goal, and choose `argmax` of those same raw outputs. Keep the original actor/critic, DG weights and BatchNorm buffers, CA3 rule, graph, manager and detector unchanged. Training target networks add storage and compute, not deployed representational capacity. Original PPO logits have unconstrained absolute offsets; their old numeric values are not already calibrated first-hit values. Fixed-width representability, off-policy extrapolation, optimization stability and coverage remain empirical gates. Goal-conditioned value approximation itself is established prior work; the proposed contribution is an implementation-specific minimality test. [Schaul et al., 2015](https://proceedings.mlr.press/v37/schaul15.html).

## Affine deadline witness and its precise scope

At a fixed state and goal the current auxiliary is affine in positive remaining time because it is linear in `[hidden, hidden*clock, clock]`. Consider two states $s,g$, with actions `direct` from $s$ to $g$ and `wait` from $s$ to $s$. The exact finite-horizon direct values are all one for $h=1,\ldots,H$; wait values are $[0,1,\ldots,1]$. Independent least-squares projection of the latter onto $a+bh$ with uniform weight over these deadlines gives

$$
a=\frac{H-4}{H},\qquad b=\frac{6}{H(H+1)},\qquad \widehat Q_H(s,\mathrm{wait},g)=1+\frac{2(H-2)}{H(H+1)}>1\quad(H>2).
$$

For $H=4$, the projected wait values are $[.3,.6,.9,1.2]$, so the projected actor initially prefers waiting over direct arrival. **The true finite-horizon success values of both actions are tied at one for $h\ge2$.** This is overshoot and a spurious strict preference that can add latency; it is not a proof of worse deadline success. With a decreasing clock this example switches to direct in time. It is an exact-value regression witness, not a derived fixed point of the actual bootstrapped learner. A final-step-only sequence $[0,0,0,1]$ instead projects to endpoint $.7$ at $H=4$ and does not itself show overshoot above immediate success.

The stationary objective gives direct value one and wait value $.99$, with no deadline approximation. It removes this clock misspecification and supplies a strict shortest-path preference on deterministic graphs. Neither observation establishes the cause of any DMLab outcome.

## Proposed bounded next gate

Before a new fit, declare a separate stationary run using the exact saved all-seven-landmark fork replay and sparse-context replay, seeds 8/99/123, the original `[128,128]` decoder, original action-head class, $\gamma=.99$, and no deadline input. A reasonable initial budget is the preceding toy budget: 4096 factual rows per seed/task, 2500 updates, batch size 256, Adam learning rate $.003$, and target copy every 100 updates. Freeze that resolved configuration and source/data hashes before training; report target exposures, parameter counts and runtime explicitly. This is six fits, not a new sweep. The graph inputs/action counts necessarily use the graph task's dimensions, as in the earlier legacy toy, while retaining the exact source module classes and hidden widths. This is not a transfer of the 1165-input/five-output donor weights into differently sized graphs; production reuse would preserve those exact donor dimensions and initial tensors.

Train only the stationary absorbing first-hit objective. The earlier finite-horizon legacy trainer also jointly trained a synthetic continuing-value head through the same decoder. Removing that interfering objective is an additional difference: the comparison tests the complete smaller candidate and must not be described as an isolated clock ablation or a loss-matched causal estimate of clock removal.

Evaluate frozen policies on every predeclared start-goal query, shortest arrival length as well as success, the saved competing-target FIRST contrast, and the sparse memory-removal control. Compare success and latency to the existing finite-horizon artifacts while acknowledging the different objectives; do not equate their numeric values or call the optimizer comparison capacity-matched. Record saved-weight replay checks and report a failure without tuning the same evaluation set. Passing would support the smaller candidate on these tasks; it would not establish universal Markov sufficiency or full DMLab navigation. At this note's finalization no stationary fit or rollout has been performed. The already-authorized finite-horizon study remains unchanged.
