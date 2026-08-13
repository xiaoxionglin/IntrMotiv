# Transition-Distance Reward Problem Plan

Date: 2026-06-05

This plan focuses on the transition-distance reward as the main feature of IntrMotiv. The goal is not to replace it with auxiliary losses. The goal is to make the transition-distance mechanism strong enough that it can drive both exploration and DG landmark formation.

The key theoretical claim is:

```text
If DG units become stable landmarks, and if the decoder learns to move efficiently between those landmarks,
then transition time in the DG-CA3 sequence system becomes a useful internal approximation of geodesic distance.
```

That is the central mechanism to protect.



## Core Principle

The transition-distance reward should remain the central intrinsic motivation signal.

Auxiliary terms should only support the conditions under which transition-distance has the intended meaning:

- DG activations should be sparse and landmark-like.
- DG landmarks should be distributed across the environment.
- The decoder should learn trajectories between landmarks, preferably near-geodesic rather than circular or arbitrary.
- Multi-activation cases should be handled cleanly, not patched case by case.

The transition-distance reward should answer:

```text
Did this action and DG activation produce a useful transition between internal landmarks?
```

The supporting losses should answer only:

```text
Are the DG landmarks and CA3 transitions well-formed enough for transition-distance to be meaningful?
```

## Current Problems

### 1. Transition Time Must Become Geodesic-Like

The original idea is not merely that temporal distance is a rough proxy for spatial distance. The stronger idea is that the decoder loss should encourage the agent to move efficiently between DG landmarks. If that works, then transition time between landmark activations becomes meaningful as an internal approximation of geodesic distance.

Problem:

- If the decoder does not learn efficient landmark-to-landmark movement, transition time may reflect behavior quirks rather than spatial structure.
- Circular trajectories can create repeated, temporally separated activations without broad geodesic coverage.
- High transition-distance reward may not imply that the agent learned useful routes between landmarks.

Derivation:

Let `z_i` denote a DG landmark event and let `R_i(phi)` be the physical region that produces that landmark under encoder parameters `phi`.

For a fixed encoder, define the policy-dependent hitting time:

$$
T_\theta(i \rightarrow j)
= \mathbb{E}_{\pi_\theta}
\left[\tau_j \mid x_0 \in R_i(\phi)\right]
$$

where `tau_j` is the first time the trajectory reaches landmark region `R_j(phi)`.

The decoder reward has the form:

$$
r_{\mathrm{dec}} = B - d_{\mathrm{CA3}}(i \rightarrow j)
$$

where `d_CA3(i -> j)` is the elapsed DG-CA3 transition time between landmark events. If CA3 progression tracks elapsed action steps, then:

$$
\mathbb{E}\left[d_{\mathrm{CA3}}(i \rightarrow j)\right]
\approx T_\theta(i \rightarrow j)
$$

Thus:

$$
\begin{aligned}
\max_\theta \mathbb{E}[r_{\mathrm{dec}}]
&= \max_\theta \mathbb{E}\left[B - d_{\mathrm{CA3}}\right] \\
&\approx \min_\theta T_\theta(i \rightarrow j)
\end{aligned}
$$

For fixed landmarks, minimizing `T_theta(i -> j)` is a stochastic shortest-path problem:

$$
V_j(x)=
\begin{cases}
0, & x \in R_j(\phi) \\
1 + \min_a \mathbb{E}\left[V_j(x') \mid x,a\right], & \text{otherwise}
\end{cases}
$$

So transition time becomes geodesic-like only if decoder learning actually reduces hitting time between stable landmark regions. This is why route-efficiency diagnostics are central.

Plan:

1. Treat landmark-to-landmark traversal as the main behavioral object, not just episode-level coverage.
2. For each newly activated DG landmark, record the previous relevant landmark and the number of action steps between them.
3. Measure whether the policy tends to reduce transition time between frequently connected landmark pairs over training.
4. Compare learned transition paths with physical/geodesic shortest paths only as evaluation, not as reward.
5. Check whether high reward episodes contain diverse landmark transitions or only repeated loops.

Design implication:

The decoder reward should continue to be delayed and action-aligned. Its job is to make actions produce useful transitions through the current landmark graph. We should avoid adding an external coordinate reward, but we should evaluate whether the learned internal graph behaves like a geodesic graph.

### 2. Reward Depends On DG Landmark Quality

Transition-distance reward is meaningful only if DG activations are landmark-like. If DG activations are noisy, fragmented, or overly broad, the distance signal becomes hard to interpret.

Problem:

- A DG unit may fire in multiple disconnected locations.
- Multiple DG units may fire for the same location.
- Too few DG units may be active, leaving the map underrepresented.
- Too many DG units may activate at once, making transition-distance ambiguous.

Derivation:

The transition-distance reward assumes that a DG event identifies a landmark:

$$
z_t = i \quad \Longleftrightarrow \quad x_t \in R_i(\phi)
$$

This assumption is useful only when the encoder-induced regions satisfy approximate landmark conditions:

$$
\begin{aligned}
\text{localization:} \quad & R_i(\phi) \text{ is spatially compact} \\
\text{separation:} \quad & R_i(\phi) \cap R_j(\phi) \approx \varnothing,\quad i \neq j \\
\text{coverage:} \quad & \bigcup_i R_i(\phi) \text{ covers behaviorally relevant space} \\
\text{sparsity:} \quad & |A_t| \text{ is small, ideally } |A_t| \in \{0,1\}
\end{aligned}
$$

If localization fails, then the same unit `i` can refer to multiple places:

$$
R_i(\phi) = R_i^1 \cup R_i^2 \cup \cdots
$$

and transition distance is no longer a distance between physical landmarks. If separation fails, many DG units can represent the same place, and transition reward can reward representational redundancy. If coverage fails, the policy can only learn routes in a partial map.

Therefore the landmark-quality metrics are not cosmetic. They test the assumptions under which `d_CA3` can be interpreted as a landmark transition distance.

Plan:

1. Track DG activity maps for each sequence.
2. Measure per-unit spatial localization, fragmentation, and activation frequency.
3. Measure population coverage of summed DG activity.
4. Keep punishment-style encoder shaping as the current best-supported baseline for single-field-like activity.
5. Add explicit diagnostics for multi-field units instead of assuming every DG unit is a clean single place field.

The transition-distance reward should remain primary; landmark-quality losses should be treated as regularizers.

### 3. Multi-Field DG Units Need Robust Disambiguation

A DG unit can have more than one place field. Punishment-style encoder reward appears to encourage near-single-field behavior, but the algorithm should not break when a DG unit has multiple fields.

Problem:

- If one DG unit fires in two distant places, sequence identity alone is not enough to identify location.
- Transition-distance reward may assign the same landmark identity to physically different places.
- This can corrupt the internal landmark graph.

Derivation:

Suppose DG unit `i` has two disconnected fields:

$$
R_i(\phi) = R_i^A \cup R_i^B,
\qquad
R_i^A \cap R_i^B = \varnothing
$$

A unit-only landmark representation maps both fields to the same event:

$$
g(x)=i,
\qquad
x \in R_i^A \cup R_i^B
$$

This creates an aliasing error:

$$
d_{\mathrm{CA3}}(i,j \mid x \in R_i^A)
\quad \text{and} \quad
d_{\mathrm{CA3}}(i,j \mid x \in R_i^B)
$$

are treated as the same transition, even though their physical geodesic distances to `j` may be very different.

The ambiguity can be reduced by conditioning the event on memory context:

$$
z_t = (i, c_t),
\qquad
c_t = \operatorname{Context}(p_t, A_{t-1}, A_{t-2}, \ldots)
$$

If the recent DG-CA3 context differs between fields, then:

$$
(i,c^A) \neq (i,c^B)
$$

even though both use the same DG unit. This preserves an internal-only mechanism while avoiding the false assumption that every DG unit is perfectly single-field.

Plan:

1. Use the broader DG memory context to disambiguate location.
2. Represent a landmark event as the active DG unit plus recent DG-CA3 context, not just the unit id.
3. Diagnose multi-field ambiguity by checking whether the same DG unit appears with different neighboring DG context patterns.
4. Prefer encoder learning pressure toward single-field activity, but make transition evaluation robust to imperfect single-field tuning.

Candidate mechanism:

$$
\operatorname{landmark\_event}_t
= \operatorname{active\_dg\_unit}_t
+ \operatorname{recent\_active\_context}_t
$$

The recent context could be a small set of recently active DG units, a CA3 progression vector, or a compact signature of the active memory state. This keeps the model internal: disambiguation comes from DG-CA3 history, not from external coordinates.

### 4. Multi-Activation At One Timestep Needs A Clean Rule

The current reward logic has special handling for single versus multiple new activations. That kind of branching can become fragile. The algorithm needs a clean definition that works for zero, one, or many newly activated DG units.

Problem:

- Multiple new activations create ambiguous landmark events.
- Case-specific if/else rules are easy to make inconsistent at boundaries.
- A hard single-activation assumption may be too brittle during learning.

Derivation:

Let:

$$
A_t = \{i : \text{DG unit } i \text{ is newly activated at } t\},
\qquad
C_t = \text{prior active memory context}
$$

The single-activation rule is a special case:

$$
d_t = \min_{j \in C_t} D(i,j),
\qquad
A_t = \{i\}
$$

A general rule should define `d_t` for all `|A_t|`:

$$
d_t(A_t, C_t)
= \operatorname{Agg}_{i \in A_t,\; j \in C_t} D(i,j)
$$

and should include an ambiguity penalty:

$$
P_{\mathrm{col}}(A_t)
= \lambda_{\mathrm{col}} \max(0,\ |A_t|-1)
$$

Then:

$$
r_t = d_t(A_t, C_t) - P_{\mathrm{col}}(A_t)
$$

For `|A_t| = 0`, define `d_t` as baseline/no-event. For `|A_t| = 1`, the rule reduces to the original transition-distance reward. For `|A_t| > 1`, the same equation still applies, avoiding fragile case-specific reward definitions.

Plan:

1. Define transition reward over a set of new activations, not a single activation.
2. Use a generic aggregation rule that handles all cardinalities:
   - zero new activations: baseline/no event;
   - one new activation: standard transition-distance reward;
   - many new activations: aggregate over the set and apply a collision penalty.
3. Prefer vectorized set operations over special case checkers.
4. Keep collision penalty small and interpretable: it should discourage ambiguous simultaneous landmarks, not replace transition-distance reward.

Candidate rule:

$$
\begin{aligned}
A_t &= \{\text{DG units newly activated at } t\} \\
d_{\mathrm{trans}}(A_t, C_t)
&= \operatorname{Agg}_{i \in A_t,\; j \in C_t} D(i,j) \\
P_{\mathrm{col}}(A_t)
&\propto \max(0,\ |A_t|-1) \\
r_t &= d_{\mathrm{trans}}(A_t, C_t) - P_{\mathrm{col}}(A_t)
\end{aligned}
$$

The exact aggregation can be min or mean over valid new-context pairs. The important part is that the same rule handles all cases without hand-coded branches.

### 5. Minimum-Distance Rule Is Probably Not The Main Problem

The current reward uses the nearest relevant sequence distance. Earlier drafts treated this as potentially too local. But if the policy-gradient driving direction is equivalent up to a constant between the closest and other valid neighbors, changing closest-neighbor selection may not change the qualitative policy update.

Problem:

- Replacing minimum distance with k-nearest, soft-min, or percentile distance may add complexity without changing the core learning pressure.
- The bigger issue is whether the transition event and landmark identity are meaningful.

Derivation:

Let the decoder reward for a transition to landmark `i` be:

$$
r_i = B - d_i
$$

Suppose two candidate context references differ by a term that is independent of the current action for the local policy update:

$$
d_{\mathrm{closest}} = d_{\mathrm{other}} + C
$$

Then:

$$
\begin{aligned}
r_{\mathrm{closest}}
&= B - d_{\mathrm{closest}} \\
&= B - d_{\mathrm{other}} - C \\
&= r_{\mathrm{other}} - C
\end{aligned}
$$

Policy-gradient updates use:

$$
\nabla_\theta J
= \mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\; A_t
\right]
$$

A constant shift in reward that is absorbed into the value baseline does not change the qualitative action preference:

$$
A'_t
= (Q_t - C) - (V_t - C)
= Q_t - V_t
= A_t
$$

Therefore changing nearest-neighbor aggregation is unlikely to fix the main issue if the difference is effectively a baseline shift. The more important question is whether `i`, `C_t`, and the transition event are valid landmark objects.

Plan:

1. Do not prioritize changing the minimum-distance rule.
2. Keep closest-distance reward as the default unless evidence shows instability.
3. Log alternative summaries only as diagnostics, not as near-term replacements.
4. Focus first on geodesic traversal, landmark quality, multi-field disambiguation, and multi-activation handling.

This keeps the work centered on the transition-distance mechanism rather than overfitting the scalar distance aggregator.

### 6. Baseline And Sign Choices Affect Encoder Activity

The report shows that reward shaping strongly changes DG activation levels. Punishment makes DG sparse and exploratory, encouragement increases activity, and mean/baseline methods sit between them.

Problem:

- The transition-distance reward is entangled with activity-level control.
- A reward method can appear better because it changes sparsity, not because it improves transition-distance structure.
- Too much punishment can create sparse but fragmented landmarks.

Derivation:

Let the encoder activation probability be:

$$
q_i(x;\phi) = P(y_i = 1 \mid x)
$$

Expected DG activity is:

$$
A(\phi)
= \mathbb{E}_x \left[\sum_i q_i(x;\phi)\right]
$$

The encoder objective can be written schematically as:

$$
J_{\mathrm{enc}}(\phi)
= \mathbb{E}\left[d_{\mathrm{CA3}}(\phi)\right]
- \lambda_{\mathrm{act}} A(\phi)
+ \lambda_{\mathrm{use}} U(\phi)
$$

where `lambda_act` is the punishment/overactivity pressure and `U(phi)` is population usage.

If `lambda_act` is too small:

$$
\lambda_{\mathrm{act}} \text{ too small}
\quad \Longrightarrow \quad
A(\phi) \text{ high}
\quad \Longrightarrow \quad
\text{many simultaneous activations}
$$

If `lambda_act` is too large:

$$
\lambda_{\mathrm{act}} \text{ too large}
\quad \Longrightarrow \quad
A(\phi) \text{ low}
\quad \Longrightarrow \quad
\text{too few landmarks}
$$

Thus baseline/sign choices are not only reward-scaling details. They determine whether the encoder operates in the regime where transition-distance is defined over sparse but sufficiently numerous landmarks.

Plan:

1. Treat activity level as a first-class diagnostic.
2. For every run, report:
   - mean number of active DG units per step;
   - fraction of DG units ever used;
   - transition-distance reward distribution;
   - DG landmark coverage;
   - localization and fragmentation metrics.
3. Keep punishment-style encoder shaping as the default, because the report suggests it promotes sparse landmark-like activity.
4. Monitor under-activation and multi-field fragmentation explicitly.

The goal is not sparsity for its own sake. The goal is sparse DG landmarks that make transition-distance interpretable.

### 7. Encoder And Decoder Must Remain Intertwined

The intertwined update is a main part of the project. The encoder and decoder are driven somewhat oppositely by the transition-distance signal, and the interesting scientific question is whether this coupled system converges to a non-trivial solution.

Problem:

- If encoder and decoder are fully decoupled, we may remove the main mechanism.
- If they are coupled naively, the system may oscillate or collapse.
- The encoder may change the landmark map while the decoder is still learning routes through that map.

Derivation:

Let `theta` be decoder parameters and `phi` be encoder parameters. The coupled update is:

$$
\begin{aligned}
\theta_{k+1}
&= \theta_k + \alpha \nabla_\theta J_{\mathrm{dec}}(\theta_k;\phi_k) \\
\phi_{k+1}
&= \phi_k + \beta \nabla_\phi J_{\mathrm{enc}}(\phi_k;\theta_k)
\end{aligned}
$$

Linearizing near a candidate equilibrium gives:

$$
\begin{bmatrix}
\delta\theta_{k+1} \\
\delta\phi_{k+1}
\end{bmatrix}
=
\begin{bmatrix}
I + \alpha H_{\theta\theta} & \alpha H_{\theta\phi} \\
\beta H_{\phi\theta} & I + \beta H_{\phi\phi}
\end{bmatrix}
\begin{bmatrix}
\delta\theta_k \\
\delta\phi_k
\end{bmatrix}
$$

The cross terms are:

$$
\begin{aligned}
H_{\theta\phi}
&= \frac{\partial}{\partial \phi}
\nabla_\theta J_{\mathrm{dec}} \\
H_{\phi\theta}
&= \frac{\partial}{\partial \theta}
\nabla_\phi J_{\mathrm{enc}}
\end{aligned}
$$

`H_tp` means decoder gradients change when landmarks move. `H_pt` means encoder gradients change when behavior changes. If these cross terms are large relative to the own-system terms `H_tt` and `H_pp`, simultaneous training can oscillate.

This derivation explains why coupling is both necessary and risky. The project needs nonzero cross-coupling, but training may require timescale control.

Plan:

1. Keep the coupled encoder-decoder transition-distance training as the main condition.
2. Use detached delayed labels for timing correctness, but do not interpret that as full algorithmic decoupling.
3. Track convergence of both:
   - landmark stability in the encoder;
   - transition efficiency in the decoder.
4. Use alternating or frozen-component training only as an ablation if coupled training fails.

Important distinction:

```text
Detached reward label = correct delayed credit assignment.
Fully decoupled encoder/decoder training = different algorithm.
```

We should keep the first. We should not adopt the second without strong evidence.

### 8. Iterative Encoder-Decoder Training May Stabilize The Coupled Objective

The original training intuition was to update encoder and decoder weights iteratively: train one side until semi-convergence, freeze it, switch to the other side, and repeat. This is not the same as abandoning the coupled mechanism. It is a block-coordinate version of the same transition-distance objective.

Theoretical setup:

Let:

- `phi` be encoder parameters that define DG landmark activations.
- `theta` be decoder/policy parameters that define behavior.
- `M_phi` be the landmark map induced by the encoder.
- `d_phi(s_t, s_{t+1})` be the transition-distance signal computed from DG-CA3 state.
- `J_dec(theta; phi)` be the decoder objective under a fixed landmark map.
- `J_enc(phi; theta)` be the encoder objective under a fixed behavior distribution.

The project's coupled objective is approximately:

```text
decoder: improve theta so behavior moves efficiently through the current landmark graph M_phi
encoder: improve phi so landmarks become sparse, distributed, and transition-informative under behavior pi_theta
```

Derivation:

The simultaneous update can be viewed as one-step joint gradient dynamics:

$$
\begin{aligned}
\theta_{k+1}
&= \theta_k + \alpha \nabla_\theta J_{\mathrm{dec}}(\theta_k;\phi_k) \\
\phi_{k+1}
&= \phi_k + \beta \nabla_\phi J_{\mathrm{enc}}(\phi_k;\theta_k)
\end{aligned}
$$

The iterative idea instead approximates block-coordinate best responses:

$$
\begin{aligned}
\theta_{m+1}
&\approx \arg\max_\theta J_{\mathrm{dec}}(\theta;\phi_m) \\
\phi_{m+1}
&\approx \arg\max_\phi J_{\mathrm{enc}}(\phi;\theta_{m+1})
\end{aligned}
$$

With partial/semi-convergence:

$$
\begin{aligned}
\theta_{m+1}
&= \operatorname{BR}_{\mathrm{dec}}^K(\theta_m,\phi_m) \\
\phi_{m+1}
&= \operatorname{BR}_{\mathrm{enc}}^L(\phi_m,\theta_{m+1})
\end{aligned}
$$

where `K` and `L` are the number of learner updates, or the time until a plateau criterion is met.

For fixed `phi`, `M_phi` is stable, so the decoder phase optimizes:

$$
J_{\mathrm{dec}}(\theta;\phi)
= \mathbb{E}_{\pi_\theta}
\left[B - d_{\mathrm{CA3}}(M_\phi)\right]
$$

and this is approximately:

$$
\min_\theta
\mathbb{E}_{\pi_\theta}
\left[T(i \rightarrow j \mid M_\phi)\right]
$$

For fixed `theta`, the behavior distribution `rho_theta(x)` is stable, so the encoder phase optimizes:

$$
J_{\mathrm{enc}}(\phi;\theta)
= \mathbb{E}_{x \sim \rho_\theta}
\left[
d_{\mathrm{CA3}}(\phi)
+ \operatorname{regularizers}(\phi)
\right]
$$

The mathematical effect is to replace a moving-target game with two temporarily stationary subproblems.

With simultaneous updates, both sides change on the same training timescale:

$$
\begin{aligned}
\theta_{k+1}
&= \theta_k + \alpha \nabla_\theta J_{\mathrm{dec}}(\theta_k;\phi_k) \\
\phi_{k+1}
&= \phi_k + \beta \nabla_\phi J_{\mathrm{enc}}(\phi_k;\theta_k)
\end{aligned}
$$

This creates a moving-target problem. The decoder is learning routes through a landmark graph that the encoder is changing, while the encoder is learning landmark allocation from trajectories generated by a decoder that is also changing. This can still converge, but it can also oscillate, collapse, or produce a non-trivial but hard-to-interpret equilibrium.

Iterative training changes the timescale:

$$
\begin{aligned}
\phi \text{ fixed:} \quad
& \theta \leftarrow \operatorname{train}
\left(\theta;\ M_\phi\right)
\text{ until decoder semi-convergence} \\
\theta \text{ fixed:} \quad
& \phi \leftarrow \operatorname{train}
\left(\phi;\ \pi_\theta\right)
\text{ until encoder semi-convergence}
\end{aligned}
$$

Expected effect during decoder phase:

- The landmark graph is stable.
- Transition-distance reward has a consistent meaning.
- The decoder can learn shorter paths between currently defined landmarks.
- If this works, transition time becomes closer to geodesic distance on the current landmark graph.

Expected effect during encoder phase:

- The behavior distribution is more stable.
- The encoder can reshape DG landmarks based on a more consistent sample of visited trajectories.
- Sparse/punishment pressure can sharpen landmarks without the policy immediately adapting to exploit transient encoder changes.

Possible benefit:

Iterative training may create a clearer bootstrapping loop:

$$
\text{stable landmarks}
\rightarrow
\text{better landmark traversal}
\rightarrow
\text{better behavioral coverage}
\rightarrow
\text{better landmark allocation}
$$

This could make the transition-distance mechanism more likely to converge to a non-trivial solution.

Possible risk:

- If the decoder semi-converges to paths through a poor early landmark map, it may overfit to bad landmarks.
- If the encoder changes too much during its phase, the decoder's learned routes become obsolete.
- If phases are too long, training may alternate between stale behavior and stale landmarks.
- If phases are too short, the method degenerates back to simultaneous updates.

Plan:

1. Treat simultaneous training as the current baseline.
2. Add iterative training as a high-priority ablation, not as a replacement before evidence.
3. Test several phase lengths:
   - short phases: switch every fixed number of learner updates;
   - medium phases: switch after reward/landmark metrics plateau;
   - long phases: near semi-convergence of one side before switching.
4. During decoder phase:
   - freeze encoder/DG projection;
   - keep DG-CA3 transition-distance reward active;
   - train policy/value to improve landmark traversal.
5. During encoder phase:
   - freeze decoder/policy as much as practical;
   - train encoder/DG projection on detached transition labels and landmark-quality regularizers;
   - monitor whether landmarks move too much.
6. Compare against simultaneous training using the same transition-distance reward and auxiliary terms.

Key diagnostics:

- landmark stability before and after encoder phases;
- route efficiency improvement during decoder phases;
- whether transition time between repeated landmark pairs decreases;
- whether phase switches cause reward collapse;
- whether DG map coverage improves without increasing fragmentation.

Decision rule:

Iterative training is worth keeping if it improves landmark stability and route efficiency while preserving the coupled transition-distance mechanism. It should not be considered successful merely because scalar intrinsic reward increases.

### 9. Auxiliary Losses Can Obscure The Main Mechanism

The report suggests batch/resource-utilization pressure helps, but too many auxiliary losses make it unclear what is actually producing exploration and landmarks.

Problem:

- Multi-activation, unused-sequence, batch loss, sparsity, and reward shaping overlap.
- If all are active, improvement may no longer be attributable to transition-distance reward.
- The algorithm risks becoming a collection of corrections rather than a clean intrinsic motivation principle.

Derivation:

Write the training signal as:

$$
J_{\mathrm{total}}
= J_{\mathrm{TD}}
+ \lambda_{\mathrm{sparse}} J_{\mathrm{sparse}}
+ \lambda_{\mathrm{use}} J_{\mathrm{use}}
+ \lambda_{\mathrm{col}} J_{\mathrm{col}}
+ \lambda_{\mathrm{extra}} J_{\mathrm{extra}}
$$

where `J_TD` is the transition-distance objective and the other terms are supports.

The parameter update is:

$$
\nabla J_{\mathrm{total}}
= \nabla J_{\mathrm{TD}}
+ \lambda_{\mathrm{sparse}} \nabla J_{\mathrm{sparse}}
+ \lambda_{\mathrm{use}} \nabla J_{\mathrm{use}}
+ \lambda_{\mathrm{col}} \nabla J_{\mathrm{col}}
+ \lambda_{\mathrm{extra}} \nabla J_{\mathrm{extra}}
$$

The transition-distance mechanism remains primary only if the support gradients are mostly aligned with it or small enough not to dominate:

$$
\cos\left(\nabla J_{\mathrm{TD}},\ \nabla J_{\mathrm{support}}\right) > 0
$$

or:

$$
\left\|\lambda_{\mathrm{support}}\nabla J_{\mathrm{support}}\right\|
\ll
\left\|\nabla J_{\mathrm{TD}}\right\|
$$

If support terms dominate, the learned behavior may be explained by sparsity, usage, or collision avoidance rather than transition-distance reward. This is why the plan treats auxiliary terms as regularizers and ablates them one at a time.

Plan:

1. Define a minimal transition-distance baseline.
2. Add auxiliary terms one at a time.
3. Keep only terms that make transition-distance produce better landmark traversal and DG maps.
4. Report auxiliary terms as regularizers, not as primary intrinsic motivation.

Recommended hierarchy:

```text
primary: transition-distance reward
support: sparse DG landmark formation
support: balanced DG sequence usage
support: clean multi-activation/collision handling
```

### 10. Boundary And Timing Semantics Need To Stay Explicit

The transition-distance reward is computed after the DG-CA3 transition but assigned back to the action/DG activation that caused it.

Problem:

- Off-by-one errors can silently change what is being optimized.
- The reward can appear to work while training the wrong timestep.
- Boundary steps can introduce artifacts.

Derivation:

Let:

$$
\begin{aligned}
s_t
&= \text{environment/DG-CA3 state before action } a_t \\
a_t
&= \text{action sampled by decoder} \\
s_{t+1}
&= \text{state after action and DG-CA3 transition} \\
r_{t+1}
&= \text{transition-distance reward computed from } s_{t+1}
\end{aligned}
$$

The decoder objective should assign:

$$
r_{t+1} \rightarrow a_t
$$

because `a_t` caused the transition into `s_{t+1}`.

The encoder objective should assign:

$$
r_{t+1} \rightarrow DG_t
$$

because `DG_t` contributed to the CA3 transition that produced the reward.

The correct dependency is:

$$
(DG_t,\ a_t,\ CA3_t)
\rightarrow CA3_{t+1}
\rightarrow r_{t+1}
$$

but the encoder label should be detached:

$$
\mathcal{L}_{\mathrm{enc}}
= -\operatorname{stopgrad}(r_{t+1})\, f(DG_t)
$$

This gives delayed credit assignment without differentiating through the discrete reward calculator. An off-by-one shift would instead optimize:

$$
r_t \rightarrow a_t
$$

or:

$$
r_{t+2} \rightarrow DG_t
$$

which breaks the causal interpretation of transition-distance reward.

Plan:

1. Keep named outputs:
   - `decoder_reward_for_action_t`
   - `encoder_reward_for_dg_t`
2. Keep tests that verify shifted timing.
3. Add small synthetic examples that describe expected transition-distance reward in human terms.
4. Make first-step and bootstrap behavior explicit.

This is an algorithmic issue, not just a code issue, because the theory depends on delayed transition credit.

## Proposed Experiment Sequence

### Stage 1: Baseline Transition-Distance Diagnostics

Run the current transition-distance baseline and log:

- transition-distance reward distribution;
- scalar intrinsic reward over time;
- DG active units per step;
- fraction of DG units used;
- simultaneous new activations;
- landmark-to-landmark transition counts;
- transition time between repeated landmark pairs;
- physical arena coverage;
- trajectory entropy;
- DG map coverage;
- per-unit localization and fragmentation.

Goal:

Establish whether high transition-distance reward predicts both broader exploration and better DG maps.

### Stage 2: Geodesic Traversal Evaluation

For trained agents, evaluate landmark transitions as graph edges:

- nodes: DG landmark events;
- edges: observed transitions between landmark events;
- edge weight: action steps or CA3 progression time between events;
- evaluation only: compare edge paths with physical/geodesic movement when coordinates are available.

Goal:

Determine whether the decoder is learning efficient movement between landmarks, which is the condition that makes transition time meaningful as an internal distance.

### Stage 3: Multi-Field And Multi-Activation Stress Tests

Construct diagnostics for:

- DG units with multiple disconnected fields;
- locations with multiple DG activations;
- timesteps with multiple new activations;
- ambiguous landmark identities with different recent contexts.

Goal:

Design robust internal handling for imperfect DG landmark maps before adding more auxiliary objectives.

### Stage 4: Iterative Training Test

Compare simultaneous training against block-wise iterative training while keeping the transition-distance reward fixed.

Phase sample sizes:

Use learner optimizer steps as the implementation unit, but interpret them as environment-transition samples. With current defaults:

$$
\text{phase\_env\_samples}
\approx
\frac{\text{phase\_train\_steps} \cdot \text{batch\_size}}{\text{num\_epochs}}
$$

For the current IntrMotiv defaults:

$$
\text{batch\_size}=1024,
\qquad
\text{num\_epochs}=1
$$

so:

$$
\text{phase\_env\_samples}
\approx
1024 \cdot \text{phase\_train\_steps}
$$

Recommended first real schedule:

| Phase | Train steps | Approx. transitions | Purpose |
| --- | ---: | ---: | --- |
| Initial encoder warmup | 128 | 131k | Establish a first sparse landmark map before decoder route learning. |
| Decoder phase | 512 | 524k | Let the policy learn repeated landmark-to-landmark traversal on a stable map. |
| Encoder phase | 128 | 131k | Let DG projection reshape landmarks cautiously under a partially learned behavior distribution. |

This gives a `4:1` decoder-to-encoder ratio after warmup. The rationale is:

- decoder route learning needs more samples because it must improve behavior over trajectories;
- encoder updates should be more conservative because large landmark drift can invalidate decoder routes;
- `128` encoder steps is enough to observe DG usage/localization trends without over-moving the landmark graph;
- `512` decoder steps gives PPO enough samples to start reducing repeated landmark-pair transition times.

Smoke-test schedule:

| Phase | Train steps |
| --- | ---: |
| Initial encoder warmup | 4 |
| Decoder phase | 8-16 |
| Encoder phase | 4-8 |

Serious ablation sweep:

| Schedule | Decoder steps | Encoder steps | Approx. decoder transitions | Approx. encoder transitions |
| --- | ---: | ---: | ---: | ---: |
| Short | 512 | 128 | 524k | 131k |
| Medium | 1024 | 256 | 1.05M | 262k |
| Long | 2048 | 512 | 2.10M | 524k |

Do not start with true metric-based semi-convergence. Fixed sample sizes should come first because they make the schedule reproducible and isolate whether iterative training helps. Plateau switching should be added only after phase curves show which metrics are stable enough to drive switching.

Conditions:

1. simultaneous encoder-decoder updates, current baseline;
2. short alternating phases, fixed number of learner updates per phase;
3. metric-triggered phases, switch when decoder reward or encoder landmark metrics plateau;
4. long semi-convergence phases, switch only after clear stabilization;
5. frozen-control runs:
   - fixed encoder, decoder only;
   - fixed decoder, encoder only.

During decoder phases:

- freeze encoder/DG projection;
- keep the current DG landmark map stable;
- train policy/value on delayed transition-distance reward;
- measure whether repeated landmark-pair transition times decrease.

During encoder phases:

- freeze decoder/policy as much as practical;
- train DG projection using detached transition labels and landmark-quality regularizers;
- measure whether landmarks become more localized and more evenly distributed;
- measure whether the landmark graph changes too abruptly.

Goal:

Test the original hypothesis that a semi-converged decoder can learn shorter paths between a stable set of landmarks, making transition time closer to geodesic distance. This test should be run before broad auxiliary-loss combinations, otherwise schedule effects and loss effects will be confounded.

### Stage 5: Auxiliary Loss Ablation

Use the same transition-distance reward and the best-supported training schedule from Stage 4. Vary only support terms:

1. transition-distance only;
2. transition-distance plus punishment-style encoder shaping;
3. transition-distance plus batch/resource usage;
4. transition-distance plus punishment plus batch usage;
5. transition-distance plus punishment plus batch plus clean collision penalty.

Goal:

Identify which support terms help transition-distance reward produce landmark traversal and distributed DG maps.

### Stage 6: Clean Internal Transition Objective

Only after diagnostics and ablations, consolidate support terms into the DG-CA3 interface.

The internal output should still center transition distance:

```python
DGCA3TransitionObjective(
    transition_distance_reward,
    encoder_transition_label,
    decoder_transition_reward,
    landmark_context,
    dg_usage_regularizer,
    collision_regularizer,
    overactivity_regularizer,
    diagnostics,
)
```

Goal:

Make transition-distance reward the source of intrinsic motivation, with named regularizers that protect its assumptions.

## What To Keep For Now

Keep:

- transition-distance reward as the main reward;
- delayed decoder reward assignment;
- detached encoder transition label;
- coupled encoder-decoder training as the main condition;
- punishment-style encoder shaping as the default baseline;
- batch/resource-utilization pressure as the most evidence-supported auxiliary term.

Keep as ablations:

- unused-sequence loss;
- multi-activation loss as a separate auxiliary loss;
- extra decoder loss;
- alternative encoder reward baselines;
- frozen encoder-only or decoder-only controls.

Treat as high-priority schedule ablation:

- iterative encoder-decoder training with semi-convergence phases.

## What To Change

Change the framing first:

- transition-distance reward is the main algorithm;
- the decoder should make transition time geodesic-like by learning landmark-to-landmark traversal;
- auxiliary losses are regularizers that make the transition-distance signal interpretable.

Then change diagnostics:

- track landmark transition graph structure;
- track route efficiency between landmarks;
- track multi-field DG units;
- track multi-activation events as set-valued transitions.

Then test training schedule:

- compare simultaneous updates with iterative semi-convergence phases;
- keep transition-distance reward fixed during this comparison;
- accept iterative training only if it improves landmark stability and route efficiency, not merely scalar reward.

Then change objective structure:

- move support terms into named DG-CA3 internal outputs;
- keep transition-distance reward as the first-class output;
- represent landmark events with recent DG-CA3 context when needed;
- handle multiple new activations with a generic set rule;
- avoid adding external coordinate coverage as reward;
- log physical coverage only as evaluation.

Only change the scalar distance aggregator if diagnostics show it matters. The nearest-distance rule is not currently the highest-priority problem.

## Learner Design Evaluation

The current two-learner setup is not efficient for implementing iterative updates.

Current structure:

- `DistanceLearnerReward` is the active learner for the main transition-distance reward path.
- `DoubleDistanceLearnerReward` is a separate older learner selected by `--double_value=True`.
- Both learners duplicate reward, encoder-loss, decoder-loss, and training-loop logic.
- The newer `DistanceLearnerReward` already uses the cleaner `DGCA3Interface`; `DoubleDistanceLearnerReward` still contains older direct reward construction.

Recommendation:

Do not add a third learner and do not implement iterative updates in both learners.

Implement iterative updates only in `DistanceLearnerReward`. Leave `DoubleDistanceLearnerReward` as a legacy ablation path until there is a specific reason to revive double-value training.

Reasoning:

1. Adding another learner would increase duplicated training-loop code.
2. Extending both existing learners would require maintaining two reward-timing implementations.
3. Iterative update is mainly a schedule over decoder and DG-projection gradients, not a fundamentally different learner architecture.
4. `DistanceLearnerReward` already computes decoder and encoder losses separately, so it is the natural place to add phase-aware updates.

The current two-forward design inside `DistanceLearnerReward` is theoretically correct but computationally inefficient:

- full forward pass: computes decoder/core/value losses and progression diagnostics;
- second `head_only` forward pass: recomputes encoder/DG output so encoder loss has a fresh gradient path.

This second forward is useful during simultaneous or encoder phases, but it is wasteful during decoder-only phases. Therefore the iterative implementation should make loss computation phase-aware:

- simultaneous mode: keep the current two-forward behavior;
- decoder phase: skip the second encoder-only forward and skip encoder loss;
- encoder phase: compute only the minimum full/core information needed for progression diagnostics, then run the encoder-loss path.

For v1, keep one optimizer and use gradient masking rather than separate optimizers:

- one optimizer preserves Sample Factory checkpoint compatibility;
- one optimizer avoids introducing separate learning-rate scheduler and optimizer-state logic;
- inactive gradients can be cleared before `optimizer.step()`.

Separate encoder/decoder optimizers may become useful later if momentum state causes phase leakage, but that should be tested after the fixed-step iterative schedule works.

## Decision Criteria

A change should be accepted only if it improves at least one of these without damaging the others:

- better landmark-to-landmark transition efficiency;
- higher physical coverage during intrinsic pretraining;
- higher DG population coverage;
- more localized DG landmarks;
- lower landmark fragmentation;
- more DG units used;
- lower simultaneous activation collisions;
- stable or improved transition-distance reward.

The central criterion is not scalar intrinsic reward alone. The correct target is:

```text
transition-distance reward produces geodesic-like landmark traversal and distributed DG landmarks
```

## Bottom Line

The transition-distance reward is the main feature and should remain the explanatory center of the project. The current problems are not mainly about replacing the closest-distance calculation. They are about making the prerequisites of the transition-distance idea actually hold:

1. DG units need to behave like useful landmarks.
2. The decoder needs to learn efficient movement between those landmarks.
3. Multi-field and multi-activation cases need robust internal handling.
4. Auxiliary losses should protect the transition-distance mechanism, not become the mechanism.

The immediate plan is to diagnose landmark traversal, robustify landmark identity and multi-activation handling, ablate support losses around the transition-distance reward, and then consolidate only the useful supports into a cleaner DG-CA3 internal objective.
