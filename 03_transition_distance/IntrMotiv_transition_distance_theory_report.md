# Theory Report: Transition-Distance Reward And Iterative Encoder-Decoder Training

Date: 2026-06-05

## Executive Findings

The transition-distance reward is theoretically meaningful only under a coupled condition:

```text
DG activations must become landmarks, and the decoder must learn efficient paths between them.
```

If this condition holds, transition time in the DG-CA3 sequence system becomes an internal approximation of geodesic distance. This is the core mechanism of IntrMotiv.

The current simultaneous training scheme updates the encoder and decoder at the same time. That preserves the coupled nature of the algorithm, but it creates a moving-target problem: the decoder is learning routes through a landmark graph that the encoder is changing, while the encoder is reshaping landmarks under a behavior policy that is also changing.

The original iterative idea is theoretically well motivated. It can be interpreted as block-coordinate training or approximate best-response dynamics:

1. Fix encoder landmarks and train the decoder until it semi-converges to efficient landmark traversal.
2. Fix decoder behavior and train the encoder until landmarks become more sparse, stable, and distributed under that behavior.
3. Repeat.

This does not remove the encoder-decoder coupling. It separates the timescales so each side can make progress against a temporarily stable counterpart.

The key prediction is:

```text
If iterative training is working, decoder phases should reduce transition time between stable landmark pairs,
and encoder phases should improve landmark coverage/localization without destroying the learned landmark graph.
```

## 1. Formal Setup

Let the environment state be:

$$
x_t \in \mathcal{X}
$$

The agent observes:

$$
o_t = O(x_t)
$$

The encoder with parameters `phi` maps observations to sparse DG activity:

$$
y_t = E_\phi(o_t)
$$

For the idealized analysis, treat DG activity as binary:

$$
y_t \in \{0,1\}^N
$$

where `N` is the number of DG/CA3 sequence inputs.

Let:

$$
A_t = \{i : y_{t,i}=1 \text{ and } i \text{ is newly activated}\}
$$

be the set of newly activated DG units at time `t`.

The CA3 sequence core stores a progression variable for each sequence:

$$
p_{t,i} \in \{0,1,\ldots,H\},
\qquad
H = L + R - 1
$$

where:

- `p_{t,i} = 0` means sequence `i` just received DG input;
- larger `p_{t,i}` means the sequence has progressed farther since its last activation;
- `p_{t,i} = H` is the inactive/faded baseline.

For a newly activated sequence `i`, the transition-distance metric is approximately the distance from `i` to recently active sequence context:

$$
d_t(i)
= \min_{j \in C_t}
\left|p_{t,i} - p_{t,j}\right|
$$

Because `p_{t,i} = 0` for a new activation, this is close to:

$$
d_t(i)
= \min_{j \in C_t} p_{t,j}
\qquad \text{when } p_{t,i}=0
$$

where `C_t` is the set of valid recent context sequences.

For multiple new activations, the natural extension is set-valued:

$$
d_t(A_t,C_t)
= \operatorname{Agg}_{i \in A_t,\;j \in C_t}
\left|p_{t,i} - p_{t,j}\right|
$$

with an additional collision term if `|A_t| > 1`.

## 2. Landmark Interpretation

Assume first that each useful DG unit corresponds to one spatial landmark region:

$$
R_i(\phi)
= \{x \in \mathcal{X} : E_\phi(O(x))_i = 1\}
$$

If the encoder is good, each `R_i(phi)` is:

- sparse;
- localized;
- stable across visits;
- not too overlapping with other `R_j(phi)`;
- collectively part of a broad spatial cover.

Then a newly activated DG unit can be interpreted as a landmark event:

$$
e_t = i,
\qquad
x_t \in R_i(\phi)
$$

In the imperfect case where one DG unit has multiple fields, a better landmark event is:

$$
e_t = (i,c_t)
$$

where `c_t` is recent DG-CA3 context. The context disambiguates which field of unit `i` is active.

Thus the true object is not just the unit id. It is:

$$
\operatorname{landmark\ event}
= \operatorname{DG\ identity}
+ \operatorname{memory\ context}
$$

This matters because transition-distance is only meaningful if landmark identity is meaningful.

## 3. Decoder Objective As Shortest-Path Learning

The decoder/policy has parameters `theta`:

$$
a_t \sim \pi_\theta(a \mid o_t,h_t)
$$

where `h_t` includes recurrent/CA3 state.

The original transition-distance metric is:

$$
d_t
= \text{time since the relevant previous landmark activation}
$$

For the decoder, the reward is shaped as:

$$
r_{\mathrm{dec},t} = B - d_t
$$

where `B` is a baseline such as `H = L + R - 1`.

So maximizing decoder reward is equivalent to minimizing transition distance:

$$
\max r_{\mathrm{dec},t}
\quad \Longleftrightarrow \quad
\min d_t
$$

Now fix the encoder `phi`. This fixes the landmark regions `R_i(phi)` and therefore fixes the current landmark graph.

For two landmark events `i` and `j`, define the policy-dependent hitting time:

$$
T_\pi(i \rightarrow j)
= \mathbb{E}_\pi
\left[
\text{number of action steps to reach landmark } j
\text{ after landmark } i
\right]
$$

In a deterministic environment with fixed target landmark `j`, minimizing hitting time satisfies a shortest-path Bellman equation:

$$
V_j(x)=
\begin{cases}
0, & x \in R_j(\phi) \\
1 + \min_a V_j(f(x,a)), & \text{otherwise}
\end{cases}
$$

In a stochastic environment:

$$
V_j(x)=
\begin{cases}
0, & x \in R_j(\phi) \\
1 + \min_a \mathbb{E}\left[V_j(x') \mid x,a\right], & \text{otherwise}
\end{cases}
$$

This is the stochastic shortest-path problem.

Therefore, under fixed landmarks:

```text
decoder training on B - d_t approximates learning short paths between landmark events.
```

This is the theoretical basis of the geodesic claim. If the decoder semi-converges while the landmark graph is fixed, observed transition time can become closer to geodesic distance on that graph.

## 4. Encoder Objective As Landmark Allocation

Now fix the decoder/policy `theta`. This fixes the behavior distribution:

$$
\rho_\theta(x)
= \text{state occupancy induced by } \pi_\theta
$$

The encoder then chooses DG landmark regions under a stable distribution of observations.

The encoder's transition-distance pressure is roughly:

$$
\max_\phi d_t(\phi)
$$

while sparse/punishment pressure and regularizers constrain the solution:

```text
low total DG activity
low simultaneous activation collision
high enough population usage
localized fields
```

A simplified encoder objective can be written:

$$
J_{\mathrm{enc}}(\phi;\theta)
= \mathbb{E}_{x_t \sim \rho_\theta}
\left[
\lambda_d d_t(\phi)
+ \lambda_{\mathrm{use}} U_t(\phi)
- \lambda_{\mathrm{col}} C_t(\phi)
- \lambda_{\mathrm{act}} A_t(\phi)
\right]
$$

where:

- `d_t(phi)` is transition distance from DG-CA3 dynamics;
- `U_t(phi)` rewards balanced sequence usage;
- `C_t(phi)` penalizes multiple new activations at the same time;
- `A_t(phi)` penalizes diffuse overactivity.

This resembles a constrained landmark-allocation problem:

```text
choose sparse fields R_i(phi) that cover the behaviorally sampled environment
while producing useful transition distances.
```

If the decoder has learned efficient paths between current landmarks, then `rho_theta` should include trajectories between landmark regions. That gives the encoder samples from corridors and underrepresented regions, allowing it to allocate better landmarks.

## 5. The Pull-Push Mechanism

The central interaction is:

```text
encoder: push landmarks farther apart / make them selective
decoder: pull transitions shorter / move efficiently between landmarks
```

This is not contradictory. It is the main constructive tension.

If the encoder alone maximized transition distance, it might push landmarks too far apart or make too few landmarks active.

If the decoder alone minimized transition distance, it might exploit the nearest landmark pair or a repetitive loop.

Together, a useful equilibrium would look like:

```text
DG landmarks spread across the environment,
and the policy learns efficient transitions between them.
```

In that equilibrium, transition time approximates an internal geodesic distance because:

1. landmarks are spatially meaningful;
2. the policy learns short paths between them;
3. CA3 progression records elapsed transition time.

## 6. Simultaneous Updates

The current training scheme updates encoder and decoder together:

$$
\begin{aligned}
\theta_{k+1}
&= \theta_k + \alpha \nabla_\theta J_{\mathrm{dec}}(\theta_k;\phi_k) \\
\phi_{k+1}
&= \phi_k + \beta \nabla_\phi J_{\mathrm{enc}}(\phi_k;\theta_k)
\end{aligned}
$$

This preserves the coupled mechanism, but creates moving targets.

Near an equilibrium `(theta*, phi*)`, write small deviations:

$$
\delta\theta_k = \theta_k - \theta^\ast,
\qquad
\delta\phi_k = \phi_k - \phi^\ast
$$

The linearized dynamics are:

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

where:

$$
\begin{aligned}
H_{\theta\theta}
&= \frac{\partial}{\partial \theta}
\nabla_\theta J_{\mathrm{dec}} \\
H_{\theta\phi}
&= \frac{\partial}{\partial \phi}
\nabla_\theta J_{\mathrm{dec}} \\
H_{\phi\theta}
&= \frac{\partial}{\partial \theta}
\nabla_\phi J_{\mathrm{enc}} \\
H_{\phi\phi}
&= \frac{\partial}{\partial \phi}
\nabla_\phi J_{\mathrm{enc}}
\end{aligned}
$$

The cross terms are the problem:

$$
\begin{aligned}
H_{\theta\phi}
&:\ \text{decoder gradient changes when encoder landmarks move} \\
H_{\phi\theta}
&:\ \text{encoder gradient changes when decoder behavior changes}
\end{aligned}
$$

If these cross terms are large, simultaneous updates can oscillate. The decoder chases landmarks that are moving, and the encoder adapts to behavior that is changing.

This does not mean simultaneous training is wrong. It means it has a non-trivial stability condition:

$$
\text{same-timescale coupled learning works only if cross-coupling is not too destabilizing}
$$

## 7. Iterative Training As Block-Coordinate Optimization

The proposed iterative scheme is:

$$
\begin{aligned}
\phi \text{ fixed:} \quad
& \theta \leftarrow \operatorname{train}(\theta;M_\phi)
\text{ until decoder semi-convergence} \\
\theta \text{ fixed:} \quad
& \phi \leftarrow \operatorname{train}(\phi;\pi_\theta)
\text{ until encoder semi-convergence}
\end{aligned}
$$

Mathematically, this approximates best-response dynamics:

$$
\begin{aligned}
\theta_{m+1}
&\approx \arg\max_\theta J_{\mathrm{dec}}(\theta;\phi_m) \\
\phi_{m+1}
&\approx \arg\max_\phi J_{\mathrm{enc}}(\phi;\theta_{m+1})
\end{aligned}
$$

or, with partial convergence:

$$
\begin{aligned}
\theta_{m+1}
&= \operatorname{BR}_{\mathrm{dec}}^K(\phi_m,\theta_m) \\
\phi_{m+1}
&= \operatorname{BR}_{\mathrm{enc}}^L(\theta_{m+1},\phi_m)
\end{aligned}
$$

where `K` and `L` are the number of training updates or a plateau criterion for each phase.

This changes the problem from:

```text
both players chase each other every minibatch
```

to:

```text
each player improves against a temporarily stable counterpart
```

## 8. Why Iteration Could Help The Geodesic Mechanism

### Decoder Phase

When `phi` is fixed, the landmark graph is stable:

$$
G_\phi = (V_\phi, E_\phi)
$$

where:

- `V_phi` are landmark events;
- `E_phi` are observed transitions between them.

The decoder phase can then reduce observed transition times:

$$
T_{\pi_\theta}(i \rightarrow j)
$$

for repeated landmark pairs.

If this phase works, we should observe:

$$
T_{\pi_\theta}(i \rightarrow j)
\text{ decreases over decoder training}
$$

and route efficiency should improve:

$$
\eta(i,j)
= \frac{\operatorname{geodesic\_distance}(i,j)}
{\operatorname{observed\_path\_length}(i,j)}
$$

where `eta` is only an evaluation metric, not a reward.

### Encoder Phase

When `theta` is fixed, the behavior distribution is more stable:

$$
\rho_\theta(x)
$$

The encoder phase can reshape landmarks under this distribution:

$$
R_i(\phi_m) \rightarrow R_i(\phi_{m+1})
$$

If this phase works, we should observe:

- more DG units used;
- better spatial coverage;
- more localized fields;
- lower collision rate;
- controlled landmark drift.

The key is controlled drift. The encoder should improve the landmark map without destroying the graph the decoder just learned.

## 9. Expected Training Signatures

If iterative training is working, metrics should show a structured pattern.

During decoder phases:

```text
transition time between stable landmark pairs decreases
decoder reward improves
route efficiency improves
landmark map remains stable
```

During encoder phases:

```text
landmark coverage improves
unused DG units become recruited
single-field quality improves
decoder reward may temporarily drop
landmark graph changes, but not catastrophically
```

Across full cycles:

```text
decoder recovers faster after each encoder phase
landmark graph becomes more stable over cycles
physical coverage increases
DG coverage increases
transition-distance reward remains meaningful
```

This may produce a sawtooth pattern:

```text
encoder phase changes landmarks -> decoder reward drops
decoder phase relearns routes -> decoder reward recovers and improves
next encoder phase makes smaller changes
```

A decreasing sawtooth amplitude would be evidence of convergence.

## 10. Failure Modes

### Bad Early Landmark Lock-In

If the first encoder map is poor, a long decoder phase may overfit to bad landmarks.

Symptom:

```text
route efficiency improves, but coverage remains poor
```

Mitigation:

- start with shorter early phases;
- allow batch/resource pressure during encoder phases;
- do not freeze encoder too early for too long.

### Landmark Drift Too Large

If the encoder changes too much during its phase, decoder learning becomes obsolete.

Symptom:

```text
decoder reward collapses after every encoder phase
landmark identity matching across phases is poor
```

Mitigation:

- reduce encoder learning rate during encoder phase;
- add landmark-stability regularization;
- switch phases earlier;
- preserve recent context for landmark identity.

### Loop Exploitation

The decoder may find a short loop between a few easy landmarks.

Symptom:

```text
transition time decreases
scalar reward increases
physical coverage and DG coverage remain low
```

Mitigation:

- keep resource-usage pressure;
- evaluate graph entropy;
- penalize repeated use of the same small transition set only if needed, and keep it internal.

### Over-Sparsity

Punishment can create clean but too few landmarks.

Symptom:

```text
high localization
low fraction of DG units used
low map coverage
```

Mitigation:

- keep batch/resource utilization;
- tune punishment relative to usage pressure;
- track active unit fraction explicitly.

## 11. Experimental Tests Implied By The Theory

### Test 1: Fixed Encoder, Decoder-Only Semi-Convergence

Freeze the encoder and train only the decoder.

Prediction:

```text
transition times between repeated landmark pairs should decrease
```

If this does not happen, the decoder reward is not producing shortest-path-like learning.

### Test 2: Fixed Decoder, Encoder-Only Semi-Convergence

Freeze the decoder and train only the encoder.

Prediction:

```text
DG map quality should improve under the fixed behavior distribution
```

If this does not happen, the encoder reward is not producing better landmark allocation.

### Test 3: Alternating Short Phases

Switch encoder and decoder phases frequently.

Prediction:

```text
more stable than simultaneous only if moving-target effects are significant
```

If short phases behave like simultaneous training, phase length is too short to reveal the mechanism.

### Test 4: Alternating Semi-Convergence Phases

Switch phases after plateau criteria.

Decoder plateau candidates:

```text
mean transition time stops decreasing
decoder reward stops improving
route efficiency stops improving
```

Encoder plateau candidates:

```text
DG coverage stops increasing
landmark localization stops improving
collision rate stops decreasing
landmark drift becomes small
```

Prediction:

```text
semi-convergence phases should produce clearer landmark graph and route learning if the theory is correct
```

### Test 5: Simultaneous Versus Iterative Matched-Loss Comparison

Keep reward/loss terms fixed and vary only training schedule:

```text
condition A: simultaneous updates
condition B: alternating short phases
condition C: alternating plateau phases
condition D: long semi-convergence phases
```

Prediction:

Iterative training should outperform simultaneous training specifically on:

- landmark stability;
- route efficiency;
- transition-time consistency;
- physical coverage;
- DG population coverage.

Scalar reward alone is insufficient.

## 12. Metrics

### Landmark Stability

For each DG unit `i`, define its spatial activation map before and after a phase:

$$
M_i^{\mathrm{before}}(x),
\qquad
M_i^{\mathrm{after}}(x)
$$

Stability can be measured by correlation:

$$
S_i
= \operatorname{corr}
\left(
M_i^{\mathrm{before}},
M_i^{\mathrm{after}}
\right)
$$

or by matched overlap.

Population stability:

$$
S = \frac{1}{N}\sum_i S_i
$$

### Transition Efficiency

For landmark pair `(i, j)`:

$$
\begin{aligned}
T_\pi(i,j)
&= \text{observed action steps between } i \text{ and } j \\
G(i,j)
&= \text{geodesic distance between landmark regions} \\
\eta(i,j)
&= \frac{G(i,j)}{T_\pi(i,j)}
\end{aligned}
$$

Higher `eta` means more efficient movement.

Use physical/geodesic distance only for evaluation.

### Landmark Graph Entropy

Let:

$$
q(i,j)
= \text{empirical frequency of transition } i \rightarrow j
$$

Graph entropy:

$$
H_{\mathrm{graph}}
= -\sum_{i,j} q(i,j)\log q(i,j)
$$

Low graph entropy with high reward suggests loop exploitation.

### DG Usage

Let:

$$
u_i
= \text{fraction of rollout steps where DG unit } i \text{ is active}
$$

Usage entropy:

$$
H_{\mathrm{DG}}
= -\sum_i u_i \log u_i
$$

This should be high enough to indicate broad sequence use, but not so high that DG activity becomes diffuse.

### Multi-Activation Collision

Let:

$$
c_t = \max(0,\ |A_t|-1)
$$

Collision rate:

$$
C = \mathbb{E}[c_t]
$$

High collision rate means transition events are ambiguous.

## 13. Theoretical Conclusion

The iterative training idea has a clear theoretical role. It gives the transition-distance mechanism the stable objects it needs:

$$
\begin{aligned}
\text{fixed encoder phase}
&\rightarrow
\text{stable landmark graph for decoder shortest-path learning} \\
\text{fixed decoder phase}
&\rightarrow
\text{stable behavior distribution for encoder landmark allocation}
\end{aligned}
$$

Simultaneous training may still work, and it remains the natural baseline. But simultaneous training asks both sides to solve a moving-target game at the same time. Iterative training tests whether the mechanism becomes clearer when the decoder is allowed to semi-converge on a stable landmark graph before the encoder changes that graph.

The most important empirical test is not whether iterative training gives higher intrinsic reward. The important test is:

$$
\text{Does iterative training make transition time behave more like geodesic distance between stable DG landmarks?}
$$

If yes, it directly supports the original theory of IntrMotiv. If no, then the project should look first at landmark identity, multi-field ambiguity, and multi-activation handling before changing the transition-distance reward itself.
