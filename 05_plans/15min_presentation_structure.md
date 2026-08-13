# 15-Minute Presentation Structure

Title: **From DG-CA3 Navigation Models to Intrinsic Motivation**

Goal: explain how the main DG-CA3 navigation project motivates the IntrMotiv project, and how transition-distance intrinsic reward fits into the standard actor-critic objective.

## Slide 1: Thesis

**Message:** The IntrMotiv project asks whether a DG-CA3 sequence model can generate its own reward for exploration and spatial representation learning.

Key points:

- Main project: DG-CA3 model supports navigation with sparse DG input and CA3 sequence dynamics.
- Limitation: external reward can bias spatial representations toward task goals.
- IntrMotiv: use internal DG-CA3 transition structure as reward.

## Slide 2: Main DG-CA3 Navigation Model

**Message:** DG provides sparse landmark-like input; CA3 turns those events into temporal sequence structure.

Model sketch:

$$
o_t \xrightarrow{\text{visual encoder}} z_t
\xrightarrow{\text{DG projection}} DG_t
\xrightarrow{\text{CA3 sequence core}} CA3_t
\xrightarrow{\text{decoder}} \pi(a_t \mid s_t), V(s_t)
$$

Key terms:

- `DG_t`: sparse activation, ideally landmark-like.
- `CA3_t`: sequence state tracking recent DG activations.
- decoder: policy/value readout for navigation.

## Slide 3: Standard A3C / Actor-Critic Objective

**Message:** In standard actor-critic, external reward trains both the critic and policy through the advantage.

Policy loss:

$$
\mathcal{L}_{\pi}
=
-\log \pi_\theta(a_t \mid s_t)\, A_t
$$

Advantage:

$$
A_t
=
R_t - V_\theta(s_t)
$$

n-step return:

$$
R_t
=
\sum_{k=0}^{n-1} \gamma^k r_{t+k+1}
+ \gamma^n V_\theta(s_{t+n})
$$

Value loss:

$$
\mathcal{L}_V
=
\frac{1}{2}
\left(R_t - V_\theta(s_t)\right)^2
$$

Entropy regularization:

$$
\mathcal{L}_{H}
=
-\beta\, H\left(\pi_\theta(\cdot \mid s_t)\right)
$$

Total actor-critic loss:

$$
\mathcal{L}_{A3C}
=
\mathcal{L}_{\pi}
+ c_V \mathcal{L}_{V}
+ \mathcal{L}_{H}
$$

Transition to IntrMotiv:

$$
r_{t+1}^{\mathrm{external}}
\quad \Longrightarrow \quad
r_{t+1}^{\mathrm{intrinsic}}
$$

## Slide 4: Why Intrinsic Reward?

**Message:** Exploration should improve the internal map, not only chase an external goal.

Problem with external-only learning:

- place fields can overrepresent goal regions;
- exploration may stop once task reward is found;
- representation quality becomes entangled with one task.

Desired intrinsic pretraining:

- explore broadly;
- form sparse DG landmarks;
- distribute landmarks across the space;
- create a useful CA3 transition graph before external task learning.

## Slide 5: Transition-Distance Intrinsic Reward

**Message:** Intrinsic reward is computed from the DG-CA3 sequence state after a transition.

CA3 progression for each DG sequence:

$$
p_{t,i} \in \{0,1,\ldots,H\},
\qquad
H = L + R - 1
$$

where:

- $p_{t,i}=0$: sequence $i$ just received DG input;
- larger $p_{t,i}$: sequence has progressed farther since last input;
- $H$: inactive/faded baseline.

For newly activated DG sequence $i$:

$$
d_t(i)
=
\min_{j \in C_t}
\left|p_{t,i} - p_{t,j}\right|
$$

Since a new activation has $p_{t,i}=0$:

$$
d_t(i)
\approx
\min_{j \in C_t} p_{t,j}
$$

Interpretation:

- small distance: new landmark is close/redundant with recent sequence context;
- large distance: new landmark is farther from recent context in internal sequence time.

## Slide 6: Decoder Reward From Transition Distance

**Message:** The decoder/policy is trained to move efficiently between internal landmarks.

The intrinsic decoder reward is shaped as:

$$
r_{t+1}^{\mathrm{dec}}
=
B - d_{t+1}
$$

where:

$$
B = H = L + R - 1
$$

So maximizing decoder reward approximately minimizes transition time:

$$
\max_\theta \mathbb{E}
\left[r_{t+1}^{\mathrm{dec}}\right]
=
\max_\theta \mathbb{E}
\left[B - d_{t+1}\right]
\approx
\min_\theta
T_\theta(i \rightarrow j)
$$

This is the key geodesic idea:

$$
\text{stable DG landmarks}
+ \text{efficient decoder policy}
\Rightarrow
\text{transition time approximates geodesic distance}
$$

## Slide 7: Encoder Reward / Landmark Formation

**Message:** The encoder/DG projection is trained to produce sparse, useful landmarks.

Encoder uses a detached delayed label:

$$
\mathcal{L}_{\mathrm{enc}}
=
-\operatorname{stopgrad}
\left(r_{t+1}^{\mathrm{enc}}\right)
\, f(DG_t)
$$

The timing is important:

$$
(DG_t,\ a_t,\ CA3_t)
\rightarrow
CA3_{t+1}
\rightarrow
r_{t+1}^{\mathrm{intrinsic}}
$$

Meaning:

- reward is computed after the DG-CA3 transition;
- credit is assigned back to the DG activation and action that caused it;
- the reward label is detached, avoiding gradients through discrete CA3 reward logic.

## Slide 8: Pull-Push Mechanism

**Message:** Encoder and decoder are driven in productive tension.

Decoder:

$$
\max_\theta \left(B - d_{t+1}\right)
\quad \Rightarrow \quad
\text{shorter landmark-to-landmark paths}
$$

Encoder:

$$
\text{learn sparse, separated, reusable DG landmarks}
$$

Conceptual equilibrium:

$$
\text{DG landmarks spread across space}
\quad +
\quad
\text{policy learns routes between landmarks}
$$

This creates a closed loop:

$$
\text{movement}
\rightarrow
\text{DG activation}
\rightarrow
\text{CA3 transition}
\rightarrow
\text{intrinsic reward}
\rightarrow
\text{better movement}
$$

## Slide 9: Evidence From Jannek's Report

**Message:** Existing results suggest the mechanism is promising but sensitive to loss design.

Evidence to highlight:

- Punishment/sparsity made DG activity more landmark-like.
- Punished agents explored more of the arena.
- Batch/resource-utilization pressure recruited more DG sequences and improved coverage.
- Some conditions still showed fragmented fields or loop-like behavior.

Interpretation:

The transition-distance reward is the main idea, but it needs well-formed DG landmarks and stable decoder learning.

## Slide 10: Current Problem: Moving Target

**Message:** Simultaneous encoder-decoder updates may destabilize the transition-distance mechanism.

Current coupled update:

$$
\begin{aligned}
\theta_{k+1}
&=
\theta_k
+ \alpha
\nabla_\theta
J_{\mathrm{dec}}(\theta_k;\phi_k) \\
\phi_{k+1}
&=
\phi_k
+ \beta
\nabla_\phi
J_{\mathrm{enc}}(\phi_k;\theta_k)
\end{aligned}
$$

Issue:

- decoder learns routes through landmarks that the encoder is changing;
- encoder learns landmarks under behavior that the decoder is changing.

This motivates iterative training.

## Slide 11: Proposed Iterative Training

**Message:** Stabilize one side while the other learns.

Block-coordinate idea:

$$
\begin{aligned}
\phi \text{ fixed:}
\quad
&\theta
\leftarrow
\operatorname{train}
\left(\theta; M_\phi\right)
\quad
\text{decoder learns routes} \\
\theta \text{ fixed:}
\quad
&\phi
\leftarrow
\operatorname{train}
\left(\phi; \pi_\theta\right)
\quad
\text{encoder improves landmarks}
\end{aligned}
$$

Default phase sizes:

| Phase | Train steps | Approx. transitions |
| --- | ---: | ---: |
| Initial encoder warmup | 128 | 131k |
| Decoder phase | 512 | 524k |
| Encoder phase | 128 | 131k |

## Slide 12: Evaluation Metrics

**Message:** Scalar intrinsic reward is not enough; we need behavior and representation metrics.

Behavior:

- physical coverage;
- trajectory entropy;
- landmark-to-landmark transition counts;
- transition time between repeated landmark pairs.

Representation:

- fraction of DG units used;
- DG population coverage;
- per-unit localization;
- landmark fragmentation;
- simultaneous activation collisions.

Key test:

$$
\text{Does transition-distance reward produce}
\quad
\text{geodesic-like landmark traversal}
\quad
\text{and distributed DG landmarks?}
$$

## Slide 13: Closing

**Message:** IntrMotiv reframes navigation pretraining as internal map-building.

Takeaways:

- Standard actor-critic can be driven by intrinsic reward instead of external reward.
- The intrinsic reward comes from DG-CA3 transition distance.
- The central hypothesis is that stable DG landmarks plus efficient decoder traversal make sequence time approximate geodesic distance.
- The next experimental step is iterative encoder-decoder training.

Final one-line summary:

$$
\boxed{
\text{Explore to build the map, and use the map to guide exploration.}
}
$$

## Timing Guide

| Section | Slides | Time |
| --- | --- | ---: |
| Main project and motivation | 1-4 | 4 min |
| Intrinsic reward mechanism | 5-8 | 5 min |
| Evidence and problems | 9-10 | 3 min |
| Next step and evaluation | 11-13 | 3 min |

