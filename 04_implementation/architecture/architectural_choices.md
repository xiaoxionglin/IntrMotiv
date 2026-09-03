# IntrMotiv Architectural Choices

This catalogue records selectable design branches, their exact scope, and the
comparison needed to interpret them. It is intentionally not a run manifest:
put concrete parameter combinations in `06_experiments/`.

## 1. Fixed Foundation

| Choice | Current intended setting | Details |
|---|---|---|
| Environment | DMLab openfield, no external reward | Five reduced actions, frameskip 8. Use matched episode/telemetry settings when comparing behavior. |
| Visual trunk | `layer2_resnet18`, ImageNet-pretrained, frozen | This is deliberate. It supplies fixed visual features; changing it is a separate representation-learning experiment. |
| DG projection | Trainable linear projection + BatchNorm + thresholded ReLU | DG rows are renormalized to unit norm after updates. BatchNorm running statistics are part of the trainable landmark layer. |
| Landmark count | `F=16` in current comparisons | One DG unit is one landmark and one possible target. There is no separate option-count hyperparameter. |
| CA3 memory | Fixed shift register, `R=8`, `L=64` | Width is `F(R+L-1)`. It has no learned weights. `R` defines local event/recent-history behavior; `L` defines the longer temporal trace and default unknown-edge horizon. |
| Bypass | Depth and map-id features | Nonrecurrent task context reaches the decoder alongside CA3 and target conditioning. |
| Policy | One categorical worker policy and value head | The manager is deterministic code, not a learned policy, and has no separate reward/value/optimizer. |
| Optimization | PPO worker plus separate DG encoder pass | PPO-to-DG gradients are cleared; DG is trained only by encoder objectives/structural updates. |

## 2. DG Representation Choices

| Choice | CLI / values | What changes | Required comparison |
|---|---|---|---|
| Threshold | DG intercept, commonly `2.43`; lower variants | Controls the activity regime before any loss acts. It is a hyperparameter, not a learning mechanism. | Matched threshold with the same encoder method/losses. |
| Encoder reward method | `encourage`, `punish`, `mean`, legacy adjusted variants | Changes the sign/centering of temporal-distance encoder feedback. Does not change worker PPO directly. | Keep batch loss and DG threshold fixed. |
| Batch recruitment | `encoder_batch_loss` | Encourages valid-minibatch-unused units through a pre-threshold softplus, so silent rows retain gradient. | On/off; keep `encoder_batch_loss_temperature=0.5` unless explicitly testing its scale. |
| Population control | `encoder_population_usage_loss` plus usage/density/collision coefficients | Adds balanced unit use, target-density, and same-step collision objectives. | Each coefficient should be isolated before a combined sweep. |
| Anti-collapse regularization | global punishment, row repulsion, CA3 temporal margin | Separately controls logit pressure, projection-direction redundancy, and the minimum reinforced distance between dominant DG onsets. A margin coefficient of 1 centers `encourage` feedback at `Hippo_R`. | Match threshold and encoder feedback; nonzero temporal margin currently requires `encourage`. |
| Orthogonal recruitment | `dg_orthogonal_recruitment` | Non-gradient replacement of one silent DG row at an `L`-step novelty endpoint. | Same loss configuration, with graph invalidation enabled. |
| Motion localization | path scatter coefficient/distance/straightness | Suppresses same-unit far reactivation only on straight traces. | Requires action integration; compare graph-only motion vs scatter. |

## 3. Flat And Direct-Target HRL Choices

| Mode | `hrl_controllable_graph` / manager | State and behavior | Notes |
|---|---|---|---|
| Flat intrinsic | disabled | No target condition or graph state. Worker learns from dense temporal-distance reward. | Required baseline for every HRL claim. |
| Direct landmark HRL | enabled + `visit_direct` | Manager selects an eligible least-visited landmark. Worker is conditioned on that DG target. | `T_ctrl` constrains feasibility/deadlines, not curiosity ranking. |
| HRL worker reward | `hit` or `hit_distance` | Deliberate target/waypoint hits receive a positive reward, optionally plus clipped temporal-distance bonus. | Compare only with the same manager target process. |
| Deadline | bootstrap horizon, learned `T_ctrl` plus margin | Unknown edges use fallback horizon. Feasible deliberate edges use an empirical arrival time plus fixed/relative margin. | Travel time gates/control only; it must not be added to curiosity ranking. |
| Exploration option | `hrl_exploration_mode`, probability, horizon | A reserved all-zero target selects bounded dense-reward exploration. It is chosen probabilistically at boundaries and forced after target timeout. | Must be matched when comparing manager-selection rules. |

## 4. Graph Memory Scope And Replay Contract

| Scope | `hrl_graph_memory` | What persists | Replay rule |
|---|---|---|---|
| Episode | `episode` | Full graph in `rnn_states`; normal terminal reset clears it. | Learner reconstructs state from stored RNN state. Compatibility default. |
| Persistent stream suffix | opt-in Sample Factory suffix | Selected graph fields survive terminal reset per actor stream. | Only the suffix survives; default Sample Factory behavior is unchanged at size zero. |
| Policy buffer | `policy_buffer` | Learner-owned, non-gradient graph buffers shared by rollout streams of one policy; checkpointed and parameter-synchronized. | Actor target/waypoint is stored in stream RNN state. Learner teacher-forces that stored condition rather than re-selecting against a newer graph. |

For all HRL modes, `rnn_state_t.target` is the condition for sampled action
`a_t`. A manager update writes the next target into `rnn_state_{t+1}`. This
prevents graph changes between sampling and learning from changing old action
log-probabilities, values, or worker reward inputs.

## 5. Manager And Graph Choices

| Manager mode | Passive graph | Target rule | Routing / validation |
|---|---|---|---|
| `visit_direct` | none | Least-visited eligible DG | Direct target only. |
| `topology_visit_direct` | yes | Novelty-only inverse visit rank | Same passive evidence, validation protocol, and bounded exploration as frontier modes. Causal control for UCB scoring. |
| `frontier_direct` | yes | Frontier score `N_i + beta_U U_i + 0.5Y_i` | Direct worker conditioning on final frontier. |
| `frontier_waypoint` | yes | Same frontier score | Shortest path through deliberately validated edges; worker conditions on next hop and replans after each hit. |

### Passive versus controllable topology

A passive edge `i -> j` requires distinct exclusive DG activations in one
physical episode, elapsed time no greater than `L`, and optional local-motion
filters. Passive confidence at the threshold creates a candidate only.

The manager must deliberately perform `RETURN(j -> i)` then
`VALIDATE(i -> j)`. Only a target-conditioned success creates a controllable
edge. Planning uses controllable edges exclusively. This separation is the
central protection against treating accidental co-occurrence as navigability.

### Frontier selection

For a reachable visited node `i`, inverse visit rank `N_i`, attempt count `E_i`,
and discovery count `D_i`:

$$
U_i=\min\left(1,\sqrt{\frac{\log(2+\sum_k E_k)}{1+E_i}}\right),
\qquad
Y_i=\frac{D_i+1}{E_i+2},
$$

$$
S_i=N_i+\beta_U U_i+0.5Y_i.
$$

Travel time affects reachability and deadline construction, never `S_i`.
Manager priority is pending validation, reachable candidate validation,
highest-scoring reachable frontier, then local bounded exploration.

## 6. Action Path Integration And Geometry Controls

| Choice | CLI | Effect | Not used for |
|---|---|---|---|
| Previous action observation | enabled by action integration | The observation carries the action that produced it; reset uses a no-action sentinel. It keeps actor and replay aligned without Sample Factory changes. | Ground-truth DMLab position. |
| Fixed command integration | `hrl_action_path_integration` | Integrates fixed body-frame reduced-action transforms with midpoint heading. Tracks local displacement, full traveled path length, turn, and straightness. | Learned global coordinates. |
| Motion-conditioned worker | `hrl_motion_policy_input` | Appends normalized local displacement, path, heading, and straightness features to worker conditioning. | A separate learned navigation policy. |
| Graph-only motion | integration on, policy input off | Motion filters passive edges and supplies scatter diagnostics without expanding worker input. | Testing a metric representation in the policy. |
| SE(2) control | `hrl_landmark_geometry=se2` | Fits landmark poses outside PPO from passive transforms; proposes nearest unvalidated neighbors. | Promoting passive/geometric proposals without deliberate validation. |

Action integration is an egomotion cue and local edge-length estimate. It is
not the main algorithmic claim and must be compared to the topological,
non-coordinate condition. Debug DMLab positions are telemetry only.

## 7. Update Schedules And Training Scope

| Choice | Meaning | Interpretation requirement |
|---|---|---|
| Simultaneous update | Worker and DG encoder objectives train on every learner update. | Baseline schedule. |
| Iterative update | Alternates encoder-only and decoder-only phases after an optional encoder warm-up. | Compare against simultaneous at matched total frames and phase schedule. |
| One policy / no PBT | Current controlled batch setting | Keeps graph scope and parameter history unambiguous. |
| Policy-global graph | One fast-weight buffer per policy | Not synchronized across independent policies. PBT/global cross-policy graphs are deliberately deferred. |

## 8. Explicit Non-Choices And Boundaries

- No learned manager, no manager policy gradient, and no manager value function.
- No HER, PPO goal relabeling, or behavior-cloning shortcut loss.
- No learned DG coordinate embedding in the topological condition.
- No passive edge may enter a planned route before deliberate validation.
- No graph-buffer update occurs per PPO minibatch or PPO epoch; it occurs once
  per newly accepted rollout outside autograd.
- Orthogonal recruitment changes landmark identity. It must invalidate
  incident graph/pose state and stale active options.

## 9. How To Define A Clean Ablation

Change one causal layer at a time:

1. representation pressure: threshold or DG loss;
2. worker conditioning/reward: flat versus direct HRL;
3. graph evidence: no graph versus passive/validated graph;
4. selection: visit rank versus UCB frontier;
5. navigation: direct final target versus waypoint routing;
6. motion: graph-only integration versus worker-conditioned integration;
7. geometry: topology-only versus matched SE(2) control.

Keep encoder method, batch loss, fixed visual trunk, `F/R/L`, reward mode,
episode process, seeds, and frame budget matched unless they are the explicit
factor.
