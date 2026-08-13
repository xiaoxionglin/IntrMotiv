# IntrMotiv Algorithmic Evaluation

Date: 2026-06-05

This note evaluates the intrinsic motivation algorithm itself, not the implementation details. The central question is whether the current DG-CA3 reward mechanism can drive exploration and form DG landmark representations that spread across the whole environment.

## Core Algorithmic Idea

The model uses a sparse DG projection into a CA3-like sequence reservoir. Each DG unit can be interpreted as selecting or reactivating a CA3 sequence. Once activated, that sequence progresses forward for a fixed number of theta-like steps.

The intrinsic reward is based on the temporal relation between sequence activations. When a DG sequence is newly activated, the system asks: how recently were other sequences active? A new activation close in CA3 progression to another activation means the agent has likely activated nearby or redundant landmarks in internal sequence-time. A new activation that is farther from recently active sequences means the agent has expanded the internal temporal map.

So the intrinsic reward is not simple visual novelty. It is a reward over the internal sequence geometry:

```text
good DG landmark allocation = sparse sequence activations that are temporally separated and collectively cover experience
```

The intended behavior is a pull-push system:

- The encoder/DG projection is pushed to allocate sparse, separated landmarks across the environment.
- The decoder/policy is pushed to move through the environment in ways that trigger useful DG-CA3 transitions.
- The batch/unused-sequence losses push the system to use available DG sequence resources rather than collapsing onto a few active units.

This is a strong idea because it ties representation learning and exploration together. The agent does not first explore randomly and then build a map. Instead, the learned map creates the intrinsic objective that shapes exploration.

## Why This Can Produce Exploration

The policy receives reward from internal transitions. To obtain more intrinsic reward, the agent must cause DG-CA3 events that are valued by the reward rule. Since DG activations are driven by visual observations, the policy is indirectly rewarded for moving into observations that cause meaningful sequence activations.

This can create exploration through three mechanisms.

First, repeated local behavior becomes less useful. If the agent stays in the same region and repeatedly triggers the same small set of DG sequences, then new activations are temporally close to recent activations or collide with already-used sequence resources. The intrinsic signal should become less favorable.

Second, moving to different parts of the environment can activate different DG units. If those activations are sparse and separated from recent sequence activity, the intrinsic signal can improve. That gives the policy a reason to leave over-visited regions.

Third, as more DG landmarks are learned, the policy has more internal targets to traverse between. The CA3 sequence state provides an internal temporal metric, so the agent is not only rewarded for seeing new pixels; it is rewarded for moving through a structured set of internal landmark transitions.

This is different from curiosity based on prediction error. Prediction-error curiosity can be distracted by stochastic or visually complex regions. Here the reward is tied to the hippocampal state allocation problem: activate useful sparse landmarks and traverse between them.

## Why This Can Form DG Landmark Representations

DG landmarks should emerge because the encoder is trained to control which visual observations activate which sequence inputs. A useful DG unit should become active for a recognizable subset of observations, and inactive elsewhere. Over training, this can become a spatially localized activation field.

The algorithm encourages this through sparsity and competition:

- A sparse DG projection prevents every observation from activating many sequences.
- Punishing or reducing excessive activity favors only the strongest visual-to-DG matches.
- Multi-activation penalties discourage several DG units from representing the same moment.
- Unused-sequence and batch losses encourage idle DG units to acquire roles.

Together, these pressures can make DG units specialize. If the agent visits many parts of the environment, each useful DG unit can become associated with a different region or landmark. If enough units are recruited, their fields can tile the arena.

The batch loss is especially important algorithmically. Without it, the model can satisfy sparsity by using only a few reliable DG units and ignoring the rest. That may produce clean activity, but not a rich map. Encouraging unused sequences creates pressure to distribute representational responsibility across the available DG population.

## How The Algorithm Spreads Landmarks Across Space

The algorithm does not explicitly know physical coordinates. It spreads landmarks through an indirect loop:

1. The visual encoder maps observations to sparse DG activations.
2. DG activations inject events into the CA3 sequence reservoir.
3. The reward evaluates temporal spacing and reuse of those sequence events.
4. The policy learns trajectories that produce better internal sequence events.
5. Those trajectories expose the encoder to broader visual samples.
6. Broader samples allow more DG units to become tuned to different locations.

This feedback loop can spread DG landmarks over the environment if the policy explores enough and if the encoder has enough pressure to use different sequence units.

The important point is that behavioral coverage and representational coverage reinforce each other. Better exploration gives the encoder more spatial samples. Better distributed DG landmarks give the policy a richer internal objective for further exploration.

This also explains why the representation can become skewed. If the policy learns a repetitive circular path, the encoder will mostly see observations along that path. DG landmarks may become clean but concentrated on the trajectory, leaving other parts of the environment underrepresented. Conversely, if the intrinsic policy visits more quadrants, DG landmarks have a better chance to cover the whole map.

## Why The Current Reward Can Work

The current reward can work because it uses a meaningful internal variable: time since sequence activation. In navigation, temporal separation often correlates with spatial separation under movement. If two DG activations occur close together in sequence-time, they may correspond to nearby or redundant observations. If they occur farther apart while the agent is moving, they are more likely to correspond to distinct parts of the environment.

This gives the algorithm a useful approximation:

```text
temporal spacing of DG-CA3 events can act as a proxy for spatial spacing of landmarks
```

That proxy is imperfect, but useful. It allows the agent to build spatial structure without external coordinates or goal reward.

The pull-push design is also important. If the encoder and decoder optimized exactly the same reward in the same way, they could collapse into trivial behavior. The encoder is shaped toward useful, sparse, distributed activations. The decoder is shaped toward behavior that moves through those activations. This division gives the model a chance to simultaneously improve map quality and exploration.

## Biological Interpretation

Algorithmically, the model resembles a hippocampal exploration system:

- DG provides sparse, high-threshold pattern separation.
- CA3 provides sequence continuation and temporal context.
- Intrinsic reward comes from the structure of internal hippocampal activity rather than external goals.
- Delayed reward assignment resembles feedback after a transition, not direct gradient through the biological circuit.

The biological appeal is that the agent can value states because they improve its internal map, not because they are externally rewarded. That matches the idea that exploratory behavior can be driven by the need to construct useful spatial representations.

The landmark interpretation also fits DG sparsity. A DG unit should not fire everywhere. It should mark a distinct situation, visual context, or place-like event. The algorithm tries to create many such sparse events and arrange them so they are not all clustered in the same behavioral region.

## Main Weaknesses

The biggest weakness is that temporal separation is not identical to spatial coverage.

An agent can create temporally separated sequence activations while moving in a repetitive loop. This may satisfy the internal metric without covering the full physical space. The report's observation that some reward methods produce circular behavior is consistent with this failure mode.

A second weakness is that sparse landmark formation and broad exploration can fight each other. Strong punishment can create sharp, selective DG landmarks, but it can also make activity too sparse or noisy. Encouragement can recruit more units, but it can also increase overlap and reduce landmark specificity.

A third weakness is that the reward is local to recently active sequence events. It does not directly measure global map coverage. The batch loss partially addresses this by encouraging unused sequences, but unused sequence recruitment is not the same as uniform spatial tiling.

A fourth weakness is aliasing. If two visually similar locations map to the same DG activation, the algorithm may treat them as the same landmark even if they are physically distinct. Conversely, one physical location may activate multiple DG units if viewpoint, direction, or visual details vary strongly.

A fifth weakness is that the decoder can exploit the reward rule. If some behavior reliably triggers internally favorable activations, the policy may repeat that behavior even if it does not improve global coverage. This is a general risk for intrinsic motivation: the agent can optimize the intrinsic proxy rather than the intended property.

## Redundant Or Overlapping Algorithmic Pressures

Several intrinsic pressures partly target the same thing.

The base reward, multi-activation loss, and sparsity/punishment all discourage crowded DG activity. They are not identical, but they overlap. If all are active and strong, the model may over-suppress DG activity.

The unused-sequence loss and batch loss both try to recruit inactive DG units. The batch loss is more global over a minibatch, while unused-sequence loss is more event-specific. Their shared goal is resource utilization, so they should be tuned as a pair rather than treated as independent additions.

The decoder reward and extra decoder loss both shape behavior around DG activation events. If both are used, they may double-count avoidance of bad activation patterns.

The encoder reward methods are also partially redundant ways of controlling activity level:

- encouragement increases activity and recruitment;
- punishment suppresses activity and sharpens landmarks;
- mean/baseline adjustment tries to balance the two.

These are better understood as different activity-control regimes, not as fundamentally different intrinsic motivation theories.

## What Would Make The Algorithm Stronger

The algorithm would be stronger if the intrinsic objective measured both local landmark quality and global map coverage.

A good target would preserve the current local sequence-event reward but add or integrate a coverage term:

```text
reward useful new landmark events,
penalize redundant simultaneous activations,
encourage use of inactive DG resources,
and discourage repeated coverage of only one behavioral loop
```

The key is to avoid adding external position supervision. Coverage should still be internal if the goal is a pure intrinsic motivation system. Possible internal coverage proxies include:

- entropy of DG unit usage over a rollout;
- diversity of CA3 progression states visited;
- novelty of DG activation patterns relative to recent memory;
- balanced occupancy of learned landmark clusters;
- reduction in overlap between DG fields while preserving sufficient activation frequency.

The batch loss points in this direction, but it is currently an auxiliary constraint. A cleaner version would make resource usage and coverage part of the intrinsic reward definition itself.

## Evaluation Criteria

For this algorithm, success should be measured on both behavior and representation.

Behavior metrics:

- fraction of arena visited per episode;
- quadrant or room coverage entropy;
- path diversity across episodes;
- avoidance of collapsed loops;
- ability to reach all regions without external reward.

Representation metrics:

- fraction of DG units used;
- spatial coverage of summed DG activity;
- localization of individual DG fields;
- low correlation between DG unit activation maps;
- stability of landmarks across episodes;
- low fragmentation, meaning one unit should not fire in many unrelated places.

The important combined metric is whether high behavioral coverage and high DG landmark coverage occur together. A model that explores widely but forms noisy landmarks has not solved the representation problem. A model that forms clean landmarks only along a narrow path has not solved the exploration problem.

## Bottom Line

The current intrinsic motivation algorithm is plausible and interesting because it links exploration directly to hippocampal representation formation. Its core strength is the closed loop between behavior and DG-CA3 sequence allocation: the agent explores to improve internal sequence structure, and that structure creates the reward landscape for further exploration.

The mechanism can form DG landmarks that spread through space if three conditions hold:

1. DG activity is sparse enough to make landmarks selective.
2. Enough DG units are recruited to cover the environment.
3. The policy's intrinsic reward does not collapse into repetitive loops.

The current design addresses all three, but with partially overlapping reward and auxiliary-loss terms. The most important algorithmic next step is to make coverage and resource utilization part of one coherent intrinsic objective, rather than relying on several separate losses that each correct one failure mode.
