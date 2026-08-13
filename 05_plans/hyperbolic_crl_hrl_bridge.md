# Hyperbolic CRL As A Bridge To Hierarchical RL

Date: 2026-07-31

## Core Idea

Contrastive reinforcement learning (CRL) and related goal-conditioned representation-learning methods can amortize away some explicit planning. Instead of learning a separate policy or value for every landmark pair, the agent learns a reusable reachability representation:

$R(s,g) \approx \phi(s)^\top \psi(g)$

or a goal-conditioned controller:

$\pi(a \mid s,g)$

This is powerful because all source-goal pairs share parameters. But it also creates a weakness: a single smooth latent geometry has to represent long-horizon reachability across rooms, corridors, bottlenecks, loops, and branching paths. In such environments, Euclidean embeddings can distort distances, especially when the environment has tree-like or hierarchical structure.

The hypothesis:

> Shortcomings of long-horizon distance representation in CRL-style learning can be mitigated by hyperbolic representations. Hyperbolic CRL can be viewed as a middle ground between flat goal-conditioned RL and explicit hierarchical RL.

## Why Flat CRL Can Struggle

CRL-style methods often learn a score between current state and goal:

$f(s,g) = \phi(s)^\top \psi(g)$

or a distance:

$d(s,g) = \|\phi(s)-\psi(g)\|$

This works well when reachability is locally smooth and roughly Euclidean. But long-horizon navigation often has non-Euclidean structure:

- rooms connected by narrow doors;
- branching corridors;
- bottleneck states shared across many routes;
- local visual similarity but different global reachability;
- long paths whose useful abstraction is a sequence of subgoals;
- state spaces that grow exponentially with decision depth.

The issue is not only approximation error. It is representational geometry. If the true reachability structure is graph-like or hierarchical, Euclidean embeddings may require high dimension or may collapse important bottleneck relations.

## Hyperbolic Representation

Hyperbolic space is well suited for tree-like and hierarchical structures because volume grows exponentially with radius. This makes it natural for representing branching reachability:

```text
root / hub / bottleneck
    -> branch 1
    -> branch 2
    -> branch 3
```

In a hyperbolic CRL variant, the state and goal embeddings live in a hyperbolic space:

$z_s = \phi(s) \in \mathbb{H}^d$

$z_g = \psi(g) \in \mathbb{H}^d$

and the CRL score depends on hyperbolic distance:

$f(s,g) = -d_{\mathbb{H}}(z_s,z_g)$

instead of Euclidean dot product or Euclidean distance.

For example, in the Poincare ball:

$d_{\mathbb{H}}(u,v) = \operatorname{arcosh}\left(1 + 2\frac{\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$

This gives a representational bias:

```text
near origin: abstract hubs / bottlenecks / reusable subgoals
near boundary: specific leaf states / local details
```

## Hyperbolic CRL As Middle Ground

Flat CRL:

```text
learn one reusable goal-conditioned representation
act directly toward goal
```

Hierarchical RL:

```text
learn subgoals/options
plan or choose among high-level goals
execute low-level controller
```

Hyperbolic CRL:

```text
learn one reusable goal-conditioned representation
but the geometry itself supports hierarchical/branching reachability
```

So hyperbolic CRL sits between the two:

- It keeps the amortized, shared-function advantage of CRL.
- It adds a structural bias similar to HRL.
- It may reduce the need for an explicit between-landmark planner.
- It can still expose implicit hierarchy through embedding radius and geodesic structure.

In this view, HRL and CRL are not opposites. They are points on a continuum:

```text
flat CRL
    shared reachability representation, weak hierarchy

hyperbolic CRL
    shared reachability representation with hierarchical geometry

explicit HRL
    discrete subgoals/options and high-level planning
```

## Relation To Landmarks

In explicit landmark planning, landmarks are nodes and the planner computes routes between them:

$g_1 \rightarrow g_2 \rightarrow g_3$

In CRL, the representation tries to absorb this into a direct goal-conditioned score:

$f(s,g)$

In hyperbolic CRL, landmarks may reappear implicitly:

- high-level bottlenecks may lie near central regions of hyperbolic space;
- local states may spread toward the boundary;
- branches may correspond to rooms, corridors, or behavioral contexts;
- subgoal-like states may be those with high betweenness or small hyperbolic radius.

So the planner is not gone exactly. Some of the hierarchical planning structure is moved into geometry.

## Why This Matters For The Current Project

This gives a cleaner theoretical umbrella:

> Long-horizon navigation requires representing reachability over a graph-like state space. One can solve this with explicit hierarchy, with amortized CRL, or with a curved reachability geometry that makes hierarchy implicit.

Your DG/CA3 sequence-memory idea can then be positioned as a complementary mechanism:

```text
hyperbolic CRL:
    slow learned geometry for reusable reachability

DG/CA3 sparse sequence memory:
    fast event-based memory for local route fragments and recent structure
```

The two could be combined:

1. DG events provide sparse candidate addresses.
2. CA3/fast weights store local event transitions.
3. Hyperbolic CRL embeds events/states into a curved reachability space.
4. Planning can be done by:
   - direct hyperbolic goal pursuit,
   - few sequence-memory rollouts,
   - or explicit HRL when needed.

This avoids framing the project as purely opposed to CRL. Instead:

> CRL amortizes reachability, HRL discretizes reachability, and sparse sequence memory can supply fast local evidence that helps either representation.

## Potential Paper Contribution

A strong theoretical/conceptual contribution could be:

> We unify CRL and HRL as different ways to represent long-horizon reachability. Flat CRL amortizes reachability in a dense representation; HRL externalizes reachability as subgoals and options; hyperbolic CRL offers a middle ground by embedding hierarchical transition structure directly into the goal-conditioned representation.

Then the DG/CA3 angle becomes:

> Sparse hippocampal-style events can serve as discrete addresses that write local transition evidence into fast memory, while hyperbolic or hierarchical representations organize those addresses for long-horizon reuse.

## Concrete Algorithm Variants

### Variant 1: Hyperbolic CRL Baseline

Replace Euclidean embeddings with Poincare embeddings:

$z_s = \phi(s), \quad z_g = \psi(g)$

$f(s,g) = -d_{\mathbb{H}}(z_s,z_g)$

Use the same contrastive objective as the Euclidean baseline, but with hyperbolic distance as the score.

Expected benefit:

```text
better long-horizon ranking of goals in branching mazes
```

### Variant 2: Event-Level Hyperbolic CRL

Use DG-like event codes $e_t$ to define discrete or sparse event states, then learn hyperbolic embeddings of events:

$z_e = \phi(e)$

The transition graph among events is embedded in hyperbolic space.

Expected benefit:

```text
cleaner hierarchy because events remove irrelevant visual detail
```

### Variant 3: Hyperbolic CRL + Fast Sequence Memory

Use hyperbolic distance as a slow global prior and CA3-like fast weights as recent local evidence:

$\mathrm{score}(g \mid s) = -d_{\mathbb{H}}(\phi(s),\psi(g)) + \lambda \, a_g^{(K)}$

where $a_g^{(K)}$ is arrival strength after $K$ recurrent memory steps.

Expected benefit:

```text
global structure from hyperbolic CRL
local adaptivity from fast sequence memory
```

### Variant 4: HRL Approximation Of Hyperbolic CRL

Extract implicit hierarchy from hyperbolic radius:

- central nodes become high-level subgoals;
- boundary nodes become local states;
- geodesic paths suggest subgoal chains.

Then use explicit HRL on top of the embedding.

This tests the claim that HRL approximates what hyperbolic geometry represents continuously.

## Evaluation

### Geometry Metrics

Compare Euclidean CRL and hyperbolic CRL on:

- rank correlation between learned distance and oracle shortest-path distance;
- distortion of bottleneck distances;
- ability to represent branching mazes in low dimension;
- separation of rooms/branches;
- radius correlation with graph centrality or betweenness.

Possible metric:

$\rho = \operatorname{corr}(d_{\mathbb{H}}(\phi(s_i),\psi(s_j)), d^*(s_i,s_j))$

### Control Metrics

Evaluate:

- goal-reaching success;
- sample efficiency after unsupervised pretraining;
- long-horizon success vs short-horizon success;
- performance in branching mazes vs open arenas;
- robustness under weak exploration.

### Hierarchy Metrics

Ask whether hyperbolic radius reveals useful hierarchy:

$\operatorname{corr}(-\|z_s\|_{\mathbb{H}}, \mathrm{Betweenness}(s))$

or whether geodesics pass through oracle bottlenecks.

## Predictions

1. Euclidean CRL should work well in open or near-Euclidean environments.
2. Hyperbolic CRL should outperform Euclidean CRL in tree-like, room-like, or bottleneck-heavy environments.
3. Explicit HRL should outperform both when discrete subgoals are clean and options are reliable.
4. Hyperbolic CRL should close part of the gap to HRL without requiring explicit subgoal extraction.
5. Fast sequence memory should help most when local transition evidence changes faster than the slow CRL embedding.

## Risks

- Hyperbolic embeddings may not help if the environment is not sufficiently hierarchical or tree-like.
- CRL objectives may not naturally place bottlenecks near the origin without additional pressure.
- Hyperbolic optimization can be finicky.
- If explicit graph planning is allowed, it may still outperform both CRL variants.
- The connection to DG/CA3 could become too diffuse unless event codes and fast memory are central to the experiments.

## When Would HRL And Hyperbolic CRL Be Equivalent?

The useful question is not whether the algorithms are identical. They are not. HRL makes the decomposition explicit:

```text
state -> high-level subgoal -> low-level controller
```

Hyperbolic CRL keeps a flat goal-conditioned interface:

```text
state + goal -> action
```

They become equivalent only in a weaker behavioral or representational sense:

> Hyperbolic CRL is equivalent to HRL if its geometry induces the same intermediate waypoints, same reachability ordering, or same actions that an explicit hierarchy would produce.

### Three Levels Of Equivalence

#### 1. Distance Equivalence

Hyperbolic CRL learns distances that are monotonically related to the HRL abstract graph distance:

$d_{\mathbb{H}}(\phi(s),\psi(g)) = m(d_{\mathrm{HRL}}(s,g))$

where $m$ is monotone. This is enough for ranking goals and choosing nearer subgoals.

#### 2. Waypoint Equivalence

The geodesic from $s$ to $g$ in hyperbolic space passes through the same bottleneck/subgoal states that HRL would select:

$w_{\mathbb{H}}(s,g) \approx w_{\mathrm{HRL}}(s,g)$

where $w_{\mathbb{H}}$ is a point or data state near the hyperbolic geodesic from $\phi(s)$ to $\psi(g)$.

This is the strongest useful equivalence for planning.

#### 3. Policy Equivalence

The flat goal-conditioned policy conditioned on the final goal produces the same action distribution as the low-level policy conditioned on the HRL subgoal:

$\pi_{\mathrm{CRL}}(a \mid s,g) \approx \pi_{\ell}(a \mid s,w_{\mathrm{HRL}}(s,g))$

This is close to planning invariance: the action needed to reach a far goal is the same as the action needed to reach the next good waypoint on the route.

## Assumptions Needed For Equivalence

### Assumption 1: The Reachability Graph Is Hierarchical Or Tree-Like

Hyperbolic geometry helps when the state-transition graph has low-dimensional hierarchical structure:

- rooms connected by doors;
- branching corridors;
- nested regions;
- hub-and-spoke topology;
- bottlenecks shared by many shortest paths.

If the environment is a flat open field, Euclidean CRL may be enough. If the environment is a grid with many loops and no dominant hierarchy, hyperbolic CRL may not correspond cleanly to HRL.

A useful diagnostic is graph hyperbolicity or tree-likeness. Hyperbolic CRL should help most when oracle shortest-path structure is well approximated by a negatively curved geometry.

### Assumption 2: Subgoals Correspond To Bottlenecks Or Abstract Hubs

For HRL and hyperbolic CRL to agree, the HRL subgoals must be the same states that hyperbolic geometry naturally emphasizes:

```text
central / low-radius states = hubs, doors, bottlenecks, reusable connectors
boundary / high-radius states = specific local states
```

If HRL subgoals are task-specific manipulation poses, temporally extended skills, or semantic objects that are not graph bottlenecks, hyperbolic distance alone may not recover them.

### Assumption 3: Local Goal Reaching Is Reliable

Both systems require a reusable local controller. HRL assumes the low-level policy can reach nearby subgoals. Hyperbolic CRL assumes local progress in embedding space corresponds to real controllable progress.

Formally, for nearby embedded states:

$d_{\mathbb{H}}(\phi(s),\psi(g)) \le \epsilon \Rightarrow P(\text{reach } g \mid s,\pi) \text{ is high}$

If this fails, the geometry is decorative: it may rank states correctly but not support control.

### Assumption 4: The Contrastive Objective Learns Near-Optimal Reachability, Not Just Behavior Frequency

Flat contrastive learning can learn policy-dependent future occupancy. HRL, especially with planning over subgoals, often aims closer to shortest-path or best-available reachability.

Equivalence requires the CRL distance to approximate a stitched or optimized distance:

$d_{\mathrm{CRL}}(s,g) \approx d^*(s,g)$

not merely:

$d_{\mathrm{CRL}}(s,g) \approx \Delta t_{\pi_{\mathrm{data}}}(s,g)$

This probably requires TD backups, multistep stitching, quasimetric constraints, graph augmentation, or strong data coverage.

### Assumption 5: The Hyperbolic Geodesic Is Action-Relevant

The geodesic should not only interpolate embeddings. It should correspond to feasible intermediate states:

$\operatorname{Geo}_{\mathbb{H}}(\phi(s),\psi(g)) \cap \phi(\mathcal{D})$

should contain states that lie on plausible routes in the dataset or environment.

If geodesics pass through empty latent regions, HRL cannot be recovered without an additional projection onto real states.

### Assumption 6: Directionality Is Handled

Standard hyperbolic distance is symmetric:

$d_{\mathbb{H}}(s,g) = d_{\mathbb{H}}(g,s)$

but control reachability can be asymmetric:

$d^*(s,g) \neq d^*(g,s)$

For navigation this may be mild, but for manipulation or one-way transitions it matters. A better middle ground may be **hyperbolic quasimetric CRL**:

$d(s,g) = d_{\mathbb{H}}(u_s,u_g) + q(s,g)$

where $q$ captures directional cost, constraints, or action asymmetry.

## Tweaks That Make Equivalence More Likely

### 1. Use Hyperbolic Distance As The CRL Score

Replace dot-product CRL:

$f(s,g) = \phi(s)^\top \psi(g)$

with:

$f(s,g) = -d_{\mathbb{H}}(\phi(s),\psi(g))$

This makes the learned value directly interpretable as a curved reachability distance.

### 2. Add Multi-Horizon Contrastive Positives

Use positives at multiple temporal scales:

```text
near future -> local controller structure
mid future -> subgoal structure
far future -> global hierarchy
```

This encourages the embedding to represent hierarchy rather than only immediate dynamics.

### 3. Add Radius/Hierarchy Regularization

Encourage graph-central or high-betweenness states to have smaller hyperbolic radius:

$\|z_s\|_{\mathbb{H}} \downarrow \quad \text{for reusable connector states}$

This is important because hyperbolic models do not always discover hierarchy automatically. The radius needs a reason to mean abstraction.

### 4. Extract HRL Subgoals From Hyperbolic Geodesics

Given $s$ and $g$, compute a hyperbolic geodesic midpoint or fractional waypoint:

$z_w = \operatorname{Geo}_{\mathbb{H}}(z_s,z_g;\alpha)$

Then select the nearest dataset/event state:

$w = \arg\min_{x \in \mathcal{D}} d_{\mathbb{H}}(\phi(x),z_w)$

This turns hyperbolic CRL into HRL by discretizing its implicit hierarchy.

### 5. Train With Planning-Invariance

Encourage:

$\pi(a \mid s,g) \approx \pi(a \mid s,w(s,g))$

where $w(s,g)$ is the hyperbolic waypoint. If this holds, then explicit HRL and flat hyperbolic CRL produce the same local action.

### 6. Quantize Hyperbolic Space Into Shells And Sectors

HRL can be approximated by discretizing hyperbolic space:

- radial shells = abstraction level;
- angular sectors = branch identity;
- cells = subgoal regions.

Then explicit HRL is a quantized version of hyperbolic CRL.

### 7. Add A Quasimetric Or Directional Head

For asymmetric control problems, use hyperbolic geometry for hierarchy and a directional residual for action-conditioned reachability:

$d(s,g,a) = d_{\mathbb{H}}(\phi(s,a),\psi(g)) + r_{\rightarrow}(s,a,g)$

This preserves the HRL analogy while avoiding the false assumption that reachability is symmetric.

## Where The Equivalence Breaks

Hyperbolic CRL and HRL will not be equivalent when:

- subgoals are not bottlenecks but semantic/task-specific states;
- the environment is mostly Euclidean rather than hierarchical;
- the transition graph has many equally good loops, making tree geometry misleading;
- the policy has irreversible or strongly asymmetric dynamics;
- CRL learns behavior-frequency distances from poor data rather than shortest-path-like distances;
- the learned geodesic passes through latent regions with no reachable states;
- long-horizon policy extraction is noisy, so direct conditioning on $g$ gives a different action than conditioning on the nearest useful waypoint.

This matters for the current project because DG/CA3 fast sequence memory may be most useful exactly where hyperbolic CRL is not enough:

```text
hyperbolic CRL:
    slow, global, approximate hierarchy

HRL:
    explicit subgoal decomposition

DG/CA3 fast memory:
    recent/local route fragments and event-specific corrections
```

## Evaluation Plan For The Equivalence Claim

### 1. Synthetic Graphs

Use graphs with controlled structure:

- trees;
- trees plus loops;
- room graphs;
- grid worlds;
- directed graphs.

Compare Euclidean CRL, hyperbolic CRL, and explicit HRL.

Metrics:

$\rho_d = \operatorname{corr}(d_{\mathrm{model}}(i,j), d^*(i,j))$

$\rho_w = \operatorname{corr}(\mathbf{1}[w_{\mathrm{model}} \in P^*_{ij}], \mathbf{1}[w_{\mathrm{HRL}} \in P^*_{ij}])$

where $P^*_{ij}$ is an oracle shortest path.

### 2. Waypoint Agreement

For each source-goal pair, extract:

- HRL high-level subgoal;
- hyperbolic geodesic waypoint;
- nearest dataset state to geodesic waypoint.

Measure agreement:

$\Pr[w_{\mathbb{H}}(s,g) \in \mathrm{Neighborhood}(w_{\mathrm{HRL}}(s,g))]$

### 3. Policy Invariance

Measure action-distribution similarity:

$D_{\mathrm{KL}}(\pi(a \mid s,g) \,\|\, \pi(a \mid s,w_{\mathbb{H}}(s,g)))$

If this is small, then the hyperbolic representation has absorbed the HRL decomposition.

### 4. Failure-Mode Map

Vary:

- graph hyperbolicity;
- loop density;
- bottleneck strength;
- directional asymmetry;
- data-policy quality;
- embedding dimension and curvature.

The expected result:

```text
hyperbolic CRL approaches HRL when hierarchy is strong and local control is reliable;
explicit HRL wins when subgoal choices require discrete search or task-specific options;
Euclidean CRL wins or ties in open, near-Euclidean spaces.
```

## Related Works

### Goal-Conditioned / Contrastive RL

- **C-Learning** learns goal reaching by recursive classification of future observations. This is an important precursor because it views goal-reaching probabilities through density/classification rather than explicit rewards.
  - https://openreview.net/forum?id=tc5qisoB-C

- **Contrastive Learning as Goal-Conditioned RL** shows that a contrastive objective can learn representations whose inner product corresponds to a goal-conditioned value function.
  - https://arxiv.org/abs/2206.07568

- **DWSL** learns the distribution of time steps between states and uses it to approximate shortest-path distances for goal-conditioned policies.
  - https://openreview.net/forum?id=46MiXApQr3

- **Multistep Quasimetric Learning / MQE** is directly relevant because it treats GCRL as learning temporal distances with quasimetric structure and multistep stitching.
  - https://openreview.net/forum?id=UElh7vzgKX

- **Contrastive Representations for Temporal Reasoning** is relevant because it asks whether learned contrastive representations can support temporal reasoning and reduce search.
  - https://proceedings.neurips.cc/paper_files/paper/2025/hash/9d75de47462ffe77addaa7b985fc6d8e-Abstract-Conference.html

### HRL And Goal-Conditioned Hierarchy

- **HIQL** is central: it explicitly argues that long-horizon offline GCRL suffers from noisy values for faraway goals, and addresses this by decomposing goal reaching into high-level latent subgoals and low-level control.
  - https://seohong.me/projects/hiql/

- **Test-Time Graph Search for GCRL** shows that a frozen goal-conditioned policy can be made stronger on long horizons by adding an explicit graph-search wrapper over dataset states.
  - https://ktolnos.github.io/ttgs/

- **Adjacency-Constrained Subgoals** is relevant for the assumption that HRL should restrict high-level goals to locally reachable regions.
  - https://huggingface.co/papers?q=multi-dimensional+goal+space

- **Latent Landmark Graph / HILL** is relevant because it combines contrastive representations with dynamically constructed landmark graphs in goal-conditioned HRL.
  - https://doi.org/10.1007/S11633-023-1482-0

### Hyperbolic Representation Learning

- **Poincare Embeddings** is the core representation-learning reference: hyperbolic space can embed hierarchical symbolic structures with low distortion.
  - https://papers.nips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html

- **Lorentz Model Continuous Hierarchies** improves hyperbolic hierarchy learning and is useful if Poincare optimization is unstable.
  - https://proceedings.mlr.press/v80/nickel18a.html

- **Hyperbolic Representation Learning: Revisiting and Advancing** is a cautionary reference: hyperbolic models do not automatically infer hierarchy unless the objective provides usable hierarchical information.
  - https://openreview.net/forum?id=9CZZ8tIhSv

### Hyperbolic RL

- **Hyperbolic Deep Reinforcement Learning** applies hyperbolic latent representations to deep RL and reports performance/generalization benefits, while also noting optimization challenges.
  - https://openreview.net/forum?id=TfBHFLgv77

- **Understanding and Improving Hyperbolic Deep RL / Hyper++** analyzes instability in hyperbolic RL and proposes fixes for stable training.
  - https://openreview.net/forum?id=7rfdenlP1L

- **Hyperbolic Embeddings for Learning Options in HRL** is an early direct link between hyperbolic embeddings and option discovery.
  - https://www.catalyzex.com/paper/hyperbolic-embeddings-for-learning-options-in

## Relation To The Fundamental Contribution Directions

This idea complements the memory-address thesis:

```text
hyperbolic CRL solves slow global reachability geometry
fast sequence memory solves recent/local associative route evidence
DG sparsity supplies low-interference event addresses
```

So the broader umbrella becomes:

> Long-horizon navigation can be represented by explicit hierarchy, amortized contrastive reachability, curved latent geometry, or fast sparse sequence memory. These are not separate tricks; they are different ways to manage the same reachability-composition problem.
