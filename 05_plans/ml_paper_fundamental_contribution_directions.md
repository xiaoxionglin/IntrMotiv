# ML Paper: Fundamental Contribution Directions

Date: 2026-07-31

This note ignores the Bernstein abstract and focuses on what could become a sufficiently distinct ML/neuroscience-workshop paper. The key question is not whether the current ideas are intuitive. They are. The question is whether they can be stated as a contribution that is not already covered by latent-landmark planning, graph-based subgoal discovery, successor representations, intrinsic motivation, differentiable planning, or hippocampal replay models.

## Short Diagnosis

The broad package is not unique enough:

```text
landmarks + intrinsic motivation + replay/rollouts + graph planning
```

This overlaps with L3P, SoRB, PRM-RL, successor options, Laplacian option discovery, InfoBot, HER-style relabeling, differentiable planners, memory-augmented agents, and hippocampal replay models.

The unique contribution must be more structural:

> Landmark abstractions are not first discovered and then planned over. Instead, sparse events are selected because they are useful write/read addresses for sequence memory.

That sentence is much closer to a fundamental contribution.

## Prior-Art Pressure Points

### Existing Landmark And Subgoal Literature

Several lines already cover "find useful intermediate states":

- L3P learns latent landmarks and an explicit pairwise reachability function $V(g_1,g_2)$ for graph planning.
- SoRB builds a graph over replay-buffer states and uses goal-conditioned distances for planning.
- Graph/roadmap RL methods combine local controllers with global planning.
- Option discovery finds bottlenecks, access states, eigenpurposes, or successor-based subgoals.
- InfoBot identifies decision states by where goal information matters for a goal-conditioned policy.

So "we discover landmarks for planning" is not enough.

### Existing Sequence And Replay Literature

Several lines already cover "use replay or rollouts":

- Successor representations and predictive maps encode future occupancy.
- Prioritized replay and replay-for-map-building models connect hippocampal replay to planning and cognitive map construction.
- Recurrent planning models sample imagined trajectories.
- Differentiable planners unroll value iteration, graph message passing, or latent rollouts.

So "we use sequences for planning" is not enough.

### Existing Fast-Weight Literature

Fast weights, differentiable plasticity, and linear-transformer-style associative memories already show that temporary outer-product memories can be written and queried efficiently:

$F_t \leftarrow \lambda F_{t-1} + \eta k_t v_t^\top$

So "fast weights implement memory" is not enough.

## Fundamental Direction 1: Event Selection As Memory Address Learning

### Core Claim

Most ML landmark methods treat landmarks as useful states. The stronger claim is:

> A good landmark is a sparse event that functions as a reliable address for writing and querying sequence memory.

This reframes the problem from state abstraction to memory-address learning.

### Mechanism

The agent observes high-dimensional input $x_t$, produces sparse DG-like events $e_t$, and writes a sequence memory:

$F_t = \lambda F_{t-1} + \eta e_t e_{t+1}^\top$

or with a trace:

$F_t = \lambda F_{t-1} + \eta r_t e_t^\top$

where $r_t$ summarizes recent sparse events.

A landmark is good if it improves future sequence completion:

$a_{k+1} = \sigma(F_t^\top a_k)$

not if it merely reconstructs $x_t$, clusters observations, or maximizes novelty.

### Why This Could Be Unique

The contribution is not a new planner. It is a new criterion for event abstraction:

```text
select events that make associative route memory readable
```

This is different from:

- reconstruction bottlenecks: preserve sensory information;
- contrastive RL representations: preserve controllable or temporally predictive information;
- landmark planners: choose graph nodes after learning reachability;
- option discovery: identify useful subgoals in an existing or learned transition graph.

### Testable Result

Compare event representations learned by:

- reconstruction or contrastive prediction;
- novelty / count-based intrinsic reward;
- latent clustering;
- bottleneck/decision-state methods;
- sequence-memory address objective.

Measure:

- retrieval purity: does cueing one event retrieve coherent successors?
- route composition: do recurrent steps predict multi-step reachability?
- interference: do unrelated routes superpose destructively?
- downstream sparse-goal navigation.

### Risk

Reviewers may still call this "representation learning for planning." To avoid that, the paper needs a metric that only your framing predicts well: memory-address quality.

Possible metric:

$Q_{\mathrm{addr}} = \mathbb{E}_{i,j}[\mathrm{AUC}(\hat R_{ij}, R^*_{ij})] - \alpha \, \mathrm{Interference}(F)$

where $\hat R_{ij}$ is reachability from recurrent memory propagation and $R^*_{ij}$ is oracle reachability.

## Fundamental Direction 2: The Weak-Policy Counterfactual Gap

### Core Claim

Raw temporal proximity under behavior policy $\pi$ is not a reliable landmark distance:

$\Delta t_\pi(g_1,g_2) \neq d^*(g_1,g_2)$

This gap is fundamental for unsupervised landmark discovery because a weak policy generates crooked, looping trajectories. Many landmark methods implicitly rely on observed transition statistics becoming meaningful enough.

### Proposed Contribution

Define and study the **weak-policy counterfactual gap**:

$G_\pi(i,j) = \Delta t_\pi(i,j) - d^*(i,j)$

or rank distortion:

$D_\pi = 1 - \rho(\Delta t_\pi, d^*)$

Then show that local sequence edges plus recurrent composition reduce this gap:

```text
bad trajectories give unreliable global delays
but still contain recoverable local adjacency fragments
```

The model should learn:

$A_{ij} \leftarrow A_{ij} + \eta \, \mathbf{1}[\Delta t(i,j) \le H]$

or an optimistic edge statistic:

$c_{ij} \approx \operatorname{softmin}_m \Delta t_{ij}^{(m)}$

Then infer counterfactual reachability by recurrent propagation:

$a_{k+1} = \sigma(A^\top a_k)$

### Why This Could Be Unique

This is a clean problem statement that ML reviewers can evaluate. It also distinguishes the work from papers that use temporal distance or goal-conditioned value estimates after collecting stronger data.

The claim becomes:

> In weak-policy exploration, global temporal distance is a biased estimator of reachability. Sparse sequence memory can recover usable counterfactual structure by composing local adjacency fragments.

### Testable Result

Construct mazes where random/weak policies often loop:

- open arena with visual aliasing;
- four rooms with bottlenecks;
- cul-de-sac maze;
- loop maze where long observed delay hides a short shortcut;
- egocentric visual observations with no coordinate input.

Compare:

- raw time-to-next-landmark;
- average temporal distance;
- contrastive temporal distance;
- explicit goal-conditioned $V(g_1,g_2)$ if trained;
- graph shortest path over oracle states;
- sequence-memory recurrence over learned events.

Key plot:

```text
x-axis: policy quality / trajectory tortuosity
y-axis: rank correlation with oracle shortest path
```

If your method degrades slower than raw temporal baselines, that is a real contribution.

### Risk

If explicit graph methods over learned landmarks win easily, the paper must emphasize compute/biology/sample constraints:

```text
we are not beating full graph search;
we are replacing all-pairs value learning with a local recurrent estimator
```

## Fundamental Direction 3: Planning As Sparse Recurrent Sampling, Not Search

### Core Claim

Planning need not be full search over landmarks or differentiable value iteration. A more realistic and cheaper mechanism is:

```text
sparse cue retrieval + few biased recurrent rollouts
```

This is not just an implementation detail. It changes what representation should be learned. Good landmarks are those that support useful low-budget rollouts.

### Mechanism

From current event $e_c$, sample $K$ short rollouts:

$g_{h+1}^{(k)} \sim p(g \mid g_h^{(k)}, F, b_{\mathrm{goal}}, b_{\mathrm{novelty}})$

with small $K$ and horizon $H$:

$K \in \{4,8,16\}, \quad H \in \{4,8,12\}$

Then choose actions or subgoals from the best rollout according to a simple score:

$S(\tau) = \lambda_g \, \mathrm{GoalProgress}(\tau) + \lambda_n \, \mathrm{Novelty}(\tau) + \lambda_c \, \mathrm{Confidence}(\tau)$

### Why This Could Be Unique

Many ML planners emphasize optimality or differentiability. Your paper could emphasize a different computational regime:

```text
planning under a tiny internal simulation budget
```

The representation is evaluated by how much planning benefit it gives per rollout, not by asymptotic optimality.

### Testable Result

Plot performance as a function of rollout budget:

```text
rollout budget K x H vs downstream navigation success
```

The strong result would be:

```text
with sparse sequence-memory landmarks, K=4 to 8 gives most of the benefit
of much heavier graph planning
```

### Risk

This could be seen as "sampling from a model." To make it distinct, the rollouts should be sequence-memory completion over event codes, not image-level or action-level world-model rollout.

## Fundamental Direction 4: Sparse Codes As Anti-Interference For Fast Weights

### Core Claim

Sparsity is not just for biological plausibility or regularization. It is necessary for reliable fast-weight route memory because dense events create destructive superposition in outer-product memories.

### Mechanism

Fast weights store transitions by superposition:

$F = \sum_t e_t e_{t+1}^\top$

Query error depends on cross-talk:

$F^\top e_i = e_{i+1} + \sum_{t \neq i} e_{t+1}(e_t^\top e_i)$

If sparse DG-like events reduce $e_t^\top e_i$ for unrelated events, they reduce interference.

### Why This Could Be Unique

This gives a fundamental role to DG sparsification in ML terms:

```text
sparsity protects temporary associative planning memory from cross-talk
```

That is stronger than saying "sparsity helps representation learning."

### Testable Result

Systematically vary:

- sparsity threshold $\theta$;
- event dimensionality;
- event overlap;
- number of stored transitions;
- sequence branching factor.

Measure:

- retrieval error;
- route completion accuracy;
- branching ambiguity;
- downstream planning success;
- memory capacity before collapse.

Expected signature:

```text
there is an intermediate sparsity regime:
too dense -> cross-talk;
too sparse -> disconnected graph;
middle -> best route memory
```

This could turn the existing threshold/rotation toy work into a principled fast-memory-capacity story.

### Risk

Sparse associative memory is old. The novelty must be the link to learned landmark selection and route planning under egocentric RL, not just capacity analysis.

## Fundamental Direction 5: Suppressive Feedback As Repulsive Code Allocation

### Core Claim

The "punishment" case may be better understood as a repulsive rotation rule for fixed-norm DG projections:

```text
active units are rotated away from currently over-selected inputs
```

This is not merely sparsity regularization. It is an online code-allocation mechanism.

### Mechanism

With fixed-norm rows $w_i$, batch normalization, and thresholding:

$e_i(x) = [\mathrm{BN}(w_i^\top x) - \theta]_+$

Punishing active units creates a tangent-space update away from the currently active observation manifold:

$w_i \leftarrow \operatorname{Proj}_{\|w_i\|=1}(w_i - \eta e_i x)$

This can be interpreted as preventing a few easy-to-activate directions from monopolizing event addresses.

### Why This Could Be Unique

The contribution would be:

> Baseline-subtracted suppressive feedback under fixed-norm thresholded projections performs online landmark-address allocation by repelling overused event detectors.

This is more specific than "anti-Hebbian learning creates sparse codes."

### Testable Result

Tie the toy phenomenon to navigation:

- Does punishment reduce event-address monopolization?
- Does it improve fast-weight retrieval by reducing transition-memory cross-talk?
- Does it improve landmark graph coverage without making events too rare?
- Does it improve downstream goal transfer?

The most important metric is not density. It is:

$\mathrm{Interference}(F) = \mathbb{E}_{i \neq j}|e_i^\top e_j|$

or route retrieval error after storing transitions.

### Risk

If punishment only lowers activity but does not improve memory-readout quality, it should remain a side result.

## Fundamental Direction 6: Intrinsic Motivation As Representation-Utility Maximization

### Core Claim

Curiosity often rewards prediction error, novelty, empowerment, or information gain. Your model could instead define intrinsic motivation as:

> seek experience that improves the utility of the agent's own sequence representation for future planning.

This is a different intrinsic reward target:

$r_t^{\mathrm{int}} = \Delta U_{\mathrm{plan}}(E, F)$

where $E$ is the event encoder and $F$ is sequence memory.

### Possible Utility Functions

Reachability improvement:is 

$U_{\mathrm{reach}} = \operatorname{corr}(\hat R, R^*)$

In real agents, $R^*$ is unavailable, so use internal proxies:

- improved consistency of recurrent arrivals across repeated experience;
- reduced route-memory interference;
- increased connectedness without event collapse;
- improved prediction of future sparse events;
- improved controllability of event transitions.

One possible internal objective:

$U(E,F) = \mathrm{Connectivity}(F) - \alpha \mathrm{Interference}(F) - \beta \mathrm{Collapse}(E)$

### Why This Could Be Unique

This reframes intrinsic motivation from "visit novel states" to "improve the internal route data structure." That is both ML-relevant and neuroscience-relevant.

The contribution would be strongest if you show novelty can fail:

```text
novelty explores visually complex but topologically useless regions;
sequence-memory utility seeks reusable event transitions
```

### Risk

If the utility depends on oracle topology, it becomes supervised. The challenge is to design a proxy that is internal and still works.

## Fundamental Direction 7: Event Graph Geometry From Egocentric Views

### Core Claim

The model learns a graph-like geometry from egocentric sensory streams without coordinate supervision:

```text
sensory stream -> sparse event graph -> sequence geometry
```

The geometry is not a Euclidean embedding first and graph second. It is an event-transition geometry produced by sparse sequence memory.

### Why This Could Fit NeurReps

NeurReps emphasizes geometry, topology, and representation structure. The paper could analyze whether learned event dynamics preserve environment topology:

- graph geodesics correlate with physical geodesics;
- bottlenecks become high-betweenness event nodes;
- loops in the environment become cycles in the event graph;
- sequence distances align with successor-like distances;
- event codes remain stable under egocentric view changes.

### Possible Metric

Topological preservation:

$\rho_{\mathrm{geo}} = \operatorname{corr}(d_F(i,j), d_{\mathrm{env}}(i,j))$

Cycle recovery:

$\beta_1(G_E) \approx \beta_1(G_{\mathrm{env}})$

Bottleneck recovery:

$\operatorname{corr}(\mathrm{Betweenness}_E, \mathrm{Betweenness}_{\mathrm{env}})$

### Why This Could Be Unique

Many RL papers measure task return. Many neuroscience papers show place-like fields. The more fundamental contribution is showing that a sparse sequence memory learns a useful **topological data structure** from egocentric experience.

### Risk

If this becomes only an analysis section, it is not enough. It must feed back into algorithmic claims: this geometry makes planning or transfer better.

## Candidate Paper Theses

### Thesis A: Memory-Address Learning

> We introduce memory-address learning for navigation: sparse event codes are trained not to reconstruct observations or optimize external rewards, but to serve as reliable write/read addresses for fast sequence memory. This yields landmarks that support route completion and downstream goal transfer.

This is the most fundamental thesis.

### Thesis B: Weak-Policy Counterfactual Recovery

> Under weak exploration, temporal proximity is a biased estimate of reachability. We show that sparse local sequence memories can compose short adjacency fragments into useful counterfactual route estimates, reducing the weak-policy counterfactual gap.

This is the cleanest ML problem formulation.

### Thesis C: Sparsity Protects Planning Memory

> DG-like sparsification improves planning not simply by producing selective features, but by reducing cross-talk in fast associative transition memory. This creates an intermediate sparsity regime where route memory is maximally useful.

This is the cleanest mechanistic/theoretical thesis.

### Thesis D: Intrinsic Motivation Builds Data Structures

> Intrinsic motivation can be formulated as improving the agent's internal route data structure. Sparse sequence feedback drives exploration toward experiences that make event memory more connected, less interfering, and more useful for future goals.

This is the broadest thesis, but also the riskiest because it is harder to make precise.

## Recommended Stack

The strongest paper may combine A, B, and C:

```text
memory-address learning
    explains what landmarks are for

weak-policy counterfactual gap
    explains why existing temporal/reachability shortcuts are insufficient

sparsity protects fast route memory
    explains why the DG-like mechanism matters
```

The paper should not try to solve all of intrinsic motivation, hierarchical RL, and hippocampal planning. A narrower paper with a strong conceptual kernel is more likely to survive review.

## Minimal Experiments For A Strong First Paper

### Experiment 1: Fast-Memory Capacity

Synthetic event sequences with controlled sparsity and branching.

Show:

- dense codes fail by cross-talk;
- overly sparse codes fail by disconnectedness;
- thresholded sparse codes maximize route completion.

This validates Thesis C.

### Experiment 2: Weak-Policy Maze

Use egocentric observations and a weak exploratory policy.

Show:

- raw temporal distance is distorted by tortuous trajectories;
- learned local sequence memory recovers better reachability;
- recurrent propagation predicts oracle graph distance better than raw delay.

This validates Thesis B.

### Experiment 3: Landmark Learning Objective

Compare event encoders:

- random sparse projections;
- autoencoder / contrastive predictive representation;
- latent clustering;
- novelty-driven events;
- sequence-memory-address objective;
- optionally explicit L3P-like reachability baseline.

Show:

- better retrieval purity;
- better graph topology;
- better downstream sparse-goal navigation.

This validates Thesis A.

### Experiment 4: Realistic Visual Navigation

Use DeepMind Lab or MiniGrid/Procgen-style egocentric navigation.

Show:

- faster adaptation to new goals after reward-free exploration;
- useful landmark fields;
- few-rollout planning benefit;
- ablation of thresholding, fixed-norm rotation, fast weights, and intrinsic feedback.

This makes the paper relevant beyond toy mazes.

## What To Avoid

- Do not sell this as "a hippocampal model that does landmark planning." Too broad and too easy to dismiss.
- Do not make $V(g_1,g_2)$ the central object. That invites direct comparison to goal-conditioned value learning.
- Do not claim optimal planning. Claim low-budget recurrent route inference.
- Do not make punishment/encouragement the headline until it improves memory utility, not just sparsity.
- Do not depend on coordinate supervision or oracle topology except for evaluation.

## Working Title Candidates

- Sparse Events as Memory Addresses for Route Learning
- Intrinsic Landmark Discovery by Fast Sequence Memory
- Landmark Learning Without Pairwise Reachability
- Sparse Sequence Memory for Weak-Policy Route Inference
- DG-Inspired Sparsity Reduces Cross-Talk in Fast Route Memory

## Most Defensible Unique Contribution

The best single-sentence contribution is:

> We show that sparse event codes can be learned as memory addresses for fast sequence transition storage, allowing an agent to recover useful counterfactual route structure from weak exploratory trajectories without explicit all-pairs reachability learning.

This is specific enough to be different from standard landmark discovery, explicit graph planning, successor representations, and generic replay.

## References To Keep Nearby

- L3P: latent landmarks plus explicit reachability, https://arxiv.org/abs/2011.12491
- SoRB: graph over replay buffer, https://arxiv.org/abs/1906.05253
- Laplacian option discovery: https://proceedings.mlr.press/v70/machado17a.html
- Successor options: https://openreview.net/forum?id=Byxr73R5FQ
- InfoBot: decision states via information bottleneck, https://research.google/pubs/infobot-structured-exploration-in-reinforcementlearning-using-information-bottleneck/
- Predictive map: https://pubmed.ncbi.nlm.nih.gov/28967910/
- Fast weights / linear transformers: https://proceedings.mlr.press/v139/schlag21a.html
- Differentiable plasticity: https://proceedings.mlr.press/v80/miconi18a.html
- Replay and map building: https://www.sciencedirect.com/science/article/pii/S0896627325007093

