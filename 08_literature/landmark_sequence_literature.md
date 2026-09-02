# Landmark Sequence Literature Map

## Core Framing

The useful comparison is not simply "our model also learns landmarks." The sharper framing is:

> Engineering models often learn landmark nodes plus an explicit reachability/value function between pairs of landmarks. A hippocampal alternative may learn sparse landmark-like events and compute pairwise reachability implicitly by sequence propagation, replay, and local temporal associations.

For the current project, the key missing bridge is counterfactual reachability. Raw elapsed time from $g_1$ to $g_2$ under a weak policy is contaminated by wandering, pauses, and loops. The relevant literature suggests a safer decomposition:

```text
local temporal adjacency from experience
        +
compressed replay / graph propagation / predictive map update
        =
implicit estimate of counterfactual landmark reachability
```

## Direct Comparison: Latent Landmarks and Graph Planning

### Zhang, Yang, and Stadie 2021: L3P

- Local copy: [papers/zhang_2021_world_model_as_graph_latent_landmarks.pdf](papers/zhang_2021_world_model_as_graph_latent_landmarks.pdf)
- Project page: https://sites.google.com/view/latent-landmarks/
- arXiv: https://arxiv.org/abs/2011.12491

**What it contributes.** L3P learns a graph-structured world model with latent landmarks as nodes. Edges are reachability estimates distilled from goal-conditioned value functions, and graph search composes short-horizon controllers into long-horizon plans.

**Why it matters here.** This is the clean engineering target for comparison. Their $V(g_1,g_2)$ is explicit, learned from goal-conditioned RL, and used as an edge cost. Your model should not claim to reproduce that value function directly. The more defensible claim is that DG/CA3-like sparse sequences could provide a neuroplausible substrate for landmark formation and implicit reachability.

**Useful contrast.**

```text
L3P:
    learn latent metric / landmarks / explicit pairwise reachability

IntrMotiv:
    learn sparse event candidates / intrinsic sequence associations /
    implicit reachability by propagation or replay
```

### Eysenbach, Salakhutdinov, and Levine 2019: Search on the Replay Buffer

- arXiv: https://arxiv.org/abs/1906.05253
- NeurIPS page/reviews: https://proceedings.neurips.cc/paper/2019/file/5c48ff18e0a47baaf81d8b8ea51eec92-Reviews.html

**What it contributes.** Builds a graph over replay-buffer states and uses learned goal-conditioned distances/policies for long-horizon planning.

**Why it matters here.** It is close to L3P but uses stored states rather than learned landmark centroids. It makes the same conceptual point: long-horizon navigation can be reduced to planning over sparse intermediate states if local reachability is available.

### Faust et al. 2018 / Francis et al. 2020: PRM-RL

- Google Research page: https://research.google/pubs/prm-rl-long-range-robotic-navigation-tasks-by-combining-reinforcement-learning-and-sampling-based-planning/
- Follow-up indoor navigation: https://research.google/pubs/long-range-indoor-navigation-with-prm-rl/

**What it contributes.** Combines local RL controllers with roadmap planning for long-range robot navigation.

**Why it matters here.** It is a robotics reference for the "local controller plus global graph" decomposition. Your model differs because the graph is not sampled geometrically or built from a hand-designed planner; it is proposed to emerge from sparse events and intrinsic sequences.

### Bonnavaud, Albore, and Rachelson 2023: Learning State Reachability as a Graph

- OpenReview: https://openreview.net/forum?id=znb7lccBdq

**What it contributes.** Gradually builds a reachability graph for goal-reaching tasks using a reusable local goal-reaching policy.

**Why it matters here.** Useful support for treating reachability as graph structure rather than as one monolithic value function.

### Demir, Cilden, and Polat 2019: Automatic Landmark Discovery Under Partial Observability

- Cambridge Core: https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/automatic-landmark-discovery-for-learning-agents-under-partial-observability/B81C420EE533D1D19FC5E7E82779DA09

**What it contributes.** Treats landmarks as compact state information useful under partial observability.

**Why it matters here.** Helps connect "landmark" to partial observability rather than only to Euclidean navigation. This is relevant because DG sparse events may be useful precisely when raw egocentric input is partial and ambiguous.

## Differentiable and Neural Planning Mechanisms

This section is useful for implementation choices. The question is how to replace explicit graph search or an explicit $V(g_1,g_2)$ with a computation that can be trained end-to-end, or at least implemented as recurrent neural dynamics.

### Tamar et al. 2016: Value Iteration Networks

- Paper: https://arxiv.org/abs/1602.02867
- NeurIPS listing: https://paperswithcode.com/paper/value-iteration-networks

**What it contributes.** Value iteration is unrolled as a differentiable recurrent/convolutional computation. Instead of calling a planner outside the network, the planner is a module whose repeated updates resemble Bellman backups.

**Why it matters here.** This is the cleanest engineering analogy for CA3 sequence propagation as planning. A landmark graph version would update landmark activities by repeated soft backups:

$u_i^{(k+1)} = \operatorname{softmax}_i \left(r_i + \gamma \sum_j A_{ij} u_j^{(k)}\right)$

or, for distances:

$d_i^{(k+1)} = \operatorname{softmin}_j \left(c_{ij} + d_j^{(k)}\right)$

This makes planning differentiable, but it is still fairly algorithmic. The hippocampal version would replace explicit Bellman iterations with recurrent sequence propagation over learned associations.

### Lee et al. 2018: Gated Path Planning Networks

- arXiv: https://arxiv.org/abs/1806.06408
- PMLR: https://proceedings.mlr.press/v80/lee18e.html

**What it contributes.** Recasts VIN-style planning as a recurrent-convolutional network and replaces brittle value-iteration-style updates with gated recurrent updates.

**Why it matters here.** Gating is relevant because CA3 sequence dynamics are not pure value iteration. A gated recurrent planner is closer to a neural dynamical system: it can accumulate, suppress, or preserve route information over multiple propagation steps.

### Graph Neural Planning / Message Passing

- Graph attention networks: https://mlanthology.org/iclr/2018/velickovic2018iclr-graph/
- E(2)-equivariant graph planning example: https://lhy.xyz/e2-planning/

**What it contributes.** Planning can be expressed as iterative message passing over graph nodes. Each node sends reachability/value information to neighbors, and multiple rounds approximate multi-step planning.

**Why it matters here.** This is probably the most natural engineering form for your landmark model:

$h_i^{(k+1)} = \phi \left(h_i^{(k)}, \sum_j \alpha_{ij}^{(k)} \psi(h_j^{(k)}, A_{ji})\right)$

where nodes are DG landmark events and edges are CA3 sequence associations. This keeps the graph explicit for analysis, while the computation is neural and differentiable.

**Neuroplausible translation.** Replace attention weights $\alpha_{ij}$ with synaptic efficacy, eligibility traces, or sequence-gated recurrent excitation. Then planning becomes spreading activation rather than symbolic graph search.

### Graves et al. 2016: Differentiable Neural Computer

- Nature: https://www.nature.com/articles/nature20101
- DeepMind summary: https://deepmind.google/blog/differentiable-neural-computers/

**What it contributes.** A neural controller reads from and writes to differentiable external memory, and can learn graph traversal and shortest-path-like tasks.

**Why it matters here.** DNC is too engineered to be a direct hippocampal model, but the memory operation is conceptually relevant. CA3/DG could be framed as a biological memory system where sparse events write associations and later route queries read them by content-addressed sequence completion.

### Parisotto and Salakhutdinov 2017: Neural Map

- arXiv: https://arxiv.org/abs/1702.08360

**What it contributes.** Uses a spatially structured neural memory for deep RL in partially observable navigation.

**Why it matters here.** It is a useful engineering comparison if you want to discuss external map memory. Your model differs by using event/sequence structure rather than writing dense spatial features into an explicit 2D memory image.

### Silver et al. 2017: Predictron

- PMLR: https://proceedings.mlr.press/v70/silver17a.html

**What it contributes.** Learns an abstract model that is rolled forward internally for multiple imagined planning depths, accumulating internal rewards and values.

**Why it matters here.** This is close to the "intrinsic sequence" framing. The network need not reconstruct observations; it only needs a latent rollout useful for value prediction. A CA3-like sequence reservoir could be interpreted as a biological analog of abstract latent rollout.

### Farquhar et al. 2018: TreeQN and ATreeC

- OpenReview: https://openreview.net/forum?id=H1dh6Ax0Z
- arXiv: https://arxiv.org/abs/1710.11417

**What it contributes.** Builds differentiable look-ahead trees in learned abstract state space, then backs up predicted rewards and values.

**Why it matters here.** This is a differentiable version of counterfactual branching. It is less hippocampal than sequential replay, but useful if you want a computational bridge from one observed trajectory to many possible routes.

### Weber et al. 2017/2018: Imagination-Augmented Agents

- arXiv: https://arxiv.org/abs/1707.06203
- DeepMind summary: https://deepmind.google/blog/agents-that-imagine-and-plan/

**What it contributes.** Uses learned environment rollouts as "imagination" inputs, then lets a policy network learn how to interpret those rollouts rather than imposing a fixed planning algorithm.

**Why it matters here.** This is a good comparison for imperfect internal models. Your sequence reservoir may not need to be a correct simulator if a downstream system learns how to use its imagined sequences.

### Srinivas et al. 2018: Universal Planning Networks

- PMLR: https://proceedings.mlr.press/v80/srinivas18b.html

**What it contributes.** Differentiable planning is performed by gradient descent in learned latent space using a forward model.

**Why it matters here.** This is useful for robotics audiences because it links learned representations, image goals, and differentiable planning. It is less neuroplausible than replay/message passing, but it supports the argument that latent planning can be learned without hand-designed geometry.

### Amos et al. 2018: Differentiable MPC

- NeurIPS: https://papers.neurips.cc/paper_files/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html
- arXiv: https://arxiv.org/abs/1810.13400

**What it contributes.** Embeds model-predictive control as a differentiable policy class, allowing costs and dynamics to be trained end-to-end.

**Why it matters here.** This is mainly an engineering contrast. MPC gives a principled differentiable optimizer, but it assumes a control-style model. Your model is more naturally an event graph or sequence memory than a continuous-control optimizer.

## Options For This Project

### Option A: Soft Graph Propagation Over Landmarks

Keep landmark nodes explicit but replace hard graph search with differentiable spreading:

$a^{(k+1)} = \sigma \left( \beta A^\top a^{(k)} + b \right)$

The implicit reachability from $g_1$ to $g_2$ is the arrival strength after $K$ steps:

$\hat V(g_1,g_2) = \max_{k \le K} a_{g_2}^{(k)}$

**Best use.** First implementation. It is simple, differentiable, and directly tests whether learned sequence edges support counterfactual route computation.

### Option B: Soft Bellman Backup On The Landmark Graph

Learn edge costs $c_{ij}$ from sequence associations and use a differentiable shortest-path recurrence:

$d_i^{(k+1)} = -\tau \log \sum_j \exp \left(-\frac{c_{ij}+d_j^{(k)}}{\tau}\right)$

As $\tau \to 0$, this approaches a shortest-path update.

**Best use.** If you want a closer analog of $V(g_1,g_2)$ while staying differentiable.

### Option C: Gated Recurrent Sequence Planner

Use a small recurrent module whose hidden state is initialized by $g_1$ and queried at $g_2$:

$h^{(k+1)} = \operatorname{GRU}(h^{(k)}, A^\top h^{(k)})$

**Best use.** If pure Bellman updates are too brittle or too algorithmic. This also resembles CA3 dynamics more than explicit value iteration.

### Option D: Replay-Trained Successor Map

Use replayed landmark sequences to update a successor-like matrix:

$M \leftarrow M + \eta \left(e_t + \gamma M_{g_{t+1},:} - M_{g_t,:}\right)$

Then $M_{ij}$ is an implicit reachability estimate from landmark $i$ to landmark $j$.

**Best use.** Strongest neuroscience bridge, but still policy-dependent unless replay is prioritized, compressed, or biased toward unexplored/goal-relevant routes.

### Option E: Learned Latent Rollout

Train a model that rolls landmark states forward:

$\hat g_{k+1} = F(\hat g_k, a_k)$

and use the rollout to train intrinsic policies or choose subgoals.

**Best use.** More engineering-heavy. Useful later if the simple sequence graph cannot capture action-contingent reachability.

## Recommendation

For the next version, implement **Option A** or **Option B** on top of the existing landmark events:

1. Learn local directed edges from short-window DG/CA3 sequence co-activation.
2. Freeze or slowly update the landmark graph.
3. Run differentiable spreading or soft Bellman backups from $g_1$.
4. Evaluate whether $g_2$ arrival predicts oracle shortest-path distance better than raw behavioral delay.

This directly addresses the weak-policy problem: observed trajectories only provide local edges, while counterfactual route estimates come from internal propagation.

## Counterfactual Routes, Replay, and Predictive Maps

### Stachenfeld, Botvinick, and Gershman 2017: Hippocampus as a Predictive Map

- Nature Neuroscience: https://www.nature.com/articles/nn.4650
- PubMed: https://pubmed.ncbi.nlm.nih.gov/28967910/

**What it contributes.** Frames hippocampal representations as successor-like predictive maps: states are represented by expected future occupancy, not only current location.

**Why it matters here.** This is the strongest conceptual bridge from explicit $V(g_1,g_2)$ to a neural representation. A successor-like code is already an implicit all-goals reachability object, but it can be learned locally from transitions.

**Use cautiously.** Standard successor representations are policy-dependent. For your problem, this is a feature and a limitation: it explains why behavior shapes representation, but it does not automatically solve shortest-path reachability under a bad policy.

### Mattar and Daw 2018: Prioritized Memory Access

- PubMed: https://pubmed.ncbi.nlm.nih.gov/30349103/

**What it contributes.** A normative account of hippocampal replay as prioritized sequential memory access for planning and value update.

**Why it matters here.** This supports the idea that replay can do the counterfactual work that raw behavioral trajectories cannot. Instead of taking observed time as distance, internally generated sequences can propagate local updates across a graph-like representation.

### Sagiv, Akam, Witten, and Daw 2025: Replay for Future Goals / Geodesic Representation

- PubMed: https://pubmed.ncbi.nlm.nih.gov/38496674/
- Neuron article: https://www.sciencedirect.com/science/article/pii/S0896627325007093

**What it contributes.** Extends replay accounts from immediate planning to map-building for possible future goals, using a geodesic/successor-like representation.

**Why it matters here.** This is especially relevant to the user's current concern. It treats replay as a mechanism for building route information before a particular external reward is available, closer to intrinsic landmark discovery.

### Pfeiffer and Foster 2013: Future Paths to Remembered Goals

- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3990408/

**What it contributes.** Shows hippocampal sequences before movement can represent paths from current location toward remembered goals.

**Why it matters here.** Empirical support for prospective sequence generation as route-like computation. This is useful background, but it is goal-directed and not by itself an intrinsic landmark-discovery mechanism.

### Olafsdottir, Bush, and Barry 2018: Replay Review

- UCL Discovery: https://discovery.ucl.ac.uk/id/eprint/10041101/

**What it contributes.** Review of replay in memory, planning, decision making, and reinforcement learning.

**Why it matters here.** Useful as a broad neuroscience citation when introducing replay to a mixed audience.

### Shin, Tang, and Jadhav 2019: Hippocampal-Prefrontal Replay

- DOI: https://doi.org/10.1016/j.neuron.2019.09.012
- PubMed: https://pubmed.ncbi.nlm.nih.gov/31677957/

**What it contributes.** Awake replay supports retrospective and prospective components of spatial learning and decision making.

**Why it matters here.** Helps justify a two-part mechanism: reverse/recent sequence traces update local structure, while forward replay can support future route evaluation.

## Intrinsic Sequences and DG/CA3 Mechanisms

### Lin, Yiu, and Leibold 2025/2026: Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator

- arXiv: https://arxiv.org/abs/2510.09951
- Search mirror: https://papers.cool/arxiv/2510.09951

**What it contributes.** Sparse egocentric input coupled to a hippocampus-inspired sequence generator supports externally rewarded navigation and produces place-cell-like hidden representations. The current arXiv v3 is accepted at ICLR 2026.

**Why it matters here.** This is the immediate predecessor. The new project can be framed as asking whether a similar sparse-input sequence motif can not only support navigation once reward is present, but also help select landmark-like events through intrinsic feedback.

### Yiu and Leibold 2023: Intrinsic and Extrinsic Theta Sequences

- eLife DOI: https://doi.org/10.7554/eLife.86837

**What it contributes.** A DG/CA3 recurrent mechanism can generate intrinsic sequences distinct from movement-driven extrinsic sequences. The paper also links intrinsic sequences to stable landmark decoding.

**Why it matters here.** This is one of the closest mechanistic neuroscience anchors for the idea that intrinsic sequences can provide landmark-like structure without being merely a readout of the animal's current trajectory.

### Leibold 2020: Reservoir of Hippocampal Sequences

- DOI: https://doi.org/10.1016/j.neunet.2020.01.014

**What it contributes.** A hippocampal-sequence reservoir model for navigation in unknown environments.

**Why it matters here.** Supports treating CA3-like sequences as a reservoir that can transform sparse inputs into temporally extended, useful state.

## Intrinsic Motivation and Goal Relabeling

### Andrychowicz et al. 2017: Hindsight Experience Replay

- OpenAI summary: https://openai.com/index/hindsight-experience-replay/
- arXiv: https://arxiv.org/abs/1707.01495

**What it contributes.** Failed trajectories can be relabeled as successful for goals actually reached, enabling learning with sparse rewards.

**Why it matters here.** HER is a computationally explicit way of using counterfactual goals. It is not neuroplausible as written, but it clarifies the problem: learning $V(g_1,g_2)$ depends on asking what would happen under goals other than the one originally pursued.

### Sekar et al. 2020: Plan2Explore

- Project page: https://ramanans1.github.io/plan2explore/
- arXiv: https://arxiv.org/abs/2005.05960

**What it contributes.** Uses a self-supervised world model to plan for expected future novelty and adapt to later tasks.

**Why it matters here.** Useful engineering comparison for intrinsic motivation. Your model can be positioned as a more circuit-level and sequence-based route to intrinsic exploratory structure.

### Fang et al. 2019: Curriculum-Guided HER

- NeurIPS: https://papers.neurips.cc/paper_files/paper/2019/hash/83715fd4755b33f9c3958e1a9ee221e1-Abstract.html

**What it contributes.** Combines hindsight goals with curiosity/diversity curricula.

**Why it matters here.** Useful if you want to discuss how intrinsic motivation can choose which achieved events should become useful goals.

## Practical Takeaways For The Current Model

### 1. Do not equate raw sequence delay with shortest-path value

Observed time between landmarks is policy-dependent:

$\Delta t_{\pi}(g_1,g_2) \neq d^*(g_1,g_2)$

unless the behavioral policy is already close to efficient.

### 2. Let sequences learn local edges, not global distance

Use intrinsic sequences to support local adjacency:

$A_{ij} \leftarrow A_{ij} + \eta \, e_i(t-\tau:t) \, e_j(t)$

where $e_i$ and $e_j$ are sparse event activations. Short windows and repeated evidence should dominate; long wandering trajectories should contribute weakly or not at all.

### 3. Use replay/propagation for counterfactual reachability

Counterfactual $V(g_1,g_2)$ can be replaced by arrival under internally generated sequence dynamics:

$a_{k+1} = f(A^\top a_k), \quad a_0 = e_{g_1}$

and then:

$\hat d(g_1,g_2) = \min \{k : a_k(g_2) > \theta \}$

This is closer to a hippocampal mechanism than directly regressing all-pairs values.

### 4. Add optimism or compression

To reduce contamination from poor trajectories, edge learning should favor short-latency, reliable transitions:

$d_{ij} \approx \operatorname{softmin}_m \Delta t_{ij}^{(m)}$

or strengthen only transitions inside a short temporal eligibility window.

### 5. Strong abstract-level claim

The strongest defensible selling point is:

> Sparse DG-like events and CA3-like intrinsic sequences may provide a circuit-level mechanism for transforming egocentric experience into landmark candidates and directed temporal associations. This offers a neuroplausible counterpart to explicit latent-landmark graph planners, where reachability is computed by value functions.

## Watchouts

- Short-window sequence association gives local adjacency, not guaranteed shortest paths.
- Successor-like maps are policy-dependent unless replay, exploration, or off-policy correction changes the statistics.
- A bad policy can still hide useful edges that have never been experienced.
- The current model should avoid claiming full graph planning unless an explicit replay/propagation readout is implemented.
- The novelty claim should be about the mechanism: sparse thresholded event selection plus intrinsic sequence feedback, not just "landmarks help planning."
