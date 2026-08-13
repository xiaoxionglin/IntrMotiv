# Iteration 17

## Abstract Draft

Hippocampal sequence activity is often studied as a correlate of movement, replay, or planning. We investigate a complementary possibility: intrinsic sequence propagation may provide a memory buffer for sparse sensory events resembling landmarks. Building on prior work showing that sparse egocentric visual input coupled to a CA3-like sequence generator supports navigation and yields place-cell-like representations under task reward, we ask whether the same motif can organize spatial representations without external reward.

The model maps egocentric visual observations through a batch-normalized, thresholded sparse projection inspired by DG activity. Active events are injected into a CA3-like recurrent sequence state and propagate over short temporal windows, forming an internal memory of recently selected events. Because sequence progression tracks temporal distance between sparse events, it can proxy spatial separation without oracle physical-state information. The agent can use sequence distance to shape dispersed landmarks and favor transitions between landmarks, rather than relying on external reward or raw sensory novelty. For robotics, this addresses a central bottleneck: useful landmarks are formed from the agent's own sensory stream and sequence memory, without privileged coordinates, hand labels, or reward-defined goals. Once formed, such landmarks can compact continuous experience into a graph-like substrate for efficient planning.

Preliminary intrinsic-reward experiments show that encoder feedback baselines strongly alter DG activity and exploration. Punishment-style feedback produced the sparsest DG activations and broader behavioral coverage, suggesting that suppressing excessive DG activity can help landmark-like representations emerge. We evaluate sparse activity maps, sequence usage, activation sparsity, behavioral coverage, and simplified rotation analyses that isolate thresholding and suppression. The emerging picture is that landmark-like organization is not produced by thresholding or suppression alone, but by interactions among sparsification, delayed sequence-based feedback, behavioral sampling, and balanced sequence usage.

This framework links hippocampal sequence theory with intrinsic motivation: sparse sensory events are not only stored or replayed, but can be selected and organized into navigational abstractions through the agent's own sequence dynamics.

## Critic

### Factual Alignment

- Preserves the user's trimmed abstract structure.
- Adds the robotics claim in a constrained way: landmarks are candidate abstractions formed without oracle coordinates, labels, or task reward.
- "Graph-like substrate for efficient planning" is a motivation and intended use, not a claim that final planning performance has already been demonstrated.
- Keeps the caution that thresholding or suppression alone is insufficient.

### Neuroscience Interest

- Maintains the hippocampal sequence framing and the Lin-style prior result.
- Makes the engineering relevance more explicit without displacing the neuroscience story.
- The abstract now has two clear selling points: intrinsic landmark formation and planning-relevant abstraction.

### Clarity

- The robotics sentences are simple and placed immediately after the mechanism they depend on.
- The third paragraph remains concise after the user's trimming.
- Check final character count against the 2500-character limit before submission.

## Revision Direction

Use this if the abstract should foreground the robotics payoff more clearly. If the submission form counts characters strictly, first compress the two added robotics sentences rather than cutting the Jannek-result paragraph.
