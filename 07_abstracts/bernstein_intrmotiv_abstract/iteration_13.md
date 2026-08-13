# Iteration 13

## Abstract Draft

Hippocampal sequence activity is usually discussed in relation to movement, replay, and planning. We ask whether sequence propagation over sparse sensory events can also provide an intrinsic signal for building landmarks. This question follows naturally from Lin, Yiu, and Leibold, who showed that sparse egocentric visual input coupled to a CA3-like sequence generator supports navigation and yields place-cell-like representations under task reward. Here we investigate whether the same sparse-input sequence motif can organize spatial representations when external rewards are absent or not yet informative.

The model receives egocentric visual observations and maps them through a batch-normalized, thresholded sparse projection inspired by dentate-gyrus activity. Active sparse events enter a CA3-like sequence state, where they propagate over short temporal windows. This sequence state provides memory of recent sparse events. When a new event occurs, its temporal distance from recently active sequences becomes an intrinsic feedback signal. Rather than rewarding image novelty directly, the agent evaluates whether a sparse event improves the temporal organization of its internal landmark map.

The work has two main selling points. For neuroscience, it suggests that hippocampal-like sequence dynamics can participate in selecting landmarks, not only in representing or replaying trajectories between them. For robotics and autonomous navigation, it provides a biologically grounded intrinsic motivation objective for discovering compact landmarks without predefined goals. We compare feedback rules that encourage or suppress sparse activations, and population-use terms that keep the sequence reservoir from collapsing onto a small set of channels. Preliminary analyses show that simple thresholding or activity suppression is not enough; landmark-like organization depends on interactions among sparsification, delayed sequence-based feedback, behavioral sampling, and resource balancing.

Together, these results frame sparse-input hippocampal sequence dynamics as a minimal but expressive mechanism for self-organizing navigational abstractions.

## Critic

### Factual Alignment

- Good overall.
- "Preliminary analyses show" should maybe be "suggest" to avoid overclaiming.
- "minimal but expressive" is good but could sound promotional.
- Again uses "two main selling points" explicitly, which the user asked for conceptually but may not belong in final abstract.

### Neuroscience Interest

- Strongest direct statement of neuroscience selling point.
- Good robotics bridge.
- Nice distinction between selecting landmarks and replaying trajectories.

### Clarity

- Very clear.
- "selling points" should be rewritten as prose.

## Revision Direction

Remove explicit "selling points" wording while preserving the two-audience structure.

