# Iteration 14

## Abstract Draft

Hippocampal sequence activity is often studied as a correlate of movement, replay, or planning. We investigate a complementary possibility: sequence propagation over sparse sensory events may provide an intrinsic signal for deciding which events should become landmarks. This question builds on Lin, Yiu, and Leibold, who showed that sparse egocentric visual input coupled to a CA3-like sequence generator supports navigation and yields place-cell-like representations under task reward. We ask whether the same sparse-input sequence motif can help organize spatial representations when external rewards are absent or not yet informative.

The model maps egocentric visual observations through a batch-normalized, thresholded sparse projection inspired by dentate-gyrus activity. Active events are injected into a CA3-like recurrent sequence state, where they propagate over short temporal windows. This sequence state acts as an internal memory of recently selected events. When a new event occurs, its temporal distance from recent sequence activity provides a delayed intrinsic feedback signal. The agent can therefore evaluate sparse events by their relation to its own evolving sequence structure, rather than by external reward or raw sensory novelty alone.

This creates a bridge between hippocampal theory and intrinsic motivation. On the neuroscience side, the model suggests a role for sparse-input sequence dynamics in landmark selection, not only in navigation between already-defined locations. On the engineering side, it gives a compact intrinsic objective for autonomous landmark discovery, a central challenge in robotics and navigation. We compare feedback rules that encourage or suppress sparse activations, together with population-use terms that keep sequence channels broadly available. Preliminary analyses suggest that neither thresholding nor activity suppression alone produces well-organized landmark codes. Instead, landmark-like organization depends on interactions among sparsification, delayed sequence-based feedback, behavioral sampling, and balanced sequence usage.

This framework points to hippocampal-like sequence dynamics as a candidate mechanism for self-organizing navigational abstractions from intrinsic experience.

## Critic

### Factual Alignment

- Strong and careful.
- No disliked explicit disclaimer.
- Does not overclaim final success.
- "well-organized landmark codes" is a bit stronger than current evidence; "landmark-like organization" is safer.

### Neuroscience Interest

- Very strong.
- Selling points are integrated naturally.
- Good Lin et al. introduction.

### Clarity

- Clear and polished.
- Within target length.
- One of the best candidates.

## Revision Direction

Make a final variant with "well-organized landmark codes" softened and slightly stronger mention of current readouts/analyses.

