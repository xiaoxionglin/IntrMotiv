# Iteration 15

## Abstract Draft

Hippocampal sequence activity is often studied as a correlate of movement, replay, or planning. We investigate a complementary possibility: sequence propagation over sparse sensory events may provide an intrinsic signal for deciding which events should become landmarks. This question builds on Lin, Yiu, and Leibold, who showed that sparse egocentric visual input coupled to a CA3-like sequence generator supports navigation and yields place-cell-like representations under task reward. We ask whether the same sparse-input sequence motif can help organize spatial representations when external rewards are absent or not yet informative.

The model maps egocentric visual observations through a batch-normalized, thresholded sparse projection inspired by dentate-gyrus activity. Active events are injected into a CA3-like recurrent sequence state, where they propagate over short temporal windows. This state acts as an internal memory of recently selected events. When a new event occurs, its temporal distance from recent sequence activity provides a delayed intrinsic feedback signal. The agent can therefore evaluate sparse events by their relation to its own evolving sequence structure, rather than by external reward or raw sensory novelty alone.

This creates a bridge between hippocampal theory and intrinsic motivation. On the neuroscience side, the model suggests a role for sparse-input sequence dynamics in landmark selection, not only in navigation between already-defined locations. On the engineering side, it gives a compact intrinsic objective for autonomous landmark discovery, a central challenge in robotics and navigation. We compare feedback rules that encourage or suppress sparse activations, together with population-use terms that keep sequence channels broadly available. We evaluate emerging sparse activity maps, sequence usage, activation sparsity, and simplified rotation analyses that isolate the effect of thresholding and suppression. Preliminary results suggest that landmark-like organization is not produced by thresholding or suppression alone, but by interactions among sparsification, delayed sequence-based feedback, behavioral sampling, and balanced sequence usage.

This framework points to hippocampal-like sequence dynamics as a candidate mechanism for self-organizing navigational abstractions from intrinsic experience.

## Critic

### Factual Alignment

- Very well aligned with current research and toy analyses.
- The evaluation list is accurate but makes the abstract a little more internal-project-like.
- "behavioral sampling" still assumes behavior analyses will be available; likely acceptable.

### Neuroscience Interest

- Strong.
- Clear neuroscience and engineering selling points.
- Slightly more methods-heavy than iteration 14.

### Clarity

- Clear but dense in the third paragraph.
- Good target length.

## Revision Direction

Use iteration 14 for elegance or iteration 15 for more concrete current-work specificity. Recommended: combine iteration 14's prose with iteration 15's "evaluate emerging..." sentence only if the final poster needs more methodological specificity.

