# Iteration 12

## Abstract Draft

Hippocampal sequences are commonly interpreted through movement, replay, and planning, yet sequence propagation over sparse inputs may also provide an intrinsic way to decide which sensory events should become landmarks. Lin, Yiu, and Leibold recently showed that sparse egocentric visual input to a CA3-like sequence generator can support visual navigation and produce place-cell-like internal representations under task reward. We extend this line of work from task-driven navigation to autonomous landmark formation: can a sparse-input sequence circuit generate its own feedback for organizing spatial representations?

Our model projects visual observations through a batch-normalized and thresholded sparse layer inspired by dentate-gyrus activity. Active events are injected into a CA3-like recurrent sequence state. Because each event propagates through a short sequence, the system can estimate how recently other events were active. We use this temporal relation as an internal transition-distance signal: newly activated events that occur too close to recent sequence activity can be treated differently from events that expand the internal structure of experience. The signal is delayed, local to the agent's own sequence state, and does not require an external reward location.

This provides two contributions. First, it proposes a biologically motivated intrinsic objective for landmark discovery, connecting hippocampal sequence dynamics with engineering problems in autonomous navigation. Second, it gives a mechanistic test bed for separating the roles of sparse thresholding, activity suppression, sequence memory, and population resource use. We compare feedback baselines that encourage or suppress sparse events and add terms that prevent collapse onto too few sequence channels. Preliminary analyses suggest that suppression alone is insufficient: useful landmark-like organization arises from the closed-loop interaction between sparse event selection, delayed sequence-based feedback, behavioral sampling, and balanced sequence usage.

The result is a compact computational account in which hippocampal-like sequence dynamics help not only to navigate between landmarks, but also to define candidate landmarks intrinsically.

## Critic

### Factual Alignment

- Good. It does not claim final solution.
- "expand the internal structure of experience" is slightly abstract but defensible.
- "does not require an external reward location" is safe for intrinsic objective framing.
- "provides two contributions" is clearer than iteration 11 but still somewhat grant-like.

### Neuroscience Interest

- Strong neuroscience setup.
- Highlights biological plausibility and hippocampal sequence state.
- Good bridge to navigation engineering.

### Clarity

- Clearer explanation of transition distance.
- Strong final sentence.
- Good length and selling points.

## Revision Direction

This is a strong candidate. Make another version with less explicit "two contributions" language and more polished conference prose.

