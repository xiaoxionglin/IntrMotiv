# Iteration 04

## Abstract Draft

How can an agent discover useful landmarks before external task rewards define what matters? We approach this question through a deliberately simplified hippocampal motif: sparse dentate-gyrus-like input to a CA3-like recurrent sequence reservoir. This is not intended as a complete model of hippocampal navigation, but as a minimal circuit whose computational consequences may be underestimated. Lin, Yiu, and Leibold showed that sparse egocentric inputs coupled to hippocampal sequence dynamics can support visual navigation and produce place-cell-like internal representations. We ask whether the same motif can support intrinsic landmark formation.

Our model projects visual observations through a thresholded, batch-normalized sparse layer and injects the resulting events into a CA3-like sequence state. Newly activated events are evaluated by their temporal relation to recently active sequences, producing an intrinsic transition-distance signal. This signal provides delayed feedback for shaping sparse event allocation and can guide behavior toward transitions that reveal useful structure in experience.

We analyze baseline-shifted encoder feedback rules that either broadly encourage or suppress sparse activations, together with population-use terms that prevent collapse onto a few sequence channels. Preliminary results suggest that landmark-like representations require more than thresholding or suppression alone: they emerge from interactions among sparse event selection, sequence memory, delayed feedback, behavioral sampling, and resource balancing. This work links hippocampal sequence circuitry to intrinsic motivation and autonomous landmark discovery, two active problems in robotics and navigation.

## Critic

### Factual Alignment

- Very good caution around simplified motif.
- "can guide behavior" is somewhat prospective but acceptable.
- "landmark-like representations require..." is a bit assertive; could say "appear to require".

### Neuroscience Interest

- Strong opening question.
- Lin et al. background is clear and accessible.
- Robotics/navigation relevance is well placed.

### Clarity

- Clearest so far.
- Good abstract arc: question, prior work, method, preliminary result, relevance.
- Could be slightly more explicit about why temporal distance is meaningful.

## Revision Direction

Refine "transition-distance" explanation in one phrase and soften "require" to "appear to require".

