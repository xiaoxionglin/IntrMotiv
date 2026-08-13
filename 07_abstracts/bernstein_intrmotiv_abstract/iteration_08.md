# Iteration 08

## Abstract Draft

Sparse input to hippocampal sequence circuitry is often treated as a prerequisite for navigation, but it may also help determine what becomes a landmark. We investigate this idea in a deliberately simplified motif: dentate-gyrus-like sparse event selection coupled to a CA3-like recurrent sequence reservoir. Lin, Yiu, and Leibold showed that this sparse-input sequence architecture supports egocentric visual navigation and produces place-cell-like representations under task reward. Here we ask whether the motif can also generate intrinsic objectives for landmark discovery.

The model maps visual observations through a batch-normalized thresholded sparse projection and injects active events into a CA3-like sequence state. When an event newly activates a sequence, its temporal separation from recently active sequences defines an internal transition-distance signal. This signal provides delayed feedback about how the event relates to recent experience, allowing representation learning and action selection to be shaped by the structure of the agent's own sequence dynamics.

We compare baseline-shifted feedback rules that encourage or suppress sparse activations, together with population-use pressures that keep sequence resources available. Current analyses suggest that neither thresholding nor suppression alone is sufficient to produce well-aligned landmark codes. Instead, useful organization appears to require the closed loop between sparse event selection, sequence memory, delayed intrinsic feedback, behavioral sampling, and resource balancing. This positions a reduced DG-CA3-inspired circuit as an algorithmic motif for intrinsic motivation and autonomous landmark discovery in navigation, while leaving open the larger hippocampal mechanisms that complement it.

## Critic

### Factual Alignment

- Very well aligned.
- Explicitly leaves room for larger hippocampal mechanisms.
- Does not overstate current toy results.

### Neuroscience Interest

- Strong and accessible neuroscience frame.
- Good Lin et al. introduction.
- Good relevance to intrinsic motivation.

### Clarity

- Clear, coherent, conference-appropriate.
- "agent's own sequence dynamics" is a useful phrase.
- Could be slightly shorter, but length is reasonable.

## Revision Direction

Make one version more neuroscience-first and less robotics-first, emphasizing hippocampal theories of sequence generation and self-organization.

