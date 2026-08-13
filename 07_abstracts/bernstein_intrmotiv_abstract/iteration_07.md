# Iteration 07

## Abstract Draft

Sparse input to hippocampal sequence circuitry is often treated as a representational prerequisite for navigation, but it may also provide a mechanism for discovering what should count as a landmark. We investigate this idea in a deliberately simplified motif: dentate-gyrus-like sparse event selection coupled to a CA3-like recurrent sequence reservoir. Lin, Yiu, and Leibold showed that this sparse-input sequence architecture supports egocentric visual navigation and produces place-cell-like representations under task reward. Here we ask whether the motif can also generate intrinsic objectives for landmark discovery.

The model maps visual observations through a batch-normalized thresholded sparse projection and injects active events into a CA3-like sequence state. When an event newly activates a sequence, its temporal separation from recently active sequences defines an internal transition-distance signal. This signal turns sequence dynamics into feedback: sparse events that are redundant, isolated, or useful for structuring experience can be treated differently by the encoder and policy.

We compare baseline-shifted feedback rules that encourage or suppress sparse activations, together with population-use pressures that keep sequence resources available. Current analyses suggest that neither thresholding nor suppression alone is sufficient to produce well-aligned landmark codes. Instead, useful organization appears to require the closed loop between sparse event selection, sequence memory, delayed intrinsic feedback, behavioral sampling, and resource balancing. The result is a minimal DG-CA3-inspired algorithmic motif for intrinsic motivation, linking hippocampal sequence models to autonomous landmark discovery in navigation.

## Critic

### Factual Alignment

- "redundant, isolated, or useful" is conceptually accurate but may imply implemented classification; okay if understood as signal interpretation.
- "require" is a bit strong; "appears to require" is included.
- No excessive biological claim.

### Neuroscience Interest

- Strong opening: sparse input may discover landmarks.
- Nice bridge from representation to intrinsic objective.
- Good use of Lin et al.

### Clarity

- Very clear and polished.
- Slightly abstract in paragraph two; maybe "different feedback" could be explained less categorically.

## Revision Direction

Keep this framing but soften the "redundant, isolated, useful" phrase and mention robotics/navigation engineering in the final sentence.

