# Iteration 01

## Abstract Draft

Sparse hippocampal input and recurrent sequence dynamics may form a compact computational motif for navigation. Previous work by Lin, Yiu, and Leibold showed that a model with sparse dentate-gyrus-like input to a CA3-like sequence generator can support egocentric visual navigation and produce place-cell-like internal representations. Here we ask whether the same simplified motif can do more than support task-driven navigation: can it also drive intrinsic landmark formation and exploration?

We study an agent in which visual observations are projected through a thresholded, batch-normalized sparse input layer and injected into a CA3-like sequence reservoir. Newly activated sparse events are evaluated by their temporal relation to recently active sequences, yielding an intrinsic transition-distance signal. This signal can shape both representation learning and action selection: the encoder is encouraged to allocate useful, distributed landmark-like events, while the policy is encouraged to move through the environment in ways that reveal informative transitions.

Our preliminary analyses suggest that sparse activity suppression alone is not sufficient to explain landmark formation. Rather, useful DG-like representations appear to depend on the interaction between thresholded sparsification, delayed sequence-based feedback, policy sampling, and population-usage pressure. This work positions a reduced DG-CA3-inspired circuit as a candidate algorithmic primitive for autonomous landmark discovery, connecting hippocampal sequence models with intrinsic motivation problems in robotics and navigation.

## Critic

### Factual Alignment

- Accurate about Lin et al. at a high level.
- Correctly avoids claiming DG-CA3 is complete.
- "The encoder is encouraged" is slightly ambiguous because some variants suppress activity; better to say "shaped".
- "policy is encouraged" may overstate current state if policy results are still developing.

### Neuroscience Interest

- Good hippocampal hook.
- Mentions DG-like input, CA3-like sequence generator, and place-cell-like representations.
- Could more explicitly say the motif is deliberately simplified and overlooked.

### Clarity

- Clear problem and contribution.
- Engineering relevance appears only at the end.
- Needs sharper novelty sentence.

## Revision Direction

Make the minimal-circuit framing more explicit and reduce overcommitment about policy performance.

