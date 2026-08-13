# Iteration 09

## Abstract Draft

Hippocampal sequences are usually interpreted through movement, replay, or planning, but intrinsic sequence propagation over sparse inputs may itself shape spatial representations. We study this possibility with a deliberately reduced DG-CA3-inspired motif: sparse event selection followed by CA3-like recurrent sequence propagation. This is not a complete model of hippocampal navigation; rather, it isolates a simple circuit computation that may be easy to overlook. Lin, Yiu, and Leibold showed that sparse egocentric inputs coupled to such sequence dynamics can support visual navigation and produce place-cell-like representations under task reward. We ask whether the same motif can also support intrinsic landmark formation.

Visual observations are mapped to thresholded sparse events, which activate and propagate through a CA3-like sequence state. The temporal separation between newly active and recently active sequences defines an internal transition-distance signal. This converts sequence state into delayed feedback for representation learning and, in the full agent, for behavior that samples informative transitions.

We compare feedback baselines that encourage or suppress sparse activations and population-use terms that distribute activity across sequence channels. Preliminary analyses indicate that useful landmark-like organization is not explained by thresholding or suppression alone. Instead, it depends on interactions among sparsification, sequence memory, delayed feedback, behavioral sampling, and resource balancing. The work suggests that even a reduced hippocampal sequence motif can provide a substrate for self-organizing spatial landmarks and intrinsic exploration.

## Critic

### Factual Alignment

- Very safe and neuroscience-oriented.
- "in the full agent" again depends on final behavioral evidence but remains cautious.
- Could underplay robotics/navigation engineering relevance.

### Neuroscience Interest

- Strongest neuroscience entry so far.
- Good for Bernstein if the audience is hippocampus/computational neuroscience.
- Less explicit about hot robotics topics.

### Clarity

- Clear and elegant.
- Slightly less concrete about engineering motivation.

## Revision Direction

Combine iteration 08's engineering bridge with iteration 09's neuroscience-first opening.

