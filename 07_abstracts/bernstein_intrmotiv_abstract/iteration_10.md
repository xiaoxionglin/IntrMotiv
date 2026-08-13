# Iteration 10

## Abstract Draft

Hippocampal sequences are often interpreted through movement, replay, or planning, but intrinsic sequence propagation over sparse inputs may itself help shape spatial representations. We study this possibility with a deliberately reduced DG-CA3-inspired motif: dentate-gyrus-like sparse event selection followed by CA3-like recurrent sequence propagation. This is not a complete model of hippocampal navigation; rather, it isolates a simple circuit computation that may be easy to overlook. Lin, Yiu, and Leibold showed that sparse egocentric inputs coupled to such sequence dynamics can support visual navigation and produce place-cell-like representations under task reward. We ask whether the same motif can also support intrinsic landmark formation.

Visual observations are mapped through a batch-normalized thresholded sparse projection, whose active events enter a CA3-like sequence state. The temporal separation between newly active and recently active sequences defines an internal transition-distance signal. This converts sequence state into delayed feedback for representation learning and, in the full agent, for behavior that samples informative transitions.

We compare baseline-shifted feedback rules that encourage or suppress sparse activations, together with population-use terms that distribute activity across sequence channels. Preliminary analyses indicate that useful landmark-like organization is not explained by thresholding or suppression alone. Instead, it depends on interactions among sparsification, sequence memory, delayed intrinsic feedback, behavioral sampling, and resource balancing. This positions a reduced hippocampal sequence motif as a substrate for self-organizing spatial landmarks and intrinsic exploration, connecting hippocampal theory with current problems in autonomous navigation and intrinsic motivation.

## Critic

### Factual Alignment

- Best balance of caution and contribution.
- Avoids claiming DG-CA3 is complete.
- Accurately presents Lin et al. as prior support for task-reward navigation.
- "behavior that samples informative transitions" should be supported by poster results or softened if needed.

### Neuroscience Interest

- Strong neuroscience entry.
- Good introduction of Lin et al.
- Clear claim that the motif is simplified yet computationally interesting.

### Clarity

- Clear, polished, and coherent.
- Strong final sentence linking neuroscience and robotics/navigation.
- Suitable as a top candidate.

## Revision Direction

Use this as the recommended base. If final behavioral results are weak, replace "for behavior that samples informative transitions" with "for testing how behavior might sample informative transitions."

