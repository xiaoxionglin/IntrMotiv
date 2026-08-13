# Iteration 11

## Abstract Draft

Hippocampal sequences are often interpreted through movement, replay, or planning, but sparse-input sequence dynamics may also provide an intrinsic scaffold for discovering spatial structure. We build on Lin, Yiu, and Leibold, who showed that sparse egocentric visual inputs coupled to CA3-like sequence propagation can support navigation and produce place-cell-like internal representations under task reward. Here we ask a complementary question: can this sparse-input sequence motif generate its own training signal for landmark discovery before external rewards define task goals?

The model maps egocentric visual observations through a batch-normalized, thresholded sparse projection inspired by dentate-gyrus activity. Active sparse events enter a CA3-like recurrent sequence state, where each event leaves a short temporal trace through sequence propagation. When a new event occurs, its temporal separation from recently active sequences defines an internal transition-distance signal. This signal turns sequence dynamics into delayed feedback: it can shape which sensory situations become sparse landmarks and can bias behavior toward transitions that expose structure in the environment.

The first selling point is computational: a compact hippocampal motif supplies a candidate intrinsic objective for autonomous landmark discovery, a central problem in robotics and navigation. The second is mechanistic: the model lets us separate sparsification from sequence-based feedback. We compare baseline-shifted feedback rules that encourage or suppress sparse activations, together with population-use terms that distribute activity across available sequence channels. Preliminary analyses indicate that thresholding or suppression alone does not explain useful landmark-like organization; the effect depends on the interaction between sparse event selection, sequence memory, delayed intrinsic feedback, behavioral sampling, and resource balancing.

This work therefore connects hippocampal sequence models to intrinsic motivation: sparse events do not merely represent the environment, but can become candidates for self-organized landmarks whose utility is evaluated by the agent's own sequence dynamics.

## Critic

### Factual Alignment

- Avoids the disliked disclaimer while still staying cautious.
- Strongly aligned with actual research direction.
- "can bias behavior" is acceptable but should be backed by poster experiments or softened if needed.
- "first selling point" and "second selling point" are too explicit for an abstract; useful internally but stylistically blunt.

### Neuroscience Interest

- Good Lin et al. setup.
- Strong mechanistic pitch: sparse events become evaluated by sequence dynamics.
- Neuroscience readers may like the "self-organized landmarks" phrasing.

### Clarity

- Clear and longer than prior drafts.
- Selling points are clear but too labeled.

## Revision Direction

Keep the two selling points but integrate them naturally instead of naming them as "first" and "second".

