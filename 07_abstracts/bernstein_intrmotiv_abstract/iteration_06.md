# Iteration 06

## Abstract Draft

Autonomous navigation requires agents to discover landmarks before external rewards specify task goals. We study this problem through a deliberately reduced hippocampal circuit motif: sparse dentate-gyrus-like event selection coupled to a CA3-like recurrent sequence reservoir. This motif is not meant to exhaust hippocampal navigation circuitry; instead, it asks what a minimal sparse-input sequence system can already compute. Lin, Yiu, and Leibold showed that such sparse egocentric inputs and hippocampal sequence dynamics can support visual navigation and produce place-cell-like representations. Here we ask whether the same motif can support intrinsic landmark discovery.

Visual observations are projected through a batch-normalized thresholded sparse layer and injected as events into a CA3-like sequence state. A newly activated event is evaluated by its temporal separation from recently active sequences, yielding an internal transition-distance signal. This delayed signal provides feedback for shaping sparse event maps and for behavior that samples informative transitions.

We compare baseline-shifted feedback rules that encourage or suppress sparse activations, together with resource-use pressures that promote coverage of available sequence channels. We evaluate emerging DG-like activity maps, sequence usage, sparsity, and behavioral coverage. Preliminary analyses suggest that suppression alone does not align sparse inputs with meaningful environmental structure. Instead, landmark-like organization appears to depend on the closed-loop interaction between thresholded sparsification, sequence memory, delayed feedback, behavioral sampling, and population balancing. This positions a simplified DG-CA3-inspired motif as a biologically grounded primitive for intrinsic motivation and landmark discovery.

## Critic

### Factual Alignment

- Safe and broad enough for future poster work.
- "We evaluate behavioral coverage" should only remain if this will definitely be in the poster.
- No unsupported final performance claim.

### Neuroscience Interest

- Good enough for neuroscience crowd.
- "DG-like activity maps" and "sequence usage" add concrete observables.
- Could more explicitly state why this is overlooked or new.

### Clarity

- Slightly long and list-heavy in final paragraph.
- Strong but maybe less elegant than iteration 05.

## Revision Direction

Compress the evaluation list and add one sentence about why this motif is overlooked: it is often treated as an implementation detail rather than an intrinsic objective generator.

