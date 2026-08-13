# Iteration 05

## Abstract Draft

Autonomous navigation requires agents to discover landmarks before external rewards specify task goals. We study this problem through a deliberately reduced hippocampal circuit motif: sparse dentate-gyrus-like event selection coupled to a CA3-like recurrent sequence reservoir. The motif is not meant to exhaust hippocampal navigation circuitry; rather, it asks what a minimal sparse-input sequence system can already compute. Previous work by Lin, Yiu, and Leibold showed that such sparse egocentric inputs and hippocampal sequence dynamics can support visual navigation and produce place-cell-like representations. We ask whether the same motif can also support intrinsic landmark discovery.

In our model, visual observations are projected through a batch-normalized thresholded sparse layer and injected as events into a CA3-like sequence state. A newly activated event is evaluated by its temporal separation from recently active sequences, yielding an internal transition-distance signal. This delayed signal provides feedback for sparse event allocation and for behavior that samples informative transitions.

We compare baseline-shifted feedback rules that encourage or suppress sparse activations, plus resource-use pressures that promote coverage of sequence channels. Preliminary analyses suggest that suppression alone does not align sparse inputs with meaningful environmental structure. Instead, landmark-like organization appears to depend on the closed-loop interaction between thresholded sparsification, sequence memory, delayed feedback, behavioral sampling, and population balancing. The results frame a simplified DG-CA3-inspired motif as a biologically grounded primitive for intrinsic motivation and landmark discovery in navigation.

## Critic

### Factual Alignment

- Strong and careful.
- "behavior that samples informative transitions" is still partly aspirational but framed as model goal.
- "environmental structure" is safer than "latent causes".

### Neuroscience Interest

- Good bridge between hippocampus and autonomous navigation.
- Could include place fields more explicitly in the current-work goal, not just Lin et al.
- Strong "minimal sparse-input sequence system" line.

### Clarity

- Clear, concise, and probably conference-ready.
- Some terms are technical but appropriate for Bernstein.

## Revision Direction

Add a sentence that current readouts include DG activity maps/place-field-like landmark maps, without overclaiming final quality.

