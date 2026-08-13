# Iteration 02

## Abstract Draft

The hippocampal formation contains many interacting circuits, but even simplified subcircuits may reveal useful computational principles. We focus on a reduced motif: sparse dentate-gyrus-like input to a CA3-like sequence reservoir. Lin, Yiu, and Leibold previously showed that this sparse-input sequence architecture supports egocentric visual navigation and gives rise to place-cell-like internal representations under task reward. Building on this result, we ask whether the same motif can support intrinsic landmark formation before, or without, reliance on external reward.

In our model, visual observations are mapped to sparse thresholded events that enter a CA3-like shift-register sequence state. The temporal relation between a newly activated event and recently active sequences defines an internal transition-distance signal. This signal provides feedback for shaping sparse landmark allocation and for biasing behavior toward informative transitions. We compare encoder feedback rules that suppress or encourage sparse events, as well as population-use pressures that prevent collapse onto a small subset of sequences.

Preliminary results and simplified mechanistic analyses suggest that activity suppression alone is not enough to align sparse inputs with meaningful latent causes. Instead, landmark formation depends on the closed-loop interaction between thresholding, sequence memory, delayed intrinsic feedback, and resource-balancing terms. The work therefore frames a minimal DG-CA3-inspired circuit not as a complete hippocampal model, but as an overlooked computational primitive for intrinsic motivation and autonomous landmark discovery.

## Critic

### Factual Alignment

- Stronger than iteration 01 on biological caution.
- Correctly includes population-use pressure and toy-analysis limitation.
- "before, or without, reliance on external reward" is safe if framed as a question, not result.

### Neuroscience Interest

- Good opening for neuroscience audience.
- The "overlooked computational primitive" phrase is compelling.
- Could mention hippocampal sequences and place fields more vividly.

### Clarity

- Clean, balanced structure.
- Slightly dense in the methods paragraph.
- "latent causes" may be too ML-specific for the abstract unless explained.

## Revision Direction

Replace "latent causes" with "environmental or sensory structure" and sharpen the neuroscience motivation around hippocampal sequences.

