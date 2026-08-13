# Iteration 03

## Abstract Draft

Hippocampal sequences are often discussed in relation to trajectories, replay, and planning, but they may also reflect an intrinsic recurrent mechanism for preserving sparse events over time. We investigate this possibility in a deliberately reduced circuit motif: sparse dentate-gyrus-like input to a CA3-like sequence reservoir. Prior work by Lin, Yiu, and Leibold showed that this motif can support egocentric visual navigation and produce place-cell-like representations. Here we ask whether it can also support intrinsic landmark discovery.

The model maps egocentric visual observations through a thresholded and batch-normalized sparse projection, then propagates sparse events through a CA3-like sequence state. When an event newly activates a sequence, its temporal relation to recently active sequences defines an internal transition-distance signal. This signal is used as delayed feedback for shaping the sparse representation and, in the full agent, for biasing behavior toward transitions that improve the internal landmark structure.

We compare feedback baselines that suppress or encourage sparse activity and resource-use terms that promote coverage of available sequence channels. Current analyses indicate that simple suppression of activity, by itself, does not explain useful landmark formation; rather, the effect depends on the interaction between sparse thresholding, sequence memory, delayed feedback, and population balancing. These results suggest that a simplified DG-CA3-inspired motif can already implement ingredients of autonomous landmark formation and intrinsic motivation, linking hippocampal sequence models with current challenges in navigation engineering.

## Critic

### Factual Alignment

- Good on "may" and "investigate".
- "in the full agent" is useful but may imply full behavioral results; still acceptable if poster will include them.
- "Current analyses indicate" is appropriately preliminary.

### Neuroscience Interest

- Stronger neuroscience entry through hippocampal sequences.
- Good bridge to place-cell-like representations.
- Could mention the motif is not a complete hippocampal model to avoid overclaiming.

### Clarity

- Clear and readable.
- "feedback baselines" might be opaque; "baseline-shifted feedback rules" could be more precise.
- Ending is strong.

## Revision Direction

Add one explicit disclaimer that the motif is deliberately simplified and not the full hippocampal circuit.

