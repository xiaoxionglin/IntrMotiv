# Source Context For Bernstein Abstract

Date: 2026-06-30

## Purpose

This folder contains iterative abstract drafts for a Bernstein Conference poster on intrinsic landmark formation with a simplified DG-CA3-inspired circuit motif.

The abstract should be forward-looking. The poster is still 2-3 months away, so the text should leave room for additional experiments while remaining accurate about the current state of the work.

## Prior Work To Introduce

Lin, Yiu, and Leibold introduced an egocentric visual navigation model using sparse input to a CA3-like sequence generator.

Key background points to use:

- Sparse DG-like input interacts strongly with a CA3-like recurrent sequence reservoir.
- The architecture can support egocentric visual navigation.
- It produces place-cell-like internal representations.
- It gives a mechanistic interpretation of hippocampal sequences as intrinsic recurrent propagation over sparse inputs, not only as readout of externally supplied trajectories or explicit planning.
- The current arXiv v3 citation is `Lin, Yiu, and Leibold, Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator, arXiv:2510.09951`; it is accepted at ICLR 2026.

Citation-year caution:

- The user refers to this as Lin et al. 2026.
- The locally available report cites an arXiv version from 2025.
- The abstract should use "Lin et al." unless the final formal citation year is confirmed.

## Current Work

The current work asks whether the same simplified sparse-input/sequence-reservoir motif can support intrinsic landmark formation and exploration.

Important framing:

- Do not claim that DG-CA3 alone is the complete hippocampal navigation algorithm.
- Present the circuit as a deliberately reduced and overlooked motif.
- The point is: even this simplified motif may already support useful computations for landmark discovery and intrinsic motivation.
- This is relevant both to neuroscience and to robotics/navigation engineering.

Current algorithmic ingredients:

- Fixed visual frontend.
- Thresholded, batch-normalized DG-like projection.
- Sparse events injected into CA3-like sequence state.
- Intrinsic transition-distance signal from the temporal relation between newly active and recently active sequences.
- Different learning pressure on representation and behavior.
- Population/resource-use pressure to avoid collapse onto too few sequences.

Current mechanistic evidence:

- Full model notes suggest punishment-style encoder shaping produces sparse, landmark-like DG activity and broader exploration.
- Toy thresholded-rotation analyses suggest punishment alone does not explain landmark alignment.
- Useful representations likely depend on the closed-loop interaction between thresholding, sequence memory, delayed intrinsic feedback, policy sampling, and population-use pressure.

## Claims To Avoid

- Avoid saying DG-CA3 is the whole hippocampal algorithm.
- Avoid saying results are final or definitive.
- Avoid saying punishment alone causes good landmark formation.
- Avoid saying the toy model proves the full model.
- Avoid claiming external-reward-free navigation is solved unless final experiments support it.
- Avoid overclaiming biological implementation details such as exact plasticity rules.

## Desired Abstract Tone

The abstract should:

- Introduce Lin et al. to a neuroscience audience.
- Present the current work as a next step from task-reward navigation to intrinsic landmark discovery.
- Appeal to Bernstein computational neuroscience readers.
- Mention robotics/navigation engineering only as a motivating bridge, not as the sole audience.
- Be general enough to survive 2-3 months of continued work.
