# Iteration 16

## Abstract Draft

Hippocampal sequence activity is often studied as a correlate of movement, replay, or planning. We investigate a complementary possibility: intrinsic sequence propagation may provide a memory buffer for sparse sensory events resembling landmarks. Building on work showing that sparse egocentric visual input coupled to a CA3-like sequence generator supports navigation and yields place-cell-like representations under task reward, we ask whether the same motif can organize spatial representations without external reward.

The model maps egocentric visual observations through a batch-normalized, thresholded sparse projection inspired by DG activity. Active events are injected into a CA3-like recurrent sequence state and propagate over short temporal windows, forming an internal memory of recently selected events. Because sequence progression tracks temporal distance between sparse events, it can proxy spatial separation without oracle physical-state information. The agent can use sequence distance to shape dispersed landmarks and favor transitions between landmarks, rather than relying on external reward or raw sensory novelty.

Preliminary intrinsic-reward experiments show that encoder feedback baselines strongly alter DG activity and exploration. Punishment-style feedback produced the sparsest DG activations and broader behavioral coverage, suggesting that suppressing excessive DG activity can help landmark-like representations emerge. Yet punishment alone can under-use the sequence reservoir or fragment fields. Resource-use terms, especially batch-level pressure to recruit inactive sequences, appear important for sequence participation and place-field coverage. We therefore compare feedback rules that encourage or suppress sparse activations together with population-use terms that keep sequence channels broadly available. We evaluate sparse activity maps, sequence usage, activation sparsity, behavioral coverage, and simplified rotation analyses that isolate thresholding and suppression. The emerging picture is that landmark-like organization is not produced by thresholding or suppression alone, but by interactions among sparsification, delayed sequence-based feedback, behavioral sampling, and balanced sequence usage.

This framework links hippocampal sequence theory with intrinsic motivation: sparse sensory events are not only stored or replayed, but can be selected and organized into navigational abstractions through the agent's own sequence dynamics.

## Critic

### Factual Alignment

- Aligns with the user's edited first two paragraphs, with only light compression.
- Accurately incorporates Jannek's report: punishment produced sparse DG activity and broader exploration; batch/resource pressure helped sequence participation and coverage.
- Avoids claiming that punishment alone is sufficient.
- "favor transitions between landmarks" is safer than claiming shortest-path learning is already proven.

### Neuroscience Interest

- Strong focus on hippocampal sequence dynamics as an intrinsic memory buffer for sparse landmark-like events.
- Clearly connects Lin et al.-style task navigation to intrinsic landmark formation.
- Jannek report content gives empirical motivation without making the abstract too engineering-heavy.

### Clarity

- Paragraph three carries the report-specific empirical story while staying below the character limit.
- Clear final takeaway.
- Character length is in the target range.

## Revision Direction

Use this as the current best version if the abstract should foreground Jannek's report results. If length must be reduced, compress the evaluation-list sentence before cutting conceptual framing.
