# Simpler goal selector and pull gate

[Compare the earlier four layouts](overview.png) · [editable overview](overview.svg)

[Compare the two side-gate layouts](gate_comparison.png) · [editable gate comparison](gate_comparison.svg). These are the current candidates: the blue $\Delta\to\pi$ path stays continuous, a small $\times$ marks where its pull signal is masked, and a separate $j=g$ comparator controls that mask.

These six alternatives depict one selected DG goal, $g$, with two uses. It conditions the policy readout, and it gates the pull signal from a DG transition $\Delta_{i\to j}$ according to its destination $j$. The blue output represents pull eligibility. Each SVG is a native vector drawing at **53 × 20 mm**. The policy's blue wedge and the temporal event's teal rounded frame follow the [architecture v3 SVG](../../../architecture_v3.svg).

| Layout | Editable SVG | Preview | Main difference |
| --- | --- | --- | --- |
| Branch | [SVG](01_branch.svg) | [PNG](01_branch.png) | The goal output branches off the direct route to the policy. |
| Stacked | [SVG](02_stacked.svg) | [PNG](02_stacked.png) | The goal output enters the policy and gate on separate rows. |
| Readout elbow | [SVG](03_labeled_elbow.svg) | [PNG](03_labeled_elbow.png) | Keeps the event below the readout and returns pull via the blue right-angle arrow. |
| DG event elbow | [SVG](04_dg_event_elbow.svg) | [PNG](04_dg_event_elbow.png) | Uses the architecture's colored $i\to j$ event cue within the same elbow layout. |
| Side gate | [SVG](05_side_gate.svg) | [PNG](05_side_gate.png) | Leaves the blue pull route intact; a separate comparator controls the $\times$ node. |
| Two-input gate | [SVG](06_two_input_gate.svg) | [PNG](06_two_input_gate.png) | Also draws the destination $j$ from the DG event into the comparator. |

NG disables the goal selector: the policy has no goal input and the pull gate accepts every destination $j$. UG draws $g$ uniformly from DG units, and FG chooses $g$ using the frontier rule. In both goal-conditioned cases, the comparator supplies the binary mask $\mathbf{1}[j=g]$ to the $\times$ node, so only matching transitions are pull-eligible. The diagrams show the active goal-conditioned circuit; the NG bypass and the two selection rules should be stated in any final figure caption.

The images are candidates only; they have not been inserted into Figure 1. To rebuild from this directory:

```bash
MPLCONFIGDIR=/tmp/cosyne_simple_goal_gate_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_simple_goal_gate.py
```
