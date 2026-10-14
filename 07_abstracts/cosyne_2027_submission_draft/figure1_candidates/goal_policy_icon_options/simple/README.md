# Simpler goal selector and pull gate

[Compare both layouts](overview.png) · [editable overview](overview.svg)

These two alternatives depict one selected DG goal, $g$, with two uses. It conditions the policy readout, and it gates the pull signal from a DG transition $\Delta_{i\to j}$ according to its destination $j$. The blue output represents pull eligibility. Both SVGs are native vector drawings at **53 × 20 mm**:

| Layout | Editable SVG | Preview | Main difference |
| --- | --- | --- | --- |
| Branch | [SVG](01_branch.svg) | [PNG](01_branch.png) | The goal output branches off the direct route to the policy. |
| Stacked | [SVG](02_stacked.svg) | [PNG](02_stacked.png) | The goal output enters the policy and gate on separate rows. |

NG disables the goal selector: the policy has no goal input and the pull gate accepts every destination $j$. UG draws $g$ uniformly from DG units, and FG chooses $g$ using the frontier rule. In both goal-conditioned cases, only transitions with $j=g$ are pull-eligible. The diagram shows the active goal-conditioned circuit; the NG bypass and the two selection rules are explained in the overview and should be stated in any final figure caption.

The images are candidates only; they have not been inserted into Figure 1. To rebuild from this directory:

```bash
MPLCONFIGDIR=/tmp/cosyne_simple_goal_gate_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_simple_goal_gate.py
```
