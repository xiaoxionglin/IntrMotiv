# NG, UG, and FG icon candidates

[Simpler goal-selector diagrams](simple/README.md) · [comparison preview](simple/overview.png). These new alternatives show a single goal selector whose output conditions the policy and gates pull according to the destination in $\Delta_{i\to j}$.

[Compare all three designs](overview.png) · [editable overview](overview.svg) · [PDF](overview.pdf)

Each row is a separate three-icon design. The individual strips are native SVGs at **53 mm × 14.72 mm**, sized to fit the current Figure 1 right side:

| Design | Editable strip | Preview | What it emphasizes |
| --- | --- | --- | --- |
| Two lanes | [SVG](01_two_lanes.svg) | [PNG](01_two_lanes.png) | Policy goal input above; eligible DG events and pull arrows below. |
| Timeline | [SVG](02_event_timeline.svg) | [PNG](02_event_timeline.png) | Matching events can produce pull repeatedly over time. |
| Goal gate | [SVG](03_goal_gate.svg) | [PNG](03_goal_gate.png) | Only the selected DG event passes the pull gate. |

In every design, the gray arrow is the readout's usual input. NG has no goal token at the policy readout; any DG event can supply pull. UG samples a DG goal uniformly and conditions the readout on it; only events from that goal can supply pull. FG uses an open leaf in the event graph to choose the goal; the readout and pull eligibility then work as in UG. The red unit is only a schematic example of a selected goal. The blue arrows mark *eligibility* for pull, not its numerical magnitude.

These are standalone experiments and have not been inserted into the submission figure. The SVGs use native shapes and text; each condition is a named group within its strip. To regenerate:

```bash
MPLCONFIGDIR=/tmp/cosyne_goal_icons_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_goal_policy_icons.py
```
