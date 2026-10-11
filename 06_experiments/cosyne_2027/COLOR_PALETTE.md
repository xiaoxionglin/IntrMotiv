# COSYNE 2027 color palette

Use color for **one variable per panel**. Keep labels, marker shapes, and line styles so the meaning survives grayscale printing. See [figure standard](FIGURE_STANDARD.md) for size and typography.

| Context | Role | Swatch | Hex | Encoding |
| --- | --- | --- | --- | --- |
| Architecture | C01 | <span style="background-color:#0072B2;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#0072B2` | Blue |
| Architecture | C05 | <span style="background-color:#D55E00;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#D55E00` | Vermilion |
| Architecture | C15 | <span style="background-color:#009E73;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#009E73` | Green; label every architecture curve or group. |
| Mechanism | Push | <span style="background-color:#AE453E;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#AE453E` | Arrows and mechanism labels |
| Mechanism | Pull | <span style="background-color:#236D9D;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#236D9D` | Arrows and mechanism labels |
| Mechanism | CA3 memory | <span style="background-color:#257873;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#257873` | Background structure of the sequence |
| Landmark | DG event 1 | <span style="background-color:#B58C22;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#B58C22` | Ochre; repeat on CA3 event 1. |
| Landmark | DG event 2 | <span style="background-color:#8B6CA8;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#8B6CA8` | Violet; repeat on CA3 event 2. |
| Landmark | DG event 3 | <span style="background-color:#4C9380;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#4C9380` | Sea green; repeat on CA3 event 3. |
| Condition | Normal / own policy | <span style="background-color:#236D9D;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#236D9D` | Filled circle; solid line. |
| Condition | Control / random | <span style="background-color:#75838C;border:1px solid #52606D;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#75838C` | Open square; dashed line. |
| Structure | Pair connector | <span style="background-color:#BBC4CA;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#BBC4CA` | Connect matched seeds. |
| Structure | Text | <span style="background-color:#183043;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#183043` | Main text and outlines. |
| Structure | Grid / unvisited | <span style="background-color:#E2E8EB;border:1px solid #75838C;">&nbsp;&nbsp;&nbsp;&nbsp;</span> | `#E2E8EB` | Background only. |

Number all three landmark events and show their CA3 order by position and arrows as well as color. Keep backgrounds white or nearly white.

**Factorial panels:** When C01/C05/C15 are compared, architecture owns the hue; encode temporal versus constant or intact versus ablated with solid/filled versus dashed/open marks. Within one architecture, the tested push or pull component may use its mechanism color against gray control. For a $2\times2$ C05 plot, use color for the factor named in the question and line or marker style for the crossed factor. Avoid four unrelated condition colors.

**Current adoption:** The architecture curves and compact push/pull schematic already follow their respective color rows. The control-versus-own-policy [behavior candidate](02_behavior_expression.md#candidate-control-and-experimental-values) uses the condition row. The three landmark colors are the target for the next DG-to-CA3 sequence illustration; keep the event numbers visible. Do not recolor archived diagnostic figures merely to match this page.
