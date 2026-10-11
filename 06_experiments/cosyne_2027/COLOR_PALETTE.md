# COSYNE 2027 color palette

Use color for **one variable per panel**. Keep labels, marker shapes, and line styles so the meaning survives grayscale printing. See [figure standard](FIGURE_STANDARD.md) for size and typography.

![Architecture, mechanism, landmark, condition, and neutral swatches](COLOR_PALETTE.svg)

| Context | Role | Color | Encoding |
| --- | --- | --- | --- |
| Architecture comparison | C01 / C05 / C15 | `#0072B2` blue / `#D55E00` vermilion / `#009E73` green | Label every curve or group. |
| Mechanism schematic | Push / pull / CA3 memory | `#AE453E` red / `#236D9D` blue / `#257873` teal | Use for arrows and mechanism labels, not architecture identities. |
| Illustrative landmarks | DG events 1 / 2 / 3 | `#B58C22` ochre / `#8B6CA8` violet / `#4C9380` sea green | Repeat each event color on its corresponding CA3 sequence segment. |
| Condition comparison | Normal or own policy / control or random | `#236D9D` blue / `#75838C` slate gray | Normal: filled circle, solid line. Control: open square, dashed line. |
| Neutral structure | Text / pair connector / grid or unvisited area | `#183043` / `#BBC4CA` / `#E2E8EB` | Pair matched seeds with light-gray connectors. Keep grid gray for backgrounds. |

Number all three landmark events and show their CA3 order by position and arrows as well as color. Keep backgrounds white or nearly white.

**Factorial panels:** When C01/C05/C15 are compared, architecture owns the hue; encode temporal versus constant or intact versus ablated with solid/filled versus dashed/open marks. Within one architecture, the tested push or pull component may use its mechanism color against gray control. For a $2\times2$ C05 plot, use color for the factor named in the question and line or marker style for the crossed factor. Avoid four unrelated condition colors.

**Current adoption:** The architecture curves and compact push/pull schematic already follow their respective color rows. The control-versus-own-policy [behavior candidate](02_behavior_expression.md#candidate-control-and-experimental-values) uses the condition row. The three landmark colors are the target for the next DG-to-CA3 sequence illustration; keep the event numbers visible. Do not recolor archived diagnostic figures merely to match this page.
