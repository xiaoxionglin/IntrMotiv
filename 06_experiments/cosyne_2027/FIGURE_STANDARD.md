# COSYNE 2027 figure standard

Use this as the starting layout for figures intended for the two-page US-letter abstract. The [question index](README.md) sets the figure order, the [color palette](COLOR_PALETTE.md) assigns semantic colors, and [the compact renderer](render_compact_figures.py) is the current executable reference. Larger diagnostic plots remain available in the source analyses.

## Physical layout and typography

| Figure type | Panel size at final placement | Type |
| --- | --- | --- |
| Behavior-expression paired-seed dot plot | 1.5 × 1.8 inches | Arial, 10 pt for titles, axes, ticks, and annotations |
| Denser factorized contrasts or learning curve | Up to 1.9 inches wide per panel; set height from content | Arial, 10 pt |

Render at the **final physical size**. Do not make a large plot and shrink it into the abstract. Verify that Arial resolves to an installed scalable TTF/OTF font; fail explicitly if it does not. For simple dot plots, put the measured quantity and unit on the y-axis and use the top position only for a panel letter. Expand any short axis wording in the caption. Give each panel one comparison, align neighboring axes, and leave enough space for 10 pt tick labels without clipping or overlap. The current behavior-expression pair is 3.0 × 1.8 inches in total.

## Evidence encoding

- In paired-difference plots, training seeds are individual colored dots; a short black segment marks their mean. A gray horizontal line marks zero difference. State the contrast direction in the caption.
- An absolute-value candidate can place gray open-square controls and blue filled-circle experimental values side by side, joined by a thin line for each training seed. Label both groups directly or in a shared legend, begin nonnegative metric axes at zero, and keep the paired-difference source table available.
- Keep the training seed as the independent unit. Average evaluation repeats within that seed; show the actual $n$ for each factor cell. Show missing cells rather than silently dropping an architecture or manipulation.
- For learning curves, use environment steps on the horizontal axis, unsmoothed seed means, and a translucent $\pm1$ SEM band. Define interpolation and support in the caption.
- Use the same factor names, colors, and metric direction across panels. Make signs readable in grayscale through the zero line, marker positions, and direct labels.
- State whether measurements use each policy's own observations, a common observation panel, or shared occupied bins. Do not merge these protocols into one effect estimate.

## Files and review

Save editable SVG and a 300-dpi PNG preview from the same renderer. Preserve vector text, deterministic SVG IDs and metadata, the exact source CSV paths and hashes, font path, physical size, and software versions in the figure manifest. Use only cached analysis tables for layout revisions; a style change should not trigger new telemetry.

Before committing, inspect the exported PNG at its expected display size and check clipped glyphs, tick collisions, empty panels, data hidden by labels, and a sensible zero/reference line. Verify the PNG pixel dimensions against the declared inches and dpi, and check that source-table hashes have not changed. When evidence changes, update the question report and caption together with the plot.
