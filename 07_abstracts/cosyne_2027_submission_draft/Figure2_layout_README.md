# Selected Figure 2: icons in the y-axis gutters

[Editable SVG](Figure2.svg) · [print PDF](Figure2.pdf) · [preview](Figure2.png) · [placement manifest](Figure2_layout_manifest.json) · [builder](build_figure2.py)

This **7.5 × 3.21-inch** selected figure starts from the [H-palette icon layout](figure_palette_candidates/priority_rgb_orange_violet_figure2.svg). In A–D, each metric icon sits between its own rotated y label and numeric y tick labels, visually linking the drawing to the statistic. `main.tex` uses this layout through `figures/Figure2.pdf` in the Overleaf package.

This layout builder keeps the result axes, box plots, dots, vertical scales, and sample values of its H-palette input unchanged. Panel D's representation plots include audited NG five-seed temporal-minus-constant differences alongside UG and FG; its NG reliable-edge position remains N/A. The A path-efficiency plot moves 24 pt right. B's heading, schematic, and result columns move 15.13 pt right as a unit, aligning its last plot's right edge with C's (D ends 0.69 pt earlier). Within B, spatial information moves another 8 pt and reliability another 10.75 pt to make room for their icon gutters. C/D spatial information moves 10 pt; overlap moves 25.46 pt; coverage and edges move 19.46 pt. The path-efficiency, reliability, and overlap icons retain their selected-figure artwork sizes (23.85, 20.75, and 27.46 pt wide). Spatial information uses 12 pt artwork; coverage and reliable edges use 14 pt. A geometry check on the exported SVG confirms a 2 pt clear gap from each icon to its y label and numeric tick labels. All visible artwork stays within the 540 pt canvas.

The C and D spatial-information y labels are positioned 2 pt below their original vertical positions. Other labels and plot geometry retain their earlier positions.

The compact arrangement makes the metric association immediate and keeps the figure at its original page width. The DG 1/DG 2 labels under the overlap maps are at their original size, though still visually dense at print size. The [right-of-plot icon layout](Figure2_story_icons_README.md) remains available as an earlier variant.

Rebuild from this directory with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_figure2.py`. The builder reads the fixed H-palette preview, uses Inkscape's rendered bounds to position the rotated labels and icons, then writes SVG, PDF, PNG, and a placement manifest. Its result plots are translated only horizontally. Rebuilding the palette previews first also regenerates this selected layout.
