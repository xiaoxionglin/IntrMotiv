# Figure 2 trial: icons in the y-axis gutters

[Editable SVG](Figure2_yaxis_icons_trial.svg) · [print PDF](Figure2_yaxis_icons_trial.pdf) · [preview](Figure2_yaxis_icons_trial.png) · [placement manifest](Figure2_yaxis_icons_trial_manifest.json) · [builder](build_figure2_yaxis_icon_trial.py)

This **7.5 × 3.19-inch** trial starts from the selected H-palette [Figure 2](Figure2.pdf). In B–D, each metric icon sits between its own rotated y label and numeric y tick labels, visually linking the drawing to the statistic. Panel A's path-efficiency icon stays beside its plot because its y axis is at the page's left edge. The selected `Figure2.svg`/`Figure2.pdf` and `main.tex` remain unchanged.

The result axes, box plots, dots, vertical scales, and sample values are unchanged. The B plots move 8 pt right; C/D spatial information, coverage, and edges move 10 pt right; C/D overlap moves 16 pt right. The icons are scaled to fit the gutters: 12 pt wide for spatial information, 18 pt for overlap and reliability, and 14 pt for coverage and reliable edges. A geometry check on the exported SVG confirms a 2 pt clear gap on both sides of each B–D icon.

The compact arrangement makes the metric association immediate and keeps the figure at its original page width. Its main cost is smaller artwork: the DG 1/DG 2 labels under the overlap maps are difficult to read at print size. The current right-of-plot placement leaves those labels more legible. This trial is an editable layout option, not a replacement for the selected manuscript figure.

Rebuild from this directory with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_figure2_yaxis_icon_trial.py`. The builder reads the selected SVG, uses Inkscape's rendered bounds to position the rotated labels and icons, then writes SVG, PDF, PNG, and a placement manifest. Its result plots are translated only horizontally.
