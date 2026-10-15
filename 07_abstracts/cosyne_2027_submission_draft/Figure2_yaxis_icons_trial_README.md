# Figure 2 trial: icons in the y-axis gutters

[Editable SVG](Figure2_yaxis_icons_trial.svg) · [print PDF](Figure2_yaxis_icons_trial.pdf) · [preview](Figure2_yaxis_icons_trial.png) · [placement manifest](Figure2_yaxis_icons_trial_manifest.json) · [builder](build_figure2_yaxis_icon_trial.py)

This **7.5 × 3.21-inch** trial starts from the selected H-palette [Figure 2](Figure2.pdf). In A–D, each metric icon sits between its own rotated y label and numeric y tick labels, visually linking the drawing to the statistic. The selected `Figure2.svg`/`Figure2.pdf` and `main.tex` remain unchanged.

The result axes, box plots, dots, vertical scales, and sample values are unchanged. The A path-efficiency plot moves 24 pt right. B spatial information moves 8 pt right; B reliability moves 10.75 pt. C/D spatial information moves 10 pt; overlap moves 25.46 pt; coverage and edges move 19.46 pt. The path-efficiency, reliability, and overlap icons retain their selected-figure artwork sizes (23.85, 20.75, and 27.46 pt wide). Spatial information uses 12 pt artwork; coverage and reliable edges use 14 pt. A geometry check on the exported SVG confirms a 2 pt clear gap from each icon to its y label and numeric tick labels. All visible artwork stays within the 540 pt canvas.

The compact arrangement makes the metric association immediate and keeps the figure at its original page width. The DG 1/DG 2 labels under the overlap maps are at their original size, though still visually dense at print size. This trial is an editable layout option, not a replacement for the selected manuscript figure.

Rebuild from this directory with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_figure2_yaxis_icon_trial.py`. The builder reads the selected SVG, uses Inkscape's rendered bounds to position the rotated labels and icons, then writes SVG, PDF, PNG, and a placement manifest. Its result plots are translated only horizontally.
