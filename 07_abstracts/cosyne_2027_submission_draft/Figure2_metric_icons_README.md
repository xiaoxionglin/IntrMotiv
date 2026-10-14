# Figure 2 metric icons

[Enlarged preview](Figure2_metric_icons_preview.png) · [editable preview](Figure2_metric_icons_preview.svg) · [builder](build_figure2_metric_icons.py)

The three individual vector icons are **10 × 30 mm** each, including their transparent margins. They contain no measured data or effect direction. The double-headed link indicates a comparison, not an increase.

| Icon | SVG | PDF | What it depicts |
| --- | --- | --- | --- |
| Map cosine | [SVG](Figure2_metric_icon_cosine.svg) | [PDF](Figure2_metric_icon_cosine.pdf) | Two DG spatial rate maps. The plotted value is the mean pairwise cosine of active units' maps; greater cosine means more similar spatial patterns. |
| Peak coverage | [SVG](Figure2_metric_icon_coverage.svg) | [PDF](Figure2_metric_icon_coverage.pdf) | Equal numbers of clustered and dispersed dominant DG peaks with Gaussian blur. The plotted value is the effective visited-area fraction of eligible peak density, not a count of peaks. |
| Reliable edges | [SVG](Figure2_metric_icon_edges.svg) | [PDF](Figure2_metric_icon_edges.pdf) | Two directed stored graphs with different counts of reliable edges. The plotted value counts stored reliable directed edges, not executed goal arrivals. |

The icon pairs illustrate what each statistic compares within a condition. In Figure 2C and 2D, the plotted $\Delta$ is then **temporal minus constant credit** for that statistic. The cosine and coverage support differs between the common-observation push comparison and the own-policy pull comparison; see the [story-figure notes](Figure2_story_README.md) and [focused metric definitions](figure2_candidates/08_focused_canvas/README.md).

The PDFs are ready for `\includegraphics[width=10mm,height=30mm,keepaspectratio]{figures/Figure2_metric_icon_cosine.pdf}` (replace the filename for the other icons). Build all SVG, PDF, and PNG assets with `build_figure2_metric_icons.py`; `build_overleaf_package.py` includes them in both Overleaf ZIPs. The preview sheet is enlarged for inspection; use the individual files at print size.
