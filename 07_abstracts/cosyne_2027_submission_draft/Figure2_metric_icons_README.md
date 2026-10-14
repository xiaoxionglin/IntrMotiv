# Figure 2 metric icons

[Enlarged preview](Figure2_metric_icons_preview.png) · [editable preview](Figure2_metric_icons_preview.svg) · [builder](build_figure2_metric_icons.py)

Map cosine, peak coverage, and reliable edges are **10 × 30 mm** each; reliability is **10 × 20 mm**, including transparent margins. The upper example in each icon has the higher value of that statistic, and the arrow points up to match the plot's y-axis direction. The icons contain no measured data or claim that either intervention increases a statistic.

| Icon | SVG | PDF | What it depicts |
| --- | --- | --- | --- |
| Map cosine | [SVG](Figure2_metric_icon_cosine.svg) | [PDF](Figure2_metric_icon_cosine.pdf) | Two **pairs** of DG place fields without outline boxes: similar fields above and different fields below. The plotted value is the mean pairwise cosine of active units' maps; greater cosine means more similar spatial patterns. |
| Peak coverage | [SVG](Figure2_metric_icon_coverage.svg) | [PDF](Figure2_metric_icon_coverage.pdf) | Equal numbers of dispersed peaks above and clustered peaks below, with Gaussian blur. The plotted value is the effective visited-area fraction of eligible peak density, not a count of peaks. |
| Reliable edges | [SVG](Figure2_metric_icon_edges.svg) | [PDF](Figure2_metric_icon_edges.pdf) | Two directed stored graphs with more reliable arrows above than below. The plotted value counts stored reliable directed edges, not executed goal arrivals. |
| DG reliability | [SVG](Figure2_metric_icon_reliability.svg) | [PDF](Figure2_metric_icon_reliability.pdf) | One DG spatial response compared between two episode-disjoint halves: matching maps above and mismatched maps below. The plot uses split-half map correlation. |

The icon pairs illustrate what each statistic compares within a condition. In Figure 2C and 2D, the plotted $\Delta$ is **temporal minus constant credit** for that statistic; in Figure 2B, reliability's $\Delta$ is **learned policy minus random actions**. The cosine and coverage support differs between the common-observation push comparison and the own-policy pull comparison; see the [story-figure notes](Figure2_story_README.md) and [focused metric definitions](figure2_candidates/08_focused_canvas/README.md).

The PDFs are ready for `\includegraphics[width=10mm,height=30mm,keepaspectratio]{figures/Figure2_metric_icon_cosine.pdf}` (replace the filename for the other icons; reliability has 20 mm height). Build all SVG, PDF, and PNG assets with `build_figure2_metric_icons.py`; `build_overleaf_package.py` includes them in both Overleaf ZIPs. The preview sheet is enlarged for inspection; use the individual files at print size.
