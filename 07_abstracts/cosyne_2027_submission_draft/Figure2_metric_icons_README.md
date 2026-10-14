# Figure 2 metric icons

[Enlarged preview](Figure2_metric_icons_preview.png) · [editable preview](Figure2_metric_icons_preview.svg) · [builder](build_figure2_metric_icons.py)

Path efficiency, map overlap, and reliability are **10 × 20 mm**; peak coverage and reliable edges are **10 × 30 mm**, including transparent margins. The upper example in each icon has the higher value of that statistic, and the arrow points up to match the plot's y-axis direction. The icons contain no measured data or claim that either intervention increases a statistic. In the DG icons, colored circles identify units: Figure 1 blue, red, and yellow, plus purple as a fourth unit.

| Icon | SVG | PDF | What it depicts |
| --- | --- | --- | --- |
| Path efficiency | [SVG](Figure2_metric_icon_path_efficiency.svg) | [PDF](Figure2_metric_icon_path_efficiency.pdf) | A direct route above a detour between the same endpoints. The pale dashed line below is displacement; the dark route is traveled length. The plotted statistic sums segment end-to-end displacement and divides by total traveled length across segments. |
| Map overlap | [SVG](Figure2_metric_icon_overlap.svg) | [PDF](Figure2_metric_icon_overlap.pdf) | Each row compares **two different DG units**, red and yellow (colored circles 1 and 2). Their spatial fields align above and diverge below. The plotted statistic is mean pairwise cosine of active units' maps. |
| Peak coverage | [SVG](Figure2_metric_icon_coverage.svg) | [PDF](Figure2_metric_icon_coverage.pdf) | The same four color-coded units have dispersed peaks above and clustered peaks below, with Gaussian blur. The plotted value is effective visited-area fraction of eligible peak density, not a count of peaks. |
| Reliable edges | [SVG](Figure2_metric_icon_edges.svg) | [PDF](Figure2_metric_icon_edges.pdf) | The same four color-coded DG nodes form two directed stored graphs with more reliable arrows above than below. The plotted value counts stored reliable directed edges, not executed goal arrivals. |
| DG reliability | [SVG](Figure2_metric_icon_reliability.svg) | [PDF](Figure2_metric_icon_reliability.pdf) | Both rows compare **the same red DG unit** at two episode-disjoint times $t_1$ and $t_2$. The maps match above and diverge below. The plot uses split-half map correlation. |

The icons illustrate what each statistic compares within a condition. In Figure 2C and 2D, the plotted $\Delta$ is **temporal minus constant credit** for that statistic; in Figure 2B, reliability's $\Delta$ is **learned policy minus random actions**. Overlap is still computed as map cosine; its support, and that of coverage, differs between the common-observation push comparison and the own-policy pull comparison. See the [story-figure notes](Figure2_story_README.md) and [focused metric definitions](figure2_candidates/08_focused_canvas/README.md).

The PDFs are ready for `\includegraphics[width=10mm,height=20mm,keepaspectratio]{figures/Figure2_metric_icon_overlap.pdf}` (replace the filename for the other icons; coverage and edges have 30 mm height). Build all SVG, PDF, and PNG assets with `build_figure2_metric_icons.py`; `build_overleaf_package.py` includes them in both Overleaf ZIPs. The preview sheet is enlarged for inspection; use the individual files at print size.
