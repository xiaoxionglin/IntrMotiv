# COSYNE 2027 submission layout draft

Open [main.pdf](main.pdf) to review the two-page layout. Page 1 contains the author's latest GitHub title and 304-word abstract, the original-size architecture, and a compact illustration of the three policy families. Page 2 groups the results by claim and includes the temporal-hit work-in-progress panel. [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) records claim checks and analyses for discussion.

Upload cosyne_2027_overleaf_sources.zip as a new Overleaf project and set main.tex as the main document. The ZIP contains every image used by the draft, their SVG sources, the figure-rendering script and its lightweight inputs, and the broader COSYNE plot library as candidate PDF/SVG pairs and PNG-only examples. FIGURE_OPTIONS.md in the ZIP indexes these assets. Raw NEMO2 evaluator arrays are deliberately excluded; PANEL_PROVENANCE.json lists the verified source hashes.

## Figure provenance and selection

| Draft element | Source and treatment |
| --- | --- |
| Architecture | Newer author-uploaded architecture SVG, SHA-256 107e97deab90cdba634a0442c706226dbef89a98340d677963f825cbb14d566d. The converted PDF is displayed at its original 6.59-inch width. |
| Variant strip | Rendered by [render_submission_panels.py](render_submission_panels.py); baseline C01, goal-conditioned C05, and goal-conditioned policy with frontier goal choice C15. |
| A: DG course | Five-seed online summary from cosyne_2027_online_wandb/arm_summaries.csv. |
| B: frozen-policy expression | Fresh decoder-1 absolute values for 38 models, averaged over two repeats from branch commit 6412596. Random-action controls and own-policy values are paired within each model for information and split-half reliability; the derived 38-row table is included in the ZIP. |
| C: selected peak-map example | Baseline seed 8, the largest peak-area temporal-minus-constant difference among five matched seeds at the 0.25% activity cutoff: 0.384 versus 0.230 (difference +0.155). This example is selected for effect size and is not presented as typical. |
| D: DG paired statistics | Constant-credit controls and temporal-credit values for every matched seed, plotted directly for DG score, active-map cosine, and peak area on common observations. The baseline mean peak-area difference is +0.109 across five positive pairs. |
| E: decoder credit | Constant-credit controls and temporal-credit values for baseline decoder-1 information; five positive pairs, mean gain +0.680 bits per activation. |
| F: worker-hit placeholder | C05 visited fractions for temporal and constant worker-hit credit, with each encoder arm held fixed. This is a provisional coverage panel while goal-control analyses are completed. |

The composite was rendered at its final 7.5-by-4.95-inch print size using scalable Arial with 9–10 pt labels. Individual dot-plot widths are 1.43 inches (behavior), 0.86 inches (DG), 1.20 inches (decoder), and 1.10 inches (worker). Open squares show controls, filled circles show tested conditions, and a line links the same training seed; the plots use actual values rather than a difference axis. Its selected example was extracted from two SHA-256-verified evaluator arrays into lightweight peak coordinates and an occupancy mask. The full source plots remain available in the repository and the Overleaf candidate library for later selection.

Rebuild the panels with the project SF_git Python environment using render_submission_panels.py, then rebuild the ZIP with build_overleaf_package.py. Compile main.tex with Tectonic or XeLaTeX.
