# COSYNE 2027 submission layout draft

Open [main.pdf](main.pdf) to review the two-page layout. Page 1 contains the author's latest GitHub title and 312-word abstract (not yet shortened to COSYNE's 300-word limit) and the original-size architecture. Page 2 groups every selected result plot by claim, including temporal-hit coverage. [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) records claim checks and analyses for discussion.

Upload cosyne_2027_overleaf_sources.zip as a new Overleaf project and set main.tex as the main document. The updated ZIP is a compact, self-contained version: it contains the two figure PDFs actually used by `main.tex`, their original SVG sources, submission notes, and panel provenance. FIGURE_OPTIONS.md indexes the included figures. Rendering scripts and underlying CSV inputs remain in the full repository; the earlier full candidate-library ZIP is preserved as `cosyne_2027_overleaf_sources_full_backup_20261012.zip` in this directory. The backup predates this figure revision, while the compact ZIP contains the current panels. `build_overleaf_package.py` refreshes the compact ZIP by default; `--full` writes a separate expanded candidate-library ZIP without replacing it. Raw NEMO2 evaluator arrays are deliberately excluded; PANEL_PROVENANCE.json lists the verified source hashes.

## Figure provenance and selection

| Draft element | Source and treatment |
| --- | --- |
| Architecture | Newer author-uploaded architecture SVG, SHA-256 107e97deab90cdba634a0442c706226dbef89a98340d677963f825cbb14d566d. The converted PDF is displayed at its original 6.59-inch width. |
| A: DG course | Five-seed online summary from cosyne_2027_online_wandb/arm_summaries.csv. |
| B: frozen-policy expression | Fresh decoder-1 absolute values for 38 models, averaged over two repeats from branch commit 6412596. Random-action controls and own-policy values are paired within each model for information and split-half reliability; the derived 38-row table is in this repository's `data/` directory. |
| C: selected peak-map example | Baseline seed 8, the largest peak-area temporal-minus-constant difference among five matched seeds at the 0.25% activity cutoff: 0.384 versus 0.230 (difference +0.155). This example is selected for effect size and is not presented as typical. |
| D: DG paired statistics | Constant-credit controls and temporal-credit values for every matched seed, plotted directly for DG score, active-map cosine, and peak area on common observations. The baseline mean peak-area difference is +0.109 across five positive pairs. |
| E: decoder credit | Constant-credit controls and temporal-credit values for baseline decoder-1 information; five positive pairs, mean gain +0.680 bits per activation. |
| F: temporal-hit coverage | C05 visited fractions for temporal and constant worker-hit credit, with each encoder arm held fixed. Goal-control interpretation requires separate matched-command and physical-arrival analyses. |

The composite was rendered at its final 7.5-by-4.48-inch print size using scalable Arial with 9–10 pt labels. Individual dot-plot widths are 1.43 inches (behavior), 0.86 inches (DG), 1.20 inches (decoder), and 1.10 inches (worker). Open squares show controls, filled circles show tested conditions, and a line links the same training seed; the plots use actual values rather than a difference axis. The selected example was extracted from two SHA-256-verified evaluator arrays into lightweight peak coordinates and an occupancy mask. The full source plots remain available in the repository and the backed-up candidate library for later selection.

Rebuild the panels with the project SF_git Python environment using render_submission_panels.py, then run build_overleaf_package.py to refresh the compact ZIP. Pass `--full` only when an expanded candidate-library ZIP is needed. Compile main.tex with Tectonic or XeLaTeX.
