# COSYNE 2027 two-page layout draft

Open `main.pdf` for the current two-page preview. `main.tex` and the six PDFs under `figures/` form a self-contained LaTeX source package. Scientific checks and missing content are recorded in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

The figure inputs are copied from the existing analyses:

| Draft figure | Source |
|---|---|
| Circuit | `06_experiments/cosyne_2027/architecture_compact.pdf` |
| Worker timing placeholder | `06_experiments/results/cosyne_2027_compact_figures/04_worker_coverage_values_and_delta.svg` |
| Online course | `06_experiments/results/cosyne_2027_compact_figures/01_online_spatial_course.svg` |
| Peak-density example | `06_experiments/results/cosyne_2027_checkpoint_fields/common_panel_analysis/figures/c01_seed99_common_peak_density.svg` |
| Encoder-credit differences | `06_experiments/results/cosyne_2027_compact_figures/03_encoder_credit_deltas.svg` |
| Frozen-policy expression | `06_experiments/results/cosyne_2027_compact_figures/02_behavior_expression_fresh.svg`, from commit `6412596` on branch `codex/cosyne-behavior-fresh-seeds`; the complete six-panel original remains in that analysis branch. |

The SVG plots were converted to PDF without changing their contents. The LaTeX source crops the lower row of the frozen-policy figure for space. Compile from this directory with `tectonic main.tex` or a compatible XeLaTeX installation.
