# COSYNE 2027 two-page submission draft

Open [main.pdf](main.pdf) for the two-page preview. The first page contains the abstract, model text, and the author's uploaded architecture. The second places all result plots together as panels A--F under one caption, with main text around them. [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) records claim checks and missing work. The self-contained Overleaf package is `cosyne_2027_overleaf_sources.zip`.

## Figure provenance

| Draft element | Source and treatment |
| --- | --- |
| Architecture | The newer uploaded `architecture.svg` (`ec6b31dd-636e-48d6-9250-9664fbb0026b`, SHA-256 `107e97deab90cdba634a0442c706226dbef89a98340d677963f825cbb14d566d`). Its unchanged vector is saved as `figures/architecture_uploaded.svg` and converted to `figures/architecture.pdf`. |
| A: DG course | `06_experiments/results/cosyne_2027_compact_figures/01_online_spatial_course.svg` |
| B: peak maps | `06_experiments/results/cosyne_2027_checkpoint_fields/common_panel_analysis/figures/c01_seed99_common_peak_density.svg` |
| C: downstream encoder credit | `06_experiments/results/cosyne_2027_temporal_credit_layers/encoder_paired_decoder_1.svg`; this adds the new CA3/decoder common-panel analysis to the original draft plots. |
| D: frozen-policy behavior | The complete six-panel `02_behavior_expression_fresh.svg` from branch commit `6412596` (`codex/cosyne-behavior-fresh-seeds`). Both the information and split-half-reliability rows are retained. |
| E: DG encoder contrasts | `06_experiments/results/cosyne_2027_compact_figures/03_encoder_credit_deltas.svg` |
| F: worker timing placeholder | `06_experiments/results/cosyne_2027_compact_figures/04_worker_coverage_values_and_delta.svg` |

The plots are the original vector panels converted to PDF without changing their data or graphics. `main.tex` arranges them in one grouped figure, keeping every original plot from the first draft and adding the complete decoder-1 encoder-credit plot. The dot-plot panels are intentionally small at page size, as requested. Their full-size source plots remain at the paths above for inspection and later panel selection.

Compile `main.tex` from this directory with `tectonic main.tex` or a compatible XeLaTeX installation. The TeX file and its seven included PDFs are sufficient for Overleaf.
