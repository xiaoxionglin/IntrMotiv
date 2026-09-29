# A0 poster review choices — 29 September 2026

[Three current variations of the attached B](layout_versions/README.md) apply
the supplied critique except the rejected condition-label change. All include
strict-field counts and mobility, three-arm DG/package transfer, the training-only
predictor, pooled versus within-family associations and command lift. The
choices differ in dominance, 12-model spatial concentration, or concentration
in the main nine C01/C05/C15 models. The last is recommended for review.
The attached header, introduction and left-panel data are preserved; an
incorrect activity-maximum key is corrected. Figure fonts declare Helvetica
with a verified Nimbus Sans substitute; references remain boxed. These new
review choices continue from B and do not revive the formerly dropped version.

[Assessment, sources and limits](layout_versions/critique_variations/README.md)
explain why transfer supports a package comparison and field-shape results
support dissociation rather than a causal claim that concentration harms behaviour.

The earlier filled poster and candidate sources remain below for reference.

[Editable Inkscape poster](Bernstein2026_IntrMotiv_filled.svg) · [Full-page preview](Bernstein2026_IntrMotiv_filled.png)

[Optional plot gallery](candidate_plots/README.md) contains nine separate editable candidates for transfer, predictor loops, graph diagnostics, compact fields, and within-family associations. They reuse the current physical panel sizes and 30 pt plot text, with 40 pt captions on the selection sheets. The gallery does not modify this poster; choose panels after rearranging its columns.

The title/author/logo strip and left column are preserved from the supplied poster. Equal-size Inkscape renders show **zero changed pixels** in those regions. Only the right-column plots, headings, captions, conclusion, and references were replaced. The adapter accepts either the original supplied SVG or the current filled poster. Updating the current poster replaces only its generated right-column layer, retaining edits elsewhere. The manifest records the input hash before replacement.

## Earlier filled poster: content and evidence

- **Exploration and returns:** four compact plots aligned horizontally. Dots show seeds 8, 99, and 123; black segments show equal-weight seed means. Coverage AUC is the time-averaged cumulative visited-cell count. The shared caption separates final-10M training summaries from frozen-policy mobile-window returns and states the mobility fractions.
- **Goal specificity:** C05/C15 target-hit-lift and raw-logit-sensitivity plots together occupy 191.5 mm, half the right column. Explanations occupy the other half. Target events do not establish physical arrival; the designs also differ in regularization and manager structure.
- **Architectural survey:** two smaller scatters show the same 168 eligible DG-16 online runs (56 variants), plotting spatial score and distinct peak bins against target-event hits / attempts. Pooled correlations are $\rho=0.583$ and $\rho=-0.424$, respectively. Color and shape identify study family; capacity is fixed throughout.
- **Conclusions:** broader exploration can coexist with weak commanded-target evidence. More peak locations do not imply more target hits. The positive spatial-score association weakens within CPD ($\rho=0.094$, 81 runs) and DGP ($\rho=-0.001$, 24). Correlations describe this survey rather than causal effects.
- **Typography:** all new plot labels, ticks, legends, and annotations are 30 pt at final A0 size. Explanatory text is 40 pt; headings are 48 pt. Short labels and smaller data areas provide room for larger text and slightly larger markers. The preserved left column retains its original fonts.

All included panels have complete local evidence; there are no unresolved placeholders. Weak distance-dependent similarity profiles were removed from the poster, without changing their source data or stretching their vertical scale. The earlier `layerwise_displacement.svg` and `kernel_band_points.csv` remain here as supplementary assets; they are no longer inputs to the current build. Full transfer and additional galleries remain supplementary.

Each new panel is a named native SVG group inside the Inkscape layer **Right column · large type and DG-16 survey · 29 Sep**. Text remains editable and the poster has no external image dependencies. Individual plot SVGs are saved here. Numerical inputs, hashes, font path, software versions, and typography are in [build_manifest.json](build_manifest.json). Every scatter observation is in [architecture_scatter_points.csv](architecture_scatter_points.csv); counts, ages, and correlations are in [architecture_scatter_statistics.json](architecture_scatter_statistics.json).

The main survey uses latest saved DG-16 online ages spanning 25–150M frames, with historical target counters and recent spatial windows. A/B repeat the same cohort. Holding capacity fixed does not match age, family, or outcome rules. DG 32/64 and replay/probe protocols are screened separately in [dg_capacity_audit.csv](dg_capacity_audit.csv). The old mixed-capacity replay-score/coverage null disappears under the DG-16 filter ($\rho=0.321$, 18 runs), so that wording and panel are retired. [Discussion points](discussion_points.md) explain the revised story and additional visitor topics.

## Earlier filled poster: rebuild

From the repository root, use the current poster or provide the original supplied SVG:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python \
  06_experiments/compose_a0_poster_20260929.py \
  --source 05_plans/poster_20260929/Bernstein2026_IntrMotiv_filled.svg

inkscape 05_plans/poster_20260929/Bernstein2026_IntrMotiv_filled.svg \
  --export-area-page --export-type=png --export-width=1800 \
  --export-filename=05_plans/poster_20260929/Bernstein2026_IntrMotiv_filled.png
```

## Earlier filled poster: editorial review

The latest revision prioritizes the user's 30/40 pt typography and fixed-capacity comparison. Two informative scatter metrics replace the previous four-panel mixed-capacity block. The take-home asks which goal representation can reliably select a physical place. See [critique.md](critique.md) for interpretation limits and [discussion_points.md](discussion_points.md) for measured visitor discussion candidates.

## Earlier filled poster: checks and reusable lessons

The build verifies the selected nine runs and common checkpoint, unique SVG IDs, unchanged retained source subtrees, and one observation per run within each scatter panel. The full page and right-column crop were inspected for clipping and overlap; family shapes reinforce colors in grayscale. New plot text is 30 pt; explanatory text is 40 pt at final A0 size. Existing left-column figures retain their original appearance. Recorded checks are in [quality_checks.json](quality_checks.json).

What worked: pinned point tables, the canonical scatter renderer, physical font units, measured text wrapping, and protected-region pixel checks. What failed in earlier drafts: 13 pt plot fonts were technically readable but too small for the intended poster experience, and mixed capacities obscured the interpretation. Set typography before layout, shorten repeated labels, and recompute claims after every cohort filter. Inkscape's export/object queries and SF_git font metrics were authoritative; no new probes or rendering dependencies were needed.
