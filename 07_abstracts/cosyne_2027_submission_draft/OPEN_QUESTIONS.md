# COSYNE 2027 draft: checks for discussion

This two-page layout keeps the author's 12 October GitHub abstract edits, including their stronger claims and the deletion of neutral open-question sentences. The new downstream headline is supported by the [38-checkpoint common-panel analysis](../../06_experiments/cosyne_2027/temporal_credit_multilayer_20261012.md). Comments marked `CHECK` in `main.tex` point here.

## Scientific claims to confirm

1. **Temporal pull and discounting.** The C05 temporal versus constant target-hit comparison still rewards hits in both arms, and PPO discounting remains active. The separate longer-credit study changes both PPO discount $\gamma$ and GAE $\lambda$; it does not directly test whether discounting substitutes for temporal hit weighting. The abstract keeps the author's proposed “can largely substitute” interpretation. Decide whether that strength of language is justified before submission.
2. **Spatial information under longer credit.** The three C15 longer-credit runs have lower measured spatial information on separate policy rollouts. Because their policies visit different places, this alone does not isolate a change in the fixed-input DG code. The abstract's “measured spatial information” wording is intentionally narrower.
3. **Policy and representation.** The own-policy versus random-action intervention freezes network weights. It supports a policy effect on the *expression* of decoder information. It does not establish that feedback changes DG weights during training. The abstract and Figure 2D use the expression claim.
4. **Worker temporal hit credit.** Figure 2F is a coverage placeholder, not a goal-control result. The five-seed C05 coverage and DG-field effects are mixed; the [CA3 and decoder factorial](../../06_experiments/cosyne_2027/05_downstream_spatial_information.md) finds mixed normalized effects and lower amplitude-weighted decoder scores in all five temporal-encoder worker pairs. Replace or revise this panel after matched-command outcomes, physical-arrival labels, and C15 worker controls are verified.
5. **C15 encoder comparison.** The main text identifies three matched C15 seed pairs. Their downstream effects are mixed. Refresh the wording after the two replacement encoder checkpoints and common-panel replays complete; do not infer their absent results from the five temporal models.

## Missing content for a submission-ready version

- **Prior-work placement and citations.** The two-page layout has no reference line. Candidates already in `08_literature/references.bib` are Lin et al. (2025), Leibold (2020), Yiu and Leibold (2023), Stachenfeld et al. (2017), and Zhang et al. (2021). Select the few that directly support the mechanism and contrast, then check the 2027 COSYNE citation rules. The second page has room for a compact reference line.
- **Worker control outcome.** The current panel is explicitly work in progress. A completed commanded-target contrast, with actual destination and timing, is the main missing result for the “pull” story.
- **Final submission rules and identity.** Confirm the 2027 template, page/word limits, anonymity, title, and author block. The 2026 attached example supplied the visual starting point; this draft uses a provisional title.
- **Panel priority.** Figure 2D now retains both decoder-1 information and split-half reliability, and Figure 2C adds the new downstream encoder-credit comparison. All original plots are present. The grouped dot plots are small at page size; decide later which need more width in the final submission.

## Layout decision

The author's architecture fills the width of page 1. Figure 2 groups all original result plots plus the new decoder comparison as six labeled panels without enlarging any source plot. The grouped figure leaves space for main text on page 2. Full-size plot files are linked from the README; at submission size, some axis labels in the dot plots require zooming.
