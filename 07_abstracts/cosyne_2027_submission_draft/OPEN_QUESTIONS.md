# COSYNE 2027 draft: checks for discussion

This layout reproduces the author's current 12 October GitHub title and 312-word abstract verbatim in substance, including the new push--pull explanation and behavior-dependent spatial-coding conclusion. It preserves unresolved scientific claims in the second-page text and comments. The title and summary still require conference-length edits before submission.

## Scientific claims to confirm

1. **Temporal pull and discounting.** The C05 temporal-versus-constant target-hit comparison rewards hits in both arms, and PPO discounting remains active. The separate longer-credit study changes both PPO discount and GAE lambda, so it does not isolate discounting as a substitute for temporal hit weighting. Decide whether the abstract's stronger substitution interpretation is justified.
2. **Spatial information under longer credit.** Three C15 longer-credit runs have lower measured spatial information on separate policy rollouts. Those policies visit different places, so this result alone does not isolate a fixed-input DG-code change.
3. **Policy and representation.** The own-policy-versus-random intervention freezes network weights. It supports behavior-dependent decoder spatial information, not a training-policy effect on DG weights.
4. **Worker temporal-hit credit.** Figure 2F compares coverage. Its mean C05 contrast is +0.026 for a temporal encoder and -0.087 for a constant encoder, each with five paired seeds. Matched-command outcomes, physical-arrival labels, and C15 worker controls are needed for a goal-control claim.
5. **C15 encoder comparison.** Figure 2D has only three matched C15 encoder seed pairs. Downstream effects are mixed; refresh after replacement checkpoints and common-panel replays complete.

## Missing submission elements

- The LaTeX summary is 312 words versus the official 300-word requirement; the title is 128 characters versus the 100-character maximum. Shorten only after author review. Select a few relevant prior works from [references.bib](../../08_literature/references.bib), then check the final COSYNE 2027 citation and format rules.
- Finish commanded-target and physical-arrival contrasts before interpreting temporal-hit coverage as goal control or making a directional pull claim.
- COSYNE 2027 specifies two pages (US Letter/A4), 12-point minimum body, 10-point captions/references permitted, margins at least 0.5 inch, and double-blind anonymity. The comprehensive review draft is currently three pages by the author's choice; shorten the title and summary and select what to condense before submission, then check the final PDF and its metadata.
- Review panel priority at final submission size. The figure keeps all scientifically distinct plotted contrasts; the earlier full candidate library remains in the backed-up expanded ZIP.

## Layout and example choice

The architecture is displayed at its original 6.59-inch width. The results are rendered as one 7.5-by-4.48-inch figure with 9–10 pt internal type at final size. The dot plots show paired control and tested-condition values in smaller individual panels, rather than difference axes. Panel C deliberately selects seed 8 because its peak-area difference is the largest of five baseline pairs. Panel D shows all paired seeds so the selected example cannot be mistaken for the distribution.

Official COSYNE 2027 rules: https://www.cosyne.org/abstracts-submission (checked 2026-10-12).
