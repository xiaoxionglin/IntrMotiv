# COSYNE 2027 draft: checks for discussion

This layout uses the author's latest 12 October GitHub title and 321-word abstract, including the removal of neutral open-question claims. It marks incomplete analyses in the main text or comments rather than changing those abstract sentences.

## Scientific claims to confirm

1. **Temporal pull and discounting.** The C05 temporal-versus-constant target-hit comparison rewards hits in both arms, and PPO discounting remains active. The separate longer-credit study changes both PPO discount and GAE lambda, so it does not isolate discounting as a substitute for temporal hit weighting. Decide whether the abstract's stronger substitution interpretation is justified.
2. **Spatial information under longer credit.** Three C15 longer-credit runs have lower measured spatial information on separate policy rollouts. Those policies visit different places, so this result alone does not isolate a fixed-input DG-code change.
3. **Policy and representation.** The own-policy-versus-random intervention freezes network weights. It supports a policy effect on expression of decoder information, not a training-policy effect on DG weights.
4. **Worker temporal-hit credit.** Figure 2F is a coverage placeholder. Its mean C05 contrast is +0.026 for a temporal encoder and -0.087 for a constant encoder, each with five paired seeds. Matched-command outcomes, physical-arrival labels, and C15 worker controls are needed for a goal-control claim.
5. **C15 encoder comparison.** Figure 2D has only three matched C15 encoder seed pairs. Downstream effects are mixed; refresh after replacement checkpoints and common-panel replays complete.

## Illustrations to discuss later

- A tiny schematic distinguishing the identical-observation replay used for encoder tests from policy rollouts used for behavioral expression.
- A target-hit timing sketch explaining temporal versus constant worker credit and how PPO discounting also favors earlier hits.
- A commanded-versus-actual-goal sketch for the planned goal-control analysis.

## Missing submission elements

- The abstract is about 321 words; confirm the final word limit and shorten only after author review. Select a few relevant prior works from [references.bib](../../08_literature/references.bib), then check the final COSYNE 2027 citation and format rules.
- Finish commanded-target and physical-arrival contrasts before replacing the temporal-hit placeholder or making a directional pull claim.
- Confirm the 2027 template, page and word limits, anonymity, title, and author block.
- Review panel priority at final submission size. The figure keeps all scientifically distinct plotted contrasts; the full candidate library remains in the Overleaf ZIP.

## Layout and example choice

The architecture is displayed at its original 6.59-inch width. The result panels are rendered as one 7.5-inch-wide figure with 9–10 pt internal type at final size. Panel C deliberately selects seed 8 because its peak-area difference is the largest of five baseline pairs. Panel D shows all paired seeds so the selected example cannot be mistaken for the distribution.
