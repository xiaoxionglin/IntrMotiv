# COSYNE 2027 draft: checks for discussion

This is a layout draft based on the author's 12 October GitHub revision of the prose abstract. The abstract retains the author's claims and omits the neutral open-question sentences they removed. Comments marked `CHECK` in `main.tex` point here. The current PDF fits two letter-size pages with the 2026 example's 12 pt type and half-inch margins.

## Scientific claims to confirm

1. **Temporal pull and discounting.** The C05 temporal versus constant target-hit comparison still rewards hits in both arms, and PPO discounting remains active. The separate longer-credit study changes both PPO discount $\gamma$ and GAE $\lambda$; it does not directly test whether discounting substitutes for temporal hit weighting. The abstract keeps the author's proposed interpretation. Decide whether the current evidence is sufficient for “can largely substitute” before submission.
2. **Spatial information under longer credit.** The three C15 longer-credit runs have lower measured spatial information on separate policy rollouts. Because their policies visit different places, this alone does not isolate a change in the fixed-input DG code. The abstract's “measured spatial information” wording is intentionally narrower.
3. **Policy and representation.** The own-policy versus random-action intervention freezes network weights. It supports a policy effect on the *expression* of the learned decoder code. It does not establish that policy feedback changes DG weights during training. The abstract and Figure 5 caption use the expression claim.
4. **Worker temporal hit credit.** Figure 2 is a placeholder using currently available C05 coverage results. The paired coverage and DG-field effects are mixed. A new [five-seed common-panel analysis](../../06_experiments/cosyne_2027/05_downstream_spatial_information.md) also finds mixed CA3 and decoder bits-per-activation effects, with lower amplitude-weighted decoder scores in all five temporal-encoder worker pairs. Replace or revise the placeholder after matched-command outcomes, physical-arrival labels, and the pending C15 worker controls are complete. Avoid a claim of improved goal following until those analyses are available.
5. **C15 encoder comparison.** Figure 4 includes the three matched C15 seed pairs currently available; the five-seed comparison and interpretation should be refreshed when the remaining analyses arrive.

## Missing content for a submission-ready version

- **Prior-work placement and citations.** The current two-page layout has no reference line. Select a small set that establishes the predecessor model, fixed hippocampal sequences, predictive maps, and learned landmark or reachability approaches. Existing candidates in `08_literature/references.bib` are Lin et al. (2025), Leibold (2020), Yiu and Leibold (2023), Stachenfeld et al. (2017), and Zhang et al. (2021). Check exact relevance and the current COSYNE citation rules before adding them. A reference line will require a small layout cut.
- **Worker control outcome.** The current panel is explicitly work in progress. A completed commanded-target contrast, with actual destination and timing, is the main missing result for the “pull” story.
- **Final submission rules and identity.** Confirm the 2027 template, page/word limits, anonymity, title, and author block. The 2026 attached example supplied the visual starting point; this draft uses a provisional title.
- **Figure selection.** Figure 5 uses only the top spatial-information row of the six-panel frozen-policy result. Its split-half reliability row is omitted for space; retain it if this becomes central to the claim.

## Layout decision

The current two-page version includes five figures: circuit, worker placeholder, online course plus a place-field example, matched encoder-credit contrasts, and frozen-policy behavior. At page size the worker panel is the smallest. Enlarging it enough to match the other figures produces a third page, so the final cut should choose between a larger worker result and one of the secondary panels.
