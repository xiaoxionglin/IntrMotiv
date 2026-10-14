# Figure 1 and Figure 2 palette trials

[Color key](palette_key.png) · [Figure 1 comparison](comparison_figure1.png) · [Figure 2 comparison](comparison_figure2.png) · [source and substitution manifest](manifest.json)

These trials use the latest author-edited [Figure 1 base](../figure1_candidates/06_architecture_v3_goal_gate_header.svg) and the [Figure 2 icon layout](../Figure2_story_icons.svg). Both sources are left unchanged. The script replaces only listed vector colors and, in option C, makes pull arrows dashed. It retains geometry, plotted values, sample points, text, the Viridis step scale in Figure 2A, and place-field image colors. In the key, DG 1–3 mean the original blue, red, and yellow DG identities; DG 4 is the additional identity used in Figure 2 metric icons. The mapping keeps each identity consistent across figures.

| Option | Main change | Figure 1 | Figure 2 |
| --- | --- | --- | --- |
| A: Slate NG | Neutral NG; preserve red push and blue pull; DG identities become violet, ochre, teal, and rose | [SVG](slate_ng_figure1.svg) · [PDF](slate_ng_figure1.pdf) · [PNG](slate_ng_figure1.png) | [SVG](slate_ng_figure2.svg) · [PDF](slate_ng_figure2.pdf) · [PNG](slate_ng_figure2.png) |
| B: Purple pull | Preserve saturated architecture colors; push becomes umber, pull purple; DG identities become cyan, rose, ochre, and moss | [SVG](purple_pull_figure1.svg) · [PDF](purple_pull_figure1.pdf) · [PNG](purple_pull_figure1.png) | [SVG](purple_pull_figure2.svg) · [PDF](purple_pull_figure2.pdf) · [PNG](purple_pull_figure2.png) |
| C: Neutral mechanisms | Follow the existing [COSYNE color standard](../../../06_experiments/cosyne_2027/COLOR_PALETTE.md): keep NG/UG/FG hues, make push neutral and solid, pull neutral and dashed; DG identities become ochre, violet, rose, and slate | [SVG](role_neutral_figure1.svg) · [PDF](role_neutral_figure1.pdf) · [PNG](role_neutral_figure1.png) | [SVG](role_neutral_figure2.svg) · [PDF](role_neutral_figure2.pdf) · [PNG](role_neutral_figure2.png) |

**Recommendation for review:** C gives the clearest role separation across both figures and matches the existing project color standard: architecture alone owns the saturated result colors, while shape and line style distinguish push from pull. A is the less disruptive choice if keeping the familiar red-push/blue-pull mnemonic matters more; its NG marks are intentionally quieter than UG and FG. B keeps NG prominent but changes both mechanism colors. These are optional assets in the expanded Overleaf ZIP; none has replaced the current submission figures.

Rebuild both options from the current sources with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_palette_previews.py`. The manifest records exact source hashes, role colors, and substitutions; the script fails when an expected source role color is missing.
