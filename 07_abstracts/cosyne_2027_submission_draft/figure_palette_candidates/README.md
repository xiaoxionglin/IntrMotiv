# Figure 1 and Figure 2 palette trials

[Opposing-hue Figure 1 comparison](comparison_figure1_opponent_dg.png) · [opposing-hue Figure 2 comparison](comparison_figure2_opponent_dg.png) · [color key](palette_key.png) · [earlier Figure 1 comparison](comparison_figure1.png) · [earlier Figure 2 comparison](comparison_figure2.png) · [source and substitution manifest](manifest.json)

These trials use the latest author-edited [Figure 1 base](../figure1_candidates/06_architecture_v3_goal_gate_header.svg) and the [Figure 2 icon layout](../Figure2_story_icons.svg). Both sources are left unchanged. The script replaces only listed vector colors and, in option C, makes pull arrows dashed. It retains geometry, plotted values, sample points, text, the Viridis step scale in Figure 2A, and place-field image colors. In the key, DG 1–3 mean the original blue, red, and yellow DG identities; DG 4 is the additional identity used in Figure 2 metric icons. The mapping keeps each identity consistent across figures.

| Option | Main change | Figure 1 | Figure 2 |
| --- | --- | --- | --- |
| A: Slate NG | Neutral NG; preserve red push and blue pull; DG identities become violet, ochre, teal, and rose | [SVG](slate_ng_figure1.svg) · [PDF](slate_ng_figure1.pdf) · [PNG](slate_ng_figure1.png) | [SVG](slate_ng_figure2.svg) · [PDF](slate_ng_figure2.pdf) · [PNG](slate_ng_figure2.png) |
| A2: Violet and rose swap | Keep A's four DG colors, assigning violet and rose to the two units highlighted in the overlap and reliability icons; ochre remains on the third Figure 1 unit | [SVG](slate_ng_violet_rose_figure1.svg) · [PDF](slate_ng_violet_rose_figure1.pdf) · [PNG](slate_ng_violet_rose_figure1.png) | [SVG](slate_ng_violet_rose_figure2.svg) · [PDF](slate_ng_violet_rose_figure2.pdf) · [PNG](slate_ng_violet_rose_figure2.png) |
| A3: Tuned DG colors | Keep violet and rose on those highlighted units; use muted taupe for the third Figure 1 unit and cyan for the fourth icon unit | [SVG](slate_ng_tuned_dg_figure1.svg) · [PDF](slate_ng_tuned_dg_figure1.pdf) · [PNG](slate_ng_tuned_dg_figure1.png) | [SVG](slate_ng_tuned_dg_figure2.svg) · [PDF](slate_ng_tuned_dg_figure2.pdf) · [PNG](slate_ng_tuned_dg_figure2.png) |
| B: Purple pull | Preserve saturated architecture colors; push becomes umber, pull purple; DG identities become cyan, rose, ochre, and moss | [SVG](purple_pull_figure1.svg) · [PDF](purple_pull_figure1.pdf) · [PNG](purple_pull_figure1.png) | [SVG](purple_pull_figure2.svg) · [PDF](purple_pull_figure2.pdf) · [PNG](purple_pull_figure2.png) |
| C: Neutral mechanisms | Follow the existing [COSYNE color standard](../../../06_experiments/cosyne_2027/COLOR_PALETTE.md): keep NG/UG/FG hues, make push neutral and solid, pull neutral and dashed; DG identities become ochre, violet, rose, and slate | [SVG](role_neutral_figure1.svg) · [PDF](role_neutral_figure1.pdf) · [PNG](role_neutral_figure1.png) | [SVG](role_neutral_figure2.svg) · [PDF](role_neutral_figure2.pdf) · [PNG](role_neutral_figure2.png) |
| D: RGB / magenta–teal | DG IDs are blue, red, green, and gold; push is magenta, pull is teal | [SVG](rgb_magenta_teal_figure1.svg) · [PDF](rgb_magenta_teal_figure1.pdf) · [PNG](rgb_magenta_teal_figure1.png) | [SVG](rgb_magenta_teal_figure2.svg) · [PDF](rgb_magenta_teal_figure2.pdf) · [PNG](rgb_magenta_teal_figure2.png) |
| E: RGB / orange–violet | DG IDs are blue, red, green, and cyan; push is orange, pull is violet | [SVG](rgb_orange_violet_figure1.svg) · [PDF](rgb_orange_violet_figure1.pdf) · [PNG](rgb_orange_violet_figure1.png) | [SVG](rgb_orange_violet_figure2.svg) · [PDF](rgb_orange_violet_figure2.pdf) · [PNG](rgb_orange_violet_figure2.png) |
| F: CMY / red–blue | DG IDs are cyan, magenta, yellow, and green; A's red push and blue pull remain | [SVG](cmy_red_blue_figure1.svg) · [PDF](cmy_red_blue_figure1.pdf) · [PNG](cmy_red_blue_figure1.png) | [SVG](cmy_red_blue_figure2.svg) · [PDF](cmy_red_blue_figure2.pdf) · [PNG](cmy_red_blue_figure2.png) |

## Two-page vector PDFs for direct review

Each PDF below has Figure 1 on page 1 and Figure 2 on page 2, both at their existing 7.5-inch width. Individual editable SVGs and vector PDFs are linked in the table above.

| Version | Two-page PDF |
| --- | --- |
| A2: literal violet–rose reassignment | [review_slate_ng_violet_rose.pdf](review_slate_ng_violet_rose.pdf) |
| A3: violet–rose with neutral remaining IDs | [review_slate_ng_tuned_dg.pdf](review_slate_ng_tuned_dg.pdf) |
| D: opposing RGB DG; magenta–teal push/pull | [review_rgb_magenta_teal.pdf](review_rgb_magenta_teal.pdf) |
| E: opposing RGB DG; orange–violet push/pull | [review_rgb_orange_violet.pdf](review_rgb_orange_violet.pdf) |
| F: opposing CMY DG; red–blue push/pull | [review_cmy_red_blue.pdf](review_cmy_red_blue.pdf) |

The later D–F options respond to the preference for strongly distinct DG unit hues and contrasting push/pull roles. NG/UG/FG retain A's neutral-slate/orange/green mapping and are identified by box-plot position and labels. At intended figure size, D separates the three Figure 1 DG identities most plainly; E makes push/pull particularly different from one another, but orange push resembles UG; F retains A's mechanism mnemonic, with cyan and magenta DG units closer to pull and push. No submission figure source or Overleaf ZIP was modified for these new trials.

Rebuild the previews and two-page PDFs from the current sources with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_palette_previews.py`. The manifest records exact source hashes, role colors, and substitutions; the script fails when an expected source role color is missing and skips unchanged vector exports.
