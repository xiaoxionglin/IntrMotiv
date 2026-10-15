# Selected COSYNE palette source

The selected palette is **H: role-first orange–violet**. Orange identifies push, violet identifies pull, blue/red/green/cyan identify DG units, and slate/olive/rose identify NG/UG/FG. The published artwork is [Figure 1](../Figure1.svg) and [Figure 2](../Figure2.svg); the [manifest](manifest.json) records source hashes and color substitutions.

The retained [Figure 1 palette SVG](priority_rgb_orange_violet_figure1.svg) and [Figure 2 palette SVG](priority_rgb_orange_violet_figure2.svg) are construction sources. Figure 2's selected y-axis-gutter layout is built from the latter by [build_figure2.py](../build_figure2.py); see its [layout notes](../Figure2_layout_README.md). The input layouts are the [author-edited Figure 1 base](../figure1_candidates/06_architecture_v3_goal_gate_header.svg) and the [Figure 2 icon arrangement](../Figure2_story_icons.svg).

Earlier palette export files were removed because they varied color without changing the scientific content. [build_palette_previews.py](build_palette_previews.py) and the manifest preserve the reproducible palette experiment if a new visual comparison is needed. Regeneration may create the old trial exports again; keep only a variant whose palette communicates a distinct scientific meaning.
