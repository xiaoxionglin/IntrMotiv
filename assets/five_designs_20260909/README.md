# Architecture diagrams

Five conceptual diagrams supporting [the design report](../../05_plans/five_designs_representation_control_exploration_20260909.md). These contain no simulation results or empirical comparisons.

- `plan_1.png` through `plan_5.png`: report images, 1640 × 870 pixels; intended display width about 820 pixels.
- Matching SVGs: scalable copies with font outlines for portable display.
- Matching previews: 820 × 435 pixels, inspected at the intended reading size.
- `render_architectures.py`: regenerates diagrams with the project `SF_git` Python environment. Resolves and verifies DejaVu Sans TTF; checks that every box label fits. Minimum text is 16 pt before report-scale rendering.

Common backbone is shown in the top row. Lower boxes summarize the distinctive learning/planning components. They are functional schematics; the report specifies exact inputs, frozen targets, and gradient destinations. Plan 5 alone adds prior-context feedback into DG.
