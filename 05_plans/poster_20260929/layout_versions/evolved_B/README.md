# Selected B: field shape, sensitivity and commanded success

[Editable A0 poster](Bernstein2026_B_evolved.svg) · [Full-page preview](Bernstein2026_B_evolved.png)

This evolves the selected B. Transfer, the training-only predictor, the 168-run DG-16 architectural survey, and minimum-threshold field dominance versus exploration and recorded target hits remain. Section 2 is tightened to fit a new command row. The header and introduction are unchanged; all nine left-panel map, peak and trajectory data fingerprints are unchanged. A shared blue-circle **Option start** / gold-star **Goal hit** key labels the bottom-left trajectory panels. Goal hits mean internal DG events. References are boxed.

All generated plot and diagram labels now declare **Helvetica**, at 30 pt; conclusion captions remain 40 pt. Helvetica is not installed on this machine. Rendering and glyph measurement use the verified scalable **Nimbus Sans** substitute explicitly; the SVG keeps `Helvetica,Nimbus Sans` so Helvetica is used when available. This substitution and font path are recorded in [manifest.json](manifest.json). The original header/introduction fonts remain unchanged.

## Two different aspects of controllability

- **Action-probability TV** measures sensitivity to swapping the goal at the same policy state: $\frac{1}{2}\sum_a |p(a)-p'(a)|$. The saved training diagnostic rolls the goal across the minibatch and averages valid original-goal rows. It is invariant to a constant logit shift. It measures instantaneous probability change, not successful command execution.
- **Command lift** is executed success minus matched shuffled-command success, in percentage points. It tests target-specific internal DG outcomes over command trials. The shuffled comparator is retrospective, physical starts are not guaranteed identical, and ordered-pair coverage is incomplete.

The main command row shows all **12 models / four variants / three seeds**, DG 16 at 75,038,720 frames: First versus Hit DG policies, and Source versus Arrival credit. Every saved model has positive command lift, **+11.3 to +29.4 p.p.** Lines join matching training seeds; black bars are means. This is an absolute difference, distinct from section 2b's training hit-lift ratio.

TV is locally available for the **six overlapping Source/Arrival credit models**, from the 70M–74,973,180 training window: **0.029–0.408%** probability mass. Their frozen command lifts are **19.1–25.7 p.p.** Low instantaneous goal sensitivity can coexist with measurable command specificity. These diagnostics use different time windows and interventions. No TV values are saved for the CPD cohort used in the field-dominance panel; missing values are not filled in.

## Field dominance versus these measures

[Command-lift scatter](plots/dominance_command.svg) · [TV scatter](plots/dominance_tv.svg)

These are separate candidate figures, leaving the main poster readable. Replay mean-threshold dominance versus command lift has $\rho=-0.31$ for DG policy and $\rho=+0.77$ for credit assignment, with six points per family. Credit dominance versus TV gives $\rho=-0.26$, also six points. Opposite command-lift associations do not support a general claim that dominant fields harm control. The poster's message remains **field dominance does not reliably order better behaviour**.

The command representations all use the same saved 10,001-observation replay panel. This **mean-threshold dominance** differs from B's **minimum-threshold dominance** in recent online CPD maps; these scores and protocols remain separate. DG capacity stays at 16. No causal claim or significance mark is added. [Continuous-field evidence](../../continuous_fields/README.md) documents sampling and protocol limitations.

## Verification and reuse

[Quality checks](quality_checks.json) verify protected pixels, data fingerprints, page bounds, Helvetica declarations and text collisions. [Plot checks](plots/quality_checks.json) verify native vectors and a 30 pt minimum. [Exact plotted coordinates](plots/plotted_points.csv), [TV joins](plots/action_tv_per_run.csv) and [diagnostic definitions/statistics](plots/command_diagnostics_statistics.json) retain the evidence. No new telemetry or cluster access was needed.

Regenerate from the repository root:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/revise_poster_layout_20260929.py --evolve-b
```

**Reusable experience:** audit which control diagnostic is actually exported before choosing its cohort; raw-logit differences cannot supply missing probability TV. Exact model joins reuse the offline command evidence. Declare the requested font, verify the installed substitute, measure real glyph bounds and preserve data fingerprints when typography changes intentionally.
