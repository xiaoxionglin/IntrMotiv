# Mean intrinsic reward during COSYNE training

[Linear SVG](01b_online_intrinsic_reward_course.svg) · [Linear PDF](01b_online_intrinsic_reward_course.pdf) · [Log-y SVG](01c_online_intrinsic_reward_course_log.svg) · [Log-y PDF](01c_online_intrinsic_reward_course_log.pdf) · [Log-y preview](01c_online_intrinsic_reward_course_log.png) · [Seed-level course](interpolated_by_seed.csv) · [Mean and SEM](arm_summaries.csv) · [Collection provenance](collection_manifest.json) · [Render provenance](render_manifest.json)

These are **selectable figure candidates**, not replacements for the current Figure 2. Both match the 1.9 × 1.9-inch physical size, Arial typography, axis placement, line width, translucent one-SEM bands, right-end labels, and architecture colors of the existing [online DG spatial course](../cosyne_2027_compact_figures/01_online_spatial_course.svg). NG/C01 is blue, UG/C05 orange, and FG/C15 green. The original version uses a shared **linear** vertical scale, on which UG is close to zero. The added version uses a shared **logarithmic vertical axis** from 0.01 to 5 intrinsic-reward units per decision. This makes UG's trend visible and shows FG's early rise followed by a roughly level later course.

The log version changes only the display axis. Both versions plot the same arithmetic five-seed means and additive ±1 SEM bands; neither takes logarithms before averaging. Every plotted mean-minus-SEM bound is positive (minimum 0.0131), so the log version hides no value on the shared 1–99M-frame grid. The raw histories contain zero at frame zero, which is outside the plotted range of **both** variants. On a log axis, equal vertical distances represent equal multiplicative ratios rather than equal reward differences. The linear version remains the appropriate view for comparing absolute reward-unit differences.

## Measured quantity and cohort

The exact logged scalar is `intrmotiv/reward/intrinsic_mean`, the internal IntrMotiv reward **before PPO-specific reward routing**, rather than external DMLab reward or episode return. Each logged point is the learner's mean over **2,048 valid decisions, including zero-reward decisions**. The separately logged `intrmotiv/reward/intrinsic_sum` divided by its corresponding nonzero mean is 2047.99977–2048.00022 across the archived points, confirming the denominator. The vertical axis is the logged mean in intrinsic-reward units per decision. NG, UG, and FG use different intrinsic-reward schemes, so the height of one curve does not by itself measure control quality or task performance.

The cohorts are the same verified terminal training runs as the online DG course: temporal DG credit in all three designs, temporal worker-hit credit in UG and FG, and five seeds per design (8, 17, 41, 99, 123). The [collection manifest](collection_manifest.json) names all 15 exact W&B run IDs, StudySpec identities and hashes, source URLs, archived history hashes, and first/last environment frame. Each run supplies 3,051 paired reward records from zero through 100,007,940 frames; there are no missing runs or plotted training points.

For each run, the unsampled logged reward history is linearly interpolated onto the same 1M-frame grid used by the DG course, from 1M through 99M frames. At every grid point, the line is the arithmetic mean across five **training seeds** and the colored band is ±1 standard error across those seeds. Interpolation aligns asynchronously logged updates; it is not temporal smoothing and it does not add independent observations. The [seed-level table](interpolated_by_seed.csv) and [arm summary](arm_summaries.csv) retain exactly the plotted values. At 99M frames, the means ± SEM are NG 2.599 ± 0.044, UG 0.050 ± 0.012, and FG 1.440 ± 0.245 intrinsic-reward units per decision.

The existing DG spatial-course cache did not contain reward scalars. The [collector and renderer](../../cosyne_2027/plot_intrinsic_reward_course.py) therefore read only the named reward tags from the same audited W&B run identities, archived lightweight per-run CSVs under [`histories/`](histories/), and render locally from that cache. The current training-run catalog contains later additions and has a different whole-file hash from the earlier DG archive; the collector checked the 15 selected run identities, StudySpec hashes, factor settings, seeds, and verified-completion status row by row rather than treating unrelated catalog additions as a mismatch.

To rerender without W&B access, run from the repository root:

```bash
MPLCONFIGDIR=/tmp/cosyne_intrinsic_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/cosyne_2027/plot_intrinsic_reward_course.py
```

Add `--collect` only when intentionally refreshing the pinned W&B histories. The renderer verifies history hashes and complete five-seed coverage before exporting both editable SVGs, PDFs, and PNGs. It rejects a log-y render if any mean or SEM bound is nonpositive or outside its printed range.
