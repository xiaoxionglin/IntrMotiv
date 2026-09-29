# Poster critique and applied revisions

[Revised editable poster](Bernstein2026_IntrMotiv_filled.svg) · [Preview](Bernstein2026_IntrMotiv_filled.png)

The selected C01 → C05 → C15 contrast is a useful poster story because its coverage ordering holds in all three seeds, while commanded-target evidence shows a different ordering. The most defensible claim remains **broader exploration without established reliable landmark control**. At fixed DG 16, spatial score and peak diversity associate with recorded target hits in opposite directions. This makes **what defines a useful landmark?** a good discussion question, while leaving architectural and measurement differences explicit.

## Changes applied to the right column

- Condensed the four exploration/return plots into a horizontal row with a shared caption. These low-information panels no longer occupy two large rows.
- Put the two goal-specificity plots together in half the right-column width and explanatory text beside them.
- Set all new plot text to 30 pt and explanatory text to 40 pt at final A0 size; section headings use 48 pt. Short labels, smaller data areas, and slightly larger markers replace the previous small-font layout.
- Highlighted C15's approximately 2.1× coverage AUC and 2.7× unique cells relative to C01. These are ratios of three-seed means, not statistically tested effect estimates.
- Defined coverage AUC as time-averaged cumulative visited-cell count. Kept the final-10M training means distinct from frozen-policy returns.
- Corrected the sensitivity label to **mean absolute raw-logit change**. It is not action-probability distance and its scale alone cannot establish how strongly actions depend on a goal.
- Retained the shuffled-target reference, all seed observations, and the C05 low-mobility qualification.
- Removed the displacement profiles. Their clearest difference is an overall similarity shift; the small distance-dependent change does not justify a prominent spatial-distance claim. The supplementary assets are retained without modifying or visually amplifying the data.
- Restricted the main architectural survey to DG 16. Two panels now show spatial score and distinct peak bins versus the same target-event outcome for the same 168 runs. Color and shape identify families; capacity is fixed.
- Recomputed the capacity-specific story and removed the old mixed-capacity replay-score/coverage null. Smaller, heterogeneous replay/probe cohorts and DG 32/64 are documented separately.
- Added a visitor question about DG identity versus sequence context. The concrete next test remains changing the commanded goal from matched physical/memory starts and measuring the reached landmark; context is a candidate hypothesis, not an established solution.

## Highest-priority critiques of the preserved left column

### 1. “Learns place fields” overstates the displayed evidence

The C01 seed-99 maps show spatial modulation, but zero units satisfy the shared strict mono-field criterion. The displayed terminal C05 and C15 seed-99 runs also have zero qualifying mono-field units. Distinct peak coordinates do not establish compact, unique destinations.

Suggested C01 heading:

> **Spatially modulated DG activity with frequent short cycles**

This still supports an interesting representation result without claiming classical place cells. Source: [matched report and mono-field results](../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/report.md#mono-field-peak-locations).

### 2. “Goal conditioning alleviates looping” is too causal and too general

C05 also adds DG global punishment and row repulsion. C15 changes manager/topology machinery; it is not C05 with only its sampler replaced. The three rows are design packages, not an isolated progression of one factor.

C05 has lower 20-decision mobile-window returns, but only 19.8% of its valid windows are mobile, compared with C01's 99.5%. Its mean 40-decision returns are higher than C01's. The lower short-return rate is meaningful but cannot establish generally better motion or elimination of loops.

Suggested C05 heading:

> **Uniform-goal design: fewer short returns, modest coverage gain**

Suggested C15 heading:

> **Frontier-goal design: broader exploration, weak goal specificity**

Source: [matched per-run table](../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/matched_terminal_per_run.csv).

### 3. The left-row figures are too small for comfortable poster reading

The three aligned rows aid comparison, but each row includes four tiny rate maps, a labeled peak atlas, and a trajectory. Unit IDs, tick labels, and colorbars require close inspection. For a future edit, retain two rate maps per condition or one small population summary, then enlarge the trajectory and the shared figure labels. Do not select fields merely because they look compact; disclose the ranking rule and retain population statistics.

The event-marked C05/C15 paths are fresh replays, while C01's is archived. They do not have identical reset/history/action sequences. The star counts are illustrative event flags, not a fully adjudicated trial success rate.

### 4. The introduction could give more space to the scientific comparison

The architecture is repeated at several levels, and the external-versus-intrinsic objectives are dense. A later revision could use one architecture schematic and one paired learning-objective diagram. The current header and left content were preserved under the user's explicit constraint.

## Interpretation limits for the right column

- A target-hit lift below one is a strong boundary on the target-control claim. It does not prove that the frontier manager is irrelevant, nor diagnose the cause of failure.
- Raw-logit sensitivity is scale-dependent. A future action-probability total-variation diagnostic would be easier to interpret if compatible saved data are available. Do not substitute an invented probability effect.
- Holding DG capacity at 16 changes score/hit correlation from $\rho=0.769$ across mixed capacities to $\rho=0.583$ across 168 runs. Within CPD it remains $\rho=0.094$ (81), and within DGP $\rho=-0.001$ (24). The positive pooled association is not a general within-family result.
- DG-16 peak diversity versus hits has $\rho=-0.424$. Within CPD/DGP it weakens to $-0.130/-0.239$. The defensible statement is that more peak locations do not imply more hits; do not claim diversity causes poor control. Peak bins describe maxima, including broad/multifield units, and do not establish unique destinations.
- Spatial score includes an activation-amplitude factor. It is not pure spatial selectivity or normalized information per activation. Comparing a score and a peak count does not compare equivalent properties of the representation.
- The old mixed-capacity replay-score/coverage null ($\rho=-0.0003$, 38 runs) changes to $\rho=0.321$ in DG 16 (18 runs). Calling this a null result after filtering would be wrong. Replay score/command benefit is $\rho=-0.226$ in the same small DG-16 sample; this is supplementary screening evidence.
- DG 64 has pooled score/hit $\rho=0.070$ across 57 runs and positive associations within some smaller families. DG 32 has nine runs from one family. These are neither interchangeable architectural variants nor a matched capacity intervention; their separate audit does not justify adding a second poster storyline.
- Main-panel ages range from 25M to 150M frames. Historical target counters differ in scope from recent spatial windows; outcome rules also vary across designs. Keeping capacity fixed does not remove these limitations or make the correlations causal.
- A/B are two measurements of the same eligible cohort, not independent replication. All 56 variants have three run rows here. Other DG-16 rows without prospective counters are unavailable for these plots; corridor geometries are not pooled into the original environment.
- A field-score versus grounded-controllability scatter would be partly circular because field eligibility appears in the outcome definition. The selected target-event outcome avoids that specific definitional dependence, while still falling short of verified physical goal control.
- The present comparison has no isolated sequence/timing ablation. It demonstrates behavior of a sequence-based intrinsic system, not that sequence timing is necessary or uniquely responsible for the gain.

The main take-home therefore remains:

> **Broader exploration does not yet establish reliable control over landmark goals.**

## Verification and reusable experience

The reviewed build preserves the supplied header and left column pixel for pixel, checks native SVG object bounds and unique IDs, and uses final A0 sizes of 30 pt for plot text and 40 pt for explanations. Saved scatter points and correlations are checked against the pinned survey table; each panel contains only DG 16. Full-page and right-column renders are the authoritative layout checks. [Visitor notes](discussion_points.md) link the supplementary predictor/looping and graph/execution observations to their original per-seed records.

What worked: explicit physical font units, measured text wrapping, pinned points, capacity-specific recomputation, and a two-panel metric comparison. What failed in earlier drafts: technically legible but undersized 13 pt fonts and an architecture story that pooled capacity. Set poster typography first, simplify labels, and re-evaluate the narrative after filtering. Consult the [metric reference](../../04_implementation/IntrMotiv_metric_reference.md) and per-seed records before promoting an observation to a control claim.
