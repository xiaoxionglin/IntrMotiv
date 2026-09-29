# A0 poster plan — revised 29 September 2026

## Title and central claim

# Intrinsic Motivation and Landmark Formation from Hippocampal Sequence Dynamics

Keep the submitted title. An optional subtitle or result banner is:

> **Broader exploration without reliable landmark control**

### Central question

> **Can hippocampal sequence dynamics organize useful landmarks and behavior without an external task reward?**

### Recommended headline

> **The frontier-based intrinsic architecture explores more broadly than the non-goal-conditioned baseline, but broader coverage does not establish controllable landmarks.**

The poster's positive result is the three-stage **exploration** comparison: C01 → C05 → C15. Its scientific qualification is that exploration, spatial coding, and commanded-target control do not improve together. This gives the progression a clear endpoint and an open computational question.

The current `Bernstein2026_IntrMotiv.svg` was reviewed on 29 September. Retain its introduction and three condition rows as the core. The latest revision preserves the supplied header and left column, uses 30 pt plot text and 40 pt explanations, and restricts the supporting architectural survey to DG 16. The weak displacement-dependent similarity profiles are removed. Other capacities and a major transfer section remain supplementary.

---

## 1. Introduction: from external reward to an internal learning signal

Retain the short connection to Lin et al. (2026): sparse visual DG input and a fixed CA3-like sequence generator supported rewarded navigation and spatial tuning. Here the task reward is removed, while sequence timing supplies an intrinsic signal.

Show one compact architecture and the two learning pressures:

- **DG representation:** encourage temporal separation between landmark events.
- **Policy:** shorten behavioral transitions between events; in the goal-conditioned versions, reward depends on the requested target.

Caption:

> CA3 sequence progression provides a temporal relationship between DG events. Representation learning favors separated events, while behavioral learning favors short transitions.

Elapsed event time is not established geodesic distance. Label these arrows as learning objectives, rather than guaranteed behavioral or representational outcomes. The visual trunk is fixed and ImageNet-pretrained; “without task reward” describes this IntrMotiv training phase.

Keep the introduction small enough for the result to dominate. Earlier rewarded-navigation benchmarks and the LSTM comparison belong to the predecessor citation, not the current conclusion.

---

## 2. Main comparison: three designs, three behavior patterns

Use plain condition names in the poster, with codes retained for figure matching:

| Poster label | Condition | Design distinction that must be disclosed |
| --- | --- | --- |
| Non-goal-conditioned RL | C01 | Temporal-distance reward for reaching a DG event, without an explicit requested goal |
| Goal-conditioned RL · uniform goals | C05 | Immediate target-conditioned worker; also includes DG global punishment and row repulsion |
| Goal-conditioned RL · frontier goals | C15 | UCB frontier manager with passive topology and direct final-target conditioning |

These are a progression of **implemented designs**, not a one-factor ablation. C05 differs from C01 in more than goal conditioning, and C15 is not simply C05 with its goal sampler replaced. C15 does not use explicit multi-hop waypoint planning. Reserve isolated causal claims about goal conditioning or frontier selection for controlled contrasts that hold the other components fixed.

Evidence identity: the original corrected-core study, F16 DG, seeds **8 / 99 / 123**, exact **100,040,704-frame** checkpoints, fixed 900-decision no-task-reward episodes. Later studies reuse “C15” for other architectures; do not mix those results into this comparison.

### Row A — C01: spatial modulation with repeated short cycles

**Replace the current heading** “Flat intrinsic motivation learns place fields, but just runs in loops” with:

> **Non-goal-conditioned RL: spatially modulated DG activity with frequent short cycles**

Keep the DG maps, active-unit peaks, and seed-99 trajectory. The maps show spatially varying activity, often fragmented or multi-field. The strict classifier identifies **zero mono-field units in seed 99** and only **1/48 units across the three runs**. Calling these examples compact place fields would overstate the evidence.

The three-seed mean 20-decision return fraction is **69.9% among mobile windows**. This supports frequent short cycles in the tested probes, rather than the general claim that the policy only loops or never explores.

**Interpretation, labeled as a hypothesis:** shortening time to any landmark event can be satisfied by revisiting familiar events. This is a plausible objective-level explanation of the observed cycling; the comparison does not isolate that mechanism.

### Row B — C05: fewer short returns, modest coverage gain

**Replace the current heading** “Goal conditioning alleviates looping, but the exploration is still bad” with:

> **Uniform-goal RL: fewer short returns, modest coverage gain**

Training coverage and unique cells increase over C01 in all three seeds, but only modestly. The 20-decision return fraction falls to **13.1%**, while only **19.8% of valid windows are mobile**, compared with 99.5% in C01. Its 40-decision return fraction is higher on average than C01's.

Put the mobility qualification beside the short-return result. Do not let a lower return rate imply uniformly better movement or elimination of loops at every timescale.

The option-hit markers illustrate recorded DG target events. They are not arrivals at a unique physical destination. C05's modest target-hit lift above shuffled targets is more encouraging than C15's, but does not establish robust landmark navigation.

### Row C — C15: broader exploration, weak commanded-target evidence

**Replace the current heading** “Selecting frontier goals helps with exploration, but success is rare” with:

> **Frontier-goal RL: broader exploration, weak commanded-target control**

C15 increases coverage AUC and unique cells over both C01 and C05 in **every paired training seed**. Relative to C01, the means are about **2.1× coverage AUC** and **2.7× unique cells per episode**. Its archived frozen probes also visit more bins than C01 in every seed.

C15's target-hit lift is below the shuffled-target reference in all three seeds. This is the central dissociation: a system can acquire broader experience while showing little evidence of selecting the commanded landmark.

The seed-99 event replay records **5 option-target hits and 123 option starts** over 10,001 observation-time poses. Use this as an illustration, not a replicated success estimate or a 5/123 success rate: starts and hit flags are not a fully adjudicated trial denominator. These flags include any current option target, including return/validation targets; they are not restricted to final frontier goals.

---

## 3. Quantify the progression and its boundary

### Evidence table for preparing captions

All values below are means of three independent seed values. Keep the table in the plan; use compact plots with individual seed dots on the poster.

| Measure | C01 | C05 | C15 | Meaning |
| --- | ---: | ---: | ---: | --- |
| Coverage AUC per episode | 38.0 | 42.0 | 78.6 | Final-10M-frame training-window mean |
| Unique cells per episode | 47.1 | 53.3 | 128.1 | Same training window |
| 20-decision returns among mobile windows | 69.9% | 13.1% | 5.9% | Archived frozen-policy probes |
| Mobile fraction of valid 20-decision windows | 99.5% | 19.8% | 58.4% | Required denominator context for returns |
| 40-decision returns among mobile windows | 25.7% | 35.9% | 14.6% | Shows the timescale qualification |
| Target-hit lift versus shuffled targets | n/a | 1.091 | 0.898 | Final training window; 1 is the reference |
| Active-only DG map cosine | 0.062 | 0.127 | 0.116 | Frozen probes; higher means more redundant maps |

Source: [matched comparison report](../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/report.md), [per-run measurements](../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/matched_terminal_per_run.csv), and [paired differences](../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/c01_paired_changes.csv).

The return test uses within-episode windows that travel more than 500 DMLab position units and end within 100 units of their start. Windows crossing resets are excluded. Retain this operational definition in a caption or QR-linked methods note. The windows overlap; they are not independent experimental replicates.

### Right-column panel A — exploration improves across the selected designs

Rename the current **“Spatial Representation”** heading above the coverage plots to:

> **Exploration across the three designs**

Reuse the matched data behind the [C01/C05/C15 summary SVG](../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/c01_c05_c15_summary.svg), rendered as **four compact plots in one horizontal row**. Each occupies about 96 mm of the 383 mm right column. Show every seed and its mean. Short two-line labels use **30 pt** at final A0 size; explanatory text uses **40 pt**, and markers are slightly larger. Coverage AUC and unique cells carry the main result. Keep the 20- and 40-decision return plots together, with mobile-window fractions and definitions in a shared caption underneath.

Suggested caption:

> In the final 10M training frames, coverage AUC and unique cells increase from C01 to C05 to C15 in all three seeds. Frozen-policy probes show fewer 20-decision returns in C05 and C15, but the mobility and 40-decision results qualify the loop interpretation.

### Right-column panel B — broader coverage does not establish goal control

Use a small **C05 versus C15 target-hit-lift** plot with all three seed dots and a reference line at 1. C01 is **not applicable**, since it has no requested target. These numbers already exist in the matched per-run table; no new replay is needed.

Pair it with raw-logit goal sensitivity. Both plots together occupy **half the right column** (191.5 mm, about one quarter of A0 width). Put explanatory text in the other half: C15 lift is below one in every seed, raw-logit sensitivity is small, and a DG target event does not verify physical arrival. Keep the design-package qualification underneath.

Suggested caption:

> C15 explores most broadly, yet commanded-target activation does not exceed the shuffled-target reference. C05 has modest positive lift. Coverage and target control therefore show different orderings.

These are online command-specificity diagnostics, not an exact-start causal intervention. A stronger control claim requires matched physical/memory starts and reliable commanded-node arrival. A dense stored graph does not substitute for this test.

### Right-column panel C — broader survey of architecture variants

Use **two small scatter panels at fixed DG capacity 16**, re-rendered from the [pinned architectural survey](../06_experiments/results/A0_poster_analysis_20260926/cross_run_scatter/report.md). Both show the same 168 eligible runs, representing 56 variants, with 30 pt plot text and larger markers. Color and shape now identify study family; capacity no longer contributes a hidden family grouping.

| Scatter | Cohort and measurements | Descriptive result |
| --- | --- | --- |
| A | DG-16 online spatial score versus recorded target-event hits / attempts | Spearman $\rho=0.583$, 168 runs |
| B | Distinct active DG peak bins versus the same hit fraction, for the same runs | $\rho=-0.424$, 168 runs |

Within the two largest families, spatial-score/hit correlations remain weak: CPD (CA3 feedback) $\rho=0.094$, 81 runs; DGP (DG policy) $\rho=-0.001$, 24 runs. The body caption states these values using the expanded family labels. Neither scatter establishes that spatial score causes hits or that diverse peaks harm control. Peak/hit correlation also weakens within CPD ($-0.130$) and DGP ($-0.239$).

Suggested message:

> At fixed DG capacity, spatial score and peak diversity tell different stories about target-event outcomes. More distinct peaks do not imply more target hits, and the positive score association weakens within the largest study families. A spatial summary alone does not establish useful goal identity.

The capacity filter changes the story: pooled score/hit correlation falls from $0.769$ (mixed capacities) to $0.583$ (DG 16). More importantly, the old mixed-capacity replay-score/coverage null ($\rho\approx0$, 38 runs) becomes $\rho=0.321$ in DG 16 (18 runs). Remove the old null wording rather than carrying it forward after filtering. DG-16 replay score versus command advantage remains weakly negative ($\rho=-0.226$, 18 runs), but this small, mixed-family probe sample is supplementary.

DG 64 has 57 online runs across different families and ages, with pooled score/hit $\rho=0.070$. Several within-family associations are positive, but the cohorts do not provide a clean, matched capacity comparison. Keep them separate in the [capacity audit](poster_20260929/dg_capacity_audit.csv) and [discussion notes](poster_20260929/discussion_points.md), rather than introducing another capacity storyline on the poster. DG 32 has only nine online rows, all from the capacity study.

These are **descriptive associations**, not causal tests or evidence of independence. Main-panel ages range from 25M to 150M frames. The target counters aggregate historical DG-event evidence, whereas spatial fields use recent saved windows; outcome rules and other architectural factors vary. A/B repeat the same runs and do not constitute independent evidence. Avoid grounded-controllability versus field-score here because field eligibility is part of that outcome definition.

The removed displacement profiles mainly show an overall similarity difference and little distance dependence. Keep their source assets supplementary; do not enlarge the vertical scale to emphasize a weak trend.

**Space priority:** concise three-design summary → concise command specificity → architectural survey with grouping/protocol controls → take-home and references.

---

## 4. Layout based on the current poster

Keep the two-column structure and the aligned C01/C05/C15 visual rows. The existing header and left side are protected under the user's instruction. The condensed scalar summaries support those rows; the broader survey fills the reclaimed right-column space with more observations.

| Region | Content | Current placement below the header |
| --- | --- | --- |
| Left column | Existing introduction, architecture, and three design rows | Preserved exactly |
| Upper right | Four exploration/return plots in one row; 40 pt caption underneath | Approximately 220–455 mm vertically |
| Middle right | Two goal-specificity plots in the left half; 40 pt explanation in the right half | Approximately 480–665 mm |
| Lower right | Two DG-16 architectural scatters; family color/shape legend and 40 pt interpretation | Approximately 696–975 mm |
| Bottom right | Take-home and visitor question; references | Approximately 988–1189 mm |

The right column begins at 443 mm and is 383 mm wide. Render plot sources at their final physical widths so compact panels keep readable fonts.

### Concrete changes to the current SVG when editing it

1. Preserve the supplied header and left-column objects. The alternative row headings in Section 2 remain proposed future edits, not changes applied to this SVG.
2. Align the four exploration/return plots horizontally and place their shared definitions and mobility qualification below.
3. Place the two goal-specificity plots together in half the right-column width; use the other half for their interpretation.
4. Use two smaller DG-16 scatters with 30 pt labels, legends, annotations, and ticks. A/B show different coding metrics for the same run cohort.
5. State that these design packages differ in more than goal choice; fixing capacity leaves family, age, and outcome-rule differences in the broader survey.
6. Keep the take-home focused on broad exploration and unresolved goal control; ask how a goal reliably identifies a physical destination. The concrete next test remains a matched-start command intervention.
7. Check SVG IDs, physical text sizes, clipping, and protected-region pixel equality after an Inkscape export. The preserved left-row labels remain small and their scientific wording remains a critique for later discussion.

Use the same seed-99 row illustrations already chosen. Label them **illustrative single-seed frozen probes**, while the quantitative claim uses all three seeds. C01's path is archived; C05/C15 option-marked paths are fresh replays from the same checkpoints. Their start/history/action sequences are not matched, and event markers must stay on their original replay paths.

The all-active peak maps show strongest mean-activation bins, not unique destinations or mono-field locations. Four highest-score maps are selected illustrations rather than population summaries. Independent color scales show relative field shape; equal colors do not imply equal activation amplitude. Preserve these distinctions in a shared caption.

---

## 5. Claim ledger

| Proposed claim | Decision | Recommended wording or reason |
| --- | --- | --- |
| The selected designs show progressive improvement | Use for episode coverage | “Coverage increases from C01 to C05 to C15 in all three training seeds.” |
| Frontier selection solves exploration | Narrow | “The frontier-based architecture improves coverage in this environment.” The C01/C05/C15 comparison bundles other differences. |
| Goal conditioning eliminates loops | Do not use | C05 has much lower mobility and higher mean 40-decision returns; other goal-conditioned cells do not reproduce this ordering. |
| Intrinsic learning produces place cells | Narrow | “Intrinsic learning produces spatially modulated DG activity.” The selected terminal seed-99 runs have no strict mono-field units. |
| Better coverage produces better spatial representations | Do not use | Strict mono-field counts and policy-conditioned map metrics do not improve monotonically. Common-history layer differences are a separate descriptive result. |
| The agent learns reliable landmark navigation | Do not use | C15's target-hit lift is below reference; peak locations, hits, and stored edges do not prove directed physical arrival. |
| Internal sequence timing causes the coverage gain | Unresolved | Timing supplies the objective, but this selected comparison has no sequence/timing ablation isolating its necessity or causal contribution. |
| Exploration and landmark control are separate requirements | Use | “Broader experience can emerge before internal goals become reliably controllable.” |
| Higher spatial score predicts better control across architectures | Do not use as a general rule | DG-16 pooled score/hit $\rho=0.583$, but within CPD/DGP it weakens; recorded DG hits do not establish physical control. |
| More distinct DG peaks improve control | Do not use | At fixed DG 16, peak/hit $\rho=-0.424$, also family-sensitive. Do not infer that diversity itself is harmful. |
| Spatial coding and behavior are statistically independent | Do not use | Limited heterogeneous samples and small descriptive correlations cannot establish independence. |

The neuroscience-facing contribution is a computational distinction: sparse event coding, the experience generated by an intrinsic objective, and behaviorally useful goal identity are separate requirements. Sequence timing is an available internal teaching signal; the data do not establish a biological mechanism or a task-general cognitive map.

---

## 6. Poster-ready conclusions and spoken story

### Replace the current conclusion with three points

- **Exploration:** the frontier-based intrinsic architecture increases coverage over the non-goal-conditioned and uniform-goal designs in all three seeds.
- **Spatial coding:** spatially modulated DG activity and distributed peaks occur, but compact place-field structure does not improve consistently with coverage.
- **Control:** the strongest explorer shows weak commanded-target evidence; broader exploration alone does not establish useful, controllable landmarks.

### Take-home box

> **Intrinsic sequence-based learning supports broad exploration in the frontier design. Turning spatial event codes into reliable goals remains unresolved.**

### Thirty-second explanation

> Our previous work used external reward to train a hippocampus-inspired navigation agent. Here sequence timing provides the internal learning signal. We compare a policy without an explicit goal, a policy given uniformly chosen internal goals, and a frontier-based goal architecture. Episode coverage increases across these designs, with the largest gain in the frontier system. But the strongest explorer is not the best at following landmark commands, and the spatial maps do not become uniformly better. The next problem is making internal landmarks into goals that reliably change where the agent goes.

### One future question

> **What goal representation makes different commands reliably select different outcomes?**

Recent CA3 context is one candidate for disambiguating repeated DG events. Keep this to a sentence or small arrow schematic, labeled as a hypothesis.

---

## 7. Supplementary material and reusable workflow

Keep the full transfer comparison, Direct/Waypoint contrasts, detailed DGP diagnostics, other C15 variants, graph matrices, field-criterion sensitivity curves, other capacities, and displacement kernels in supplementary material. The two DG-16 online survey views now belong in the main poster; the complete scatter gallery and replay/probe views remain supplementary. The [visitor discussion notes](poster_20260929/discussion_points.md) offer additional measured observations without adding plots to the main layout.

Use the [matched C01/C05/C15 report](../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/report.md) as the entry point, the [latest poster batch summary](../06_experiments/results/A0_poster_analysis_20260926/batch_summary.md) for supplementary outcomes, and the [telemetry contract](../04_implementation/reusable_place_field_telemetry.md) for metric meanings. Avoid relying on older narrative summaries when exact matched tables and current captions exist.

**What worked:** the rendered SVG exposed the actual space constraints and stale conclusion text; the saved matched table established which outcomes improve in every seed. The existing loop report and event provenance prevented confusing low motion, target flags, and directed arrival.

**What was slow:** broad historical-report searches produced long outputs and repeated old architecture inventories. Start future revisions with the selected comparison's report, table, and figure before expanding to other families.

**Authoritative checks:** recompute three-seed means and paired differences from `matched_terminal_per_run.csv`; check field claims against `mono_field_peak_counts.csv`; check option-marker semantics against `goal_option_provenance.json`; render the actual poster with Inkscape and visually inspect it. No new training, cluster access, or telemetry sweep is required to implement this plan.

For density revisions, replot from `cross_run_scatter/all_run_metrics.csv` rather than rebuilding the survey from all historical exports. The saved [poster point table](poster_20260929/architecture_scatter_points.csv), [statistics](poster_20260929/architecture_scatter_statistics.json), and [capacity audit](poster_20260929/dg_capacity_audit.csv) identify every selected observation and association. Set physical typography first: 30 pt plot text and 40 pt body text. Shorten repeated labels and decrease the data area rather than shrinking text. Recompute both the statistics and narrative whenever eligibility changes.
