# High policy-credit worker timing

[Compact Figure-2-style PDF](high_credit_worker_compact.pdf) · [SVG](high_credit_worker_compact.svg) · [PNG preview](high_credit_worker_compact.png) · [paired seed values](paired_seed_differences.csv) · [source mapping](sources.json) · [compact renderer](plot_high_credit_compact.py)

**Relative-count alternative:** [PDF](high_credit_worker_cplus1_compact.pdf) · [SVG](high_credit_worker_cplus1_compact.svg) · [PNG preview](high_credit_worker_cplus1_compact.png) · [per-arm counts and relative values](high_credit_cplus1_seed_values.csv)

**Shared hit-rate axis, dual graph-edge axes:** [PDF](high_credit_worker_shared_hit_rate_dual_edges_compact.pdf) · [SVG](high_credit_worker_shared_hit_rate_dual_edges_compact.svg) · [PNG preview](high_credit_worker_shared_hit_rate_dual_edges_compact.png) · [paired per-arm hit rates](high_credit_hit_rate_pairs.csv) · [renderer](plot_high_credit_dual_axes.py)

**DG representation, including Gaussian field-peak coverage:** [five-panel PDF](high_credit_worker_representation_compact.pdf) · [SVG](high_credit_worker_representation_compact.svg) · [PNG preview](high_credit_worker_representation_compact.png) · [individual peak-coverage card](representation_cards/peak_coverage.svg) · [individual map-overlap card](representation_cards/map_overlap.svg) · [per-arm and paired-seed values](high_credit_representation_pairs.csv) · [renderer](plot_high_credit_representation.py)

**Actual peak locations and spacing check:** [two illustrative seed pairs](high_credit_peak_examples.pdf) · [all C05/UG pairs](high_credit_peak_locations_c05.pdf) · [all C15/FG pairs](high_credit_peak_locations_c15.pdf) · [area-versus-spacing diagnostic](high_credit_peak_area_vs_spacing.pdf) · [renderer](plot_high_credit_peak_locations.py)

**Earlier dual C/D axes:** [PDF](high_credit_worker_dual_axis_hit_rate_compact.pdf) · [SVG](high_credit_worker_dual_axis_hit_rate_compact.svg) · [PNG preview](high_credit_worker_dual_axis_hit_rate_compact.png)

The selected compact figure is 7.5 × 2.5 inches. It follows the submitted Figure 2's Arial typography, C05 orange/C15 green palette, quartile boxes with individual seed dots, min–max whiskers, and explicit signed $\Delta$ axes. All four panels show **only** the high policy-credit regimen, with C05 and C15 on each horizontal axis. Each panel uses its own vertical unit and range.

## Compact figure caption

**Temporal rather than constant worker-hit credit under high policy credit.** The temporal DG encoder setting is fixed in both worker arms. The two arms share $\gamma=0.999$ and GAE $\lambda=0.995$; the figure therefore shows within-regimen worker timing differences. The change from the original $\gamma=0.99$, $\lambda=0.95$ cohort raises both parameters together, and that older cohort is not plotted here. Every colored dot is one temporal-minus-constant matched training-seed difference (seeds 8, 17, 41, 99, and 123; $n=5$ pairs per architecture). Boxes span the first to third quartiles, the line marks the median, and whiskers extend to the observed minimum and maximum. The gray line is zero; an upward difference means a larger measured value in the temporal worker arm. No significance test is encoded.

**A, DG information:** active-unit mean spatial information, in bits, from each model's own-policy 100,000-decision snapshot at 100M training frames. **B, Coverage AUC:** logged coverage area under the curve over 90–100M training frames. **C, Correct hits:** total training observations with a correct internally detected target hit over that window. These repeated observations are **not independent physical goal arrivals**. **D, Reliable edges:** the stored graph's reliable-edge count in the 100M own-policy snapshot; this describes detector-memory structure, not physical goal arrival. The own-policy snapshots follow different trajectories, and the four readouts measure distinct outcomes.

The [paired table](paired_seed_differences.csv) retains the five high-credit differences for each architecture, along with the earlier original-cohort rows. Only `policy_credit=high` enters this figure. All 40 high-credit seed/readout differences were recomputed from the [source files and hashes](sources.json). Regenerate with:

```bash
MPLCONFIGDIR=/tmp/cosyne_high_compact_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_high_credit_compact.py
```

## Relative-count alternative: panels C and D

The second compact figure retains the same A and B panels. For **C, correct-hit observations**, and **D, reliable graph edges**, each seed instead shows $100(T-C)/(C+\epsilon)$ on the **relative change (%)** axis, where $T$ and $C$ are the temporal- and constant-bonus arm counts for that matched architecture and seed, and $\epsilon=1$ count. The added one is **one observation** in C and **one edge** in D. Zero means equal counts; positive values favor the temporal arm. The visible C scale runs from −25% to +100%, and the D scale from −100% to +100%; extra headroom holds the numeric labels. An upward triangle at the top marks **each seed above +100%**, with its true value written beside it. Such values, including their box or whisker portions, extend beyond the displayed data scale; none are silently omitted. Open diamonds mark arithmetic means across the five seed-level ratios. When a mean also exceeds +100%, its diamond is placed at the top and its true value is labeled. These are ratios on a percent scale, but they are not literal percentage changes from the constant arm when $C$ is small or zero. Boxes, medians, min–max whiskers, and dots otherwise have the same definitions as in the absolute-difference figure. The high-credit setting, five seeds, observation windows, and own-policy graph protocol are unchanged.

The [per-arm table](high_credit_cplus1_seed_values.csv) records all 20 counts and ratios. Its C and D source CSVs are the pinned `a26852f` files named in [source mapping](sources.json). The renderer reloads those files from Git history, checks every $T-C$ against the original paired table, then exports the relative figure and table. Regenerate with:

```bash
MPLCONFIGDIR=/tmp/cosyne_high_relative_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_high_credit_compact.py --relative-counts
```

**Interpretation limit:** at 100M frames, the C05 constant graph has zero reliable edges in seeds 8, 41, and 99. The added one makes their plotted values finite: 0%, 300%, and 300% respectively. These are count differences divided by a stabilizing one-edge denominator, not estimates of a conventional percent change from a nonzero baseline. C05 seed 41 also appears at 586% for correct hits because its constant-arm count is only 1,687. The corresponding C05 mean values are 131% for hits and 116% for graph edges; both lie above the drawn +100% tick and are labeled. The raw counts and absolute-difference figure should accompany any cross-architecture reading of these panels.

## Shared hit-rate scale with dual graph-edge axes

**C now puts C05 and C15 hit-rate differences on one shared y-axis.** Their observed ranges fit within the same −15 to +15 hits-per-1,000 scale, so their heights can be compared directly. **D retains separate graph-edge scales:** orange left ticks apply only to C05, green right ticks only to C15, with zero lines aligned. Equal vertical heights in D do **not** imply equal edge-count differences. A and B remain the same absolute-difference panels. Each C or D box spans its architecture's five seed-paired differences; whiskers reach the observed minimum and maximum, dots are seeds, and the open diamond is their arithmetic mean. The figure remains 7.5 × 2.5 inches and includes only the high policy-credit regimen ($\gamma=0.999$, GAE $\lambda=0.995$). The [earlier candidate](high_credit_worker_dual_axis_hit_rate_compact.pdf) used separate C05 and C15 axes for C as well; its hit-rate data and D panel were otherwise the same.

**C, commanded target-hit rate:** for each arm and seed, divide the sum of `target_hit_numerator` by the sum of `target_hit_event_count` over the 90–100M-frame online window, then multiply by 1,000. The source summaries average both components over the same 306 learner updates, so their ratio equals the ratio of those sums. The paired value is the temporal-arm rate minus the constant-arm rate, in **hits per 1,000 eligible logged events**. It is an online training diagnostic, **not a per-episode success fraction or independent physical-goal arrival rate**. It replaces the earlier sum of repeated learner-batch correct-hit observations, which had no meaningful episode denominator. The rate uses a distinct target-hit numerator and eligible-event denominator; it is not the earlier correct-hit count divided by 306. The [per-arm table](high_credit_hit_rate_pairs.csv) records both components, both rates, and every paired difference. The mean paired changes are +5.19 per 1,000 for C05 and +2.06 per 1,000 for C15; C05 has four positive seed differences and C15 has three. C15 seed 8's denominator also changes substantially between arms (1,276 to 606 mean eligible events per update), so the rate contrast should be read alongside its components.

**D, reliable graph edges:** the original temporal-minus-constant 100M own-policy snapshot difference, without the relative-count offset. C05 differences range from −12 to +3 edges (mean −0.8); C15 differences from +11 to +54 edges (mean +21.6). The dual axes show each architecture's spread without imposing a shared count range. They do not turn graph edges into a normalized measure of control or make the two graph regimes directly equivalent.

The C source is each architecture's pinned `online_per_run.csv` at commit `a26852f`; the D source is the high-credit portion of the [paired difference table](paired_seed_differences.csv), itself checked against the pinned spatial summaries. The relevant source CSVs and hashes are listed in [source mapping](sources.json). Regenerate the shared-C candidate and hit-rate table with:

```bash
MPLCONFIGDIR=/tmp/cosyne_high_shared_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_high_credit_dual_axes.py --shared-hit-axis
```

Omit `--shared-hit-axis` to regenerate the earlier dual C/D candidate.

## Additional DG representation measures at high policy credit

The [five-panel comparison](high_credit_worker_representation_compact.pdf) uses the same five C05 and five C15 matched training-seed pairs (8, 17, 41, 99, 123) as the preceding timing comparison. The plotted architecture labels are **UG** for C05 (uniform goal) and **FG** for C15 (frontier goal); source tables retain C05/C15. Both worker-bonus arms have temporal DG encoder credit, policy $\gamma=0.999$, and GAE $\lambda=0.995$. Every DG readout comes from the model's own-policy, 100,000-decision spatial snapshot at the 100M-frame checkpoint. Each colored dot is one temporal-minus-constant worker-bonus seed difference, the open diamond is the arithmetic mean, the box spans the first and third quartiles with a median line, and whiskers reach the observed minimum and maximum. The gray line is zero. Positive values mean more of the named quantity under temporal worker credit; no significance test or confidence interval is encoded. Both architectures share a vertical scale **within** each panel, but the five panels use different units and ranges.

Every panel also has an [individual SVG/PDF/PNG card](representation_cards/) in the Figure 2 loose-canvas style. **These cards are the primary reusable layout assets; the five-panel sheet is a supplemental overview.** Each card is 2.25 × 1.95 inches, with a measured plot region no larger than 1.2 × 1.0 inches **including tick and axis labels but excluding the title**. Titles are 12 pt, ticks 8 pt, axis labels 10 pt, seed dots have opacity 0.7, and box outlines, whiskers, and medians sit above the dots. The overview is 7.5 × 5.0 inches; the earlier four-panel sheet was 7.5 × 2.5 inches. The [new field-peak coverage card](representation_cards/peak_coverage.svg) and [matching map-overlap card](representation_cards/map_overlap.svg) can be laid out separately without expanding the submitted compact Figure 2.

| Measure, temporal minus constant worker bonus | UG/C05 mean; positive seeds | FG/C15 mean; positive seeds |
| --- | ---: | ---: |
| A. Active-only mean DG map cosine | +0.0185; 2/5 | +0.0175; 3/5 |
| B. Gaussian effective field-peak area fraction | −0.0019; 3/5 | +0.0045; 1/5 |
| C. Distinct active DG peak bins | −0.6 bins; 1/5 (one tie) | +0.2 bins; 1/5 (three ties) |
| D. Single-field fraction of field-eligible units | +0.0625; 4/5 | −0.0375; 0/5 (two ties) |
| E. Median nearest-neighbor distance among the same peaks used for B | +8.3 DMLab units; 2/5 (one tie) | +15.1 units; 3/5 |

**A, map overlap:** mean pairwise cosine of the active units' post-inhibition spatial rate maps over each run's occupied bins. Lower cosine means less overlap among unit maps within that run. C05 seed 123 contributes a +0.212 difference, while three of its other four seeds have negative differences. **B, Gaussian field-peak coverage:** for each snapshot, retain DG units whose active fraction is at least 0.25%, which meet the evaluator's field-eligibility test of at least 20 active observations in at least three spatial bins, and whose dominant peak coordinates are finite. Give every retained peak equal mass, blur it by a Gaussian of width 100 DMLab units on the 19 × 19 grid, normalize the resulting density on the **same 318-bin pooled visited mask for all 20 snapshots**, and divide the exponential of its Shannon entropy by 318. C05 has 15 coverage-qualified peaks per arm and seed; C15 has 16. This is the same metric definition and cutoff used for the encoder-credit Figure 2 field-peak coverage panel, with a separately declared high-credit mask. **C, distinct bins** counts spatial bins containing a dominant active-unit peak; this count does not describe how evenly those bins cover the arena. **D, single fields** is the evaluator's fraction among field-eligible DG units. **E, cutoff-matched peak spacing** is the median nearest-other-peak distance among exactly the peaks used for B. The archived source's original spacing measure included all field-eligible peaks, including one C05 unit per run below B's 0.25% activity cutoff. Both versions remain in the [paired table](high_credit_representation_pairs.csv). Distinct bins and peak spacing provide different concentration checks; neither substitutes for panel B.

Panel B's exact paired differences, in seed order 8, 17, 41, 99, 123, are **UG:** +0.035655, +0.087194, −0.065139, +0.018900, −0.086216; **FG:** −0.026894, −0.070327, +0.192990, −0.044917, −0.028365. The UG median is +0.0189 despite its near-zero negative mean; the FG median is −0.0284 despite its slightly positive mean, which is driven by seed 41. Thus the new coverage measure does **not** show a consistent gain under temporal worker credit in either architecture.

### Why coverage and nearest-peak distance appeared to disagree

The original E card used the canonical snapshot summary's all-eligible nearest-peak distance. The B coverage card excludes peaks below 0.25% activity. In C05, exactly one otherwise eligible DG unit in **each** of the ten snapshots falls below that cutoff. Recomputing nearest-peak distance on B's 15 peaks changes the C05 paired mean from **+67.2 to +8.3 DMLab units** (positive in 2/5 rather than 3/5 seeds). For seed 99, the original +114.8-unit difference becomes **0**: the constant and temporal arms both have a 316.2-unit median among the coverage-qualified peaks. The [paired geometry diagnostic](high_credit_peak_area_vs_spacing.pdf) shows every seed before and after matching the cutoff. The original all-eligible values remain in the table as `*_median_dominant_peak_nearest_neighbor_distance`; the revised panel E uses `*_cutoff_median_nearest_peak_distance`. A source cross-check reproduces every original snapshot median from the archived coordinates to within $10^{-4}$ units.

This cutoff mismatch explains much of the **C05 mean** discrepancy, but the two statistics also ask different spatial questions. Area is entropy of a 100-unit-blurred, equal-mass density across the fixed visited mask; nearest-peak distance is a median of local raw-coordinate distances. A group mean near zero also combines positive and negative area changes; no seed has exactly zero area change. In C15 seed 123, all 16 peaks pass the cutoff, yet area changes by −0.0284 while median nearest-peak distance rises by 100 units. The [C15 peak sheet](high_credit_peak_locations_c15.pdf) shows its temporal peaks farther apart locally but more concentrated in separated edge groups; greater local spacing need not spread probability mass evenly over the full mask.

The [example comparison](high_credit_peak_examples.pdf) makes the two mechanisms easier to inspect; the [C05 peak sheet](high_credit_peak_locations_c05.pdf) and [C15 peak sheet](high_credit_peak_locations_c15.pdf) show **all five seed pairs**, not selected exemplars. Constant and temporal worker arms use identical x/y world-coordinate axes and the same Gaussian density scale across the two full sheets. Filled architecture-colored circles are the peaks used in B and revised E; a red cross marks an eligible C05 peak excluded by the activity cutoff, and a number inside a circle identifies coincident peaks. Gray shading is the normalized Gaussian peak density on the shared mask; unvisited bins are gray. Each panel prints its area fraction and cutoff-matched median nearest-peak distance. Unit identities are not aligned across independently trained networks. Regenerate the four diagnostic figures from the archived lightweight snapshot inputs with:

```bash
MPLCONFIGDIR=/tmp/cosyne_peak_locations_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_high_credit_peak_locations.py
```

The preceding figure's DG spatial-information axis denotes **amplitude-weighted bits per decision** averaged over active units, not bits per activation. Its evaluator computes occupancy-weighted $\sum_x p(x)r(x)\log_2[r(x)/\bar r]$ without dividing by mean activity $\bar r$; “$\Delta$ bits” there should be read as $\Delta$ bits/decision. The own-policy snapshots here follow different trajectories between arms. Their mean visited-cell-fraction differences are nearly zero (C05 +0.0011; C15 0.0000), but the sampled observations and occupancy weights are not matched. Panel B's shared mask supplies a **common spatial denominator**, not identical experiences or a fixed-input representation intervention. The earlier encoder-credit Figure 2 panel used a different 242-bin common-replay mask, so its absolute area fractions should not be compared directly with these 318-bin high-credit fractions.

The [paired table](high_credit_representation_pairs.csv) retains both arms, all seed differences, eligible-peak counts, the shared-mask hash, and each source NPZ hash. The [lightweight peak-input archive](high_credit_peak_snapshot_inputs.json) contains the 20 occupied-bin masks, active fractions, eligibility flags, and peak coordinates required to regenerate panel B; raw NPZs remain under the allocated NEMO2 workspace. The [provenance record](high_credit_peak_area_provenance.json) records the 318-bin mask hash (`fc9481bbb296fae9facc6890267a828266e5d0554493ee98d471e4cc7a454e5a`), cutoff, field criterion, Gaussian width, and input hash. A separate direct-NPZ recalculation of all 20 areas agreed with the archived-input results to a maximum absolute error of $5.0\times10^{-16}$ and reproduced the mask hash. [Source mapping](sources.json) ties the pinned `a26852f` CSVs and NPZ paths to the comparison. To refresh the lightweight input archive from NEMO2, then regenerate the table and figures:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python collect_high_credit_peak_inputs.py
MPLCONFIGDIR=/tmp/cosyne_high_representation_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_high_credit_representation.py
```

Only the first command needs NEMO2 access. The second imports the existing canonical Gaussian peak-area implementation and uses the archived lightweight inputs locally.

## Earlier full-size comparison

[C05 PDF](c05_worker_regimen_boxplots.pdf) · [C15 PDF](c15_worker_regimen_boxplots.pdf) · [combined overview](worker_regimen_boxplots.pdf) · [full-size renderer](plot_worker_regimen_boxplots.py)

These earlier figures compare original and high-credit cohorts and have editable SVG and 300-dpi PNG variants. The family-specific versions give the hit-observation and graph-edge panels more vertical resolution. They are retained for the cohort comparison; the compact figure above is the high-credit-only result.

### Earlier figure caption

**Seed-paired effects of temporal rather than constant worker-hit credit in C05 and C15.** Each point is the temporal-minus-constant difference for one matched training seed (8, 17, 41, 99, or 123); marker shape identifies the seed throughout. Both worker-bonus arms use temporal DG encoder credit. The original policy-credit setting is $\gamma=0.99$, GAE $\lambda=0.95$; the high-credit setting is $\gamma=0.999$, GAE $\lambda=0.995$. **Both policy parameters change together.** Boxes show the 25th–75th percentiles, black bars the median, whiskers the observed minimum and maximum, and orange crosses the mean. Every group has five training-seed pairs. Horizontal lines mark zero; no hypothesis test or significance marker is shown.

**A,** Active-unit mean DG spatial information in bits from each network's own-policy, 100,000-decision snapshot at the 100M-frame checkpoint. **B,** Logged coverage area under the curve over training frames 90–100M. The positive mean for original C15 is driven by seed 123 ($+41.1343$ AUC units); its other four seed differences are negative. **C,** Total logged observations with correct target hits during training frames 90–100M, summed within each seed. These are repeated training telemetry observations, **not independent physical goal arrivals**. **D,** Reliable stored graph-edge counts from the 100M own-policy snapshot, which describe detector-memory graph structure rather than physical goal arrival. The different readouts have separate vertical scales.

The original C15 constant-bonus runs used a recovery learner patch absent from the original temporal-bonus runs. The original and high-credit settings are separate training cohorts; their differences cannot isolate the effect of $\gamma$, $\lambda$, or that source patch. Own-policy spatial snapshots also follow different trajectories. The graph, hit, and coverage measures address distinct outcomes and should not be interpreted as repeated estimates of one quantity.

### Earlier data and regeneration

The [paired table](paired_seed_differences.csv) contains the 20 plotted rows and identifies the relevant source bundle for each row. [Source mapping](sources.json) lists the per-metric source files and locally available SHA-256 hashes. All 40 high-credit values and 15 original-C05 spatial/hit/graph values were recomputed from those files. Original-C05 AUC and all original-C15 differences retain the rounded seed values supplied with the source task; the automatic approval review blocked exporting the private NEMO2 summary CSVs for independent reloading in this worktree. Original-C05 spatial differences were replaced with exact locally recomputed values. The source task had already calculated the remote differences using the listed canonical summaries and cached NPZs.

Rebuild the vector figure and 300-dpi preview with:

```bash
MPLCONFIGDIR=/tmp/cosyne_regimen_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_worker_regimen_boxplots.py
```

The plot uses the paired CSV directly and introduces no smoothing, normalization, or significance calculation. The family figures are designed for 7.2-inch width and the overview for 8.4-inch width; all labels are at least 12 pt at their intended size.
