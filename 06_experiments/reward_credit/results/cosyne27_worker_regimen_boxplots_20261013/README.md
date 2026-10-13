# High policy-credit worker timing

[Compact Figure-2-style PDF](high_credit_worker_compact.pdf) · [SVG](high_credit_worker_compact.svg) · [PNG preview](high_credit_worker_compact.png) · [paired seed values](paired_seed_differences.csv) · [source mapping](sources.json) · [compact renderer](plot_high_credit_compact.py)

**Relative-count alternative:** [PDF](high_credit_worker_cplus1_compact.pdf) · [SVG](high_credit_worker_cplus1_compact.svg) · [PNG preview](high_credit_worker_cplus1_compact.png) · [per-arm counts and relative values](high_credit_cplus1_seed_values.csv)

**Shared hit-rate axis, dual graph-edge axes:** [PDF](high_credit_worker_shared_hit_rate_dual_edges_compact.pdf) · [SVG](high_credit_worker_shared_hit_rate_dual_edges_compact.svg) · [PNG preview](high_credit_worker_shared_hit_rate_dual_edges_compact.png) · [paired per-arm hit rates](high_credit_hit_rate_pairs.csv) · [renderer](plot_high_credit_dual_axes.py)

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
