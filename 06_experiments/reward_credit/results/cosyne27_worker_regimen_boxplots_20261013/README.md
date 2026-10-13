# High policy-credit worker timing

[Compact Figure-2-style PDF](high_credit_worker_compact.pdf) · [SVG](high_credit_worker_compact.svg) · [PNG preview](high_credit_worker_compact.png) · [paired seed values](paired_seed_differences.csv) · [source mapping](sources.json) · [compact renderer](plot_high_credit_compact.py)

The selected compact figure is 7.5 × 2.5 inches. It follows the submitted Figure 2's Arial typography, C05 orange/C15 green palette, quartile boxes with individual seed dots, min–max whiskers, and explicit signed $\Delta$ axes. All four panels show **only** the high policy-credit regimen, with C05 and C15 on each horizontal axis. Each panel uses its own vertical unit and range.

## Compact figure caption

**Temporal rather than constant worker-hit credit under high policy credit.** The temporal DG encoder setting is fixed in both worker arms. The two arms share $\gamma=0.999$ and GAE $\lambda=0.995$; the figure therefore shows within-regimen worker timing differences. The change from the original $\gamma=0.99$, $\lambda=0.95$ cohort raises both parameters together, and that older cohort is not plotted here. Every colored dot is one temporal-minus-constant matched training-seed difference (seeds 8, 17, 41, 99, and 123; $n=5$ pairs per architecture). Boxes span the first to third quartiles, the line marks the median, and whiskers extend to the observed minimum and maximum. The gray line is zero; an upward difference means a larger measured value in the temporal worker arm. No significance test is encoded.

**A, DG information:** active-unit mean spatial information, in bits, from each model's own-policy 100,000-decision snapshot at 100M training frames. **B, Coverage AUC:** logged coverage area under the curve over 90–100M training frames. **C, Correct hits:** total training observations with a correct internally detected target hit over that window. These repeated observations are **not independent physical goal arrivals**. **D, Reliable edges:** the stored graph's reliable-edge count in the 100M own-policy snapshot; this describes detector-memory structure, not physical goal arrival. The own-policy snapshots follow different trajectories, and the four readouts measure distinct outcomes.

The [paired table](paired_seed_differences.csv) retains the five high-credit differences for each architecture, along with the earlier original-cohort rows. Only `policy_credit=high` enters this figure. All 40 high-credit seed/readout differences were recomputed from the [source files and hashes](sources.json). Regenerate with:

```bash
MPLCONFIGDIR=/tmp/cosyne_high_compact_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_high_credit_compact.py
```

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
