# Worker timing across policy-credit settings

[C05 PDF](c05_worker_regimen_boxplots.pdf) · [C15 PDF](c15_worker_regimen_boxplots.pdf) · [combined overview](worker_regimen_boxplots.pdf) · [paired seed values](paired_seed_differences.csv) · [source mapping](sources.json) · [plot script](plot_worker_regimen_boxplots.py)

Each figure also has an editable SVG and a 300-dpi PNG with the same stem. The family-specific figures give the hit-observation and graph-edge panels enough vertical resolution; the combined overview uses one scale per metric and shows the full four-group comparison.

## Figure caption

**Seed-paired effects of temporal rather than constant worker-hit credit in C05 and C15.** Each point is the temporal-minus-constant difference for one matched training seed (8, 17, 41, 99, or 123); marker shape identifies the seed throughout. Both worker-bonus arms use temporal DG encoder credit. The original policy-credit setting is $\gamma=0.99$, GAE $\lambda=0.95$; the high-credit setting is $\gamma=0.999$, GAE $\lambda=0.995$. **Both policy parameters change together.** Boxes show the 25th–75th percentiles, black bars the median, whiskers the observed minimum and maximum, and orange crosses the mean. Every group has five training-seed pairs. Horizontal lines mark zero; no hypothesis test or significance marker is shown.

**A,** Active-unit mean DG spatial information in bits from each network's own-policy, 100,000-decision snapshot at the 100M-frame checkpoint. **B,** Logged coverage area under the curve over training frames 90–100M. The positive mean for original C15 is driven by seed 123 ($+41.1343$ AUC units); its other four seed differences are negative. **C,** Total logged observations with correct target hits during training frames 90–100M, summed within each seed. These are repeated training telemetry observations, **not independent physical goal arrivals**. **D,** Reliable stored graph-edge counts from the 100M own-policy snapshot, which describe detector-memory graph structure rather than physical goal arrival. The different readouts have separate vertical scales.

The original C15 constant-bonus runs used a recovery learner patch absent from the original temporal-bonus runs. The original and high-credit settings are separate training cohorts; their differences cannot isolate the effect of $\gamma$, $\lambda$, or that source patch. Own-policy spatial snapshots also follow different trajectories. The graph, hit, and coverage measures address distinct outcomes and should not be interpreted as repeated estimates of one quantity.

## Data and regeneration

The [paired table](paired_seed_differences.csv) contains the 20 plotted rows and identifies the relevant source bundle for each row. [Source mapping](sources.json) lists the per-metric source files and locally available SHA-256 hashes. All 40 high-credit values and 15 original-C05 spatial/hit/graph values were recomputed from those files. Original-C05 AUC and all original-C15 differences retain the rounded seed values supplied with the source task; the automatic approval review blocked exporting the private NEMO2 summary CSVs for independent reloading in this worktree. Original-C05 spatial differences were replaced with exact locally recomputed values. The source task had already calculated the remote differences using the listed canonical summaries and cached NPZs.

Rebuild the vector figure and 300-dpi preview with:

```bash
MPLCONFIGDIR=/tmp/cosyne_regimen_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python plot_worker_regimen_boxplots.py
```

The plot uses the paired CSV directly and introduces no smoothing, normalization, or significance calculation. The family figures are designed for 7.2-inch width and the overview for 8.4-inch width; all labels are at least 12 pt at their intended size.
