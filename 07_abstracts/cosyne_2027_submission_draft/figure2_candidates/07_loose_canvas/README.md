# Figure 2 loose canvas

[Main canvas: boxes and illustrations](figure2_loose_canvas.svg) · [PDF](figure2_loose_canvas.pdf) · [preview](figure2_loose_canvas.png)

[Scatter canvas: raw pairs and peak locations](figure2_scatter_canvas.svg) · [PDF](figure2_scatter_canvas.pdf) · [preview](figure2_scatter_canvas.png)

This folder starts from [candidate 06](../06_compact_vertical_boxes.pdf) and exposes its elements separately for manual layout. The **main canvas** contains the box plots, online spatial-emergence course, archived trajectory and place-field tiles, and movable headings and color key. **All scatter plots**—the raw control-versus-tested pairs and the two field-peak location maps—are on the separate scatter canvas. Both canvases group components by spatial emergence, frozen behavior, DG encoder **push** ablation, worker-credit **pull** ablation, longer PPO/GAE credit, or spatial examples as applicable. Each component is an independent movable SVG group; the boards are inventories, not proposed final figure layouts. Every element also has its own SVG, PDF, and PNG under [`components/`](components/). The original compact candidate, submitted `figures/results_sections.*`, `main.tex`, and Overleaf package remain unchanged.

All **26 paired-difference box plots** have the same **0.68 × 0.774 inch data axes**, matching the smallest box axes in candidate 06. Their surrounding component cards are larger so 12 pt titles, ticks, and axis labels remain legible while you arrange them. The ten statistical topics from candidate 06 are retained; a `+` on a card marks one of 16 extra candidate statistical groups. **Each box has a separate raw-pair scatter plot** with the same filename plus `_raw_pairs` on the scatter canvas, so every group has a direct view of its measured control and tested values. Both boards and all components can be edited independently.

In the **push ablation** components, the displayed groups are **C01, C05, C15**. Here C05 means the **C05-wT arm**, where the temporal worker bonus is held fixed while DG encoder credit changes. The C05-wC arm is omitted from these two canvases at your request. Its archived measurements and [candidate 06](../06_compact_vertical_boxes.pdf) remain intact; this is a display selection, not a claim that the arm was absent or that all C05 runs were pooled.

The board also includes [separate A](components/label_a.svg), [B](components/label_b.svg), [C](components/label_c.svg), [D](components/label_d.svg), [E](components/label_e.svg), and [F](components/label_f.svg) headings plus a [color key](components/label_color_key.svg), so those can move independently too.

## Elements from candidate 06

| Section | Statistical panels | Companion evidence |
| --- | --- | --- |
| A. Emergence | [Online DG course](components/A01_spatial_emergence.svg) | Five-seed mean and $\pm1$ s.e.m. |
| B. Frozen behavior | [Readout MLP layer 1 information](components/B01_readout_policy_information.svg) · [map reliability](components/B02_policy_map_reliability.svg) | [Information raw pairs](components/B01_readout_policy_information_raw_pairs.svg) · [reliability raw pairs](components/B02_policy_map_reliability_raw_pairs.svg) |
| C. Spatial examples | — | [Archived trajectory](components/C01_archived_trajectory.svg) · [DG 11](components/C02_archived_dg11.svg) · [DG 0](components/C03_archived_dg0.svg) |
| D. Temporal DG credit | [DG bits/step](components/D01_dg_information.svg) · [active-map overlap](components/D02_map_overlap.svg) · [field-peak coverage](components/D03_peak_area.svg) | [DG raw pairs](components/D01_dg_information_raw_pairs.svg) · [overlap raw pairs](components/D02_map_overlap_raw_pairs.svg) · [peak-area raw pairs](components/D03_peak_area_raw_pairs.svg) · [temporal peak locations](components/C04_peaks_temporal.svg) · [constant peak locations](components/C04_peaks_constant.svg) |
| E. Downstream readout | [Readout MLP layer 1 bits/activation](components/E01_readout1_information.svg) | [Raw pairs](components/E01_readout1_information_raw_pairs.svg) |
| F. Worker and PPO credit | [Worker online coverage](components/F01_worker_coverage.svg) · [worker online DG score](components/F02_worker_online_dg.svg) · [longer PPO coverage](components/F03_longer_ppo_coverage.svg) · [longer PPO frozen DG score](components/F04_longer_ppo_dg.svg) | [Coverage raw pairs](components/F01_worker_coverage_raw_pairs.svg) · [online DG raw pairs](components/F02_worker_online_dg_raw_pairs.svg) · [PPO coverage raw pairs](components/F03_longer_ppo_coverage_raw_pairs.svg) · [PPO DG raw pairs](components/F04_longer_ppo_dg_raw_pairs.svg) |

The raw-pair views show the measured control value on $x$, tested value on $y$, and an equality line. One mark is one paired training seed or frozen model, matching its companion box. Blue, orange, green, and purple denote C01, C05, C15, and prescribed-goal agents. The constant-encoder C05-eC worker-pull arm uses diamonds in the raw-pair views; the other displayed groups use circles. The B views contain 38 frozen models (10 C01, 20 C05, eight C15). D and E contain five paired seeds per displayed group. Worker-credit F plots contain five paired seeds per fixed-encoder group; longer-PPO F plots contain three paired seeds per family.

The peak-location pair is the direct example for **field-peak coverage**. It uses the same C01 seed-8, 10,001-observation common replay and 0.25% unit-activity cutoff as candidate 05. Colored dots are the eligible DG units' field peaks; grayscale shows their Gaussian-smoothed equal-mass density on shared visited bins, with a common density scale between the two maps. Gray outside the sampled arena is unvisited. The effective-area fractions are 0.384 (temporal) and 0.230 (constant). This seed was chosen because its positive peak-area difference is the largest among five C01 pairs, so it illustrates an upper-end effect, not a typical seed. The archived trajectory and two response maps in C come from a **different C01 seed-8 run** and are not matched examples for D.

## Extra candidate statistical panels

These use the same paired-difference box size. They were not in candidate 06 because of space. Direction, negative effects, and mixed seeds are retained. Each has a matching raw-pair SVG/PDF/PNG in `components/`; for example, [CA3 raw pairs](components/E02_ca3_information_raw_pairs.svg) and [sampled-path raw pairs](components/F08_path_efficiency_raw_pairs.svg).

| Area | Additional plots |
| --- | --- |
| DG activity and overlap | [Mean active fraction](components/D04_dg_activity.svg) · [prethreshold map overlap](components/D05_prethreshold_overlap.svg) |
| Peak distribution | [Peak-density Gini](components/D06_peak_gini.svg) · [mean pairwise peak distance](components/D07_peak_separation.svg) |
| Other network layers | [CA3 bits/activation](components/E02_ca3_information.svg) · [readout MLP layer 2 bits/activation](components/E03_readout2_information.svg) · [CA3 weighted bits/step](components/E04_ca3_weighted.svg) · [readout MLP layer 1 weighted bits/step](components/E05_readout1_weighted.svg) · [readout MLP layer 2 weighted bits/step](components/E06_readout2_weighted.svg) |
| Worker credit on common replay and control diagnostics | [DG score on common replay](components/F05_worker_replay_dg.svg) · [stored-graph reachability](components/F06_graph_reachability.svg) · [stored-graph prospective success](components/F07_graph_prospects.svg) · [sampled-path efficiency](components/F08_path_efficiency.svg) · [readout MLP layer 1 weighted bits/step](components/F10_worker_readout1_weighted.svg) · [readout MLP layer 2 weighted bits/step](components/F11_worker_readout2_weighted.svg) |
| Joint longer PPO/GAE credit | [Internally detected hits per 1,000 decisions](components/F09_longer_ppo_hits.svg) |

## Reading the statistics

In every box, the vertical axis explicitly shows $\Delta=\text{tested}-\text{control}$. The box spans the first and third quartiles, its center line is the median, whiskers reach the observed minimum and maximum, and all paired differences are shown as dots. The horizontal line is zero. These are descriptive summaries, not confidence intervals. Each metric has a separate symmetric vertical scale; color and distance should be compared **within** a panel, not used to compare effect sizes across panels. A positive value is not automatically favorable: more map overlap or larger peak-density Gini, for example, mean greater redundancy or concentration.

The definition of the pair depends on the test:

- **B:** Each frozen network executes its own learned policy or random actions, with weights fixed. $\Delta=\text{own policy}-\text{random actions}$. Two 50,000-decision evaluations are averaged within each model. The intervention changes the sampled trajectory.
- **D/E push:** $\Delta=\text{temporal DG encoder credit}-\text{mean-matched constant DG credit}$. The displayed C01, C05, and C15 groups each have five matched training seeds. C05 is specifically **C05-wT**, the fixed temporal worker-hit bonus arm; C05-wC is not displayed here. Separate terminal networks replay the **same 10,001 observations** at approximately 100 million training frames. DG maps and CA3/readout activity come from that common replay. Field-eligible information per activation and all-channel amplitude-weighted bits per step have different denominators; they need not agree.
- **F worker:** $\Delta=\text{temporal worker-hit credit}-\text{mean-matched constant worker-hit credit}$ in C05, with encoder credit fixed. eT/eC name that **fixed encoder setting**. The online panels average 90–99 million training frames; the common-replay, stored-graph, and sampled-trajectory panels have separate protocols. Stored-graph prospects and sampled-path efficiency do not establish executed physical goal arrival.
- **F PPO:** A separate episode-long prescribed-goal/C15 study changes PPO discount $\gamma$ from 0.99 to 0.999 **and** GAE $\lambda$ from 0.95 to 0.99 together. $\Delta=\text{joint longer credit}-\text{baseline}$. Online coverage AUC and detector-hit rate use policy-sampled experience. Frozen learned-unit DG scores come from separate 10,000-decision policy-driven paths, so the DG difference does not isolate a fixed-input network effect. Internally detected hits do not establish physical goal arrival.

## Rebuild and source record

[render_canvas.py](render_canvas.py) loads the same archived tables and source graphics as the candidate gallery, validates group sizes and paired arithmetic, and produces the individual assets and both boards. [manifest.json](manifest.json) maps every component to its canvas group, protocol, displayed architecture groups, source keys, SHA-256 hashes, and mean paired differences. It also records the exact source-file hashes and board membership. No training or new evaluator run was used.

From the repository root:

```bash
MPLCONFIGDIR=/tmp/cosyne_figure2_canvas_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python 07_abstracts/cosyne_2027_submission_draft/figure2_candidates/07_loose_canvas/render_canvas.py
```
