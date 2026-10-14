# Figure 1 v3 candidate: DG information, intrinsic reward, and fields

[SVG](04_architecture_v3_dg_reward_fields.svg) · [Print PDF](04_architecture_v3_dg_reward_fields.pdf) · [PNG preview](04_architecture_v3_dg_reward_fields.png) · [Source manifest](04_architecture_v3_dg_reward_fields_manifest.json)

This is a separate **8.5 × 1.9-inch** candidate, within the width of US letter paper. It retains the authored [architecture v3](../architecture_v3.svg) at its original 5.41-inch width and essentially its original 1.9-inch height. All new panels are to its right. The two training curves are each 0.90 inch wide, and the three fields are closely spaced. If inserted into the current 7.5-inch submission figure area, the complete candidate would be reduced to 88% of its native size. It has not replaced the submitted Figure 1 or changed the TeX source.

## Suggested caption

**Architecture, spatial information, intrinsic reward, and example DG response maps.** The architecture panel shows an egocentric 96 × 72 RGB observation processed by an ImageNet-pretrained ResNet-18 visual trunk fixed through layer 2. A trainable DG projection and sparsifier convert the visual representation into sparse events. The fixed CA3-like recurrent memory retains event history, and a trainable two-layer, 128-unit readout predicts values and chooses among five actions. The red temporal push strengthens eligible DG events as a function of time between events; the blue temporal pull shapes policy learning around transitions to DG events. The schematic is the original v3 artwork, including its internal symbols and feedback path.

**B, DG spatial information.** Amplitude-weighted online spatial information of the active DG units, in bits per step, from each policy's latest 10,000 observations during training. **C, intrinsic reward.** The learner-logged `intrmotiv/reward/intrinsic_mean`, measured *before PPO reward routing*, is the mean over 2,048 valid transitions per update, including transitions with zero intrinsic reward. Its vertical display axis is logarithmic; the curves and bands still use arithmetic values. Each course uses the same 15 verified runs, with temporal DG credit in all groups and temporal worker-hit credit in UG and FG: five seeds each for no goal (NG/C01), uniform goal (UG/C05), and frontier goal (FG/C15). The horizontal axis is training environment frames in millions. Lines show the five-seed mean at each 1-million-frame point from 1 to 99 million; translucent bands show plus or minus one standard error across seeds. Interpolation aligns separately logged updates and is not temporal smoothing. The intrinsic-reward scales differ across architectures, so relative curve height does not by itself measure navigation performance. All plotted means and SEM bounds are positive on the log axis.

**D, C15/FG place-field examples.** DG 4, 1, and 11 come from the **temporal-credit C15, seed-123** model at 100,007,936 training frames. Its independent, policy-driven terminal probe lasted 10,001 decisions and visited 307 of 361 spatial bins. The maximum-response cells are separated at approximately $(2,4)$, $(6,15)$, and $(17,3)$ in the displayed 19 × 19 grid, respectively. DG 4 satisfies the atlas's strict monofield criterion; DG 1 and 11 were selected for distinct peak locations and do **not** satisfy that strict criterion. Their spatial-information scores are 0.258, 0.168, and 0.122 bits, respectively, and their active fractions are 1.7%, 2.1%, and 2.0%. Maps show post-threshold occupancy-corrected responses. Each map uses its **own visited-bin min–max color range**; gray bins were unvisited, and colors cannot be compared as absolute response amplitudes across units. These are three selected examples from one seed, not a population estimate of field shape or a fixed-trajectory stability test.

## Regeneration and source checks

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/cosyne_figure1_v3_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python 07_abstracts/cosyne_2027_submission_draft/figure1_candidates/render_architecture_v3_course_candidate.py
```

The renderer checks complete five-seed, 99-point online courses, positivity of the reward mean and SEM bounds, seed-123 probe coverage, selected-unit eligibility, and the three atlas peak cells. The manifest pins SHA-256 hashes of the architecture, plotted summaries, min–max atlas, and unit and seed metric tables. The architecture and map tiles are high-resolution raster content within the PDF/SVG; plot lines, labels, axes, and bands remain vector elements.
