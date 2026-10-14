# Standalone Figure 2 trajectory candidates

These assets remain separate from Figure 2 and both Overleaf ZIPs. NG, UG, and FG mean no-goal, uniform-goal, and frontier-goal architecture. The three candidate paths come from comparable 100,000-observation online-spatial windows near the 100M-frame checkpoint. The [current Figure 2 story](../../Figure2_story_README.md) instead uses an independent 10,001-observation frozen-policy UG path and FG seed 17; its composite is unchanged. This candidate uses the same NG seed 99 and UG seed 123 runs as before, and **FG temporal-credit seed 8**, selected by full-window occupancy uniformity among five comparable FG seeds.

| Candidate | Three-run comparison | Interchangeable 38 pt square panels |
| --- | --- | --- |
| 65 complete fragments per architecture | [PDF](65_fragments.pdf) · [SVG](65_fragments.svg) · [PNG](65_fragments.png) | [NG](NG_65_fragments_tile.svg) · [UG](UG_65_fragments_tile.svg) · [FG](FG_65_fragments_tile.svg) |
| Verified-hit **separate archive** | [PDF](archived_verified_goal_hits.pdf) · [SVG](archived_verified_goal_hits.svg) · [PNG](archived_verified_goal_hits.png) | [C05](archived_hits_C05_seed99_tile.svg) · [C15](archived_hits_C15_seed99_tile.svg) |

Each trajectory tile is also available as a PDF and PNG with the same basename. **Every trajectory panel is exactly 38 × 38 pt**, both in the three-run sheet and in its individual tile file. This matches `trajectory_tile_size=38` in [the current Figure 2 icon-variant builder](../../build_figure2_story_icon_variant.py); the latest checked `main` commit, `a49c677`, has the same setting. Sheet labels and the shared colorbar sit outside those squares. Individual tiles contain only the 38 pt path panel; use the sheet's shared colorbar when assembling a composite. The sheet is 270 × 140 pt. PDF and SVG retain vector paths and text; PNG is a 300-dpi preview. All panels use the same square **100–2000 DMLab-unit** x/y world bounds and equal aspect, with no zoom or crop.

The path uses a **Viridis sequential gradient along every observed line segment**. `step` is the original observation/decision row index **within each run's 100,000-observation source window**, from 0 to 99,999. A segment joining two consecutive observations receives the midpoint of their row indices. The normalization and colorbar range are identical across NG, UG, and FG. The random order of selected fragments does not determine color; no colored line joins distinct fragments. Color does not encode absolute training frames, success, or path efficiency.

## Exact fragment selection and interpretation

The source is the pinned 100,000-observation online-spatial snapshot from frames 99,207,936–100,007,936 for NG `COS27_C01_TEMPORAL_S99`, UG `COS27_C05_TEMPORAL_TEMPORAL_S123`, and FG `COS27_C15_TEMPORAL_S8`. This is an online evaluation window, not the full training trajectory or the separate 10k frozen probe. A change in `segment_id` or the preceding `done` closes a fragment, matching the canonical renderer; fragments of one observation are ineligible. Most eligible fragments contain 64 observations.

For each architecture, the renderer makes a uniform random permutation of eligible fragment indices without replacement, using NumPy seeds **2027, 2028, and 2029** for NG, UG, and FG. It takes the first 65 eligible fragments in each permutation. There is no fragment choice based on map coverage, path shape, goal hits, or figure appearance. Every chosen fragment is plotted in full, with no joins across fragments. Blank areas mean only that the **sampled fragments** do not pass through them; they are not evidence that the complete 100k window left those areas unvisited. Equal fragment counts do not imply equal observation counts:

| Architecture | Selected fragments | Plotted observations |
| --- | ---: | ---: |
| NG, seed 99 | 65 | 3,864 |
| UG, seed 123 | 65 | 4,000 |
| FG, seed 8 | 65 | 3,752 |

These are **legibility trials**, not new estimates of path efficiency or coverage. The [manifest](manifest.json) gives each original snapshot SHA-256, every chosen source-fragment ordinal and starting row, exact observation totals, seed-screen scores, the shared bounds, color normalization, panel geometry, font, software versions, and output hashes. The lightweight [selected-fragment archive](selected_fragments.npz) contains the chosen x/y paths, original-row starts and offsets, run identities and source hashes, and seed-screen scores, not the complete source NPZs. The renderer checks those cached identities and hashes before plotting, preventing a later seed edit from relabeling stale coordinates. [Editable renderer](render_trajectory_candidates.py) regenerates every SVG/PDF/PNG locally with:

```bash
MPLCONFIGDIR=/tmp/cosyne_trajectory_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_trajectory_candidates.py
```

The preparation mode `--prepare-from DIRECTORY` recreates the lightweight archives after staging the exact NG and UG NPZs as `NG/online_spatial.npz` and `UG/online_spatial.npz`, all five C15 temporal-credit snapshots as `C15_candidates/S8.npz`, `S17.npz`, `S41.npz`, `S99.npz`, and `S123.npz`, and the two older verified pose-event CSVs as `c05_seed99_pose_events.csv` and `c15_seed99_pose_events.csv`. It checks every full-source hash and contract before extracting coordinates and scores. The full snapshots and pose-event streams remain in the allocated NEMO2 workspace. The checked-in lightweight archive permits further layout trials without another remote read.

## FG seed selection

All five comparable C15 temporal DG-credit/temporal worker-credit runs have the same 100M-frame target and 100,000-observation protocol. The screen bins every full-window x/y observation into the fixed 19 × 19 world grid. For visited bins with occupancy proportions $p_i$, effective bins are $N_{\mathrm{eff}}=\exp(-\sum_i p_i\log p_i)$; uniformity is $N_{\mathrm{eff}}$ divided by the number of visited bins. This distinguishes nearly saturated *visited-bin fraction* from even use of those bins. It is descriptive occupancy, not a behavioral outcome score. The same calculation on the 65 sampled fragments shows what the panel actually displays.

| FG seed | Full-window visited bins | Full-window effective bins | Uniformity | Sampled visited bins | Sampled effective bins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **8, selected candidate** | **318** | **136.37** | **0.429** | **249** | **101.54** |
| 17, prior Figure 2A example | 315 | 18.10 | 0.057 | 110 | 15.54 |
| 41 | 317 | 29.54 | 0.093 | 174 | 25.14 |
| 99 | 318 | 126.59 | 0.398 | 257 | 95.46 |
| 123 | 318 | 39.58 | 0.124 | 211 | 32.18 |

**Selection criterion:** maximize full-window effective bins divided by visited bins across these five FG seeds, then visually inspect the 65-fragment candidate at the common bounds and final 38 pt size. Seed 8 ranks first (0.429), ahead of seed 99 (0.398), and its sampled fragments visit 249 bins. At final size, the seed-8 paths remain visible across multiple arena regions. The former seed 17 scores 0.057 and its sampled fragments visit 110 bins. Path efficiency and other Figure 2A outcomes were not used to select the seed. Exact source paths and full SHA-256 hashes for **all five** seeds are in the manifest.

## Verified goal-hit example and its limit

The current selected online-spatial NPZs have no observation-aligned goal-hit flag, and the selected UG frozen pose has no such flag either. A goal ID or stored candidate target is **not** a verified hit, so the 65-fragment candidate carries no hit markers. The separate hit example instead uses the previously captured **2026-09-28 C05 and C15 seed-99 frozen probes**, which are *different runs* from the selected Figure 2 paths. It plots the same episode index (`num_traj=1`, frames 900–1799, 900 observations) in both. Open circles mark option starts **without** a hit; magenta stars mark the observation-aligned `HRLStateLayout.target_hit` flag, with 44 C05 and 1 C15 hits. These are hits of the currently commanded option target, not certified physical arrival at a final external goal. [Archived event summaries](../../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/report.md) document the hook and full 10k probes. The [lightweight aligned-episode archive](verified_hit_episode_inputs.npz) preserves only the plotted x/y positions and event flags; its source CSV hashes are in the manifest.
