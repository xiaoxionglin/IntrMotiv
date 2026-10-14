# Standalone Figure 2 trajectory candidates

These assets are **not** inserted into Figure 2 or either Overleaf ZIP. NG, UG, and FG mean no-goal, uniform-goal, and frontier-goal architecture. Every trajectory uses the original selected run and its 100M-frame checkpoint. The [current Figure 2 story](../../Figure2_story_README.md) embeds NG and FG paths from 100,000-observation online-spatial windows but the UG path from a separate 10,001-observation frozen-policy probe. For this density comparison, the UG candidate uses its **same-run, same-checkpoint online-spatial window** so all three panels have the same source protocol. The original frozen UG tile remains in the current figure, unchanged.

| Candidate | Three-run comparison | Small interchangeable tiles |
| --- | --- | --- |
| 9 fragments | [PDF](09_fragments.pdf) · [SVG](09_fragments.svg) · [PNG](09_fragments.png) | [NG](NG_09_fragments_tile.svg) · [UG](UG_09_fragments_tile.svg) · [FG](FG_09_fragments_tile.svg) |
| 33 fragments | [PDF](33_fragments.pdf) · [SVG](33_fragments.svg) · [PNG](33_fragments.png) | [NG](NG_33_fragments_tile.svg) · [UG](UG_33_fragments_tile.svg) · [FG](FG_33_fragments_tile.svg) |
| 65 fragments | [PDF](65_fragments.pdf) · [SVG](65_fragments.svg) · [PNG](65_fragments.png) | [NG](NG_65_fragments_tile.svg) · [UG](UG_65_fragments_tile.svg) · [FG](FG_65_fragments_tile.svg) |
| Verified-hit **separate archive** | [PDF](archived_verified_goal_hits.pdf) · [SVG](archived_verified_goal_hits.svg) · [PNG](archived_verified_goal_hits.png) | [C05](archived_hits_C05_seed99_tile.svg) · [C15](archived_hits_C15_seed99_tile.svg) |

Each tile is also available as a PDF and PNG with the same basename. The three-run sheets retain x/y ticks; the 2.2-inch tiles omit text so they can be inserted beside a manuscript caption without shrinking labels. All assets are vector except their PNG previews. Their x/y limits are identically **100–2000 DMLab units**; no zoom or crop changes the map scale.

## Exact fragment selection and interpretation

The source is the pinned 100,000-observation online-spatial snapshot from frames 99,207,936–100,007,936 for NG `COS27_C01_TEMPORAL_S99`, UG `COS27_C05_TEMPORAL_TEMPORAL_S123`, and FG `COS27_C15_TEMPORAL_S17`. This is an online evaluation window, not the full training trajectory or the separate 10k frozen probe. NG, UG, and FG have 1,675, 1,689, and 1,679 original contiguous fragments. A change in `segment_id` or the preceding `done` closes a fragment, matching the canonical renderer; fragments of one observation are ineligible. Most eligible fragments contain 64 observations.

For each architecture, the renderer makes a uniform random permutation of eligible fragment indices without replacement, using NumPy seeds **2027, 2028, and 2029** for NG, UG, and FG. The first 9, 33, and 65 draws form nested versions. There is no choice based on map coverage, path shape, goal hits, or figure appearance. Every chosen fragment is plotted in full, with no joins across fragments. Color identifies architecture; it does not encode time or success. Blank areas in a candidate mean that the **sampled fragments** do not pass through them; they are not evidence that the complete 100k window left those areas unvisited. The same number of fragments does not always mean the exact same number of observations:

| Architecture | 9 fragments | 33 fragments | 65 fragments |
| --- | ---: | ---: | ---: |
| NG, seed 99 | 576 observations | 2,044 | 3,864 |
| UG, seed 123 | 544 | 2,080 | 4,000 |
| FG, seed 17 | 516 | 2,016 | 3,808 |

Thus these are **legibility trials**, not new estimates of path efficiency or coverage. The [manifest](manifest.json) gives each original snapshot SHA-256, every chosen source-fragment ordinal and starting row, exact observation totals, the shared bounds, font, software versions, and output hashes. The lightweight [selected-fragment archive](selected_fragments.npz) contains only the chosen x/y paths and offsets, not the complete source NPZs. [Editable renderer](render_trajectory_candidates.py) regenerates every SVG/PDF/PNG locally with:

```bash
MPLCONFIGDIR=/tmp/cosyne_trajectory_mpl /home/xiaoxiong/miniforge3/envs/SF_git/bin/python render_trajectory_candidates.py
```

The preparation mode `--prepare-from DIRECTORY` recreates the lightweight archives after the three exact source NPZs have been staged as `NG/online_spatial.npz`, `UG/online_spatial.npz`, and `FG/online_spatial.npz`, and the two older verified pose-event CSVs as `c05_seed99_pose_events.csv` and `c15_seed99_pose_events.csv`. It checks every full-source hash before extracting coordinates. The full snapshots and pose-event streams remain in the allocated NEMO2 workspace. This checked-in archive and renderer make subsequent layout trials possible without another remote read.

## Verified goal-hit example and its limit

The current selected online-spatial NPZs have no observation-aligned goal-hit flag, and the selected UG frozen pose has no such flag either. A goal ID or stored candidate target is **not** a verified hit, so the 9/33/65-fragment candidates carry no hit markers. The separate hit example instead uses the previously captured **2026-09-28 C05 and C15 seed-99 frozen probes**, which are *different runs* from the selected Figure 2 paths. It plots the same episode index (`num_traj=1`, frames 900–1799, 900 observations) in both. Open circles mark option starts **without** a hit; magenta stars mark the observation-aligned `HRLStateLayout.target_hit` flag, with 44 C05 and 1 C15 hits. These are hits of the currently commanded option target, not certified physical arrival at a final external goal. [Archived event summaries](../../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/report.md) document the hook and full 10k probes. The [lightweight aligned-episode archive](verified_hit_episode_inputs.npz) preserves only the plotted x/y positions and event flags; its source CSV hashes are in the manifest.
