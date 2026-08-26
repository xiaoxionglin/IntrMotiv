# Candidate Telemetry at 50M Frames

**Date:** 2026-08-26. **Batch:** SF_IntrMotiv_EncourageDGRegularizers.
**Telemetry array:** 7868970, 18 of 18 completed with exit code 0.

## Protocol

This checkpoint evaluation did not modify training. Each task loaded the nearest retained milestone to 50M frames (actual range 48.3M to 52.2M) and performed one stochastic 10,001-decision DMLab rollout.

The selected conditions were global fixed HRL at thresholds 2.20 and 2.43, each with CTRL and global punishment 0.01 plus row repulsion 1.0 (G001_R100), plus the corresponding flat threshold-2.43 control/candidate pair. Every condition has seeds 8, 99, and 123.

Telemetry records a 19 x 19 occupancy grid, occupancy-corrected current DG rate maps, pre-threshold DG-logit maps, spatial information, active fraction, map cosine redundancy, and peak-bin diversity. These are representation-health probes. Since policy rollouts are stochastic and do not follow an identical coverage trajectory, they do not establish a causal behavioral comparison.

## Aggregate Results

Values are mean +/- sample standard deviation across three seeds.

| Condition | Visited cells | DG SI bits | Active fraction | Silent units | Map cosine | Unique peak bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat T2.43 CTRL | 230.0 +/- 64.2 | 0.214 +/- 0.225 | 0.0179 +/- 0.0187 | 0.0 | 0.807 +/- 0.174 | 2.7 +/- 1.5 |
| Flat T2.43 G001_R100 | 308.7 +/- 12.9 | 0.196 +/- 0.070 | 0.0145 +/- 0.0069 | 1.3 +/- 1.5 | 0.626 +/- 0.117 | 3.3 +/- 0.6 |
| Global HRL T2.20 CTRL | 301.3 +/- 11.4 | 0.252 +/- 0.245 | 0.0543 +/- 0.0623 | 0.0 | 0.833 +/- 0.289 | 3.0 +/- 2.6 |
| Global HRL T2.20 G001_R100 | 308.0 +/- 6.1 | 0.161 +/- 0.094 | 0.0345 +/- 0.0269 | 0.0 | 0.809 +/- 0.230 | 2.7 +/- 2.1 |
| Global HRL T2.43 CTRL | 296.7 +/- 4.0 | 0.337 +/- 0.210 | 0.0324 +/- 0.0148 | 0.0 | 0.818 +/- 0.228 | 2.0 +/- 1.7 |
| Global HRL T2.43 G001_R100 | 295.3 +/- 16.4 | 0.520 +/- 0.437 | 0.0421 +/- 0.0217 | 0.0 | 0.734 +/- 0.235 | 4.0 +/- 3.0 |

Lower map cosine and more unique peak bins indicate less redundant DG maps. Neither one is sufficient alone, because a near-silent representation can also have low cosine.

## Interpretation

### Global HRL T2.43

This is the strongest current representation candidate, but it is not replicated. Relative to its control, G001_R100 has higher mean spatial information (0.520 versus 0.337 bits), higher active fraction (0.042 versus 0.032), zero silent units, lower map redundancy (0.734 versus 0.818), and more peak-bin diversity (4.0 versus 2.0). The two groups visit essentially the same number of cells, so this is not explained by broader occupancy.

The mean is dominated by seed 123. Combined minus control spatial-information changes are:

| Seed | Difference |
| --- | ---: |
| 8 | -0.069 bits |
| 99 | -0.234 bits |
| 123 | +0.851 bits |

Seed 123 is a genuine positive example: mean SI 0.977 bits, map cosine 0.647, and four peak bins. Seeds 8 and 99 do not improve. Keep the condition as the main candidate, but do not yet claim the regularizer reliably creates stable place fields.

### Global HRL T2.20

The lower-threshold combined condition is weaker at 50M: mean SI falls from 0.252 to 0.161 bits, active fraction falls, and peak diversity does not improve. This conflicts with its better online option-success metrics, showing that its option-event signal has not translated into stronger DG spatial selectivity.

### Flat T2.43

The flat combined arm visits more cells and reduces map redundancy, but has no SI improvement (0.196 versus 0.214 bits) and averages 1.3 silent DG units. It is not the primary follow-up candidate.

## Artifacts

All outputs are in the NEMO workspace:

    /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/
      encourage_dg_regularizers_candidates_50m_20260826/

The directory contains the common-checkpoint manifest, raw pose/map arrays, CSV and Markdown summaries, the comparison figure, per-run DG rate-map grids, and pre-threshold-logit grids.

## Decision

Continue the batch unchanged. At 100M, repeat this exact common-checkpoint telemetry for the three matched control/candidate pairs. Add a fixed scripted coverage trajectory, or repeated rollouts per checkpoint, so a future analysis can separate trajectory differences from representation differences.

