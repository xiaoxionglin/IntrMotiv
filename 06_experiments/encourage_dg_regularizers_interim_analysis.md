# Encourage DG regularizers: interim online and 50M telemetry

**Snapshot date:** 2026-08-26.  
**W&B project:** `SF_IntrMotiv_EncourageDGRegularizers`.  
**Batch:** `intrmotiv_encourage_dg_regularizers_20260825`.

## Scope

All 60 production jobs were running at the snapshot. The available terminal
frame counts ranged from 55.4M to 71.5M frames (median about 56M), so this is
an interim comparison only. Each value below is the mean over the last
`min(10M frames, 20% of observed frames)` of each run, then averaged over the
three seeds. Coverage is the fixed-length physical-episode telemetry.

The factorial design is:

| Factor | Values |
| --- | --- |
| Architecture | flat `FREG`; fixed/global HRL `GREG` |
| DG threshold | 2.20; 2.43 |
| Regularizer arm | `CTRL`; global punishment 0.01 (`G001`); global punishment 0.03 (`G003`); row repulsion 1.0 (`R100`); both 0.01 and 1.0 (`G001_R100`) |
| Seeds | 8, 99, 123 |

All conditions use `encourage`, the batch-usage term, the fixed pretrained
ResNet layer-2 trunk, and simultaneous encoder/decoder updates.

## Interim Findings

### DG activity has not globally collapsed

All arms retain nontrivial DG density (0.0185 to 0.0541) and high usage
entropy (0.94 to 0.99). The lowest threshold's controls have zero
minibatch-silent units in both architectures. This is substantially healthier
than the prior punishment/mean anti-collapse batch.

Global punishment is not uniformly protective. At threshold 2.43 it produces
minibatch silent fractions in several flat arms:

| Architecture | Threshold | Arm | DG density | Silent fraction | Usage entropy |
| --- | --- | --- | ---: | ---: | ---: |
| Flat | 2.43 | CTRL | 0.0206 | 0.0000 | 0.9802 |
| Flat | 2.43 | G001 | 0.0265 | 0.0625 | 0.9577 |
| Flat | 2.43 | G003 | 0.0206 | 0.0625 | 0.9548 |
| Flat | 2.43 | G001_R100 | 0.0185 | 0.0833 | 0.9402 |
| Global HRL | 2.43 | G001 | 0.0405 | 0.0426 | 0.9761 |
| Global HRL | 2.43 | G001_R100 | 0.0306 | 0.0000 | 0.9880 |

Thus the combined arm is presently the only threshold-2.43 global-HRL
regularized condition that both preserves nonzero coverage improvement and
avoids observed minibatch silence. This is a health-screen result, not yet a
place-field result.

### Current coverage signal

The table gives the matched-seed difference from `CTRL` at the same
architecture and threshold. It is still confounded by unequal training
progress between jobs and has only three seeds.

| Architecture | Threshold | Arm | Coverage AUC delta | Unique-cell delta | Interim reading |
| --- | --- | --- | ---: | ---: | --- |
| Flat | 2.20 | G001 | +12.7 +/- 31.7 | +20.1 +/- 53.9 | Directionally positive but too variable. |
| Flat | 2.20 | G003 | +1.6 +/- 13.0 | +6.3 +/- 28.5 | No reliable effect. |
| Flat | 2.20 | R100 | -2.2 +/- 5.4 | -2.6 +/- 6.2 | No benefit. |
| Flat | 2.43 | G001_R100 | +26.9 +/- 18.7 | +51.9 +/- 41.7 | Large but accompanied by 8.3% silent units. |
| Flat | 2.43 | G003 | +30.9 +/- 35.9 | +59.5 +/- 60.8 | Large variance and 6.3% silent units. |
| Global HRL | 2.20 | G001 | -7.5 +/- 10.3 | -13.7 +/- 18.0 | Negative early signal. |
| Global HRL | 2.20 | G003 | -11.7 +/- 11.6 | -23.1 +/- 21.9 | Negative early signal. |
| Global HRL | 2.20 | G001_R100 | +0.5 +/- 17.5 | 0.0 +/- 29.0 | No evidence of benefit yet. |
| Global HRL | 2.43 | G001_R100 | +9.8 +/- 7.2 | +16.6 +/- 11.4 | Best stable HRL signal so far; no minibatch silence. |
| Global HRL | 2.43 | G003 | +3.9 +/- 2.7 | +6.9 +/- 4.8 | Small positive signal, but some silence. |
| Global HRL | 2.43 | R100 | +1.4 +/- 10.8 | +1.8 +/- 20.0 | No clear benefit. |

The preliminary architecture-level comparison is compatible with global HRL
being competitive or better than flat at both thresholds, but it is not a
valid architecture winner yet because the runs have progressed unequally and
the effect of regularizers is not isolated from training time.

### HRL worker signal remains sparse

The graph does form confidence-qualified edges in every global-HRL arm:
known-edge fraction is about 0.14 to 0.17. This alone does not demonstrate
goal-directed control. Target-hit and option-success rates remain low and
variable.

The strongest interim worker funnel is threshold 2.20 `G001_R100`:

| Threshold | Arm | Hit rate | Timeout rate | Option success fraction | Known-edge fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| 2.20 | CTRL | 0.0038 | 0.1262 | 0.0296 | 0.1582 |
| 2.20 | G001_R100 | 0.0043 | 0.1006 | 0.0433 | 0.1534 |
| 2.43 | CTRL | 0.0032 | 0.1033 | 0.0306 | 0.1482 |
| 2.43 | G001_R100 | 0.0015 | 0.0987 | 0.0141 | 0.1652 |

This does **not** select the 2.20 combination as the best overall HRL
condition: it has no observed coverage advantage yet. It instead shows that
coverage and target-hit metrics are currently not aligned, which is a core
diagnostic question for the final analysis.

## Provisional Decision

Do not stop or alter the running batch from this snapshot.

At completion, prioritize these comparisons:

1. Global HRL, threshold 2.43: `CTRL` versus `G001_R100`, as the current
   best balanced coverage/representation candidate.
2. Global HRL, threshold 2.20: `CTRL` versus `G001_R100`, to test whether its
   higher option success converts into later exploration.
3. Flat threshold 2.43 global-punishment arms, as positive coverage candidates
   that may be invalidated by silent DG units or unstable place fields.
4. Row repulsion alone as a meaningful negative control; it has not shown an
   exploration benefit so far.

The final analysis must align all runs at a shared frame checkpoint, use the
10k-decision place-field evaluation, and compare matched seeds. Until then,
the interim coverage differences are directional evidence only.

## Artifacts

The reusable evaluator snapshot is stored in the NEMO workspace:

```text
/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/
  encourage_dg_regularizers_interim_20260826/
```

It contains `per_run_terminal.csv`, family/condition summaries, a manifest,
and the generic diagnostic report. Its automatic condition parser predates
the `FREG`/`GREG` names, so the architecture/arm grouping in this note was
computed directly from the per-run data.

## Matched 50M candidate telemetry — 26 August 2026

The online snapshot above spans unequal training ages. The matched
50M checkpoint telemetry below assesses a six-condition subset with
10,001-decision rollouts; it does not replace the wider factorial or establish
a finished 100M outcome. The source note was
`encourage_dg_regularizers_candidate_telemetry_50m.md`.

**Date:** 2026-08-26. **Batch:** SF_IntrMotiv_EncourageDGRegularizers.
**Telemetry array:** 7868970, 18 of 18 completed with exit code 0.

### Protocol

This checkpoint evaluation did not modify training. Each task loaded the nearest retained milestone to 50M frames (actual range 48.3M to 52.2M) and performed one stochastic 10,001-decision DMLab rollout.

The selected conditions were global fixed HRL at thresholds 2.20 and 2.43, each with CTRL and global punishment 0.01 plus row repulsion 1.0 (G001_R100), plus the corresponding flat threshold-2.43 control/candidate pair. Every condition has seeds 8, 99, and 123.

Telemetry records a 19 x 19 occupancy grid, occupancy-corrected current DG rate maps, pre-threshold DG-logit maps, spatial information, active fraction, map cosine redundancy, and peak-bin diversity. These are representation-health probes. Since policy rollouts are stochastic and do not follow an identical coverage trajectory, they do not establish a causal behavioral comparison.

### Aggregate Results

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

### Interpretation

#### Global HRL T2.43

This is the strongest current representation candidate, but it is not replicated. Relative to its control, G001_R100 has higher mean spatial information (0.520 versus 0.337 bits), higher active fraction (0.042 versus 0.032), zero silent units, lower map redundancy (0.734 versus 0.818), and more peak-bin diversity (4.0 versus 2.0). The two groups visit essentially the same number of cells, so this is not explained by broader occupancy.

The mean is dominated by seed 123. Combined minus control spatial-information changes are:

| Seed | Difference |
| --- | ---: |
| 8 | -0.069 bits |
| 99 | -0.234 bits |
| 123 | +0.851 bits |

Seed 123 is a genuine positive example: mean SI 0.977 bits, map cosine 0.647, and four peak bins. Seeds 8 and 99 do not improve. Keep the condition as the main candidate, but do not yet claim the regularizer reliably creates stable place fields.

#### Global HRL T2.20

The lower-threshold combined condition is weaker at 50M: mean SI falls from 0.252 to 0.161 bits, active fraction falls, and peak diversity does not improve. This conflicts with its better online option-success metrics, showing that its option-event signal has not translated into stronger DG spatial selectivity.

#### Flat T2.43

The flat combined arm visits more cells and reduces map redundancy, but has no SI improvement (0.196 versus 0.214 bits) and averages 1.3 silent DG units. It is not the primary follow-up candidate.

### Artifacts

All outputs are in the NEMO workspace:

    /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/
      encourage_dg_regularizers_candidates_50m_20260826/

The directory contains the common-checkpoint manifest, raw pose/map arrays, CSV and Markdown summaries, the comparison figure, per-run DG rate-map grids, and pre-threshold-logit grids.

### Decision

Continue the batch unchanged. At 100M, repeat this exact common-checkpoint telemetry for the three matched control/candidate pairs. Add a fixed scripted coverage trajectory, or repeated rollouts per checkpoint, so a future analysis can separate trajectory differences from representation differences.
