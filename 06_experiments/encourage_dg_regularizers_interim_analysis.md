# Encourage DG Regularizers: Interim Analysis

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
