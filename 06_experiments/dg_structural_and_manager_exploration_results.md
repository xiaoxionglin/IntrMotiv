# Structural Diversity And Manager Exploration Results

**Analysis date:** 2026-08-27  
**Run horizon:** 100M environment frames per run  
**Replicates:** seeds 8, 99, and 123 in every condition

## Analysis Status

| Batch | W&B project | Production runs | Status |
| --- | --- | ---: | --- |
| DG structural diversity | `SF_IntrMotiv_DGStructuralDiversity` | 48/48 | Complete and analyzed |
| HRL manager exploration | `SF_IntrMotiv_HRLManagerExploration` | 24/24 | Complete and analyzed |

All 72 runs finished at approximately 100M environment frames. Results below
are means of each run's final 10% of W&B history, followed by a mean and sample
standard deviation across the three seeds. Coverage statistics are directly
comparable because both batches use the same fixed-length no-reward DMLab
environment and telemetry scope.

`coverage_auc` is the time-average cumulative number of unique spatial cells:

```text
U_t = number of unique cells observed through decision t
coverage_auc = (1 / T) * sum_t U_t
```

It rewards both spatial extent and early discovery. `unique cells` is final
extent and occupancy entropy measures how evenly the visited cells are used.

## Main Conclusions

1. The strongest flat exploration result is `FSD G001_R100 X0 O0`: coverage
   AUC `71.6 +/- 15.4` and `116.8 +/- 31.7` unique cells. It remains the best
   mean in these batches.
2. The strongest HRL coverage result is `GSD G001_R100 X1 O0`: AUC
   `66.4 +/- 9.2` and `112.7 +/- 17.8` cells. However, option success is only
   `1.22 +/- 0.29%` and the target-hit rate is `0.11 +/- 0.05%`. Its behavior
   is therefore not evidence that HRL learned reliable target navigation.
3. The best structural condition for the intended HRL mechanism remains
   `GSD CTRL X0 O1`: AUC `58.8 +/- 6.4`, option success
   `20.1 +/- 16.3%`, and hit rate `2.31 +/- 1.84%`. It is much less stable in
   option success than in coverage, so it is a candidate rather than a solved
   architecture.
4. `CTRL_X1_O1 P010` is the best manager condition by external behavior:
   AUC `65.5 +/- 7.1`, `108.6 +/- 11.8` cells, and entropy `3.86 +/- 0.23`.
   Its paired AUC gain over its deadline-only control is
   `+9.2 +/- 21.6`, which is not robust with three seeds.
5. Forced exploration is too dominant. Even `p=0` spends 37-40% of transitions
   in exploration because every target timeout triggers it. At `p=0.10`,
   87% of exploration selections are still timeout-forced. This reduces known
   graph edges from about 15% to 6-9% and does not improve target success.
6. The longer deadline reduces timeouts and raises completed-option success,
   but does not reliably improve exploration. The old-to-new timeout reduction
   is `-6.39 +/- 3.37` percentage points for `X0` and
   `-6.90 +/- 2.09` for `X1`. The `X0` coverage AUC falls by
   `8.08 +/- 1.90` in all matched seeds.
7. Online DG activity does not show severe population collapse: late density
   is generally 2.2-4.4%, usage entropy is 0.95-0.99, and minibatch silent-unit
   fractions are 0-6.25%. These metrics do not establish spatial place-field
   diversity; a matched 10k-decision checkpoint telemetry evaluation remains
   necessary.

## Structural Diversity Batch

`X1` enables CA3 temporal exclusion and `O1` enables orthogonal recruitment.
`G001_R100` combines global pre-threshold punishment coefficient 0.01 with DG
row-repulsion coefficient 1.0.

### Flat policy

| Background | Exclusion | Recruitment | Coverage AUC | Unique cells | Late minus midpoint AUC |
| --- | --- | --- | ---: | ---: | ---: |
| CTRL | X0 | O0 | 50.8 +/- 27.9 | 78.9 +/- 43.5 | +2.6 +/- 3.9 |
| CTRL | X0 | O1 | 55.0 +/- 13.6 | 88.3 +/- 26.4 | -6.2 +/- 6.5 |
| CTRL | X1 | O0 | 54.6 +/- 36.4 | 87.7 +/- 62.0 | +3.1 +/- 3.1 |
| CTRL | X1 | O1 | 64.6 +/- 15.5 | 104.8 +/- 29.7 | -2.9 +/- 7.3 |
| G001_R100 | X0 | O0 | **71.6 +/- 15.4** | **116.8 +/- 31.7** | +0.7 +/- 12.7 |
| G001_R100 | X0 | O1 | 56.8 +/- 11.6 | 84.3 +/- 23.7 | +1.8 +/- 18.0 |
| G001_R100 | X1 | O0 | 52.8 +/- 8.4 | 84.0 +/- 13.6 | -15.9 +/- 16.7 |
| G001_R100 | X1 | O1 | 57.0 +/- 16.6 | 85.9 +/- 31.0 | -0.7 +/- 10.1 |

The established regularizers help only in their simple `X0 O0` combination.
Adding either structural mechanism removes that advantage. Averaged over the
full flat factorial, recruitment changes AUC by only `+0.9`, exclusion by
`-1.3`, and `G001_R100` by `+3.3`; these averages hide strong interactions.

### Fixed/global HRL

| Background | Exclusion | Recruitment | Coverage AUC | Unique cells | Option success | Hit rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CTRL | X0 | O0 | 55.3 +/- 2.7 | 91.5 +/- 5.4 | 2.19 +/- 0.10% | 0.23 +/- 0.04% |
| CTRL | X0 | O1 | 58.8 +/- 6.4 | 89.5 +/- 14.2 | **20.10 +/- 16.26%** | **2.31 +/- 1.84%** |
| CTRL | X1 | O0 | 58.7 +/- 3.1 | 97.7 +/- 4.4 | 1.18 +/- 0.56% | 0.14 +/- 0.08% |
| CTRL | X1 | O1 | 53.1 +/- 9.0 | 84.3 +/- 15.3 | 11.33 +/- 17.97% | 1.69 +/- 2.82% |
| G001_R100 | X0 | O0 | 59.0 +/- 2.4 | 99.6 +/- 3.6 | 1.26 +/- 0.50% | 0.15 +/- 0.07% |
| G001_R100 | X0 | O1 | 41.1 +/- 5.3 | 60.9 +/- 10.3 | 11.68 +/- 7.45% | 1.50 +/- 1.14% |
| G001_R100 | X1 | O0 | **66.4 +/- 9.2** | **112.7 +/- 17.8** | 1.22 +/- 0.29% | 0.11 +/- 0.05% |
| G001_R100 | X1 | O1 | 56.2 +/- 9.6 | 89.4 +/- 16.9 | 7.65 +/- 10.90% | 1.04 +/- 1.58% |

Orthogonal recruitment replaces all 16 DG rows in every enabled run, but its
HRL main effect is `-7.6` AUC. In particular, it destroys the coverage gain of
`G001_R100 X1 O0`. Recruitment is therefore too aggressive in its current
form, despite the one-per-rollout rate limit.

CA3 exclusion has a positive HRL main effect of `+5.1` AUC, driven chiefly by
`G001_R100 X1 O0`; it is not consistently beneficial in the other cells. The
new manager batch directly logs actual conflicting activations and shows
approximately 79% for both `X0` and `X1` controls. Coefficient 1 therefore does
not measurably reduce the behavior it was intended to suppress.

## Manager Exploration Batch

All manager runs use the longer target deadline. `DCTRL` disables manager
exploration, `FORCED` explores only after a target timeout, and `P010`/`P025`
also make random exploration selections with the stated probability.

| Structure | Mode | Coverage AUC | Unique cells | Option success | Hit rate | Known edges | Exploration occupancy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CTRL_X0_O1 | DCTRL | 50.7 +/- 6.7 | 78.8 +/- 13.9 | **36.9 +/- 16.9%** | **1.17 +/- 0.75%** | **14.9 +/- 1.6%** | 0% |
| CTRL_X0_O1 | FORCED | 56.5 +/- 17.3 | 90.2 +/- 36.3 | 15.3 +/- 11.2% | 0.14 +/- 0.12% | 8.8 +/- 3.2% | 40.2 +/- 2.3% |
| CTRL_X0_O1 | P010 | 51.6 +/- 16.2 | 78.4 +/- 31.5 | 18.4 +/- 12.2% | 0.16 +/- 0.11% | 7.3 +/- 1.3% | 45.6 +/- 3.4% |
| CTRL_X0_O1 | P025 | 55.3 +/- 14.8 | 84.9 +/- 28.0 | 17.2 +/- 10.3% | 0.14 +/- 0.09% | 8.6 +/- 3.4% | 50.0 +/- 9.1% |
| CTRL_X1_O1 | DCTRL | 56.2 +/- 27.3 | 91.5 +/- 51.9 | 19.1 +/- 11.9% | 0.21 +/- 0.11% | **15.6 +/- 2.1%** | 0% |
| CTRL_X1_O1 | FORCED | 54.9 +/- 3.3 | 85.2 +/- 7.4 | 15.6 +/- 4.5% | 0.12 +/- 0.04% | 8.0 +/- 2.2% | 37.0 +/- 3.2% |
| CTRL_X1_O1 | P010 | **65.5 +/- 7.1** | **108.6 +/- 11.8** | 17.8 +/- 7.4% | 0.14 +/- 0.07% | 6.4 +/- 0.2% | 41.4 +/- 5.1% |
| CTRL_X1_O1 | P025 | 54.6 +/- 7.8 | 81.5 +/- 11.9 | **33.5 +/- 22.6%** | **0.33 +/- 0.28%** | 8.3 +/- 2.9% | 51.8 +/- 5.1% |

Manager exploration provides a much denser worker signal than target hitting:
15-19% of exploration transitions receive nonzero flat distance reward, while
target-hit rates remain 0.12-0.33% in exploration-enabled conditions. Any
coverage gain can therefore arise from the flat exploration branch without an
improvement in DG-target control.

The manager also starves the graph of target data. Compared with deadline-only
controls, all exploration settings reduce known-edge coverage by 6-9
percentage points. `P010` is the only promising behavioral setting, and only
for `X1`; it does not improve option success or graph completeness. The current
forced-after-every-timeout rule should not be treated as a successful manager.

## Deadline Effect

Relative to the matching short-deadline structural runs:

| Structure | Target deadline, new | Coverage AUC change | Option-success change | Timeout-rate change |
| --- | ---: | ---: | ---: | ---: |
| CTRL_X0_O1 | 45.0 +/- 7.5 | **-8.1 +/- 1.9** | +16.8 +/- 29.2 pp | -6.39 +/- 3.37 pp |
| CTRL_X1_O1 | 99.4 +/- 12.4 | +3.1 +/- 36.1 | +7.8 +/- 11.1 pp | -6.90 +/- 2.09 pp |

The deadlines now behave as intended mechanically: options get more time and
timeout much less often. They also complete less frequently, and the target-hit
rate per transition does not improve. For `X0`, every seed loses coverage.
Deadline calibration should therefore optimize useful target attempts per unit
experience, not completed-option success alone.

## Architecture Selection

There is no single winner across objectives:

- **Flat exploration benchmark:** `FSD G001_R100 X0 O0`.
- **Highest HRL external coverage:** `GSD G001_R100 X1 O0`, but it is not
  navigating reliably to designated DG targets.
- **Best evidence of the intended target mechanism:** short-deadline
  `GSD CTRL X0 O1`, with high uncertainty across seeds.
- **Manager candidate worth one focused follow-up:** `CTRL_X1_O1 P010`, after
  reducing forced exploration occupancy and preserving target/graph samples.

The flat benchmark still exceeds every HRL condition's mean AUC. The present
results support continued mechanism work, not a claim that HRL outperforms the
Jannek-compatible flat architecture.

## Remaining Analysis

- Run matched 10k-decision place-field telemetry at several checkpoints for
  the four selected architectures above. Online density and entropy cannot
  answer whether DG fields cover distinct locations.
- Measure chance-corrected and target-shuffled hit baselines to separate
  deliberate navigation from incidental DG activation.
- Measure target attempts, hits, timeouts, and graph updates per unit target
  occupancy. Per-transition event rates are confounded when exploration takes
  40-50% of policy decisions.
- Calibrate predicted deadline against realized hit and timeout times by edge.
- Repeat any selected manager comparison with more seeds; the current paired
  manager effects have standard deviations larger than their means.

