# Recent Batch Statistics Report

Date: 2026-08-24

This report explains the statistics used in the fixed flat baseline and the
HRL persistence comparison. This is the final 100M-frame terminal analysis:
all 18 fixed-flat, 18 fixed/global-HRL, 18 long/per-stream-HRL, and 6
long-flat runs completed. The source output is the reusable evaluator at:

```text
/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/
evaluation_recent_batches_20260824/
```

The report retains strict limits on interpretation. Existing scalars can show
coverage, DG health, and an HRL option funnel, but cannot establish that a
target hit was intentional, that DGs have place fields, or that graph
deadlines are calibrated.

## Data and Aggregation

The source data are TensorBoard event files. This avoids dependence on W&B
upload state.

For each run, the `intrmotiv/eval/v1` evaluator averages each logged scalar
over its late terminal window:

```text
terminal window = last min(10M frames, 20% of observed frames)
```

The tables then average per-run terminal means. A `+/-` value is a standard
deviation across the listed three-seed condition, not a confidence interval.
Family averages pool conditions and therefore must not be read as independent
replicate estimates.

All runs use `env_frameskip=8`: one policy decision normally advances eight
DMLab engine frames. Thus a 900-decision telemetry window is normally 7,200
engine frames.

## Spatial Coverage

### Cell definition

At policy decision `t`, the wrapper receives the first two coordinates of the
DMLab position telemetry and assigns the agent to one square cell:

```text
cell_t = floor(position_t[:2] / 100)
```

The grid width is 100 DMLab position units. Let `V_t` be the set of cells seen
from the beginning of the current episode or telemetry window through `t`, and
let `C_t = |V_t|`.

### Unique cells

```text
coverage_unique_cells = C_T
```

This is the number of distinct discretized cells visited in the measurement
interval. It ignores how often each cell is revisited.

### Coverage AUC

The logged AUC is not a geometric area. It is the time-average cumulative
number of cells discovered:

```text
coverage_auc = (1 / T) * sum_{t=1..T} C_t
```

The implementation accumulates `C_t` after each step and divides by the
number of steps. It rewards discovering cells early. For example, an agent
that reaches 20 distinct cells in the first 100 steps and then remains there
gets a much larger AUC than one that first reaches the same 20 cells at the
end, although both have identical final unique-cell counts.

For a fixed `T`, the AUC is bounded above by `T` only when every step visits a
new cell. It is not normalized to `[0, 1]` and should only be compared at the
same grid size, environment, and interval length.

### Occupancy entropy

If `n_c` is the number of steps in cell `c`, with
`p_c = n_c / sum_c n_c`, then:

```text
coverage_entropy = -sum_c p_c log(p_c)
```

Entropy measures how evenly time is spread across the cells already visited.
It is complementary to unique-cell count: an agent can have many cells with
low entropy if it spends nearly all its time in one of them.

### Fixed episodes versus long windows

The statistic has two different scopes in the current experiments.

| Run family                        | Measurement interval                    | Reset behavior                                               | TensorBoard namespace                     |
| --------------------------------- | --------------------------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| Fixed flat and fixed/global HRL   | One physical fixed-length DMLab episode | Environment and coverage counters reset at terminal          | `policy_stats/avg_z_..._coverage_*`       |
| Long/per-stream HRL and long flat | 900 policy decisions                    | Only window counters reset; environment and RNN state do not | `intrmotiv/exploration/window/coverage_*` |

The long-window metric measures local exploration over each 900-decision
period. It does **not** report cumulative map coverage from the beginning of
the long episode, because its visited-cell set is cleared after every emitted
window. The fixed metric is conditioned on physical respawn; the long metric
is not. Therefore, compare fixed/global HRL only with fixed `encourage` flat,
and long/per-stream HRL only with long flat controls.

### Sample Factory smoothing

Each completed fixed episode or emitted long window is forwarded through the
standard Sample Factory episodic-statistics path. Sample Factory keeps up to
`stats_avg=100` recent reports per policy and logs their mean. The values in
TensorBoard are therefore already trailing averages. The terminal analysis
adds a second average over late logged snapshots. They are stable outcome
summaries, not raw single-episode measurements.

## DG Activity Diagnostics

All DG statistics are learner-minibatch diagnostics, computed from the DG
head output `a` after thresholding (`a > 0`). They are not direct map-place
field measurements.

| Metric | Definition | Interpretation |
|---|---|---|
| `dg_density` | Mean of `1[a > 0]` over sequence positions and DG units | Fraction of active DG units per sampled transition. |
| `dg_multi_activation_rate` | Fraction of sampled transitions with more than one active DG unit | Measures non-one-hot events; not necessarily an error. |
| `dg_silent_unit_fraction` | Fraction of DG units never active anywhere in the sampled minibatch | Detects within-minibatch collapse. It is not a lifetime silent-unit rate. |

The fixed flat result illustrates why these must be read together with
exploration: `mean` and `punish` have high coverage but roughly 80% silent DG
units. They may make reasonable flat policies but cannot provide usable DG
subgoals.

## Worker Reward and HRL Events

### Flat decoder reward

For non-HRL runs, the decoder reward is derived from the temporal internal
distance `d_t`:

```text
r_flat,t = reward_scale * (L + R - 1 - d_t)
```

Here `L=64`, `R=8`, and `reward_scale=0.1`, so the baseline is 71. This is a
dense temporal-distance signal and is not an external exploration score.

### HRL `hit_distance` worker reward

For HRL runs, the worker reward is target-gated. With target-hit indicator
`h_t`, target reward 1.0, distance-bonus coefficient 0.1, and the same
baseline:

```text
r_HRL,t = h_t * [1.0 + 0.1 * max(0, 0.1 * (71 - d_t))]
```

Thus a non-hit has exactly zero worker reward. `intrinsic_nonzero_fraction`
is consequently expected to be close to `target_hit_rate`; both quantify the
sparsity of worker supervision rather than map coverage.

| HRL metric | Exact sampled quantity |
|---|---|
| `active_target_fraction` | Fraction of valid replay transitions whose stored target ID is nonempty. |
| `target_hit_rate` | Mean stored `target_hit` flag, so hits per sampled valid policy transition. |
| `option_timeout_rate` | Mean stored option-expired flag, so timeouts per sampled valid policy transition. |
| `option_success_fraction` | `hits / (hits + timeouts)` for the learner minibatch. This is per completed option, unlike hit rate. |
| `tctrl_update_rate` | Mean stored `tctrl_updated` flag. It measures graph updates per sampled transition, not edge quality. |

Final target-hit rates are `0.00237` for fixed/global HRL and `0.00223` for
long/per-stream HRL: roughly one stored target match per 422 and 448 policy
transitions respectively. Fixed/global option success is only `0.0236`; the
corresponding long/per-stream value is `0.0125`. This is sufficient to
populate graph entries but is not evidence that the worker learned reliable
subgoal navigation. The current logs do not provide the chance-hit baseline
needed to distinguish deliberate arrivals from incidental DG matches.

## Controllability Graph Statistics

The graph uses `F=16` DG landmarks. Diagonal edges are excluded from graph
fractions.

| Metric | Definition | Caution |
|---|---|---|
| `node_visit_weight_mean` | Mean DG visit weight | Its scale depends on graph scope, event count, and decay; do not compare it as an outcome score. |
| `known_edge_fraction` | Fraction of off-diagonal pairs with `T_ctrl > 0` and confidence above threshold | A nonzero edge says an arrival was recorded, not that it is useful or causal. |
| `forgotten_edge_fraction` | Seen edges below confidence threshold | Meaningful only when confidence decay is enabled. |
| `known_controllability_time_mean` | Mean `T_ctrl` over known edges | Travel-time estimate in policy decisions. Compare only within matched graph semantics. |
| `edge_confidence_mean` | Mean off-diagonal confidence | Fast-weight support, not probability of reachability. |

For the learner-owned global graph, one accepted rollout updates model buffers
outside autograd. On each global option completion/timeout, strengths decay by
`gamma = 0.5^(1 / half_life_options)`. On success from `i` to `j` in `tau`
steps, confidence and arrival time follow the weighted running update:

```text
C_ij <- gamma * C_ij + 1
T_ij <- (gamma * C_old_ij * T_old_ij + tau) / C_ij
```

The current global runs therefore genuinely test half-lives measured in global
option events.

The long/per-stream branch does not currently execute this fast-weight rule:
`hrl_persistent_fast_weights=False` activates the legacy best-time update
path. It does not decay weights, does not use the confidence threshold for
selection, and only marks `tctrl_updated` for an arrival that improves the
stored time. This invalidates its nominal 5k/10k/20k half-life comparison.

## Final Results And Conclusions

All values below are terminal-window means over completed 100M-frame runs.
Coverage values have the scope described above: fixed/global values are
physical-episode statistics, while long values are 900-decision telemetry
windows.

| Family | Runs | Coverage AUC | Unique cells | DG density | Silent DG fraction | Hit rate | Option success | Known edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed flat, `encourage` only | 6 | 47.3 | 74.7 | 0.0233 | 0.0000 | n/a | n/a | n/a |
| Fixed/global HRL | 18 | 58.5 | 96.6 | 0.0405 | 0.0069 | 0.00237 | 0.0236 | 0.1515 |
| Long/per-stream HRL | 18 | 48.3 | 85.5 | 0.0462 | 0.0104 | 0.00223 | 0.0125 | 0.1881 |
| Long flat | 6 | 33.1 | 54.9 | 0.0407 | 0.0000 | n/a | n/a | n/a |

### Flat encoder controls

`mean` and `punish` give the highest fixed-episode coverage, between 84.3 and
92.9 AUC across schedule cells, but leave 76-84% of DG units silent in the
learner minibatches. They are therefore unsuitable as HRL landmark controls.
`encourage` maintains a non-silent DG population but has lower and more
variable coverage: 53.5 AUC for iterative and 41.0 for simultaneous updates.

There is no uniform iterative-update result in the flat batch. Iteration helps
`encourage`, hurts `mean`, and has negligible effect on `punish`. It should not
be selected as a generally superior update schedule from this batch alone.

### Fixed/global HRL

Fixed/global HRL is the only valid fast-weight half-life sweep. It keeps DG
activity healthy, uses confidence-qualified edges, and has numerically higher
fixed-episode coverage than the matched `encourage` flat control (58.5 versus
47.3 AUC). This is promising, but not yet a causal planning result: the HRL
family pools six condition cells, the fixed flat comparison is not a completed
seed-paired statistical test, and target success remains very low.

No global half-life or update schedule wins reproducibly. The six three-seed
coverage means range narrowly from 57.2 to 59.8 AUC:

| Half-life, global option events | Iterative AUC | Simultaneous AUC |
|---|---:|---:|
| 5k | 57.2 +/- 4.0 | 59.8 +/- 5.3 |
| 10k | 58.4 +/- 2.8 | 58.3 +/- 2.5 |
| 20k | 58.5 +/- 8.7 | 58.6 +/- 4.6 |

The final maximum is 5k/simultaneous, not the earlier provisional 10k cell,
but the variation is too large and the cell separation too small to select a
half-life or update schedule.

### Long/per-stream HRL

Long/per-stream HRL has higher long-window coverage than long flat (48.3
versus 33.1 AUC), but both have substantial variation and cannot be compared
to fixed-episode values. More importantly, this branch ran with
`hrl_persistent_fast_weights=False`. Its nominal 5k/10k/20k factor is inert:
it uses the legacy best-time update, no confidence decay, and no
confidence-threshold target gating. Its half-life cells must therefore not be
used as a sensitivity study or architecture selection criterion.

### Supported Conclusion

The fixed/global graph is the most promising current HRL architecture because
it preserves DG activity, maintains confidence-gated graph state, and has the
best valid HRL external-coverage result. The evidence is still insufficient to
claim goal-directed navigation or graph-based planning. The next comparison
must add chance-corrected target hits, stored option events, target-shuffle
policy probes, spatial DG-field evaluation, and predicted-versus-realized
arrival-time calibration.

## Metrics That Must Not Be Overinterpreted

- `distance_metric` is an internal temporal-distance statistic, not spatial
  coverage or an achievement score.
- Raw intrinsic reward is not comparable between flat and HRL runs because
  flat uses dense temporal reward while HRL uses a hit-gated reward.
- High known-edge fraction does not establish controllability; it must be
  accompanied by rising option success and coverage.
- Higher coverage AUC does not prove planning or target following. It is an
  external behavior outcome; target-hit and option-success trajectories are
  needed for that causal claim.
