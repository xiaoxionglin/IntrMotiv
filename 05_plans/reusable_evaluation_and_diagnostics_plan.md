# Reusable IntrMotiv Evaluation And Diagnostics Plan

Date: 2026-08-24

## Purpose

Create a reusable evaluation and diagnostics system for IntrMotiv runs and
architectural variants. It should establish whether the model is behaving as
intended, rather than treating a single outcome metric or internal loss as
evidence of success.

The system must distinguish these claims:

1. Training was technically valid.
2. The agent explores the environment better externally.
3. DG units form stable, spatially useful landmarks.
4. The worker responds to and reaches supplied DG targets above chance.
5. The controllability graph improves target selection and deadlines.

The present batch-specific analysis scripts remain useful historical tools,
but should not become the foundation for new evaluation work.

## Location And Compatibility

The existing `sf_working_directories/IntrMotiv/analysis/` directory contains
ad hoc batch analysis and historical reports. Add a distinct reusable package:

```text
sf_working_directories/IntrMotiv/evaluation/
|-- README.md
|-- METRIC_SCHEMA.md
|-- protocols.py
|-- schema.py
|-- cli.py
|-- collect/
|   |-- online_metrics.py
|   |-- trajectory_recorder.py
|   `-- checkpoint_evaluator.py
|-- diagnostics/
|   |-- exploration.py
|   |-- dg_fields.py
|   |-- target_control.py
|   |-- controllability_graph.py
|   `-- learning_health.py
`-- report/
    |-- run_report.py
    |-- batch_report.py
    `-- figures.py
```

The package must use TensorBoard event files as the canonical online scalar
source. W&B may mirror metrics but must not be necessary to analyze a run.

Existing `intrmotiv/` TensorBoard/W&B names remain unchanged. The evaluation
package provides a documented alias map for old `train/...` tags and produces
the existing terminal-window CSV fields during the transition. Existing
`analysis/analyze_recent_batches.py` can later become a thin compatibility
entry point into the new package, without invalidating prior outputs.

Avoid Sample Factory changes unless the existing generic periodic-statistics
forwarding cannot support a required scalar. Any generic hook must be
default-off and preserve normal Sample Factory behavior.

## Versioned Metric Contract

Define a versioned evaluation schema, initially `intrmotiv/eval/v1/`. Every
metric definition records:

- full metric name, unit, numerator/denominator, and aggregation method;
- source: learner minibatch, actor stream, terminal episode, telemetry window,
  or checkpoint evaluator;
- temporal scope: update, option, episode, telemetry window, or full
  evaluation rollout;
- applicability: flat, HRL, graph scope, and optional architectural module;
- whether the metric is an outcome, diagnostic, or validity check.

Each output directory contains a manifest with:

- source revision and dirty-worktree state;
- fully resolved training configuration;
- evaluation protocol ID and schema version;
- environment, map, episode, and telemetry semantics;
- graph scope: none, episode, per-stream, or policy-global;
- target and reward semantics.

Reports must refuse direct numerical comparison when protocol scopes differ,
for example fixed physical episodes versus long non-terminal telemetry windows.

## Online Training Metrics

Online logs should be cheap scalar summaries or bounded histograms. They
should not store full trajectories from every training actor.

### Training validity and learning health

Log and report:

- environment frames, policy decisions, throughput, and learner/actor lag;
- PPO policy/value loss, entropy, KL, clipping fraction, and explained
  variance;
- encoder and decoder gradient norms, parameter-update norms, and iterative
  update phase;
- invalid-sample fraction and NaN/Inf counters;
- reward mean, variance, nonzero fraction, and return-normalizer state.

This distinguishes an algorithmic failure from a run that did not train as
configured.

### DG representation health

Retain current DG density, multi-activation fraction, and minibatch silent
unit fraction. Add:

- per-unit activation-count distribution and active-unit entropy;
- dominant-DG distribution;
- lifetime DG usage and lifetime silent-unit fraction;
- DG persistence and dominant-unit switch rate;
- pairwise DG co-activation summary;
- CA3 register magnitude/occupancy by lag;
- feedback and each auxiliary-loss contribution relative to the encoder loss.

These metrics detect both collapse to silent units and the converse failure in
which most units activate nearly everywhere.

### HRL option funnel

Represent the complete target-control chain as rates and counts:

```text
valid source
-> eligible target candidates
-> feasible candidates
-> target selected
-> target-conditioned action
-> option completion
-> target hit or timeout
-> graph update
```

For every completed option, aggregate:

- source type: current DG or CA3-trace fallback;
- eligible and feasible candidate counts;
- no-candidate and unreachable-fallback rates;
- selected target novelty rank and visit weight;
- known versus fallback deadline, deadline length, and completion elapsed
  time;
- reset reason: initialization, hit, timeout, terminal, or explicit policy
  reset;
- success rate by deadline bucket and predicted-time bucket;
- target hit rate and expected chance hit rate;
- policy-global graph version and age at the point of sampling.

The hit-rate statistic alone cannot separate learned target navigation from an
incidental activation of the selected DG.

### Graph usefulness and calibration

Log:

- observed, confidence-qualified, and feasible edge fractions;
- edge age, graph-update count, and update provenance;
- update type: intended hit, incidental observation, hindsight graph-only
  update, or timeout;
- predicted arrival time versus observed elapsed time;
- deadline calibration: success frequency and elapsed-time residual by
  predicted-time bucket;
- actor graph-version lag for policy-global graphs;
- node/edge decay and forgotten-edge fractions.

An edge count is evidence that a transition was seen, not evidence that the
graph is useful for control. The metrics must make this distinction explicit.

## Checkpoint Evaluation Protocol

Use Sample Factory's existing checkpoint/evaluation mechanism rather than
creating a new trainer. Add an IntrMotiv diagnostic wrapper around the
evaluation environment and model outputs.

Evaluate checkpoints at:

```text
0, 1M, 5M, 10M, 25M, 50M, 75M, and 100M environment frames
```

Use a fixed, versioned suite of map seeds and start locations. For each
checkpoint, run:

- stochastic evaluation for exploration behavior;
- greedy evaluation for target-following behavior.

Evaluation must not update model parameters, optimizer state, graph buffers,
BatchNorm running statistics, or training counters.

Store bounded detailed artifacts only in the NEMO workspace:

```text
/work/classic/fr_xl1014-train/IntrMotiv/.../_evaluation/
  <run_id>/<checkpoint_step>/<protocol_id>/
    manifest.json
    summary.json
    episodes.parquet
    option_events.parquet
    dg_field_maps.npz
```

The initial default is 16 capped episodes per checkpoint. This produces enough
data for diagnosis without making artifact storage a training bottleneck.

## Offline Evaluation Probes

### External exploration

For each evaluation protocol, calculate:

- coverage AUC, final unique cells, occupancy entropy, and revisitation ratio;
- early versus late cell-discovery rate;
- displacement from start and spatial extent;
- occupancy heatmaps;
- seed-wise curves and seed-paired condition deltas.

Coverage AUC remains:

```text
coverage_auc = (1 / T) * sum_t unique_cells_seen_by_t
```

It must always be shown beside final unique-cell count and occupancy entropy.
It measures early discovery, not geometric area.

### DG place-field quality

Using position and DG traces saved by checkpoint evaluation, compute per DG:

- activation rate;
- occupancy-corrected spatial information;
- field compactness and disconnected-component count;
- field coverage;
- split-half spatial-map reliability;
- dominant-DG spatial confusion matrix;
- lifetime silent-unit fraction.

Use temporal or visit split halves for spatial-map reliability. Do not claim a
place field only because the same trajectory both estimated and scored it.

### Target conditioning and intentional control

For every selected target, compute:

```text
chance_corrected_hit_lift = observed_target_hit_rate /
                            expected_hit_rate_from_target_marginal_frequency
```

Also report:

- option success and timeout survival curves by deadline;
- spatial progress toward the target's independently estimated field;
- target-switch, self-target, and repeated-failed-target rates;
- target sensitivity of the worker policy:

```text
KL(pi(action | state, selected_target) ||
   pi(action | state, shuffled_valid_target))
```

Near-zero target sensitivity means that the worker ignores target conditioning.
High target sensitivity without above-chance hit lift means the policy responds
to targets but has not learned to navigate to them.

### Graph calibration and causal contribution

For graph-enabled runs, calculate:

- predicted versus realized first-arrival time for one-hop and multi-hop paths;
- feasibility precision and recall;
- deadline calibration;
- whether newly feasible nodes are explored subsequently;
- graph/target contribution ablations using the same recorded evaluation
  states: recorded targets, blank target, and shuffled valid target.

The recorded-target condition isolates worker execution. Blank and shuffled
conditions test whether the target input materially changes the policy. They
are evaluation-only and never alter PPO data or replayed log-probabilities.

## Standard Reports And Conclusions

`evaluation cli report-run` should produce Markdown/HTML in this order:

1. configuration and validity;
2. learning curves;
3. external exploration;
4. DG representation;
5. option funnel and chance-corrected control;
6. graph calibration;
7. supported conclusions, failure diagnosis, and unresolved questions.

`evaluation cli report-batch` should produce:

- seed-paired condition comparisons;
- confidence or bootstrap intervals;
- learning curves in addition to terminal summaries;
- a machine-readable `conclusion_ledger.json` with evaluated runs, protocol,
  validity status, supported conclusion, unresolved questions, and analysis
  version.

This should become the structured source for conclusions while the current
`RECENT_BATCH_ANALYSIS_LEDGER.md` remains historical context.

## Implementation Sequence

1. **Foundation**
   - Create the package, metric schema, protocol manifests, CLI, TensorBoard
     loader, and legacy-tag adapter.
   - Test schema validation, scope checking, and flat-run handling when HRL
     metrics are absent.

2. **High-value online diagnostics**
   - Add the option funnel, chance-hit baseline, graph-update provenance, and
     DG lifetime usage metrics.
   - Preserve all present TensorBoard tags.

3. **Checkpoint evaluation**
   - Integrate with existing Sample Factory evaluation/checkpoint APIs.
   - Add compact trajectory, DG, and option-event recording under workspace
     paths only.

4. **Offline probes and reports**
   - Implement exploration, DG-field, target-control, and graph-calibration
     diagnostics.
   - Generate one report for a flat baseline and one for fixed/global HRL.

5. **Preflight validation**
   - Run flat, fixed/global HRL, and long/per-stream HRL preflights.
   - Verify artifact bounds, schema labeling of graph and episode scope, and
     refusal to compare incompatible protocols.

## Priority

Implement checkpoint evaluation, the option funnel, and chance-corrected hit
analysis first. These directly resolve the central uncertainty in current HRL
runs: whether the worker is genuinely following DG targets or merely producing
incidental target matches.
