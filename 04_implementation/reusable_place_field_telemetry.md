# Reusable DG Place-Field Telemetry

## Purpose

This is the standard offline workflow for diagnosing whether IntrMotiv DG
units form active, spatially distributed, and reasonably stable receptive
fields. Use it across flat, HRL, manager, loss, threshold, and update-schedule
variants instead of writing batch-specific `enjoy` scripts.

The workflow separates four questions:

1. **Activity:** do DG units cross threshold during a sufficiently long probe?
2. **Selectivity:** do active units carry spatial information?
3. **Diversity:** do different units peak in different parts of the map?
4. **Stability:** do field maps persist across checkpoints on shared occupancy?

Online minibatch density and usage entropy cannot answer these questions. A
policy can have healthy online DG activity while all units respond to the same
few locations.

Training may request the same privileged DMLab `(x, y, yaw)` signal for
environment-only coverage and viewpoint metrics. In that path the reward
wrapper consumes the pose from `info` and removes it before Sample Factory
stores observations or learner rollouts. These online exploration summaries
complement but do not replace the 10k manifest-driven DG place-field protocol:
they contain no DG activation maps and cannot establish spatial selectivity or
map stability.

## Authoritative Locations

NEMO2 source checkout:

```text
/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/
  sf_working_directories/IntrMotiv/evaluation/
```

| Component | Role |
| --- | --- |
| `place_fields.py` | Loads one checkpoint, runs DMLab, and writes raw spatial arrays. |
| `submit_place_field_sweep.py` | Validates a manifest and submits one ordinary Slurm job per selected row. |
| `run_place_field_sweep_single.sh` | Executes one zero-based manifest row inside an ordinary Slurm job. |
| `run_place_field_sweep_array.sh` | Compatibility wrapper for historical or already queued arrays only. |
| `build_place_field_sweep.py` | Provides retained-checkpoint discovery and nearest-checkpoint selection. |
| `summarize_place_fields.py` | Produces standard per-artifact metrics, DG maps, and pre-threshold maps. |
| `analyze_place_field_manifest.py` | Adds active-only diversity metrics and replicated terminal/trajectory tables. |
| `plot_place_field_trajectories.py` | Produces five-checkpoint contact sheets and trajectory figures. |
| `map_stability.py` | Compares occupancy-weighted maps with a final reference checkpoint. |
| `target_control_interventions.py` | Runs frozen-policy, frozen-graph target interventions from exclusive source events and writes ordered-pair trial records. |
| `run_target_control_intervention_single.sh` | Executes one 75M intervention-manifest row as an ordinary Slurm job. |

The derived analyzer is also mirrored in this vault at
[[../06_experiments/analyze_place_field_manifest.py|analyze_place_field_manifest.py]]
so its metric implementation remains visible with the reports.

All raw and generated telemetry data must be written below:

```text
/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/
```

Do not write DMLab rollout artifacts, logs, caches, or temporary files into the
NEMO home checkout.

## Standard Protocol

Unless a report states otherwise, use:

- 10,000 policy decisions per checkpoint; the evaluator currently records
  10,001 occupancy samples because the endpoint is inclusive;
- stochastic policy actions, matching previous candidate telemetry;
- seed 99 at checkpoints nearest 5M, 25M, 50M, 75M, and 100M frames;
- seeds 8 and 123 at the final checkpoint;
- the same environment, grid size, action stochasticity, and rollout length
  for every condition in a comparison;
- one short ordinary Slurm preflight before the production jobs.

This is seven tasks per architecture. It gives a five-checkpoint trajectory and
a three-seed terminal result without evaluating every seed at every checkpoint.
Use all-seed trajectories only when seed-specific dynamics are the research
question.

Two thousand decisions are not enough for normal field-diversity evaluation.
They can detect total silence, but often leave too much of the map unvisited.

## Manifest Contract

`submit_place_field_sweep.py` and `run_place_field_sweep_single.sh` expect a
tab-separated manifest with this exact column order:

```text
condition
family
schedule
feedback
half_life
seed
target_frames
checkpoint_frames
checkpoint
run_dir
label_suffix
```

Important requirements:

- `label_suffix` must be unique for every row and should include condition,
  seed, and actual checkpoint frames.
- `checkpoint` and `run_dir` must be absolute workspace paths.
- `target_frames` is the requested comparison point; `checkpoint_frames` is
  the actual retained checkpoint loaded.
- Use `select_checkpoints(run_dir)` from `build_place_field_sweep.py` rather
  than reimplementing checkpoint discovery.
- Keep a full analysis manifest and a seed-99 trajectory-only manifest. The
  latter prevents final checkpoints from other seeds being interpreted as a
  temporal sequence by stability tools.

Before submission, verify every checkpoint exists, all labels are unique, the
expected row count is present, and every output/log/temp path is in the
workspace.

For studies using `intrmotiv/study/v1`, generate both manifests with the
canonical workflow rather than creating another batch-specific builder:

```bash
python -m hpc_runs.intrmotiv_study render-telemetry \
  hpc_runs/studies/STUDY.study.json BATCH_ROOT OUTPUT_ROOT
```

This reuses `select_checkpoints(run_dir)`, exact study run identities, the
standard five-plus-two seed/checkpoint protocol, workspace validation, and the
study fingerprint. Existing batch-specific manifest builders are historical
compatibility tools; extend `hpc_runs/intrmotiv_study/telemetry.py` when a new
generally useful selection component is needed.

### Relationship to Jannek's telemetry launcher

`dmlab/experiments/run_generate_telemetry.py` in IntrMotiv is currently a
byte-for-byte copy of Jannek's run description and still contains Jannek's
`fr_js1764` workspace and encoder paths. The current manifest-driven
`place_fields.py` evaluator was not derived from that file: it was built to
preserve explicit checkpoint selection, raw NPZ maps, continuous pre-threshold
maps, and the downstream manifest-analysis contract.

The standard workflow does retain the useful scheduling property of Jannek's
Sample Factory launcher: every evaluation is submitted as an independent
ordinary Slurm job. The small telemetry-specific submitter is the documented
exception to the normal rule against private submission loops; training and
ordinary experiment batches must continue to use Sample Factory's launcher.

## Rollout Submission

Set the common locations on NEMO2:

```bash
REPO=/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam
PY=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
ROOT=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/<analysis_name>
```

The submitter is print-only by default. It validates the manifest, selected
rows, checkpoints, run directories, unique labels, and workspace paths, then
prints one independent `sbatch` command per row. Review this plan before adding
`--submit`.

Run one representative checkpoint as a cheap preflight. The established
telemetry resource profile is 4 CPUs and 16 GB:

```bash
cd "$REPO"

PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/submit_place_field_sweep.py \
  --manifest "$ROOT/analysis_manifest.tsv" \
  --output-dir "$ROOT/preflight" \
  --row <representative_row> \
  --max-num-frames 500 \
  --time-limit 00:20:00

# After reviewing the single printed sbatch command:
PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/submit_place_field_sweep.py \
  --manifest "$ROOT/analysis_manifest.tsv" \
  --output-dir "$ROOT/preflight" \
  --row <representative_row> \
  --max-num-frames 500 \
  --time-limit 00:20:00 \
  --submit
```

Require exit code zero, one `place_fields.npz`, thresholded and pre-threshold
arrays, and no traceback before production. Then print and review the full 10k
ordinary-job plan:

```bash
PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/submit_place_field_sweep.py \
  --manifest "$ROOT/analysis_manifest.tsv" \
  --output-dir "$ROOT" \
  --row 0-<last_row> \
  --max-num-frames 10000 \
  --time-limit 01:00:00

# Submit only after the printed plan is correct:
PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/submit_place_field_sweep.py \
  --manifest "$ROOT/analysis_manifest.tsv" \
  --output-dir "$ROOT" \
  --row 0-<last_row> \
  --max-num-frames 10000 \
  --time-limit 01:00:00 \
  --submit
```

The submitter pauses one second between calls and writes a timestamped
`submission_manifest_*.tsv` containing every row, label, ordinary job ID, and
exact command. It never adds `--array` or dependencies. This follows the
code-driven, one-job-per-run scheduling pattern used by Sample Factory's
launcher while preserving the newer manifest and NPZ contracts.

Do not use `run_place_field_sweep_array.sh` for a new NEMO2 batch. It remains
only so already queued and historical array jobs stay reproducible.

## Postprocessing

Run lightweight postprocessing from the source checkout after all submitted
jobs complete:

```bash
cd "$REPO"

PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/summarize_place_fields.py \
  --input-dir "$ROOT/raw" \
  --out-dir "$ROOT/summary"

PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/analyze_place_field_manifest.py \
  --input-dir "$ROOT" \
  --manifest "$ROOT/analysis_manifest.tsv" \
  --out-dir "$ROOT/summary"
```

Generate checkpoint sheets with the trajectory-only manifest:

```bash
cd "$REPO/sf_working_directories/IntrMotiv/evaluation"

PYTHONPATH="$REPO:$PWD" "$PY" plot_place_field_trajectories.py \
  --input-dir "$ROOT" \
  --manifest "$ROOT/trajectory_manifest.tsv" \
  --out-dir "$ROOT/trajectories"
```

Run `map_stability.py` once per condition, again using the trajectory-only
manifest:

```bash
cd "$REPO"

PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/map_stability.py \
  --input-dir "$ROOT" \
  --manifest "$ROOT/trajectory_manifest.tsv" \
  --condition <condition> \
  --out-dir "$ROOT/stability/<condition>"
```

## Target-Control Intervention Protocol

Studies that declare `telemetry.intervention.protocol` as
`target-control-intervention-v1` reuse the standard 75M rows in
`analysis_manifest.tsv`. Submit them with the existing ordinary-job submitter
and the intervention runner; do not build a second run/checkpoint inventory:

```bash
PYTHONPATH=. "$PY" \
  sf_working_directories/IntrMotiv/evaluation/submit_place_field_sweep.py \
  --manifest "$ROOT/intervention_manifest.tsv" \
  --output-dir "$ROOT/interventions" \
  --runner sf_working_directories/IntrMotiv/evaluation/run_target_control_intervention_single.sh \
  --max-num-frames 100000 \
  --time-limit 04:00:00 \
  --job-name-prefix intrmotiv-control
```

Review the printed ordinary-job plan before adding `--submit`. Each row:

- loads the exact manifest checkpoint and keeps policy parameters and graph
  buffers frozen;
- begins a trial only on an exclusive DG source event;
- balances commands over all 15 off-diagonal targets, up to five trials per
  ordered pair within 100,000 decisions;
- uses the same reliable-edge/passive/bootstrap deadline rules as training;
- records source pose and bins, success, elapsed decisions, physical path
  length, endpoint, and every DG hit during the trial;
- evaluates all alternate targets on the same core state to measure
  counterfactual action sensitivity;
- constructs a deterministic nearest-context, same-source shuffled-target
  control from the completed trial table.

The row output is `intervention_trials.csv` plus
`intervention_summary.json`. When `study_manifest.json` is adjacent to the
input TSV, the summary carries its schema, workflow version, study ID, and
study SHA-256.

## Raw Artifact Contract

Each rollout writes `raw/<label>/place_fields.npz`. Current analysis expects:

| Array | Meaning |
| --- | --- |
| `occupancy` | Decision count in each 19 x 19 position cell. |
| `rate_maps` | Occupancy-corrected post-threshold DG activity per cell and unit. |
| `spatial_information` | Occupancy-corrected information in bits per DG unit. |
| `active_fraction` | Post-threshold duty cycle per DG unit. |
| `pre_threshold_rate_maps` | Occupancy-corrected continuous DG logits per cell and unit. |
| `pre_threshold_mean`, `pre_threshold_std` | Per-unit rollout logit moments. |
| `pre_threshold_above_fraction` | Per-unit fraction above the configured threshold. |
| `control_edge_confidence`, `control_attempts`, `control_tctrl` | Optional frozen policy controllability-graph snapshot. |
| `passive_confidence`, `passive_elapsed`, `birth_support` | Optional passive recruitment evidence and maturity snapshot. |
| `recruitment_row_counts` | Optional per-DG-row assignment counts. |

Treat additions as backward-compatible optional arrays. Do not silently change
the meaning or shape of existing arrays; update tests and this document when a
schema change is necessary.

## Derived Metrics

For active unit set `A`, the analyzer computes pairwise map cosine over cells
visited in the rollout:

```text
cos(i, j) = <r_i, r_j> / (||r_i|| ||r_j||),  i,j in A
```

Lower cosine means less redundant maps. Units with zero thresholded events are
excluded so silent zero vectors cannot lower the score artificially.

For each active unit, the peak is the maximum positive value of its
occupancy-corrected rate map. If `n_b` units peak in cell `b`, normalized peak
entropy is:

```text
p_b = n_b / |A|
H_peak = -sum_b p_b log(p_b) / log(|A|)
```

`active_unique_peak_bins` counts occupied peak cells and
`active_pairwise_peak_distance_bins` averages Euclidean peak distance in grid
cells. Since one grid cell is 100 DMLab position units, multiply by 100 for
environment units.

The same cosine, peak count, entropy, and distance are computed over all
continuous pre-threshold logit maps. Agreement between thresholded and
pre-threshold results shows that a diversity result is not merely caused by the
hard threshold.

## Outputs

| Output | Use |
| --- | --- |
| `summary/place_field_summary.csv` | Standard per-checkpoint activity, SI, cosine, and peak count. |
| `summary/derived_place_field_metrics.csv` | Active-only and pre-threshold diversity per checkpoint. |
| `summary/terminal_three_seed_aggregate.csv` | Mean and sample SD for terminal replicated conditions. |
| `summary/seed99_checkpoint_trajectory.csv` | Numerically ordered checkpoint trajectory. |
| `summary/place_fields_*.png` | Per-checkpoint thresholded DG maps. |
| `summary/pre_threshold_logits_*.png` | Per-checkpoint continuous logit maps. |
| `trajectories/field_evolution_*.png` | Five-checkpoint seed-99 contact sheets. |
| `stability/<condition>/rate_map_stability_summary.csv` | Correlation and peak displacement to final maps. |

Copy only lightweight CSVs and report figures into this Obsidian vault. Keep
raw NPZs, complete plot sets, and Slurm logs in the workspace.

## Interpretation Rules

- Do not infer spatial diversity from online DG density or usage entropy.
- Always report activity/silence with map cosine. Silent maps can make an
  all-unit cosine summary look deceptively low.
- Always inspect both thresholded and pre-threshold maps.
- Compare identical rollout lengths, position grids, environments, and action
  stochasticity.
- Report individual seeds as well as mean +/- sample SD. Three seeds are not a
  high-powered significance test.
- A selected-candidate comparison is not a causal factor ablation. Exact loss
  or architecture attribution requires matched conditions differing in one
  factor.
- Policy-driven checkpoint rollouts do not follow identical paths. Stability
  correlations use shared occupancy but cannot replace a fixed observation
  panel or scripted trajectory.
- High spatial diversity is necessary for DG subgoals but does not prove the
  worker can intentionally reach them. Pair telemetry with chance-corrected
  target hits, option success, graph coverage, and external exploration.

## Extension Points

Build new diagnostics on the manifest and NPZ contracts rather than creating a
parallel evaluation path. High-value extensions are:

1. Fixed observation panels or scripted trajectories for causal field drift.
2. Split-half reliability within one 10k rollout.
3. Bootstrap uncertainty over trajectory segments and occupancy cells.
4. Target-conditioned probes comparing intended and accidental DG hits.
5. Per-unit alignment between field stability, visit count, reward, and graph
   confidence.
6. HTML or notebook exploration generated from the same CSV outputs.

Keep new metrics additive, add focused tests under IntrMotiv's `tests/`, update
the raw artifact contract when needed, and preserve old manifests and NPZs.

## Validated Example

The 2026-08-27 run evaluated 28 tasks with no failures under Slurm job
`7881719`. Its report, figures, limitations, and artifact root are recorded in
[[../06_experiments/dg_structural_manager_place_field_telemetry|Structural And Manager Place-Field Telemetry]].
