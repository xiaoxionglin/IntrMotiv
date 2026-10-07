# Frozen-policy behavior and field expression — 7 October 2026

## Question and interpretation

For the same final checkpoint, do the agent's own actions expose more spatially
informative DG, CA3, and decoder activity than random actions? The actor,
encoder, decoder, and BatchNorm statistics are frozen. The action replacement
is an evaluation-time intervention; it cannot establish how policy sampling
changed the weights during training.

## Pinned cohort and protocol

The cohort is the Bernstein poster's corrected-core C01, C05, and C15 models,
each with training seeds 8, 99, and 123. Every checkpoint is the exact
100,040,704-frame final checkpoint already named in the
[poster terminal table](../results/A0_poster_analysis_20260926/flat_goal_comparison/matched_terminal_per_run.csv).
The [54-row evaluation manifest](../../hpc_runs/studies/behavior_field_expression_20261007.tsv)
has SHA-256 `db5797830cca1ec4f8366fa5e429d226488d735347f8f936bba32fdd9f44900b`.
It crosses nine models, two independent evaluation seeds (51,000 and 52,000),
and three action policies. Each row requests exactly 50,000 agent decisions.
Both evaluation seeds are repeat measurements, not extra training seeds.

The primary contrast replaces each independently sampled stochastic policy
action with an independent uniform draw from that checkpoint's allowed action
set. The secondary control holds each random draw for eight decisions. In both
arms the frozen model still processes every observation and updates its
recurrent/controller state; only the motor action sent to DMLab changes.
Environment starts use the same seed within each model/evaluation-seed pair.
Recurrent state resets at episode boundaries. All new checkpoint copies,
rollout arrays, Slurm logs, temporary data, and caches are under
`/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/behavior_field_expression_20261007/`.
The former `fr_xl1014-train` allocation is read-only input provenance.

## Measurements and decision rule

The [probe](../../hpc_runs/behavior_field_probe.py) saves aligned pose,
episode boundaries, post-threshold DG, continuous DG logits, full CA3 trace,
decoder hidden layers 1 and 2, predicted value, action logits/probabilities,
proposed action, and executed action. Full arrays remain in the active
workspace. The [postprocessor](../../hpc_runs/behavior_field_analysis.py)
calculates 19-by-19 occupancy-corrected maps. Signed downstream activities are
rectified for rate/field measurements; their original signed traces are saved.

The primary score is the canonical amplitude-weighted spatial score divided
by that unit's mean activation, yielding bits per activation. Each paired
comparison uses equal weight on bins with at least ten visits in **both**
policies and the intersection of canonically eligible units. A comparison is
undefined below 20 shared bins or without a jointly eligible active unit.
This shared-support score is distinct from the historical native-occupancy
score, which remains in the per-run table for continuity. The postprocessor
also saves 10k/20k/30k/40k/50k paired prefixes, split-half reliability,
silence, mono-fields, components, dominant mass, distinct peaks, active-only
cosine, occupancy, headings, movement, and actions. Value and action outputs
are archived exploratory traces, with no field claim predeclared for them.
The deployed probe schema is `intrmotiv/behavior-field-probe/v1` and its
source SHA-256 is `df1f52f624cbf26a27503d764013ef38e4e7f4fb073e7b346310f672d9254efa`;
the deployed postprocessor source SHA-256 is
`aafab7ce9f54f42818d36a309854fe4cd37ad6f97e4134b7ff08d4a4c902680d`.

A favorable result requires a consistent positive own-minus-uniform difference
across the three training seeds of a condition, stable direction late in the
rollout, and adequate common spatial support. The two evaluation seeds assess
measurement stability and are averaged within each training seed. The
eight-decision random control tests sensitivity to random-action persistence.
Mixed or unsupported layers are reported as such. The old 10k probes provide
context and are not pooled with these 50k measurements.

## Execution and verification status

- The nine historical `config.json` files and final checkpoints were copied
  into the active workspace with SHA-256 verification. The input inventory is
  `staged_inputs.json` beside the workspace `inputs/` folder.
- Two 256-decision compute-node preflights completed successfully: own-policy
  Slurm job `8294306` and uniform-random job `8294307`, both exit 0. Their
  first poses and neural readouts match. All layer/value/action arrays have
  exactly 256 aligned finite rows; action probabilities sum to one within
  $1.2\times10^{-7}$. Uniform random replaced 213 of 256 proposed actions.
  The checkpoint's learned parameters and BatchNorm running statistics had
  identical hashes before and after each probe.
- On that trace, the new DG occupancy matches the established evaluator
  exactly; rate maps differ by at most $9.6\times10^{-8}$ and the canonical
  spatial score by at most $3.1\times10^{-8}$. The eight focused tests pass
  locally and in the NEMO2 runtime.
- Print-only review validated 18 own-policy and 18 uniform-random independent
  50k-decision jobs against the pinned manifest and active-workspace paths.
  The first submissions were cancelled before producing any 50k trace:
  inspecting the archived C01 config revealed restored `train_dir`, DMLab
  cache, and W&B paths under the retired allocation. The adapter now overrides
  those three paths before the canonical loader constructs DMLab, disables
  W&B, and records the resolved paths in every probe's metadata. A second
  short preflight completed (jobs `8294361`/`8294362`, exit 0), and its
  metadata records only active-workspace runtime paths. The 36 primary jobs
  were resubmitted; their authoritative job IDs are in
  `probes/submission_20261007T160713Z.tsv` and
  `probes/submission_20261007T160743Z.tsv`.
  The cancelled submission records are
  `probes/submission_20261007T155751Z.tsv` and
  `probes/submission_20261007T155828Z.tsv`; their 36 jobs were explicitly
  cancelled, without affecting other studies. Results are pending. The
  persistent-random arm follows primary completion and health review.

## Reusable workflow note

The canonical checkpoint loader and map contract allowed the new policy
contrast to stay a thin evaluator adapter. The critical validation was a
paired short rollout: it caught layer-hook, action-override, and alignment
errors before the large matrix. Future behavior-sampling probes should reuse
the same staged checkpoint identities and matched-support normalization,
then change only the declared action controller.
