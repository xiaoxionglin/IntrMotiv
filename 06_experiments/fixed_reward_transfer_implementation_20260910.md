# Fixed-reward transfer implementation and launch, 2026-09-10

## Scientific question

The study tests whether IntrMotiv pretraining accelerates learning in
`openfield_map2_fixed_loc3` relative to the same architecture trained from
scratch. Success is defined on downstream external reward learning curves,
measured against environment frames. The primary evidence is earlier
time-to-threshold and larger early area under the learning curve across matched
seeds; final performance is secondary.

Waypoint selection is bypassed. The fixed task uses a trainable 16-dimensional
constant task vector, initialized uniformly, through the existing
`target_id_film` conditioning path. The controllable graph, intrinsic reward,
and goal/waypoint selection are disabled. This preserves the goal-conditioned
policy interface without asking a selector to rediscover a goal already fixed
by the environment.

## Production design

The matrix contains seven conditions at seeds 42, 1234, and 9999, for 21 runs:

- scratch;
- SCR checkpoint: frozen DG, trainable DG, and policy fine-tuning;
- SAT checkpoint: frozen DG, trainable DG, and policy fine-tuning.

The two sources are the selected 75,038,720-frame checkpoints from
`SCR_C15_ARR_DIRS_S123` and `SAT_C15_ARR_DIRO_FILM_S8`. Their SHA-256 values are
`ec95b4319f17fa1d8c2c10b0a7778df2932a4f710a8c615f06dc95930a56da20`
and `7b3275e28ef78df68d523d6203925fbb6ee01c0edd2bc380bcbc54107a43b3fb`,
respectively.

Each run trains for 108,000 seconds with a 32-hour Slurm allocation. It uses
APPO, 32 workers, eight environments per worker, rollout and recurrence 64,
batch size 2,048, learning rate 0.0002, frameskip 4, the reduced five-action
set, one policy, external reward only, and no PBT. The critic, optimizer, and
training counters always start fresh. DG transfer loads only the DG projection
and normalization tensors. Policy transfer additionally loads the decoder and
action head.

The source of truth is
`hpc_runs/studies/fixed_reward_transfer.study.json`, schema
`intrmotiv/study/v1`, workflow 1.5.0. Its final SHA-256 is
`b7e5c5db2547cf5e05763718479dfa8a54e9dbd83a6b30012d33a9c0c2920d73`.

## Runtime changes and verification

The runtime now supports an explicit transfer path, `dg` or `policy` scope,
optional DG freezing, and fixed-task conditioning. Transfer is applied only to
a new downstream run; an existing downstream checkpoint takes precedence on
resume. Tensor names and shapes are checked fail-closed. Frozen legacy
BatchNorm uses stored checkpoint statistics in inference mode, preventing both
parameter and running-statistic drift.

The synchronized NEMO2 runtime passed 285 focused IntrMotiv tests, and the
canonical/study suite passed 30 tests. The dedicated tests cover packed and
plain constant conditioning, gradients, selective DG loading, policy loading
with a fresh critic, DG freezing, and immutable legacy BatchNorm statistics.

Two earlier preflight namespaces were cancelled after they exposed concrete
compatibility issues. The first showed that old source checkpoints do not have
newer feature-centering buffers, so the legacy transfer inventory was narrowed
to tensors those checkpoints define. The second showed that freezing weights
alone still lets legacy BatchNorm running statistics drift, which led to the
module-level frozen-statistics fix. Neither failed namespace was reused.

The final preflight used jobs 8040513, 8040514, 8040515, 8040516, 8040517,
8040521, and 8040522. All completed with exit `0:0`, produced 2,031,616-frame
terminal checkpoints, and passed the seven-of-seven runtime audit. The audit
checked configuration, external-reward routing, frame counts, finite loss and
gradient telemetry, readable source checkpoints, logged scope-specific loads,
and exact terminal DG equality for frozen arms. Its StudySpec SHA-256 is
`49978bf565e08a76cd37609b3e0210e6c568afe7bf868d49ceec6b06e162ef4f`.

## Production submission

The required print-only review generated 21 scripts and the canonical audit
reported `commands_match_study: true` and `workspace_paths_valid: true`. All
scripts carry `#SBATCH --time=32:00:00`.

The production submission directory is:

`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_fixed_reward_transfer_20260910/20260910T021828Z`

The jobs are 8040710 through 8040730, in StudySpec order. The submitted audit
reports 21 rows, 21 submitted job IDs, exact command matching, valid workspace
paths, and `submitted_complete: true`. All 21 jobs were running at the first
scheduler check.

## Analysis plan

Use the fixed-location episode score against environment frames. For every
transfer arm, compare its three matched-seed curves with scratch using:

1. time to predefined score thresholds chosen without consulting condition
   labels;
2. early learning-curve area over common frame windows;
3. score at fixed frame budgets with matched-seed differences;
4. terminal score to distinguish faster learning from a final-performance
   change.

Report individual seeds and the across-seed estimate. A promising transfer
result requires a consistent early advantage over scratch; one favorable seed
or higher throughput is insufficient.

## Reusable workflow lessons

The canonical StudySpec, launcher print-only output, `jobs.tsv`, and submitted
audit were the authoritative records. Exact source checkpoint hashes and an
ordinary-job 2M-frame preflight caught two issues that unit parsing could not.
For future weights-only transfer studies, test old-checkpoint tensor inventory
and normalization-state freezing explicitly before the production matrix.
When the training backend does not save a frame-zero checkpoint, combine
focused initialization tests and load logs with terminal frozen-tensor checks;
do not claim a tuned terminal checkpoint proves its exact initial state.
