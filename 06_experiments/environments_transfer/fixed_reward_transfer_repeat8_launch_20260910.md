# Repeat-8 fixed-reward transfer replacement

## Status

The previous repeat-4 transfer batch, jobs 8040710–8040730, was cancelled at
the user's request. Slurm accounting confirms all 21 cancellations. Existing
checkpoints and logs are retained. Unrelated algorithm-search jobs were not
targeted.

The replacement seven-cell, three-seed study is validated. All seven ordinary
preflight jobs completed with exit `0:0` and passed the full runtime audit.
Production jobs 8047305–8047325 are submitted. The first startup audit found
18 running and three queued; every started job logged five actions and repeat 8
without a startup traceback.

## Controlled change

The only production training-argument change is `env_frameskip=4` to
`env_frameskip=8`; run/output names and the W&B group use a new namespace.
The five-action interface, source checkpoint files, transfer scopes, constant
task conditioning, normalization behavior, learning settings, seeds, and
108,000-second training budget remain identical. A focused test compares every
expanded old/new command with only repeat and namespace differences excluded.

The 21 runs are generated from the StudySpec: scratch and three transfer
scopes (frozen DG, tuned DG, policy) for SCR and SAT, each at seeds 42, 1234,
and 9999. A new repeat-8 scratch control is included. The previous repeat-4
scratch runs remain secondary timing references, not the matched control.

This isolates the timing change. It does not claim that the constant task
vector is equivalent to the source waypoint selector, or that frozen and
trainable normalization paths implement identical forward functions.

## Provenance

- Production: `hpc_runs/studies/fixed_reward_transfer_repeat8.study.json`;
  schema `intrmotiv/study/v1`, workflow 1.5.0; SHA-256
  `99910170169d07d10d584d0aa645b3993189daaa9dbe78ba28b9ad1e2c74b56d`.
- Preflight: `hpc_runs/studies/fixed_reward_transfer_repeat8_preflight.study.json`;
  schema `intrmotiv/study/v1`, workflow 1.5.0; SHA-256
  `12ec5ef81e177d8e6a25086e05f7ba61ca2bdec0ae565b14bfe4dbd3ced905f5`.
- SCR source SHA-256:
  `ec95b4319f17fa1d8c2c10b0a7778df2932a4f710a8c615f06dc95930a56da20`.
- SAT source SHA-256:
  `7b3275e28ef78df68d523d6203925fbb6ee01c0edd2bc380bcbc54107a43b3fb`.

Bulk artifacts are under
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/`.
The lightweight source-interface audit and rendered production commands are
under `analysis/fixed_reward_transfer_repeat8_deployment_20260910/` there.

## Validation

NEMO2 passed 33 canonical/study tests and 8 focused transfer/action tests.
All 28 preflight/production commands parsed through the real training entry
point. Actual source configs agree on repeat, action flags, memory settings,
visual encoder, sensor settings, and conditioning module. Source checkpoint
hashes agree with the original study.

The runtime auditor now derives frame repeat from the declared run and can
reject source/destination interface differences. Tests specifically reject
repeat, action-set, and memory-horizon mismatches. This is a study-adapter
extension; the canonical StudySpec schema and package version are unchanged.

All seven real preflight environments logged `Discrete(5), frameskip=8` and
expected transfer-load counts without startup exceptions. Preflight retention
is raised to 100 checkpoints to preserve initialization evidence throughout
the bounded 2M-frame run.

All seven frame-zero checkpoints were copied outside checkpoint rotation to
the preflight submission's `initialization/` directory. Its `audit.json`
confirms zero environment/train steps, a uniform task vector, exact equality
of all selected tensors (4 for DG, 11 for policy), and fresh critics in every
transfer arm. This also confirms that the earlier missing initial checkpoints
were a retention problem, not an absence of frame-zero saving in this backend.

## Submission records

Preflight submitted directory:
`_slurm/intrmotiv_fixed_reward_transfer_repeat8_preflight_20260910/20260910T115629Z`.
Jobs: 8046996–8047002. The submitted audit confirms exactly seven matching
commands, unique job IDs, and workspace-valid paths.
All seven saved 2,064,384-frame terminal checkpoints, with recorded telemetry
through 2,031,616 frames. The runtime audit passed 7/7 with no errors, including
initialization, source-interface matching, finite learning metrics, and exact
terminal equality of frozen DG tensors. Slurm elapsed times were 16:43–23:37.

Production print-only directory:
`_slurm/intrmotiv_fixed_reward_transfer_repeat8_20260910/20260910T115736Z`.
The audit confirms 21 matching commands and workspace-valid paths. All 21
generated scripts request 32 hours to accommodate 30 hours of training and
shutdown.

Production submitted directory:
`_slurm/intrmotiv_fixed_reward_transfer_repeat8_20260910/20260910T122201Z`.
Jobs: 8047305–8047325. The submitted canonical audit confirms all 21 exact
commands, unique job IDs, and workspace-valid paths, with the production
fingerprint above. `jobs.tsv` is the authoritative job-to-condition mapping.
`startup_snapshot.json` records scheduler status and available initial
checkpoints preserved under `initialization/` before checkpoint rotation.
Production outcome and learning-speed comparisons remain pending.

## Reusable lesson

Actual source configs, exact expanded-command comparisons, and runtime
environment logs are authoritative for transfer compatibility. Tensor shapes
alone cannot detect a changed action duration. Preserve a frame-zero artifact
before checkpoint rotation; do not silently drop initialization checks when
retention removes the evidence. During a short preflight, lightweight
scheduler/checkpoint snapshots avoid repeatedly loading all TensorBoard data;
load full metrics once for the final runtime gate.
