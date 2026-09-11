# IntrMotiv recurrent Double DQN / HER: implementation and batch record

The first executable stage tests whether the selected **existing representation**
supports a better local worker with persistent replay and genuine future-goal
HER. It does not yet implement the final adaptive-DG/manager architecture.
The proposed six-run comparison is DDQN versus DDQN+HER, at child seeds
8, 99, and 123, for 5M child frames each. There is no matched PPO learner here;
intact-parent evaluation is a separate reference. No PPO-superiority claim is
supported by this design.

## Source and exact parents

The live source was inspected at
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam`, commit
`636be3a5862529bfb510b3e52e2f41f8cae9e78d`, with existing tracked and untracked
modifications. Its tracked diff SHA-256 is
`3c95df7b321800cfc416ba8aee104da14c50bcf39687adbd31f8fbb57e9815a4`.
The untracked goal-write module was included in the scoped source snapshot;
a tracked diff alone is insufficient provenance. This is current-runtime
provenance, not proof of the exact source at parent training time.

All three exact 25,001,984-frame model checkpoints exist. Paths, hashes,
source seeds, and learner seeds are recorded in
[data/intrmotiv_ddqn_her_20260911/parent_manifest.json](data/intrmotiv_ddqn_her_20260911/parent_manifest.json).
The primary is `DGC_DIRECT_WORKER_F16_S99`; secondary parents are
`DGC_WAYPOINT_DG_F64_S8` and `DGC_WAYPOINT_DG_F64_S99`. The compact 25M NPZs
were not substituted for model checkpoints. Development goals 1/4/11 remain
explicitly post-hoc selected; they are not held-out discoveries.

Training runs from the independent source copy
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ddqn_her_20260911`.
The evaluator has a separate copy ending in `_ddqn_her_eval_20260911`.
Existing DG-capacity and CRL/L3P runs and their imported source were not changed.
The new runtime incorporates the reviewed graph-planning optimization; its
39 focused tests pass there.

## Causal links inspected and changes made

1. **Recognition semantics:** the selected `frontier_direct` parent uses
   exclusive positive landmark recognition, just like `frontier_waypoint`.
   Dominant-positive recognition belongs to legacy `visit_direct`. Inferring
   this rule from the word “direct” would silently change HER labels.
2. **Reset semantics:** DMLab returns a cached preterminal image after shutdown.
   The new patch explicitly marks it invalid. The vector helper separates
   final/reset observations and copies valid final data before reset. Replay
   stores missing final successors as `None` and excludes them from TD/HER;
   it never bootstraps from a reset image or copies the preceding feature as
   a fabricated successor.
3. **Recurrent HER:** physical stream and episode IDs, contiguous indices,
   logged goals and options, and a 71-step washout prefix define reconstruction.
   HER intervenes after the real prefix, holds one virtual command and budget,
   recomputes first arrival, and ends loss terms on the first hit. Online and
   target write-conditioned memories are rebuilt with their own parameters.
4. **Representation/control boundary:** exact source preprocessing, DG
   preactivation, depth bypass, and map-number context are preserved. The first
   encoded batch must exactly match the original actor head. The entire
   reference encoder, including normalization state, is frozen. Decoder/FiLM
   and the new Q head learn; source policy logits and critic values are excluded.
5. **Finite horizons:** neutral-initialized inputs expose option budget / 64
   and physical episode decision index / 1,800, without clipping. The latter
   is an input scale; it does not guess the engine's exact last decision.
   The source fixed-length Lua wrapper retains its 120-second timeout.
6. **Replay eligibility and cost:** uncompleted chunks do not qualify as episode
   ends. HER waits for genuine future context; unavailable future goals fall
   back to original-goal samples. Ring sampling uses constant-time slots instead
   of rebuilding a 200k-entry key list per sample. Prefix rebuilding advances
   memory only, avoiding thousands of unnecessary decoder calls per update.
7. **Preflight duration:** 250k frames cannot exercise the first target hard-copy
   at 16,384-decision warmup, one update / 64 decisions, and 1,000 updates / copy.
   The correctness gate is therefore 500k frames.
8. **Process startup:** the first preflight hit `AF_UNIX path too long` in
   `forkserver`. Its TMPDIR is now `/work/classic/fr_xl1014-train/tmp/ddqn_JOBID`,
   which leaves space for Python's socket suffix. The retry has a new namespace.

The implementation is in [hpc_runs/intrmotiv_offpolicy](../hpc_runs/intrmotiv_offpolicy/README.md).
Shared runtime changes are reviewable patches:
[terminal contract](../hpc_runs/source_snapshots/intrmotiv_ddqn_terminal_contract_20260911.patch)
and [matched evaluation adapter](../hpc_runs/source_snapshots/intrmotiv_ddqn_matched_evaluation_20260911.patch).

## Study and submission provenance

All studies use `intrmotiv/study/v1`, canonical workflow `1.8.0`, and the
existing Sample Factory Slurm launcher. Version 1.8.0 was synchronized and
its focused tests rerun in the isolated checkout; the active source was not
upgraded by this task.

| Study | Runs / child frames | SHA-256 |
| --- | --- | --- |
| preflight | 2 × 500k | `e39104e4e8beb5f3f96cc01b8072aa906ee826239fe661d17353681bac1eea52` |
| preflight2 | 2 × 500k | `e7de99e78d56c8d7cc2c51550dd79d5ef4f09fdab95ff4f9a0155970cab632c0` |
| frozen_pilot | 6 × 5M | `a9771c14657cf6e59c6e9d6358b96d4d53452951dfd4235112bc01324ea18bef` |

Specs are under `hpc_runs/studies/intrmotiv_ddqn_her_*.study.json`.
Rendered plans are retained in [data/intrmotiv_ddqn_her_20260911](data/intrmotiv_ddqn_her_20260911).
Every training/checkpoint/log/cache/replay destination is in the allocated
workspace. The source-only checkout remains in home.

Original preflight jobs **8055861 / 8055862** failed after parent loading and
conversion, before collector startup, due to the Unix-socket path. Their
artifacts remain under `intrmotiv_ddqn_her_preflight_20260911`.

Retry jobs **8055873 / 8055874** use
`intrmotiv_ddqn_her_preflight2_20260911`. Their submitted manifest passed the
canonical audit at
`.../train_dir/_slurm/intrmotiv_ddqn_her_preflight2_20260911/20260911T165031Z/jobs.tsv`.
Both arms produced finite TD updates, real arrivals, original-goal samples, and
physical resets. The HER arm also produced genuine future relabels. Full
runtime completion and calibrated control conclusions are recorded separately
below when available.

Frozen-parent qualification job **8055902** was submitted using
`submit_place_field_sweep.py` and its reviewed `--runner` extension, as one
ordinary job. Its workspace is
`.../train_dir/analysis/intrmotiv_ddqn_her_parent_20260911`.
The existing exact-start evaluator now accepts an optional worker callback,
common deadline, and registry filter; defaults retain the original behavior.
Six evaluator tests pass locally and in the isolated evaluator checkout,
including a deliberately wrong callback that reverses measured command lift.
This evaluates detector arrival and physical-start equality. Independent
spatial destination qualification remains necessary for claims about unique
places; perceptual aliases are not silently relabeled as navigation success.

## Validation and limitations

Sixteen learner/contract tests plus 30 canonical workflow tests pass locally
and in the separate production checkout, including explicit pose-sidecar
isolation and runtime-gate rejection tests. The optimized source passes
39 planner tests remotely. Six exact-start evaluator tests pass in both
locations. Tests cover disagreeing DDQN heads, finite budget and first arrival,
invalid terminal data, physical boundaries and eviction, replay RNG restoration,
washout reconstruction, separate target/write state, and a tiny two-destination
learning problem. The toy uses a smooth decoder: its initial tiny random ReLU
fixture collapsed to dead units and could not test the intended TD contract.
This is not evidence that the real pretrained decoder learns useful control.

The production-runtime branch intentionally rejects `dg_goal_input=write`
collection until actor-memory rebuilding after write updates is qualified.
It also does not implement adaptive DG, manager integration, a matched PPO
control, or exact resume. Checkpoints explicitly say
`warm_restart_requires_refill`; replay content is not included. Replay caches
frozen preactivations, so it cannot be repurposed for adaptive DG merely by
turning on an optimizer. An adaptive stage must retain trunk features and
reconstruct current working-DG traces while preserving reference labels.

Representation, exploration, and control therefore have explicit roles in this
first stage: reuse and measure the selected frozen representation; collect with
an epsilon-greedy fixed-command policy; train first-arrival control with/without
HER. Advancing to adaptive representation or planner integration requires
measured commanded control, not lower critic loss or more HER positives.

## Reusable workflow lessons

The authoritative evidence is the real checkpoint hash, source-head parity,
canonical StudySpec fingerprint, generated/submitted `jobs.tsv`, Slurm exit
state, metrics, and initial/terminal checkpoint state. Repository unit tests
alone did not expose the multiprocessing path limit. Include canonical study
fixtures when synchronizing tests: the first isolated workflow run failed
because its reference StudySpec was omitted, not because the package failed.

Next time, resolve source and checkpoint contracts once, calculate the
preflight length from the slowest cadence, bound process temporary paths,
and reuse the canonical discovery/submit/evaluator extensions. Preserve every
failed namespace. Evaluate learner-active throughput separately from initial
collection speed; do not interpret a cumulative average as steady-state FPS.

## Completed 500k runtime gate

Both retry jobs completed with `ExitCode=0:0`: DDQN in 9:55 and DDQN+HER in
11:16. Each processed **500,096 frames / 125,024 real decisions**, made
**1,696 learner updates and one target copy**, and excluded **64 invalid final
transitions** across two resets per collector. The frozen encoder checksums
remained unchanged, and the two arms' converted initialization hashes match.
The retained child checkpoints are at 0, 250,112, and 500,096 frames.

The [runtime audit](data/intrmotiv_ddqn_her_20260911/runtime_audit.json) is separate
from scientific qualification. Learner-active throughput, computed between
recorded active points rather than from the initial collection burst, was
**954 frames/s for DDQN** and **826 frames/s for HER**. Peak batch-process RSS
was 15,776,380 KiB and 14,443,456 KiB respectively (Slurm accounting; not a
standalone replay-memory measurement). HER used 8,581 relabeled sequences,
31.62% of sampled sequences; the requested 80% is reduced by eligible-future
event availability and legitimate original-goal fallback.

Final observed TD losses were 0.0091 and 0.0378, but HER had 96.8% out-of-range
selected-action Q predictions in the final logged minibatch, versus 14.0% for
DDQN. This diagnostic does not reveal the magnitude or sign and cannot by
itself diagnose divergence. The next-source instrumentation adds Q min/max/mean,
actual original/HER loss-position counts, and collection/learning timing.
These logging additions do not alter the targets or losses and were not patched
into the completed jobs mid-run. A small exact-start parent/child panel is
being used to inspect real action choices and Q magnitudes before release.

The machine-readable [config delta](data/intrmotiv_ddqn_her_20260911/config_delta.json)
records intentional differences, including 32 collectors versus 64 source
actor environments. The complete system is still in the controlled frozen
stage. Passing the runtime gate is not evidence for adaptive-DG or manager
integration, nor for PPO superiority.

## Matched smoke and production-diagnostic release

Ordinary matched-smoke jobs **8056008 / 8056011 / 8056012** all completed with
exit `0:0`. They reused one exactly reproduced physical start and three
commands, with the same passive source inventory. The
[retained summaries and trial rows](data/intrmotiv_ddqn_her_20260911/matched_smoke/raw)
show:

| Policy | Commanded detector arrivals | Paired command lift | Q range over evaluated actions |
| --- | --- | --- | --- |
| Intact parent | 1 / 3 | 0 | Not a Q model |
| DDQN at 500,096 child frames | 0 / 3 | 0 | −0.248 to 0.345 |
| DDQN+HER at 500,096 child frames | 0 / 3 | 0 | −0.817 to 0.292 |

HER's value-range issue is primarily negative predictions on this panel,
not an exploding positive value scale. Both children selected the same initial
action across commands at this start. One start cannot establish general
failure or success; it establishes that useful deliberate control has **not**
yet been demonstrated. The 500k preflight also ends while epsilon is still
0.55, before its 1M-frame annealing endpoint. The broader parent qualification
job 8055902 is separate from this smoke.

Proceeding to 5M is a **controlled learning experiment**, not promotion of an
established successful controller. The six-run diagnostic was submitted as
**8056085–8056090** after the runtime gate, checkpoint-load/command smoke,
updated-source tests, repeated print-only review, and canonical submitted audit.
The ordered mapping is DDQN seeds 8/99/123, then DDQN+HER seeds 8/99/123.
The source is
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ddqn_her_production_20260911`.
Its new instrumentation logs Q extrema/mean, actual valid original/HER loss
positions, and collection/learning time. No loss or target changes were made
between the passing preflight and production diagnostic.

The [submitted audit](data/intrmotiv_ddqn_her_20260911/production_submission_audit.json)
confirms all six exact commands and workspace paths. Source hashes are retained
in [production_source_files.sha256](data/intrmotiv_ddqn_her_20260911/production_source_files.sha256).
The submission directory is
`.../train_dir/_slurm/intrmotiv_ddqn_her_frozen_pilot_20260911/20260911T171402Z/`.
Study fingerprint remains
`a9771c14657cf6e59c6e9d6358b96d4d53452951dfd4235112bc01324ea18bef`.

The next scientific decision must use all three child seeds at common frames,
exact-start commanded arrival, failure-inclusive first-arrival cost, and
independent spatial destination qualification. The current manifest runner is
explicitly a one-start smoke adapter; use the evaluator's declared larger panel
for scientific assessment. Do not treat the parent's preserved-head advantage,
the small development panel, nominal HER fraction, or critic-loss reduction as
a comparison of mature controllers. Do not enable online DG updates on the
cached-preactivation replay contract, import PPO reliability into a new worker,
or enable a waypoint manager before real controller-specific revalidation.

At the final startup inspection all six production-diagnostic jobs were RUNNING,
at approximately 122k–149k frames with finite TD updates. The new timing records
showed roughly 0.123–0.127 seconds for prefix rebuilding plus batch preparation,
versus 0.020–0.031 seconds for the learner update itself. This identifies a
measured next optimization target: batch prefix reconstruction across sequences
and verify numerical equivalence before another batch. Do not change these
running jobs to chase throughput. The broader parent evaluation 8055902 remained
running at that inspection.

[Preflight receipts](data/intrmotiv_ddqn_her_20260911/preflight_receipts) retain
conversion key inventories, exact run configs, metrics, and terminal gate
records. The parent manifest's qualification field records discovery-time
status; the matched-smoke results above supersede that initial status for the
primary parent.
