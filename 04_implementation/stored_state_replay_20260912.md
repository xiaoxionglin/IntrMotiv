# Stored-state DDQN and auxiliary HER trial — 12 September 2026

The user approved trying batch-stored recurrent states instead of reconstructing
online/target representation histories. Stop and discuss any substantial new
infrastructure requirement before building it. This supersedes the earlier exact
representation reconstruction requirement for this explicitly labeled trial; it
does not retroactively qualify the earlier production matrix.

Start from the original decoder-only Sample Factory integration at
`/tmp/intrmotiv_decoder_only_20260912`, not the sequence of performance candidates.
Candidate: `/tmp/intrmotiv_stored_state_20260912`.
Flag: `--controller_replay_state=stored`; default `reconstruct` remains unchanged.

## Contract

- Retain the action-time, goal-independent worker output through SF's existing
  extra-policy-output buffers and owned physical replay. Do not retain observations
  in this replay mode; the original fresh-data DG learner still receives them.
- Feed the stored worker state and actual command directly to the existing
  TargetFiLMDecoder and online/target Q heads. Main rewards are the original
  fresh-data rewards, including exploration rewards and temporal magnitude.
- HER changes only the decoder subgoal, auxiliary reward and auxiliary terminal
  mask. Reuse the parent's extracted `worker_reward_from_magnitude` formula and
  the fresh transition's recorded reward magnitude. Preserve independent HER RNG
  and additional update budget; main positions are never replaced.
- Use stored canonical activity for future achievements within the remaining
  option budget. Exclude already-achieved goals. Retain structural-generation
  rejection, physical ordering and actual successor commands.
- Certified terminal observations receive one batched DG-label evaluation under
  the published learner snapshot during ingestion, before fresh DG learning.
  Record that publication separately. No recurrent history is reconstructed, and
  reset observations cannot serve as terminal successors. Main and auxiliary
  bootstrap stop at physical termination/truncation.
- Both Q networks consume the same behavior-time state. This intentionally accepts
  representation lag and does not claim representation-snapshot equivalence.
  Replay gradients reach decoder/Q parameters only; DG trains through its original
  fresh-data objective. Require STOP and `dg_goal_input=none` explicitly.
- Use ordinary SF actor memory continuation across publications in stored mode;
  skip the custom publication-history reconstruction. PPO remains unchanged.
- Existing reconstruction checkpoints cannot initialize this trial: they did not
  retain worker outputs. Fail clearly on a replay-mode checkpoint mismatch and
  start the trial fresh. Stored-state checkpoints preserve state and terminal
  provenance and start new physical episodes on restart.

## Qualification

387 local runtime tests pass, including six new stored-replay tests covering
frozen-reference Q/reward parity, no encoder/core replay calls, gradient ownership,
main/auxiliary terminal targets, separate sampling RNG and checkpoint retention.
This is not an end-to-end throughput qualification. A bounded fresh SF trial is
required before reporting throughput or defining a new production release.

Earlier throughput candidates and their Slurm comparisons are historical
performance diagnostics. They do not constitute qualification for stored-state
training. The older production heartbeat was paused after explicit user approval. Its
first pause attempt was rejected by automatic approval review; the approved
second attempt succeeded. Do not resume it against the superseded contract.

## Reusable lesson

Separate the off-policy algorithm from the representation-freshness assumption.
Stored-state replay needs neither a new recurrent core nor a new HER engine.
Keep the approximation explicit: it learns a decoder over behavior-time states,
not end-to-end recurrent gradients through the generating history. Preserve
terminal provenance and never infer a post-observation state from SF's pre-input
state without checking temporal alignment.

## Fresh SF feasibility trial

Canonical StudySpec: `hpc_runs/studies/full_system_controller_stored_trial.study.json`.
Schema `intrmotiv/study/v1`, workflow `1.8.1`, SHA-256
`05a68eb91d17699580d65c47836b33c77e2dcb50ba37082758d7f3f89f3ff127`.
Four fresh runs: direct F16 and waypoint decoder F64, DDQN and DDQN+HER, seed 99,
262,144 frames each. Warm-up/cadence/main TD budget are unchanged. Remote runtime
and audit tests: 392 passed; 35 local workflow/audit tests passed. Canonical
print-only review and workspace/submission checks passed before submission.
Source snapshot: `hpc_runs/source_snapshots/controller_stored_replay_trial_20260912.patch`
with a sibling JSON hash manifest. The patch dry-runs cleanly against the original
decoder-only source. Remote source:
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_stored_state_20260912`.
Trial output remains in the allocated workspace under
`train_dir/intrmotiv_full_system_controller_stored_trial_20260912`.

### First trial results and telemetry correction

Jobs 8057485–8057488 all completed, in 4:28, 4:32, 5:15 and 5:17 respectively.
All four checkpoints reached 311,296 frames because SF drained in-flight work
past the 262,144-frame stop threshold. The W&B server reports all four finished,
`stored_state_replay=1`, 959 main updates each, and 113 DG metric keys per run.
HER learned 2,617 additional positions for direct and 7,368 for waypoint; the
plain DDQN arms learned zero auxiliary positions. This is an early-training
workload with relatively few eligible HER positions, not a worst-case full
256-auxiliary-position throughput measurement.

Post-warm-up end-to-end rates measured between the first 65,536-frame log point
and the first 278,528-frame log point, excluding startup and shutdown-only log
repeats:

| Architecture | DDQN FPS | DDQN+HER FPS |
|---|---:|---:|
| Direct F16 | 1,774.93 | 1,703.94 |
| Waypoint decoder F64 | 1,419.95 | 1,374.13 |

Canonical runtime audit job 8057490 found the first trial lacked the declared
262,144-frame place-field artifact: the inherited 100,000-observation window
cannot fill in this short trial. This is a trial configuration error, not a
missing telemetry backend. Do not label the first trial as fully qualified.

R2 uses exactly the same learner, a 32,768-observation artifact window and
131,072-frame scalar cadence. It starts fresh in a separate namespace.
Study: `hpc_runs/studies/full_system_controller_stored_trial_r2.study.json`;
SHA `8272c2097863e16fecd0417fca886d842a16745a81057b4e6f95d2227bb1cc4c`.
Schema/workflow remain `intrmotiv/study/v1` / `1.8.1`.

The reusable auditor now consumes the StudySpec's declared spatial targets,
validates stored worker tensors and certified terminal-label publication, and
requires zero custom actor rebuilds for stored mode. Reconstruction mode retains
its original rebuild and terminal-observation checks. Six controller-audit and
30 workflow tests pass (36 total); runtime code is unchanged for R2.

Reusable short-trial rule: compare the artifact window in observations against
the horizon divided by frameskip, as well as checking snapshot/scalar cadence.
The original 100k-observation window remains appropriate for longer studies.

R2 submitted jobs: 8057491 direct DDQN, 8057492 direct HER, 8057493 waypoint
DDQN, 8057494 waypoint HER. Dependent canonical audit: 8057495. Its immutable
source is `SF_hipposlam_controller_stored_state_r2_20260912`; only the trial
configuration and compatible audit checks differ from the original trial source.
Canonical print-only and submitted audits pass for the R2 SHA above.

The first trial's completed controller audit confirmed exactly 245,504 main TD
positions, 38 fresh DG optimizer steps and 19 graph batches in every arm, zero
update debt, finite losses, learned DG projection, unchanged fixed trunk,
certified terminal labels and zero actor-history rebuilds. Its only failed gate
was the missing short-trial spatial snapshot. End-to-end checkpoint restart and
longer-horizon learning quality are still separate qualification work; the local
stored-state checkpoint round-trip test alone does not establish those results.

## R2 completed qualification

Jobs 8057491–8057494 and audit 8057495 completed successfully. The canonical
runtime gate passes all four runs, including the declared place-field snapshot.
Each checkpoint contains 311,296 frames (the 262,144-frame stopping threshold
plus SF's in-flight collection), 959 main updates, 245,504 main TD positions,
38 fresh DG steps and 19 graph batches. Update debt, custom actor rebuilds and
actor version failures are zero. HER adds 3,357 direct and 6,586 waypoint TD
positions. W&B confirms all four runs finished with stored-state mode enabled,
959 main updates and 113 DG metric keys each.

Measured post-warm-up throughput from distinct runner frame timestamps,
65,536–278,528 frames:

| Architecture | DDQN FPS | DDQN + auxiliary HER FPS |
| --- | ---: | ---: |
| Direct F16 | 1,704 | 1,638 |
| Waypoint decoder F64 | 1,420 | 1,420 |

These short trials exercise relatively few eligible HER examples; they do not
measure the worst-case full auxiliary budget or establish long-horizon learning
quality. Stored representations remain behavior-time approximations, and replay
trains the decoder/Q heads, not the recurrent representation. A real process
restart qualification remains outstanding. The original production automation
remains paused with user approval; this trial does not authorize silently
restoring the superseded reconstruction production contract.

Evidence is archived under
`06_experiments/data/intrmotiv_full_system_controller_20260912/stored_state/`:
`controller_stored_trial_r2_gate.json`,
`controller_stored_trial_r2_online_result.json`, and
`controller_stored_trial_r2_submission_audit.json`.

Reusable lesson: reuse SF's batched action-time worker outputs when accepting
representation lag. A thin decoder replay adapter removed the dominant history
reconstruction cost without another transport or learner framework. For short
qualification runs, size the existing spatial observation window to fit the
available decisions before submission; the canonical audit and W&B server
summaries are the authoritative completion checks.

## Online deployment after user request

The user requested putting the current implementation online and checking for
obvious obstacles. Deploy the passing stored-state implementation unchanged;
the proposed current-DG finite-window reconstruction is not part of this run.

Study `hpc_runs/studies/full_system_controller_stored_online.study.json`:
`intrmotiv/study/v1`, workflow `1.8.1`, SHA
`d0059468b4703790ec58f44f565ec4e60288b0a7498e4e46b4f3675949e22978`.
Four fresh seed-99 runs, direct F16 and waypoint decoder F64, each DDQN/HER,
with a 2M-frame horizon. Restore the ordinary 100,000-observation spatial window
and require 1M/2M snapshots. Immutable source:
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_stored_online_20260912`.

All 36 focused deployment/workflow/audit tests passed. Canonical print-only
review matched the StudySpec and verified all output/cache/temp paths under
allocated workspace storage. Training jobs 8058060–8058063 were submitted and
entered RUNNING; dependent runtime audit 8058064 checks the completed 2M runs.
The previous production automation remains paused.

No immediate correctness blocker was found. Two material limitations remain:
behavior-time DG identity/feature drift is accepted by this baseline, and the
main sampler shuffles the full replay key list per update, so its cost grows
until the 200,000-decision capacity is reached. The longer run tests capacity
and epsilon annealing; short-trial throughput is not a saturated-buffer promise.
Checkpoint size/save time also grows with stored worker tensors. Process-restart
qualification remains distinct from successful saving and fresh startup.

Live startup verification: all four runs advanced to 294,912–409,600 logged
frames without logged exceptions. Recent 60-second throughput was 1,365–1,638
FPS. W&B server summaries confirmed finite main losses, 831–1,215 main updates,
zero update debt, continuing fresh DG/graph updates, and auxiliary positions
5,131 direct / 1,754 waypoint. Early zero auxiliary counts were transient lack
of eligible future achievements, not a disabled HER arm. These are asynchronous
live summaries, so differing counts at this wall time are not a budget mismatch.
Evidence: `controller_stored_online_startup.json` in the archived stored-state
folder. Full-capacity performance, 1M/2M spatial gates and long-run control quality
remain pending; do not infer success from the startup check alone.

Efficient follow-up: use the canonical jobs.tsv and W&B group
`intrmotiv_full_system_controller_stored_online_20260912`, then inspect dependent
audit 8058064 and `analysis/controller_stored_online_gate.json`. No runtime
patch was necessary for this deployment; keep proposed sampler improvements
separate until full-buffer measurements establish the bottleneck.

## Authorized production qualification and release follow-up

The user now requests auditing current progress and bringing production online
when possible. This authorizes a new stored-state production release after its
own gates; it does not revive the superseded reconstruction implementation.

At approximately 1M frames, replay reached its 200,000-decision capacity in all
arms with finite losses and zero update debt. Five-minute throughput was about
550–1,150 FPS across the four arms; short-trial throughput overestimated sustained
performance. Full-key-list shuffling remains a measured scaling concern, not a
reason to silently change the algorithm in this release. Stored worker-state
checkpoints can reach about 4 GB for F64. Existing `ControllerLearner.save_milestone`
already bounds periodic archives using `keep_checkpoints` while pinning canonical
frame targets. Production must retain this behavior and workspace-only paths.

Both online and qualification source files match every SHA in
`hpc_runs/source_snapshots/controller_stored_replay_trial_20260912.json`.
The original 387 runtime tests remain applicable; the unchanged deployed
qualification source additionally passed all 36 focused workflow/audit tests.

Remaining qualification uses isolated source
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_stored_qualification_20260912`:

- Fresh online DDQN/HER: jobs 8058060–8058063; 2M audit 8058064;
  `analysis/controller_stored_online_gate.json`.
- Fresh PPO on this exact source: jobs 8058067–8058068; audit 8058071;
  `analysis/controller_stored_ppo_gate.json`. Study
  `full_system_controller_stored_ppo.study.json`, SHA
  `7f31786ee7d549ae2d78a946ebc20d4ba1813a78486b1963a7ef5d145caaccde`.
- Exact GPU restore **passed**, job 8058070, all four 311,296-frame R2 checkpoints.
  It verifies online/target parameters, normalization buffers, optimizer, full
  replay rows, CPU/CUDA/Python/NumPy and replay/HER RNGs, counters and pending-tail
  discard. The initial helper job 8058069 failed before loading any model due to
  a wrong source path in the adapted shell script; correction required no runtime
  change. Immutable baselines and certificate:
  `analysis/restart_baselines/intrmotiv_full_system_controller_stored_restart_20260912/`.
- Real resumed-training checks: jobs 8058072–8058075; resume R2 into independent
  output directories, preserve original initial checkpoints, start new physical
  episodes, advance to the 442,368-frame threshold. Study
  `full_system_controller_stored_restart.study.json`, SHA
  `ed18afcad7819ad79e050522e9999d5789c4a5bdf369c6950f8abab240efe368`.
  Check `analysis/controller_stored_restart_gate.json`, bound to the immutable
  baselines and exact-reload certificate. Snapshot window is 32,768 observations;
  if the valid-observation ring cannot fill at the declared target, correct only
  that qualification configuration and repeat canonical review before resubmission.

All paths above with `analysis/` are relative to
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/`.
All new studies use schema `intrmotiv/study/v1`, workflow `1.8.1`.

After these three runtime gate documents pass, verify paired main-update
accounting at matched accepted-decision counts, positive HER positions, DG and
fixed-trunk checks, W&B delivery, spatial snapshots and physical-session restart.
Only then render the canonical production StudySpec: 18 fresh runs, direct F16
and waypoint decoder F64, PPO/DDQN/DDQN+HER, seeds 8/99/123. Both memories remain
goal-independent, STOP routing, stored replay. Inherit 300M frames and the parent
5M/25M/75M/150M/300M checkpoints/telemetry targets, 300-second regular saves,
1,800-second milestones and keep=8; use terminal telemetry for seeds 8 and 123.
Set `controller_preflight=False` in the qualified production study, never skip a
failed gate. Production gets its own immutable source and output/W&B namespace.
Use the established SF launcher with 48-hour L40S allocations, canonical
print-only review and submission audit. Verify real training and W&B before
claiming launch. The 300M horizon can exceed a Slurm allocation; checkpoint
continuation must use the existing resume workflow, not fresh reinitialization.

Reusable approach: run exact restore and short real resumes alongside long
preflights. Reuse existing GPU loader and canonical restart auditors; verify
adapted source paths before submitting helper jobs. Do not optimize or replace
the learner while qualifying an unchanged release.

Real restart audit job: **8058076**. The existing production heartbeat was
updated to this stored-state contract and reactivated every 15 minutes under the
new user authorization. It waits for the three qualification gates and then
performs canonical production review, submission and live verification without
another confirmation. It pauses after verified launch.

Qualification follow-up: **PPO audit 8058071 and real restart audit 8058076 both
passed**. Both PPO arms finished at 2,048,000 frames. All four resumed DDQN arms
finished at 491,520 frames with exactly 1,662 main updates, 425,472 main TD
positions, 60 fresh DG steps and 30 graph batches. All exact checkpoint-bound
reload checks passed; new physical sessions advanced correctly, with zero debt
and no actor version failures. HER retained/advanced auxiliary positions to
41,582 direct and 58,685 waypoint. Gate JSONs are archived in the stored-state
experiment data folder.

The fresh online gate remains pending: both plain DDQN arms completed, each with
7,743 main updates, 250 fresh DG steps and 125 graph batches. Direct HER crossed
2M and is finishing; waypoint HER reached 1,589,248 logged frames at roughly
500 FPS over five minutes. All four have finite main losses and zero update debt;
both HER arms have substantial positive auxiliary training. No new runtime fix
is indicated. Wait for audit 8058064 before production rendering/submission.
