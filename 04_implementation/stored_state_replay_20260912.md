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
training. The existing production heartbeat still requires qualification and
must read this revised scientific contract through the active experiment record.

## Reusable lesson

Separate the off-policy algorithm from the representation-freshness assumption.
Stored-state replay needs neither a new recurrent core nor a new HER engine.
Keep the approximation explicit: it learns a decoder over behavior-time states,
not end-to-end recurrent gradients through the generating history. Preserve
terminal provenance and never infer a post-observation state from SF's pre-input
state without checking temporal alignment.
