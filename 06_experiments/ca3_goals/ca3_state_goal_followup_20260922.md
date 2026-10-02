# CA3 state-goal follow-up release

## Release state

The corrected four-cell CA3 state-goal factorial is in production on the G500
workstation. The production queue contains twelve fresh 300M-frame runs: four
are active on two RTX PRO 6000 Blackwell GPUs and eight are pending in the
resource-aware direct queue. Qualification checkpoints were not continued.

The scientific cells are fixed/EMA anchors crossed with dominant/unique-context
candidate recognition. Each cell uses one flat, seed-independent
`config.wandb_tags` value:

| Condition | Anchor mode | Candidate mode |
|---|---|---|
| `G500_CTX_FIXED_DOM_H32` | fixed | dominant |
| `G500_CTX_FIXED_UNIQUE_H32` | fixed | unique contextual |
| `G500_CTX_EMA_DOM_H32` | EMA | dominant |
| `G500_CTX_EMA_UNIQUE_H32` | EMA | unique contextual |

All runs use W&B project `SF_IntrMotiv_CA3StateGoals` and production group
`ca3_state_goal_followup_20260923_g500_production_v2`.

## Source snapshots

- Mature-replay throughput qualification: `2bad4082`.
- Matched encoder-credit vectorization: `a0426a95`.
- Replay authority snapshot: `16e705fc`.
- Production runtime: `9fbdcdc3`.
- General checkpoint-reload helper: `5d4f9833`.
- Terminal-transport-aware audit correction: `7ee88152`.
- Published branch: `codex/ca3-state-goal-followup-20260922`.
- Production source export:
  `/scratch/lin/IntrMotiv/src/SF_hipposlam_ca3_state_goal_followup_g500_9fbdcdc3`.

The production runtime remains pinned to the code that passed throughput and
runtime qualification. The later audit correction changes only release-gate
interpretation and is isolated in a separate source snapshot.

## G500 qualification

- StudySpec: `hpc_runs/studies/ca3_state_goal_followup_20260923_g500_preflight_v6.study.json`.
- Study SHA-256: `8c3987079dc44b69d7b66bcd414d3fa5085366f3454c7aaad005385a45a1f0e7`.
- Reviewed direct-manifest SHA-256: `8cd66db2a7647147fd1e971a5883726733beebaf9db3c5020b865728f6a369e0`.
- Output root:
  `/scratch/lin/IntrMotiv/train_dir/ca3_state_goal_followup_20260923_g500_preflight_v6`.
- Result: all four seed-99 runs completed beyond 2M frames without traceback,
  NaN, path leakage, or watchdog failure.

All four cells reached calibration readiness with 512 retained positive pairs,
registered all 64 anchor slots, exercised confirmation, contextual HER positive
and wrong-context paths, and exercised multi-activation recognition. Both EMA
cells recorded thousands of refinement comparisons. Zero accepted refinement
or activation would have remained a scientific result rather than a release
failure.

Each final checkpoint passed an independent exact learner restore covering
model and buffers, optimizer, counters, target controller, RNG state, replay,
calibration state, anchors, generations, EMA prototypes, and active masks. The
four 10k-decision privileged evaluators completed and emitted rollout-conditioned
alias diagnostics without feeding coordinates into training.

The first audit exposed a stale inherited requirement for a certified terminal
CA3 successor. This follow-up explicitly introduces no terminal raw-CA3
transport and rejects contextual terminal HER targets without a certified
successor. Commit `7ee88152` makes this requirement an explicit audit parameter:
older controller studies remain strict, while this follow-up still requires a
real terminal event and every other stored-replay invariant. Focused tests pass
on desktop and G500, and the corrected four-cell qualification audit passes
with zero errors.

## Throughput correction

Mature replay exposed two GPU synchronization defects that short probes missed:

1. matched encoder credit converted CUDA scalars inside the DG-event loop;
2. stored main replay read the selectable mask and anchor generation from CUDA
   separately for every commanded row.

The runtime now vectorizes matched-credit validation and snapshots the two
64-element authority vectors once per replay transaction. The snapshot retains
the exact active-mask and generation checks. Main replay example construction
fell from 4.86 seconds to roughly 0.03--0.06 seconds, and the matched four-cell
qualification reached 17.75k aggregate FPS in a fully aligned mature 60-second
window, 9.8% above the 16.16k historical reference. The sum of per-run mature
60-second medians was 16.11k, within 0.3% of that reference. The relevant
runtime fixes are `a0426a95` and `16e705fc`.

## G500 production

- StudySpec: `hpc_runs/studies/ca3_state_goal_followup_20260923_g500_production_v2.study.json`.
- Study SHA-256: `3eae7061195617daadf54669526923656f9e34cf3bdfa77a1a970f5558062a34`.
- Reviewed direct-manifest SHA-256: `192fd3d240b07455dbab3fcd1e15d2d71e52df1ce57623a40576245c55b3dcd9`.
- Output root:
  `/scratch/lin/IntrMotiv/train_dir/ca3_state_goal_followup_20260923_g500_production_v2`.
- Queue geometry: four slots `[0, 1, 0, 1]`, 32 workers, 8 environments per
  worker, 8 worker splits, one epoch, batch size 2,048, two minibatches, and
  rollout/recurrence 64.
- Runtime placement: `device=gpu` puts the learner and model updates on GPU;
  Sample Factory actor workers remain CPU-side and `actor_worker_gpus=[]`.

At the first post-launch check, four jobs were running and eight were pending.
The live runs had accumulated 1,900,544 aggregate frames with no traceback. The
sum of their first complete 60-second FPS values was approximately 16.38k while
the fourth run was still warming up. After every live run crossed 2.5M frames,
each reported 4,096 FPS over 60 seconds: 16,384 aggregate FPS, above the 16,163
reference, with no runtime, traceback, or CUDA-memory error. `jobs.tsv`,
`submission.json`, submission audit, source commit, StudySpec SHA, manifest SHA,
and the mature throughput audit are preserved in the production output root.

The earlier NEMO CPU and L40S qualifications remain separate historical tracks;
this workstation release neither resumes nor mutates their jobs or namespaces.

## Reusable lessons

- Measure throughput after contextual activation and replay maturity. Short
  startup probes do not exercise event-density-dependent synchronization.
- Treat Python scalar conversion from CUDA tensors inside per-event or per-row
  loops as a synchronization defect. Snapshot small immutable authority vectors
  once per transaction without weakening eligibility checks.
- Exact reload of 200k replay rows represented as Python objects takes about
  24 minutes per 4 GB checkpoint. Start independent certificates as cells finish;
  a future format should store replay in columnar tensor blocks.
- Direct evaluator commands must pin the qualified source root in `PYTHONPATH`
  and create isolated `TMPDIR` directories before DMLab starts.
- Release gates must declare optional transport contracts explicitly. A study
  that intentionally rejects uncertified terminal targets must not inherit a
  requirement to synthesize the forbidden successor state.
