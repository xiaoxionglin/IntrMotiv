# Full-system IntrMotiv controller integration — 12 September 2026

**Status: implementation in progress; not a qualified DDQN training release.**
No production run was started from this integration. The original running source
and its jobs are unchanged. The learner factory currently rejects non-PPO modes
rather than silently train a DDQN actor with the PPO learner.

## Scope and provenance

The intervention is controller learning only. All three historical parents are
included: `DGC_DIRECT_WORKER_F16_S99`, `DGC_WAYPOINT_DG_F64_S8`, and
`DGC_WAYPOINT_DG_F64_S99`, at 25,001,984 frames. These are a historical
compatibility shortlist, not a new performance ranking.

The source is the actual NEMO2 checkout at
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam`, revision
`636be3a5862529bfb510b3e52e2f41f8cae9e78d`, including its tracked modifications
and untracked goal-write implementation. The isolated implementation checkout is
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_20260912`, branch
`codex/full-system-controller`. This is not a revert to public master.

[Evidence](data/intrmotiv_full_system_controller_20260912/) contains complete
resolved parent configs, verified checkpoint hashes, source-file hashes, per-parent
preservation manifests, the implementation patch, and test results. Current
source provenance is recorded; exact launch-time source equivalence remains a
separate provenance check. Configuration replication and checkpoint continuation
must remain distinct initialization modes.

## Objective decision

The user approved **a separate auxiliary HER head**, sharing original worker
features with the acting main DDQN head. Main Q-values select actions and estimate
the parent's continuing return. Auxiliary Q-values never select actions or
replace the main Bellman target. Shared-feature gradients can still interact;
this is an explicit experimental factor, not a claim of complete independence.

Source authority: `DistanceLearnerReward._calculate_internal_reward` and
`_prepare_batch` in the original `custom_learner.py`. The selected parents use
`hit_distance`, `gamma=0.99`, and `value_bootstrap=False`.

| Boundary/event | Main return | Auxiliary HER |
|---|---|---|
| Physical termination | Stop bootstrap | Stop bootstrap |
| Physical truncation for these parents | Stop bootstrap (`value_bootstrap=False`) | Stop bootstrap |
| Manager retargeting | Continue with the actual successor command/context | Does not substitute a new manager objective |
| Next waypoint | Continue with actual successor waypoint/context | Virtual worker task remains separately specified |
| Option timeout | Continue according to original physical return | Can end the explicitly bounded virtual task |
| Target/other-landmark arrival | Preserve original reward components and continuation | Virtual task's achievement/boundary must be recomputed consistently |
| SF rollout/replay chunk ending | Not a physical terminal; require valid continuation | Not a manufactured terminal |
| Missing successor observation | Reject required bootstrap example; never use reset observation | Same validity requirement |

The main target is $r_t+\gamma(1-d_t)Q^-(s_{t+1},\arg\max_aQ(s_{t+1},a))$.
There is no 0–1 Q clamp, three-goal vocabulary, or replacement first-arrival reward.
The auxiliary head includes a remaining-budget scalar and state-budget interactions
because its proposed objective is bounded. Its horizon construction, loss weighting,
and equal-TD-budget study configuration are **not yet qualified**; no fixed
64-decision task has been introduced into the acting controller.

## Implemented and checked

- Opt-in original actor/controller extension: main action-value head and optional
  auxiliary head over the actual decoder. PPO and shadow acting paths retain
  their original parameter initialization and action calculation.
- STOP/JOINT controller boundary reused; goal-write uses the actual worker view.
- Separate Bellman boundary functions; native SF categorical action distribution
  and sampling reused. Zero-epsilon/masked-action entropy stays finite.
- Original DG auxiliary loss block extracted into one shared method without
  copying its formulas. Fresh optimization scheduling is not replaced.
- Existing no-extra-encoder-loss telemetry bug fixed: disabled loss components
  now have zero telemetry instead of undefined local variables.
- Whole-model snapshots include DG, write modulation, normalization, and graph
  buffers. Differentiable replay reaches online parameters through an isolated
  model with cloned snapshot buffers. Version mismatch is rejected; replay does
  not advance the parent's RNG or update its live caches/statistics/graph.
- Reconstruction through original preprocessing, encoder, packed core, and decoder
  for the selected finite simple-core families. Complete worker histories are
  rebuilt separately under online/target parameters. A packed call retains
  gradients through earlier goal writes; repeated single-step calls would detach
  those histories in the original core.
- Per-config delta validation rejects unclassified changes, including freezing DG,
  reducing its vocabulary, changing manager families, or misspelled new options.

The history operation is a component, not a finished ingress or actor-publication
system. Physical stream/episode identity and snapshot-consistent reward/label
construction must be supplied and checked by the pending integration.

## Validation evidence and remaining gates

Local full IntrMotiv suite: **318 passed**. The original model parity audit also
passed for all three parents using their actual eight-action navigation vocabulary,
real pretrained encoder, DG, core, and decoder. It compares all state tensors,
initialization RNG, and outputs over five synthetic-observation steps in two
streams. It also checks differentiable replay through each actual model: gradients
reach its original decoder while source state tensors remain unchanged. It covers
default PPO and shadow-head initialization. This is not a
DMLab rollout, full learner-update parity test, or checkpoint-continuation test.

Remaining required work:

1. Native SF physical replay ingress, terminal provenance, and interaction/TD counters.
2. Snapshot-consistent original reward/event reconstruction and HER sample construction.
3. Actor memory rebuilding at parameter publication, including stream identity and
   failure telemetry; the replay helper alone does not solve actor memory mixing.
4. Actual DDQN learner/optimizer ownership and scheduling, fresh DG-only forward path,
   target-update clock, checkpoint/resume state, and verified shadow isolation.
5. Full-parent checkpoint initialization with explicit new-parameter reporting.
6. Forced-action full-system preservation tests, bounded Slurm preflights, and
   canonical matched PPO/DDQN/DDQN+auxiliary-HER StudySpec after those gates pass.

Do not remove the launch guard or label the old frozen diagnostic as satisfying
these gates. No training-ready command is supplied for this unfinished release.

## Reusable experience

Use the actual runtime snapshot and per-parent config manifests first. Local Python
imports initially resolved the older editable checkout because the snapshot omitted
its parent package `__init__.py` files; restoring those package markers corrected
source resolution. The actual-model replay audit also caught integer instruction
channels being converted to floats by direct normalization; replay now reuses SF
`prepare_and_normalize_obs` for the original type/device restoration. One initial suite failure was a missing copied Lua environment
patch, not a controller regression; retrieving the original patch made the complete
suite runnable. Check imported module paths before trusting a local preservation test.

The authoritative checks are the full original IntrMotiv test directory, actual-model
parity against the saved pre-change actor source, exact checkpoint/source hashes,
and generated config deltas. Reuse these artifacts rather than re-inventorying the
historical batch scripts. Every subsequent claim should distinguish module tests,
full learner parity, environment preflight, and production qualification.
