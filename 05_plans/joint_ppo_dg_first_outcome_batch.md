# Joint PPO-to-DG and First-Outcome Control Batch

Status: the original eight-cell 5M seed-99 preflight passed. After review, the production curriculum was simplified to directed local successors. A two-cell 1M compatibility smoke (`7995912`–`7995913`) was submitted on 2026-09-06; production waits only for that smoke to show healthy learning telemetry.

## Scientific question

The study cleanly crosses two possible causes of failed target control: the worker objective ignores wrong first outcomes (`HIT` versus chance-centered `FIRST`), and PPO cannot reshape DG when CA3 is detached (`STOP` versus `JOINT`). Legacy one-hot and target-ID FiLM conditioning remain a third crossed factor. All other relevant choices are fixed: C15 ARR credit, legacy BatchNorm, immediate targets, balanced least-tested commands among directed passive first successors, MON retirement with zero replacements, 75M fresh training, and seeds 8/99/123.

## Implementation contract

- `--ppo_dg_gradient=stop|joint` changes only the CA3 boundary presented to the decoder. The frozen ResNet trunk remains outside the gradient path.
- `--hrl_control_outcome=target_hit|first_distinct` preserves HIT behavior or terminates at the first distinct exclusive landmark. If a source has `K` observed candidate successors, FIRST assigns `+q(d)` to the commanded outcome and `-q(d)/max(1,K-1)` to a wrong outcome. The behavior-time `K` is replayed, so later graph changes cannot alter the sampled objective.
- `--hrl_direct_target_selection=local_successor` restricts commands from source `j` to directed alternatives with observed passive first-successor evidence `passive_confidence[j,k] > 0`, then balances those candidates by decayed controlled-attempt evidence. It never routes or uses learned controllability for candidate selection.
- `Tctrl` is used only as a learned deadline/cost for a selected local edge; it never defines which observation is treated as the outcome.
- FIRST wrong outcomes and timeouts complete attempts; only correct outcomes add confidence. Free exploration updates neither controlled attempts nor confidence.
- The frozen evaluator reads the intervention protocol from the manifest sidecar, restricts each source to its observed directed passive successors, terminates on the first distinct exclusive outcome, and records the realized outcome and completion reason.

## Validation and artifacts

- Algorithm/evaluator suite on NEMO2: 233 passed.
- Standardized study suite on NEMO2: 30 passed.
- Preflight StudySpec SHA-256: `1d7db721017c8900b4dee7ff1b2aa1ebad92f6844276989406b0bf2aa54ea1a3`.
- Local-successor smoke StudySpec SHA-256: `d592db66497954afc2b7b0f98c5387a563ef103d671054068ba4aac5dfe07884`.
- Production StudySpec SHA-256: `2e3104c975188e7cddeb71bce8816c0f4f0d6eb96688c44e0ea2b7560b5447b5`.
- Preflight Slurm jobs: `7994175`–`7994182`.
- Workspace output root: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_dg_policy_gradient_first_outcome_preflight_20260906`.

The original preflight analyzer checked finite losses, replay agreement, one DG minibatch forward, STOP/JOINT gradient separation, ARR gradient activity, FIRST event activity, command entropy, observed-pair coverage, DG-row activity, and zero MON replacement. The local-successor smoke additionally checks candidate discovery, behavior-time candidate counts, finite signed FIRST rewards, correct STOP/JOINT norms, and continued DG-row activity before the 24 production jobs are released.
