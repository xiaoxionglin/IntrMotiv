# Joint PPO-to-DG and First-Outcome Control Batch

Status: eight-cell 5M seed-99 preflight submitted on 2026-09-06. Production is blocked on the declared runtime gates and has not been submitted.

## Scientific question

The study cleanly crosses two possible causes of failed target control: the worker objective ignores wrong first outcomes (`HIT` versus chance-centered `FIRST`), and PPO cannot reshape DG when CA3 is detached (`STOP` versus `JOINT`). Legacy one-hot and target-ID FiLM conditioning remain a third crossed factor. All other relevant choices are fixed: C15 ARR credit, legacy BatchNorm, immediate targets, balanced least-tested direct commands, MON retirement with zero replacements, 75M fresh training, and seeds 8/99/123.

## Implementation contract

- `--ppo_dg_gradient=stop|joint` changes only the CA3 boundary presented to the decoder. The frozen ResNet trunk remains outside the gradient path.
- `--hrl_control_outcome=target_hit|first_distinct` preserves HIT behavior or terminates at the first distinct exclusive landmark. FIRST assigns `+q(d)` to the commanded outcome and `-q(d)/15` to a wrong outcome.
- `--hrl_direct_target_selection=least_tested` commands only observed alternative landmarks and balances each source row by decayed controlled-attempt evidence. It never routes or uses graph topology for selection.
- FIRST wrong outcomes and timeouts complete attempts; only correct outcomes add confidence. Free exploration updates neither controlled attempts nor confidence.
- The frozen evaluator reads the intervention protocol from the manifest sidecar, restricts commands to observed alternatives, terminates on the first distinct exclusive outcome, and records the realized outcome and completion reason.

## Validation and artifacts

- Algorithm/evaluator suite on NEMO2: 233 passed.
- Standardized study suite on NEMO2: 30 passed.
- Preflight StudySpec SHA-256: `1d7db721017c8900b4dee7ff1b2aa1ebad92f6844276989406b0bf2aa54ea1a3`.
- Production StudySpec SHA-256: `e38e88dfeca86426b25a91ad63b0bbc361b4e2cec0ec4466eae5150b87419f86`.
- Preflight Slurm jobs: `7994175`–`7994182`.
- Workspace output root: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_dg_policy_gradient_first_outcome_preflight_20260906`.

The preflight analyzer checks finite losses, replay agreement, one DG minibatch forward, STOP/JOINT gradient separation, ARR gradient activity, FIRST event activity, command entropy, observed-pair coverage, DG-row activity, and zero MON replacement. Production submission is a separate manual action after the preflight report passes; there is no dependent production job.
