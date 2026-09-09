# Navigation8 algorithm screen implementation

## Status

The six-cell navigation8 algorithm screen is implemented in the canonical
StudySpec workflow. The production matrix is ready but remains gated on six
submitted 2M-frame preflights.

- Production: 6 configurations × seeds 8, 99, and 123 = 18 runs at 300M
  simulator frames each.
- Preflight: one seed-99 run per configuration = 6 runs at 2M frames each.
- Action interface: eight ordered locomotion/yaw actions, no fire, frame repeat
  4.
- Original decision-based discount, CA3, recurrence, rollout, and goal/option
  horizons are unchanged. Their physical durations are therefore half those of
  the repeat-8 predecessor runs.
- Ground-truth position is explicitly excluded from policy input with
  `--with_pos_obs=False`; pose remains available only to monitoring.

Canonical production fingerprint:
`c435ddac609945336d1e42ca16ed0bcc8fd2d46be13eef167a0085b39b682094`.

Canonical preflight fingerprint:
`379522c86e5c23f89c7b63a5d30f522d93778ac0b5ad9a7f828612569ba81bce`.

## Runtime implementation

The authoritative NEMO2 runtime at git revision
`c684685f75b37ef8325f27623f54f9cc32e97186` received a minimal extension that
matches the historical reduced-action implementation:

1. `NAVIGATION_ACTION_SET` is defined beside `ACTION_SET` and
   `REDUCED_ACTION_SET`.
2. `--dmlab_navigation_action_set=True` selects it in the custom DMLab
   environment.
3. The set contains forward, backward, both strafes, pure yaw ±20, and forward
   plus yaw ±20, in that order.
4. The actor obtains eight logits from the resulting
   `gym.spaces.Discrete(8)`. Loading a five-action actor head into it fails on
   tensor shape mismatch.

Command validation exposed a pre-existing parser defect: Python
`type=bool` interprets the string `"False"` as true. Both the historical
reduced-action switch and the new navigation switch now use the existing
`str2bool` parser. Omitted flags and explicit true values retain their intended
behavior; explicit false values now work.

Deployed file hashes:

- `custom_params.py`: `14ab520456b3418b74eac3981ee0744aef32d93c9bf80963b7e7b91c6d22ccdd`
- `dmlab_env.py`: `bd906eea63bd0f526283e90d6dae88053b6d0c5c426d208fe233bf368702d7a0`
- `dmlab_gym.py`: `a75665a83046fcec7011d7744cc5fe148005be2ebbaa08da65108dcaf0e9e743`
- Complete deployed git diff: `44577c1d401fc6d99afdbd5f9607db6bf67fb87abad05b1d6deccf48b80b7359`

## Study and evaluation contract

The production specification is
`hpc_runs/studies/navigation8_algorithm_screen.study.json`; its thin launcher
adapter is `hpc_runs/navigation8_algorithm_screen.py`. A separate preflight
StudySpec and adapter provide the six-run 2M gate.

Tests compare the selected configurations with their historical source runs.
Every inherited argument must remain identical except the declared batch
boundaries: action flags, frame repeat, training cap, W&B destination, and
online-spatial milestones. The W_REF cells retain the exact shared reference
checkpoint at 75,038,720 frames and differ only in STOP versus JOINT DG
gradient routing.

Production online-spatial targets are 5M, 25M, 75M, 150M, and 300M frames. The
standard five-checkpoint seed-99 plus terminal seeds 8 and 123 protocol expands
to 42 offline field/trajectory evaluations. Persistent matched-command
interventions select only the six terminal W_REF checkpoints. The four
graph-controller families retain graph outcomes and diagnostics; their
graph-specific terminal controller probes must be run through the existing
graph evaluator after their checkpoints exist, rather than being mislabeled as
persistent-goal interventions.

The common-observation panel is pinned to the seed-99 `SCR_ARR_DIRS` trajectory
at 25M frames. The final analysis must report representation, control, and
exploration independently and retain pair-level directed costs and regional
command support.

## Validation and submission record

NEMO2 validation completed before submission:

- 36 focused action, study, and canonical workflow tests passed.
- Full IntrMotiv suite: 277 passed, 23 warnings.
- All six generated preflight commands parsed through the authoritative
  training entry point with navigation true, reduced/extended false, repeat 4,
  pose input false, and a 2M-frame cap.
- The frozen SCR reference checkpoint exists.
- Print-only review generated six scripts under
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_navigation8_algorithm_screen_preflight_20260909/20260909T163639Z`.
- Canonical print-only audit reported exact command agreement and valid
  workspace paths.

Submitted preflight directory:
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_navigation8_algorithm_screen_preflight_20260909/20260909T163801Z`.

Jobs `8035135`–`8035140` were submitted successfully and passed the canonical
post-submission audit. At the final check they were pending for resources, with
no scheduler or startup failure. Production submission remains blocked until
all six finish with exit 0 and runtime artifacts confirm the expected action
space, finite learning signals, correct frame accounting, and spatial snapshot
generation. `hpc_runs/audit_navigation8_algorithm_screen_preflight.py` performs
that fail-closed post-run audit from the submitted `jobs.tsv`.

## Reusable lesson

Parsing complete generated commands in the actual cluster runtime found a bug
that unit tests over action constants could not detect. Future action-interface
changes should test literal true and false CLI values, parse every StudySpec
cell through the production entry point, and then verify the selected action
count in a real environment preflight before launching long runs.
