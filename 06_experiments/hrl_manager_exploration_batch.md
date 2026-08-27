# HRL Manager Exploration Batch

## Purpose

This batch tests whether an explicit exploration manager action can reduce the
sparse-reward failure of fixed/global HRL while retaining successful DG-target
navigation. It keeps the two most credible HRL structural-diversity conditions
from the running 2026-08-26 batch:

- `CTRL_X0_O1`: encouragement plus orthogonal recruitment;
- `CTRL_X1_O1`: the same condition with CA3 temporal exclusion coefficient 1.

The old running batch remains the reference for the original shorter deadline.

## Manager Exploration Option

The option is enabled by:

```text
--hrl_exploration_mode=True
--hrl_manager_exploration_probability=p
--hrl_exploration_horizon=64
```

At each option reset, the fixed manager selects exploration with probability
`p`. A missed DG-target deadline forces exploration regardless of `p`. After
64 exploration decisions, the manager returns to novelty-first DG-target
selection, subject to another probabilistic exploration choice.

Exploration is stored as reserved target ID `F` in compact RNN option state and
is supplied to the worker as the existing all-zero target vector. It therefore
adds no decoder input dimension. PPO replay uses the stored manager action and
does not reconstruct it from the learner's newer graph or random generator.

The worker reward is mode-dependent:

```text
DG-target option: hit_distance reward on target completion
exploration option: existing flat temporal-distance decoder reward
```

The default is off, and exploration is restricted to
`hrl_graph_memory=policy_buffer` because that mode teacher-forces stored targets
during replay.

## Increased Deadlines

All new conditions, including the no-exploration deadline controls, use:

```text
--hrl_timeout_margin_ratio=1.0
--hrl_timeout_margin_steps=8
--hrl_bootstrap_horizon=96
```

For a learned controllability time `T`, the DG-target deadline is now:

```text
ceil(2 * T) + 8
```

The exploration deadline is independently fixed at 64 decisions.

## Production Matrix

W&B project: `SF_IntrMotiv_HRLManagerExploration`

Batch: `intrmotiv_hrl_manager_exploration_20260826`

| Factor | Values |
| --- | --- |
| Structural condition | `CTRL_X0_O1`, `CTRL_X1_O1` |
| Manager setting | deadline control, forced-only, `p=0.10`, `p=0.25` |
| Seed | 8, 99, 123 |

This is `2 x 4 x 3 = 24` jobs. Every job runs 100M environment frames with
frozen ImageNet-pretrained `layer2_resnet18`, `F=16`, `R=8`, `L=64`, threshold
2.43, encouragement, batch loss, simultaneous updates, one policy, no PBT, 32
workers times two environments, and the existing CPU Slurm profile.

Production jobs are `7873418` through `7873442` (the scheduler skipped
`7873441`). The submission manifest is under:

```text
/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/
intrmotiv_hrl_manager_exploration_20260826/20260826T165527Z/
```

All training output, W&B staging, Slurm logs, caches, and temporary files point
into `/work/classic/fr_xl1014-train`.

The complete IntrMotiv suite passed after implementation: `74 passed` with
five pre-existing dependency/deprecation warnings. Immediately after
submission all 24 production jobs were running, all expected log files were
present, and no traceback or runtime exception was found.

## Preflight

Jobs `7873413` and `7873414` completed 524,288 frames with exit code zero.

| Check | Forced-only | `p=1` stress test |
| --- | ---: | ---: |
| Active option fraction | 0.9985 | 0.9980 |
| Exploration mode fraction | 0.0459 | 0.9980 |
| Exploration selection fraction | 0.1087 | 1.0000 |
| Forced fraction of exploration selections | 1.0000 | 0.0000 |
| Exploration deadline | 64 | 64 |
| Target deadline mean | 85.5 | not applicable |
| Exploration reward mean | 0.0479 | 0.3604 |
| Exploration reward nonzero fraction | 0.0213 | 0.0597 |

The forced-only run confirms that target timeouts activate exploration even at
`p=0`. The stress test confirms sustained exploration conditioning and reward.
Target-only option success excludes exploration horizon completions.

## Primary Analysis

Compare each exploration setting first against the increased-deadline control
within the same structural condition and seed, then compare the deadline
control to the original running batch.

Primary behavioral outcomes:

- coverage AUC, unique cells, and occupancy entropy;
- target-only option success and hit rate;
- exploration occupancy and nonzero reward fraction;
- known-edge fraction and target deadline;
- DG density, silent-unit fraction, usage entropy, and place-field diversity.

The key diagnostic is whether exploration improves coverage without driving
target-option occupancy and success toward zero. A high exploration fraction
alone is not success.

## Final Analysis Status

All 24 production runs completed at approximately 100M environment frames and
have been analyzed over their final 10% of W&B history. Conclusions, matched
deadline comparisons, and the three-seed tables are in
[[dg_structural_and_manager_exploration_results|Structural Diversity And Manager Exploration Results]].

`CTRL_X1_O1 P010` has the best manager coverage, but forced timeout recovery
dominates manager selection, graph coverage falls, and target success does not
improve. The current manager is therefore a promising behavioral probe rather
than a successful hierarchical controller.
