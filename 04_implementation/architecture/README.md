# IntrMotiv Architecture Reference

This directory is the maintained, code-oriented reference for the IntrMotiv
architecture. It separates mechanisms that are easy to conflate during batch
design: reward signals, optimization losses, non-gradient state updates, and
architectural switches.

## Reading Order

1. [[architectural_choices]]: model, state, manager, graph, motion, and
   optimization choices. Start here when defining a run.
2. [[losses]]: every implemented reward-derived objective, auxiliary loss, and
   structural update. Start here when interpreting a loss curve or adding an
   ablation.
3. [[../IntrMotiv_metric_reference|IntrMotiv Metric Reference]]: exact logged
   metric definitions and denominators.
4. [[../current_hrl_architecture_summary|Current HRL Architecture Summary]]:
   the older, long-form implementation snapshot. It is useful for historical
   detail, but its batch-specific sections are not the current source of truth.
5. [[../core_logic_audit_20260901|Core Logic Audit, 2026-09-01]]: known
   implementation inconsistencies, affected historical conclusions, and the
   required fix order before the next clean comparison.

## Scope And Source Of Truth

- The implementation is in NEMO2 source checkout
  `SF_git_XXL/SF_hipposlam/sf_working_directories/IntrMotiv/`.
- Primary files are `custom_encoder.py`, `custom_core.py`,
  `custom_learner.py`, `hrl_controllable_graph.py`,
  `topological_frontier.py`, and `custom_params.py`.
- This reference was reconciled with those files on 2026-08-31. When code and
  this vault disagree, code wins; update this directory as part of the code
  change.

## Terminology

| Term | Meaning |
|---|---|
| DG activity | Post-threshold non-negative output of the trainable DG projection. |
| DG logit | Batch-normalized DG projection output before the thresholded ReLU. |
| DG node | One DG unit, treated as one landmark and one possible worker subgoal. |
| CA3 state | Fixed, non-parameterized DG shift register of width `F * (R + L - 1)`. |
| Worker | The only learned policy: PPO actor and value network conditioned on an optional waypoint/target. |
| Manager | Deterministic code that selects targets, routes, validates edges, or invokes bounded exploration. It has no optimizer or value head. |
| Passive edge | An observed local landmark transition. It is evidence only, never usable for planning until deliberately validated. |
| Controllable edge | A successful target-conditioned transition, retained as a fast-weight graph statistic. |

## Maintenance Rules

- Give every new loss its formula, mask/denominator, gradient destination,
  default setting, and interaction constraints in [[losses]].
- Give every new CLI-controlled architecture branch its state scope, replay
  contract, default, and comparison/control in [[architectural_choices]].
- Do not call a reward a loss, or call a buffer mutation a learned objective.
  The distinction matters for causal interpretation.
- Record batch-specific parameter sets in `06_experiments/`; keep this folder
  batch-independent.
