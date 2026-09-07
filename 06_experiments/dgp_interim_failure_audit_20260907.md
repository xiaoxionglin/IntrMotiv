# IntrMotiv Batch Regression and Trivial-Minimum Audit

**Date:** 2026-09-07  
**Current study:** `dg_policy_gradient_first_outcome_20260906`  
**Study SHA-256:** `2e3104c975188e7cddeb71bce8816c0f4f0d6eb96688c44e0ea2b7560b5447b5`  
**Current status at audit:** 24/24 jobs running; synchronized W&B comparison through 23M environment steps

## Executive diagnosis

The current batch has not frozen numerically, but it has converged early to
uninformative equilibria. The decisive common-factor failure is the
`local_successor` candidate definition:

```text
passive_confidence[source, target] > 0
```

Because any observed passive first-successor qualifies and decayed positive
confidence remains positive, the candidate set is effectively cumulative. By
1--5M, it already contains essentially all 240 directed off-diagonal pairs;
from 5M onward every source has all 15 alternatives. The intended local control
problem therefore became a uniformly sampled all-pairs problem.

This creates two different trivial solutions:

- `FIRST` receives approximately zero expected reward when outcomes are
  command-independent. Its success remains at chance, about `1/15`.
- `HIT` ignores wrong first outcomes and rewards any later target occurrence.
  Broad, recurrent DG fields therefore produce many accidental hits and an
  apparently controllable dense graph without commanded control.

The early flattening is consequently evidence of a stationary proxy solution,
not a stopped optimizer. DG activity, encoder gradients, and JOINT PPO-to-DG
gradients remain finite and nonzero.

## Current batch at the common 20M--23M window

| Outcome | Option success | Target action sensitivity | Commanded evidence | Reliable reachable pairs | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| FIRST | 0.0676 | 0.0112 | first outcome 0.0606 vs shuffled 0.0673 (`-9.9%`) | 0.023 | Chance-level local control; graph correctly remains mostly empty |
| HIT | 0.4949 | 0.00432 | target hit 0.01995 vs shuffled 0.01983 (`+0.6%`) | 0.959 | High completion and graph reachability are accidental-hit proxies |

Both have exactly 15 candidates per source and complete observed-pair coverage.
The external coverage metric falls from about 79--80 in the 1M--5M window to
42.2 for FIRST and 45.6 for HIT at 20M--23M, then changes little. Replacing the
old C15 UCB frontier curriculum with uniform least-tested all-pair commands is
the main reason the current batch no longer reproduces C15's high coverage.

All 24 synchronized 5M spatial snapshots have 16/16 active DG rows, but mean
mono-field fraction is only `0.78%` and grounded controllability is exactly
zero. Thus this is not population silence; the active landmark identities are
not spatially grounded enough for the dense HIT graph to be meaningful.

### JOINT versus STOP

At 20M--23M, pooled seed-matched `JOINT - STOP` effects are:

- coverage: `+0.20 +/- 7.78`;
- target action sensitivity: `+0.00145`;
- target-versus-shuffle advantage: `-0.00529`;
- PPO-to-DG / encoder-to-DG norm ratio: about `1.47`;
- gradient cosine: `-0.0195`;
- rowwise gradient conflict: `51.8%`.

JOINT is active, but has not produced useful control. At 5M it has lower
spatial information than STOP (`0.045` vs `0.066`) and slightly higher map
cosine (`0.346` vs `0.335`). This suggests modest early representation harm,
not catastrophic collapse. The current batch cannot fairly decide the general
value of PPO-to-DG gradients because both gradient conditions are trained on
the saturated candidate problem.

### FiLM versus legacy goal input

At 20M--23M, FiLM raises action sensitivity by about `0.00843`, but lowers
target-versus-shuffle advantage by about `0.0110`. At 5M it also has lower
spatial information (`0.052` vs `0.060`) and higher map cosine (`0.355` vs
`0.326`). FiLM makes the policy react more strongly to target identity, but the
reaction is not aligned with reaching that identity. It is an interface
effect, not a control improvement.

## Historical design ledger

| Change | Apparent benefit | What actually worsened | Verdict |
| --- | --- | --- | --- |
| Mean/punish encoder rewards -> `encourage` | Kept DG population active | Mean/punish left 73--76% of units silent | `encourage` was a real repair; retain ARR encourage |
| Delayed -> immediate target (C02 -> C03) | Option success `0.277 -> 0.453` | Coverage `45.1 -> 23.2`; sensitivity `0.0303 -> 0.0125`; lift stayed about one | Immediate timing is cleaner causally, but was not a performance win |
| Global punishment + row repulsion (C03 -> C05) | Coverage `23.2 -> 42.0`; lift `1.003 -> 1.091`; sensitivity doubled | C05 still had multi-component fields and sink destinations | Modest useful baseline, not solved control |
| CA3 temporal exclusion (C05 -> C06; later X1) | Occasionally raised hit lift or spatial separation | C05 -> C06 success `0.317 -> 0.196`, sensitivity halved; later X1 raised silence, lowered entropy/success/sensitivity, and caused heavy recruitment churn | Clear regression; reject |
| Aggressive orthogonal recruitment | Produced 10--12 distinct spatial peaks instead of 3--4 | Replaced all 16 identities and reduced HRL coverage AUC by `7.6`; policy/graph could not track identity changes | Representation proxy improved while control worsened |
| Strict graph-stabilized recruitment | Avoided churn | C13/C15 made zero replacements; the silent-endpoint gate made retirement nearly inert | Safe but ineffective |
| Open DIR/PRED retirement | Allowed active endpoints in principle | Only one replacement in 36 later runs; that replacement caused persistent stale-generation sample dropping; PRED replacements hurt C13 | Mechanism unproven and operationally unsafe; keep disabled |
| Long deadlines | Fewer timeouts and higher option completion | No target-lift gain; encouraged looping and accidental completion | Proxy inflation, not control |
| Timeout recovery / forced exploration (C12 -> C13) | More explicit exploration time | Coverage `48.1 -> 38.5`, sensitivity `0.075 -> 0.0073`, success `0.759 -> 0.640`, fewer graph edges | Clear regression |
| Separate exploration head | Cleaner conceptual branch separation | In the edge batch, success fell `0.0259` and reachability `0.0197` relative to shared head | No evidence to retain |
| Geometry features | Reduced looping | Success fell `0.0342` and reachability `0.0156`; no control benefit | Added complexity without solving objective |
| Passive UCB topology (C14 -> C15) | Coverage `48.1 -> 78.6`; graph nearly saturated | Lift fell below one (`0.898`) and sensitivity was `0.0019` | Useful exploration curriculum only; not controllability |
| Waypoint planning (C15 -> C16) | Lift returned to about one | Coverage `78.6 -> 41.6`; still no control | Planning cannot repair a missing local controller |
| Source encoder credit (ARR -> SRC) | Coverage `+9.75`, option success `+0.0566` | Spatial information `-0.0265`, map cosine `+0.0663`, mono-field fraction `-0.0461`, grounded control lower | False-positive behavioral improvement; retain ARR |
| Running-consistent BatchNorm | Cleaner-looking actor/learner semantics | Option success `0.409 -> 0.061`, reachability `0.999 -> 0.120`, density `0.0282 -> 0.0125`, spatial information `0.108 -> 0.011` in matched ARR comparisons | Largest implementation regression; already reverted to legacy BatchNorm |
| Explicit one-forward update contract with legacy BatchNorm | Removes double BatchNorm mutation and makes STOP explicit | No comparable collapse; replay match about `0.982`, all rows active | Correctness repair worth retaining |
| Current local FIRST objective | Exposes the HIT graph as a false positive; increases action sensitivity | At chance and below shuffled outcomes because the candidate set is global and first outcomes are often not controllable | Useful diagnostic, failed training formulation as implemented |

## Root causal chain across batches

The recurring failure is not primarily a missing manager heuristic:

```text
broad or multi-component DG identities
    -> passive transitions become dense and nonlocal
    -> accidental landmark hits are easy
    -> HIT certifies false controllability edges
    -> manager favors proxy-reachable destinations
    -> graph/coverage/option-success rise without command dependence
```

FIRST removes the accidental-hit loophole, but the present candidate rule then
turns the problem into balanced prediction/control over all 15 outcomes:

```text
positive passive evidence never truly expires
    -> all 240 candidate edges saturate
    -> most first outcomes are command-independent
    -> centered expected reward is approximately zero
    -> early stationary chance solution
```

This is why adding PPO gradients to DG does not rescue the batch. End-to-end
gradients cannot manufacture a useful causal signal when the commanded event
is usually outside the action's immediate sphere of influence. The slight
negative gradient cosine shows that JOINT can instead oppose the ARR
push--pull objective.

## What should be retained

- frozen ImageNet ResNet-18 trunk;
- legacy BatchNorm behavior;
- one differentiable DG/CA3 forward and explicit gradient boundary;
- ARR temporal-distance credit;
- no recruitment while identity-reset semantics remain unresolved;
- source-matched shuffled evaluation and grounded-controllability diagnostics;
- FIRST as an evaluation outcome and possibly a training outcome only after a
  genuinely local, expiring candidate relation is defined.

## Immediate design conclusion

Do not add another manager patch to this formulation. The next experiment
should isolate one minimal question: can a commanded DG identity alter the
distribution of outcomes within a short causal action window? Candidate
eligibility must be based on recent, substantial transition evidence rather
than `> 0`, and must expire under the same evidence decay. Until that is true,
neither FIRST nor PPO-to-DG is being tested on a meaningful local-control task.

The current 24-run batch remains useful as a negative diagnostic: HIT supplies
a false dense graph, FIRST removes it, and neither STOP/JOINT nor LEG/FiLM
creates grounded control. It should not be used to conclude that end-to-end
PPO-to-DG gradients are generally ineffective.

## Is longer training likely to rescue it?

The apparent late learning in the completed DPR and Saturday studies was
mostly an increase in how strongly the policy reacted to target identity, not
an increase in selecting the commanded outcome. Between the 45--55M and
65--75M windows:

- action sensitivity increased in 42/54 DPR runs and 32/36 Saturday runs;
- mean action-sensitivity gains were `+0.00330` and `+0.00279` respectively;
- DPR coverage decreased by `1.60` on average and Saturday coverage was flat
  (`+0.13`);
- Saturday target-versus-shuffle advantage decreased by `0.00181` on average;
- only 12/36 Saturday runs improved that advantage, only one crossed from
  negative to positive, and none crossed above a `5%` advantage.

Thus the user's visual observation of post-50M improvement is real, but the
improving curves are chiefly conditioning and proxy curves. They did not turn
into reliable causal control by 75M.

The current batch provides stronger evidence against rescue by duration. FIRST
commanded-versus-shuffled first-outcome advantage progresses from about
`-3.4%` at 1--5M to `-5.8%`, `-8.5%`, `-8.4%`, and `-9.9%` in successive
windows through 23M. HIT remains within roughly one percent of its shuffled
target rate while its graph saturates. The common candidate set is already
complete, so more experience repeats the same ill-posed all-pairs problem.

Recommendation: let the already-running study reach its declared 75M horizon
if the 50M/75M representation checkpoints and complete factorial diagnosis are
worth the remaining compute, but do not extend it beyond 75M and do not expect
late training to rescue control. If compute conservation dominates, 50M is a
sufficient early-stop point after the synchronized snapshot. Any next training
budget should go to correcting candidate locality rather than increasing this
study's horizon.

## Artifacts

- W&B trajectory collector:
  `06_experiments/analyze_dgp_interim_wandb_20260907.py`
- synchronized per-run windows:
  `06_experiments/results/dgp_interim_20260907/per_run_window.csv`
- condition trajectories:
  `06_experiments/results/dgp_interim_20260907/condition_window_summary.csv`
- exact provenance:
  `06_experiments/results/dgp_interim_20260907/analysis_manifest.json`
- completed-study late-window audit:
  `06_experiments/results/late_learning_audit_20260907/late_deltas_with_crossings.csv`
- NEMO2 spatial summary:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/dgp_interim_spatial_5m_20260907/`
