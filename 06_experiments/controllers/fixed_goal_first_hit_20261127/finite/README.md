# Exact finite-system first-hit control

Date: 2026-11-27. These are **tabular mechanism simulations**, not a simulation of the current PPO learner or proof that learned DG/CA3 supports DMLab navigation. The [theory](../../../../02_algorithm/fixed_goal_first_hit_control.md) preceded collection and fitting; its SHA-256 is retained in `prerun.json` and `provenance.json`.

## Design and evidence boundary

All methods share reward-free one-step transition samples at seeds 8, 99 and 123, with 100 outcomes per observed state-action pair. Collection resets independently to each nonterminal state: this is a **generative coverage experiment**, not evidence that autonomous exploration discovers the required actions. Every recorded episode contains one action, so longer successful routes require composition across episodes. The collector marks its one-step cuts as truncations and preserves the genuine successor; those cuts do not terminate Bellman backups. Physical termination does.

The seven-state fork has two requested goal states, reversible branch routes and self-loops. Its two targets are a small mechanism test, not an exhaustive claim over every DG identity in IntrMotiv. The stochastic witness compares a 0.1-success risky action with a reliable two-action route. A missing-support condition omits the hub-to-right action without removing its affected evaluation pairs. The alias witness has identical observations in two equally likely contexts requiring opposing actions.

Policies are frozen during exact evaluation against the true transition kernel. The kernel is available to the simulator and independent recursive oracle; fitted controllers use empirical transition counts only. All declared start-goal pairs remain in `per_pair.csv`; initially achieved goals are flagged and excluded from headline means, while unreachable-within-deadline and unsupported cases remain visible. `command_trials.csv` includes every executed command/evaluated-target combination. An alternative trial stops when its own command is reached; hitting the evaluated target first counts as success. An alternative already achieved at the starting state stops at time zero and is excluded from the primary lift. Thus the fork has 14 declared pairs, 12 primary success pairs, and 10 nontrivial paired-command comparisons.

The continuing comparison explicitly retargets to goal zero after **every decision**, allows repeated hit rewards, and optimizes those subsequent rewards. It is a deliberately simple counterexample to objective interchangeability, **not a replica of the current C15 manager**. The fixed-policy-successor comparison predicts uniform-policy future hits and then acts greedily using those predictions; it is policy improvement from a fixed-policy predictor, not uniform behavior itself.

## Results

At the eight-decision deadline, all three seeds give the same complete-fork results:

| Controller | Successful probability averaged over 12 noninitial-goal pairs | Worst pair | Maximum oracle regret | Mean executed-command lift |
| --- | ---: | ---: | ---: | ---: |
| Same-goal first hit | 1.0000 | 1.0000 | 0.0000 | 0.8000 |
| Fixed-policy successor predictions, greedy deployment | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| Continuing one-decision manager retargeting | 0.5000 | 0.0000 | 1.0000 | 0.0000 |
| Uniform random actions | 0.2356 | 0.0098 | 0.9902 | 0.0004 |

The fixed-policy predictor also solves this graph: these results **do not establish superiority of optimal backups over that comparison**. Undiscounted first-hit values can have several optimal actions, so lowest-index tie-breaking can produce slower paths and incidental alternative-target hits while retaining perfect deadline success. The rows at deadlines 1, 2 and 4 remain in the full tables.

Removing one observed action leaves four of the twelve start-goal routes unsupported. The first-hit controller succeeds on 8/12, with worst-pair success zero. The oracle still succeeds on all twelve. This failure is retained in the denominator. `missing_support_witness.json` constructs two kernels that agree on every observed action and differ only on the omitted action, demonstrating the identification limit.

In the stochastic witness, the risky samples contain 7/100, 6/100 and 11/100 successes at seeds 8, 99 and 123. Retaining failures produces the reliable two-step choice and true success 1 at deadline two in every seed. Filtering to one-step hits makes the risky estimate 1, while its deployed true success remains 0.1. This is an explicit outcome-selection bias illustration, **not an implementation or evaluation of full HER**. At deadline one, the risky action is correctly optimal with true success 0.1.

Exhaustively enumerating the two shared observation-only deterministic policies gives mean success 0.5 for each in the alias witness; randomized mixtures cannot exceed their convex hull. Distinguishing contexts by one bit permits oracle success 1 in both. This is an information boundary, not learned neural performance or evidence that an action register fixes the actual DMLab representation.

Independent recursive-oracle agreement is exact to the recorded precision: maximum discrepancy **0.0**. There are nine datasets, 1,392 per-pair rows, and 144 summary groups. Fourteen focused tests pass, including terminal-success precedence, composition from one-step episodes, horizon-dependent actions, failure retention, omitted support, alternate-command stopping, nontrivial lift denominators, the alias bound, and the subsequent analytic witness below.

## Subsequent fixed-policy counterexample

Because the original fork did not distinguish fixed-policy prediction from optimal backups, a separate analytic diagnostic was specified **after inspecting that result**. It is not added retrospectively to the original prospective nine-dataset suite. At a two-action deadline, action zero immediately succeeds with probability 0.75; action one reaches a state where one action succeeds and the other fails. Uniform continuation predicts action one's value as 0.5 and greedily selects the 0.75 risky action. Optimal continuation predicts its value as 1 and selects the reliable composed route. Exact execution yields 0.75 versus 1. The design was saved before calculation in `post_audit_successor_witness_design.json`; `post_audit_successor_witness_result.json` records both action-value vectors and outcomes. This identifies a mathematical distinction, not empirical superiority on the original fork.

## Reproduction and provenance

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xiaoxiong/miniforge3/envs/SF_git/bin/python -m unittest hpc_runs.test_first_hit_finite -v
PYTHONDONTWRITEBYTECODE=1 /home/xiaoxiong/miniforge3/envs/SF_git/bin/python -m hpc_runs.first_hit_control.finite --output 06_experiments/controllers/fixed_goal_first_hit_20261127/finite --theory 02_algorithm/fixed_goal_first_hit_control.md
```

`prerun.json` is persisted before collection. `provenance.json` verifies the theory and simulation source remained unchanged during execution. `dataset_manifest.json` records each raw NPZ hash, canonical data-payload hash, environment hash, row count and state-action support. `oracle_audits.json` preserves each oracle check. The source file is `hpc_runs/first_hit_control/finite.py`; tests are `hpc_runs/test_first_hit_finite.py`.

The useful shortcut is to verify the exact target semantics and full denominators on these tiny systems before interpreting neural approximation or DMLab failures. The alternate-command audit caught a protocol mismatch during development: holding an alternative command alive after it had succeeded would not match the proposed deployment contract. The regression test now enforces stopping at that success. A second review removed already-satisfied alternative commands from the primary lift; `lift_denominator_correction.json` preserves the initial uncommitted result and source hashes, and the tables were regenerated with the corrected denominator.
