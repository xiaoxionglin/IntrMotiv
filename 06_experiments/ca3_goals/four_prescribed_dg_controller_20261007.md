# Four prescribed DG goals: matched controller screen

**Status, 7 October 2026:** implementation and six-run StudySpec are prepared locally. No training result is reported here. Production release requires native field/transition preflight and a short full-learner qualification on a NEMO2 compute node.

## Question and comparison

Can the existing C15 direct FiLM worker acquire command-specific control when four goal identities are fixed, compact and unambiguous? The intervention is privileged: engine position defines the four DG activities, but neither position nor target coordinates enter the worker, manager or reward code. A positive result shows that this supplied representation package helps the current controller; it does not show that visual DG can learn it.

The [workflow-1.14.1 StudySpec](../../hpc_runs/studies/four_prescribed_dg_controller_20261007.study.json) declares one matched contrast, ORACLE minus LEARNED, with seeds 8, 99 and 123. Both arms have 16 DG channels, the same fixed ImageNet visual trunk, CA3 size, FiLM target interface, C15 frontier manager, reward rule, action set, 64-decision fallback horizon, and four eligible manager identities (IDs 0–3). Both expose a four-value `dg_prescribed` observation slot, so their observation structure matches. In LEARNED, all 16 DG channels are learned and this slot is zero. In ORACLE, channels 0–3 are replaced by fixed fields; channels 4–15 remain learned CA3 context. Those context channels cannot become graph nodes or manager goals. The existing encoder batch-use loss excludes only the fixed four rows; the other configured DG losses retain their source semantics. Recruitment is monitor-only with zero permitted replacements in both arms.

This is a six-run screen, not an architecture sweep. The historical odor/C15 result is context, not a capacity- and goal-vocabulary-matched control. The source design discussed 4 fixed plus 28 learned units; the user explicitly selected **4 fixed plus 12 learned** for this batch.

## Prescribed events and feasibility gate

For each center $c_i$, the environment computes $g_i(p)=\exp(-\|p-c_i\|^2/(2\sigma^2))$ and sends

$$
a_i(p)=\frac{\max(0,g_i(p)-e^{-2})}{1-e^{-2}},\qquad \sigma=20.
$$

The field peaks at one and is exactly zero at and beyond radius 40 DMLab units. Centers are provisionally $(550,550)$, $(1450,550)$, $(550,1450)$ and $(1650,1650)$, selected from non-wall cells of the fixed third map. Static Lua inspection is insufficient to certify the world-coordinate transform, collision margin, or decision-time event rate. Before training, run a compute-node native DMLab preflight that verifies center positions, support, episode resets, and that all four events can be sampled at repeat eight. Record per-field event counts and transitions under the actual 64-decision horizon. If a field is inaccessible or routinely skipped between observations, fix the declared geometry and regenerate the StudySpec fingerprint before any production submission.

The prescribed values enter after the learned DG threshold and before CA3. The visual trunk and projection do not receive privileged coordinates. The first four projection outputs in ORACLE are unused, and gradients from the fixed fields are blocked; the remaining twelve learned outputs still update. Manager recognition masks context-only channels while CA3 retains them. The same goal mask applies to LEARNED, so selecting among only four goal IDs is held fixed.

## Outcomes and release gates

The primary outcome is **command-caused arrival** by the declared option deadline, measured from matched starts under alternative target commands. Report target-macro arrival probability and lift, all failed/time-out trials, each goal and seed, first distinct event, time to arrival, and initial-action distribution change. A first event is diagnostic; arrival need not be first for a goal to be useful. The existing intervention manifest requests all three alternatives per source and 20 attempts per ordered pair, subject to a bounded preflight and an exact-start verification. Online option success and graph edges are secondary. External coverage AUC is the exploration outcome, paired by seed over early and terminal windows.

Qualification proceeds in this order:

1. Verify the four fixed fields, reset/observation alignment, exact zero outside support, no direct coordinate bypass, and matching manager goal masks in both arms.
2. Run one short full-learner job per arm on a compute node. Require finite losses/rewards, at least some events for each prescribed field, nonempty goal candidate opportunities, learned-context updates, and a checkpoint reload with identical fixed fields and goal mask.
3. Run the six declared jobs to 75M with milestones at 5M, 25M, 50M and 75M. Use the canonical StudySpec and print-only submission audit; keep all bulk outputs in `/work/classic/fr_xl1014-corridor-geometry`.
4. Use the standard manifest-driven 10k-decision field evaluator and matched-command intervention. Report fixed-oracle versus learned-context maps separately, including silence, active-only overlap, peak diversity and pre-threshold maps. The standard 100-unit grid is too coarse to certify an 80-unit support, so verify the prescribed field shape directly from position/activity samples as an additional diagnostic.

If the oracle arm has clear command lift while LEARNED does not, unstable or ambiguous learned goal identity becomes the leading bottleneck. If both arms show little action change and little lift despite adequate events and candidate exposure, worker goal-conditioning or credit is implicated. If the oracle fields are rarely encountered or goals are infeasible under the deadline, the controller comparison is inconclusive; report that gate rather than interpreting a null result.

## Provenance and reusable lesson

The current StudySpec validates as six unique cells under schema `intrmotiv/study/v1`, declared workflow `1.14.1`, SHA-256 `009cd13111d886839f04624be0913f95773ef4e672ce5c86b6853724afca6b1d`. It derives from the corrected odor C15 configuration without its odor and Hebbian-selector factors. The runtime implementation is isolated in `/tmp/intrmotiv_four_oracle_20261007` pending source commit, remote synchronization, and compute-node qualification. Local focused tests currently pass; they do not establish native field feasibility or training quality.

The efficient path for a future prescribed-goal study is to reuse the canonical StudySpec and evaluator, and preflight event availability before launching a large matrix. A perfectly shaped field is not a useful controller test if its event is absent from the actual decision stream.
