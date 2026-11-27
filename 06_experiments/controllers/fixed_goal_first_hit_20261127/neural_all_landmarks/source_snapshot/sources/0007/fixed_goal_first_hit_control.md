# Fixed-goal first-arrival control after reward-free exploration

Date: 2026-11-27. Status: theory and prospective simulation protocol, written before the new simulations. Scope confirmed by the user: **every reachable learned DG landmark**, with no task reward or learning during goal evaluation. The full DMLab claim remains open.

## Candidate and existing evidence

Preserve the frozen visual trunk, learned DG identities, fixed CA3 sequence memory, and existing goal-conditioned decoder. Train and deploy an action-value head for reaching the **same requested goal** before a deadline. Use recorded reward-free transitions and all goal labels, including failures. Freeze DG weights and BatchNorm during this first control experiment so the goal definitions and replay states remain stationary.

This is a change to the controller's learning and acting contract, not a proposal to add a manager, transport memory, or world model. Minimality is conditional on the existing representation and goal interface being sufficient; it is not a global architecture-minimality theorem.

The audited desktop runtime is `/home/xiaoxiong/SFgit/SF_hipposlam`, commit `206ee833` (full identity will accompany experiment artifacts). Its `dmlab/controller_q.py` states that the main head predicts continuing parent return and the hindsight auxiliary never selects actions. `controller_stored_replay.py` keeps the real successor command for main replay; virtual goals are fixed on both states only for the auxiliary. `custom_actor_critic.py` acts through the main head. Thus adding HER again would not isolate the missing contract.

Stopping PPO value/GAE at a hit was already tested in [18 completed 100M runs](../06_experiments/reward_credit/goal_value_stop_dg_20261009.md), without replicated goal-selective improvement. The proposed contrast additionally changes supervision coverage, the same-goal Bellman target, and which objective actually selects actions. These must be separated in the small tests rather than attributing any gain to termination alone. The desktop source is not the isolated release used by every historical study.

## Objective and conditional result

Let $s_t$ be the available physical controller state: DG/CA3 and only the baseline's existing context inputs. Let $G_g$ be the set recognized by frozen DG detector $g$, and let $y_g(s_{t+1})$ indicate its occurrence in the genuine successor observation. Starts already in the requested goal are reported separately. A detector set can have multiple physical components: reaching its identity does not establish navigation to a particular component.

Define the first-arrival time $\tau_g=\inf\{k\geq1:s_{t+k}\in G_g\}$. The primary objective is maximizing $P(\tau_g\leq H)$ before physical episode termination. For $h$ remaining decisions, define

$$
Q_0(s,a,g)=0,
$$

$$
Q_h(s,a,g)=\mathbb E\left[y_g(s')+(1-y_g(s'))(1-d_{\rm phys})\max_b Q_{h-1}(s',b,g)\mid s,a\right].
$$

The successor retains $g$, even when the exploration manager selected a different command. A successful terminal successor earns one; a failed physical terminal transition earns zero. A rollout buffer boundary is not a physical terminal. Reset observations must never stand in for an unavailable terminal observation. The controller selects $\arg\max_a Q_h(s,a,g)$ and decrements $h$ until success, termination, or the evaluation deadline. Passing another DG landmark is not failure for this arbitrary-destination objective; first-distinct outcome is an additional diagnostic.

**Proposition.** If $s$ is a Markov sufficient state for the controlled process and goal detectors, the detector and transitions are stationary, every necessary state-action transition is identified, and backups are exact, then $Q_h$ is the optimal probability of first reaching $G_g$ within $h$ decisions after taking action $a$. No additional recurrent state or predictive model is required under these conditions.

**Proof.** For $h=0$, there are no future decisions and the value is zero. For $h\geq1$, partition the successor into success (value one), physical failure (zero), and a live non-goal state. In the third case the induction hypothesis supplies the optimal remaining-$h-1$ success probability; choosing the maximizing action gives the displayed backup. Taking the expectation completes the induction.

A discount below one defines a different objective, $\mathbb E[\gamma^{\tau_g-1}\mathbf1\{\tau_g\leq H\}]$, trading arrival speed against probability. It is not silently substituted for the finite-deadline probability. Deadline conditioning is already expressible by the existing auxiliary's hidden/clock interactions; its old weights and reward targets are not assumed calibrated for this new objective.

If $\|\widehat Q_h-Q_h^*\|_\infty\leq\epsilon_h$, the loss of greedily choosing from $\widehat Q_h$ and subsequently using the approximate greedy policies is at most $2\sum_{j=1}^{H}\epsilon_j$, capped at one. At each step the selected action loses at most $2\epsilon_h$ relative to the optimal action, followed by the remaining policy's loss. This is a conditional error bound, not a convergence guarantee for the neural decoder or incomplete replay.

## What reward-free data can and cannot identify

An unlabeled transition $(s,a,s',d)$ supplies a target for **every** frozen DG identity. Enumerate the goal set, or draw goals independently of the realized successor given $(s,a)$. In either case the conditional distribution of successors remains the physical transition kernel. Selecting only goals achieved later can alter that distribution in stochastic environments. Positive examples alone cannot distinguish a reliable action from a lucky action.

Coverage of states is insufficient: the relevant actions at those states must also have support. Two environments can agree on every recorded transition and disagree on the consequence of an untried action. No offline method can promise arbitrary-goal control in both. Support masks can limit extrapolation but cannot recover missing outcomes. Report unsupported decisions and unreachable goals with the full denominator.

Representation sufficiency is a separate condition. If two histories produce identical available controller inputs but require opposing actions to reach the same goal, a shared decoder cannot solve both. With equally likely contexts and deterministic incompatible correct actions, one-step success is at most one half. A goal label cannot resolve a missing state distinction. Only a demonstrated failure of this type motivates an additional action/history feature; changing memory is not the first intervention by default.

Likewise, multiple physical destinations sharing one DG identity cannot be selected independently through that identity. The agreed target is the landmark detector set; physical grounding and false detector hits remain required diagnostics. Freeze learning throughout evaluation, execute alternative commands from identical starts, and distinguish detection, physical arrival, and accidental long-horizon arrival.

## Prospective small-scale tests

Use matched reward-free transition datasets and seeds 8, 99, and 123; goals are assigned only after collection. Record dataset/source/config hashes. Freeze all learning for evaluation, include every declared start-goal pair, and keep supported, unsupported, already-at-goal, and unreachable denominators separate.

1. **Exact finite systems:** directed branches, loops, bottlenecks, a stochastic risky-versus-reliable route, and incomplete action support. Compare fixed-goal first-hit backups with continuing reward across manager commands, fixed-policy successor predictions, uniform random actions, and an oracle that is explicitly excluded from learning. Evaluate multiple deadlines and executed alternative-goal commands. The oracle is a correctness reference, not a trained competitor.
2. **Representation boundary:** matched observation histories with incompatible action requirements. Show when unchanged DG/CA3 is sufficient, when it necessarily aliases states, and whether the smallest action-history addition removes that particular alias. Do not count a tabular state identifier as evidence that the real CA3 decoder learned control.
3. **Learned decoder:** use the established CA3 shift adapter and a goal-conditioned neural readout with the same observations, actions, data, update budget, and seeds across the tested objective/routing conditions. Report full goal-conditioned arrival matrices, worst-goal success, alternative-command lift, latency, Bellman calibration, and train/evaluation support. A tabular success alone does not release DMLab production.

**Gates declared before results.** Exact backups must match exhaustive oracle values to numerical tolerance on supported finite systems. The stochastic test must retain failure outcomes. Any representation impossibility witness must remain a failure for the unmodified observation-only state. The learned unchanged-memory candidate must reach at least 90% success on every supported deterministic start-goal pair in each of the three seeds, with positive paired command lift and without goal-evaluation updates, before it is called sufficient on that toy. Failure triggers diagnosis of supervision, approximation, or aliasing; no tuning on evaluation pairs and no expanding the architecture without identifying the failure. A qualification may proceed to diagnose a failed gate, but full DMLab production requires the relevant scientific gate to pass.

## DMLab continuation

Reuse a provenance-pinned reward-free checkpoint and the existing recorded-state replay path. First qualify frozen DG/BN and unchanged CA3, state/goal/action timing, genuine terminal observations, same-goal successors, reward independence from exploration bonuses, and acting-head identity. Inspect decoder expressiveness before deciding whether the existing deadline head or a main-head extension is the smaller compatible implementation. Evaluate old continuing-return behavior and the new controller on the same frozen representation, exploration data, start states, goals, deadlines, and seeds. Offline training cost and any newly collected reward-free data must be reported separately.

Use the canonical [StudySpec workflow](../04_implementation/standardized_study_workflow.md) and [manifest-driven telemetry](../04_implementation/reusable_place_field_telemetry.md). Keep the prior 80-run action-transport study separate; it asks a different question. The new study must include place fields, trajectories, graph support, and executed frozen command interventions. NEMO2 access awaits the user's manual login confirmation; local work continues. Never interpret a rendered plan as a submitted or successful DMLab experiment.

## Prior-art boundary

Goal-conditioned action values, hindsight experience reuse, and first-passage dynamic programming are established ideas. The research question here is whether their precise learning-to-acting connection repairs this existing architecture with a minimal change. References will be source-verified in the experiment report; no mathematical novelty is claimed.
