# Bastankhah 2026: empowerment representations in IntrMotiv

Read: 2026-09-08. **Mahsa Bastankhah, Sophie Broderick & Benjamin Eysenbach, “Learning to Perceive the World Through Control: Empowerment-Based Representation Learning,” arXiv:2605.30656v1 (28 May 2026).** [Paper](https://arxiv.org/abs/2605.30656) · [Full text](https://arxiv.org/html/2605.30656v1).

**Takeaway for IntrMotiv:** evaluate whether sparse DG events preserve distinctions needed for reliable control through CA3 memory. Temporal separation, field diversity, and novelty reward are insufficient evidence by themselves.

## What the paper establishes

Empowerment maximizes skill–future-state information, \(\max_{p(z\mid s_0)} I(Z;S^+\mid s_0)\). A skill discriminator factors through forward \(\phi(s_0)\) and backward \(\psi(s^+)\) representations: the former concerns controllable futures, the latter how outcomes are reached. They need not agree. Invariance requires minimal representations/policies and specified dynamics/noise assumptions; an invariant optimum exists, but not every optimum is invariant. Even their combination can lose control-relevant information. Backward sufficiency requires deterministic control-relevant dynamics and common-start, exact-horizon reachability. METRA experiments demonstrate noise robustness; downstream control uses frozen representations with SAC/PPO. These are not APPO or DG/CA3 experiments. See §§3–6, especially Definition 4.1, Corollary 5.6, and Propositions 5.7–5.8. [Full text](https://arxiv.org/html/2605.30656v1).

## Synthesis against the project

The following are project-specific interpretations and proposals, not results established by the paper. Context is from local design/implementation records, not a fresh audit of the NEMO2 runtime.

| IntrMotiv question | Consequence for design and interpretation |
|---|---|
| **Elapsed time / geodesic distance** | Retain the project's conditional claim: CA3 elapsed time approximates useful travel distance only after landmarks stabilize and the controller learns efficient routes. Measure directed first-passage time, success probability, and timeout together. A long delay may mean a detour or failed control. Do not turn an embedding distance into a travel-time claim. |
| **Goal-conditioned control** | Treat the current control state and desired outcome as different roles. Here DG supplies sensory events and CA3 supplies history; a candidate extension is a current-state head on CA3 and a separate goal head. Do not equate the paper's forward representation with DG or its backward representation with CA3. Retain goal-independent DG identities so the command cannot redefine what counts as arrival. |
| **DG formation / separation** | Test whether merging two DG identities impairs commanded outcomes, rather than rewarding separation indiscriminately. Redundant codes can be merged; visually similar situations may still require different actions or histories. Under partial observation, ask what distinctions DG must inject into CA3, not whether a single image contains the complete control state. Spatial localization remains necessary for the project's physical-landmark interpretation. |
| **Curiosity / novelty** | Distinguish unfamiliarity from controllable outcome diversity. An identity absent from finite CA3 memory may simply have been forgotten; a never-seen identity may be unreachable. Noise-driven onsets or repeated long cycles can increase intrinsic reward without extending useful control. Preserve coverage, raw-field, and intervention diagnostics. |
| **Managerless HRL** | The September 7 design already provides the clean comparison: flat intrinsic control versus persistent, absent-identity goal commands, without a graph manager. Goal sampling and termination still exist. Earlier “no learned manager” designs had deterministic managers; do not use that phrase to claim manager absence. Persistent commands demonstrate temporal organization only when they causally change achieved outcomes. |
| **APPO practicality** | Start with the existing STOP-versus-JOINT goal comparison and exact-command replay. Keep the frozen visual trunk. An empowerment-inspired objective would be an additional experiment, not a drop-in replacement justified by this paper. Specify its gradient path, outcome definition, horizon, and sampling prior before implementation. |

### Keep the distance claim explicit

For fixed landmark regions and a specified source-state distribution, use

\[
T_\pi(i\to j)=\mathbb E_\pi[\tau_j\mid i],\qquad
T^*(i\to j)=\inf_\pi T_\pi(i\to j).
\]

The first is policy-dependent; the second is optimal expected travel time. Neither is automatically symmetric or equal to spatial geodesic length. The project's positive success payment decreasing with latency also trades off arrival probability against speed. Reporting only successful latencies can favor a policy that fails often. Use the existing 8/16/32/64-decision arrival curves and route-efficiency evaluation, with physical geometry used only for evaluation.

### Smallest useful empirical test

Reuse the September 7 matched-start command intervention and common observation panel. At each verified start/CA3 state, execute alternative commands with matched sampling conditions and measure

\[
I(G;Y\mid h)=H(Y\mid h)-\mathbb E_G H(Y\mid h,G),
\]

where the outcome vocabulary and horizon are fixed independently of the encoder being assessed, and failure/timeout remains an outcome. This extends the project's existing control-representation principle: different commands should reliably produce different outcomes. Also report commanded success; mutual information alone can score a consistent permutation of goal labels highly.

Compare STOP and JOINT on commanded-versus-other hit rates, qualified physical arrivals, route efficiency, and common-panel DG stability. A change in policy action probabilities alone is insufficient. On the same panel, test harmless visual perturbations separately from changes to available routes or task context. Preserve common physical outcomes across checkpoints so DG drift cannot manufacture apparent improvement. Estimate information cautiously with repeated trials and finite-sample uncertainty.

For a later discriminator experiment, a possible reward is \(\log q(g\mid h_t,Y)-\log p(g\mid h_t)\). Store the actual command, its behavior-time prior, horizon, and reward/version information. Replay the behavior condition; do not resample goals or casually apply hindsight PPO. Fix reward evaluation within a learner update and audit asynchronous representation drift and DG BatchNorm semantics. In JOINT, prevent representation changes from changing the success labels. This is a proposed engineering contract, not a tested implementation.

## Related literature: SGCRL mechanism

**Bastankhah, Liu, Arumugam, Griffiths & Eysenbach, “Demystifying the Mechanisms Behind Emergent Exploration in Goal-conditioned RL,” ICLR 2026; arXiv:2510.14129.** [Paper](https://arxiv.org/abs/2510.14129) · [Mechanism, §4](https://arxiv.org/html/2510.14129v1#S4).

Preserve the earlier insight: **the actor pursues states that look goal-like in the learned representation; contrastive updates reduce that similarity along unsuccessful trajectories, while successful paths acquire a goal-similarity trace.** Their low-rank tabular analysis helps isolate this mechanism. “Unexplored states are precisely the goal-like states” is too strong: this is a learned exploration dynamic, not an exact novelty detector or a general coverage guarantee.

For IntrMotiv, this motivates testing whether representation plasticity itself changes exploration, beyond explicit CA3 novelty gating. Freezing DG while preserving the policy's training procedure is a useful causal comparison, provided the induced changes to reward labels are tracked. SGCRL's contrastive shaping should not be conflated with the current elapsed-time encoder/decoder objectives.

## Local context and reusable reading route

Relative links below resolve in the project's `08_literature` folder:

- [Transition-distance reward plan](../03_transition_distance/IntrMotiv_transition_distance_reward_plan.md): stable landmarks, efficient control, and the geodesic hypothesis.
- [Control as a representation principle](../05_plans/control_representation_principle.md): DG-to-CA3 information path and command/outcome interventions.
- [CA3 memory novelty and goal implementation, September 7](../06_experiments/ca3_memory_novelty_goal_implementation.md): graph-free baseline, STOP/JOINT, actual-command replay, and qualified-arrival evaluation.
- [Explicit update contract, September 5](../05_plans/explicit_dg_controller_update_contract.md): gradient and representation-version contracts.
- [Architecture reference](../04_implementation/architecture/README.md): catalogue; older snapshots are not current batch defaults.
- [Existing control-relevance literature review](control_relevance_fisher_place_fields_20260908.md): broader prior-art positioning.

Workflow lesson: begin with the literature index and latest dated implementation/study record; then inspect primary-paper definitions, assumptions, and experiments. The historical architecture snapshot alone would have misidentified the current manager and gradient choices. arXiv HTML supplied the relevant equations directly. This note adds interpretation only; no training change or new outcome claim is implied.
