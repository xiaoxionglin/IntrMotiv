# Future Scope and Research Inspirations

Created: 2026-09-08. Living collection of paper ideas emerging from IntrMotiv batch design and interpretation.

## Purpose and selection criteria

Preserve ideas that could support a distinct scientific explanation or a separate paper, especially ideas that generalize beyond spatial navigation. An entry should pose a substantive question, identify a plausible mechanism, and suggest evidence that could distinguish it from alternatives. Routine fixes, parameter adjustments, and untested combinations of modules belong in implementation or batch plans.

The leading inspiration, explicitly raised by the user, is **place-field formation as a way to establish reliable control**. This could become more compelling than the currently planned main paper. Preserve that possibility without deciding the paper's direction in this note.

The [current scientific program](scientific_program_after_corrected_core_20260902.md) centers on sequence-derived intrinsic feedback organizing sensory events and exploration before external reward. The ideas below ask broader questions about what representations are for, how control shapes them, and why useful abstractions emerge. They can become spin-offs or motivate a later change in the main thesis if the evidence warrants it.

This is a synthesis of project notes and recovered discussions, not a comprehensive literature review or a claim of novelty. “Hypothesis” and “proposed test” below denote future work; observations are attributed to existing reports. Earlier design proposals are preserved as inspirations even when their implemented variants have not succeeded.

## 1. Place fields form to make control reliable

**Literature check:** [Control relevance, Fisher sensitivity, and place fields](../08_literature/control_relevance_fisher_place_fields_20260908.md) records close precedents and a narrower possible contribution. In particular, reward-driven place-field reorganization and empowerment-based representation learning already have direct treatments.

**Expanded mathematical note:** [Control as a principle for representation: equations and recovered inspirations](control_representation_principle.md). This develops the user's squared-gradient/Fisher-information intuition, the Lin et al. continuation, and related ideas recovered from past tasks.

**Possible paper thesis:** Localized fields emerge because an agent needs sensory identities that support repeatable, distinguishable behavioral outcomes.

The deeper motivation is to continue the control-based explanation in Lin et al.: ask why learning to act should select particular representational distinctions. The central intuition is **represent distinctions that make a difference to what the agent can deliberately accomplish**.

For the baseline, DG encodes $z_t=f_\theta(x_t)$ and CA3 forms $h_t=\mathcal M(h_{t-1},z_t)$; the control state is $s_t=h_t$. Two complementary quantities make the control principle concrete. With matched source state $s$, randomized command $U$, and subsequent outcome $Y$,

$$
\mathcal D(s)=I(U;Y\mid S=s)
=H(Y\mid S=s)-\mathbb E_u H(Y\mid S=s,\operatorname{do}(U=u))
$$

measures whether different commands produce distinguishable consequences. The entropy terms use the same intervention distribution. High outcome diversity is useful here only when it depends on the command. Meanwhile, for a representation cell $B$,

$$
\mathcal R(B,g)=
\mathbb E_{s\mid B}\max_a Q_g(s,a)
-\max_a\mathbb E_{s\mid B}Q_g(s,a)\geq0
$$

measures the value lost by requiring those states to share one action choice. Positive $\mathcal R$ supplies a reason to split an identity. These are proposed conceptual formulations, not current implemented losses. The companion note states their assumptions and explains why policy entropy alone is insufficient.

The causal idea is that a landmark activated in several behaviorally incompatible situations is an unreliable command or starting state. Learning to control transitions could favor representations that separate those situations. Spatially localized fields would then be one expression of a more general requirement: the same internal identity should have coherent consequences under the same action or command.

This is stronger than showing that place-like activity accompanies navigation. It asks whether the demand for reliable control *causes* field formation. It also makes a useful prediction: spatial unimodality need not be universal. Multiple physical locations could legitimately share an identity if they support equivalent control, while one location could require different identities when history changes what can happen next.

**Where it came from:** The [contextual-landmark design](contextual_landmark_state_design.md) identifies transitionally coherent contextual states as the transferable requirement. The [recent batch audit](../06_experiments/recent_batches_design_audit_20260906.md) separates broad, ambiguous fields from commanded control. The [late spatial outliers](../06_experiments/late_training_outliers_20260908.md) suggest that policy gradients can help spatial differentiation, with an effect that depends on the goal interface.

**Evidence boundary:** The outlier's localized fields cluster in one corner, and its target-versus-shuffled performance remains near chance. Current results motivate the question; they do not show that reliable control produced the fields, or that localized fields are sufficient for control.

**Proposed decisive test:** Compare representation learning under a genuinely command-specific objective with a matched passive or eventual-hit objective. Evaluate field formation on common observations and test frozen controllers under paired goal interventions from matched starting states. Separately use fixed localized landmarks as a diagnostic to distinguish a deficient controller from deficient learned identities. Privileged spatial labels would belong only to that explicitly labeled diagnostic.

**What would weaken the thesis:** Control improves without any increase in transition coherence, or fields form equally well when the control-learning signal is removed. A spatial change alone would not establish the proposed mechanism.

## 2. A place field may be a contextual state, not a visual detector

**Hypothesis:** Recent sequence memory resolves perceptual aliasing by distinguishing occurrences of the same sensory event that permit different next outcomes.

There are two conceptually different possibilities: DG detects visual events and CA3 context supplies the actual state identity; or memory feeds back into DG so that individual DG events already denote contextual states. This distinction changes what a “landmark” means and could support a paper about the functional role of recurrent hippocampal feedback.

**Where it came from:** [Contextual-landmark design](contextual_landmark_state_design.md) and the [CA3-feedback predictive-DG proposal](ca3_feedback_predictive_dg_batch.md).

**Proposed decisive test:** First ask whether recent context separates the incompatible successor distributions of a shared event on held-out occurrences. Then compare contextual graph identities with context-conditioned DG, controlling representation capacity. Test repeated visual observations with different histories in both navigation and an abstract state graph. Freeze learned identities for the subsequent control readout.

**Evidence boundary:** Adding feedback or improving passive outcome prediction does not establish controllability. Context can also fragment one useful state into many route-specific identities. The key outcome is reduced ambiguity with reusable control, not simply more states or cleaner maps.

## 3. Representation and controller can settle into a narrow local solution

**Hypothesis:** Joint learning can produce a self-reinforcing loop: the representation becomes useful for a restricted behavior, that behavior repeatedly samples a small part of the environment, and the resulting data further specialize the representation.

This could explain why locally differentiated fields or increased relative selectivity coexist with weak broad exploration. A separate paper could study when reciprocal adaptation produces transferable skills and when it produces a narrow solution.

**Where it came from:** The [late spatial outliers](../06_experiments/late_training_outliers_20260908.md), the [late target-hit-lift audit](../06_experiments/late_target_hit_lift_audit_20260908.md), and the [explicit DG–controller update contract](explicit_dg_controller_update_contract.md).

**Proposed decisive test:** Use pre-specialization and late checkpoints to compare joint learning, freezing one component, and common replayed experience. Evaluate on held-out starts and regions. Test whether the apparent specialization persists when observation coverage is held fixed, and whether either component transfers to a separately trained counterpart.

**Evidence boundary:** The corner-field outlier and the sustained C05 selectivity candidate are different runs. Neither establishes an encoder–controller equilibrium. The C05 candidate lacks the legacy activation counters needed to resolve its lift increase; large lift spikes elsewhere were mostly denominator artifacts. These observations motivate a hypothesis, not a demonstrated dynamical phase transition.

## 4. Learn sparse events as reliable addresses for sequence memory

**Possible paper thesis:** The purpose of a learned landmark is to provide a reliable cue for writing and retrieving experience.

An event can be visually distinctive yet be a poor memory address if its repeated occurrences mix incompatible routes. Conversely, an event need not encode all sensory detail to retrieve the right continuation. This gives representation learning a concrete functional target: coherent retrieval and useful route composition.

**Where it came from:** [Fundamental contribution directions](ml_paper_fundamental_contribution_directions.md), especially memory-address learning and sparsity protecting route memory.

**Proposed decisive test:** Compare event encoders at matched capacity and activity levels, measuring retrieval purity, interference between routes, composition of held-out routes, and downstream transfer. Vary sparsity independently to ask whether its benefit comes from protecting memory rather than merely reducing activity.

**Evidence boundary:** This is an earlier conceptual proposal, not an established outcome of the recent batches. It connects naturally to idea 1: memory needs coherent continuations and control needs coherent consequences, but these requirements should be compared rather than assumed equivalent.

## 5. Intrinsic motivation can improve the agent's usable knowledge

**Hypothesis:** An agent can seek experience because it improves the reliability and reuse of its internal transition knowledge, even when that experience is no longer novel.

Repeatedly verifying a difficult transition could be intrinsically valuable if it makes later behavior dependable. This suggests a general exploration principle for physical environments, web workflows, and abstract graphs: improve what the agent can predict or deliberately accomplish.

**Where it came from:** Representation-utility maximization in [fundamental contribution directions](ml_paper_fundamental_contribution_directions.md), qualified by the false-positive connectivity results in the [recent batch audit](../06_experiments/recent_batches_design_audit_20260906.md).

**Proposed decisive test:** Construct a setting where novelty favors distracting observations while repeated experience improves a reusable transition. Compare held-out prediction or control improvement with novelty-based exploration. Start from one measurable utility, rather than a sum of connectivity, diversity, and collapse penalties.

**Evidence boundary:** More edges, coverage, or option completions are insufficient utility measures: the existing system can increase them without commanded control. Any learned internal proxy must be checked against independent behavioral outcomes.

## 6. Observed transitions and controllable transitions are different knowledge

**Possible paper thesis:** A useful cognitive graph must distinguish “this happened after that” from “the agent can make this happen.”

The batch designs exposed how eventual target hits and broad event identities can generate apparently reliable edges even when commands barely affect outcomes. This could become a general methodological or learning-objective paper about establishing causal control over learned abstractions.

**Where it came from:** The [recent batch audit](../06_experiments/recent_batches_design_audit_20260906.md), [contextual-landmark design](contextual_landmark_state_design.md), and [late lift audit](../06_experiments/late_target_hit_lift_audit_20260908.md).

**Proposed decisive test:** Hold starting state and relevant history fixed, intervene on the requested goal, and compare first distinct outcomes with a matched baseline. Include attainable alternatives and report absolute success as well as goal specificity. A first-outcome objective is a candidate mechanism; changing the outcome definition alone does not make an impossible command learnable.

**Evidence boundary:** Online shuffled activation diagnostics are not paired behavioral interventions. Spatially valid endpoints do not make a graph causal. This direction could also supply the evaluation foundation for idea 1 rather than requiring its own paper.

## 7. Compose useful routes from weak exploration

**Hypothesis:** Local sequence knowledge can support routes that the behavior policy has never traversed end to end, overcoming the bias of experienced temporal distance toward the current policy.

**Where it came from:** The weak-policy counterfactual-recovery proposal in [fundamental contribution directions](ml_paper_fundamental_contribution_directions.md) and the distinction between occupancy and reachability in the [hyperbolic CRL/HRL bridge](hyperbolic_crl_hrl_bridge.md).

**Proposed decisive test:** Collect the same weak-policy data for all methods, ensure the necessary local transitions are observed, and withhold full route combinations. Test composed route predictions and actual execution against direct temporal-distance estimates and a simple explicit graph baseline.

**Evidence boundary:** Composition cannot recover an unobserved connection without additional assumptions, and execution still needs a capable local controller. Hyperbolic geometry is an optional representation hypothesis, not the contribution by itself.

## 8. Suppressive learning may allocate events rather than simply sparsify them

**Hypothesis:** Feedback that repels overused event detectors could improve how limited representational capacity is allocated, but only when coupled to useful temporal or behavioral feedback.

**Where it came from:** The suppressive address-allocation proposal in [fundamental contribution directions](ml_paper_fundamental_contribution_directions.md) and the [threshold-rotation toy report](../06_experiments/threshold_rotation_toy_report.md).

**Proposed decisive test:** Connect controlled toy inputs to sequence retrieval and compare allocation, route interference, and transfer while separating normalization and temporal-feedback effects.

**Evidence boundary:** The toy report already rejects a simple universal density explanation: punishment often produces sparser activity and weaker alignment to latent directions, with distribution- and threshold-dependent exceptions. A useful paper would explain the interaction that improves memory or behavior; sparsity alone would not establish it.

## How to develop this collection

Start with **idea 1** as the leading spin-off question. Ideas **2 and 3** are competing or complementary explanations to investigate; **idea 6** supplies essential evidence about control. Ideas **4 and 5** preserve broader functional interpretations. Ideas **7 and 8** remain separate, more speculative directions. This ordering reflects the user's stated interest and conceptual fit, not a claim about publication readiness.

Promote an idea into a dedicated proposal when it has a clear causal claim, a minimal discriminating experiment, and a checked relationship to prior work. Define any actual training study through the [standardized study workflow](../04_implementation/standardized_study_workflow.md); the tests above are conceptual sketches, not submission matrices.

For future additions, record the originating discussion or design note, the hypothesis, the strongest supporting and opposing evidence, and the smallest test that could change our mind. Keep historical inspiration separate from the latest empirical status.

### Reusable experience from this collection

Existing design notes were the fastest source of mechanisms; the latest dated audits were authoritative for what the batches actually support. Targeted heading and phrase searches with `rg`, followed by bounded reads of linked reports, were more useful than broad concatenated reads, which produced truncated output. No new training, telemetry collection, or literature search was needed. Next time, start here and update evidence from newer reports rather than re-inventorying runs or treating an older proposal as a result.
