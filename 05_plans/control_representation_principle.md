# Control as a Principle for Representation

Date: 2026-09-08. Mathematical development of [Future Scope and Research Inspirations](future_scope_and_inspirations.md), with provenance from earlier discussions and implemented designs.

**Prior-art update:** [Targeted literature review](../08_literature/control_relevance_fisher_place_fields_20260908.md) identifies close precedents for control-based spatial representations, reward-driven field reorganization, task-dependent Fisher precision, and empowerment-based representation learning. The squared-gradient/policy-Fisher proportionality below remains a conjecture; the Fisher identities are standard mathematics. Lin is the immediate architectural predecessor, not the first control-based account of spatial representation.

## Why this could be a deeper continuation of Lin et al.

The user's motivation is explanatory: control might determine why a nervous system needs particular distinctions in the first place. A field would be valuable because it makes action reliable, with spatial tuning emerging when that requires distinguishing locations. This could advance the Lin storyline beyond another source of training reward.

[Lin, Yiu & Leibold](https://arxiv.org/html/2510.09951v3) demonstrate that sparse sensory input and a fixed sequence generator support rewarded navigation, with spatial fields and task-dependent remapping emerging during learning. Their DG projection and downstream controller are trained; the visual trunk is fixed. Their main navigation task uses a fixed rewarded destination, rather than an explicit goal input. The paper discusses predictive and other accounts; its control task supplies a functional setting for studying the emergence of representation.

The proposed continuation is our interpretation, not a conclusion already isolated by that paper:

> Place fields emerge because reliable behavior requires preserving some distinctions and discarding others. Control shapes both the representation and the experience from which that representation is learned.

That is why the idea has value beyond “add a control loss.” It could explain which states separate, which states merge, why fields remap with task demands, and when temporal memory can substitute for explicit geometry. We should describe Lin as the immediate step in this research lineage, without claiming historical priority for all control-based explanations of place fields.

## Notation and status of the equations

For the current design discussed here, sensory features enter the DG encoder, and the controller's state is the CA3 memory of those DG representations:

$$
z_t=f_\theta(x_t),\qquad
h_t=\mathcal M(h_{t-1},z_t),\qquad
s_t=h_t,\qquad
\pi_\phi(a_t\mid h_t,g).
$$

For the fixed sequence generator, a schematic recurrence is

$$
h_t=A h_{t-1}+B f_\theta(x_t).
$$

Here $A$ propagates the trace and $B$ injects DG activity. The control state does not concatenate raw sensory features with memory. Current sensory information reaches it through the current DG injection. The goal conditions the policy, not the visual DG identity. Optional historical bypasses and CA3-to-DG feedback variants are separate architectural factors, not part of this baseline notation.

Thus $z_t$ is a sensory event code, while $h_t$ is the control representation. Memory contains a transformed history of DG events, not a separate store of raw observations or an assumed path integrator. Throughout the control quantities below, $s$ means $h$. Equal memory states can still conceal different environmental situations, so matching $h$ alone does not guarantee identical physical starts; interventions must control the relevant underlying trial conditions as well.

Let $g$ denote a desired outcome, $a$ a primitive action, and $u$ an intervention: either an action or a commanded subgoal executed by a fixed controller. Let $Y$ be an outcome at a specified horizon or the first distinct landmark plus timeout. For causal comparisons, draw $u\sim\rho(u\mid s)$ independently of uncontrolled differences between trials at matched $s$. Write

$$
P_s^u(y)=P(Y=y\mid S=s,\operatorname{do}(U=u)).
$$

All information quantities below depend on this explicitly chosen distribution, controller, horizon, and outcome vocabulary. Hold the outcome definition fixed for comparisons; if the learned encoder can redefine success, it can improve a score by changing labels.

The equations below are **new formalizations of recovered ideas**, except where an existing design equation is explicitly identified. They are alternatives and diagnostics, not a proposal to combine every term into one loss. The user's remembered sentence about maximal differences was not recovered verbatim in the inspected tasks; its closely related mechanisms were.

## 1. From policy entropy to differences that matter

The user clarified that the intended left-hand side was a **squared gradient**, a measure of local representational sensitivity. For a scalar unit $f_i$, the natural scalar is $\|\nabla_x f_i(x)\|_2^2$; for the full DG population, sum across units:

$$
\boxed{
\mathcal S_{\mathrm{DG}}(x)
=\sum_i\|\nabla_x f_{\theta,i}(x)\|_2^2
=\|J_f(x)\|_F^2.
}
$$

The closest faithful correction of the original intuition is therefore

$$
\mathcal S_{\mathrm{DG}}(x_t)
\overset{\mathrm{hypothesis}}{\propto}
H[\pi_\phi(\cdot\mid h_t,g)].
$$

Both sides are scalars, and the policy now depends on DG-derived memory. This repairs the mathematical form, but entropy alone remains an imperfect indicator of useful sensitivity: a random policy can have high entropy everywhere, while a competent near-deterministic policy may depend critically on precise sensory distinctions.

A closer connection to the intended **control relevance** is to compare this squared sensitivity with an input Fisher sensitivity of the policy, developed in section 2. This retains the user's squared-gradient idea while distinguishing uncertainty from the behavioral effect of perturbing sensory evidence.

### Different goals should make different actions appropriate

At a fixed contextual state define

$$
\bar\pi(a\mid s)=\sum_g\rho(g\mid s)\pi(a\mid s,g),
$$

$$
\begin{aligned}
\mathcal D_{\mathrm{action}}(s)
&=\sum_g\rho(g\mid s)
D_{\mathrm{KL}}\!\left(\pi(\cdot\mid s,g)\Vert\bar\pi(\cdot\mid s)\right)\\
&=H[\bar\pi(\cdot\mid s)]
-\mathbb E_g H[\pi(\cdot\mid s,g)]\\
&=I(G;A\mid S=s).
\end{aligned}
$$

This is high when each goal gives a confident action choice but different goals give different choices. It is zero when every goal produces the same random policy. With two equally likely goals choosing opposite deterministic actions, it is one bit; if both use the same fair coin, it is zero.

This formalizes a decision point. It does **not** establish that the chosen actions achieve different outcomes. It is also not a new information-theoretic concept: [InfoBot](https://arxiv.org/abs/1901.10902) identifies decision states through goal dependence of behavior. The proposed contribution would concern how control-dependent information requirements cause field formation in the DG/CA3 system.

### Different commands should make different outcomes happen

Define $\bar P_s=\sum_u\rho(u\mid s)P_s^u$. Then

$$
\boxed{
\mathcal D_{\mathrm{outcome}}(s)
=\sum_u\rho(u\mid s)D_{\mathrm{KL}}(P_s^u\Vert\bar P_s)
=I(U;Y\mid S=s).
}
$$

Equivalently,

$$
\mathcal D_{\mathrm{outcome}}(s)
=H(\bar P_s)-\sum_u\rho(u\mid s)H(P_s^u).
$$

The first term rewards distinguishable possibilities; the second subtracts variability remaining under a fixed command. A random outcome independent of the command gives zero. Several commands reliably reaching different outcomes give a large value. All commands reliably reaching the same sink also give zero.

This is the closest formulation of **“a state is interesting when actions/subgoals make maximal differences.”** Maximizing over the intervention distribution gives channel capacity,

$$
\mathcal E(s)=\max_\rho I(U;Y\mid S=s),
$$

an empowerment-style quantity. [Empowerment](https://arxiv.org/abs/1310.1863) already defines control through action-to-sensory channel capacity. For our comparisons, a fixed balanced $\rho$ may be more revealing because optimization can ignore difficult commands. Goal-channel information can also be high if goal names are systematically permuted, so commanded success $P(Y=g\mid\operatorname{do}(G=g),s)$ remains a separate test.

## 2. A state distinction is worth keeping when merging it loses control

Large differences between commands at one state do not by themselves explain why two states need separate representations. If every location has the same control consequences, the agent might have high control capacity everywhere while needing no location code.

The complementary question is:

> Would confusing these two situations force the agent to choose the wrong action?

For a representation cell $B$, let $Q_g(s,a)$ be expected return from taking $a$ and then following a specified continuation policy for goal $g$. Using the same conditional state distribution in both terms, define

$$
\boxed{
\mathcal R(B,g)
=\mathbb E_{s\mid B,g}\left[\max_a Q_g(s,a)\right]
-\max_a\mathbb E_{s\mid B,g}\left[Q_g(s,a)\right]
\geq 0.
}
$$

The first term permits a different action for each state; the second forces one shared action. Their difference is the immediate decision value of resolving that ambiguity, with continuation held fixed. It is not a full retraining or long-horizon abstraction-error theorem.

For two equally likely states, consider:

| State | Left | Right |
|---|---:|---:|
| $s_1$ | 1 | 0 |
| $s_2$ | 0 | 1 |

A merged identity achieves at most 0.5; separate identities achieve 1. Thus $\mathcal R=0.5$. If Left is best in both states, $\mathcal R=0$, even if their images differ greatly. This is a concrete explanation for allocating representational capacity according to behavioral need.

A complementary outcome-based distance is

$$
d_{\mathrm{ctrl}}(s,s')
=\mathbb E_{u\sim\rho}\operatorname{JS}(P_s^u,P_{s'}^u),
$$

using a common intervention set and weights. A coherent identity should avoid merging states with incompatible consequences under the same intervention. Outcome differences can exist without changing the best action, so this criterion is stronger than decision regret in some tasks.

### Squared gradients, Fisher information, and the DG-to-memory path

A squared encoder gradient is not automatically Fisher information. Fisher information is an expected squared **log-probability gradient**. Its usual score definition is summarized in [Ly et al., A Tutorial on Fisher Information](https://arxiv.org/abs/1705.01064). Here the differentiation variable is sensory input, not the trainable parameters.

For a fixed prior memory and fixed goal, define the input-indexed action distribution

$$
p_x(a)=\pi_\phi\!\left(a\mid\mathcal M(h_{t-1},f_\theta(x)),g\right).
$$

Its Fisher matrix and scalar trace are

$$
F_x^\pi
=\mathbb E_{a\sim p_x}
[\nabla_x\log p_x(a)\,\nabla_x\log p_x(a)^\top],
$$

$$
\boxed{
\mathcal S_\pi(x;h_{t-1},g)
=\operatorname{tr}F_x^\pi
=\mathbb E_{a\sim p_x}\|\nabla_x\log p_x(a)\|_2^2.
}
$$

This asks: **how strongly would a small change in sensory evidence change the policy through DG and its memory?** For smooth distributions with common support, in nats,

$$
D_{\mathrm{KL}}(p_x\Vert p_{x+\delta})
=\tfrac12\delta^\top F_x^\pi\delta+o(\|\delta\|^2).
$$

The scalar trace averages sensitivity over input directions; a specific unit direction $v$ has sensitivity $v^\top F_x^\pi v$. These quantities depend on input coordinates and scaling, so comparisons require the same normalized feature space.

Define $F_h^\pi=\mathbb E_a[\nabla_h\log\pi(a\mid h,g)\nabla_h\log\pi(a\mid h,g)^\top]$ and $K_t=\partial h_t/\partial x_t$. The chain rule gives

$$
K_t=BJ_f(x_t),\qquad
\boxed{
F_{x_t}^\pi
=J_f(x_t)^\top B^\top F_{h_t}^\pi B J_f(x_t).
}
$$

This is the exact local relation for the stated linear-memory architecture, wherever derivatives exist. It shows why raw DG sensitivity and control sensitivity differ: an encoder direction can vary strongly while the controller ignores its memory trace. For an earlier input, holding other observations fixed,

$$
\frac{\partial h_t}{\partial x_{t-k}}
=A^kB J_f(x_{t-k}).
$$

Consequently, a sensory event can matter through its later trace even if it has little immediate effect on action. This derivative follows the recorded observation sequence; it does not include changes to future observations caused by changed actions.

The proposed scalar principle closest to the user's intent is

$$
\boxed{
\sum_i\|\nabla_x f_{\theta,i}(x)\|_2^2
\overset{\mathrm{hypothesis}}{\propto}
\mathbb E_{h_{t-1},g\mid x}\,
\mathbb E_{a\sim p_x}
\|\nabla_x\log p_x(a)\|_2^2.
}
$$

The context/goal average matters: a visual-only encoder assigns one sensitivity to $x$, even if its control significance varies across histories and tasks. Interpret this as **allocate sensory resolution where it supports behavioral distinctions**, under fixed scale and capacity. It is a normative hypothesis, not an identity or a ready-made loss. The current policy already depends on the encoder, so fitting the two sides jointly could yield zero sensitivity on both sides or amplify useless sensitivity. A competent frozen reference policy, or independent outcome/value diagnostics, is needed to test usefulness rather than merely current dependence.

For a direct statistical interpretation of the encoder itself, one can introduce an explicitly hypothetical noisy code:

$$
\widetilde Z\mid x\sim\mathcal N(f_\theta(x),\sigma^2I),\qquad
F_x^{\mathrm{enc}}=\sigma^{-2}J_f(x)^\top J_f(x),\qquad
\operatorname{tr}F_x^{\mathrm{enc}}=\sigma^{-2}\mathcal S_{\mathrm{DG}}(x).
$$

With fixed isotropic noise, squared encoder sensitivity is proportional to encoder Fisher information. This noise model is an explanatory construction, not part of the current implementation.

The stronger behavioral test still asks whether sensitivity preserves **consequences**, not merely changes action probabilities. An analogous Fisher matrix can be defined for fixed-intervention outcome distributions, or one can use the decision-regret and outcome-divergence quantities above. High policy sensitivity can reflect a fragile or wrong controller. Likewise, the goal-information quantities in section 1 measure differences across commands at fixed memory, whereas Fisher sensitivity measures differences across sensory inputs at fixed command; neither replaces the other.

For thresholded DG, gradients exist only away from activation boundaries. Use finite perturbations to test event switches, keep encoder scale fixed, and never infer narrow or useful place fields from a large gradient alone. The proposed comparison is a scientific hypothesis about allocation of representation, not a change to training.

## 3. Same state plus same intervention should have coherent consequences

This was recovered directly from **feedback to DG** and the [contextual-landmark design](contextual_landmark_state_design.md). The [predictive-DG implementation plan](ca3_feedback_predictive_dg_batch.md) operationalizes part of it through source-state outcome prediction.

For an occurrence assigned identity $Z=j$, recent context $C$ should not still reveal major differences in the outcome of the same command if $j$ is intended to be the complete control state:

$$
\mathcal A(j)=I(Y;C\mid Z=j,U).
$$

High $\mathcal A$ indicates unresolved contextual ambiguity. A candidate remedy in a feedback variant is $Z=f_\theta(X,C)$; the baseline remains $Z=f_\theta(X)$. If CA3 already resolves the ambiguity downstream, a visual DG identity need not satisfy this condition by itself; the criterion should then be applied to the combined state.

The earlier chat explicitly proposed the held-out diagnostic

$$
\Delta_{\mathrm{state}}
=\mathrm{CE}(Y\mid U)-\mathrm{CE}(Y\mid Z,U).
$$

With Bayes-optimal predictors under one distribution, this equals $I(Y;Z\mid U)$. For finite fitted predictors it is a measured predictive improvement, not an exact information estimate. It asks whether state contributes beyond the command. In contrast, $I(U;Y\mid Z)$ asks whether the command contributes beyond state. Both are needed to distinguish useful control state from either a predictor that ignores goals or a predictor that ignores observations.

The CPD PASS/GOAL heads are predictive objectives, not implemented causal-information objectives. Their data were policy-generated. Controlled interventions would be needed for the stronger reading above.

A compact normative formulation considers an optional abstraction $V=\psi(H)$ of the memory state. This is not a redefinition of the DG encoder. Using a finite or stochastic abstraction to avoid deterministic-continuous mutual-information pathologies, it is

$$
\min_{\psi,q} I(H;V)
\quad\text{subject to}\quad
\mathbb E_{h,u}D_{\mathrm{KL}}\!\left(P_h^u\Vert q(Y\mid \psi(h),u)\right)\leq\epsilon.
$$

Preserve as little memory-state information as possible while retaining the consequences of interventions. This is a proposed abstraction principle, not an added module or a demonstrated objective of the sparse DG. It also clarifies how prediction can serve a control explanation: the object preserved is the consequence of intervening, rather than every predictable sensory detail.

## 4. The policy can make temporal memory spatially informative

Recovered from **Goal conditioned localization explanation**, **long runs & goal factorization**, and **High-level Planner**. This is a second potential paper-level insight: control changes the data distribution, so it can create the conditions under which an otherwise ambiguous temporal code becomes useful.

Let $H_t$ be CA3 history and $P_t$ discretized physical position, used only for evaluation. A landmark age $(j,\tau)$ alone need not localize the agent. Under a consistent controller,

$$
P(P_t\mid j\text{ occurred at }t-\tau,\pi_g)
$$

may become concentrated because the agent repeatedly follows similar routes. The proposed positive feedback loop is

$$
\text{better control}\longrightarrow
\text{more reproducible routes}\longrightarrow
\text{more informative temporal memory}\longrightarrow
\text{better control}.
$$

A diagnostic for goal-dependent localization is

$$
\Delta_{\mathrm{goal\text{-}localization}}
=H(P_t\mid H_t)-H(P_t\mid H_t,G)
=I(P_t;G\mid H_t).
$$

Estimate it on matched spatial coverage, or compare held-out location-probe errors. A policy confined to one corner can make localization trivially easy, so raw entropy reduction is insufficient. If goals switch, current $G_t$ is not necessarily the controller that generated the remembered history; past goals/actions may matter.

The user's single-goal intuition should be retained as **a simplification of the policy-conditioned problem**, not a claim that Lin's environment literally becomes one-dimensional or that one fixed goal guarantees unique routes. The separate heads and persistent commands in the [persistent-control implementation](../06_experiments/persistent_intrinsic_control_implementation.md) are related experimental probes, not evidence that this mechanism has succeeded.

## 5. A measured place field contains both tuning and behavior

Recovered explicitly from **High-level Planner**. With position $p$, heading $\alpha$, memory $h$, and unit activation $z_i$,

$$
M_i^\pi(p)
=\mathbb E_{\alpha,h\sim d_\pi(\alpha,h\mid p)}
[z_i(p,\alpha,h)].
$$

Training can change $z_i$, the sampling distribution $d_\pi$, or both. Thus a more localized map can arise from sharper tuning, more consistent heading/history at each location, or their interaction.

The theoretical opportunity is to explain **representation–behavior co-adaptation** as a mechanism, rather than treating occupancy only as a nuisance. The discriminating measurement is a common-distribution map,

$$
M_i^{\mathrm{ref}}(p)
=\mathbb E_{\alpha,h\sim d_{\mathrm{ref}}(\alpha,h\mid p)}
[z_i(p,\alpha,h)],
$$

using the same observation/action sequences and memory initialization across checkpoints. Contextual encoders require replaying histories, not just shuffling static images. Compare these maps with on-policy maps. The [late spatial outliers](../06_experiments/late_training_outliers_20260908.md) currently provide policy-driven evidence, so they cannot isolate these paths.

## 6. An endogenous target must retain its meaning while it is learned

Recovered from **feedback to DG** and concretely reflected in the [persistent-control implementation](../06_experiments/persistent_intrinsic_control_implementation.md). This is more than an engineering detail: the system must distinguish changing the world from changing the criterion of achievement.

With a moving event detector, an apparent success score is

$$
J(\theta,\phi)=\mathbb E_{\tau\sim\pi_{\phi,\theta}}
\left[\mathbf 1\{f_\theta(X_\tau)=g\}\right].
$$

It can rise because the policy reaches the intended event more often or because $f_\theta$ expands the event to include what the policy already does. With hard labels this is a conceptual decomposition of changes, not an ordinary differentiable reward path.

A fixed reference $\bar f$ makes the achievement test

$$
J_{\mathrm{ref}}(\theta,\phi)=\mathbb E_{\tau\sim\pi_{\phi,\theta}}
\left[\mathbf 1\{\bar f(X_\tau)=g\}\right].
$$

The live representation can improve while the target definition stays fixed. This does not repair a poor reference vocabulary, but it separates learning a skill from relabeling its outcome. The actual reference implementation uses its specified onset semantics; the indicator above abstracts those details.

## 7. A useful push–pull equilibrium needs identifiable control

Recovered from **High-level Planner** and **Minimalist**. The implemented family starts from

$$
r_{\mathrm{enc}}=\beta d,\qquad r_{\mathrm{dec}}=\beta(E-d).
$$

The inspiration is reciprocal pressure between making meaningful distinctions and learning efficient transitions. But a flat decoder optimizing time until *some* event solves a different problem from reaching a specified event:

$$
\min_\pi\mathbb E[T_{\mathrm{any}}]
\quad\text{versus}\quad
\min_\pi\mathbb E[T_g\mid g].
$$

The former admits short loops. Likewise, if two behaviors induce the same distribution of the interval $D$, any expected reward depending only on that interval agrees:

$$
P_{\pi_1}(D)=P_{\pi_2}(D)
\quad\Longrightarrow\quad
\mathbb E_{\pi_1}[r(D)]=\mathbb E_{\pi_2}[r(D)].
$$

For per-time return, event rate, discounting, and termination must also match; equality of interval distributions alone is an event-level statement. This makes precise the earlier insight that temporal spacing alone cannot identify exploration versus repetition.

The deeper continuation is to ask whether reciprocal learning can produce **distinct controllable consequences and coherent reusable identities**, measured by sections 1–3. This is a sharper target than assuming an antagonistic reward automatically produces a useful equilibrium. The [FIRST/JOINT design](joint_ppo_dg_first_outcome_batch.md) tests parts of the missing structure, but none of the mutual-information or regret objectives above has been implemented by that batch.

## Scientific predictions worth preserving

- **Task-dependent splitting:** locations should separate when merging them incurs decision regret; changing goal demands should alter that need while sensory input is held constant.
- **Legitimate multi-fields:** disconnected positions may share a code when they support equivalent control. Euclidean mono-fields are a possible consequence, not the normative definition.
- **Context-dependent remapping:** identical observations should separate when history changes the effect of the same intervention, unless downstream memory already preserves the distinction.
- **Control-induced localization:** temporal history may become more informative as routes stabilize, even with the encoder frozen. Matched coverage distinguishes this from confinement.
- **Reliable alternatives:** a random policy and a universal sink can both have poor command-to-outcome information for different reasons; successful learning should separate commands and stabilize their consequences.

These support a leading thesis: **representational distinctions emerge where confusing situations would compromise reliable control; behavior can simultaneously make those distinctions easier to infer from sparse temporal evidence.** This is the explanatory advance to investigate, not a claim already established by the current batches.

## Recovered discussion provenance and reusable workflow

The following task titles were retrieved verbatim. IDs allow future retrieval through `read_thread`; the summaries below are paraphrases, not quotations of every original proposal.

| Task | ID | Idea recovered |
|---|---|---|
| High-level Planner | `01a07c21-7ece-7052-83f3-0a8b4afba664` | Trajectory shaping versus control gradients; projected rate maps; missing target in the original push–pull argument |
| feedback to DG | `01a0794f-bc4f-71e1-8b07-6b534e9b93a4` | Same state/intervention, coherent outcomes; source predictive gain; goal-independent landmark identity |
| Goal conditioned localization explanation | `6a9f544e-b328-83eb-9149-ac8d4173aa8b` | Policy-conditioned temporal localization and the route–memory feedback loop |
| long runs & goal factorization | `01a07e51-13b0-7191-b6dc-b8c9da71e6af` | User's fixed-goal versus many-controller hypothesis; retrieved task preview, supplemented by the localization discussion and implementation record |
| Diagnose C15 recruitment criteria | `01a06d12-985f-7922-8edc-b8054d1a8674` | Source identities with predictable controlled consequences; context inconsistency rather than graph degree |
| Minimalist | `01a07bfc-fa7b-7ad1-bb13-7f520b3a01c3` | Interval-only rewards cannot distinguish matching repetitive and exploratory event statistics |

Use this table and the linked implementation notes for future retrieval. Task previews efficiently locate ideas; older discussion turns often precede long implementation logs. Filter retrieved content to user and assistant messages before printing it, and use bounded pages rather than dumping tool histories. Earlier assistant interpretations are inspiration, not authoritative results; dated audits remain the evidence source. The Lin paper and the two closest information-theoretic sources were checked directly for this revision. No new runs or training changes were made.

Notation maintenance: the user clarified that the baseline control state is DG-derived memory, and that the original sensitivity intuition uses squared gradients. Preserve $z=f_\theta(x)$ and $s=h$ when extending this note; label contextual-DG and additional state-abstraction variants separately. Use `$...$` for inline math and `$$...$$` for display math.
