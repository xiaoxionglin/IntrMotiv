# Control-Relevant Representation, Fisher Sensitivity, and Place Fields

Research date: 2026-09-08. Targeted prior-art review for [Control as a Principle for Representation](../05_plans/control_representation_principle.md).

## Finding

**Yes: several parts of the idea have already been expressed, including close mathematical and place-field precedents.** The strongest matches are task-dependent efficient coding, control-oriented information bottlenecks, empowerment-based representation learning, and reward-driven place-field reorganization.

I did not find the exact proposed proportionality between squared DG gradients and policy input-Fisher trace in the inspected sources. This is a bounded search result, not evidence of novelty. The Fisher definition, Gaussian-code identity, and chain-rule transformation are standard mathematics; writing them for DG/CA3 is an application rather than a new theorem. The proportionality itself still needs justification from an objective and resource constraint.

The broad statement that representations should preserve distinctions needed for behavior is established. The possible contribution is a specific explanation and causal test of how sparse sensory events and sequence memory acquire the distinctions required for reliable endogenous control.

## Closest papers, ordered by relevance

### 1. Kumar, Bordelon, Zavatone-Veth & Pehlevan — ICML 2025

[A Model of Place Field Reorganization During Reward Maximization](https://proceedings.mlr.press/v267/kumar25a.html).

This is the closest place-field precedent. Gaussian radial-basis fields have learned centers, widths, and amplitudes; an actor–critic uses their activity. Reward-driven updates reorganize the fields and account for reward-site accumulation, backward elongation, and representational drift. The paper also compares reward maximization with successor and metric representation objectives.

**Overlap:** explains spatial representation through behavioral optimization. **Difference:** begins with a spatial Gaussian field parameterization. Our proposed question concerns fields emerging from egocentric features through sparse DG and sequence memory, with no external task reward. That difference must yield an explanatory result, not merely a different architecture.

Read first for the neuroscience positioning. Publisher abstract and indexed paper passages were inspected; direct full-PDF retrieval was blocked during this review, so detailed derivation comparison remains outstanding.

### 2. Bastankhah, Broderick & Eysenbach — 2026

[Learning to Perceive the World Through Control: Empowerment-Based Representation Learning](https://arxiv.org/html/2605.30656v1). The [author project page](https://mahsa-bastankhah.github.io/MISL/) identifies it as ICML 2026; arXiv v1 was posted May 28, 2026.

This is the closest reward-free conceptual precedent. It studies forward and backward representations induced by empowerment optimization, with invariance to control-irrelevant factors under its assumptions. Active interaction is central to the argument.

**Overlap:** control itself organizes representation and data collection. **Difference:** the paper does not establish a DG/CA3 place-field mechanism or the proposed Fisher-allocation law. Its invariance guarantees should not be transferred to our partially observed, finite-memory system without checking assumptions. This paper makes a broad claim that “control-driven interaction learns useful representation” insufficient as our novelty statement.

### 3. Schaffner et al. — Nature Human Behaviour 2023

[Sensory perception relies on fitness-maximizing codes](https://www.nature.com/articles/s41562-023-01584-y).

This is the closest empirical/normative match to allocating sensory precision according to decision consequences. The study changes reward structure while keeping stimulus statistics fixed, and examines human behavior, neural responses, and constrained networks. Encoding precision is expressed through Fisher information:

$$
J(s)\propto k\,p(s)^q.
$$

Notation is adapted: $p(s)$ is stimulus frequency, $k$ a resource scale, and the exponent depends on the objective. The important match is that the cost of errors changes the allocation of precision. It does not derive squared DG sensitivity proportional to policy Fisher. This is a strong methodological template: manipulate behavioral stakes while holding sensory exposure constant.

### 4. Pensia, Jog & Loh — 2019 preprint

[Extracting robust and accurate features via a robust information bottleneck](https://arxiv.org/abs/1910.06893), especially section 2.2 of the [full text](https://arxiv.org/pdf/1910.06893).

A direct mathematical precedent uses input-parameterized Fisher sensitivity of a stochastic feature:

$$
\Phi(Z\mid X)
=\mathbb E_{x,z}\|\nabla_x\log p(z\mid x)\|_2^2.
$$

The objective penalizes sensitivity while preserving predictive relevance. The direction matters: useful representations should discard unnecessary sensitivity, not maximize gradients indiscriminately. This paper supplies established language for the user's squared-gradient intuition. It studies supervised feature robustness, not sequential control or hippocampal fields. Our fixed-noise Gaussian identity is a specialization of this statistical framework, not a novel information measure.

### 5. Pacelli & Majumdar — RSS 2020

[Learning Task-Driven Control Policies via Information Bottlenecks](https://arxiv.org/abs/2002.01428), [full text](https://arxiv.org/pdf/2002.01428).

This is a close architectural and objective-level precedent. A recurrent representation is updated from observations, and actions depend on that representation. Their objective trades expected control cost against information retained about the underlying state. In adapted notation:

$$
\min_{q,\pi}\;
\beta\,\mathbb E[C(\tau)]+\sum_t I(X_t;Z_t),
\qquad
q(z_t\mid z_{t-1},o_t),\quad \pi(a_t\mid z_t).
$$

Here $X_t$ means underlying state in their paper, not our frozen visual features. **Overlap:** learn only what control needs and retain it in recurrent state. **Difference:** a learned recurrent bottleneck and external task costs, rather than fixed CA3 dynamics and learned sparse events.

### 6. Achille & Soatto — 2017 preprint

[A Separation Principle for Control in the Age of Deep Learning](https://arxiv.org/abs/1711.03321).

The paper explicitly proposes a finite-complexity state representation retaining the information needed for control while discarding nuisance variability, including the dynamic setting and memory. This is a conceptual antecedent of $s_t=h_t$: the agent need not reconstruct every property of the world to act well.

It does not identify the specific DG/CA3 mechanism or derive our sensitivity proportionality. This entry is based on the primary abstract, not a full theorem audit. It is sufficient to establish that minimal memory representations for control are an existing research direction.

### 7. Goyal et al. — InfoBot, ICLR 2019

[InfoBot: Transfer and Exploration via the Information Bottleneck](https://arxiv.org/abs/1901.10902).

InfoBot identifies decision states through the dependence of behavior on the goal. The quantity used in our note,

$$
I(G;A\mid S)
=\mathbb E_{s,g}D_{\mathrm{KL}}
[\pi(\cdot\mid s,g)\Vert\bar\pi(\cdot\mid s)],
$$

belongs directly to this line. Their bottleneck discourages unnecessary goal dependence while identifying states where it matters; it is not simply a prescription to maximize goal dependence everywhere. This is the closest precedent to “different subgoals require different actions here.” Outcome controllability and place-field formation remain different questions.

### 8. Rakelly, Gupta, Florensa & Levine — NeurIPS 2021

[Which Mutual-Information Representation Learning Objectives are Sufficient for Control?](https://proceedings.neurips.cc/paper/2021/hash/dd45045f8c68db9f54e70c67048d32e8-Abstract.html). The [authors' explanation](https://bair.berkeley.edu/blog/2021/11/19/mi-sufficiency-analysis/) details the counterexamples and assumptions.

This is essential for evaluating our proposed information objectives. Their analysis shows that some inverse-information and passive temporal-information objectives admit representations insufficient for optimal control. A traffic-light example is revealing: action prediction can succeed while discarding a light the agent cannot influence but must observe to choose correctly.

**Implication:** distinguish “features the agent can change” from “features needed to act well.” Their offline, fully observed analysis does not directly refute every interactive empowerment formulation; it identifies a sufficiency question our endogenous goals and memory must address explicitly.

### 9. Vijayabaskaran & Cheng — PLOS Computational Biology 2022

[Navigation task and action space drive the emergence of egocentric and allocentric spatial representations](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010320).

Deep RL from visual input produces different spatial representations depending on task and action space, including place-like and egocentric vector-like units. This directly supports investigating how behavioral demands select a representation rather than assuming one spatial code for every task.

It also means the general claim that navigation learning generates task-dependent spatial codes predates Lin. The distinct question for this project is why sparse DG events plus a fixed temporal memory yield particular fields, and how controlled interventions separate learning gradients from changed experience.

### 10. Gustafson & Daw — PLOS Computational Biology 2011

[Grid Cells, Place Cells, and Geodesic Generalization for Spatial Reinforcement Learning](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002235).

The authors evaluate spatial codes as bases for value-function learning. Navigation-relevant geometry follows paths around obstacles, so generalization should respect geodesic rather than merely Euclidean distance. This is an early normative argument that the downstream RL problem constrains spatial representation.

It analyzes suitable spatial basis functions rather than deriving sparse visual DG fields from intrinsic control. Its lesson for our theory is nevertheless direct: representational similarity should be judged against behavioral consequences, not just physical proximity.

### 11. Li, Walsh & Littman — ISAIM 2006; Zhang et al. — ICLR 2021

[Towards a Unified Theory of State Abstraction for MDPs](https://thomasjwalsh.net/pub/aima06Towards.pdf) formalizes abstraction schemes with different preservation guarantees, including value- and policy-related distinctions. Our merge-regret quantity is a decision-theoretic diagnostic consistent with this tradition, not a newly discovered abstraction principle. Preserving a current action choice is weaker than preserving what learning and planning need.

[Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742) uses bisimulation-based behavioral similarity to learn representations from rich observations. This is a close precedent for comparing same-action consequences across states. Bisimulation includes reward and recursive transition structure; our finite-horizon outcome JS divergence is not automatically equivalent to it. The ICLR paper's primary abstract and publication record were inspected; no equivalence of guarantees is claimed.

## What the research changes about our equation

### Attribution of all the proposed control formulations

The user's follow-up clarified that the question concerns the formulations introduced during our discussion, not just the squared-gradient equation. **Most of those formulations are established quantities or close adaptations of existing frameworks.** “New formalization” in the companion note means newly written for this project, not mathematically original.

| Formulation in our discussion | Closest established formulation | Assessment |
|---|---|---|
| $\mathcal D_{\mathrm{action}}(s)=I(G;A\mid S=s)$ | InfoBot's conditional goal–action information and divergence from a goal-marginalized policy | Direct mathematical match after choosing the averaging distribution. InfoBot regularizes unnecessary dependence; it does not maximize it everywhere. |
| $\mathcal D_{\mathrm{outcome}}(s)=I(U;Y\mid S=s)$ | The intervention/skill-to-outcome information underlying empowerment | Same information-theoretic structure. With a fixed intervention distribution this is mutual information; maximization over that distribution gives channel capacity. |
| $\mathbb E[\max_a Q]-\max_a\mathbb E[Q]$ | Expected value of perfect information, conditional on the currently merged state cell | A decision-theoretic value-of-information expression applied to state distinctions, not a new regret principle. |
| $d_{\mathrm{ctrl}}(s,s')=\mathbb E_u\operatorname{JS}(P_s^u,P_{s'}^u)$ | Behavioral state abstraction and bisimulation | A related finite-horizon diagnostic. It is not identical to reward-aware recursive bisimulation and inherits none of its guarantees automatically. |
| $\min I(H;V)$ subject to small behavioral/predictive distortion | Rate–distortion and control-oriented information bottlenecks | An adaptation of established compression-versus-performance principles. |
| $I(Y;C\mid Z,U)$ and $\mathrm{CE}(Y\mid U)-\mathrm{CE}(Y\mid Z,U)$ | Conditional sufficiency and conditional predictive information | Standard measures applied to contextual landmark ambiguity and source-state usefulness. Cross-entropy gain equals conditional MI only for the appropriate Bayes-optimal predictors. |

Primary matches: [InfoBot, equations 1–2](https://arxiv.org/pdf/1901.10902), [empowerment-based representation learning](https://arxiv.org/html/2605.30656v1), [state-abstraction theory](https://thomasjwalsh.net/pub/aima06Towards.pdf), and [control information bottlenecks](https://arxiv.org/pdf/2002.01428).

Two additional sources sharpen the attribution. [Abel et al., State Abstraction as Compression in Apprenticeship Learning, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4179) explicitly develops abstraction as a trade-off between compression and performance using rate–distortion and information bottlenecks. [Information Density in Decision Analysis](https://pubsonline.informs.org/doi/10.1287/deca.2022.0465) connects information value to sensitivity of a decision's expected utility. These precedents support interpreting the merge-regret expression as the value of revealing which state one occupies, with the continuation policy fixed.

For the especially inspirational phrase “a state is worth representing when actions/subgoals make maximal differences,” **InfoBot is closest when “differences” means action choice across goals; empowerment is closest when it means distinguishable consequences across commands.** Value of information is closest when the emphasis is why two situations deserve different representations. These are three complementary questions, not one newly introduced principle. None alone establishes spatial field formation.

The potentially distinctive contribution therefore lies in a causal explanation and measurable prediction for the sparse DG-to-CA3 architecture, not in introducing these quantities. The 2026 empowerment representation paper is already close even to the broader claim that reward-free control objectives organize representation; it must be compared directly.

The proposed expression is

$$
\mathcal S_{\mathrm{DG}}(x)
=\sum_i\|\nabla_x f_{\theta,i}(x)\|^2
\overset{?}{\propto}
\mathbb E_{h,g}\operatorname{tr}F_x^\pi.
$$

There are three different claims here:

1. **Squared encoder sensitivity relates to Fisher information under an explicit noise model.** Established statistical structure.
2. **Representational precision should reflect behavioral relevance under resource limits.** Established normative principle, with direct empirical precedents above.
3. **Optimal precision is linearly proportional to the current policy's input Fisher trace.** Not established by this review or by our previous derivation.

The third claim should be treated as a provisional conjecture, not the strongest result of the idea. The existing policy can be insensitive because it has not learned, or sensitive for the wrong reasons. Input units, encoder scaling, noise, capacity, training distribution, and the chosen loss all affect an allocation law. A scalar trace also discards which directions matter.

### A more defensible mathematical bridge

The following is our synthesis using the standard local Fisher/KL expansion, not an equation attributed to a particular control paper above. Let $p_x$ be a **fixed competent reference policy** evaluated at the same prior memory and goal, and let a small zero-mean sensory perturbation have covariance $\Sigma_x$. With smooth distributions, common support, and natural logarithms,

$$
\mathbb E_\delta D_{\mathrm{KL}}(p_x\Vert p_{x+\delta})
\approx\tfrac12\operatorname{tr}(F_x^\pi\Sigma_x).
$$

This gives a control-dependent distortion cost: noise in some directions changes behavior much more than noise in others. A coding problem can allocate precision to reduce this distortion under a stated resource budget. It does **not** yield one universal proportionality law without specifying that budget and the mapping from code to effective noise. Policy KL still measures behavioral change rather than regret; independent reward or outcome tests must establish that preserving this behavior is useful.

In our baseline, the sensitivity must pass through the actual memory architecture:

$$
z_t=f_\theta(x_t),\qquad h_t=A h_{t-1}+Bz_t,\qquad s_t=h_t,
$$

$$
F_{x_t}^\pi
=J_f(x_t)^\top B^\top F_{h_t}^\pi B J_f(x_t).
$$

This is a chain-rule identity at differentiable points. The research opportunity is to test whether this constrained pathway shapes field formation and which past sensory distinctions remain useful through their traces. A DG gradient with respect to visual features is not automatically a spatial tuning gradient: viewpoint, visual statistics, memory, and occupancy must be accounted for.

## Positioning relative to Lin and the possible spin-off

[Lin, Yiu & Leibold](https://arxiv.org/html/2510.09951v3) remain the immediate architectural predecessor: sparse DG input and a fixed sequence generator support rewarded visual navigation with emergent spatial structure. The research lineage should now acknowledge earlier control-based spatial models and the close 2025–2026 work above.

The motivating intuition remains valuable, but its strongest form is more specific:

> Explain how a resource-limited sensory-event system, whose controller only sees event memory, learns distinctions required for reliable behavior—and when those distinctions appear as localized, contextual, or task-dependent fields.

Possible discriminating contributions, rather than novelty claims already earned:

- **Field emergence without a spatial basis:** explain why control gradients acting on sensory DG features yield fields, and what predicts their extent or splitting.
- **Two causal paths:** isolate representation updates from policy-induced concentration of headings and histories. Neither must be inferred from on-policy field maps alone.
- **Memory-specific allocation:** test whether an event's later control relevance through CA3 predicts where the encoder becomes selective, beyond immediate action sensitivity.
- **Endogenous control with stable meanings:** demonstrate reusable command-dependent outcomes while preventing the event detector from redefining success around existing behavior.
- **A real allocation law:** derive a sensitivity/precision prediction from a specified objective and capacity constraint, then test it against occupancy, predictive relevance, and reward proximity.

These should not all be added as mechanisms. One clean causal result, measured against the closest alternatives, is a stronger paper than combining their terminology.

## Recommended reading sequence and research limits

Read **Kumar 2025**, **Bastankhah 2026**, and **Schaffner 2023** first: they are closest to the place-field explanation, reward-free representation principle, and precision-allocation intuition respectively. Follow with **Pacelli 2020**, **Pensia 2019**, and **Rakelly 2021** to specify the architecture, sensitivity measure, and sufficiency requirements. Use InfoBot for the decision-state interpretation.

This was a targeted primary-source search across neuroscience, control representation, state abstraction, empowerment, and input Fisher sensitivity, including citation-following and a recent-paper check. It is not an exhaustive systematic review. Full-text sections were inspected where accessible; abstract-only and blocked-PDF limitations are identified above. Conference/publication dates were taken from primary records rather than search-engine crawl-age labels. Absence of the exact proportionality in this selection does not establish novelty.

Reusable workflow: search conceptual equivalents in several literatures rather than only the proposed formula. Author project pages and publisher records exposed the closest recent work; primary full-text equations distinguished genuine input-Fisher methods from unrelated parameter-Fisher applications. Some PMC/OpenReview pages blocked access, so publisher pages and arXiv were the efficient alternatives. Next time start with this shortlist and resolve the specific remaining comparison—especially Kumar's learning equations and Bastankhah's assumptions—rather than repeating a broad inventory. No new experiments or training changes were made.
