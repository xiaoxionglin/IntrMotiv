# IntrMotiv: boss briefing and publication plan

Prepared 10 September 2026. Synthesis of saved project reports through 10 September; no new cluster check or experiment analysis. Running/submitted statuses below refer to their dated audits. Publication suggestions are scientific framing, not verified venue deadlines, format rules, or acceptance predictions.

**A two-minute spoken update**

“We have made progress in separating what the model actually learns. We can maintain sparse sensory codes, and some configurations explore much more broadly. We also have spatially differentiated activity, including a few promising localized units. But those three things—exploration, spatial representation, and deliberately reaching a chosen landmark—do not automatically improve together.

“That is the main finding so far. Some of our best explorers have poor landmark fields, while our best localization candidate gives up some coverage. We also audited the apparent late breakthroughs in goal control: several huge improvements were caused by unstable ratios, so we are being more careful about what counts as control.

“The original scientific question is still open: can intrinsic sequence dynamics teach useful sensory landmarks before a task reward is available? The newer hypothesis is that reliable control over diverse destinations may supply the missing pressure on representation. That is a proposal we can test, rather than a result we already have.

“For the paper, the most direct next result is downstream transfer. The matched transfer batch has now passed its preflights and been submitted. It compares learning from scratch with transferring the DG representation or policy. Alongside that, we need fixed-observation representation tests and interventions that change the commanded goal from matched starts.

“My suggestion is to use Bernstein to present the current question and the separation we have found; develop Cosyne around a causal sequence-memory or control-demand experiment; and make the ICLR story depend on replicated utility and mechanism evidence. We have enough for a substantive progress discussion, but the central causal paper result is still missing.”

**What we have actually found**

| Finding | Concrete evidence | Interpretation and boundary |
|---|---|---|
| Sparse, differentiated activity is feasible | Corrected-core C05 terminal probes: all 16 units active in all three seeds; active-map cosine $0.127\pm0.065$, approximately 14.3 distinct peak bins; continuous maps also differentiated | Diversity is not solely a thresholding artifact. However, 12/16, 15/16, and 14/16 units have multiple half-peak connected regions. Peak diversity is not one-place-per-unit identity. |
| Exploration and localization separate | September 9 G_SHARED recent coverage AUC 90.95–92.29 across seeds, but zero mono-field units in all three 200M snapshots | Strong exploration within the studied setting does not establish localized goal identities. These are descriptive observations, not proof of a universal trade-off. |
| There is a partial localization candidate | F_GATE S123: 7/16 mono-field units at 300M, active-map cosine 0.094, 15 distinct peak bins, zero silent units | A checkpoint/seed worth independent inspection. Recent coverage AUC 47.5 versus about 64.9 at 55–75M; neither replication nor stable fixed-panel localization is established. |
| Allowing policy learning to affect DG may help exploration | Saved September 9 JSON shows W_REF_JOINT exceeds STOP coverage in two of three seed pairs at matched 180–200M: +4.97, −0.44, +16.55 for seeds 8/99/123 | The earlier status prose incorrectly said all three; see the [technical briefing](../06_experiments/promising_architectures_technical_briefing_20260910.md). Does not establish improved goal accuracy or the proposed control-based explanation of fields. |
| Nominal target success can be misleading | C05 known edges funnel into only 3–5 qualified destinations; September 8 audit inspected 543 lift-bearing runs and ten full histories | Broad/multi-location target identities and unsupported denominators can create misleading progress signals. |
| Huge lift spikes are not control breakthroughs | CPD ADD_CA3_DIR_PASS S99 at 65–75M: mean logged lift 2,363.6, pooled activation-rate ratio 0.702 | Averaging ratios with zero baselines can reverse the apparent conclusion. Even pooled activation lift is not a commanded-arrival intervention. |

Coverage AUC here is average cumulative visited-cell count within an episode, not a percentage. Recent windows differ in training age; do not use this table as a cross-family performance leaderboard. Online policy-driven field snapshots and independent offline probes are different evidence types.

Sources: [corrected-core field probes](../06_experiments/corrected_core_candidate_place_field_telemetry_20260902.md), [September 9 status](../06_experiments/persistent_intrinsic_control_status_20260909.md), [full-history lift audit](../06_experiments/late_target_hit_lift_audit_20260908.md).

**Claims we can make, and how far away the stronger claims are**

| Proposed claim | Current status | Evidence needed to advance |
|---|---|---|
| In these experiments, sparse activity, spatial differentiation, coverage, and target control are distinct outcomes | Supported descriptive conclusion | Consolidate matched-condition figures and preserve individual seeds and measurement definitions. |
| Reward-free navigation training with a DG/CA3-inspired system yields candidate spatial codes | Supported in a restricted, descriptive sense | Describe them as candidate codes; do not attribute their origin specifically to sequence feedback yet. The visual trunk is pretrained, not learned from scratch without supervision. |
| Structured sequence feedback causes useful landmark formation | Open central hypothesis | Encoder/policy component controls and disrupted-sequence control; occupancy-matched and fixed-observation representation tests. |
| Intrinsic pretraining improves later task learning | Direct test now submitted; outcome pending | Matched-budget transfer curves, replication, and comparison with equally budgeted alternative pretraining. |
| Diverse control demands shape useful representations | Promising hypothesis; early exploration signal only | Stable independent outcomes, matched start/goal interventions, and a controlled gradient-routing or task-demand manipulation with common-panel maps. |
| Learned landmarks support reliable intentional navigation and route composition | Not demonstrated | Full start–goal success matrix, all-attempt arrival cost, held-out approaches, and multi-leg routes. |
| The model explains biological place-field formation | Too strong | A narrower mechanism claim with distinguishable predictions and causal tests; physiological agreement beyond selected maps. |
| Temporal sequence distance is spatial or optimal control distance | Not established | Directed reachability/cost validation. Elapsed event time is policy-dependent and can be large inside a loop. |

The distance to a paper is best expressed as missing evidence, not a percentage complete or a promised number of weeks. The current material supports a progress poster narrative. A mechanistic claim needs controlled perturbations; an ML utility claim needs replicated downstream benefit. Fully autonomous landmark navigation is a substantially larger target.

**Keep the two scientific stories explicit**

The original story asks whether endogenous sequence feedback can organize sensory events and behavior before external navigation reward. The newer story asks whether the distinctions needed for reliable, diverse control organize the sensory representation. The second changes the teaching principle; success there would not retrospectively validate temporal-distance reward.

The immediate architectural predecessor already establishes sparse-input/sequence-memory benefits for rewarded navigation and emergent spatial tuning. Our contribution must isolate what intrinsic learning adds. [Lin, Yiu & Leibold, arXiv v3](https://arxiv.org/abs/2510.09951v3).

Recommend retaining the minimal sequence-feedback question as the current paper's organizing hypothesis while the existing transfer study resolves utility. Treat anchored control as the next bounded alternative if diagnosis supports it. The [five designs](five_designs_representation_control_exploration_20260909.md) are alternatives, not a plan to combine five mechanisms. There is no established novelty claim for control-based representation learning in general.

**What is already in motion**

- The [navigation8 screen](../06_experiments/navigation8_algorithm_screen_implementation_20260909.md) has six configurations, three seeds, and 300M-frame budgets; its September 9 record confirms preflights and production startup. It screens algorithms under a changed action/timing interface; it does not isolate sequence causality.
- The [September 10 repeat-8 transfer replacement](../06_experiments/fixed_reward_transfer_repeat8_launch_20260910.md) compares scratch with frozen-DG, tuned-DG, and policy transfer from each of two selected sources: 21 runs, three downstream seeds. Seven preflights passed. Production was submitted; startup recorded 18 running and three queued. Outcomes remain pending.
- The earlier repeat-4 transfer batch was cancelled after discovering a source/destination timing mismatch. Use the repeat-8 replacement for the matched comparison.

Transfer interpretation needs special care: three downstream seeds for one selected source checkpoint do not establish robustness across pretraining seeds. The current study tests selected-checkpoint utility, not sequence-specific benefit. Normalization and conditioning differences also limit a pure frozen-versus-tuned mechanistic interpretation. Report downstream frames, decisions, wall time, and pretraining cost separately; a downstream sample-efficiency gain does not automatically mean lower total interaction cost.

Canonical transfer provenance: schema `intrmotiv/study/v1`, workflow `1.5.0`, production SHA-256 `99910170169d07d10d584d0aa645b3993189daaa9dbe78ba28b9ad1e2c74b56d`. Navigation8 provenance is preserved in its linked launch record. This briefing defines no new StudySpec or training matrix.

**ICLR writing plan: mechanism plus utility**

Working question: “Can intrinsic sequence memory organize representations that help later learning?”

Write the motivation, exact mechanism, and evaluation definitions now. Build the results narrative around four figure slots:

1. Minimal architecture and the hypothesized feedback paths, distinguishing trained DG from fixed visual trunk and sequence dynamics.
2. Causal component/sequence controls on representation and exploration, with matched capacity and interaction budgets.
3. Representation reliability and the exploration–control separation, including the informative failure cases.
4. Downstream learning curves, first for the current selected sources, then a confirmatory comparison with independent pretraining seeds and matched alternative pretraining.

The preferred headline is conditional: intrinsic sequence-based pretraining improves downstream learning, with evidence that structured sequence feedback matters. A transfer win alone supports a narrower selected-pretraining result. If only representation improves, write a representation/mechanism story with that boundary. If neither mechanism nor utility survives controls, do not present the current algorithm screen as the missing contribution.

Keep full HRL route composition secondary unless it passes the command-control gate. Generalization to new goals/layouts and an established intrinsic-pretraining comparator strengthen the ML case. These are proposed scientific standards, not formal conference requirements.

**Bernstein poster plan: a clear question and an honest current result**

Working title: “Spatial selectivity and exploration dissociate in a sparse sequence-memory agent.”

Use four panels: the original biological question; the minimal DG/CA3-inspired model; contrasting maps and trajectories with all-seed summaries; the missing control link and the next decisive test. Select candidate maps transparently and put population summaries beside them. Separate DG activity from CA3 sequence-state activity.

Take-home sentence: “Broad exploration and differentiated sensory activity can emerge without establishing stable, individually controllable landmarks.”

The July [iteration 17 abstract](../07_abstracts/bernstein_intrmotiv_abstract/iteration_17.md) needs revision before reuse. Change sequence distance as spatial separation to a tested temporal proxy; make efficient graph planning an intended use; replace the old punishment-centered preliminary paragraph with corrected and current evidence. Proposed replacement result text:

“Across the tested configurations, spatial selectivity, behavioral coverage, and target-conditioned control dissociate. Some configurations explore broadly despite fragmented sensory fields, whereas selected checkpoints show partial localization without demonstrated reliable destination control. These findings motivate independent tests of landmark identity and of the causal contribution of sequence-derived feedback.”

**Cosyne poster plan: a mechanism that makes a discriminating prediction**

Prefer one focused question: does temporal sequence structure causally organize the code, or do control demands determine the distinctions retained by that code? Do not make both untested hypotheses the conclusion.

For the sequence story, perturb sequence order or memory length while holding inputs, capacity, and sampling as comparable as possible. For the control story, manipulate required destinations or approach ambiguity with stable outcome definitions, then measure DG and CA3 on common observations/histories. A predicted result is that behaviorally necessary distinctions change under the manipulation; this has not been established.

Use panels for the hypothesis, controlled manipulation, population-level representation effect, and behavioral consequence. Show whether effects persist under common-panel evaluation so changed trajectories cannot fully explain remapping. If these tests are not ready, use the narrower observed dissociation and present the causal account as a hypothesis. Do not substitute attractive fields for an explanation of their formation.

**Next decisions, in priority order**

1. Read the current transfer study at matched interaction budgets and retain failures and every seed. It is the nearest direct test of practical usefulness.
2. Qualify a small set of candidates with the existing common-panel and matched-command evaluators. Stable goal recognition and command-caused arrival should precede claims about useful graph nodes.
3. Choose one causal experiment according to the intended paper: sequence-feedback controls for the original claim, or diversified anchored-control demands for the newer claim. Resolve known normalization confounds before interpreting that comparison.
4. Commit to the publication headline after those results. Avoid another broad mechanism sweep merely because the strongest desired claim is still missing.

The decision to discuss with the boss is scientific priority: pursue the nearest reward-free pretraining/transfer story first, or deliberately shift the main project toward control-driven representation. My recommendation is the former while keeping the latter as a focused follow-up.

**Reusable workflow lesson**

Dated result audits and launch records were the fastest authoritative sources for this briefing; old abstracts and theoretical design notes describe intent, not current findings. Broad concatenated file reads truncated important context, so bounded reads of dated records worked better. A final date-based file search found the September 10 transfer replacement and prevented reporting a stale plan. Next time start with this briefing, inspect newer dated experiment notes and canonical study artifacts, and refresh only the affected claims. Reuse the cached full-history lift audit instead of repeating the 543-run screen. No new analyses, jobs, or infrastructure changes were needed; this note is the reusable synthesis.
