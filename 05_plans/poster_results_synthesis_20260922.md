# From spatial codes to controllable goals

## IntrMotiv: results synthesis and ongoing-work poster plan

**Evidence cutoff: 22 September 2026.** This document synthesizes retained experiment reports and study definitions, together with Xiaoxiong's retrospective account of the initial push–pull experiments. It does not report new rollouts, a live cluster audit, or a fresh reanalysis of raw histories. Numerical claims below refer to their explicitly dated/checkpointed sources, not necessarily each run's latest state. Repository snapshot reviewed: `323261646ce56094c59720066467fe3281293cf5` on `main`.

**Recommended title:** *From spatial codes to controllable goals in a hippocampus-inspired agent*  
**Subtitle:** *Ongoing work on elapsed-time intrinsic motivation and predictive CA3 goal representations*

**Main message:** We can obtain substantially differentiated and, in selected configurations, localized DG activity without an external navigation reward. But a spatially selective code is not automatically a stable, distinguishable destination that an agent can deliberately reach. The emerging problem is not only how to form fields, but how to turn experience into useful goals.

This is a progress story with a positive representation result, a concrete behavioral limitation, and a motivated next experiment—not a catalogue of failed tricks and not a claim that the latest architecture already works.

## 1. The scientific story

### Starting point: what Lin et al. (2026) already established

[Lin, Yiu & Leibold, ICLR 2026](https://openreview.net/forum?id=li1vfqDzRD), *Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator*, provides the architectural premise: a sparsely driven, fixed sequence-generating memory can support rewarded visual navigation and emergent spatial tuning. Its sparse-versus-dense comparison makes the interaction between input sparsity and memory architecture important, rather than proposing that a sequence generator universally outperforms learned recurrence.

**The follow-up question is different:** can sequence-derived feedback organize representations and behavior before the eventual task reward is specified, and can the learned system subsequently help goal-directed navigation?

Do not present emergent spatial tuning under rewarded navigation as the new contribution. Likewise, successful rewarded navigation in the predecessor is not evidence that the current intrinsically trained DG targets are controllable.

### Initial idea: opposite pressures from one internal clock

Use consistent notation throughout the poster:

$$
u_t=f_\theta(o_t),\qquad S_{t+1}=P S_t+J u_{t+1}.
$$

Here $u_t$ is sparse DG activity, $S_t$ is CA3 memory, $P$ is the fixed shift operator, and $J$ injects a DG event across the first $R$ sequence positions. Let $\Delta_e$ denote the elapsed time associated with an eligible DG transition event.

The intended push–pull principle is:

- **Representation pressure:** favor sensory events separated by longer elapsed intervals, rather than redundant events occurring close together.
- **Behavioral pressure:** learn to traverse between selected events more quickly.

A concrete historical `encourage` implementation illustrates these opposing preferences:

$$
\mathcal L_{\rm enc,event}=-\frac{\beta}{N_{\rm valid}}\sum_{e\in\mathcal E}\Delta_e\,u_{j_e,t_e},
\qquad r^{\rm flat}_{\pi,e}=\beta(E-\Delta_e).
$$

Here $E=R+L-1$ is the finite sequence-register length. Event masks, source-versus-arrival credit, population-use losses, and behavior-time labels matter; this is not the complete objective of every historical or current arm. Punishment and mean-centered variants belong to the same broad development history but are not numerically interchangeable with this example. Later goal-conditioned worker rewards also differ from the original flat reward. See the [checked architecture/loss account](../06_experiments/02_architectures_losses_place_fields_trajectories_and_graphs_20260910.md).

“Push–pull” means opposing learning pressures on event timing, **not literally equal-and-opposite encoder and decoder weight-update vectors**. The encoder and policy have different parameter spaces and gradient paths.

The attractive hypothesis is that improving the controller makes elapsed time increasingly informative about navigational separation, while representation learning distributes useful event identities. However, $\Delta$ is initially a policy-dependent temporal quantity, not geodesic distance. Turning, detours, loops, event aliasing, and missing history can all break that correspondence. The empty-history sentinel $E$ must never be plotted as a measured long-distance transition.

### What happened

The initial experiments produced interesting DG tuning but fragmented/multi-region fields and imperfect spatial distribution. The [June review of Jannek's report](../02_algorithm/IntrMotiv_loss_recommendations_from_report.md) already describes that combination: sparse, more selective responses with noisy or fragmented maps, and an important role for population recruitment. These are historical qualitative observations, not newly rescored baseline measurements.

Subsequent fixes and design changes produced stronger representation candidates. Yet good-looking fields, target activation, broad exploration, and useful transfer did not consistently appear together. Some selected checkpoints show local target-reaching evidence, but the relevant destinations are geographically concentrated. Others have broad exploration without clean individual fields.

**The latest step changes the goal definition:** DG units remain sparse sensory/event features and possible registry addresses; the goal's identity is carried by an observed CA3 state and its learned predictive representation. The question is whether temporal context can distinguish destinations that an instantaneous DG ID conflates.

## 2. Claims: what can be said now, and what would strengthen it

| Claim | Current status | Safe poster wording | Missing evidence for the stronger claim |
|---|---|---|---|
| C1. Intrinsic training can produce differentiated and locally concentrated DG activity | Supported descriptively in selected checkpoints; not uniformly across runs | “Selected configurations develop sparse, differentiated DG fields without external navigation reward.” | Common-observation evaluation, all-seed summaries, explicit comparison with initialization and an appropriately matched no-intrinsic-learning control |
| C2. The representation fixes improved the usable set of candidate fields | Qualitative development result; individual causal contributions incompletely isolated | “Correctness repairs and subsequent design changes yielded stronger spatial-code candidates.” | Matched ablations before attributing the gain to one loss, recruitment rule, normalization change, or push–pull itself |
| C3. High target-hit scores do not establish command-specific navigation | Strongly supported as an evaluation warning | “Frequent eventual DG hits can coexist with little target-specific activation advantage.” | Frozen matched-start interventions with independently evaluated destinations; activation-shuffle comparisons alone are not causal tests |
| C4. Some apparent representation/control successes are geographically local | Supported for named selected checkpoints | “Distinct codes and locally reliable transitions can be concentrated within a small part of the environment.” | Whole-episode coverage, destination spread relative to the accessible map, and evaluation from varied starts/approaches |
| C5. The tested intrinsic checkpoints have not demonstrated a transfer advantage | Supported for the retained interim SCR/SAT comparison, not for all candidates | “The tested sources did not improve mean fixed-reward performance at the matched evaluation window.” | Updated learning curves, newer source checkpoints, independent pretraining seeds, and transfer scopes that preserve the relevant learned system |
| C6. CA3-derived predictive goals may resolve part of the goal-identity problem | Implemented and launched; efficacy remains open | “We are testing whether predictive temporal context provides more useful goal identities than individual DG activations.” | Correct contextual HER semantics, readout health, alias/route tests, matched command effects, and downstream utility |

**Do not claim yet:** task-general map learning; reliable navigation to arbitrary locations; route composition; geodesic-distance learning; a causal biological explanation of place-field formation; or successful CA3-goal transfer.

“No external navigation reward” does not mean learning everything from scratch: the current IntrMotiv frontend uses a fixed ImageNet-pretrained visual trunk, and the historical controller also receives a depth bypass. Keep that qualification in Methods, with coordinate telemetry clearly outside the training pathway. [Architecture source](../06_experiments/02_architectures_losses_place_fields_trajectories_and_graphs_20260910.md).

## 3. Evidence ledger: the results worth retaining

The following are evidence examples, **not a leaderboard**. They mix different action interfaces, ages, managers, capacities, and measurement windows. Use matched comparisons within a study for performance claims.

### 3.1 Representation progress: stronger candidates really exist

| Evidence | Retained result | What it establishes / does not establish |
|---|---|---|
| Corrected-core C05, three seeds | All 16 units active; active-map cosine $0.127\pm0.065$; approximately 14.3 distinct peak bins | Differentiation beyond silence alone; most units nevertheless have multiple half-peak components |
| SCR ARR DIRS, seed 123, 75M gallery | 11/16 single-field units; active-map cosine 0.0605; 13 distinct peaks | A strong selected localization example; fields concentrate toward a limited region, not 11 uniformly distributed destinations |
| SAT ARR DIRO FiLM, seed 8, 75M gallery | 9/16 single-field units; cosine 0.1147; 15 distinct peaks | Another useful selected representation source; do not credit the enabled replacement rule because the recorded replacement count is zero |
| DGC WAYPOINT_DG F64, seed 8, 25M | Cosine 0.083; 45.3% single-field among eligible units; all 64 active | A newer representation candidate worth transfer testing; not a replicated whole-arena goal system |
| Same DGC arm, seed 99, 25M | Cosine 0.094; 28.8% single-field among eligible units; 58/64 active | A second promising seed, but seed 123 is weaker: cosine 0.185 and 7.8% single-field |

Sources: [corrected-core qualification](../06_experiments/corrected_core_candidate_place_field_telemetry_20260902.md), [historical matched gallery and architecture account](../06_experiments/02_architectures_losses_place_fields_trajectories_and_graphs_20260910.md), [DGC all-seed analysis](../06_experiments/dg_capacity_goal_conditioning_interim_20260911.md).

Use “single-field under the stated connected-component criterion,” not mathematically proven “unimodal.” One connected high-response region can contain multiple peaks. Also distinguish three meanings of “spread”: the extent of one unit's field, the distribution of fields across units, and the area sampled by the animal/agent. Improving one does not establish improvement in the others.

**Figure-ready message:** representation learning progressed beyond the original fragmented examples, but the strongest fields must be shown together with occupancy, spatial scale, and all-seed summaries.

### 3.2 The most revealing behavioral result: hits without specificity

In the historical DGP comparison:

| Run | Eventual option success, 65–75M | Pooled target / shifted-target activation ratio |
|---|---:|---:|
| DGP_C15_HIT_JOINT_FILM_S99 | 56.25% | 0.9958 |
| DGP_C15_HIT_JOINT_LEG_S123 | 55.52% | 0.9973 |
| DGP_C15_HIT_STOP_LEG_S8 | 52.07% | 0.9928 |

The available target set included all 15 alternatives per source. Thus these scores cannot simply be dismissed as selecting only a few eligible DG IDs. However, a broad vocabulary of IDs is not a broad set of physical destinations, and eventual HIT permits many opportunities and intervening wrong events.

The corresponding FIRST conditions have only about 6.8–7.1% option success, but they are **separately trained conditions**, not counterfactual scoring of the same policy. A $1/15$ reference is appropriate only under its single-outcome, uniform-command assumptions; it is not the correct chance level for eventual HIT.

[Source and exact measurement definitions](../06_experiments/06_high_option_success_goal_sets_and_controls_20260914.md).

There is a separate metric pathology: the September 8 audit found a selected apparent lift of 2,363.6 whose pooled activation-rate ratio was only 0.702. Means of ratios with nearly zero denominators can be misleading. Do not make the poster depend on the largest logged lift. [Audit summary](01_findings_scientific_claims_and_publication_plan_20260910.md).

**Figure-ready message:** show both the hit score and the near-one activation ratio, labeling them as different measurements. Do not put them on a shared “success” axis or call this a matched-start intervention.

### 3.3 Local skill candidates: preserve the positive evidence without enlarging its scope

DGC DIRECT_WORKER F16 seed 99 at 25M is the clearest retained small-subset candidate. Its recorded edge outcomes include 225/241 hits for 0→1, 252/292 for 2→4, and 36/43 for 6→11. The corresponding target peaks are at $(250,550)$, $(150,550)$, and $(550,550)$: distinct but geographically local destinations. These are adaptive training-history counts, not independent frozen-policy trials.

At this checkpoint the single-field fraction is 66.7% **among eligible units**, but nine units peak in the three most populated exact bins. At 75M the single-field fraction falls to 6.25% and the grounded-control score to zero. Changing behavior windows prevent a pure representation-drift conclusion, but the checkpoint advantage is clearly not enough to declare the whole run successful.

[Source, raw counts, and candidate-selection limitations](../06_experiments/dg_capacity_goal_conditioning_interim_20260911.md).

A small, spatially meaningful set of reliable goals is a legitimate success target. **Do not require full connectivity across every allocated DG unit** before acknowledging useful control. The relevant requirements are meaningful destination identity, command-caused arrival, and a clearly stated start/goal domain.

### 3.4 Which “fixes” changed the picture—and which explanations did not survive controls

| Intervention family | What the retained evidence says | Presentation role |
|---|---|---|
| Core correctness, pretrained-feature preservation, recruitment gradients, event-credit alignment | Necessary to interpret later runs. Older silent-unit recovery and mean/punish objectives had concrete failure modes; configuration differences were not clean ablations | Brief Methods/infrastructure qualification, not a separate grand scientific claim |
| Source/arrival credit, recruitment, FiLM, gradient routing | Produced interesting selected fields, but several factors vary together. Under the historical LEG interface, JOINT versus STOP improved map cosine in all six paired outcome/seed comparisons; these involve only three distinct seeds | A bounded mechanism lead, not “JOINT always helps” |
| Recent CA3→DG feedback | The matched no-feedback baseline has mean cosine 0.170; none of the compared feedback variants improves it. Selected-seed benefits did not generalize | Demote the earlier success interpretation; do not use these results to predict the new CA3-goal readout's outcome |
| More DG capacity / waypoint control | DGC F64 waypoint gives useful representation candidates; other F64 arms have weak control | Capacity is not a standalone solution; distinguish combined architectural changes |
| DDQN/HER and replay cadence | CPU2048 at 25M: Direct F16 has much greater graph reachability but poor individual fields; Waypoint F64 has lower overlap but sparse reliable graphs. The grounded diagnostic is zero in all 24 runs | Evidence that changing the learner or increasing replay alone did not solve the joint problem |
| Navigation8 interface | At 75M, SAT has low mean overlap (0.209) while DGP has the strongest connected graph; neither is a qualified command-specific controller | Interim comparison, not a final ranking or a matched five-action replication |
| Corridor geometry | At 95–100M, corridor coverage AUC is lower than open layouts in all nine paired architecture/layout comparisons | The proposed geometric rescue did not work in this screen; action sensitivity alone is insufficient |

Sources: [collapse and normalization diagnosis](../06_experiments/03_mean_punishment_and_silent_units_jannek_comparison_20260910.md), [gradient/interface comparison](../06_experiments/02_architectures_losses_place_fields_trajectories_and_graphs_20260910.md), [matched feedback comparison](../06_experiments/05_ca3_feedback_matched_results_and_full_state_gap_20260913.md), [CPU2048](../06_experiments/cpu2048_analysis_20260917.md), [Navigation8](../06_experiments/navigation8_algorithm_screen_interim_20260916.md), [completed corridor training](../06_experiments/corridor_geometry_analysis_20260921.md).

For a poster, keep this development history to a small inset or omit most of it. The audience needs the scientific inference, not every configuration acronym.

### 3.5 Geometry gives a concrete measure of behavioral confinement

At the common 95–100M training window, accessible-area-normalized coverage AUC is:

| Architecture | Corridor | Open |
|---|---:|---:|
| SAT FiLM | 2.70% | 13.97% |
| DGP joint | 2.34% | 10.67% |
| Waypoint F64 HER | 7.03% | 15.81% |

SAT and DGP end corridor episodes after visiting approximately six and five accessible cells respectively, out of 199; Waypoint visits about 21. This is not just an artifact of the corridor containing fewer floor cells. Yet SAT/DGP retain almost fully connected proxy graphs.

These are three layout seeds with training seed 99, not three independent learner seeds. They do not establish performance against a uniform-random policy: the report records those frozen evaluations as submitted, not completed. [Full evidence and caveats](../06_experiments/corridor_geometry_analysis_20260921.md).

### 3.6 Transfer: a bounded negative result, not a verdict on every candidate

The retained repeat-8 comparison evaluates all 21 runs over **50,784,640–60,784,640 environment frames**, using three downstream seeds per condition. Values are mean length-weighted scores, not success percentages.

| Transfer condition | Mean score | Difference from scratch |
|---|---:|---:|
| Scratch | 9.115 | 0 |
| SCR DG frozen | 6.103 | −3.012 |
| SCR DG tuned | 8.547 | −0.568 |
| SCR policy tuned | 7.342 | −1.773 |
| SAT DG frozen | 5.355 | −3.761 |
| SAT DG tuned | 8.522 | −0.593 |
| SAT policy tuned | 9.031 | −0.084 |

Frozen DG trails scratch in all three paired downstream seeds for both sources; tuned variants are closer, especially SAT policy transfer. This does **not** establish final convergence, absence of an early learning advantage, or the transferability of newer checkpoints. It also does not identify spatial confinement as the cause: that is a plausible connection requiring a dedicated comparison.

[Matched-window report and reusable collector](../06_experiments/fixed_reward_transfer_latest_common_20260911.md). Three downstream seeds from one chosen pretrained source are not three independent pretraining replications.

## 4. The latest direction: retain DG, change what counts as a goal

### Why the change follows from the results

An instantaneous DG target can denote several visual situations or disconnected field components. Rewarding any activation of that unit can therefore reward reaching the wrong physical place—or repeatedly visiting a convenient local patch. Making the field visually cleaner helps, but does not by itself validate the goal identity used by the controller.

The proposed alternative is:

$$
z_t=W\,\operatorname{vec}(S_t),\qquad g_j=W\,\operatorname{vec}(A_j),
$$

where $A_j$ is a **real observed raw CA3 state** stored in slot $j$. The slot indexes storage; it does not mean every occurrence of DG unit $j$ is the same landmark. One landmark per slot remains the intended initial design; contextual clones are not required here.

Train the readout with executed-action-conditioned prediction of future DG inputs:

$$
\widehat U_{t,H}=F_\psi(z_t,a_{t:t+H-1}),\qquad
U_{t,H}=[u_{t+1},\ldots,u_{t+H}].
$$

The fixed CA3 transition need not be learned again. The new learning targets the incoming environmental information rather than the predictable shift of existing memory. The planned auxiliary objective includes prediction and variance/covariance regularization; its initial gradient ownership is the readout/predictor, not DG or the worker objective.

To recognize goals online, compare predicted-future signatures under a shared action-probe bank:

$$
\sigma(S)=\operatorname{concat}_{p\in\mathcal P}\widehat U(W\operatorname{vec}(S),p).
$$

Registration/confirmation uses realized futures after they become available; immediate recognition uses the causal signature. Raw CA3 anchors are retained so their embeddings can be recomputed with the current readout. This is not merely “replace a DG one-hot by a larger vector and reward cosine similarity.” [Design](ca3_state_readout_innovation_sample_factory_plan.md).

### What has actually been launched

The [22 September release record](../06_experiments/ca3_predictive_active_goals_20260922.md) documents seven architectures × seeds 8, 99, 123 = **21 fresh 300M-frame production runs**, following corrected qualification. All use F64 capacity and stored DDQN+HER. The recorded production-startup checks passed; no mature behavioral advantage is established by that record.

The [actual StudySpec](../hpc_runs/studies/ca3_predictive_active_goals_production.study.json) includes `BASE_ID`, `PRED_SHADOW_H16`, `ZSTATE_ID_H16`, `ZGOAL_FIXED_H16`, `CTX_FULL_H16`, `CTX_NOACTION_H16`, and `CTX_FULL_H32`. The primary contrast is `CTX_FULL_H16 - ZGOAL_FIXED_H16`.

**Important contrast limitation:** the primary contrast changes contextual recognition and anchor updating (`champion` versus fixed), not only the representation vector. Interpret it as a mechanism-package comparison unless a matched anchor-policy control is added. Likewise, continuous-goal conditioning is an interface change as well as a representational change.

### Correctness qualifications before interpreting efficacy

The [same-day follow-up handoff](ca3_state_goal_followup_20260922.md) identifies two urgent issues: HER still using DG-slot identity in the contextual path, and omitted readout variance/covariance penalties. Treat these as documented issues whose correction must be verified in the evaluated runtime; do not assume a design note means the fix has shipped.

The detailed handoff's contextual HER rule compares the stored goal CA3 state with the current/successor CA3 state at an eligible DG event; **matching the DG ID alone cannot establish success**. Its detailed A1 section also explicitly removes a same-slot requirement for contextual HER, unlike the abbreviated formula at the top of that note. Use the detailed contract and its regression tests as authoritative for this synthesis; resolve that documentation inconsistency before implementation reuse.

Keep the existing production lineage intact. Corrected follow-up runs need an identifiable fresh lineage rather than silent changes to running jobs. Fixed versus EMA-signature-refined real anchors and unique-contextual-match recognition are optional follow-up factors, not established results.

### What the new representation still cannot guarantee

Equal predicted futures over a finite probe bank do not prove physical identity. Different places can share short-horizon futures, and different approaches to one place can produce different histories. Action probes can also be unsupported by replay. Therefore test **both alias rejection and approach tolerance**. A predictor can improve while the goals remain unusable; low average prediction loss can also reflect mostly silent DG targets.

The goal of this batch is not prettier DG maps. It is better goal identity and command-caused reaching, without sacrificing the useful extent of exploration.

## 5. Missing analyses and tests: run the independent work in parallel

The numbering is a priority order for attention, not a serial launch schedule. Do not wait for representation analysis before testing transfer or command control on existing candidates. Only semantic/correctness prerequisites should block a result from being interpreted.

### P0 — Assemble and refresh evidence already generated or submitted

- [ ] Recover a source/checkpoint manifest for every selected figure: study, exact run, learner seed, layout seed, training frames, action set/repeat, normalization, source revision, and evaluator version.
- [ ] Refresh the existing repeat-8 transfer study using the canonical `--latest-common` collector, retaining full curves and all downstream seeds. Keep the September 11 table as historical, not as an assertion about present terminal performance.
- [ ] Collect the already submitted corridor evaluations before launching replacements: the September 21 report lists 135 spatial jobs and 54 command-control jobs, with full episodes and geometry-matched random controls attached to terminal rows. Audit completion/eligibility and report missing rows rather than silently dropping them.
- [ ] Locate comparable original push–pull artifacts for the before/after panel. Current evidence supports the qualitative history; a quantitatively matched original-versus-corrected panel is still missing. Keep Jannek's provenance and contribution explicit.
- [ ] Verify the contextual-HER and anti-collapse fixes against the runtime/checkpoint lineage actually evaluated. Report older results as implementation-limited where those defects remain.

### P1-A — Show representation quality independently of where the policy happened to go

Evaluate initialization, a historical selected source, and newer candidates on a common observation/history panel covering accessible space and multiple headings. Include DG pre-threshold and post-threshold activity; freeze the intended evaluation normalization. For CA3 or history-conditioned DG, replay complete aligned histories—independent shuffled frames are not a valid state test.

Report field-component count, field area/dispersion, active-only map overlap, silent/eligible-unit counts, and population destination coverage. Show online occupancy-driven maps separately because they answer a different question. Quantify stability on the same panel across checkpoints; changed-policy snapshots alone cannot establish drift.

**Decision unlocked:** “The representation becomes more localized/distributed” rather than only “the sampled activity maps look better.”

### P1-B — Test whether changing the command changes where the agent arrives

Use frozen checkpoints with exact reset/prefix matching to reproduce the same physical start and memory state. Vary only the commanded goal. Include the correct command, alternate commands, and a goal-blind or shuffled-command condition; match horizon and evaluation randomness as far as the environment permits.

Keep the existing full command protocol—up to 16 sources, four targets, five repeats—where applicable. Small meaningful subsets are acceptable, but report which sources/targets were unavailable or excluded. Keep failures, timeouts, ambiguity, and already-achieved starts explicit.

Evaluate destination arrival independently of the training hit predicate. Offline pose can define/check frozen diagnostic target regions, without entering training or recognition. For CA3 goals, report both contextual-goal recognition and physical-place arrival; a history-specific goal may deliberately impose a stricter condition than location alone.

**Minimum display:** a source/command/outcome matrix, per-goal success with denominators, matched-command arrival advantage, and complete example episodes on the full map. Immediate action-probability sensitivity is secondary: different action distributions need not reach different destinations.

**Decision unlocked:** command-specific local navigation, with its actual domain stated. This is the most valuable missing behavioral panel.

### P1-C — Test newer transfer sources now, not only the old winners

The following shortlist is grounded in retained reports, not claimed to exhaust Xiaoxiong's newer promising runs:

| Candidate | Why retain it | Transfer status in the evidence reviewed |
|---|---|---|
| Historical SCR/SAT sources | Reproduce/extend the existing selected-source result | Tested in the retained repeat-8 study; no mean advantage at the reported window |
| DGC WAYPOINT_DG F64, seeds 8 and 99, 25M | Low-overlap representation candidates from a different lineage | Not covered by that SCR/SAT transfer result; prioritize source-checkpoint verification and testing |
| DGC DIRECT_WORKER F16, seed 99, 25M | Strongest retained small-local-skill candidate | Not covered by that transfer result; test the saved checkpoint, not an assumed stronger later state |
| CPU2048 waypoint DDQN+HER, cadence 2048, matched 75M checkpoints | Later field improvement is documented; a useful alternate learner lineage | Transfer outcome not established by the reviewed report |
| Corrected CA3-goal lineage | Tests the newest semantic-goal proposal | Efficacy and transfer remain pending; verify semantic correctness first |

Freeze candidate-selection rules before looking at downstream outcomes. Selected-checkpoint screening can begin immediately; confirmation should include independent pretraining seeds, not merely more downstream seeds from the same source.

Retain scratch, frozen-DG, tuned-DG, and policy transfer as relevant. For the CA3 system add the planned task-general transfer scope—DG, readout, predictor, usable anchors/graph and universal worker—while leaving external-reward-specific bindings/heads fresh. Specify which components are actually consumed downstream and how a reward location is bound to learned goals; copying unused weights is not a valid utility test. Treat this full-system transfer as a different scope from DG-only transfer.

Match action set/repeat, input interface, normalization handling, and downstream interaction budget. Show full learning curves, an early-learning summary, and a common late window; retain pretraining cost separately. A sample-efficiency gain after pretraining need not be a gain in total interactions.

**Decision unlocked:** whether any current intrinsic representation/controller helps a later fixed-reward task, and which component carries that benefit.

### P1-D — Diagnose the CA3-goal batch in parallel

Measure readout variance/effective rank, active-event versus zero-event prediction errors, state-shuffle and action-shuffle deltas, calibration support, active-goal count, ambiguity/abstention, and anchor changes. These separate a healthy predictor from trivial or unused latent codes.

Using held-out offline pose labels, measure recognition of the same place across different approaches and rejection of different places that share a DG ID. Plot positive and background similarity distributions, not only the threshold-passing rate. Keep these labels diagnostic; they must not leak into training, threshold tuning on the test set, or online recognition.

Compare arms with **a common independent destination criterion**. Otherwise changing the definition of “hit” can change apparent success even if physical behavior does not improve. Report both semantic precision and recall: a highly selective recognizer that almost never activates is not enough.

**Decision unlocked:** whether context improves destination identity, whether the worker uses it, and whether any coverage/transfer benefit accompanies it.

### P2 — One focused causal experiment for a paper

For the original push–pull claim, isolate event-timing feedback rather than adding another large heuristic sweep. A compact study can compare intact timing feedback with encoder-side timing removed, policy-side timing removed, and timing shuffled within valid event strata. Match feature initialization, population-use terms, activity/event statistics as far as possible, action timing, and training budget. Evaluate representations on common histories and behavior separately. Distinguish teacher-signal perturbations from changes to the memory dynamics.

For the CA3-goal claim, keep the existing batch contrasts and add only the missing matched control needed to disentangle contextual recognition from anchor refinement. A frozen/random readout control with matched interface can test whether predictive learning adds more than compression or dimensionality change.

An oracle-goal diagnostic is useful only as an explicitly privileged positive control: demonstrate that the worker can reach stable, unambiguous targets under the same motor interface. Do not count oracle-supervised fields as intrinsic representation results. This distinguishes representation/recognition limitations from a controller that cannot solve even a well-specified task.

### P3 — Stronger generalization and mechanism claims

After a positive effect is identifiable, test new reward locations, held-out approach routes, independent training seeds and layouts, and selected multi-leg routes. For elapsed time as distance, validate directed travel-time estimates against independently measured costs on fixed goals; report failure/censoring and policy changes. Route composition and biological predictions are additions to an already interpretable result, not prerequisites for an ongoing-work poster.

## 6. The poster itself: five panels, one question

A simple three-column layout is enough. Put the original mechanism on the left, the representation–control discrepancy in the center, and transfer plus the CA3 follow-up on the right. Use plain labels instead of the full run acronyms in the main panels; preserve provenance in captions or a linked supplement.

| Panel | Heading / purpose | Suggested content | Status |
|---|---|---|---|
| A | Can an internal sequence clock organize learning before task reward? | Small DG→CA3→controller diagram; opposing encoder/policy timing pressures; predecessor result labeled “Lin et al., 2026” | Can write now; carefully separate inherited result from new hypothesis |
| B | Intrinsic training produces candidate spatial codes | Initial fragmented example beside SCR/SAT and one newer F64 candidate; all units or a clearly selected subset plus all-seed population summary; common arena/occupancy scale | Existing galleries available; matched original/control panel and common-history evaluation missing |
| C | Recognizing a feature is not the same as reaching a destination | Eventual-hit versus shuffled-activation diagnostic; local-target map; ideally the new matched-command matrix | Historical diagnostic ready; causal command panel pending |
| D | Does the learned system help a later task? | Scratch versus selected transfer learning curves, preserving every seed and matched budgets | Historical interim summary ready; refreshed curves/new-source tests pending |
| E | From DG IDs to predictive CA3 goals | Real CA3 anchor → learned readout → predictive signature → goal-conditioned worker; current seven-arm study summarized in plain language | Method/experiment underway; leave efficacy explicitly unresolved |

**Optional inset:** corridor exploration did not rescue the current algorithms. Use the normalized three-architecture comparison only if space permits; it supports the bottleneck diagnosis but should not overwhelm the main narrative.

Useful starting assets: [historical field/graph/trajectory gallery](../06_experiments/late_outlier_spatial_gallery_20260908.md), [DGC candidate report](../06_experiments/dg_capacity_goal_conditioning_interim_20260911.md), [Navigation8 all-seed atlas](../06_experiments/data/navigation8_algorithm_screen_interim_20260916/visual_atlas_75m.md), [CPU2048 atlas](../06_experiments/data/cpu2048_analysis_20260917/atlas.md), and [corridor figures](../06_experiments/corridor_geometry_analysis_20260921.md). These are source assets to inspect/reformat, not newly rendered poster panels.

### Draft abstract / overview paragraph

Building on a hippocampus-inspired sequence-memory agent for rewarded navigation, we investigate whether internally generated temporal feedback can organize representations and behavior before a task reward is specified. Our initial push–pull scheme uses elapsed time to favor temporally separated sensory events while encouraging the controller to shorten transitions. Subsequent refinements produced differentiated and, in selected configurations, localized DG activity. However, spatial tuning, broad exploration, and deliberate target reaching did not consistently improve together. High eventual target-hit scores could coexist with negligible shuffled-target activation advantage, while some promising local transitions were concentrated within a small spatial region. The tested pretrained sources have not yet shown a mean downstream fixed-reward advantage at the retained matched window. These observations motivate replacing individual DG identities as goals with predictive representations of observed CA3 states. We are testing whether this context-sensitive goal definition improves alias rejection, command-specific navigation, and transfer without requiring coordinates during training.

### Thirty-second explanation

“Our earlier work showed that sparse sequence memory can support rewarded navigation. Here we ask whether its internal clock can help learn before we know the reward location. We can obtain interesting spatial fields, but the hard part is turning them into destinations: a unit can activate in several places, or many units can describe one small region. So we are separating field quality from deliberate reaching and transfer, and testing goals defined by predictive CA3 context rather than a DG ID alone.”

### Final take-home box

**Spatial selectivity is an intermediate achievement. The next test is whether the learned representation specifies distinct, deliberately reachable, reusable goals.**

## 7. Routes from the poster to a paper

**Representation/mechanism paper:** demonstrate a reproducible change in spatial coding on common histories, plus a matched manipulation showing that the proposed temporal feedback matters. Transfer can remain secondary, but the paper must not infer mechanism from selected maps.

**Goal-representation/control paper:** demonstrate that CA3-derived goals reduce aliasing and produce greater matched-command arrival advantage than a well-matched DG-ID baseline. Preserve destination extent and report approach dependence. Better prediction loss alone is not the result.

**Intrinsic-pretraining/utility paper:** demonstrate improved downstream learning across independently pretrained sources and downstream seeds, with transparent cost accounting and an equally budgeted comparison. A full-system improvement supports a full-system claim; it does not automatically prove that DG field quality or elapsed-time feedback caused it.

The current poster need not wait for all three stories. Its coherent present conclusion is the representation-to-goal gap, with the latest architecture presented as a testable response. Upgrade the headline only when the relevant missing evidence arrives.

## 8. Reporting conventions and reusable workflow

Use one compact claim-to-artifact manifest for the final poster. Each plotted number needs its source, checkpoint/window, denominator, unit, and evidence status: historical measurement, current qualified result, candidate, or hypothesis. Canonical StudySpecs and saved collector exports should supply run identities; this synthesis does not define a replacement run matrix or authorize new submissions.

Coverage requires special care. Historical raw coverage AUC is an episode-time average number of visited cells, not a percentage. Accessible-normalized AUC divides by that layout's accessible area. Pooled occupancy across parallel streams and resets is neither of those and cannot establish single-episode exploration. Similarly, graph connectivity, prospective hit rate, spatially grounded proxy scores, and matched-command arrival are different quantities.

Report all attempts when computing control. A useful bounded cost summary is $\mathbb E[\min(T_g,H)]$ together with $P(T_g\le H)$; do not report only travel times of successful episodes. State whether simultaneous/incorrect events end an attempt. Show the eligible/start/goal counts so filtering cannot masquerade as improvement.

For a small predeclared goal set, one proposed matched-command summary is

$$
A_{\rm cmd}=\frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\left[P(R_g\text{ reached by }H\mid g)-\frac{1}{|\mathcal G|-1}\sum_{g'\ne g}P(R_g\text{ reached by }H\mid g')\right],
$$

with the same start/history distribution in every term and independently fixed diagnostic regions $R_g$. This is a proposed evaluation statistic, not a retrospectively available result. Report the full outcome matrix beside it because an average can hide a single easy destination.

**Reuse lesson:** dated matched reports were more informative than old abstracts, launch-success notes, or selected-outlier summaries. In particular, the newer matched feedback analysis downgrades a selected-seed interpretation, and the detailed CA3 handoff is more precise than its abbreviated introduction. Future updates should refresh the affected claim rows and artifact links, not append an ever-growing chronological run diary. No training code, active run configuration, or existing implementation plan was changed by this synthesis.
