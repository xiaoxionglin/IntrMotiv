# From spatial codes to controllable goals

## IntrMotiv: updated results synthesis and poster plan

**Evidence cutoff: 25 September 2026.** This synthesis is organized by scientific factors rather than experiment chronology. It uses the maintained experiment map in [[../06_experiments/README|the factorized experiment synthesis]] and the dated reports linked below. Repository state reviewed immediately before this update: e06cff298adcc9fd47441ac38f86e8eeeb06dc6a on main.

**Recommended title:** *From spatial codes to controllable goals in a hippocampus-inspired agent*  
**Subtitle:** *Intrinsic spatial representation, goal aliasing, and predictive CA3 state*

**Poster thesis:** Intrinsic sequence-based learning can produce differentiated spatial event codes, but spatial tuning is not equivalent to a useful goal representation. Across several architectures, field quality, graph connectivity, deliberate target control, and downstream transfer dissociate. The current hypothesis is that DG should remain a sparse event/address code while temporal CA3 state supplies a more specific goal identity.

The poster should therefore tell a **representation-to-goal story**, not an architecture leaderboard and not a chronology of engineering fixes.

---

## 1. Scientific question and architecture

### Starting point

[Lin, Yiu & Leibold, ICLR 2026](https://openreview.net/forum?id=li1vfqDzRD) showed that sparse input to a hippocampus-inspired sequence generator can support rewarded navigation and emergent spatial tuning. The new question is different:

> Can internal temporal structure organize representations before a task reward is specified, and can those representations later support flexible goal-directed behavior?

The current IntrMotiv family shares a common scaffold:

$$
u_t=f_\theta(o_t),\qquad S_{t+1}=P S_t+J u_{t+1},
$$

where $u_t$ is sparse DG activity and $S_t$ is the fixed CA3 sequence state. Early variants use elapsed event time $\Delta$ as an intrinsic signal: DG learning favors temporally separated events while the controller is trained to shorten selected transitions.

The attractive idea is that behavior and representation can improve each other: as control improves, elapsed time becomes a better approximation to navigational separation; as landmarks become more useful, the controller receives better subgoals. But $\Delta$ is policy-dependent, not geodesic distance. Loops, aliasing, missing events, and different approaches to the same place can all break the approximation.

### Factorized architecture

The poster should use these factors instead of historical acronyms whenever possible.

| Factor | Main levels in the current evidence |
| --- | --- |
| Sensory scaffold | Fixed ImageNet ResNet-18 features; depth/instruction bypasses in relevant historical lines |
| DG representation | Sparse learned projection; F16 or F64; different encoder/recruitment/normalization rules |
| CA3 state | Fixed shift-register memory of recent DG events |
| Goal representation | DG ID; target-ID FiLM; continuous readout state $z=WS$; contextual CA3 anchor; external reward instruction |
| Controller | PPO; stored-state DDQN; DDQN+HER |
| Graph / manager | Direct landmark target; controllability graph; waypoint/frontier manager; contextual graph |
| Training signal | Intrinsic temporal feedback; target hit/first outcome; predictive readout loss; external reward |
| Evaluation | Spatial fields; physical coverage; graph structure; matched command outcomes; downstream transfer |

A family label such as SCR, SAT, DGP, Direct F16, or Waypoint F64 bundles several of these factors. Cross-family differences are descriptive unless the report explicitly identifies a matched contrast.

---

## 2. The central result: the four axes dissociate

The results are most coherent when separated into four questions:

1. **Representation:** are DG events spatially differentiated?
2. **Goal identity:** does a goal denote a distinct destination rather than an aliased feature?
3. **Controllability:** does changing the goal change the policy and reliably select the intended internal node?
4. **Utility:** does the learned system help when a new external reward is introduced?

The project has positive evidence on the first axis, strong evidence that axes 1–3 can dissociate, and only preliminary evidence on the fourth. Physical localization is an external grounding test; for the controllability pillar itself, success means that a goal changes actions and preferentially produces the intended learned node.

### 2.1 Spatial representation can become substantially differentiated

Selected corrected and later configurations produce sparse, non-silent, spatially differentiated DG maps without external navigation reward.

Examples worth retaining:

| Evidence | Result | Safe interpretation |
| --- | --- | --- |
| Corrected-core C05, three seeds | All 16 units active; active-map cosine $0.127\pm0.065$; about 14.3 distinct peak bins | Representation is differentiated beyond simple sparse firing |
| SCR ARR DIRS, seed 123 at 75M | 11/16 single-field units; cosine 0.0605; 13 distinct peaks | Strong selected localization example, but fields are spatially concentrated |
| SAT ARR DIRO FiLM, seed 8 at 75M | 9/16 single-field units; cosine 0.1147; 15 distinct peaks | A second strong selected representation example |
| DGC Waypoint-DG F64, seed 8 at 25M | Cosine 0.083; 45.3% single-field among eligible units; 64/64 active | F64 can produce a larger, more differentiated landmark vocabulary |
| Same F64 lineage, seed 99 | Cosine 0.094; 28.8% single-field; 58/64 active | Replicates the qualitative promise, with substantial seed variability |

Sources: [[../06_experiments/corrected_core_candidate_place_field_telemetry_20260902|corrected-core telemetry]], [[../06_experiments/02_architectures_losses_place_fields_trajectories_and_graphs_20260910|architecture/results account]], and [[../06_experiments/dg_capacity_goal_conditioning_interim_20260911|DG-capacity analysis]].

Do not equate “single field” with mathematically unimodal tuning. Also distinguish field shape, population distribution across the environment, and physical coverage by the policy.

**Poster wording:** “Intrinsic training can produce sparse, differentiated, and in selected cases localized DG event codes.”

### 2.2 The three pillars can separate—and the combination can be transient

The DG-capacity study now has a complete newer 75M panel for all 18 **direct** runs. This strengthens the dissociation story but also warns against treating one attractive checkpoint as a stable architecture property.

The clearest local-control candidate remains **DIRECT_WORKER F16 seed 99 at 25M**: 66.7% mono-field among eligible units, 100% reachable ordered pairs, 61.3% prospective edge success, and several heavily sampled edges above 80% observed success. Its useful target peaks are nevertheless geographically clustered. By 75M, the same run has only 6.25% mono-field units, 81.25% reachable pairs, 56.0% prospective success, and zero grounded-controllability proxy.

Across the three-seed DIRECT_WORKER F16 arm, the same pattern is visible more weakly: mean map cosine improves from 0.190 to 0.159 from 25M to 75M, but mono-field fraction falls from 24.3% to 6.3% and the grounded proxy from 0.106 to 0.005. In contrast, DIRECT_DG F16 improves mono-field fraction from 2.1% to 12.5% while retaining about 90% graph reachability at 75M.

**Poster interpretation:** we can find checkpoints that temporarily combine two desirable properties, but the joint state is not yet stable. This is more informative than assigning a permanent “good/bad” label to an architecture.

Source: [[../06_experiments/dg_capacity_goal_conditioning_interim_20260911|DG-capacity analysis, including the newer 75M direct-only follow-up]].

### 2.3 A good graph or high hit rate can still be goal-insensitive

This is the strongest negative/diagnostic result.

In the historical DGP HIT line:

| Run | Eventual option success | Target / shifted-target activation ratio |
| --- | ---: | ---: |
| DGP HIT JOINT FiLM S99 | 56.25% | 0.9958 |
| DGP HIT JOINT LEG S123 | 55.52% | 0.9973 |
| DGP HIT STOP LEG S8 | 52.07% | 0.9928 |

The available goal vocabulary covered all 15 alternatives per source, but target activation was essentially unchanged by shifting the command. Separately trained FIRST variants fell to about 6.8–7.1% option success. These are not matched counterfactual evaluations of one policy, but together they expose a central failure mode: an agent can eventually encounter the requested DG event without the request exerting much causal control over behavior.

The same dissociation appears elsewhere:

- In the 25M CPU2048 comparison, Direct F16 has high graph reachability but poor individual field structure; Waypoint F64 has lower map overlap but only sparse reliable graph connectivity. Grounded controllability is zero in all 24 runs at the matched checkpoint.
- In Navigation8, DGP develops the strongest connected graph while SAT has lower map overlap; neither is a demonstrated command-specific controller.
- Corridor geometry does not rescue the problem: normalized exploration is lower in corridors than open layouts in all nine architecture–layout pairs.

Sources: [[../06_experiments/06_high_option_success_goal_sets_and_controls_20260914|high-hit analysis]], [[../06_experiments/cpu2048_analysis_20260917|CPU2048]], [[../06_experiments/navigation8_algorithm_screen_interim_20260916|Navigation8]], and [[../06_experiments/corridor_geometry_analysis_20260921|corridor geometry]].

**Poster message:** “Spatial selectivity and graph connectivity are not sufficient for controllability: the goal must change actions and preferentially select the intended internal node.”

This is a scientifically useful result rather than merely a failed controller: it separates the emergence of a state/event code from the emergence of a behaviorally meaningful goal.

---

## 3. Why DG identity is an imperfect goal

The simplest IntrMotiv architecture treats one DG unit as both:

- a sparse landmark/event detector, and
- the identity of a destination to be reached.

The experiments show why those roles need not coincide.

A DG unit can fire at multiple physical locations. Several units can cluster in one part of the environment. A broad or common unit can become an easy graph sink. The same physical location can also be encountered through different recent histories. Therefore a DG index is a useful **address**, but not necessarily a complete goal state.

A small number of useful DG goals can still be valuable. The DGC Direct-Worker F16 seed-99 checkpoint, for example, contains strong incoming event counts for several distinct but geographically local targets. That is legitimate local skill evidence, but not arbitrary-location navigation.

The current conceptual revision is:

> **DG says “an event of this type occurred”; CA3 context says “which occurrence/state was this?”**

This motivates learning a compact state

$$
z_t = W S_t
$$

from CA3 and using observed CA3/readout states as goal identities while retaining DG slots for sparse event detection and graph bookkeeping.

---

## 4. Predictive CA3 goals: promising decomposition, no navigation win yet

### 4.1 Seven-way predictive-state experiment

The predictive CA3 study holds F64, waypoint management, and stored DDQN+HER fixed while separating several factors:

| Cell | Worker state | Goal | Context rule |
| --- | --- | --- | --- |
| BASE_ID | Parent memory | DG ID | None |
| PRED_SHADOW_H16 | Parent memory | DG ID | Readout learned but unused |
| ZSTATE_ID_H16 | Predictive $z$ | DG ID | None |
| ZGOAL_FIXED_H16 | Predictive $z$ | Continuous $z$ goal | Fixed comparator |
| CTX_FULL_H16 | Predictive $z$ | Continuous $z$ goal | Contextual anchors/hits |
| CTX_NOACTION_H16 | Predictive $z$ | Continuous $z$ goal | Predictor omits action input |
| CTX_FULL_H32 | Predictive $z$ | Continuous $z$ goal | Longer predictive horizon |

At the matched 75M checkpoint:

| Architecture | Map cosine | Mono-field fraction | Reliable edges | Grounded control |
| --- | ---: | ---: | ---: | ---: |
| BASE_ID | 0.272 | 0.094 | 13.0 | 0 |
| ZSTATE_ID_H16 | 0.230 | 0.177 | 39.7 | 0 |
| ZGOAL_FIXED_H16 | 0.227 | 0.120 | 43.3 | 0 |
| CTX_FULL_H16 | 0.259 | 0.017 | 2.0 | 0 |
| CTX_FULL_H32 | 0.207 | 0.053 | 0.3 | 0 |

The important result is not that “CA3 goals work.” They do not yet demonstrate control. Instead, the experiment localizes the failure:

- putting the predictive readout into the worker state or using a continuous readout goal is compatible with richer graph evidence;
- adding the current contextual registration/hit machinery sharply reduces reliable graph evidence;
- every arm still has zero grounded controllability at 75M.

Source: [[../06_experiments/ca3_predictive_active_goals_interim_analysis_20260924|predictive CA3 interim analysis]].

### 4.2 Anchor/candidate follow-up

The follow-up fixes the predictive H32 state-goal architecture and crosses two factors:

- **Anchor maintenance:** first confirmed anchor (FIXED) versus EMA prototype-guided representative refinement.
- **Candidate admission:** strongest raw DG event (DOM) versus exactly one recognized contextual anchor (UNIQUE).

At the balanced 25M checkpoint:

| Anchor / candidate | Map cosine | Mono-field fraction | Reliable edges | Grounded control |
| --- | ---: | ---: | ---: | ---: |
| Fixed / dominant | 0.186 | 0.047 | 23.0 | 0 |
| Fixed / unique | 0.209 | 0.102 | 0.3 | 0 |
| EMA / dominant | 0.276 | 0.052 | 28.0 | 0 |
| EMA / unique | 0.280 | 0.188 | 1.0 | 0 |

UNIQUE admission increases mono-field fraction in several paired comparisons but almost eliminates graph attempts and reliable edges. The likely computational issue is a **precision–coverage trade-off**: stricter contextual recognition may reject ambiguous events, but if it abstains too often the controller receives too little graph/replay support. That interpretation is consistent with the data but still requires the recognition and HER diagnostics for confirmation.

Source: [[../06_experiments/ca3_state_goal_followup_interim_analysis_20260924|CA3 state-goal follow-up]].

### Neuroscience-facing interpretation

This architecture gives a useful computational analogy without requiring a literal biological claim:

- DG-like sparse codes can emphasize event separation.
- CA3-like temporal context can potentially disambiguate repeated or aliased sensory events.
- Too much separation/abstention can destroy usable transition statistics.
- Successful behavior may therefore require a balance between **pattern separation** and **generalization/completion**, rather than maximal representational distinctness.

The current experiments support the existence of this trade-off as an engineering/computational problem. They do **not** establish that the biological DG–CA3 circuit implements the specific readout, HER, or graph algorithm used here.

---

## 5. Perceptual landmarks: making observations easier helps selectively

The easy-landmark maze asks whether the current failures are partly caused by severe visual aliasing. The same geometry is rendered either with 10 distinct decals plus 10 colored wall faces or with neutral versions at the same sites.

In the completed seed-99 2M frozen qualification:

| Architecture | Neutral coverage AUC | Rich cues | Rich − neutral |
| --- | ---: | ---: | ---: |
| SCR | 0.326 | 0.461 | +0.134 |
| DGP | 0.437 | 0.410 | -0.027 |
| Waypoint | 0.349 | 0.577 | +0.228 |

This is only two frozen episodes from one training seed per architecture, and the synchronized training-window comparison gives different directions. Therefore it is not a replicated cue-effect result.

Still, it is useful for the poster as a mechanistic test: **perceptual distinctiveness interacts with architecture**. A visually easier world can help some policies without repairing a goal-insensitive learning rule.

Source: [[../06_experiments/easy_landmark_maze_qualification_analysis_20260924|easy-landmark qualification]].

This is more informative than saying the original environment was simply “too hard.” The emerging question is which part of difficulty matters: perceptual aliasing, goal identity, temporal recognition, or controller learning.

---

## 6. Transfer: single fixed goals are too narrow to carry the main claim

### 6.1 Historical fixed-reward transfer

At the latest shared 50.8–60.8M window of the earlier repeat-8 study, scratch has the highest mean downstream score (9.115). SAT policy transfer is almost tied (9.031), while frozen-DG and most other transfer conditions trail scratch.

Source: [[../06_experiments/fixed_reward_transfer_latest_common_20260911|latest-common transfer comparison]].

That result is correctly described as “no mean transfer advantage in the tested window,” not as evidence that intrinsic pretraining is useless.

### 6.2 Corrected fixed-site transfer reveals why the task is weak

The corrected 2026-09-24/25 transfer implementation fixes the external-reward path and tests scratch, DG transfer, worker/graph transfer, and full transfer at source-aligned reward sites.

The early seed-42 training episodes show that the single fixed reward is learned very rapidly even from scratch:

| Site / arm | Raw score near 2M | Raw score near 4M |
| --- | ---: | ---: |
| DG50 Waypoint scratch | 7.6 | 10.0 |
| DG50 Waypoint full transfer | 1.0 | 1.6 |
| DG50 Flat scratch | 9.2 | 9.9 |
| DG50 Flat DG transfer | 9.1 | 9.9 |
| DG51 Waypoint scratch | 7.8 | 10.0 |

A raw score of 10 means essentially all recent episodes collected the +10 reward. These are online training episodes from one downstream seed, not the prespecified held-out evaluation.

Two lessons follow:

1. **The task is easy enough that scratch has little headroom to lose.** This makes one fixed reward location a weak test of reusable intrinsic structure.
2. **Transfer is not automatically beneficial.** Source worker/graph priors can be poorly aligned to one downstream reward, while DG-only transfer can be neutral rather than harmful.

The zero-shot command probes also show detectable command-dependent action probabilities without reliable physical-arrival advantage, reinforcing the earlier separation between policy sensitivity and navigation.

Source: [[../06_experiments/fixed_reward_dg_peak_transfer_execution_20260924|corrected fixed-reward execution record]].

### 6.3 Five-cue transfer is the more informative test

The new five-cue task keeps the same map but samples one of five invisible reward locations each episode and presents a stable number instruction. It therefore asks whether a learned representation/controller can support **multiple changing task goals**, rather than optimizing one fixed location.

The validated design contains eight transfer/scratch arms, three downstream seeds, and 48 runs at 75M. Qualification passed, but production was still being submitted/running at the evidence cutoff. There is no result to plot yet.

Source: [[../06_experiments/cued_reward5_transfer_20260925|five-cue reward transfer]].

**Poster interpretation:** use the single-goal result mainly to motivate why transfer should be tested across multiple reward locations. Do not make it the headline negative result.

---

## 7. Claim ledger for the poster

| Claim | Status on 25 Sep | Safe wording |
| --- | --- | --- |
| Intrinsic sequence-based learning can produce differentiated DG spatial codes | Supported descriptively and in several replicated summaries, but not uniformly | “Intrinsic training can produce sparse, differentiated spatial event codes.” |
| Spatial tuning alone gives a useful navigation goal | Contradicted by multiple dissociations | “Spatial selectivity does not guarantee a distinct or controllable destination.” |
| Dense/reliable graph structure proves goal-directed control | Not supported | “Internal graph connectivity can be high even when command-specific evidence is weak.” |
| Predictive CA3 state is a better goal representation | Mechanistically plausible; partial representation/graph effects; control still zero | “Predictive temporal context is being tested as a richer goal identity than DG ID.” |
| Unique contextual recognition improves goal identity | May increase localization, but strongly reduces graph support in early results | “Stricter contextual matching trades ambiguity against usable event coverage.” |
| Perceptual landmark cues solve the problem | Not established; architecture-dependent qualification effect | “Visual distinctiveness can help some architectures but is not a universal rescue.” |
| Intrinsic pretraining improves fixed-reward transfer | Not shown in current single-goal tests | “Single fixed-goal transfer shows little or no advantage; the task is rapidly learnable from scratch.” |
| Intrinsic structure helps across many later goals | Open | “A five-cue transfer test now targets reusable multi-goal structure.” |

Do not claim arbitrary-location navigation, route composition, learned geodesic distance, task-general cognitive maps, or a biological mechanism of DG/CA3 function.

---

## 8. Poster design: five panels, one argument

The poster should read left to right as a sequence of scientific questions.

| Panel | Heading | Content | Status |
| --- | --- | --- | --- |
| A | **Can an internal sequence clock organize learning before reward?** | Minimal visual trunk → DG → CA3 → controller diagram; representation-vs-control push–pull principle; predecessor result clearly labeled as prior work | Ready |
| B | **Intrinsic learning produces spatial event codes** | One early fragmented example, one strong F16 example, one F64 example; population summary with map overlap, mono-field fraction, and occupancy | Mostly ready; common-history evaluation remains stronger future evidence |
| C | **A spatial code is not automatically a controllable goal** | DGP ~56% eventual hit beside target/shuffle ratio ~1; graph-connectivity/control dissociation; one map showing clustered/local targets | Strongest current result |
| D | **Temporal context changes the goal problem** | BASE_ID → z-state → continuous z-goal → contextual anchor diagram; compact 75M predictive table and 25M DOM/UNIQUE trade-off | Interim but informative |
| E | **Does intrinsic structure help when rewards change?** | Single-goal scratch learns rapidly; five-cue task schematic with five instructed rewards; label five-cue results as ongoing | Single-goal evidence ready; five-cue outcome pending |

**Optional inset:** easy-landmark rich-versus-neutral qualification. It visually supports the idea that sensory distinctiveness and goal identity are separate bottlenecks.

Avoid a panel listing every historical architecture. Acronyms belong in captions/supplementary links; main graphics should show the factor that changed.

### Figure priorities

1. One matched representation atlas with occupancy.
2. DGP hit-versus-command-specificity plot.
3. Predictive CA3 factor diagram and the two compact result tables.
4. Single-goal transfer learning curves showing fast scratch learning.
5. Five-cue environment schematic when results are not yet available.

Useful source assets: [[../06_experiments/data/navigation8_algorithm_screen_interim_20260916/visual_atlas_75m|Navigation8 atlas]], [[../06_experiments/data/cpu2048_analysis_20260917/atlas|CPU2048 atlas]], [[../06_experiments/recent_architecture_batches_synthesis_20260924|recent architecture synthesis]], and the figure links inside the predictive/follow-up reports.

---

## 9. Poster-ready abstract

Building on a hippocampus-inspired sequence-memory agent for rewarded navigation, we ask whether internally generated temporal signals can organize representations before the eventual task reward is known. Intrinsic training produces differentiated and, in selected configurations, localized DG activity, but spatial tuning, internal graph connectivity, and deliberate target reaching do not reliably emerge together. In particular, high eventual landmark-hit rates can coexist with little target-specific activation advantage, indicating that an instantaneous DG identity is often an inadequate goal description. We therefore separate sparse event identity from temporal goal identity by learning a predictive readout of recent CA3 state and using observed readout states as continuous goals. Early experiments show that predictive state and continuous goals can alter representation and graph structure, whereas strict contextual recognition currently reduces usable graph evidence and has not yet produced grounded control. Single fixed-reward transfer is rapidly learned from scratch, motivating a new multi-goal transfer test in which reward locations change across episodes. Together, these results frame spatial selectivity as an intermediate representation problem and goal identity as a distinct computational problem for flexible navigation.

### Thirty-second explanation

“Our previous model showed that sparse hippocampal-style sequence memory can support rewarded navigation. Here we ask whether the same temporal structure can help organize learning before the reward is known. We do get interesting spatial event codes, but a place-like unit is not automatically a useful destination: high hit rates and dense graphs can appear even when changing the target barely changes the outcome. We are therefore testing whether recent CA3 context gives a more specific goal identity than a DG unit alone, and whether that structure helps when the reward changes between several locations.”

### Take-home box

> **Spatial selectivity is not the endpoint. Flexible navigation requires a representation that distinguishes useful goals, enough experience to learn transitions between them, and a controller whose commands causally change destinations.**

---

## 10. Neuroscience framing

The neuroscience contribution should be framed as a computational hypothesis rather than a claim of circuit-level correspondence.

### What the model suggests

- A sparse DG-like code can emerge as a useful event basis without every unit becoming an ideal place cell.
- Spatial coding and behavioral control can dissociate; an organism does not need every internal feature to be optimized for one current task.
- Temporal context in a CA3-like sequence state can, in principle, distinguish otherwise aliased sensory events.
- Very strict identity separation can reduce usable transition evidence, producing a trade-off between discrimination and generalization.

This creates a natural connection to **pattern separation versus pattern completion/generalization**: the best goal representation may not be the maximally separated one. It must be specific enough to reject aliases while broad enough to collect repeated experience across approaches and routes.

### What the model does not establish

The current readout, replay, HER, graph, and controller are algorithmic devices, not proposed literal biological mechanisms. The experiments do not show that hippocampal place fields are caused by the implemented losses, nor that CA3 computes the exact goal embedding used here.

The useful neuroscience claim is narrower:

> A hippocampus-inspired agent exposes a computational tension between forming distinct event representations and turning them into reusable, behaviorally controllable goals.

That tension is itself relevant to biological navigation, where representations must serve many future goals rather than overfit one reward location.

---

## 11. Evidence still needed before upgrading the headline

### Highest priority

1. **Matched node-control intervention.** From identical physical/memory starts, change only the goal, measure immediate action-distribution change, and record the first learned node reached. Report the full source × command × reached-node matrix. Physical destination is a stronger grounding analysis, not required for the basic controllability criterion.
2. **CA3 readout health.** Show prediction loss by active/zero targets, state- and action-shuffle deltas, latent variance/effective rank, recognition positive/background distributions, and contextual HER support.
3. **Common-history representation evaluation.** Compare DG fields on the same observation/history panel rather than only policy-dependent online occupancy.
4. **Five-cue transfer results.** Compare scratch and transfer across all five instructions and downstream seeds, emphasizing early sample efficiency and per-goal generalization.

### Stronger paper-level tests

- A compact causal timing-feedback ablation for the original intrinsic mechanism.
- Frozen/random readout controls for the CA3-goal line.
- Oracle unambiguous goals as a positive controller control.
- Independent pretraining seeds and held-out reward locations/routes.
- Directed travel-time versus independently measured navigation cost if making a distance/geodesic claim.

---

## 12. Reporting rules

Use [[../06_experiments/README|the factorized experiment synthesis]] as the report map and [[../04_implementation/IntrMotiv_metric_reference|the metric dictionary]] for definitions.

Every poster number should have:

- exact study/run identity;
- checkpoint or window;
- denominator;
- whether it is training, frozen evaluation, or offline telemetry;
- whether the comparison is matched or cross-family/descriptive.

Keep these quantities distinct:

- spatial field quality;
- physical coverage;
- graph connectivity;
- target-conditioned action sensitivity;
- command-caused learned-node arrival;
- physical arrival/location as a separate grounding check;
- downstream reward learning.

For transfer, always show all downstream seeds and full learning curves. Report pretraining cost separately from downstream sample efficiency.

For control, retain failures and timeouts in denominators. If summarizing time-to-goal, report both $P(T_g\le H)$ and a censored/bounded cost such as $\mathbb E[\min(T_g,H)]$.

The goal of future updates is not to append another chronological status section. Update the relevant factor, claim row, and poster panel when new evidence changes the scientific interpretation.
