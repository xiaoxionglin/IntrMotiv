# A0 poster analysis plan — 26 September 2026

This plan supports [[A0_poster_plan_20260926|the A0 poster plan]]. The goal is not to analyze every IntrMotiv run. It is to produce a small, matched set of figures that answer the poster's scientific questions.

## 1. Poster questions to answer

1. Can intrinsic sequence-based learning produce spatially organized DG landmarks without external reward?
2. How do representation, exploration, and node controllability dissociate?
3. Does apparently good target reaching actually depend on the commanded goal?
4. Does a spatially better frozen DG improve downstream reward learning?
5. Where in the DG → CA3 → decoder hierarchy does spatial structure appear or disappear?

The main figures should come from matched all-seed contrasts. Single runs are used only as visual exemplars.

---

## 2. Primary run sets

### A. Representation versus exploration: ARR vs SRC

Use the completed Saturday batch at 75M.

**Primary matched cells**

- SAT_C15_ARR_MON_FILM_S8
- SAT_C15_ARR_MON_FILM_S99
- SAT_C15_ARR_MON_FILM_S123
- SAT_C15_SRC_MON_FILM_S8
- SAT_C15_SRC_MON_FILM_S99
- SAT_C15_SRC_MON_FILM_S123

This holds C15, MON retirement, FiLM goal interface, environment, and training horizon fixed; encoder temporal credit changes ARR ↔ SRC.

Existing all-seed result: SRC increases coverage while worsening spatial information / map overlap / mono-field structure.

**Poster exemplar:** seed 8 is a useful matched visual pair because SRC_MON_FILM_S8 is a strong exploration case while ARR retains substantially cleaner spatial organization. The all-seed paired statistics remain the claim.

**Checkpoint:** 75M spatial snapshot; use 65–75M scalar window for coverage/control summaries.

---

### B. Representation versus internal control structure: Direct F16 vs Waypoint F64

Use CPU2048, DDQN+HER, cadence 2048, all three seeds.

**Primary matched cells**

- CPU2048_DIRECT_F16_DDQN_HER_S8
- CPU2048_DIRECT_F16_DDQN_HER_S99
- CPU2048_DIRECT_F16_DDQN_HER_S123
- CPU2048_WAYPOINT_DECODER_F64_DDQN_HER_S8
- CPU2048_WAYPOINT_DECODER_F64_DDQN_HER_S99
- CPU2048_WAYPOINT_DECODER_F64_DDQN_HER_S123

**Primary checkpoint:** 75M, because all six rows are available and the dissociation is mature.

At 75M, Direct F16 / HER has broad reliable reachability but essentially no qualified mono-fields; Waypoint F64 / HER has more single-field structure but only about 1% reachable ordered pairs.

This is a bundled architecture comparison: F, manager, and goal interface differ together. Present it as a system-level trade-off, not a pure capacity effect.

**Poster exemplar:** seed 99 unless the final all-seed analysis shows it is qualitatively atypical. If atypical, choose one common seed for both architectures by a predeclared median-distance rule across map cosine, mono-field fraction, coverage, and reachable-pair fraction.

**Secondary checkpoint:** 25M only to show that the same qualitative trade-off is not created by selecting 75M.

---

### C. Apparent target success versus goal dependence: DGP HIT vs FIRST

Use the C15 JOINT LEG family at 75M.

**Primary matched cells**

- DGP_C15_HIT_JOINT_LEG_S8/S99/S123
- DGP_C15_FIRST_JOINT_LEG_S8/S99/S123

Use the 65–75M option/control window and the 75M spatial snapshot.

Primary message: eventual HIT success can be high while target/shuffled activation is approximately equal; FIRST success is much lower.

**Poster exemplar:** DGP_C15_HIT_JOINT_LEG_S123 plus its seed-matched FIRST counterpart. S123 has a visually interpretable dense graph and several mono-fields, but the all-seed HIT/FIRST comparison remains primary.

---

### D. Candidate positive node-control checkpoint

Evaluate this checkpoint explicitly rather than inferring control from graph counts:

- DGC_DIRECT_WORKER_F16_S99 at 25M.

Existing evidence: strong tested local edges and a transient combination of field structure and graph evidence.

Run a new matched **node-control intervention** from identical source/history starts:

- nominated command;
- alternate/shuffled command;
- immediate action-distribution change;
- first distinct DG node reached;
- time to intended node;
- failures/timeouts retained.

This run is shown as a positive controllability example only if the intervention confirms command-dependent node outcomes. Otherwise keep it as passive transition evidence only.

Optional negative comparator:

- DGC_WAYPOINT_DG_F64_S99 at 25M.

---

### E. Downstream utility: frozen source DG vs frozen random DG

Use the complete five-cue frozen-DG control matrix at 75M.

**D50 source family**

- CR5C_D50_W_RAND_DG_S42/S1234/S9999
- CR5C_D50_W_SOURCE_DG_S42/S1234/S9999

**D51 source family**

- CR5C_D51_W_RAND_DG_S42/S1234/S9999
- CR5C_D51_W_SOURCE_DG_S42/S1234/S9999

Both arms have a fresh worker, fresh reward manager, empty graph, and frozen DG. The controlled factor is calibrated random versus source DG initialization.

Primary message: source and random DG representations differ spatially, but source DG shows no clear downstream reward-learning advantage.

Analyze D50 and D51 separately first, then show the six paired source-minus-random differences.

---

## 3. Two distinct evaluation datasets

Do not use one dataset for both representation and behavior.

### A. Policy-driven frozen evaluation

Use each checkpoint's frozen policy for:

- trajectory examples;
- episode coverage;
- occupancy;
- occupancy flow field;
- node-control interventions;
- graph / command outcomes.

Within each matched comparison, use the same reset seeds and evaluation horizon.

### B. Common observation/history panel

Use the same replayed sensory/action histories for every checkpoint within a comparison for:

- DG place fields;
- pre-threshold DG maps;
- DG / CA3 / decoder-1 spatial kernels;
- population-vector comparisons.

This prevents a cleaner field map from being merely a consequence of a different policy visiting different places.

The replay must preserve complete observation history so CA3 is evaluated on valid sequences rather than shuffled independent frames.

---

## 4. Standard analysis stack

### 4.1 DG place fields

For every primary run:

- pre-threshold DG activation maps;
- post-threshold DG activation maps;
- spatial information;
- active-only map cosine;
- silent-unit fraction;
- mono-field fraction;
- number of field components;
- field area / dominant-component mass;
- distinct peak bins;
- nearest-neighbor peak distance;
- population peak coverage / concentration.

For the poster, show only a compact unit atlas plus 2–4 population metrics.

Use the common observation/history panel for quantitative comparison. Retain policy-driven maps as descriptive supplementary figures.

---

### 4.2 Trajectory and occupancy

From frozen-policy episodes:

- full arena occupancy heatmap;
- several complete representative trajectories;
- fixed-length individual trajectory segments;
- accessible-area coverage AUC;
- endpoint coverage;
- occupancy entropy / effective number of occupied bins;
- concentration of occupancy mass in the most visited bins;
- stationary fraction;
- displacement/path-length efficiency.

Use identical reset seeds within each pair.

---

### 4.3 Occupancy flow field

For every spatial bin, using valid within-episode transitions:

- mean displacement vector $(\Delta x,\Delta y)$;
- mean step length;
- flow coherence $||E[\Delta x,\Delta y]|| / E[||\Delta x,\Delta y||]$;
- sample count / occupancy.

Poster rendering:

- occupancy heatmap as background;
- arrows show mean local displacement;
- arrow magnitude or opacity encodes coherent flow;
- mask low-count bins.

This is intended to expose confinement, repetitive circulation, dead zones, and directional channels that a trajectory spaghetti plot can hide.

Do not interpret flow as policy optimality.

---

### 4.4 Graph and node controllability

Produce two complementary graph views.

**All-node outcome matrix**

- source × target prospective attempts;
- prospective success fraction;
- reliable-edge mask;
- unattempted pairs explicitly masked.

**Spatially grounded graph**

- position only nodes with qualified field peaks;
- draw reliable edges between those nodes;
- keep ambiguous/multi-field nodes separate rather than forcing them onto one coordinate.

Report:

- reliable edge count;
- reachable-pair fraction;
- SCC size;
- prospective success;
- degree concentration;
- target coverage.

For selected checkpoints run the matched node-control intervention and report:

- command → first reached node confusion matrix;
- commanded versus shuffled success;
- first-step action-probability TV;
- failure/timeout fraction;
- time to intended node.

Node controllability is the primary criterion. Physical-place arrival is a stronger optional grounding analysis.

---

## 5. Layerwise learned spatial kernel

Reuse the concept in Jannek's analysis_kernel.ipynb, but implement it as a reusable script.

For a layer $l$, first construct a population activity map $A_l(x,y)$ on the common observation/history panel. Then compute

$$
K_l(\Delta x,\Delta y)
=
E_{x,y}
[
\operatorname{corr}
(
A_l(x,y),
A_l(x+\Delta x,y+\Delta y)
)
].
$$

Analyze:

1. **DG post-threshold activity**
2. **full CA3 sequence state**
3. **decoder-1 hidden activation**

Optional supplementary layer: DG pre-threshold logits.

### Important controls

- same spatial grain and arena bounds for all compared runs;
- require both displaced bins to have sufficient samples;
- z-score across population dimensions before Pearson correlation;
- compute a heading-matched kernel (e.g. 8–12 heading bins) and average across heading to reduce orientation confounding;
- keep an all-heading kernel as supplementary;
- for goal-conditioned decoder-1, compute kernels separately for a balanced set of goals/cues and average them rather than using the policy's uneven goal distribution.

### Poster outputs

For each layer:

- 2-D displacement kernel heatmap;
- radial profile $K_l(r)$;
- one scalar spatial length-scale summary, e.g. half-maximum radius or positive-kernel area;
- optional anisotropy index.

The important question is not whether every layer has a pretty kernel, but **where spatial organization appears, sharpens, or disappears across DG → CA3 → decoder**.

---

## 6. Transfer-specific mechanistic analysis

The frozen source-vs-random DG experiment deserves more than reward curves.

### Behavioral utility

For every site × seed pair:

- five-cue reward learning curve;
- early reward AUC (0–10M and 0–20M);
- full 0–75M AUC;
- terminal 65–75M reward / success;
- per-cue success;
- time-to-reward where available.

Plot all paired source-vs-random runs; do not show only group means.

### Representation difference

On the common panel compare source vs random DG:

- spatial information;
- map cosine;
- mono-field fraction / field components;
- peak distribution;
- DG spatial kernel.

### Does the difference survive downstream processing?

At the same 75M checkpoints compute:

- CA3 spatial kernel;
- decoder-1 spatial kernel.

This is especially informative if source and random DG differ strongly at DG but converge in CA3/decoder-1 despite similar reward learning.

Do not assume this outcome in advance.

### Descriptive association

Across the six matched site × seed pairs, compare source-minus-random representation differences with source-minus-random reward AUC. Treat this as descriptive only; $n=6$ is too small for a strong correlation claim.

---

## 7. Matched-checkpoint rules

1. Compare conditions only at a checkpoint available for every declared row in that contrast.
2. Use the exact saved milestone, not each run's latest checkpoint.
3. Scalars use the same trailing frame window within a contrast.
4. All-seed summaries are primary; exemplar runs illustrate them.
5. Do not mix online policy-driven field metrics with common-panel field metrics in one numerical comparison.
6. Do not compare graph reachability across different node capacities without also reporting node count and edge density.
7. Never infer causal control from graph counters alone.
8. Transfer source-vs-random comparisons are paired by source site and downstream seed.

---

## 8. Output structure

Write all new analysis under one batch root, for example:

06_experiments/results/A0_poster_analysis_20260926/

Recommended contents:

- manifest.tsv — exact run/checkpoint/evaluator identities;
- summary.md — concise factual findings;
- tables/ — all-seed and paired statistics;
- representation/ — place fields and common-panel metrics;
- behavior/ — occupancy, trajectories, flow fields;
- control/ — graph matrices and matched node interventions;
- kernels/ — DG/CA3/decoder-1 kernels and radial profiles;
- transfer/ — random-vs-source reward and representation comparisons;
- poster_candidates/ — small set of figure-ready PDF/SVG/PNG exports.

Every poster candidate should have a caption file containing run, checkpoint, seed, evaluator, denominator, and whether the evidence is policy-driven or common-panel.

---

## 9. Priority order

### P0 — must have

1. Frozen source-DG vs random-DG transfer comparison, including representation metrics and reward curves.
2. Saturday ARR vs SRC matched representation/exploration comparison.
3. CPU2048 Direct F16 vs Waypoint F64 HER at 75M.
4. DGP HIT vs FIRST plus target/shuffle diagnostic.
5. Common-panel DG place fields for all above.
6. Matched node-control intervention for DGC_DIRECT_WORKER_F16_S99 at 25M.

### P1 — high-value mechanistic figures

7. Occupancy flow fields for the primary exemplars.
8. DG → CA3 → decoder-1 spatial kernels.
9. Spatially grounded graph overlays.

### P2 — only if P0/P1 leave a clear question

10. Layerwise kernel evolution across multiple checkpoints.
11. Cue-specific decoder kernels in the five-cue transfer study.
12. Physical-place command intervention in addition to node-level control.
