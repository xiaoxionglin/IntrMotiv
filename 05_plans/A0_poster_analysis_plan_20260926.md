# A0 poster analysis plan — revised 26 September 2026

This plan supports [[A0_poster_plan_20260926|the A0 poster plan]]. The goal is not to analyze every IntrMotiv run. It is to produce a small set of mature, interpretable figures plus matched comparisons that answer the poster's scientific questions.

## 1. Poster questions to answer

1. Can intrinsic sequence-based learning produce spatially organized DG landmarks without external reward?
2. How do representation, exploration, and node controllability dissociate?
3. What do the mature systems converge to after long training?
4. Does apparently good target reaching actually depend on the commanded goal?
5. Does a spatially better frozen DG improve downstream reward learning?
6. Where in the DG → CA3 → decoder hierarchy does spatial structure appear or disappear?

---

## 2. Run-selection principle

Use two complementary evidence tracks.

### A. Matched comparisons

For a quantitative architecture claim, use the **latest saved checkpoint shared by every required condition × seed**.

Rules:

- refresh checkpoint/snapshot availability before any analysis;
- do not default to the checkpoint used in an older report;
- do not compare each run at its individual latest age;
- if 75M is the latest common checkpoint despite some runs reaching 150–300M, use 75M for the matched claim;
- use all seeds as the primary evidence.

### B. Mature exemplars

For visual/mechanistic exemplar figures, preferentially use **mature 100–300M checkpoints** when they exist.

Purpose:

- show what a trained system ultimately develops into;
- avoid selecting an attractive early transient state;
- support detailed place-field, trajectory, flow, graph, and layerwise-kernel figures.

A mature exemplar does not need to be part of a complete factorial comparison, but it must be labeled as an exemplar rather than an architecture-wide effect.

### C. Earlier checkpoints

Use 5M/25M/75M checkpoints only when:

- they are the latest common age for a matched comparison; or
- the scientific point is explicitly learning dynamics, emergence, transience, or forgetting.

Never choose an earlier checkpoint merely because it looks better.

---

## 3. First task: refresh the candidate inventory

Before running new expensive analyses, scan all current StudySpecs, saved checkpoints, online-spatial snapshots, and completed W&B runs.

Produce one table with:

- run / architecture;
- seed;
- training frames reached;
- saved spatial milestones;
- latest common milestone for each proposed comparison;
- latest mature spatial checkpoint for each run;
- existing place-field / trajectory / graph / control analyses;
- missing analyses.

Pay special attention to runs that have reached 100M, 150M, 300M or later.

Known mature candidates to verify rather than assume:

- CPU2048 Waypoint F64 DDQN cadence-2048, including seed 99 with a 300M spatial snapshot;
- CPU2048 Direct F16 DDQN / DDQN+HER runs around 150M;
- full-system Waypoint F64 PPO, including seed 123 at 150M;
- DG-capacity Waypoint F64 runs with later training than the old 25M screen;
- completed 100M corridor runs;
- completed five-cue transfer controls at 75M.

Do not restrict the mature-exemplar search to the runs shortlisted in the 25 September screening report.

---

## 4. Primary matched comparisons

### A. Representation versus exploration: ARR vs SRC

Use the completed Saturday C15 MON FiLM batch, seeds 8/99/123.

Conditions:

- SAT_C15_ARR_MON_FILM
- SAT_C15_SRC_MON_FILM

Use the **latest common checkpoint after refreshing inventory**. If no newer common spatial checkpoint exists, retain 75M.

Primary message:

> SRC can increase exploration while worsening DG spatial differentiation.

Use all-seed paired statistics. Choose a visual seed only after confirming it is qualitatively representative of the paired effect.

---

### B. Representation versus internal control structure: Direct F16 vs Waypoint F64

Use CPU2048 DDQN+HER cadence-2048, seeds 8/99/123.

Conditions:

- CPU2048_DIRECT_F16_DDQN_HER
- CPU2048_WAYPOINT_DECODER_F64_DDQN_HER

Use the **latest common saved spatial milestone across all six runs**.

If 75M remains the latest complete matched checkpoint, use 75M for the architecture contrast even though individual runs trained much longer.

Primary message:

> Better-separated spatial representations and stronger internal graph reachability can develop in different systems.

This is a bundled system comparison; F, manager, and goal interface differ together.

---

### C. Apparent target success versus goal dependence: DGP HIT vs FIRST

Use C15 JOINT LEG, seeds 8/99/123.

Conditions:

- DGP_C15_HIT_JOINT_LEG
- DGP_C15_FIRST_JOINT_LEG

Use the latest common checkpoint/window available. If unchanged, retain the 65–75M scalar window and 75M spatial snapshot.

Primary message:

> High eventual target-hit rates can coexist with little command-specific advantage.

---

### D. Downstream utility: frozen source DG vs frozen random DG

Use the complete five-cue frozen-DG control matrix.

Conditions:

- CR5C_D50_W_SOURCE_DG vs CR5C_D50_W_RAND_DG
- CR5C_D51_W_SOURCE_DG vs CR5C_D51_W_RAND_DG
- seeds 42/1234/9999

Use the completed 75M endpoint unless a later deliberately matched endpoint exists.

Analyze D50 and D51 separately first, then the six paired source-minus-random differences.

Primary message:

> A spatially different / better-organized source DG does not currently yield a clear downstream reward-learning advantage over calibrated random DG.

---

## 5. Mature exemplar pool

Select 2–4 mature runs after the refreshed inventory. Do not preselect solely by attractive metrics.

### Candidate 1: mature Waypoint F64 DDQN

Strong candidate:

- CPU2048 Waypoint F64 DDQN cadence-2048, seed 99, 300M spatial snapshot.

Why useful:

- genuinely long training;
- distributed DG fields remain;
- graph has only limited strongly screened edges;
- useful for showing that long training does not automatically resolve representation/control organization.

### Candidate 2: mature full-system Waypoint PPO

Strong candidate:

- full-system Waypoint F64 PPO, seed 123, 150M spatial snapshot.

Why useful:

- many apparently strong graph edges;
- multiple targets concentrate in common physical regions;
- useful example of dense transition evidence / common sinks without distinct landmark identity.

### Candidate 3: mature Direct F16

Choose the best-supported 150M Direct F16 DDQN or DDQN+HER run after inventory refresh.

Use it to compare mature direct-controller behavior with the mature Waypoint examples.

Do not force a Direct exemplar if the saved 150M spatial evidence is incomplete or uninformative.

### Candidate 4: optional mature geometry example

Use a 100M corridor run only if the final poster retains an environment-geometry point.

Otherwise keep corridor results off the main poster.

### Earlier DGC F16 S99

DGC_DIRECT_WORKER_F16_S99 at 25M is **not a default positive exemplar**.

Use it only as a developmental/transience example:

- 25M looked promising;
- the same run degraded by 75M.

If a node-control intervention is run, evaluate both the attractive 25M checkpoint and the latest available mature checkpoint so persistence is explicit.

---

## 6. Two distinct evaluation datasets

Do not use one dataset for both representation and behavior.

### A. Policy-driven frozen evaluation

Use each checkpoint's frozen policy for:

- trajectories;
- episode coverage;
- occupancy;
- occupancy flow field;
- graph / command outcomes;
- node-control interventions.

Within a matched comparison, use identical reset seeds and evaluation horizons.

### B. Common observation/history panel

Use the same replayed sensory/action histories within each comparison for:

- DG place fields;
- pre-threshold DG maps;
- DG / CA3 / decoder-1 spatial kernels;
- population-vector comparisons.

This prevents representation comparisons from being confounded by different visitation.

Replay complete histories so CA3 is evaluated on valid sequences rather than shuffled independent frames.

For mature single-run exemplars, also compute the common-panel representation when a meaningful comparator exists; otherwise clearly label policy-driven representation plots as descriptive.

---

## 7. Standard analysis stack

Apply the full stack to the selected mature exemplars. For matched families, fill only missing analyses needed for the poster.

### 7.1 DG place fields

Compute:

- pre-threshold DG maps;
- post-threshold DG maps;
- spatial information;
- active-only map cosine;
- silent-unit fraction;
- mono-field fraction;
- field-component count;
- dominant-component mass / field area;
- distinct peak bins;
- nearest-neighbor peak distance;
- peak concentration / spatial coverage of the DG population.

Poster output: compact atlas + 2–4 population metrics.

---

### 7.2 Trajectory and occupancy

From frozen-policy episodes:

- occupancy heatmap;
- complete representative trajectories;
- fixed-length trajectory segments;
- accessible-area coverage AUC;
- occupancy entropy / effective occupied bins;
- occupancy concentration;
- stationary fraction;
- displacement/path-length efficiency.

Use identical reset seeds for paired conditions.

---

### 7.3 Occupancy flow field

For valid within-episode transitions in each spatial bin, compute:

- mean displacement vector $(\Delta x,\Delta y)$;
- mean step length;
- flow coherence $||E[\Delta x,\Delta y]|| / E[||\Delta x,\Delta y||]$;
- occupancy/sample count.

Render occupancy underneath local flow arrows and mask low-count bins.

Purpose: expose repetitive loops, confinement, sinks, directional channels, and dead regions that trajectory overlays can hide.

---

### 7.4 Graph and node controllability

Produce:

**All-node outcome matrix**

- source × target attempts;
- prospective success;
- reliable-edge mask;
- explicitly masked unattempted pairs.

**Spatially grounded graph**

- position only nodes with qualified spatial fields;
- retain ambiguous/multi-field nodes as ambiguous rather than forcing one coordinate;
- overlay reliable directed edges.

Report:

- reliable edge count/density;
- reachable-pair fraction;
- SCC size;
- prospective success;
- degree concentration;
- target coverage.

Where checkpoint support permits, run matched command interventions:

- command → first reached node confusion matrix;
- commanded versus shuffled success;
- first-step action-probability TV;
- failures/timeouts;
- time to intended node.

Graph counters alone are never called causal control.

---

## 8. Layerwise learned spatial kernel

Reuse the concept in Jannek's analysis_kernel.ipynb as a reusable script.

For layer $l$, construct population activity map $A_l(x,y)$ on the common observation/history panel and compute

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

1. DG post-threshold activity;
2. full CA3 sequence state;
3. decoder-1 hidden activation.

Optional: DG pre-threshold logits.

Controls:

- same arena bounds/grain;
- sufficient occupancy in both displaced bins;
- z-score across population dimensions;
- heading-matched kernel, averaged across heading bins;
- all-heading kernel as supplementary;
- for goal-conditioned decoder-1, balance goals/cues before averaging.

Poster outputs:

- 2-D displacement kernel;
- radial profile $K_l(r)$;
- spatial length-scale summary;
- optional anisotropy index.

Primary question:

> Where does spatial organization appear, sharpen, broaden, or disappear across DG → CA3 → decoder?

Prioritize this analysis for mature exemplars and the source-vs-random transfer comparison.

---

## 9. Transfer-specific mechanistic analysis

For every D50/D51 × seed pair:

### Behavioral utility

- five-cue reward-learning curve;
- 0–10M and 0–20M AUC;
- full 0–75M AUC;
- terminal 65–75M reward/success;
- per-cue success;
- time-to-reward if available.

Show paired source/random runs, not only means.

### Representation

On a common panel:

- spatial information;
- map cosine;
- mono-field / component structure;
- peak distribution;
- DG spatial kernel.

### Downstream representation

At matched 75M checkpoints:

- CA3 spatial kernel;
- decoder-1 spatial kernel.

This tests whether source/random differences present in DG survive downstream processing.

### Descriptive association

Across the six site × seed pairs, compare source-minus-random representation changes with source-minus-random reward AUC. Treat this as descriptive only.

---

## 10. Checkpoint and exemplar rules

1. **Refresh first.** Never rely on an old report's checkpoint inventory if newer runs/snapshots may exist.
2. **Matched claim:** latest milestone present for every required condition × seed.
3. **Mature exemplar:** latest informative saved spatial checkpoint, preferably 100–300M.
4. Never compare conditions at different ages and call it an architecture effect.
5. Never choose an earlier checkpoint because it gives prettier fields or stronger control.
6. Earlier checkpoints are allowed only for explicitly longitudinal claims.
7. All-seed matched summaries are primary; exemplars illustrate mechanisms.
8. Do not mix policy-driven and common-panel field metrics in one numerical contrast.
9. Report node count/edge density alongside graph reachability when capacities differ.
10. Transfer source/random results are paired by source family and downstream seed.

---

## 11. Output structure

Write new analyses under:

06_experiments/results/A0_poster_analysis_20260926/

Recommended contents:

- manifest.tsv — refreshed run/checkpoint/evaluator inventory;
- mature_candidates.tsv — all candidate runs ≥100M with available spatial evidence;
- summary.md — concise factual findings;
- tables/ — paired/all-seed summaries;
- representation/ — place fields/common-panel metrics;
- behavior/ — occupancy/trajectories/flow;
- control/ — graph/interventions;
- kernels/ — DG/CA3/decoder-1 kernels;
- transfer/ — source/random analyses;
- poster_candidates/ — figure-ready exports.

Every poster candidate must record:

- exact run;
- checkpoint age;
- seed;
- evaluator/protocol;
- denominator;
- policy-driven vs common-panel evidence;
- whether it is a matched result, mature exemplar, or longitudinal example.

---

## 12. Priority order

### P0 — must do first

1. Refresh the full checkpoint/snapshot inventory.
2. Identify all mature ≥100M poster candidates.
3. Recompute the exact latest-common checkpoint for every matched comparison.
4. Complete the frozen source-vs-random five-cue transfer analysis.
5. Complete the latest-common Saturday ARR vs SRC comparison.
6. Complete the latest-common CPU2048 Direct F16 vs Waypoint F64 HER comparison.
7. Complete DGP HIT vs FIRST / target-shuffle analysis.

### P1 — mature exemplar figures

8. Full place-field + trajectory + occupancy + flow + graph stack for 2–4 mature exemplars.
9. DG → CA3 → decoder-1 spatial kernels for those mature exemplars.
10. Common-panel representation comparison where a mature matched comparator exists.
11. Matched node-control intervention on the strongest mature candidate if feasible.

### P2 — developmental/secondary analyses

12. 25M → 75M DGC F16 S99 transience analysis.
13. Kernel evolution across checkpoints.
14. Corridor 100M analysis if geometry remains in the poster.
15. Physical-place command interventions beyond node-level control.
