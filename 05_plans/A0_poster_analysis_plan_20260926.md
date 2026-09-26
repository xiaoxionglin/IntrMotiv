# A0 poster analysis plan — concise 26 September 2026

This plan supports [[A0_poster_plan_20260926|the A0 poster plan]]. The poster has room for only a few detailed examples. Most runs should contribute to aggregate statistics, not receive a full analysis stack.

## 1. Selection rule

Before new analysis, refresh the checkpoint/snapshot inventory.

- **Matched claims:** use the latest saved checkpoint shared by every required condition × seed.
- **Visual exemplars:** prefer mature 100–300M checkpoints.
- Do not choose an earlier checkpoint because it looks better.
- Earlier checkpoints are shown only for an explicit learning-dynamics/transience point.

The final poster should contain **at most two detailed run exemplars**.

---

## 2. Aggregate comparisons

These mostly need compact summary plots, not individual galleries.

### A. Representation vs exploration

Saturday C15 MON FiLM, ARR vs SRC, seeds 8/99/123.

Use the latest common checkpoint.

Show only a paired/all-seed summary of:

- coverage;
- spatial information / map cosine;
- mono-field fraction.

Message: broader exploration can accompany worse spatial differentiation.

### B. Representation vs internal control structure

CPU2048 DDQN+HER cadence-2048:

- Direct F16;
- Waypoint F64;
- seeds 8/99/123.

Use the latest common checkpoint.

Show only aggregate:

- map cosine / mono-field fraction;
- reliable reachable-pair fraction;
- prospective success.

Message: spatial representation and graph/control structure can dissociate.

### C. Apparent success vs command specificity

DGP C15 JOINT LEG:

- HIT vs FIRST;
- seeds 8/99/123.

Use latest common checkpoint/window.

Show:

- option success;
- commanded vs shuffled target activation/outcome;
- optionally graph reachability as context.

Message: frequent target hits do not establish goal-dependent control.

### D. Downstream transfer

Five-cue frozen DG:

- SOURCE_DG vs calibrated RAND_DG;
- D50 and D51;
- three seeds each.

Show:

1. compact DG spatial-quality comparison;
2. paired downstream reward-learning curves/AUC.

Message: better spatial organization does not currently produce a downstream reward-learning advantage.

Do not run a full trajectory/graph/kernel analysis on every transfer run unless a specific mechanistic question remains after these two plots.

---

## 3. Detailed exemplars

After refreshing the inventory, choose **two** mature runs.

### Exemplar 1 — best mature spatial/landmark example

Choose the most interpretable mature run with localized/differentiated DG structure and adequate behavior sampling.

Prefer a ≥100M checkpoint.

This exemplar should visually establish that nontrivial spatial structure can emerge without external reward.

### Exemplar 2 — mature dissociation/failure example

Choose one mature run that most clearly shows a mechanistically interesting failure, e.g.:

- strong graph but common/broad landmark sinks;
- broad exploration but poor fields;
- good fields but weak graph/control.

Current candidates to inspect include:

- CPU2048 Waypoint F64 DDQN seed 99 at 300M;
- full-system Waypoint F64 PPO seed 123 at 150M;
- mature Direct F16 checkpoints around 150M.

Choose only after checking all mature candidates; do not show all of them.

DGC Direct-Worker F16 seed 99 at 25M is not a main exemplar. Use it only if we explicitly show the transient 25M → 75M deterioration.

---

## 4. Full analysis stack — exemplars only

For the two selected exemplars, complete:

### Place fields

- DG pre- and post-threshold maps;
- spatial information;
- map cosine;
- mono-field / field-component structure;
- peak distribution.

### Behavior

- representative trajectories;
- occupancy map;
- occupancy flow field;
- coverage / occupancy concentration.

### Graph/control

- reliable graph / source-target outcome matrix;
- prospective success;
- reachable-pair fraction;
- spatially grounded nodes where meaningful.

Only run matched command interventions if the exemplar is being used to claim controllability.

### Layerwise spatial kernel

Reuse Jannek's population-vector displacement-correlation analysis for:

1. DG;
2. CA3;
3. decoder-1.

Use the same observation/history panel across layers. Include a 2-D kernel and radial profile only if it adds a clear poster message.

---

## 5. Poster-output target

The downstream analysis should aim to produce approximately:

1. **one aggregate representation/exploration plot** — ARR vs SRC;
2. **one aggregate representation/control plot** — Direct F16 vs Waypoint F64;
3. **one command-specificity plot** — DGP HIT/FIRST/shuffle;
4. **one transfer panel** — SOURCE vs RAND spatial metrics + reward curves;
5. **one detailed exemplar figure** — fields + trajectory/flow;
6. **optionally one second detailed exemplar or DG→CA3→decoder kernel figure**.

Anything beyond this should remain supplementary.

Save new outputs under:

06_experiments/results/A0_poster_analysis_20260926/

Include a concise summary with:

- latest-common checkpoints used for each aggregate comparison;
- the two selected mature exemplars and why they were chosen;
- exact figure candidates for the poster.
