# A0 poster plan — 26 September 2026

## Working title

# From spatial codes to useful goals

**Subtitle:** Learning representation, exploration, and control from sensory sequences without a spatial map

**One-line message:** Spatial structure can emerge from online sensory-sequential learning, but representation quality, exploration, controllability, and downstream reward utility can dissociate.

---

## 1. Top-left: why this problem?

### Two routes to spatial structure

**A major route in navigation models**

self-motion / path integration → grid-like or coordinate-like spatial state → navigation

**Our route**

non-spatial sensory input → sparse DG events → intrinsic CA3 sequences → temporal relationships → spatially useful structure?

Key constraints:

- no coordinates, map, geodesic distance, or privileged spatial labels in learning;
- useful goals are not predefined;
- learning is online rather than from one fixed batch of experience;
- the changing policy determines the data from which the representation is learned.

**Framing sentence:**  
> Space is not an input to the model; it is a possible structure discovered in sequences of sensory experience.

---

## 2. Define success: three coupled pillars

A useful intrinsically learned goal system needs:

### Representation
Are internal events distinct, sparse, and spatially structured?

### Exploration
Does the policy collect broad and useful experience rather than repeatedly sample a narrow subset?

### Controllability
Does changing the internal goal change the policy and reliably select the intended internal node?

Physical localization is an external validation of node meaning, not part of the training definition.

Show the closed loop:

representation → possible goals → control → exploration / experience → representation

**Main conceptual point:** failure of one pillar changes the training signal available to the other two.

---

## 3. Main results: the pillars dissociate

Do not organize this section by architecture acronyms. Use 2–3 concrete failure patterns.

### A. Spatial organization versus exploration

Use a matched example where broader exploration accompanies worse field separation, or where good fields occupy only a restricted useful part of the experienced space.

Message:

> Better exploration does not automatically produce cleaner spatial codes, and cleaner codes do not guarantee a useful distribution of goals.

### B. Representation versus node control

Use the Direct F16 versus Waypoint F64 contrast:

- Waypoint F64: better-separated DG maps / more single-field units, but a very sparse reliable transition graph.
- Direct F16: broad graph reachability and substantial prospective node success, but poorly localized DG identities.

Message:

> Better spatial representation and stronger internal controllability can occur in different architectures.

### C. Apparent success versus command-specific control

Use DGP as the cautionary example:

- eventual target-hit rate can be around 50%;
- commanded versus shuffled target activation is approximately equal.

Message:

> Reaching a node frequently is not enough: controllability requires the requested goal to change behavior.

**Central result banner:**  
> We can obtain strong components of representation, exploration, and node control, but not yet a robust architecture that combines all three.

---

## 4. Transfer: spatially better does not yet mean more useful

This should now be a main result panel.

### Experiment

Five-cue downstream reward task, with identical fresh controller / reward learning and a frozen DG representation:

- **SOURCE DG:** pretrained DG weights and normalization are frozen.
- **RANDOM DG:** calibrated random DG weights and normalization are frozen.

The source and random DGs differ substantially in spatial organization, including spatial information and place-field separation.

### Result

> **The learned source DG shows no downstream reward-learning advantage over the frozen random DG.**

Show two aligned pieces of evidence:

1. **Representation panel:** source versus random DG spatial information / field-separation metrics.
2. **Reward-learning panel:** source versus random learning curves or summary showing no meaningful advantage.

### Interpretation

This is not evidence that spatial representation is irrelevant. It shows that the spatial properties we currently optimize and measure are **not sufficient to predict downstream utility**.

Possible explanations to keep for discussion, not as claims:

- random high-dimensional features may already provide enough discrimination for this task;
- current spatial metrics may measure organization that the controller does not exploit;
- the downstream task may not demand the particular structure learned by DG;
- useful transfer may require representation + controllability, not representation alone.

**Poster message:**  
> Spatial organization is an intermediate property, not a guarantee of reusable control.

Reference: [[../06_experiments/cued_reward5_transfer_20260925|five-cue transfer campaign]].

---

## 5. Small final panel: what is missing?

The remaining bottleneck is not simply “make prettier place fields.”

A DG identity can be spatially organized yet still be an imperfect goal because the same event can occur in different temporal contexts.

### Current hypothesis

DG event → recent CA3 sequence state → predictive readout → context-specific goal identity

Keep this conceptual. Do not show the full seven-arm CA3 architecture matrix.

One current lesson can be stated:

> Increasing contextual specificity may reduce aliasing, but excessive specificity can starve transition learning.

This motivates a balance between **separation** and **generalization / reusable experience**.

---

## Suggested A0 portrait layout

### Top strip — 15%
Title, one-sentence question, and literature / architecture contrast.

### Left column — 25%
Three-pillar framework + minimal DG→CA3 architecture + one strong representation figure.

### Middle column — 35%
Main empirical story: 2–3 dissociations among representation, exploration, and controllability.

This should be the visual center of the poster.

### Right column — 25%
Frozen source-DG versus random-DG transfer result, followed by the small CA3-context next-hypothesis box.

---

## Figures to prioritize

1. Minimal sensory → DG → CA3 → controller schematic.
2. Three-pillar closed-loop diagram.
3. One strong place-field / occupancy example.
4. Direct F16 versus Waypoint F64 representation-versus-graph comparison.
5. DGP target-hit versus shuffled-command diagnostic.
6. Source-DG versus random-DG spatial metrics.
7. Source-DG versus random-DG downstream reward-learning curves.
8. Small conceptual CA3-context goal diagram.

Avoid large tables. Prefer 1–2 numbers per comparison and direct visual contrasts.

---

## Take-home box

> **A spatially organized representation is not sufficient by itself. Useful intrinsic spatial learning must jointly solve representation, exploration, and controllability—and the spatial properties of a learned DG do not yet predict downstream reward-learning advantage.**

---

## Keep off the main poster

- detailed SCR/SAT/DGP configuration history;
- normalization and recruitment implementation history;
- controller replay cadence;
- FIXED versus EMA anchor details;
- Slurm / qualification records;
- StudySpec hashes;
- the full CA3 seven-arm table;
- transfer implementation debugging.

Keep these in captions, QR-linked reports, or discussion.
