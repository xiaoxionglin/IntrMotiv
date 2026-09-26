# A0 poster plan — updated 26 September 2026

## Title

# Intrinsic Motivation and Landmark Formation from Hippocampal Sequence Dynamics

### Central question

> **Can hippocampal sequence dynamics organize useful landmarks without an external reward?**

One-sentence setup:

> Lin et al. (2026) showed that sparse egocentric visual input coupled to a CA3-like sequence generator supports rewarded navigation and emergent spatial tuning. Here we remove the task reward and ask whether internally available sequence timing can organize representation and behavior on its own.

This should remain close to the submitted abstract. The poster is an updated answer to that original question, not a new cognitive-map story.

---

## 1. Start from Lin et al. 2026: remove the reward

### Previous work

Show the original architecture:

visual input → sparse DG → fixed CA3 sequence dynamics → actor–critic ← external task reward

Main point:

- sparse sensory input + CA3-like sequence memory can support rewarded navigation;
- spatially tuned DG activity emerges during task learning.

### This work

Visually remove the external reward arrow.

New question:

> **What happens when the task reward is absent?**

The architecture still has internally available temporal structure:

visual input → sparse DG events → CA3 sequence propagation → elapsed event time / recent history

### Intrinsic hypothesis

Use sequence timing to create two coupled pressures:

- **representation:** discourage redundant / overly frequent landmark events and favor more separated events;
- **behavior:** learn transitions between selected landmarks more efficiently.

Do not oversell elapsed time as geodesic distance. Phrase it as:

> Sequence progression provides a locally available temporal relationship between sparse sensory events that may help organize landmarks without oracle spatial information.

### Small literature-positioning note

Keep this brief:

> Many navigation models obtain spatial structure from self-motion/path integration, explicit state geometry, predictive objectives, or predefined visual goals. Here we test a complementary possibility: whether an internally generated hippocampal-like sequence can itself provide a useful organizing signal.

No need to make this a large comparison panel.

---

## 2. What would count as useful landmark formation?

The experiments made clear that one scalar “success” measure is not enough.

Use three visually simple pillars:

### Representation
Are DG events sparse, differentiated, and spatially organized?

### Exploration
Does the policy acquire experience broadly rather than repeatedly sampling a narrow region?

### Controllability
Does changing the internal goal change the policy and reliably select the intended internal node?

Physical place identity is an external validation of what a node means; coordinates are not part of the training definition.

Show the closed loop:

representation → possible landmarks/goals → behavior → exploration/experience → representation

**Key point:**

> Because learning is online, representation and behavior shape each other's training data.

---

## 3. Main result: the desired properties do not automatically emerge together

This should be the largest results panel.

Do not organize the poster by SCR / SAT / DGP / Waypoint acronyms. Use a few interpretable examples.

### A. Good landmark structure can be spatially confined

Use the strongest representation + local-transition candidate, e.g. Direct-Worker F16 seed 99 around 25M.

Show:

- localized / single-field DG examples;
- strong selected node-transition counts;
- occupancy / peak locations showing that useful landmarks concentrate in a restricted region.

Message:

> **Locally useful landmark structure does not guarantee broad exploration or spatial coverage.**

Label the control evidence as **node-level transition evidence**, not verified arbitrary-location navigation.

### B. Better representation can coexist with weaker control structure

Use the Direct F16 versus Waypoint F64 comparison.

- Waypoint F64: better-separated maps / more single-field units, but sparse reliable graph reachability.
- Direct F16: broad graph reachability and substantial prospective node success, but poorly localized DG identities.

Message:

> **Representation quality and internal controllability can improve separately.**

### C. Apparent target success can be misleading

Use DGP:

- eventual target-hit rate can be around 50%;
- commanded versus shuffled target activation is approximately equal.

Message:

> **Frequent target hits do not establish goal-dependent behavior.**

Operational definition for the poster:

> Controllability requires both a goal-dependent change in policy and reliable selection of the intended internal node.

### Main result banner

> **Intrinsic sequence-based learning can produce strong components of landmark representation, exploration, and node control, but these properties do not yet robustly coincide in one system.**

---

## 4. Stronger functional test: do better spatial fields help a new reward task?

This should be a major result panel.

### Five-cue frozen-DG comparison

Same downstream task and fresh reward-learning machinery, but freeze the representation:

- **SOURCE DG:** pretrained DG projection and normalization;
- **RANDOM DG:** calibrated random DG projection and normalization.

The two representations differ in spatial organization, including spatial information and place-field separation.

### Result

> **The learned source DG shows no clear downstream reward-learning advantage over the frozen random DG.**

Present two aligned plots:

1. **Representation difference:** source versus random DG field/spatial metrics.
2. **Behavioral utility:** downstream five-cue reward-learning curves showing no corresponding advantage.

### Interpretation

Do not frame this as “spatial representation does not matter.”

Frame it as:

> **The spatial properties we currently measure are not sufficient to predict downstream reward utility.**

Useful discussion possibilities:

- random features may already discriminate the observations needed by this downstream task;
- field separation may not be the property the controller needs;
- useful transfer may require representation plus controllability / transition structure;
- the downstream task may not exploit the learned organization.

This is a strong negative result because the representation difference is real while the functional advantage is absent.

Reference: [[../06_experiments/cued_reward5_transfer_20260925|five-cue transfer campaign]].

---

## 5. Small final panel: what might be missing?

Keep this as a hypothesis, not another architecture-results section.

A DG event can be a useful sparse landmark detector without being a complete goal identity. The same event may occur in different temporal contexts.

### Current hypothesis

DG event → recent CA3 sequence context → predictive/readout state → context-specific goal identity

One sentence is enough:

> **Temporal context may help distinguish different occurrences of an otherwise aliased landmark.**

Optional current lesson:

> More specific contextual recognition may reduce aliasing, but can also reduce repeated transition evidence; useful goals may require a balance between separation and generalization.

Do not show the full seven-arm CA3 table or FIXED/EMA details.

---

## Suggested A0 portrait layout

### Top strip — 15%
Title, central question, Lin et al. 2026 → “remove reward” schematic.

### Left column — 25%
Intrinsic sequence-timing hypothesis + three-pillar framework + one strong representation example.

### Middle column — 35%
Main dissociation panel:
- confined but structured landmarks;
- representation versus node-control contrast;
- DGP apparent-hit versus command-specificity result.

This is the visual center.

### Right column — 25%
Frozen source-DG versus random-DG transfer result, then a small CA3-context “what next?” box.

---

## Figures to prioritize

1. **Lin et al. 2026 architecture**, with the external reward visibly removed for the present study.
2. Minimal intrinsic sequence-timing / push–pull schematic.
3. Three-pillar diagram: representation × exploration × controllability.
4. One strong landmark field + occupancy example.
5. Direct F16 versus Waypoint F64 representation/control contrast.
6. DGP target-hit versus shuffled-command diagnostic.
7. Source-DG versus random-DG spatial metrics.
8. Source-DG versus random-DG five-cue reward-learning curves.
9. Small DG-event → CA3-context goal schematic.

Avoid large tables. Use the detailed run names only in captions.

---

## Suggested verbal story

> Our previous work showed that sparse hippocampal-like sequence memory can support rewarded navigation and emergent spatial representations. Here we asked a simpler exploratory question: what happens if the external reward is removed and the sequence dynamics themselves provide the internal learning signal? We found that spatially organized landmark-like representations can emerge, but representation, exploration, and controllability do not automatically develop together. More importantly, a pretrained DG with better spatial organization does not currently improve downstream reward learning relative to a frozen random DG. These results suggest that place-like organization is only one ingredient of a useful landmark system, motivating a closer look at how temporal context defines behaviorally useful goal states.

---

## Take-home box

> **Hippocampal-like sequence dynamics can organize landmark-like representations without task reward, but spatial organization alone does not guarantee exploration, controllability, or downstream usefulness.**

---

## Keep off the main poster

- detailed architecture chronology;
- SCR / SAT / DGP configuration taxonomy;
- normalization and recruitment implementation history;
- controller replay cadence;
- FIXED versus EMA anchor details;
- full CA3 ablation matrix;
- cluster / qualification / StudySpec details;
- transfer debugging history.

Keep these in captions, QR-linked reports, or discussion.
