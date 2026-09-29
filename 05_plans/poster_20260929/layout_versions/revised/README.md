# Two revised A0 posters: transfer, predictor, and field structure

Both versions combine the transfer result from the former version 1 with the predictor result from version 2. The former version 3 is dropped from the current selection. The header and entire left column match the previous recommended poster.

![Two-version overview](comparison_overview.png)

| Version | Section 3b | Recommendation |
| --- | --- | --- |
| [A: single-field fraction](Bernstein2026_A_transfer_predictor_monofields.svg) | Exploration and control across all 81 CA3-feedback runs | Preferred: broadest direct check of the monofield message |
| [B: continuous field dominance](Bernstein2026_B_transfer_predictor_dominance.svg) | Same two outcomes across 12 locally available runs | Alternative: avoids the binary field cutoff |

SVGs are editable A0, 841 × 1189 mm. Plot text and diagram labels are 30 pt; conclusion captions are 40 pt. New markers are 78 pt². The overview is a reduced preview. Existing 383 × 76.2 mm and 191.5 × 76.2 mm canvases are retained.

## Common content

**2a Exploration and revisits.** C15 has 2.1× the final-window coverage AUC of C01. Frozen maps at that checkpoint have one strict single-field unit out of 48 across three seeds in both C01 and C15; C05 has two. This supports improvement without increased monofield counts, not a causal claim that monofields have no benefit. Returns count mobile path windows with path >500 units and displacement <100; the plots distinguish 20 and 40 decisions.

**2b Goal specificity.** The two small plots retain issued-versus-shuffled target-hit lift and mean absolute raw-logit change under a goal swap. The diagram holds state fixed and changes the requested goal, making action sensitivity understandable. Lift 1 is the shuffled reference. Action scores are raw logits, not probabilities. The diagram is schematic and contains no fabricated measurements. The caption now makes one short point: C15 has little goal-specific advantage in this diagnostic.

**2c Transfer.** WORKER transfers learned DG, worker and graph; RAND_DG starts with random DG, a fresh worker and an empty graph. Both use frozen DG 64 and 75M downstream frames. The package diagram replaces the long explanatory paragraph. Full-run mean reward gains are +6.1% for D50 and +15.8% for D51; WORKER wins two of three downstream seed pairs in each. Points are training seeds, lines pair the seeds and black bars are means. This tests the joint system and includes source pretraining. It does not isolate DG learning, show an immediate head start, or supply a matched WORKER heldout evaluation. See [the transfer evidence](../../worker_random_transfer/README.md).

**2d Training-only goal-conditioned outcome prediction.** This is the CPD `dg_transition_prediction=goal` head, not the historical shadow CA3 predictor or the later predictive-state readout. It receives source DG activity and the requested goal and predicts the first distinct exclusive DG outcome or timeout. Cross-entropy is added to the encoder loss with coefficient 0.1. The main branch trains the DG projection, context-feedback adapter and predictor head; the goal-only control trains its own head. DIRECT mode detaches the input history. The fixed ImageNet trunk stays fixed, and predictor outputs do not select actions or manager goals. DG also receives its existing ARR and joint PPO training gradients. The return arrow in the diagram means a training gradient, not planning.

The Off/On contrast uses the same GATE_ACT_DIR design, with and without that auxiliary loss. Frozen 10k probes at 75,038,720 frames show more 20-decision mobile returns (62.4% → 83.2%) and less visited-bin coverage (69.3% → 55.3%), in all three seed pairs. Starts and stochastic paths are not identical. Reinforcing familiar cycles is a hypothesis; the result is not a general verdict on prediction. [predictor_role.json](predictor_role.json) pins the role, study and inspected source evidence.

**3a Architecture context.** The original two survey measures and all 168 runs / 56 variants remain. DG capacity stays at 16. Families differ in ages and outcome rules; spatial score and peak diversity should not be treated as causal control evidence.

## The monofield message covers exploration and control

> **More single-field structure does not reliably track better behaviour in these comparisons.**

Version A uses the complete CA3-feedback family: 27 variants × three seeds, all 75,005,952 frames. Single-field fraction is the fraction of eligible units passing the existing 80% dominant-component criterion at every 30/50/70% peak threshold. Its rank association is $ho=-0.064$ with visited coverage and $ho=-0.324$ with recorded target hits. The x axis displays 0–50% (all observed values fit); y axes retain 0–100%. Coverage spans only 84.2–88.1%, so there is little headroom in this particular exploration measure.

Version B retains a continuous score before classification: minimum dominant-component mass over those thresholds, averaged over eligible units. The four locally available CA3-feedback variants give 12 seed-level points: $ho=-0.616$ for online coverage and $ho=-0.266$ for target hits. The 0–1 x axis and 0–100% y axes are retained. This score is not spatial concentration and not mean-threshold dominance. [Continuous-field evidence](../../continuous_fields/README.md) records the broader availability and protocol sensitivity.

Both views show architectural associations, not interventions on field shape. Recent policy-driven maps and historical internal target-event counters cover different time spans; DG target hits do not verify physical arrival. A weak association cannot establish equivalence or prove that single fields are unnecessary. The shorter public caption expresses the observed dissociation without claiming that monofields cause harm.

## Verification and reuse

[manifest.json](manifest.json) fingerprints sources and all nine retained left data groups. [quality_checks.json](quality_checks.json) checks protected pixels, page bounds and text collisions. [Plotted points](plots/plotted_points.csv) and [field statistics](plots/field_behavior_statistics.json) retain exact coordinates, cohorts and measurement definitions. No training, cluster access or new telemetry was used.

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/revise_poster_layout_20260929.py --source 05_plans/poster_20260929/layout_versions/input_layout.svg
```

**Reusable experience:** reuse the pinned survey and saved trajectories, verify the exact predictor variant before drawing gradient arrows, and regenerate compact plots at their intended physical size. Compare the entire left column against the previous delivered poster, not just the older attachment. Keep diagram labels short and inspect actual Inkscape glyph bounds after rendering.
