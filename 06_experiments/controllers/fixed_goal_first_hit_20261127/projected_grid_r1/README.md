# Fixed GeoMem grid: projected versus unprojected first-hit targets

The projected condition reached every one of the 240 nontrivial start-goal queries in all three seeds. The unprojected condition reached 240, 220 and 235. This is a paired comparison on one fresh, separately trained 4×4 GeoMem grid with sixteen fixed synthetic DG identities; it is not policy transfer, learned visual-field evidence, or DMLab validation.

Both conditions use the exact original decoder/action-head classes, 35,588 trainable parameters, frozen CA3, the same reward-free replay and initial weights, and raw greedy action selection. Only the projected condition clamps target-network values to $[0,1]$ **before** multiplying by $.99$. No clock or new inference head is added. The [prospective design](../runtime/projected_bootstrap_design.md), [pre-data source/configuration manifest](pre_data_manifest.json), and [prepared replay receipt](prepared.json) precede the six fixed fits.

## Complete outcomes

Arrival and shortest-route columns count nontrivial queries out of 240. Worst-goal success counts successful starts out of fifteen. FIRST is the mean competing-target ordering contrast over all 3360 ordered comparisons per seed/condition. Each condition also retains sixteen separately counted initially satisfied queries; these do not inflate the primary arrival denominator.

| Seed | Target | Arrival | Worst goal | Shortest arrival | FIRST |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | Unprojected | 240/240 | 15/15 | 178/240 | 0.748810 |
| 8 | Projected | 240/240 | 15/15 | 189/240 | 0.756548 |
| 99 | Unprojected | 220/240 | 1/15 | 144/240 | 0.641369 |
| 99 | Projected | 240/240 | 15/15 | 172/240 | 0.747619 |
| 123 | Unprojected | 235/240 | 11/15 | 167/240 | 0.728274 |
| 123 | Projected | 240/240 | 15/15 | 175/240 | 0.739286 |

The preregistered gate requires at least $.9$ success for **every deterministic pair in every seed**, which here requires each pair to succeed. Projected passes all three fits; unprojected passes seed 8 only. Pooled descriptive counts are 720/720 versus 695/720 arrivals and 536/720 versus 489/720 shortest arrivals. These 720 queries are not 720 independent training replications. The three per-seed results above remain the primary replication record; no significance or population-generalization claim is made.

## Range and calibration diagnostics

These ranges use every usable factual action-goal label in the saved replay, excluding currently satisfied commands. Fresh targets use the same final frozen model on both sides of each condition's operator. They are not stale training targets or held-out observations.

| Seed | Target | Raw prediction range | Fresh target range |
| --- | --- | --- | --- |
| 8 | Unprojected | [0.929768, 1.014538] | [0.934596, 1.006377] |
| 8 | Projected | [0.929114, 1.016526] | [0.934412, 1.000000] |
| 99 | Unprojected | [0.931629, 1.051325] | [0.932287, 1.032714] |
| 99 | Projected | [0.917515, 1.014729] | [0.936095, 1.000000] |
| 123 | Unprojected | [0.923056, 1.023438] | [0.932545, 1.015251] |
| 123 | Projected | [0.928824, 1.023659] | [0.926380, 1.000000] |

Projection bounds targets, while raw predictions still exceed one. The maximum projected value errors on the 960 declared fresh-start action queries are 0.04007, 0.03625 and 0.03838. These exceed the sufficient uniform shortest-action error bound of about 0.004755 at graph distance six. Successful arrival therefore does not establish calibrated values, universal shortest routing, or a guarantee over all CA3 histories. Both projected and unprojected fresh-operator residuals are saved for each final model in its `fresh_residuals.json`.

## Matching and reproducibility

- Every seed's 4096-transition replay covers all 64 physical node-action combinations, including wall self-loops. All 128 segment ends are collector cuts; none is a physical failure. This is coarse node-action exposure, not exhaustive coverage of CA3 histories.
- Actual initial parameter hashes, sampled-row hashes, usable-mask hashes and dataset hashes match across each pair. Both conditions use 9,600,000 usable goal labels per seed, 2500 updates, batch 256, Adam learning rate $.003$, target refresh every 100 updates and $\gamma=.99$.
- All six saved checkpoints reproduce their complete frozen evaluation traces exactly. Model hashes are unchanged by evaluation. Source archives, live source hashes and all three replay files are unchanged at the final audit.
- Training took 196.3 seconds total for the unprojected fits and 194.1 seconds for projected fits on one numerical thread. These timings are descriptive; they do not establish an efficiency advantage.
- Eight focused interface/numerical tests passed before the study. No fit was repeated, extended or selected by its outcomes.

The authoritative [results](results.json) have SHA-256 `edfe2acb6d3a114afe0e2a732026ae695fdce353dbe3eb26415f7f8579de0eaa`. Per-seed folders retain checkpoints, all trajectory/control rows, independent value calibration and every competing-FIRST comparison. The code entrypoint is [projected_grid.py](../../../../hpc_runs/first_hit_control/projected_grid.py).

This supports target-range projection as a useful minimal change on this fully observed task. It does not resolve the earlier sparse-memory failures: any later reuse of those tasks must be labeled post-hoc regression. The reusable procedure is to archive the protocol before data, pair actual initialization/batch/mask hashes, distinguish fresh residuals from stale-target loss, and retain per-goal failures rather than relying on mean success.
