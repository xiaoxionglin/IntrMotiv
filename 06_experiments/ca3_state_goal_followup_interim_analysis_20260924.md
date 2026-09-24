# CA3 state-goal follow-up: matched 25M interim analysis

**Analysis date:** 24 September 2026. **Scope:** NEMO2 CPU production StudySpec `ca3_state_goal_followup_20260922_production`, twelve 300M-frame runs. All twelve have 5M and 25M online spatial snapshots; ten have 75M. The two missing 75M rows are the fixed-anchor seed-8 conditions, so the 25M checkpoint is the balanced comparison. The separate G500 production release is documented in [its release record](ca3_state_goal_followup_20260922.md) and is not pooled here.

## What changed from the predictive batch

This follow-up holds the readout-goal architecture fixed: the worker receives a 16-dimensional state readout of raw CA3 memory; the prediction horizon is 32 decisions; the predictor receives actions; goals are continuous readout states; active graph slots and contextual hit checks are enabled. The learned readout receives prediction loss and variance/covariance anti-collapse terms. Stored DDQN+HER remains the worker learner. The change under test is how an accepted contextual identity is maintained and which candidate event is allowed to update the graph.

An **anchor** is a stored CA3 state associated with one active graph slot. `FIXED` keeps the first confirmed anchor. `EMA` maintains a moving prototype of confirmed predictive signatures and may replace the stored anchor with a more representative confirmed occurrence, preserving the slot's semantic identity and graph edges. This is a same-identity refinement, not a new goal slot.

`DOM` takes the strongest raw DG candidate. `UNIQUE` evaluates contextual recognition over selectable active anchors and accepts an event when exactly one anchor matches. It can rescue a valid landmark when several DG units fire, but it can also abstain when zero or several contextual anchors match. The factorial is:

| Cell | Anchor maintenance | Candidate admission |
|---|---|---|
| `CTX_FIXED_DOM_H32` | Fixed first anchor | Dominant DG event |
| `CTX_FIXED_UNIQUE_H32` | Fixed first anchor | Exactly one contextual match |
| `CTX_EMA_DOM_H32` | EMA signature refinement | Dominant DG event |
| `CTX_EMA_UNIQUE_H32` | EMA signature refinement | Exactly one contextual match |

Each cell has seeds 8, 99, and 123. The declared contrasts are the EMA main effect, unique-candidate main effect, their interaction, and unique-versus-dominant within each anchor mode. Those paired contrasts are the correct way to judge the two factors.

## Matched 25M online spatial results

Values are three-seed means at the declared 25M snapshot. `Mono` is the fraction of eligible DG units with one field. `Cosine` is active-only map overlap. `Reliable edges` and `reachable pairs` describe the internal graph; neither by itself proves command-caused travel.

| Anchor / candidate | Cosine | Mono | Distinct DG peak bins | Reliable edges | Reachable pairs | Grounded control |
|---|---:|---:|---:|---:|---:|---:|
| Fixed / dominant | 0.186 | 0.047 | 42.3 | 23.0 | 0.007 | 0.000 |
| Fixed / unique | 0.209 | 0.102 | 45.0 | 0.3 | 0.000 | 0.000 |
| EMA / dominant | 0.276 | 0.052 | 40.0 | 28.0 | 0.008 | 0.000 |
| EMA / unique | 0.280 | 0.188 | 38.0 | 1.0 | 0.000 | 0.000 |

Within fixed anchors, unique contextual admission raises mono-field fraction in all three paired seeds but removes 36, 17, and 15 reliable edges relative to dominant admission. Within EMA anchors, it removes 32, 21, and 28 edges; mono-field fraction improves in two seeds and is effectively unchanged in one. This is a strong early graph-density association with the candidate rule. Because the graph construction depends on accepted events, the result is consistent with frequent `UNIQUE` abstention, but the table alone does not measure abstention.

At the per-run level, unique arms have only 0–2 reliable edges, whereas dominant arms have 15–36. The graph gap is consistent across both anchor modes and all three paired seeds; the mono-field advantage is less consistent.

Fixed anchors have lower mean map overlap than EMA in both candidate-rule strata, but the fixed-minus-EMA difference is not consistent in all three seeds for the dominant rule. The EMA mechanism needs its actual refinement counts and recognition-calibration diagnostics before it can be credited with any representation effect. All four cells have zero mean grounded controllability at 25M.

The candidate-rule graph gap grows across the first two complete checkpoints. Under fixed anchors, dominant/unique mean reliable edges are 1.3/0 at 5M and 23.0/0.3 at 25M. Under EMA, they are 16.0/4.3 at 5M and 28.0/1.0 at 25M. Thus the unique rule is already graph-sparse early, and the dominant arms add edges while unique arms do not. The 75M panel remains incomplete and is excluded from this paired trajectory.

## Interpretation and next test

The unique-context rule may be too selective for graph-building at this early training age. It is also plausible that its accepted events are fewer but cleaner; the current online graph output cannot decide between those explanations. Compare `context_zero_match`, `context_multi_match`, `context_unique_rescues`, accepted events, anchor refinements, calibration thresholds, and contextual HER positive/wrong-context rates at the same matched age. Then use frozen place fields and matched-command evaluation to test whether the accepted graph edges correspond to reproducible destinations.

At 75M, ten snapshots are available, but both fixed/seed-8 rows are absent. The 75M table is therefore an inventory, not a balanced factorial outcome. Canonical 25M–30M and 5M–10M TensorBoard scalar scans both failed because the discovered step history for fixed/dominant seed 99 ends at 5,865,472 while its later spatial snapshots exist. No matched online-scalar contrast is inferred from this report. Production remains active toward 300M.

## Provenance and reusable lesson

- Study schema `intrmotiv/study/v1`; declared workflow 1.11.0; StudySpec SHA-256 `68a0911fbc4719cc20eca4cf4145ffe0405b570d2ac7bf5a49a84643ab9f8437`.
- Canonical definition: [ca3_state_goal_followup_20260922_production.study.json](../hpc_runs/studies/ca3_state_goal_followup_20260922_production.study.json). The scientific rationale and semantics are in [the follow-up plan](../05_plans/ca3_state_goal_followup_20260922.md).
- Authoritative interim output: `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/ca3_state_goal_followup_20260924_interim_spatial/` (`per_snapshot.csv`, `snapshot_inventory.csv`, `analysis_manifest.json`). The already verified workflow 1.12.0 checkout read this 1.11.0 study.

The main NEMO2 checkout was only workflow 1.10.1 and could not load this study; the verified 1.12.0 checkout avoided a redundant collector implementation. A lightweight copy of the exact StudySpec was staged under the allocated corridor workspace for analysis. Future repeated collection should use that checked version pairing and the canonical CSV outputs.
