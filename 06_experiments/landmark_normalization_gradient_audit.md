# Landmark normalization gradient audit

Date: 2026-09-05  
Source checkpoint: C15 ARR-MON, source-credit/retirement batch, seed 99, 75M  
Rollout: 4,096 DMLab decisions, 3,843-dimensional frozen ResNet/instruction features  
DG threshold: 2.43  

## Result

The audit supports a normalization-gradient failure rather than excessive
sample removal as the common cause of the latest MON regression.

| Normalization view | Forward mask agreement with legacy | Gradient cosine with legacy | Mean absolute row-to-feature-mean cosine | Pairwise DG-row gradient cosine |
|---|---:|---:|---:|---:|
| Legacy differentiable batch moments | 1.000 | 1.000 | 0.044 | 0.018 |
| Current fixed running moments | 0.995 | 0.162 | 0.951 | 0.901 |
| Post-step-aligned fixed moments | 1.000 | 0.180 | 0.951 | 0.901 |
| Input-centered fixed moments | 1.000 | 0.589 | 0.035 | -0.043 |

The feature mean norm was `15.96`, compared with centered per-coordinate RMS
`0.134`. With fixed projected statistics, positive DG credit therefore sends
nearly every row along the same large common feature direction. Recalibrating
projected moments after the optimizer step restores a consistent published
state, but does not change that gradient. Subtracting a checkpointed running
feature mean removes the common direction without making actor outputs depend
on a learner minibatch.

Legacy and input-centered forward masks agreed exactly on this audit. Their
gradients were not identical (cosine `0.589`), so the input-centered condition
is a principled replacement for the hidden centering effect, not a claim to
reproduce every BatchNorm derivative.

## Relation to sample rejection

Global generation filtering directly caused both ARR-DIR crashes: after a
replacement it left only `18.3%` or `9.0%` of a batch, and an unguarded sample
standard deviation received an empty or singleton selection. It cannot explain
the MON degradation, because MON made no replacement, changed no generation,
and rejected no PPO samples. Terminal scheduled-to-applied encoder-credit match
was approximately `98%` in the surviving MON runs.

The corrected implementation rejects mixed representation generations at
whole-rollout boundaries and defers learning before graph updates, bootstrap,
reward construction, GAE, or normalization when fewer than one normal
minibatch of fresh decisions remains. This may intentionally skip one complete
learner batch after a rare replacement; it prevents training on a tiny biased
fragment and is separate from the normalization study.

## Artifacts

- NEMO2 audit job: `7989312` (completed successfully).
- Raw JSON: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/normalization_gradient_audit_20260905/real_feature_gradient_audit.json`.
- Three-cell StudySpec SHA-256: `1b97f4cf964271202f0579d7051bf97fb3c268bb7a803065c05c4f6b2a8cf6b9`.
- Training jobs: `7989323` (legacy), `7989324` (post-step atomic), and `7989325` (input-centered atomic).
