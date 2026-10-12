# Temporal-credit ablations across CA3 and the decoder

[Question index](README.md) · [Encoder-credit report](03_encoder_credit.md) · [Worker-bonus report](04_worker_bonus_and_scope.md) · [DG checkpoint analysis](checkpoint_fields_20261011.md)

**Status, 12 October 2026:** The 38 verified original fresh terminal checkpoints have been replayed through CA3, decoder 1, and decoder 2 on one saved 10,001-observation sequence. Every replay reproduced its archived DG spatial score and the 242-bin occupancy panel after accounting for the established axis convention. This completes the original C01 and C05 factor cells and gives three paired C15 encoder seeds **for downstream layers**. The two C15 constant-encoder replacement trainings completed at 100M, and their DG fields now pass the same 10,001-observation replay, giving five paired C15 DG seeds in the [field report](checkpoint_fields_20261011.md). Their seed-8 CA3/decoder preflight [completed with exit zero](../results/cosyne_2027_temporal_credit_layers/raw/COS27_C15_CONSTANT__s8__f100007936.json): the exact 100,007,936-frame checkpoint reproduced the archived DG score within $5.1\times10^{-9}$ per unit, and all three downstream layers have finite metrics on the same 242-bin, 10,001-observation panel. The reviewed seed-99 replay was [submitted once as job 8345388](../results/cosyne_2027_temporal_credit_layers/submission_20261012T103050Z.tsv). Neither replacement enters the paired downstream summary until both pass the complete-source audit. The separate C15 constant-worker checkpoints remain absent from this layer table. Missing cells are **unmeasured**, not zero.

## Answer at the current evidence boundary

Temporal **encoder** credit improves decoder-1 spatial information in C01 on all five paired seeds, both per activation (+0.680 ± 0.244 bits) and in the amplitude-weighted all-unit score (+0.101 ± 0.029 bits per step). Decoder 2 shows the same five-seed direction (+0.636 ± 0.303 and +0.024 ± 0.014). CA3 is more metric dependent: per-activation information rises in four of five C01 pairs (+0.088 ± 0.159), whereas its all-unit amplitude-weighted score rises in all five (+0.256 ± 0.097). C01 has no worker hit bonus, so these are encoder-credit contrasts without that factor.

C05 partly extends the encoder result. Decoder 1 gains per activation in 5/5 pairs with temporal worker bonus and 4/5 with constant worker bonus; its all-unit score rises in 4/5 and 5/5 respectively. CA3 normalized information rises in 4/5 pairs under either fixed worker arm, but its all-unit score is mixed with constant worker bonus. Decoder 2 depends more on the worker arm. Temporal **worker** bonus does not provide a consistent spatial-information gain at fixed encoder credit. With temporal encoder credit it **lowers** the amplitude-weighted decoder-1 and decoder-2 scores in all five worker pairs, even though normalized information has mixed signs. The C05 paired interaction is small and mixed for CA3, and negative on average but dispersed for the decoder; it is not a stable cross-layer interaction claim.

C15 has three matched encoder seeds (17, 41, 123) in its temporal-worker downstream-layer table. CA3 and both decoder layers have mixed normalized signs (2/3 higher); amplitude-weighted decoder-2 information is slightly lower on average (1/3 higher). Replacement seeds 8 and 99 have verified 100M checkpoints and DG replays but await downstream-layer replay. The separate constant-worker extension has no measured layer pairs yet, so C15 worker timing and interaction remain unavailable.

![Encoder-credit seed pairs for decoder 1](../results/cosyne_2027_temporal_credit_layers/encoder_paired_decoder_1.png)

[Editable decoder-1 figure](../results/cosyne_2027_temporal_credit_layers/encoder_paired_decoder_1.svg) · [CA3 encoder figure](../results/cosyne_2027_temporal_credit_layers/encoder_paired_ca3.svg) · [decoder-2 encoder figure](../results/cosyne_2027_temporal_credit_layers/encoder_paired_decoder_2.svg) · [CA3 worker figure](../results/cosyne_2027_temporal_credit_layers/worker_paired_ca3.svg) · [decoder-1 worker figure](../results/cosyne_2027_temporal_credit_layers/worker_paired_decoder_1.svg) · [decoder-2 worker figure](../results/cosyne_2027_temporal_credit_layers/worker_paired_decoder_2.svg)

Dots are paired training seeds, black segments are the seed means, and shaded positions have no verified paired replay. Figure labels give the measured/declared seed counts. All panels compare independently trained frozen networks on the same external observation and saved action-history sequence at 100,007,936 training frames. The panel comes from one earlier C15 policy; it removes evaluation-trajectory differences without covering every reachable state or equalizing the networks' internally evolving goals.

### Absolute original values on the common panel

These are run-level means ± sample SD across **all available** training seeds in each cell. C15's T/T row has five models and its C/T row only three, so subtracting those two displayed means is **not** the paired C15 effect. The paired tables below use only seeds 17, 41, and 123 for that contrast. T and C denote temporal and constant encoder / worker settings; C01 has no worker bonus.

| Architecture; encoder / worker | n | CA3 bits/activation | Decoder 1 bits/activation | Decoder 2 bits/activation |
| --- | ---: | ---: | ---: | ---: |
| C01; T / — | 5 | 1.876 ± 0.171 | 1.931 ± 0.182 | 1.844 ± 0.328 |
| C01; C / — | 5 | 1.789 ± 0.092 | 1.251 ± 0.235 | 1.209 ± 0.160 |
| C05; T / T | 5 | 2.025 ± 0.355 | 2.160 ± 0.261 | 1.548 ± 0.192 |
| C05; T / C | 5 | 1.883 ± 0.317 | 2.103 ± 0.520 | 1.751 ± 0.261 |
| C05; C / T | 5 | 1.547 ± 0.289 | 1.722 ± 0.209 | 1.420 ± 0.154 |
| C05; C / C | 5 | 1.404 ± 0.354 | 1.545 ± 0.332 | 1.257 ± 0.108 |
| C15; T / T | 5 | 1.606 ± 0.152 | 1.181 ± 0.391 | 1.194 ± 0.214 |
| C15; C / T | 3 | 1.627 ± 0.371 | 0.843 ± 0.233 | 0.885 ± 0.090 |

| Architecture; encoder / worker | n | CA3 weighted bits/step | Decoder 1 weighted bits/step | Decoder 2 weighted bits/step |
| --- | ---: | ---: | ---: | ---: |
| C01; T / — | 5 | 0.485 ± 0.070 | 0.208 ± 0.026 | 0.066 ± 0.021 |
| C01; C / — | 5 | 0.230 ± 0.046 | 0.108 ± 0.015 | 0.042 ± 0.010 |
| C05; T / T | 5 | 0.507 ± 0.102 | 0.237 ± 0.064 | 0.082 ± 0.027 |
| C05; T / C | 5 | 0.545 ± 0.145 | 0.329 ± 0.114 | 0.139 ± 0.051 |
| C05; C / T | 5 | 0.362 ± 0.053 | 0.170 ± 0.040 | 0.070 ± 0.016 |
| C05; C / C | 5 | 0.428 ± 0.134 | 0.205 ± 0.106 | 0.076 ± 0.039 |
| C15; T / T | 5 | 0.469 ± 0.103 | 0.203 ± 0.043 | 0.088 ± 0.032 |
| C15; C / T | 3 | 0.314 ± 0.116 | 0.179 ± 0.075 | 0.084 ± 0.038 |

## Factorized paired estimates

Temporal minus mean-matched constant is the sign convention for each encoder or worker contrast. Each table cell is **paired mean ± sample SD; positive seeds / paired seeds**. The C05 interaction is computed within each seed as $(T/T-T/C)-(C/T-C/C)$, with encoder setting first and worker setting second. [All per-seed differences](../results/cosyne_2027_temporal_credit_layers/paired_by_seed.csv), [factor summary](../results/cosyne_2027_temporal_credit_layers/factor_summary.csv), and [absolute values with activity and silence](../results/cosyne_2027_temporal_credit_layers/absolute_by_seed.csv) retain exact values.

### Bits per activation, eligible units

The occupancy-weighted spatial score is divided by each unit's mean rectified response before averaging over field-eligible units. Eligibility requires at least 20 positive observations across three bins. This normalization removes a uniform gain change, but the eligible unit set can differ between independently trained models.

| Contrast | CA3 | Decoder 1 | Decoder 2 |
| --- | ---: | ---: | ---: |
| C01 encoder T − C | +0.088 ± 0.159; 4/5 | +0.680 ± 0.244; 5/5 | +0.636 ± 0.303; 5/5 |
| C05 encoder T − C; worker T | +0.477 ± 0.606; 4/5 | +0.438 ± 0.366; 5/5 | +0.128 ± 0.333; 3/5 |
| C05 encoder T − C; worker C | +0.480 ± 0.585; 4/5 | +0.559 ± 0.736; 4/5 | +0.494 ± 0.278; 5/5 |
| C05 worker T − C; encoder T | +0.141 ± 0.148; 4/5 | +0.057 ± 0.740; 3/5 | −0.203 ± 0.362; 2/5 |
| C05 worker T − C; encoder C | +0.144 ± 0.334; 3/5 | +0.177 ± 0.489; 3/5 | +0.163 ± 0.198; 3/5 |
| C05 paired interaction | −0.003 ± 0.204; 3/5 | −0.120 ± 1.016; 3/5 | −0.366 ± 0.540; 2/5 |
| C15 encoder T − C; worker T | +0.047 ± 0.331; 2/3 | +0.217 ± 0.462; 2/3 | +0.185 ± 0.213; 2/3 |
| C15 encoder T − C; worker C | pending | pending | pending |
| C15 worker T − C; encoder T | pending | pending | pending |
| C15 worker T − C; encoder C | pending | pending | pending |

### Amplitude-weighted bits per step, all units

This is the original unnormalized spatial score, averaged over **all** units, including silent ones. It changes when response amplitude or silence changes, even if information per activation is fixed. The per-step and per-activation tables therefore answer different questions; neither should silently replace the other.

| Contrast | CA3 | Decoder 1 | Decoder 2 |
| --- | ---: | ---: | ---: |
| C01 encoder T − C | +0.256 ± 0.097; 5/5 | +0.101 ± 0.029; 5/5 | +0.024 ± 0.014; 5/5 |
| C05 encoder T − C; worker T | +0.144 ± 0.116; 4/5 | +0.067 ± 0.089; 4/5 | +0.012 ± 0.036; 3/5 |
| C05 encoder T − C; worker C | +0.117 ± 0.219; 2/5 | +0.124 ± 0.077; 5/5 | +0.063 ± 0.044; 5/5 |
| C05 worker T − C; encoder T | −0.038 ± 0.101; 1/5 | −0.093 ± 0.112; 0/5 | −0.057 ± 0.058; 0/5 |
| C05 worker T − C; encoder C | −0.066 ± 0.096; 1/5 | −0.035 ± 0.100; 2/5 | −0.006 ± 0.041; 3/5 |
| C05 paired interaction | +0.028 ± 0.154; 3/5 | −0.057 ± 0.095; 2/5 | −0.051 ± 0.044; 1/5 |
| C15 encoder T − C; worker T | +0.114 ± 0.201; 2/3 | +0.010 ± 0.095; 2/3 | −0.002 ± 0.064; 1/3 |
| C15 encoder T − C; worker C | pending | pending | pending |
| C15 worker T − C; encoder T | pending | pending | pending |
| C15 worker T − C; encoder C | pending | pending | pending |

## Activity, silence, and relation to DG

C01's higher decoder-1 and decoder-2 information per activation does not come from firing more often: temporal encoder credit lowers their positive-sample fractions by 0.083 and 0.079 on average; decoder-2 is lower in all five pairs. C01 CA3 mean rectified response rises by 0.163 in 5/5 pairs, and it has 33 more eligible channels on average. Its normalized CA3 gain is small and mixed, so the strong amplitude-weighted CA3 gain should not be described as a uniform selectivity gain. The CA3 trace is the full 1,136-channel core state block, including the DG slots; it is not independent of DG.

In C05 with a **constant** worker bonus, two temporal-encoder seeds have 71 completely silent CA3 channels on this panel; the constant-encoder peers have none. The per-activation CA3 mean excludes ineligible channels, while the all-channel amplitude score retains them. This is one reason the two CA3 metrics disagree on the strength of that arm. C05 worker-temporal versus worker-constant differences in decoder positive fraction and mean response are mixed, but decoder-1/2 all-unit spatial scores fall in every temporal-encoder seed. [Per-run counts and response magnitudes](../results/cosyne_2027_temporal_credit_layers/absolute_by_seed.csv) allow that distinction to be audited.

The [DG fixed-panel analysis](checkpoint_fields_20261011.md#identical-observations-strongest-representational-result) provides the input-layer context: temporal encoder credit changes its amplitude-weighted spatial score by +0.0575 (5/5) in C01, +0.0333 (4/5) in C05 with temporal worker bonus, +0.0195 (2/5) in C05 with constant worker bonus, and +0.0358 (2/3) in C15. Holding C05 encoder credit fixed, temporal worker bonus changes DG score by −0.012 and −0.026 in the temporal- and constant-encoder arms. DG's score is an amplitude-weighted measure; the downstream normalized readout is a separate calculation.

## What the comparison does and does not establish

The absolute layer comparison freezes each trained model and replays one identical saved input sequence, including action history and episode resets. It tests the **between-trained-model** effect of credit timing on representation under that input. Training behavior can still mediate the learned weights; internal recurrent and goal states remain model dependent. One saved C15-policy state distribution is not a comprehensive state-space test. Five paired training seeds support C01 and C05 estimates; C15 has three pairs, and the five C05 seeds recur across its factor arms. The sample SD is descriptive, not a confidence interval over environments.

The separate [own-policy versus uniform-random frozen intervention](02_behavior_expression.md) changes executed actions **within** a model and measures behavior-conditioned expression. Its positive decoder-1 own-minus-random effects cannot be counted as temporal-credit gains. Policy-native 10k [checkpoint fields](checkpoint_fields_20261011.md#policy-specific-10k-decision-comparison) and 50k behavior probes sample different visited states; they are useful sensitivity checks, but the fixed-panel tables above carry the absolute representation comparison. Stored graph and executed-command results stay in the [worker and control report](04_worker_bonus_and_scope.md).

## Provenance and open cells

The [analysis manifest](../results/cosyne_2027_temporal_credit_layers/analysis_manifest.json) records every replay JSON and checkpoint hash, the saved-panel SHA-256, the canonical `training_runs.csv` hash, StudySpec fingerprints, schema `intrmotiv/study/v1`, and workflow version `1.14.1`. [Availability by factor cell](../results/cosyne_2027_temporal_credit_layers/availability.csv) separates verified training from completed layer replay. The [compute-node replay](../../hpc_runs/common_panel_layer_probe.py) uses the established frozen-policy decoder hooks and shared spatial metrics; the [ordinary-job launcher](../../hpc_runs/submit_common_panel_layers.py) generated a print-only plan and skipped completed preflights. The C01/C15 source ran in the allocated NEMO2 workspace after local and remote focused tests; [two exact submission ledgers](../results/cosyne_2027_temporal_credit_layers/submission_20261012T011345Z.tsv) [record all 18 jobs](../results/cosyne_2027_temporal_credit_layers/submission_20261012T011622Z.tsv), which finished with Slurm exit `0:0`. The original C05 20-run replay reused its separately staged source and passed the same DG reference check. All 38 accepted layer replays are at 100,007,936 frames; the C15 replacement source patch remains visible in the training catalog when its future cells are admitted.

**Next evidence gate:** Audit seed-99 layer job 8345388 and its JSON, then recompute C15 matched-age downstream encoder contrasts after both replacement JSONs pass checkpoint, DG parity, and finite-layer audits. The replacements and constant-worker extension use an isolated C15 learner recovery patch, whereas the original temporal-worker arms were trained from the earlier source; retain that source distinction in C15 interpretation. Do not substitute missing cells with zeros or compare different saved ages as terminal peers. No new training or duplicate layer job is required for the 38 completed original checkpoints.
