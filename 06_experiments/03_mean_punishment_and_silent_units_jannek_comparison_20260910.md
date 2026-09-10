# 03 — Mean, punishment, and silent units: comparison with Jannek

10 September 2026. Read-only investigation of saved reports and local source; no new training or runtime comparison. The evidence supports several concrete contributors, but not a uniquely identified cause of the Jannek-to-IntrMotiv difference.

**Most direct documented configuration difference:** the August 24 anti-collapse field-analysis slice explicitly disabled `encoder_batch_loss`, used mean/punish, and ran to 80M. Its previous healthy reference used encourage plus batch loss and 100M. Jannek's section 4.2 identifies batch loss as the strongest auxiliary effect under punishment. Therefore the collapsed slice did not preserve the earlier push–pull balance. It cannot isolate the effect of global punishment or row repulsion. This explanation applies to that slice, not every later run.

Sources: [anti-collapse field report](dg_anti_collapse_place_fields.md), [original HRL audit](hrl_batch1_results_and_next_iteration.md), original PDF section 4.2, and [core logic audit](../04_implementation/core_logic_audit_20260901.md).

**Direct sparsifying gradient:** with post-threshold activity $z=[v-b]_+$ and fixed event mask $M$, punishment contributes

$$
L_{punish}=\beta\langle M(E-\Delta)z\rangle.
$$

For an eligible active event with $\Delta<E$, its derivative with respect to $v$ is positive, so gradient descent suppresses that response. The event objective alone has no positive term maintaining activity. Under BatchNorm and shared projection weights, this does not mean every other observation's response must decrease; it changes the sampled response distribution and threshold crossings.

**Recovery weakness was inherited, not newly introduced:** Jannek's retained local code and the September 1 audit both show the old unused-batch objective acting on post-ReLU activity. Its direct gradient is zero for a unit below threshold across the learner batch. It may reinforce a unit that is absent in behavior but crosses threshold in learner replay, yet cannot guarantee revival. Shared data/statistics/optimizer changes mean silence is not an absolute absorbing state of the complete system. The corrected pre-threshold softplus loss supplies a revival gradient. It was not present in the early collapse runs.

The old `unused_sequence_loss` also combines a preceding `>=2R` occupied-slot mask with an `==R` condition in the inspected local Jannek function, making that particular mask empty for positive R. This is a local-source finding, not verification that every report checkpoint used these exact lines. It further illustrates why flags cannot be treated as proof of effective positive learning.

**Mean's sentinel contamination:** both retained Jannek code and the sealed September 8 `legacy_reward_streams` calculate mean over the entire internal-distance tensor, including entries equal to empty/no-event sentinel $E$. Let $q$ be the fraction of non-sentinel entries and $\mu$ their mean. Exactly for that tensor,

$$
\bar\Delta=(1-q)E+q\mu.
$$

The average mean-centered reward over its non-sentinel entries is

$$
\beta(\mu-\bar\Delta)=-\beta(1-q)(E-\mu).
$$

As $q\to0$, this approaches punishment. The encoder's additional event mask can further change which entries receive gradients; this identity is not a claim that gradient averages equal reward averages. Nor is $q$ DG activation density. Example: $E=71,\mu=8,q=.1,\beta=.1$ gives baseline 64.7 and mean finite-entry reward −5.67, versus punishment −6.3. Positive centered values can reside on sentinel entries that are not eligible for event credit. This mechanism is shared with Jannek, so it amplifies a sparsity shift rather than independently explaining its origin.

**Additional changes with different evidential status:**

- Global softplus punishment is an extra suppressive gradient even below threshold, unlike event-only punishment. Population/collision terms and removal of positive auxiliaries change the balance; identify actual flags by batch. Row-angle repulsion alone is not a general activity-restoration mechanism.
- The frontend changed from Jannek's checkpoint-loaded `pretrained_resnet` path to `layer2_resnet18`. The September 1 audit confirmed that affected early layer2 trunks were reinitialized by the policy initializer and then frozen. Jannek's path was also potentially exposed; actual Jannek checkpoint weights were not verified here. These are not established matched pretrained feature distributions.
- Threshold 2.43 was already used by Jannek. The numerical threshold itself is not a new explanation. Changes in feature tails, visits, projected moments, and actor/learner normalization can change its effective sparsity substantially.
- Actor running moments versus learner batch moments, repeated moment updates, and pre-step row normalization were inherited implementation weaknesses. They can interact with a changed feature distribution; they are not proven newly introduced causes.
- The separate September 5 `running_consistent` experiment supplies direct evidence of a later normalization-driven regression. Matched MON density fell .0282→.0125 (ARR) and .0285→.0112 (SRC), with zero replacements. A real-feature audit found near-identical forward masks but markedly different gradients; fixed-stat row gradients aligned strongly with the common feature mean. This occurred after the August collapse and cannot explain it retroactively. Legacy normalization was subsequently restored.
- Sparse worker reward acts indirectly through behavior when PPO-to-DG is stopped. DG still receives its own event credit. Reduced/biased visitation can reduce opportunities for recovery; this must not be described as simply gating the encoder loss by target reward.
- All-transition averaging makes event-loss magnitude shrink with event frequency. This attenuates both positive and negative event terms, not uniquely positive reinforcement. It changes their balance relative to losses with different masks and denominators.

**Scope:** not every new condition collapsed. August 25 encourage-plus-batch-loss runs retained substantial activity, and corrected-core encourage runs had about .021–.028 online density. Conversely, the anti-collapse probes found 13–16 silent units in selected deterministic 2,001-decision evaluations. Silence within a learner batch or short probe does not establish permanent global silence.

**Conclusion:** the most concrete explanation for the anti-collapse slice is removal of positive batch recruitment while retaining mean/punish, compounded by a weak inherited revival path. Mean itself becomes more punishment-like as sentinel entries dominate. Changed visual features and normalization provide additional routes into low-activity regimes; only the later normalization regression has its own targeted gradient evidence. A single causal attribution for the original-versus-new comparison remains unproven.

**Authoritative local code inspected:**

- `/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/jannek/dmlab/custom_learner.py`: `_extra_encoder_loss`, `_encoder_loss`, and mean-baseline calculation around lines 1530–1598 and 2000–2015.
- Same source tree `custom_encoder.py`, checkpoint-loading path around lines 764–799, and `experiments/distance_metric_run_internal.py` for the retained launch defaults. These are present-day retained files, not an authenticated original run archive.
- `hpc_runs/source_snapshots/persistent_intrinsic_control_hotfix_20260908.tar.gz`: `custom_learner.py::legacy_reward_streams`, inspected by reading selected files into `/tmp`.
- [September 5 gradient audit](landmark_normalization_gradient_audit.md) and [matched update-contract diagnosis](encoder_decoder_update_contract_batch_diagnosis.md).

**Reusable lesson:** start with exact positive-loss flags and feature identity; distinguish a newly changed factor from an inherited failure amplifier. Inspect every baseline's averaging support, including sentinel entries, before interpreting it as centered event feedback. Restore a truly matched reference before allocating causal credit to one regularizer. No files in the runtime checkout, checkpoints, study definitions, or jobs were modified.
