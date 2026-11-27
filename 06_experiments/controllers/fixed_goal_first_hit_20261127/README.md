# Minimal control change after reward-free exploration

Date: 2026-11-27. Owner: this report. Scope confirmed by the user: every reachable learned DG landmark. **Status: theory and initial graph simulations complete; all-seven-goal toy passes in three seeds; full DMLab control validation open.** The user requests GeoMem-style controlled simulations first and DMLab execution on NEMO. No DMLab installation was performed locally.

## Question and candidate

Can the existing DG/CA3 state support arbitrary landmark commands when the acting controller is trained directly to reach the requested landmark? The selected candidate preserves the visual trunk, DG identities, CA3 dynamics, and current goal-conditioned MLP. It changes the controller's objective, replay goal labels, and acting-head selection together as an explicit contract: fixed requested goal, unit first-arrival target, and action selection from that trained value. The first prototype used the available FiLM interface; the subsequent matched legacy-decoder test shows that addition is unnecessary on these tasks.

The [theory and prospective gates](../../../02_algorithm/fixed_goal_first_hit_control.md) were written before simulations. They give a finite-horizon Bellman proof and state the representation and action-support assumptions. The result is conditional sufficiency, not proof that the current learned DG/CA3 state satisfies those assumptions. The test of minimality is whether this controller change works before adding memory or a world model.

Reward-free means the exploration transitions contain no external task reward. Goal labels are generated from frozen DG events. Offline control fitting uses this data; goal evaluation freezes learning. Fitting a new controller is not claimed to transfer the original worker policy. The representation and goal-command interface are retained. Synthetic DG cues in the small tests are controlled stand-ins, not learned DMLab place fields.

## Why this differs from previous attempts

The audited clean desktop runtime is `/home/xiaoxiong/SFgit/SF_hipposlam` at `206ee8330f048b910c19ea9b82e6215a58a63c8d`. Its `controller_q.py` declares that hindsight values never choose actions. Main replay keeps behavior and successor manager commands; virtual-goal replay rewrites both commands but trains only the auxiliary. Acting uses the main continuing-return head. This is a verified implementation distinction, not evidence that correcting it will solve DMLab navigation.

The [goal-value-stop study](../../reward_credit/goal_value_stop_dg_20261009.md) already trained eighteen 100M runs. Its terminal PPO/GAE change did not establish replicated control benefit; its executed command-intervention evidence is incomplete. Therefore this record does **not** treat the absence of a positive result as proof that option termination has no effect. The [episode-long DG comparison](../../dg_goal_control/episode_long_dg_comparison_20261009.md) supplies separate negative exact-start evidence for prescribed fields and FiLM. No completed run or TensorBoard analysis is repeated here.

The proposed all-goal labeling also differs from future-selected hindsight: it retains failed stochastic transitions for every goal. Goal-conditioned value functions and reward-free exploration are established methods ([UVFA](https://proceedings.mlr.press/v37/schaul15.html), [Jin et al.](https://proceedings.mlr.press/v119/jin20d.html)); stochastic hindsight bias is documented by [USHER](https://proceedings.mlr.press/v205/schramm23a.html). This study makes no general algorithmic novelty claim.

## Local experiment record

The first candidate succeeds on the declared graph tests while retaining the existing CA3 update and runtime FiLM/head classes. It does not establish that a particular pretrained DMLab representation is sufficient.

| Test | Result | Evidence boundary |
| --- | --- | --- |
| Exact covered branching graph, deadline 8 | 12/12 nontrivial pairs; oracle discrepancy zero | Tabular mechanism test using independent one-step resets, not natural exploration |
| Same graph, one essential action absent from replay | 8/12; four failures retained | Missing support cannot be repaired by the objective |
| Stochastic safe versus risky route | All-outcome replay selects success 1.0; successful-transitions-only subset selects success 0.1 | Illustrates outcome-selection bias; not a full HER comparison |
| Neural factual replay, two endpoint goals, deadline 6 | 12/12 pairs in each of seeds 8, 99, 123 | Fixed synthetic DG; 4,096 random-exploration transitions per seed |
| Neural factual replay, sparse cue-history task | 4/4 in each seed; history removal 2/4; shuffled command 0/4 | Existing CA3 is necessary and sufficient for these fresh-cue trials |
| Neural factual replay, **all seven DG identities as goals** | **42/42 nontrivial pairs in every seed**; every goal succeeds | One fixed seven-node graph with distinct current DG cues; not visual representation learning |

The all-seven-goal extension uses the exact same reward-free trajectory files as the two-target graph experiment. It assigns all goals after collection, enumerates goal labels independently of outcomes, and performs no evaluation updates. Its informative alternative-command lift is $26/35=0.743$ in every seed; seven alternatives already satisfied at the initial state are excluded from that contrast and counted explicitly. There are 49 declared source/goal combinations, seven trivial initial-goal cases, and 42 nontrivial cases per seed.

The exact finite test also finds that greedy fixed-policy successor predictions solve the original branching graph. That graph alone does not establish a need for optimal Bellman backups. A separately labeled post-audit counterexample gives optimal-control success 1.0 versus 0.75 for greedy fixed-policy prediction. See the [finite record](finite/README.md), [pair table](finite/per_pair.csv), and [stochastic witness](finite/stochastic_witness.json).

The neural learner uses actual source `TargetFiLMDecoder` and `ControllerQHeads`, with the unchanged CA3 adapter. Its fixed settings are memory length 8, injection repeat 1, hidden width 128, deadline 6, 2,500 Adam updates, batch size 256, learning rate 0.003, and target refresh every 100 updates. These toy settings are not the DMLab checkpoint configuration. Distillation from an empirical value table is reported separately as an approximation upper bound; the replay-TD arm constructs targets from stored CA3, actions, successor CA3/DG and physical termination, without a latent-state table lookup. A latent-state relabeling test verifies that TD updates do not use hidden node identities.

![Frozen replay control and interventions](figures/replay_control.png)

Each point is one training/data seed, with exact deterministic evaluation over 12 or four nontrivial pairs. No confidence interval is inferred from three copies of the same task. Correct and shuffled commands use the same trained head. The history ablation removes earlier CA3 slots. The blind control also removes the goal input and trains a synthetic continuing reward objective: it is **not** a reproduction of the old controller or an isolated head-routing contrast. Editable [SVG](figures/replay_control.svg), [PDF](figures/replay_control.pdf), and [data/font mapping](figures/replay_control.mapping.json) accompany the inspected PNG.

All 15 trained readouts (12 initial fits plus three all-goal TD fits) were reloaded and reproduced their recorded trials exactly, with unchanged parameter hashes during evaluation. Results and audits: [initial neural results](neural/results.json), [saved-weight replay](neural/replay_audit.json), [all-landmark results](neural_all_landmarks/results.json).

**Limits.** Successful action selection does not establish calibrated probabilities: maximum absolute per-deadline value errors are 0.45–0.58 in the initial suite and 0.56–0.61 in the all-goal extension. The existing deadline head is affine in positive remaining time at fixed state/goal and cannot exactly represent arbitrary first-hit probability curves. The all-goal run uses 4.48M first-hit labels versus 1.28M in the two-goal TD run; equal optimizer steps do not mean equal supervision cost. Distillation also has a different target count. The sparse task evaluates fresh histories within the finite memory window; one seed-8 replay transition had already lost its initial cue. No universal CA3 sufficiency, held-out-map transfer, or training-efficiency advantage is claimed.

## Runtime qualification

The actual current C05 checkpoint uses the ordinary **legacy MLP**, not FiLM. Its full input already includes the target one-hot. A six-fit follow-up therefore repeats all-seven-goal and sparse-context replay-TD with the actual `MlpDecoderJit` architecture (`[128,128]`, ReLU, JIT enabled), using the same saved exploration data. It reaches 42/42 and 4/4 pairs respectively in every seed; removing memory gives 2/4 in the contextual task. [Legacy results and saved-weight audits](neural_legacy_r1/results.json). These are fresh toy decoder weights, not transferred pretrained weights. This supports preserving the original decoder in DMLab; adding FiLM is unnecessary for the tested repair. Including this follow-up, 21 fits completed successfully. An initial JIT target-copy failure occurred before optimization and is retained as an infrastructure failure.

The [GeoMem reference audit](geomem_reference/README.md) directly reuses its graph constructors, separates observations from latent nodes, distinguishes controlled action kernels from fixed-policy edge probabilities, and checks retained-cue versus expired-cue versus undisclosed-context failures. It involved no native DMLab. Those are the relevant conventions for the beginning of this investigation; its graph correctness labels are not imported as external-reward supervision for IntrMotiv.

The copied-head adapter passes terminal-success, same-goal, deadline-zero, missing-successor, unsupported-continuation, source-immutability and training-gradient checks. A separate structural gate rejects goal-dependent geometry, goal-written worker states and unverified state layouts; replacing only the goal slice is not valid for those variants. [Adapter certificate](runtime/adapter_qualification.json) · [core-layout certificate](runtime/core_eligibility.json).

A pre-existing native DMLab installation reproduced two 64-decision no-reward prefixes exactly during an initial compatibility check. No checkpoint or learned controller participated, and no installation was performed. That check was unnecessary for the initial graph tests; following the user's correction, further native DMLab execution is confined to NEMO. The [historical local smoke receipt](runtime/native_smoke.json) is infrastructure evidence only.

## Full DMLab execution boundary

The user confirmed successful manual NEMO2 login on 27 November. The source audit selected `COS27_C05_TEMPORAL_TEMPORAL_S99`, checkpoint `checkpoint_000006104_100007936.pth`, SHA-256 `f35464c7e058df3ac51b676d5560470aa3cddd3c57f02ed1c0012a5fbcfaed38`. Its exact frozen source/configuration and parent StudySpec are recorded in the [qualification protocol](runtime/qualification_protocol.json). The source uses winner-based DG arrivals; exclusive single-active arrivals are a separate secondary metric. No new DMLab production jobs have been submitted by this study.

A suitable reward-free pretrained checkpoint was not found in the inspected local transfer directories. The local online C15 trajectory arrays were produced while DG changed and omit the full actor state; they cannot substitute for stationary frozen replay. The existing remote 60k prediction dataset was collected after externally rewarded fine-tuning and is not eligible reward-free evidence.

The first NEMO qualification uses 5,000 frozen factual decisions, 100 fit updates, and a tiny matched two-goal panel. This qualifies integration only. Job `8949867` failed before loading a model: the older donor tree lacked `sf_shared.checkpoints`, a dependency of the current evaluator. The [release checks](runtime/remote_release_qualification.json) and [submission receipt](runtime/qualification_submission.json) are retained; the failure produced no replay, fitted controller or control evidence. An import-only regression now checks the complete canonical entry points before a corrected isolated release is submitted.

The pipeline preserves full actor state, executed actions, genuine successors, terminal validity, native per-step zero reward and source hashes. The saved readout binds checkpoint, configuration, detector semantics, source arithmetic and weights; relocation is accepted only with identical module bytes. Compare original and new controllers from identical physical and recurrent starts, with the same goals and deadlines. Before efficacy results, the larger budget is fixed at 100,000 new reward-free decisions (seed 73002, 20% uniform action mixture), 30,000 Adam updates, batch 128, learning rate 0.0003, horizon 256, and target refresh every 100 updates. A whole-episode 10% holdout is diagnostic only; use final weights without selection. This permits 300 target generations and at most 61.44M goal labels, but does not establish action support, convergence or calibrated probabilities.

Corrected qualification `8950150` loaded the exact checkpoint and collected all 5,000 decisions. It passed the frozen-state, zero-native-reward and complete-transition checks, then failed while serializing a NumPy action-count integer into JSON. The saved replay lacks its completed certificate and is excluded from fitting. Normalize that count at the snapshot and metadata boundary, then repeat the unchanged bounded qualification; no control results exist from this job.

Keep the separate [80-run action-transport study](../../../05_plans/ca3_action_transport_dmlab_handoff_20261113.md) independent. Its most recent inspected chat record (13 November) reports local qualification and **0/80 production jobs submitted**. It is not a receipt for this experiment or a reason to duplicate that matrix.

New training studies must use schema `intrmotiv/study/v1`, the canonical workflow (currently `1.15.2`), and a generated SHA-256. Do not create a nominal production StudySpec before the selected controller is qualified and the actual source checkpoint/config are pinned. All remote outputs go under `/work/classic/fr_xl1014-corridor-geometry`; synchronize and test a release-specific workflow before use. Follow [the standard workflow](../../../04_implementation/standardized_study_workflow.md).

## Required full-scale evidence

| Component | Required result | Current status |
| --- | --- | --- |
| Place fields | Frozen 10k-decision checkpoint maps, pre-threshold activity, silent units, information and peak diversity, with complete unit/occupancy denominators | Exact-checkpoint common-panel and source-policy artifacts audited and reusable |
| Trajectories | Matched-command paths, occupancy, first requested-hit latency, failures and censoring | Requires executed frozen interventions |
| Graphs | Stored support distinguished from prospectively executed command outcomes; unattempted edges retained | Requires frozen checkpoint and replay support audit |
| Goal control | Every eligible DG identity, identical-start alternative commands, fixed deadlines, detector versus physical arrival, worst-goal success and complete denominator | Open; primary completion gate |

The current evaluator's headline lift filters censored trials. The new analysis must retain them explicitly, report success per issued trial and censoring separately, and preserve missing starts rather than silently dropping them. First-distinct landmark outcomes remain secondary: routes to arbitrary requested goals can legitimately pass other landmarks. In DMLab an unobserved landmark or action is not thereby proved unreachable; it remains outside the identified support, with that denominator reported. Only the finite systems have an exhaustive reachability oracle.

The [field reuse audit](runtime/field_reuse_audit.json) verifies the donor checkpoint/configuration, shared observation panel, and existing field NPZ bytes. Both 10k-decision panels have all 16 DG units active and 16 distinct peak bins, but none qualifies as a single field under the saved component criteria. The common panel visits 242/361 bins, with active-map cosine 0.0253 and mean spatial information 0.0949 bits; the source-policy panel visits 190/361, with cosine 0.0515 and information 0.2000 bits. These continuous post-inhibition maps are distinct from winner-event maps. Their half-peak components can describe physical endpoints secondarily, but cannot establish an exact winner footprint without per-step DG. New command trials therefore record processed DG and pose together; pose remains telemetry only. No repeated native field sweep is needed for the unchanged representation.

## Reuse and next action

The authoritative starting points are the current runtime's Q/replay/decoder code, the unchanged CA3 adapter in `exports/intrmotiv_prediction_starter`, GeoMem's controlled environment/sensing contracts, and the canonical study/evaluator workflow. Prior value-stop and HER reports prevent repeating already-tested mechanisms. The next step is the bounded NEMO qualification using the pinned reward-free donor and certified frozen replay. Remote access is confirmed.

Parallel edits initially changed two source-file hashes while the neural job was running. Exact original bytes were recovered and archived; function-level equivalence checks distinguish harmless report/docstring changes from executed logic. The all-goal extension used the new source-snapshot helper before training and completed without source drift. Future runs should archive dependencies before starting and check both saved-file integrity and loaded/current source identity.
