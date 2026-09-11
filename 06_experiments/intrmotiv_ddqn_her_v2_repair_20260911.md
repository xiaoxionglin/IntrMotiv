# DDQN/HER v2 repair and qualification

The bounded frozen-reference repair is implemented. This is still a selected-parent local-control diagnostic: no DG adaptation, neurogenesis, manager, new goals, parent demonstrations, new reward, or new RL algorithm. DMLab performance remains unqualified until matched-command evaluation.

## Provenance and scope

The supplied audit was extracted under `/tmp/intrmotiv_repair_audit`, and its original `diagnostics.py` ran unchanged with the `SF_git` Python. Its three archived files match the **live v1 production source**, not only the old report: contracts `bac5a766…`, worker `cafc265c…`, replay `f1a189cc…`. Full hashes and scheduler evidence are in [v1 live provenance](data/intrmotiv_ddqn_her_v2_20260911/v1_live_provenance.txt). Archived source was never installed over current code.

Development checkout: `/tmp/intrmotiv_ddqn_her_v2_20260911`, branch `codex/ddqn-her-v2`. Runtime checkout: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_ddqn_her_v2_20260911`, copied from the actual isolated v1 runtime before applying the repair. Existing v1 jobs, source, checkpoints and outputs were preserved. The inherited runtime includes the previously documented DMLab terminal/vector-worker and matched-evaluator patches; these external runtime files differ from the unpatched vault base. [Source comparison](data/intrmotiv_ddqn_her_v2_20260911/source_comparison.json) separates inherited runtime hashes from repaired package/study files, which match byte-for-byte.

Parent remains `DGC_DIRECT_WORKER_F16_S99`, exactly **25,001,984** source frames, SHA-256 `04ccec999e7b9748a960b6b421a91e9bcf6ce3ce9c44cea246359d34f0633585`. Its authoritative path remains in [the canonical parent manifest](data/intrmotiv_ddqn_her_20260911/parent_manifest.json). Seeds 8/99/123 are child learner seeds, not independent parent pretraining seeds.

## Reproduced findings and repair

| Finding | Classification and v1 evidence | v2 behavior |
| --- | --- | --- |
| Additive clock readout | Verified representational limitation: changing budget shifts Q identically across states; fixture difference only $1.79\times10^{-7}$ | Decoder output and both clocks enter one joint hidden layer with tanh before the fresh Q head |
| Duplicated evaluator readout | Verified maintenance/parity gap | Collection and TD call `step` → `readout` → `readout_state`; evaluation calls the same `readout_state` on already-updated memory, with no second CA3 advance |
| Distant HER hit discarded | Verified supervision limitation: hit at decision 40 retained 16 positions and zero reward | Materialize through the first verified hit, at most 64 decisions; split only with a preserved virtual start/deadline and loss suffix |
| Virtual budget restarted in suffixes | Verified remaining-budget coverage limitation | A suffix at offset 24 starts at budget 40; exact physical prefix rebuilt independently for online and target |
| Invalid final row hides prior observed success | Verified bookkeeping bug | Known valid hit remains eligible even before an invalid final row; episode completion recorded independently of successor validity |
| Default environment remaining = 1800 | Verified false metadata | Unknown is `None`; 1800 is only the physical clock's input scale |
| Certified final observation discarded | Verified collector bug; also verified outer vector fields are dropped by legacy `_select_info` | Read certified outer final observation separately, encode it only as a successor, and retain reset image only for the next episode |
| Infrequent target copies | Optimization hypothesis, not a correctness bug | Exposed period in optimizer-update units; selected 100 using controlled neural sparse-path evidence below |
| Longer HER batches get more TD positions | Comparison confound if nominal batch count retained | Exact 256 valid TD positions/update in both arms; complete remaining suffix carried into the next update |
| Sparse achieved-goal coverage | Unresolved empirical hypothesis | Per-goal real events, arrivals, contributing episodes, HER request/realization, hit offsets, reward segments and remaining-budget histograms now logged |

Eligible HER goals are sampled uniformly over distinct observed registry goals, rather than weighting a goal by repeated event occurrences; its earliest verified hit defines termination. This is an intentional v1-to-v2 sampling change.

Child checkpoint, replay, conversion and run schemas are versioned to v2. Old v1 child checkpoints are rejected with an explicit incompatibility error; no migration is claimed. Parent actor logits are never treated as Q-values. Parent decoder weights are retained, while joint/Q layers receive matched fresh initialization for each paired child seed.

`PositionBatcher` never discards a reward to make an optimizer batch fit. If replay sampling fails midway, it retains already-selected positions for retry. A final incomplete pending attempt can leave at most 63 positions queued at training stop; its length is reported. Requested/realized replay counters refer to segments, including consistent splits. Both gradient updates and actual TD positions are reported separately.

## Boundary and bootstrap contract

| Boundary | Reward/loss | Bootstrap |
| --- | --- | --- |
| First observed commanded hit | Reward 1, include that position; stop all later loss | No |
| Virtual budget expires | Include final valid position, reward 0 unless hit | No |
| True physical termination, certified successor | Include position and observed reward | No |
| Environmental truncation, certified successor | Include position and observed reward | Yes from certified final state, unless hit/budget also terminates |
| Missing/uncertified final successor | Exclude row entirely; no unknown reward | No TD target for excluded row; preceding valid transition remains an ordinary bootstrap |
| Replay chunk/future currently incomplete | Include only known valid transitions | Yes across nonterminal chunk boundary |
| Autoreset image | New physical episode only | Never used as previous episode's successor |

Current F16 memory writes are goal-independent. `learn_batch` explicitly rejects relabeled write-conditioned workers; future support requires reconstruction of virtual command history rather than using recorded original commands. The production collector continues to reject write-conditioned parents.

## Numerical qualification

58 focused tests pass locally and on the isolated NEMO2 runtime: original contracts, new boundary/suffix/parity/expressivity tests, and canonical StudySpec tests. The 40-decision fixture produces exactly one terminal reward, no post-arrival loss, and budget-40 suffix training beginning at physical offset 24. Histogram tests count 64 original positions across budgets 1–64 and 40 HER positions across budgets 25–64. Identical-state training/evaluation Q parity is exact, with unchanged memory.

The supervised expressivity fixture uses the production readout with known one-/two-step values and a strict budget-dependent action switch (immediate 0.5 success probability versus certain two-step arrival). The acceptance threshold remains maximum absolute error <0.04. Constant-rate fitting initially passed locally but failed remotely (0.0866). An overly early learning-rate reduction also failed. The final fixture uses 6,000 supervised fitting steps, reducing the rate after 4,000; remote error is **0.00337**, with the correct action switch. This is fixture fitting, not a change to the DMLab optimizer or reward.

The deterministic sparse-path test uses the production `QWorker` and `DoubleDQNLearner`, fresh neural Q initialization, one-step Double-DQN targets, two actions (advance/fail), and the same fixed transition dataset, optimizer-update count and TD-position count within every cadence pair. It is not a tabular propagation argument. At 10,000 updates:

| Path length | Known start value | Start Q, copy every 1000 | Start Q, copy every 100 | Max value error, period 100 | Greedy path success, period 100 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 | 0.97030 | 0.96991 | 0.97021 | 0.00048 | Yes |
| 16 | 0.86006 | 0.23527 | 0.85479 | 0.00535 | Yes |
| 40 | 0.67573 | 0.00091 | 0.67646 | 0.00571 | Yes |
| 64 | 0.53091 | -0.00168 | 0.52705 | 0.00524 | Yes |

At 5,000 updates even period 100 failed the 64-step fixture; both periods were extended equally to 10,000. Period 1000 failed the longer value tests and greedy control on 40/64 steps. Select **100 optimizer updates per target copy** for this diagnostic. This supports faster refresh under these fixtures, not a general neural propagation bound or a promised DMLab rescue. Numerical artifacts retain both budgets; [final-source qualification](data/intrmotiv_ddqn_her_v2_20260911/qualification_final.json) reruns the complete suite after fixture stabilization: [5k](data/intrmotiv_ddqn_her_v2_20260911/qualification_5000.json), [10k](data/intrmotiv_ddqn_her_v2_20260911/qualification_10000.json), and [final expressivity](data/intrmotiv_ddqn_her_v2_20260911/expressivity_final.json).

Reproduce with:

```bash
python -m unittest hpc_runs.test_intrmotiv_offpolicy hpc_runs.test_intrmotiv_offpolicy_v2 hpc_runs.test_intrmotiv_study
python -m hpc_runs.intrmotiv_offpolicy.qualify --output /tmp/ddqn_v2_qualification.json
```

## Actual v1 5M evidence

All six v1 jobs **8056085–8056090** completed with exit 0. Each reached **5,000,064 child frames**, **19,265 updates**, **19 target copies**, and **672 excluded invalid-final rows**; frozen encoder hashes remained unchanged. All runtime gates pass. See [runtime audit](data/intrmotiv_ddqn_her_v2_20260911/v1_runtime_audit.json).

Canonical `collect-online --latest-common` uses **4,750,064–5,000,064** frames across all six runs:

| Three-seed mean | DDQN | DDQN+HER |
| --- | ---: | ---: |
| TD loss | 0.002309 | 0.002302 |
| Sampled selected-action Q outside [0,1] | 5.26% | 16.59% |
| Realized HER segment fraction | 0 | 35.44% |
| Cumulative throughput, frames/s | 808.8 | 782.6 |

Per-seed numbers and exact provenance are in [canonical terminal analysis](data/intrmotiv_ddqn_her_v2_20260911/ddqn_v1_terminal_for_v2_20260911/per_run.csv). These online metrics cannot establish independent commanded control. The old 500k one-start smoke is not a 5M evaluation. No completed multi-start 5M child-control artifact was obtained for this repair, so no 5M success claim or improvement claim is made.

## Canonical v2 matrix and resources

- Schema `intrmotiv/study/v1`, workflow **1.8.0**.
- [Six-run StudySpec](../hpc_runs/studies/intrmotiv_ddqn_her_v2.study.json), SHA-256 `716caaf3d3358a7813d2af37c691dbf92eb0193c851f3f965638f40ac1056cab`.
- DDQN versus DDQN+HER; child seeds 8/99/123; **5M additional frames per run**. Run names come only from StudySpec; [rendered matrix](data/intrmotiv_ddqn_her_v2_20260911/staged_runs.json).
- [Two-run preflight](../hpc_runs/studies/intrmotiv_ddqn_her_v2_preflight.study.json), SHA-256 `a95d0e5ab85ebb3568f09ad42bb69409fe2b5b1e0c7d8f82835fad1367f79516`: both arms, seed 99, 250k frames, enough for a physical reset and seven target copies.
- 32 environments, CPU, 8 learner threads, 40 allocated CPUs, 80G/job, no GPU. Preflight wall limit **1 hour/job**. Staged full runs use **8 hours/job**, a cap rather than measured duration. Maximum staged allocation: 6 × 40 × 8 = 1,920 CPU-hours; actual throughput must be checked in preflight. No increase in simultaneous job count versus the authorized six-run scope.
- Interaction budget, frozen source/normalization, goal registry 1/4/11, action repeat 4, permitted observations and epsilon schedule are unchanged.
- Intentional update differences: nonlinear joint readout; coherent HER/full valid original sequences up to 64; boundary corrections; **100** target period; **256 valid positions per optimizer update**, every **64 accepted aggregate decisions** after 16,384 accepted warmup decisions. Thus 4 valid positions/accepted post-warmup decision, versus approximately 3.4 realized previously. This is controlled across v2 arms, not an equal-compute v1/v2 comparison.
- New data root `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_ddqn_her_v2_20260911`. All caches, temporary files and logs remain in the allocated workspace.

Ordinary preflight jobs **8056773 / 8056774** were submitted after print-only inspection and passing source tests. Submission: `_slurm/intrmotiv_ddqn_her_v2_preflight_20260911/20260911T193626Z`. Full matrix was first printed and audited at `_slurm/intrmotiv_ddqn_her_v2_20260911/20260911T193631Z`. After both preflights passed all required runtime gates, the unchanged six-run matrix was submitted at `_slurm/intrmotiv_ddqn_her_v2_20260911/20260911T194512Z`. Job IDs: **8056789, 8056790, 8056791, 8056792, 8056793, 8056794**. [Submission audit](data/intrmotiv_ddqn_her_v2_20260911/production_submission_audit.json) confirms six jobs, exact commands and workspace-only output paths.

Commands from the isolated NEMO2 v2 checkout:

```bash
python -m hpc_runs.intrmotiv_offpolicy.audit_runtime hpc_runs/studies/intrmotiv_ddqn_her_v2_preflight.study.json /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_ddqn_her_v2_preflight_20260911
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh hpc_runs.intrmotiv_ddqn_her_v2 --print-only --slurm_sbatch_template=hpc_runs/intrmotiv_offpolicy/nemo2_ddqn_v2.sh --slurm_timeout=08:00:00
```

The full sweep was released only after the preflight runtime gate passed. Repeat print-only review if the study changes. Preserve the new source fingerprint; do not patch a running source checkout.

## Completed v2 runtime preflight

Both jobs completed with exit 0: DDQN in **5:43**, HER in **6:37**. Each reached **250,112 frames**, **720 updates**, **7 target copies**, **184,320 valid TD positions** and **32 invalid-final exclusions**. Encoder freeze and paired initialization passed. The [formal runtime audit](data/intrmotiv_ddqn_her_v2_20260911/preflight_runtime_audit.json) passed; learner-active throughput was **971.4 / 776.1 frames/s** for DDQN/HER. Pending suffixes contained 11/18 positions at shutdown and are explicitly accounted for.

HER's achieved-event counts for goals 1/4/11 were **451/118/906**, spanning **27/29/42 distinct stream-episodes**. Its replay supplied **212/363/1,589 reward-bearing segments**, and **71/170/532 valid loss positions at remaining budgets 1–8**. Thus all three goals have observed successes and lower-budget supervision, although coverage is uneven. These are overlapping replay segments and collector events, not independent successes. Training commanded arrivals were 17/336, 29/372 and 104/356 completed attempts; these epsilon-greedy behavior rates do not qualify independent goal control. [Full counters](data/intrmotiv_ddqn_her_v2_20260911/preflight_final_metrics.json).

## Control evaluation and recommendation

The v2 manifest adapter defaults to **four source landmarks × three repeated exact starts**, with alternative commands from the unchanged registry and the same 64-decision deadline. It reuses the canonical matched-intervention evaluator and ordinary independent-job telemetry submitter. It now reports per-goal successes, denominators, censoring, failure-inclusive arrival time and child seed, as well as aggregate paired arrival lift. Partial source coverage is reported rather than hidden. Exact paired physical and recurrent starts must be verified in the returned rows; this is not a one-start smoke.

Use a fresh telemetry manifest built from actual checkpoint discovery after completion, and `run_evaluation_v2.sh` as the runner. Detector arrival remains explicitly separate from independently qualified spatial arrival; no spatial qualification has been added or claimed. Evaluate terminal children before claiming success or adding representation/manager mechanisms. If achieved-goal coverage is inadequate, first report the per-goal/episode evidence, then propose broader goals or parent collection as separate experiments.

## Reusable workflow lessons

Live source SHA-256, checkpoint SHA-256, StudySpec fingerprints, submitted `jobs.tsv`, runtime receipts and exact-frame canonical analysis are authoritative. Archived audit assertions reproduced old behavior; they were not reused as new acceptance tests. Test the whole vector autoreset path, because preserving a final observation in the worker is insufficient when a later selector drops it. Compare target cadence on multi-step neural fixtures with equal budgets, and retain failed short-budget results. Supervised expressivity tests need a convergence phase on both Python/PyTorch environments; do not loosen numerical thresholds to hide optimizer sensitivity. Fixed valid-position batching with persistent suffixes makes longer HER unrolls a measurable data change rather than an accidental compute advantage.
