# Five-cue reward transfer: campaign and matched frozen-DG controls

## Architecture factorization

| Factor | Levels / setting | Interpretation |
| --- | --- | --- |
| Downstream task | Five invisible reward locations, uniformly selected per episode and cued by stable number instruction | Broader multi-goal transfer than the earlier single fixed reward |
| Geometry / sensory map | Original openfield_map2 retained | Reward location changes without changing the base visual geometry |
| Source representation | Two surviving source checkpoints plus scratch/transfer variants in the eight declared arms | Source identity must be tracked separately from transfer operation |
| Goal signal | Number instruction specifies the external reward location | Instruction is downstream task context, not privileged coordinate input |
| Controller / transfer | Waypoint manager or flat goal mixture depending arm; external reward drives learning | Different arms reuse different amounts of source structure |
| Evaluation | 48-run, 8-arm × 3-seed production design, 75M frames | This is designed to test reusable multi-goal structure rather than one-site specialization |

Cross-report context: [[README|factorized experiment synthesis]].


## Task and study

The source `openfield_map2` geometry and visual assets are retained. Each episode uniformly selects one of five invisible $+10$ reward cells, ends on first entry, and presents its stable number instruction to the high-level waypoint manager or the flat arm's goal mixture. All five cells are excluded from spawning. The worker and DG source-map identity remains the original map-3 input.

| Instruction | Reward center $(x,y)$ | Origin                         |
| ----------- | --------------------- | ------------------------------ |
| 1           | $(350,1950)$          | DG-capacity target-50 site     |
| 2           | $(250,1950)$          | Full-system PPO target-51 site |
| 3           | $(1550,1250)$         | Additional reachable cell      |
| 4           | $(650,1450)$          | Additional reachable cell      |
| 5           | $(850,250)$           | Additional reachable cell      |

The validated StudySpec is `hpc_runs/studies/cued_reward5_transfer_20260925.study.json` in the isolated source checkout. Its schema is `intrmotiv/study/v1`, workflow version `1.12.0`, and SHA-256 is `9922ce6fed52d11a6d7cff0d860a03860b219444417279976d6792602a721d59`. It declares both surviving source checkpoints, eight arms, seeds 42/1234/9999, and 75M frames per run: 48 runs total. W&B project: `SF_IntrMotiv_CuedReward5Transfer`.

Source branch: `codex/cued-reward5-transfer-20260925` at NEMO2 commit `52a6b6d6`, checkout `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_fixed_reward_transfer_20260924`. All outputs and caches are in `/work/classic/fr_xl1014-fixed-reward-transfer`. The production submission directory is `/work/classic/fr_xl1014-fixed-reward-transfer/IntrMotiv/SF_hipposlam/train_dir/_slurm/cued_reward5_transfer_20260925/production_submitted/`; `jobs.tsv` is the live accepted-job manifest, `submit.log` records quota retries, and `scancel.sh` cancels all accepted jobs.

The first jobs started just before that commit was created from the same deployed files, so their automatically captured W&B Git hash may be the parent `e389a7ab`, while the committed implementation is `52a6b6d6`. Use the StudySpec fingerprint and submission manifest alongside the source commit for provenance.

## Qualification and launch

The combined print-only audit found all 48 commands matched the StudySpec and all output paths stayed in the workspace. Four 262,144-frame CPU/GPU qualifications covering waypoint full refinement and flat full transfer completed with exit code 0. The cue-aware CUDA input issue found in the first GPU attempt was fixed before the clean GPU qualification. At shared TensorBoard steps, the PPO reward mean equaled environment reward mean; positive reward batches occurred in both waypoint qualifications and the CPU flat qualification. Source graph and policy tensors loaded in the relevant arms.

The native engine preflight confirmed cue changes, map resets, and physical reward contact for instruction 3. It was stopped once production training and the four qualifications showed reward pickups, to free the Slurm slot; it did not certify physical contact for every instruction individually. The structural map test checked all five reward markers and spawn exclusion.

The prior fixed-location production jobs were cancelled before this launch. On September 25, Slurm accepted the new production jobs beginning at ID `8196560`; CPU and GPU jobs entered `RUNNING`, and W&B run files appeared in the new campaign. The per-user `QOSMaxSubmitJobPerUserLimit` was reached after the first 21 accepted jobs. The quota-aware submitter retried as slots opened and ultimately recorded all 48 job IDs in `jobs.tsv`; check that manifest for exact identities and live Slurm for current states.

## Matched frozen-DG controls

The follow-up StudySpec `hpc_runs/studies/cued_reward5_frozen_dg_controls_20260925.study.json` has schema `intrmotiv/study/v1`, workflow `1.12.0`, SHA-256 `e3d4f5a1d29240dcb415b9ee8a163f236b1b41c145f6c766f4f3056698a1a32a`, and 12 runs: two source architectures, seeds 42/1234/9999, and two waypoint arms. Both arms use a fresh worker, fresh reward manager, empty graph, identical five-cue task, and 75M frames. `W_RAND_DG` freezes freshly initialized random DG projection weights, calibrates BatchNorm moments on one unlabeled learner minibatch before the PPO update, then freezes the moments. `W_SOURCE_DG` loads only source DG projection and BatchNorm tensors, then freezes both. Their paired difference tests whether the pretrained DG initialization helps relative to a calibrated random representation; the original `W_FIXED` arm also transfers a worker and therefore is not its direct control.

The add-on source is isolated from running production at NEMO2 checkout `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_cued_reward5_frozen_dg_20260925`, commit `20fbcbe0`. The first 262,144-frame qualification showed a frozen random DG was completely silent with untouched default BatchNorm moments; those four jobs and their original StudySpec remain as failure evidence. The one-minibatch calibration revision passed 19 focused tests on NEMO2 and fresh 4-run and 12-run print-only audits. The revised qualification StudySpec SHA-256 is `bbb4ad077375835a3c810a5de8c1ff2df0a70482e87deb9f64b924e9cdffe7b0`; jobs `8205826`–`8205829` all completed successfully. Random-DG density was 0.0127 on CPU and 0.0111 on GPU, with active-transition fractions 0.528 and 0.489. Both random DGs had exactly one normalization update and unchanged weights; both source DGs kept weights and moments fixed. All four runs had positive external reward batches and exact PPO/external reward equality at shared steps.

The qualification gate report is in the production submission directory. It released all 12 audited production controls, Slurm IDs `8208173`–`8208186` with gaps in the cluster's global ID sequence. The submitted audit reports `submitted_complete=true`, 12 unique IDs, exact StudySpec commands, and valid workspace paths. The September 26 matched 50M control analysis and policy-input probes are integrated below. The later exact 75M frozen-DG endpoints, common-history representation panels, and heldout reward trials are in the [poster batch analysis](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings). That analysis covers the frozen-DG controls; the wider eight-arm campaign needs its own completed matched analysis.

## Frozen-DG control results

The September 26 collection compared `W_SOURCE_DG` with calibrated
`W_RAND_DG` across all three paired downstream seeds within each source
architecture. DG50 uses goal-written worker memory; DG51 uses the canonical
CA3 trace, so their results are separate. DG50 loads the 130,662,400-frame
DG-capacity seed-123 source; DG51 loads the 179,142,656-frame full-system PPO
seed-123 source. Each source checkpoint is reused across its three downstream
seeds, so those are downstream replications rather than independent source
pretraining replicates. The shared 40–50M reward window and
50,003,968-frame online spatial snapshot are developmental. At that collection
cutoff, DG51 had 75M snapshots and DG50 had not yet completed. The later
[exact 75M comparison](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings)
includes both families and heldout matched-reset trials. Source DG's stronger
50M fields and prospective option hits did not turn into a 75M reward advantage
in those frozen-DG controls.

The original production StudySpec omitted `telemetry.online_spatial_target_frames`
although its training commands requested 5M, 20M, 50M, and 75M snapshots.
The collector used an analysis-only StudySpec copy with those declared targets:
SHA-256 `774dd8275e0b5da821611078bf2802d834462f40dc55a7fea8ef351196932858`.
The submitted StudySpec and every training command remained unchanged.

### Place fields and graph at matched 50M frames

All values below are means of three matched downstream seeds. Spatial information is divided by each unit's mean activity before averaging across active units; this avoids treating the amplitude-weighted standard score as normalized bits per activity. The field cosine and peak counts are computed on the policy-driven 100k-observation windows.

| Source architecture | Frozen DG | Spatial information, bits/activity | Active-map cosine | Distinct peak bins / 64 | Reliable directed edges | Reachable ordered node pairs | Prospective goal-hit fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DG50, goal-write | Random | 1.79 | 0.212 | 50.0 | 100.0 | 0.083 | 0.143 |
| DG50, goal-write | Source | 2.73 | 0.097 | 49.0 | 69.7 | 0.057 | 0.271 |
| DG51, standard trace | Random | 1.85 | 0.201 | 50.0 | 108.0 | 0.136 | 0.137 |
| DG51, standard trace | Source | 4.48 | 0.101 | 36.3 | 47.3 | 0.031 | 0.390 |

Source DG has greater normalized spatial information, lower average inter-unit map cosine, and higher prospective goal-hit fraction in all three paired seeds of both architectures. Random DG creates more reliable graph edges and more reachable node pairs in every paired seed. Thus source DG produces more effective tested commands, but a sparser learned graph. DG51 source fields also crowd into fewer peak bins at 50M; its lower map cosine should not be read as complete spatial coverage.

At the completed DG51 75M endpoint, the same pattern remains: normalized information is 4.48 versus 1.88 bits/activity (source versus random); prospective goal hits are 0.400 versus 0.138; reliable edges are 34 versus 99; and reachable ordered pairs are 0.013 versus 0.107. Source DG's active-unit fraction falls to 0.880 at this endpoint, versus 1.000 for calibrated random DG. The strictly grounded graph score is near zero in all arms because it requires *both* endpoints of a reliable edge to pass the strict mono-field criterion. It does not negate the measured prospective option hits.

Prospective hits are actual target activations after a previously known edge was commanded, accumulated during training. They are conditioned on the manager's selected targets and visited states; a matched-command versus shuffled-command intervention is still needed to measure causal target reachability from identical starts. These online place fields are policy-driven windows, so checkpoint differences are also affected by trajectory and viewpoint coverage.

Representative 50M visual audits (seed 42): [DG50 random fields](assets/cued_reward5_frozen_dg_interim_20260926/selected_figures/figures/CR5C_D50_W_RAND_DG_S42/target_000050000000_policy_00_place_fields_page01.png), [DG50 source fields](assets/cued_reward5_frozen_dg_interim_20260926/selected_figures/figures/CR5C_D50_W_SOURCE_DG_S42/target_000050000000_policy_00_place_fields_page01.png), [DG51 random fields](assets/cued_reward5_frozen_dg_interim_20260926/selected_figures/figures/CR5C_D51_W_RAND_DG_S42/target_000050000000_policy_00_place_fields_page01.png), [DG51 source fields](assets/cued_reward5_frozen_dg_interim_20260926/selected_figures/figures/CR5C_D51_W_SOURCE_DG_S42/target_000050000000_policy_00_place_fields_page01.png). The [DG51 random](assets/cued_reward5_frozen_dg_interim_20260926/selected_figures/CR5C_D51_W_RAND_DG_S42_50m_graph.png) and [DG51 source](assets/cued_reward5_frozen_dg_interim_20260926/selected_figures/CR5C_D51_W_SOURCE_DG_S42_50m_graph.png) graph outcome maps show the denser but weaker random-DG command evidence.

### Worker input probes: CA3 and depth

Both architectures receive a 10-channel depth bypass and a CA3 trace derived from DG. DG51's worker reads the canonical 4,544-value CA3 trace. DG50's goal-write architecture appends a separate 4,544-value goal-conditioned worker trace, which the worker reads; the canonical trace remains available to the manager. The task cue enters the manager and the goal ID conditions the worker through FiLM.

I loaded the frozen checkpoints, ran 512 deterministic decisions for each of the 12 runs, and replayed their actual core outputs through the frozen action head. The replayed action probabilities matched the rollout to a maximum absolute error below `6e-7`. Five within-run shuffles independently replaced the action-visible CA3 trace or the depth bypass while preserving each observation's task cue and selected goal. The table reports mean total variation of the eight-action distribution, averaged across the three seeds. The DG50 probes use 50M checkpoints; DG51 probes use completed 75M checkpoints.

| Source architecture | Frozen DG | CA3 shuffle | Depth shuffle | Greedy-action flips: CA3 | Greedy-action flips: depth |
| --- | --- | ---: | ---: | ---: | ---: |
| DG50 | Random | 0.366 | 0.0062 | 61.8% | 1.34% |
| DG50 | Source | 0.433 | 0.0005 | 44.5% | 0.01% |
| DG51 | Random | 0.280 | 0.0097 | 55.3% | 2.07% |
| DG51 | Source | 0.355 | 0.0015 | 41.9% | 0.08% |

The worker clearly uses its CA3 trace in both architectures and both DG groups. On these sampled states, direct depth-bypass changes have little effect on the action distribution. DG50's canonical trace alone has zero direct effect on its worker head, as expected from the goal-write routing; its appended goal-written trace has a large effect. This is a one-step input-dependence test. It does not measure the change in closed-loop reward success when a signal is removed during a whole episode, and it does not measure how much CA3 itself encodes depth-like visual cues.

### Reward learning and goal selection at 50M

The stronger source-DG fields and option hits have not yielded a clear external-reward advantage in the common 40–50M window. External-reward nonzero fractions are 0.000349 (source) versus 0.000529 (random) for DG50, with source lower in all three seeds. DG51 is nearly tied: 0.000597 (source) versus 0.000585 (random), with mixed paired-seed differences. These are online training fractions, not held-out episode success.

The 50M checkpoints show a direct bottleneck in converting reward to goal choice. For each instruction and current DG source, the manager stores an external-return estimate for each of the 64 final goals. It samples a final goal with score $v+2/\sqrt{n+1}$ passed through a softmax at temperature 2, excluding the current goal. Mean learned values are 0.012–0.017 across groups; the optimism term is about 0.55 at 12 goal tests. Despite nonzero reward-table updates, the **visit-weighted maximum goal probability is only 0.0167–0.0177** across the four groups, versus 0.0159 for uniform choice among 63 goals. Entropy on cue/source rows with updates is 99.84–99.96% of uniform. The mean pairwise distance between different instructions' goal distributions, weighted by current-source visits, is only 0.009–0.018 in total variation. The source DG arms are not more cue selective than the random DG arms by this measure.

This explains the apparent mismatch: the source representation improves the chance of reaching a DG target *when an established option is commanded*, but the manager still samples nearly all goals with nearly equal probability for each cue. The source graph is also less connected, so fewer sampled distant goals can use a learned route. The reward task requires choosing goals that lead to one of five physical cells, not just reaching arbitrary DG events. Sparse reward-to-go estimates, the optimism term, and softmax smoothing keep the current selector broad; the checkpoint probabilities establish the broad selection, while the separate contribution of each mechanism remains an inference. A more selective cue-conditioned manager is the next mechanism to test before expecting the source DG's option advantage to improve task reward.

### Data and reusable method

- [Spatial snapshots and per-unit summaries](data/cued_reward5_frozen_dg_interim_20260926/spatial/)
- [Cue-conditioned reward-manager probabilities from all 12 checkpoints](data/cued_reward5_frozen_dg_interim_20260926/spatial/reward_manager_diagnostics.csv)
- [Matched online reward summaries](data/cued_reward5_frozen_dg_interim_20260926/online/)
- [Twelve frozen-policy input probes](data/cued_reward5_frozen_dg_interim_20260926/input_sensitivity/)
- [Input probe implementation](probe_cued_reward5_policy_inputs.py) and [Slurm wrapper](probe_cued_reward5_policy_inputs.sbatch)

The full raw NPZ snapshots, edge tables, TensorBoard histories, and Slurm logs remain in `/work/classic/fr_xl1014-fixed-reward-transfer/IntrMotiv/SF_hipposlam/train_dir/analysis/cued_reward5_frozen_dg_controls_20260925/`. The canonical spatial collector and the saved action-head replay checks were authoritative. Two workflow issues slowed collection: the missing online target declaration required an analysis-only StudySpec copy, and the split CPU/GPU run roots required workspace symlinks directly to each nested `00_RUN` summary directory. Future studies should declare the online targets in the StudySpec and retain that declaration in print-only review.

## Reusable lesson

For a cue-conditioned transfer campaign, preserve source DG/worker tensor shapes by carrying the task cue separately into the planner and flat mixture. Verify the actual PPO reward path with positive-reward batches. The print-only audit and quota-aware live manifest are authoritative for whether a large study is fully submitted; Slurm acceptance can be partial even when the study is valid.
