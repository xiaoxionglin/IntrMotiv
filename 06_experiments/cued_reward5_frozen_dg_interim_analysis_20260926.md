# Frozen source DG versus calibrated random DG: interim map and control analysis

## Scope and status

This compares the matched `W_SOURCE_DG` and `W_RAND_DG` waypoint controls in the five-cue reward task. Each arm has a fresh worker, fresh reward manager, and initially empty graph; DG initialization is the controlled difference. The source DG projection and BatchNorm are frozen. The random DG projection is frozen after one unlabeled BatchNorm calibration minibatch. The source visual trunk, task, starts, actions, and 75M-frame budget are matched within each source architecture and downstream seed (42, 1234, 9999).

DG50 transfers the DG-capacity seed-123 checkpoint at 130,662,400 frames; DG51 transfers the full-system PPO seed-123 checkpoint at 179,142,656 frames. The two source architectures are analyzed separately because DG50 uses goal-written worker memory and DG51 uses the canonical CA3 trace.

The production StudySpec is `intrmotiv/study/v1`, workflow `1.12.0`, SHA-256 `e3d4f5a1d29240dcb415b9ee8a163f236b1b41c145f6c766f4f3056698a1a32a`. On September 26, all six DG51 GPU runs had completed 75M frames, while the six DG50 CPU runs were still running. All 12 runs have spatial snapshots through 50M frames: 42 of the planned 48 snapshots exist. The 50M snapshot actual frame count is 50,003,968; DG51 also has 75,005,952-frame snapshots. Each snapshot retains the latest 100,000 behavior observations. The online reward comparison uses the common 40–50M-frame window.

The submitted StudySpec omitted the `telemetry.online_spatial_target_frames` declaration even though its training arguments explicitly requested 5M, 20M, 50M, and 75M snapshots. An analysis-only copy under the workspace adds those four target declarations so the canonical `collect-spatial --include-details` command can validate and collect the already written snapshots. It changes no training command or run identity. Its separate analysis SHA-256 is `774dd8275e0b5da821611078bf2802d834462f40dc55a7fea8ef351196932858`.

## Place fields and graph at matched 50M frames

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

## Does the worker use CA3 or depth?

Both architectures receive a 10-channel depth bypass and a CA3 trace derived from DG. DG51's worker reads the canonical 4,544-value CA3 trace. DG50's goal-write architecture appends a separate 4,544-value goal-conditioned worker trace, which the worker reads; the canonical trace remains available to the manager. The task cue enters the manager and the goal ID conditions the worker through FiLM.

I loaded the frozen checkpoints, ran 512 deterministic decisions for each of the 12 runs, and replayed their actual core outputs through the frozen action head. The replayed action probabilities matched the rollout to a maximum absolute error below `6e-7`. Five within-run shuffles independently replaced the action-visible CA3 trace or the depth bypass while preserving each observation's task cue and selected goal. The table reports mean total variation of the eight-action distribution, averaged across the three seeds. The DG50 probes use 50M checkpoints; DG51 probes use completed 75M checkpoints.

| Source architecture | Frozen DG | CA3 shuffle | Depth shuffle | Greedy-action flips: CA3 | Greedy-action flips: depth |
| --- | --- | ---: | ---: | ---: | ---: |
| DG50 | Random | 0.366 | 0.0062 | 61.8% | 1.34% |
| DG50 | Source | 0.433 | 0.0005 | 44.5% | 0.01% |
| DG51 | Random | 0.280 | 0.0097 | 55.3% | 2.07% |
| DG51 | Source | 0.355 | 0.0015 | 41.9% | 0.08% |

The worker clearly uses its CA3 trace in both architectures and both DG groups. On these sampled states, direct depth-bypass changes have little effect on the action distribution. DG50's canonical trace alone has zero direct effect on its worker head, as expected from the goal-write routing; its appended goal-written trace has a large effect. This is a one-step input-dependence test. It does not measure the change in closed-loop reward success when a signal is removed during a whole episode, and it does not measure how much CA3 itself encodes depth-like visual cues.

## Reward task implication

The stronger source-DG fields and option hits have not yielded a clear external-reward advantage in the common 40–50M window. External-reward nonzero fractions are 0.000349 (source) versus 0.000529 (random) for DG50, with source lower in all three seeds. DG51 is nearly tied: 0.000597 (source) versus 0.000585 (random), with mixed paired-seed differences. These are online training fractions, not held-out episode success.

The 50M checkpoints show a direct bottleneck in converting reward to goal choice. For each instruction and current DG source, the manager stores an external-return estimate for each of the 64 final goals. It samples a final goal with score $v+2/\sqrt{n+1}$ passed through a softmax at temperature 2, excluding the current goal. Mean learned values are 0.012–0.017 across groups; the optimism term is about 0.55 at 12 goal tests. Despite nonzero reward-table updates, the **visit-weighted maximum goal probability is only 0.0167–0.0177** across the four groups, versus 0.0159 for uniform choice among 63 goals. Entropy on cue/source rows with updates is 99.84–99.96% of uniform. The mean pairwise distance between different instructions' goal distributions, weighted by current-source visits, is only 0.009–0.018 in total variation. The source DG arms are not more cue selective than the random DG arms by this measure.

This explains the apparent mismatch: the source representation improves the chance of reaching a DG target *when an established option is commanded*, but the manager still samples nearly all goals with nearly equal probability for each cue. The source graph is also less connected, so fewer sampled distant goals can use a learned route. The reward task requires choosing goals that lead to one of five physical cells, not just reaching arbitrary DG events. Sparse reward-to-go estimates, the optimism term, and softmax smoothing keep the current selector broad; the checkpoint probabilities establish the broad selection, while the separate contribution of each mechanism remains an inference. A more selective cue-conditioned manager is the next mechanism to test before expecting the source DG's option advantage to improve task reward.

## Data and reusable method

- [Spatial snapshots and per-unit summaries](data/cued_reward5_frozen_dg_interim_20260926/spatial/)
- [Cue-conditioned reward-manager probabilities from all 12 checkpoints](data/cued_reward5_frozen_dg_interim_20260926/spatial/reward_manager_diagnostics.csv)
- [Matched online reward summaries](data/cued_reward5_frozen_dg_interim_20260926/online/)
- [Twelve frozen-policy input probes](data/cued_reward5_frozen_dg_interim_20260926/input_sensitivity/)
- [Input probe implementation](probe_cued_reward5_policy_inputs.py) and [Slurm wrapper](probe_cued_reward5_policy_inputs.sbatch)

The full raw NPZ snapshots, edge tables, TensorBoard histories, and Slurm logs remain in `/work/classic/fr_xl1014-fixed-reward-transfer/IntrMotiv/SF_hipposlam/train_dir/analysis/cued_reward5_frozen_dg_controls_20260925/`. The canonical spatial collector and the saved action-head replay checks were authoritative. Two workflow issues slowed collection: the missing online target declaration required an analysis-only StudySpec copy, and the split CPU/GPU run roots required workspace symlinks directly to each nested `00_RUN` summary directory. Future studies should declare the online targets in the StudySpec and retain that declaration in print-only review.
