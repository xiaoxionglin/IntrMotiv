# Four prescribed DG goals: matched controller screen

**Status, 8 October 2026:** all six matched runs completed at 75,005,952 frames with exit 0. Their four declared checkpoints and online spatial snapshots are present. The terminal online evidence still shows sparse oracle hits and weak command sensitivity. The 10k frozen sweep was cancelled after the user requested a replacement with orthogonally initialized FiLM goal weights; exact-start causal arrival remains unmeasured in this zero-initialized cohort. See the [replacement study](orthogonal_film_dg_20261008.md).

## Question and comparison

Can the existing C15 direct FiLM worker acquire command-specific control when four goal identities are fixed, compact and unambiguous? The intervention is privileged: engine position defines the four DG activities, but neither position nor target coordinates enter the worker, manager or reward code. A positive result shows that this supplied representation package helps the current controller; it does not show that visual DG can learn it.

The [workflow-1.14.1 StudySpec](../../hpc_runs/studies/four_prescribed_dg_controller_20261007.study.json) declares one matched contrast, ORACLE minus LEARNED, with seeds 8, 99 and 123. Both arms have 16 DG channels, the same fixed ImageNet visual trunk, CA3 size, FiLM target interface, C15 frontier manager, reward rule, action set, 64-decision fallback horizon, and four eligible manager identities (IDs 0–3). Both expose a four-value `dg_prescribed` observation slot, so their observation structure matches. In LEARNED, all 16 DG channels are learned and this slot is zero. In ORACLE, channels 0–3 are replaced by fixed fields; channels 4–15 remain learned CA3 context. Those context channels cannot become graph nodes or manager goals. The existing encoder batch-use loss excludes only the fixed four rows; the other configured DG losses retain their source semantics. Recruitment is monitor-only with zero permitted replacements in both arms.

This is a six-run screen, not an architecture sweep. The historical odor/C15 result is context, not a capacity- and goal-vocabulary-matched control. The source design discussed 4 fixed plus 28 learned units; the user explicitly selected **4 fixed plus 12 learned** for this batch.

## Prescribed events and feasibility gate

For each center $c_i$, the environment computes $g_i(p)=\exp(-\|p-c_i\|^2/(2\sigma^2))$ and sends

$$
a_i(p)=\frac{\max(0,g_i(p)-e^{-2})}{1-e^{-2}},\qquad \sigma=20.
$$

The field peaks at one and is exactly zero at and beyond radius 40 DMLab units. Centers are $(550,550)$, $(1450,550)$, $(550,1450)$ and $(1650,1650)$, selected from non-wall cells of the fixed third map. Native DMLab job `8291772` completed on a compute node (exit 0) using the training level and eight engine steps per action. Across 200 reset seeds followed by a random-action rollout, 12,201 decision-time observations contained respectively 10, 7, 11 and 59 positive samples, with 4, 4, 5 and 14 entries. Minimum center distances were 9.1, 18.3, 0 and 0 DMLab units. Thus all four fields are accessible and detectable, but detections are sparse. These counts mix reset observations with the random trajectory, so they are an availability check, not a policy encounter-rate estimate. Per-field exposure under the learned training policy remains an explicit early-milestone check.

The prescribed values enter after the learned DG threshold and before CA3. The visual trunk and projection do not receive privileged coordinates. The first four projection outputs in ORACLE are unused, and gradients from the fixed fields are blocked; the remaining twelve learned outputs still update. Manager recognition masks context-only channels while CA3 retains them. The same goal mask applies to LEARNED, so selecting among only four goal IDs is held fixed. In the standard frozen evaluator, `raw_dg_*` and `pre_threshold_*` arrays reconstruct those unused learned projection rows; **for ORACLE IDs 0–3, only post-replacement `active_fraction`, `rate_maps`, and activity-aligned samples describe the prescribed fields**.

## Outcomes and release gates

The primary outcome is **command-caused arrival** by the declared option deadline, measured from matched starts under alternative target commands. Report target-macro arrival probability and lift, all failed/time-out trials, each goal and seed, first distinct event, time to arrival, and initial-action distribution change. A first event is diagnostic; arrival need not be first for a goal to be useful. The existing intervention manifest requests all three alternatives per source and 20 attempts per ordered pair, subject to a bounded preflight and an exact-start verification. Online option success and graph edges are secondary. External coverage AUC is the exploration outcome, paired by seed over early and terminal windows.

Qualification proceeds in this order:

1. Verify the four fixed fields, reset/observation alignment, exact zero outside support, no direct coordinate bypass, and matching manager goal masks in both arms.
2. Run one short full-learner job per arm on a compute node. Require finite losses/rewards, nonempty goal opportunities, and a checkpoint reload. Check per-field policy encounters, learned-context updates, and exact fixed-field alignment at the first production milestone; the short qualification telemetry does not establish all three.
3. Run the six declared jobs to 75M with milestones at 5M, 25M, 50M and 75M. Use the canonical StudySpec and print-only submission audit; keep all bulk outputs in `/work/classic/fr_xl1014-corridor-geometry`.
4. Use the standard manifest-driven 10k-decision field evaluator and matched-command intervention. Report fixed-oracle versus learned-context maps separately, including silence, active-only overlap, peak diversity and pre-threshold maps. The standard 100-unit grid is too coarse to certify an 80-unit support, so verify the prescribed field shape directly from position/activity samples as an additional diagnostic.

If the oracle arm has clear command lift while LEARNED does not, unstable or ambiguous learned goal identity becomes the leading bottleneck. If both arms show little action change and little lift despite adequate events and candidate exposure, worker goal-conditioning or credit is implicated. If the oracle fields are rarely encountered or goals are infeasible under the deadline, the controller comparison is inconclusive; report that gate rather than interpreting a null result.

## Interim evidence at 5M and 25M

The canonical StudySpec online collector read all six TensorBoard histories at the declared early ages, using windows 2.5–7.5M and 22.5–27.5M. The spatial collector read all 12 retained online snapshots, each containing 100,000 policy observations. The pinned [5M](results/four_prescribed_dg_20261007/online_5m_per_run.csv), [25M](results/four_prescribed_dg_20261007/online_25m_per_run.csv), [per-snapshot](results/four_prescribed_dg_20261007/spatial_5m_25m_per_snapshot.csv), and [per-unit](results/four_prescribed_dg_20261007/spatial_5m_25m_per_unit.csv) tables preserve the individual seeds. Their adjacent manifests preserve the study fingerprint and collector protocol; bulk NPZs remain in the allocated workspace.

| Age | Arm | Mean coverage AUC | Mean online option success | Mean target-action probability TV |
| --- | --- | ---: | ---: | ---: |
| 5M | LEARNED | 62.3 | 51.4% | 0.00028 |
| 5M | ORACLE | 81.9 | 2.23% | 0.00026 |
| 25M | LEARNED | 35.0 | 33.9% | 0.00047 |
| 25M | ORACLE | 71.1 | 2.57% | 0.00168 |

These are three-seed means of online window metrics, not matched-start success probabilities. At 25M the oracle coverage AUC is higher for each paired seed (differences 27.2, 40.5 and 40.3), while its option success is lower for each (differences −26.2, −27.1 and −40.8 percentage points). The target-action TV remains small in both arms. Within each arm, commanded and shuffled instantaneous target-hit rates are nearly identical; this is a diagnostic of weak command sensitivity, not the planned matched-start intervention result.

In the ORACLE snapshots, fixed-field positive observations per 100,000 policy observations were:

| Age | DG 0 | DG 1 | DG 2 | DG 3 |
| --- | ---: | ---: | ---: | ---: |
| 5M, range over seeds | 139–216 | 59–81 | 181–266 | 167–413 |
| 25M, range over seeds | 98–223 | 34–75 | 139–311 | 155–263 |

All four prescribed fields therefore occur in every seed and age, and all twelve learned context channels remain active. The stored post-replacement activities match the declared Gaussian function at their recorded poses with maximum absolute error below $3\times10^{-8}$ across the six oracle snapshots. The LEARNED arm's first four units instead had roughly 1,930–4,712 positive observations per 100,000, depending on unit, seed and age. This large event-frequency difference makes online option success an unequal-difficulty comparison. In particular, DG 1 is sampled only 34–81 times per 100,000 in ORACLE.

The oracle graph has **zero reliable directed edges at both ages in all three seeds**; LEARNED has 2–9. The oracle seed-99 25M graph recorded about 5,718 directed attempt mass across the four goal IDs, but its maximum off-diagonal estimated edge reliability was 0.07, well below the declared 0.5 threshold. Its frontier discovery counts were zero. The graph's separate prospective-attempt array was also zero, so these stored attempts should not be presented as completed prospective control trials. The standard 19-by-19 place-field grid and its mono-field eligibility test do not resolve the narrow oracle fields; interpret exact pose/activity checks above rather than the grid's zero mono-field score for those four units.

The [50M](results/four_prescribed_dg_20261007/online_50m_per_run.csv) and [75M](results/four_prescribed_dg_20261007/online_75m_per_run.csv) online per-run tables and [all-age spatial summaries](results/four_prescribed_dg_20261007/spatial_all_per_snapshot.csv) extend this pattern. At 75M, oracle coverage AUC averaged 69.9 versus 36.0 for LEARNED, higher in every paired seed. Oracle online option success averaged 3.05% versus 41.9%, but the fields differ greatly in encounter frequency. Mean target-action probability TV was still only 0.00460 in ORACLE and 0.00189 in LEARNED. The terminal oracle graph had zero reliable edges and zero recorded prospective attempts in all seeds; LEARNED averaged 1.67 reliable edges. Adjacent manifests preserve the source StudySpec fingerprint and collection windows. These are online training diagnostics, **not** matched-start arrival effects.

Fixed, clean fields therefore did not yield frequent online goal hits under the original controller by 75M. This does not identify a unique cause: the destinations are far apart, events are rare, the 64-decision option rule may be too short, and the goal-specific FiLM matrix starts at zero. Higher coverage is an exploration observation, not evidence that commands caused arrivals. The [orthogonal-FiLM replacement](orthogonal_film_dg_20261008.md) isolates the initial goal-matrix change at the same finite horizon and separately compares finite and episode-long control under that initialization.

At the earlier scheduled audit, all six jobs were running between 41.3M and 43.1M frames with finite latest train, policy and value losses. All later 50M and 75M checkpoints and online snapshots were produced before training completed. The 12 manifest-driven frozen 10k jobs, IDs `8308169`–`8308180`, were cancelled at the user's request before their outputs could be treated as a complete comparison. The original intervention manifest omitted the exact-start evaluator selection, so its legacy hit counter would not have answered the declared causal-arrival question; no such intervention was submitted.

## Provenance and reusable lesson

The current StudySpec validates as six unique cells under schema `intrmotiv/study/v1`, declared workflow `1.14.1`, SHA-256 `009cd13111d886839f04624be0913f95773ef4e672ce5c86b6853724afca6b1d`. It derives from the corrected odor C15 configuration without its odor and Hebbian-selector factors. Source commit `c3980b69` is pushed as `codex/four-oracle-dg-20261007` in `SF_hipposlam`; the isolated NEMO2 source is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_four_prescribed_dg_20261007`. A content-only comparison found no differences between the pushed source and the isolated runtime, excluding caches and transient inspection scripts. The exact NEMO2 focused tests passed (36 tests). Seed-99 qualification jobs `8291787` (LEARNED) and `8291788` (ORACLE) completed at 524,288 frames with exit 0 and both checkpoint milestones. The last learner window showed finite train losses (0.123 and 0.477), active manager targets (0.469 and 0.313), and some completed options (10 and 5); the ORACLE window had zero target hits, while LEARNED had five. These are small, unmatched online windows and must not be interpreted as an outcome contrast. Standard frozen-checkpoint smoke job `8291806` loaded the ORACLE 524,288-frame checkpoint, completed 500 decisions, and exited 0. Its post-replacement DG fractions for IDs 0–3 were 0, 0, 0.004 and 0, confirming the short rollout's limited exposure. The submitted six-row StudySpec audit passed with exact commands, unique numeric job IDs `8291816`–`8291821`, and workspace-only paths; all six later completed. Training root: `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_four_prescribed_dg_controller_20261007`. Submission manifest and audit: `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_four_prescribed_dg_controller_20261007/20261007T124123Z/`. These checks establish runtime health, not control improvement.

The efficient path for a future prescribed-goal study is to reuse the canonical StudySpec and evaluator, and preflight event availability before launching a large matrix. A perfectly shaped field is not a useful controller test if its event is absent from the actual decision stream.
