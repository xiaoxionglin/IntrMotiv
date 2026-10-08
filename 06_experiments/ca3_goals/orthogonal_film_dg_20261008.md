# Orthogonal FiLM goal initialization: replacement DG studies

**Status, 8 October 2026:** three corrected compute-node qualifications and exact checkpoint reloads passed. All six finite replacements completed 75M, and all twelve episode-long replacements have complete 75M online snapshots and continue toward 150M. The comparisons below are online training evidence. Frozen matched-command arrivals have not yet been measured. No recurring monitor is active.

## Interim 75M result

The canonical collector used the same 70–75M-frame window for all twelve episode-long runs. Each row below averages seeds 8, 99 and 123. Coverage AUC is the external exploration measure. Action TV compares policy action distributions under different goals at the same observed state. The last column is the commanded minus retrospectively shuffled target-hit rate per 1,000 logged events; it is a diagnostic, not a matched-command causal effect.

| Episode-long arm | Coverage AUC | Goal-action TV | Commanded minus shuffled hits / 1k |
| --- | ---: | ---: | ---: |
| Prescribed FiLM | 64.0 | 0.00317 | approximately 0.00 |
| Learned-4 FiLM | 38.1 | 0.00064 | +0.28 |
| C15 all-16 FiLM | 45.0 | 0.00097 | −0.01 |
| C05 all-16 FiLM | 58.3 | 0.00802 | −3.14 |

Prescribed versus learned-4 coverage increased in every paired seed by 22.8, 30.8 and 24.2 AUC points. C05 exceeded C15 by 9.8, 15.4 and 14.7 points, but this package changes manager selection and two DG losses together. Neither advantage establishes command-driven arrival. The prescribed arm's target-hit rate was only 0.42 per 1,000 events and indistinguishable from its shuffled rate. C05's larger mean action TV is driven substantially by seed 123 (0.0194; the other two seeds were 0.00430 and 0.00035), and its commanded hit rate was below its shuffled rate in all three seeds. These retrospective rates cannot substitute for executing alternative commands from matched starts.

The complete finite study and its zero-$M$ parent use exactly the same 70–75M online window, seeds, goal vocabulary and horizon. The only scientific change is initializing $M$ with orthonormal rows. The three-seed means are:

| Finite arm | Coverage AUC: zero $M$ → orthogonal $M$ | Option success: zero → orthogonal | Goal-action TV: zero → orthogonal |
| --- | ---: | ---: | ---: |
| Prescribed | 69.9 → 64.7 | 3.05% → 1.48% | 0.00460 → 0.00257 |
| Learned-4 | 36.0 → 37.3 | 41.9% → 43.5% | 0.00189 → 0.00268 |

For prescribed, coverage changed by +7.4, −14.1 and −9.2 AUC points in seeds 8, 99 and 123. Orthogonal initialization therefore did not consistently improve exploration or online command sensitivity and halved the already-low oracle option-hit fraction. That hit fraction is strongly confounded by the narrow fields' rarity. In the orthogonal cohort, changing from the finite horizon to episode-long commands changed prescribed coverage by −15.0, +3.1 and +9.9 points across the same seeds; learned-4 changed by −2.1, −2.8 and +7.4. There is no consistent 75M coverage benefit from removing the deadline. These are separately trained policies, so only a frozen matched-command arrival test can isolate goal-following ability.

The [twelve terminal checkpoint matrices](results/orthogonal_film_dg_20261008/finite_75m_film_goal_rows.csv) show that orthogonal initialization did preserve more **parameter** separation. Across the four eligible oracle goal rows at 75M, mean row norm was 1.77 and mean absolute pairwise row cosine was 0.639, compared with norm 1.59 and cosine 0.976 for the zero-initialized oracle controls. Initial orthogonal rows had norm one and cosine zero; neither property is constrained during PPO. Thus the new rows neither stayed orthogonal nor collapsed to zero. Their greater separation did **not** translate into stronger measured action modulation or goal hits. Possible causes include downstream cancellation and sparse or poorly assigned goal credit; the online data do not distinguish them.

All four fixed DG fields were encountered in every 75M episode-long snapshot. Their positive observations per 100,000 behavior observations ranged from 167–233, 52–76, 116–156 and 162–220 for IDs 0–3, respectively; all twelve learned context units remained active. The narrow fields are therefore real but rare, especially goal 1. The 19-by-19 mono-field score still cannot resolve their 40-unit support. In the finite replacement's 75M snapshots, goal 1 appeared only 32–59 times per 100,000. The earlier finite 45–50M window had a prescribed seed-123 coverage of only 26.9 versus 81.9 and 65.5 in the other seeds; by 75M, seed 123 had recovered to 62.3. This transient illustrates why the planned age-matched 75M contrast is preferable to a single early window.

Among learned all-16 representations at 75M, the online mono-field fraction was highly seed-dependent: C15 had 0.500, 0.0625 and 0; C05 had 0, 0.625 and 0 in seeds 8, 99 and 123 respectively. All units were active, but the three-seed mean active-only map cosine was 0.135 in C15 and 0.205 in C05. C05's coverage gain therefore does not coincide with a consistent mono-field or map-separation gain in these online samples. Frozen 10k maps are needed for a stronger field-quality comparison.

### The episode-long graph counter is censored

At 75M the episode-long prescribed and learned-4 arms each report all 12 possible directed edges among their four eligible goals as reliable in all three seeds. Their prospective success fractions are exactly 1.0, as are C15's and C05's, despite the negligible online command advantage above. The source outcome update counts prospective attempts on a hit or option expiration. Episode-long normal goals do not expire, and a physical episode reset does not enter that attempted-outcome branch. Therefore the prospective denominator contains recorded hits but omits unresolved goals censored at episode end. The resulting 100% fraction and reliability-based edge count cannot establish navigation success. The derived `graph_grounded_controllability` also multiplies this biased fraction and must not be used as a control conclusion here. A minimal general repair is to record issued commands and episode-end censoring explicitly, then report physical arrival by fixed observation windows with its denominator and censoring; the planned frozen matched-start intervention already measures the relevant causal contrast.

The [finite 75M per-run table](results/orthogonal_film_dg_20261008/finite_online_70_75m_per_run.csv), [episode-long C15 table](results/orthogonal_film_dg_20261008/episode_online_70_75m_per_run.csv), [C05 table](results/orthogonal_film_dg_20261008/c05_online_70_75m_per_run.csv), adjacent manifests, and their [finite](results/orthogonal_film_dg_20261008/finite_spatial_all_per_snapshot.csv), [C15](results/orthogonal_film_dg_20261008/episode_spatial_all_per_snapshot.csv) and [C05](results/orthogonal_film_dg_20261008/c05_spatial_all_per_snapshot.csv) spatial snapshots contain seed-level evidence and all available ages. Full graph arrays and original NPZs remain in the allocated NEMO2 workspace. The original zero-$M$ finite cohort's [75M table](results/four_prescribed_dg_20261007/online_75m_per_run.csv) is the matched initialization reference.

The finite frozen-checkpoint and intervention manifests contain 12 field and six matched-command rows. Representative 500-decision field preflight `8314889` completed with exit 0 and a valid NPZ containing thresholded and pre-threshold arrays. After print-only review, the 12 independent 10k-decision field jobs `8314905`–`8314916` were submitted. Exact-start prescribed-goal intervention preflight `8314918` was submitted separately with a 20k-decision cap; its result gates the six full finite interventions. The canonical telemetry renderer cannot yet produce the episode-long frozen manifest: future 100M and 150M targets currently select the same latest checkpoint as 75M, causing duplicate labels. Keep the immutable StudySpecs unchanged and render their manifests when the declared checkpoints exist. The 75M online result is therefore provisional with respect to controllability; frozen place-field and matched-command tests and the 150M training result are pending.

## Question and intervention

The target-ID FiLM decoder computes

$$
h=\operatorname{ReLU}(W_s x+b_s),\quad
(\Delta\gamma_g,\beta_g)=e_g^\top M,\quad
z=h\odot(1+\Delta\gamma_g)+\beta_g.
$$

The earlier graph-HRL runs initialized the $16\times256$ goal-to-modulation matrix $M$ to zero, so every goal had the same effect at the start. The replacement uses `--hrl_film_goal_init=orthogonal`: rows of $M$ are orthonormal with gain one. The all-zero no-goal vector still yields identity modulation. A seed-specific local random generator initializes only $M$, preserving the random stream and initial values of the state, output and action layers for paired seeds. The historical zero setting remains the default in code. Orthogonality is **initialization only**; PPO can change the rows afterward.

## Declared matrix

| Study | Arms | Seeds | Length | Main comparison |
| --- | --- | --- | ---: | --- |
| [Finite-horizon FiLM](../../hpc_runs/studies/orthogonal_film_finite_dg_20261008.study.json) | ORACLE, LEARNED4 | 8, 99, 123 | 75M frames | Within-new-study field replacement; at 75M, same-seed initialization contrast against the completed zero-$M$ finite study |
| [Episode-long C15 FiLM](../../hpc_runs/studies/orthogonal_film_episode_dg_20261008.study.json) | ORACLE, LEARNED4, C15 all-goal | 8, 99, 123 | 150M frames | At 75M, finite versus episode-long horizon under the same orthogonal initialization |
| [Episode-long C05 FiLM](../../hpc_runs/studies/orthogonal_film_episode_c05_20261008.study.json) | C05 all-goal | 8, 99, 123 | 150M frames | C05 versus C15 controller-family package at paired ages |

The finite and episode-long source arguments were compared run by run with their corresponding parent studies. Apart from run/tracking identity and orthogonal $M$ initialization, scientific arguments match. C05 versus C15 retains three jointly changing settings: `visit_direct` versus `frontier_direct`, DG global punishment $0.01$ versus $0$, and DG row repulsion $1$ versus $0$. Thus that comparison cannot isolate any one setting. The cancelled legacy-9 runs reached their 5M milestone only; carry them forward as early context, **not** as a 75M or 150M matched control.

The six-run finite StudySpec uses a 64-decision fallback deadline. The twelve episode-long FiLM runs keep a command until a hit or physical episode end. All arms retain CA3 $L=64$, the reward rule, field centers and width, eligible goal sets, and the previous controller settings. The finite and C15/C05 studies keep their previously declared online, 10k frozen field, trajectory, graph, and matched-command protocols; the finite replacement now explicitly selects the exact-start executed-alternative evaluator. Primary control evidence must be commanded versus alternative-command physical arrival from matched starts, with failures, censoring, field exposure, and action change. Online hit-only success is secondary.

## Release gates and provenance

The [finite qualification](../../hpc_runs/studies/orthogonal_film_finite_dg_qualification_20261008.study.json), [episode-long C15 qualification](../../hpc_runs/studies/orthogonal_film_episode_dg_qualification_20261008.study.json), and [C05 qualification](../../hpc_runs/studies/orthogonal_film_episode_c05_qualification_20261008.study.json) each ran one seed-99 arm for 1,048,576 frames. The earlier 524,288-frame qualification window was too short to fill a 100,000-observation online spatial snapshot at frameskip eight; all three new qualifications wrote complete snapshots. Training jobs `8308758` (finite oracle), `8308756` (episode-long oracle) and `8308757` (episode-long C05) completed with exit 0. Exact reload jobs `8309967`, `8309969` and `8309968` respectively certified model, optimizer and counters against their saved milestone checkpoints.

| Qualification | Latest train loss | Active-target fraction | Goal-action TV | Prescribed-field positive observations per 100k | Context units active |
| --- | ---: | ---: | ---: | --- | ---: |
| Finite ORACLE | 0.099 | 0.273 | 0.000191 | 54, 25, 48, 196 | 12/12 |
| Episode-long ORACLE | 0.137 | 0.326 | 0.000287 | 59, 27, 41, 208 | 12/12 |
| Episode-long C05 | 0.650 | 0.998 | 0.000530 | Not prescribed; 16 learned DG units | 12/12 channels 4–15 |

The losses are finite, all four oracle fields were observed, and $M$ row norms remained near one. [Finite](results/orthogonal_film_dg_20261008/finite_online_per_run.csv), [episode-long C15](results/orthogonal_film_dg_20261008/episode_c15_online_per_run.csv), and [C05](results/orthogonal_film_dg_20261008/episode_c05_online_per_run.csv) online tables, spatial unit/snapshot summaries, adjacent manifests, and exact reload certificates preserve the evidence. These one-seed short windows show runtime health. Goal-action TV is nonzero but small, so orthogonal $M$ has not yet demonstrated effective command-driven behavior; the two oracle qualifications had no target hits in their final learner window.

Source branch `codex/orthogonal-film-dg-20261008`, commit `f93c112d`, is mirrored at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_orthogonal_film_dg_20261008`. Focused tests passed locally and remotely, including row orthogonality, initial command distinction, historical zero initialization, and preservation of other weights and the global RNG stream. All nine unique arm/qualification configurations passed the real training argument parser. The workflow schema is `intrmotiv/study/v1`, version `1.14.1`. Production StudySpec SHA-256 values are `21c3e50c20876b575dae711845512fb1d927e5c0bbed8c629c1e3c2431e859f5` (finite), `ffa7a53f4b8aef4e14b38bbf24f6f00fe18e4d5a25006ad705b93efcc6cb0af2` (episode-long C15), and `778b816aaaf0914d68bcb254cbfa82f9fa951534cb44470af08d1691848a604d` (episode-long C05).

The first three qualification submissions, `8308748`–`8308750`, used a global generator for $M$. They were cancelled before completion because consuming that stream would change later layer initialization. Fresh qualification namespaces and the completed jobs above use the corrected local-generator implementation. The production print-only and submitted audits matched all commands and checked workspace-only paths. The finite six-run jobs are `8309970`–`8309975` with a 48-hour limit; the episode-long C15 nine-run jobs are `8309976`–`8309984` and C05 three-run jobs are `8309985`–`8309987`, both with 72-hour limits under the CPU partition's four-day maximum. Their immutable submission manifests are under `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/_slurm/{orth_fdg_20261008,orth_eldg_20261008,orth_elc05_20261008}/20261008T0127Z_submitted/`. No recurring monitor was restarted; production follow-up is on demand.

## Reusable lesson

An initialization intervention must preserve the random stream used for every other parameter. Otherwise a nominal one-matrix change is also a whole-policy initialization change. Compare rendered training arguments and initial unaffected tensors, then use compute-node qualifications before an expensive training matrix.
