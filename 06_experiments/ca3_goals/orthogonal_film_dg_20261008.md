# Orthogonal FiLM goal initialization: replacement DG studies

**Status, 8 October 2026:** the earlier episode-long C15 and C05 production jobs were cancelled at the user's request; the finite-horizon six-run study had already completed. Three corrected 1,048,576-frame compute-node qualifications and exact checkpoint reloads passed. All 18 replacement runs have been submitted and were running at the first scheduler check. No replacement performance result is available yet.

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
