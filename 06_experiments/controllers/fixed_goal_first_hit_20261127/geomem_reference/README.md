# GeoMem reference method and graph-contract audit

Date: 2026-11-27. This bounded audit reuses GeoMem's task constructors and observation/action contracts. It performs **no training, native DMLab execution, installation, or modification of GeoMem**. It follows GeoMem's method of identifying the information requirement before comparing memory mechanisms; it does not claim transfer between the projects.

## What was reused

Source repository: `/home/xiaoxiong/Desktop/Projects/geomem`, commit `b60fc27bd177ee8bfffc54bdfa37f171627aa17a`. The canonical sources are `geomem-env/src/geomem_env/task_graphs.py` and `graph_process.py`. Their exact bytes were copied before execution and imported from `source_snapshot/`, with the bridge and provenance helper also archived. `source_audit.json` checks archive integrity separately from changes to original paths.

The adapter in `hpc_runs/first_hit_control/geomem_tasks.py` preserves these distinctions:

- A latent node and its sensory identity are separate. A requested sensory identity denotes every node with that observation, including aliases.
- GeoMem's process arrays contain fixed-policy joint edge probabilities. For example, a 0.25 outcome of one of two uniformly selected actions has joint mass 0.125. The controlled adapter reconstructs the conditional probability 0.25 from the exported edge.
- Missing actions remain unavailable, with zero transition rows and an explicit action mask. They do not become invented self-loops.
- Event and reset labels are environmental scheduling operations. The purely controlled adapter refuses these graphs until an explicit scheduler is defined. It never gives their labels to the agent as controllable actions.
- GeoMem's effective observation includes the available-action mask. Any future comparison must keep this additional information identical across methods and distinguish it from DG-only sensing.

## Reused witnesses and implications

| Constructor | Export | Opposing-action alias pairs | Information result | Implication for first-hit control |
| --- | --- | ---: | --- | --- |
| `delayed_match(delay=4)` | 22 nodes, 8 action edges, 12 event edges | 2 | Same current probe requires opposing responses; observed onset cues distinguish the histories. With categorical CA3 repeat 1, length 4 loses the cue and length 8 preserves it. | The objective cannot recover an expired cue. A scheduler must advance observations without inventing actor choices. |
| `context_decision(delay=2)` | 40 nodes, 16 action edges, 16 event edges | 2 | Conflicting modalities require remembering the initially observed rule. | Same-goal targets are conditionally sufficient only when retained history identifies the relevant rule. The public action mask also changes observation classes. |
| `alternation(length=4)` | 14 nodes, 20 action edges, 4 reset edges | 1 | First-trial observation histories are identical despite opposing correct choices. A previous public outcome distinguishes the contexts only if memory survives the reset. | No larger finite observation memory resolves an undisclosed initial previous side. Reset semantics must match the actual controller. |

Each listed pair uses equally likely hidden contexts, so any policy depending only on the common current observation has one-step success at most 0.5. These are exact information-limit witnesses, not learned-agent measurements. The first two constructors contain informative earlier cues; their loss from a finite trace is a separate, explicitly measured boundary. Alternation's initial ambiguity is stronger: the entire available initial observation history is identical. Its subsequent-trial history contains informative outcome feedback, but clearing memory at reset removes that information.

The source tasks' correctness/success labels are used only to identify incompatible action requirements. They are not imported as reward-free IntrMotiv training supervision. All three graph documents are accepted by GeoMem's process validator and intentionally rejected by the purely controlled adapter because they contain event/reset edges. This is a contract finding, not a failed controller experiment.

`summary.json` records node/edge counts, observation classes, scheduling assumptions and hashes of every graph export, process NPZ and witness JSON. The witness files preserve both latent-node paths and their public observation sequences, allowing the alias and CA3-expiry checks to be reconstructed without another simulation.

## How to use the GeoMem approach for the next IntrMotiv simulation

GeoMem's dependency-free `environment.py` already provides `aliased_rooms`, `aliased_loop_rooms`, `aliased_t_maze`, `landmark_gap_detour`, `shortcut_route_choice`, and `sparse_visual_feature_channel`. These are preferable starting points to recreating navigation graphs. Preserve the constructors' explicit wall-attempt/self-loop contract: this interface differs from TaskGraph's absent-edge contract. In reward-free collection, use `Environment.step` and an explicitly declared physical termination rule; `step_result` otherwise supplies the source task's success reward and goal termination.

Begin with matched random-action trajectories and frozen sparse sensory identities. Compare the existing CA3 plus first-hit objective with the existing objective on those same histories and update budgets. Vary cue availability and history length around the observed alias/expiry boundary. Add action-history information only after an exact same-input/opposing-action witness demonstrates that the retained state is insufficient. Test loops, backtracking and held-out complete histories rather than shuffled rows. Keep goal-detector sets separate from unique physical locations.

For a later held-out transition-law test, `landmark_navigation.py` supplies the existing four-action movement law, sparse aliased landmark sensor, separate random streams and route digests. Its even/odd row-sign split withholds combinations of familiar local rules; it does **not** withhold graph topology. A held-out topology claim requires a separately declared graph-family split. The existing reference knows the transition/sensor family and is an information ceiling, not an equally trained competitor.

Two implementation details require care. The sparse sensor's random seed must vary by observation event to obtain temporal dropout; a fixed seed/state pair repeats its outcome. The generic T-maze symbol function contains a fixed `position >= 8` threshold, so changing corridor length while blindly reusing that sensor changes its meaning. Keep the canonical length or specify and test a parameter-aware observation adapter.

The `landmark_row_budget` workflow provides the relevant experiment organization: immutable seed/condition shards, exact data and label digests, matched examples and optimization budgets, validation-only model selection, complete evaluation cells, and refusal to assemble incomplete or mismatched artifacts. Reuse these principles rather than launching or rebuilding its historical experiment matrix. Small fitting failures should trigger an approximation/data diagnostic before an architectural conclusion.

## Reproduction

From the IntrMotiv repository root, with existing `SF_git` dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xiaoxiong/miniforge3/envs/SF_git/bin/python -m unittest hpc_runs.test_first_hit_geomem -v
PYTHONDONTWRITEBYTECODE=1 /home/xiaoxiong/miniforge3/envs/SF_git/bin/python -m hpc_runs.first_hit_control.geomem_tasks --output /tmp/intrmotiv-geomem-reference-new
```

The output directory must be new. Eight focused tests pass: probability semantics, unavailable actions, sensory alias classes, event/reset rejection, terminal validation, probability validation, delayed-cue expiry and alternation's reset-history distinction. The task documents and witness JSON are deterministic; NPZ archive hashes bind the actual files written in this run.
