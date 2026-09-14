# Oracle DG landmarks: four precise goals

14 September 2026. Design proposal; no implementation or training submission.
Based on saved run results and checkpoint/config extraction, not a fresh cluster audit.

## Recommendation and question

Use the **DGP C15 HIT / JOINT / target-ID FiLM** configuration, with
`DGP_C15_HIT_JOINT_FILM_S99` as the concrete reference. Start with four fixed,
compact oracle DG fields near the four accessible corners, within a total of
32 DG channels: **4 oracle + 28 learned**. Only those four identities are
manager goals. The learned channels continue to provide CA3 context. The
initial proposal has **no option deadline**: retain the commanded goal until
arrival or the existing physical episode boundary. These revisions follow the
user’s review; they are proposed experimental settings, not runtime changes.

The question is whether stable, spatially unambiguous goal landmarks allow the
existing worker to learn command-specific navigation. This is an oracle
diagnostic: a positive result establishes sufficiency under privileged landmark
input, not that the learned visual representation can acquire these fields.
Precision, stability, orientation invariance, and spatial placement change
together; the first experiment does not isolate width alone.

## Scope correction after checking R3

The user’s successful randomized three-goal task materially narrows the value
of this proposal. A live read-only W&B API recheck of
[HippoSLAM_R3_2](https://wandb.ai/xiaoxionglin-bernstein-center-freiburg/HippoSLAM_R3_2)
on 14 September found 30 runs. At 32 DG units, mean raw scores are 9.91 and
9.83, with length-weighted scores 8.3651 and 8.2337 for instruction scales 1
and 9. Each value averages eight PBT policy summaries within a run and then
five seeds. These are unequal-age terminal summaries, not success percentages
or matched evaluations. They reproduce the prior saved audit.

A representative live config confirms `BypassSS`, `R=8`, `L=64`, numerical
instructions, repeat 8, eight PBT policies, and a frozen loaded visual encoder
(`pretrained_resnet` with an explicit load path). This is not the exact frozen
ImageNet layer-2 trunk/FiLM/JOINT setup proposed here. W&B configs and scores
alone still do not reconstruct the archived task randomization or reward code;
the three-randomized-goal description is supplied by the user.

**Do not present four fixed oracle destinations as a new demonstration that
DG–CA3 can support goal-directed navigation.** R3 already provides strong prior
evidence for that broad ability. The justified remaining question is whether
stable, localized landmark identities repair command dependence within the
otherwise failing IntrMotiv objective, memory, and self-selected-goal system.

Accordingly, treat this as a bounded diagnostic, not a new large training/count
sweep by default. Keep the 32-unit/no-option-deadline settings matched, preserve
IntrMotiv goal selection and learning mechanics, and compare learned versus
oracle goal channels. Removing passive discovery and manually/randomly supplying
all goals, as suggested in conversation, would further turn it into a supplied-
goal navigation task; do not silently fold that into the primary experiment.
If discovery prevents testing the worker, document that failure and label an
externally commanded worker probe as a separate localization of the bottleneck.
Any benefit relative to historical DGP cannot be assigned solely to precision
because width and option expiration also changed; use the matched new control.

Success would support the oracle landmark package as a repair in this IntrMotiv
setting, not prove precision alone or autonomous landmark discovery. Failure
would show that representation replacement is insufficient here despite R3’s
successful navigation; it would not imply DG–CA3 cannot navigate.

Recheck artifact: [compact per-run and grouped scores](../06_experiments/results/hipposlam_r3_2_20260910/oracle_planning_recheck_20260914.json).
Process lesson: check already successful supplied-goal baselines before expanding
an oracle study. The first sandboxed API connection could not verify the token;
the authorized network-enabled read succeeded. Reuse compact API summaries for
this question; no history scan, training modification or cluster login was needed.

## Why this parent

Saved 65–75M behavior and 75M, 100k-observation spatial snapshots show:

| Seed | Option success | Target/shuffle activation lift | Mono-field units /16 | Stationary observations |
|---|---:|---:|---:|---:|
| 8 | 51.40% | 1.0019 | 0 | 1.67% |
| 99 | 56.25% | 0.9958 | 2 | 1.82% |
| 123 | 49.49% | 1.0028 | 1 | 2.24% |

This family remains mobile and learns goal-varying FiLM parameters, but has
almost no activation advantage for the commanded goal. It is an informative
failed-control baseline, not a proven successful controller. Seed 99 is a
selected descriptive example; use all three paired seeds for conclusions.

Prefer this over FIRST training, which also changes the reward/termination
problem, and over adding DDQN/HER or waypoint routing to this first test.
The spatial outliers `SCR_C15_ARR_DIRS_S123` and
`SAT_C15_ARR_DIRO_FILM_S8` are useful illustrations, but introduce different
retirement/manager histories. The recent direct DG64 runs also show substantial
late control and movement failure, making them less clean starting points.

Retain the parent environment `openfield_map2_fixed_loc3_fixedlength_noreward`,
repeat 8, frozen ImageNet ResNet-18 through layer 2,
`R=8`, `L=64`, depth/instruction bypass, immediate target-ID FiLM, APPO,
JOINT gradients on learned DG, ARR encourage, legacy BatchNorm for learned
channels, and `hit_distance` reward. Keep recruitment replacements disabled.
Increase total DG width from the historical 16 to 32 in both primary arms,
and disable option expiration in both. Preserve complete expanded parent
arguments and explicitly record these deltas, not just this summary.
With `R=8`, `L=64`, CA3 grows from 16×71=1,136 to 32×71=2,272 values;
this also changes decoder input and graph/goal-table dimensions. More context
capacity is plausible, but an improvement is not established by the prior runs.

Use **fresh training with the revised parent configuration** for the main comparison.
Replacing four learned identities inside the trained checkpoint would invalidate
their old CA3 histories, FiLM meanings, value estimates, and graph edges. A
checkpoint rescue experiment is a separate transfer question requiring explicit
reset/load scopes and matched continuation controls.

## Oracle field definition

Reserve zero-based channels 0–3. For horizontal world position $p_t$ and a
fixed center $c_i$, use a thresholded, peak-normalized Gaussian:

$$
g_i(p_t)=\exp\left(-\frac{\|p_t-c_i\|^2}{2\sigma^2}\right),\qquad
a_i(p_t)=\frac{\max(0,g_i(p_t)-e^{-2})}{1-e^{-2}},\qquad \sigma=20.
$$

Each field peaks at one and is exactly zero at and beyond radius 40 DMLab
units. Its support diameter is 80 units; its disk area is approximately 5,027
square units, about half a 100×100 tile. An unthresholded Gaussian is unsuitable:
positive tails would make an `activity > 0` detector active everywhere.

Place centers at centers of the four outermost accessible corner floor tiles,
with the support clear of walls and collision margins. The telemetry bounds
`[100,2000] × [100,2000]` suggest provisional coordinates
`(150,150), (1850,150), (150,1850), (1850,1850)`, but these are **not verified
walkable coordinates**. Verify the actual level geometry, position convention,
and reachability before freezing the study. Retain the chosen centers unchanged
across training seeds; do not choose them from a successful trajectory.

Inject these values at the post-threshold DG interface before CA3 writes.
Do not pass them through learned BatchNorm, row normalization, recruitment, or
DG losses. Learned-row losses must explicitly mask oracle output rows and use
the learned-row denominator. Oracle values can remain fixed context when a
learned-row temporal loss needs them. Record that this is a mixed representation
objective, distinct from the 32-learned-row control objective.

Only the field transform may consume privileged position. The worker sees
oracle activity through ordinary CA3 and target IDs; it receives no coordinates,
target coordinates, heading oracle, distance-to-goal shaping, or oracle route.
The existing telemetry pose channel is deliberately removed before policy
input: this variant therefore needs an explicit opt-in observation/replay path,
with contemporaneous pose and reset-safe alignment. Do not repurpose the
monitoring channel silently. Save centers, width, channel IDs, and enablement in
checkpoint/config state; default behavior must remain unchanged.

## Goal vocabulary and recognition

For the restricted-goal arms, apply the same identity mask to manager landmark
recognition, source/target identities, passive evidence, hit detection, and
graph updates. Channels outside the mask remain CA3 context, not manager nodes.
This prevents a broad learned context unit from winning an all-channel argmax or
breaking an exclusive oracle event. Keep tensor dimensions fixed at 32 in the primary comparison and
mask unused graph rows/columns rather than changing the controller architecture.

Preserve the parent's least-tested selection among observed passive successors,
intersected with the allowed identity mask. Thus the four IDs are the allowed
goal vocabulary; the instantaneous candidate set can be smaller until passive
discovery. Do not silently prepopulate edges or give the manager geometric
knowledge. With four identities there are at most 12 directed distinct pairs.

Log source availability, candidate counts, empty-candidate time, and visits to
each field from the start. Preserve the parent's no-candidate behavior and
verify it actually explores. A failure to discover candidates is a manager/
exploration bottleneck, not evidence that a worker cannot use oracle fields.
If discovery blocks the experiment, a separately declared follow-up may make
all allowed noncurrent goals available without passive evidence, using the
same selector in both representation arms.

Count an option hit once at arrival using the old commanded goal and the
correct action/outcome alignment. Exclude currently occupied goals at command
selection. Do not reward dwelling, reset teleports, or repeated samples within
one arrival as new completions.

## Parent name and the meaning of context

- **DGP** identifies the DG policy-gradient experiment family.
- **C15** is the inherited configuration label, not a count of DG units. In
  this DGP batch the target rule was changed to least-tested observed passive
  successors with direct control; it should not be confused with the original
  C15 frontier-UCB curriculum.
- **HIT** rewards eventual activation of the commanded identity; encountering
  another identity does not terminate the option as a wrong FIRST outcome.
- **JOINT** lets PPO gradients reach learned DG through the worker’s CA3 input,
  alongside the separate DG objective. The visual ResNet remains fixed, and
  oracle fields have no trainable parameters.
- **FiLM** uses the commanded target ID to modulate worker hidden activations.

“Context” means the activity histories of the 28 learned DG channels in CA3,
which can help distinguish observations and recent routes between oracle fields.
They are not 28 additional goals. For example, at oracle corner 0 its activity
could be 0.8 while a broad learned channel is 1.4. Recognition based on a global
argmax could select the learned channel, while an all-channel exclusivity test
could reject the event entirely. This is an interface hazard to test against the
chosen runtime, not a newly verified bug. Restrict manager recognition to the
four goal channels; preserve all 32 channels for worker CA3 context.

## No-deadline initial variant

No bootstrap or learned-edge option expiration: an active command persists
through other landmark encounters until its own hit or a physical episode end.
Do not encode this by setting a huge horizon or changing `Hippo_L`. Keep
`L=64`, rollout/recurrence length, and the physical episode duration unchanged.
The fixed-length level’s 120-second boundary still bounds a failed attempt
(about 900 decisions at repeat 8). At reset, terminate the pending attempt
without hit credit and record episode termination separately from option timeout.

Longer attempts can make accidental eventual arrival easier, so compare
commanded and shuffled-command arrival curves versus elapsed decisions, not
only final success. Use a common bounded evaluation observation window in both
arms; that measurement window is not a training timeout. Report time to hit,
pending/unfinished attempts, and goal exposure as well as completed attempts.
A bad command can occupy the remainder of an episode; quantify that behavior
before deciding whether deadlines should become a later experimental factor.

The passive-successor temporal eligibility limit is a separate mechanism from
option expiration. Retaining its parent value can still prevent discovery of
widely separated goals even with unlimited option duration. Log this explicitly
and qualify discovery before interpreting learning; removing option deadlines
does not remove that gate. Likewise, CA3 memory still has a finite horizon.

## Minimal comparison and optional count sweep

The primary comparison has two matched arms, each with seeds 8, 99, and 123.
The historical configuration is a reference, not a capacity/deadline-matched
control; an exact fresh parent rerun is optional:

| Arm | DG input to CA3 | Allowed manager identities | Purpose |
|---|---|---|---|
| Historical parent (reference) | 16 learned | All 16, parent discovery | Original finite deadlines; contextual comparison only |
| Learned-4 | 32 learned | Fixed IDs 0–3 | Matched width, restricted vocabulary, no option deadline |
| Oracle-4 | 4 fixed oracle + 28 learned | Fixed IDs 0–3 | Same width/vocabulary/deadline setting; oracle representation intervention |

Predeclare learned IDs; do not select the four best-looking fields after
training. Learned-4 controls vocabulary size and recognition mechanics, but
does not match the physical goal regions. Its identity success must not be
compared with physical corner success as though they were the same task.
Evaluate command dependence against each arm's own matched null, and ground
learned identities with frozen evaluation maps.

Run a short seed-99 functional preflight for each arm, then a 25M pilot with
saved 5M/25M checkpoints. After runtime and intervention qualification, use the
same paired three-seed 75M budget as the parent. Retain checkpoints at
1M, 5M, 25M, 50M, and 75M for the standard five-checkpoint seed-99 trajectory;
retain terminal checkpoints for seeds 8/123. Use a 65–75M primary online window.
An early null at 25M is diagnostic, not a final impossibility claim.

Do not initially cross controller, gradient mode, width, and oracle count.
After the four-goal test, a useful count series is $K=2,4,8$ at fixed total
$F=32$ and fixed field width: two opposite corners; four corners; then those
four plus four accessible edge midpoints. Match learned-mask controls at each
$K$. This changes goal count, landmark coverage, and learned context capacity
($32-K$), so report it as such rather than a pure capacity effect. Avoid $K=1$:
it cannot test choosing between destinations. For precision specifically, the
next control is broader fixed fields at the same four centers and peak amplitude,
with a common narrow physical-arrival evaluation region for both widths.

## Preflight and evaluation gates

1. Verify field values, support, orientation invariance, and position alignment
   at centers/boundaries and across resets; fail closed on missing pose. Check
   actor, learner replay, checkpoint reload, and evaluator equality for oracle
   activity and manager events. Confirm unchanged trunk parameters and buffers,
   fixed oracle fields, learned-row updates, and zero oracle replacement.
2. Exercise simultaneous learned-context and oracle activation. Confirm only
   masked identities affect manager events and the commanded arrival is detected.
   Test empty candidate sets, source absence, episode ends, and no repeated hits.
   Verify goals survive both the old bootstrap deadline and learned-edge
   deadlines, remain unchanged at wrong landmarks, and end correctly on hit/reset.
3. Check travel feasibility at repeat 8 within a physical episode, and
   separately qualify the inherited 64-decision passive-transition limit.
   Widely separated corners may exceed the passive discovery window;
   an 80-unit field may also be crossed between decision observations. Measure
   attainable path times and decision-time detection using a scripted/manual
   environment preflight independent of the trained policy. Do not infer
   feasibility from a straight-line distance or the parent’s mean movement.
   If discovery is inadequate, revise and declare the common eligibility
   protocol in both restricted arms before production and repeat print-only
   review. Do not reintroduce an option deadline as an implicit fix, silently
   change `L` or action repeat, or use swept-path hit credit.
4. Run frozen-policy command interventions with all three alternative commands
   from each oracle source region: 12 ordered pairs, initially at least 20
   matched attempts per pair per seed, balanced over heading and episode/start
   contexts. Match complete recurrent histories, not just $(x,y)$. Verify that
   the environment supports exact state restoration before promising cloned
   starts; otherwise report randomized, start-stratified trials as approximate
   matching. Reuse/extend the established intervention evaluator compatibly.
5. Primary evidence is goal-macro physical arrival probability before a common
   declared evaluation window, and its difference from a command-shuffled execution
   control. In that control, give a balanced alternative command to the worker
   while scoring the originally assigned target from the same starting context.
   Keep graph and normalization frozen. Include every unfinished/failed trial and report
   trial counts, per-goal/per-seed results and uncertainty. A post-hoc shuffled
   activation curve is not this intervention.
6. Also record first distinct goal arrival, eventual arrival, time/path length,
   visitation/occupancy, stationarity, empty-candidate exposure, and online
   option success. FIRST chance is $1/3$ only for a uniformly assigned alternative
   goal and one command-independent outcome; it is not the baseline for eventual
   HIT. Action sensitivity alone is insufficient. Sparse landmarks may leave
   CA3 without a useful location/heading history between fields, even if their
   recognition is perfect.

A positive result requires improved commanded physical arrivals relative to the
execution null across seeds, not merely more hits or reliable graph edges.
Accurate fields plus no command advantage points toward worker credit assignment,
memory, or exploration; missing candidates or unreachable goal regions make the
worker test inconclusive. An overall negative result does not rule out precise
representations generally: four sparse corners are not a complete state code.

For spatial telemetry use the existing manifest-driven 10k-decision evaluator
and its standard active-only cosine, peak diversity, silence, amplitude-weighted
spatial score, and pre-threshold diagnostics. Separate oracle and learned rows.
The ordinary 100-unit map bins cannot resolve an 80-unit support well: add a
declared fine-resolution field check from saved positions/activities while
preserving standard NPZ/map meanings. Oracle raw Gaussians must be labeled
separately from learned pre-threshold BN logits. Keep policy-driven maps distinct
from fixed-observation stability checks.

## Provenance and implementation handoff

Parent StudySpec: [dg_policy_gradient_first_outcome.study.json](../hpc_runs/studies/dg_policy_gradient_first_outcome.study.json).
Validated locally on 14 September: schema `intrmotiv/study/v1`, declared workflow
`1.4.1`, collector/validator implementation `1.8.1`, 24 parent runs, SHA-256
`2e3104c975188e7cddeb71bce8816c0f4f0d6eb96688c44e0ea2b7560b5447b5`.

Reference checkpoint from the saved extraction: 75,038,720 frames,
`checkpoint_000004580_75038720.pth`, SHA-256
`0cbb3ce047909276d7281346621217bac7474260d74176e8afbd32a7ac3447eb`.
Its full workspace path and config are in
[checkpoint_extract.json](../06_experiments/results/film_input_audit_20260914/checkpoint_extract.json).
The spatial snapshot has a different actual frame count (75,005,952); do not
claim exact checkpoint/snapshot identity.

New oracle CLI/runtime support and verified centers are not established by
this planning task. Once implemented, put the complete executable StudySpec
under `hpc_runs/studies/`, using schema `intrmotiv/study/v1` and a compatible
declared workflow version. Make that validated spec the sole source of run
names, factors, seeds, targets, metrics, contrasts and intervention metadata.
Record its generated SHA-256 here; no new study fingerprint exists yet.

Follow [the canonical workflow](../04_implementation/standardized_study_workflow.md):
`validate`, `render-runs`, SF print-only generation, `audit-submission`, ordinary
Slurm preflight, then production only under a submission request. Reuse
`collect-online` and `render-telemetry`; telemetry rows are ordinary independent
jobs, not arrays. Synchronize and test newly required workflow/evaluator pieces
in an isolated NEMO2 source checkout before use. All training, logs, caches,
temporary data and raw telemetry belong under `/work/classic/fr_xl1014-train`.

Evidence: [terminal joined data](../06_experiments/results/late_outliers_20260908/dgp_terminal_rankings.csv),
[goal-set audit](../06_experiments/06_high_option_success_goal_sets_and_controls_20260914.md),
[FiLM audit](../06_experiments/07_film_goal_parameters_and_ca3_depth_weights_20260914.md),
[DGP failure analysis](../06_experiments/dgp_interim_failure_audit_20260907.md),
[DG64 health](../06_experiments/dg_capacity_health_20260913.md).

## Reusable workflow lesson

Saved canonical terminal tables plus extracted config/provenance were sufficient
for candidate selection; no new TensorBoard scan or SSH was needed. Validate
the parent StudySpec and distinguish checkpoint frames from snapshot targets.
Read only named metadata/config keys from checkpoint extracts: dumping a
list-backed extraction accidentally includes large parameter arrays and obscures
the evidence. For oracle interventions, define activity support, vocabulary,
recognition, replay alignment and physical-episode feasibility before a count sweep.
The user review also separates DG width, goal count, option expiration, passive
discovery windows and CA3 memory length: changing one does not change the others.
Use matched 32-unit/no-deadline arms rather than attributing their difference
from the historical 16-unit parent entirely to the oracle fields.
The planning checks passed; runtime feasibility and scientific outcomes remain
untested. Reuse this contract and the canonical workflow on implementation.
