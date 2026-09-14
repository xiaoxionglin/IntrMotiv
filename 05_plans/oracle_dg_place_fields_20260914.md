# Oracle DG landmarks: four precise goals

14 September 2026. Design proposal; no implementation or training submission.
Based on saved run results and checkpoint/config extraction, not a fresh cluster audit.

## Recommendation and question

Use the **DGP C15 HIT / JOINT / target-ID FiLM** configuration, with
`DGP_C15_HIT_JOINT_FILM_S99` as the concrete reference. Start with four fixed,
compact oracle DG fields near the four accessible corners, within a total of
16 DG channels. Only those four identities are manager goals. The remaining
12 learned channels continue to provide CA3 context.

The question is whether stable, spatially unambiguous goal landmarks allow the
existing worker to learn command-specific navigation. This is an oracle
diagnostic: a positive result establishes sufficiency under privileged landmark
input, not that the learned visual representation can acquire these fields.
Precision, stability, orientation invariance, and spatial placement change
together; the first experiment does not isolate width alone.

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
repeat 8, frozen ImageNet ResNet-18 through layer 2, 16 DG channels,
`R=8`, `L=64`, depth/instruction bypass, immediate target-ID FiLM, APPO,
JOINT gradients on learned DG, ARR encourage, legacy BatchNorm for learned
channels, and `hit_distance` reward. Keep recruitment replacements disabled.
Preserve complete expanded parent arguments, not just this summary.

Use **fresh training with the parent configuration** for the main comparison.
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
objective, not the original 16-learned-row objective.

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
This prevents a broad learned context unit from winning an all-16 argmax or
breaking an exclusive oracle event. Keep tensor dimensions fixed at 16 and
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

## Minimal comparison and optional count sweep

The proposed main comparison has three arms, each with seeds 8, 99, and 123:

| Arm | DG input to CA3 | Allowed manager identities | Purpose |
|---|---|---|---|
| Parent | 16 learned | All 16, parent discovery | Reproduce the historical configuration on the qualified source |
| Learned-4 | 16 learned | Fixed IDs 0–3 | Control for restricting the manager vocabulary/recognition |
| Oracle-4 | 4 fixed oracle + 12 learned | Fixed IDs 0–3 | Test stable precise goal landmarks at the same vocabulary size |

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
$F=16$ and fixed field width: two opposite corners; four corners; then those
four plus four accessible edge midpoints. Match learned-mask controls at each
$K$. This changes goal count, landmark coverage, and learned context capacity
($16-K$), so report it as such rather than a pure capacity effect. Avoid $K=1$:
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
   Test empty candidate sets, source absence, timeouts, and no repeated hits.
3. Check travel feasibility at repeat 8 and the inherited 64-decision bootstrap
   and passive-transition limits. Widely separated corners may exceed them;
   an 80-unit field may also be crossed between decision observations. Measure
   attainable path times and decision-time detection using a scripted/manual
   environment preflight independent of the trained policy. Do not infer
   feasibility from a straight-line distance or the parent’s mean movement.
   If inadequate, revise and declare common horizons in both restricted arms
   before production and repeat print-only review. Do not silently change
   `L`, action repeat, or use swept-path hit credit.
4. Run frozen-policy command interventions with all three alternative commands
   from each oracle source region: 12 ordered pairs, initially at least 20
   matched attempts per pair per seed, balanced over heading and episode/start
   contexts. Match complete recurrent histories, not just $(x,y)$. Verify that
   the environment supports exact state restoration before promising cloned
   starts; otherwise report randomized, start-stratified trials as approximate
   matching. Reuse/extend the established intervention evaluator compatibly.
5. Primary evidence is goal-macro physical arrival probability before a common
   declared deadline, and its difference from a command-shuffled execution
   control. In that control, give a balanced alternative command to the worker
   while scoring the originally assigned target from the same starting context.
   Keep graph and normalization frozen. Include every timeout/failure and report
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
memory, or exploration; missing candidates or impossible deadlines make the
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
recognition, replay alignment and deadline feasibility before a count sweep.
The planning checks passed; runtime feasibility and scientific outcomes remain
untested. Reuse this contract and the canonical workflow on implementation.
