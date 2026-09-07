# CA3 finite-memory novelty and minimal goal control — implementation

Date: 2026-09-07. Canonical production definition:
`hpc_runs/studies/ca3_memory_novelty_goal.study.json`.
Schema `intrmotiv/study/v1`, workflow `1.5.0`.

## Design retained

50 production runs: ten conditions × paired seeds 8, 23, 57, 99, 123;
75M environment frames each. Separate ten-condition seed-99 preflight: 2M
frames each. Frames are not policy decisions: the environment frame skip is 8.

| Conditions | Encoder reward | Decoder mechanism | PPO → DG |
|---|---|---|---|
| F_D_BASE/GATE/SOFT/HARD | 0.1 d | Baseline / novelty gate / gate + soft inhibition / gate + hard inhibition | STOP |
| F_DMR_BASE/GATE/SOFT/HARD | 0.1(d − R) | Same four levels | STOP |
| G_DMR_STOP | 0.1(d − R) | Intrinsically sampled absent DG identity | STOP |
| G_DMR_JOINT | 0.1(d − R) | Same goal mechanism | JOINT |

Common: F=16, R=8, L=64, CA3 length E=71 decisions, DG threshold 2.43,
fixed pretrained ResNet-18 through layer 2, trainable DG and legacy BN,
arrival-recipient local-predecessor encoder credit, batch-use and multi-onset
regularizers. No external task reward, graph, manager, learned context,
recruitment, prediction auxiliary, geometry-conditioned policy, or hindsight.
The JSON owns all actual arguments and fourteen seed-paired contrasts.

## Implemented semantics

CA3 absence is tested **before injection** for reward gating. Only the dominant
refractory-qualified onset is eligible; the established flat decoder distance
reward 0.1(E − d) is preserved and multiplied by the binary gate. Encoder
distance scaling remains unchanged. Reward gating does not prohibit DG firing.

Inhibition uses the detached maximum previous CA3 amplitude for each identity.
Continuous activity is allowed. On re-entry, SOFT subtracts that amplitude and
rectifies; HARD vetoes activity while the identity remains remembered.
Both inhibit the DG signal entering CA3, not merely the decoder reward.
Remembered identities become eligible again after finite-memory expiry.

Goal conditions uniformly sample an identity absent from the updated CA3;
never-seen identities are eligible. An empty candidate set gives a null command.
The command enters the immediate target-ID FiLM adapter and persists for at
most 64 decisions. A dominant target onset at latency τ receives
`0.1(65 − τ)` once. Intermediate landmarks do not terminate or pay. A
non-dominant target activation invalidates the command without reward. Success,
invalidation, and timeout resample; episode boundaries clear active goal state.
PPO replays the actual behavior command, including the reset-safe descriptor,
instead of a command resampled during the learner forward pass.

Tests cover continuity, memory-detached gradients, pre-injection gating,
empty candidate sets, intermediate arrivals, ambiguous arrivals, last-decision
success, one-time reward, packed/single execution, unequal sequence lengths,
checkpoint round-trips, state allocation, and command replay.

## Evaluation support and interpretation

The established place-field evaluator now stores raw rectified-DG maps alongside
post-inhibition maps and continuous pre-threshold maps. Its observation-panel
record/replay options retain full policy observations and episode boundaries.
Predeclared common panel: F_D_BASE seed 99 at 5M, 10k decisions. Replay that same
panel through every checkpoint; free-policy trajectories are not drift tests.
The current-observation pose is paired with the current DG output (an existing
one-step offset was corrected), and the saved training environment is retained
instead of overriding it with the parser's bootstrap environment.

The intervention manifest selects the two goal conditions at 75M for **all
five seeds**. Other field telemetry remains seed 99 at 5/25/50/75M and seeds
8/123 at 75M. Additional goal-probe seeds use the same checkpoint inventory.

Graph-free interventions replay deterministic action prefixes from 16 seeded
starts. From each exactly verified starting observation and CA3 state, execute
every absent command for 64 decisions, with matched policy-sampling streams.
Other commands provide within-start controls. Report target-macro-averaged
matched-minus-other hit rates at 8/16/32/64 decisions and normalized trapezoidal
CTC AUC. Censored or incomplete command sets are not treated as failures or
retained selectively. Raw event times, event positions, trajectories, and path
lengths are retained. This is an evaluation intervention, not a training rule.

`qualify_absent_goal_interventions.py` additionally scores arrivals inside each
identity's independent raw-DG largest-mass half-peak connected component.
Without common-panel field evidence, spatially qualified CTC is explicitly
missing. Identity activation alone is not a claim of spatial control.

Important limits: finite memory cannot rule out cycles longer than E; suppression
can reduce counted onsets without improving physical exploration; a higher
reward or fewer multi-fields does not establish efficient goal control. Compare
coverage and physical trajectories, event repetition, silence, entropy,
active-only map cosine, peak diversity, information, and raw/post-inhibition
field components together. Short preflight results are correctness evidence,
not outcome evidence for the paper.

## Reusable workflow lessons

Use StudySpec expansion, exact run-directory discovery, and the launcher
submission audit; do not rediscover or parse condition factors from run names.
`audit_ca3_memory_preflight.py` checks finite logged metrics, nonzero decoder
learning signals, zero forbidden gradients, nonzero JOINT gradients, and exact
goal-command replay. Replay mismatch is not applicable to flat policies; its
absence there must not be classified as missing goal telemetry.

Run the full authoritative NEMO2 tests: the local Sample Factory installation
differs from the runtime. First deployment passed 254 IntrMotiv tests and 27
workflow tests. Run DMLab integration checks only as bounded ordinary Slurm
jobs, with all observations, logs, and caches in the allocated workspace.

Exact-start intervention checks exposed that repeated environment reset can
retain more state than its seed suggests. Fail closed on observation/state
mismatch; never silently downgrade this to observational label shuffling.
Record detailed per-input differences before retrying. The evaluator's old
hard-coded environment override and pose offset are general infrastructure
issues, isolated from the scientific training changes.

Training implementation lives in the NEMO2 source checkout under
`sf_working_directories/IntrMotiv/`. A scoped deployment-source archive is
retained with the submission record for reproducibility; it is not an
alternative runtime or workflow. Preserve unrelated dirty runtime changes.

## Submission and verification record

Preflight submission: ordinary jobs 7998917–7998926, fingerprint
`d0f8efc18b49f616d0d3a373b4eefeb04ff6bae18c4cead35d1262fa05d60ef3`.
Its reviewed/submitted launcher directory is
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_ca3_memory_novelty_goal_20260907_preflight/20260907T170621Z`.

Final runtime audit, evaluation smoke, production fingerprint, and production
job IDs are appended after their actual verification. Do not interpret this
intermediate record as confirmation of production submission.
