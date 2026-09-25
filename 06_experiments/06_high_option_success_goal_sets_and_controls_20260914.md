# 06 — High option success: candidate goals, accidental hits, and matched controls

## Architecture factorization

| Factor | Levels / setting in this report | Interpretation |
| --- | --- | --- |
| Family | Selected DGP HIT runs plus matched FIRST/STOP controls; one Saturday frontier-manager case | Not a balanced architecture ranking |
| Goal representation | DG target ID, LEG or FiLM depending selected run | Candidate vocabulary can be broad while physical destinations remain ambiguous |
| Outcome rule | HIT allows intervening wrong landmarks; FIRST terminates on first distinct outcome | This difference is central to why eventual success can look high |
| Controller | PPO; JOINT or STOP DG routing in selected DGP rows | Controller and DG-learning path differ in some selected cases |
| Graph | Passive/controllability eligibility and reliability statistics | Edge density/coverage are proxy evidence, not verified skills |
| Primary question | Does high option success imply command-specific navigation? | The report answers this with matched controls and says no |

Cross-report context: [[README|factorized experiment synthesis]].


14 September 2026. Analysis of saved September 6–8 artifacts; not a fresh runtime collection. Select the highest terminal DGP HIT run, the strongest spatial HIT candidate, and a STOP control. Add the highest Saturday terminal option-success run as a different-manager comparison. This is targeted descriptive analysis, not a statistical selection-independent ranking.

DGP uses passive_confidence[source,target] > 0 eligibility, balanced by decayed attempt evidence. The set saturated to all 15 alternatives per source / 240 directed pairs, so these terminal HIT scores were not confined to a small eligible-goal subset. Eligibility does not mean every goal is successfully controlled. HIT permits intervening wrong identities; FIRST ends on the first distinct exclusive outcome. A 1/15 baseline pertains to a uniform command and a single command-independent outcome, not to eventual HIT over multiple opportunities.

| HIT run | Option success, 65–75M | Target/shuffle activation lift | Mono-fields at75M /16 | Reliable edges /240 | Matched FIRST option success | FIRST outcome lift |
|---|---:|---:|---:|---:|---:|---:|
| DGP_C15_HIT_JOINT_FILM_S99 | 56.25% | 0.9958 | 2 | 168 | 7.06% | 0.9447 |
| DGP_C15_HIT_JOINT_LEG_S123 | 55.52% | 0.9973 | 5 | 166 | 6.78% | 0.8964 |
| DGP_C15_HIT_STOP_LEG_S8 | 52.07% | 0.9928 | 1 | 143 | 6.86% | 0.9596 |

The FIRST rows are separately trained seed-matched conditions, not counterfactual evaluation of the same checkpoint. HIT lift is pooled per-decision target activation versus shifted-target activation; it is not the ratio of option-success probabilities. It provides no positive command-specificity evidence here, but is not a matched-start intervention. Spatial maps use 100k policy-driven observations at75M; graph edges reflect the proxy success definition and are not verified skills.

Saturday SRC MON FILM S8: terminal summary option success56.996%, target activation2.3832% versus shuffled2.3771% (ratio1.0026); top-three incoming share0.6785, attempt coverage0.9567, zero mono-fields. These use five summary rows after70M, unlike DGP65–75M. Frontier selection is adaptive, and high aggregate attempted-pair coverage is not proof of uniform command frequency or destination-specific success. The available summaries do not reconstruct per-destination success denominators.

Interpretation: high eventual landmark activation coexists with negligible command specificity. In DGP the candidate vocabulary was broad; the successful physical destinations may still be narrow or ambiguous. Saturday also shows graph concentration. No near50% aggregate should be described as50% success at arbitrary physical goal locations.

Next evidence: per-source/target attempt and success matrices with deadlines, first-versus-eventual arrivals, macro averages over goals, independently grounded target regions, and matched-start command interventions. Use identical windows and preserve failures/timeouts in denominators.

Sources: [DGP joined terminal results](results/late_outliers_20260908/dgp_terminal_rankings.csv), [candidate audit](dgp_interim_failure_audit_20260907.md), [Saturday results](results/recent_batches_audit_20260906/saturday_terminal_per_run.csv). DGP StudySpec schema intrmotiv/study/v1, workflow1.4.1, SHA2e3104c975188e7cddeb71bce8816c0f4f0d6eb96688c44e0ea2b7560b5447b5.

Workflow lesson: distinguish eligible goals, attempted goals, successful identities, and physical destinations. Inspect candidate counts and shuffled controls before interpreting aggregate success; reuse canonical saved results before collecting histories. No training or configuration was changed.

## Poster-exemplar evidence update, 25 September 2026

For `DGP_C15_HIT_JOINT_FILM_S99`, the canonical saved 75M snapshot is at
75,005,952 frames (100,000 retained behavior samples). Its detailed row is in
[`late_outliers_20260908/dgp_snapshots/per_snapshot.csv`](results/late_outliers_20260908/dgp_snapshots/per_snapshot.csv),
with the per-unit map export at
[`DGP_C15_HIT_JOINT_FILM_S99_spatial.csv.gz`](results/late_outliers_20260908/DGP_C15_HIT_JOINT_FILM_S99_spatial.csv.gz).
The map has 88.1% grid coverage, 16/16 active units, active-only cosine 0.147,
14 unique peak bins, 2/16 mono-field units (12.5%), stationary fraction 1.8%, and path
efficiency 0.097. The median nearest-neighbor distance among dominant peaks is
0, indicating coincident peaks despite the count of distinct bins.

The graph has 168 reliable directed edges (70.0% edge density), 93.75% ordered
pair reachability, and 56,856/107,982 prospective successes (52.7%). The
existing 65–75M option-success summary is 56.25%. Those aggregate HIT values
do not show whether success depends on the commanded node: available saved
artifacts contain no matched-start alternative-goal intervention, action
probability TV, or first-reached internal-node command-by-node matrix. Therefore
command specificity remains **unavailable**; the aggregate activation lift
(0.9958 in the existing report) also supplies no positive specificity evidence.

The snapshot bundle provides occupancy and path scalars but no separately
rendered exact-run trajectory/segment panel or coverage AUC. No new DMLab
rollout or intervention was performed for this update.
