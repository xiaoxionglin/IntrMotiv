# C15 variants: control structure and behavior

## What to put in the poster

**Best control tradeoff: DGP FIRST versus HIT, JOINT LEG.** At the shared 75M online window, FIRST has **58.9% prospective success versus 50.8%**, with the improvement in all three seeds. Reachability moves in the opposite direction: **4.7% versus 96.1%**, also in all three seeds. Grounded score is **7.7% versus 1.9%**, but FIRST's nonzero score comes entirely from seed 99, so the grounding advantage is not replicated. This is a useful dissociation between target-event outcomes and connected graph structure. See the [eight-variant control SVG](dgp_online_100k_training_window_control.svg), [declared paired contrasts](declared_paired_contrasts_per_seed.csv), and [bounded command evidence SVG](dgp_command_evidence.svg).

**Best local behavior comparison: the four CPD variants with terminal probes.** BASE, GATE ACT DIR, GATE ACT DIR GOAL, and ADD CA3 DIR have all three seeds at exactly 75,038,720 frames. GATE ACT DIR has the highest mean grounded score in the 27-condition saved-window survey (**16.3%**), but it is **0 / 48.8 / 0%** for seeds 8 / 99 / 123. Its mean prospective success is **67.8%** and reachability **12.1%**. Use it as a seed-sensitive candidate, not a replicated control improvement. See [all CPD control variants](cpd_online_100k_training_window_control.svg), [terminal behavior](cpd_frozen_10k_policy_probe_behavior.svg), and the [full run index](run_index.md).

**Useful looping result within CPD:** Adding the goal predictor to GATE ACT DIR raises the 20-step mobile return fraction from **62.4% to 83.2%**, with an increase in every paired seed. Frozen arena visitation falls from **69.3% to 55.3%** on average. A goal predictor is a different experimental factor from introducing goal conditioning into the C01 baseline. This result supports a mechanism-specific failure example; it does not support saying every C15 descendant reduces loops.

**Source-credit candidate:** SCR ARR DIRS combines **90.1% reachability**, **45.9% prospective success**, and **13.1% mean grounded score**. Its grounding is **0 / 0 / 39.2%** across seeds 8 / 99 / 123. It is another individual-run candidate with weak replication. [Control SVG](source_credit_online_100k_training_window_control.svg).

**Keep the original C15 as the exploration anchor.** C14 changes frontier scoring to visit novelty; C16 enables explicit waypoint planning. Original C15 has the strongest terminal training coverage of the three; [training comparison](corrected_core_training_comparison.svg) and [original analysis](../flat_goal_comparison/report.md). In its archived 100M frozen probes, mean 20-step return is **5.9%** and 40-step return **14.6%**. CPD probes loop substantially more, but the 75M versus 100M ages and changed mechanisms make that a cross-family description, not a matched architecture effect.

## Scope and variant identities

The catalogue contains **67 conditions / 201 declared runs**. **66 conditions / 198 runs** have existing control or training diagnostic records. Persistent C15 continuation is declared but has no matching measurement in these local exports. [Variant catalogue](variant_catalogue.md) gives every condition and its settings; [registry](variant_registry.csv) preserves schema, workflow version, and StudySpec SHA-256. This survey covers explicitly C15-based studies and C14/C16 controls, rather than treating every later controller as the same architecture.

| Family | Conditions | What changes | Primary available evidence |
| --- | ---: | --- | --- |
| Corrected core | 3 | C14 visit novelty; C15 UCB direct; C16 UCB waypoints | 100M terminal scalars; three C15 frozen probes |
| DPR | 6 | MON/DIR/PRED recruitment × legacy/FiLM goal input | Terminal training diagnostics; six exploratory seed-99 frozen probes |
| Source credit | 10 | Arrival/source encoder credit × five recruitment/endpoint rules | Shared 75,005,952-frame retained training window |
| Saturday | 12 | Arrival/source credit × MON/DIR-open/PRED-open × legacy/FiLM | Final scalar window beginning at 70M, reported last sample at 74,973,180 |
| DGP | 8 | HIT/FIRST worker outcome × STOP/JOINT DG gradient × legacy/FiLM | Shared 75,005,952-frame retained window; selected 75M command probes |
| CPD | 27 | BASE; passive/goal prediction; gate/add feedback; CA3/activity history; direct/BPTT gradients | All-condition shared 75,005,952-frame retained window; four-condition terminal probes |
| Persistent C15 | 1 | Declared continuation of the C15 base | No matching local measurement; not ranked |

All these C15 descendants use goal-conditioned workers. C15-based StudySpecs here use direct frontier management; **C16 is the explicit waypoint control**. The catalogue reads exact goal interfaces, target selection, outcome definitions, gradients, and recruitment settings from StudySpecs. Names such as FIRST, FiLM, or goal predictor are not interchangeable interventions.

## Control figures and meanings

| Comparison | Editable SVG | Protocol |
| --- | --- | --- |
| CPD: 27 conditions | [Control](cpd_online_100k_training_window_control.svg) | Online retained 100k-observation training window |
| DGP: eight conditions | [Control](dgp_online_100k_training_window_control.svg) | Online retained 100k-observation training window |
| Source credit: ten conditions | [Control](source_credit_online_100k_training_window_control.svg) | Online retained 100k-observation training window |
| Saturday: 12 conditions | [Control](saturday_final_training_scalar_window_control.svg) | Mean terminal scalar diagnostics; prospective counters unavailable |
| DPR: six conditions | [Control](dpr_final_training_scalar_window_control.svg) | Mean terminal scalar diagnostics; prospective and grounded metrics unavailable |
| C14 / C15 / C16 | [Coverage and worker diagnostics](corrected_core_training_comparison.svg) | Final 10M training scalars; modern prospective/grounded metrics unavailable |
| DGP command probes | [Executed, shuffled, pair coverage](dgp_command_evidence.svg) | Bounded frozen command trials, selected JOINT LEG arms |
| Saturday command probes | [Executed, shuffled, pair coverage](saturday_command_evidence.svg) | Bounded frozen command trials, selected MON FiLM arms |

Each point is one trained seed; black marks are equal-weight seed means. These are descriptive comparisons with three seeds. The [per-run table](control_per_run.csv), [condition summaries](control_by_condition.csv), and [canonical declared contrast summary](declared_paired_contrasts_summary.csv) retain the values, counts, protocols, and source provenance. Contrasts use the StudySpec's original selectors and the shared pairing engine; unavailable metrics remain missing.

- **Prospective success:** hits divided by attempts in the saved prospective command-event buffers. This is not option-success rate, checkpoint-stored confidence/attempts, or necessarily arrival at a unique physical location. The underlying HIT/FIRST definitions differ by condition.
- **Graph reachability:** fraction of ordered distinct DG-node pairs connected by a directed path in the reliable graph. Reliability uses each saved run's thresholds. It does not measure policy execution of every path.
- **Grounded score:** $G=S_{prospective}f_{mono\ endpoints}$, where $f_{mono\ endpoints}$ is the fraction of reliable edges whose two endpoints pass the canonical monofield criterion with valid peak coordinates. A zero can reflect no qualifying endpoint pair. A high value does not by itself establish physical target-specific control.
- **Frozen stored graph:** matrices from confidence and attempt buffers are labeled *stored confidence / attempts*. Frozen archives without prospective counters do not acquire a prospective score or a zero success estimate.

In bounded DGP command probes, FIRST versus HIT has executed-minus-shuffled success **+24.5 versus +15.2 percentage points**, but initial-action sensitivity is only **0.00123 versus 0.00359**. Complete ordered-pair coverage is **35.7% versus 78.5%**. Saturday ARR versus SRC command advantages are **+20.7 versus +22.1 points**, with only **1.9% versus 8.1%** of pairs complete. Selection and bounded trial coverage limit these outcomes; pair them with graph and field panels rather than claiming exhaustive controllability. [Original per-seed command records](command_evidence_per_run.csv).

## Trajectories and the full diagnostic stack

**33 run/protocol analyses are rendered here:** 12 CPD terminal frozen probes, 12 CPD retained training windows, three original C15 terminal frozen probes, and six DPR seed-99 frozen probes at unequal ages. Every row in the [diagnostic index](run_index.md) links its editable full and first-segment trajectory, occupancy, flow, four DG fields, all-active peaks, strict mono peaks, DG kernel, available graph matrices, and original local input files.

| Group | Editable behavior summary | Sampling |
| --- | --- | --- |
| CPD terminal | [SVG](cpd_frozen_10k_policy_probe_behavior.svg) | Three seeds × four conditions, 10,001 recorded observations per probe |
| CPD developmental | [SVG](cpd_online_100k_training_window_behavior.svg) | Three seeds × four conditions, retained 100k-observation windows |
| Original C15 | [SVG](corrected_core_frozen_10k_policy_probe_behavior.svg) | Three terminal frozen-policy probes |
| DPR exploratory | [SVG](dpr_frozen_unmatched_age_probe_behavior.svg) | Seed 99 only; 73,826,304–75,038,720 frames, not a matched architecture contrast |

| CPD terminal condition | Visited arena bins | 20-step mobile return | 40-step mobile return |
| --- | ---: | ---: | ---: |
| BASE | 69.3% | 65.1% | 34.5% |
| GATE ACT DIR | 69.3% | 62.4% | 26.0% |
| GATE ACT DIR GOAL | 55.3% | 83.2% | 42.1% |
| ADD CA3 DIR | 61.5% | 86.0% | 43.3% |

The looping definition matches the C01/C05/C15 comparison: after 20 or 40 decisions, displacement below 100 DMLab units after travelling more than 500 units, within one contiguous episode/segment. Rates use mobile windows; [trajectory metrics](trajectory_per_run.csv) also preserve the all-window rates, mobile fractions, and eligible-window counts. Reset connections never enter path lines or loop denominators. Training-window segments may be shorter than physical episodes; their rates stay separate from frozen probe rates. Coverage divides by all 361 coarse arena bins.

DG field maps are occupancy-normalized mean activation. Each of the four panels uses **0 to its own unit maximum**, with four separate colorbars; units are selected by the existing amplitude-weighted spatial score. All-active peak maps use the strongest original unsmoothed activation bin of each active unit. Strict mono maps retain the established eligibility and 80% dominant-component mass at 30/50/70% thresholds. DG kernels reuse the original displacement-correlation method, radius eight bins, at least five observations per cell and ten eligible cell pairs per offset. These policy-driven kernels are not fixed-history or downstream-layer comparisons; [existing layerwise results](../exemplar_gallery/historical_extension/report.md) remain the source for fixed-history DG/CA3/decoder contrasts.

**Event-marker limitation:** These archived paths lack aligned option-reset and target-hit flags. No goal-hit markers are inferred from bends or peak locations. The existing [C15 marked replay](../flat_goal_comparison/c15_seed99/trajectory_option_events_full.svg) remains a separate stochastic replay, with any current option-target hit and no planned waypoints. CPD's issued-goal summaries in the historical report are also distinct from actual target-hit events.

## Availability and remaining acquisition

The scalar/control survey is available for 66 conditions. Full raw trajectory/field inputs are currently local for the 33 rows above. **Full DGP/Saturday/source-credit trajectory galleries, the other 23 CPD conditions, and C14/C16 frozen paths are not rendered in this batch because their raw inputs are not locally staged.** Existing selected DGP/Saturday figures and frozen graph summaries remain linked from the [earlier gallery](../exemplar_gallery/selection_summary.md); they do not substitute for new SVG trajectories.

NEMO2 rejected automated authentication. A manual `ssh nemo2` login with OTP is needed before acquiring missing archived files. No new training or evaluator jobs are required for already saved telemetry. [Run availability](run_availability.csv) lists every declared run and its available evidence. Missingness is explicit, including Persistent C15's absent measurements. The local raw cache, hashes, original cluster paths where recorded, and pose provenance are in [the trajectory table](trajectory_per_run.csv).

## Sources and reproduction

[Original-file source index](source_index.md) provides clickable links to original files and pinned copies.
[Input provenance](input_provenance.json) records hashes and pinned copies of the original control CSVs and per-seed command JSONs. [Variant registry](variant_registry.csv) binds factors to the validated study schema, workflow version, and study hash. The [complete catalogue](variant_catalogue.md) links original study definitions; [run index](run_index.md) links raw local NPZs and pose files. The original C01/C05/C15 report is [here](../flat_goal_comparison/report.md).

Reproduce the survey and figures using `06_experiments/render_c15_variants.py` in the `SF_git` environment, with the repository as `--input-root`, the staged `06_experiments/data/c15_variants_20260928/raw` as `--raw-roots`, and this directory as `--output`. `--catalogue-only` updates control tables/plots without rerendering trajectories. The adapter reuses validated StudySpecs, canonical declared contrasts, graph outcome rendering, the C01/C05/C15 trajectory/field/peak helpers, established flow calculations, and the existing DG kernel calculation. No raw data or training state is changed.

**Reusable experience:** Saved canonical exports made the full variant survey inexpensive. Keeping online, scalar, frozen, and bounded-command protocols separate prevented missing prospective buffers from being reported as zero success. A clean clone exposed eager environment imports in an array-only kernel routine and a missing graph label argument; lazy replay imports and compatible explicit SVG/estimand options now fix those reusable paths. Check raw availability before promising a full gallery, retain original source hashes, and verify delivery in the user's actual local folder as well as the pushed checkout.
