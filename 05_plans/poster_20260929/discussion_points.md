# Visitor discussion points and DG-capacity review

The main poster keeps the C01/C05/C15 story and two DG-16 survey panels. The points below are conversation material, with existing measurements and limits stated separately from hypotheses. No new training or probes were run.

## 1. What makes a spatial landmark useful?

At DG 16, 168 online runs (56 variants, three run rows each) give spatial-score/target-hit $\rho=0.583$ and distinct-peak/hit $\rho=-0.424$. Within the largest families, score/hit is only $0.094$ in CPD (81 runs) and $-0.001$ in DGP (24). Peak/hit is also weaker within CPD ($-0.130$) and DGP ($-0.239$).

**Visitor question:** Does landmark quality mean a high spatial score, many distinct peaks, compact fields, or goals that reliably change the reached location?

**Interpretation:** The coding metrics measure different things. More peak locations do not imply more target-event success in this survey. These are descriptive associations; neither diversity's negative correlation nor score's positive correlation establishes causation. Spatial score includes activation amplitude, and peak bins can belong to broad or multifield units.

Source: [selected point table](architecture_scatter_points.csv), [exact statistics](architecture_scatter_statistics.json), and [original survey metric definitions](../../06_experiments/results/A0_poster_analysis_20260926/cross_run_scatter/report.md). The main legend labels the fixed-DG-16 DGC subset as “Goal interfaces”: the retained designs are DIRECT DG, DIRECT WORKER, and WAYPOINT DG. No DG-capacity difference is pooled into that family in the poster.

## 2. Can better prediction accompany worse exploration?

In CPD GATE ACT DIR, adding the goal predictor changes frozen-probe means as follows at the same 75,038,720-frame checkpoint age:

| Measurement | Without goal predictor | With goal predictor |
| --- | ---: | ---: |
| 20-decision returns among mobile windows | 62.4% | 83.2% |
| Visited arena bins / all bins | 69.3% | 55.3% |

Returns increase and visited coverage decreases in **each of the three paired training seeds**. These probes match age, environment, and length, not stochastic starting states or action histories.

**Visitor question:** Can prediction stabilize familiar cycles rather than encourage the discovery of new transitions?

**Interpretation:** The predictor variant performs worse on these two measurements. Reinforcing easy cycles is a hypothesis, not an identified cause. This is a goal-predictor factor in an already goal-conditioned controller, not evidence against goal conditioning in general.

Source: [per-seed trajectories](../../06_experiments/results/A0_poster_analysis_20260926/c15_variants/trajectory_per_run.csv), filtered to `CPD_C15_GATE_ACT_DIR` and `CPD_C15_GATE_ACT_DIR_GOAL`, `frozen_10k_policy_probe`; [variant report](../../06_experiments/results/A0_poster_analysis_20260926/c15_variants/report.md#what-to-put-in-the-poster).

## 3. Is a connected internal graph an executable map?

The selected DGP FIRST/HIT JOINT LEG designs have opposite orderings at the shared 75,005,952-frame online window:

| Measurement | FIRST | HIT |
| --- | ---: | ---: |
| Recorded prospective successes / attempts | 58.9% | 50.8% |
| Stored-graph reachable pairs | 4.7% | 96.1% |

FIRST has more recorded success and much lower reachability in every paired seed.

**Visitor question:** Which graph edges represent actions the agent can execute, rather than event relationships it has stored?

**Interpretation:** Stored connectivity and event outcomes are distinct diagnostics. FIRST/HIT also change the worker-outcome rule, so this is not a common-definition causal success comparison or proof of physical target arrival. The grounded-score advantage is concentrated in one seed; do not present it as replicated grounding.

Source: [per-seed control records](../../06_experiments/results/A0_poster_analysis_20260926/c15_variants/control_per_run.csv), filtered to `DGP_C15_FIRST_JOINT_LEG` and `DGP_C15_HIT_JOINT_LEG`, `online_100k_training_window`; [declared contrasts](../../06_experiments/results/A0_poster_analysis_20260926/c15_variants/declared_paired_contrasts_per_seed.csv).

## 4. Must each internal goal correspond to one compact field?

Across the matched C01/C05/C15 terminal probes, only **4 of 144 DG units** satisfy the shared strict mono-field criterion: 1/48 in C01, 2/48 in C05, and 1/48 in C15. C15's coverage advantage therefore does not coincide with a replicated increase in compact single-field units. Counts depend on the declared criterion and the sampled policy trajectories.

**Visitor question:** Could population sequence context disambiguate repeated DG events when individual units have broad or multiple fields?

**Next test:** From matched physical and memory states, change DG identity and context separately, then measure the reached landmark. Context is a candidate, not an established fix; the existing goal-predictor result cautions against assuming extra representation machinery helps.

Source: [matched mono-field counts](../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/mono_field_peak_counts.csv) and [classification method](../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/mono_field_method.json).

## Why the main poster keeps DG 16

The [separate capacity audit](dg_capacity_audit.csv) is regenerated from the pinned survey and retains pooled/within-family estimates by protocol and capacity.

| Eligible online cohort | Runs with score and target counters | Score/hit Spearman $\rho$ |
| --- | ---: | ---: |
| DG 16 | 168 | 0.583 |
| DG 32 | 9 | -0.150 |
| DG 64 | 57 | 0.070 |

DG 32 comes from one study family. DG 64 combines CA3-state, controller, capacity, and transfer families at heterogeneous ages. Within DG 64, score/hit is positive in controllers ($0.580$, 12 runs), the capacity study ($0.700$, 9), and five-cue transfer ($0.678$, 12), while CA3-state is weaker ($0.119$, 24). That grouping sensitivity is interesting conversation material, but does not establish a DG-capacity effect or a clean additional poster result. The capacity study itself contains three DG-64 rows at 75M while its DG-16/32 rows are all at 150M; even that pooled family is not fully age-matched.

Filtering also overturns the earlier replay-score/coverage null: mixed capacities gave $\rho=-0.0003$ (38 runs), DG 16 gives $0.321$ (18), and DG 64 gives $-0.475$ (20). These cohorts differ in family and common-history panel, so the opposite signs cannot be interpreted as a capacity-induced reversal. DG-16 replay score versus held-out command advantage is $-0.226$ across 18 runs, still a small mixed-family screen. The old null wording was removed from the poster.

**Reusable lesson:** Fix capacity before comparing architecture families, recalculate every displayed association after filtering, and distinguish missing command counters from zero success. For poster discussion candidates, read the original per-seed table and verify direction in each paired seed before quoting a three-seed mean.
