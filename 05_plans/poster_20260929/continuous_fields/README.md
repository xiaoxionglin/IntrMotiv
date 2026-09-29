# Continuous place-field candidates

Use **01** to replace the binary mono-field count panel, and **03** for the
exploration/control scatter. **04** broadens the architectural comparison using
saved continuous component-dominance scores. **02** supplies C01/C05/C15 behavior
context. **07** is the preferred compact command result, using all 12 saved
models; **06** shows their continuous field-dominance comparison. **05** remains
a seed-99 illustration and should not carry a general trend claim. Everything
here can be regenerated locally without NEMO. The existing posters have not been edited.

Plots use editable SVG text at **30 pt**, scatter markers **78 pt²**, and the
existing physical canvases: **191.5 × 76.2 mm** for the compact pair,
**383 × 76.2 mm** for a row of two scatters, and **383 × 165.1 mm** for four
scatter panels. Selection-sheet captions are **40 pt**.
Import at 100% physical size; shrinking a panel also shrinks its text.

![Main candidates](gallery_continuous_1.png)

| Candidate | Editable figure | Use |
| --- | --- | --- |
| 01 Continuous field structure | [SVG](01_continuous_fields_core.svg) | Compact alternative to strict counts |
| 02 Core field structure and behavior | [SVG](02_core_concentration_and_behavior.svg) | Main-story discussion |
| 03 Concentration, exploration, control | [SVG](03_concentration_exploration_control.svg) | Preferred continuous scatter |
| 04 Dominance across variants | [SVG](04_dominance_across_variants.svg) | Wider available survey |
| 05 Matched command examples | [SVG](05_concentration_command_comparisons.svg) | Exploratory supplement |
| 06 All saved seeds: dominance and commands | [SVG](06_dominance_command_all_saved_seeds.svg) | Replicated discussion, two variants per family |
| 07 Seed-paired command lift | [SVG](07_command_lift_seed_pairs.svg) | Preferred positive command result |

## What the measures mean

**Spatial concentration, $C$:** normalize each unit’s unsmoothed,
occupancy-corrected rate map to equal-weight rate mass over sampled bins.
For $N$ sampled bins and $p_b=r_b/\sum_b r_b$,

$$
A_{\mathrm{eff}}=\frac{1}{\sum_b p_b^2},\qquad
C=\frac{N-A_{\mathrm{eff}}}{N-1}.
$$

Effective area, $A_{\mathrm{eff}}$, is the number of equally active bin equivalents.
$C=0$ means uniform activity across the sampled arena; $C=1$ means all mass in
one bin. Multiplying a unit’s activity by any positive gain leaves it unchanged.
This avoids the amplitude dependence of the existing spatial score.

**Single-field dominance, $D$:** the existing continuous `field_mono_score`,
before the $D\geq0.8$ classification. At 30%, 50%, and 70% of the smoothed peak,
find connected components; take the largest component’s share of superlevel
rate mass, then the minimum across the three thresholds. It distinguishes one
dominant component from several competing ones. It still uses peak thresholds,
but the output is continuous and needs no mono-field pass/fail cutoff.

**Mean-threshold dominance, $\bar D$ (candidate 06):** the existing replay
summary `mean_dominant_component_mass`. This averages the dominant-component
share across all three thresholds and eligible units, instead of taking the
minimum threshold share per unit. It is a continuous connected-field measure,
but it is **not $D$ and not spatial concentration $C$**. The mean is available
for 12 saved replay summaries; local raw maps support $C$ and minimum $D$ for
only four of those models. Missing minimum scores cannot be recovered from the mean.

Use the two together. A broad connected field can have high $D$; several small,
distant peaks can have high $C$. $C$ is spatial activity concentration, not a
geometric radius or a guarantee of one contiguous place field. Neither measure
captures diversity across DG units; keep peak diversity or map cosine as a
separate population diagnostic.

Entropy concentration and 80%-mass area are retained as alternative summaries
in the data tables. They answer related questions with different tail sensitivity.
They are not independent evidence for the same claim. The computation and
support rules are documented in the
[canonical telemetry guide](../../../04_implementation/reusable_place_field_telemetry.md#continuous-field-concentration-saved-map-analysis).

## What the data support

For C01/C05/C15, each number below first averages eligible DG units within a
run, then averages three training seeds. Models are at 100,040,704 frames and
maps are archived 10k frozen-policy probes. Probe histories differ by policy.

| Measure | C01 | C05 | C15 |
| --- | ---: | ---: | ---: |
| Spatial concentration $C$ | 0.925 | 0.918 | 0.877 |
| Effective area / sampled bins | 7.9% | 8.7% | 12.6% |
| Area containing 80% of rate mass / sampled bins | 6.8% | 7.6% | 11.2% |
| Single-field dominance $D$ | 0.372 | 0.482 | 0.373 |
| Final-window training coverage AUC | 37.97 | 42.03 | 78.57 |

C15 is less concentrated than both C01 and C05 in **all three seed comparisons**.
It does not improve mean dominance over C01. C05 has higher dominance than C15
in each of the three seeds. These statements concern the selected probe
measurements; they do not establish equivalence of representations.

The strongest poster wording is:

> **C15 broadens exploration without sharpening DG fields.**

The continuous measurements make this statement more informative than “only
four units pass a mono-field criterion.” They do not justify “localized fields
are unnecessary” across tasks or a causal claim that broader fields improve
exploration.

In the four locally available CA3-feedback variants (12 seeds, common
75,005,952-frame online age):

| Association | Concentration $C$ | Dominance $D$ |
| --- | ---: | ---: |
| Visited grid fraction | $\rho=-0.772$ | $\rho=-0.616$ |
| Recorded target-event hit fraction | $\rho=-0.238$ | $\rho=-0.266$ |

These are descriptive Spearman coefficients across seed-level run means.
Coverage is near the ceiling: **85.3–88.1%** of the total 19 × 19 grid, so the
coverage plot retains its full 0–1 y axis. $C$ axes use the disclosed 0.65–1
detail range. Maps describe a recent 100k training window; target counters
accumulate historical attempts. Internal DG target hits are not verified
physical arrival or downstream reward success.

The available algorithm screen has six variants and 18 seeds for dominance
versus coverage ($\rho=0.258$), and four goal variants / 12 seeds for dominance
versus target hits ($\rho=-0.322$). The two worker-reference variants have no
target-hit counters. Missing counters are omitted from that panel, not treated
as failed commands. Shapes identify variants within each family: CPD baseline
circle, CA3 feedback square, goal context triangle, predictor diamond; algorithm
screen CA3 circle, DG-policy square, arrival-credit triangle, source-credit
diamond, worker joint-gradient filled plus, worker stopped-gradient cross.

The defensible discussion point is **field shape does not strongly order the
recorded target-hit rates in these available subsets**. This is weaker than
“field shape does not matter.” Four CPD architecture means give a concentration
versus hit rank coefficient of −0.8, but only four points; dominance gives +0.4.
The seed-level coefficients and three-seed architecture means both remain in
[correlations.csv](correlations.csv). Grouping is part of the interpretation.

## Matched-command results available offline

![Optional command comparisons](gallery_continuous_2.png)

All four variants have saved command outcomes and mean-threshold dominance
for seeds **8, 99 and 123**: **12 models**, all DG 16 at **75,038,720 frames**.
All their representations used the **same recorded 10,001-observation panel**,
`reduced_sat.npz`. The earlier statement that each family used a different
panel was incorrect; the raw-map metadata, survey and replay manifest agree
on a shared panel. Families remain separate because their controller designs
and evaluation decision caps differ (30k DGP, 20k Saturday).

### Preferred command claim: target specificity

[Candidate 07](07_command_lift_seed_pairs.svg) shows every seed, with gray
lines connecting matching seed IDs and black mean bars. No error bars or
significance marks are implied.

| Family | Variant | Mean command lift | Range across three seeds |
| --- | --- | ---: | ---: |
| DG policy | First outcome | +24.5 p.p. | +18.9 to +29.4 p.p. |
| DG policy | Target hit | +15.2 p.p. | +11.3 to +19.1 p.p. |
| Credit assignment | Source | +22.1 p.p. | +19.9 to +25.7 p.p. |
| Credit assignment | Arrival | +20.7 p.p. | +19.1 to +21.8 p.p. |

> **Executed commands show target specificity in all 12 evaluated models.**

All 12 observed lifts are positive relative to matched shuffled-command
success. First-outcome lift exceeds target-hit lift in all three seed pairs;
the mean difference is **9.3 p.p.** Source credit exceeds arrival credit in
two of three seeds, with a smaller **1.4 p.p.** mean difference. The latter is
a discussion point rather than a strong ordering.

Lift measures the saved intervention’s **internal DG target-event success**.
The shuffled comparator is selected retrospectively from completed trials;
it does not ensure identical physical starts. Ordered-pair coverage is
incomplete, and evaluation opportunity depends on source recognition. These
results describe command specificity, not verified physical arrival. Neither
contrast by itself isolates the effect of DG learning.

### Continuous field structure versus command lift

[Candidate 06](06_dominance_command_all_saved_seeds.svg) uses all 12 saved
mean-threshold dominance scores, six points per family. Descriptive Spearman
associations are **−0.31 for DG policy** and **+0.77 for credit assignment**.
Blue circles identify First or Source; orange squares identify Hit or Arrival.
These opposite signs do not support the common negative association suggested
by the seed-99 examples. With only two variants per family, the positive
credit-assignment association also cannot establish a general field–control rule.

The appropriate discussion question is **which representation properties
predict controllable transitions after architecture and seed are accounted for?**
For a poster with limited space, prefer 07 over 05; use 06 only if the field-shape
discussion is central. Do not pool the two score definitions or mix these
command outcomes with training-time target-hit counters.

### Seed-99 raw-map illustration

Candidate 05 uses only seed 99, where raw maps are locally available and both
$C$ and minimum-threshold $D$ can be computed:

| Pair | Variant | Concentration $C$ | Dominance $D$ | Executed minus shuffled success |
| --- | --- | ---: | ---: | ---: |
| DG policy | First outcome | 0.873 | 0.382 | +29.4 p.p. |
| DG policy | Target hit | 0.919 | 0.539 | +19.1 p.p. |
| Credit assignment | Source | 0.907 | 0.392 | +25.7 p.p. |
| Credit assignment | Arrival | 0.917 | 0.435 | +21.2 p.p. |

Within both available pairs, the more concentrated representation accompanies
the smaller matched-command benefit. **One seed per variant is insufficient for
a general trend claim.** The replicated mean-dominance comparison in 06 does
not reproduce a common negative ordering across both families. No fitted line,
rank coefficient, or significance claim is displayed in 05. A DG-16
CPU model at 300M also has concentration and command data in the tables, but
is omitted from these comparisons because its family and checkpoint age differ.

## Availability and sampling checks

There are **63 run/protocol means**: 30 online (12 CPD + 18 algorithm-screen),
28 frozen (15 corrected-core + 12 CPD + one CPU), and five common replay.
Raw maps support concentration for **45** of them; the additional 18 online
algorithm-screen means support continuous dominance only. Capacity is fixed at
DG 16. This is not the entire 168-run poster survey. The
[survey availability table](survey_availability.csv) also retains unavailable
DG-16 run/protocol rows, including excluded corridor geometries.

The additional [command summary table](command_dominance_per_run.csv) retains
12 exact replay/command joins separately from these 63 raw-map/unit-CSV means.
Its scores have a different threshold aggregation. Four seed-99 summary
scores were checked against local raw maps; the eight other seeds rely on the
saved derived CSV, without inventing unit scores or unavailable concentration.
Expanding to all eight DGP variants requires command probes for six variants
that have no saved matched-command outcomes. The broader CPD survey has
training counters but no matched-command outcomes; it cannot fill that gap.

Unknown bins are excluded. Rate maps already divide activity by visits; map
mass gives equal weight to spatial bins rather than dwell time. Shapes of
silent units are undefined. Means use units with at least 20 active observations
and activity in at least three bins, preserving the existing eligibility rule;
they do not select units based on mono-field status. Eligibility and silence
counts are exported for every run.

Restricting support to bins with at least five observations preserves CPD’s
online concentration ordering closely ($\rho=0.993$ between scores), with a
maximum run-mean change of 0.0082. The coverage association remains negative
($\rho=-0.725$) and target-hit association remains weak ($\rho=-0.273$).
Entropy concentration gives −0.694 and −0.273, respectively. These checks
support the interpretation within this selected online subset.

Frozen CPD maps do **not** reproduce the online coverage association: $C$
versus frozen visited coverage is $\rho=0.151$; entropy concentration gives
−0.354. Shorter, policy-specific probe paths and the choice of tail weighting
matter. Do not combine frozen and online maps into one universal concentration
trend. There is no fixed-trajectory causal control for the main exploration
scatter. The common-replay command comparisons provide better input comparability
and now include three independently trained seeds per variant; they still
cover only two selected variants per family.

## Sources, verification, and reuse

[per_unit.csv](per_unit.csv) retains unit values and raw-map or canonical-CSV
source identities. [per_run.csv](per_run.csv) retains checkpoint ages, protocols,
study fingerprints where available, eligibility counts, alternative measures,
and mean/median summaries. [plotted_points.csv](plotted_points.csv) maps exact
values to panels. [raw_map_availability.csv](raw_map_availability.csv) reports
duplicates, unavailable exact matches and capacity exclusions.
[command_dominance_per_run.csv](command_dominance_per_run.csv) retains the
12 summary joins, replay counts, eligibility, command trial coverage, model
identity and exact shared panel. [command_dominance_statistics.json](command_dominance_statistics.json)
records the descriptive associations and every seed-paired lift difference.
[manifest.json](manifest.json) fingerprints every selected source and records
formula, physical size, font, and package versions.
[quality_checks.json](quality_checks.json) verifies editable vectors, unique SVG
IDs, resolved references, typography, and that the poster remained unchanged.

Regenerate the figures from the repository root:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/render_field_concentration_20260929.py
```

Six focused scientific-invariant tests cover uniform/peak endpoints, gain
invariance, silent units, unknown-bin handling, geometric fragmentation, and
invalid inputs. The canonical study suite plus these tests passed **43 checks**.

**Reusable experience:** saved per-unit scores are authoritative for continuous
dominance, while effective area requires raw maps. Join by run/model age/protocol
and explicit policy, deduplicate identical maps, keep capacity fixed, and audit
measurement availability before making a cross-run claim. Frozen summary rows
omit policy ID; selected checkpoint paths verify policy 0. Large rotated labels
can exceed a compact physical canvas at 30 pt: use short axis names, captioned
definitions and one-line panel statistics, then inspect the final SVG render.
For offline command plots, first check existing derived summaries: missing
raw maps need not mean missing run-level measures. In replay exports, `frames`
counts observations; join model age through `checkpoint_frames`. Validate
metric aggregation and replay-panel identity explicitly, then retain the
average-threshold score under a distinct name from minimum-threshold dominance.
