# Reused frozen DG fields and stored graph

These outputs reuse the certified common-observation-panel archive for C05 seed 99 at 100,007,936 frames. No environment, actor, or new rollout was executed. The raw NPZ remains outside the repository; its SHA-256 is `a7848b1e8ed0cd526bd87c90d41f348af4008c7f1a8074a51801756df2dc3eb7`.

## Complete unit summary

The panel contains 10,001 observation samples and visits 242/361 spatial bins. All 16 units are active and satisfy the saved field-eligibility rule; none is silent. There are 16 distinct unsmoothed peak bins and active-map cosine .02527. **Mono-fields are 0/16 eligible units and 0/16 total units.** Distinct peaks and low map overlap do not establish single-field coding.

[All 16 unit rows](units.csv) retain activity counts/fractions, occupancy-weighted mean activity, spatial score, unsmoothed and dominant-component peaks, component counts at all three saved thresholds, and mono-field eligibility. The mean amplitude-weighted spatial score is .09487. This canonical score retains activity amplitude; it is **not bits per activation**.

## Selected maps

Units 0, 4, 8, and 12 were selected by index before the main control outcomes. Each figure has two vertical panels: saved post-inhibition mean activity and saved pre-threshold mean logit. There is no additional smoothing or per-unit normalization. Limits are shared across all 16 units: activity 0–17.0196; logits −1.6589–18.2105, with the diverging logit scale centered at zero. Gray bins were unvisited, not inactive. Coordinates are DMLab world units.

| Unit | PNG | Editable SVG |
| --- | --- | --- |
| 0 | [DG 00](dg_00.png) | [DG 00 vector](dg_00.svg) |
| 4 | [DG 04](dg_04.png) | [DG 04 vector](dg_04.svg) |
| 8 | [DG 08](dg_08.png) | [DG 08 vector](dg_08.svg) |
| 12 | [DG 12](dg_12.png) | [DG 12 vector](dg_12.svg) |

Each PNG is 1000×1900 pixels, with minimum 30-point text at 100 DPI (approximately 42 pixels). The exact scalable DejaVu Sans file and its hash are recorded. All four exports were inspected at their intended width; labels remain legible without clipping or overlap. [Visual QA receipt](visual_qa.json).

## Stored graph, separate from executed control

[All 256 directed entries](stored_graph.csv) preserve the three original graph arrays, including all 16 diagonal entries and 240 off-diagonal possibilities. The exact donor source confirms that `control_attempts` stores decayed completed-attempt mass, `control_edge_confidence` stores decayed successful-option mass, and `control_tctrl` stores the success-weighted mean completion time in decisions. Confidence reaches 1061.5; **it is not a probability**.

The canonical descriptive ratio is $(\mathrm{success\ mass}+1)/(\mathrm{attempt\ mass}+2)$. Using the canonical .5 success-mass and .5 ratio thresholds gives 15/240 directed edges, largest weak component 10, largest strong component 1, and 16/240 graph-reachable ordered pairs. These are frozen pretrained estimates, not current first-hit success or independently validated controllability. Prospective outcome arrays are absent and remain unavailable, rather than being replaced by zero success.

[Protocol and source mapping](protocol.json), [complete summary](summary.json), and archived sources preserve the exact hashes, axes, thresholds, and units. The donor graph source SHA is `d36dde771c78ecd5d19b91d91c17550b332412e823b2b1d01b3f916b53c58efe`, independently checked against the actual remote donor file. The renderer reuses canonical field-diversity and graph calculations; two focused tests verify axis masking and complete graph denominators.
