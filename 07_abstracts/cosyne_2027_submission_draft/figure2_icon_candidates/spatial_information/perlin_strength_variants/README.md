# Perlin lower-map strength trials

[Original range](comparison.png) · [stronger range](stronger_comparison.png) · [editable stronger comparison](stronger_comparison.svg) · [print stronger comparison](stronger_comparison.pdf) · [manifest](manifest.json)

These seven **10 × 20 mm** icons vary only the contrast of the lower Perlin field. They share the same upper field on a white background, DG red, noise seed, spatial pattern, and mean lower-map tint. Every lower-map cell gets a uniform **0.1 red-opacity tint**, so the background never becomes white. The numbers below are rendering amplitudes, not measured spatial-information values. At higher strengths, the spatially varying component is clipped and offset to retain its mean; the added tint makes the lower map's overall mean higher than the upper map's. **Current** matches the existing [Perlin DG map candidate](../05_perlin_dg_maps.svg).

| Level | Amplitude | Editable SVG | Print PDF |
| --- | ---: | --- | --- |
| Subtle | 0.045 | [SVG](01_subtle.svg) | [PDF](01_subtle.pdf) |
| Medium | 0.070 | [SVG](02_medium.svg) | [PDF](02_medium.pdf) |
| Current | 0.095 | [SVG](03_current.svg) | [PDF](03_current.pdf) |
| Strong | 0.110 | [SVG](04_strong.svg) | [PDF](04_strong.pdf) |
| High | 0.140 | [SVG](05_high.svg) | [PDF](05_high.pdf) |
| Higher | 0.200 | [SVG](06_higher.svg) | [PDF](06_higher.pdf) |
| Very high | 0.280 | [SVG](07_very_high.svg) | [PDF](07_very_high.pdf) |

Each comparison shows its icons enlarged and at actual page size. The stronger comparison includes **Current** as a reference. All PDFs contain vector artwork. Rebuild from the parent directory with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_spatial_information_icons.py`.
