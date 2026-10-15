# Perlin lower-map strength trials

[Comparison preview](comparison.png) · [editable comparison](comparison.svg) · [print comparison](comparison.pdf) · [manifest](manifest.json)

These four **10 × 20 mm** icons vary only the contrast of the lower Perlin field. They share the same upper field, DG red, noise seed, spatial pattern, and mean lower-map tint. The numbers below are rendering amplitudes, not measured spatial-information values. **Current** matches the existing [Perlin DG map candidate](../05_perlin_dg_maps.svg).

| Level | Amplitude | Editable SVG | Print PDF |
| --- | ---: | --- | --- |
| Subtle | 0.045 | [SVG](01_subtle.svg) | [PDF](01_subtle.pdf) |
| Medium | 0.070 | [SVG](02_medium.svg) | [PDF](02_medium.pdf) |
| Current | 0.095 | [SVG](03_current.svg) | [PDF](03_current.pdf) |
| Strong | 0.110 | [SVG](04_strong.svg) | [PDF](04_strong.pdf) |

The comparison shows each icon enlarged and at its actual page size. All PDFs contain vector artwork. Rebuild from the parent directory with `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python build_spatial_information_icons.py`.
