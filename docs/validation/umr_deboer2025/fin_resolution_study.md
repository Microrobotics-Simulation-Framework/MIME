# Fin Resolution Sensitivity Study

Pre-T2.6 gate check: UMR fin resolution at candidate LBM grid spacings.

## Physical geometry

- Vessel diameter (domain): 9.4 mm
- Body radius: 0.87 mm, length: 4.1 mm
- Fin outer radius: 1.42 mm
- Fin length: 2.03 mm, width (circ. arc): 0.55 mm, thickness (radial overlap): 0.15 mm
- Helix pitch: 8.0 mm (MIME-ANO-002: assumed)

## Key geometry insight

The 6 fins (2 sets x 3) each span 2.03mm axially in a 4.1mm body.
Fin spacing = 0.68mm < fin length = 2.03mm,
so fins overlap axially. The correct distinguishability metric is
angular bands per z-cross-section (not z-bands).

The thinnest free-standing fin dimension in the code is the circumferential
arc length = fin_width = 0.55mm at the body surface.
The `fin_thickness` parameter (0.15mm) controls a radial overlap
zone with the body, not a free-standing thin feature.

## Resolution comparison

| Resolution | dx (mm) | Circ. arc (lu) | Radial (lu) | Angular bands | Fin nodes | Solid % |
|---|---|---|---|---|---|---|
| 64x64x82 | 0.1469 | 1.4 | 2.5 | 3 | 428 | 1.233% |
| 96x96x124 | 0.0979 | 2.0 | 5.5 | 3 | 1395 | 1.195% |
| 128x128x164 | 0.0734 | 2.6 | 6.0 | 3 | 3469 | 1.227% |
| 192x192x246 | 0.0490 | 4.1 | 10.0 | 3 | 11265 | 1.236% |

**Minimum viable resolution**: 128^3 (dx = 0.0734 mm, circ. arc = 2.6 lu, radial = 6.0 lu)

## Parameter interpretation note

The code uses `fin_width` (0.55mm) as circumferential arc and `fin_thickness`
(0.15mm) as a radial overlap with the body. The paper's "fin thickness" likely
refers to the blade thickness (circumferential), which would make the fin thinner
than the code implements. If the interpretation is revised so that the circumferential
extent = 0.15mm instead of 0.55mm, the resolution requirements increase significantly
(0.15mm / dx at 128^3 = 2.0 lu — marginal). This should be resolved before T2.6.

