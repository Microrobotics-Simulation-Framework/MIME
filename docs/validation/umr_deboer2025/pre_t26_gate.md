# Pre-T2.6 Gate: Fin Resolution and Stability

## Fin resolution study (DONE)

All 4 tested resolutions (64^3 through 192^3) produce distinguishable fins
with 3 angular bands per z-slice (matching 3 fins per set at 120 degree spacing).

| Resolution | Circ. arc (lu) | Radial extent (lu) | Angular bands |
|---|---|---|---|
| 64^3 | 5.1 | 3.1 | 3 |
| 96^3 | 7.3 | 5.3 | 3 |
| 128^3 | 9.7 | 6.8 | 3 |
| 192^3 | 14.7 | 11.1 | 3 |

**Minimum viable resolution**: 64^3 (circumferential arc = 5.1 lu, radial = 3.1 lu)

**Caveat**: The code interprets `fin_width` (0.55mm) as the circumferential extent.
If the paper's "fin thickness" (0.15mm) is actually the circumferential blade
thickness (which would make the fin 3.7x thinner), the minimum viable resolution
escalates to 128^3 (0.15mm / 0.073mm = 2.0 lu — marginal). This parameter
interpretation should be resolved before T2.6 accuracy validation.

## 128^3 stability check (DONE — PASS)

- **Setup**: 128x128x128, pipe wall (R=64 lu) + UMR body (no fins, no rotation, stationary), initial velocity 0.001 lu in z
- **Result**: 500 steps on RTX 2060 GPU, **no NaN, no Inf, density conserved to 0.006%**
- **Step time**: 1.6s per step at 128^3 (pipe + UMR + simple BB)
- **Drag on UMR**: -0.077 lattice units (opposing flow direction — correct sign)
- **Velocity**: u_max = 0.001 lu (stable, well below compressibility limit)

## Decision: proceed with 128^3

128^3 provides:
- 9.7 lu circumferential fin arc (well-resolved)
- 6.8 lu radial fin extent (well-resolved)
- Numerically stable combined pipe + UMR bounce-back
- 1.6s/step on RTX 2060 — feasible for moderate sweeps (~1000 steps x 4 ratios = ~1.8 hours)

**Risk**: If fin_thickness interpretation is revised to circumferential = 0.15mm,
128^3 gives only 2.0 lu — marginal. Would need 192^3 (3.1 lu) or adaptive
refinement near fins.

## Performance note for T2.6 planning

Measured step time at 128^3 on RTX 2060 GPU: **1.6s/step** (Python-loop LBM with
bounce-back, no Guo forcing). With Guo forcing, estimated ~2.5s/step due to
(128,128,128,19,3) intermediates.

For a confinement sweep:
- 4 confinement ratios x 1000 steps each = 4000 steps = **1.8 hours** (no force)
- With rotation (geometry recomputation each step): estimated **3-5 hours**
- On A100: estimated 10-20x faster → **10-30 minutes**
