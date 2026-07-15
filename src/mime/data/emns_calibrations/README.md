# eMNS calibrations (MPEM)

Calibrated Multipole Electromagnet Model (MPEM) parameters for real electromagnetic
navigation systems, consumed by `mime.nodes.actuation.coil_array_field.CoilArrayFieldNode`
via `load_mpem_calibration()`.

These are **third-party data files**, redistributed unmodified except where noted below.

## Provenance and licence

| File | System | Coils | Upstream | Upstream licence |
|------|--------|-------|----------|------------------|
| `navion_2.yaml` | Navion_2 (3-coil planar) | 3 | [ethz-msrl/Workspace_eMNS](https://github.com/ethz-msrl/Workspace_eMNS) `models/Navion_2_Calibration_24-02-2020.yaml` | Apache-2.0 (repository LICENSE) |
| `octomag.yaml` | OctoMag (8-coil) | 8 | [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public) `mag_control/mpem/cal/OctoMag_Calibration.yaml` | Apache-2.0 (per-file header, retained) |
| `cardiomag.yaml` | CardioMag Jun-2015, no-offset variant | 8 | [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public) `mag_control/mpem/cal/C_Mag_Calibration_06-25-2015_no_offset.yaml` | Apache-2.0 (per-file header, retained) |

Copyright 2018 Multi-Scale Robotics Lab, ETH Zurich (Prof. Bradley J. Nelson).
Licensed under the Apache Licence, Version 2.0. See <http://www.apache.org/licenses/LICENSE-2.0>.

MIME is LGPL-3.0-or-later. Apache-2.0 is one-way compatible with LGPL-3.0-or-later, so
redistribution here is permitted. (It is **not** compatible with LGPL-2.1 — see the note
in the repository's licence file.)

### Changes made to the upstream files (Apache-2.0 §4(b))

* `navion_2.yaml` — a trailing newline was appended. Content otherwise byte-identical to
  upstream. No licence header was present upstream; the file inherits Apache-2.0 from the
  Workspace_eMNS repository LICENSE.
* `octomag.yaml`, `cardiomag.yaml` — copied verbatim, upstream Apache-2.0 headers retained.

### Why Navion comes from Workspace_eMNS, not Tesla_core_public

The same `Navion_2_Calibration_24-02-2020.yaml` appears in both repositories with
**numerically identical** contents (verified: the two differ only in `System_Name`
(`Navion_2` vs `Navion`) and a leading `---` document marker). Tesla_core_public's
top-level licence is **BSD-4-clause**, whose advertising clause is GPL-incompatible and
therefore unusable here; the Workspace_eMNS copy is Apache-2.0. We take the Apache-2.0
copy. The `octomag.yaml` and `cardiomag.yaml` files carry their *own* Apache-2.0
headers, which govern over the Tesla_core_public repository default.

**Do not vendor `Tesla_core_public/mag_control/mpem/src/scalorPotential.cpp`** or any other
BSD-4-clause source from that repository. The MPEM evaluator in MIME is an independent
reimplementation from the published formulation (Petruska, Edelmann & Nelson 2017;
Charreyron et al. 2021).

## Format

Each file is the upstream MPEM YAML schema:

```yaml
System_Name: Navion_2
Coil_List: [Coil_0, Coil_1, Coil_2]
Coil_0:
  Source_List: [Src_0, Src_1, Src_2]
  Src_0:
    A_Coeff: ~                    # interior (solid-harmonic) coefficients; null in all shipped models
    B_Coeff: [-0.000777105750805] # exterior coefficients, B_Coeff[i] has order l = i+1
    Source_Direction: [...]       # 3-vector, NOT guaranteed unit-normalised
    Source_Position:  [...]       # 3-vector, metres
Workspace_Dimensions: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
```

Units are strict SI throughout: positions in **m**, currents in **A**, field in **T**.
`B_Coeff` has units T·m³/A. (mT appears only in upstream plotting code.)

## Notes that matter for the implementation

* **Every shipped calibration is order 1 only** (`A_Coeff: ~`, `len(B_Coeff) == 1`). An
  order-1 exterior term is exactly a point dipole, so in practice these models are a sum of
  3–8 fitted point dipoles per coil. `octomag.yaml` says so in its own name:
  `System_Name: OctoMag_Calibration_Order_1`.
* `Source_Position` is a **fitted** multipole location, not a physical coil centre. Navion
  coils 0 and 1 each contain near-cancelling dipole pairs, so the net moment is ~100×
  smaller than the individual terms. Treat sources as basis functions, not hardware.
* `Source_Direction` is **not** normalised upstream; normalise on load.
* The upstream C++ `getGradient()` returns the *field* `B = -grad(Phi)`, not the gradient.
  Do not be misled when cross-checking against `mag_manip`.
* Gradient 5-vector packing upstream is `[dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz]`, with
  `dBz/dz` recovered by the trace-free closure.
* `Workspace_Dimensions` is a **declared region of interest, NOT a fit-validity box** —
  despite the obvious reading. Upstream never fits it: `calibrate()` in
  `electromagnet_calibration.cpp` never writes `workSpace` (only the constructor and the
  YAML loader do); the hand-written fitter input `cal_ic_files/Navion_IC.yaml` declares it
  **unbounded** (±99e99); all four shipped Navion calibrations carry a *byte-identical* box
  despite being different machines fitted on different dates; and the synthetic `mns_*`
  models declare a round ±0.1 m with no fit at all. Being inside it is not evidence of
  accuracy; being outside it is not evidence of extrapolation. The true fit domain of the
  shipped models is not published. (MPEM is separately an exterior expansion and does
  diverge as `r -> 0` at a source, but sources sit inside the iron cores.)
* `cardiomag.yaml` is the **no-offset** variant (8 coils). The upstream offset variant has a
  9th `Offset` block of current-independent sources modelling remanent/ambient field, which
  the upstream forward model ignores unless explicitly enabled. Zero current gives exactly
  zero field here.

## Regression oracle

`tests/data/emns_mpem_oracle.npz` holds reference actuation matrices `A_b(p)` and gradient-5
vectors `G5(p, i)` for **all three** systems (Navion_2, CardioMag, OctoMag) at 50
in-workspace positions each, generated from the upstream `mag_manip` 3.0.0 library. Used by
MIME-VER-140 to pin our reimplementation against upstream. `mag_manip` itself is **not** a
MIME dependency: its wheel is built against the numpy 1.x ABI and hard-fails on numpy >= 2
(MIME is on numpy 2.4), so the oracle is generated once in an isolated environment and
checked in.

Note `computeFieldGradient5FromCurrents` returns the stacked **8-vector** `[B(3); G5(5)]`,
not a 5-vector — verified against `computeFieldFromCurrents`. Take the trailing 5.

### Agreement is conditioning-limited, not implementation-limited

Two independent implementations of the same algebra sum the same near-cancelling source
contributions in different orders, so they cannot agree better than
`eps / field_cancellation_ratio`. Measured against `mag_manip` in float64:

| model | field cancellation | predicted floor | measured |
|---|---|---|---|
| `cardiomag` | 2.5e-1 | 8.8e-16 | 8.6e-16 |
| `navion_2` | 8.0e-3 | 2.8e-14 | 4.0e-14 |
| `octomag` | 6.9e-5 | 3.2e-12 | 1.5e-12 |

The prediction tracks measurement to ~1.5× across a 3600× spread in conditioning, which is
why MIME-VER-140's tolerance is derived per-system rather than pinned flat. OctoMag's
larger disagreement is arithmetic, not error: it is the worst-conditioned fit shipped
(sources ~245 km from a 2 cm workspace), and the same ratio predicts its float32 behaviour
— which is why `CoilArrayFieldNode` refuses to construct it in single precision.
