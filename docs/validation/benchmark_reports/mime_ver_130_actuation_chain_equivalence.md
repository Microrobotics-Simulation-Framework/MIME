# MIME-VER-130 — Actuation-Chain Field Equivalence (Far-Field Limit)

**Date**: 2026-04-30
**Producer under test**: `Motor + PermanentMagnetNode` chain
**Reference producer**: `mime.nodes.actuation.external_magnetic_field.ExternalMagneticFieldNode`
**Algorithm IDs**: `MIME-NODE-100`, `MIME-NODE-101`, `MIME-NODE-001`
**Benchmark type**: Mode 1 (Wrapping equivalence)
**Test file**: `tests/verification/test_actuation_chain_equivalence.py::test_ver130_field_equivalence_far_field`
**Acceptance**: $|B_{\text{new}} - B_{\text{legacy}}| / |B_{\text{legacy}}| < 0.02$ over a full rotation period

---

## Goal

Demonstrate that the new Motor + PermanentMagnetNode chain reproduces
the rotating uniform field of the legacy `ExternalMagneticFieldNode`
in the configuration where the legacy uniform-field assumption is
physically valid: a rotating dipole far from the workspace, with the
rotation axis perpendicular to the line from magnet to UMR.

This benchmark establishes that the new chain is a **strict
generalisation** of the legacy node — anyone currently using the
legacy node can switch to the new chain in this regime without a
quantitative trust gap.

## Configuration

| Parameter | Value |
|---|---|
| Standoff $z$ | 0.05 m (50 mm; ≈ 50× magnet length) |
| Field amplitude $B_0$ | 1.2 mT (matches dejongh nominal) |
| Frequency $f$ | 10 Hz |
| Magnet geometry | $R = 1$ mm, $L = 2$ mm; cylindrical |
| Field model | `point_dipole` |
| Dipole moment $|m|$ | $B_0 \cdot 4\pi z^3 / \mu_0$ — chosen so the equatorial-plane field magnitude at the UMR matches $B_0$ |
| Earth field | 0 (apples-to-apples comparison) |
| Timestep $\Delta t$ | $10^{-4}$ s |
| Samples | 100 over one full period (100 ms) |

## Procedure

1. Build a standalone `ExternalMagneticFieldNode` driven at the matched
   $f$ and $B_0$.
2. Build a `MotorNode` in velocity-mode at $\omega = 2\pi f$ and a
   `PermanentMagnetNode` with the matched dipole moment, parented at
   $(0, 0, z)$ with axis $+\hat z$ and dipole along the rotor-frame
   $+\hat x$.
3. Step both at the same $\Delta t$; sample the field at the UMR
   (origin) 100 times per period.
4. Compute (a) per-sample magnitude error and (b) per-sample direction
   angle between the two field vectors.

## Result

**PASS**.

- The relative magnitude error stays below 2% over the full period —
  the residual is dipole-vs-uniform field anisotropy and the small
  motor startup transient at $t=0^+$.
- The direction agreement is within 5° of phase — the new chain's
  motor needs ~one rotation period to fully lock into velocity-mode
  steady-state under the default PI gains. Tuning the gains
  (specifically `velocity_kp`) tightens this further; defaults are
  sufficient for the 2 % acceptance.

## Scope and Limitations

- Far-field configuration only: `point_dipole` is faithful at $z \gg
  R_{\text{magnet}}$. Closer in, the bench needs the `current_loop`
  or `coulombian_poles` model — see `MIME-VER-110` / `MIME-VER-111`.
- Validates the *producer* side. Whether downstream physics (UMR
  magnetic response, rigid body, drag) reproduce the legacy result
  is the subject of `MIME-VER-131`.

## Reproducibility

- JAX precision: x64.
- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_actuation_chain_equivalence.py::test_ver130_field_equivalence_far_field -x -q`.
