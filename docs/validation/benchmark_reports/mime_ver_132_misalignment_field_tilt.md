# MIME-VER-132 — Misalignment-Induced Field Tilt at the UMR

**Date**: 2026-04-30
**Producer under test**: `mime.nodes.actuation.permanent_magnet.PermanentMagnetNode`
**Algorithm ID**: `MIME-NODE-101`
**Benchmark type**: Mode 2 (Analytical)
**Test file**: `tests/verification/test_actuation_chain_equivalence.py::test_ver132_misalignment_field_tilt`
**Acceptance**: field-tilt angle at the UMR position grows strictly monotonically with the magnet's lateral offset over a 0–6 mm sweep; on-axis tilt < 1°

---

## Goal

Verify the *upstream cause* of the misalignment-induced step-out
reduction described in the approved plan
(`/home/nick/.claude/plans/hi-familiarize-yourself-with-giggly-toucan.md`).

The user's stated phenomenon:

> Slight misalignment with the actuator (i.e. not directly above) causes
> the UMR propulsion vector to be slightly non-parallel with the vessel.
> This effect increases as the offset increases.

This benchmark isolates the **field-side cause** — the dipole field at
the UMR location tilts off the vessel-perpendicular plane as the
magnet moves laterally. Downstream, that tilt drives the UMR axis to
follow it (via $T = m \times B$), which produces the wobble and the
lateral wall pressure that ultimately reduces step-out.

A *direct* benchmark of step-out reduction requires a calibrated
`ContactFrictionNode`, which is **out of scope for this plan** (per
plan §"Out of Scope"). The plan therefore declares MIME-VER-132's
quantitative tolerance on UMR-axis tilt deferred until that
calibration lands; here we assert the upstream cause's monotonicity,
which is a falsifiable claim about the dipole field formula.

## Configuration

| Parameter | Value |
|---|---|
| Magnet standoff $z$ | 0.05 m |
| Magnet rotation rate | 10 Hz about $+\hat z$ |
| Magnet geometry | $R = 1$ mm, $L = 2$ mm |
| Field model | `point_dipole` |
| Lateral offsets swept | 0, 1, 2, 4, 6 mm (in $+\hat x$) |
| Target | UMR at origin |
| $\Delta t$ | $10^{-4}$ s |

## Procedure

For each offset:

1. Spin the motor up to its commanded $\omega = 2\pi \cdot 10$ Hz at
   the standoff with the lateral offset applied to its parent pose.
2. Sample the magnet's instantaneous $B$-field at the UMR (origin)
   over one full rotation period.
3. Take a sample after spin-up and compute the tilt angle of $B$ off
   the xy plane: $\arcsin(|B_z| / |B|)$.

Assert:

- The list of tilt angles is *strictly monotonically increasing* with
  offset.
- The on-axis (offset = 0) tilt is below 1°.

## Result

**PASS** when both checks hold.

The on-axis tilt is small (not exactly zero because the motor is in a
brief PI-startup transient when the snapshot is taken, and the dipole
formula evaluated at one instant is not exactly co-planar with the
xy plane unless the dipole vector is also in the xy plane at that
instant).

## Scope and Limitations

- This benchmark validates the **field-side** chain only (i.e., the
  output of `PermanentMagnetNode`). Whether the UMR's axis tilts as
  predicted by $T = m \times B$ requires a coupled simulation, which
  is the subject of follow-on work once `ContactFrictionNode` is
  calibrated.
- The dipole-only formula is valid here because the standoff is in
  the far-field regime ($z \approx 50 R_{\text{magnet}}$). At smaller
  standoffs the bench would need to switch to `current_loop` or
  `coulombian_poles`.

## Reproducibility

- JAX precision: x64.
- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_actuation_chain_equivalence.py::test_ver132_misalignment_field_tilt -x -q`.
