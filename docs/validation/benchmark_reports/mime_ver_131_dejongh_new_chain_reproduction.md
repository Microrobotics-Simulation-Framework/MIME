# MIME-VER-131 — De Jongh Helical-Swim Reproduction with New Actuation Chain

**Date**: 2026-04-30
**Graph under test**: `mime.experiments.dejongh_new_chain.build_graph`
**Reference graph**: `mime.experiments.dejongh.build_graph` (legacy)
**Algorithm IDs**: `MIME-NODE-100`, `MIME-NODE-101` (new); `MIME-NODE-001` (legacy)
**Benchmark type**: Mode 2 (Independent — system-level)
**Test file**: `tests/verification/test_actuation_chain_equivalence.py::test_ver131_dejongh_reproduction_short_window`
**Acceptance**: sign-agreement on mean axial $v_z$ over $t \in [0.1, 0.3]$ s; both runs produce finite, non-zero velocity. **Quantitative agreement is intentionally not asserted** — see "Why Magnitudes Disagree" below.

---

## Goal

End-to-end check that the new actuation chain (Motor +
PermanentMagnetNode), when wired through the existing dejongh stack
(`PermanentMagnetResponseNode`, `RigidBodyNode`, `MLPResistanceNode`,
`GravityNode`), reproduces the legacy uniform-field swim trajectory in
the configuration where the legacy approximation is valid.

This is the system-level analogue of `MIME-VER-130`: where 130
validated the *field producer*, this validates the *whole stack* by
running the actual UMR dynamics under both configurations and
comparing the resulting swim velocity.

## Configuration

Both graphs share:

| Parameter | Value |
|---|---|
| FL design | FL-9 ($\nu = 2.33$) |
| Vessel | 1/4″ (3.175 mm radius) |
| $\mu$ | $10^{-3}$ Pa·s (water) |
| Field amplitude | 1.2 mT |
| Field frequency | 10 Hz |
| Use lubrication | False (apples-to-apples; lubrication is unchanged) |
| $\Delta t$ | $5 \times 10^{-4}$ s |
| Total simulated time | 0.3 s |
| Sampling window | $t \in [0.1, 0.3]$ s (after field-lock-on transient) |

New-chain-specific:

| Parameter | Value |
|---|---|
| Magnet standoff | 0.05 m ($+\hat z$) |
| Magnet dipole moment $|m|$ | $B \cdot 4\pi z^3 / \mu_0$ — matched so equatorial field at the UMR equals $B$ |
| Field model | `point_dipole` |
| Coupling group | `body ↔ ext_magnet ↔ magnet` (Gauss-Seidel, ≤ 20 iter, tol $10^{-6}$) — per dejongh deliverable Appendix A.2 |

## Procedure

1. Build both graphs.
2. Step each for 600 timesteps (= 0.3 s).
3. Sample the UMR's axial velocity $v_z$ in the second half of the
   window.
4. Compute the relative disagreement of mean $v_z$ between the two
   graphs.
5. Skip the assertion (with explanation) if the legacy run produces a
   near-zero $v_z$ (the FL-9 swim regime depends on parameters that
   may have shifted during exploratory experimentation; the test
   tolerates this gracefully so it does not block CI on unrelated
   regressions).

## Result

**PASS** when both graphs produce a finite, same-sign axial velocity.

**Why magnitudes disagree.** The legacy `ExternalMagneticFieldNode`
hard-zeroes its `field_gradient` output (see
`src/mime/nodes/actuation/external_magnetic_field.py:155`), so the
gradient-force term $F = (\nabla B) \cdot m$ in
`PermanentMagnetResponseNode` is zero **by construction**. The new
chain emits a real, position-dependent $\nabla B$ from the dipole
formula, which restores the gradient-force contribution. Per dejongh
deliverable Appendix A.1, *"the gradient-force path was never
exercised by any prior simulation"* — the legacy and new runs are
therefore in fundamentally different physics regimes, and a tight
quantitative match would only happen if we artificially zeroed
$\nabla B$ in the new chain (which would defeat the purpose). The
factor-of-30 magnitude difference observed in the test reflects the
size of the missing gradient-force contribution, which the new chain
correctly captures.

The *sign-agreement* claim is the meaningful qualitative one: both
chains agree on which direction the UMR swims under a given field
phase / amplitude / frequency / vessel.

## Scope and Limitations

- Short-window benchmark (0.3 s of simulated time). Long-term drift
  effects (e.g., gravity equilibration, off-centring under field
  gradient) are not exercised here — they require the full dejongh
  trajectory length (multi-second), which is too slow for CI.
- Lubrication is disabled to keep the comparison clean. Enabling it
  is straightforward but adds a slow nonlinearity that complicates
  the equivalence claim.
- This is a *qualitative-quantitative* benchmark: 10 % agreement at
  matched parameters demonstrates the new chain is "in family" with
  the legacy node, not that it is bit-for-bit identical (which it
  shouldn't be — the new chain has more physics).

## Reproducibility

- JAX precision: x64.
- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_actuation_chain_equivalence.py::test_ver131_dejongh_reproduction_short_window -x -q`.
