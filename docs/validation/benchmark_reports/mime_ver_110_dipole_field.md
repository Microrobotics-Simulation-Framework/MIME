# MIME-VER-110 — Permanent-Magnet Field vs Analytical

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.permanent_magnet.PermanentMagnetNode`
**Algorithm ID**: `MIME-NODE-101`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_permanent_magnet.py::test_ver110_dipole_field_far_field`
**Acceptance**: $|B_{\text{node}} - B_{\text{analytic}}| / |B_{\text{analytic}}| < 10^{-4}$

---

## Goal

Verify that each of the three field models (`point_dipole`,
`current_loop`, `coulombian_poles`) returns the **B field** at a target
point in agreement with its own analytical reference, in the far-field
regime $r/R_{\text{magnet}} \in \{10, 20, 50\}$.

This benchmark validates the *value* output of the node. The
companion benchmarks MIME-VER-111 and MIME-VER-112 cover the gradient
and the Earth-field superposition respectively.

## Configuration

| Parameter | Value |
|-----------|-------|
| `dipole_moment_a_m2` | $1.0$ A·m² |
| `magnet_radius_m` | $10^{-3}$ m |
| `magnet_length_m` | $2 \times 10^{-3}$ m |
| `magnetization_axis_in_body` | $(0,0,1)$ |
| `earth_field_world_t` | $(0,0,0)$ — magnet-only check |
| Magnet pose | identity quaternion at origin |
| Target points | $(0,0,z)$ with $z \in \{10, 20, 50\}\,R_{\text{magnet}}$ |
| JAX precision | x64 (enabled at module load) |

## Analytical references (numpy, double precision)

**Point dipole** (any displacement $\mathbf{r}$):

$$
\mathbf{B}_{\text{dipole}} = \frac{\mu_0}{4\pi} \cdot
\frac{3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}}{r^3}.
$$

**Current loop**, on-axis:

$$
B_z = \frac{\mu_0 I R^2}{2(R^2+z^2)^{3/2}},\quad I_{\text{eff}} = \frac{|\mathbf{m}|}{\pi R^2}.
$$

**Coulombian poles**, on-axis ($M = |\mathbf{m}| / (\pi R^2 L)$):

$$
B_z = \frac{\mu_0 M}{2}\left[
\frac{z+L/2}{\sqrt{R^2+(z+L/2)^2}}
- \frac{z-L/2}{\sqrt{R^2+(z-L/2)^2}}
\right].
$$

## Method

1. Construct `PermanentMagnetNode` with the model under test and
   `earth_field_world_t = (0, 0, 0)`.
2. Call `update(state, boundary_inputs, dt)` with the magnet pose and
   target. Read `state["field_vector"]`.
3. Compute the analytical reference in float64 numpy.
4. Compare per-point: $\|B_{\text{node}} - B_{\text{ref}}\| / \|B_{\text{ref}}\| < 10^{-4}$.

A second test (`test_ver110_dipole_field_off_axis`) checks the
`point_dipole` model at three off-axis configurations
($r \in \{(3,0,5), (3,4,5), (0,50,0)\}$ mm) — these are 3D
configurations where coordinate-frame errors would manifest as
cross-component leakage.

## Results

| Model | $z = 10\,R$ | $z = 20\,R$ | $z = 50\,R$ |
|-------|------------|------------|-------------|
| `point_dipole`     | $< 10^{-4}$ | $< 10^{-4}$ | $< 10^{-4}$ |
| `current_loop`     | $< 10^{-4}$ | $< 10^{-4}$ | $< 10^{-4}$ |
| `coulombian_poles` | $< 10^{-4}$ | $< 10^{-4}$ | $< 10^{-4}$ |

Off-axis (`point_dipole` only):

| Target $(x,y,z)$ mm | Relative error |
|---------------------|----------------|
| $(3, 0, 5)$ | $< 10^{-4}$ |
| $(3, 4, 5)$ | $< 10^{-4}$ |
| $(0, 50, 0)$ | $< 10^{-4}$ |

All eight configurations pass the $10^{-4}$ acceptance threshold.

## Verdict: PASS

The three field models match their analytical references to better
than $10^{-4}$ relative error in the far field, on-axis. The
point-dipole model also matches off-axis to the same tolerance. The
v1 implementation is faithful in its declared validity envelope.

## Related notes

- Float32 is **insufficient** for this benchmark; the
  `coulombian_poles` model differs from its float64 reference by
  $3.6 \times 10^{-4}$ at $z = 20\,R_{\text{magnet}}$ in float32.
  The test module enables `jax_enable_x64 = True` at import time.
- The `current_loop` and `coulombian_poles` models reduce to dipole +
  near-field correction *off axis*; this benchmark only exercises
  the on-axis closed forms for those two models.
- For **near-field** behaviour (z < 5 R), the point-dipole formula
  diverges from the finite-magnet truth — see
  `docs/deliverables/dejongh_benchmark_summary.md` lines 393–413
  for the calibrating numbers (~ 4 % at z = 3 R, < 1 % at z = 10 R).
  This benchmark therefore exercises only z ≥ 10 R, where all three
  models agree with their respective analytical truths.
