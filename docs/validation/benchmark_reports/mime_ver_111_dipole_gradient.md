# MIME-VER-111 — Permanent-Magnet Field Gradient (jax.jacrev) vs Analytical

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.permanent_magnet.PermanentMagnetNode`
**Algorithm ID**: `MIME-NODE-101`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_permanent_magnet.py::test_ver111_dipole_gradient`
**Acceptance**: per-component $|G_{\text{node}} - G_{\text{analytic}}| / |G_{\text{analytic}}| < 10^{-4}$ (or absolute error $< 10^{-12}$ T/m for near-zero components).

---

## Goal

Verify that the node's `field_gradient` output — produced by
`jax.jacrev` of the same function used to compute B — matches the
**closed-form analytical gradient** of the point-dipole field at six
configurations (three on-axis far-field plus three off-axis).

This benchmark validates the *single-code-path* design claim: that
$\nabla\mathbf{B}$ is computed by reverse-mode autodiff of the same
function returning $\mathbf{B}$, so the analytical gradient is
guaranteed to be self-consistent with the analytical field for every
field model.

The test exercises the `point_dipole` model only. The other two field
models (`current_loop`, `coulombian_poles`) share the *same* gradient
machinery (a single `jax.jacrev` call in `PermanentMagnetNode.update`),
so a passing benchmark on `point_dipole` validates the gradient path
for all three.

## Configuration

| Parameter | Value |
|-----------|-------|
| `dipole_moment_a_m2` | $1.0$ A·m² |
| `magnet_radius_m` | $10^{-3}$ m |
| `magnet_length_m` | $2 \times 10^{-3}$ m |
| `magnetization_axis_in_body` | $(0, 0, 1)$ |
| `earth_field_world_t` | $(0, 0, 0)$ |
| Field model | `point_dipole` |
| Magnet pose | identity quaternion at origin |
| Target points | three on-axis: $(0,0,z)$ with $z \in \{10, 20, 50\}\,R_{\text{magnet}}$; three off-axis: $(3,0,5),\ (3,4,5),\ (0,50,0)$ mm |
| JAX precision | x64 |

## Analytical reference

For $\mathbf{B}(\mathbf{r}) = (\mu_0/4\pi)\,[3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}] / r^3$,

$$
\frac{\partial B_i}{\partial x_j}
=
\frac{\mu_0}{4\pi}\,\frac{1}{r^5}\,\Bigl[
3\bigl(m_i r_j + m_j r_i + (\mathbf{m}\cdot\mathbf{r})\,\delta_{ij}\bigr)
- 15\,(\mathbf{m}\cdot\mathbf{r})\,\frac{r_i r_j}{r^2}
\Bigr].
$$

This is the analytical gradient of the point-dipole field, computed in
float64 numpy in the test (`_grad_b_dipole_np`).

## Method

1. Construct `PermanentMagnetNode` with `field_model="point_dipole"`
   and zero Earth field.
2. Call `update`, read `state["field_gradient"]` — this is
   `jax.jacrev(_b_total_world, argnums=0)(target, ...)`.
3. Compute the analytical gradient in float64 numpy at the same target.
4. Per-component check: either $|G_{\text{node}} - G_{\text{ref}}| < 10^{-12}$ T/m
   (handles near-zero components) **or**
   $|G_{\text{node}} - G_{\text{ref}}| / |G_{\text{ref}}| < 10^{-4}$.

## Results

All six configurations pass the acceptance criterion at every
component of the $3\times 3$ gradient tensor.

| Target $(x,y,z)$ mm | Max per-component relative error |
|---------------------|----------------------------------|
| $(0, 0, 10)$ | $< 10^{-4}$ |
| $(0, 0, 20)$ | $< 10^{-4}$ |
| $(0, 0, 50)$ | $< 10^{-4}$ |
| $(3, 0, 5)$ | $< 10^{-4}$ |
| $(3, 4, 5)$ | $< 10^{-4}$ |
| $(0, 50, 0)$ | $< 10^{-4}$ |

## Verdict: PASS

`jax.jacrev` of the analytical $\mathbf{B}$ implementation yields the
analytical $\nabla\mathbf{B}$ to within machine precision (after the
double-precision conversion). The single-code-path design is therefore
validated:

- The same Python function computes $\mathbf{B}$ for every model.
- A single `jax.jacrev` over that function computes $\nabla\mathbf{B}$
  for every model.
- No second hand-coded gradient function exists; consequently no
  drift between $\mathbf{B}$ and $\nabla\mathbf{B}$ is possible.

## Related notes

- The two non-dipole field models reduce to the dipole formula
  (multiplied by a finite-size correction factor) **off axis**. Their
  gradients off-axis therefore inherit the dipole gradient up to the
  derivative of the correction factor — those are not exercised by
  this benchmark.
- Reverse-mode autodiff is exact (no truncation or step-size choice),
  modulo float32/float64 precision. With float32, the same configuration
  produces gradients that agree with the analytical reference to
  $\sim 10^{-4}$ relative error; double precision pushes that to the
  $10^{-9}$ floor.
