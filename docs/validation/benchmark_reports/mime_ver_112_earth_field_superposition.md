# MIME-VER-112 — Earth-Field Superposition Exactness

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.permanent_magnet.PermanentMagnetNode`
**Algorithm ID**: `MIME-NODE-101`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_permanent_magnet.py::test_ver112_earth_field_superposition`
**Acceptance**:
$\mathbf{B}_{\text{total, with earth}} - \mathbf{B}_{\text{total, no earth}} \equiv \mathbf{B}_{\text{earth}}$ (atol $10^{-12}$);
$\mathbf{B}_{\text{total}}(+\mathbf{B}_{\text{earth}}) - \mathbf{B}_{\text{total}}(-\mathbf{B}_{\text{earth}}) \equiv 2\mathbf{B}_{\text{earth}}$ (atol $10^{-12}$).

---

## Goal

Verify that the node's Earth-field background superposes **exactly**
onto the magnet's field — no scaling, no rotation, no leakage into the
$\nabla\mathbf{B}$ output (the Earth field is uniform, so its gradient
is zero).

In hardware-bench experiments, the static Earth field
($\sim 50\,\mu\text{T}$) is comparable in magnitude to a small distant
permanent magnet's stray field at the experimental working distance,
and the response of a magnetised UMR depends on the *vector sum* of
the two. This benchmark guarantees that sum is computed correctly.

## Configuration

| Parameter | Value |
|-----------|-------|
| `dipole_moment_a_m2` | $1.0$ A·m² |
| `magnet_radius_m` | $10^{-3}$ m |
| `magnet_length_m` | $2 \times 10^{-3}$ m |
| `field_model` | `point_dipole` |
| Magnet pose | identity quaternion at origin |
| Target | $(0, 0, 10\,R_{\text{magnet}})$ |
| Test Earth field | $\mathbf{B}_{\text{earth}} = (2 \times 10^{-5}, 1 \times 10^{-5}, -4.5 \times 10^{-5})$ T (asymmetric — all three components nonzero) |
| JAX precision | x64 |

## Method

Three node instances are constructed, identical in every parameter
*except* the Earth field:

1. `node_no_earth` — `earth_field_world_t = (0, 0, 0)`.
2. `node_with_earth` — `earth_field_world_t = +B_earth`.
3. `node_neg_earth` — `earth_field_world_t = -B_earth`.

Each evaluates `field_vector` at the same target. The differences are
checked element-wise against the analytical expectation:

$$
\mathbf{B}_{\text{pos}} - \mathbf{B}_{\text{no}} = \mathbf{B}_{\text{earth}},
$$

$$
\mathbf{B}_{\text{pos}} - \mathbf{B}_{\text{neg}} = 2\mathbf{B}_{\text{earth}}.
$$

Both checks use absolute tolerance $10^{-12}$ T (a few orders of
magnitude tighter than the float64 floor for these magnitudes).

## Results

Both checks pass to absolute tolerance $10^{-12}$ T. The Earth field
adds linearly without scaling or rotation; the magnet contribution is
unchanged when the Earth field is flipped.

## Verdict: PASS

The node's Earth-field treatment is an exact additive superposition.
Two corollaries:

- A consumer can recover the magnet-only field by subtracting the
  configured `earth_field_world_t` from the node's output. (See
  `test_grad_wrt_amplitude` in the same test file for an example.)
- The amplitude scale acts on the *magnet contribution only* — it
  does not multiply the Earth field. This is the documented behaviour
  in `_b_total_world`.

## Related notes

- The Earth field is uniform; its gradient is therefore zero, and
  the node's `field_gradient` output is unaffected by the Earth-field
  background. This is enforced automatically because the
  `_b_total_world` function on which `jax.jacrev` operates includes
  the Earth field as a constant, whose derivative w.r.t. the target
  position is zero. (No separate test is required — it is a
  derivative-of-a-constant identity.)
- Default Earth field in the constructor is
  $(2 \times 10^{-5}, 0, -4.5 \times 10^{-5})$ T — rough mid-latitude
  magnitudes. Users can override or zero this for clean
  magnet-only verification (as VER-110 and VER-111 do).
