"""The Cartesian MAC grid kind's contract: the weighted inner product (gate 12), the exact projector, the operators'
order, and the convective term's energy conservation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from fvm_cases import divergence_free, eps, sampled, trig_grid


def _norm(grid, a):
    return float(jnp.sqrt(grid.inner(a, a)))


def test_the_inner_product_is_the_cell_volume_times_the_sum_over_every_face(dtype):
    grid = trig_grid(8)
    rng = np.random.default_rng(3)
    a = tuple(jnp.asarray(rng.standard_normal(grid.cells), dtype=dtype) for _ in range(3))
    b = tuple(jnp.asarray(rng.standard_normal(grid.cells), dtype=dtype) for _ in range(3))
    plain = sum(np.sum(np.asarray(x, np.float64) * np.asarray(y, np.float64)) for x, y in zip(a, b, strict=True))
    scale = sum(np.sum(np.abs(np.asarray(x, np.float64) * np.asarray(y, np.float64))) for x, y in zip(a, b, strict=True))
    assert abs(float(grid.inner(a, b)) - grid.cell_volume * plain) <= 64 * eps(dtype) * grid.cell_volume * scale
    assert float(grid.inner_magnitude(a, b)) == pytest.approx(grid.cell_volume * scale, rel=64 * eps(dtype))


def test_control_an_inner_product_missing_one_component_fails_the_same_comparison(dtype):
    grid = trig_grid(8)
    rng = np.random.default_rng(4)
    a = tuple(jnp.asarray(rng.standard_normal(grid.cells), dtype=dtype) for _ in range(3))
    full = grid.inner(a, a)
    partial = grid.cell_volume * (jnp.sum(a[0] * a[0]) + jnp.sum(a[1] * a[1]))
    assert abs(float(full - partial)) > 1e3 * eps(dtype) * float(full)


def test_the_projector_is_exact_idempotent_and_symmetric(dtype):
    grid = trig_grid(16)
    rng = np.random.default_rng(5)
    u = tuple(jnp.asarray(rng.standard_normal(grid.cells), dtype=dtype) for _ in range(3))
    v = tuple(jnp.asarray(rng.standard_normal(grid.cells), dtype=dtype) for _ in range(3))
    pu = grid.project(u)
    size = _norm(grid, u)
    divergence = float(jnp.max(jnp.abs(grid.divergence(pu)))) * grid.spacing
    assert divergence <= 1e3 * eps(dtype) * float(jnp.max(jnp.abs(u[0])))
    again = grid.project(pu)
    assert _norm(grid, tuple(a - b for a, b in zip(again, pu, strict=True))) <= 1e2 * eps(dtype) * size
    left, right = float(grid.inner(grid.project(u), v)), float(grid.inner(u, grid.project(v)))
    assert abs(left - right) <= 1e3 * eps(dtype) * size * _norm(grid, v)


def test_a_discretely_divergence_free_field_is_left_as_it_is(dtype):
    grid = trig_grid(16)
    u = divergence_free(grid, dtype)
    pu = grid.project(u)
    assert _norm(grid, tuple(a - b for a, b in zip(pu, u, strict=True))) <= 1e3 * eps(dtype) * _norm(grid, u)


def _order(errors, ns):
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(ns[i + 1] / ns[i])) for i in range(len(ns) - 1)]


def test_the_operators_converge_at_second_order(float64):
    ns = (8, 16, 32)
    lap_errors, grad_errors, conv_errors = [], [], []
    for n in ns:
        grid = trig_grid(n)
        u = divergence_free(grid, "float64")
        # The Laplacian of U1 is -3 U1, of U2 -2 U2: with a = 1, b = 0.5, compare component by component.
        u1 = divergence_free(grid, "float64", 1.0, 0.0)
        u2 = divergence_free(grid, "float64", 0.0, 1.0)
        exact = tuple(-3.0 * p - 2.0 * 0.5 * q for p, q in zip(u1, u2, strict=True))
        lap = grid.laplacian(u)
        lap_errors.append(max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(lap, exact, strict=True)))
        centres = grid.cell_points("float64")
        p = jnp.sin(centres[..., 0]) * jnp.cos(centres[..., 1]) * jnp.sin(2 * centres[..., 2])
        g = grid.gradient(p)
        exact_g = sampled(grid, (lambda q: jnp.cos(q[..., 0]) * jnp.cos(q[..., 1]) * jnp.sin(2 * q[..., 2]),
                                 lambda q: -jnp.sin(q[..., 0]) * jnp.sin(q[..., 1]) * jnp.sin(2 * q[..., 2]),
                                 lambda q: 2 * jnp.sin(q[..., 0]) * jnp.cos(q[..., 1]) * jnp.cos(2 * q[..., 2])),
                          "float64")
        grad_errors.append(max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(g, exact_g, strict=True)))
        conv_errors.append(conv_error(grid, u1))
    # Second order is asymptotic: at N 8 the convective term's order is still 1.71 (1.93 at 16 to 32, 1.98 at 32 to 64),
    # so the finest pair carries the claim and the coarse one only bounds how far from it the start is.
    for errors in (lap_errors, grad_errors, conv_errors):
        coarse, fine = _order(errors, ns)
        assert 1.9 <= fine <= 2.2
        assert coarse >= 1.6


def conv_error(grid, u1):
    """``-(U1 . grad) U1`` exactly, for ``U1 = (sin x cos y cos z, cos x sin y cos z, -2 cos x cos y sin z)``."""
    convection = grid.convection(u1)

    def exact(c):
        def f(q):
            x, y, z = q[..., 0], q[..., 1], q[..., 2]
            uu = (jnp.sin(x) * jnp.cos(y) * jnp.cos(z), jnp.cos(x) * jnp.sin(y) * jnp.cos(z),
                  -2 * jnp.cos(x) * jnp.cos(y) * jnp.sin(z))
            grads = {
                0: ((jnp.cos(x) * jnp.cos(y) * jnp.cos(z), -jnp.sin(x) * jnp.sin(y) * jnp.cos(z),
                     -jnp.sin(x) * jnp.cos(y) * jnp.sin(z))),
                1: ((-jnp.sin(x) * jnp.sin(y) * jnp.cos(z), jnp.cos(x) * jnp.cos(y) * jnp.cos(z),
                     -jnp.cos(x) * jnp.sin(y) * jnp.sin(z))),
                2: ((2 * jnp.sin(x) * jnp.cos(y) * jnp.sin(z), 2 * jnp.cos(x) * jnp.sin(y) * jnp.sin(z),
                     -2 * jnp.cos(x) * jnp.cos(y) * jnp.cos(z))),
            }[c]
            return -(uu[0] * grads[0] + uu[1] * grads[1] + uu[2] * grads[2])
        return f

    reference = sampled(grid, (exact(0), exact(1), exact(2)), "float64")
    return max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(convection, reference, strict=True))


def _random_divergence_free(grid, dtype, seed):
    rng = np.random.default_rng(seed)
    return grid.project(tuple(jnp.asarray(rng.standard_normal(grid.cells), dtype=dtype) for _ in range(3)))


@pytest.mark.parametrize("seed", range(4))
def test_the_convective_term_does_no_work_on_a_divergence_free_field(dtype, seed):
    grid = trig_grid(16)
    u = _random_divergence_free(grid, dtype, seed)
    c = grid.convection(u)
    assert abs(float(grid.inner(c, u))) <= 1e3 * eps(dtype) * float(grid.inner_magnitude(c, u))


@pytest.mark.parametrize("seed", range(4))
def test_control_a_convective_term_shifted_one_cell_does_work(dtype, seed):
    """The divergence form conserves energy component by component, so a component scaled leaves the work zero and
    cannot be the control here (the manufactured solution catches it); a one-cell index shift does not."""
    grid = trig_grid(16)
    u = _random_divergence_free(grid, dtype, seed)
    c = grid.convection(u)
    shifted = (jnp.roll(c[0], 1, axis=1), c[1], c[2])
    assert abs(float(grid.inner(shifted, u))) > 1e3 * eps(dtype) * float(grid.inner_magnitude(c, u))
