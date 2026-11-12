"""The patch inside an outer Krylov solve that JAX transposes (the coordinator's finding for F5's product): the outer
solve's matvec holds a patch column, so reverse mode transposes the patch's own solve nested inside it. The patch's CG
goes through ``projected_solve`` on a grid with no weighted product, so it is a symmetric ``custom_linear_solve`` (its
transpose solve is itself, and JAX never transposes its matvec), its stop a frozen tolerance, its threshold made inside
the solve from its own right-hand side, never in the matvec. The dot-product identity ``<J v, w> = <v, J^T w>`` of
the whole nested map holds to the solves' tolerance. Control ``PATCH_PLANT=loose_transpose``: the patch's transpose
solve given apart and capped at two iterations (``_solve.transposed``, the seam the flow's own gradient gate plants
through), so the nested transpose is not the transpose."""

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.flow.fvm import _solve
from mime.nodes.flow.fvm._patch import RefinementPatch, patch_reflection, patch_stopping, placed
from mime.nodes.flow.fvm._solve import REPORT_SIZE


class Plane:
    region = "wall"

    def __init__(self, at):
        self.at = at

    def signed_distance(self, points):
        return self.at - points[..., 2]


@pytest.fixture
def float64(monkeypatch):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    if os.environ.get("PATCH_PLANT") == "loose_transpose":
        monkeypatch.setattr(_solve, "transposed", lambda stopping: dataclasses.replace(stopping, cap=2))
    yield
    jax.config.update("jax_enable_x64", held)


def test_the_patch_column_nested_in_an_outer_krylov_solve_transposes(float64):
    rng = np.random.default_rng(3)
    h = 5e-5
    patch = RefinementPatch((12, 12, 12), h, read="point")
    points = jnp.asarray(rng.uniform(4.0, 8.0, size=(6, 3)) * h + np.array([0.0, 0.0, 2.0 * h]))
    placement = placed(patch, [Plane(4.3 * h)], jnp.zeros(3, jnp.int32), 0, points, (0.3 * h) ** 2, 0.1,
                       shift=1260.0 / 1e-3)
    grid = patch.grid()
    zero = tuple(jnp.zeros(grid.given_count(c)) for c in range(3))
    stopping = patch_stopping(patch, jnp.float64, tolerance=1e-12)
    tap = jnp.zeros(REPORT_SIZE)

    def column(f):
        return patch_reflection(patch, placement, zero, f.reshape(-1, 3), None, stopping, tap).values.reshape(-1)

    probe = jnp.asarray(rng.normal(size=18))
    scale = 0.5 / float(jnp.linalg.norm(column(probe)) / jnp.linalg.norm(probe))

    def outer(b):
        solved, _ = jax.scipy.sparse.linalg.gmres(lambda x: x + scale * column(x), b, tol=1e-12, atol=0.0,
                                                  restart=18, maxiter=4)
        return solved

    b = jnp.asarray(rng.normal(size=18))
    v = jnp.asarray(rng.normal(size=18))
    w = jnp.asarray(rng.normal(size=18))
    _, forward = jax.jvp(outer, (b,), (v,))
    _, pullback = jax.vjp(outer, b)
    (backward,) = pullback(w)
    left, right = float(jnp.dot(forward, w)), float(jnp.dot(v, backward))
    size = float(jnp.linalg.norm(forward) * jnp.linalg.norm(w) + jnp.linalg.norm(v) * jnp.linalg.norm(backward))
    assert abs(left - right) <= 1e-8 * size, (left, right, abs(left - right) / size)
