"""The F5-patch interface's column gate: the patch's reflection and its transpose take one column of forces and are
vmappable over a column axis; vmapped over seven columns they equal the loop over the columns to rounding, in both
dtypes, under the step operator (a shift of rho/dt) with a wall in the box. The base column's history rides with its
column only."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.flow.fvm._patch import PatchHistory, RefinementPatch, patch_reflection, patch_reflection_transpose, \
    patch_stopping, placed
from mime.nodes.flow.fvm._solve import REPORT_SIZE

COLUMNS = 7


class PlaneSolid:
    region = "wall"

    def __init__(self, at):
        self.at = at

    def signed_distance(self, points):
        return self.at - points[..., 2]


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    yield jnp.dtype(request.param)
    jax.config.update("jax_enable_x64", held)


def case(dtype):
    """A 12-cell patch of 50 um cells, a plane wall a third of the way up, 9 force points above it in water-like glycerol
    (0.1 Pa s) at a step of 1 ms (rho/dt = 1.26e6); each column's given values and forces drawn at a body's scales."""
    rng = np.random.default_rng(7)
    h = 5e-5
    patch = RefinementPatch((12, 12, 12), h, read="point")
    points = jnp.asarray(rng.uniform(4.0, 8.0, size=(9, 3)) * h + np.array([0.0, 0.0, 2.0 * h]), dtype=dtype)
    placement = placed(patch, [PlaneSolid(4.3 * h)], jnp.zeros(3, jnp.int32), 0, points, (0.3 * h) ** 2, 0.1,
                       shift=1260.0 / 1e-3)
    grid = patch.grid()
    boundary = tuple(jnp.asarray(rng.normal(size=(COLUMNS, grid.given_count(c))) * 1e-4, dtype=dtype)
                     for c in range(3))
    forces = jnp.asarray(rng.normal(size=(COLUMNS, 9, 3)) * 1e-6, dtype=dtype)
    cotangent = jnp.asarray(rng.normal(size=(COLUMNS, 9, 3)), dtype=dtype)
    return patch, placement, boundary, forces, cotangent


def close(a, b, dtype):
    """Within the CG's own tolerance of each other, on the column's scale."""
    scale = float(jnp.max(jnp.abs(b))) or 1.0
    return float(jnp.max(jnp.abs(a - b))) <= 10.0 * patch_tolerance(dtype) * scale


def patch_tolerance(dtype):
    return 1e-8 if dtype == jnp.float64 else 2e-4


def test_vmapped_over_seven_columns_the_reflection_equals_the_loop(dtype):
    patch, placement, boundary, forces, _ = case(dtype)
    stopping = patch_stopping(patch, dtype)
    tap = jnp.zeros(REPORT_SIZE, dtype=dtype)

    def one(b0, b1, b2, f):
        return patch_reflection(patch, placement, (b0, b1, b2), f, None, stopping, tap).values

    mapped = jax.jit(jax.vmap(one))(*boundary, forces)
    looped = jnp.stack([jax.jit(one)(boundary[0][k], boundary[1][k], boundary[2][k], forces[k])
                        for k in range(COLUMNS)])
    assert close(mapped, looped, dtype)


def test_vmapped_over_seven_columns_the_transpose_equals_the_loop(dtype):
    patch, placement, _, _, cotangent = case(dtype)
    stopping = patch_stopping(patch, dtype)
    tap = jnp.zeros(REPORT_SIZE, dtype=dtype)

    def one(c):
        given, force, _ = patch_reflection_transpose(patch, placement, c, None, stopping, tap)
        return given, force

    mapped = jax.jit(jax.vmap(one))(cotangent)
    for k in range(COLUMNS):
        given, force = jax.jit(one)(cotangent[k])
        assert close(mapped[1][k], force, dtype)
        for c in range(3):
            assert close(mapped[0][c][k], given[c], dtype)


def test_the_base_columns_history_moves_only_its_own_column(dtype):
    """A history on the base column (one of the seven) changes that column alone."""
    patch, placement, boundary, forces, _ = case(dtype)
    stopping = patch_stopping(patch, dtype)
    tap = jnp.zeros(REPORT_SIZE, dtype=dtype)
    rng = np.random.default_rng(11)
    grid = patch.grid()
    field = tuple(jnp.asarray(rng.normal(size=grid.interior_shape(c)) * 1e-4, dtype=dtype) for c in range(3))
    history = PatchHistory(field, jnp.asarray(rng.normal(size=patch.capacity()) * 1e-4, dtype=dtype))
    base = jax.jit(lambda b0, b1, b2, f: patch_reflection(patch, placement, (b0, b1, b2), f, None, stopping, tap,
                                                          history).values)
    column = jax.jit(lambda b0, b1, b2, f: patch_reflection(patch, placement, (b0, b1, b2), f, None, stopping,
                                                            tap).values)
    with_history = base(boundary[0][0], boundary[1][0], boundary[2][0], forces[0])
    without = column(boundary[0][0], boundary[1][0], boundary[2][0], forces[0])
    assert float(jnp.max(jnp.abs(with_history - without))) > 100.0 * patch_tolerance(dtype) * float(
        jnp.max(jnp.abs(without)))
