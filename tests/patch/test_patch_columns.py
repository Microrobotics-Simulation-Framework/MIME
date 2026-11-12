"""The F5-patch interface's column gate and its rules, held by the patch's functions (the review's M5).

**vmap against the loop.** The patch's reflection and its transpose take one column of forces and are vmappable over a
column axis. Vmapped over seven columns they are the loop's program batched, whose last bits differ (the solve-free
parts agree to 1e-16); the CG, whose iterates are sensitive to rounding at its tolerance, then stops at a different
iterate within that tolerance, as the loop does under a one-ulp change of its own input. So at the default tolerance
the columns are the loop's to the solve's tolerance, and at the floor (1e-14 in float64, float32's default) to
rounding (4.7e-16 in float64). Bitwise equality would need a sequential batching rule, which has no transpose: reverse
mode through the vmapped columns would fail. The control ``PATCH_PLANT=short_cap`` stops the vmapped solves at 20
iterations.

**The starts** (the interface's rule 3 and the PR's column rule): a column starts at zero; the base column at zero or
from its own history's field; a start on a column, a base start from any other field (a product's solution), and a
start on a transpose are refused. The base column's history rides with its column only."""

import dataclasses
import os

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


#: How many of the solve's tolerances the vmapped columns may depart from the loop by, on each column's scale. Two
#: roundings of one CG stop at different iterates within its tolerance: measured 1.1e-9 (this laptop) and 4.5e-8 (a
#: 4-vCPU runner, run 43240080205) in float64 at 1e-8, where a one-ulp nudge of the loop's own given values moved it
#: 4.4e-8 here and 3.0e-10 there; 2.0e-5 in float32 at its 1e-5 floor; 4.7e-16 in float64 at 1e-14. The short_cap plant
#: departs by 4.6e-4 in float64 (in float32, whose solve takes about 23 iterations, 4.9e-4, under this line).
TOLERANCES = 50.0


def patch_tolerance(dtype):
    return 1e-8 if dtype == jnp.float64 else 2e-4


def floor_tolerance(dtype):
    """The floor: 1e-14 in float64; float32's default stops at its own rounding floor."""
    return 1e-14 if dtype == jnp.float64 else None


def line(stopping):
    return TOLERANCES * float(stopping.tolerance)


def worst(a, b):
    """The largest difference over the columns, each on its own column's scale."""
    return max(float(jnp.max(jnp.abs(a[k] - b[k])) / jnp.max(jnp.abs(b[k]))) for k in range(COLUMNS))


@pytest.mark.parametrize("at", ["default", "floor"])
def test_vmapped_over_seven_columns_the_reflection_is_the_loop_within_its_rounding(dtype, at):
    """At the default tolerance the vmapped columns are the loop's to the solve's tolerance; at the floor (float64's
    1e-14, float32's default) to rounding."""
    patch, placement, boundary, forces, _ = case(dtype)
    stopping = patch_stopping(patch, dtype, tolerance=None if at == "default" else floor_tolerance(dtype))
    tap = jnp.zeros(REPORT_SIZE, dtype=dtype)
    mapped_stopping = dataclasses.replace(stopping, cap=20) if os.environ.get("PATCH_PLANT") == "short_cap" \
        else stopping

    def one(b0, b1, b2, f, held=stopping):
        return patch_reflection(patch, placement, (b0, b1, b2), f, None, held, tap).values

    mapped = jax.jit(jax.vmap(lambda b0, b1, b2, f: one(b0, b1, b2, f, mapped_stopping)))(*boundary, forces)
    loop = jax.jit(one)
    looped = jnp.stack([loop(boundary[0][k], boundary[1][k], boundary[2][k], forces[k]) for k in range(COLUMNS)])
    gap = worst(mapped, looped)
    assert gap <= line(stopping), gap
    if at == "floor" and dtype == jnp.float64:
        assert gap <= 1e3 * float(jnp.finfo(dtype).eps), gap


def test_vmapped_over_seven_columns_the_transpose_is_the_loop_within_its_rounding(dtype):
    patch, placement, _, _, cotangent = case(dtype)
    stopping = patch_stopping(patch, dtype)
    tap = jnp.zeros(REPORT_SIZE, dtype=dtype)

    def one(c):
        given, force, _ = patch_reflection_transpose(patch, placement, c, None, stopping, tap)
        return given, force

    mapped = jax.jit(jax.vmap(one))(cotangent)
    loop = jax.jit(one)
    looped = [loop(cotangent[k]) for k in range(COLUMNS)]
    for part in range(4):
        def pick(found):
            return found[1] if part == 3 else found[0][part]

        held = mapped[1] if part == 3 else mapped[0][part]
        gap = worst(held, jnp.stack([pick(looped[k]) for k in range(COLUMNS)]))
        assert gap <= line(stopping), (part, gap)


def test_a_start_is_refused_on_a_column_and_from_any_field_but_the_base_columns_history(dtype):
    patch, placement, boundary, forces, cotangent = case(dtype)
    stopping = patch_stopping(patch, dtype)
    tap = jnp.zeros(REPORT_SIZE, dtype=dtype)
    grid = patch.grid()
    given = (boundary[0][0], boundary[1][0], boundary[2][0])
    field = grid.zeros(dtype)
    history = PatchHistory(field, jnp.zeros(patch.capacity(), dtype=dtype))
    with pytest.raises(ValueError, match="a column of the patch starts at zero"):
        patch_reflection(patch, placement, given, forces[0], field, stopping, tap)
    with pytest.raises(ValueError, match="starts only from its history's own field"):
        patch_reflection(patch, placement, given, forces[0], grid.zeros(dtype), stopping, tap, history)
    with pytest.raises(ValueError, match="a column of the patch starts at zero"):
        patch_reflection_transpose(patch, placement, cotangent[0], field, stopping, tap)
    found = patch_reflection(patch, placement, given, forces[0], history.field, stopping, tap, history)
    assert float(found.report[0]) >= 1


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
