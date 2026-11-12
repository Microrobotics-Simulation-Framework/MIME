"""The F5-patch interface's column gate and its rules, held by the patch's functions (the review's M5).

**vmap against the loop.** The patch's reflection and its transpose take one column of forces and are vmappable over a
column axis. Vmapped over seven columns they are the loop's program batched, whose last bits differ (the solve-free
parts agree to 1e-16); the CG then stops at a different iterate within its tolerance. So the vmapped columns are judged
against the loop's own spread under a one-ulp change of its given values (the same program, its input moved by
rounding): at float64's default 1e-8 that spread is about 4e-8 at equal iterations and the vmapped 1e-9; at the floor
(1e-14 in float64, float32's default) both are rounding (5e-16 in float64). The control ``PATCH_PLANT=short_cap``
stops the vmapped solves at 20 iterations.

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


#: How many of the loop's own spreads (the larger of a one-ulp nudge up and down of its input) the vmapped columns may
#: depart by: each is one draw of the same rounding's effect (measured: the vmapped 1.1e-9 against the nudges' 4.4e-8
#: in float64 at 1e-8; the transpose's given values 7.5e-8 against 1.6e-8); the short_cap plant departs by 4.6e-4.
SPREADS = 10.0


def patch_tolerance(dtype):
    return 1e-8 if dtype == jnp.float64 else 2e-4


def floor_tolerance(dtype):
    """The floor: 1e-14 in float64; float32's default stops at its own rounding floor."""
    return 1e-14 if dtype == jnp.float64 else None


def worst(a, b):
    """The largest difference over the columns, each on its own column's scale."""
    return max(float(jnp.max(jnp.abs(a[k] - b[k])) / jnp.max(jnp.abs(b[k]))) for k in range(COLUMNS))


@pytest.mark.parametrize("at", ["default", "floor"])
def test_vmapped_over_seven_columns_the_reflection_is_the_loop_within_its_rounding(dtype, at):
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
    spread = 0.0
    for way in (np.inf, -np.inf):
        towards = jnp.asarray(way, dtype)
        nudged = jnp.stack([loop(*(jnp.nextafter(boundary[c][k], towards) for c in range(3)), forces[k])
                            for k in range(COLUMNS)])
        spread = max(spread, worst(nudged, looped))
    gap = worst(mapped, looped)
    eps = float(jnp.finfo(dtype).eps)
    assert gap <= SPREADS * max(spread, 64.0 * eps), (gap, spread)
    if at == "floor" and dtype == jnp.float64:
        assert gap <= 1e3 * eps, gap


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
    nudges = [[loop(jnp.nextafter(cotangent[k], jnp.asarray(way, dtype))) for k in range(COLUMNS)]
              for way in (np.inf, -np.inf)]
    eps = float(jnp.finfo(dtype).eps)
    for part in range(4):
        def pick(found):
            return found[1] if part == 3 else found[0][part]

        stacked = jnp.stack([pick(looped[k]) for k in range(COLUMNS)])
        spread = max(worst(jnp.stack([pick(nudged[k]) for k in range(COLUMNS)]), stacked) for nudged in nudges)
        held = mapped[1] if part == 3 else mapped[0][part]
        assert worst(held, stacked) <= SPREADS * max(spread, 64.0 * eps), (part, worst(held, stacked), spread)


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
