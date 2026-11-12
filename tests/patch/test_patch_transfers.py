"""Gate 4 of the patch phase: the two transfers are an exact transpose pair. A coarse field read at the patch's given
values and its spread back, the patch's field read at points and over the cells of a quadrature and their spreads,
judged by the adjoint identity on the scale of its own terms, over drawn grids (periodic, and open along z), fine cells,
origins, fields and points. The control plants a transpose that is not one, ``(1 + 1e-6 cos k)`` on each spread entry,
switched on by ``PATCH_PLANT=nontranspose``."""

import os
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.nodes.flow.fvm._grid import MACGrid
from mime.nodes.flow.fvm._nested import NestedMACGrid
from mime.nodes.flow.fvm._open import OpenMACGrid
from mime.nodes.flow.fvm._patch_transfers import Quadrature, cell_read, cell_read_transpose, patch_boundary, \
    patch_boundary_transpose, patch_read, patch_read_transpose

#: The identity's line, in units of the dtype's rounding on the scale of its terms.
ROUNDINGS = 64


DTYPES = ("float32", "float64")


@contextmanager
def run_dtype(name):
    """The run's dtype for one draw, restored after (a fixture is not reset between a property's draws)."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield jnp.dtype(name)
    finally:
        jax.config.update("jax_enable_x64", held)


def planted(spread):
    """The spread as the control plants it: each entry times ``1 + 1e-6 cos k``, ``k`` its flat index."""
    if os.environ.get("PATCH_PLANT") != "nontranspose":
        return spread
    return tuple(a * (1.0 + 1e-6 * jnp.cos(jnp.arange(a.size, dtype=a.dtype).reshape(a.shape))) for a in spread)


def identity_error(forward, backward, dtype):
    """``|<B x, y> - <x, B^T y>|`` over the sum of its terms' magnitudes, both sides': the rounding's scale."""
    (bx, y), (x, bty) = forward, backward
    left = sum(float(jnp.sum(a * b)) for a, b in zip(bx, y))
    right = sum(float(jnp.sum(a * b)) for a, b in zip(x, bty))
    scale = sum(float(jnp.sum(jnp.abs(a * b))) for a, b in zip(bx, y)) + \
        sum(float(jnp.sum(jnp.abs(a * b))) for a, b in zip(x, bty))
    return abs(left - right) / scale, ROUNDINGS * float(jnp.finfo(dtype).eps)


#: The boxes the draws take their shapes from (a few, so that each operation compiles once a shape: the values, the
#: spacings, the ratio and the origins are drawn freely).
COARSE_CELLS = ((8, 9, 10), (12, 7, 9))
PATCH_CELLS = ((5, 6, 7), (8, 8, 8))
POINT_COUNTS = (5, 17)


@st.composite
def nestings(draw):
    """A coarse grid (periodic, or open along z), a fine cell, a patch inside it and a seed: a body's scales, a coarse
    cell of 50 um to 0.5 mm, the fine cell 1.5 to 6 times smaller."""
    open_axis = draw(st.booleans())
    cells = draw(st.sampled_from(COARSE_CELLS))
    spacing = draw(st.floats(5e-5, 5e-4))
    coarse = OpenMACGrid(cells, spacing) if open_axis else MACGrid(cells, spacing)
    fine = spacing / draw(st.floats(1.5, 6.0))
    patch_cells = draw(st.sampled_from(PATCH_CELLS))
    grid = NestedMACGrid(patch_cells, fine)
    origin = []
    for axis in range(3):
        extent = cells[axis] * spacing / fine
        if axis == 2 and open_axis:
            low, high = 1, int(extent) - patch_cells[axis] - 2
            origin.append(draw(st.integers(low, max(low, high))))
        else:
            origin.append(draw(st.integers(int(-extent), int(2 * extent))))
    return coarse, grid, np.asarray(origin, dtype=np.int32), draw(st.integers(0, 2 ** 31 - 1))


def fields(shapes, rng, dtype):
    return tuple(jnp.asarray(rng.normal(size=s), dtype=dtype) for s in shapes)


@pytest.mark.parametrize("name", DTYPES)
@settings(deadline=None)
@given(nesting=nestings())
def test_gate4_the_boundary_transfer_and_its_spread_are_a_transpose_pair(name, nesting):
    with run_dtype(name) as dtype:
        boundary_pair(dtype, nesting)


def boundary_pair(dtype, nesting):
    coarse, grid, origin, seed = nesting
    rng = np.random.default_rng(seed)
    origin = jnp.asarray(origin)
    shapes = tuple((OpenMACGrid.shape(coarse, c) if isinstance(coarse, OpenMACGrid) else coarse.cells) for c in range(3))
    corner = jnp.asarray(rng.uniform(-2.0, 2.0, size=3) * coarse.spacing, dtype=dtype)
    x = fields(shapes, rng, dtype)
    y = fields([(grid.given_count(c),) for c in range(3)], rng, dtype)
    bx, _ = patch_boundary(grid, origin, coarse, x, corner)
    bty = planted(patch_boundary_transpose(grid, origin, coarse, shapes, corner, y))
    error, line = identity_error((bx, y), (x, bty), dtype)
    assert error <= line, (error, line)


@pytest.mark.parametrize("name", DTYPES)
@settings(deadline=None)
@given(nesting=nestings(), count=st.sampled_from(POINT_COUNTS))
def test_gate4_the_read_and_the_cell_read_and_their_spreads_are_transpose_pairs(name, nesting, count):
    with run_dtype(name) as dtype:
        read_pairs(dtype, nesting, count)


def read_pairs(dtype, nesting, count):
    _, grid, origin, seed = nesting
    rng = np.random.default_rng(seed)
    origin = jnp.asarray(origin)
    h = grid.spacing
    corner = np.asarray(origin) * h
    inside = [corner + rng.uniform(0.5, np.asarray(grid.cells) - 0.5, size=(count, 3)) * h]
    points = jnp.asarray(inside[0], dtype=dtype)
    x = fields([grid.padded_shape(c) for c in range(3)], rng, dtype)
    y = jnp.asarray(rng.normal(size=(count, 3)), dtype=dtype)
    bx, _ = patch_read(grid, origin, x, points)
    bty = planted(patch_read_transpose(grid, origin, points, y))
    error, line = identity_error(((bx,), (y,)), (x, bty), dtype)
    assert error <= line, ("point", error, line)
    owners = rng.integers(0, count, size=4 * count)
    owners[:count] = np.arange(count)
    weights = rng.uniform(0.1, 1.0, size=owners.size) * h * h
    areas = np.bincount(owners, weights=weights, minlength=count)
    quadrature = Quadrature(jnp.asarray(corner + rng.uniform(0.5, np.asarray(grid.cells) - 0.5, size=(owners.size, 3)) * h,
                                        dtype=dtype), jnp.asarray(weights, dtype=dtype),
                            jnp.asarray(owners, dtype=jnp.int32), jnp.asarray(areas, dtype=dtype))
    cx, _ = cell_read(grid, origin, x, quadrature)
    cty = planted(cell_read_transpose(grid, origin, quadrature, y))
    error, line = identity_error(((cx,), (y,)), (x, cty), dtype)
    assert error <= line, ("cell", error, line)
