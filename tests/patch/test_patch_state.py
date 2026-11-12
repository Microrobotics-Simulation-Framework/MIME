"""The patch's state and its small rules: the held field moved by whole cells at a snap, the uncovered slab taken from
the coarse grid's fill (gate 10's slab, here exactly); the snapped origin; the whole-cell dilations and their blend
weight; the build report's numbers and the window's lines (flags, never refusals)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.flow.fvm._patch import RefinementPatch, moved, patch_numbers, rule_two_cell, snapped_origin, \
    whole_cell_dilations


@pytest.fixture(autouse=True)
def float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.mark.parametrize("by", [(1, 0, 0), (0, -2, 1), (3, 1, -1), (0, 0, 0)])
def test_a_held_field_moves_by_whole_cells_and_the_uncovered_slab_is_the_fill(by):
    patch = RefinementPatch((6, 7, 8), 1e-4)
    grid = patch.grid()
    rng = np.random.default_rng(sum(by) + 7)
    field = tuple(jnp.asarray(rng.normal(size=grid.interior_shape(c))) for c in range(3))
    fill = tuple(jnp.asarray(rng.normal(size=grid.interior_shape(c))) for c in range(3))
    found = moved(field, jnp.asarray(by, jnp.int32), fill)
    for c in range(3):
        old, new, refill = np.asarray(field[c]), np.asarray(found[c]), np.asarray(fill[c])
        for index in np.ndindex(new.shape):
            source = tuple(i + b for i, b in zip(index, by))
            inside = all(0 <= s < n for s, n in zip(source, old.shape))
            assert new[index] == (old[source] if inside else refill[index])


def test_the_origin_snaps_to_whole_fine_cells_with_the_box_centred():
    patch = RefinementPatch((20, 20, 20), 5e-5)
    centre = jnp.asarray([1.0e-3 + 0.37 * 5e-5, 2.0e-3, 0.5e-3 + 0.51 * 5e-5])
    origin = np.asarray(snapped_origin(patch, centre))
    assert origin.dtype == np.int32
    assert np.all(np.abs((origin + 10) * 5e-5 - np.asarray(centre)) <= 0.5 * 5e-5 + 1e-15)


def test_the_two_whole_cell_dilations_and_their_weight_are_continuous_and_carry_the_derivative():
    patch = RefinementPatch((8, 8, 8), 2e-5)
    for dilation, lower, weight in ((0.0, 0, 0.0), (1.4 * 2e-5, 1, 0.4), (2.999 * 2e-5, 2, 0.999)):
        found = whole_cell_dilations(patch, jnp.asarray(dilation))
        assert int(found[0]) == lower and abs(float(found[1]) - weight) <= 1e-9
    derivative = jax.grad(lambda d: whole_cell_dilations(patch, d)[1])(jnp.asarray(1.4 * 2e-5))
    assert abs(float(derivative) - 1.0 / 2e-5) <= 1e-6 / 2e-5


def test_the_numbers_flag_a_contact_below_the_window_and_never_refuse_it():
    a = 0.7e-3
    cell = rule_two_cell(a)
    patch = RefinementPatch((110, 110, 110), cell)
    inside = patch_numbers(patch, 156.25e-6, a, a, 2.0 * cell, 0.25 * a)
    assert inside.radius_holds and inside.handover_holds
    assert abs(inside.local_radius_cells - 20.0) <= 1e-12 and abs(inside.handover_radii - 0.1) <= 1e-12
    assert abs(inside.ratio - 156.25e-6 / cell) <= 1e-12
    below = patch_numbers(patch, 156.25e-6, a, 0.5 * a, 2.0 * cell, 0.25 * a)
    assert not below.radius_holds and not below.handover_holds
    assert inside.read_holds
    pointed = RefinementPatch((110, 110, 110), cell, read="point")
    assert not patch_numbers(pointed, 156.25e-6, a, a, 2.0 * cell, 0.25 * a).read_holds
    assert patch_numbers(pointed, 156.25e-6, a, a, 2.0 * cell, 0.1 * a).read_holds
