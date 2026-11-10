"""The switch both near-singular evaluators share (``mime.nodes.hydro._near_singular``: the wall's self-image and the
near field's cross-kernel, design #308), its mechanics tested once here; each caller's conformance cases are its own
tests' (``tests/near_field/test_reach.py``, ``tests/oracles/test_mrs_near_wall.py``).

- **The blend is C1 across the reach**: one within half the reach, zero beyond it, its value and first derivative
  continuous at both ends of the band. The control, a linear ramp over the same band, has a derivative that jumps.
- **No derivative passes through the selection**: the budget's choice (``nearest_cells``, ``nearest_points``) is
  integers and masks, so a value read through it alone has a zero derivative in the distances, while the blend's
  weight keeps its own.
- **The shortfall**: ``nearest_cells`` reports a cell within the reach left out of the budget, and none where the
  first left out lies beyond it; ``nearest_points`` leaves out a cell whose points do not fit.
- **The two branches agree with nothing in reach**: every weight is zero there, so the near branch returns the far
  one exactly, and the ``cond`` takes the far one."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.hydro._near_singular import blended, nearest_cells, nearest_points, smoothstep, within_reach

REACH = 2.0


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def slope(weight, at: float) -> float:
    return float(jax.grad(lambda d: weight(d, REACH))(jnp.asarray(at)))


def jumps(weight) -> float:
    """The largest jump of ``weight``'s value or derivative across either end of its band."""
    found = 0.0
    for edge in (0.5 * REACH, REACH):
        below, above = edge - 1e-9, edge + 1e-9
        found = max(found, abs(float(weight(jnp.asarray(below), REACH)) - float(weight(jnp.asarray(above), REACH))),
                    abs(slope(weight, below) - slope(weight, above)))
    return found


def test_the_blend_is_one_within_half_the_reach_zero_beyond_and_c1_across_both_ends():
    assert float(smoothstep(jnp.asarray(0.3 * REACH), REACH)) == 1.0
    assert float(smoothstep(jnp.asarray(1.2 * REACH), REACH)) == 0.0
    assert jumps(smoothstep) < 1e-6


def test_control_a_linear_ramp_over_the_band_has_a_derivative_that_jumps():
    def ramp(distance, reach):
        return jnp.clip((reach - distance) / (0.5 * reach), 0.0, 1.0)

    assert jumps(ramp) > 0.5


def test_no_derivative_passes_through_the_selection_and_the_blend_keeps_its_own():
    values = jnp.arange(6.0)
    sizes = jnp.asarray([2, 3, 1, 4, 2, 2])

    def through_cells(distance):
        chosen, short = nearest_cells(distance, 3, REACH)
        return jnp.sum(values[chosen]) + short.astype(distance.dtype)

    def through_points(distance):
        cell, within, valid, fits = nearest_points(distance, sizes, 6)
        return jnp.sum(jnp.where(valid, values[cell] + within, 0.0)) + jnp.sum(fits.astype(distance.dtype))

    distance = jnp.asarray([0.4, 3.0, 1.2, 0.9, 2.5, 1.5])
    assert not np.any(np.asarray(jax.grad(through_cells)(distance)))
    assert not np.any(np.asarray(jax.grad(through_points)(distance)))
    weight = jax.grad(lambda d: jnp.sum(blended(jnp.zeros((1, 3)), jnp.ones((1, 3)), smoothstep(d, REACH))))
    assert float(weight(jnp.asarray([1.5]))[0]) != 0.0


def test_the_budget_of_cells_reports_a_cell_within_the_reach_left_out():
    distance = jnp.asarray([0.4, 3.0, 1.2, 0.9, 2.5, 1.5])
    chosen, short = nearest_cells(distance, 3, REACH)
    assert sorted(np.asarray(chosen).tolist()) == [0, 2, 3] and bool(short), "1.5 lies within the reach, left out"
    assert not bool(nearest_cells(distance, 4, REACH).short), "the first left out, 2.5, lies beyond it"
    assert not bool(nearest_cells(distance, 6, REACH).short)


def test_the_budget_of_points_takes_whole_cells_nearest_first_while_they_fit():
    distance = jnp.asarray([0.4, 3.0, 1.2, 0.9, 2.5, 1.5])
    sizes = jnp.asarray([2, 3, 1, 4, 2, 2])
    fits = np.asarray(nearest_points(distance, sizes, 6).fits)
    assert fits.tolist() == [True, False, False, True, False, False], "cells 0 then 3 fill six slots; 2 does not fit"


def test_with_nothing_in_reach_the_near_branch_returns_the_far_one_exactly():
    far, near = jnp.asarray(np.random.default_rng(7).normal(size=(4, 6))), jnp.ones((4, 6))
    distance = jnp.asarray([REACH, 2.5 * REACH])
    assert np.array_equal(np.asarray(blended(far, near, smoothstep(distance, REACH))), np.asarray(far))
    taken = within_reach(distance, REACH, lambda: near, lambda: far)
    assert np.array_equal(np.asarray(taken), np.asarray(far))
    assert np.array_equal(np.asarray(within_reach(jnp.asarray([0.5, 9.0]), REACH, lambda: near, lambda: far)),
                          np.asarray(near))
