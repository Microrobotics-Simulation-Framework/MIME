"""The patch on the plane oracle at a CI size (8 fine cells across the radius, the sphere plus 4 a side; the probes
take the operating point, 20 cells): gate 1 (the patch's share of the normal entry, against the same MRS with the exact
image read at its points; control ``PATCH_PLANT=no_source``, the forces' own field left out), gate 9 (a warm start
gives the cold answer and iterates; control ``accept_start``, a start within the tolerance accepted with no
iteration), gate 12 (float32's solve reaches its floor unflagged and within a line of float64's; control
``below_floor``, a tolerance under the floor, which must stop flagged)."""

import dataclasses
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from patch_cases import Case, built_patch, image_reader, normal_k, patch_reader, regularised_free, image_at, \
    resistance  # noqa: E402
from plane_oracles import brenner  # noqa: E402

from mime.nodes.flow.fvm._patch import patch_reflection, patch_stopping  # noqa: E402
from mime.nodes.flow.fvm._solve import REPORT_FIELDS, REPORT_SIZE  # noqa: E402

CASE = Case(spacing=0.25, cells=8.0, gap=0.3, margin=4)
#: Gate 1's band at this size (2.4 fine cells): the window spike's -4.2 % at 2 cells and -1.2 % at 3, widened.
SHARE_BAND = (-6.0, -1.0)


@pytest.fixture(scope="module")
def float64_module():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def test_gate1_the_patchs_share_of_the_normal_entry_is_the_bands(float64_module):
    image = normal_k(resistance(CASE, image_reader(CASE, "point"), ("Uz",)))
    reader, record = patch_reader(CASE)
    patch = normal_k(resistance(CASE, reader, ("Uz",)))
    share = 100.0 * (patch - image) / brenner(CASE.gap)
    assert SHARE_BAND[0] <= share <= SHARE_BAND[1], (share, image, patch)
    assert min(record["iterations"]) >= 1


def column_case(dtype):
    """One column at the oracle's pose: the forces of a unit normal translation in free space, the plane's exact
    given values for them."""
    patch, placement, centre, sphere = built_patch(CASE)
    placement = jax.tree.map(lambda a: a.astype(dtype) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating)
                             else a, placement)
    grid = patch.grid()
    sources = sphere.force + centre
    forces = np.tile([0.0, 0.0, 1e-9], (sphere.count, 1))
    given = []
    for c in range(3):
        at = np.asarray(grid.given_points(c, jnp.float64)) + np.asarray(placement.origin) * patch.spacing
        above = at[:, 2] > 0.0
        values = np.zeros(at.shape[0])
        values[above] = image_at(at[above], sources, sphere.eps_squared, forces, 0.0, 1e-3)[:, c]
        values[~above] = -regularised_free(at[~above], sources, sphere.eps_squared, forces, 1e-3)[:, c]
        given.append(jnp.asarray(values, dtype=dtype))
    return patch, placement, (given[0], given[1], given[2]), jnp.asarray(forces, dtype=dtype)


def report(found):
    return dict(zip(REPORT_FIELDS, np.asarray(found.report).tolist()))


def test_gate9_a_warm_start_gives_the_cold_answer_and_iterates(float64_module):
    patch, placement, given, forces = column_case(jnp.float64)
    stopping = patch_stopping(patch, jnp.float64)
    if os.environ.get("PATCH_PLANT") == "accept_start":
        stopping = dataclasses.replace(stopping, least_iterations=0)
    solve = jax.jit(lambda b, f, s: patch_reflection(patch, placement, b, f, s, stopping, jnp.zeros(REPORT_SIZE)))
    first = solve(given, forces, None)
    # The next problem a hundredth of the tolerance away: the warm start is within the tolerance of its answer.
    nudged = tuple(g * (1.0 + 1e-10) for g in given)
    cold = solve(nudged, forces, None)
    warm = solve(nudged, forces, first.field)
    held = report(warm)
    scale = float(jnp.max(jnp.abs(cold.values)))
    assert float(jnp.max(jnp.abs(warm.values - cold.values))) <= 10.0 * stopping.tolerance * scale
    assert held["iterations"] >= 1 and held["residual"] <= stopping.tolerance, held
    assert held["iterations"] < report(cold)["iterations"]


def test_gate12_float32_reaches_its_floor_unflagged_within_a_line_of_float64(float64_module):
    patch, placement, given, forces = column_case(jnp.float64)
    exact = patch_reflection(patch, placement, given, forces, None, patch_stopping(patch, jnp.float64),
                             jnp.zeros(REPORT_SIZE))
    jax.config.update("jax_enable_x64", False)
    try:
        patch, placement, given, forces = column_case(jnp.float32)
        stopping = patch_stopping(patch, jnp.float32)
        if os.environ.get("PATCH_PLANT") == "below_floor":
            stopping = dataclasses.replace(stopping, tolerance=1e-9, at_floor=False, cap=600)
        found = patch_reflection(patch, placement, given, forces, None, stopping, jnp.zeros(REPORT_SIZE,
                                                                                            dtype=jnp.float32))
        held = report(found)
        values = np.asarray(found.values, dtype=np.float64)
    finally:
        jax.config.update("jax_enable_x64", True)
    assert held["at_cap"] == 0 and held["stagnated"] == 0 and held["not_finite"] == 0, held
    scale = float(np.max(np.abs(np.asarray(exact.values))))
    assert float(np.max(np.abs(values - np.asarray(exact.values)))) <= 1e-4 * scale


def test_gate8_the_cell_reads_normal_overshoot_near_contact_is_under_its_line(float64_module):
    """The read near contact (decided 4 (3)): the MRS with the plane's exact image read over each force point's cell, at
    a gap of 0.1 radii and MIME's default force spacing, overshoots Brenner's normal entry by under 5 % (measured +2.3 %
    on the probe); read at the points it is +47 % (the control ``point_read``)."""
    case = Case(spacing=0.25, cells=8.0, gap=0.1, margin=4)
    read = "point" if os.environ.get("PATCH_PLANT") == "point_read" else "cell"
    found = normal_k(resistance(case, image_reader(case, read), ("Uz",)))
    overshoot = 100.0 * (found / brenner(case.gap) - 1.0)
    assert 0.0 <= overshoot <= 5.0, overshoot
