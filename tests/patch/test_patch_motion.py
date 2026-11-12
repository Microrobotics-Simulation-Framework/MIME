"""The patch moving with a body (gates 3, 5, 6 and 11 at a CI size: MIME's 338-point sphere at 6 fine cells across its
radius, the patch the sphere plus 4 cells a side, a plane wall on the fine lattice's faces, its exact given values in
the program). The metric is the sphere's first reflection, ``m = sum F . v(x)`` over its force points for its free
normal-translation forces ``F``: one patch solve a pose.

- gate 3: 50 poses with 18 snaps trace the step once; control ``larger``, a patch two cells larger, retraces;
- gate 5: a snap moves ``m`` by a small share of it (read across the corner's flip, not over the sweep's interval),
  and between snaps its derivative carries no period-one wobble; controls ``continuous``, the wall moved with the body
  by a fraction of a cell (the lattice moving through the band), and ``stale_given``, the given values read at the
  corner of 0.04 cells before (a snap that moves the box without re-reading its faces);
- gate 6: the blend of the two whole-cell dilations is continuous across a whole cell and its derivative in the
  dilation agrees with differences; control ``dilation_stopped``;
- gate 11: ``m``'s derivative in the body's position between snaps agrees with central differences, with the point
  read and the cell read (the default, the review's M3); control ``read_stopped``, the points' derivative stopped."""

import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).parent))
from lubrication_cases import RADIUS  # noqa: E402
from patch_cases import mrs_sphere  # noqa: E402

from mime.nodes.flow.fvm._patch import RefinementPatch, blended_reflection, patch_reflection, patch_stopping, \
    placed, snapped_origin, whole_cell_dilations  # noqa: E402
from mime.nodes.flow.fvm._patch_transfers import Quadrature  # noqa: E402
from mime.nodes.flow.fvm._solve import REPORT_SIZE  # noqa: E402
from mime.nodes.hydro.mrs._plane import plane_image  # noqa: E402

CELLS = 6.0
#: Gate 5's lines at this size, regression lines set between the clean sweep's and the plant's (the operating point's
#: are the probes'): the wobble metric, and a snap's jump over the metric, read across the flip of the snapped corner
#: (``SNAP_STRADDLE`` fine cells either side, where the slope's share is under 1e-7 of the metric): clean 1.7e-4, the
#: stale_given plant 1.7e-2.
WOBBLE_LINE = 1e-2
SNAP_LINE = 2e-3
SNAP_STRADDLE = 1e-6
MARGIN = 4
VISCOSITY = 1e-3


@pytest.fixture(scope="module", autouse=True)
def float64_module():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def plant():
    return os.environ.get("PATCH_PLANT") or None


class Plane:
    region = "wall"

    def __init__(self, at):
        self.at = at

    def signed_distance(self, points):
        return self.at - points[..., 2]


def setting(extra=0, read="point"):
    sphere = mrs_sphere(0.25)
    h = RADIUS / CELLS
    n = int(math.ceil(2 * (CELLS + MARGIN))) + extra
    patch = RefinementPatch((n, n, n), h, read=read)
    free = sla.lu_solve(sla.lu_factor(sphere.matrix / VISCOSITY), np.tile([0.0, 0.0, 1e-3], sphere.count))
    forces = jnp.asarray(free.reshape(-1, 3) * sphere.area[:, None])
    return patch, sphere, forces


def exact_given(patch, origin, sources, eps2, forces, wall):
    """The plane's exact given values at the patch's faces: MIME's regularised image above the wall, less the forces'
    own regularised Stokeslets below (in the program)."""
    grid = patch.grid()
    corner = origin.astype(sources.dtype) * patch.spacing
    out = []
    for c in range(3):
        at = grid.given_points(c, sources.dtype) + corner
        parameters = {"normal": jnp.asarray([0.0, 0.0, 1.0]), "offset": jnp.asarray(wall)}
        image = jnp.einsum("tsij,sj->ti", plane_image(at, sources, parameters, VISCOSITY, eps2).matrix, forces)[:, c]
        r = at[:, None, :] - sources[None, :, :]
        r2 = jnp.sum(r * r, axis=-1)
        s = 1.0 / (8.0 * math.pi * VISCOSITY * (r2 + eps2[None, :]) ** 1.5)
        free = jnp.sum(s * ((r2 + 2.0 * eps2[None, :]) * forces[None, :, c] + r[..., c]
                            * jnp.sum(r * forces[None, :, :], axis=-1)), axis=1)
        out.append(jnp.where(at[:, 2] > wall, image, -free))
    return (out[0], out[1], out[2])


TRACES = {"count": 0}


def metric_function(patch, sphere, forces, wall_offset=0.0):
    """``m(centre, dilation)``: the first reflection at a pose, the wall at ``z = wall_offset`` (m) dilated by the
    seam's continuous ``dilation`` (m) through the whole-cell blend."""
    body = jnp.asarray(sphere.force)
    cells = jnp.asarray(sphere.points)
    eps2 = jnp.asarray(sphere.eps_squared)
    stopping = patch_stopping(patch, jnp.float64, tolerance=1e-11)

    def m(centre, dilation):
        TRACES["count"] += 1
        origin = snapped_origin(patch, centre)
        points = body + centre
        quadrature = None
        at_cells = cells + centre
        if plant() == "read_stopped":
            points = jax.lax.stop_gradient(points)
            at_cells = jax.lax.stop_gradient(at_cells)
        if patch.read == "cell":
            quadrature = Quadrature(at_cells, jnp.asarray(sphere.weights), jnp.asarray(sphere.owners, dtype=jnp.int32),
                                    jnp.asarray(sphere.area))
        if plant() == "dilation_stopped":
            dilation = jax.lax.stop_gradient(dilation)
        lower, weight = whole_cell_dilations(patch, dilation)
        placements = tuple(placed(patch, [Plane(wall_offset)], origin, lower + k, points, eps2, VISCOSITY,
                                  quadrature=quadrature) for k in (0, 1))
        # The given values are the undilated wall's, as the coarse grid hands them (its solve is untouched).
        # The stale_given plant reads the given values at the corner the patch had 0.04 cells of motion before: a snap
        # that moves the box without re-reading its faces (its second flip inside the sweep's snap interval).
        at = snapped_origin(patch, centre - jnp.asarray([0.0, 0.0, 0.04 * patch.spacing])) \
            if plant() == "stale_given" else origin
        given = exact_given(patch, at, points, eps2, forces, wall_offset)
        values, _, _ = blended_reflection(patch, placements, weight, given, forces, None, stopping,
                                          jnp.zeros(REPORT_SIZE))
        return jnp.sum(forces * values)

    return m


def pose(fraction_cells_up, along=0.0):
    """The sphere's centre (m) with its surface 0.3 radii above the wall plus a share of a fine cell up, offset along
    the wall by ``along`` fine cells, and (0.37, 0.21) fine cells off the lattice laterally."""
    h = RADIUS / CELLS
    return jnp.asarray([(0.37 + along) * h, 0.21 * h, RADIUS * 1.3 + fraction_cells_up * h])


def test_gate3_fifty_poses_with_snaps_trace_the_step_once():
    patch, sphere, forces = setting(2 if plant() == "larger" else 0)
    step = jax.jit(metric_function(patch, sphere, forces))
    TRACES["count"] = 0
    if plant() == "larger":
        small, _, _ = setting()
        jax.jit(metric_function(small, sphere, forces))(pose(0.0), jnp.asarray(0.0)).block_until_ready()
    origins = set()
    for k in range(50):
        centre = pose(0.0, along=0.37 * k)
        origins.add(tuple(np.asarray(snapped_origin(patch, centre)).tolist()))
        step(centre, jnp.asarray(0.0)).block_until_ready()
    assert len(origins) >= 18
    assert TRACES["count"] == 1, TRACES["count"]


def period_one_amplitude(values, step, keep=None):
    """The period-one (one fine cell) part of a derivative sampled every ``step`` cells (the intervals ``keep``
    marks), over the derivative's mean magnitude, after a quadratic trend is removed (the window spike's wobble
    metric)."""
    derivative = np.diff(values) / step
    x = (np.arange(derivative.size) + 0.5) * step
    if keep is not None:
        derivative, x = derivative[keep], x[keep]
    basis = np.stack([np.ones_like(x), x, x * x, np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)], axis=1)
    coefficients, *_ = np.linalg.lstsq(basis, derivative, rcond=None)
    return float(np.hypot(coefficients[3], coefficients[4]) / np.mean(np.abs(derivative)))


def snap_point(patch, low, high):
    """The offset (fine cells) between ``low`` and ``high`` where the snapped corner flips, by bisection to 1e-12."""
    start = np.asarray(snapped_origin(patch, pose(low)))
    for _ in range(60):
        middle = 0.5 * (low + high)
        if np.array_equal(np.asarray(snapped_origin(patch, pose(middle))), start):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def test_gate5_a_snap_moves_m_a_little_and_between_snaps_its_derivative_is_smooth():
    patch, sphere, forces = setting()
    m = jax.jit(metric_function(patch, sphere, forces))
    h = RADIUS / CELLS
    step = 0.125
    offsets = np.arange(17) * step
    if plant() == "continuous":
        values = [float(m(pose(0.0), jnp.asarray(-u * h))) for u in offsets]
    else:
        values = [float(m(pose(u), jnp.asarray(0.0))) for u in offsets]
    snaps = [k for k in range(1, len(offsets)) if plant() != "continuous" and not np.array_equal(
        np.asarray(snapped_origin(patch, pose(offsets[k - 1]))), np.asarray(snapped_origin(patch, pose(offsets[k]))))]
    keep = np.asarray([k + 1 not in snaps for k in range(len(offsets) - 1)])
    amplitude = period_one_amplitude(np.asarray(values), step, keep)
    print("gate 5 wobble", amplitude, "snaps", snaps)
    assert amplitude <= WOBBLE_LINE, amplitude
    if plant() in (None, "stale_given"):
        assert snaps, "the sweep crosses no snap"
        flip = snap_point(patch, offsets[snaps[0] - 1], offsets[snaps[0]])
        below = float(m(pose(flip - SNAP_STRADDLE), jnp.asarray(0.0)))
        above = float(m(pose(flip + SNAP_STRADDLE), jnp.asarray(0.0)))
        print("gate 5 snap", abs(above - below) / abs(below), "at", flip)
        assert abs(above - below) <= SNAP_LINE * abs(below), (above, below)


def test_gate6_the_blend_is_continuous_across_a_whole_cell_and_its_derivative_agrees():
    patch, sphere, forces = setting()
    m = metric_function(patch, sphere, forces)
    h = RADIUS / CELLS
    centre = pose(0.3)
    value = jax.jit(m)
    across = [float(value(centre, jnp.asarray((1.0 + s) * h))) for s in (-1e-7, 1e-7)]
    assert abs(across[0] - across[1]) <= 1e-5 * abs(across[0])
    at = jnp.asarray(1.4 * h)
    _, tangent = jax.jit(lambda d: jax.jvp(lambda q: m(centre, q), (d,), (jnp.ones_like(d),)))(at)
    step = 1e-3 * h
    difference = (float(value(centre, at + step)) - float(value(centre, at - step))) / (2 * step)
    assert abs(float(tangent) - difference) <= 1e-3 * abs(difference), (float(tangent), difference)


@pytest.mark.parametrize("read", ["point", "cell"])
def test_gate11_the_reflections_derivative_in_the_bodys_position_agrees_with_differences(read):
    patch, sphere, forces = setting(read=read)
    m = metric_function(patch, sphere, forces)
    h = RADIUS / CELLS
    centre = pose(0.3)
    direction = jnp.asarray([0.3, -0.2, 1.0])
    _, tangent = jax.jit(lambda c: jax.jvp(lambda q: m(q, jnp.asarray(0.0)), (c,), (direction,)))(centre)
    value = jax.jit(m)
    step = 1e-6 * h
    difference = (float(value(centre + step * direction, jnp.asarray(0.0)))
                  - float(value(centre - step * direction, jnp.asarray(0.0)))) / (2 * step)
    assert abs(float(tangent) - difference) <= 1e-5 * abs(difference), (float(tangent), difference)
