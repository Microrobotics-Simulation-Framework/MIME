"""Gates 2 and 3 on straight periodic tubes driven by a body force on the fluid: Poiseuille's core curvature (aligned and
tilted, time-stepped and through the steady solve) with the 1 % viscosity plant, and the quarter-cell wall plant judged by
the wall's measured shift against the band's planar calibration.

The oracles remove the wall's offset (B10 Q-b): the core curvature of a least-squares paraboloid over ``r < 0.6 R`` is
``G / (4 nu)`` wherever the wall sits; the wall's own position is judged against the planar calibration, the shift the
same band gives a plane channel at the same spacing, averaged over the wall's sub-cell positions."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import DRIVES, Waveform, driven, in_dtype, run, tube_node

from mime.nodes.flow.fvm import ImplicitFlowNode, Jacobi, MACGrid, Stopping, band_named, periodic_tube_layout, \
    projected_solve, wall_fields
from mime.nodes.flow.fvm._box import same_discretisation
from mime.scene._sampling import counted

G, NU = 8.0, 1.0
#: The gate's line on the core curvature's viscosity (S1's, B10).
CURVATURE_LINE = 5e-3
#: The wall plant's line: the measured shift against the planar calibration, in cells (S1: 20 of 20 for (2h, 0.3h)).
SHIFT_LINE = 0.15


def _velocity(state):
    return tuple(np.asarray(state[f"velocity_{c}"], np.float64) for c in "xyz")


def core_viscosity(node: ImplicitFlowNode, state, g: float = G) -> float:
    """``G / (4 k)`` from the paraboloid ``a + b x + c y - k r^2`` fitted to the axial velocity at the cells' centres
    within ``0.6 R`` of the axis (the axis's own direction, so tilted tubes too)."""
    c = node.constants()
    h = c["spacing"]
    grid = MACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), h)
    u = _velocity(state)
    centred = [0.5 * (u[0] + np.roll(u[0], -1, 0)), 0.5 * (u[1] + np.roll(u[1], -1, 1)),
               0.5 * (u[2] + np.roll(u[2], -1, 2))]
    axis = np.asarray(c["tube_axis"], np.float64)
    axis = axis / np.linalg.norm(axis)
    axial = sum(axis[k] * centred[k] for k in range(3))
    points = np.asarray(grid.cell_points("float64"), np.float64)
    extents = np.asarray(grid.cells) * h
    offset = points - np.asarray(c["tube_centre"])
    offset = offset - np.round(offset / extents * np.asarray([1, 1, 0])) * extents * np.asarray([1, 1, 0])
    along = offset @ axis
    radial = offset - along[..., None] * axis
    r2 = np.sum(radial ** 2, axis=-1)
    inside = r2 < (0.6 * c["tube_radius"]) ** 2
    e1 = np.cross(axis, [0.0, 1.0, 0.0] if abs(axis[1]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    x, y = radial @ e1, radial @ e2
    design = np.stack([np.ones(int(inside.sum())), x[inside], y[inside], -r2[inside]], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, axial[inside], rcond=None)
    return g / (4.0 * coefficients[3])


@functools.cache
def _steady_state(dtype: str, offset: tuple[float, float], lattice: tuple[int, int], scheme: str, plant: float,
                  dh: int = 16):
    with in_dtype(dtype):
        if lattice == (0, 1):
            node = tube_node(dh, offset=offset, dt=0.5, convection=False, scheme=scheme)
        else:
            layout = periodic_tube_layout(0.5, 1.0, cells_across=dh, lattice=lattice, offset_cells=offset)
            node = driven(ImplicitFlowNode("flow", 0.5, cells_x=layout.cells[0], cells_y=layout.cells[1],
                                           cells_z=layout.cells[2], spacing=layout.spacing, viscosity=NU, density=1.0,
                                           tube_centre=layout.centre, tube_axis=layout.axis,
                                           tube_radius=layout.radius, convection=False, scheme=scheme,
                                           periodic_images="transverse"), Waveform(steady=G, along=layout.axis))
        if plant != 1.0:
            original = ImplicitFlowNode.operator_of
            ImplicitFlowNode.operator_of = lambda self, p, grid, sigma: original(  # type: ignore[method-assign]
                self, {**p, "viscosity": plant * p["viscosity"]}, grid, sigma)
        try:
            state = run(node, 40 if scheme == "imex_euler" else 2)
        finally:
            if plant != 1.0:
                ImplicitFlowNode.operator_of = original  # type: ignore[method-assign]
        return node, state


OFFSETS = ((0.0, 0.0), (0.25, 0.0), (0.37, 0.19), (0.5, 0.5))


@pytest.mark.parametrize("offset", OFFSETS)
def test_poiseuilles_core_curvature_gives_the_viscosity(dtype, offset):
    node, state = _steady_state(dtype, offset, (0, 1), "imex_euler", 1.0)
    assert abs(core_viscosity(node, state) / NU - 1.0) < CURVATURE_LINE


@pytest.mark.parametrize("offset", OFFSETS)
def test_control_a_one_percent_viscosity_plant_fails_the_core_curvature(dtype, offset):
    node, state = _steady_state(dtype, offset, (0, 1), "imex_euler", 1.01)
    assert abs(core_viscosity(node, state) / NU - 1.0) >= CURVATURE_LINE


@pytest.mark.parametrize("offset", OFFSETS[:2])
def test_the_steady_solve_is_the_time_stepped_fixed_point(dtype, offset):
    stepped_node, stepped = _steady_state(dtype, offset, (0, 1), "imex_euler", 1.0)
    steady_node, steady = _steady_state(dtype, offset, (0, 1), "steady", 1.0)
    assert abs(core_viscosity(steady_node, steady) / NU - 1.0) < CURVATURE_LINE
    a, b = _velocity(stepped), _velocity(steady)
    difference = math.sqrt(sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b, strict=True)))
    size = math.sqrt(sum(float(np.sum(y ** 2)) for y in b))
    assert difference / size < (1e-6 if dtype == "float64" else 1e-4)


@pytest.mark.parametrize("offset", OFFSETS[:2])
def test_poiseuilles_core_curvature_on_a_tilted_tube(dtype, offset):
    node, state = _steady_state(dtype, offset, (1, 3), "steady", 1.0)
    assert abs(core_viscosity(node, state) / NU - 1.0) < CURVATURE_LINE


def test_control_a_one_percent_viscosity_plant_fails_on_the_tilted_tube(dtype):
    node, state = _steady_state(dtype, (0.0, 0.0), (1, 3), "steady", 1.01)
    assert abs(core_viscosity(node, state) / NU - 1.0) >= CURVATURE_LINE


# -- the wall plant ----------------------------------------------------------------------------------------------------

class Slab:
    """A plane channel's solid, ``|x - centre| - half_width`` (positive in the solid): the planar calibration's wall."""

    def __init__(self, centre: float, half_width: float):
        self.centre, self.half_width = centre, half_width

    def signed_distance(self, points):
        return jnp.abs(points[..., 0] - self.centre) - self.half_width


@functools.cache
def planar_calibration(dh: int, dtype: str) -> float:
    """The shift (cells) of a plane channel's wall under the default band, from its flux against plane Poiseuille's,
    averaged over eight sub-cell positions of the wall."""
    shifts = []
    with in_dtype(dtype):
        h = 1.0 / dh
        half = 0.5
        for k in range(8):
            cells = 2 * int(math.ceil(half / h)) + 12
            grid = MACGrid((cells, 2, 2), h)
            centre = cells * h / 2 + (k / 8.0) * h
            fields = wall_fields(grid, [Slab(centre, half)], band_named("smooth_2h_0.3h"), 0.1, dtype)
            sigma = tuple(NU * f for f in fields.inverse_permeability)
            force = (jnp.zeros(grid.cells, dtype=dtype), jnp.zeros(grid.cells, dtype=dtype), G * (1.0 - fields.chi[2]))

            def apply(u):
                lap = grid.laplacian(u)
                return tuple(-NU * a + s * b for a, s, b in zip(lap, sigma, u, strict=True))

            diagonal = tuple(-NU * grid.laplacian_diagonal() + s for s in sigma)
            tolerance = 1e-10 if dtype == "float64" else 1e-5
            u, _ = jax.jit(lambda f: projected_solve(grid, apply, diagonal, f, None,
                                                     Stopping(tolerance=jnp.asarray(tolerance, dtype=dtype)),
                                                     jnp.zeros(6, dtype=dtype), Jacobi()))(force)
            inside = np.asarray(fields.signed_distance[2]) < h
            flux = float(np.sum(np.where(inside, np.asarray(u[2], np.float64), 0.0))) * h / (grid.cells[1] * grid.cells[2])
            effective = (3.0 * NU * flux / (2.0 * G)) ** (1.0 / 3.0)
            shifts.append((effective - half) / h)
    return float(np.mean(shifts))


def tube_shift(node: ImplicitFlowNode, state, radius: float) -> float:
    """The tube wall's shift (cells) from its flux against Poiseuille's at ``radius``: ``R_eff = (8 nu Q / (pi G))^(1/4)``."""
    flux = float(state["lumen_flux"])
    effective = (8.0 * NU * flux / (math.pi * G)) ** 0.25
    return (effective - radius) / node.constants()["spacing"]


@functools.cache
def _plant_state(dtype: str, offset: tuple[float, float], quarter: float, dh: int = 16):
    with in_dtype(dtype):
        node = tube_node(dh, offset=offset, dt=0.5, convection=False, scheme="steady")
        planted = driven(ImplicitFlowNode(**{**_arguments(node), "tube_radius": 0.5 + quarter / dh}), DRIVES[node])
        return node, run(planted, 2)


def _arguments(node: ImplicitFlowNode) -> dict:
    c = dict(node.constants())
    for key in ("solve_report",):
        c.pop(key, None)
    return {"name": "flow", "timestep": node.delta_t, **c}


@pytest.mark.parametrize("offset", OFFSETS)
def test_the_tube_walls_shift_matches_the_planar_calibration(dtype, offset):
    node, state = _plant_state(dtype, offset, 0.0)
    assert abs(tube_shift(node, state, 0.5) - planar_calibration(16, dtype)) < SHIFT_LINE


@pytest.mark.parametrize("offset", OFFSETS)
def test_control_a_quarter_cell_outward_wall_plant_fails_the_shift(dtype, offset):
    """The wall a quarter cell outward of where the oracle puts it (the geometry the solver was given, not the one
    judged): never judged by the core curvature, which is blind to it by construction."""
    node, state = _plant_state(dtype, offset, 0.25)
    assert abs(tube_shift(node, state, 0.5) - planar_calibration(16, dtype)) >= SHIFT_LINE


# -- the layout's count slack (M2) ----------------------------------------------------------------------------------

#: The round inputs of the build's cases: (radius, period, cells across, lattice); the walk's and the build's.
ROUND_INPUTS = [(0.5e-3, 0.25e-3, 8, (0, 1)), (0.5e-3, 0.125e-3, 32, (0, 1)), (0.5e-3, 0.25e-3, 16, (1, 3)),
                (0.5, 2.0 / 8, 8, (0, 1))]


@pytest.mark.parametrize(("radius", "length", "across", "lattice"), ROUND_INPUTS)
@pytest.mark.parametrize("moved", ["radius", "length"])
@pytest.mark.parametrize("by", [1e-6, -1e-6, 5e-5, -5e-5])
def test_a_round_input_moved_within_the_count_slack_keeps_its_discretisation(radius, length, across, lattice, moved,
                                                                          by):
    """At round inputs every count is a whole ratio, so a ceiling would sit on its own boundary: the two-builds walk's
    relative 1e-6 and a central difference's step under the slack (5e-5), up or down, in the radius or the period, keep
    the cells (a move of the slack itself lands on the next count's boundary). The radius
    of an aligned tube keeps the spacing too (one discretisation, as the dQ/dR gate's two sides need); the period's
    spacing is the period over its cells, and a tilted tube's the radius's, so there the cells alone, the spacing a
    leaf of the step."""
    base = periodic_tube_layout(radius, length, cells_across=across, lattice=lattice)
    scale = 1.0 + by
    other = periodic_tube_layout(radius * scale if moved == "radius" else radius,
                                 length * scale if moved == "length" else length, cells_across=across, lattice=lattice)
    if lattice == (0, 1) and moved == "radius":
        assert same_discretisation(base, other), (base.cells, other.cells, base.spacing, other.spacing)
    else:
        assert base.cells == other.cells


def test_control_a_bare_ceiling_moves_the_cells_at_a_round_input(monkeypatch):
    """The control: with no slack (a bare ceiling), the walk's 1e-6 up in the radius adds cells across, and in the
    period a cell along it, as the review measured (15x15x2 to 17x17x2; 19x19x3)."""
    import mime.nodes.flow.fvm._box as box
    import mime.scene._sampling as sampling

    monkeypatch.setattr(sampling, "COUNT_SLACK", 0.0)
    base = box.periodic_tube_layout(0.5e-3, 0.25e-3, cells_across=8)
    assert base.cells != box.periodic_tube_layout(0.5e-3 * (1 + 1e-6), 0.25e-3, cells_across=8).cells
    assert base.cells != box.periodic_tube_layout(0.5e-3, 0.25e-3 * (1 + 1e-6), cells_across=8).cells


def test_the_count_slack_is_a_relative_share_of_the_ratio():
    """The MRS's rule (``counted``, one helper for both): a whole ratio keeps its count; the next starts a relative
    COUNT_SLACK above it."""
    assert counted(19.0 * (1 + 0.9e-4)) == 19 and counted(19.0 * (1 + 1.1e-4)) == 20
    assert counted(18.5) == 19 and counted(19.0) == 19
