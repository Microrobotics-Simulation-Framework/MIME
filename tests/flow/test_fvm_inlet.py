"""The prescribed inlet (the maintainer's inflow decision (A), 2026-11-12; predictions in
``predictions_prescribed_inlet.json``): an Inflow's opening holds the developed profile scaled to the law's flux, its
tangential velocity zero, the projector Neumann on its plane; the grid's pieces exact to rounding, the profile's
Stokes-layer form past the series' line, the opening's work booked under its law, the development flag, the
refusals, and the trees' float32 tolerance."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import in_dtype
from open_cases import draw, open_node, point_node, scan, tube_resistance

from mime.nodes.flow._womersley import VALIDITY
from mime.nodes.flow.fvm import _inlet
from mime.nodes.flow.fvm._open import HIGH, LOW, OpeningLayout, OpenMACGrid
from mime.nodes.flow.fvm._open_node import TREE_FLOAT32_TOLERANCE, OpenFlowNode

PLANES = (LOW, HIGH)


def _held_field(grid: OpenMACGrid, seed: int) -> tuple:
    """A divergence-free field whose prescribed plane carries a drawn face: the subspace's projection plus the lift."""
    rng = np.random.default_rng(seed)
    dtype = "float64" if jnp.zeros(()).dtype == jnp.float64 else "float32"
    drawn = tuple(jnp.asarray(rng.standard_normal(grid.shape(k)), dtype=dtype) for k in range(3))
    face = jnp.asarray(rng.standard_normal(grid.cells[:2]), dtype=dtype)
    return tuple(a + b for a, b in zip(grid.project(drawn), grid.prescribed_field(face), strict=True)), face


@pytest.mark.parametrize("plane", PLANES)
def test_the_prescribed_planes_pieces_are_exact(dtype, plane):
    """The lift keeps its face and is divergence-free; the projection zeroes the face and is idempotent; the scheme's
    convective power telescopes to the openings' kinetic fluxes with the prescribed plane's held tangential velocity
    (``OpenMACGrid._given_edges``); each to rounding on drawn fields. The control, the face formula (the edge terms
    left out), misses by far more than rounding on every draw."""
    eps = float(jnp.finfo(jnp.zeros(()).dtype).eps)
    grid = OpenMACGrid((6, 5, 7), 0.1, OpeningLayout(), plane)
    at = 0 if plane == LOW else -1
    for seed in range(4):
        held, face = _held_field(grid, seed)
        lift = grid.prescribed_field(face)
        assert float(jnp.max(jnp.abs(lift[2][..., at] - face))) <= 8 * eps * float(jnp.max(jnp.abs(face)))
        scale = max(float(jnp.max(jnp.abs(a))) for a in held)
        assert float(jnp.max(jnp.abs(grid.divergence(held)))) * grid.spacing <= 200 * eps * scale
        projected = grid.project(held)
        assert float(jnp.max(jnp.abs(projected[2][..., at]))) == 0.0
        again = grid.project(projected)
        assert max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(again, projected, strict=True)) <= 200 * eps * scale
        power = float(grid.inner(grid.convection(held), held))
        kinetic, size = grid.opening_kinetic_fluxes(held)
        assert abs(power + float(jnp.sum(kinetic))) <= 200 * eps * float(jnp.sum(size))
        wb = held[2][..., at]
        formula = (1.0 if plane == LOW else -1.0) * grid.spacing ** 2 * float(jnp.sum(0.5 * wb ** 3))
        others = float(jnp.sum(kinetic)) - float(kinetic[0 if plane == LOW else 1])
        assert abs(power + others - formula) > 1e3 * eps * float(jnp.sum(size))


def test_the_stokes_layer_is_the_series_past_its_line():
    """Past the series' line the harmonic's shape is the Stokes layer's: at alpha 30 and 45 (inside float64's series)
    the two forms, each scaled to unit flux on a plane at 16 cells across, agree within 1e-4 of the shape's peak (the
    module's measured 9.5e-5 at 25 and 2.4e-5 at 50, against SciPy)."""
    with in_dtype("float64"):
        h = 1.0 / 16.0
        x = (jnp.arange(20) + 0.5) * h - 0.625
        points = jnp.stack(jnp.meshgrid(x, x, indexing="ij"), axis=-1)
        fluid = jnp.ones((20, 20))
        rho = jnp.sum(points * points, axis=-1) / 0.25
        for alpha in (30.0, 45.0):
            series = _inlet._scaled(fluid, _inlet.profile(jnp.clip(rho, 0.0, 1.0), alpha ** 2, 80), h * h)
            layer = _inlet._scaled(fluid, _inlet._stokes_layer(jnp.clip(rho, 0.0, 1.0), jnp.asarray(alpha)), h * h)
            peak = float(jnp.max(jnp.hypot(series[0], series[1])))
            gap = float(jnp.max(jnp.hypot(series[0] - layer[0], series[1] - layer[1])))
            assert gap <= 1e-4 * peak, (alpha, gap / peak)
    assert VALIDITY["float32"][0] == 25.0 and VALIDITY["float64"][0] == 50.0


def test_the_prescribed_face_is_the_profile_at_the_laws_flux(dtype):
    """On a drawn open tube every step's face is the profile at the law's flux to rounding, the flux the law's; the
    unit fields are gone with the Inflow's row (one outlet: the field's velocity does not see its pressure level)."""
    node = point_node(draw(3), ledger=False)
    assert node.prescribed_opening() == 0 and node.others() == []
    p = node._step_constants(None)  # noqa: SLF001
    final, _ = scan(node, 12)
    grid = node.open_grid(p)
    profile = node.inlet_profile(p, grid, node.fields(p, grid))
    instant = jnp.asarray(final["steps"]).astype(final["velocity_z"].dtype) * node.delta_t
    face = profile.at(instant)
    eps = float(jnp.finfo(final["velocity_z"].dtype).eps)
    assert float(jnp.max(jnp.abs(final["velocity_z"][..., 0] - face))) <= 64 * eps * float(jnp.max(jnp.abs(face)))
    row = np.asarray(p["opening_parameters"], dtype=np.float64)[0]
    flux = row[0] + row[1] * math.cos(2.0 * math.pi * row[2] * float(instant) + row[3])
    assert abs(float(final["opening_flux"][0]) + flux) <= 1e3 * eps * abs(flux)


@pytest.mark.parametrize("plant", [None, "no_holding_force"])
def test_the_inlets_pressure_is_the_tubes_drop(float64, monkeypatch, plant):
    """Steady Stokes through the open tube at 8 and 16 cells across: the prescribed opening's pressure less the
    outlet's, at the flux a given pressure drop drove through the same tube, is that drop within 3 % (the inlet's
    adjustment: 2.1 % and 1.5 % while writing); the control, the faces' holding force left out of the opening's pressure
    (the field's pressure in the end cells alone), read 4.8 %: the test resolves an error in the inlet's pressure of
    about 1 % of the drop, no finer."""
    if plant:
        monkeypatch.setattr(OpenFlowNode, "given_face_pressures",
                            staticmethod(lambda grid, remainder, pressure, rho: rho * pressure[..., 0]))
    resistance = tube_resistance(2.0)
    misses = []
    for cells in (8, 16):
        driven = open_node(cells, laws=("pressure", "pressure"), parameters=[[100.0, 0, 0, 0], [0.0, 0, 0, 0]],
                           dt=1.0, convection=False, scheme="steady", tolerance=1e-11)
        flux = float(scan(driven, 2)[0]["opening_flux"][1])
        node = open_node(cells, parameters=[[flux, 0.0, 0.0, 0.0], [resistance, 1e-9, resistance, 0.0]], dt=1.0,
                         convection=False, scheme="steady", tolerance=1e-11)
        final, _ = scan(node, 3)
        misses.append(abs(float(final["opening_pressure"][0] - final["opening_pressure"][1]) / 100.0 - 1.0))
    assert (max(misses) <= 0.03) == (plant is None), misses


def _reach_node(reach: float, reynolds: float) -> OpenFlowNode:
    flux = reynolds * math.pi / 4.0
    resistance = tube_resistance(2.0)
    return open_node(8, parameters=[[flux, 0.0, 0.0, 0.0], [resistance, 1.0 / resistance, resistance, 0.0]],
                     dt=1e-3 / max(reynolds, 1.0), inlet_reach=reach)


def test_the_development_flag_fires_inside_its_length_and_is_quiet_past_it(float64):
    """``inlet_development_flagged``: a reach of a tenth of a diameter at Re 100 is inside the measured development
    length (fires); a reach of two hundred diameters at Re 1 is past it (quiet)."""
    fired, _ = scan(_reach_node(0.1, 100.0), 2)
    quiet, _ = scan(_reach_node(200.0, 1.0), 2)
    assert float(fired["inlet_development_flagged"]) == 1.0 and float(fired["inlet_development"]) > 1.0
    assert float(quiet["inlet_development_flagged"]) == 0.0 and 0.0 < float(quiet["inlet_development"]) < 1.0


def test_a_trees_own_end_takes_no_inflow():
    """A tube tree's own end at a plane is not a section the profile is made for (flow.inflow-section, the node's)."""
    with pytest.raises(ValueError, match="flow.inflow-section"):
        OpenFlowNode("flow", 0.01, cells_x=8, cells_y=8, cells_z=8, spacing=0.125, viscosity=1.0, density=1.0,
                     tree_points=[[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]], tree_radii=[[0.3, 0.3]], tree_fillet=0.1,
                     tree_segments=[(0, 1)], openings=("a", "b"), opening_laws=("inflow", "windkessel"),
                     opening_parameters=[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])


def test_float32_on_an_open_tree_takes_its_measured_tolerance():
    """A float32 open vessel walled by a tube tree with no tolerance given solves to 1e-4 (as measured, run
    43171688923); float64 keeps 1e-8, and a given tolerance is held."""
    from test_fvm_region import tree_vessel

    for dtype, expected in (("float32", TREE_FLOAT32_TOLERANCE), ("float64", 1e-8)):
        with in_dtype(dtype):
            node = tree_vessel(ledger=False).build().graph.get_node("water.flow")
            assert float(node.constants()["tolerance"]) == pytest.approx(expected)


def test_an_inflow_sharing_its_plane_is_refused(float64):
    """A region whose root and one end both turn to the low face: the Inflow's opening is prescribed and holds its whole
    plane, so it is refused (flow.inflow-plane), naming the end it shares with."""
    from test_fvm_region import _refused, tree_vessel

    from mime.scene import TubeTree
    from mime.units import mm

    tree = TubeTree(points=mm((0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 2.5, 0.0, 0.8)),
                    segments=((0, 1), (1, 2), (1, 3)), radii=mm((0.5, 0.5, 0.4, 0.4, 0.35, 0.35)))
    assert _refused(tree_vessel(tree=tree, ledger=False)) == ["flow.inflow-plane"]
