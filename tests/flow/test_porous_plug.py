"""The static porous plug's gates (``tests/flow/predictions_porous_plug.json`` holds their predictions, committed before
they ran): a plug of declared permeability held in the flow's step, sampled through the band onto the faces, its
resistivity added to the wall's on its share of the fluid.

1. **Darcy's pressure drop**: a plug one diameter long filling the tube, in a period of four, gives the series of its
   resistance (Brinkman's, with his wall factor) and the open tube's; control, its permeability read 10 % high.
2. **Brinkman's profile** in a plug filling the period, where ``sqrt(K)`` is resolved; control, the same plant.
3. **The annulus**: a coaxial plug of half the lumen's radius splits the flux as a Brinkman core matched to a Stokes
   annulus; control, its radius read a quarter cell larger.
4. **Resistivities add**: a drawn plug overlapping the wall's band reads the sum at every face; control, the larger of
   the two in place of the sum.
5. **Positivity**: a drawn permeability of zero or below, or not finite, is flagged where it is sampled, refused after
   the run and refused when declared; control, the sampled check removed.
6. **The ledger**: the identity closes with the plug's Darcy loss booked under its law, and in a steady flow that loss
   is the drive's work through the plug; control, the loss booked with its sign flipped.
7. **Gradients**, float64: the box's flux in the permeability, the plug's offset across the axis and its length,
   forward and reverse, against central differences of one discretisation; control, ``stop_gradient`` on the field.
8. **Float32**: gates 1 and 3 within their rounding of float64.

Each plant is switched by ``MIME_PLANT_<NAME>``; each drawn case by ``DETECTION_SEED`` (``porous_cases``)."""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import METERED, Waveform, in_dtype, run, step_of
from hypothesis import given, settings
from hypothesis import strategies as st
from porous_cases import G, LAW, R, box_flux, brinkman_profile, composite, experiment, open_experiment, plant, \
    plug_node, plug_resistance, series_flux, steady

import mime.nodes.flow.fvm._band as band
import mime.nodes.flow.fvm._node as node_module
from mime.build import BuildRefused
from mime.core import ledger as books
from mime.interactions import Darcy
from mime.scene import Plug, Z
from mime.things import Structure
from mime.units import m2, mm, ms
from mime.nodes.flow.fvm import band_named, wall_fields
from mime.testing import ledger_closes

#: The gates' lines, each from the first step's measurements (``predictions_porous_plug.json``'s ``outcomes``).
DARCY_LINE = 0.04
PROFILE_LINE = 0.02
ANNULUS_LINE = 0.03
#: The derivative against the central difference: in the permeability, 1e-6 (the difference's truncation at a relative
#: step of 1e-4); in the geometry, 1e-4, the difference straddling the band's kinks within a step.
PERMEABILITY_LINE, GEOMETRY_LINE = 1e-6, 1e-4
FLOAT32_LINE = 1e-4


def _scaled(K: float) -> float:
    """The permeability the node is given: ``K``, or read high by the plant ``PLUG_K``."""
    return K * (1.0 + float(plant("PLUG_K") or 0.0))


@pytest.mark.parametrize(("root", "interior"), [(0.3, "0.01h"), (1.0, "0.01h"), (0.3, "0.1h"), (0.03, "0.1h")])
def test_a_plug_filling_the_tube_gives_darcys_pressure_drop(float64, root, interior):
    """A plug half a diameter long in a period of two (the probe's arms take one in four, to D/h 64). The flux the run
    reports, ``lumen_flux``, is read, and it is the box's whole flux: past a closing plug the lumen's own mean carries
    the loop through the wall's solid (review B2). Two more plants: the plug's depth not carried into the wall's solid
    (``NO_REACH``, a leak around it through the default interior's solid) and the lumen's own mean reported
    (``LUMEN_LOOP``)."""
    dh, length, period = 16, 0.5, 2.0
    h = 1.0 / dh
    K = (root * h) ** 2
    with pytest.MonkeyPatch.context() as patched:
        if plant("NO_REACH"):
            patched.setattr(band, "wall_reach", lambda phi: jnp.zeros_like(phi))
        if plant("LUMEN_LOOP"):
            patched.setattr(node_module.ImplicitFlowNode, "reported_flux", lambda self, lumen, solid: lumen)
        node = plug_node(dh, [dict(radius=R, length=length, permeability=_scaled(K))], length_cells=int(period * dh),
                         interior=interior)
        state = steady(node)
    q = box_flux(node, state)
    assert float(state["solve_flagged"]) == 0.0
    assert abs(q / series_flux(K, length, period) - 1.0) < DARCY_LINE
    assert abs(float(state["lumen_flux"]) / series_flux(K, length, period) - 1.0) < DARCY_LINE
    # The plug's pressure drop as its face pressures read it, with the drive over its length and its band's two half
    # widths, is Darcy's at the measured flux.
    faces = np.asarray(state["porous_face_pressure"], np.float64)[0]
    drop = faces[0] - faces[1] + G * (length + 2 * h)
    assert abs(drop / (q * plug_resistance(K, length)) - 1.0) < DARCY_LINE


@pytest.mark.parametrize("root", [2.0, 4.0])
def test_where_sqrt_k_is_resolved_the_profile_is_brinkmans(float64, root):
    dh = 16
    h = 1.0 / dh
    K = (root * h) ** 2
    node = plug_node(dh, [dict(radius=R, length=100.0, permeability=_scaled(K))])
    state = steady(node)
    c = node.constants()
    w = np.asarray(state["velocity_z"], np.float64)[:, :, 0]
    xs = (np.arange(c["cells_x"]) + 0.5) * h - c["tube_centre"][0]
    ys = (np.arange(c["cells_y"]) + 0.5) * h - c["tube_centre"][1]
    r = np.hypot(xs[:, None], ys[None, :])
    inside = r < R - 2 * h
    error = np.max(np.abs(w[inside] - brinkman_profile(K, r[inside]))) / float(brinkman_profile(K, 0.0))
    assert error < PROFILE_LINE


@pytest.mark.parametrize("root", [1.0, 3.0])
def test_a_plug_with_an_annular_gap_splits_the_flux_as_a_brinkman_core_in_a_stokes_annulus(float64, root):
    dh = 16
    h = 1.0 / dh
    K = (root * h) ** 2
    radius = 0.5 * R + (0.25 * h if plant("PLUG_RADIUS") else 0.0)
    node = plug_node(dh, [dict(radius=radius, length=100.0, permeability=K)], interior="0.01h")
    state = steady(node)
    total, _ = composite(K, 0.5 * R)
    assert abs(float(state["lumen_flux"]) / total - 1.0) < ANNULUS_LINE
    # The wall's leak against B3's line (at the 0.1h interior it reads 1.1e-3 to 1.8e-3 here, flagged).
    assert float(state["bypass_flagged"]) == 0.0


@settings(deadline=None)
@given(root=st.floats(0.01, 10.0), radius=st.floats(-0.9, 2.0), offset=st.floats(0.0, 0.5),
       length=st.floats(0.5, 20.0))
def test_a_plug_overlapping_the_walls_band_reads_the_sum_of_the_two_at_every_face(root, radius, offset, length):
    """Drawn: the plug's radius within -0.9 to 2 cells of the lumen's, so its band overlaps the wall's."""
    with in_dtype("float64"):
        _sum_read(root, radius, offset, length)


def _sum_read(root: float, radius: float, offset: float, length: float) -> None:
    dh = 16
    h = 1.0 / dh
    node = plug_node(dh, [dict(radius=R + radius * h, length=length, permeability=(root * h) ** 2,
                               offset=(offset * h, 0.0))], length_cells=4)
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    with pytest.MonkeyPatch.context() as patched:
        if plant("POROUS_MAX"):
            patched.setattr(band, "resistivities_added",
                            lambda walls, porous: jnp.maximum(walls, sum(porous[1:], start=porous[0])))
        fields = node.fields(p, grid)
    c = node.constants()
    walls = wall_fields(grid, node.solids(p, grid), band_named(c["band"]), 0.1, jnp.float64)
    (region,) = node.porous_regions(p, grid)
    for component in range(3):
        alone = band.porous_field(region, grid.face_points(component, jnp.float64), walls.chi[component],
                                  walls.signed_distance[component], band_named(c["band"]).width * h)
        expected = np.asarray(walls.inverse_permeability[component]) + np.asarray(alone)
        assert np.array_equal(np.asarray(fields.inverse_permeability[component]), expected)
    assert np.any(np.asarray(fields.porous[0][2]) * np.asarray(fields.walls[2]) > 0.0)


@settings(deadline=None)
@given(K=st.one_of(st.just(0.0), st.just(-0.0), st.floats(-1.0, -1e-30), st.just(math.nan), st.just(-math.inf)))
def test_a_permeability_of_zero_or_below_is_flagged_where_it_is_sampled(K):
    with in_dtype("float64"):
        _flagged(K)


def _flagged(K: float) -> None:
    node = plug_node(16, [dict(radius=R, length=1.0, permeability=K)], length_cells=4)
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    fields = node.fields(p, grid)
    check = band.negative_resistivity
    if plant("POSITIVITY_OFF"):
        check = lambda fields, permeabilities=(): band.negative_resistivity(  # noqa: E731
            band.WallFields(fields.chi, fields.walls, fields.signed_distance))
    assert bool(check(fields, node.permeabilities(p)))


def test_a_run_whose_permeability_is_not_above_zero_is_refused_after_it(float64):
    node = plug_node(16, [dict(radius=R, length=1.0, permeability=-1e-6)], length_cells=8)
    state = steady(node)
    assert float(state["negative_resistivity"]) == 1.0
    problem = node.stop_problem(state)
    assert problem is not None and problem.code == "flow.permeability-not-positive"


def test_a_declared_permeability_of_zero_or_below_is_refused_at_the_build():
    for value in (0.0, -1e-14):
        with pytest.raises(BuildRefused) as raised:
            experiment(permeability=value).build()
        assert "value.not-positive" in {r.code for r in raised.value.refusals}


def _flip_porous_losses(monkeypatch) -> None:
    """Gate 6's plant: each porous region's loss booked with its sign flipped."""
    original = node_module.ImplicitFlowNode.porous_losses

    def flipped(fields, solved, nu, rho, dt, grid):
        held = original(fields, solved, nu, rho, dt, grid)
        return {k: (-v if k.startswith(node_module.POROUS) else v) for k, v in held.items()}

    monkeypatch.setattr(node_module.ImplicitFlowNode, "porous_losses", staticmethod(flipped))


def _metered_plug(dtype: str, sign_flipped: bool, monkeypatch) -> tuple:
    if sign_flipped:
        _flip_porous_losses(monkeypatch)
    with in_dtype(dtype):
        delivery = books.LedgerDelivery("porous_0", LAW, books.DISSIPATIVE, books.HOSTED).encoded()
        node = plug_node(8, [dict(radius=0.5 * R, length=100.0, permeability=(0.5 / 8) ** 2)], length_cells=4,
                         scheme="imex_euler", dt=0.002, ledger=(*METERED, delivery))
        return node, run(node, 10)


def test_the_ledger_books_the_plugs_darcy_loss_under_its_law_and_the_identity_closes(dtype, monkeypatch):
    sign = bool(plant("POROUS_SIGN"))
    node, state = _metered_plug(dtype, sign, monkeypatch)
    judged = ledger_closes(state)
    assert judged.passed, judged.detail
    rows = books.laws(books.deliveries(node.constants()))
    booked = float(np.asarray(state[f"{books.RECORDS}step_work"])[rows.index(LAW)])
    assert booked < 0.0  # dissipated


def test_control_the_plugs_loss_booked_with_its_sign_flipped_is_caught(dtype, monkeypatch):
    _, state = _metered_plug(dtype, True, monkeypatch)
    judged = ledger_closes(state)
    assert judged.failed


def _flux_of(node, p):
    state = node.initial_state()
    ended = step_of(node, p, Waveform(steady=G))(dict(state))
    return jnp.sum(ended["velocity_z"]) * p["spacing"] ** 3 / (node.constants()["cells_z"] * p["spacing"])


@pytest.mark.parametrize("leaf", ["porous_0_permeability", "porous_0_centre", "porous_0_length"])
def test_the_flux_derivative_in_the_plug_forward_and_reverse_is_the_central_difference(float64, leaf):
    dh = 16
    h = 1.0 / dh
    # Placed off the faces (its ends 0.3 and 0.37 cells from a face's plane, its axis 0.3 cells off a cell's centre),
    # away from the switches a face on a medial set or an end plane would sit on (R3 item 4).
    if leaf == "porous_0_length":
        regions = [dict(radius=R, length=0.5 + 0.3 * h, permeability=(1.0 * h) ** 2, at=0.5 + 0.035 * h)]
        cells = 2 * dh
    else:
        regions = [dict(radius=0.5 * R, length=100.0, permeability=(1.0 * h) ** 2, offset=(0.3 * h, 0.0))]
        cells = 2
    node = plug_node(dh, regions, length_cells=cells, tolerance=1e-12)
    p = node._step_constants(None)  # noqa: SLF001
    value = jnp.asarray(p[leaf], jnp.float64)
    direction = jnp.zeros_like(value).at[0].set(1.0) if value.ndim else jnp.ones_like(value)
    step = 1e-4 * (value[0] if value.ndim else value) if leaf != "porous_0_centre" else 1e-3 * h

    def flux_at(moved):
        return _flux_of(node, {**p, leaf: moved})

    with pytest.MonkeyPatch.context() as patched:
        if plant("POROUS_STOP"):
            original = band.porous_field
            patched.setattr(band, "porous_field", lambda *a: jax.lax.stop_gradient(original(*a)))
        _, forward = jax.jvp(flux_at, (value,), (direction,))
        reverse = jnp.vdot(jax.grad(flux_at)(value), direction)
    central = (flux_at(value + step * direction) - flux_at(value - step * direction)) / (2 * step)
    scale = abs(float(central))
    line = PERMEABILITY_LINE if leaf == "porous_0_permeability" else GEOMETRY_LINE
    assert abs(float(forward) - float(central)) < line * scale
    assert abs(float(reverse) - float(central)) < line * scale


@pytest.mark.parametrize("case", ["darcy", "annulus"])
def test_float32_gives_the_plugs_flux_within_its_rounding_of_float64(case):
    fluxes = []
    for dtype in ("float64", "float32"):
        with in_dtype(dtype):
            if case == "darcy":
                node = plug_node(16, [dict(radius=R, length=0.5, permeability=(0.3 / 16) ** 2)], length_cells=32,
                                 interior="0.01h")
            else:
                node = plug_node(16, [dict(radius=0.5 * R, length=100.0, permeability=(1.0 / 16) ** 2)])
            fluxes.append(box_flux(node, steady(node)))
    assert abs(fluxes[1] / fluxes[0] - 1.0) < FLOAT32_LINE


def test_the_build_report_says_where_the_plugs_layer_is_under_a_cell_and_where_its_flux_is_unmeasured():
    h = 1.0 / 16
    said = {}
    for root, interior in ((2.0, "0.1h"), (0.1, "0.1h"), (0.005, "0.1h"), (0.005, "0.01h"), (0.002, "0.1h")):
        node = plug_node(16, [dict(radius=R, length=1.0, permeability=(root * h) ** 2)], length_cells=4,
                         interior=interior)
        said[root, interior] = [line for line in node.built_lines(None, None) if LAW in line]
    assert said[2.0, "0.1h"] == []
    resolved = "are resolved, measured down to sqrt(K)/h"
    assert len(said[0.1, "0.1h"]) == 1 and resolved in said[0.1, "0.1h"][0]
    assert len(said[0.005, "0.1h"]) == 1 and resolved in said[0.005, "0.1h"][0]
    assert all(len(said[key]) == 1 and "unmeasured here" in said[key][0] for key in ((0.005, "0.01h"), (0.002, "0.1h")))


def _codes(exp) -> set[str]:
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    return {r.code for r in raised.value.refusals}


def test_the_plug_runs_through_the_experiment_its_flux_and_face_pressures_reported(float64):
    built = experiment(permeability=1e-10).build()
    final = built.run(ms(20.0)).final["water.flow"]
    faces = np.asarray(final["porous_face_pressure"])
    assert faces.shape == (1, 2) and faces[0, 0] > faces[0, 1]  # the drop along the drive
    assert float(final["lumen_flux"]) > 0.0 and float(final["negative_resistivity"]) == 0.0


def test_a_porous_region_of_another_geometry_kind_is_refused():
    exp = experiment(plug=False)
    exp.attach(Darcy(), medium="water", structure="tube")
    exp.declare("tube.permeability", m2(1e-12), reason="a test's value")
    assert "flow.porous-kind" in _codes(exp)


def test_a_plug_held_by_rk3_is_refused():
    assert "flow.porous-scheme" in _codes(experiment(scheme="rk3"))


def test_a_plug_in_a_tilted_tube_is_refused():
    exp = experiment(plug=False, axis=(1.0, 0.0, 3.0))
    exp.add_structure(Structure("clot", geometry=Plug(axis=+Z, radius=mm(0.5), length=mm(1.0))))
    exp.declare("clot.permeability", m2(1e-12), reason="a test's value")
    exp.attach(Darcy(), medium="water", structure="clot")
    assert "flow.porous-axis" in _codes(exp)


def test_a_plug_in_an_open_vessel_reads_darcys_drop_on_its_faces_and_its_books_close(float64, monkeypatch):
    """The open vessel (an inflow at its start, a Windkessel at its end, no body force), two millimetres long, a plug
    half a millimetre long filling it at 16 cells across: its face pressures' drop is Darcy's at the inflow (measured
    -3.9 % on a vessel twice as long, its own drop between its openings -4.0 % of the series at this resolution); with
    the energy ledger its identity closes, the plug's loss booked under its law (review minor 5). Control, gate 6's: the
    loss booked with its sign flipped (``POROUS_SIGN``)."""
    from mime.observations import EnergyLedger

    if plant("POROUS_SIGN"):
        _flip_porous_losses(monkeypatch)
    radius, h = 0.5e-3, 1e-3 / 16
    K = h ** 2
    exp = open_experiment(K)
    exp.observe(EnergyLedger())
    result = exp.build().run(ms(1.0))
    final = result.final["water.flow"]
    faces = np.asarray(final["porous_face_pressure"], np.float64)[0]
    darcy = 1e-9 * plug_resistance(K, 0.5e-3, radius, 1e-3)
    assert abs((faces[0] - faces[1]) / darcy - 1.0) < 0.06
    judged = ledger_closes(result)
    assert judged.passed, judged.detail


def test_a_plug_anywhere_along_a_periodic_tube_is_read_at_its_image_in_the_box(float64):
    """A plug one millimetre long in a periodic box of two (D/h 8, K 1e-10): placed at z 0, 4 (its image straddling the
    box's end), 5 and -3 mm, every run reads the same flux and face pressures (review B1: at 5 and -3 the clot was not
    in the simulation, at 4 half of it); control, the centre not placed in the box (``NO_WRAP``)."""
    read = {}
    with pytest.MonkeyPatch.context() as patched:
        if plant("NO_WRAP"):
            patched.setattr(node_module.ImplicitFlowNode, "porous_centre", lambda self, centre, grid: centre)
        for z in (0.0, 4.0, 5.0, -3.0):
            exp = experiment(permeability=1e-10, length=1.0, period=2.0, cells_across=8, centre=(0.0, 0.0, z))
            final = exp.build().run(ms(20.0)).final["water.flow"]
            read[z] = (float(final["lumen_flux"]), np.asarray(final["porous_face_pressure"], np.float64)[0])
    flux, faces = read[0.0]
    for z, (moved, moved_faces) in read.items():
        assert abs(moved / flux - 1.0) < 1e-9, z
        assert np.allclose(moved_faces, faces, rtol=0.0, atol=1e-9 * float(np.max(np.abs(faces)))), z


def test_a_plug_longer_than_a_periodic_boxs_period_is_refused():
    assert "flow.porous-period" in _codes(experiment(length=2.0, period=2.0))


def test_a_plug_outside_an_open_vessels_span_is_refused():
    assert "flow.porous-outside" in _codes(open_experiment((1.0 / 16) ** 2 * 1e-6, centre=(0.0, 0.0, 5.0)))


@pytest.mark.parametrize("radius", [0.25, 0.5])
def test_a_region_leaving_a_gap_below_its_measured_line_is_flagged_at_run_time_with_a_warning(float64, radius):
    """At sqrt(K)/h 0.1 (D/h 8) a coaxial core of half the lumen's radius leaves a gap whose flux read 12 to 16 % low
    (review M2): ``porous_flagged`` is raised with a warning, and the build's line says so; a plug closing the lumen at
    the same K is within its measured line and is not flagged. Control: no region read as leaving a gap
    (``NO_GAP``)."""
    import mime.nodes.flow.fvm._porous as porous

    K = (0.1 * 1e-3 / 8) ** 2
    with pytest.MonkeyPatch.context() as patched:
        if plant("NO_GAP"):
            patched.setattr(porous, "leaves_a_gap", lambda radius, offset_squared, lumen_radius: False)
        built = experiment(permeability=K, radius=radius).build()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = built.run(ms(20.0))
    gap = radius < 0.5
    flagged = float(result.final["water.flow"]["porous_flagged"])
    warned = any("porous_flagged" in str(w.message) for w in caught)
    line = next(line for line in built.report.text().splitlines() if "the porous region" in line)
    assert (flagged, warned, "leaves a gap" in line) == (float(gap), gap, gap)
