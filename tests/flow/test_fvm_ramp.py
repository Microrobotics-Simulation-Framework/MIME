"""The stiff interior's ramp from the band's edge (h) to 1.5h, adopted under row #315 behind four gates, each through
the node's own step against the ramp it replaced (from h to 3h, ``wall_cases.ramp_end(3.0)``):

- **R1**, the quarter-cell wall plant (the radius a quarter cell outward of where the oracle puts it) still fails the
  wall's shift against the planar calibration (both under the new ramp) at a drawn sub-cell offset, and the clean
  wall passes it; the detection rate over drawn offsets is the gate's (``MIME_PLANT_WALL=wall``);
- **R2**, a 75 % stenosis and a full occlusion: the solid's bypass no larger than under the old ramp, the stenosis's
  flux within ``FLUX_LINE`` of the old ramp's, the occlusion's leak no larger, at 0.1h and 0.01h; the control plants a
  ramp ending at 6h for the new (``MIME_PLANT_WALL=ramp:6``), which leaks more than the old;
- **R3**, float32: correct runs raise no solve flag and their fluxes lie within ``FLOAT32_LINE`` of float64's; the
  control holds a float32 tolerance below its rounding floor (``MIME_PLANT_WALL=tolerance``), which the stop flags;
- **R4**, the CG's iterations at the 0.01h interior through the stenosis, steady and IMEX Euler, at most
  ``ITERATION_LINE`` times the old ramp's; the control stiffens the interior tenfold (``MIME_PLANT_WALL=stiffer``).

The predictions are ``predictions_general_wall.json``'s, committed before these ran."""

from __future__ import annotations

import functools

import numpy as np
import pytest
from fvm_cases import DRIVES, driven, in_dtype, run, tube_node
from test_fvm_tube import _arguments, planar_calibration, tube_shift
from wall_cases import box_flux, draw, occlusion, plant, plant_value, ramp_end, seeds, stenosis, tree_node

import mime.nodes.flow.fvm._band as band_module
from mime.nodes.flow.fvm import REPORT_FIELDS, ImplicitFlowNode

#: R1's line: the wall's measured shift against the planar calibration, in cells (S1's, as the tube's gate).
SHIFT_LINE = 0.15
#: R2's line on the stenosis's flux, new ramp against old, relative.
FLUX_LINE = 3e-3
#: R3's line on float32's flux against float64's, relative.
FLOAT32_LINE = 1e-4
#: R4's line: the new ramp's iterations over the old's.
ITERATION_LINE = 1.2
#: The old ramp's end, in cells.
OLD_END = 3.0
ITERATIONS = REPORT_FIELDS.index("iterations")
FLAGS = [REPORT_FIELDS.index(name) for name in ("at_cap", "stagnated", "not_finite")]


def _new_end() -> float:
    return plant_value("ramp", band_module.INTERIOR_RAMP_END) if plant("ramp") else band_module.INTERIOR_RAMP_END


# -- R1 -----------------------------------------------------------------------------------------------------------
@functools.cache
def _drawn_tube(dtype: str, offset: tuple[float, float], quarter: float):
    with in_dtype(dtype):
        node = tube_node(16, offset=offset, dt=0.5, convection=False, scheme="steady")
        given = driven(ImplicitFlowNode(**{**_arguments(node), "tube_radius": 0.5 + quarter / 16}), DRIVES[node])
        return node, run(given, 2)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("seed", seeds(101, 102, 103))
def test_r1_the_walls_shift_holds_the_planar_calibration_at_a_drawn_offset(dtype, seed):
    """The tube at a drawn sub-cell offset: its shift within ``SHIFT_LINE`` of the planar calibration, both under the
    new ramp. **Control** (``MIME_PLANT_WALL=wall``): the solver's wall a quarter cell outward."""
    offset = draw(seed)[1][:2]
    quarter = 0.25 if plant("wall") else 0.0
    node, state = _drawn_tube(dtype, offset, quarter)
    shift, planar = tube_shift(node, state, 0.5), planar_calibration(16, dtype)
    print(f"\nseed {seed} {dtype} offset {offset}: shift {shift:+.4f}, planar {planar:+.4f} cells")
    assert abs(shift - planar) < SHIFT_LINE


# -- R2 -----------------------------------------------------------------------------------------------------------
def _tree_run(geometry: str, interior: str, end: float, seed: int, dtype: str = "float64"):
    tree = {"stenosis": stenosis(), "occlusion": occlusion()}[geometry]
    with in_dtype(dtype), ramp_end(end):
        placed = tree_node(tree, 16, offset=draw(seed)[1], interior=interior,
                           tolerance=1e-10 if dtype == "float64" else None)
        return placed, run(placed.node, 1)


@pytest.mark.parametrize("interior", ["0.1h", "0.01h"])
@pytest.mark.parametrize("seed", seeds(201))
def test_r2_the_stenosis_leaks_no_more_and_keeps_its_flux(interior, seed):
    _, new = _tree_run("stenosis", interior, _new_end(), seed)
    _, old = _tree_run("stenosis", interior, OLD_END, seed)
    moved = box_flux(new) / box_flux(old) - 1.0
    print(f"\nstenosis {interior} seed {seed}: bypass {float(new['bypass_ratio']):.3e} against "
          f"{float(old['bypass_ratio']):.3e}; flux moved {moved:+.2e}; iterations "
          f"{float(new['solve_report'][ITERATIONS]):.0f} and {float(old['solve_report'][ITERATIONS]):.0f}")
    assert float(new["wall_inexact"]) == 0.0 and float(old["wall_inexact"]) == 0.0
    assert abs(float(new["bypass_ratio"])) <= abs(float(old["bypass_ratio"]))
    assert abs(moved) <= FLUX_LINE


@pytest.mark.parametrize("interior", ["0.1h", "0.01h"])
@pytest.mark.parametrize("seed", seeds(301))
def test_r2_the_occlusion_leaks_no_more(interior, seed):
    _, new = _tree_run("occlusion", interior, _new_end(), seed)
    _, old = _tree_run("occlusion", interior, OLD_END, seed)
    print(f"\nocclusion {interior} seed {seed}: leak {box_flux(new):.4e} against {box_flux(old):.4e} "
          f"({box_flux(new) / box_flux(old):.3f})")
    assert 0.0 < box_flux(new) <= box_flux(old)


# -- R3 -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("geometry", ["tube", "stenosis"])
@pytest.mark.parametrize("seed", seeds(401))
def test_r3_float32_raises_no_flag_and_holds_float64s_flux(geometry, seed):
    """Float32 at its default stop (its rounding floor where 1e-5 is out of reach), against float64 at 1e-10.
    **Control** (``MIME_PLANT_WALL=tolerance``): float32 held to 1e-9, below its floor, which the stop flags."""
    offset = draw(seed)[1]
    held = 1e-9 if plant("tolerance") else None
    if geometry == "tube":
        fluxes, reports = [], []
        for dtype, tolerance in (("float64", 1e-10), ("float32", held)):
            with in_dtype(dtype):
                node = tube_node(16, offset=offset[:2], dt=0.5, convection=False, scheme="steady",
                                 tolerance=tolerance)
                state = run(node, 1)
                fluxes.append(float(state["lumen_flux"]))
                reports.append(np.asarray(state["solve_report"]))
    else:
        fluxes, reports = [], []
        for dtype, tolerance in (("float64", 1e-10), ("float32", held)):
            with in_dtype(dtype):
                placed = tree_node(stenosis(), 16, offset=offset, tolerance=tolerance)
                state = run(placed.node, 1)
                fluxes.append(box_flux(state))
                reports.append(np.asarray(state["solve_report"]))
    moved = fluxes[1] / fluxes[0] - 1.0
    print(f"\n{geometry} seed {seed}: float32 against float64 {moved:+.2e}; float32's report {reports[1]}")
    assert all(float(reports[1][k]) == 0.0 for k in FLAGS)
    assert abs(moved) <= FLOAT32_LINE


# -- R4 -----------------------------------------------------------------------------------------------------------
def _iterations(end: float, scheme: str, seed: int) -> float:
    held = band_module.INTERIORS
    if plant("stiffer") and end != OLD_END:
        band_module.INTERIORS = {**held, "0.01h": 0.001}  # type: ignore[assignment]
    try:
        with in_dtype("float64"), ramp_end(end):
            placed = tree_node(stenosis(), 16, offset=draw(seed)[1], interior="0.01h", scheme=scheme, dt=0.02,
                               tolerance=1e-8)
            if scheme == "steady":
                return float(run(placed.node, 1)["solve_report"][ITERATIONS])
            ended, reports = run(placed.node, 3, record=lambda s: s["solve_report"])
            return float(np.mean(np.asarray(reports)[:, ITERATIONS]))
    finally:
        band_module.INTERIORS = held


@pytest.mark.parametrize("scheme", ["steady", "imex_euler"])
@pytest.mark.parametrize("seed", seeds(501))
def test_r4_the_cg_takes_no_more_iterations_through_the_stiff_stenosis(scheme, seed):
    new, old = _iterations(_new_end(), scheme, seed), _iterations(OLD_END, scheme, seed)
    print(f"\n{scheme} seed {seed}: iterations {new:.1f} against {old:.1f} ({new / old:.3f})")
    assert new <= ITERATION_LINE * old
