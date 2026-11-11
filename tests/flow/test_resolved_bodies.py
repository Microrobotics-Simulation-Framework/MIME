"""F4 PR 1's gates: a resolved body on the FVM's grid (``Drag.Penalty``), its coupling lagged and declared. Predictions
committed first (``tests/flow/predictions_f4.json``). The controls honour ``MIME_PLANT_<NAME>``
(``resolved_cases.plant``), for ``.github/probe/detection_rate.py``.

- **The Re -> 0 ladder** (gate 1): the steady sphere's drag against Haberman and Sayre at 3 and 6 cells a radius, the
  radius the drag reads (smaller by a shift in cells, which falls with N: the shape-and-position form, never a bound on
  the nominal radius).
- **The density ratios** (gate 2): lagged, R3's case at Re 76 is stable and smooth at the measured range's edges, 0.3
  and 1.4; the rigid-interior shortcut, its control, diverges at 1.2.
- **The displaced share** (gate 3): a neutral sphere in a free box driven by a uniform force accelerates with the
  fluid (control: the share left out).
- **The books** (gate 4): metered, every receiver's books close, the drag judged at one velocity by its host.
- **The ripple and the position derivative** (gates 5 and 6), **float32** (gate 7), the spacing rule and the
  refusals."""

from __future__ import annotations

import contextlib
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from resolved_cases import (FreeBox, HeldSphere, deeper_band, haberman_sayre, patched, plant, r3_experiment,
                            r3_history, rigid_interior, shift_cells, stable)

from dataclasses import dataclass

from mime.build import BuildRefused, RunFailed
from mime.core import run_dtype
from mime.interactions import Drag
from mime.registry import register_interaction
from mime.vocabulary import register_namespace
from mime.nodes.flow.fvm import _resolved_node
from mime.nodes.flow.fvm._resolved_node import ResolvedFlowNode
from mime.units import ms


# -- gate 1: the Re -> 0 ladder ---------------------------------------------------------------------------------------
def test_the_sphere_reads_haberman_and_sayre_at_a_radius_smaller_by_a_shift_that_falls_with_n(float64):
    """Gate 1: a sphere at a/R 0.375 on a periodic tube's axis, Stokes, flux-free with its displaced share: its drag
    against Haberman and Sayre at 3 and 6 cells a radius (R3: -6.9 % and -1.0 %), and the radius it reads, smaller by a
    shift in cells that falls with N. The plant doubles the body's band's penalty depth, a shift the gate must see."""
    readings = {}
    if plant("BODY_BAND"):
        with patched(_resolved_node, "body_faces", deeper_band(2.0)):
            readings = {cells: HeldSphere(0.375, cells).k() for cells in (3.0, 6.0)}
    else:
        readings = {cells: HeldSphere(0.375, cells).k() for cells in (3.0, 6.0)}
    exact = haberman_sayre(0.375)
    errors = {cells: k / exact - 1.0 for cells, k in readings.items()}
    shifts = {cells: shift_cells(k, 0.375, cells) for cells, k in readings.items()}
    print(f"F4 gate 1: K errors {errors}, shifts in cells {shifts}")
    assert -0.085 < errors[3.0] < -0.055, errors
    assert -0.020 < errors[6.0] < -0.003, errors
    assert 0.0 < shifts[6.0] < shifts[3.0], shifts


# -- gate 2: the density ratios at Re 76, lagged -----------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.3, 1.4])
def test_the_lagged_coupling_is_stable_and_smooth_at_the_measured_ranges_edges(dtype, ratio):
    """Gate 2: R3's case (a sphere of 3 cells a radius on a periodic tube's axis, Re 76's drive, from rest, 200 steps of
    IMEX Euler), lagged, the load the penalty's plus the interior's momentum change: at density ratios 0.3 and 1.4, the
    range down to which the rung admits the lagged coupling, the body's speed is finite, under twice the flow's largest
    and its step-to-step change never alternates in sign over four steps in a row (a lagged loop ringing or
    diverging; the moving body's ripple dips once a cell). The plant is the rigid-interior shortcut."""
    if plant("RIGID_INTERIOR"):
        with patched(ResolvedFlowNode, "stepped", rigid_interior(ratio, ResolvedFlowNode.stepped)):
            velocities, top = r3_history(ratio)
    else:
        velocities, top = r3_history(ratio)
    held, longest = stable(velocities, top)
    print(f"F4 gate 2 [{dtype}] ratio {ratio}: stable {held}, the longest run of alternating changes {longest}, V "
          f"{velocities[-1]:.4g} m/s, the flow's largest {top:.4g} m/s")
    assert held, (longest, velocities[-1], top)


def test_the_rigid_interior_shortcut_diverges_where_the_field_formulation_holds(float64):
    """Gate 2's control: the rigid-interior shortcut (the interior's momentum change taken as m_f (V^n - V^n-1)/dt)
    at a density ratio of 1.2 is not smooth: R3 measured it diverging, its amplification about -1.6 a step."""
    with patched(ResolvedFlowNode, "stepped", rigid_interior(1.2, ResolvedFlowNode.stepped)):
        velocities, top = r3_history(1.2)
    held, longest = stable(velocities, top)
    print(f"F4 gate 2's control: the shortcut at 1.2 stable {held}, the longest run of alternating changes {longest}")
    assert not held, (longest, velocities[-1], top)


# -- gate 3: the displaced share ---------------------------------------------------------------------------------------
def test_a_neutral_sphere_in_a_uniformly_driven_free_box_accelerates_with_the_fluid(dtype):
    """Gate 3: a neutral sphere of 3 cells a radius in a free 24^3 box (the tube outside it) driven by a uniform force
    g: over the run's second half its acceleration is g's (the share restores the drive on the fluid it displaces,
    which the field's drive, on the fluid alone, leaves out). The plant leaves the share out (about 16 % short)."""
    box = FreeBox(share=not plant("NO_SHARE"))
    velocities = box.run(160)
    acceleration = (velocities[-1] - velocities[79]) / (80 * box.dt)
    error = acceleration / box.g - 1.0
    print(f"F4 gate 3 [{dtype}]: the second half's acceleration over g - 1 = {error:.3g}")
    assert abs(error) < 0.005, error


# -- gate 4: the books ----------------------------------------------------------------------------------------------
def _gift(original):  # noqa: ANN001, ANN202 - the plant: the body's penalty load taken at half the twist solved at
    def body_books(self, p, fields, velocity, index):  # noqa: ANN001, ANN202
        halved = dataclasses.replace(fields, twists=tuple(0.5 * twist for twist in fields.twists))
        return original(self, p, halved, velocity, index)
    return body_books


def test_the_books_close_and_the_drag_is_judged_at_one_velocity_by_its_host(dtype):
    """Gate 4: metered, a sphere (ratio 1.2) carried in R3's driven tube for 40 steps: every receiver's books close at
    every step (``ledger_closes``); the drag booked under its id on the fluid's row (held in its step) and the body's, its
    share under ``drag:ball.displaced``; its sign judged by its host at one velocity, no gift step on either row. The
    plant takes the body's penalty load at half the twist the field was solved at (a load at a velocity the field did
    not see, which gives the body work the fluid never lost: the defect a judgement at one velocity exists for). A gift
    added to the delivered load alone is not this gate's to see: the lagged exchange's unpaired work is booked in the
    drag's rows by design, unjudged until the affine coupling pairs it (measured: such a plant caught 0 of 5 runs in
    each dtype)."""
    from mime.observations import EnergyLedger
    from mime.testing import ledger_closes

    exp = r3_experiment(1.2)
    exp.observe(EnergyLedger())
    stopped: RunFailed | None = None
    with patched(ResolvedFlowNode, "body_books", _gift(ResolvedFlowNode.body_books)) if plant("GIFT") \
            else contextlib.nullcontext():
        try:
            result = exp.build().run(ms(6.0))
        except RunFailed as failed:  # the books up to a stop, so that a gift reads as a gift, not as the stop it causes
            stopped, result = failed, failed.result
    assert result is not None and result.energy is not None, stopped
    rows = {(law.law, law.receiver): law for law in result.energy.laws}
    assert {("drag:ball", "water.flow"), ("drag:ball", "ball"), ("drag:ball.displaced", "ball")} <= set(rows), rows
    gifts = {receiver: rows[("drag:ball", receiver)].gift_steps for receiver in ("ball", "water.flow")}
    print(f"F4 gate 4 [{dtype}]: the drag's gift steps {gifts}" + (" (the run stopped)" if stopped else ""))
    assert gifts == {"ball": 0, "water.flow": 0}, gifts
    assert stopped is None, stopped
    verdict = ledger_closes(result)
    assert verdict.status == "PASS", verdict.detail


# -- gates 5 and 6: the ripple and the derivative in position ------------------------------------------------------------
def test_the_drag_of_a_sphere_moved_through_a_cell_stays_within_the_ripples_line(float64):
    """Gate 5: a sphere of 6 cells a radius at a/R 0.375 at eight offsets through a cell along the axis: the drag's
    spread over them, (max - min)/mean, and the largest change of the drag a cell along it (its derivative times the
    cell, by autodiff), the quality line a derivative in the body's position carries. The plant stiffens the body's
    band (its depth a tenth), the staircase a stiff penalty makes of a smooth band (B3)."""
    if plant("STIFF_BAND"):
        with patched(_resolved_node, "body_faces", deeper_band(0.1)):
            spread, largest = _ripple()
    else:
        spread, largest = _ripple()
    print(f"F4 gate 5: the drag's spread through a cell {spread:.3g}, its largest change a cell {largest:.3g}")
    assert spread < 0.01, spread
    assert largest < 0.05, largest


def _ripple() -> tuple[float, float]:
    held = HeldSphere(0.375, 6.0)
    h = held.layout.spacing
    offsets = np.linspace(0.0, 1.0, 8, endpoint=False)
    drags, slopes = [], []
    for offset in offsets:
        centre = held.centre + np.asarray((0.0, 0.0, offset * h))
        drag, slope = jax.value_and_grad(lambda z: held.column(2, jnp.asarray(centre).at[2].set(z))[2])(centre[2])
        drags.append(float(drag))
        slopes.append(float(slope))
    mean = float(np.mean(drags))
    return (max(drags) - min(drags)) / mean, max(abs(slope) for slope in slopes) * h / mean


def test_a_derivative_in_the_bodys_position_equals_a_central_difference(float64):
    """Gate 6: the axial drag of a sphere off the axis (its centre a fifth of the tube's radius across, 3 cells a
    radius) in its lateral position: autodiff through the steady step against a central difference of a hundredth of
    a cell, the derivative of the discrete drag (the window's corner held out of it, the band smooth). Within a cell the
    derivative itself moves (gate 5's ripple): a difference of a tenth of a cell already reads 5 % off it. The plant
    stops the derivative at the body's position in the lift."""
    held = HeldSphere(0.375, 3.0)
    h = held.layout.spacing
    centre = jnp.asarray(held.centre).at[0].add(0.2 * held.radius / 0.375)

    def drag(x: object) -> object:
        moved = jax.lax.stop_gradient(x) if plant("STOP_POSITION") else x
        return held.column(2, centre.at[0].set(moved))[2]

    derivative = float(jax.grad(drag)(centre[0]))
    step = 0.01 * h
    difference = (float(drag(centre[0] + step)) - float(drag(centre[0] - step))) / (2.0 * step)
    print(f"F4 gate 6: d F_z / d x {derivative:.6g} against {difference:.6g}")
    assert abs(derivative - difference) <= 2e-3 * abs(difference), (derivative, difference)


# -- gate 7: float32 ---------------------------------------------------------------------------------------------------
def test_the_steady_drag_in_float32_is_within_its_line_of_float64s():
    """Gate 7: the steady drag at 6 cells a radius in float32 within 1e-3 of float64's (B10's line for a penalty
    drag)."""
    held = jax.config.jax_enable_x64
    try:
        values = {}
        for dtype in ("float64", "float32"):
            jax.config.update("jax_enable_x64", dtype == "float64")
            values[dtype] = HeldSphere(0.375, 6.0).k()
    finally:
        jax.config.update("jax_enable_x64", held)
    difference = abs(values["float32"] / values["float64"] - 1.0)
    print(f"F4 gate 7: float32's drag against float64's {difference:.3g}")
    assert difference < 1e-3, values


# -- the build: the spacing rule and the refusals -----------------------------------------------------------------------
@pytest.mark.parametrize(("cells", "across"), [(3, 16), (6, 32)])
def test_the_fvm_takes_the_finer_of_the_lumens_spacing_and_n_cells_across_the_bodys_radius(float64, cells, across):
    """B8.2: with no spacing written, the FVM's grid is the finer of D/h ``cells_across`` (16 here) and N cells across
    the resolved body's radius: a 0.1875 mm sphere in a 1 mm lumen at 3 cells a radius takes 16 across, at the rung's
    own 6, 32."""
    built = r3_experiment(1.0, feature_cells=cells, dt=1.5e-4 * 16 / across).build()
    node = built.graph.get_node("water.flow")
    assert isinstance(node, ResolvedFlowNode)
    assert math.isclose(float(node.constants()["spacing"]), 1e-3 / across, rel_tol=1e-3), node.constants()["spacing"]


def _refusals(exp: object) -> dict[str, str]:
    with pytest.raises(BuildRefused) as refused:
        exp.build()  # type: ignore[attr-defined]
    return {refusal.code: f"{refusal.message} | {refusal.action}" for refusal in refused.value.refusals}


def test_a_body_lighter_than_the_lagged_ranges_edge_is_refused_naming_the_route(float64):
    """The lagged coupling is declared over the measured range: a body at 0.2 times the fluid's density is refused,
    naming the routes (a denser body, the MRS, the affine coupling this one comes with next)."""
    found = _refusals(r3_experiment(0.2))
    assert "drag.penalty-lagged" in found, found


def test_a_hull_with_no_exact_solid_region_is_refused_naming_the_mrs(float64):
    from mime.scene import Box
    from mime.units import m

    found = _refusals(r3_experiment(1.0, hull=Box(size=m((2e-4, 2e-4, 2e-4)))))
    assert "drag.penalty-kind" in found, found


def test_one_way_is_refused_on_a_resolved_body(float64):
    found = _refusals(r3_experiment(1.0, one_way=True))
    assert any("one_way" in message for message in found.values()), found


def test_a_grid_coarser_than_the_thinnest_feature_line_is_refused(float64):
    from mime.units import m

    found = _refusals(r3_experiment(1.0, spacing=m(1e-4)))
    assert "drag.penalty-thin" in found, found


def test_rk3_is_refused_for_a_resolved_body(float64):
    from mime.dynamics import Scheme

    found = _refusals(r3_experiment(1.0, scheme=Scheme.RK3()))
    assert "drag.penalty-scheme" in found, found


register_namespace("resolvedtest", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PointBeside(Drag):
    """The stokes rung under a name of its own, so that a point body's drag (which delivers its Maxey-Riley added mass)
    can be attached beside a resolved body's: the case the added-mass refusal guards."""

    name = "resolvedtest.point_beside"


def test_an_added_mass_delivered_to_a_resolved_body_is_refused_naming_the_added_mass_phase(float64):
    """A6 item 9: a resolved body's grid carries its fluid's inertia, so a law delivering an added mass to it (here a
    point body's drag, whose Maxey-Riley added mass rides on its resistance) is refused, naming the route (I2's
    delivered-inertia port, for a medium whose solver carries no inertia)."""
    exp = r3_experiment(1.0)
    exp.attach(PointBeside(rung="stokes"), body="ball", medium="water")
    found = _refusals(exp)
    assert "drag.penalty-added-mass" in found and "I2" in found["drag.penalty-added-mass"], found


def test_a_body_whose_window_would_meet_its_periodic_image_is_refused(float64):
    from mime.units import m

    found = _refusals(r3_experiment(1.0, length=m(5e-4)))
    assert "drag.penalty-box" in found, found


def test_an_open_vessel_takes_no_resolved_body_yet(float64):
    """The open vessel's one elimination of openings and bodies is the affine coupling's (the next part): until then
    its flow refuses a resolved body, naming the periodic box."""
    found = _refusals(r3_experiment(1.0, opened=True))
    assert "flow.open-resolved" in found, found


def test_a_tube_trees_flow_takes_no_resolved_body_yet(float64):
    """A tube tree's node holds the tree's solid alone: a resolved body in its flow is refused by name, naming the
    straight tube and the MRS (without the refusal the build stopped on an entry defect, its reads unconnected)."""
    found = _refusals(r3_experiment(1.0, tree=True))
    assert "flow.tree-resolved" in found, found
