"""The open vessel through the experiment (F2b's PRs 2 and 3): ``FVM(domain=VesselRegion())`` on a tube cut at its own
ends, an ``Inflow`` at ``tube.start`` and a ``Windkessel`` at ``tube.end``, built, run and metered; the account names the
inflow as what supplied the work, the Windkessel's stored energy and its loss by name; the plan's refusals, each a way
the openings cannot be stepped; a law over both openings (``SharedDistal``, ``surface=`` a tuple or the lumen's
openings) and a closed ``Valve``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mime.build import BuildRefused
from mime.dynamics import FVM, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce, Inflow, SharedDistal, Valve, Windkessel
from mime.nodes.flow.fvm import OpenFlowNode
from mime.observations import EnergyLedger
from mime.scene import PeriodicBox, Scene, Tube, VesselRegion, Z
from mime.testing import ledger_closes
from mime.things import Medium, Structure
from mime.units import Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_Pa, m3_per_s, m_per_s2, mm, ms

RADIUS, LENGTH, MU = 0.5e-3, 1e-3, 1e-3
#: The inflow at a mean speed of 1 mm/s, and the tube's own Poiseuille resistance.
FLUX = 1e-3 * math.pi * RADIUS ** 2
RESISTANCE = 8.0 * MU * LENGTH / (math.pi * RADIUS ** 4)


def vessel(*, domain=None, axis=+Z, scheme=None, laws=("start", "end"), inflows=("start",), dt=None) -> Experiment:
    exp = Experiment("open_tube", scene=Scene.new(up=Z, units="m"), dt=ms(2.0) if dt is None else dt)
    exp.add_structure(Structure("tube", geometry=Tube(axis=axis, inner_radius=mm(RADIUS * 1e3), wall=mm(0.2),
                                                      length=mm(LENGTH * 1e3))))
    flow = FVM(domain=VesselRegion() if domain is None else domain, cells_across=8,
               scheme=Scheme.IMEXEuler() if scheme is None else scheme)
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(MU), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    for where in laws:
        if where in inflows:
            exp.attach(Inflow(steady=m3_per_s(FLUX if where == "start" else -FLUX)), medium="water",
                       surface=f"tube.{where}")
        else:
            exp.attach(Windkessel(proximal=Pa_s_per_m3(0.3 * RESISTANCE), compliance=m3_per_Pa(1e-13),
                                  distal=Pa_s_per_m3(3.0 * RESISTANCE)), medium="water", surface=f"tube.{where}")
    return exp


def test_an_open_vessel_runs_with_its_laws_and_its_books_close(float64):
    exp = vessel()
    exp.observe(EnergyLedger())
    built = exp.build()
    result = built.run(ms(20.0))
    final = result.final["water.flow"]
    np.testing.assert_allclose(np.asarray(final["opening_flux"]), [-FLUX, FLUX], rtol=1e-12)
    assert ledger_closes(result).passed
    account = result.energy
    assert account is not None
    assert set(account.supplied) == {"inflow:tube.start"}
    assert "windkessel:tube.end" in account.stored and "windkessel:tube.end" in account.lost
    assert account.lost["windkessel:tube.end"] > 0.0
    text = built.report.text()
    assert "metered (fluid and opening ports)" in text and "supplied by inflow:tube.start" in text


def test_a_saved_open_vessel_loads_and_runs_bit_for_bit(float64, tmp_path):
    exp = vessel()
    exp.save(tmp_path / "saved")
    loaded = Experiment.load(tmp_path / "saved")
    finals = [each.build().run(ms(10.0)).final["water.flow"] for each in (exp, loaded)]
    for name in ("velocity_z", "opening_pressure", "opening_state"):
        assert np.array_equal(np.asarray(finals[0][name]), np.asarray(finals[1][name]))
    text = (tmp_path / "saved" / "experiment.mime.yaml").read_text()
    assert "group: vessel_region" in text and "name: windkessel" in text and "name: inflow" in text


def refused(exp: Experiment) -> list[str]:
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    return [r.code for r in raised.value.refusals]


def test_each_opening_takes_one_law(float64):
    assert "flow.opening-law" in refused(vessel(laws=("start",)))
    doubled = vessel()
    doubled.attach(Windkessel(proximal=Pa_s_per_m3(RESISTANCE), compliance=m3_per_Pa(1e-13),
                              distal=Pa_s_per_m3(RESISTANCE)), medium="water", surface="tube.start")
    assert "flow.opening-law" in refused(doubled)


def test_a_law_at_a_surface_that_is_no_opening_is_refused(float64):
    exp = vessel()
    exp.attach(Inflow(steady=m3_per_s(FLUX)), medium="water", surface="tube.inner_surface")
    assert "flow.opening-unknown" in refused(exp)


def test_inflows_at_every_opening_must_cancel(float64):
    """Two inflows that cancel (one giving what the other takes) are well posed through the gauge's row; two that do
    not are refused."""
    exp = vessel(inflows=("start", "end"))
    built = exp.build()
    final = built.run(ms(4.0)).final["water.flow"]
    np.testing.assert_allclose(np.asarray(final["opening_flux"]), [-FLUX, FLUX], rtol=1e-12)
    unbalanced = vessel(laws=("start",), inflows=("start",))
    unbalanced.attach(Inflow(steady=m3_per_s(FLUX)), medium="water", surface="tube.end")
    assert "flow.openings-ill-posed" in refused(unbalanced)


def test_a_periodic_box_takes_no_law_at_an_opening(float64):
    exp = vessel(domain=PeriodicBox(length=mm(0.25)), laws=("end",))
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    assert "flow.no-openings" in refused(exp)


def test_an_open_vessel_stepped_by_rk3_runs_and_each_step_dissipates(float64):
    """``Scheme.RK3()`` on an open vessel builds the open node of the explicit stage kind: the inflow's face flux is its
    waveform, route 1 judges every step to have dissipated, the Windkessel's own books are workless against the flow's
    to rounding; past RK3's viscous number the step is refused."""
    exp = vessel(scheme=Scheme.RK3())
    exp.observe(EnergyLedger())
    built = exp.build()
    result = built.run(ms(20.0))
    final = result.final["water.flow"]
    np.testing.assert_allclose(np.asarray(final["opening_flux"]), [-FLUX, FLUX], rtol=1e-12)
    judged = ledger_closes(result)
    assert judged.passed, judged.detail
    assert float(final["ledger_route_one_negative_steps"]) == 0.0
    account = result.energy
    assert account is not None and set(account.supplied) == {"inflow:tube.start"}
    assert account.lost["windkessel:tube.end"] > 0.0
    assert "flow.viscous-limit" in refused(vessel(scheme=Scheme.RK3(), dt=ms(5.0)))


def test_an_open_vessel_square_to_the_box_axis_is_refused(float64):
    """A tilted vessel is built through the region builder (``test_fvm_region.py``); one square to z cannot be turned
    onto either face by one bend."""
    assert "flow.region-face" in refused(vessel(axis=(1.0, 0.0, 0.0)))


def test_dt_auto_on_an_open_vessel_is_refused_naming_its_missing_step_rule(float64):
    """dt='auto' is refused with the open vessel's own reason, never "nothing in this experiment" bounds its step."""
    with pytest.raises(BuildRefused) as raised:
        vessel(dt="auto").build()
    (refusal,) = [r for r in raised.value.refusals if r.code == "run.dt-auto-unset"]
    assert "no step rule yet" in str(refusal) and "nothing in this experiment" not in str(refusal)


def test_the_cfl_stop_on_an_open_vessel_names_its_own_routes(float64):
    """The CFL stop's routes on an open vessel: a timestep from the inflow's peak, never dt='auto' or a steady drive."""
    built = vessel().build()
    node = built.graph.get_node("water.flow")
    assert isinstance(node, OpenFlowNode)
    state = {**dict(node.initial_state()), "stopped_at": np.int32(5), "stopped_cfl": np.float64(0.9)}
    problem = node.stop_problem(state)
    assert problem is not None and problem.code == "flow.cfl-exceeded"
    assert "dt='auto' is refused here" in problem.action and "steady drive" not in problem.action
    assert "inflow's peak" in problem.action


# -- a law over several openings (F2b's PR 3; the record's "F2's rows, 2", item 1) --------------------------------

def loop(surface, *, distal=5.0, scheme=None, metered=True) -> Experiment:
    """The tube closed on itself through a shared distal law across both its ends (``first`` at the start, ``second``
    at the end), driven by a body force on its fluid: a loop, whose flux the shared resistance cannot change (the two
    openings' fluxes sum to zero, so ``R_d (q_1 + q_2)`` vanishes) and the off-diagonals' sign can."""
    exp = Experiment("open_loop", scene=Scene.new(up=Z, units="m"), dt=ms(2.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(RADIUS * 1e3), wall=mm(0.2),
                                                      length=mm(LENGTH * 1e3))))
    flow = FVM(domain=VesselRegion(), cells_across=8, stokes=True,
               scheme=Scheme.Steady() if scheme is None else scheme)
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(MU), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    exp.attach(SharedDistal(first=Pa_s_per_m3(RESISTANCE), second=Pa_s_per_m3(3.0 * RESISTANCE),
                            distal=Pa_s_per_m3(distal * RESISTANCE)), medium="water", surface=surface)
    if metered:
        exp.observe(EnergyLedger())
    return exp


def test_a_law_over_two_openings_binds_both_and_its_shared_resistance_drops_out(float64, monkeypatch):
    """The shared distal law attached across the tube's two ends, as a tuple and as the lumen's openings group: one
    law, one row in the books (``shared_distal:tube.start+tube.end``), the books closing; the loop's flux the same at
    two distal resistances a decade apart (the discriminator: the exact law's split does not move with R_d), and moved
    by the off-diagonals dropped (each R_d folded into its own opening's row: the record's control (a))."""
    fluxes = {}
    for surface in (("tube.start", "tube.end"), "tube.lumen"):
        for distal in (5.0, 50.0):
            result = loop(surface, distal=distal).build().run(ms(6.0))
            assert ledger_closes(result).passed
            assert "shared_distal:" + ("tube.start+tube.end" if isinstance(surface, tuple) else "tube.lumen") \
                in result.energy.lost
            fluxes[surface, distal] = float(np.asarray(result.final["water.flow"]["opening_flux"])[1])
    reference = fluxes["tube.lumen", 5.0]
    assert all(abs(value / reference - 1.0) < 1e-9 for value in fluxes.values()), fluxes
    from twin_cases import PLANTS as TWIN_PLANTS

    TWIN_PLANTS["dropped"](monkeypatch.setattr)
    dropped = float(np.asarray(loop("tube.lumen").build().run(ms(6.0)).final["water.flow"]["opening_flux"])[1])
    assert abs(dropped / reference - 1.0) > 0.1, (dropped, reference)


def test_a_law_given_another_count_of_openings_is_refused(float64):
    """A one-opening law given two surfaces, and the two-opening law given one, are each refused naming the count; a
    region with no declared openings given as a law's openings is refused. Each alone: the flow does not then say that
    no law is attached at the openings the refused law named (it is missing from the flow's laws only because it was
    refused)."""
    one = vessel(laws=())
    one.attach(Windkessel(proximal=Pa_s_per_m3(RESISTANCE), compliance=m3_per_Pa(1e-13),
                          distal=Pa_s_per_m3(RESISTANCE)), medium="water", surface=("tube.start", "tube.end"))
    assert refused(one) == ["binding.opening-count"]
    two = vessel(laws=("start",))
    two.attach(SharedDistal(first=Pa_s_per_m3(RESISTANCE), second=Pa_s_per_m3(RESISTANCE),
                            distal=Pa_s_per_m3(RESISTANCE)), medium="water", surface="tube.end")
    assert refused(two) == ["binding.opening-count"]
    wall = vessel(laws=())
    wall.attach(SharedDistal(first=Pa_s_per_m3(RESISTANCE), second=Pa_s_per_m3(RESISTANCE),
                             distal=Pa_s_per_m3(RESISTANCE)), medium="water", surface="tube.wall")
    assert refused(wall) == ["binding.no-openings"]


def test_a_flux_law_at_every_opening_is_refused_naming_the_laws_to_replace(float64):
    """An Inflow at the start and a Valve at the end have no solution; the route names both flux laws and never sends a
    Windkessel to the opening the Valve holds."""
    exp = vessel(laws=("start",))
    exp.attach(Valve(), medium="water", surface="tube.end")
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    (refusal,) = [r for r in raised.value.refusals if r.code == "flow.openings-ill-posed"]
    assert "inflow:tube.start" in refusal.action and "valve:tube.end" in refusal.action, refusal.action
    assert "surface='tube.end'" not in refusal.action, refusal.action


def test_a_saved_law_over_two_openings_loads_and_runs_bit_for_bit(float64, tmp_path):
    exp = loop(("tube.start", "tube.end"), metered=False)
    exp.save(tmp_path / "saved")
    loaded = Experiment.load(tmp_path / "saved")
    finals = [each.build().run(ms(4.0)).final["water.flow"] for each in (exp, loaded)]
    assert np.array_equal(np.asarray(finals[0]["opening_flux"]), np.asarray(finals[1]["opening_flux"]))


def test_a_closed_valve_holds_its_opening_shut(float64):
    """A valve at the tube's end and a Windkessel at its start, the fluid pushed by a body force: no flux at either
    end (the valve's row, and the incompressible field), the pressure behind the valve solved, the books closing."""
    exp = vessel(laws=("start",), inflows=())
    exp.attach(Valve(), medium="water", surface="tube.end")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    exp.observe(EnergyLedger())
    result = exp.build().run(ms(20.0))
    final = result.final["water.flow"]
    assert np.max(np.abs(np.asarray(final["opening_flux"]))) < 1e-12 * FLUX
    assert float(np.asarray(final["opening_pressure"])[1]) > 0.0
    assert ledger_closes(result).passed
