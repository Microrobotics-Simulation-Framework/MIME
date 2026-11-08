"""The FVM through the experiment: ``Medium(fields={"flow": FVM(...)})`` built with ``exp.build``, driven by a
``BodyForce`` law bound to the medium; the gates' physics read through the build (Poiseuille's core curvature on the
built node), the saved form's round trip, the CFL check's refusal of a run, and the plan's refusals."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from fvm_cases import in_dtype
from test_fvm_tube import CURVATURE_LINE, core_viscosity

from mime.build import ArgumentRefused, BuildRefused, RunFailed
from mime.dynamics import FVM, Scheme
from mime.dynamics.flow import IMEXEulerScheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.scene import PeriodicBox, Scene, Tube, X, Z
from mime.things import Medium, Structure
from mime.units import kg_per_m3, m_per_s2, mm, mPa_s, ms

RADIUS, NU, G = 0.5e-3, 1e-6, 1.0


def experiment(*, axis=+Z, fills: str | None = "tube.lumen", dt=ms(20.0), steady=True, g=G, **settings):
    exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=dt)
    exp.add_structure(Structure("tube", geometry=Tube(axis=axis, inner_radius=mm(RADIUS * 1e3), wall=mm(0.2),
                                                      length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=mm(0.25)), cells_across=16, stokes=steady,
               scheme=Scheme.Steady() if steady else Scheme.IMEXEuler(), **settings)
    exp.add_medium(Medium("water", fills=fills, fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(g)), medium="water")
    return exp


def test_the_built_flows_core_curvature_gives_the_viscosity(dtype):
    with in_dtype(dtype):
        built = experiment().build()
        result = built.run(ms(40.0))
        node = built.graph.get_node("water.flow")
    assert abs(core_viscosity(node, result.final["water.flow"], g=G) / NU - 1.0) < CURVATURE_LINE


def test_a_saved_flow_loads_and_runs_bit_for_bit(float64, tmp_path: pathlib.Path):
    """The flow saves in the form it has (dev1's form 10, a medium's ``dynamics.flow``), settings groups included,
    and loads as the same medium: never dropped, never refused."""
    exp = experiment()
    exp.save(tmp_path / "saved")
    loaded = Experiment.load(tmp_path / "saved")
    finals = []
    for each in (exp, loaded):
        finals.append(each.build().run(ms(40.0)).final["water.flow"])
    for name in ("velocity_x", "velocity_y", "velocity_z"):
        assert np.array_equal(np.asarray(finals[0][name]), np.asarray(finals[1][name]))
    text = (tmp_path / "saved" / "experiment.mime.yaml").read_text()
    assert "name: fvm" in text and "group: periodic_box" in text and "group: steady" in text


def test_a_run_past_the_cfl_line_is_refused_naming_its_step(float64):
    """Convection on, a drive strong enough that the flow passes the CFL line within the run: refused, the step and the
    route named; the run's result is carried with the refusal."""
    exp = experiment(steady=False, g=2000.0, dt=ms(1.0))
    with pytest.raises(RunFailed) as raised:
        exp.build().run(ms(200.0))
    codes = [r.code for r in raised.value.refusals]
    assert codes == ["flow.cfl-exceeded"]
    assert "at its step" in raised.value.refusals[0].message and "dt <=" in raised.value.refusals[0].action


def test_a_run_inside_the_cfl_line_is_not_refused(float64):
    exp = experiment(steady=False, g=1.0, dt=ms(1.0))
    result = exp.build().run(ms(20.0))
    assert int(np.asarray(result.final["water.flow"]["stopped_at"])) == -1


@pytest.mark.parametrize(("change", "code"), [
    ({"steady": True, "stokes": False}, "flow.steady-convection"),
    ({"band": "smooth_1h_0.1h"}, "flow.band"),
    ({"interior": "0.5h"}, "flow.interior"),
    ({"preconditioner": "multigrid"}, "flow.preconditioner"),
    ({"cells_across": 4}, "flow.resolution"),
])
def test_a_flow_its_settings_rule_out_is_refused_when_planned(change, code):
    steady = change.pop("steady", True)
    stokes = change.pop("stokes", None)
    exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=ms(20.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=mm(0.25)), stokes=steady if stokes is None else stokes,
               scheme=Scheme.Steady() if steady else IMEXEulerScheme(), **change)
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water")
    with pytest.raises(BuildRefused) as raised:
        exp.build()
    assert code in [r.code for r in raised.value.refusals]


def test_a_flow_in_a_medium_that_fills_no_tube_is_refused():
    with pytest.raises(BuildRefused) as raised:
        experiment(fills=None).build()
    assert "flow.region" in [r.code for r in raised.value.refusals]


def test_a_tube_along_an_axis_the_box_cannot_hold_is_refused():
    with pytest.raises(BuildRefused) as raised:
        experiment(axis=(0.0, 0.6, 0.8)).build()
    assert "flow.box-axis" in [r.code for r in raised.value.refusals]


def test_a_tube_along_x_is_refused_with_its_route():
    with pytest.raises(BuildRefused) as raised:
        experiment(axis=+X).build()
    assert "flow.box-axis" in [r.code for r in raised.value.refusals]


def test_a_medium_takes_its_fields_by_role():
    flow = FVM(domain=PeriodicBox(length=mm(0.25)))
    medium = Medium("water", fields={"flow": flow})
    assert medium.flow is flow and medium.dynamics is flow and dict(medium.fields) == {"flow": flow}
    assert dict(Medium("bath", flow=flow).fields) == {"flow": flow}
    with pytest.raises(ArgumentRefused) as raised:
        Medium("water", fields={"acoustic": flow})
    assert raised.value.refusals[0].code == "medium.field-role"
    with pytest.raises(ArgumentRefused) as raised:
        Medium("water", flow=FVM(domain=PeriodicBox(length=mm(1.0))), fields={"flow": flow})
    assert raised.value.refusals[0].code == "medium.flow-twice"
