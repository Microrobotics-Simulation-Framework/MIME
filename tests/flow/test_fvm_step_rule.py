"""The flow's step rule through the experiment (F1b's gate 3 (a), (b) and the auto step): the scheme from the viscous
limit (RK3 first), the CFL predicted at build from the drive and refused before any step past the chosen scheme's line
with the route in plain words, ``stokes=True`` exempt from the line, ``dt="auto"`` the largest stable step (dividing
the drive's period), its save, and the controls (a prediction that reads the drive's mean, or the wrong spacing,
misses the refusal)."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

import mime.dynamics.flow._steps as steps
from mime.build import BuildRefused
from mime.dynamics import FVM, Scheme, Womersley
from mime.dynamics.flow import chosen_scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.nodes.flow.fvm import ExplicitFlowNode, ImplicitFlowNode
from mime.scene import PeriodicBox, Scene, Tube, Z
from mime.things import Medium, Structure
from mime.units import Hz, kg_per_m3, m_per_s2, mm, mPa_s, ms, s

RADIUS, NU, CELLS = 0.5e-3, 1e-6, 16
H = 0.25e-3 / math.ceil(0.25e-3 / (2 * RADIUS / CELLS))


def experiment(*, dt=ms(0.1), g=2.0, amplitude=None, frequency=None, field=None, viscosity=1.0, **settings):
    exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=dt)
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(RADIUS * 1e3), wall=mm(0.2),
                                                      length=mm(10.0))))
    flow = field if field is not None else FVM(domain=PeriodicBox(length=mm(0.25)), cells_across=CELLS, **settings)
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(viscosity), reason="water near 20 C, or a viscous test fluid")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    pulsatile = {} if amplitude is None else {"amplitude": m_per_s2(amplitude), "frequency": Hz(frequency)}
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(g), **pulsatile), medium="water")
    return exp


def _codes(raised):
    return [r.code for r in raised.value.refusals]


@pytest.mark.parametrize(("number", "expected"), [(0.01, "rk3"), (0.15, "rk3"), (0.1501, "imex_euler"), (1.0, "imex_euler")])
def test_the_scheme_is_rk3_where_its_viscous_number_holds(number, expected):
    assert chosen_scheme(None, NU, H, number * H * H / NU) == expected


def test_a_named_scheme_is_kept_whatever_the_viscous_number():
    assert chosen_scheme(Scheme.IMEXEuler(), NU, H, 0.01 * H * H / NU) == "imex_euler"
    assert chosen_scheme(Scheme.RK3(), NU, H, 1.0 * H * H / NU) == "rk3"


def test_the_build_steps_rk3_on_the_explicit_node_and_imex_euler_on_the_implicit_one(float64):
    rk3 = experiment(dt=s(0.1 * H * H / NU)).build()
    imex = experiment(dt=s(0.5 * H * H / NU), g=0.05).build()
    assert isinstance(rk3.graph.get_node("water.flow"), ExplicitFlowNode)
    assert type(imex.graph.get_node("water.flow")) is ImplicitFlowNode


def test_a_dt_past_the_predicted_line_is_refused_before_any_step_with_its_route(float64):
    """Poiseuille's centreline at g 2 m/s^2 is 0.125 m/s; RK3's line 1.7 along z: dt above 1.7 h / 0.125 is refused,
    and the route's step builds."""
    peak = 2.0 * RADIUS ** 2 / (4 * NU)
    limit = RK3_LINE * H / peak
    viscous = 0.15 * H * H / NU
    g = 2.0 * 4.0  # a drive four times stronger, so that RK3's viscous limit is the larger bound
    peak4 = g * RADIUS ** 2 / (4 * NU)
    over = 1.05 * RK3_LINE * H / peak4
    assert over < viscous
    with pytest.raises(BuildRefused) as raised:
        experiment(dt=s(over), g=g).build()
    assert "flow.cfl-predicted" in _codes(raised)
    refusal = next(r for r in raised.value.refusals if r.code == "flow.cfl-predicted")
    assert "use dt <=" in refusal.action and "dt='auto'" in refusal.action
    experiment(dt=s(0.95 * RK3_LINE * H / peak4), g=g).build()
    del limit


RK3_LINE = 1.7


def test_a_harmonic_drives_peak_enters_the_prediction(float64):
    """A pulsatile drive at alpha 6 whose steady part alone is within the line and whose harmonic takes it past."""
    omega = (6.0 / RADIUS) ** 2 * NU
    f = omega / (2 * math.pi)
    g, amplitude = 1.0, 40.0
    peak = steps.predicted_peak(steps.Drive(g, amplitude, f), RADIUS, NU)
    dt = 1.1 * RK3_LINE * H / peak
    assert g * RADIUS ** 2 / (4 * NU) * dt / H < RK3_LINE  # the mean alone would pass
    with pytest.raises(BuildRefused) as raised:
        experiment(dt=s(dt), g=g, amplitude=amplitude, frequency=f, scheme=Scheme.RK3()).build()
    assert "flow.cfl-predicted" in _codes(raised)


def test_control_a_prediction_from_the_drives_mean_misses_the_refusal(float64, monkeypatch):
    omega = (6.0 / RADIUS) ** 2 * NU
    f = omega / (2 * math.pi)
    peak = steps.predicted_peak(steps.Drive(1.0, 40.0, f), RADIUS, NU)
    dt = 1.1 * RK3_LINE * H / peak
    import mime.dynamics.flow._fvm as fvm

    monkeypatch.setattr(fvm, "predicted_peak", lambda drive, radius, nu: abs(drive.steady) * radius ** 2 / (4 * nu))
    experiment(dt=s(dt), g=1.0, amplitude=40.0, frequency=f, scheme=Scheme.RK3()).build()


def test_control_a_prediction_on_twice_the_spacing_misses_the_refusal(float64, monkeypatch):
    import mime.dynamics.flow._fvm as fvm

    g = 8.0
    over = 1.5 * RK3_LINE * H / (g * RADIUS ** 2 / (4 * NU))
    original = fvm.periodic_tube_layout

    def coarse(*args, **kwargs):
        layout = original(*args, **kwargs)
        return type(layout)(layout.cells, 2.0 * layout.spacing, layout.centre, layout.axis, layout.radius,
                            layout.axial_length, layout.images)

    with pytest.raises(BuildRefused):
        experiment(dt=s(over), g=g).build()
    monkeypatch.setattr(fvm, "periodic_tube_layout", coarse)
    experiment(dt=s(over), g=g).build()


def test_a_stokes_run_is_exempt_from_the_line(float64):
    experiment(dt=s(0.5 * H * H / NU), g=2000.0, stokes=True).build()


def test_rk3_named_past_its_viscous_number_is_refused_with_its_route(float64):
    with pytest.raises(BuildRefused) as raised:
        experiment(dt=s(0.3 * H * H / NU), scheme=Scheme.RK3()).build()
    refusal = next(r for r in raised.value.refusals if r.code == "flow.viscous-limit")
    assert "Scheme.IMEXEuler()" in refusal.action and "dt='auto'" in refusal.action


def test_the_report_states_the_scheme_its_margins_and_the_alternatives_price(float64):
    built = experiment(dt="auto", g=2.0, amplitude=1.0, frequency=50.0).build()
    lines = built.report.checked["water.flow"]
    assert any(line.startswith("the flow's scheme: rk3") for line in lines)
    assert any("the predicted CFL" in line and "margin" in line for line in lines)
    assert any("steps a period" in line for line in lines)
    assert any("imex_euler would be" in line for line in lines)


# -- dt="auto" ------------------------------------------------------------------------------------------------------

def test_auto_takes_rk3s_largest_stable_step_dividing_the_drives_period(float64):
    built = experiment(dt="auto", g=2.0, amplitude=1.0, frequency=50.0).build()
    dt = built.report.dt
    period = 1.0 / 50.0
    assert abs(period / dt - round(period / dt)) < 1e-9
    assert isinstance(built.graph.get_node("water.flow"), ExplicitFlowNode)
    viscous = 0.15 * H * H / NU
    assert dt <= viscous * (1 + 1e-12)
    assert dt > viscous * period / (period + viscous)  # the largest whole division under the bound


def test_auto_takes_imex_euler_where_its_step_is_the_larger(float64):
    """A viscous fluid (cell Re well under 2.7): RK3's viscous step is the smaller, so IMEX Euler's line wins."""
    built = experiment(dt="auto", g=0.02, viscosity=100.0).build()
    assert type(built.graph.get_node("water.flow")) is ImplicitFlowNode


def test_auto_with_nothing_bounding_the_step_is_refused(float64):
    with pytest.raises(BuildRefused) as raised:
        experiment(dt="auto", stokes=True).build()
    assert "run.dt-auto-unset" in _codes(raised)
    with pytest.raises(BuildRefused) as raised:
        experiment(dt="auto", field=Womersley()).build()
    assert "run.dt-auto-unset" in _codes(raised)


def test_auto_saves_the_step_it_chose_and_loads_bit_for_bit(float64, tmp_path: pathlib.Path):
    exp = experiment(dt="auto", g=2.0, amplitude=1.0, frequency=50.0)
    built = exp.build()
    exp.save(tmp_path / "saved")
    loaded = Experiment.load(tmp_path / "saved")
    again = loaded.build()
    assert again.report.dt == built.report.dt
    a = built.run(ms(20.0)).final["water.flow"]["velocity_z"]
    b = again.run(ms(20.0)).final["water.flow"]["velocity_z"]
    assert np.array_equal(np.asarray(a), np.asarray(b))
