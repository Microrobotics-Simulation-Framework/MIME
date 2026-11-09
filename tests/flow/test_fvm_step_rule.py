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
from fvm_cases import in_dtype
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
#: The tube the prediction takes: widened by the band's reach (``WALL_REACH_CELLS``).
WIDE = RADIUS + steps.WALL_REACH_CELLS * H


def experiment(*, dt=ms(0.1), g=2.0, amplitude=None, frequency=None, phase=0.0, field=None, viscosity=1.0, seed=0,
               duration=None, **settings):
    exp = Experiment("periodic_tube", scene=Scene.new(up=Z, units="m"), dt=dt, seed=seed, duration=duration)
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(RADIUS * 1e3), wall=mm(0.2),
                                                      length=mm(10.0))))
    flow = field if field is not None else FVM(domain=PeriodicBox(length=mm(0.25)), cells_across=CELLS, **settings)
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(viscosity), reason="water near 20 C, or a viscous test fluid")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    pulsatile = {} if amplitude is None else {"amplitude": m_per_s2(amplitude), "frequency": Hz(frequency),
                                              "phase": phase}
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
    viscous = 0.15 * H * H / NU
    g = 2.0 * 4.0  # a drive four times stronger, so that RK3's viscous limit is the larger bound
    peak4 = g * WIDE ** 2 / (4 * NU)
    over = 1.05 * RK3_LINE * H / peak4
    assert over < viscous
    with pytest.raises(BuildRefused) as raised:
        experiment(dt=s(over), g=g).build()
    assert "flow.cfl-predicted" in _codes(raised)
    refusal = next(r for r in raised.value.refusals if r.code == "flow.cfl-predicted")
    assert "use dt <=" in refusal.action and "dt='auto'" in refusal.action
    experiment(dt=s(0.95 * RK3_LINE * H / peak4), g=g).build()


RK3_LINE = 1.7


def test_a_harmonic_drives_peak_enters_the_prediction(float64):
    """A pulsatile drive at alpha 6 whose steady part alone is within the line and whose harmonic takes it past."""
    omega = (6.0 / RADIUS) ** 2 * NU
    f = omega / (2 * math.pi)
    g, amplitude = 1.0, 40.0
    peak = steps.predicted_peak(steps.Drive(g, amplitude, f), WIDE, NU)
    dt = 1.1 * RK3_LINE * H / peak
    assert g * WIDE ** 2 / (4 * NU) * dt / H < RK3_LINE  # the mean alone would pass
    with pytest.raises(BuildRefused) as raised:
        experiment(dt=s(dt), g=g, amplitude=amplitude, frequency=f, scheme=Scheme.RK3()).build()
    assert "flow.cfl-predicted" in _codes(raised)


def test_control_a_prediction_from_the_drives_mean_misses_the_refusal(float64, monkeypatch):
    omega = (6.0 / RADIUS) ** 2 * NU
    f = omega / (2 * math.pi)
    peak = steps.predicted_peak(steps.Drive(1.0, 40.0, f), WIDE, NU)
    dt = 1.1 * RK3_LINE * H / peak
    import mime.dynamics.flow._fvm as fvm

    monkeypatch.setattr(fvm, "predicted_peak", lambda drive, radius, nu: abs(drive.steady) * radius ** 2 / (4 * nu))
    experiment(dt=s(dt), g=1.0, amplitude=40.0, frequency=f, scheme=Scheme.RK3()).build()


def test_control_a_prediction_on_twice_the_spacing_misses_the_refusal(float64, monkeypatch):
    import mime.dynamics.flow._fvm as fvm

    g = 8.0
    over = 1.5 * RK3_LINE * H / (g * WIDE ** 2 / (4 * NU))
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


# -- the prediction against a run from rest -------------------------------------------------------------------------

def _largest_speed_from_rest(alpha, *, phase, periods=1.5):
    """dt="auto" at D/h 16, a harmonic drive at ``alpha`` from rest at ``phase``: the largest axial speed over the run
    (m/s) and the prediction's, which takes the periodic peak doubled (``START_FROM_REST``)."""
    import jax
    import jax.numpy as jnp

    f = (alpha / RADIUS) ** 2 * NU / (2 * math.pi)
    built = experiment(dt="auto", g=0.0, amplitude=20.0, frequency=f, phase=phase).build()
    node = built.graph.get_node("water.flow")
    assert isinstance(node, ExplicitFlowNode)
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    dt = built.report.dt
    law = next(name for name in built.graph.node_names if name.startswith("body_force"))
    drive = jnp.asarray(built.graph.get_node(law).initial_state()["waveform"])

    def body(state, _):
        ended = node.advance(p, state, {"waveform": drive}, dt)
        return ended, jnp.max(jnp.abs(ended["velocity_z"]))

    _, speeds = jax.jit(lambda s0: jax.lax.scan(body, s0, None, length=round(periods / (f * dt))))(
        dict(node.initial_state()))
    return float(np.max(np.asarray(speeds))), steps.predicted_peak(steps.Drive(0.0, 20.0, f), WIDE, NU)


@pytest.mark.parametrize("alpha", [6.0, 14.0])
@pytest.mark.parametrize("phase", [0.0, math.pi / 2])
def test_the_prediction_bounds_a_run_from_rest_at_any_phase(float64, alpha, phase):
    """The speed from rest stays under the steady centreline plus twice the harmonic's periodic peak (the maximum
    principle); the worst phase starts the drive at zero. Measured: 1.08 to 1.97 times the largest speed."""
    largest, predicted = _largest_speed_from_rest(alpha, phase=phase)
    assert predicted >= largest


@pytest.mark.parametrize("alpha", [6.0, 14.0])
def test_control_the_periodic_peak_alone_misses_a_run_from_rest(float64, alpha):
    """Without the doubling the prediction falls under the run's speed at the worst phase (measured 0.54 to 0.56 of
    it): the test above can fail."""
    largest, predicted = _largest_speed_from_rest(alpha, phase=math.pi / 2)
    assert predicted / steps.START_FROM_REST < 0.8 * largest


# -- the step the flow's node takes (its rate=) ---------------------------------------------------------------------

#: A drive whose predicted CFL is 0.68 at 5e-5 s, so 6.8 at ten times that step.
G_AT_RATE = 0.4 * RK3_LINE * H / 5e-5 * 4 * NU / WIDE ** 2


def test_the_cfl_line_is_judged_at_the_flows_rate_not_the_base_step(dtype):
    """``FVM(rate=...)`` steps the node at ten base steps: the line is judged there (CFL 6.8, refused), where the same
    base step without a rate builds at CFL 0.68."""
    built = experiment(dt=s(5e-5), g=G_AT_RATE).build()
    assert any("max|u| dt / h = 0.68" in line for line in built.report.checked["water.flow"])
    with pytest.raises(BuildRefused) as raised:
        experiment(dt=s(5e-5), g=G_AT_RATE, rate=s(5e-4)).build()
    refusal = next(r for r in raised.value.refusals if r.code == "flow.cfl-predicted")
    assert "= 6.8 at the flow's step 0.0005 s" in refusal.message
    assert "every 10 base steps" in refusal.action and "use dt <= 1.25e-05 s" in refusal.action


def test_rk3_named_past_its_viscous_number_at_its_rate_is_refused(dtype):
    """RK3 at a rate of twenty base steps: ``nu dt / h^2`` 0.256 at its own step (past 0.15) is refused; the base step
    alone (0.0128) builds."""
    experiment(dt=s(5e-5), g=0.05, scheme=Scheme.RK3()).build()
    with pytest.raises(BuildRefused) as raised:
        experiment(dt=s(5e-5), g=0.05, rate=s(1e-3), scheme=Scheme.RK3()).build()
    refusal = next(r for r in raised.value.refusals if r.code == "flow.viscous-limit")
    assert "nu dt / h^2 = 0.256" in refusal.message


def test_the_scheme_built_at_a_rate_is_the_one_the_report_names(dtype):
    """No scheme named, a rate of twenty base steps: the build steps IMEX Euler at the flow's step, and the report names
    it, the step and its viscous number there."""
    built = experiment(dt=s(5e-5), g=0.05, rate=s(1e-3)).build()
    assert type(built.graph.get_node("water.flow")) is ImplicitFlowNode
    lines = built.report.checked["water.flow"]
    assert "the flow's step: 0.001 s, every 20 base steps (its rate=)" in lines
    assert any(line.startswith("the flow's scheme: imex_euler, at nu dt / h^2 = 0.256") for line in lines)


def test_auto_refuses_a_rate_on_the_field_whose_step_it_chooses(dtype):
    with pytest.raises(BuildRefused) as raised:
        experiment(dt="auto", g=2.0, rate=ms(1.0)).build()
    assert "run.dt-auto-rate" in _codes(raised)


def _with_a_second_medium(rate):
    exp = experiment(dt="auto", g=2.0, amplitude=1.0, frequency=50.0)
    exp.add_structure(Structure("other", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0),
                                                       centre=mm((5.0, 0.0, 0.0)))))
    exp.add_medium(Medium("saline", fills="other.lumen", fields={"flow": Womersley(rate=rate)}))
    exp.declare("saline.viscosity", mPa_s(1.0), reason="saline near 20 C")
    exp.declare("saline.density", kg_per_m3(1000.0), reason="saline near 20 C")
    return exp


def test_auto_keeps_a_rate_declared_elsewhere_or_refuses_it_with_its_reason(dtype):
    """A rate planned before ``dt="auto"``'s step is chosen is made a whole multiple of it once it is: three base steps
    builds a node at three steps; two and a half is refused as any rate that is not a whole multiple."""
    dt = experiment(dt="auto", g=2.0, amplitude=1.0, frequency=50.0).build().report.dt
    built = _with_a_second_medium(s(3.0 * dt)).build()
    assert built.graph.get_node("saline.flow").delta_t == pytest.approx(3.0 * dt, rel=1e-12)
    with pytest.raises(BuildRefused) as raised:
        _with_a_second_medium(s(2.5 * dt)).build()
    assert "run.rate-not-multiple" in _codes(raised)


def test_auto_checks_the_seed(dtype):
    with pytest.raises(BuildRefused) as raised:
        experiment(dt="auto", g=2.0, seed="x").build()
    assert "run.seed" in _codes(raised)


def test_auto_divides_a_declared_duration_and_the_drives_period(dtype):
    """With a steady drive the step divides the declared duration; with a periodic one, both (0.0101 s against a 0.02 s
    period: 200 steps a period)."""
    for pulsatile, duration in ((False, 1.0), (True, 1.0), (True, 0.0101)):
        exp = experiment(dt="auto", g=2.0, duration=s(duration),
                         **({"amplitude": 1.0, "frequency": 50.0} if pulsatile else {}))
        dt = exp.build().report.dt
        assert abs(duration / dt - round(duration / dt)) < 1e-9 * duration / dt
        if pulsatile:
            assert abs(0.02 / dt - round(0.02 / dt)) < 1e-9 * 0.02 / dt


# -- the prediction's dtype and its line ----------------------------------------------------------------------------

def _scipy_peak(alpha):
    import scipy.special

    rho = np.linspace(0.0, 1.0, 20001)
    b = np.exp(-1j * np.pi / 4) * alpha
    return float(np.max(np.abs((1 - scipy.special.jv(0, b * rho) / scipy.special.jv(0, b)) / alpha ** 2)))


def _run_dtype_peak(alpha):
    """The prediction as it was: the series evaluated in the run's dtype, with no line."""
    import jax.numpy as jnp

    from mime.nodes.flow._womersley import profile, series_terms

    rho = np.linspace(0.0, 1.0, 1001)
    re, im = profile(jnp.asarray(rho * rho), jnp.asarray(alpha * alpha), series_terms(np.float64))
    return float(np.max(np.hypot(np.asarray(re, float), np.asarray(im, float))))


def _holds(peak, alpha, reference):
    """The prediction's harmonic peak at ``alpha`` is float64's (``reference``) and a bound: at least the periodic
    peak (SciPy's, on 20 times the radii: the prediction's 1001 radii read it 3.6e-6 under at alpha 50) and, doubled,
    at least the maximum principle's ``2 / alpha^2`` from rest."""
    return (math.isfinite(peak) and abs(peak / reference - 1.0) < 1e-12 and peak >= _scipy_peak(alpha) * (1 - 1e-5)
            and steps.START_FROM_REST * peak >= 2.0 / alpha ** 2)


FLOAT32_ALPHAS = [60.0, 65.0, 70.0, 75.0]


@pytest.fixture(scope="module")
def float64_peaks():
    import jax

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return {alpha: steps.harmonic_peak(alpha) for alpha in [40.0, 50.0, *FLOAT32_ALPHAS]}
    finally:
        jax.config.update("jax_enable_x64", held)


@pytest.mark.parametrize("alpha", [40.0, 50.0, *FLOAT32_ALPHAS])
def test_the_prediction_is_float64_arithmetic_and_a_bound_in_float32(alpha, float64_peaks):
    with in_dtype("float32"):
        assert _holds(steps.harmonic_peak(alpha), alpha, float64_peaks[alpha])


@pytest.mark.parametrize("alpha", FLOAT32_ALPHAS)
def test_control_the_series_in_the_runs_dtype_fails_in_float32(alpha, float64_peaks):
    """The review's float32 readings of the old prediction: -1.8 % at 60, +166 % at 65, zero at 70, NaN at 75."""
    with in_dtype("float32"):
        assert not _holds(_run_dtype_peak(alpha), alpha, float64_peaks[alpha])


@pytest.mark.parametrize("alpha", [55.0, 80.0, 120.0, 400.0])
def test_past_the_line_the_peak_is_the_lines_scaled_and_a_bound(alpha):
    peak = steps.harmonic_peak(alpha)
    assert not steps.series_holds(alpha)
    assert peak == steps.series_peak(steps.SERIES_LINE) * (steps.SERIES_LINE / alpha) ** 2
    assert steps.START_FROM_REST * peak >= 2.0 / alpha ** 2
    if alpha <= 120.0:
        assert peak >= _scipy_peak(alpha)


def test_control_past_the_line_the_maximum_principles_bound_halved_is_under_the_periodic_peak():
    """The bound the scaled value is held to can fail: ``1 / alpha^2`` (half of it) is under the periodic peak
    ``1.07 / alpha^2`` at every alpha the line passes."""
    assert all(1.0 / alpha ** 2 < _scipy_peak(alpha) for alpha in (55.0, 80.0, 120.0))


def _pulsatile(alpha, dt):
    f = (alpha / RADIUS) ** 2 * NU / (2 * math.pi)
    return experiment(dt=dt, g=0.0, amplitude=2.0e5, frequency=f, scheme=Scheme.RK3())


def _auto_steps(alpha):
    chosen = {}
    for name in ("float64", "float32"):
        with in_dtype(name):
            built = _pulsatile(alpha, "auto").build()
            assert any("past the series' line of 50" in line for line in built.report.checked["water.flow"])
            chosen[name] = built.report.dt
    return chosen


@pytest.mark.parametrize("alpha", [65.0, 82.0])
def test_a_float32_build_past_the_series_line_is_refused_and_steps_as_float64_does(alpha):
    """R 0.5 mm, A 2e5 m/s^2: dt 5e-4 s is refused in both dtypes, and dt='auto' takes float64's step to the bit in
    float32 (the float32 series read +166 % at 65 and NaN at 82, where it built at 5e-4 and took a step 37 times
    float64's)."""
    for name in ("float64", "float32"):
        with in_dtype(name):
            with pytest.raises(BuildRefused) as raised:
                _pulsatile(alpha, s(5e-4)).build()
            assert "flow.cfl-predicted" in _codes(raised)
    chosen = _auto_steps(alpha)
    assert chosen["float32"] == chosen["float64"]


@pytest.mark.parametrize("alpha", [65.0, 82.0])
def test_control_the_series_in_the_runs_dtype_moves_float32s_auto_step(alpha, monkeypatch):
    """With the series in the run's dtype and no line, float32's dt='auto' is not float64's: 3.6 times smaller at 65,
    refused at 82 (its NaN, which the guard on a non-finite prediction catches)."""
    float64_step = _auto_steps(alpha)["float64"]
    monkeypatch.setattr(steps, "harmonic_peak", _run_dtype_peak)
    with in_dtype("float32"):
        try:
            moved = _pulsatile(alpha, "auto").build().report.dt
        except BuildRefused:
            moved = None
    assert moved != float64_step


def test_a_prediction_that_is_not_finite_is_refused_and_bounds_no_auto_step(dtype, monkeypatch):
    import mime.dynamics.flow._fvm as fvm

    experiment(dt=s(1e-4)).build()  # the control: the same case predicted builds
    monkeypatch.setattr(fvm, "predicted_peak", lambda *_: math.nan)
    for dt in (s(1e-4), "auto"):
        with pytest.raises(BuildRefused) as raised:
            experiment(dt=dt).build()
        assert _codes(raised) == ["flow.cfl-predicted"]


def test_a_series_that_is_not_finite_inside_the_line_counts_as_past_it(monkeypatch):
    line = steps.series_peak(steps.SERIES_LINE)
    original = steps.series_peak
    monkeypatch.setattr(steps, "series_peak", lambda alpha: original(alpha) if alpha == steps.SERIES_LINE else math.nan)
    assert not steps.series_holds(20.0)
    assert steps.harmonic_peak(20.0) == line * (steps.SERIES_LINE / 20.0) ** 2


@pytest.mark.parametrize("axis", [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 3.0)])
def test_the_builds_line_is_the_nodes_cfl_limit(float64, axis):
    from mime.dynamics.flow._fvm import host_line
    from mime.nodes.flow.fvm import RK3

    assert host_line("rk3", axis) == pytest.approx(float(RK3().cfl_limit(axis)), rel=1e-15)
