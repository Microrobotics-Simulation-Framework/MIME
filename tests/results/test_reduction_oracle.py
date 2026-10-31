"""The gate's oracle (design #184 item 7): an output the graph reduces is the series of the per-step output it reduces,
taken over the output's window by the one inclusion rule and reduced by the node's own rule, to the rounding of the
node's sums. ``SwimSpeed``'s ``mean_speed`` against ``speed.between(after, end).mean()``, its ``distance`` against
``.integral()`` and its ``observed`` against the window's steps.

The control: the same window shifted by one step, either way, fails the same check at every draw. The bead falls with
nothing to resist it, so its speed grows by ``g' dt`` every step (``g'`` gravity less buoyancy), and a window one step
longer or shorter moves the mean of a linear ramp by exactly half a step's gain, ``g' dt / 2``. The bound is the node's
own rounding: its sums of ``c`` terms in its dtype, each within ``c`` units of rounding of the largest speed, so
``8 (c + 1) eps max|v|`` for the mean (a factor of four over the worst case). Over the draws (at most ``n = 200``
steps, the start speed within one run's gain of zero, so ``max|v| <= 2 g' n dt``), ``g' dt / 2`` over that bound is
at least ``1 / (32 (c + 1) n eps)``, above six in float32: the control cannot pass, whatever is drawn.

The first property draws window starts off the grid, at least a twentieth of a step from a step's end. On the grid the
node decides a tie by its own rounding of ``k dt`` in its float type (in float32 it counts the tied step for about
57 % of starts at the base steps here, and a float64 comparison would not), and the series decides a start the same
way, so a second property, over float32 starts on the grid and nothing else, holds the two equal; its control is the
float64 rule applied to a float32 run, on starts where the node counts the tied step, which fails at every draw by a
whole step. A third has a body whose speed is not linear in time, a sphere under Stokes drag approaching its terminal
speed, where the mean is not the midpoint of the window's first and last speeds, and fails if the series' mean is.
The float64 on-grid cases and the constant-speed cases with ``every=`` close the rest."""

import importlib
from unittest import mock

import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from rigs import GRAVITY, WATER, falling_bead, gain_per_step, viscosity_for

import mime.results._series as series_module
from mime.results._layout import position
from mime.testing.toys import coupled_toy
from mime.units import ms, s

PROPERTY = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def bound(speeds: np.ndarray) -> float:
    """The node's rounding over its window's ``len(speeds)`` steps, on the scale of the largest speed."""
    eps = float(np.finfo(speeds.dtype).eps)
    return 8.0 * (len(speeds) + 1) * eps * float(np.max(np.abs(speeds.astype(np.float64))))


def mean_ratio(result, swim, start: float) -> float:
    """How far the node's ``mean_speed`` is from the series' mean over (start, end], in units of the node's rounding."""
    window = result[swim].speed.between(s(start))
    return abs(result[swim].mean_speed.value - window.mean().value) / bound(window.raw)


def mean_failures(result, swim, start: float) -> list[str]:
    """How the node's ``mean_speed`` differs from the series' mean over (start, end], beyond the node's rounding."""
    window = result[swim].speed.between(s(start))
    node, series = result[swim].mean_speed.value, window.mean().value
    tolerance = bound(window.raw)
    return [] if abs(node - series) <= tolerance else [
        f"mean_speed {node!r} m/s, the series over {window.start!r} to {window.end!r} s gives {series!r} m/s: "
        f"{abs(node - series):.3g} apart, beyond {tolerance:.3g}"]


def nodes_steps(result, swim) -> int:
    """The steps the node counted: the time it observed over the base step."""
    return round(float(result[swim].observed.value) / result.dt)


def reduction_failures(result, swim, start: float) -> list[str]:
    """The distance and the time observed, against the series' integral and its window's steps."""
    window = result[swim].speed.between(s(start))
    observed, distance = result[swim].observed.value, result[swim].distance.value
    steps = len(window) * window.dt
    found = []
    if abs(distance - window.integral().value) > bound(window.raw) * steps:
        found.append(f"distance {distance!r} m against the integral {window.integral().value!r} m")
    if abs(observed - steps) > 8.0 * (len(window) + 1) * float(np.finfo(window.raw.dtype).eps) * steps:
        found.append(f"observed {observed!r} s against the window's {len(window)} steps, {steps!r} s")
    return found


@PROPERTY
@given(dt=st.floats(1e-4, 2e-3), steps=st.integers(20, 200), density=st.floats(1100.0, 8000.0),
       volume=st.floats(1e-12, 1e-8), fall=st.floats(-1.0, 1.0), data=st.data())
def test_the_mean_speed_is_its_series_mean_over_its_window_and_a_window_one_step_off_is_not(dt, steps, density,
                                                                                           volume, fall, data):
    start = data.draw(st.integers(1, steps - 3), label="the window starts in step")
    after = (start + data.draw(st.floats(0.05, 0.95), label="and this far into it")) * dt
    experiment, swim = falling_bead(dt, after, density, volume, fall * gain_per_step(dt, density) * steps)
    result = experiment.build().run(s(steps * dt), record=[swim.speed])
    assert len(result[swim].speed) == steps
    assert mean_failures(result, swim, after) == []
    assert reduction_failures(result, swim, after) == []
    for shifted in (after - dt, after + dt):
        assert mean_failures(result, swim, shifted), f"the window shifted to {shifted!r} s still passed"


def node_takes_the_tied_step(dt: float, start: int) -> bool:
    """Whether the node, in float32, counts the step that ends exactly at ``start dt``: its own test, restated."""
    return bool(np.float32(start) * np.float32(dt) > np.float32(start * dt))


def float64_rule(steps, start, dt, clock):
    """The rule the series decided a start by before it followed the node's: exact in float64, the tied step left
    out."""
    return steps > position(start, dt)


RAMP = {"dt": st.floats(1e-4, 2e-3), "steps": st.integers(20, 200), "density": st.floats(1100.0, 8000.0),
        "volume": st.floats(1e-12, 1e-8), "fall": st.floats(-1.0, 1.0)}


@PROPERTY
@given(**RAMP, data=st.data())
def test_on_the_grid_in_float32_the_mean_speed_is_its_series_mean_and_the_series_has_the_steps_the_node_counted(
        dt, steps, density, volume, fall, data):
    start = data.draw(st.integers(1, steps - 3), label="the window starts at the end of step")
    after = start * dt
    experiment, swim = falling_bead(dt, after, density, volume, fall * gain_per_step(dt, density) * steps)
    result = experiment.build().run(s(steps * dt), record=[swim.speed])
    window = result[swim].speed.between(s(after))
    assert window.raw.dtype == np.float32
    assert len(window) == nodes_steps(result, swim)
    assert mean_failures(result, swim, after) == []
    assert reduction_failures(result, swim, after) == []


#: The least a start the float64 rule gets wrong is from the node's, over the node's rounding: a whole step more or
#: less moves a ramp's mean by half a step's gain, which is at least 1 / (32 (c + 1) n eps), above six in float32, at
#: any draw.
TIED_STEP_FLOOR = 6.0


@PROPERTY
@given(**RAMP, data=st.data())
def test_control_the_float64_rule_on_a_float32_run_is_a_step_off_at_every_start_where_the_node_counts_the_tie(
        dt, steps, density, volume, fall, data):
    ties = [j for j in range(1, steps - 2) if node_takes_the_tied_step(dt, j)]
    assume(ties)
    start = data.draw(st.sampled_from(ties), label="the window starts at the end of step, one the node ties on")
    after = start * dt
    experiment, swim = falling_bead(dt, after, density, volume, fall * gain_per_step(dt, density) * steps)
    result = experiment.build().run(s(steps * dt), record=[swim.speed])
    assert mean_failures(result, swim, after) == []
    with mock.patch.object(series_module, "ended_after", float64_rule):
        window = result[swim].speed.between(s(after))
        assert nodes_steps(result, swim) == len(window) + 1, "the node counted the tied step; the float64 rule did not"
        assert mean_ratio(result, swim, after) > TIED_STEP_FLOOR
        assert mean_failures(result, swim, after) != []


def test_the_oracles_own_bead_on_the_grid_in_float32_where_the_node_counts_the_tie_agrees_with_its_series():
    """Review finding 1's case: ``dt`` 1e-3 s, 60 steps, a start at the end of step 5, 9 or 10, where the node counts
    the tied step. The series held 55 samples against the node's 56 and its mean was off by 156 to 171 times the
    oracle's bound."""
    dt = 1e-3
    for start in (5, 9, 10):
        assert node_takes_the_tied_step(dt, start)
        experiment, swim = falling_bead(dt, start * dt, 2500.0, 1e-9, 0.0)
        result = experiment.build().run(s(60 * dt), record=[swim.speed])
        assert len(result[swim].speed.between(s(start * dt))) == nodes_steps(result, swim) == 60 - start + 1
        assert mean_failures(result, swim, start * dt) == [] and reduction_failures(result, swim, start * dt) == []


NONLINEAR = {"dt": st.floats(1e-4, 2e-3), "steps": st.integers(20, 200), "density": st.floats(1100.0, 8000.0),
             "volume": st.floats(1e-12, 1e-8), "fall": st.floats(-1.0, 0.3)}


@PROPERTY
@given(**NONLINEAR, time_constant=st.floats(0.1, 0.5), data=st.data())
def test_a_speed_that_is_not_linear_in_time_has_its_mean_as_its_series_mean_and_not_as_its_ends_midpoint(
        dt, steps, density, volume, fall, time_constant, data):
    """A sphere in water under Stokes drag, its speed approaching the terminal speed as an exponential of time
    constant ``time_constant`` of the run, from a start speed between the terminal speed reversed and a third of it,
    so the speed is curved over a window of at least half the run. Starts on the grid or off it."""
    start = data.draw(st.integers(1, steps // 2), label="the window starts at the end of step")
    into = data.draw(st.one_of(st.just(0.0), st.floats(0.05, 0.95)), label="and this far into the next")
    after = (start + into) * dt
    tau = time_constant * steps * dt
    terminal = (1.0 - WATER / density) * GRAVITY * tau
    experiment, swim = falling_bead(dt, after, density, volume, fall * terminal, viscosity_for(density, volume, tau))
    result = experiment.build().run(s(steps * dt), record=[swim.speed])
    window = result[swim].speed.between(s(after))
    assert len(window) == nodes_steps(result, swim)
    assert mean_failures(result, swim, after) == []
    assert reduction_failures(result, swim, after) == []
    midpoint = (float(window.raw[0]) + float(window.raw[-1])) / 2.0
    assert abs(result[swim].mean_speed.value - midpoint) > bound(window.raw), "the rig cannot tell mean from midpoint"


@pytest.mark.parametrize("every", [3, 5])
def test_a_constant_speed_integrates_to_its_speed_times_the_span_whatever_the_spacing_of_its_samples(every):
    """Neutrally buoyant, released at 0.25 m/s with nothing to change it: its speed is the same at every step, so the
    integral of samples taken every ``every`` steps is the speed times the span, and the same as the node's distance. A
    weight of one base step on each sample would give the span over ``every``, which the control shows this test
    tells from the right one."""
    dt, steps, speed = 1e-3, 60, 0.25
    experiment, swim = falling_bead(dt, 0.0, WATER, 1e-9, speed)
    result = experiment.build().run(s(steps * dt), record=[swim.speed], every=ms(every * dt * 1e3))
    series = result[swim].speed
    span = steps * dt
    tolerance = 8.0 * (steps + 1) * float(np.finfo(series.raw.dtype).eps) * speed * span
    assert len(series) == steps // every and series.every == every
    assert abs(series.integral().value - speed * span) <= tolerance
    assert abs(result[swim].distance.value - speed * span) <= tolerance
    assert abs(series.mean().value - speed) <= bound(series.raw)
    wrong = float(np.asarray(series.raw, dtype=np.float64).sum()) * dt
    assert abs(wrong - speed * span) > 1e3 * tolerance, "a weight of dt per sample is not told from every * dt"


@pytest.fixture
def float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


@pytest.mark.parametrize(("dt", "steps", "start"), [(5e-4, 60, 20), (1e-3, 41, 1), (2.5e-4, 100, 97), (0.1, 30, 7)])
def test_on_the_grid_the_step_ending_at_the_windows_start_is_left_out_as_the_node_leaves_it(float64, dt, steps,
                                                                                            start):
    after = start * dt
    experiment, swim = falling_bead(dt, after, 2500.0, 1e-9, 0.0)
    result = experiment.build().run(s(steps * dt), record=[swim.speed])
    assert result[swim].speed.raw.dtype == np.float64
    assert len(result[swim].speed.between(s(after))) == steps - start
    assert mean_failures(result, swim, after) == [] and reduction_failures(result, swim, after) == []
    assert mean_failures(result, swim, after - dt) and mean_failures(result, swim, after + dt)


def lab_fixture():
    return importlib.import_module("mime_lab_fixture")


def test_a_labs_observation_reduces_by_its_own_rule_and_its_series_gives_the_same():
    """The lab fixture's ``labx.height`` declares its integral the way MIME's observations declare theirs, and gets the
    same accessors with no MIME edit: its ``height_integral`` is its ``height`` series' integral over the run, and a
    window one step short is not."""
    experiment = coupled_toy()
    height = experiment.observe(lab_fixture().Height(body="probe"))
    result = experiment.build().run(s(0.6), record=[height.height])
    series = result[height].height
    node = result[height].height_integral
    assert node.unit == "m s" and series.integral().unit == "m s"
    tolerance = bound(series.raw) * len(series) * series.dt
    assert abs(node.value - series.integral().value) <= tolerance
    assert abs(node.value - series.between(s(series.dt)).integral().value) > tolerance


def steady_harness():
    """The differential harness's window rules, read from its own file (``tests/differential/script_a_steady.py``),
    which this phase does not change."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).parents[1] / "differential" / "script_a_steady.py"
    spec = importlib.util.spec_from_file_location("script_a_steady_rules", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_harness_displacement_rule_agrees_with_the_window_rule_and_its_mean_velocity_rule_is_its_own(float64):
    """Script A's harness keeps its rules (preregistered): its history starts with the state at t = 0, and its window
    is closed, ``t >= t_start``. Its speed by displacement between the samples at ``t_start`` and ``t_end`` is the
    mean over the steps that end after ``t_start``, so it agrees with this phase's rule to rounding; its mean
    velocity counts the sample at ``t_start`` as well, one sample more, and so differs by more than rounding."""
    harness = steady_harness()
    dt, steps, start = 1e-3, 100, 40
    experiment, swim = falling_bead(dt, start * dt, 2500.0, 1e-9, 0.0)
    result = experiment.build().run(s(steps * dt), history=True)
    held = result.history["bead"]
    time = np.arange(0, steps + 1) * dt
    zero = np.zeros((1, 3))
    history = harness.History(time, np.vstack([zero, np.asarray(held["pose"])[:, :3]]),
                              np.vstack([zero, np.asarray(held["twist"])[:, :3]]), None, None, None)
    down = harness.Frame(axis=(0.0, 0.0, -1.0), angle_zero=(1.0, 0.0, 0.0), moment_in_body=(1.0, 0.0, 0.0))
    mean = result[swim].mean_speed.value
    tolerance = bound(result[swim].speed.between(s(start * dt)).raw)
    assert abs(harness.speed_between(history, down, start * dt, steps * dt) - mean) <= tolerance
    closed = float(np.mean(history.velocity[time >= start * dt - 1e-12] @ np.asarray(down.axis)))
    assert abs(closed - mean) > 1e3 * tolerance
