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

Window starts are drawn off the grid, at least a twentieth of a step from a step's end, since there the node decides
inclusion by its own rounding of ``k dt`` (float32's is a thousandth of a step here); on the grid, the rule's tie is
tested in float64, where the node's ``k dt`` and the series' are the same product."""

import importlib

import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from rigs import falling_bead, gain_per_step

from mime.testing.toys import coupled_toy
from mime.units import s

PROPERTY = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def bound(speeds: np.ndarray) -> float:
    """The node's rounding over its window's ``len(speeds)`` steps, on the scale of the largest speed."""
    eps = float(np.finfo(speeds.dtype).eps)
    return 8.0 * (len(speeds) + 1) * eps * float(np.max(np.abs(speeds.astype(np.float64))))


def mean_failures(result, swim, start: float) -> list[str]:
    """How the node's ``mean_speed`` differs from the series' mean over (start, end], beyond the node's rounding."""
    window = result[swim].speed.between(s(start))
    node, series = result[swim].mean_speed.value, window.mean().value
    tolerance = bound(window.raw)
    return [] if abs(node - series) <= tolerance else [
        f"mean_speed {node!r} m/s, the series over {window.start!r} to {window.end!r} s gives {series!r} m/s: "
        f"{abs(node - series):.3g} apart, beyond {tolerance:.3g}"]


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
