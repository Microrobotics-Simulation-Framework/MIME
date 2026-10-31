"""The results phase's gate, measured over many runs: the oracle (``mean_speed`` against the speed series' mean over
its window) and its controls, over the property tests' own draws, run after run. A run is 100 draws of each arm; for
each
it prints whether the oracle held at every draw, whether each control failed at every draw, and the closest margins
(a ratio above one means the check failed: the difference over the node's rounding).

- ``off the grid``: the first property (window starts at least a twentieth of a step off the grid); its control the
  window shifted one step either way.
- ``on the grid``: float32 starts on the grid and nothing else; its control the window closed at its start. ``on the
  grid, ties``: the same, on the starts where the node counts the tied step, drawn from them, and its control the
  float64 rule on a float32 run.
- ``not linear``: a sphere under Stokes drag, starts on the grid and off it; its controls the series' mean as the
  midpoint of the window's first and last speeds, as its middle sample, over a window open at its end, and its integral
  by the trapezoid rule.
- ``every``: a constant speed sampled every 3 and 5 steps; its control the integral weighted by a base step.

    JAX_PLATFORMS=cpu python tests/differential/measure_results_oracle.py [runs [arm ...]]
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "results"))
sys.path.insert(0, str(HERE.parent))

import mime.results._series as series_module  # noqa: E402
from rigs import GRAVITY, WATER, falling_bead, gain_per_step, viscosity_for  # noqa: E402
from test_reduction_oracle import bound, float64_rule, node_takes_the_tied_step  # noqa: E402

from mime.units import ms, s  # noqa: E402

DRAWS = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow], database=None)


@dataclass
class Arm:
    """What one arm measured over a run: the oracle's largest ratio and each control's smallest, with how many draws
    each failed to fail."""

    draws: int = 0
    oracle: float = 0.0
    oracle_failures: int = 0
    controls: dict[str, float] = field(default_factory=dict)
    control_passes: dict[str, int] = field(default_factory=dict)

    def oracle_saw(self, ratio: float) -> None:
        self.draws += 1
        self.oracle = max(self.oracle, ratio)
        self.oracle_failures += ratio > 1.0

    def control_saw(self, name: str, ratio: float) -> None:
        self.controls[name] = min(self.controls.get(name, float("inf")), ratio)
        self.control_passes[name] = self.control_passes.get(name, 0) + (ratio <= 1.0)


def ratio_of(node: float, plant: float, tolerance: float) -> float:
    return abs(node - plant) / tolerance


def off_the_grid() -> Arm:
    arm = Arm()

    @settings(DRAWS)
    @given(dt=st.floats(1e-4, 2e-3), steps=st.integers(20, 200), density=st.floats(1100.0, 8000.0),
           volume=st.floats(1e-12, 1e-8), fall=st.floats(-1.0, 1.0), data=st.data())
    def draw(dt, steps, density, volume, fall, data):
        start = data.draw(st.integers(1, steps - 3))
        after = (start + data.draw(st.floats(0.05, 0.95))) * dt
        experiment, swim = falling_bead(dt, after, density, volume, fall * gain_per_step(dt, density) * steps)
        result = experiment.build().run(s(steps * dt), record=[swim.speed])
        node = result[swim].mean_speed.value
        window = result[swim].speed.between(s(after))
        tolerance = bound(window.raw)
        arm.oracle_saw(ratio_of(node, window.mean().value, tolerance))
        for shifted in (after - dt, after + dt):
            arm.control_saw("shifted a step", ratio_of(node, result[swim].speed.between(s(shifted)).mean().value,
                                                       tolerance))

    draw()
    return arm


def on_the_grid() -> Arm:
    arm = Arm()

    @settings(DRAWS)
    @given(dt=st.floats(1e-4, 2e-3), steps=st.integers(20, 200), density=st.floats(1100.0, 8000.0),
           volume=st.floats(1e-12, 1e-8), fall=st.floats(-1.0, 1.0), data=st.data())
    def draw(dt, steps, density, volume, fall, data):
        start = data.draw(st.integers(1, steps - 3))
        after = start * dt
        experiment, swim = falling_bead(dt, after, density, volume, fall * gain_per_step(dt, density) * steps)
        result = experiment.build().run(s(steps * dt), record=[swim.speed])
        node = result[swim].mean_speed.value
        window = result[swim].speed.between(s(after))
        tolerance = bound(window.raw)
        arm.oracle_saw(ratio_of(node, window.mean().value, tolerance))
        everything = result[swim].speed
        closed = np.asarray(everything.raw[everything.steps >= start], dtype=np.float64)
        if not node_takes_the_tied_step(dt, start):
            arm.control_saw("closed at its start", ratio_of(node, float(closed.mean()), tolerance))

    draw()
    return arm


def on_the_grid_ties() -> Arm:
    """The control test's own draws: a start the node ties on, drawn from the starts that tie."""
    arm = Arm()

    @settings(DRAWS)
    @given(dt=st.floats(1e-4, 2e-3), steps=st.integers(20, 200), density=st.floats(1100.0, 8000.0),
           volume=st.floats(1e-12, 1e-8), fall=st.floats(-1.0, 1.0), data=st.data())
    def draw(dt, steps, density, volume, fall, data):
        ties = [j for j in range(1, steps - 2) if node_takes_the_tied_step(dt, j)]
        assume(ties)
        start = data.draw(st.sampled_from(ties))
        after = start * dt
        experiment, swim = falling_bead(dt, after, density, volume, fall * gain_per_step(dt, density) * steps)
        result = experiment.build().run(s(steps * dt), record=[swim.speed])
        node = result[swim].mean_speed.value
        tolerance = bound(result[swim].speed.between(s(after)).raw)
        arm.oracle_saw(ratio_of(node, result[swim].speed.between(s(after)).mean().value, tolerance))
        with mock.patch.object(series_module, "ended_after", float64_rule):
            tied = result[swim].speed.between(s(after))
            arm.control_saw("float64 rule, where the node ties", ratio_of(node, tied.mean().value, tolerance))

    draw()
    return arm


def not_linear() -> Arm:
    arm = Arm()

    @settings(DRAWS)
    @given(dt=st.floats(1e-4, 2e-3), steps=st.integers(20, 200), density=st.floats(1100.0, 8000.0),
           volume=st.floats(1e-12, 1e-8), fall=st.floats(-1.0, 0.3), time_constant=st.floats(0.1, 0.5),
           data=st.data())
    def draw(dt, steps, density, volume, fall, time_constant, data):
        start = data.draw(st.integers(1, steps // 2))
        into = data.draw(st.one_of(st.just(0.0), st.floats(0.05, 0.95)))
        after = (start + into) * dt
        tau = time_constant * steps * dt
        terminal = (1.0 - WATER / density) * GRAVITY * tau
        experiment, swim = falling_bead(dt, after, density, volume, fall * terminal,
                                        viscosity_for(density, volume, tau))
        result = experiment.build().run(s(steps * dt), record=[swim.speed])
        node, distance = result[swim].mean_speed.value, result[swim].distance.value
        window = result[swim].speed.between(s(after))
        raw = np.asarray(window.raw, dtype=np.float64)
        tolerance = bound(window.raw)
        arm.oracle_saw(ratio_of(node, window.mean().value, tolerance))
        arm.control_saw("mean as the ends' midpoint", ratio_of(node, (raw[0] + raw[-1]) / 2, tolerance))
        arm.control_saw("mean as the middle sample", ratio_of(node, raw[len(raw) // 2], tolerance))
        arm.control_saw("window open at its end", ratio_of(node, raw[:-1].mean(), tolerance))
        span = tolerance * len(raw) * dt
        arm.control_saw("integral by the trapezoid rule",
                        ratio_of(distance, (raw.sum() - (raw[0] + raw[-1]) / 2) * dt, span))

    draw()
    return arm


def every() -> Arm:
    arm = Arm()
    dt, steps, speed = 1e-3, 60, 0.25
    for spacing in (3, 5):
        experiment, swim = falling_bead(dt, 0.0, WATER, 1e-9, speed)
        result = experiment.build().run(s(steps * dt), record=[swim.speed], every=ms(spacing * dt * 1e3))
        series = result[swim].speed
        tolerance = 8.0 * (steps + 1) * float(np.finfo(series.raw.dtype).eps) * speed * steps * dt
        arm.oracle_saw(abs(series.integral().value - speed * steps * dt) / tolerance)
        arm.control_saw(f"weight of a base step, every {spacing}",
                        abs(float(np.asarray(series.raw, dtype=np.float64).sum()) * dt - speed * steps * dt)
                        / tolerance)
    return arm


ARMS: dict[str, Callable[[], Arm]] = {"off the grid": off_the_grid, "on the grid": on_the_grid,
                                      "on the grid, ties": on_the_grid_ties, "not linear": not_linear,
                                      "every": every}


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for name in list(ARMS):
        if len(sys.argv) > 2 and name not in sys.argv[2:]:
            del ARMS[name]
    held: dict[str, int] = dict.fromkeys(ARMS, 0)
    failed: dict[tuple[str, str], int] = {}
    closest: dict[tuple[str, str], float] = {}
    worst: dict[str, float] = dict.fromkeys(ARMS, 0.0)
    for run in range(1, runs + 1):
        for name, measure in ARMS.items():
            arm = measure()
            held[name] += arm.oracle_failures == 0
            worst[name] = max(worst[name], arm.oracle)
            for control, ratio in arm.controls.items():
                key = (name, control)
                failed[key] = failed.get(key, 0) + (arm.control_passes[control] == 0)
                closest[key] = min(closest.get(key, float("inf")), ratio)
            print(f"run {run:2d}, {name}: {arm.draws} draws; oracle failures {arm.oracle_failures}, worst "
                  f"{arm.oracle:.3g} of its bound; " + "; ".join(
                      f"{control}: {arm.control_passes[control]} passes, closest {ratio:.3g}"
                      for control, ratio in arm.controls.items()), flush=True)
    print()
    for name in ARMS:
        print(f"{name}: the oracle held at every draw in {held[name]} of {runs} runs (worst {worst[name]:.3g} of its "
              "bound)")
        for (arm_name, control), count in failed.items():
            if arm_name == name:
                print(f"    {control}: failed at every draw in {count} of {runs} runs, closest "
                      f"{closest[(arm_name, control)]:.3g} times the bound")


if __name__ == "__main__":
    main()
