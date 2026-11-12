"""Forward-mode derivatives of the in-loop camera's ``expected`` against central differences of the same run, through the
built graph (``Study.gradient(mode="forward")``, the camera's path acknowledged as the unverified entry it is declared).

A ball settles in the tube's water under gravity (a bead 1.1 times water's density), seen by a camera that sees the whole
scene, a frame every base step, for 5 ms. Three objectives, each a scalar of the last frame's ``expected``:

* **silhouette, the ball's pose** (``ball.parts.steel.density``): the bead's settling (the buoyancy-corrected acceleration
  over the frame's time) moves its silhouette by a fraction of a pixel per kg/m^3, so the box-filtered edge measure (the
  disc's exact coverage of each pixel it crosses) carries the derivative through the dynamics, the ring's pose and the frame;
* **silhouette, a leaf** (``camera:.detector.cx``): the principal point moves the same silhouette across the pixels;
* **interior** (``camera:.detector.fy``): the tube wall's rows 9 to 12, between its two edges (rows 7 and 14.5 at the focal
  length of 750, which a step of 20 moves by 0.1 px): the pixels there change by the chord's absorption, with no edge in them.

The silhouette objectives are the first moments of the window about its centre (a total of the darkness is the same wherever
the ball sits in the window, so its derivative is the tube's absorption under the ball, not the edge's).

**The step and the tolerance** come from two mechanisms. *Interior* (no edge in the window): the central difference errs by
``C h^2`` relative, ``C`` read from float64, where rounding is nothing (measured in this file's development, relative
errors 2.5e-5, 1.0e-4, 4.0e-4 at h = 0.5, 1, 2 for fy, a fourfold rise per doubling); the tolerance is ten times that.
*Silhouette*: the box edge is piecewise, so the central difference over +-h errs where the interval holds a pixel corner or
a tangent to a pixel's side (the coverage's derivative jumps there, and rises as a square root at a tangent): the error is
linear in h up to a saturation, not quadratic (float64, the moment objective: cx 7.3e-5, 7.0e-4, 1.9e-3, 3.8e-3, 3.8e-3 at
h = 0.05, 0.1, 0.2, 0.5, 1 px; the settling 3.5e-6, 2.6e-3, 9.5e-3, 1.8e-2 at h = 0.25, 0.5, 1, 2 kg/m^3), at most about
``KINK`` = 1e-2 of the derivative per unit of h; the tolerance is that, and the steps are the smallest the dtype's
rounding allows (the corner is why a larger step is no better). *Rounding*: the objective sums about 2400 pixels in the run's dtype, so
the difference of two objectives is known to ``SPACING`` (the square root of the pixels) units in the last place of their
size, and the central difference to that over ``2 h``: added to either tolerance (measured float32 excess over float64:
0.8 units for the settling, 15 for cx, 1 for fy). All relative to the derivative; the mechanism of the jumps is
not isolated here (the pixel corner and the tangent are what the box edge has, not a measured cause).

**The control** (``CAMERA_PLANT=edge-stopped``) puts ``stop_gradient`` on the edge measure (the line's box coverage and the
disc's): the silhouette derivatives are then those of the absorption under the ball and the shift of the window, a small
share of the edge's, and fail by orders of magnitude on every run; the interior case, which has no edge in it, is unchanged."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, planted

import mime.sensing._coverage as coverage
from mime import Body, Experiment, Medium, State, Structure
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.observations import Camera, World
from mime.scene import Scene, Sphere, Tube, X, Z
from mime.study import Study, grid
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

if planted("edge-stopped"):
    covered, disc = coverage.covered, coverage.disc
    coverage.covered = lambda *a, **k: jax.lax.stop_gradient(covered(*a, **k))  # type: ignore[assignment]
    coverage.disc = lambda *a, **k: jax.lax.stop_gradient(disc(*a, **k))  # type: ignore[assignment]

#: The ball's window (rows 40 to 79, columns 15 to 74) and the wall's (rows 9 to 12, every column) in the 160 x 120 frame.
BALL = (slice(40, 80), slice(15, 75))
WALL = (slice(9, 13), slice(None))
#: The interior case's relative truncation coefficient measured in float64 (error / (h / unit)^2), times ten.
TRUNCATION = 1.0e-4 * 10
#: The silhouette cases' relative error per unit of step: the piecewise box edge's allowance (float64 measured at most
#: 9.5e-3 per unit).
KINK = 1.0e-2
#: How many units in the last place of the objective's size the dtype's difference of two objectives may be off by: the
#: objective sums about 2400 pixels, whose rounding adds as a random walk, sqrt(2400) = 49.
SPACING = 50.0
#: path, centre, unit of the parameter, window, whether the objective is the window's first moment
CASES = {"density": ("ball.parts.steel.density", 1100.0, 1.0, BALL, True),
         "cx": ("camera:.detector.cx", 79.5, 1.0, BALL, True),
         "fy": ("camera:.detector.fy", 750.0, 1.0, WALL, False)}
#: The step in the parameter's unit.
STEPS = {"density": 0.25, "cx": 0.05, "fy": 1.0}


def scene(density: float = 1100.0, **leaves: float):
    """The ball (a sphere of 20 um at 50 um upstream of the tube's centre, 2 mm/s along it) in the tube's water, settling
    under gravity, seen by the camera; ``density`` the ball's in kg/m^3 and ``leaves`` the detector's replaced numbers."""
    exp = Experiment("settling", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10), length=um(480))))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((-50.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((2e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(20)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(density), reason="a bead 1.1 times water's density")
    exp.add_medium(Medium("water", fills="tube.lumen"))
    exp.declare("water.density", kg_per_m3(1000.0), reason="water")
    exp.attach(Gravity(), body="ball", medium="water")
    exp.observe(Camera(detector=dataclasses.replace(DETECTOR, **leaves),
                       mount=World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS), frame_period=ms(1),
                       absorbers=(("tube", 3000.0),)))
    return exp


def objective(window: tuple[slice, slice], moment: bool, expected: Any) -> Any:
    block = expected[window]
    if not moment:
        return jnp.sum(block)
    rows, columns = jnp.indices(block.shape)
    return jnp.sum(((rows - (block.shape[0] - 1) / 2) + (columns - (block.shape[1] - 1) / 2)) * block)


def settle(name: str, value: float) -> float:
    """The objective of the last frame of the case's scene with the parameter at ``value``."""
    path, _, _, window, moment = CASES[name]
    built = (scene(density=value) if path.endswith("density") else scene(**{path.rsplit(".", 1)[1]: value})).build()
    built.run(ms(5))
    return float(objective(window, moment, jnp.asarray(built.current_state()["camera:"]["expected"])))


def forward(name: str) -> float:
    """The forward-mode derivative of the case's objective in its parameter, at the case's centre."""
    path, centre, _, window, moment = CASES[name]
    exp = scene()
    point = kg_per_m3(centre) if path.endswith("density") else centre
    result = Study(exp).sweep(grid({path: [point]}), duration=ms(5)).gradient(
        lambda final: objective(window, moment, final["camera:"]["expected"]), mode="forward",
        acknowledged={"camera:": "a forward derivative checked here against central differences"})
    return float(list(result.members[0].gradient.values())[0])


@pytest.mark.parametrize("name", list(CASES))
def test_the_forward_derivative_of_expected_agrees_with_central_differences(dtype, name):
    _, centre, unit, _, _ = CASES[name]
    h = STEPS[name] * unit
    d = forward(name)
    up, down = settle(name, centre + h), settle(name, centre - h)
    central = (up - down) / (2 * h)
    size = max(abs(up), abs(down))
    truncation = TRUNCATION * (h / unit) ** 2 if name == "fy" else KINK * h / unit
    tolerance = truncation + SPACING * float(np.spacing(np.asarray(size, dtype=dtype))) / (2 * h * abs(d))
    error = abs(d - central) / abs(d)
    assert abs(d) > 0.0 and error <= tolerance, (name, dtype, d, central, error, tolerance)
