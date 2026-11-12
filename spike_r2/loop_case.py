"""The closed loop of camera spike R2: field -> robot -> camera -> tracker -> controller, all inside the compiled step.

**The case**: the smallest existing experiment with a coil array and a body the camera can see, the multipole oracle's
``sphere_in_the_array("navion_2", 52, torque=False)`` (``tests/oracles/test_calibrated_multipole.py``): a 1 mm sphere
of 4e-6 kg with a fixed moment of 2.8e-4 A m^2 at oracle point 52, in Navion_2's field, the force alone (its torque
off, so that the moment holds its direction), no gravity, no medium, dt = 1 ms. Added: a declared camera
(:data:`DETECTOR`, 96 x 72, 0.1 m above the sphere looking down, a frame every :data:`PERIOD_STEPS` base steps) and a
GRAPH controller (:class:`CameraPID` reading the camera's ``expected``, or :class:`StatePID` reading the sphere's true
pose: the STATE arm), every :data:`PERIOD_STEPS` steps, latency one step, writing the coils' currents.

**The law** (the same in both arms but for what it measures): the measured point (the tracker's soft centroid
back-projected on the plane ``z = z0``, so that depth is never observed; or the sphere's true position, the STATE arm),
a PID on the 3-D error to a reference :func:`reference` in the plane, derivative on the measurement, a force
``m (kp e + ki int e - kd dp/dt)``, and the JAX bounded allocation (``mime.control._allocation``) of the force rows at
the measured point within ``limit`` (A).
``kp, ki, kd, limit`` are leaves of the controller's node (a study's axes would make them so).

SPIKE_VARIANT picks the allocation's variant (its controls); SPIKE_PLANT=edge stops the derivative of the disc
coverage's edge pixels (the camera's edge measure)."""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests" / "oracles"))

from mime.build._declarations import ControllerDeclaration, ThingDeclaration  # noqa: E402
from mime.control._allocation import bounded_lsq, force_rows  # noqa: E402
from mime.data import asset  # noqa: E402
from mime.fields._coil_array import calibration_of  # noqa: E402
from mime.gradients import GradientStatus  # noqa: E402
from mime.nodes.actuation import placed_unit_fields  # noqa: E402
from mime.observations import Camera, Detector, World  # noqa: E402
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller  # noqa: E402
from mime.sensing._tracker import tracked  # noqa: E402
from mime.units import m, ms  # noqa: E402
from mime.vocabulary import register_namespace  # noqa: E402

jnp: Any = jax.numpy
VARIANT = os.environ.get("SPIKE_VARIANT", "double_where")
PLANT = os.environ.get("SPIKE_PLANT", "")

if PLANT == "edge":
    from mime.sensing import _coverage

    _disc = _coverage.disc

    def _edge_stopped(corners_x: Any, corners_y: Any, centre: Any, radius: Any) -> Any:
        share = _disc(corners_x, corners_y, centre, radius)
        edge = (share > 0.0) & (share < 1.0)
        return jnp.where(edge, jax.lax.stop_gradient(share), share)

    _coverage.disc = _edge_stopped

from test_calibrated_multipole import oracle, sphere_in_the_array  # noqa: E402

register_namespace("spike", "0.0.1")

INDEX = 52
PERIOD_STEPS = 5
DT = 1e-3
WIDTH, HEIGHT, FOCAL = 96, 72, 400.0
HEIGHT_ABOVE = 0.1
START = np.asarray(oracle("navion_2", "points")[INDEX], dtype=np.float64)
MOMENT = np.asarray(oracle("navion_2", "moments")[INDEX], dtype=np.float64)
MASS = 4e-6
#: The camera's optical frame (x right, y down, z out of the lens) looking down -z: a half turn about x.
LOOK_DOWN = (0.0, 1.0, 0.0, 0.0)
CAMERA_AT = (float(START[0]), float(START[1]), float(START[2] + HEIGHT_ABOVE))
DETECTOR = Detector(width=WIDTH, height=HEIGHT, fx=FOCAL, fy=FOCAL, cx=(WIDTH - 1) / 2, cy=(HEIGHT - 1) / 2)
FULL = 1000.0
#: The gains' defaults: a double integrator closed at about 8 Hz (kp = w^2, kd = 1.6 w), a slow integral.
GAINS = {"kp": 2500.0, "ki": 3000.0, "kd": 80.0, "limit": 25.0}
#: The reference: a step of STEP (m) at t = 0, then a circle of radius RADIUS at OMEGA (rad/s) about the stepped point.
STEP = (3e-3, -1.5e-3)
RADIUS, OMEGA = 1.5e-3, 2 * math.pi * 2.0

_cal = calibration_of(asset("navion_2"))
POSITIONS = np.asarray(_cal.positions)
VECTORS = np.asarray(_cal.coefficients)[..., None] * np.asarray(_cal.directions)


def reference(t: Any) -> Any:
    """The reference ``x, y, z`` at time ``t`` (s): the start, stepped by ``STEP`` (read when the law is made: a
    draw sets it before the build) at t = 0, then a circle about it in the plane ``z = z0``."""
    step = STEP
    return jnp.stack([START[0] + step[0] + RADIUS * (jnp.cos(OMEGA * t) - 1.0),
                      START[1] + step[1] + RADIUS * jnp.sin(OMEGA * t), jnp.full(jnp.shape(t), START[2])])


def _law(ctx: Any, measure: Any) -> ControlLaw:
    kp, ki, kd, limit = (ctx.scalar(name) for name in ("kp", "ki", "kd", "limit"))
    period = ctx.timestep
    dtype = jnp.zeros(()).dtype
    positions, vectors = jnp.asarray(POSITIONS, dtype=dtype), jnp.asarray(VECTORS, dtype=dtype)
    moment = jnp.asarray(MOMENT, dtype=dtype)

    def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        t = state["t"]
        xy = measure(reads)
        point = xy
        error = reference(t) - point
        integral = state["integral"] + error * period
        rate = (point - state["previous"]) / period
        wanted = MASS * (kp * error + ki * integral - kd * rate)
        _, gradients = placed_unit_fields(point, jnp.zeros(3, dtype=dtype), jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype),
                                          positions, vectors)
        rows = force_rows(gradients, moment)
        currents = bounded_lsq(rows, wanted, jnp.full((3,), 1.0, dtype=dtype) * limit, variant=VARIANT).currents
        return ({"t": t + period, "integral": integral, "previous": xy}, {"currents": currents.astype(dtype)})

    start = {"t": jnp.zeros((), dtype=dtype), "integral": jnp.zeros((3,), dtype=dtype),
             "previous": jnp.asarray(START, dtype=dtype)}
    return ControlLaw(state=start, held={"currents": jnp.zeros((3,), dtype=dtype)}, step=step)


_RECORD = GradientRecord(where="its step: the tracker, the PID and the bounded allocation",
                         reason="camera spike R2", route="none: a spike")


@register_controller
@dataclass(frozen=True, kw_only=True)
class CameraPID(Controller):
    """The PID through the JAX allocation, measuring by the tracker on the camera's ``expected``."""

    name = "spike.camera_pid"
    version = "0.1.0"
    differentiability = GradientStatus.UNVERIFIED
    gradient = _RECORD
    builds = {"": ()}
    reads = {"image": Signal("1", (HEIGHT, WIDTH))}
    writes = {"currents": Signal("A", (3,))}
    kp: float = GAINS["kp"]
    ki: float = GAINS["ki"]
    kd: float = GAINS["kd"]
    limit: float = GAINS["limit"]

    def law(self, ctx: Any) -> ControlLaw:
        def measure(reads: Mapping[str, Any]) -> Any:
            found = tracked(reads["image"], full=FULL, fx=FOCAL, fy=FOCAL, cx=(WIDTH - 1) / 2, cy=(HEIGHT - 1) / 2,
                            position=CAMERA_AT, orientation=LOOK_DOWN, plane_point=(0.0, 0.0, float(START[2])),
                            plane_normal=(0.0, 0.0, 1.0))
            return found.point

        return _law(ctx, measure)


@register_controller
@dataclass(frozen=True, kw_only=True)
class StatePID(CameraPID):
    """The same law measuring the sphere's true ``x, y`` (ground truth, not deployable): the STATE arm."""

    name = "spike.state_pid"
    reads = {"pose": Signal("m; 1", (7,))}

    def law(self, ctx: Any) -> ControlLaw:
        return _law(ctx, lambda reads: reads["pose"][:3])


def experiment(arm: str = "camera", leaves: tuple[str, ...] = ("kd", "ki", "kp", "limit"), **gains: float) -> Any:
    """The case with the camera and the controller of ``arm`` (``"camera"`` or ``"state"``), ``leaves`` its node's
    leaves."""
    exp = sphere_in_the_array("navion_2", INDEX, torque=False)
    # The coils' constant drive is replaced by the controller's writes (a command has one writer).
    exp._declarations[:] = [ThingDeclaration(dataclasses.replace(d.thing, commands={}))
                            if isinstance(d, ThingDeclaration) and d.thing.name == "coils" else d
                            for d in exp._declarations]
    exp.observe(Camera(detector=DETECTOR, mount=World(position=m(CAMERA_AT), orientation=LOOK_DOWN),
                       frame_period=ms(PERIOD_STEPS)))
    controller = (CameraPID if arm == "camera" else StatePID)(**gains)
    reads = {"image": "camera:.expected"} if arm == "camera" else {"pose": "sphere.pose"}
    exp.set_controller(controller, reads=reads, writes={"currents": "coils.currents"}, mode="graph",
                       rate=ms(PERIOD_STEPS), latency=ms(1))
    exp._declarations[:] = [dataclasses.replace(d, leaves=tuple(sorted(leaves))) if isinstance(d, ControllerDeclaration)
                            else d for d in exp._declarations]
    return exp


def camera_case() -> Any:
    """The camera arm at the defaults (mem_estimate.py's and node_measure.py's factory)."""
    return experiment("camera")


def state_case() -> Any:
    """The STATE arm at the defaults."""
    return experiment("state")
