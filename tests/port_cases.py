"""Shared pieces of the ports' tests: the planted errors their controls switch on (``PORT_PLANT``, read by
``.github/probe/detection_rate.py --plant-var PORT_PLANT``, the draw's seed in ``DETECTION_SEED``), NumPy oracles of the
two stages' rungs written from the design's order, and the experiments they run: the coil-array sphere with a port on
its currents, its loop closed by a controller through a measurement port and a command port, and a kinematic two-joint arm
commanded by a square wave its controller writes through a port."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from chain_cases import C_ARM

from mime import Apparatus, Body, Experiment, State
from mime.data import asset as packaged
from mime.drives import Constant
from mime.dynamics import Arm, RigidInertial
from mime.fields import CalibratedMultipole
from mime.interactions import Magnetic
from mime.gradients import GradientStatus
from mime.registry import ControlLaw, Controller, Signal, register_controller
from mime.scene import Scene, Sphere, Z, asset
from mime.things import Part
from mime.units import A, A_m2, kg, kg_m2, m, m_per_s, m_per_s2, ms, rad, rad_per_s
from mime.vocabulary import register_namespace

register_namespace("oracle", "0.0.1")

#: The planted error a control switches on, by name; empty when nothing is planted.
PLANT = os.environ.get("PORT_PLANT", "")
#: The seed of a control's draw (``DETECTION_SEED``, which detection_rate.py sets for each run).
SEED = int(os.environ.get("DETECTION_SEED", "0") or 0)


def planted(name: str) -> bool:
    return PLANT == name


@dataclass(frozen=True)
class Link:
    """A command port's numbers, in base steps and SI: what the oracle and the stage are both given."""

    period: int = 1
    latency: int = 0
    low: float | None = None
    high: float | None = None
    trip: bool = False
    rest: float = 0.0
    slew: float | None = None
    tau: float | None = None


def sampled_and_delayed(written: np.ndarray, period: int, latency: int, rest: Any) -> np.ndarray:
    """The design's first two rungs: the writer sampled every ``period`` steps and held, then delayed by ``latency``,
    the rest before."""
    out = np.empty_like(written)
    for n in range(len(written)):
        source = n - latency
        out[n] = rest if source < 0 else written[period * (source // period)]
    return out


def command_oracle(written: np.ndarray, link: Link, dt: float, *, order: tuple[str, ...] = (
        "sample", "delay", "bound", "slew", "response")) -> np.ndarray:
    """What a command port delivers at each step of ``written`` (steps by elements), its rungs in ``order``: the
    design's by default (sample, delay, bound, slew, response: a first-order lag of ``tau`` from the rest, exact for a
    value held over the step)."""
    held = np.asarray(written, dtype=np.float64)
    rest = np.full(held.shape[1:], link.rest)

    def sample(x: np.ndarray) -> np.ndarray:
        return np.stack([x[link.period * (n // link.period)] for n in range(len(x))])

    def delay(x: np.ndarray) -> np.ndarray:
        return np.stack([rest if n < link.latency else x[n - link.latency] for n in range(len(x))])

    def bound(x: np.ndarray) -> np.ndarray:
        if link.low is None or link.high is None:
            return x
        if not link.trip:
            return np.clip(x, link.low, link.high)
        out, tripped = np.empty_like(x), False
        for n in range(len(x)):
            tripped = tripped or bool(np.any((x[n] < link.low) | (x[n] > link.high)))
            out[n] = rest if tripped else x[n]
        return out

    def slew(x: np.ndarray) -> np.ndarray:
        if link.slew is None:
            return x
        out, before = np.empty_like(x), rest
        for n in range(len(x)):
            before = before + np.clip(x[n] - before, -link.slew * dt, link.slew * dt)
            out[n] = before
        return out

    def response(x: np.ndarray) -> np.ndarray:
        if link.tau is None:
            return x
        out, before = np.empty_like(x), rest
        for n in range(len(x)):
            before = x[n] + (before - x[n]) * math.exp(-dt / link.tau)
            out[n] = before
        return out

    rungs = {"sample": sample, "delay": delay, "bound": bound, "slew": slew, "response": response}
    for rung in order:
        held = rungs[rung](held)
    return held


def command_flags(written: np.ndarray, link: Link) -> np.ndarray:
    """The bound's flag at each step of ``written``, in the design's order: whether the clip acted on what the link
    delivered (any element), or whether the trip has latched by then; zeros for a port with no bounds."""
    if link.low is None or link.high is None:
        return np.zeros(len(written))
    delivered = sampled_and_delayed(np.asarray(written, dtype=np.float64), link.period, link.latency,
                                    np.full(np.shape(written)[1:], link.rest))
    beyond = np.any((delivered < link.low) | (delivered > link.high), axis=tuple(range(1, delivered.ndim)))
    return (np.maximum.accumulate(beyond) if link.trip else beyond).astype(np.float64)


def stage_of(link: Link, shape: tuple[int, ...], dt: float) -> Any:
    """A command stage holding one port, ``currents``, of ``link``'s numbers."""
    from mime.nodes.control import CommandStageNode

    leaves: dict[str, Any] = {"rest_currents": link.rest}
    if link.low is not None:
        leaves.update(low_currents=link.low, high_currents=link.high)
    if link.slew is not None:
        leaves["slew_currents"] = link.slew
    if link.tau is not None:
        leaves["response_currents_tau"] = link.tau
    return CommandStageNode(
        "ports:coils", dt, commands=["currents"], units={"currents": "A"}, shapes={"currents": shape},
        periods={"currents": link.period}, latencies={"currents": link.latency},
        bounded={"currents": "" if link.low is None else ("trip" if link.trip else "clip")},
        slewed={"currents": link.slew is not None},
        responses={"currents": ("response.first_order_lag", "0.1.0") if link.tau is not None
                   else ("response.ideal", "0.1.0")},
        carries={"currents": ("lagged",) if link.tau is not None else ()}, **leaves)


def stepped(node: Any, written: np.ndarray, dt: float, field: str = "currents", *,
            flag: str | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """``node``'s delivered ``field`` at each step of ``written``, stepped through its own update; with ``flag``, the
    state field of that name at each step beside it."""
    state = node.initial_state()
    out, flags = [], []
    for value in written:
        state = node.fresh_update(state, {field: jnp.asarray(value)}, dt)
        out.append(np.asarray(state[field], dtype=np.float64))
        if flag is not None:
            flags.append(float(state[flag]))
    return (np.stack(out), np.asarray(flags)) if flag is not None else np.stack(out)


def square_wave(draw: np.random.Generator, steps: int, size: int, *, beyond: float,
                halves: tuple[int, int] = (3, 8), past: tuple[float, float] = (1.2, 2.0)) -> np.ndarray:
    """A drawn square wave of ``steps`` rows: each element jumps between two levels, one ``past`` times ``beyond``
    (outside the bounds) and one inside, on a drawn half period of ``halves`` steps (the upper excluded), so a bound and
    a slew both act and the command comes back."""
    out = np.empty((steps, size))
    for j in range(size):
        half = int(draw.integers(*halves))
        high, low = beyond * draw.uniform(*past), beyond * draw.uniform(-0.5, 0.5)
        out[:, j] = [high if (n // half) % 2 == 0 else low for n in range(steps)]
    return out


@register_controller
@dataclass(frozen=True, kw_only=True)
class Replayed(Controller):
    """Writes the rows of ``table`` one a tick, its first at the first tick, then holds its last: a command whose
    every value the test knows."""

    name = "oracle.replayed"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": ()}
    reads = {}
    writes = {"joint_positions": Signal("1", (-1,))}
    table: tuple[tuple[float, ...], ...]

    def law(self, ctx: Any) -> ControlLaw:
        rows = jnp.asarray(self.table)

        def step(state: Any, reads: Any) -> tuple[Any, Any]:
            count = state["count"]
            row = rows[jnp.minimum(count, rows.shape[0] - 1)]
            return {"count": count + 1}, {"joint_positions": row}

        return ControlLaw(state={"count": jnp.asarray(0, dtype=jnp.int32)},
                          held={"joint_positions": rows[0]}, step=step)


@register_controller
@dataclass(frozen=True, kw_only=True)
class PoseFeedback(Controller):
    """Writes the coils' currents in proportion to the body's position (``gain`` amperes a metre, the same for each
    coil and axis), and its age, ``age`` in base steps, into nothing: a controller cheap to trace whose writes depend on
    what it reads, for the ports' loops."""

    name = "oracle.pose_feedback"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,)), "age": Signal("1", ())}
    writes = {"currents": Signal("A", (3,))}
    gain: float = -2.0e4

    def law(self, ctx: Any) -> ControlLaw:
        gain = self.gain

        def step(state: Any, reads: Any) -> tuple[Any, Any]:
            return {"ages": state["ages"] + reads["age"]}, {"currents": gain * reads["pose"][:3]}

        return ControlLaw(state={"ages": jnp.asarray(0.0)}, held={"currents": jnp.zeros(3)}, step=step)


def fed_back_sphere(ports: dict[str, Any], *, mode: str, name: str = "fed_back") -> Experiment:
    """The coil sphere, its currents written by :class:`PoseFeedback` through its ports, reading its pose and age
    through the measurement port ``tracker``."""
    exp = coil_sphere(ports, name=name, commands={})
    exp.set_controller(PoseFeedback(), reads={"pose": "coils.tracker", "age": "coils.tracker.age"},
                       writes={"currents": "coils.currents"}, mode=mode, rate=ms(1))
    return exp


#: The two-joint arm's revolute joints' limits (tests/descriptions/c_arm.urdf), rad, and its start (its slide, then its
#: two revolute joints).
C_ARM_LIMITS = ((-2.0, 2.0), (-3.0, 3.0))
C_ARM_START = (0.1, 0.0, 0.0)


def replayed_arm(table: np.ndarray, port: Any, *, name: str = "ported_arm", mode: str = "graph",
                 slide: Any = None) -> Experiment:
    """The two-joint arm, kinematic, its revolute joints written by :class:`Replayed` with ``table`` through ``port``
    (``None``: no port), its slide held at 0.1 m through ``slide`` (its port, where the revolute one lags: the arm's node
    lags every joint with one time constant)."""
    ports = {"arm.joint_positions": port} if port is not None else {}
    if slide is not None:
        ports["arm.joint_displacements"] = slide
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(1))
    exp.add_apparatus(Apparatus(
        "carm", chain={"arm": Arm.Kinematic(description=asset(C_ARM), initial_positions=rad(C_ARM_START))},
        commands={"arm.joint_displacements": Constant(m((0.1,)))}, ports=ports))
    exp.set_controller(Replayed(table=tuple(tuple(float(c) for c in row) for row in table)), reads={},
                       writes={"joint_positions": "carm.arm.joint_positions"}, mode=mode, rate=ms(1))
    return exp


def coil_sphere(ports: dict[str, Any] | None = None, *, name: str = "coil_sphere",
                commands: dict[str, Any] | None = None) -> Experiment:
    """A 1 mm sphere with a fixed moment at Navion's centre, in its field at held currents (``commands``, by default
    a constant drive), the magnetic force attached, no gravity and no medium; the apparatus's ``ports`` as given."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    exp.add_apparatus(Apparatus("coils", fields={"magnetic": CalibratedMultipole(calibration=packaged("navion_2"))},
                                commands={"currents": A((4.0, -2.5, 1.5))} if commands is None else commands,
                                ports=ports or {}))
    exp.add_body(Body("sphere", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(1e-3)))},
                      initial=State(position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("sphere.mass", kg(4e-6)), ("sphere.inertia", kg_m2((1.6e-12,) * 3)),
                        ("sphere.magnetic_moment", A_m2((2.8e-4, 0.0, 0.0))), ("sphere.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre sphere with a magnet at its centre")
    exp.attach(Magnetic.Force(), body="sphere", field="coils.magnetic")
    return exp


def staged_sphere(*, trip: bool = False, name: str = "staged_sphere") -> Experiment:
    """The coil sphere with every rung of both stages: its currents sampled every 2 steps, 3 steps late, bounded
    (clipped, or tripped with ``trip``) and slewed; its swim speed along x measured every 2 steps, 1 step late, with
    noise and a quantum. What the switch scan traces the stages in."""
    from mime.observations import SwimSpeed
    from mime.scene import X
    from mime.things import CommandPort, Gaussian, Measurement
    from mime.units import m_per_s, s

    ports = {"currents": CommandPort(rate=ms(2), latency=ms(3), bounds=A(3.0), on_bound="trip" if trip else "clip",
                                     slew=A(2000.0) / s(1.0), reason="every rung of the command stage, for the switch scan"),
             "speed": Measurement(of="swim_speed:sphere.speed", rate=ms(2), latency=ms(1),
                                  noise=Gaussian(sigma=m_per_s(1e-6)), quantum=m_per_s(1e-7),
                                  reason="every rung of the measure stage, for the switch scan")}
    exp = coil_sphere(ports, name=name)
    exp.observe(SwimSpeed("sphere", along=X))
    return exp
