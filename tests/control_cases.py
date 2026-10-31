"""Test-only controllers and the small experiments the controller's tests run them in. The controllers are registered
under the ``oracle`` namespace, in this process only, so the registry walk (a fresh process) holds them to no
round-trip case.

``spun_magnet`` is the cheapest closed loop with a command a controller can write and read back: a motor alone in the
world, its rotor carrying a magnet above a body, the motor's angle prescribed. The kinematic motor's state ``angle``
is the command it was given that step, so the history of the angle is the history of the writes in force: a write
``k`` (``Ticker`` writes ``k`` at its tick ``k``) shows exactly on the steps it acts on."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from chain_cases import MAGNET_MOMENT, add_body, experiment, magnet_part

from mime.dynamics import Motor
from mime.experiment import Experiment
from mime.fields import PointDipole
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.interactions.magnetic import MagneticTorque
from mime.observations import SwimSpeed
from mime.registry import (
    ControlLaw,
    Controller,
    GradientRecord,
    GroupRequest,
    Plan,
    PlanContext,
    Signal,
    register_controller,
    register_interaction,
)
from mime.scene import X, Z
from mime.things import Apparatus, PermanentMagnet
from mime.units import A_m2, m, ms, rad
from mime.vocabulary import register_namespace

register_namespace("oracle", "0.0.1")

_ABSENT = GradientRecord(where="its step", reason="a test's callback: no derivative passes through mode 2",
                         route="none: it exists to be called")


@register_controller
@dataclass(frozen=True, kw_only=True)
class Ticker(Controller):
    """Writes its tick count as the motor's angle, ``k`` at its tick ``k`` (the first is 0), holding ``held`` before
    its first write acts; reads the body's pose, and counts in its state how often it was called. Every tick also
    records what it read in ``seen``, a list the test reads."""

    name = "oracle.ticker"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"angle": Signal("1", ())}
    held: float = -1.0
    seen: list[Any] | None = None

    def law(self, ctx: Any) -> ControlLaw:
        seen = self.seen

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            if seen is not None:
                seen.append({label: np.asarray(value) for label, value in reads.items()})
            ticks = int(state["ticks"])
            return {"ticks": ticks + 1}, {"angle": np.asarray(float(ticks))}

        return ControlLaw(state={"ticks": 0}, held={"angle": np.asarray(self.held)}, step=step)


@register_controller
@dataclass(frozen=True, kw_only=True)
class Counter(Controller):
    """:class:`Ticker` in ``jax.numpy``, so that its step traces: writes its tick count as the motor's angle, ``k`` at
    its tick ``k``, holding ``held`` before its first write acts, and counts in its state. It reads the body's pose
    and does nothing with it. The same writes as ``Ticker``'s at every tick, so the two modes can be compared bit for
    bit."""

    name = "oracle.counter"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"angle": Signal("1", ())}
    held: float = -1.0

    def law(self, ctx: Any) -> ControlLaw:
        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            ticks = state["ticks"]
            return {"ticks": ticks + 1}, {"angle": jnp.asarray(ticks, dtype=jnp.zeros(()).dtype)}

        return ControlLaw(state={"ticks": jnp.asarray(0, dtype=jnp.int32)}, held={"angle": np.asarray(self.held)},
                          step=step)


@register_controller
@dataclass(frozen=True, kw_only=True)
class CalledBack(Counter):
    """:class:`Counter`, its step calling the host (``jax.debug.callback``) with what it reads."""

    name = "oracle.called_back"

    def law(self, ctx: Any) -> ControlLaw:
        counted = super().law(ctx)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[Any, Any]:
            jax.debug.callback(lambda pose: None, reads["pose"])
            return counted.step(state, reads)

        return ControlLaw(state=counted.state, held=counted.held, step=step)


#: The mutable Python state :class:`Impure` reads: a counter its step increments, which the compiled step would freeze
#: at the number of times it was traced.
IMPURE_CALLS = [0]


@register_controller
@dataclass(frozen=True, kw_only=True)
class Impure(Counter):
    """:class:`Counter`, writing instead the number of times its step has been called, from a global its step increments:
    mode 2 writes 1, 2, 3, ..., and mode 1 would write the number of times the step was traced, at every tick."""

    name = "oracle.impure"

    def law(self, ctx: Any) -> ControlLaw:
        counted = super().law(ctx)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[Any, Any]:
            IMPURE_CALLS[0] += 1
            return state, {"angle": jnp.asarray(float(IMPURE_CALLS[0]), dtype=jnp.zeros(()).dtype)}

        return ControlLaw(state=counted.state, held=counted.held, step=step)


@register_controller
@dataclass(frozen=True, kw_only=True)
class Growing(Counter):
    """:class:`Counter`, its state keeping every pose it read, so that its state grows by one row a tick: it traces,
    and its state has another shape after each tick."""

    name = "oracle.growing"

    def law(self, ctx: Any) -> ControlLaw:
        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            return ({"poses": jnp.concatenate([state["poses"], reads["pose"][None]])},
                    {"angle": jnp.asarray(0.0, dtype=jnp.zeros(()).dtype)})

        return ControlLaw(state={"poses": jnp.zeros((0, 7))}, held={"angle": np.asarray(self.held)}, step=step)


@register_controller
@dataclass(frozen=True, kw_only=True)
class Follower(Controller):
    """Writes as the motor's angle a filtered copy of the body's turn about x, ``gain`` times the pose's quaternion x:
    smooth, in ``jax.numpy`` and in the run's precision, so that a derivative of a run passes through it."""

    name = "oracle.follower"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    gradient = GradientRecord()
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"angle": Signal("1", ())}
    gain: float = 50.0
    smoothing: float = 0.5

    def law(self, ctx: Any) -> ControlLaw:
        gain, smoothing = self.gain, self.smoothing

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            filtered = state["filtered"] + smoothing * (gain * reads["pose"][4] - state["filtered"])
            return {"filtered": filtered}, {"angle": filtered}

        zero = jnp.zeros(())
        return ControlLaw(state={"filtered": zero}, held={"angle": zero}, step=step)


@register_controller
@dataclass(frozen=True, kw_only=True)
class JointWatcher(Controller):
    """Reads an arm's joint positions and writes a motor's angle, the first joint's: in :func:`watched_arm` the arm is
    driven, so nothing the controller writes reaches what it reads."""

    name = "oracle.joint_watcher"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    reads = {"joints": Signal("1", (-1,))}
    writes = {"angle": Signal("1", ())}

    def law(self, ctx: Any) -> ControlLaw:
        return ControlLaw(state={}, held={"angle": np.asarray(0.0)},
                          step=lambda state, reads: (state, {"angle": reads["joints"][0]}))


@register_controller
@dataclass(frozen=True, kw_only=True)
class RateWatcher(Controller):
    """Reads a chain segment's frames' angular velocities and writes a motor's angle, fixed: a reader of the rates,
    which the segment computes and publishes only where one reads them."""

    name = "oracle.rate_watcher"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    reads = {"rates": Signal("rad/s", (-1, 3))}
    writes = {"angle": Signal("1", ())}

    def law(self, ctx: Any) -> ControlLaw:
        return ControlLaw(state={}, held={"angle": np.asarray(0.0)},
                          step=lambda state, reads: (state, {"angle": np.asarray(0.5)}))


@register_controller
@dataclass(frozen=True, kw_only=True)
class Watcher(Controller):
    """Reads the body's pose (the physics state) and the swim speed an observation measures, and writes a fixed
    angle: the controller whose reads are of both classes."""

    name = "oracle.watcher"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,)), "speed": Signal("m/s", ())}
    writes = {"angle": Signal("1", ())}

    def law(self, ctx: Any) -> ControlLaw:
        return ControlLaw(state={}, held={"angle": np.asarray(0.0)},
                          step=lambda state, reads: (state, {"angle": np.asarray(0.5)}))


@register_controller
@dataclass(frozen=True, kw_only=True)
class TwoWrites(Controller):
    """Writes two labels, so that both can be bound to one command."""

    name = "oracle.two_writes"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = _ABSENT
    builds = {"": ()}
    writes = {"first": Signal("1", ()), "second": Signal("1", ())}

    def law(self, ctx: Any) -> ControlLaw:
        return ControlLaw(state={}, held={"first": np.asarray(0.0), "second": np.asarray(0.0)},
                          step=lambda state, reads: (state, {"first": np.asarray(0.0), "second": np.asarray(0.0)}))


@register_controller
@dataclass(frozen=True, kw_only=True)
class Misspelt(Ticker):
    """Writes under a label it does not declare."""

    name = "oracle.misspelt"

    def law(self, ctx: Any) -> ControlLaw:
        return ControlLaw(state={}, held={"angle": np.asarray(0.0)},
                          step=lambda state, reads: (state, {"angel": np.asarray(1.0)}))


class Unregistered(Ticker):
    """A controller no one registered."""


@register_interaction
@dataclass(frozen=True, kw_only=True)
class GroupedTorque(MagneticTorque):
    """:class:`~mime.interactions.Magnetic.Torque`, asking that the body's dynamics and its own node be solved together
    in one coupling group: a small rig whose controller reads the output of a coupling group."""

    name = "oracle.grouped_torque"
    family = None

    def plan(self, ctx: PlanContext) -> Plan:
        return dataclasses.replace(super().plan(ctx), groups=(
            GroupRequest(("body",), "a test's group around the body", 1e-6, 20, own_nodes=True),))


def expected_angles(steps: int, latency: int, period: int, held: float = -1.0) -> list[float]:
    """The motor's angle at each step under the latency rule, with :class:`Ticker` or :class:`Counter` writing it:
    the write of tick ``k`` (``k``, taken at step ``k * period``) acts on steps ``k * period + latency`` up to the
    next tick's, and ``held`` before the first."""
    return [held if n < latency else float((n - latency) // period) for n in range(steps)]


def spun_magnet(name: str = "spun_magnet", *, controller: Controller | None = None, reads=None, writes=None,
                mode: Any = "callback", rate=ms(3), latency=None, observe: bool = False,
                drive_angle: bool = False, torque: Any = None) -> Experiment:
    """A motor alone in the world, 10 cm above a body, its rotor carrying a magnet whose field turns the body; its
    angle prescribed, written by ``controller`` (a :class:`Ticker` by default) every ``rate``. ``observe`` adds the
    body's swim speed along x; ``drive_angle`` gives the angle a drive as well."""
    exp = experiment(name)
    exp.add_apparatus(Apparatus(
        "spin", chain={"motor": Motor.Kinematic(axis=Z, prescribed="angle", initial_angle=rad(0.0),
                                                at=m((0.0, 0.0, 0.1)))},
        parts={"magnet": magnet_part("motor.rotor")}, magnets={"magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        fields={"magnetic": PointDipole(part="magnet")},
        commands={"motor.angle": rad(0.25)} if drive_angle else {}))
    exp.declare("spin.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet")
    add_body(exp)
    exp.attach(Magnetic.Torque() if torque is None else torque, body="umr", field="spin.magnetic")
    if observe:
        exp.observe(SwimSpeed("umr", along=X))
    exp.set_controller(controller if controller is not None else Ticker(),
                       reads={"pose": "umr.pose"} if reads is None else reads,
                       writes={"angle": "spin.motor.angle"} if writes is None else writes,
                       mode=mode, rate=rate, latency=latency)
    return exp


def watched_arm(name: str = "watched_arm", **options) -> Experiment:
    """A0's apparatus with its motor's angle written by :class:`JointWatcher`, which reads the arm's joint positions:
    the arm is driven, so the controller reads what its writes do not reach."""
    from chain_cases import rpm

    exp = experiment(name)
    exp.add_apparatus(rpm(prescribed="angle", controlled=("motor.angle",)))
    exp.declare("rpm.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet")
    add_body(exp)
    exp.attach(Magnetic.Torque(), body="umr", field="rpm.magnetic")
    exp.set_controller(JointWatcher(), reads={"joints": "rpm.arm.joint_positions"},
                       writes={"angle": "rpm.motor.angle"}, **options)
    return exp
