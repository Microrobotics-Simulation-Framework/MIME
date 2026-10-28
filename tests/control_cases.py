"""Test-only controllers and the small experiments the controller's tests run them in. The controllers are registered
under the ``oracle`` namespace, in this process only, so the registry walk (a fresh process) holds them to no
round-trip case.

``spun_magnet`` is the cheapest closed loop with a command a controller can write and read back: a motor alone in the
world, its rotor carrying a magnet above a body, the motor's angle prescribed. The kinematic motor's state ``angle``
is the command it was given that step, so the history of the angle is the history of the writes in force: a write
``k`` (``Ticker`` writes ``k`` at its tick ``k``) shows exactly on the steps it acts on."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from chain_cases import MAGNET_MOMENT, add_body, experiment, magnet_part

from mime.dynamics import Motor
from mime.experiment import Experiment
from mime.fields import PointDipoleField
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.observations import SwimSpeed
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller
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


class Unregistered(Ticker):
    """A controller no one registered."""


def spun_magnet(name: str = "spun_magnet", *, controller: Controller | None = None, reads=None, writes=None,
                mode: Any = "callback", rate=ms(3), latency=None, observe: bool = False,
                drive_angle: bool = False) -> Experiment:
    """A motor alone in the world, 10 cm above a body, its rotor carrying a magnet whose field turns the body; its
    angle prescribed, written by ``controller`` (a :class:`Ticker` by default) every ``rate``. ``observe`` adds the
    body's swim speed along x; ``drive_angle`` gives the angle a drive as well."""
    exp = experiment(name)
    exp.add_apparatus(Apparatus(
        "spin", chain={"motor": Motor.Kinematic(axis=Z, prescribed="angle", initial_angle=rad(0.0),
                                                at=m((0.0, 0.0, 0.1)))},
        parts={"magnet": magnet_part("motor.rotor")}, magnets={"magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        fields={"magnetic": PointDipoleField(part="magnet")},
        commands={"motor.angle": rad(0.25)} if drive_angle else {}))
    exp.declare("spin.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet")
    add_body(exp)
    exp.attach(Magnetic.Torque(), body="umr", field="spin.magnetic")
    if observe:
        exp.observe(SwimSpeed("umr", along=X))
    exp.set_controller(controller if controller is not None else Ticker(),
                       reads={"pose": "umr.pose"} if reads is None else reads,
                       writes={"angle": "spin.motor.angle"} if writes is None else writes,
                       mode=mode, rate=rate, latency=latency)
    return exp
