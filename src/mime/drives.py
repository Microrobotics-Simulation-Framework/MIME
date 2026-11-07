"""Drives: in-graph command generators, each a node whose numbers are ``params`` leaves with one exclusive edge into
the apparatus command it writes. A quantity written for a command is short for :class:`Constant`; :class:`Ramp` moves
it at a rate from one value to another; a controller (phase 7) replaces the drive of each command it writes. A drive's
id is ``drive:<apparatus>.<command>``. The value a drive writes for a step holds over the step."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, override

from mime.core import switch
from mime.gradients import GradientStatus
from mime.nodes.actuation import ConstantNode, RampNode
from mime.registry import DESIGN, convention
from mime.registry import BuildContext, Commanded, CommandedRate, Drive, GradientRecord, register_drive
from mime.units import Magnitude, Quantity, Time, s

if TYPE_CHECKING:
    from mime.build import DriveBuilder


@register_drive
@dataclass(frozen=True, kw_only=True)
class Constant(Drive):
    """The command's value, the same every step: ``Constant(Hz(10))``."""

    name = "drive.constant"
    version = "0.1.0"
    setting_roles = {"value": DESIGN}
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": (ConstantNode,)}
    value: Annotated[Quantity, Commanded] = field(kw_only=False)

    @override
    def build(self, ctx: BuildContext, b: DriveBuilder) -> None:
        value = ctx.settings["value"]
        node = b.add_node(ConstantNode(ctx.node_name(), ctx.timestep, value=value))
        b.command(node["value"])


@register_drive
@dataclass(frozen=True, kw_only=True)
class Ramp(Drive):
    """The command held at ``start`` for ``hold``, then moved at ``rate`` (a magnitude, the command's units per second)
    until it reaches ``to``, then held there: ``Ramp(Hz(100.0), to=Hz(5.0), rate=Hz(0.25) / s(1.0), hold=s(3.0))``
    ramps a field's frequency down from 100 Hz after 3 s, at a quarter of a hertz a second. The value written for each
    step is the ramp's at the step's middle, its mean over the step, so the integrating rotating field (version 0.2.0)
    turns by the ramp's exact integral; ``uniform_rotating`` 0.1.0, whose phase is the present frequency times the time,
    refuses it on its frequency. One number per command."""

    name = "drive.ramp"
    version = "0.1.0"
    setting_roles = {"start": DESIGN, "to": DESIGN, "rate": DESIGN,
                     "hold": convention(DESIGN, "the ramp starts with the run unless a hold before it is given")}
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="ramp.start",
            indicator="the time at the middle of the step less the hold, s: the ramp moves the value from where it "
                      "is positive",
            orders={"value": 0},
            route="take the ramp's exact mean over the step in which its start falls, the integral of the clamp "
                  "over the step divided by its length, which is continuously differentiable in the hold",
            settings=("hold",)),
        switch.Declaration(
            owner="ramp.end",
            indicator="the ramp's duration less the time since its start at the middle of the step, s: the value "
                      "holds where it ends once this is negative",
            orders={"value": 0},
            route="take the ramp's exact mean over the step in which its end falls, as for its start, which is "
                  "continuously differentiable in the hold, the rate and the values at both ends",
            settings=("hold", "start", "to", "rate")),
    ))
    references = ()
    builds = {"": (RampNode,)}
    start: Annotated[Quantity, Commanded] = field(kw_only=False)
    to: Annotated[Quantity, Commanded]
    rate: Annotated[Quantity, CommandedRate]
    hold: Annotated[Quantity, Time] = s(0.0)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        for name in ("start", "to", "rate"):
            if not isinstance(settings[name], float):
                found.append(f"{name} is {settings[name]!r}: a ramp takes one number per command, not a vector")
        rate, hold = settings["rate"], settings["hold"]
        if isinstance(rate, float) and not (math.isfinite(rate) and rate > 0.0):
            found.append(f"rate is {rate!r} in SI: a ramp moves at a rate above zero (its direction is from start "
                         f"to to)")
        if not (isinstance(hold, float) and math.isfinite(hold) and hold >= 0.0):
            found.append(f"hold is {hold!r} s, not a time from 0 on")
        return found

    @override
    def build(self, ctx: BuildContext, b: DriveBuilder) -> None:
        start, to, rate = ctx.scalar("start"), ctx.scalar("to"), ctx.scalar("rate")
        node = b.add_node(RampNode(ctx.node_name(), ctx.timestep, start=start, rate=math.copysign(rate, to - start),
                                   duration=abs(to - start) / rate, hold=ctx.scalar("hold")))
        b.command(node["value"])


def as_drive(value: object) -> Drive:
    """A command as its drive: a drive as written, a quantity as :class:`Constant`."""
    if isinstance(value, Drive):
        return value
    if isinstance(value, Quantity):
        return Constant(value)
    raise TypeError(f"a command takes a drive or a quantity (short for drives.Constant), not {value!r}")


__all__ = ["Constant", "Ramp", "as_drive"]
