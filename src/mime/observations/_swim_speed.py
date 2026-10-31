from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.core import switch
from mime.gradients import GradientStatus
from mime.nodes.body import SwimSpeedNode
from mime.registry import (
    BuildContext,
    Direction,
    DirectionLike,
    GradientRecord,
    Observation,
    Output,
    OutputKind,
    Read,
    Reduction,
    RoleKind,
    register_observation,
)
from mime.units import LENGTH, TIME, VELOCITY, Magnitude, Quantity, Time, s
from mime.vocabulary import TWIST

if TYPE_CHECKING:
    from mime.build import ObservationBuilder


@register_observation
@dataclass(frozen=True, kw_only=True)
class SwimSpeed(Observation):
    """The body's velocity along ``along`` (``tube.axis``), and its mean over every step that ends after ``after``:
    ``SwimSpeed(umr, along=tube.axis)``. It reads the body's twist, adds no edge into the physics, and keeps what it
    measures in its own node's state, so a run need keep no history to report it: ``mean_speed``, ``distance`` and
    ``observed`` at the end of a run, and ``speed`` each step, a series when the run records it."""

    name = "swim_speed"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="swim.window",
            indicator="the time at the end of the step less the setting after, s: a step is counted whole once its "
                      "end lies after it",
            orders={"distance": -1, "observed": -1, "mean_speed": -1},
            route="count the step in which the setting after falls by the part of it that lies after it, with the "
                  "speed interpolated linearly between the speeds read at successive steps (the trapezoid rule): the "
                  "distance and the time observed are then continuously differentiable in after, and so is the mean "
                  "wherever some time has been observed",
            settings=("after",)),
        switch.Declaration(
            owner="swim.mean",
            indicator="the time observed less half a timestep, s: the mean is taken once some time is observed",
            orders={"mean_speed": -1},
            route="none: where observing starts the mean goes from none to the speed, which has no derivative to "
                  "compute",
            settings=("after",)),
    ))
    references = ()
    builds = {"": (SwimSpeedNode,)}
    roles = {"body": RoleKind.BODY}
    reads = {"twist": Read("body", TWIST.name)}
    outputs = (
        Output("speed", VELOCITY, "the body's velocity along the direction, each step"),
        Output("mean_speed", VELOCITY, "the mean of the speed over the steps that end after the setting after",
               kind=OutputKind.AT_END, reduces="speed", rule=Reduction.MEAN, after="after"),
        Output("distance", LENGTH, "the distance travelled along the direction over the steps that end after the "
                                   "setting after: the speed's integral over them",
               kind=OutputKind.AT_END, reduces="speed", rule=Reduction.INTEGRAL, after="after"),
        Output("observed", TIME, "the time observed: the length of the steps that end after the setting after",
               kind=OutputKind.AT_END, after="after"),
    )
    body: Any = field(default=None, kw_only=False)
    along: Annotated[DirectionLike, Direction]
    after: Annotated[Quantity, Time] = s(0.0)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        after = settings["after"]
        return [] if isinstance(after, float) and after >= 0.0 else [f"after is {after!r} s, not a time from 0 on"]

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        node = b.add_node(SwimSpeedNode(ctx.node_name(), ctx.timestep, along=ctx.vector("along"),
                                        after=ctx.scalar("after")))
        b.connect(b.read("twist"), into=node["twist"])
