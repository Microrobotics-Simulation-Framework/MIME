from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.gradients import GradientStatus
from mime.nodes.body import SwimSpeedNode
from mime.registry import (
    NOT_ESTABLISHED,
    BuildContext,
    Direction,
    DirectionLike,
    GradientRecord,
    Observation,
    Read,
    RoleKind,
    register_observation,
)
from mime.units import Magnitude, Quantity, Time, s
from mime.vocabulary import TWIST

if TYPE_CHECKING:
    from mime.build import ObservationBuilder


@register_observation
@dataclass(frozen=True, kw_only=True)
class SwimSpeed(Observation):
    """The body's velocity along ``along`` (``tube.axis``), and its mean over every step that ends after ``after``:
    ``SwimSpeed(umr, along=tube.axis)``. It reads the body's twist, adds no edge into the physics, and keeps what it
    measures in its own node's state (``speed``, ``mean_speed``), so a run need keep no history to report it."""

    name = "swim_speed"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE
    gradient = GradientRecord(
        where="at the switches of its node's step: a step is counted once its end lies after the setting after, and "
              "the mean is taken once the time observed passes half a timestep",
        reason="the steps observed are counted whole",
        route=NOT_ESTABLISHED)
    builds = {"": (SwimSpeedNode,)}
    roles = {"body": RoleKind.BODY}
    reads = {"twist": Read("body", TWIST.name)}
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
