from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as setting
from typing import TYPE_CHECKING, Annotated, Any, cast, override

from mime.core import switch
from mime.gradients import GradientStatus
from mime.nodes.body import PhaseLagNode
from mime.registry import DESIGN, convention
from mime.registry import (
    BuildContext,
    Direction,
    DirectionLike,
    GradientRecord,
    Observation,
    Output,
    OutputKind,
    Plan,
    PlanContext,
    Read,
    Reduction,
    RoleKind,
    Value,
    register_observation,
)
from mime.units import DIMENSIONLESS, FREQUENCY, TIME, Magnitude, Quantity, Time, s
from mime.vocabulary import FLUX_DENSITY, POSE, TWIST, Capability

if TYPE_CHECKING:
    from mime.build import ObservationBuilder


@register_observation
@dataclass(frozen=True, kw_only=True)
class PhaseLag(Observation):
    """The angle from the field bound to ``field`` to the body's moment, in the plane the field turns in (about
    ``about``, the field's rotation axis), positive where the field leads: ``PhaseLag(umr, field="field.magnetic",
    about=tube.axis, after=s(2.5))``. It is defined net of the lag the build declares: the magnetic torque reads the
    body's pose at the start of the step and the field at its end (the build report's lags), so the moment it turns is
    a step behind the one this observation reads, by the body's turn over the step; the lag reported is the angle the
    torque acts on, the one its balance ``m B sin(lag) = R omega`` holds for. It reads the body's spin about the axis
    too, the rate it turns at in turns per second, so that a run says whether the body follows the field (``mean_spin``
    against the field's frequency).
    It adds no edge into the physics, and keeps what it measures in its own node's state: ``lag`` and ``spin`` each
    step, and their means over the steps that end after ``after``."""

    name = "phase_lag"
    version = "0.1.0"
    setting_roles = {"about": DESIGN,
                     "after": convention(DESIGN, "the window starts at the run's start unless a later start is given")}
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="phase_lag.window",
            indicator="the time at the end of the step less the setting after, s: a step is counted whole once its "
                      "end lies after it",
            orders={"mean_lag": -1, "mean_spin": -1, "turned": -1, "lagged": -1, "observed": -1},
            route="count the step in which the setting after falls by the part of it that lies after it, with the lag "
                  "and the spin interpolated linearly between successive steps (the trapezoid rule): the window's "
                  "integrals and its length are then continuously differentiable in after, and so are the means "
                  "wherever some time has been observed",
            settings=("after",)),
        switch.Declaration(
            owner="phase_lag.mean",
            indicator="the time observed less half a timestep, s: the means are taken once some time is observed",
            orders={"mean_lag": -1, "mean_spin": -1},
            route="none: where observing starts each mean goes from none to its value, which has no derivative to "
                  "compute",
            settings=("after",)),
    ))
    references = ()
    validity = (
        "the turn back over the declared step is the body's spin about the axis times the step: exact for a body "
        "turning about the axis, as the rigid body turns by its new angular velocity; a tilting body's turn about "
        "other axes is not taken",
    )
    builds = {"": (PhaseLagNode,)}
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    net_of_lag = ("body", "field")
    reads = {"pose": Read("body", POSE.name), "twist": Read("body", TWIST.name),
             "flux_density": Read("field", FLUX_DENSITY.name)}
    outputs = (
        Output("lag", DIMENSIONLESS, "the angle from the field to the body's moment in the plane of rotation, in "
                                     "radians, positive where the field leads, net of the lag the build declares, "
                                     "each step"),
        Output("spin", FREQUENCY, "the body's rate of turning about the field's axis, in turns per second, each "
                                  "step"),
        Output("mean_lag", DIMENSIONLESS, "the mean of the lag over the steps that end after the setting after, in "
                                          "radians", kind=OutputKind.AT_END, reduces="lag", rule=Reduction.MEAN,
               after="after"),
        Output("mean_spin", FREQUENCY, "the mean of the spin over the steps that end after the setting after",
               kind=OutputKind.AT_END, reduces="spin", rule=Reduction.MEAN, after="after"),
        Output("turned", DIMENSIONLESS, "the body's turns about the field's axis over the steps that end after the "
                                        "setting after: the spin's integral over them",
               kind=OutputKind.AT_END, reduces="spin", rule=Reduction.INTEGRAL, after="after"),
        Output("lagged", TIME, "the lag's integral over the steps that end after the setting after, in radian "
                               "seconds", kind=OutputKind.AT_END, reduces="lag", rule=Reduction.INTEGRAL, after="after"),
        Output("observed", TIME, "the time observed: the length of the steps that end after the setting after",
               kind=OutputKind.AT_END, after="after"),
    )
    body: Any = setting(default=None, kw_only=False)
    about: Annotated[DirectionLike, Direction]
    after: Annotated[Quantity, Time] = s(0.0)
    field: Any = None

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        after = settings["after"]
        return [] if isinstance(after, float) and after >= 0.0 else [f"after is {after!r} s, not a time from 0 on"]

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        """A uniform field is read with no points; any other where the magnetic torque reads it, at the body's
        magnet's centroid. The moment is the body's value."""
        values = {"moment": Value("body", "magnetic_moment")}
        provider = ctx.capabilities.get("field")
        if provider is not None and Capability.UNIFORM in provider.flags:
            return Plan(reads=dict(self.reads), values=values)
        reads = {**self.reads, "flux_density": Read("field", FLUX_DENSITY.name, at=Read("body", POSE.name),
                                                    offset="magnet_centre")}
        return Plan(reads=reads, values={**values, "magnet_centre": Value("body", "magnet_centre")})

    @override
    def net_of(self, node: Any, behind: int, why: str) -> PhaseLagNode:
        return cast(PhaseLagNode, node).netted(behind, why)

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        moment = ctx.values["moment"]
        node = b.add_node(PhaseLagNode(ctx.node_name(), ctx.timestep,
                                       moment=moment if isinstance(moment, Sequence) else (moment,),
                                       about=ctx.vector("about"), after=ctx.scalar("after")))
        for label in ("pose", "twist", "flux_density"):
            b.connect(b.read(label), into=node[label])
