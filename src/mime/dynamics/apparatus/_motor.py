from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.gradients import GradientStatus
from mime.nodes.actuation import KinematicMotorNode
from mime.registry import BuildContext, ChainSegment, Command, Problem, Segment, register_dynamics
from mime.units import DIMENSIONLESS, FREQUENCY, Magnitude, Quantity
from mime.vocabulary import Capability

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration

_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
#: What each prescription commands, and the command's dimension: an angle (rad), or a speed (rad/s).
PRESCRIPTIONS = {"angle": DIMENSIONLESS, "velocity": FREQUENCY}


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class Motor(ChainSegment):
    """A motor as a chain segment: a rotor turning about ``axis`` relative to its stator. Its frames are ``base``, the
    stator, placed on the frame it mounts on, and ``rotor``; a magnet on the rotor mounts on ``<segment>.rotor``. The
    axis is stated in the motor's own base frame, the frame it mounts on turned by its ``orientation``, never in the
    world: ``axis=Z`` is that frame's +Z, so it turns with the arm that carries it.

    ``kinematic`` (``Motor.Kinematic(...)``): its motion is prescribed, ``prescribed="angle"`` by the command
    ``<segment>.angle`` (rad), or ``prescribed="velocity"`` by ``<segment>.velocity`` (rad/s, integrated over each
    step), from ``initial_angle``. An ideal source of that motion: no rotor inertia, cogging, imbalance or thermal
    terms, and no load taken; the ``dynamic`` rung (torque in, the rotor's inertia) is reserved."""

    name = "motor"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ("Sola2017",)
    builds = {"kinematic": (KinematicMotorNode,)}
    rungs = ("kinematic", "dynamic")
    reserved_rungs = {"dynamic": "the dynamic motor (torque in, its rotor's inertia, its reaction torque on the link "
                                 "it is mounted on) is reserved, not built"}
    flags = frozenset({Capability.RIGID})
    limits = ("an ideal source of the prescribed motion, with no cogging, imbalance or thermal terms (MIME-ANO-102, as "
              "it applies to a prescribed motion)",
              "a prescribed motion: a load on it is not applied, and its reaction torque on the link it is mounted on "
              "is not delivered (MIME-ANO-103)")
    rung: str | None = None
    axis: tuple[float, float, float]
    prescribed: str
    initial_angle: Annotated[Quantity, DIMENSIONLESS]

    @classmethod
    def Kinematic(cls, **settings: Any) -> Motor:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="kinematic", **settings)

    def unit_axis(self) -> tuple[float, float, float] | None:
        try:
            x, y, z = (float(c) for c in self.axis)
        except (TypeError, ValueError):
            return None
        length = math.sqrt(x * x + y * y + z * z)
        if not (math.isfinite(length) and length > 0.0):
            return None
        return (x / length, y / length, z / length)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        if self.unit_axis() is None:
            found.append(f"axis={self.axis!r} is not a direction: three numbers, not all zero, in the motor's base "
                         f"frame (axis=Z is its +Z)")
        if self.prescribed not in PRESCRIPTIONS:
            found.append(f"prescribed={self.prescribed!r} is not one of {', '.join(PRESCRIPTIONS)}")
        if not isinstance(settings["initial_angle"], float):
            found.append(f"initial_angle is {settings['initial_angle']!r}, not one angle")
        return found

    @override
    def segment(self, settings: Mapping[str, Magnitude]) -> Segment:
        axis, angle = self.unit_axis(), settings["initial_angle"]
        if axis is None or not isinstance(angle, float):
            return Segment((), problems=(Problem("chain.initial", "the motor's axis or initial angle is not one",
                                                 "give a direction and one angle"),))
        half = angle / 2.0
        rotor = (0.0, 0.0, 0.0, math.cos(half), *(math.sin(half) * c for c in axis))
        return Segment(KinematicMotorNode.FRAMES,
                       {self.prescribed: Command(self.prescribed, PRESCRIPTIONS[self.prescribed])},
                       initial=(_IDENTITY, rotor))

    @override
    def load_refusal(self) -> str | None:
        if self.rung == "kinematic":
            return ("the kinematic motor's angle is prescribed by its command, so a load would not turn it; the "
                    "dynamic rung takes torques")
        return None

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        mount = ctx.mount
        node = b.add_node(KinematicMotorNode(
            ctx.node_name(), ctx.timestep, axis=self.unit_axis() or (0.0, 0.0, 1.0), prescribed=self.prescribed,
            placement=self.placement(ctx.settings), mount_index=mount.index if mount else -1,
            mount_count=mount.count if mount else 0, initial_mount_pose=mount.initial_pose if mount else _IDENTITY,
            initial_angle=ctx.scalar("initial_angle")))
        if mount is not None:
            b.mount(into=node["mount_poses"])


__all__ = ["PRESCRIPTIONS", "Motor"]
