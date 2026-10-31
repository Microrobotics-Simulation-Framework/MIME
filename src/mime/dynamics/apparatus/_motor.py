from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, override

from mime.gradients import GradientStatus
from mime.nodes.actuation import DynamicMotorNode, KinematicMotorNode
from mime.registry import BuildContext, CapabilityRecord, ChainSegment, Command, Port, Problem, Segment, \
    Spinner, register_dynamics
from mime.units import DIMENSIONLESS, FREQUENCY, MOMENT_OF_INERTIA, ROTATIONAL_DAMPING, TORQUE, Magnitude, Quantity
from mime.vocabulary import FRAME_ANGULAR_VELOCITIES, FRAME_POSES, LINK_WRENCH, Capability

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration

_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
#: What each prescription commands, and the command's dimension: an angle (rad), or a speed (rad/s).
PRESCRIPTIONS = {"angle": DIMENSIONLESS, "velocity": FREQUENCY}
KINEMATIC_LIMITS = (
    "an ideal source of the prescribed motion, with no cogging, imbalance or thermal terms (MIME-ANO-102, as it "
    "applies to a prescribed motion)",
    "a prescribed motion: a load on it is not applied, and its reaction torque on the link it is mounted on is not "
    "delivered (MIME-ANO-103); mass on its rotor is an articulated payload, refused on a dynamic segment")
DYNAMIC_LIMITS = (
    "an ideal torque source: the torque commanded is the torque on the rotor, with no cogging, imbalance, armature, "
    "saturation or thermal terms (MIME-ANO-102)",
    "rigid bearings, and a rotor symmetric about its axis with its centre of mass on it (MIME-ANO-122): it carries the "
    "rotor's absolute spin about its axis, its inertia the declared one with the moments of the parts mounted on its "
    "rotor, and on a dynamic segment the rotor's mass and mean inertia across its axis are welded on the link it is "
    "mounted on; there a tilted or unbalanced rotor is refused, and one unequal across its axis is accepted where the "
    "worst case of the term not carried, from the link's joint speed limits and the motor's top_speed, is at most half "
    "the gyroscopic torque carried at those speeds (the build report gives both); on a kinematic segment, whose motion "
    "no mounted mass changes, every rotor is accepted, and its tilt, imbalance or inequality across its axis is not "
    "carried in its own spin",
    "the angular velocity of the frame it is mounted on is read a step late, as its pose is, for its speed relative to "
    "its stator, its bearing's damping and its gyroscopic torque",
    "its reaction (the drive's and the bearing damping's reaction about its axis, and the loads on its rotor across "
    "it) loads the segment it is mounted on a step late, what its stator passed over the step before, and only where "
    "that segment takes loads, a dynamic one (MIME-ANO-103)")


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
    terms, and no load taken.

    ``dynamic`` (``Motor.Dynamic(...)``): its rotor, of ``inertia`` about the axis, is turned by the torque commanded
    (``<segment>.torque``, N m) against its bearing's viscous ``damping`` and the loads on it about the axis, from
    ``initial_angle`` and ``initial_velocity`` (relative to its stator): an ideal torque source. It is a spinner: it
    carries the rotor's absolute spin about its axis, and the parts mounted on ``<segment>.rotor`` are its rotor, their
    moments about the axis added to ``inertia``; on a dynamic segment their mass and mean inertia across the axis are
    welded on the link the motor is mounted on, and a rotor unequal across its axis needs ``top_speed``, the most its
    spin reaches (rad/s), for the worst case of the term not carried. It takes loads on its frames
    (``apparatus.link_wrench``: the reaction on a magnet on its rotor), and its stator passes the drive's and bearing's
    reactions, the rotor's loads across its axis and its gyroscopic torque to the frame it is mounted on, which takes
    them a step late where its segment takes loads (MIME-ANO-103). A part on ``<segment>.base`` rides on that frame."""

    name = "motor"
    version = "0.2.0"
    differentiability = GradientStatus.EXACT
    references = ("Sola2017",)
    builds = {"kinematic": (KinematicMotorNode,), "dynamic": (DynamicMotorNode,)}
    rungs = ("kinematic", "dynamic")
    flags = frozenset({Capability.RIGID})
    publishes = MappingProxyType({FRAME_POSES.name: Port("frame_poses"),
                                  FRAME_ANGULAR_VELOCITIES.name: Port("frame_angular_velocities")})
    limits = KINEMATIC_LIMITS
    rung: str | None = None
    axis: tuple[float, float, float]
    prescribed: str | None = None
    initial_angle: Annotated[Quantity, DIMENSIONLESS]
    initial_velocity: Annotated[Quantity | None, FREQUENCY] = None
    inertia: Annotated[Quantity | None, MOMENT_OF_INERTIA] = None
    damping: Annotated[Quantity | None, ROTATIONAL_DAMPING] = None
    top_speed: Annotated[Quantity | None, FREQUENCY] = None

    @classmethod
    def Kinematic(cls, **settings: Any) -> Motor:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="kinematic", **settings)

    @classmethod
    def Dynamic(cls, **settings: Any) -> Motor:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="dynamic", **settings)

    @override
    def capabilities(self) -> CapabilityRecord:
        if self.rung != "dynamic":
            return super().capabilities()
        return CapabilityRecord(self.label(), frozenset({*self.flags, Capability.INERTIAL}), dict(self.publishes),
                                {LINK_WRENCH.name: Port("link_wrench")}, frozenset(), DYNAMIC_LIMITS)

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
        if not isinstance(settings["initial_angle"], float):
            found.append(f"initial_angle is {settings['initial_angle']!r}, not one angle")
        if self.rung == "dynamic":
            if self.prescribed is not None:
                found.append("prescribed is the kinematic rung's: the dynamic motor is turned by the torque commanded")
            inertia = settings.get("inertia")
            if not (isinstance(inertia, float) and math.isfinite(inertia) and inertia > 0.0):
                found.append(f"inertia is {inertia!r} kg m^2: the dynamic motor needs its rotor's moment of inertia "
                             f"about its axis, above zero")
            for name in ("damping", "initial_velocity"):
                value = settings.get(name, 0.0)
                if not (isinstance(value, float) and math.isfinite(value)) or (name == "damping" and value < 0.0):
                    found.append(f"{name} is {value!r}, not one finite number" + (" of zero or more"
                                                                                  if name == "damping" else ""))
            top = settings.get("top_speed", 1.0)
            if not (isinstance(top, float) and math.isfinite(top) and top > 0.0):
                found.append(f"top_speed is {top!r} rad/s, not one speed above zero")
            return found
        if self.prescribed not in PRESCRIPTIONS:
            found.append(f"prescribed={self.prescribed!r} is not one of {', '.join(PRESCRIPTIONS)}")
        given = [name for name in ("inertia", "damping", "initial_velocity", "top_speed") if name in settings]
        if given:
            found.append(f"{', '.join(given)} {'is' if len(given) == 1 else 'are'} the dynamic rung's: the kinematic "
                         f"motor's motion is prescribed")
        return found

    @override
    def segment(self, settings: Mapping[str, Magnitude]) -> Segment:
        axis, angle = self.unit_axis(), settings["initial_angle"]
        if axis is None or not isinstance(angle, float):
            return Segment((), problems=(Problem("chain.initial", "the motor's axis or initial angle is not one",
                                                 "give a direction and one angle"),))
        half = angle / 2.0
        rotor = (0.0, 0.0, 0.0, math.cos(half), *(math.sin(half) * c for c in axis))
        if self.rung == "dynamic":
            relative = settings.get("initial_velocity", 0.0)
            speed = relative if isinstance(relative, float) else 0.0
            return Segment(DynamicMotorNode.FRAMES, {"torque": Command("torque", TORQUE)}, initial=(_IDENTITY, rotor),
                           initial_rates=((0.0, 0.0, 0.0), (speed * axis[0], speed * axis[1], speed * axis[2])))
        prescribed = self.prescribed or "angle"
        return Segment(KinematicMotorNode.FRAMES, {prescribed: Command(prescribed, PRESCRIPTIONS[prescribed])},
                       initial=(_IDENTITY, rotor))

    @override
    def load_refusal(self) -> str | None:
        if self.rung == "kinematic":
            return ("the kinematic motor's angle is prescribed by its command, so a load would not turn it; the "
                    "dynamic rung takes torques")
        return None

    @override
    def riding(self) -> tuple[str, ...]:
        return ("base",)

    @override
    def spinner(self, settings: Mapping[str, Magnitude]) -> Spinner | None:
        if self.rung != "dynamic":
            return None
        inertia, top = settings.get("inertia"), settings.get("top_speed")
        return Spinner("rotor", self.unit_axis() or (0.0, 0.0, 1.0), inertia if isinstance(inertia, float) else 0.0,
                       top if isinstance(top, float) else None)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        mount = ctx.mount
        placed: dict[str, Any] = {
            "placement": self.placement(ctx.settings), "mount_index": mount.index if mount else -1,
            "mount_count": mount.count if mount else 0,
            "initial_mount_pose": mount.initial_pose if mount else _IDENTITY,
            "initial_mount_rate": mount.initial_rate if mount else (0.0, 0.0, 0.0)}
        if self.rung == "dynamic":
            node = b.add_node(DynamicMotorNode(
                ctx.node_name(), ctx.timestep, axis=self.unit_axis() or (0.0, 0.0, 1.0),
                inertia=ctx.spin_inertia if ctx.spin_inertia is not None else ctx.scalar("inertia"),
                damping=ctx.scalar("damping") if "damping" in ctx.settings else 0.0, **placed,
                initial_angle=ctx.scalar("initial_angle"),
                initial_angular_velocity=ctx.scalar("initial_velocity") if "initial_velocity" in ctx.settings else 0.0))
            if mount is not None:
                b.mount(into=node["mount_poses"])
                b.mount_rates(into=node["mount_rates"])
                b.reaction(node["reaction"])
            return
        node = b.add_node(KinematicMotorNode(
            ctx.node_name(), ctx.timestep, axis=self.unit_axis() or (0.0, 0.0, 1.0),
            prescribed=self.prescribed or "angle", **placed, initial_angle=ctx.scalar("initial_angle")))
        if mount is not None:
            b.mount(into=node["mount_poses"])
            b.mount_rates(into=node["mount_rates"])


__all__ = ["PRESCRIPTIONS", "Motor"]
