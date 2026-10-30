from __future__ import annotations

import hashlib
import math
from types import MappingProxyType
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, cast, override

import jax
import numpy as np

from mime.core import jnp, switch
from mime.core.multibody import ClosedChainRefused, DescriptionReading, DescriptionRefused, InvalidDescription, \
    forward_kinematics, mass_matrix, read_urdf
from mime.gradients import GradientStatus
from mime.nodes.actuation import SWEEPS, DynamicArmNode, KinematicArmNode
from mime.registry import BuildContext, CapabilityRecord, ChainSegment, Command, GradientRecord, Port, Problem, \
    RungGradient, Segment, register_dynamics
from mime.scene import Asset
from mime.units import DIMENSIONLESS, FREQUENCY, TORQUE, Magnitude, Quantity, Time
from mime.vocabulary import FRAME_POSES, JOINT_POSITIONS, JOINT_VELOCITIES, LINK_WRENCH, Capability

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration

#: Where to ask for what MIME does not do.
ISSUES = "https://github.com/Microrobotics-Simulation-Framework/MIME/issues"
_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

KINEMATIC_SWITCHES = (
    switch.Declaration(
        owner="arm.limit.low",
        indicator="a bounded joint's commanded position less its lower limit, rad (m on a prismatic joint)",
        orders={"joint_positions": 0, "frame_poses": 0},
        route="none: a commanded position below the limit is held at the limit, whose derivative in the command is "
              "zero; a smooth stop would be another model of the arm"),
    switch.Declaration(
        owner="arm.limit.high",
        indicator="a bounded joint's upper limit less its commanded position, rad (m on a prismatic joint)",
        orders={"joint_positions": 0, "frame_poses": 0},
        route="none: a commanded position above the limit is held at the limit, as at the lower one"),
)
DYNAMIC_SWITCHES = (
    switch.Declaration(
        owner="arm.stop.low",
        indicator="a bounded joint's position at the end of the step, before its stop, less its lower limit, rad "
                  "(m on a prismatic joint)",
        orders={"joint_positions": 0, "frame_poses": 0, "joint_velocities": 0},
        route="none: a rigid, inelastic stop holds the joint's velocity to the one that brings it to its limit at the "
              "end of the step, through an impulse on the step's system that moves every joint coupled to it, a "
              "kink in the velocities and positions; a compliant stop would be another model of the joint"),
    switch.Declaration(
        owner="arm.stop.high",
        indicator="a bounded joint's upper limit less its position at the end of the step, before its stop, rad "
                  "(m on a prismatic joint)",
        orders={"joint_positions": 0, "frame_poses": 0, "joint_velocities": 0},
        route="none: a rigid, inelastic stop, as at the lower limit"),
    switch.Declaration(
        owner="arm.friction.low",
        indicator="the torque that would stop a joint over the step, its inertia that with the other joints free and "
                  "their impulses as the sweep leaves them, plus its Coulomb friction, N m (N on a prismatic joint): "
                  "below zero the friction cannot stop it",
        orders={"joint_positions": 0, "frame_poses": 0, "joint_velocities": 0},
        route="none: Coulomb friction is a kink in the velocity between sticking and sliding; a smoothed friction "
              "law would be another model of the joint"),
    switch.Declaration(
        owner="arm.friction.high",
        indicator="a joint's Coulomb friction less the torque that would stop it over the step, its inertia that with "
                  "the other joints free and their impulses as the sweep leaves them, N m (N on a prismatic joint)",
        orders={"joint_positions": 0, "frame_poses": 0, "joint_velocities": 0},
        route="none: Coulomb friction is a kink between sticking and sliding, as on the other side"),
    switch.Declaration(
        owner="arm.stick",
        indicator="where the sweeps start: the largest so far, and at least 1, of the squared ratio of the impulse "
                  "that would hold a joint with friction still over the step (every such joint held, the others "
                  "free) to its friction's, less the next joint's, dimensionless; the start is scaled by its "
                  "reciprocal",
        orders={"joint_positions": 0, "frame_poses": 0, "joint_velocities": 0},
        route="none: the start is scaled to what the joints' friction can hold, a kink in the impulses the sweeps "
              "start from, which reaches the step's velocity only as far as the sweeps leave it unconverged"),
)
#: What each rung publishes and takes, and the limits its record states.
KINEMATIC_LIMITS = (
    "rigid joints and links, with no joint flexibility, backlash or structural mode, and no contact between links "
    "(MIME-ANO-101)",
    "a prescribed motion: a load on it is not applied, the reaction of a field on a magnet mounted on it and a mounted "
    "motor's reaction torque among them (MIME-ANO-103)")
DYNAMIC_LIMITS = (
    "rigid joints and links, with no joint flexibility, backlash or structural mode, and no contact between links "
    "(MIME-ANO-101)",
    "its base fixed in the world: it is its chain's first segment",
    "the masses of what is mounted on it (a motor, a magnet) are not added to its links: give them in its description; "
    "nor is a mounted motor's rotor inertia about its axis (MIME-ANO-122)",
    "every moving joint moves mass or has armature: a moving body with geometry and no <inertial> is refused, and so "
    "is a joint with no inertia about its axis at the initial positions (one whose bodies have neither geometry nor "
    "<inertial>)",
    "a joint limit is a rigid, inelastic stop: a joint driven past it arrives at it at the end of the step, and the "
    "stop takes the rest of its velocity on the next; its impulse moves every joint coupled to it; the limits on its "
    "joints' speeds and efforts in its description are not applied",
    f"the stops' and Coulomb friction's impulses are solved by {SWEEPS} projected Gauss-Seidel sweeps on the step's "
    f"system: exact for one joint and where every joint with friction sticks, and never adding kinetic energy; on "
    f"coupled joints, the change a converged solve would still make to the velocity, relative to the unconstrained "
    f"velocity, measured over drawn states at worst 2e-6 on the AR4 and 1e-2 on a strongly coupled planar arm both "
    f"of whose joints reach a stop in one step",
    "a mounted motor's reaction loads it a step late: what the motor's stator passed over the step before "
    "(MIME-ANO-103)")


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class Arm(ChainSegment):
    """A robot arm as a chain segment, general over its description: the multibody tree its URDF file
    (``description``, a file asset pinned by its SHA-256) describes, so that another arm is another file. Its frames
    are the tree's, every link's and every named frame's; what mounts on it names one (``mount="link_6"``).

    ``kinematic`` (``Arm.Kinematic(...)``): its joint positions are commanded (the command
    ``<segment>.joint_positions``, in radians, or metres on a prismatic joint), each held within its joint's limits
    where it has them; with ``servo``, a time constant, the joints follow the held command as a first-order lag,
    otherwise at once; every frame's world pose follows by forward kinematics. It has no mass matrix and takes no
    load: a law that would load it is refused, since a load cannot move a prescribed motion.

    ``dynamic`` (``Arm.Dynamic(...)``): its joints are driven by torques (the command ``<segment>.joint_torques``, N m,
    or N on a prismatic joint) against its inertia, the experiment's gravity, each joint's damping and Coulomb
    friction from the description, and the loads on its frames (``apparatus.link_wrench``: a reaction on a source
    mounted on it, a mounted motor's), from ``initial_positions`` and ``initial_velocities``; a joint limit is a rigid,
    inelastic stop, and the stops and friction act as impulses through the arm's inertia, so that what holds one joint
    moves the joints coupled to it. It publishes its joint velocities too. Its base is fixed in the world, so it is its
    chain's first segment (``chain.fixed-base``), and every moving joint must move mass: a moving body with geometry
    and no ``<inertial>`` is refused (``chain.inertia-gap``), and so is a joint with no inertia about its axis at the
    initial positions, whose bodies have neither geometry nor ``<inertial>`` and which has no armature
    (``chain.massless-joint``).

    Every element of the file is read and used, recognised and not used, or refused, as ``mime.core.multibody``'s
    reader sorts it; each kind not used is warned about once and listed in the build report and the saved form. An
    element refused because dropping it changes the mechanism (a mimic joint, an element URDF does not define) is
    refused unless ``acknowledged`` gives it a reason, ``((path, reason), ...)``, which the saved form records.

    A closed chain (a parallel mechanism), which URDF cannot describe, is refused, and is out of scope until a
    downstream user needs one: ask at the issues, or register your own chain segment (``mime.registry.ChainSegment``)
    that takes its own description and publishes the same frame poses, so that mounts work unchanged."""

    name = "arm"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(*KINEMATIC_SWITCHES, *DYNAMIC_SWITCHES))
    rung_gradients = {
        "kinematic": RungGradient(differentiability=GradientStatus.PIECEWISE_KINK,
                                  gradient=GradientRecord(switches=KINEMATIC_SWITCHES)),
        "dynamic": RungGradient(differentiability=GradientStatus.PIECEWISE_KINK,
                                gradient=GradientRecord(switches=DYNAMIC_SWITCHES)),
    }
    references = ("Sciavicco2000", "Featherstone2008")
    builds = {"kinematic": (KinematicArmNode,), "dynamic": (DynamicArmNode,)}
    rungs = ("kinematic", "dynamic")
    flags = frozenset({Capability.RIGID})
    publishes = MappingProxyType({FRAME_POSES.name: Port("frame_poses"),
                                  JOINT_POSITIONS.name: Port("joint_positions")})
    limits = KINEMATIC_LIMITS
    rung: str | None = None
    description: Asset
    initial_positions: Annotated[Quantity, DIMENSIONLESS]
    initial_velocities: Annotated[Quantity | None, FREQUENCY] = None
    servo: Annotated[Quantity | None, Time] = None
    acknowledged: tuple[tuple[str, str], ...] = ()

    @classmethod
    def Kinematic(cls, **settings: Any) -> Arm:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="kinematic", **settings)

    @classmethod
    def Dynamic(cls, **settings: Any) -> Arm:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="dynamic", **settings)

    @override
    def capabilities(self) -> CapabilityRecord:
        if self.rung != "dynamic":
            return super().capabilities()
        return CapabilityRecord(self.label(), frozenset({*self.flags, Capability.INERTIAL}),
                                {**self.publishes, JOINT_VELOCITIES.name: Port("joint_velocities")},
                                {LINK_WRENCH.name: Port("link_wrench")}, frozenset(), DYNAMIC_LIMITS)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        if not isinstance(cast(object, self.description), Asset):
            found.append(f"description={self.description!r} is not a file asset: write description=asset(\"arm.urdf\")"
                         f" with asset from mime.scene")
        servo = settings.get("servo")
        if servo is not None and not (isinstance(servo, float) and math.isfinite(servo) and servo > 0.0):
            found.append(f"servo is {servo!r} s, not a time constant above zero")
        if self.rung == "dynamic":
            if servo is not None:
                found.append("servo is the kinematic rung's: the dynamic arm follows the torques it is given")
        elif "initial_velocities" in settings:
            found.append("initial_velocities is the dynamic rung's: the kinematic arm's joints move as commanded")
        written = cast(object, self.acknowledged)  # a script is not type-checked
        pairs = list(cast(tuple[object, ...], written)) if isinstance(written, tuple) else [written]
        wrong: list[object] = [pair for pair in pairs if not _reasoned(pair)]
        if wrong:
            found.append(f"acknowledged holds {wrong[0]!r}, not a pair (element path, reason)")
        return found

    def read(self) -> DescriptionReading | Problem:
        """The description read, or the problem that refuses it."""
        description = self.description
        try:
            content = description.read()
        except OSError as error:
            return Problem("description.missing", f"cannot read the description {description.path}: {error}",
                           "pass the path of a URDF file that exists, as asset(\"arm.urdf\")")
        found = hashlib.sha256(content).hexdigest()
        if description.sha256 is not None and found != description.sha256.lower():
            return Problem("description.changed", f"the description {description.path} has SHA-256 {found}, and it is "
                                                  f"pinned to {description.sha256}",
                           "use the file the experiment was written against, or pin the new one")
        try:
            return read_urdf(description.path, sha256=found, acknowledged=dict(self.acknowledged))
        except DescriptionRefused as refused:
            return Problem("description.refused", str(refused),
                           "edit the file, or acknowledge each element named with its reason: acknowledged=((path, "
                           "reason), ...), which the saved form records")
        except InvalidDescription as error:
            return Problem("description.invalid", f"{description.path}: {error}", "correct the description file")

    @override
    def segment(self, settings: Mapping[str, Magnitude]) -> Segment:
        reading = self.read()
        if isinstance(reading, Problem):
            return Segment((), problems=(reading,))
        tree = reading.tree
        if tree.closures:
            refused = ClosedChainRefused("the robot arm", tuple(c.name for c in tree.closures))
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.closed-loop", str(refused), f"describe an open tree, ask for closed chains at {ISSUES}, or "
                                                   f"register your own chain segment publishing the same frame "
                                                   f"poses"),))
        dofs = tree.articulation.dofs
        dynamic = self.rung == "dynamic"
        gaps = [gap.link for gap in reading.classification.inertia_gaps if gap.needed_by_dynamics]
        if dynamic and gaps:
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.inertia-gap", f"{tree.name}'s moving bodies with geometry and no <inertial> ({', '.join(gaps)}) "
                                     f"have no mass the dynamic arm can move: URDF reads them as massless, and MIME "
                                     f"infers no mass from geometry",
                f"add an <inertial> to each, from CAD or a measurement, use the kinematic rung, or ask for inference "
                f"from geometry at {ISSUES}"),))
        if dynamic and self.mount is not None:
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.fixed-base", f"the dynamic arm's base is fixed in the world, and it is mounted on "
                                    f"{self.mount!r}",
                "make it its chain's first segment, placed in the world with at= and orientation="),))
        initial = _components(settings["initial_positions"])
        if len(initial) != dofs or not all(math.isfinite(c) for c in initial):
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.initial", f"initial_positions has {len(initial)} values, and {tree.name} has {dofs} moving "
                                 f"joints ({', '.join(tree.articulation.dof_names)})",
                "give one finite position per moving joint, in that order"),))
        rates = _components(settings["initial_velocities"]) if "initial_velocities" in settings else (0.0,) * dofs
        if len(rates) != dofs or not all(math.isfinite(c) for c in rates):
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.initial", f"initial_velocities has {len(rates)} values, and {tree.name} has {dofs} moving "
                                 f"joints ({', '.join(tree.articulation.dof_names)})",
                "give one finite rate per moving joint, in that order"),))
        articulation = tree.articulation
        beyond = [f"{name} = {position:g} (limits {low:g} to {high:g})" for name, position, low, high in
                  zip(articulation.dof_names, initial, np.asarray(articulation.lower), np.asarray(articulation.upper),
                      strict=True) if not low <= position <= high]
        if beyond:
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.initial", f"initial_positions lies beyond a joint's limits: {', '.join(beyond)}; the arm holds "
                                 f"every position within them, so it would jump at its first step",
                "start every joint within its limits"),))
        with cast(Any, jax).enable_x64(True):
            frames = np.asarray(forward_kinematics(tree, jnp.asarray(initial, dtype=jnp.float64)), dtype=np.float64)
            inertia = np.diagonal(np.asarray(mass_matrix(tree, jnp.asarray(initial, dtype=jnp.float64)))) \
                if dynamic else np.ones(dofs)
        unmoved = [name for name, own in zip(articulation.dof_names, inertia, strict=True)
                   if not own > _ROUNDING * float(np.max(inertia))]
        if unmoved:
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.massless-joint", f"{tree.name}'s moving joint{'s' if len(unmoved) > 1 else ''} "
                                        f"({', '.join(unmoved)}) move no inertia about their axes at the initial "
                                        f"positions: the bodies they move have neither geometry nor an <inertial>, "
                                        f"and the joints no armature, so the dynamic arm has no mass to move there",
                f"give each body it moves an <inertial>, from CAD or a measurement, or the joint an armature; or use "
                f"the kinematic rung"),))
        command = Command("joint_torques", TORQUE, (dofs,)) if dynamic else \
            Command("joint_positions", DIMENSIONLESS, (dofs,))
        return Segment(tree.frame_names, {command.port: command}, reading.classification,
                       initial=tuple(tuple(float(c) for c in row) for row in frames))

    @override
    def load_refusal(self) -> str | None:
        if self.rung == "kinematic":
            return ("the kinematic arm's joint positions are prescribed by its command, so a load would not move it; "
                    "the dynamic rung takes loads")
        return None

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        reading = self.read()
        if isinstance(reading, Problem):
            raise ValueError("phase 1 refuses an arm whose description it cannot read")
        if self.rung == "dynamic":
            tree = reading.tree
            names = tree.articulation.dof_names
            b.add_node(DynamicArmNode(
                ctx.node_name(), ctx.timestep,
                **DynamicArmNode.described(tree.articulation, [tree.joint(name).damping for name in names],
                                           [tree.joint(name).friction for name in names]),
                placement=self.placement(ctx.settings), gravity=ctx.gravity,
                initial_joint_positions=_components(ctx.settings["initial_positions"]),
                initial_joint_velocities=_components(ctx.settings["initial_velocities"])
                if "initial_velocities" in ctx.settings else None))
            return
        mount = ctx.mount
        node = b.add_node(KinematicArmNode(
            ctx.node_name(), ctx.timestep, **KinematicArmNode.described(reading.tree.articulation),
            placement=self.placement(ctx.settings), mount_index=mount.index if mount else -1,
            mount_count=mount.count if mount else 0,
            initial_mount_pose=mount.initial_pose if mount else _IDENTITY,
            initial_joint_positions=_components(ctx.settings["initial_positions"]),
            servo_time_constant=ctx.scalar("servo") if "servo" in ctx.settings else None))
        if mount is not None:
            b.mount(into=node["mount_poses"])


#: A joint's inertia about its axis no larger than this share of the arm's largest is rounding of none (64 units in the
#: last place of float64, in which the build computes it).
_ROUNDING = 64 * float(np.finfo(np.float64).eps)


def _reasoned(pair: object) -> bool:
    """Whether ``pair`` is an element's path and the reason it is read past, two strings."""
    held = cast(tuple[object, ...], pair) if isinstance(pair, tuple) else ()
    return len(held) == 2 and all(isinstance(item, str) for item in held)


def _components(value: Magnitude) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


__all__ = ["Arm"]
