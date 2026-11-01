from __future__ import annotations

import hashlib
import math
import warnings
from types import MappingProxyType
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, cast, override

import jax
import numpy as np

from mime.core import jnp, switch
from mime.core.multibody import ROOT, ClosedChainRefused, DescriptionReading, DescriptionRefused, \
    DescriptionWarning, InvalidDescription, JointKind, MultibodyTree, Payload, Transmission, articulate, \
    forward_kinematics, frame_motions, graft, mass_matrix, read_urdf, with_armatures
from mime.dynamics.apparatus._actuators import actuator
from mime.gradients import GradientStatus
from mime.nodes.actuation import SWEEPS, DynamicArmNode, KinematicArmNode
from mime.registry import DESIGN, PHYSICAL, convention
from mime.registry import BuildContext, CapabilityRecord, ChainSegment, Command, GradientRecord, Port, Problem, \
    RungGradient, Segment, SegmentVisual, register_dynamics
from mime.scene import Asset
from mime.units import DIMENSIONLESS, FORCE, FREQUENCY, LENGTH, MASS, MOMENT_OF_INERTIA, TORQUE, Magnitude, Quantity, \
    Time, unit_symbol
from mime.validity import basis_text
from mime.vocabulary import FRAME_ANGULAR_VELOCITIES, FRAME_POSES, JOINT_POSITIONS, JOINT_VELOCITIES, LINK_INERTIA, \
    LINK_MOMENTUM, LINK_WRENCH, Capability

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
        indicator="where the sweeps start: the largest so far, and never below one, of the squared ratio of the impulse "
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
    "motor's reaction torque among them (MIME-ANO-103), and no mass mounted on it changes it: every mounted mass is "
    "accepted, tilted, unbalanced or articulated, and none is carried")
DYNAMIC_LIMITS = (
    "rigid joints and links, with no joint flexibility, backlash or structural mode, and no contact between links "
    "(MIME-ANO-101)",
    "its base fixed in the world: it is its chain's first segment",
    "what is mounted on its links with a mass (a part, a stator, a dynamic motor's rotor) is welded on the link at "
    "build, the link's own inertial untouched; a rotor's inertia about its axis rides on its spin, not on the link "
    "(MIME-ANO-122); an articulated payload (a segment with joints of its own carrying mass), a tilted or unbalanced "
    "rotor, and a rotor too unequal across its axis for its speeds are refused, and the kinematic rung takes each",
    "every moving joint moves mass or has armature: a moving body with geometry and no <inertial> is refused, and so "
    "is a joint with no inertia about its axis at the initial positions (one whose bodies have neither geometry nor "
    "<inertial>)",
    "each joint's armature (a geared drive's rotor inertia reflected onto it, kg m^2 on a revolute joint and kg on a "
    "prismatic one) is on the diagonal of its mass matrix wherever its dynamics uses it, declared (armature=) or "
    "reflected through a simple transmission of its description from its actuator's rotor (actuators=); none where "
    "neither gives one, the URDF's own meaning, which the build report says; a transmission that couples joints is not "
    "read",
    "a joint limit is a rigid, inelastic stop: a joint driven past it arrives at it at the end of the step, and the "
    "stop takes the rest of its velocity on the next; its impulse moves every joint coupled to it; the limits on its "
    "joints' speeds and efforts in its description are not applied",
    f"the stops' and Coulomb friction's impulses are solved by {SWEEPS} projected Gauss-Seidel sweeps on the step's "
    f"system: exact for one joint and where every joint with friction sticks, and never adding kinetic energy; on "
    f"coupled joints, the change a converged solve would still make to the velocity, relative to the unconstrained "
    f"velocity, measured over drawn states at worst 3e-6 on the AR4 and 1.6e-2 on a strongly coupled planar arm both "
    f"of whose joints reach a stop in one step (the median exact)",
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
    mounted on it, a mounted motor's), from ``initial_positions`` and ``initial_velocities``; a prismatic joint's force
    is ``<segment>.joint_forces`` (N), and the kinematic rung's prismatic positions ``<segment>.joint_displacements``
    (m), each command over its kind's joints in their order. What is mounted on its links with a mass is welded on the
    link at build, a rigid body of its own in its tree: a part with a mass, a motor's stator, and a dynamic motor's
    rotor (its mass, and its mean inertia across its axis; its inertia about its axis rides on its spin). An
    articulated payload (a segment with joints of its own carrying mass) is refused (``chain.articulated-payload``),
    and so are a tilted or unbalanced rotor and one too unequal across its axis for its speeds; the kinematic rung,
    whose motion no mounted mass changes, takes every case. A joint limit is a rigid,
    inelastic stop, and the stops and friction act as impulses through the arm's inertia, so that what holds one joint
    moves the joints coupled to it. It publishes its joint velocities too. Its base is fixed in the world, so it is its
    chain's first segment (``chain.fixed-base``), and every moving joint must move mass: a moving body with geometry
    and no ``<inertial>`` is refused (``chain.inertia-gap``), and so is a joint with no inertia about its axis at the
    initial positions, whose bodies have neither geometry nor ``<inertial>`` and which has no armature
    (``chain.massless-joint``).

    **Armature**, on the dynamic rung: a geared joint's drive reflects its motor's rotor inertia onto the joint, ``N^2
    J`` through a reduction ``N``, which its description's links do not hold. ``armature`` declares it per joint
    (``armature={"joint_6": kg_m2(1e-4, reason=...)}``, kg on a prismatic joint), and ``actuators`` gives the rotor
    behind each simple transmission the description holds, by the actuator's name (``actuators={"motor_6":
    kg_m2(8.2e-6, reason=...)}``, or a registered actuator's name, ``"stepperonline.17hs19-2004s1"``), whose
    reduction reflects it. A joint given both ways, or twice, is refused (``chain.armature``). A transmission is judged
    only where ``actuators`` names its actuator: there one with anything wrong with it (the reader's and the tree's
    account of it, :meth:`MultibodyTree.transmission_faults`) is refused, naming what; one not used is recognised and
    not used, as every transmission was before, and the build report names what is wrong with it. A joint given
    neither has none, the URDF's own meaning: a convention, which the build report says and research lists. Each
    armature is on the diagonal of the mass matrix wherever the dynamics uses it, and on the tree a model-based
    controller reads (``ctx.models``).

    Every element of the file is read and used, recognised and not used, or refused, as ``mime.core.multibody``'s
    reader sorts it; each kind not used is warned about once and listed in the build report and the saved form. An
    element refused because dropping it changes the mechanism (a mimic joint, an element URDF does not define) is
    refused unless ``acknowledged`` gives it a reason, ``((path, reason), ...)``, which the saved form records.

    A closed chain (a parallel mechanism), which URDF cannot describe, is refused, and is out of scope until a
    downstream user needs one: ask at the issues, or register your own chain segment (``mime.registry.ChainSegment``)
    that takes its own description and publishes the same frame poses, so that mounts work unchanged."""

    name = "arm"
    version = "0.1.0"
    setting_roles = {
        "rung": DESIGN, "description": PHYSICAL, "initial_positions": DESIGN,
        "initial_velocities": convention(DESIGN, "a dynamic arm's joints start at rest unless given their rates"),
        "servo": convention(PHYSICAL, "the kinematic arm's joints take the held command at once unless a servo lag is "
                                      "given"),
        "acknowledged": convention(DESIGN, "no element of the description is read past unless acknowledged with its "
                                           "reason"),
        "armature": convention(PHYSICAL, "a joint whose armature is not declared, and whose transmission's actuator is "
                                         "not given, has none: the URDF's own meaning, no geared drive's rotor "
                                         "reflected onto it"),
        "actuators": convention(PHYSICAL, "a transmission whose actuator's rotor is not given reflects nothing onto "
                                          "its joint"),
    }
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
                                  FRAME_ANGULAR_VELOCITIES.name: Port("frame_angular_velocities"),
                                  JOINT_POSITIONS.name: Port("joint_positions")})
    limits = KINEMATIC_LIMITS
    rung: str | None = None
    description: Asset
    initial_positions: Annotated[Quantity, DIMENSIONLESS]
    initial_velocities: Annotated[Quantity | None, FREQUENCY] = None
    servo: Annotated[Quantity | None, Time] = None
    acknowledged: tuple[tuple[str, str], ...] = ()
    armature: Mapping[str, Quantity] | tuple[tuple[str, Quantity], ...] = ()
    actuators: Mapping[str, Quantity | str] | tuple[tuple[str, Quantity | str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("armature", "actuators"):
            object.__setattr__(self, name, _keyed(getattr(self, name)))

    def declared_armature(self) -> tuple[tuple[str, Quantity], ...]:
        """The armature declared, by joint, as pairs in the joints' names' order."""
        return cast(tuple[tuple[str, Quantity], ...], self.armature)

    def given_actuators(self) -> tuple[tuple[str, Quantity | str], ...]:
        """The actuators given, by name, as pairs in their names' order."""
        return cast(tuple[tuple[str, Quantity | str], ...], self.actuators)

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
                                {LINK_WRENCH.name: Port("link_wrench"), LINK_MOMENTUM.name: Port("link_momentum"),
                                 LINK_INERTIA.name: Port("link_inertia")}, frozenset(), DYNAMIC_LIMITS)

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
            found += _keyed_problems(self.armature, "armature", rotor=False)
            found += _keyed_problems(self.actuators, "actuators", rotor=True)
        else:
            if "initial_velocities" in settings:
                found.append("initial_velocities is the dynamic rung's: the kinematic arm's joints move as commanded")
            given = [name for name in ("armature", "actuators") if getattr(self, name)]
            if given:
                found.append(f"{' and '.join(given)} {'is' if len(given) == 1 else 'are'} the dynamic rung's: the "
                             f"kinematic arm's joints move as commanded, whatever their inertia")
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
                           "use the file the experiment was written against, or pin the new one: asset(path, "
                           "sha256=...)")
        try:
            return read_urdf(description.path, sha256=found, acknowledged=dict(self.acknowledged))
        except DescriptionRefused as refused:
            return Problem("description.refused", str(refused),
                           "edit the file, or acknowledge each element named with its reason: acknowledged=((path, "
                           "reason), ...), which the saved form records")
        except InvalidDescription as error:
            return Problem("description.invalid", f"{description.path}: {error}",
                           "correct the description file, then pass it again: Arm.Kinematic(description=asset(...), "
                           "...)")

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
                                                   "register your own chain segment publishing the same frame poses "
                                                   "(mime.registry.register_dynamics(...))"),))
        dofs = tree.articulation.dofs
        dynamic = self.rung == "dynamic"
        gaps = [gap.link for gap in reading.classification.inertia_gaps if gap.needed_by_dynamics]
        if dynamic and gaps:
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.inertia-gap", f"{tree.name}'s moving bodies with geometry and no <inertial> ({', '.join(gaps)}) "
                                     f"have no mass the dynamic arm can move: URDF reads them as massless, and MIME "
                                     f"infers no mass from geometry",
                "add an <inertial> to each, from CAD or a measurement, use the kinematic rung, Arm.Kinematic(...), or "
                "ask "
                f"for inference from geometry at {ISSUES}"),))
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
                "give initial_positions= one finite position per moving joint, in that order"),))
        rates = _components(settings["initial_velocities"]) if "initial_velocities" in settings else (0.0,) * dofs
        if len(rates) != dofs or not all(math.isfinite(c) for c in rates):
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.initial", f"initial_velocities has {len(rates)} values, and {tree.name} has {dofs} moving "
                                 f"joints ({', '.join(tree.articulation.dof_names)})",
                "give initial_velocities= one finite rate per moving joint, in that order"),))
        articulation = tree.articulation
        beyond = [f"{name} = {position:g} (limits {low:g} to {high:g})" for name, position, low, high in
                  zip(articulation.dof_names, initial, np.asarray(articulation.lower), np.asarray(articulation.upper),
                      strict=True) if not low <= position <= high]
        if beyond:
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.initial", f"initial_positions lies beyond a joint's limits: {', '.join(beyond)}; the arm holds "
                                 f"every position within them, so it would jump at its first step",
                "start every joint within its limits: initial_positions=rad((...))"),))
        armed = _armature(tree, self.declared_armature(), self.given_actuators()) if dynamic else None
        if armed is not None and armed.problems:
            return Segment(tree.frame_names, description=reading.classification, problems=armed.problems)
        classification = reading.using(armed.used) if armed is not None else reading.classification
        held = with_armatures(tree, armed.values) if armed is not None else tree
        with cast(Any, jax).enable_x64(True):
            frames = np.asarray(forward_kinematics(tree, jnp.asarray(initial, dtype=jnp.float64)), dtype=np.float64)
            turning = np.asarray(frame_motions(tree, jnp.asarray(initial, dtype=jnp.float64),
                                               jnp.asarray(rates, dtype=jnp.float64))[1], dtype=np.float64)
            inertia = np.diagonal(np.asarray(mass_matrix(held, jnp.asarray(initial, dtype=jnp.float64)))) \
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
                f"the kinematic rung, Arm.Kinematic(...)"),))
        revolute = sum(articulation.revolute)
        prismatic = dofs - revolute
        kinds = (("joint_torques", TORQUE, revolute), ("joint_forces", FORCE, prismatic)) if dynamic else \
            (("joint_positions", DIMENSIONLESS, revolute), ("joint_displacements", LENGTH, prismatic))
        commands = {port: Command(port, dimension, (count,)) for port, dimension, count in kinds if count}
        massive = any(body.inertial is not None and body.inertial.mass > 0.0 for body in tree.bodies)
        return Segment(tree.frame_names, commands, classification,
                       initial=tuple(tuple(float(c) for c in row) for row in frames),
                       initial_rates=tuple((float(r[0]), float(r[1]), float(r[2])) for r in turning),
                       massive=massive and not dynamic, initial_joints=tuple(float(c) for c in initial),
                       notes=armed.lines if armed is not None else transmission_lines(tree),
                       visuals=segment_visuals(reading, frames))

    def armed(self, tree: MultibodyTree) -> MultibodyTree:
        """``tree`` with each joint's armature on it, as phase 1 resolved it (its problems refused there)."""
        return with_armatures(tree, _armature(tree, self.declared_armature(), self.given_actuators()).values)

    @override
    def load_refusal(self) -> str | None:
        if self.rung == "kinematic":
            return ("the kinematic arm's joint positions are prescribed by its command, so a load would not move it; "
                    "the dynamic rung takes loads")
        return None

    @override
    def carries_mass(self) -> bool:
        return self.rung == "dynamic"

    def quietly_read(self) -> DescriptionReading | Problem:
        """The description read again, its warnings given once already, where it was first read."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DescriptionWarning)
            return self.read()

    @override
    def riding(self) -> tuple[str, ...]:
        reading = self.quietly_read()
        if isinstance(reading, Problem):
            return ()
        articulation = reading.tree.articulation
        return tuple(name for name, body in zip(articulation.frame_names, articulation.frame_bodies, strict=True)
                     if body == ROOT)

    @override
    def reads_mount_rates(self, settings: Mapping[str, Magnitude], published: bool) -> bool:
        return published

    @override
    def carried_model(self, mounted: Sequence[Payload]) -> MultibodyTree | None:
        reading = self.quietly_read()
        if isinstance(reading, Problem):
            return None
        return graft(self.armed(reading.tree) if self.rung == "dynamic" else reading.tree, tuple(mounted))

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        reading = self.read()
        if isinstance(reading, Problem):
            raise ValueError("phase 1 refuses an arm whose description it cannot read")
        if self.rung == "dynamic":
            tree = reading.tree
            names = tree.articulation.dof_names
            carried = articulate(graft(self.armed(tree), ctx.mounted), frames=tree.frame_names)
            b.add_node(DynamicArmNode(
                ctx.node_name(), ctx.timestep,
                **DynamicArmNode.described(carried, [tree.joint(name).damping for name in names],
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
            initial_mount_rate=mount.initial_rate if mount else (0.0, 0.0, 0.0),
            initial_joint_positions=_components(ctx.settings["initial_positions"]),
            servo_time_constant=ctx.scalar("servo") if "servo" in ctx.settings else None, rates=ctx.rates_read))
        if mount is not None:
            b.mount(into=node["mount_poses"])
            if ctx.rates_read:
                b.mount_rates(into=node["mount_rates"])


#: A joint's inertia about its axis no larger than this share of the arm's largest is rounding of none (64 units in the
#: last place of float64, in which the build computes it).
_ROUNDING = 64 * float(np.finfo(np.float64).eps)


def _reasoned(pair: object) -> bool:
    """Whether ``pair`` is an element's path and the reason it is read past, two strings."""
    held = cast(tuple[object, ...], pair) if isinstance(pair, tuple) else ()
    return len(held) == 2 and all(isinstance(item, str) for item in held)


#: A bone's radius, the stand-in drawn from a link's origin to each child's where the stage has no visual of it, as a
#: fraction of the segment's reach at its initial positions (the farthest frame from its base): a presentation choice,
#: thin beside the link it stands for and wide enough to see.
BONE_RADIUS_FRACTION = 0.02
_BONE_COLOUR = (0.55, 0.55, 0.6)


def _pose7(transform: Any) -> tuple[float, float, float, float, float, float, float]:
    t, q = transform.translation, transform.rotation
    return (float(t[0]), float(t[1]), float(t[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def segment_visuals(reading: DescriptionReading, frames: np.ndarray) -> tuple[SegmentVisual, ...]:
    """What the stage draws of a description, frame by frame: every link's visual (a mesh, a box, a cylinder or a
    sphere); where a link's visuals are meshes, its collision primitives; and, where any visual is a mesh or a link has
    none, a bone from each body's origin to each child joint's that is offset from it, so that the chain shows
    whatever the stage can resolve. ``frames`` are the frames' poses at the initial positions, for the bones'
    radius."""
    tree = reading.tree
    found: list[SegmentVisual] = []
    meshed: set[str] = set()
    for shape in reading.shapes:
        if shape.role == "visual":
            found.append(SegmentVisual(shape.link, shape.kind, "visual", _pose7(shape.origin), shape.mesh,
                                       shape.scale, shape.size, shape.radius, shape.length, colour=shape.colour))
            if shape.kind == "mesh":
                meshed.add(shape.link)
    for shape in reading.shapes:
        if shape.role == "collision" and shape.kind != "mesh" and shape.link in meshed:
            found.append(SegmentVisual(shape.link, shape.kind, "collision", _pose7(shape.origin), None, shape.scale,
                                       shape.size, shape.radius, shape.length, colour=_BONE_COLOUR))
    seen = {shape.link for shape in reading.shapes if shape.role == "visual"}
    if meshed or any(body.name not in seen for body in tree.bodies):
        held = np.asarray(frames, dtype=np.float64)
        reach = float(np.max(np.linalg.norm(held[:, :3], axis=1))) if len(held) else 0.0
        radius = max(reach * BONE_RADIUS_FRACTION, 1e-6)
        for joint in tree.joints:
            end = tuple(float(c) for c in joint.origin.translation)
            if math.hypot(*end) > radius:
                found.append(SegmentVisual(joint.parent, "bone", "frames", radius=radius,
                                           end=(end[0], end[1], end[2]), colour=_BONE_COLOUR))
    return tuple(found)


def _components(value: Magnitude) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


def _keyed(value: object) -> object:
    """A setting keyed by name (``{"joint_6": kg_m2(...)}``) as the pairs it is held and saved as, in the names'
    order; anything else as it is, for :meth:`Arm.setting_problems` to refuse."""
    if isinstance(value, Mapping):
        pairs = cast(Mapping[object, object], value).items()
    elif isinstance(value, (tuple, list)):
        pairs = cast(Sequence[object], value)
    else:
        return value
    held: list[object] = [tuple(cast(Sequence[object], pair)) if isinstance(pair, (tuple, list)) else pair
                          for pair in pairs]
    named = [cast(tuple[object, ...], pair) for pair in held if isinstance(pair, tuple)]
    if len(named) == len(held) and all(len(pair) == 2 and isinstance(pair[0], str) for pair in named):
        return tuple(sorted(cast(list[tuple[str, object]], named), key=lambda pair: pair[0]))
    return tuple(held)


def _keyed_problems(value: object, name: str, *, rotor: bool) -> list[str]:
    """What is wrong with a setting keyed by name as written: each entry a name and a quantity of zero or more (a
    rotor's above zero, kg m^2), or for an actuator a registered actuator's name."""
    example = ('actuators={"motor_6": kg_m2(8.2e-6, reason=...)}' if rotor else
               'armature={"joint_6": kg_m2(1e-4, reason=...)}')
    pairs = cast(tuple[object, ...], value) if isinstance(value, tuple) else (value,)
    found: list[str] = []
    for pair in pairs:
        if not (isinstance(pair, tuple) and len(cast(tuple[object, ...], pair)) == 2):
            found.append(f"{name} holds {pair!r}, not a name and its value: write {example}")
            continue
        key, held = cast(tuple[object, object], pair)
        if not isinstance(key, str):
            found.append(f"{name} is keyed by {key!r}, not a name")
        elif rotor and isinstance(held, str):
            try:
                actuator(held)
            except KeyError as error:
                found.append(f"{name}[{key!r}]: {error.args[0]}")
        elif not isinstance(held, Quantity) or not isinstance(held.value, float) or not math.isfinite(held.value) \
                or held.value < 0.0 or (rotor and held.value == 0.0):
            what = "a rotor's moment of inertia above zero, kg_m2(...), or a registered actuator's name" if rotor \
                else "a joint's armature of zero or more, kg_m2(...) on a revolute joint and kg(...) on a prismatic one"
            found.append(f"{name}[{key!r}] is {held!r}, not {what}")
        elif rotor and held.dimension != MOMENT_OF_INERTIA:
            found.append(f"{name}[{key!r}] is {held}, not a rotor's moment of inertia: kg_m2(...)")
    return found


@dataclass(frozen=True)
class _Armature:
    """Each joint's armature as an arm's settings and description give it (``values``, SI), the build report's lines
    for it, each problem found, with its action, and the transmissions it uses, by name."""

    values: dict[str, float]
    lines: tuple[str, ...]
    problems: tuple[Problem, ...]
    used: tuple[str, ...] = ()


def _armature(tree: MultibodyTree, declared: Sequence[tuple[str, Quantity]],
              given: Sequence[tuple[str, Quantity | str]]) -> _Armature:
    """Each joint's armature: declared on it, or reflected through its simple transmission from the rotor its actuator
    is given; refused where a joint or an actuator is named twice, a joint is named that the tree does not move, an
    armature is of another dimension than its joint's kind takes, an actuator is named that no simple transmission
    names, the transmission it names has anything wrong with it, or a joint is given one both ways. A transmission
    whose actuator is not given is not used, and nothing wrong with it is refused: the classification names it, with
    what is wrong with it."""
    kinds = {j.name: j.kind for j in tree.joints if j.moves}
    driving: dict[str, list[Transmission]] = {}
    for held in tree.transmissions:
        driving.setdefault(held.actuator, []).append(held)
    values: dict[str, float] = {}
    lines: list[str] = []
    problems: list[Problem] = []
    used: list[str] = []
    for setting, names in (("armature", [joint for joint, _ in declared]), ("actuators", [name for name, _ in given])):
        twice = sorted({name for name in names if names.count(name) > 1})
        for name in twice:
            problems.append(Problem(
                "chain.armature", f"{setting} names {name!r} {names.count(name)} times, so it gives "
                                  f"{'its joint' if setting == 'actuators' else name} more than one armature",
                f"give {name!r} once in Arm.Dynamic({setting}=...)"))
        if twice:
            return _Armature({}, (), tuple(problems))
    for joint, held in declared:
        if joint not in kinds:
            problems.append(Problem(
                "chain.armature", f"armature names {joint!r}, which is not a moving joint of {tree.name}; its moving "
                                  f"joints are {', '.join(kinds)}",
                "key each armature by a moving joint's name: Arm.Dynamic(armature={\"<joint>\": kg_m2(...)}, ...)"))
            continue
        revolute = kinds[joint] is JointKind.REVOLUTE
        wanted = MOMENT_OF_INERTIA if revolute else MASS
        if held.dimension != wanted:
            problems.append(Problem(
                "chain.armature", f"armature[{joint!r}] is {held}, and {joint} is a {kinds[joint].value} joint, whose "
                                  f"armature is {'a moment of inertia, kg m^2' if revolute else 'a mass, kg'}",
                f"write armature={{{joint!r}: {'kg_m2' if revolute else 'kg'}(...)}} in Arm.Dynamic(...)"))
            continue
        values[joint] = cast(float, held.value)
        reason = f" ({basis_text(held.reason)})" if held.reason is not None else ", its source not given"
        lines.append(f"armature of {joint}: {values[joint]:.6g} {unit_symbol(wanted)}, declared{reason}")
    for name, rotor in given:
        found = driving.get(name, [])
        if not found:
            named = ", ".join(sorted(driving)) or "none"
            problems.append(Problem(
                "chain.armature", f"actuators names {name!r}, and no simple transmission of {tree.name}'s description "
                                  f"names that actuator; the actuators its simple transmissions name: {named}",
                "give each actuator by the name its transmission gives it, in Arm.Dynamic(actuators={...}), or declare "
                "the joint's armature= itself"))
            continue
        transmission = found[0]
        faults = tree.transmission_faults(transmission)
        if faults:
            problems.append(Problem(
                "chain.armature", f"actuators names {name!r}, whose transmission {transmission.name!r} in "
                                  f"{tree.name}'s description cannot reflect its rotor onto a joint: "
                                  f"{'; '.join(faults)}",
                "correct the transmission in the description, or leave its actuator out of actuators= and declare "
                "the joint's armature= itself"))
            continue
        joint = transmission.joint
        if joint in values:
            problems.append(Problem(
                "chain.armature", f"{joint}'s armature is given twice: declared (armature={{{joint!r}: ...}}) and "
                                  f"reflected through transmission {transmission.name!r} from actuator {name!r} "
                                  f"(actuators={{{name!r}: ...}})",
                "keep one of the two in Arm.Dynamic(...): armature= or actuators="))
            continue
        if isinstance(rotor, str):
            record = actuator(rotor)
            inertia = record.inertia
            source = f"registered actuator {record.name}: {basis_text(cast(Any, record.rotor_inertia.reason))}"
        else:
            inertia = cast(float, rotor.value)
            source = basis_text(rotor.reason) if rotor.reason is not None else "its source not given"
        values[joint] = transmission.reflected(inertia)
        used.append(transmission.name)
        unit = "kg m^2" if kinds[joint] is JointKind.REVOLUTE else "kg"
        lines.append(f"armature of {joint}: {values[joint]:.6g} {unit}, reflected through transmission "
                     f"{transmission.name!r} (reduction {transmission.reduction:g}, squared) from actuator {name!r}'s "
                     f"rotor, {inertia:.6g} kg m^2 ({source})")
    for transmission in tree.transmissions:
        if transmission.name not in used:
            lines.append(f"transmission {transmission.name!r} (on {transmission.joint or 'no joint'}): recognised and "
                         f"not used, its actuator {transmission.actuator!r}'s rotor not given (actuators=), so it "
                         f"reflects nothing{_faults_text(tree, transmission)}")
    bare = [joint for joint in kinds if joint not in values]
    if bare:
        lines.append(f"no armature on {', '.join(bare)}: the URDF's own meaning, a convention (a geared drive's rotor "
                     f"reflected onto a joint is declared by armature=, or given through its transmission by "
                     f"actuators=)")
    return _Armature(values, tuple(lines), tuple(problems), tuple(used))


def _faults_text(tree: MultibodyTree, transmission: Transmission) -> str:
    faults = tree.transmission_faults(transmission)
    return f"; were it used, it would be refused: {'; '.join(faults)}" if faults else ""


def transmission_lines(tree: MultibodyTree) -> tuple[str, ...]:
    """The kinematic rung's lines for its description's simple transmissions: each recognised and not used, with what
    is wrong with it, since the kinematic arm's joints move as commanded."""
    return tuple(f"transmission {t.name!r} (on {t.joint or 'no joint'}): recognised and not used, the kinematic arm's "
                 f"joints moving as commanded{_faults_text(tree, t)}" for t in tree.transmissions)


__all__ = ["Arm"]
