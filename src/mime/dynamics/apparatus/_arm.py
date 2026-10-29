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
    forward_kinematics, read_urdf
from mime.gradients import GradientStatus
from mime.nodes.actuation import KinematicArmNode
from mime.registry import BuildContext, ChainSegment, Command, GradientRecord, Port, Problem, Segment, register_dynamics
from mime.scene import Asset
from mime.units import DIMENSIONLESS, Magnitude, Quantity, Time
from mime.vocabulary import FRAME_POSES, JOINT_POSITIONS, Capability

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration

#: Where to ask for what MIME does not do.
ISSUES = "https://github.com/Microrobotics-Simulation-Framework/MIME/issues"
_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


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
    load: a law that would load it is refused, since a load cannot move a prescribed motion; the ``dynamic`` rung
    (joint torques in, its inertia, gravity, damping and friction, and loads on its links) is reserved for that.

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
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="arm.limit.low",
            indicator="a bounded joint's commanded position less its lower limit, rad (m on a prismatic joint)",
            orders={"joint_positions": 0, "frame_poses": 0},
            route="none: a commanded position below the limit is held at the limit, whose derivative in the command "
                  "is zero; a smooth stop would be another model of the arm"),
        switch.Declaration(
            owner="arm.limit.high",
            indicator="a bounded joint's upper limit less its commanded position, rad (m on a prismatic joint)",
            orders={"joint_positions": 0, "frame_poses": 0},
            route="none: a commanded position above the limit is held at the limit, as at the lower one"),
    ))
    references = ("Sciavicco2000", "Featherstone2008")
    builds = {"kinematic": (KinematicArmNode,)}
    rungs = ("kinematic", "dynamic")
    reserved_rungs = {"dynamic": "the dynamic arm (joint torques in; its inertia, gravity, joint damping and friction "
                                 "from the description; loads on its links) is reserved, not built"}
    flags = frozenset({Capability.RIGID})
    publishes = MappingProxyType({FRAME_POSES.name: Port("frame_poses"),
                                  JOINT_POSITIONS.name: Port("joint_positions")})
    limits = ("rigid joints and links, with no joint flexibility, backlash or structural mode, and no contact "
              "between links (MIME-ANO-101)",
              "a prescribed motion: a load on it is not applied, the reaction of a field on a magnet mounted on it "
              "and a mounted motor's reaction torque among them (MIME-ANO-103)")
    rung: str | None = None
    description: Asset
    initial_positions: Annotated[Quantity, DIMENSIONLESS]
    servo: Annotated[Quantity | None, Time] = None
    acknowledged: tuple[tuple[str, str], ...] = ()

    @classmethod
    def Kinematic(cls, **settings: Any) -> Arm:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="kinematic", **settings)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        if not isinstance(cast(object, self.description), Asset):
            found.append(f"description={self.description!r} is not a file asset: write description=asset(\"arm.urdf\")"
                         f" with asset from mime.scene")
        servo = settings.get("servo")
        if servo is not None and not (isinstance(servo, float) and math.isfinite(servo) and servo > 0.0):
            found.append(f"servo is {servo!r} s, not a time constant above zero")
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
        initial = _components(settings["initial_positions"])
        if len(initial) != dofs or not all(math.isfinite(c) for c in initial):
            return Segment(tree.frame_names, description=reading.classification, problems=(Problem(
                "chain.initial", f"initial_positions has {len(initial)} values, and {tree.name} has {dofs} moving "
                                 f"joints ({', '.join(tree.articulation.dof_names)})",
                "give one finite position per moving joint, in that order"),))
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
        return Segment(tree.frame_names, {"joint_positions": Command("joint_positions", DIMENSIONLESS, (dofs,))},
                       reading.classification, initial=tuple(tuple(float(c) for c in row) for row in frames))

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


def _reasoned(pair: object) -> bool:
    """Whether ``pair`` is an element's path and the reason it is read past, two strings."""
    held = cast(tuple[object, ...], pair) if isinstance(pair, tuple) else ()
    return len(held) == 2 and all(isinstance(item, str) for item in held)


def _components(value: Magnitude) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


__all__ = ["Arm"]
