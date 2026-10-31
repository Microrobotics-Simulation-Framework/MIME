"""The canonical form of a multibody: one tree of bodies joined by joints, with named frames to mount parts on and loop
closures held as data. Description readers build it; the algorithms read the articulation compiled from it.

Every number here is a Python float in SI (metres, kilograms, radians, seconds), so a tree is immutable, hashable and
compares by value, and the saved form can record it."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mime.core.multibody._articulation import Articulation

type Vector3 = tuple[float, float, float]
type Quaternion4 = tuple[float, float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]

#: Rounding allowed in a description's inertia, relative to its largest principal moment: a symmetric tensor whose
#: smallest moment is above minus this much of its largest is positive semidefinite to within its own rounding.
_INERTIA_ROUNDING = 64 * float(np.finfo(np.float64).eps)


class InvalidDescription(ValueError):
    """A description that does not make a tree: a name used twice, a body with two parents, a joint naming a body that
    is not there, a joint with no axis, an inertia no body can have."""


class ClosedChainRefused(ValueError):
    """An algorithm asked of a tree that holds a loop closure. Every algorithm here is for open trees.

    Closed-chain (parallel) mechanisms such as a delta robot are out of scope until a downstream user needs one: if you
    have that use case, open an issue at https://github.com/Microrobotics-Simulation-Framework/MIME/issues. Until then,
    a lab can register its own dynamics entry for a chain segment (``mime.registry.register_dynamics``) that takes its
    own description of the mechanism and publishes the same frame poses as MIME's chains, so that mounts and everything
    downstream work unchanged."""

    def __init__(self, algorithm: str, closures: tuple[str, ...]) -> None:
        self.algorithm, self.closures = algorithm, closures
        named = ", ".join(repr(c) for c in closures)
        plural = "s" if len(closures) > 1 else ""
        super().__init__(f"{algorithm} is for open trees, and this tree holds the loop closure{plural} {named}. "
                         "Closed chains are out of scope until someone needs one: ask at "
                         "https://github.com/Microrobotics-Simulation-Framework/MIME/issues, or register your own "
                         "chain segment's dynamics (mime.registry.register_dynamics) publishing the same frame poses.")


def _vector(value: tuple[float, ...], what: str) -> Vector3:
    if len(value) != 3 or not all(math.isfinite(v) for v in value):
        raise InvalidDescription(f"{what} must be three finite numbers, not {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class RigidTransform:
    """A frame placed in another: ``x_parent = R x_frame + translation``, ``R`` the rotation of the unit quaternion
    ``rotation`` (wxyz). The quaternion is normalised on construction."""

    translation: Vector3 = (0.0, 0.0, 0.0)
    rotation: Quaternion4 = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _vector(self.translation, "a translation"))
        if len(self.rotation) != 4 or not all(math.isfinite(v) for v in self.rotation):
            raise InvalidDescription(f"a rotation must be four finite numbers (w, x, y, z), not {self.rotation!r}")
        norm = math.sqrt(sum(v * v for v in self.rotation))
        if norm == 0.0:
            raise InvalidDescription("a rotation quaternion must not be zero")
        w, x, y, z = (float(v) / norm for v in self.rotation)
        object.__setattr__(self, "rotation", (w, x, y, z))

    def matrix(self) -> np.ndarray:
        return rotation_matrix(self.rotation)

    def then(self, inner: RigidTransform) -> RigidTransform:
        """``inner`` placed in this transform's frame, expressed in this transform's parent."""
        rotation = self.matrix()
        translation = rotation @ np.asarray(inner.translation) + np.asarray(self.translation)
        return RigidTransform(as_vector3(translation), _product(self.rotation, inner.rotation))


IDENTITY = RigidTransform()


def rotation_matrix(q: Quaternion4) -> np.ndarray:
    """The rotation matrix of a unit quaternion (wxyz), in float64: for a description's constants."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _product(p: Quaternion4, q: Quaternion4) -> Quaternion4:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def as_vector3(v: np.ndarray) -> Vector3:
    return (float(v[0]), float(v[1]), float(v[2]))


def as_matrix3(m: np.ndarray) -> Matrix3:
    return (as_vector3(m[0]), as_vector3(m[1]), as_vector3(m[2]))


@dataclass(frozen=True)
class Inertial:
    """A body's mass, its centre of mass in the body's frame, and its inertia tensor about the centre of mass in the
    body frame's axes. A description that gives the tensor in a rotated frame is rotated into the body's on reading."""

    mass: float
    centre_of_mass: Vector3 = (0.0, 0.0, 0.0)
    inertia: Matrix3 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mass) and self.mass >= 0.0):
            raise InvalidDescription(f"a mass must be finite and not negative, not {self.mass!r}")
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "centre_of_mass", _vector(self.centre_of_mass, "a centre of mass"))
        tensor = np.asarray(self.inertia, dtype=np.float64)
        if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
            raise InvalidDescription(f"an inertia tensor must be a finite 3 x 3 matrix, not {self.inertia!r}")
        scale = float(np.max(np.abs(tensor)))
        if float(np.max(np.abs(tensor - tensor.T))) > _INERTIA_ROUNDING * scale:
            raise InvalidDescription(f"an inertia tensor must be symmetric, not {self.inertia!r}")
        moments = np.linalg.eigvalsh(0.5 * (tensor + tensor.T))
        if moments[0] < -_INERTIA_ROUNDING * scale:
            raise InvalidDescription(f"an inertia tensor's principal moments must not be negative; {self.inertia!r} "
                                     f"has {moments.tolist()}")
        object.__setattr__(self, "inertia", as_matrix3(0.5 * (tensor + tensor.T)))

    def placed(self, transform: RigidTransform) -> Inertial:
        """This inertial, of a frame placed by ``transform``, expressed in the parent frame."""
        rotation = transform.matrix()
        centre = rotation @ np.asarray(self.centre_of_mass) + np.asarray(transform.translation)
        return Inertial(self.mass, as_vector3(centre), as_matrix3(rotation @ np.asarray(self.inertia) @ rotation.T))


def combined(inertials: list[Inertial]) -> Inertial:
    """The inertial of rigidly joined parts, each already in one frame: masses add, the centre of mass is their
    mass-weighted mean, and each tensor moves to it by the parallel-axis theorem."""
    mass = sum(part.mass for part in inertials)
    if mass == 0.0:
        total = sum((np.asarray(part.inertia) for part in inertials), np.zeros((3, 3)))
        return Inertial(0.0, (0.0, 0.0, 0.0), as_matrix3(total))
    centre = sum((part.mass * np.asarray(part.centre_of_mass) for part in inertials), np.zeros(3)) / mass
    total = np.zeros((3, 3))
    for part in inertials:
        d = np.asarray(part.centre_of_mass) - centre
        total += np.asarray(part.inertia) + part.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return Inertial(mass, as_vector3(centre), as_matrix3(total))


@dataclass(frozen=True)
class Body:
    """A rigid body, with its own frame. ``inertial`` is ``None`` when its description declares none, which URDF reads
    as massless; a reader records where that body has geometry, since a dynamic chain may need its mass."""

    name: str
    inertial: Inertial | None = None


class JointKind(StrEnum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


@dataclass(frozen=True)
class JointLimits:
    """Position limits (radians or metres; infinite where the joint has none, as a continuous joint), and the velocity
    and effort limits as declared (``None`` where none is). An effort limit is stored as its description gives it,
    even where that is not positive (the AR4's file gives -1)."""

    lower: float = -math.inf
    upper: float = math.inf
    velocity: float | None = None
    effort: float | None = None

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise InvalidDescription(f"a joint's lower limit must not be above its upper one: {self.lower}, "
                                     f"{self.upper}")


@dataclass(frozen=True)
class Joint:
    """A joint from ``parent`` to ``child``. ``origin`` places the child's frame in the parent's at zero joint position;
    a revolute joint then turns the child's frame about ``axis`` through its origin, a prismatic joint slides it along
    ``axis``, both in the child's frame (normalised on construction), and a fixed joint welds it.

    ``damping`` (N m s/rad or N s/m), ``friction`` (N m or N, the Coulomb level) and ``armature`` (kg m^2 or kg, a
    rotor's inertia reflected through its gearing) are held for the dynamic chain. Of them, the algorithms here apply
    the armature, on the diagonal of the mass matrix and in inverse dynamics, as MuJoCo does; damping and friction are
    joint forces a dynamic chain applies itself."""

    name: str
    kind: JointKind
    parent: str
    child: str
    origin: RigidTransform = IDENTITY
    axis: Vector3 = (1.0, 0.0, 0.0)
    limits: JointLimits = field(default_factory=JointLimits)
    damping: float = 0.0
    friction: float = 0.0
    armature: float = 0.0

    def __post_init__(self) -> None:
        axis = np.asarray(_vector(self.axis, f"joint {self.name!r}'s axis"))
        length = float(np.linalg.norm(axis))
        if length == 0.0 and self.kind is not JointKind.FIXED:
            raise InvalidDescription(f"joint {self.name!r} is {self.kind.value} about a zero axis")
        object.__setattr__(self, "axis", as_vector3(axis / length) if length > 0.0 else (1.0, 0.0, 0.0))
        for what in ("damping", "friction", "armature"):
            value = getattr(self, what)
            if not (math.isfinite(value) and value >= 0.0):
                raise InvalidDescription(f"joint {self.name!r}'s {what} must be finite and not negative, not {value!r}")

    @property
    def moves(self) -> bool:
        return self.kind is not JointKind.FIXED


@dataclass(frozen=True)
class Frame:
    """A named frame fixed to a body, beyond the body's own: an MJCF site, or a mount point. ``offset`` places it in
    the body's frame."""

    name: str
    body: str
    offset: RigidTransform = IDENTITY


class ClosureKind(StrEnum):
    """What a loop closure holds together: ``CONNECT``, the two frames' origins (a ball joint); ``WELD``, the two frames
    whole."""

    CONNECT = "connect"
    WELD = "weld"


@dataclass(frozen=True)
class LoopClosure:
    """A constraint closing a loop in the tree, between two named frames: the data a closed-chain mechanism (a delta
    robot, a four-bar linkage) needs beyond its spanning tree. URDF cannot describe one; a tree built in code, or a
    later description format, can hold one.

    Closed chains are out of scope until a downstream user needs one, and every algorithm here refuses a tree that
    holds a closure, naming it (:class:`ClosedChainRefused` says what to do instead)."""

    name: str
    kind: ClosureKind
    first: str
    second: str


@dataclass(frozen=True)
class DescriptionSource:
    """The description file a tree was read from: its format, its file name, and the SHA-256 of its bytes, which pins
    it."""

    format: str
    file_name: str
    sha256: str


@dataclass(frozen=True)
class MultibodyTree:
    """A multibody as one tree: ``root`` is fixed to the world, every other body is the child of exactly one joint, and
    each body's frame and each extra frame is named, in one namespace, for parts to mount on.

    Everything is expressed in the root's frame: a chain mounted elsewhere composes its base pose with the poses here,
    and states gravity in the root's frame."""

    name: str
    root: str
    bodies: tuple[Body, ...]
    joints: tuple[Joint, ...]
    frames: tuple[Frame, ...] = ()
    closures: tuple[LoopClosure, ...] = ()
    source: DescriptionSource | None = None

    def __post_init__(self) -> None:
        for what in ("bodies", "joints", "frames", "closures"):
            object.__setattr__(self, what, tuple(getattr(self, what)))
        _unique("body or frame", [b.name for b in self.bodies] + [f.name for f in self.frames])
        _unique("joint", [j.name for j in self.joints])
        _unique("loop closure", [c.name for c in self.closures])
        bodies = {b.name for b in self.bodies}
        if self.root not in bodies:
            raise InvalidDescription(f"the root {self.root!r} is not one of the bodies")
        parent_joint: dict[str, str] = {}
        for joint in self.joints:
            for end in (joint.parent, joint.child):
                if end not in bodies:
                    raise InvalidDescription(f"joint {joint.name!r} names {end!r}, which is not one of the bodies")
            if joint.child in parent_joint:
                raise InvalidDescription(f"body {joint.child!r} is the child of two joints, "
                                         f"{parent_joint[joint.child]!r} and {joint.name!r}; a tree gives each body "
                                         "one parent (a loop is a closure)")
            parent_joint[joint.child] = joint.name
        if self.root in parent_joint:
            raise InvalidDescription(f"the root {self.root!r} is the child of joint {parent_joint[self.root]!r}")
        reached = {body for body, _ in self.walk()}
        unreached = [b.name for b in self.bodies if b.name not in reached]
        if unreached:
            raise InvalidDescription(f"bodies not connected to the root {self.root!r}: {unreached}")
        for frame in self.frames:
            if frame.body not in bodies:
                raise InvalidDescription(f"frame {frame.name!r} is fixed to {frame.body!r}, which is not one of the "
                                         "bodies")
        named = bodies | {f.name for f in self.frames}
        for closure in self.closures:
            for end in (closure.first, closure.second):
                if end not in named:
                    raise InvalidDescription(f"loop closure {closure.name!r} names {end!r}, which is not a frame")

    def walk(self) -> Iterator[tuple[str, Joint | None]]:
        """Every body reached from the root, depth first, with the joint into it (``None`` for the root); a body's
        children in the order their joints are listed."""
        children: dict[str, list[Joint]] = {}
        for joint in self.joints:
            children.setdefault(joint.parent, []).append(joint)
        stack: list[tuple[str, Joint | None]] = [(self.root, None)]
        while stack:
            body, joint = stack.pop()
            yield body, joint
            stack.extend((j.child, j) for j in reversed(children.get(body, [])))

    def body(self, name: str) -> Body:
        found = next((b for b in self.bodies if b.name == name), None)
        if found is None:
            raise KeyError(f"no body {name!r}; the bodies are {[b.name for b in self.bodies]}")
        return found

    def joint(self, name: str) -> Joint:
        found = next((j for j in self.joints if j.name == name), None)
        if found is None:
            raise KeyError(f"no joint {name!r}; the joints are {[j.name for j in self.joints]}")
        return found

    def welded(self) -> dict[str, str]:
        """For every body, the body it moves with: the child of the nearest joint that moves, at or above it, or the
        root for a body welded to the root."""
        moving_with: dict[str, str] = {}
        for body, joint in self.walk():
            moving_with[body] = body if joint is None or joint.moves else moving_with[joint.parent]
        return moving_with

    @property
    def frame_names(self) -> tuple[str, ...]:
        """Every named frame: the bodies' own, in the order listed, then the extra frames."""
        return tuple(b.name for b in self.bodies) + tuple(f.name for f in self.frames)

    @cached_property
    def articulation(self) -> Articulation:
        """The arrays the algorithms read, compiled once per tree."""
        from mime.core.multibody._articulation import articulate

        return articulate(self)


@dataclass(frozen=True)
class Payload:
    """A rigid body welded to a named frame of a tree, as a part mounted on a link is: ``name``, a body of its own in
    the tree's namespace; ``frame``, the frame it is fixed to; ``placement``, its own frame in that one; and its
    ``inertial``, in its own frame."""

    name: str
    frame: str
    placement: RigidTransform
    inertial: Inertial


def graft(tree: MultibodyTree, payloads: tuple[Payload, ...] | list[Payload]) -> MultibodyTree:
    """``tree`` with each payload welded on: a body of its own, the child of a fixed joint (named as the payload) from
    the body its frame is fixed to, at that frame's offset composed with the payload's placement. The link's own
    inertial is not edited: the articulation combines the two about their common centre of mass, as it combines every
    body welded to another. A jointed sub-tree grafted on a link (an articulated payload) is the same operation with
    moving joints, which no algorithm here takes yet."""
    if not payloads:
        return tree
    extra = {f.name: f for f in tree.frames}
    bodies, joints = list(tree.bodies), list(tree.joints)
    for payload in payloads:
        if payload.frame in extra:
            held = extra[payload.frame]
            body, offset = held.body, held.offset.then(payload.placement)
        else:
            tree.body(payload.frame)
            body, offset = payload.frame, payload.placement
        bodies.append(Body(payload.name, payload.inertial))
        joints.append(Joint(payload.name, JointKind.FIXED, body, payload.name, offset))
    return MultibodyTree(tree.name, tree.root, tuple(bodies), tuple(joints), tree.frames, tree.closures, tree.source)


@dataclass(frozen=True)
class RotorSplit:
    """A rotor's inertial split about its spin axis ``axis`` (a unit vector through its frame's origin): its moment
    about the axis ``axial`` (with its centroid's offset from the axis), the mean ``across`` of its two moments across
    the axis and half their difference ``asymmetry``, both about its centre of mass; how far its centre of mass lies
    off the axis (``off_axis``, m) and the size of its inertia's turn out of the axis (``tilt``, kg m^2: the part of
    ``I e`` across ``e``), each against its own rounding (``balanced``, ``aligned``, ``symmetric``)."""

    mass: float
    centre: Vector3
    axial: float
    across: float
    asymmetry: float
    moments: tuple[float, float]
    off_axis: float
    tilt: float
    balanced: bool
    aligned: bool
    symmetric: bool

    def carried(self, axis: Vector3) -> Inertial:
        """What the frame the rotor turns in carries of it: its mass at its centre of mass, and its mean moment across
        the axis, symmetric about it, with nothing about the axis itself (the spin carries that)."""
        e = np.asarray(axis)
        return Inertial(self.mass, self.centre, as_matrix3(self.across * (np.eye(3) - np.outer(e, e))))


#: The rounding a rotor's split allows, relative to its own scale (its size for its centre of mass, its largest
#: moment for its inertia): 64 units in the last place of float64, in which the build computes it.
ROTOR_ROUNDING = 64 * float(np.finfo(np.float64).eps)


def split_rotor(inertial: Inertial, axis: Vector3) -> RotorSplit:
    """``inertial`` (a rotor's, in a frame whose origin lies on its spin axis) split about the unit ``axis``."""
    e = np.asarray(axis, dtype=np.float64)
    e = e / float(np.linalg.norm(e))
    centre = np.asarray(inertial.centre_of_mass)
    tensor = np.asarray(inertial.inertia)
    radial = centre - float(centre @ e) * e
    off = float(np.linalg.norm(radial))
    largest = float(np.max(np.abs(np.linalg.eigvalsh(tensor)))) if np.any(tensor) else 0.0
    size = math.sqrt(largest / inertial.mass) if inertial.mass > 0.0 else 0.0
    turned = tensor @ e
    tilt = float(np.linalg.norm(turned - float(e @ turned) * e))
    plane = tensor - np.outer(turned, e) - np.outer(e, turned) + float(e @ turned) * np.outer(e, e)
    cross = np.sort(np.linalg.eigvalsh(plane))[1:]
    low, high = float(cross[0]), float(cross[1])
    scale = ROTOR_ROUNDING * max(largest, 0.0)
    return RotorSplit(inertial.mass, inertial.centre_of_mass, float(e @ turned) + inertial.mass * off * off,
                      0.5 * (low + high), 0.5 * (high - low), (high, low), off, tilt,
                      off <= ROTOR_ROUNDING * max(size, float(np.linalg.norm(centre))), tilt <= scale,
                      0.5 * (high - low) <= scale)


def _unique(what: str, names: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise InvalidDescription(f"the {what} name {name!r} is used twice")
        seen.add(name)
