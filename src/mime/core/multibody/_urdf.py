"""The URDF reader: a URDF file into the canonical tree, every element in it sorted into one of three groups.

- **Read and used**: links and their inertials; joints (revolute, continuous, prismatic, fixed) with their origins,
  parents, children, axes, limits, and dynamics (damping and friction).
- **Recognised and not used**: visuals, collisions, materials, transmissions, ``<gazebo>`` and ``<ros2_control>``
  extensions, and a joint's calibration and safety controller. MIME simulates none of them; each kind found is
  warned about once, saying what MIME does instead.
- **Refused**: anything that changes the mechanism if dropped. A mimic joint is refused: MIME's tree has no coupling
  between joints, and without its coupling the joint would move on its own. So is an element URDF does not define,
  since MIME cannot tell what dropping it would do. Both can be read past by acknowledging them, with a reason, which
  the classification records: the joint then moves freely, and the unknown element is dropped. Floating and planar
  joints, which the tree cannot hold, and unexpanded xacro macros are refused outright.

A link with geometry and no ``<inertial>`` is massless, as URDF reads it; MIME infers no mass from a mesh. Each such
link is recorded, and where the rigid body it moves with declares no inertial at all, a dynamic chain needs a mass the
file does not give, and refuses.

The file is pinned: the tree records the SHA-256 of its bytes, and a reading given the hash it expects refuses any
other file."""

from __future__ import annotations

import hashlib
import math
import warnings
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from mime.core.multibody._tree import Body, DescriptionSource, Inertial, InvalidDescription, Joint, JointKind, \
    JointLimits, MultibodyTree, RigidTransform, as_matrix3, rotation_matrix

ISSUES = "https://github.com/Microrobotics-Simulation-Framework/MIME/issues"


class ElementGroup(StrEnum):
    READ = "read and used"
    UNUSED = "recognised and not used"
    REFUSED = "refused"


@dataclass(frozen=True)
class DescriptionElement:
    """One element of a description: where it is (``joint[elbow]/mimic``: a tag, with its name where it has one, or its
    position among its siblings of that tag where there are several), what kind it is, and its group."""

    path: str
    category: str
    group: ElementGroup


@dataclass(frozen=True)
class InertiaGap:
    """A link with geometry and no ``<inertial>``. ``moves_with`` is the body it moves with (the child of its nearest
    moving joint, or the root); ``needed_by_dynamics`` is whether that rigid body moves and none of its links declares
    an inertial, so that a dynamic chain would need a mass the file does not give."""

    link: str
    moves_with: str
    needed_by_dynamics: bool


#: What MIME does instead of each element it does not use or refuses, where to look, and how to ask.
CATEGORY_MESSAGES: dict[str, str] = {
    "visual": "MIME simulates no link's appearance; the stage draws it on each link's frame (mime.viz): a box, "
              "cylinder or sphere as its USD primitive, a mesh from the asset cache where a visual set bakes it "
              "(`mime visuals fetch <set>`), and simple shapes from the link's frames where it has neither.",
    "collision": "MIME simulates no collision between a chain's links: contact is an interaction an experiment "
                 f"attaches between named things (mime.interactions.Contact). To ask for link collisions: {ISSUES}.",
    "material": "a URDF material is a colour, not a physical material: a part's material comes from an OpenMaterica "
                f"record or a value declared in the experiment (mime.materials). To ask for colours: {ISSUES}.",
    "transmission": "the dynamic chain takes joint forces, so no reduction stands between an actuator and its joint: "
                    f"scale commands by the gear ratio yourself, or ask for transmissions at {ISSUES}.",
    "gazebo": "these are Gazebo's own simulation settings; a MIME experiment declares its own laws and settings, and "
              f"no plugin runs. If one of them matters to your mechanism, ask at {ISSUES}.",
    "ros2_control": "ROS 2 control's hardware interface: a MIME controller declares what it reads and writes "
                    f"(mime.control). To ask for it to be read: {ISSUES}.",
    "calibration": "a real arm's homing references: MIME starts each joint where the experiment's initial state "
                   f"puts it. To ask for homing: {ISSUES}.",
    "safety_controller": "a real controller's soft limits: MIME keeps the joint's own <limit>, and a controller "
                         f"declares its own limits (mime.control). To ask for soft limits: {ISSUES}.",
    "mimic": "MIME's tree has no coupling between joints, so the joint would move on its own. Make it fixed or remove "
             "the mimic in the file, or acknowledge it with a reason to read the joint as independent, or ask for "
             f"coupled joints at {ISSUES}.",
    "unknown": "URDF does not define this element, and MIME cannot tell whether dropping it changes the mechanism. "
               "Remove it from the file, or acknowledge it with a reason to read past it, or ask for it to be "
               f"recognised at {ISSUES}.",
    "floating": "MIME's tree is fixed to its root and holds one-degree-of-freedom joints: a free-flying body is a body "
                f"with its own dynamics (mime.dynamics), not a joint of a chain. To ask for one: {ISSUES}.",
    "planar": "MIME's tree holds one-degree-of-freedom joints: describe a planar joint as two prismatic joints and a "
              f"revolute one between massless links, or ask at {ISSUES}.",
    "xacro": "the file holds xacro macros: expand them first (xacro robot.urdf.xacro > robot.urdf) and read the "
             "result.",
    "inertia gap": "URDF reads a link with no <inertial> as massless, and MIME infers no mass from geometry. A dynamic "
                   "chain refuses a moving body with geometry none of whose links declares an <inertial>: add one, "
                   f"from CAD or a measurement, or ask for inference from geometry at {ISSUES}.",
}

#: The refused kinds that can be read past on acknowledgement; the others the tree cannot hold.
_ACKNOWLEDGEABLE = frozenset({"mimic", "unknown"})

_READ, _UNUSED, _REFUSED = ElementGroup.READ, ElementGroup.UNUSED, ElementGroup.REFUSED
#: The elements URDF defines, by the tag of their parent, and the group of each.
_CHILDREN: dict[str, dict[str, ElementGroup]] = {
    "robot": {"link": _READ, "joint": _READ, "material": _UNUSED, "transmission": _UNUSED, "gazebo": _UNUSED,
              "ros2_control": _UNUSED},
    "link": {"inertial": _READ, "visual": _UNUSED, "collision": _UNUSED},
    "inertial": {"origin": _READ, "mass": _READ, "inertia": _READ},
    "joint": {"origin": _READ, "parent": _READ, "child": _READ, "axis": _READ, "limit": _READ, "dynamics": _READ,
              "calibration": _UNUSED, "safety_controller": _UNUSED, "mimic": _REFUSED},
}
_JOINT_KINDS = {"revolute": JointKind.REVOLUTE, "continuous": JointKind.REVOLUTE, "prismatic": JointKind.PRISMATIC,
                "fixed": JointKind.FIXED}


@dataclass(frozen=True)
class DescriptionClassification:
    """Every element of a description in its group, the links whose mass is a gap, and the refused elements read past
    on acknowledgement with their reasons. A build report and the saved form take it as it is."""

    source: DescriptionSource
    elements: tuple[DescriptionElement, ...]
    inertia_gaps: tuple[InertiaGap, ...] = ()
    acknowledged: tuple[tuple[str, str], ...] = ()

    def group(self, group: ElementGroup) -> tuple[DescriptionElement, ...]:
        return tuple(e for e in self.elements if e.group is group)

    def categories(self, group: ElementGroup) -> dict[str, tuple[str, ...]]:
        """The paths of the group's elements, by kind, in file order."""
        found: dict[str, list[str]] = {}
        for element in self.group(group):
            found.setdefault(element.category, []).append(element.path)
        return {category: tuple(paths) for category, paths in found.items()}

    def unacknowledged(self) -> tuple[DescriptionElement, ...]:
        """The refused elements not read past."""
        read_past = {path for path, _ in self.acknowledged}
        return tuple(e for e in self.group(ElementGroup.REFUSED) if e.path not in read_past)

    def warnings(self) -> list[str]:
        """One message per kind of element not used, kind of refused element read past, and for the links read as
        massless: what each warning says."""
        name, found = self.source.file_name, list[str]()
        for category, paths in self.categories(ElementGroup.UNUSED).items():
            found.append(f"{name}: {len(paths)} <{category}> element{_s(paths)} not used ({', '.join(paths)}). "
                         f"{CATEGORY_MESSAGES[category]}")
        reasons = dict(self.acknowledged)
        read_past: dict[str, list[str]] = {}
        for element in self.group(ElementGroup.REFUSED):
            if element.path in reasons:
                read_past.setdefault(element.category, []).append(f"{element.path} ({reasons[element.path]})")
        for category, paths in read_past.items():
            found.append(f"{name}: {len(paths)} refused <{category}> element{_s(paths)} read past as acknowledged: "
                         f"{', '.join(paths)}. {CATEGORY_MESSAGES[category]}")
        if self.inertia_gaps:
            links = ", ".join(g.link + (" (needed by a dynamic chain)" if g.needed_by_dynamics else "")
                              for g in self.inertia_gaps)
            found.append(f"{name}: {len(self.inertia_gaps)} link{_s(self.inertia_gaps)} with geometry and no "
                         f"<inertial>, read as massless: {links}. {CATEGORY_MESSAGES['inertia gap']}")
        return found


def _s(items: tuple[object, ...] | list[str]) -> str:
    return "" if len(items) == 1 else "s"


class DescriptionWarning(UserWarning):
    """An element of a description that MIME reads past: not simulated, or refused and acknowledged."""


class DescriptionRefused(ValueError):
    """A description holding elements that change the mechanism if dropped. ``classification`` holds all three
    groups; ``refused`` the elements refused."""

    def __init__(self, classification: DescriptionClassification) -> None:
        self.classification = classification
        self.refused = classification.unacknowledged()
        lines = [f"  {e.path}: {CATEGORY_MESSAGES[e.category]}" for e in self.refused]
        can = [e.path for e in self.refused if e.category in _ACKNOWLEDGEABLE]
        if can:
            lines.append(f"To read past {', '.join(can)}, pass acknowledged={{path: reason}}; the reason is recorded.")
        super().__init__(f"{classification.source.file_name} is refused: it holds {len(self.refused)} element"
                         f"{_s(self.refused)} that change{'s' if len(self.refused) == 1 else ''} the mechanism if "
                         "dropped.\n" + "\n".join(lines))


@dataclass(frozen=True)
class LinkShape:
    """A link's ``<visual>`` or ``<collision>`` as the file writes it, for the stage to draw (no physics reads it):
    its link, ``role`` (``visual`` or ``collision``), its origin in the link's frame, and its geometry: ``mesh`` (the
    file name as written, with ``scale``), ``box`` (``size``), ``cylinder`` (``radius``, ``length``) or ``sphere``
    (``radius``), in metres; and its colour (RGB, 0 to 1), its own material's or the robot's material it names."""

    link: str
    role: str
    origin: RigidTransform
    kind: str
    mesh: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    length: float | None = None
    colour: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class DescriptionReading:
    """A description read: its tree, the classification of every element in it, and its links' visual and collision
    shapes, which the stage draws (a shape the file writes malformed is left out: it is appearance, never physics)."""

    tree: MultibodyTree
    classification: DescriptionClassification
    shapes: tuple[LinkShape, ...] = ()


def read_urdf(path: str | Path, *, sha256: str | None = None,
              acknowledged: Mapping[str, str] | None = None) -> DescriptionReading:
    """Read the URDF file at ``path`` into a tree, with every element classified. Each kind of element not used is
    warned about once (:class:`DescriptionWarning`), and a file holding a refused element is refused
    (:class:`DescriptionRefused`) unless ``acknowledged`` maps its path to a reason. ``sha256``, when given, is the
    hash the file must have."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256.lower():
        raise InvalidDescription(f"{path} has SHA-256 {digest}, not the {sha256} it is pinned to")
    source = DescriptionSource("urdf", Path(path).name, digest)
    try:
        robot = ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise InvalidDescription(f"{path} is not well-formed XML: {error}") from error
    if robot.tag != "robot":
        raise InvalidDescription(f"{path}'s root element is <{robot.tag}>, not <robot>")

    elements: list[DescriptionElement] = []
    _classify(robot, "", elements)
    given = dict(acknowledged or {})
    refused = {e.path: e for e in elements if e.group is ElementGroup.REFUSED}
    for element_path, reason in given.items():
        if element_path not in refused:
            raise InvalidDescription(f"{element_path!r} is acknowledged, but {source.file_name} refuses no such "
                                     f"element; the refused elements are {sorted(refused)}")
        if refused[element_path].category not in _ACKNOWLEDGEABLE:
            raise InvalidDescription(f"{element_path} cannot be read past: "
                                     f"{CATEGORY_MESSAGES[refused[element_path].category]}")
        if not reason.strip():
            raise InvalidDescription(f"the acknowledgement of {element_path} gives no reason")
    classification = DescriptionClassification(source, tuple(elements),
                                               acknowledged=tuple((p, given[p]) for p in refused if p in given))
    if classification.unacknowledged():
        _warn(classification)
        raise DescriptionRefused(classification)

    tree = _tree(robot, source)
    classification = DescriptionClassification(source, tuple(elements), _inertia_gaps(robot, tree),
                                               classification.acknowledged)
    _warn(classification)
    return DescriptionReading(tree, classification, _shapes(robot))


def _colour(material: ElementTree.Element | None, named: Mapping[str, tuple[float, float, float]]) -> \
        tuple[float, float, float] | None:
    if material is None:
        return None
    color = material.find("color")
    if color is not None:
        try:
            rgba = _floats(color.get("rgba"), 4, (0.0, 0.0, 0.0, 1.0), "a material's colour")
        except InvalidDescription:
            return None
        return (min(max(rgba[0], 0.0), 1.0), min(max(rgba[1], 0.0), 1.0), min(max(rgba[2], 0.0), 1.0))
    return named.get(str(material.get("name")))


def _shape(link: str, role: str, element: ElementTree.Element,
           named: Mapping[str, tuple[float, float, float]]) -> LinkShape | None:
    """One ``<visual>`` or ``<collision>``, or ``None`` where it is malformed or holds no geometry the stage draws."""
    geometry = element.find("geometry")
    if geometry is None or len(geometry) != 1:
        return None
    held, where = geometry[0], f"link[{link}]/{role}"
    try:
        origin = _origin(element.find("origin"), where)
        colour = _colour(element.find("material"), named) if role == "visual" else None
        if held.tag == "mesh" and held.get("filename"):
            scale = _floats(held.get("scale"), 3, (1.0, 1.0, 1.0), f"{where} mesh scale")
            return LinkShape(link, role, origin, "mesh", mesh=str(held.get("filename")),
                             scale=(scale[0], scale[1], scale[2]), colour=colour)
        if held.tag == "box":
            size = _floats(held.get("size"), 3, (0.0, 0.0, 0.0), f"{where} box size")
            return LinkShape(link, role, origin, "box", size=(size[0], size[1], size[2]), colour=colour)
        if held.tag == "cylinder":
            return LinkShape(link, role, origin, "cylinder", radius=_number(held, "radius", 0.0, where),
                             length=_number(held, "length", 0.0, where), colour=colour)
        if held.tag == "sphere":
            return LinkShape(link, role, origin, "sphere", radius=_number(held, "radius", 0.0, where), colour=colour)
    except InvalidDescription:
        return None
    return None


def _shapes(robot: ElementTree.Element) -> tuple[LinkShape, ...]:
    """Every link's visuals, then its collisions, in file order; a robot-level material lends its colour by name."""
    named: dict[str, tuple[float, float, float]] = {}
    for material in robot.findall("material"):
        colour = _colour(material, {})
        if colour is not None and material.get("name"):
            named[str(material.get("name"))] = colour
    found: list[LinkShape] = []
    for link in robot.findall("link"):
        name = str(link.get("name"))
        for role in ("visual", "collision"):
            found += [shape for element in link.findall(role) if (shape := _shape(name, role, element, named))]
    return tuple(found)


def _warn(classification: DescriptionClassification) -> None:
    for message in classification.warnings():
        warnings.warn(message, DescriptionWarning, stacklevel=3)


def _element_paths(parent: ElementTree.Element, prefix: str) -> list[tuple[ElementTree.Element, str]]:
    counts: dict[str, int] = {}
    for child in parent:
        counts[child.tag] = counts.get(child.tag, 0) + 1
    seen: dict[str, int] = {}
    found: list[tuple[ElementTree.Element, str]] = []
    for child in parent:
        tag = _local(child.tag)
        if child.get("name") is not None:
            label = f"{tag}[{child.get('name')}]"
        elif counts[child.tag] > 1:
            label = f"{tag}[{seen.get(child.tag, 0)}]"
        else:
            label = tag
        seen[child.tag] = seen.get(child.tag, 0) + 1
        found.append((child, f"{prefix}/{label}" if prefix else label))
    return found


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _classify(parent: ElementTree.Element, prefix: str, elements: list[DescriptionElement]) -> None:
    defined = _CHILDREN.get(parent.tag, {})
    for child, path in _element_paths(parent, prefix):
        if child.tag.startswith("{") and "xacro" in child.tag:
            elements.append(DescriptionElement(path, "xacro", ElementGroup.REFUSED))
            continue
        group = defined.get(child.tag)
        if group is None:
            elements.append(DescriptionElement(path, "unknown", ElementGroup.REFUSED))
            continue
        category = child.tag
        if child.tag == "joint" and child.get("type") in ("floating", "planar"):
            category, group = str(child.get("type")), ElementGroup.REFUSED
        elements.append(DescriptionElement(path, category, group))
        if group is ElementGroup.READ:
            _classify(child, path, elements)


def _floats(text: str | None, count: int, default: tuple[float, ...], where: str) -> tuple[float, ...]:
    if text is None:
        return default
    try:
        values = tuple(float(v) for v in text.split())
    except ValueError as error:
        raise InvalidDescription(f"{where}: {text!r} is not {count} numbers") from error
    if len(values) != count:
        raise InvalidDescription(f"{where}: {text!r} is not {count} numbers")
    return values


def _optional_number(element: ElementTree.Element, attribute: str, where: str) -> float | None:
    text = element.get(attribute)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError as error:
        raise InvalidDescription(f"{where}: {attribute}={text!r} is not a number") from error


def _number(element: ElementTree.Element, attribute: str, default: float, where: str) -> float:
    found = _optional_number(element, attribute, where)
    return default if found is None else found


def _required(element: ElementTree.Element, attribute: str, where: str) -> str:
    value = element.get(attribute)
    if value is None:
        raise InvalidDescription(f"{where} has no {attribute!r}")
    return value


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF's ``rpy``: a roll about the fixed x axis, then a pitch about the fixed y, then a yaw about the fixed z, so
    ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr)


def _origin(element: ElementTree.Element | None, where: str) -> RigidTransform:
    if element is None:
        return RigidTransform()
    xyz = _floats(element.get("xyz"), 3, (0.0, 0.0, 0.0), f"{where}/origin xyz")
    rpy = _floats(element.get("rpy"), 3, (0.0, 0.0, 0.0), f"{where}/origin rpy")
    return RigidTransform((xyz[0], xyz[1], xyz[2]), _quaternion_from_rpy(rpy[0], rpy[1], rpy[2]))


def _inertial(element: ElementTree.Element, where: str) -> Inertial:
    mass, inertia = element.find("mass"), element.find("inertia")
    if mass is None or inertia is None:
        raise InvalidDescription(f"{where}/inertial needs both <mass> and <inertia>")
    origin = _origin(element.find("origin"), f"{where}/inertial")
    value = _optional_number(mass, "value", f"{where}/inertial/mass")
    if value is None:
        raise InvalidDescription(f"{where}/inertial/mass has no 'value'")
    moments = {k: _number(inertia, k, 0.0, f"{where}/inertial/inertia")
               for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    tensor = np.array([[moments["ixx"], moments["ixy"], moments["ixz"]],
                       [moments["ixy"], moments["iyy"], moments["iyz"]],
                       [moments["ixz"], moments["iyz"], moments["izz"]]])
    rotation = rotation_matrix(origin.rotation)
    try:
        return Inertial(value, origin.translation, as_matrix3(rotation @ tensor @ rotation.T))
    except InvalidDescription as error:
        raise InvalidDescription(f"{where}/inertial: {error}") from error


def _joint(element: ElementTree.Element) -> Joint:
    name = _required(element, "name", "a <joint>")
    where = f"joint[{name}]"
    kind_name = _required(element, "type", where)
    if kind_name not in _JOINT_KINDS:
        raise InvalidDescription(f"{where} has type {kind_name!r}; URDF's types are revolute, continuous, prismatic, "
                                 "fixed, floating and planar")
    kind = _JOINT_KINDS[kind_name]
    ends: list[str] = []
    for end in ("parent", "child"):
        found = element.find(end)
        if found is None:
            raise InvalidDescription(f"{where} has no <{end}>")
        ends.append(_required(found, "link", f"{where}/{end}"))
    axis_element = element.find("axis")
    axis = _floats(None if axis_element is None else axis_element.get("xyz"), 3, (1.0, 0.0, 0.0), f"{where}/axis")
    limit = element.find("limit")
    if limit is None and kind_name in ("revolute", "prismatic"):
        raise InvalidDescription(f"{where} is {kind_name} and has no <limit>, which URDF requires of it: give its "
                                 "lower, upper, velocity and effort limits")
    limits = JointLimits()
    if limit is not None:
        bounded = kind_name in ("revolute", "prismatic")
        limits = JointLimits(
            lower=_number(limit, "lower", 0.0, f"{where}/limit") if bounded else -math.inf,
            upper=_number(limit, "upper", 0.0, f"{where}/limit") if bounded else math.inf,
            velocity=_optional_number(limit, "velocity", f"{where}/limit"),
            effort=_optional_number(limit, "effort", f"{where}/limit"))
    dynamics = element.find("dynamics")
    damping = friction = 0.0
    if dynamics is not None:
        damping = _number(dynamics, "damping", 0.0, f"{where}/dynamics")
        friction = _number(dynamics, "friction", 0.0, f"{where}/dynamics")
    try:
        return Joint(name, kind, ends[0], ends[1], _origin(element.find("origin"), where), (axis[0], axis[1], axis[2]),
                     limits, damping=damping, friction=friction)
    except InvalidDescription as error:
        raise InvalidDescription(f"{where}: {error}") from error


def _tree(robot: ElementTree.Element, source: DescriptionSource) -> MultibodyTree:
    bodies: list[Body] = []
    for link in robot.findall("link"):
        name = _required(link, "name", "a <link>")
        inertial = link.find("inertial")
        bodies.append(Body(name, None if inertial is None else _inertial(inertial, f"link[{name}]")))
    joints = [_joint(joint) for joint in robot.findall("joint")]
    children = {j.child for j in joints}
    roots = [b.name for b in bodies if b.name not in children]
    if len(roots) != 1:
        raise InvalidDescription(f"a URDF tree has one root link, a link no joint has as its child; this one has "
                                 f"{roots or 'none'}")
    return MultibodyTree(_required(robot, "name", "<robot>"), roots[0], tuple(bodies), tuple(joints), source=source)


def _inertia_gaps(robot: ElementTree.Element, tree: MultibodyTree) -> tuple[InertiaGap, ...]:
    moves_with = tree.welded()
    declared = {moves_with[b.name] for b in tree.bodies if b.inertial is not None}
    gaps: list[InertiaGap] = []
    for link in robot.findall("link"):
        shaped = any(part.find("geometry") is not None for part in link if part.tag in ("visual", "collision"))
        if shaped and link.find("inertial") is None:
            name = str(link.get("name"))
            rigid = moves_with[name]
            gaps.append(InertiaGap(name, rigid, rigid != tree.root and rigid not in declared))
    return tuple(gaps)
