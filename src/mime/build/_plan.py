"""Phase 1: every declaration resolved and checked as data, with no graph, and every refusal collected."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

from mime import materials, vocabulary
from mime.build._control import UNBUILT, ClassedRead, ControllerPlan, Mode, ReadClass, ResolvedWrite, mode_of, \
    refused_reads
from mime.build._declarations import (
    Attachment,
    ControllerDeclaration,
    Declaration,
    ObservationDeclaration,
    RunSettings,
    ThingDeclaration,
    ValueDeclaration,
)
from mime.build._refusals import MIME_DEFECT, BuildRefused, Refusal, RefusalClass, catalogue_problem
from mime.build._report import BuildReport, OrientationLine, RunStructure
from mime.core import VELOCITY_RULES
from mime.core.multibody import RigidTransform
from mime.registry import (
    CapabilityRecord,
    ChainSegment,
    Command,
    Commanded,
    Confinement,
    Controller,
    Delivery,
    Direction,
    Drive,
    Dynamics,
    Entry,
    EntryKind,
    FieldProvider,
    Geometry,
    GroupRequest,
    Hull,
    Interaction,
    Observation,
    Plan,
    PlanContext,
    Port,
    Read,
    Region,
    Registered,
    Registration,
    RoleKind,
    Segment,
    Solid,
    Tensor,
    Value,
    implements,
    registered,
    registration_of,
    setting_value,
)
from mime.scene import (
    MAGNET_API,
    MAGNET_DIRECTION,
    ComposedLayer,
    Composition,
    Scene,
    WorldFrame,
    compose,
    meters_per_unit,
    scene_up,
    unit_vector,
    valid_part_path,
    valid_prim_path,
    world_frame,
)
from mime.things import (
    WORLD,
    Apparatus,
    Body,
    DirectionPath,
    FieldPath,
    GeometryPath,
    Handle,
    Medium,
    Part,
    PartMass,
    PermanentMagnet,
    RegionPath,
    Structure,
    SurfacePath,
    Thing,
    ThingKind,
    TypedPath,
    did_you_mean,
    prim_path_of,
)
from mime.units import FREQUENCY, LENGTH, TIME, Dimension, Magnitude, Quantity, UnitError, si, unit_symbol
from mime.vocabulary import FRAME_POSES, Capability, ReadKind, TargetKind

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOCAL = re.compile(r"[a-z_][a-z0-9_]*")
#: What a region's name in OpenMaterica's scene manifest (``<structure>_<region>``) may be.
MANIFEST_NAME = re.compile(r"[a-z0-9_]+")
_RATE_RTOL = 1e-9
#: How far a placement's quaternion's norm may be from one.
_UNIT = 1e-6
_DYNAMICS_ORDER = {ThingKind.MEDIUM: 0, ThingKind.BODY: 1, ThingKind.APPARATUS: 2, ThingKind.STRUCTURE: 3}
#: A body's values its hull's kind gives, so that declaring one beside it is refused: the extent (the hull's radial
#: reach), and the shape values a law reads from the kind's family parameters.
FROM_THE_KIND = ("extent", "radius", "length", "helix_wavenumber")


class Section(Enum):
    """The canonical order's sections, in the order their nodes are added."""

    PROVIDER = 0
    INTERACTION = 1
    DYNAMICS = 2
    OBSERVATION = 3
    DRIVE = 4
    CONTROLLER = 5


@dataclass(frozen=True)
class ResolvedTarget:
    """A target port of a built dynamics node, and who delivers into it (``by``). ``units`` are the vocabulary's,
    which every edge into the port carries as its label."""

    node: str
    field: str
    kind: TargetKind
    thing: str
    interface: str
    units: str
    by: str = ""


@dataclass(frozen=True)
class ResolvedRead:
    """Where a read comes from. A state read names the publishing node's field. A field read names the provider's
    output port for its reader slot, and ``points`` the state read its points come from. ``units`` are the
    vocabulary's, which the edge carrying the read is labelled with."""

    node: str
    field: str
    interface: str
    units: str
    points: ResolvedRead | None = None
    slot: str | None = None


@dataclass(frozen=True)
class ResolvedConfinement:
    """A confinement addressed to a body's dynamics: the confining entry's id, the confined body, the surface it
    binds (``tube.inner_surface``), the shape that surface's free side is evaluated with and its parameters in SI,
    the margin and the body's extent in metres, and the velocity rule."""

    id: str
    subject: str
    surface: str
    shape: str
    parameters: Mapping[str, Magnitude]
    margin: float
    extent: float
    rule: str


@dataclass(frozen=True)
class ResolvedMount:
    """A frame something is mounted on: the frame ``frame`` of the chain segment ``segment`` (its id,
    ``<apparatus>.<segment>``), whose node ``node`` publishes it at row ``index`` of its ``count`` frame poses, in
    ``field``; and its world pose at the start, ``(x, y, z, qw, qx, qy, qz)``."""

    segment: str
    frame: str
    node: str
    field: str
    index: int
    count: int
    initial_pose: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ResolvedPart:
    """A part of an apparatus as a provider reads it: its name, the frame it is mounted on (``None``: fixed in the
    world), its placement in that frame (a translation in metres and a unit quaternion ``(w, x, y, z)``, the part's
    frame not turned from its mount's), the radius of the smallest sphere about its centroid that holds its solid,
    and the solid's registered label."""

    apparatus: str
    name: str
    mount: ResolvedMount | None
    offset: tuple[float, float, float, float, float, float, float]
    radius: float
    solid: str


@dataclass(frozen=True)
class ResolvedCommand:
    """The input a drive writes: a field provider's node and port, the command's name, and its units."""

    node: str
    field: str
    command: str
    units: str


@dataclass(frozen=True)
class GeometryPlan:
    """A structure's geometry kind, resolved: its registration and its settings in SI."""

    structure: str
    geometry: Geometry
    registered: Registered
    settings: Mapping[str, Magnitude]


@dataclass(frozen=True)
class ResolvedMagnet:
    """A body's magnet part: its key in ``magnets``, its prim, the role's model, its magnetisation direction in the
    part's frame and in the body's, and where that direction came from."""

    body: str
    key: str
    part: str
    model: str
    in_part: tuple[float, float, float]
    in_body: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class PartLine:
    """A body's part as resolved: its solid's registration and settings in SI, its volume, its centroid in the
    body's frame and the rotation from its frame to the body's, the part it lies inside, its material's name, and
    whether it is a magnet. ``inertia_per_mass`` is about its centroid, in the body's frame."""

    body: str
    name: str
    solid: Solid
    registered: Registered
    settings: Mapping[str, Magnitude]
    volume: float
    centre: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    inertia_per_mass: Tensor
    inside: str | None
    material: str | None
    magnet: bool
    reach: float | None = None


@dataclass(frozen=True)
class MaterialsPlan:
    """The experiment's conditions, each record as read and pinned, every read the values used, each body's parts,
    and each medium's material by name, where it names a record."""

    conditions: tuple[materials.OperatingCondition, ...] = ()
    records: tuple[materials.LoadedRecord, ...] = ()
    reads: tuple[materials.ReadOutcome, ...] = ()
    parts: tuple[PartLine, ...] = ()
    media: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class ScenePlan:
    """The scene as the build resolved it: up (a unit vector), the stage's unit and metres per unit, each thing's
    prim path, the things bound to a prim an asset brings in, the referenced layers as read, and the magnets."""

    up: tuple[float, float, float]
    units: str
    meters_per_unit: float
    prims: Mapping[str, str]
    bound: frozenset[str]
    layers: tuple[ComposedLayer, ...]
    magnets: tuple[ResolvedMagnet, ...] = ()


@dataclass(frozen=True)
class Slot:
    """A reader slot of a field provider: ``points_<name>`` in, ``<prefix>_<name>`` out for each interface read."""

    name: str
    reader: str
    interfaces: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedRegistration:
    """A registration addressed to a dynamics: the registering interaction's id, the record's kind, the registered
    thing, and by label the subject's state to read and the subject's targets to deliver into."""

    id: str
    kind: str
    subject: str
    reads: Mapping[str, ResolvedRead]
    returns: Mapping[str, ResolvedTarget]


@dataclass
class EntryPlan:
    """One declared entry, resolved: who it is, what it was bound to, which nodes it will add and at what rate."""

    section: Section
    owner: str
    entry: Entry
    registered: Registered
    nodes: tuple[str, ...]
    divider: int
    declared: int
    thing: str | None = None
    thing_kind: ThingKind | None = None
    bindings: dict[str, str] = field(default_factory=dict[str, str])
    plan: Plan | None = None
    record: CapabilityRecord | None = None
    reads: dict[str, ResolvedRead] = field(default_factory=dict[str, ResolvedRead])
    delivers: dict[str, ResolvedTarget] = field(default_factory=dict[str, ResolvedTarget])
    registrations: list[ResolvedRegistration] = field(default_factory=list[ResolvedRegistration])
    slots: list[Slot] = field(default_factory=list[Slot])
    initial: dict[str, Any] = field(default_factory=dict[str, Any])
    seeds: dict[str, int] = field(default_factory=dict[str, int])
    settings: dict[str, Magnitude] = field(default_factory=dict[str, Magnitude])
    values: dict[str, materials.ResolvedValue] = field(default_factory=dict[str, materials.ResolvedValue])
    confinements: list[ResolvedConfinement] = field(default_factory=list[ResolvedConfinement])
    command: ResolvedCommand | None = None
    regions: dict[str, Region] = field(default_factory=dict[str, Region])
    hulls: dict[str, Hull] = field(default_factory=dict[str, Hull])
    position: int = 0
    segment: Segment | None = None
    mount: ResolvedMount | None = None
    parts: dict[str, ResolvedPart] = field(default_factory=dict[str, ResolvedPart])

    def node_name(self, local: str = "") -> str:
        return self.owner if not local else f"{self.owner}.{local}"

    def node_names(self) -> list[str]:
        return [self.node_name(local) for local in self.nodes]

    def reader_cls(self) -> type[Interaction] | type[Observation]:
        """The class of an interaction's or observation's plan."""
        return cast(type[Interaction] | type[Observation], self.registered.cls)

    def describe(self) -> str:
        rung = getattr(self.entry, "rung", None)
        return self.registered.cls.label() + (f" (rung {rung})" if rung else "")

    def canonical_key(self) -> tuple[int, int, str, int, str]:
        """Sections in order; dynamics by kind of thing and thing, a chain's segments in chain order; then by id."""
        if self.section is Section.DYNAMICS and self.thing_kind is not None:
            return (self.section.value, _DYNAMICS_ORDER[self.thing_kind], self.thing or "", self.position, self.owner)
        return (self.section.value, 0, "", 0, self.owner)


@dataclass(frozen=True)
class ValueLine:
    """A value the experiment's entries read: its path, its SI value and unit, where it came from and why, the ids
    of the entries that read it, and one line per input it was read or derived from."""

    path: str
    value: Magnitude
    unit: str
    source: str
    reason: str
    read_by: tuple[str, ...]
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupPlan:
    """A coupling group: the union of overlapping solve-together requests, with the settings they agree on."""

    nodes: tuple[str, ...]
    tolerance: float
    max_iterations: int
    requested_by: tuple[str, ...]


@dataclass(frozen=True)
class PlanSet:
    """Phase 1's result: everything phase 2 constructs, in canonical order and in declaration order."""

    settings: RunSettings
    dt: float
    frame: WorldFrame
    entries: tuple[EntryPlan, ...]
    declared: tuple[EntryPlan, ...]
    groups: tuple[GroupPlan, ...]
    targets: Mapping[tuple[str, str], ResolvedTarget]
    vocabulary: Mapping[str, str]
    run: RunStructure = RunStructure()
    things: Mapping[str, ThingKind] = field(default_factory=dict[str, ThingKind])
    values: tuple[ValueLine, ...] = ()
    geometries: Mapping[str, GeometryPlan] = field(default_factory=dict[str, GeometryPlan])
    fills: Mapping[str, str] = field(default_factory=dict[str, str])
    order: tuple[str, ...] = ()
    scene: ScenePlan = ScenePlan((0.0, 0.0, 1.0), "m", 1.0, MappingProxyType({}), frozenset(), ())
    orientation: tuple[OrientationLine, ...] = ()
    materials: MaterialsPlan = MaterialsPlan()
    apparatus_parts: Mapping[str, Mapping[str, ResolvedPart]] = field(
        default_factory=dict[str, Mapping[str, ResolvedPart]])
    controller: ControllerPlan | None = None

    def scene_text(self) -> str:
        """The world frame and the stage's settings, one line, for the build report."""
        up, gravity = (_vec(self.frame.up), _vec(self.frame.gravity))
        return (f"up {up} from the scene, so gravity {gravity} m/s^2; the stage is written in "
                f"{self.scene.units} (metersPerUnit {self.scene.meters_per_unit!r}), with {WORLD} its default prim")

    def owner_of(self, node: str) -> str | None:
        for plan in self.entries:
            if node in plan.node_names():
                return plan.owner
        return None


def _class_record(cls: type[Entry]) -> CapabilityRecord:
    if issubclass(cls, FieldProvider):
        return CapabilityRecord(cls.label(), frozenset(cls.flags),
                                {name: Port(prefix) for name, prefix in cls.publishes.items()})
    dynamics = cast(type[Dynamics], cls)
    return CapabilityRecord(dynamics.label(), frozenset(dynamics.flags), dict(dynamics.publishes),
                            dict(dynamics.accepts), frozenset(dynamics.registration_kinds))


def _offers(role: RoleKind) -> EntryKind:
    """The kind of entry whose capability record a role's support predicate is asked of."""
    return EntryKind.FIELD_PROVIDER if role is RoleKind.FIELD else EntryKind.DYNAMICS


def _qualifying(kind: EntryKind, predicate: Callable[[CapabilityRecord], bool]) -> str:
    names = [r.cls.label() for r in registered(kind) if predicate(_class_record(r.cls))]
    return ", ".join(names) if names else "none registered"


def _seconds(value: float) -> str:
    return f"{value!r} s"


def _vec(values: Sequence[float]) -> str:
    """A vector for the report: six significant digits, a negative zero written as zero."""
    return "(" + ", ".join(f"{float(c) + 0.0:.6g}" for c in values) + ")"


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def binding_name(value: Any) -> str:
    """The name a binding refers to, as written: a handle's name, a typed path's dotted string, or the string."""
    if isinstance(value, Handle):
        return value.name
    if isinstance(value, TypedPath):
        return str(value)
    return value if isinstance(value, str) else repr(value)


def entry_id(cls: type[Interaction] | type[Observation], bindings: Mapping[str, Any]) -> str:
    """An interaction's or observation's id: its registered name and the names bound to its id roles, never an
    attachment counter, so the canonical order cannot depend on the order of the script's calls."""
    roles = cls.roles
    id_roles = cls.id_roles
    if id_roles is None:
        id_roles = tuple(r for r, k in roles.items() if k is RoleKind.BODY) or tuple(roles)
    names = [binding_name(bindings[r]) if r in bindings and bindings[r] is not None else "?" for r in id_roles]
    return f"{cls.name}:{'-'.join(names)}"


class _Planner:
    def __init__(self, settings: RunSettings, declarations: list[Declaration]) -> None:
        self.settings = settings
        self.declarations = declarations
        self.refusals: list[Refusal] = []
        self.things: dict[str, Thing] = {}
        self.dynamics: dict[str, EntryPlan] = {}
        self.providers: dict[str, EntryPlan] = {}
        self.readers: list[EntryPlan] = []
        self.declared: list[EntryPlan] = []
        self.drives: list[EntryPlan] = []
        self.geometries: dict[str, GeometryPlan] = {}
        self.fills: dict[str, str] = {}
        self.values: dict[str, materials.Declared] = {}
        self.dt = 0.0
        self.frame: WorldFrame | None = None
        self.units = ""
        self.meters_per_unit = 1.0
        self.composition: Composition | None = None
        self.prims: dict[str, str] = {}
        self.bound: set[str] = set()
        self.magnets: list[ResolvedMagnet] = []
        self.conditions: dict[str, materials.OperatingCondition] = {}
        self.parts: dict[str, dict[str, PartLine]] = {}
        self.part_materials: dict[str, dict[str, materials.Record]] = {}
        self.materials: dict[str, materials.Record] = {}
        self.records: dict[str, materials.LoadedRecord] = {}
        self.resolver: materials.Resolver | None = None
        self.refused_problems: set[tuple[str, str, str]] = set()
        self.segments: dict[str, list[EntryPlan]] = {}
        self.apparatus_parts: dict[str, dict[str, ResolvedPart]] = {}
        self.initial_frames: dict[str, tuple[tuple[float, ...], ...]] = {}
        self.controller: EntryPlan | None = None
        self.mode: Mode | None = None
        self.latency = 0
        #: The commands the controller writes, by apparatus and command, each to its write label and path as written.
        self.written: dict[str, dict[str, tuple[str, str]]] = {}
        self.writes: dict[str, ResolvedWrite] = {}

    def refuse(self, code: str, category: RefusalClass, message: str, action: str) -> None:
        self.refusals.append(Refusal(code, category, message, action))

    def defect(self, cls: type[Entry], message: str) -> None:
        self.refuse("entry.defect", RefusalClass.INTERNAL, f"{cls.label()} ({cls.__module__}): {message}",
                    f"a defect in {cls.label()}, defined in {cls.__module__}: report it to that package. "
                    f"If that package is MIME: {MIME_DEFECT}")

    def run(self) -> PlanSet:
        self.plan_settings()
        things = [(index, d.thing) for index, d in enumerate(self.declarations) if isinstance(d, ThingDeclaration)]
        for _, thing in things:
            self.name_thing(thing)
        for _, thing in things:
            if isinstance(thing, Structure) and thing.geometry is not None and thing.name in self.things:
                self.plan_geometry(thing)
        self.plan_magnets()
        self.plan_parts()
        self.plan_values()
        self.plan_materials()
        for index, thing in things:
            if self.things.get(thing.name) is thing:
                self.plan_thing(thing, index)
        for index, declaration in enumerate(self.declarations):
            if isinstance(declaration, Attachment):
                self.plan_reader(declaration.law, dict(declaration.bindings), index, Section.INTERACTION)
            elif isinstance(declaration, ObservationDeclaration):
                observation = declaration.observation
                self.plan_reader(observation, observation.bindings(), index, Section.OBSERVATION)
        self.check_ids()
        self.resolve_values()
        self.plan_confinements()
        self.collect_registrations_and_slots()
        self.check_readers_outside_parts()
        self.declare_controller()
        for index, thing in things:
            if isinstance(thing, Apparatus) and self.things.get(thing.name) is thing:
                self.plan_commands(thing, index)
        controller = self.plan_controller()
        targets = self.collect_targets()
        self.check_exclusive()
        groups = self.union_groups()
        self.check_rates()
        self.derive_seeds()
        if self.refusals or self.frame is None:
            raise BuildRefused(self.refusals, BuildReport(self.settings.experiment, self.dt or None,
                                                          vocabulary=vocabulary.versions(),
                                                          refusals=tuple(self.refusals)))
        entries = tuple(sorted(self.declared, key=EntryPlan.canonical_key))
        order = gathered([name for plan in entries for name in plan.node_names()], [g.nodes for g in groups])
        layers = self.composition.layers if self.composition is not None else ()
        scene = ScenePlan(self.frame.up, self.units, self.meters_per_unit, dict(sorted(self.prims.items())),
                          frozenset(self.bound), layers, tuple(sorted(self.magnets, key=lambda m: (m.body, m.key))))
        values = self.value_lines()
        return PlanSet(self.settings, self.dt, self.frame, entries, tuple(self.declared), groups, targets,
                       vocabulary.versions(), self.run_structure(),
                       {name: thing.kind for name, thing in sorted(self.things.items())}, values,
                       dict(sorted(self.geometries.items())), dict(sorted(self.fills.items())), tuple(order),
                       scene, self.orientation_lines(), self.materials_plan(values),
                       {name: dict(sorted(parts.items())) for name, parts in sorted(self.apparatus_parts.items())},
                       controller)

    # -- run settings -----------------------------------------------------------------------------------------

    def plan_settings(self) -> None:
        name = self.settings.experiment
        conditions, problems = materials.operating_conditions(self.settings.conditions)
        for code, message, action in problems:
            self.refuse(code, RefusalClass.USER, message, action)
        self.conditions = conditions
        if not (isinstance(name, str) and _IDENTIFIER.fullmatch(name)):
            self.refuse("experiment.name", RefusalClass.USER, f"the experiment name {name!r} is not an identifier",
                        "name the experiment with letters, digits and underscores, starting with a letter")
        up = self.plan_scene()
        frame, problems = world_frame(up, self.settings.gravity)
        for code, message, action in problems:
            self.refuse(code, RefusalClass.USER, message, action)
        self.frame = frame
        dt = self.settings.dt
        if dt is None:
            self.refuse("run.dt-missing", RefusalClass.USER,
                        "dt, the base step, has no default: every node's timestep is a whole multiple of it",
                        "pass dt as a time to Experiment, such as dt=ms(1)")
            return
        try:
            value = si(dt, TIME, what="dt")
        except UnitError as error:
            self.refuse("run.dt-units", RefusalClass.USER, str(error), "write dt as a time, such as ms(1) or s(0.02)")
            return
        if isinstance(value, tuple) or not (math.isfinite(value) and value > 0.0):
            self.refuse("run.dt-invalid", RefusalClass.USER, f"dt = {dt} is not a finite time above zero",
                        "pass a positive dt, such as ms(1)")
            return
        self.dt = value
        if not isinstance(self.settings.seed, int) or isinstance(self.settings.seed, bool):
            self.refuse("run.seed", RefusalClass.USER, f"seed={self.settings.seed!r} is not an integer",
                        "pass an integer seed")

    def plan_scene(self) -> tuple[float, float, float] | None:
        """The scene's up, once checked, with its units and its references composed; or the refusals. Up is declared
        once, on the scene: an experiment's own ``up=`` is refused, alone or beside a scene."""
        scene = cast(object, self.settings.scene)
        if self.settings.up is not None:
            if scene is not None:
                self.refuse("frame.up-twice", RefusalClass.USER,
                            f"up is declared twice, on the scene and on the experiment (up={self.settings.up!r})",
                            "remove up= from Experiment: the scene's up is the one up")
            else:
                self.refuse("frame.up-on-experiment", RefusalClass.USER,
                            f"the experiment declares up={self.settings.up!r}, and up is declared once, on the scene",
                            "declare it on the scene: Experiment(..., scene=Scene.new(up=..., units=\"m\"))")
                return None
        if scene is None:
            self.refuse("frame.up-missing", RefusalClass.USER,
                        "the experiment declares no scene, so no up direction, and MIME assumes no axis",
                        "pass scene=Scene.new(up=+Z, units=\"m\") to Experiment, with whichever direction is up "
                        "in your world")
            return None
        if not isinstance(scene, Scene):
            self.refuse("scene.not-a-scene", RefusalClass.USER, f"scene={scene!r} is not a Scene",
                        "pass scene=Scene.new(up=..., units=...)")
            return None
        up, problems = scene_up(scene.declared_up)
        mpu, unit_problems = meters_per_unit(scene.units)
        self.units, self.meters_per_unit = str(scene.units), mpu if mpu is not None else 1.0
        self.composition = compose(scene.references, mpu)
        for code, message, action in [*problems, *unit_problems, *self.composition.problems]:
            self.refuse(code, RefusalClass.USER, message, action)
        return up

    def divider(self, rate: Any, what: str) -> int:
        """``rate`` as a whole multiple of ``dt`` (a period, or a frequency whose period is one), or a refusal."""
        if rate is None or not self.dt:
            return 1
        try:
            if isinstance(rate, Quantity) and rate.dimension == FREQUENCY:
                frequency = si(rate, FREQUENCY, what=f"the rate of {what}")
                period = 1.0 / frequency if isinstance(frequency, float) and frequency > 0 else math.nan
            else:
                period = si(rate, TIME, what=f"the rate of {what}")
        except UnitError as error:
            self.refuse("run.rate-units", RefusalClass.USER, str(error),
                        "write a rate as a period, such as ms(2), or a frequency, such as Hz(500)")
            return 1
        ratio = period / self.dt if isinstance(period, float) else math.nan
        whole = round(ratio) if math.isfinite(ratio) else 0
        if whole >= 1 and abs(ratio - whole) <= _RATE_RTOL * ratio:
            return whole
        below, above = (max(1, math.floor(ratio)), math.ceil(ratio)) if math.isfinite(ratio) else (1, 1)
        nearest = sorted({below, max(above, 1)})
        self.refuse("run.rate-not-multiple", RefusalClass.USER,
                    f"the rate of {what} is a period of {_seconds(period) if isinstance(period, float) else rate}, "
                    f"not a whole multiple of the base step dt = {_seconds(self.dt)}",
                    "use a period that is a whole multiple of dt; the nearest are "
                    + " and ".join(f"{k} step{'s' * (k > 1)} ({k * self.dt:.6g} s, {1.0 / (k * self.dt):.4g} Hz)"
                                   for k in nearest))
        return 1

    # -- things, their geometry, dynamics and providers ----------------------------------------------------

    def name_thing(self, thing: Thing) -> None:
        name = cast(object, thing.name)  # a script is not type-checked
        bound_to = thing.prim if isinstance(thing, Body) else None
        if not (isinstance(name, str) and _IDENTIFIER.fullmatch(name)):
            named = (f"the body bound to {bound_to} is named after its prim's last element, {name!r}, which is not "
                     f"an identifier" if bound_to is not None else
                     f"the {thing.kind.value} name {name!r} is not an identifier")
            self.refuse("thing.name", RefusalClass.USER, named,
                        "name things with letters, digits and underscores, starting with a letter")
            return
        if name in self.things:
            held = self.things[name]
            self.refuse("thing.duplicate", RefusalClass.USER,
                        f"two things are named {name!r}: a {held.kind.value} at {prim_path_of(held)} and a "
                        f"{thing.kind.value} at {prim_path_of(thing)} (a body bound to a prim is named after the "
                        f"prim's last element unless it is given a name)",
                        "give every thing its own name: Body(\"<name>\", prim=...)")
            return
        self.things[name] = thing
        self.place_thing(thing)

    def place_thing(self, thing: Thing) -> None:
        """The thing's prim: the prim an asset brings in that a body binds, which must exist, or ``/World/<name>``
        for a thing written Python-first, which no asset may already hold. Two things never share a prim."""
        path = prim_path_of(thing)
        where = f"{thing.kind.value} {thing.name!r}"
        composition = self.composition
        if isinstance(thing, Body) and thing.prim is not None:
            if not valid_prim_path(path):
                self.refuse("thing.prim", RefusalClass.USER, f"{where} binds {path!r}, not a prim path below {WORLD}",
                            f"name the prim as scene[\"{WORLD}/...\"]")
                return
            if composition is None or not (composition.provided(path) and composition.has(path)):
                prims = composition.prims_below(WORLD) if composition is not None else []
                self.refuse("thing.prim-missing", RefusalClass.USER,
                            f"{where} binds {path}, which no referenced asset brings into the scene",
                            did_you_mean(path, prims) + " Reference the asset that holds it: "
                                                        "scene.reference(asset(...), at=...)")
                return
            roots = [layer.at for layer in composition.layers]
            if path not in roots:
                root = next(at for at in roots if path.startswith(f"{at}/"))
                self.refuse("thing.prim-inside-asset", RefusalClass.USER,
                            f"{where} binds {path}, a prim inside the asset referenced at {root}: a body binds the "
                            f"prim its asset is referenced at, and the prims inside it are its parts",
                            f"bind {root}, and give its parts their roles: magnets={{...}}")
                return
            self.bound.add(thing.name)
        elif composition is not None and composition.provided(path):
            self.refuse("thing.prim-taken", RefusalClass.USER,
                        f"{where} is written at {path}, which a referenced asset already brings in",
                        "bind the asset's prim with prim=scene[...], or give the thing another name")
            return
        for other, held in self.prims.items():
            if held == path:
                self.refuse("thing.duplicate-prim", RefusalClass.USER, f"{other!r} and {thing.name!r} both bind {path}",
                            "bind each thing to its own prim")
                return
        self.prims[thing.name] = path

    def plan_magnets(self) -> None:
        """Each magnet part of a body, its direction read from the part's prim through the ``mime:`` schema or
        declared with its role, never both and never neither. A body written Python-first has magnets only among the
        parts it declares, with their directions declared. A body's declared moment takes its direction from its
        magnet, so a body with several magnets has its moment derived from its parts."""
        composition = self.composition
        for name, thing in self.things.items():
            if isinstance(thing, Apparatus) and thing.magnets and name in self.prims:
                self.plan_apparatus_magnets(name, thing)
            if not isinstance(thing, Body) or not thing.magnets:
                continue
            where = f"body {name!r}"
            if name not in self.bound or composition is None:
                if name in self.prims and thing.parts:
                    for key, role in thing.magnets.items():
                        self.plan_declared_magnet(name, key, cast(object, role), thing)
                elif name in self.prims:
                    self.refuse("magnet.no-prim", RefusalClass.USER,
                                f"{where} gives parts a magnet's role, and binds no prim and declares no parts: a "
                                f"magnet is a part, below the prim a body binds or among the parts it declares",
                                "bind the body to its asset's prim, Body(..., prim=scene[\"/World/...\"]), or declare "
                                "its parts: Body(..., parts={\"<name>\": Part(...)})")
                continue
            if len(thing.magnets) > 1 and not thing.parts:
                self.refuse("magnet.several", RefusalClass.USER,
                            f"{where} has {len(thing.magnets)} magnets and declares no parts: a body's moment from "
                            f"several magnets is the part resolver's sum, over the parts the magnets are",
                            "declare the body's parts (Body(..., parts={...})), or keep one magnet")
                continue
            body = self.prims[name]
            for key, role in thing.magnets.items():
                self.plan_magnet(name, body, key, cast(object, role), composition)

    def plan_apparatus_magnets(self, name: str, thing: Apparatus) -> None:
        """An apparatus's magnet, among the parts it declares, its direction declared in the part's frame: its moment
        is the apparatus's ``magnetic_moment``, declared as a magnitude along that direction, so it has one."""
        if len(thing.magnets) > 1:
            self.refuse("magnet.several", RefusalClass.USER,
                        f"apparatus {name!r} has {len(thing.magnets)} magnets ({', '.join(sorted(thing.magnets))}): an "
                        f"apparatus's moment is declared for its one magnet, {name}.magnetic_moment, and a value per "
                        f"part of an apparatus is not built", "keep one magnet in each apparatus")
            return
        for key, role in thing.magnets.items():
            self.plan_declared_magnet(name, key, cast(object, role), thing)

    def plan_declared_magnet(self, name: str, key: str, role: object, thing: Body | Apparatus) -> None:
        """A magnet among the parts a body or an apparatus written Python-first declares: a body's part's frame is
        turned with the body's, and an apparatus's part's is not turned from its mount's, so the direction declared
        in the part's frame is the body's, or the mount's, too."""
        where = f"{thing.kind.value} {name!r}'s magnet {key!r}"
        if not isinstance(role, PermanentMagnet):
            self.refuse("magnet.role", RefusalClass.USER, f"{where} is {role!r}, not a magnet's role",
                        "give the part its role: magnets={\"<part>\": PermanentMagnet(direction=...)}")
            return
        if key not in thing.parts:
            self.refuse("magnet.part-missing", RefusalClass.USER, f"{where} names no part the body declares",
                        did_you_mean(key, thing.parts))
            return
        if role.direction is None:
            self.refuse("magnet.direction-missing", RefusalClass.USER,
                        f"{where} has no direction: a part declared in the experiment has no prim to carry it, and "
                        f"MIME assumes no axis", "pass PermanentMagnet(direction=(x, y, z)) in the part's frame")
            return
        declared = unit_vector(role.direction)
        if declared is None:
            self.refuse("magnet.direction", RefusalClass.USER,
                        f"{where}: direction={role.direction!r} is not three numbers, not all zero",
                        "pass the direction as three numbers in the part's frame")
            return
        self.magnets.append(ResolvedMagnet(name, key, f"{self.prims[name]}/{key}", role.model, declared, declared,
                                           "declared in the experiment"))

    def plan_magnet(self, name: str, body: str, key: str, role: object, composition: Composition) -> None:
        where = f"body {name!r}'s magnet {key!r}"
        if not isinstance(role, PermanentMagnet):
            self.refuse("magnet.role", RefusalClass.USER, f"{where} is {role!r}, not a magnet's role",
                        "give the part its role: magnets={\"<part>\": PermanentMagnet()}")
            return
        if not valid_part_path(key):
            self.refuse("magnet.part", RefusalClass.USER, f"{where}: {key!r} is not a part's path below {body}",
                        "name the part by its path below the body's prim, such as \"Magnets\"")
            return
        part = f"{body}/{key}"
        design, problems = composition.magnet(part, body)
        for code, message, action in problems:
            self.refuse(code, RefusalClass.USER, f"{where}: {message}", action)
        if problems:
            return
        if design is not None and role.direction is not None:
            self.refuse("magnet.direction-twice", RefusalClass.USER,
                        f"{where} has its direction declared twice: {MAGNET_DIRECTION} = {_vec(design.in_part)} on "
                        f"its prim {part}, and direction={role.direction!r} in the experiment",
                        "keep the asset's design data and drop direction=, or remove the attribute from the asset")
            return
        if design is not None:
            self.magnets.append(ResolvedMagnet(name, key, part, role.model, design.in_part, design.in_body,
                                               f"the {MAGNET_API} schema on {part}"))
            return
        if role.direction is None:
            self.refuse("magnet.direction-missing", RefusalClass.USER,
                        f"{where} has no direction: its prim {part} carries no {MAGNET_DIRECTION}, and the "
                        f"experiment declares none, and MIME assumes no axis",
                        f"write it on the part's prim with the {MAGNET_API} schema, or pass "
                        f"PermanentMagnet(direction=(x, y, z)) in the part's frame")
            return
        declared = unit_vector(role.direction)
        if declared is None:
            self.refuse("magnet.direction", RefusalClass.USER,
                        f"{where}: direction={role.direction!r} is not three numbers, not all zero",
                        "pass the direction as three numbers in the part's frame")
            return
        self.magnets.append(ResolvedMagnet(name, key, part, role.model, declared,
                                           composition.in_body(part, body, declared), "declared in the experiment"))

    def directed(self) -> dict[str, tuple[tuple[float, float, float], str]]:
        """The values whose direction a body's one magnet gives, by path, with the unit vector in the body's frame and
        where it came from. A body with several magnets has no one direction to give a declared magnitude."""
        counts = Counter(m.body for m in self.magnets)
        return {f"{m.body}.magnetic_moment": (m.in_body, f"the magnet {m.key}'s direction, from {m.source}")
                for m in self.magnets if counts[m.body] == 1}

    # -- parts and materials ------------------------------------------------------------------------------

    def plan_parts(self) -> None:
        """Each part of each body: its solid checked and in SI, with its volume and inertia; its placement in the
        body's frame; the part it lies inside; and whether it is a magnet."""
        for name, thing in self.things.items():
            if not isinstance(thing, Body) or not thing.parts or name not in self.prims:
                continue
            planned: dict[str, PartLine] = {}
            for key, part in sorted(thing.parts.items()):  # by name: the order a script lists them changes nothing
                line = self.plan_part(name, key, cast(object, part), thing)
                if line is not None:
                    planned[key] = line
            for key, line in planned.items():
                host = line.inside
                if host is None:
                    continue
                if host == key or host not in thing.parts or thing.parts[host].inside is not None:
                    self.refuse("part.inside", RefusalClass.USER,
                                f"body {name!r}'s part {key!r} lies inside {host!r}, which is not another part of the "
                                f"body that lies inside none", did_you_mean(host, [k for k in thing.parts if k != key]))
            for key in thing.magnets:
                if name in self.bound and key not in thing.parts:
                    self.refuse("magnet.no-part", RefusalClass.USER,
                                f"body {name!r} declares parts, and its magnet {key!r} is not among them: its moment "
                                f"is derived from its parts", f"declare the magnet's part: parts={{{key!r}: Part(...)}}")
            self.parts[name] = planned

    def plan_part(self, name: str, key: str, part: object, thing: Body) -> PartLine | None:
        where = f"body {name!r}'s part {key!r}"
        bound = name in self.bound
        body = self.prims[name]
        if not isinstance(part, Part):
            self.refuse("part.not-a-part", RefusalClass.USER, f"{where} is {part!r}, not a Part",
                        "declare it as Part(<a solid>, material=...)")
            return None
        if bound:
            composition = self.composition
            if not valid_part_path(key) or composition is None or not composition.has(f"{body}/{key}"):
                below = [p.removeprefix(f"{body}/") for p in composition.prims_below(body) if p != body] \
                    if composition is not None else []
                self.refuse("part.missing", RefusalClass.USER,
                            f"{where}: a bound body's part is a prim below {body}, and there is no {body}/{key}",
                            did_you_mean(key, below))
                return None
        elif not _IDENTIFIER.fullmatch(key):
            self.refuse("part.name", RefusalClass.USER, f"{where}: {key!r} is not an identifier",
                        "name parts with letters, digits and underscores, starting with a letter")
            return None
        if part.mount is not None:
            self.refuse("part.mount", RefusalClass.USER,
                        f"{where} is mounted on {part.mount!r}: a body's part is placed in the body's frame, and a "
                        f"mount is an apparatus's part's", "drop mount=, and place the part with at= or by its prim")
            return None
        geometry = cast(object, part.geometry)
        record = self.registered_as(cast(Entry, geometry), EntryKind.GEOMETRY, f"{where}'s solid")
        if record is None:
            return None
        if not isinstance(geometry, Solid):
            self.refuse("part.not-a-solid", RefusalClass.USER,
                        f"{where} is a {record.cls.label()}, which gives no volume or inertia",
                        "give the part a solid: " + ", ".join(r.cls.label() for r in registered(EntryKind.GEOMETRY)
                                                              if issubclass(r.cls, Solid)))
            return None
        settings: dict[str, Magnitude] = {}
        for setting, declared in record.settings.items():
            if declared.kind is Direction:
                found = self.literal_direction(getattr(geometry, setting, None),
                                               f"{where}'s solid ({record.cls.label()}): setting {setting!r}")
                if found is not None:
                    settings[setting] = found
        settings.update(self.settings_in_si(geometry, record, f"{where}'s solid"))
        if self.setting_problems(geometry, settings, f"{where}'s solid") or not self.settings_complete(geometry,
                                                                                                       settings):
            return None
        try:
            volume, inertia = float(geometry.volume(settings)), geometry.inertia_per_mass(settings)
            reach = float(geometry.radial_reach(settings)) if implements(geometry, "radial_reach") else None
        except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
            self.defect(record.cls, f"volume(), inertia_per_mass() or radial_reach() raised {error!r}")
            return None
        if not (math.isfinite(volume) and volume > 0.0):
            self.defect(record.cls, f"volume() gave {volume!r} m^3 for settings it accepted")
            return None
        identity: tuple[tuple[float, float, float], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        if bound:
            if part.at is not None:
                self.refuse("part.at-on-asset", RefusalClass.USER,
                            f"{where} is a prim of the asset, placed by the asset's transforms, and declares at= too",
                            "drop at=: the asset places its parts")
                return None
            assert self.composition is not None
            centre, rotation, problems = self.composition.placement(f"{body}/{key}", body, self.meters_per_unit)
            for code, message, action in problems:
                self.refuse(code, RefusalClass.USER, f"{where}: {message}", action)
            if problems:
                return None
        else:
            centre, rotation = (0.0, 0.0, 0.0), identity
            if part.at is not None:
                try:
                    at = si(part.at, LENGTH, what=f"{where}'s at=")
                except (UnitError, TypeError) as error:
                    self.refuse("part.at", RefusalClass.USER, str(error),
                                "place the part's centroid with three lengths in the body's frame: at=mm((x, y, z))")
                    return None
                if not (isinstance(at, tuple) and len(at) == 3 and all(math.isfinite(c) for c in at)):
                    self.refuse("part.at", RefusalClass.USER, f"{where}'s at={part.at} is not a point (three lengths)",
                                "place the part's centroid with three lengths: at=mm((x, y, z))")
                    return None
                centre = (at[0], at[1], at[2])
        inside = cast(object, part.inside)
        if inside is not None and not isinstance(inside, str):
            self.refuse("part.inside", RefusalClass.USER, f"{where}: inside={inside!r} is not a part's name",
                        "name the part it lies inside")
            return None
        material = cast(object, part.material)
        if material is not None:
            if not isinstance(material, materials.Record):
                self.refuse("material.not-a-record", RefusalClass.USER,
                            f"{where}'s material is {material!r}, not a mime.materials.Record",
                            "name its record: Record(\"<name>\", packaged_record(...) or record_file(...))")
                return None
            self.part_materials.setdefault(name, {})[key] = material
        def turned(i: int, j: int) -> float:
            return math.fsum(rotation[i][k] * inertia[k][m] * rotation[j][m] for k in range(3) for m in range(3))

        in_body: Tensor = ((turned(0, 0), turned(0, 1), turned(0, 2)), (turned(1, 0), turned(1, 1), turned(1, 2)),
                           (turned(2, 0), turned(2, 1), turned(2, 2)))
        return PartLine(name, key, geometry, record, settings, volume, centre, rotation, in_body,
                        inside, material.name if isinstance(material, materials.Record) else None,
                        key in thing.magnets, reach)

    def plan_materials(self) -> None:
        """Every material the experiment names, from its media and its bodies' parts: one material per name, each
        record file read and checked against its pin; then the resolver every value is read through."""
        users: list[tuple[str, object]] = [
            (f"medium {name!r}", cast(object, thing.material)) for name, thing in self.things.items()
            if isinstance(thing, Medium) and thing.material is not None]
        users += [(f"body {name!r}'s part {key!r}", material) for name, held in self.part_materials.items()
                  for key, material in held.items()]
        for where, material in users:
            if not isinstance(material, materials.Record):
                self.refuse("material.not-a-record", RefusalClass.USER,
                            f"{where}'s material is {material!r}, not a mime.materials.Record",
                            "name its record: Record(\"<name>\", packaged_record(...) or record_file(...))")
                continue
            for code, message, action in materials.record_problems(material):
                self.refuse(code, RefusalClass.USER, f"{where}: {message}", action)
            held = self.materials.get(material.name)
            if held is not None and held is not material:
                self.refuse("material.name", RefusalClass.USER,
                            f"two different materials are named {material.name!r}; {where}'s is one",
                            "give each material its own name, or use the one material in both places")
                continue
            self.materials[material.name] = material
        files: dict[str, str] = {}
        for material in self.materials.values():
            loaded, problems = materials.load_record(material)
            for code, message, action in problems:
                self.refuse(code, RefusalClass.USER, message, action)
            if loaded is None:
                continue
            if files.get(loaded.file, loaded.sha256) != loaded.sha256:
                self.refuse("material.file-name", RefusalClass.USER,
                            f"two different record files are both named {loaded.file}, and the saved form writes "
                            f"each as {materials.RECORDS}/{loaded.file}", "give the record files distinct names")
                continue
            files[loaded.file] = loaded.sha256
            self.records[material.name] = loaded
        store = materials.record_store(self.records.values()) if self.records else None
        media = {name: thing.material for name, thing in self.things.items()
                 if isinstance(thing, Medium) and isinstance(cast(object, thing.material), materials.Record)
                 and thing.material is not None}
        directions = {(m.body, m.key): (m.in_body, m.source) for m in self.magnets}
        bodies = {name: {key: materials.PartPlan(
            PartMass(key, line.registered.cls.label(), line.volume, line.centre, line.inertia_per_mass, line.inside,
                     line.reach),
            self.part_materials.get(name, {}).get(key), *directions.get((name, key), (None, "")))
            for key, line in planned.items()} for name, planned in self.parts.items()}
        kinds = {name: thing.kind for name, thing in self.things.items()}
        self.resolver = materials.Resolver(kinds, self.values, self.directed(), self.conditions, media, bodies,
                                           self.records, store)

    def plan_thing(self, thing: Thing, index: int) -> None:
        if thing.dynamics is not None:
            self.plan_dynamics(thing, thing.dynamics, index)
        if isinstance(thing, Apparatus):
            self.plan_chain(thing, index)
            self.plan_apparatus_parts(thing)
            for field_name, provider in thing.fields.items():
                self.plan_provider(thing.name, field_name, provider, index)
        if isinstance(thing, Medium) and thing.fills is not None:
            region = self.geometry_part(thing.fills, RegionPath, "regions", f"medium {thing.name!r}: fills=")
            if region is not None:
                self.fills[thing.name] = region

    def plan_geometry(self, structure: Structure) -> None:
        geometry = structure.geometry
        where = f"structure {structure.name!r}'s geometry"
        record = self.registered_as(cast(Entry, geometry), EntryKind.GEOMETRY, where)
        if record is None or geometry is None:
            return
        settings: dict[str, Magnitude] = {}
        for name, declared in record.settings.items():
            if declared.kind is Direction:
                value: object = getattr(geometry, name, None)
                if isinstance(value, (TypedPath, str)):
                    self.refuse("setting.direction", RefusalClass.USER,
                                f"{where} ({record.cls.label()}): setting {name!r} refers to {value}; a geometry's own "
                                f"directions are written as three numbers in the world frame (geometries are not "
                                f"placed relative to one another yet)",
                                f"write {name} as three numbers, such as (1, 0, 0)")
                    continue
                found = self.literal_direction(value, f"{where} ({record.cls.label()}): setting {name!r}")
                if found is not None:
                    settings[name] = found
        settings.update(self.settings_in_si(geometry, record, where))
        if implements(geometry, "region_primitive") and not all(
                MANIFEST_NAME.fullmatch(f"{structure.name}_{region}") for region in geometry.regions):
            self.refuse("thing.name", RefusalClass.USER,
                        f"{where} ({record.cls.label()}): structure {structure.name!r} names its regions "
                        f"{structure.name}_<region> in OpenMaterica's scene manifest, whose names are lowercase "
                        f"letters, digits and underscores",
                        f"name the structure {re.sub('[^a-z0-9_]', '_', structure.name.lower())!r}, or with lowercase "
                        f"letters, digits and underscores")
        problems = self.setting_problems(geometry, settings, where)
        if problems or not self.settings_complete(geometry, settings):
            return
        self.geometries[structure.name] = GeometryPlan(structure.name, geometry, record, settings)

    def settings_complete(self, entry: Entry, settings: Mapping[str, Magnitude]) -> bool:
        """Whether every setting the entry needs reached SI: one that did not is refused already."""
        record = registration_of(entry)
        return record is not None and all(name in settings or (d.optional and setting_value(entry, name) is None)
                                          for name, d in record.settings.items())

    def setting_problems(self, entry: Entry, settings: Mapping[str, Magnitude], where: str) -> list[str]:
        """The entry's own judgement of its settings in SI, each problem a refusal; none asked of settings that did
        not reach SI, which are refused already."""
        record = registration_of(entry)
        if record is None or not self.settings_complete(entry, settings):
            return []
        try:
            problems = list(entry.setting_problems(settings))
        except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
            self.defect(type(entry), f"setting_problems() raised {error!r}")
            return ["defect"]
        for problem in problems:
            self.refuse("setting.value", RefusalClass.USER, f"{where} ({record.cls.label()}): {problem}",
                        "correct the setting")
        return problems

    def geometry_part[P: GeometryPath](self, value: object, kind: type[P], names: str, where: str,
                                       reader: type[Entry] | None = None) -> str | None:
        """The path ``value`` names (a typed path of ``kind``, or its string), as ``<structure>.<part>``, once
        checked against the structure's geometry; or a refusal. ``reader`` is the entry that refers to it: geometry is
        read once, at construction, so a reference to the geometry of a thing that moves is refused unless the reader
        declares ``moving``."""
        noun = {"regions": "region", "surfaces": "surface", "directions": "direction"}[names]
        if isinstance(value, TypedPath) and not isinstance(value, kind):
            self.refuse("binding.wrong-kind", RefusalClass.USER, f"{where} takes a {noun}, not {value}",
                        f"name a {noun} of a structure's geometry, as structure.{noun}")
            return None
        if not isinstance(value, (TypedPath, str)):
            self.refuse("binding.wrong-kind", RefusalClass.USER, f"{where} takes a {noun}, not {value!r}",
                        f"name a {noun} as structure.{noun}")
            return None
        path = str(value)
        owner, _, part = path.partition(".")
        thing = self.things.get(owner)
        if reader is not None and thing is not None and (isinstance(thing, Body) or thing.dynamics is not None) \
                and not reader.moving:
            moves = "a body moves with its dynamics" if isinstance(thing, Body) else \
                f"it has the dynamics {cast(Entry, thing.dynamics).label()}"
            self.refuse("geometry.moving-read", RefusalClass.USER,
                        f"{where}{path!r}: {reader.label()} reads the geometry of {thing.kind.value} {owner!r}, which "
                        f"moves ({moves}); {reader.label()} takes geometry once, at construction, and would keep it "
                        f"as it was at t = 0", f"refer to the geometry of a fixed thing, or use an entry that declares "
                                               f"moving (it reads the pose of {owner!r} every step)")
            return None
        if isinstance(thing, Body) and names == "surfaces":
            self.refuse("binding.moving-surface", RefusalClass.USER,
                        f"{where}{path!r} is a surface of body {owner!r}: a body's surfaces are reserved, not built, "
                        f"since projection between two moving things is a constraint across several bodies",
                        "bind a surface of a fixed structure, such as tube.inner_surface")
            return None
        if thing is None:
            self.refuse("binding.unknown", RefusalClass.USER, f"{where}{path!r} names no thing",
                        did_you_mean(owner, [n for n, t in self.things.items() if isinstance(t, Structure)]))
            return None
        geometry = self.geometries.get(owner)
        if geometry is None:
            self.refuse("binding.unknown", RefusalClass.USER,
                        f"{where}{path!r}: {thing.kind.value} {owner!r} has no geometry, so no {noun} {part!r}",
                        f"give {owner!r} a geometry, or name a {noun} of another structure")
            return None
        held = cast(Any, type(geometry.geometry))
        candidates: tuple[str, ...] = tuple(held.surfaces) if names == "surfaces" else tuple(getattr(held, names))
        if part not in candidates:
            self.refuse("binding.unknown", RefusalClass.USER,
                        f"{where}{path!r}: {geometry.registered.cls.label()} has no {noun} {part!r}",
                        did_you_mean(part, candidates))
            return None
        return path

    def region_of(self, medium: str) -> Region | None:
        """The region medium ``medium`` fills, as a shape and its parameters in SI, when its geometry kind gives that
        region a shape; ``None`` otherwise."""
        path = self.fills.get(medium)
        if path is None:
            return None
        owner, _, part = path.partition(".")
        plan = self.geometries.get(owner)
        if plan is None:
            return None
        shape = type(plan.geometry).region_shapes.get(part)
        if shape is None:
            return None
        try:
            parameters = plan.geometry.region_parameters(part, plan.settings)
        except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
            self.defect(plan.registered.cls, f"region_parameters({part!r}) raised {error!r}")
            return None
        return Region(path, shape, parameters)

    def literal_direction(self, value: object, what: str) -> tuple[float, float, float] | None:
        try:
            components = tuple(float(c) for c in cast(Any, value))
        except (TypeError, ValueError):
            components = ()
        norm = math.sqrt(sum(c * c for c in components)) if len(components) == 3 else 0.0
        if len(components) != 3 or not math.isfinite(norm) or norm == 0.0:
            self.refuse("setting.direction", RefusalClass.USER,
                        f"{what} is {value!r}, not a direction: a thing's direction such as tube.axis, or three "
                        f"numbers not all zero", "write the direction as tube.axis or as three numbers")
            return None
        return (components[0] / norm, components[1] / norm, components[2] / norm)

    def direction(self, value: object, what: str, reader: type[Entry] | None = None) -> tuple[float, float, float] | None:
        """A direction setting as a unit vector: a typed path or string naming a geometry's direction, or three
        numbers."""
        if isinstance(value, (TypedPath, str)):
            path = self.geometry_part(value, DirectionPath, "directions", f"{what} = ", reader)
            if path is None:
                return None
            owner, _, part = path.partition(".")
            plan = self.geometries[owner]
            return self.literal_direction(plan.geometry.direction(part, plan.settings), what)
        return self.literal_direction(value, what)

    def registered_as(self, entry: Entry, kind: EntryKind, where: str) -> Registered | None:
        record = registration_of(entry)
        if record is None or record.kind is not kind:
            self.refuse("entry.unregistered", RefusalClass.USER,
                        f"{where}: {type(entry).__qualname__} is not a registered {kind.value}",
                        f"pass a registered {kind.value}, or register it with "
                        f"mime.registry.register_{kind.name.lower()}")
            return None
        return record

    def plan_dynamics(self, thing: Thing, dynamics: Dynamics, index: int) -> None:
        where = f"{thing.kind.value} {thing.name!r}"
        record = self.registered_as(dynamics, EntryKind.DYNAMICS, where)
        if record is None:
            return
        cls = cast(type[Dynamics], record.cls)
        if thing.kind not in cls.applies_to:
            applies = ", ".join(sorted(k.value for k in cls.applies_to))
            others = [cast(type[Dynamics], r.cls) for r in registered(EntryKind.DYNAMICS)]
            self.refuse("dynamics.wrong-thing", RefusalClass.USER,
                        f"{where} has dynamics {cls.label()}, which applies to {applies} only",
                        f"choose a dynamics for a {thing.kind.value}: " + (", ".join(
                            other.label() for other in others if thing.kind in other.applies_to)
                            or "none registered"))
            return
        if not self.planned_rung(cls, dynamics, where)[0]:
            return
        try:
            capabilities = dynamics.capabilities()
        except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
            self.defect(cls, f"capabilities() raised {error!r}")
            return
        owner = f"{thing.name}.flow" if thing.kind is ThingKind.MEDIUM else thing.name
        plan = EntryPlan(Section.DYNAMICS, owner, dynamics, record, tuple(cls.nodes),
                         self.divider(dynamics.rate, f"{where}'s dynamics"), index, thing=thing.name,
                         thing_kind=thing.kind, record=capabilities,
                         settings=self.settings_in_si(dynamics, record, f"{where}'s dynamics"))
        if not self.check_locals(cls, plan.nodes):
            return
        self.setting_problems(dynamics, plan.settings, f"{where}'s dynamics")
        if isinstance(thing, Body):
            plan.initial = self.initial_state(thing, cls)
            if set(plan.initial) == set(cls.initial_state):
                try:
                    problems = dynamics.initial_problems(plan.initial)
                except Exception as error:  # noqa: BLE001
                    self.defect(cls, f"initial_problems() raised {error!r}")
                    problems = []
                for problem in problems:
                    self.refuse("state.value", RefusalClass.USER, f"body {thing.name!r}'s initial state: {problem}",
                                "correct the initial state")
        self.dynamics[thing.name] = plan
        self.declared.append(plan)

    def initial_state(self, thing: Body, cls: type[Dynamics]) -> dict[str, Any]:
        wanted = cls.initial_state
        given = dict(thing.initial)
        values: dict[str, Any] = {}
        for key, value in given.items():
            if key not in wanted:
                self.refuse("state.unknown", RefusalClass.USER,
                            f"body {thing.name!r}: its dynamics {cls.label()} takes no initial {key!r}",
                            did_you_mean(key, wanted))
                continue
            try:
                values[key] = si(value, wanted[key], what=f"body {thing.name!r}'s initial {key}")
            except UnitError as error:
                self.refuse("state.units", RefusalClass.USER, str(error), f"write {key} in units of {wanted[key]}")
            except TypeError as error:
                self.refuse("state.value", RefusalClass.USER, f"body {thing.name!r}'s initial {key}: {error}",
                            f"write {key} as a quantity of {wanted[key]}")
        missing = [key for key in wanted if key not in given]
        if missing:
            self.refuse("state.missing", RefusalClass.USER,
                        f"body {thing.name!r}: its dynamics {cls.label()} needs an initial "
                        f"{', '.join(missing)}, and MIME assumes none",
                        f"pass initial=State({', '.join(f'{k}=...' for k in missing)})")
        return values

    def planned_rung(self, cls: type[Entry], entry: Entry, where: str) -> tuple[bool, str | None]:
        """Whether the rung ``entry`` writes is one of its built rungs (``True`` for an entry without rungs), refused
        otherwise, with the rung written."""
        rungs = cast(tuple[str, ...], getattr(cls, "rungs", ()))
        rung = cast(str | None, getattr(entry, "rung", None)) if rungs else None
        reserved = cast(Mapping[str, str], getattr(cls, "reserved_rungs", {}))
        if rungs and rung not in rungs:
            valid = ", ".join(repr(r) for r in rungs if r not in reserved)
            if rung is None:
                self.refuse("rung.missing", RefusalClass.USER, f"{where} has no default rung",
                            f"write the rung: rung= one of {valid}")
            else:
                self.refuse("rung.unknown", RefusalClass.USER, f"{where}: rung={rung!r} is not one of its rungs",
                            did_you_mean(str(rung), rungs))
            return False, rung
        if rung in reserved:
            self.refuse("rung.reserved", RefusalClass.USER,
                        f"{where}: the rung {rung!r} is reserved, not built: {reserved[str(rung)]}",
                        "write one of the built rungs: " + ", ".join(repr(r) for r in rungs if r not in reserved))
            return False, rung
        return True, rung

    # -- an apparatus's chain and parts -----------------------------------------------------------------------

    def plan_chain(self, apparatus: Apparatus, index: int) -> None:
        """Each segment of the apparatus's chain, in chain order: its rung and settings checked, what its settings
        make it (its frames, its commands, the classification of its description), and the frame of the segment
        before it that it mounts on. A refused segment ends the chain there: what mounts on it cannot be placed."""
        previous: EntryPlan | None = None
        planned = self.segments.setdefault(apparatus.name, [])
        for position, (local, segment) in enumerate(apparatus.chain.items()):
            where = f"apparatus {apparatus.name!r}'s segment {local!r}"
            plan = self.plan_segment(apparatus, local, cast(object, segment), position, previous, index, where)
            if plan is None:
                return
            planned.append(plan)
            self.declared.append(plan)
            previous = plan

    def plan_segment(self, apparatus: Apparatus, local: object, segment: object, position: int,
                     previous: EntryPlan | None, index: int, where: str) -> EntryPlan | None:
        if not (isinstance(local, str) and _LOCAL.fullmatch(local)):
            self.refuse("chain.segment-name", RefusalClass.USER, f"{where}: {local!r} is not a lower-case name",
                        "name segments with lower-case letters, digits and underscores, starting with a letter")
            return None
        record = self.registered_as(cast(Entry, segment), EntryKind.DYNAMICS, where)
        if record is None:
            return None
        if not isinstance(segment, ChainSegment):
            segments = [r.cls.label() for r in registered(EntryKind.DYNAMICS) if issubclass(r.cls, ChainSegment)]
            self.refuse("chain.not-a-segment", RefusalClass.USER,
                        f"{where} is {record.cls.label()}, a dynamics that is not a chain segment",
                        "a chain holds segments (mime.registry.ChainSegment), which publish their frames' poses: "
                        + (", ".join(segments) or "none registered"))
            return None
        cls = cast(type[ChainSegment], record.cls)
        built, _ = self.planned_rung(cls, segment, where)
        settings = self.settings_in_si(segment, record, f"{where}'s dynamics")
        if not built or self.setting_problems(segment, settings, where) or not self.settings_complete(segment,
                                                                                                    settings):
            return None
        try:
            made = cast(object, segment.segment(settings))
            capabilities = segment.capabilities()
        except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
            self.defect(cls, f"segment() or capabilities() raised {error!r}")
            return None
        if not isinstance(made, Segment) or FRAME_POSES.name not in capabilities.publishes:
            self.defect(cls, "a chain segment's segment() returns a mime.registry.Segment, and it publishes "
                             f"{FRAME_POSES.name}")
            return None
        for problem in made.problems:
            uncatalogued = catalogue_problem(problem.code, RefusalClass.USER)
            if uncatalogued is not None:
                self.defect(cls, f"its segment's problem ({problem.message}) has a code it cannot be refused with: "
                                 f"{uncatalogued}")
                continue
            self.refuse(problem.code, RefusalClass.USER, f"{where} ({cls.label()}): {problem.message}", problem.action)
        if made.problems:
            return None
        plan = EntryPlan(Section.DYNAMICS, f"{apparatus.name}.{local}", segment, record, tuple(cls.nodes),
                         self.divider(segment.rate, f"{where}'s dynamics"), index, thing=apparatus.name,
                         thing_kind=ThingKind.APPARATUS, record=capabilities, settings=settings, position=position,
                         segment=made)
        if not self.check_locals(cls, plan.nodes) or not self.placement_holds(segment.orientation, where):
            return None
        if previous is None:
            if segment.mount is not None:
                self.refuse("chain.mount-first", RefusalClass.USER,
                            f"{where} is its chain's first segment and mounts on {segment.mount!r}: the first segment "
                            f"is placed in the world", "drop mount=, and place it with at= and orientation=")
                return None
            self.place_initially(plan, segment.placement(settings))
            return plan
        frames = previous.segment.frames if previous.segment is not None else ()
        mount = cast(object, segment.mount)
        if mount is None or mount not in frames:
            said = "names no frame to mount on, and MIME assumes none" if mount is None else \
                f"mounts on {mount!r}, which is no frame of {previous.owner!r}"
            self.refuse("chain.mount", RefusalClass.USER, f"{where} {said}: a segment mounts on a frame of the "
                                                         f"segment before it, {previous.owner!r}",
                        f"pass mount= one of its frames. {did_you_mean(str(mount), frames)}")
            return None
        port = cast(CapabilityRecord, previous.record).publishes[FRAME_POSES.name]
        index = frames.index(str(mount))
        start = self.initial_frames[previous.owner][index]
        plan.mount = ResolvedMount(previous.owner, str(mount), previous.node_name(port.node), port.field, index,
                                   len(frames), start)
        self.place_initially(plan, composed(start, segment.placement(settings)))
        return plan

    def place_initially(self, plan: EntryPlan, base: Sequence[float]) -> None:
        """The world pose of each of the segment's frames at the start: its base's, composed with each frame's pose in
        its base frame as the segment gives it (at its base where it gives none)."""
        assert plan.segment is not None
        local = plan.segment.initial or tuple((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) for _ in plan.segment.frames)
        self.initial_frames[plan.owner] = tuple(composed(base, pose) for pose in local)

    def placement_holds(self, orientation: object, where: str) -> bool:
        """Whether a placement's ``orientation`` is a unit quaternion ``(w, x, y, z)``, or ``None`` (no turn)."""
        if orientation is None:
            return True
        try:
            components = tuple(float(c) for c in cast(Any, orientation))
        except (TypeError, ValueError):
            components = ()
        if len(components) == 4 and abs(math.sqrt(sum(c * c for c in components)) - 1.0) <= _UNIT:
            return True
        self.refuse("chain.orientation", RefusalClass.USER,
                    f"{where}: orientation={orientation!r} is not a unit quaternion (w, x, y, z)",
                    "write the turn from the frame it is placed in as a unit quaternion (w, x, y, z)")
        return False

    def plan_apparatus_parts(self, apparatus: Apparatus) -> None:
        """Each part of the apparatus: its solid in SI, its mount (a frame of a segment of its chain, or the world),
        and its centroid there. Its material is not read: an apparatus's moment is declared."""
        segments = {plan.owner.partition(".")[2]: plan for plan in self.segments.get(apparatus.name, [])}
        frames = [f"{local}.{frame}" for local, plan in segments.items() for frame in
                  (plan.segment.frames if plan.segment is not None else ())]
        resolved = self.apparatus_parts.setdefault(apparatus.name, {})
        for key, part in sorted(apparatus.parts.items()):
            where = f"apparatus {apparatus.name!r}'s part {key!r}"
            held = cast(object, part)
            if not _IDENTIFIER.fullmatch(key) or not isinstance(held, Part):
                self.refuse("part.not-a-part", RefusalClass.USER, f"{where} is {held!r} under the name {key!r}",
                            "declare it as Part(<a solid>, mount=\"<segment>.<frame>\", at=...) under an identifier")
                continue
            if held.material is not None or held.inside is not None:
                self.refuse("part.apparatus-material", RefusalClass.USER,
                            f"{where} names a material or a part it lies inside: an apparatus's part is a solid fixed "
                            f"to a frame, its material not read (its moment is declared, as "
                            f"{apparatus.name}.magnetic_moment)", "drop material= and inside=")
                continue
            solid = self.apparatus_solid(held.geometry, where)
            if solid is None:
                continue
            label, radius = solid
            at = self.part_offset(held.at, where)
            if at is None:
                continue
            mount: ResolvedMount | None = None
            if held.mount is not None:
                local, _, frame = str(held.mount).partition(".")
                plan = segments.get(local)
                named = plan.segment.frames if plan is not None and plan.segment is not None else ()
                if plan is None or frame not in named or plan.record is None:
                    self.refuse("part.mount", RefusalClass.USER,
                                f"{where} is mounted on {held.mount!r}, which is no frame of a segment of the chain",
                                "mount it on <segment>.<frame>, or drop mount= to fix it in the world. "
                                + did_you_mean(str(held.mount), frames))
                    continue
                port = plan.record.publishes[FRAME_POSES.name]
                index = named.index(frame)
                mount = ResolvedMount(plan.owner, frame, plan.node_name(port.node), port.field, index, len(named),
                                      self.initial_frames[plan.owner][index])
            resolved[key] = ResolvedPart(apparatus.name, key, mount, (*at, 1.0, 0.0, 0.0, 0.0), radius, label)

    def apparatus_solid(self, geometry: object, where: str) -> tuple[str, float] | None:
        """A part's solid, checked, as its label and the radius of the smallest sphere about its centroid that
        holds it."""
        record = self.registered_as(cast(Entry, geometry), EntryKind.GEOMETRY, f"{where}'s solid")
        if record is None:
            return None
        if not isinstance(geometry, Solid):
            self.refuse("part.not-a-solid", RefusalClass.USER, f"{where} is a {record.cls.label()}, not a solid",
                        "give the part a solid: " + ", ".join(r.cls.label() for r in registered(EntryKind.GEOMETRY)
                                                              if issubclass(r.cls, Solid)))
            return None
        settings: dict[str, Magnitude] = {}
        for setting, declared in record.settings.items():
            if declared.kind is Direction:
                found = self.literal_direction(getattr(geometry, setting, None),
                                               f"{where}'s solid ({record.cls.label()}): setting {setting!r}")
                if found is not None:
                    settings[setting] = found
        settings.update(self.settings_in_si(geometry, record, f"{where}'s solid"))
        if self.setting_problems(geometry, settings, f"{where}'s solid") or not self.settings_complete(geometry,
                                                                                                       settings):
            return None
        try:
            radius = float(geometry.enclosing_radius(settings))
        except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
            self.defect(record.cls, f"enclosing_radius() raised {error!r}")
            return None
        if not (math.isfinite(radius) and radius > 0.0):
            self.defect(record.cls, f"enclosing_radius() gave {radius!r} m for settings it accepted")
            return None
        return record.cls.label(), radius

    def part_offset(self, at: object, where: str) -> tuple[float, float, float] | None:
        if at is None:
            return (0.0, 0.0, 0.0)
        try:
            value = si(at, LENGTH, what=f"{where}'s at=")
        except (UnitError, TypeError) as error:
            self.refuse("part.at", RefusalClass.USER, str(error),
                        "place the part's centroid with three lengths in the frame it is mounted on: at=mm((x, y, z))")
            return None
        if not (isinstance(value, tuple) and len(value) == 3 and all(math.isfinite(c) for c in value)):
            self.refuse("part.at", RefusalClass.USER, f"{where}'s at={at} is not a point (three lengths)",
                        "place the part's centroid with three lengths: at=mm((x, y, z))")
            return None
        return (value[0], value[1], value[2])

    def chain_load_refusal(self, apparatus: str) -> str | None:
        """Why the chain of ``apparatus`` takes no load, when none of its segments does; ``None`` otherwise."""
        reasons = [(plan.owner, cast(ChainSegment, plan.entry).load_refusal()) for plan in
                   self.segments.get(apparatus, [])]
        if not reasons or any(why is None for _, why in reasons):
            return None
        return "; ".join(f"{owner}: {why}" for owner, why in reasons)

    def settings_in_si(self, entry: Entry, record: Registered, where: str, *,
                       commanded: Dimension | None = None) -> dict[str, Magnitude]:
        """Each dimensioned setting of ``entry`` checked against the dimension its annotation declares and turned
        into SI; a setting of another dimension, or a bare number where a quantity is needed, is refused. A
        direction becomes a unit vector, and a drive's commanded quantity is checked against ``commanded``, the
        dimension of the command it drives. A geometry's own directions are left to :meth:`plan_geometry`."""
        values: dict[str, Magnitude] = {}
        for name, declared in record.settings.items():
            value: object = setting_value(entry, name)
            if value is None and declared.optional:
                continue
            what = f"{where} ({record.cls.label()}): setting {name!r}"
            if declared.kind is Direction:
                if not isinstance(entry, Geometry):
                    found = self.direction(value, what, type(entry))
                    if found is not None:
                        values[name] = found
                continue
            dimension = commanded if declared.kind is Commanded else declared.dimension
            if dimension is None:
                continue
            try:
                values[name] = si(value, dimension, what=what)
            except UnitError as error:
                self.refuse("setting.units", RefusalClass.USER, str(error),
                            f"write {name} as a quantity of {dimension}")
            except TypeError as error:
                self.refuse("setting.value", RefusalClass.USER, f"{what}: {error}",
                            f"write {name} as a quantity of {dimension}")
        return values

    def check_locals(self, cls: type, nodes: tuple[str, ...]) -> bool:
        if len(set(nodes)) != len(nodes) or not all(n == "" or _LOCAL.fullmatch(n) for n in nodes):
            self.defect(cls, f"its nodes {nodes!r} are not distinct lower-case local names")
            return False
        return True

    def plan_provider(self, apparatus: str, field_name: str, provider: FieldProvider, index: int) -> None:
        where = f"field {field_name!r} of apparatus {apparatus!r}"
        if not _IDENTIFIER.fullmatch(field_name):
            self.refuse("thing.field-name", RefusalClass.USER, f"{where}: the name is not an identifier",
                        "name fields with letters, digits and underscores")
            return
        record = self.registered_as(provider, EntryKind.FIELD_PROVIDER, where)
        if record is None:
            return
        try:
            capabilities = provider.capabilities()
        except Exception as error:  # noqa: BLE001
            self.defect(record.cls, f"capabilities() raised {error!r}")
            return
        path = f"{apparatus}.{field_name}"
        plan = EntryPlan(Section.PROVIDER, path, provider, record, ("",), 1, index, thing=apparatus,
                         thing_kind=ThingKind.APPARATUS, record=capabilities,
                         settings=self.settings_in_si(provider, record, where))
        self.setting_problems(provider, plan.settings, where)
        if not self.provider_parts(plan, where):
            return
        self.providers[path] = plan
        self.declared.append(plan)

    def provider_parts(self, plan: EntryPlan, where: str) -> bool:
        """Each part of its apparatus a provider's settings name, with the role it must have, resolved into
        ``plan.parts``; ``False`` once one is refused."""
        cls = cast(type[FieldProvider], plan.registered.cls)
        apparatus = cast(Apparatus, self.things[plan.thing or ""])
        resolved = self.apparatus_parts.get(apparatus.name, {})
        for setting, role in cls.parts.items():
            named = cast(object, getattr(plan.entry, setting, None))
            if not isinstance(named, str) or named not in apparatus.parts:
                self.refuse("provider.part", RefusalClass.USER,
                            f"{where} ({cls.label()}): {setting}={named!r} names no part of apparatus "
                            f"{apparatus.name!r}",
                            f"name one of its parts. {did_you_mean(str(named), apparatus.parts)}")
                return False
            held = apparatus.magnets.get(named)
            if role is not None and (held is None or getattr(held, "model", None) != role):
                self.refuse("provider.part", RefusalClass.USER,
                            f"{where} ({cls.label()}): its part {named!r} is not a {role.replace('_', ' ')}",
                            f"give the part its role: magnets={{{named!r}: PermanentMagnet(direction=...)}}")
                return False
            if named not in resolved:
                return False  # its part is refused already
            plan.parts[setting] = resolved[named]
        return True

    def plan_commands(self, apparatus: Apparatus, index: int) -> None:
        """A drive for every command of the apparatus's built providers, each refused when it drives no command,
        when its quantity is not the command's dimension, or when a command has none: MIME assumes no command's
        value. A provider that builds no node (nothing reads its field) takes no drive."""
        taken: dict[str, tuple[EntryPlan, Command]] = {}
        dropped: set[str] = set()
        for field_name, provider in apparatus.fields.items():
            path = f"{apparatus.name}.{field_name}"
            commands = type(provider).commands
            for command, spec in commands.items():
                if command in taken or command in dropped:
                    self.refuse("command.ambiguous", RefusalClass.USER,
                                f"apparatus {apparatus.name!r}: two of its fields take a command {command!r}",
                                "give the apparatus one field with that command")
                elif path in self.providers:
                    taken[command] = (self.providers[path], spec)
                else:
                    dropped.add(command)
        for plan in self.segments.get(apparatus.name, []):
            held: Mapping[str, Command] = plan.segment.commands if plan.segment is not None else {}
            for command, spec in held.items():
                taken[f"{plan.owner.partition('.')[2]}.{command}"] = (plan, spec)
        for command, drive in apparatus.commands.items():
            where = f"apparatus {apparatus.name!r}'s command {command!r}"
            if command not in taken:
                if command not in dropped:
                    self.refuse("command.unknown", RefusalClass.USER, f"{where}: no field of it takes that command",
                                did_you_mean(command, [*taken, *dropped]))
                continue
            record = self.registered_as(cast(Entry, drive), EntryKind.DRIVE, where)
            if record is None:
                continue
            provider, spec = taken[command]
            plan = EntryPlan(Section.DRIVE, f"drive:{apparatus.name}.{command}", cast(Drive, drive), record, ("",), 1,
                             index, thing=apparatus.name, thing_kind=ThingKind.APPARATUS,
                             settings=self.settings_in_si(cast(Drive, drive), record, where,
                                                          commanded=spec.dimension),
                             command=ResolvedCommand(provider.node_name(), spec.port, command,
                                                     unit_symbol(spec.dimension)))
            self.setting_problems(cast(Drive, drive), plan.settings, where)
            self.drives.append(plan)
            self.declared.append(plan)
        written = self.written.get(apparatus.name, {})
        for command, (label, path) in written.items():
            where = f"the controller's write {label!r}"
            if command not in taken:
                said = ("a command of a field nothing reads, so no node takes it" if command in dropped else
                        f"and apparatus {apparatus.name!r} takes no command {command!r}")
                self.refuse("controller.write-unknown", RefusalClass.USER, f"{where} names {path!r}, {said}",
                            did_you_mean(path, [f"{apparatus.name}.{c}" for c in taken]))
                continue
            if command in apparatus.commands:
                self.refuse("controller.written-twice", RefusalClass.USER,
                            f"apparatus {apparatus.name!r}'s command {command!r} has a drive, and {where} writes it "
                            f"too: a command has one writer, its drive or the controller, which replaces it",
                            f"remove {command!r} from the apparatus's commands, or {label!r} from the controller's "
                            f"writes")
                continue
            plan, spec = taken[command]
            self.writes[label] = ResolvedWrite(label, path, plan.node_name(), spec.port, unit_symbol(spec.dimension),
                                               spec.shape)
        for command in taken:
            if command not in apparatus.commands and command not in written:
                spec = taken[command][1]
                self.refuse("command.undriven", RefusalClass.USER,
                            f"apparatus {apparatus.name!r}'s command {command!r} has no drive, and MIME assumes no "
                            f"command's value", f"give it one: commands={{{command!r}: <a {spec.dimension} quantity, "
                                                f"or a drive>}}, or have the controller write it")

    # -- the controller ----------------------------------------------------------------------------------------

    def declare_controller(self) -> None:
        """The controller's declaration checked as data: one controller, registered, in a mode that is built, at a
        rate and a latency that are whole multiples of ``dt``, with every label it reads and writes bound once. Its
        write paths are parsed to the commands they name, which :meth:`plan_commands` resolves in place of their
        drives."""
        declared = [(i, d) for i, d in enumerate(self.declarations) if isinstance(d, ControllerDeclaration)]
        if not declared:
            return
        if len(declared) > 1:
            self.refuse("controller.more-than-one", RefusalClass.USER,
                        f"set_controller is called {len(declared)} times, and an experiment has one controller",
                        "declare one controller: it reads and writes as many labels as it needs")
            return
        index, held = declared[0]
        record = self.registered_as(cast(Entry, held.controller), EntryKind.CONTROLLER, "the controller")
        if record is None:
            return
        cls, entry = cast(type[Controller], record.cls), held.controller
        where = f"the controller {cls.label()}"
        self.mode = mode_of(held.mode)
        if self.mode is None:
            self.refuse("controller.mode-unknown", RefusalClass.USER, f"{where}: mode={held.mode!r} names no mode",
                        "use one of: " + "; ".join(f"{m.word!r} or {m.value} ({m.meaning})" for m in Mode))
        elif self.mode in UNBUILT:
            self.refuse("controller.mode-unbuilt", RefusalClass.USER,
                        f"{where}: mode {self.mode.value}, {self.mode.meaning}, is not built yet; it lands in "
                        f"{UNBUILT[self.mode]}",
                        f"run it as a callback for now: mode={Mode.CALLBACK.word!r} ({Mode.CALLBACK.value})")
        settings = self.settings_in_si(entry, record, where)
        self.setting_problems(entry, settings, where)
        self.latency = self.latency_steps(held.latency, where)
        self.controller = EntryPlan(Section.CONTROLLER, "controller", entry, record, (),
                                    self.divider(held.rate, where), index, settings=settings)
        reads = self.labels(held.reads, cls.reads, "reads", where)
        writes = self.labels(held.writes, cls.writes, "writes", where)
        self.controller.bindings = {**{f"read.{k}": v for k, v in reads.items()},
                                    **{f"write.{k}": v for k, v in writes.items()}}
        commands = {f"{a.name}.{c}": a.name for a in self.things.values() if isinstance(a, Apparatus)
                    for c in (*a.commands, *self.commands_of(a))}
        claimed: dict[tuple[str, str], str] = {}
        for label, path in writes.items():
            apparatus, _, command = path.partition(".")
            if not isinstance(self.things.get(apparatus), Apparatus) or not command:
                self.refuse("controller.write-unknown", RefusalClass.USER,
                            f"the controller's write {label!r} names {path!r}, which is no apparatus's command "
                            f"(written <apparatus>.<command>, a chain's as <apparatus>.<segment>.<command>)",
                            did_you_mean(path, commands))
                continue
            if (apparatus, command) in claimed:
                self.refuse("controller.written-twice", RefusalClass.USER,
                            f"the controller's writes {claimed[(apparatus, command)]!r} and {label!r} both write "
                            f"{path!r}: a command has one writer", "bind each write label to its own command")
                continue
            claimed[(apparatus, command)] = label
            self.written.setdefault(apparatus, {})[command] = (label, path)
        self.declared.append(self.controller)

    def commands_of(self, apparatus: Apparatus) -> list[str]:
        """The commands an apparatus's fields and chain segments take, by name, as phase 1 knows them so far."""
        found = [command for provider in apparatus.fields.values() for command in type(provider).commands]
        for plan in self.segments.get(apparatus.name, []):
            if plan.segment is not None:
                found += [f"{plan.owner.partition('.')[2]}.{command}" for command in plan.segment.commands]
        return found

    def labels(self, given: object, declared: Mapping[str, Any], kind: str, where: str) -> dict[str, str]:
        """``given``, the labels bound to paths, checked against the labels the controller declares under ``kind``:
        each bound once, to a path written as a string or a typed path."""
        if not isinstance(given, Mapping):
            self.refuse("controller.labels", RefusalClass.USER, f"{where}: {kind}={given!r} is not a mapping of its "
                                                                 f"labels to what they {kind[:-1]}",
                        f"pass {kind}={{label: path}} for its labels: {', '.join(sorted(declared)) or 'none'}")
            return {}
        bound: dict[str, str] = {}
        for label, path in cast(Mapping[object, object], given).items():
            if label not in declared:
                self.refuse("controller.labels", RefusalClass.USER, f"{where} {kind[:-1]}s no label {label!r}",
                            did_you_mean(str(label), declared))
            elif not isinstance(path, (str, TypedPath, Handle)):
                self.refuse("controller.labels", RefusalClass.USER,
                            f"{where}: {kind[:-1]} {label!r} is bound to {path!r}, not a path",
                            "bind it to a path, written as a string such as \"umr.pose\" or a typed path")
            else:
                bound[cast(str, label)] = binding_name(path)
        missing = sorted(set(declared) - set(cast(Mapping[object, object], given)))
        if missing:
            self.refuse("controller.labels", RefusalClass.USER,
                        f"{where} {kind} {', '.join(repr(m) for m in missing)}, and nothing is bound to "
                        f"{'it' if len(missing) == 1 else 'them'}: MIME binds nothing by default",
                        f"bind each in {kind}={{label: path}}")
        return bound

    def latency_steps(self, latency: object, where: str) -> int:
        """``latency`` as a whole number of base steps, none for ``None``; or a refusal naming the nearest."""
        if latency is None or not self.dt:
            return 0
        try:
            seconds = si(latency, TIME, what=f"the latency of {where}")
        except UnitError as error:
            self.refuse("controller.latency", RefusalClass.USER, str(error), "write a latency as a time, such as ms(2)")
            return 0
        ratio = seconds / self.dt if isinstance(seconds, float) else math.nan
        whole = round(ratio) if math.isfinite(ratio) else -1
        if whole >= 0 and abs(ratio - whole) <= _RATE_RTOL * max(ratio, 1.0):
            return whole
        nearest = sorted({max(0, math.floor(ratio)), max(0, math.ceil(ratio))}) if math.isfinite(ratio) else [0]
        self.refuse("controller.latency", RefusalClass.USER,
                    f"the latency of {where} is {latency}, not a whole number of base steps dt = {_seconds(self.dt)} "
                    f"from zero on", "use a whole number of steps; the nearest are "
                                     + " and ".join(_seconds(k * self.dt) for k in nearest))
        return 0

    def plan_controller(self) -> ControllerPlan | None:
        """The controller's reads, each resolved to the node field the run reads and classed, and its writes, each
        checked against the command it replaces the drive of; then the reads a profile refuses."""
        plan = self.controller
        if plan is None:
            return None
        cls = cast(type[Controller], plan.registered.cls)
        where = f"the controller {cls.label()}"
        reads: list[ClassedRead] = []
        for label, signal in sorted(cls.reads.items()):
            path = plan.bindings.get(f"read.{label}")
            found = self.controller_read(label, signal.units, signal.shape, path, where) if path else None
            if found is not None:
                reads.append(found)
        writes: list[ResolvedWrite] = []
        for label, signal in sorted(cls.writes.items()):
            write = self.writes.get(label)
            if write is None:
                continue
            if signal.units != write.units or not _fits(signal.shape, write.shape):
                self.refuse("controller.signal", RefusalClass.USER,
                            f"{where} writes {label!r} in {signal.units!r} with shape {signal.shape}, and "
                            f"{write.path} takes {write.units!r} with shape {write.shape}",
                            f"bind {label!r} to a command in its units and shape")
                continue
            writes.append(write)
        for read, reason in refused_reads(None, reads):
            self.refuse("controller.read-refused", RefusalClass.USER,
                        f"{where}'s read {read.label!r} ({read.path}, {read.read_class.value}) is refused: {reason}",
                        "read an observation's output instead")
        if self.mode is None or len(writes) != len(cls.writes) or len(reads) != len(cls.reads):
            return None
        return ControllerPlan(plan.owner, cast(Controller, plan.entry), plan.registered, dict(plan.settings), self.mode,
                              plan.divider, self.latency, tuple(reads), tuple(writes))

    def controller_read(self, label: str, units: str, shape: tuple[int, ...], path: str,
                        where: str) -> ClassedRead | None:
        """A read label's path resolved: ``<thing>.<state>`` or ``<apparatus>.<segment>.<state>``, a state interface
        its dynamics publishes, classed state; or ``<observation id>.<field>``, an observation's output, classed
        observation, its field checked once its node is built."""
        observations = {p.owner: p for p in self.readers if p.section is Section.OBSERVATION}
        states = self.state_paths()
        if ":" in path:
            owner, _, output = path.rpartition(".")
            held = observations.get(owner)
            if held is None or "" not in held.nodes or not output:
                self.refuse("controller.read-unknown", RefusalClass.USER,
                            f"{where}'s read {label!r} names {path!r}, which is no observation's output "
                            f"(written <observation id>.<field>)",
                            did_you_mean(owner, observations))
                return None
            return ClassedRead(label, path, held.node_name(), output, ReadClass.OBSERVATION,
                               f"observation {owner}'s {output}", units, shape)
        found = states.get(path)
        if found is None:
            self.refuse("controller.read-unknown", RefusalClass.USER,
                        f"{where}'s read {label!r} names {path!r}, which is no state a dynamics publishes nor an "
                        f"observation's output", did_you_mean(path, [*states, *(f"{o}.<field>" for o in observations)]))
            return None
        dynamics, interface = found
        entry = vocabulary.lookup(interface)
        port = cast(CapabilityRecord, dynamics.record).publishes[interface]
        if entry is None or entry.units != units or entry.shape is None or not _fits(shape, entry.shape):
            held_units, held_shape = (entry.units, entry.shape) if entry is not None else ("?", None)
            self.refuse("controller.signal", RefusalClass.USER,
                        f"{where} reads {label!r} in {units!r} with shape {shape}, and {path} ({interface}) is in "
                        f"{held_units!r} with shape {held_shape}", f"bind {label!r} to a state in its units and shape")
            return None
        of = dynamics.owner if dynamics.segment is not None else dynamics.thing
        return ClassedRead(label, path, dynamics.node_name(port.node), port.field, ReadClass.STATE,
                           f"{interface} of {of}", units, shape)

    def state_paths(self) -> dict[str, tuple[EntryPlan, str]]:
        """Every state a controller may read, by its path: each interface a thing's dynamics publishes as
        ``<thing>.<leaf>``, and a chain segment's as ``<apparatus>.<segment>.<leaf>``, with the dynamics and the
        interface."""
        found: dict[str, tuple[EntryPlan, str]] = {}
        held = [(plan.thing or "", plan) for plan in self.dynamics.values()]
        held += [(plan.owner, plan) for plans in self.segments.values() for plan in plans]
        for prefix, plan in held:
            if plan.record is None:
                continue
            for interface in plan.record.publishes:
                found[f"{prefix}.{interface.rpartition('.')[2]}"] = (plan, interface)
        return found

    # -- bindings -------------------------------------------------------------------------------------------

    def resolve(self, where: str, role: str, kind: RoleKind, value: Any, reader: type[Entry]) -> str | None:
        if kind is RoleKind.SURFACE:
            return self.geometry_part(value, SurfacePath, "surfaces", f"{where}: {role}=", reader)
        if kind is RoleKind.FIELD:
            path = str(value) if isinstance(value, (FieldPath, str)) else None
            if isinstance(value, Handle) or path is None:
                self.refuse("binding.wrong-kind", RefusalClass.USER,
                            f"{where}: {role}= binds a field, written apparatus.field, not {value!r}",
                            f"bind {role}= one of: {', '.join(sorted(self.providers)) or 'no field is declared'}")
                return None
            if path not in self.providers:
                self.refuse("binding.unknown", RefusalClass.USER, f"{where}: {role}={path!r} names no field",
                            did_you_mean(path, self.providers))
                return None
            return path
        if isinstance(value, TypedPath):
            self.refuse("binding.wrong-kind", RefusalClass.USER,
                        f"{where}: {role}= binds a {kind.value}, not the path {value}",
                        f"bind {role}= one of: {', '.join(self.names_of(kind)) or f'no {kind.value} is declared'}")
            return None
        name = binding_name(value)
        thing = self.things.get(name)
        if thing is None:
            self.refuse("binding.unknown", RefusalClass.USER, f"{where}: {role}={name!r} names no thing",
                        did_you_mean(name, self.names_of(kind)))
            return None
        if thing.kind is not kind.thing_kind:
            self.refuse("binding.wrong-kind", RefusalClass.USER,
                        f"{where}: {role}={name!r} is a {thing.kind.value}, and {role}= binds a {kind.value}",
                        f"bind {role}= one of: {', '.join(self.names_of(kind)) or f'no {kind.value} is declared'}")
            return None
        return name

    def names_of(self, kind: RoleKind) -> list[str]:
        if kind is RoleKind.FIELD:
            return sorted(self.providers)
        if kind is RoleKind.SURFACE:
            return sorted(f"{owner}.{surface}" for owner, plan in self.geometries.items()
                          for surface in type(plan.geometry).surfaces)
        return sorted(n for n, t in self.things.items() if t.kind is kind.thing_kind)

    def record_of(self, kind: RoleKind, name: str) -> CapabilityRecord | None:
        if kind is RoleKind.FIELD:
            return self.providers[name].record
        if kind is RoleKind.SURFACE:
            name = name.partition(".")[0]
        plan = self.dynamics.get(name)
        return plan.record if plan else None

    # -- interactions and observations --------------------------------------------------------------------

    def plan_reader(self, entry: Interaction | Observation, given: dict[str, Any], index: int,
                    section: Section) -> None:
        kind = EntryKind.INTERACTION if section is Section.INTERACTION else EntryKind.OBSERVATION
        record = self.registered_as(entry, kind, f"{kind.value} #{index + 1}")
        if record is None:
            return
        cls = cast(type[Interaction] | type[Observation], record.cls)
        owner = entry_id(cls, given)
        where = f"{kind.value} {owner}"
        settings = self.settings_in_si(entry, record, where)
        bindings: dict[str, str] = {}
        complete = True
        for role in given:
            if role not in cls.roles:
                self.refuse("binding.unknown-role", RefusalClass.USER, f"{where} has no role {role!r}",
                            did_you_mean(role, cls.roles))
                complete = False
        for role, role_kind in cls.roles.items():
            value = given.get(role)
            if value is None:
                candidates = ", ".join(self.names_of(role_kind)) or f"no {role_kind.value} is declared"
                self.refuse("binding.missing", RefusalClass.USER,
                            f"{where} needs {role}=, and MIME binds nothing by default",
                            f"bind {role}= one of: {candidates}")
                complete = False
                continue
            name = self.resolve(where, role, role_kind, value, cls)
            if name is None:
                complete = False
            else:
                bindings[role] = name
        built, rung = self.planned_rung(cls, entry, where)
        if not (complete and built):
            return
        self.setting_problems(entry, settings, where)
        records = {role: self.record_of(cls.roles[role], name) for role, name in bindings.items()}
        if section is Section.INTERACTION and not self.supported(cast(type[Interaction], cls), where, rung, records):
            return
        divider = self.divider(getattr(entry, "rate", None), where) if section is Section.OBSERVATION else 1
        up, gravity = (self.frame.up, self.frame.gravity) if self.frame else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        declared = {(role, path.partition(".")[2]): value for role, name in bindings.items()
                    if cls.roles[role].binds_a_thing for path, value in self.declared_of(name).items()}
        regions = {role: region for role, name in bindings.items() if cls.roles[role] is RoleKind.MEDIUM
                   for region in (self.region_of(name),) if region is not None}
        hulls = {role: hull for role, name in bindings.items() if cls.roles[role] is RoleKind.BODY
                 for hull in (self.hull_of(name),) if hull is not None}
        context = PlanContext(self.dt, up, gravity, dict(bindings), records, dict(settings), declared, regions,
                              hulls)
        try:
            plan = cast(object, entry.plan(context))  # an entry's own code, checked like a script's
        except Exception as error:  # noqa: BLE001
            if self.settings_complete(entry, settings):
                self.defect(cls, f"plan() raised {error!r}")
            return
        if not isinstance(plan, Plan):
            self.defect(cls, f"plan() returned {type(plan).__name__}, not a mime.registry.Plan")
            return
        if not self.check_locals(cls, tuple(plan.nodes)):
            return
        for problem in plan.problems:
            uncatalogued = catalogue_problem(problem.code, RefusalClass.USER)
            if uncatalogued is not None:
                self.defect(cls, f"its plan's problem ({problem.message}) has a code it cannot be refused with: "
                                 f"{uncatalogued}")
                continue
            self.refuse(problem.code, RefusalClass.USER, f"{where} ({cls.label()}): {problem.message}",
                        problem.action)
        entry_plan = EntryPlan(section, owner, entry, record, tuple(plan.nodes), divider, index, bindings=bindings,
                               plan=plan, settings=settings, regions=regions, hulls=hulls)
        if section is Section.OBSERVATION and (plan.delivers or plan.registrations or plan.groups):
            self.defect(cls, "an observation reads only: it delivers into nothing, registers nothing and asks "
                             "for no coupling group")
            return
        self.plan_reads(entry_plan, where)
        for label, delivery in plan.delivers.items():
            target = self.target(entry_plan, where, delivery)
            if target is not None:
                entry_plan.delivers[label] = target
        for registration in plan.registrations:
            self.plan_registration(entry_plan, where, registration)
        for request in plan.groups:
            self.check_request(entry_plan, request)
        self.readers.append(entry_plan)
        self.declared.append(entry_plan)

    def supported(self, cls: type[Interaction], where: str, rung: str | None,
                  records: Mapping[str, CapabilityRecord | None]) -> bool:
        rows = [row for row in cls.support if row.rung in (None, rung)]
        if not cls.support or any(all(test(records.get(role)) for role, test in row.when.items()) for row in rows):
            return True
        bound = "; ".join(f"{role}: {rec.describe() if rec else 'no dynamics'}" for role, rec in records.items())
        options: list[str] = []
        for row in rows:
            needs = "; ".join(f"{role} needs {test} (qualifying: {_qualifying(_offers(cls.roles[role]), test)})"
                              for role, test in row.when.items())
            options.append(f"{'rung ' + row.rung + ': ' if row.rung else ''}{needs}")
        self.refuse("support.unsupported", RefusalClass.USER,
                    f"{where}{f' with rung {rung!r}' if rung else ''} does not support what it is bound to ({bound})",
                    "the supported constructions are: " + " | ".join(options or ["none for this rung"]))
        return False

    def interface(self, cls: type[Entry], name: str, kinds: tuple[type, ...], what: str) -> vocabulary.Interface | None:
        entry = vocabulary.lookup(name)
        if entry is None:
            self.defect(cls, f"it {what} {name!r}, which is not in the vocabulary")
            return None
        if entry.reserved:
            self.refuse("vocabulary.reserved", RefusalClass.USER,
                        f"{cls.label()} {what} {name!r}, which is reserved, not built: its kind "
                        f"({entry.kind.value}), units ({entry.units}), frame ({entry.frame}) and sign are fixed and "
                        f"nothing builds it yet",
                        f"choose an entry that does not use {name!r} until it is built")
            return None
        if not isinstance(entry.kind, kinds):
            self.defect(cls, f"it {what} {name!r}, which is a {entry.kind.value} interface")
            return None
        return entry

    def state_read(self, plan: EntryPlan, where: str, read: Read) -> ResolvedRead | None:
        cls = plan.reader_cls()
        role_kind = cls.roles.get(read.role)
        if role_kind is None or role_kind is RoleKind.FIELD:
            self.defect(cls, f"it reads state through role {read.role!r}, which is not one of its thing roles")
            return None
        entry = self.interface(cls, read.interface, (ReadKind,), "reads")
        if entry is None:
            return None
        if entry.kind is not ReadKind.STATE:
            self.defect(cls, f"it reads the field {read.interface!r} as state; read it through a field role")
            return None
        thing = plan.bindings[read.role]
        dynamics = self.dynamics.get(thing)
        port = dynamics.record.publishes.get(read.interface) if dynamics and dynamics.record else None
        if dynamics is None or port is None:
            has = f"its dynamics {dynamics.describe()}" if dynamics else "it has no dynamics, so it"
            self.refuse("read.not-published", RefusalClass.USER,
                        f"{where} reads {read.interface} from {role_kind.value} {thing!r}, and {has} "
                        f"does not publish it",
                        f"give {thing!r} a dynamics that publishes {read.interface}: "
                        + _qualifying(EntryKind.DYNAMICS, lambda r: read.interface in r.publishes))
            return None
        return ResolvedRead(dynamics.node_name(port.node), port.field, read.interface, entry.units)

    def plan_reads(self, plan: EntryPlan, where: str) -> None:
        cls = plan.reader_cls()
        assert plan.plan is not None
        sources: dict[tuple[str, ResolvedRead | None], str] = {}
        for label in sorted(plan.plan.reads):
            read = plan.plan.reads[label]
            if cls.roles.get(read.role) is not RoleKind.FIELD:
                if read.at is not None:
                    self.defect(cls, f"read {label!r}: only a field is read at points")
                    continue
                resolved = self.state_read(plan, where, read)
                if resolved is not None:
                    plan.reads[label] = resolved
                continue
            entry = self.interface(cls, read.interface, (ReadKind,), "reads")
            if entry is None:
                continue
            if entry.kind is not ReadKind.FIELD:
                self.defect(cls, f"it reads the state {read.interface!r} through field role {read.role!r}")
                continue
            path = plan.bindings[read.role]
            provider = self.providers[path]
            uniform = provider.record is not None and Capability.UNIFORM in provider.record.flags
            if read.at is None and not uniform:
                self.defect(cls, f"read {label!r} of field {read.interface!r} gives no points (at=), and its provider "
                                 f"{provider.describe()} is not uniform")
                continue
            prefix = provider.record.publishes.get(read.interface) if provider.record else None
            if prefix is None:
                self.refuse("read.not-published", RefusalClass.USER,
                            f"{where} reads {read.interface} from field {path!r}, whose provider "
                            f"{provider.describe()} does not publish it",
                            f"bind a field whose provider publishes {read.interface}: "
                            + _qualifying(EntryKind.FIELD_PROVIDER,
                                          lambda r, name=read.interface: name in r.publishes))
                continue
            if plan.section is Section.OBSERVATION:
                self.refuse("observation.field-read", RefusalClass.USER,
                            f"{where} reads the field {path!r}; an observation reads a field through its own "
                            f"instance of the provider, which is not built yet",
                            "observe the state the field acts on instead, until observations' own providers land")
                continue
            points = None if uniform or read.at is None else self.state_read(plan, where, read.at)
            if points is None and not uniform:
                continue
            key = (path, points)
            slot = sources.setdefault(key, _sanitize(plan.owner) + (f"_{len(sources)}" if sources else ""))
            plan.reads[label] = ResolvedRead(provider.owner, f"{prefix.field}_{slot}", read.interface, entry.units,
                                             points, slot)

    def target(self, plan: EntryPlan, where: str, delivery: Delivery) -> ResolvedTarget | None:
        cls = plan.reader_cls()
        role_kind = cls.roles.get(delivery.role)
        if role_kind is None or role_kind is RoleKind.FIELD:
            self.defect(cls, f"it delivers through role {delivery.role!r}, which is not one of its thing roles")
            return None
        refusal = self.chain_load_refusal(plan.bindings[delivery.role]) if role_kind is RoleKind.APPARATUS else None
        if refusal is not None:
            self.refuse("target.kinematic-chain", RefusalClass.USER,
                        f"{where} delivers {delivery.target} into apparatus {plan.bindings[delivery.role]!r}, whose "
                        f"chain takes no load: {refusal}",
                        "the dynamic rung of a chain takes loads, and is reserved, not built: until it lands, attach "
                        "no law that loads the chain (a load would not move a prescribed motion)")
            return None
        entry = self.interface(cls, delivery.target, (TargetKind,), "delivers into")
        if entry is None:
            return None
        kind = entry.kind
        if not isinstance(kind, TargetKind):
            self.defect(cls, f"it delivers into {delivery.target!r}, which is a {kind.value} interface")
            return None
        if kind is TargetKind.REGISTRATION:
            self.defect(cls, f"it delivers into the registration {delivery.target!r}; plan a Registration instead")
            return None
        if kind is TargetKind.INEQUALITY_CONSTRAINT:
            self.defect(cls, f"it delivers into the inequality constraint {delivery.target!r}; the dynamics takes one "
                             f"at construction: plan a Confinement instead")
            return None
        thing = plan.bindings[delivery.role]
        dynamics = self.dynamics.get(thing)
        port = dynamics.record.accepts.get(delivery.target) if dynamics and dynamics.record else None
        if dynamics is None or port is None:
            has = f"its dynamics {dynamics.describe()}" if dynamics else "it has no dynamics, so it"
            self.refuse("target.not-accepted", RefusalClass.USER,
                        f"{where} delivers into {delivery.target} of {role_kind.value} {thing!r}, and {has} does "
                        f"not accept it",
                        f"give {thing!r} a dynamics that accepts {delivery.target}: "
                        + _qualifying(EntryKind.DYNAMICS, lambda r: delivery.target in r.accepts))
            return None
        return ResolvedTarget(dynamics.node_name(port.node), port.field, kind, thing, delivery.target, entry.units,
                              plan.owner)

    def plan_registration(self, plan: EntryPlan, where: str, registration: Registration) -> None:
        cls = plan.reader_cls()
        for role in (registration.role, registration.subject):
            if cls.roles.get(role) in (None, RoleKind.FIELD):
                self.defect(cls, f"its registration names role {role!r}, which is not one of its thing roles")
                return
        entry = self.interface(cls, registration.target, (TargetKind,), "registers through")
        if entry is None:
            return
        if entry.kind is not TargetKind.REGISTRATION:
            self.defect(cls, f"its registration target {registration.target!r} is a {entry.kind.value} target")
            return
        receiver = plan.bindings[registration.role]
        dynamics = self.dynamics.get(receiver)
        record = dynamics.record if dynamics else None
        if dynamics is None or record is None or registration.target not in record.accepts \
                or registration.kind not in record.registration_kinds:
            has = f"its dynamics {dynamics.describe()}" if dynamics else "it has no dynamics, so it"
            self.refuse("registration.not-accepted", RefusalClass.USER,
                        f"{where} registers {plan.bindings[registration.subject]!r} with {receiver!r} as "
                        f"{registration.kind}, and {has} does not accept that registration",
                        f"give {receiver!r} a dynamics that accepts {registration.kind}: "
                        + _qualifying(EntryKind.DYNAMICS, lambda r: registration.kind in r.registration_kinds))
            return
        reads = {label: self.state_read(plan, where, Read(registration.subject, interface))
                 for label, interface in registration.reads.items()}
        returns = {label: self.target(plan, where, Delivery(registration.subject, target))
                   for label, target in registration.returns.items()}
        resolved_reads = {label: read for label, read in reads.items() if read is not None}
        resolved_returns = {label: target for label, target in returns.items() if target is not None}
        if len(resolved_reads) != len(reads) or len(resolved_returns) != len(returns):
            return
        subject = plan.bindings[registration.subject]
        dynamics.registrations.append(
            ResolvedRegistration(plan.owner, registration.kind, subject, resolved_reads, resolved_returns))

    def check_request(self, plan: EntryPlan, request: GroupRequest) -> None:
        cls = plan.reader_cls()
        tolerance, cap = cast(object, request.tolerance), cast(object, request.max_iterations)  # an entry's own values
        if not (isinstance(tolerance, float) and math.isfinite(tolerance) and tolerance > 0.0):
            self.defect(cls, f"its coupling-group request gives tolerance {tolerance!r}, not a float above zero")
        if not (isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1):
            self.defect(cls, f"its coupling-group request gives max_iterations {cap!r}, not an integer >= 1")
        for role in request.roles:
            if cls.roles.get(role) in (None, RoleKind.FIELD):
                self.defect(cls, f"its coupling-group request names role {role!r}, not one of its thing roles")
            elif plan.bindings[role] not in self.dynamics:
                self.refuse("group.no-dynamics", RefusalClass.USER,
                            f"{plan.owner} asks to solve {plan.bindings[role]!r} together with it, and "
                            f"{plan.bindings[role]!r} has no dynamics",
                            f"give {plan.bindings[role]!r} a dynamics, or bind another thing")

    # -- values of things, and confinements -------------------------------------------------------------------

    def plan_values(self) -> None:
        """Each value declared in the experiment checked against the things declared: a known value of the thing,
        in its dimension and shape, with a reason, declared once."""
        kinds = {name: thing.kind for name, thing in self.things.items()}
        parts = {name: frozenset(thing.parts) for name, thing in self.things.items() if isinstance(thing, Body)}
        for declaration in self.declarations:
            if not isinstance(declaration, ValueDeclaration):
                continue
            written = cast(object, declaration.path)  # a script is not type-checked
            path = str(written) if isinstance(written, (TypedPath, str)) else repr(written)
            if path in self.values:
                self.refuse("value.duplicate", RefusalClass.USER, f"the value {path} is declared twice",
                            "declare each value once")
                continue
            if not isinstance(declaration.value, Quantity):
                self.refuse("value.units", RefusalClass.USER,
                            f"the declared value {path} is {declaration.value!r}, not a quantity",
                            "declare it as a quantity with its units, such as kg(8.5e-5)")
                continue
            held = materials.Declared(path, declaration.value, cast(str, declaration.reason))
            problems = materials.declaration_problems(held, kinds, self.directed(), parts)
            for code, message, action in problems:
                self.refuse(code, RefusalClass.USER, message, action)
            if not problems and not self.given_by_kind(path):
                self.values[path] = held

    def hull_of(self, body: str) -> Hull | None:
        """The hull of ``body``: its one part outside every other, when that part's kind gives its radial reach; the
        build's other solids give none yet, and a union of several parts is not built."""
        bulk = [line for line in self.parts.get(body, {}).values() if line.inside is None]
        if len(bulk) != 1 or bulk[0].reach is None:
            return None
        line = bulk[0]
        return Hull(body, line.name, line.solid, line.settings, line.centre, line.rotation)

    def given_by_kind(self, path: str) -> bool:
        """Whether the declared value at ``path`` is one its body's hull kind gives (its extent, the radial reach,
        and the shape values a law reads from the kind), which is refused: the kind is the one authority for it."""
        body, _, name = path.partition(".")
        hull = self.hull_of(body)
        if hull is None or name not in FROM_THE_KIND:
            return False
        given = "its extent is the hull's radial reach" if name == "extent" else \
            f"{name} is a parameter of the hull's kind, which a law reads from the kind"
        self.refuse("value.from-kind", RefusalClass.USER,
                    f"the value {path} is declared, and {hull.label()} gives it: {given}",
                    f"remove the declaration; set the kind's settings on {body}'s part {hull.part!r} instead")
        return True

    def resolution(self, path: str) -> materials.Resolution:
        if self.resolver is None:
            return materials.Resolution(None, blocked=True)
        return self.resolver.value(path)

    def declared_of(self, thing: str) -> dict[str, Magnitude]:
        """The values ``thing`` has, in SI, by path, from wherever each comes (a declaration, a record, its parts):
        what a plan reads through ``PlanContext.value``. A value that cannot be resolved is left out here and refused
        once, where an entry reads it."""
        held = self.things.get(thing)
        if held is None:
            return {}
        found: dict[str, Magnitude] = {}
        for name in materials.VALUE_KINDS.get(held.kind, {}):
            resolved = self.resolution(f"{thing}.{name}").value
            if resolved is not None:
                found[resolved.path] = resolved.value
        return found

    def refuse_problems(self, problems: Sequence[tuple[str, str, str]]) -> None:
        """The resolver's problems, each refused once, however many values it keeps from being resolved."""
        for problem in problems:
            if problem not in self.refused_problems:
                self.refused_problems.add(problem)
                self.refuse(problem[0], RefusalClass.USER, problem[1], problem[2])

    def value(self, plan: EntryPlan, label: str, thing: str, name: str) -> None:
        held = self.things.get(thing)
        if held is None:
            return
        cls = plan.registered.cls
        if materials.value_kind(held.kind, name) is None:
            self.defect(cls, f"it reads the value {name!r} of {held.kind.value} {thing!r}, which a {held.kind.value} "
                             f"does not have")
            return
        path = f"{thing}.{name}"
        found = self.resolution(path)
        self.refuse_problems(found.problems)
        if found.value is None:
            if found.problems or found.blocked:
                return
            directed = self.directed()
            meaning = materials.VALUE_KINDS[held.kind][name]
            quantity = f"<a quantity of {meaning.dimension}>"
            if path in directed:
                quantity = f"<its magnitude, a quantity of {meaning.dimension}; its direction is {directed[path][1]}>"
            sources = {ThingKind.MEDIUM: "it has no material record that holds it",
                       ThingKind.BODY: "it declares no parts it could be derived from"}.get(held.kind, "")
            self.refuse("value.undeclared", RefusalClass.USER,
                        f"{plan.owner} reads {path}, {meaning.meaning}, and nothing provides it: "
                        f"{sources + ', and ' if sources else ''}the experiment declares no value",
                        f"declare it with its reason: exp.declare({path!r}, {quantity}, reason=\"...\")")
            return
        plan.values[label] = found.value

    def resolve_values(self) -> None:
        for plan in self.dynamics.values():
            for label, name in cast(type[Dynamics], plan.registered.cls).values.items():
                self.value(plan, label, plan.thing or "", name)
        for plan in self.providers.values():
            for label, name in cast(type[FieldProvider], plan.registered.cls).values.items():
                self.value(plan, label, plan.thing or "", name)
        for plan in self.readers:
            assert plan.plan is not None
            cls = plan.reader_cls()
            for label, read in plan.plan.values.items():
                role = cls.roles.get(read.role)
                if role is None or not role.binds_a_thing:
                    self.defect(plan.registered.cls, f"it reads a value through role {read.role!r}, which binds no "
                                                     f"thing")
                    continue
                self.value(plan, label, plan.bindings[read.role], read.name)

    def plan_confinements(self) -> None:
        for plan in self.readers:
            assert plan.plan is not None
            for confinement in plan.plan.confines:
                self.plan_confinement(plan, confinement)
        for plan in self.dynamics.values():
            if len(plan.confinements) > 1:
                self.refuse("confinement.more-than-one", RefusalClass.USER,
                            f"body {plan.thing!r} is confined by {', '.join(c.id for c in plan.confinements)}; at most "
                            f"one confinement per body is built (sequential projections depend on their order)",
                            "keep one of them")

    def plan_confinement(self, plan: EntryPlan, confinement: Confinement) -> None:
        cls = plan.reader_cls()
        where = f"interaction {plan.owner}"
        declared_values: Mapping[str, Value] = plan.plan.values if plan.plan is not None else {}
        if (cls.roles.get(confinement.role) is not RoleKind.BODY or cls.roles.get(confinement.surface)
                is not RoleKind.SURFACE or confinement.extent not in declared_values):
            self.defect(cls, "its confinement names a body role, a surface role and the label of the body's extent "
                             "among its values")
            return
        entry = self.interface(cls, confinement.target, (TargetKind,), "confines through")
        if entry is None:
            return
        if entry.kind is not TargetKind.INEQUALITY_CONSTRAINT:
            self.defect(cls, f"its confinement target {confinement.target!r} is a {entry.kind.value} target")
            return
        body, surface = plan.bindings[confinement.role], plan.bindings[confinement.surface]
        owner, _, part = surface.partition(".")
        moving = self.things.get(owner)
        if isinstance(moving, Body) or (moving is not None and moving.dynamics is not None):
            self.refuse("confinement.moving-surface", RefusalClass.USER,
                        f"{where} projects {body!r} against {surface}, a surface of {moving.kind.value} {owner!r}, "
                        f"which moves: projection between two moving things is reserved (a constraint across several "
                        f"bodies), not built", "bind the surface of a fixed structure")
            return
        dynamics = self.dynamics.get(body)
        if dynamics is None or dynamics.record is None or confinement.target not in dynamics.record.accepts:
            has = f"its dynamics {dynamics.describe()}" if dynamics else "it has no dynamics, so it"
            self.refuse("target.not-accepted", RefusalClass.USER,
                        f"{where} confines body {body!r}, and {has} does not accept {confinement.target}",
                        f"give {body!r} a dynamics that accepts {confinement.target}: "
                        + _qualifying(EntryKind.DYNAMICS, lambda r: confinement.target in r.accepts))
            return
        if confinement.rule not in VELOCITY_RULES:
            self.refuse("confinement.rule", RefusalClass.USER,
                        f"{where}: the velocity rule {confinement.rule!r} is not one of the built rules",
                        did_you_mean(confinement.rule, VELOCITY_RULES))
            return
        extent = plan.values.get(confinement.extent)
        if extent is None:
            return  # refused as undeclared
        geometry = self.geometries[owner]
        shape = type(geometry.geometry).surfaces[part]
        try:
            parameters = geometry.geometry.surface_parameters(part, geometry.settings)
        except Exception as error:  # noqa: BLE001
            self.defect(geometry.registered.cls, f"surface_parameters({part!r}) raised {error!r}")
            return
        dynamics.confinements.append(ResolvedConfinement(plan.owner, body, surface, shape, parameters,
                                                         confinement.margin, cast(float, extent.value),
                                                         confinement.rule))

    def value_lines(self) -> tuple[ValueLine, ...]:
        """Every value an entry reads, with its readers; every value declared in the experiment that none reads; and
        every value of a part the part resolver took, read by that body's resolver."""
        readers: dict[str, list[str]] = {}
        resolved: dict[str, materials.ResolvedValue] = {}
        for plan in self.declared:
            for value in plan.values.values():
                readers.setdefault(value.path, []).append(plan.owner)
                resolved[value.path] = value
        lines = [ValueLine(path, value.value, value.unit, value.source, value.reason,
                           tuple(sorted(set(readers[path]))), value.details) for path, value in sorted(resolved.items())]
        listed = set(resolved)
        if self.resolver is not None:
            for body in sorted(self.parts):
                if not any(path.startswith(f"{body}.") for path in listed):
                    continue
                for path, held in sorted(self.resolver.inputs_of(body).items()):
                    if held.value is not None and path not in listed:
                        listed.add(path)
                        lines.append(ValueLine(path, held.value.value, held.value.unit, held.value.source,
                                               held.value.reason, (f"{body}'s part resolver",), held.value.details))
        for path in sorted(self.values):
            if path not in listed:
                found = self.resolution(path).value
                if found is not None:
                    lines.append(ValueLine(path, found.value, found.unit, found.source, found.reason, (),
                                           found.details))
        return tuple(sorted(lines, key=lambda line: line.path))

    def materials_plan(self, values: Sequence[ValueLine]) -> MaterialsPlan:
        """The conditions, the records read, the reads the listed values rest on, and every body's parts."""
        used = {value.path for value in values}
        reads: dict[tuple[str, str], materials.ReadOutcome] = {}
        if self.resolver is not None:
            for path in sorted(used):
                for outcome in self.resolver.value(path).reads:
                    reads[(outcome.material, outcome.field)] = outcome
        parts = tuple(line for name in sorted(self.parts) for _, line in sorted(self.parts[name].items()))
        media = {name: thing.material.name for name, thing in sorted(self.things.items())
                 if isinstance(thing, Medium) and thing.material is not None
                 and isinstance(cast(object, thing.material), materials.Record)}
        return MaterialsPlan(tuple(self.conditions.values()),
                             tuple(self.records[name] for name in sorted(self.records)),
                             tuple(reads[key] for key in sorted(reads)), parts, media)

    # -- across every plan ----------------------------------------------------------------------------------

    def check_ids(self) -> None:
        seen: dict[str, EntryPlan] = {}
        for plan in self.readers:
            if plan.owner in seen:
                self.refuse("id.duplicate", RefusalClass.USER,
                            f"two declarations share the id {plan.owner!r}: an id is the registered name and the "
                            f"things bound to it, so the same entry is bound to the same things twice",
                            "remove the duplicate declaration")
            seen[plan.owner] = plan

    def collect_registrations_and_slots(self) -> None:
        for dynamics in self.dynamics.values():
            dynamics.registrations.sort(key=lambda r: r.id)
        slots: dict[str, dict[str, Slot]] = {}
        for plan in self.readers:
            for read in plan.reads.values():
                if read.slot is None:
                    continue
                held = slots.setdefault(read.node, {}).get(read.slot)
                interfaces = (*(held.interfaces if held else ()), read.interface)
                if held and held.reader != plan.owner:
                    self.refuse("slot.collision", RefusalClass.INTERNAL,
                                f"field {read.node!r}: readers {held.reader} and {plan.owner} map to one slot name "
                                f"{read.slot!r}", MIME_DEFECT)
                slots[read.node][read.slot] = Slot(read.slot, plan.owner, tuple(dict.fromkeys(interfaces)))
        for path, provider in list(self.providers.items()):
            provider.slots = [slots[path][name] for name in sorted(slots.get(path, {}))]
            if not provider.slots:
                # A field nothing reads builds no node: there is no reader slot to give it.
                self.declared.remove(provider)
                del self.providers[path]

    def check_readers_outside_parts(self) -> None:
        """A provider whose field is valid outside its parts only (``outside_parts``) is not read by a body that starts
        inside the smallest sphere about a part's centroid that holds the part: the field is singular at the centroid
        and the model is not faithful that near (MIME-ANO-100)."""
        for path, provider in self.providers.items():
            if not cast(type[FieldProvider], provider.registered.cls).outside_parts:
                continue
            readers = [reader for reader in self.readers if any(read.node == path for read in reader.reads.values())]
            for part in provider.parts.values():
                start = composed(part.mount.initial_pose, part.offset) if part.mount else part.offset
                for reader in readers:
                    for body in reader.bindings.values():
                        found = self.dynamics.get(body)
                        position = found.initial.get("position") if found is not None else None
                        if position is None:
                            continue
                        distance = math.dist(position, start[:3])
                        if distance < part.radius:
                            self.refuse(
                                "provider.reader-inside", RefusalClass.USER,
                                f"body {body!r} starts {distance:.6g} m from the centroid of {part.apparatus!r}'s part "
                                f"{part.name!r}, inside the {part.radius:.6g} m sphere that holds the magnet: {path} "
                                f"is evaluated outside the magnet, and singular at its centroid",
                                f"start {body!r} outside the magnet, farther than its radius from {part.name!r}'s "
                                f"centroid (MIME-ANO-100 asks for five times that)")

    def collect_targets(self) -> dict[tuple[str, str], ResolvedTarget]:
        targets: dict[tuple[str, str], ResolvedTarget] = {}
        for plan in self.dynamics.values():
            assert plan.record is not None
            for name, port in plan.record.accepts.items():
                entry = vocabulary.lookup(name)
                if entry is None or not isinstance(entry.kind, TargetKind):
                    self.defect(plan.registered.cls, f"it accepts {name!r}, which is not a target in the vocabulary")
                    continue
                if entry.reserved:
                    continue
                targets[(plan.node_name(port.node), port.field)] = ResolvedTarget(
                    plan.node_name(port.node), port.field, entry.kind, plan.thing or "", name, entry.units)
        return targets

    def deliveries(self) -> list[ResolvedTarget]:
        found = [t for plan in self.readers for t in plan.delivers.values()]
        found += [t for plan in self.dynamics.values() for r in plan.registrations for t in r.returns.values()]
        return found

    def check_exclusive(self) -> None:
        by_port: dict[tuple[str, str], list[ResolvedTarget]] = {}
        for target in self.deliveries():
            if target.kind.exclusive:
                by_port.setdefault((target.node, target.field), []).append(target)
        for found in by_port.values():
            if len(found) > 1:
                first = found[0]
                self.refuse("target.exclusive", RefusalClass.USER,
                            f"{first.interface} of {first.thing!r} is an {first.kind.value} and takes one delivery; "
                            f"{', '.join(sorted(t.by for t in found))} all deliver into it",
                            "keep one of them")

    def union_groups(self) -> tuple[GroupPlan, ...]:
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            while parent.setdefault(node, node) != node:
                node = parent[node]
            return node

        requests: list[tuple[EntryPlan, GroupRequest, list[str]]] = []
        for plan in self.readers:
            assert plan.plan is not None
            for request in plan.plan.groups:
                members = [n for role in request.roles if plan.bindings.get(role) in self.dynamics
                           for n in self.dynamics[plan.bindings[role]].node_names()]
                members += plan.node_names() if request.own_nodes else []
                if not members:
                    continue
                for node in members[1:]:
                    parent[find(node)] = find(members[0])
                requests.append((plan, request, members))
        merged: dict[str, list[tuple[EntryPlan, GroupRequest, list[str]]]] = {}
        for item in requests:
            merged.setdefault(find(item[2][0]), []).append(item)
        dividers = {n: p.divider for p in self.declared for n in p.node_names()}
        groups: list[GroupPlan] = []
        for items in merged.values():
            nodes = sorted({n for _, _, members in items for n in members})
            settings = {(r.tolerance, r.max_iterations) for _, r, _ in items}
            by = tuple(sorted(f"{plan.owner}: {request.reason}" for plan, request, _ in items))
            if len(settings) > 1:
                self.refuse("group.settings", RefusalClass.USER,
                            f"the requests merged into one coupling group over {', '.join(nodes)} ask for different "
                            f"tolerances or caps: " + "; ".join(
                                f"{plan.owner}: tolerance {r.tolerance!r}, max_iterations {r.max_iterations}"
                                for plan, r, _ in items),
                            "make the merged requests agree")
                continue
            if len({dividers.get(n, 1) for n in nodes}) > 1:
                self.refuse("group.rates", RefusalClass.USER,
                            f"the coupling group over {', '.join(nodes)} has members at different rates: "
                            + ", ".join(f"{n} every {dividers.get(n, 1)} steps" for n in nodes),
                            "give every member of a coupling group the same rate")
                continue
            ((tolerance, cap),) = settings
            groups.append(GroupPlan(tuple(nodes), tolerance, cap, by))
        return tuple(groups)

    def check_rates(self) -> None:
        if not self.declared or not self.dt:
            return
        dividers = {n: p.divider for p in self.declared for n in p.node_names()}
        if dividers and min(dividers.values()) > 1:
            self.refuse("run.no-node-at-dt", RefusalClass.USER,
                        f"no node runs at the base step dt = {_seconds(self.dt)} (the fastest runs every "
                        f"{min(dividers.values())} steps); MADDENING would take the greatest common divisor of the "
                        f"rates as its base step instead of dt",
                        f"run some entry at dt, or raise dt to {_seconds(min(dividers.values()) * self.dt)}")

    def run_structure(self) -> RunStructure:
        """How a run executes: in compiled chunks of one control period, with the controller called between them, when
        the controller is a callback; otherwise as one compiled scan. A per-chunk hook given to a run chunks it too,
        without changing what it computes."""
        if self.controller is None or self.mode is not Mode.CALLBACK:
            return RunStructure()
        return RunStructure(self.controller.divider, (f"the controller {self.controller.registered.cls.label()} "
                                                      f"(mode {Mode.CALLBACK.value}, {Mode.CALLBACK.meaning})",))

    def orientation_lines(self) -> tuple[OrientationLine, ...]:
        """One line per thing: the directions declared for it or by the entries bound to it, a reference written as
        the path it refers to beside the vector it resolved to."""
        items: dict[str, list[str]] = {name: [] for name in self.things}
        for name, plan in self.geometries.items():
            for direction in type(plan.geometry).directions:
                items[name].append(f"{direction} {_vec(plan.geometry.direction(direction, plan.settings))}")
        for name, thing in self.things.items():
            dynamics = self.dynamics.get(name)
            if isinstance(thing, Body) and dynamics is not None and "orientation" in dynamics.initial:
                items[name].append(f"initial orientation (w, x, y, z) {_vec(dynamics.initial['orientation'])}")
            if isinstance(thing, Medium) and name in self.fills:
                items[name].append(f"fills {self.fills[name]}")
        for magnet in self.magnets:
            if isinstance(self.things.get(magnet.body), Apparatus):
                items[magnet.body].append(f"magnet {magnet.key} along {_vec(magnet.in_part)} in its part's frame, not "
                                          f"turned from the frame it is mounted on, from {magnet.source}")
                continue
            items[magnet.body].append(f"magnet {magnet.key} along {_vec(magnet.in_part)} in its part's frame, "
                                      f"{_vec(magnet.in_body)} in the body's, from {magnet.source}")
        for name, plans in self.segments.items():
            for plan in plans:
                held = cast(ChainSegment, plan.entry)
                on = f"on {plan.mount.segment}.{plan.mount.frame}" if plan.mount is not None else "in the world"
                at = _vec(cast(tuple[float, ...], plan.settings.get("at", (0.0, 0.0, 0.0))))
                turn = _vec(held.orientation) if held.orientation is not None else "(1, 0, 0, 0)"
                items[name].append(f"segment {plan.owner} ({plan.describe()}) placed {on} at {at} m, turned by {turn} "
                                   f"(w, x, y, z)")
        for name, parts in self.apparatus_parts.items():
            for key, part in sorted(parts.items()):
                on = f"on {part.mount.segment}.{part.mount.frame}" if part.mount is not None else "in the world"
                items[name].append(f"part {key} ({part.solid}) fixed {on}, its centroid at {_vec(part.offset[:3])} m "
                                   f"there, not turned")
        for plan in self.declared:
            if isinstance(plan.entry, Geometry):
                continue
            thing = plan.thing or next((bound for role, bound in plan.bindings.items()
                                        if role in getattr(plan.reader_cls(), "roles", {})
                                        and plan.reader_cls().roles[role].binds_a_thing), None)
            if thing not in items:
                continue
            for setting, declared in plan.registered.settings.items():
                if declared.kind is not Direction or setting not in plan.settings:
                    continue
                vector = cast(tuple[float, ...], plan.settings[setting])
                written: object = getattr(plan.entry, setting, None)
                by = f"= {written} " if isinstance(written, (TypedPath, str)) else ""
                items[thing].append(f"{plan.owner} {setting} {by}{_vec(vector)}")
        return tuple(OrientationLine(name, thing.kind.value, self.prims.get(name, prim_path_of(thing)),
                                     tuple(items[name])) for name, thing in self.things.items())

    def derive_seeds(self) -> None:
        for plan in self.readers:
            assert plan.plan is not None
            for stream in plan.plan.streams:
                plan.seeds[stream] = derive_seed(self.settings.seed, plan.owner, plan.bindings, stream)


def _fits(declared: Sequence[int], held: Sequence[int]) -> bool:
    """Whether two shapes agree, ``-1`` on either side matching any size."""
    return len(declared) == len(held) and all(a == b or -1 in (a, b) for a, b in zip(declared, held, strict=True))


def composed(outer: Sequence[float], inner: Sequence[float]) -> tuple[float, ...]:
    """``inner``, a pose ``(x, y, z, qw, qx, qy, qz)`` in ``outer``'s frame, in ``outer``'s parent, in float64."""
    held = RigidTransform((outer[0], outer[1], outer[2]), (outer[3], outer[4], outer[5], outer[6])).then(
        RigidTransform((inner[0], inner[1], inner[2]), (inner[3], inner[4], inner[5], inner[6])))
    return (*held.translation, *held.rotation)


def gathered(order: Sequence[str], groups: Sequence[Sequence[str]]) -> list[str]:
    """The canonical node order with each coupling group's members moved together to the place of its last member.

    MADDENING 0.4.x runs a group as one block at its first member's place, so a law added between a group's members
    (after a law in the group, before the dynamics in it) would run after the whole group, and the dynamics would
    read that law's load from the previous step. Gathered here, every node the canonical order puts before a member
    runs before the group, and the block keeps the order the nodes are added in."""
    out = list(order)
    for group in groups:
        members = [name for name in out if name in group]
        if not members:
            continue
        last = members[-1]
        out = [name for held in out for name in ((members if held == last else []) if held in group else [held])]
    return out


def derive_seed(seed: int, owner: str, bindings: Mapping[str, str], stream: str) -> int:
    """A stream's seed from the experiment's seed and the stream's identity (its entry's id, bindings and name), so
    attaching or removing another entry leaves it unchanged."""
    identity = "\x1f".join([str(seed), owner, *(f"{k}={v}" for k, v in sorted(bindings.items())), stream])
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def plan_experiment(settings: RunSettings, declarations: list[Declaration]) -> PlanSet:
    """Phase 1. Raises :class:`BuildRefused` with every reason found."""
    return _Planner(settings, declarations).run()
