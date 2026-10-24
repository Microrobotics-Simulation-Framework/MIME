from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, override

from maddening.core.node import SimulationNode

from mime.gradients import GradientStatus
from mime.things import ThingKind, TypedPath
from mime.units import Dimension, Magnitude, Quantity
from mime.vocabulary import Capability

if TYPE_CHECKING:
    from mime.build import (
        DriveBuilder,
        DynamicsBuilder,
        InteractionBuilder,
        ObservationBuilder,
        ProviderBuilder,
        ResolvedConfinement,
        ResolvedRegistration,
        Slot,
    )

_EMPTY: Mapping[str, Any] = MappingProxyType({})


class RoleKind(Enum):
    """What a role of an interaction or observation binds. ``SURFACE`` binds a named surface of a thing's geometry
    (``surface=tube.inner_surface``)."""

    BODY = "body"
    MEDIUM = "medium"
    STRUCTURE = "structure"
    APPARATUS = "apparatus"
    FIELD = "field"
    SURFACE = "surface"

    @property
    def thing_kind(self) -> ThingKind | None:
        return None if self in (RoleKind.FIELD, RoleKind.SURFACE) else ThingKind(self.value)

    @property
    def binds_a_thing(self) -> bool:
        return self.thing_kind is not None


@dataclass(frozen=True)
class SettingKind:
    """A marker for a setting whose meaning is not one SI dimension: ``rotation_axis: Annotated[DirectionLike,
    Direction]``, ``value: Annotated[Quantity, Commanded]``."""

    name: str
    description: str

    @override
    def __repr__(self) -> str:
        return self.name


#: A direction: a typed path to a direction of a thing's geometry (``tube.axis``), that path's string, or three numbers
#: (normalised in phase 1). Phase 1 hands the entry the unit vector in the world frame, read with ``ctx.vector``.
Direction = SettingKind("Direction", "a direction: a thing's direction such as tube.axis, or three numbers")
#: A drive's quantity, in the dimension of the command it drives, which phase 1 checks it against.
Commanded = SettingKind("Commanded", "a quantity in the dimension of the command it drives")

type DirectionLike = TypedPath | str | tuple[float, float, float]


class Port(NamedTuple):
    """A field of one of an entry's own nodes: ``node`` is its local name, ``""`` for an entry's only node."""

    field: str
    node: str = ""


@dataclass(frozen=True)
class Read:
    """A read of a bound thing: a state interface its dynamics publishes, or a field interface its provider publishes,
    evaluated at ``at`` (a state read giving the points)."""

    role: str
    interface: str
    at: Read | None = None


@dataclass(frozen=True)
class Delivery:
    """A delivery into a target interface of a bound thing's dynamics."""

    role: str
    target: str


@dataclass(frozen=True)
class Registration:
    """A record by which an interaction asks the dynamics of ``role`` to include the thing bound to ``subject`` in its
    own update. The receiving dynamics reads the subject's ``reads`` and delivers into its ``returns``, by label."""

    role: str
    target: str
    kind: str
    subject: str
    reads: Mapping[str, str] = field(default_factory=dict[str, str])
    returns: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class Value:
    """A read of a value of the thing bound to ``role``: its mass, its buoyant volume, a medium's density. Phase 1
    resolves it through ``mime.materials``, the one place that reads such values, and hands it to the entry in SI,
    with ``ctx.values``; the entry passes it to its node's constructor, so it becomes a ``params`` leaf."""

    role: str
    name: str


@dataclass(frozen=True)
class Confinement:
    """An inequality constraint: the body bound to ``role`` stays on the free side of the surface bound to
    ``surface``, by ``margin`` (SI) plus the body's extent (the value labelled ``extent`` in the plan), and its velocity
    is changed by ``rule`` where it is projected back. The body's dynamics takes it at construction."""

    role: str
    target: str
    surface: str
    margin: float
    rule: str
    extent: str = "extent"


@dataclass(frozen=True)
class Command:
    """A command a field provider (later an apparatus's chain) takes: the input ``port`` of its node, in
    ``dimension``. A drive or the controller writes it; nothing else does."""

    port: str
    dimension: Dimension
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class GroupRequest:
    """A request that the dynamics of the bound ``roles`` (and the entry's own nodes, with ``own_nodes``) be solved
    together in one coupling group. Overlapping requests are merged; every request states its tolerance and cap."""

    roles: tuple[str, ...]
    reason: str
    tolerance: float
    max_iterations: int
    own_nodes: bool = False


@dataclass(frozen=True)
class Problem:
    """What an entry's plan finds wrong with what it is bound to (a value outside a surrogate's training data, a medium
    whose region has no axis), refused in phase 1 under ``code``, its message naming the entry, with ``action``
    saying what to write instead."""

    code: str
    message: str
    action: str


@dataclass(frozen=True)
class Plan:
    """What an interaction or observation will build, as data: its nodes by local name, its reads and deliveries by
    label, its registrations, its solve-together requests and its random streams; and the problems it found, each a
    refusal."""

    nodes: tuple[str, ...] = ("",)
    reads: Mapping[str, Read] = field(default_factory=dict[str, Read])
    delivers: Mapping[str, Delivery] = field(default_factory=dict[str, Delivery])
    registrations: tuple[Registration, ...] = ()
    groups: tuple[GroupRequest, ...] = ()
    streams: tuple[str, ...] = ()
    values: Mapping[str, Value] = field(default_factory=dict[str, Value])
    confines: tuple[Confinement, ...] = ()
    problems: tuple[Problem, ...] = ()


@dataclass(frozen=True)
class CapabilityRecord:
    """What a dynamics or field provider offers, which support tables are written against."""

    entry: str
    flags: frozenset[Capability]
    publishes: Mapping[str, Port]
    accepts: Mapping[str, Port] = field(default_factory=dict[str, Port])
    registration_kinds: frozenset[str] = frozenset()

    def describe(self) -> str:
        flags = ", ".join(sorted(flag.value for flag in self.flags)) or "none"
        return f"{self.entry} (flags: {flags})"


@dataclass(frozen=True)
class Region:
    """The region a medium fills (``tube.lumen``), as a node evaluates it: the registered shape (``mime.core``) and
    its parameters in SI."""

    path: str
    shape: str
    parameters: Mapping[str, Magnitude]


@dataclass(frozen=True)
class PlanContext:
    """What an entry may consult while planning: run settings in SI, its bindings by role, the capability record of
    each bound thing's dynamics or bound field's provider, its own dimensioned settings in SI (``settings``, read
    with :meth:`scalar` and :meth:`vector`), the declared values of the things it is bound to (:meth:`value`), for a
    plan that depends on one, and the region each bound medium fills (:meth:`region`). A value it uses is still read
    through ``Plan.values``."""

    dt: float
    up: tuple[float, float, float]
    gravity: tuple[float, float, float]
    bindings: Mapping[str, str]
    capabilities: Mapping[str, CapabilityRecord | None]
    settings: Mapping[str, Magnitude] = field(default_factory=dict[str, Magnitude])
    declared: Mapping[tuple[str, str], Magnitude] = field(default_factory=dict[tuple[str, str], Magnitude])
    regions: Mapping[str, Region] = field(default_factory=dict[str, Region])

    def scalar(self, name: str) -> float:
        return _scalar(self.settings, name)

    def vector(self, name: str) -> tuple[float, ...]:
        return _vector(self.settings, name)

    def value(self, role: str, name: str) -> Magnitude | None:
        """The SI value declared for ``name`` of the thing bound to ``role``, or ``None`` when none is."""
        return self.declared.get((role, name))

    def region(self, role: str) -> Region | None:
        """The region the medium bound to ``role`` fills, or ``None`` when it fills none with a registered shape."""
        return self.regions.get(role)


@dataclass(frozen=True)
class BuildContext:
    """What an entry may consult while building. Every node it adds is named ``node_name(local)`` and stepped at
    ``timestep``; ``seeds`` holds one derived seed per declared random stream; ``settings`` holds its dimensioned
    settings as phase 1 converted them to SI, read with :meth:`scalar` and :meth:`vector`, so that its nodes are
    constructed from SI floats and an entry never converts a unit itself. ``regions`` holds the region each bound
    medium fills, as its plan saw it."""

    id: str
    dt: float
    timestep: float
    prefix: str
    bindings: Mapping[str, str] = field(default_factory=dict[str, str])
    seeds: Mapping[str, int] = field(default_factory=dict[str, int])
    initial: Mapping[str, Any] = field(default_factory=dict[str, Any])
    settings: Mapping[str, Magnitude] = field(default_factory=dict[str, Magnitude])
    values: Mapping[str, Magnitude] = field(default_factory=dict[str, Magnitude])
    confinements: tuple[ResolvedConfinement, ...] = ()
    up: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    regions: Mapping[str, Region] = field(default_factory=dict[str, Region])

    def node_name(self, local: str = "") -> str:
        return self.prefix if not local else f"{self.prefix}.{local}"

    def scalar(self, name: str) -> float:
        return _scalar(self.settings, name)

    def vector(self, name: str) -> tuple[float, ...]:
        return _vector(self.settings, name)

    def scalar_value(self, label: str) -> float:
        """The value its plan reads under ``label``, in SI: one number."""
        return _scalar(self.values, label)

    def vector_value(self, label: str) -> tuple[float, ...]:
        return _vector(self.values, label)

    def region(self, role: str) -> Region | None:
        """The region the medium bound to ``role`` fills, as phase 1 resolved it."""
        return self.regions.get(role)


def _held(settings: Mapping[str, Magnitude], name: str) -> Magnitude:
    if name not in settings:
        raise KeyError(f"no dimensioned setting {name!r} is set; those in SI are {sorted(settings)}")
    return settings[name]


def _scalar(settings: Mapping[str, Magnitude], name: str) -> float:
    value = _held(settings, name)
    if isinstance(value, tuple):
        raise TypeError(f"setting {name!r} is a vector of {len(value)} components, not a number")
    return value


def _vector(settings: Mapping[str, Magnitude], name: str) -> tuple[float, ...]:
    value = _held(settings, name)
    if not isinstance(value, tuple):
        raise TypeError(f"setting {name!r} is a number, not a vector")
    return value


@dataclass(frozen=True)
class DimensionedSetting:
    """A setting whose annotation declares its dimension, ``Annotated[Quantity, Length]``; ``optional`` when the
    annotation admits ``None`` (``Annotated[Quantity | None, Length]``). ``kind`` is set instead of a dimension for a
    setting marked :data:`Direction` or :data:`Commanded`."""

    dimension: Dimension | None
    optional: bool = False
    kind: SettingKind | None = None


@dataclass(frozen=True)
class Requires:
    """A predicate over a capability record: every flag claimed, every target accepted, every interface published."""

    flags: frozenset[Capability] = frozenset()
    accepts: frozenset[str] = frozenset()
    publishes: frozenset[str] = frozenset()

    def __call__(self, record: CapabilityRecord | None) -> bool:
        return (record is not None and self.flags <= record.flags and self.accepts <= set(record.accepts)
                and self.publishes <= set(record.publishes))

    @override
    def __str__(self) -> str:
        terms = [f"flag {flag.value}" for flag in sorted(self.flags, key=lambda f: f.value)]
        terms += [f"accepts {name}" for name in sorted(self.accepts)]
        terms += [f"publishes {name}" for name in sorted(self.publishes)]
        return " and ".join(terms) or "anything"


def requires(*flags: Capability, accepts: Sequence[str] = (), publishes: Sequence[str] = ()) -> Requires:
    return Requires(frozenset(flags), frozenset(accepts), frozenset(publishes))


@dataclass(frozen=True)
class SupportRow:
    """One supported construction: the rung (``None`` for an entry without rungs) and, per role, what the bound
    thing's dynamics or the bound field's provider must offer."""

    rung: str | None
    when: Mapping[str, Requires]
    note: str = ""


class NotEstablished(Enum):
    """The value of an entry's field that its code does not establish: declared as unknown, never guessed. Each use is a
    line, with its reason, in the docs check's list ``docs/ratchets/not_established.txt``."""

    NOT_ESTABLISHED = "not established"


NOT_ESTABLISHED = NotEstablished.NOT_ESTABLISHED


@dataclass(frozen=True)
class GradientRecord:
    """Where the derivative through an entry is not the exact derivative of what it computes, why, and the route to
    ``exact``: what an entry whose status is not ``exact`` declares beside it."""

    where: str | NotEstablished
    reason: str | NotEstablished
    route: str | NotEstablished


@dataclass(frozen=True, kw_only=True)
class Entry:
    """Every registered kind. An entry is a frozen dataclass whose fields are its settings; ``name`` and ``version``
    are its registry key, and the build records both. ``validity`` holds declared validity lines, listed in the
    build report and not yet checked.

    A dimensioned setting declares its dimension once, in its annotation (``margin: Annotated[Quantity, Length]``,
    with the markers of :mod:`mime.units`), and holds a :class:`~mime.units.Quantity`. Registration reads the
    annotations; phase 1 refuses a quantity of another dimension, naming the entry, the setting and both dimensions,
    and hands the entry its settings in SI through its context (``ctx.scalar("margin")``). ``rate`` is the one
    exception: a period or a frequency, checked against ``dt`` by its own rule.

    Three declarations are read by the docs: ``gradient``, where and why the derivative is not exact and the route to
    ``exact`` (for every status but ``exact``); ``references``, the bibliography keys of the works the entry rests on
    (``()`` for none, ``NOT_ESTABLISHED`` where its code does not say), a MIME entry's each an entry of
    ``docs/bibliography.bib``; and ``builds``, each rung's node classes
    (``""`` for an entry without rungs), which the node walk checks against what a case builds."""

    name: ClassVar[str]
    version: ClassVar[str]
    differentiability: ClassVar[GradientStatus]
    gradient: ClassVar[GradientRecord | None] = None
    references: ClassVar[tuple[str, ...] | NotEstablished] = ()
    builds: ClassVar[Mapping[str, tuple[type[SimulationNode], ...]]] = MappingProxyType({})
    validity: ClassVar[tuple[str, ...]] = ()
    #: Whether the entry follows geometry that moves: a thing's surface, region or direction it refers to is read
    #: through the thing's pose every step. Without it, a reference to the geometry of a thing with a dynamics is
    #: refused, since the entry takes geometry once, at construction, and would keep it as it was at t = 0.
    moving: ClassVar[bool] = False

    @classmethod
    def label(cls) -> str:
        return f"{cls.name} {cls.version}"

    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        """What makes settings whose dimensions are right wrong all the same (a radius below zero, a direction along
        an axis it must not lie on), given in SI, one line each: each is refused in phase 1."""
        return []


@dataclass(frozen=True, kw_only=True)
class SettingGroup:
    """Settings of an entry written together as one value (``lubrication=Lubrication(epsilon=mm(0.15))``): a frozen
    dataclass whose fields are settings, a dimensioned one annotated as an entry's is, registered by its ``group``
    name (``register_setting_group``) so that the saved form writes it by that name and reads it back. Phase 1 hands
    the entry a group's dimensioned settings in SI as ``<setting>.<field>`` (``ctx.scalar("lubrication.epsilon")``),
    refusing one of another dimension under that name. A group holds no group."""

    group: ClassVar[str]


def setting_value(entry: object, name: str) -> object:
    """The setting ``name`` of ``entry``, following a group's ``<setting>.<field>``; ``None`` when it, or the group it
    sits in, is ``None``."""
    held: object = entry
    for part in name.split("."):
        if held is None:
            return None
        held = getattr(held, part, None)
    return held


@dataclass(frozen=True, kw_only=True)
class Dynamics(Entry):
    """How one aspect of one thing evolves. The experiment builds it once, with every registration addressed to it.

    ``publishes`` maps state interfaces to the node fields that hold them, ``accepts`` target interfaces to the node
    inputs that take them (an inequality constraint is taken at construction, in ``ctx.confinements``), and
    ``registration_kinds`` names the registration records it can take. ``initial_state`` maps each value it takes
    from a body's ``State`` to that value's dimension, and :meth:`initial_problems` says what else is wrong with
    them. ``values`` reads values of its own thing (label to value name), as an interaction's plan does.
    """

    applies_to: ClassVar[frozenset[ThingKind]]
    flags: ClassVar[frozenset[Capability]] = frozenset()
    publishes: ClassVar[Mapping[str, Port]] = _EMPTY
    accepts: ClassVar[Mapping[str, Port]] = _EMPTY
    registration_kinds: ClassVar[frozenset[str]] = frozenset()
    initial_state: ClassVar[Mapping[str, Dimension]] = MappingProxyType({})
    values: ClassVar[Mapping[str, str]] = MappingProxyType({})
    nodes: ClassVar[tuple[str, ...]] = ("",)
    rate: Quantity | None = None

    def capabilities(self) -> CapabilityRecord:
        return CapabilityRecord(self.label(), frozenset(self.flags), dict(self.publishes), dict(self.accepts),
                                frozenset(self.registration_kinds))

    def initial_problems(self, initial: Mapping[str, Magnitude]) -> list[str]:
        """What is wrong with an initial state whose dimensions are right (a quaternion that is not a unit one, a
        vector of the wrong length), one line each."""
        return []

    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class FieldProvider(Entry):
    """What produces a field. ``publishes`` maps each field interface to the prefix of its output ports: a reader
    slot ``s`` reads its points into ``points_<s>`` (unless the provider claims ``Capability.UNIFORM``) and the field
    out of ``<prefix>_<s>``. ``commands`` are the inputs a drive or the controller writes, by command name."""

    flags: ClassVar[frozenset[Capability]]
    publishes: ClassVar[Mapping[str, str]]
    commands: ClassVar[Mapping[str, Command]] = MappingProxyType({})

    def capabilities(self) -> CapabilityRecord:
        return CapabilityRecord(self.label(), frozenset(self.flags),
                                {name: Port(prefix) for name, prefix in self.publishes.items()})

    def build(self, ctx: BuildContext, b: ProviderBuilder, slots: Sequence[Slot]) -> None:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Interaction(Entry):
    """One physical law among named things. It reads declared interfaces and delivers into declared targets of their
    dynamics, through a restricted builder; it never builds a dynamics or touches another entry's nodes.

    ``roles`` are bound by keyword at ``attach``. ``id_roles`` are the roles its id names (by default its body roles,
    or every role when it has none). An entry with ``rungs`` has a ``rung`` setting with no default, and ``support``
    lists the constructions it supports, each as predicates over the bound dynamics' capability records.
    ``reserved_rungs`` maps a rung whose name and meaning are fixed but which is not built to the reason, and the
    build refuses it with that reason. ``family`` names the family whose suite it passes, if any.
    """

    roles: ClassVar[Mapping[str, RoleKind]]
    family: ClassVar[str | None] = None
    id_roles: ClassVar[tuple[str, ...] | None] = None
    rungs: ClassVar[tuple[str, ...]] = ()
    reserved_rungs: ClassVar[Mapping[str, str]] = MappingProxyType({})
    support: ClassVar[tuple[SupportRow, ...]] = ()
    reads: ClassVar[Mapping[str, Read]] = _EMPTY
    delivers: ClassVar[Mapping[str, Delivery]] = _EMPTY
    values: ClassVar[Mapping[str, Value]] = MappingProxyType({})

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers), values=dict(self.values))

    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Observation(Entry):
    """What an experiment measures. It reads and never delivers: its nodes have no edge into the physics. Its
    bindings are its fields named after its roles (``Recorder(body=probe)``)."""

    roles: ClassVar[Mapping[str, RoleKind]]
    id_roles: ClassVar[tuple[str, ...] | None] = None
    reads: ClassVar[Mapping[str, Read]] = _EMPTY
    rate: Quantity | None = None

    def bindings(self) -> dict[str, Any]:
        return {role: getattr(self, role, None) for role in self.roles}

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(reads=dict(self.reads))

    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Drive(Entry):
    """A command generator: one node whose numbers are ``params`` leaves, with one exclusive edge into the command it
    drives (``drive:<apparatus>.<command>``). A quantity written for a command is short for the constant drive."""

    def build(self, ctx: BuildContext, b: DriveBuilder) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Drawing:
    """How the stage shows a geometry kind: one polygon mesh in the geometry's own frame, in metres and near its
    origin, and that frame's placement in the world (a translation, and a unit quaternion ``(w, x, y, z)``). USD
    stores points as float32, so the points stay small and local, and the placement is written as double-precision
    transform ops. ``representation`` says in words what the mesh is, for the stage's readers."""

    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    representation: str


@dataclass(frozen=True, kw_only=True)
class Geometry(Entry):
    """A geometry kind: a thing's shape as a script declares it, with its settings in SI handed to its methods. It
    names its ``regions`` (``lumen``), its ``directions`` (``axis``) and its ``surfaces``, each surface mapped to the
    shape in ``mime.core`` that its free side is evaluated with, and each region in ``region_shapes`` to the shape it
    is; typed paths to them come from the thing's handle (``tube.inner_surface``). A reader takes a surface's or a
    region's shape name and parameters into its node at construction."""

    regions: ClassVar[tuple[str, ...]] = ()
    region_shapes: ClassVar[Mapping[str, str]] = MappingProxyType({})
    surfaces: ClassVar[Mapping[str, str]] = MappingProxyType({})
    directions: ClassVar[tuple[str, ...]] = ()

    def surface_parameters(self, surface: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        raise NotImplementedError

    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        """The parameters of the shape ``region_shapes`` names for ``region``, in SI."""
        raise NotImplementedError

    def direction(self, name: str, settings: Mapping[str, Magnitude]) -> tuple[float, float, float]:
        raise NotImplementedError

    def drawing(self, settings: Mapping[str, Magnitude]) -> Drawing | None:
        """What the stage shows of this geometry, given its settings in SI; ``None`` writes its prim with its settings
        and nothing to see."""
        return None


type Tensor = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class Solid(Geometry):
    """A geometry kind with a closed-form volume and inertia: the solid a body's part is. In the part's own frame,
    centred on its centroid, it gives its ``volume`` (m^3) and ``inertia_per_mass``, the inertia tensor about its
    centroid divided by its mass (m^2), for a uniform density, given its settings in SI."""

    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        raise NotImplementedError

    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Controller(Entry):
    """Reads observations and writes apparatus commands; registered here, declared on an experiment once the
    controller's modes are built."""

    rate: Quantity | None = None
