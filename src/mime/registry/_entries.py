from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, NamedTuple, cast, override

from maddening.core.node import SimulationNode

from mime.core import shape_switches, switch
from mime.gradients import RANK, GradientStatus
from mime.things import ThingKind, TypedPath
from mime.units import Dimension, Length, Magnitude, Quantity
from mime.vocabulary import FRAME_POSES, Capability

if TYPE_CHECKING:
    from mime.build import (
        DriveBuilder,
        DynamicsBuilder,
        InteractionBuilder,
        ObservationBuilder,
        ProviderBuilder,
        ResolvedConfinement,
        ResolvedMount,
        ResolvedPart,
        ResolvedRegistration,
        Slot,
    )
    from mime.core.multibody import DescriptionClassification, MultibodyTree, Payload

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
    label, its registrations, its solve-together requests and its random streams; the shipped asset references it
    reads (``mime.assets``, by name: a wall's table), which a saved experiment's lock records; and the problems it
    found, each a refusal."""

    nodes: tuple[str, ...] = ("",)
    reads: Mapping[str, Read] = field(default_factory=dict[str, Read])
    delivers: Mapping[str, Delivery] = field(default_factory=dict[str, Delivery])
    registrations: tuple[Registration, ...] = ()
    groups: tuple[GroupRequest, ...] = ()
    streams: tuple[str, ...] = ()
    values: Mapping[str, Value] = field(default_factory=dict[str, Value])
    confines: tuple[Confinement, ...] = ()
    problems: tuple[Problem, ...] = ()
    assets: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityRecord:
    """What a dynamics or field provider offers, which support tables are written against, and the limits of its
    model that it states (``limits``: what it leaves out, each naming the anomaly that records it)."""

    entry: str
    flags: frozenset[Capability]
    publishes: Mapping[str, Port]
    accepts: Mapping[str, Port] = field(default_factory=dict[str, Port])
    registration_kinds: frozenset[str] = frozenset()
    limits: tuple[str, ...] = ()

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
class Hull:
    """A body's hull as a law reads it: the kind of the one part of the body that lies inside no other (``part``), its
    settings in SI, and where the part sits in the body's frame (its centroid, and the rotation from its frame to the
    body's, as rows). A law reads the kind's capabilities (its family parameters, its radial reach), never the
    shape's formula."""

    body: str
    part: str
    kind: Solid
    settings: Mapping[str, Magnitude]
    centre: tuple[float, float, float]
    rotation: tuple[tuple[float, ...], ...]

    def label(self) -> str:
        return f"{self.body}'s hull, part {self.part!r} ({self.kind.label()})"


@dataclass(frozen=True)
class PlanContext:
    """What an entry may consult while planning: run settings in SI, its bindings by role, the capability record of
    each bound thing's dynamics or bound field's provider, its own dimensioned settings in SI (``settings``, read
    with :meth:`scalar` and :meth:`vector`), the declared values of the things it is bound to (:meth:`value`), for a
    plan that depends on one, the region each bound medium fills (:meth:`region`), the hull of each bound body
    (:meth:`geometry`) and the segment that takes the reaction on each bound field's source (:meth:`reaction`). A value
    it uses is still read through ``Plan.values``."""

    dt: float
    up: tuple[float, float, float]
    gravity: tuple[float, float, float]
    bindings: Mapping[str, str]
    capabilities: Mapping[str, CapabilityRecord | None]
    settings: Mapping[str, Magnitude] = field(default_factory=dict[str, Magnitude])
    declared: Mapping[tuple[str, str], Magnitude] = field(default_factory=dict[tuple[str, str], Magnitude])
    regions: Mapping[str, Region] = field(default_factory=dict[str, Region])
    hulls: Mapping[str, Hull] = field(default_factory=dict[str, Hull])
    reactions: Mapping[str, str] = field(default_factory=dict[str, str])

    def scalar(self, name: str) -> float:
        return _scalar(self.settings, name)

    def vector(self, name: str) -> tuple[float, ...]:
        return _vector(self.settings, name)

    def geometry(self, of: str) -> Hull | None:
        """The hull of the body bound to the role ``of`` (``ctx.geometry(of="body")``), or ``None`` when it has none:
        no part, several outside one another, or one whose kind gives no radial reach."""
        return self.hulls.get(of)

    def reaction(self, role: str) -> str | None:
        """The chain segment that takes the reaction on the source of the field bound to ``role`` (its id,
        ``<apparatus>.<segment>``): the segment its provider's one part is mounted on, where that segment takes loads.
        ``None`` where nothing does: a source fixed in the world, or on a segment that takes no load, a kinematic
        one. A law that delivers a reaction plans the delivery ``Delivery(role, "apparatus.link_wrench")`` only where
        this names a segment."""
        return self.reactions.get(role)

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
    medium fills, and ``hulls`` the hull of each bound body, as its plan saw them. A chain segment after the first
    finds the frame it mounts on in ``mount``, and a field provider the parts its settings name, with their mounts, in
    ``parts``. A segment whose dynamics carry mass finds what is mounted on its frames in ``mounted``, each a rigid
    body to graft on its tree; a segment that spins a rotor, the rotor's moment about its axis in ``spin_inertia`` (its
    own declared and its parts') and, where its parts are unequal across the axis, their half difference across it in
    ``spin_asymmetry`` (a 3 x 3 matrix, flattened, in the segment's base frame at angle zero). ``rates_read`` is
    whether anything reads the angular velocities of its frames (``apparatus.frame_angular_velocities``): a segment
    mounted on it, a spinner, a controller or an observation; a segment that computes them only to publish them may
    skip them where nothing does. A controller finds in ``models`` the built tree of each segment that carries mass,
    what is mounted on it grafted on, by its id: a model-based controller reads that, and one that reads a bare
    description of its own declares that choice."""

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
    hulls: Mapping[str, Hull] = field(default_factory=dict[str, Hull])
    mount: ResolvedMount | None = None
    parts: Mapping[str, ResolvedPart] = field(default_factory=dict[str, "ResolvedPart"])
    mounted: tuple[Payload, ...] = ()
    spin_inertia: float | None = None
    models: Mapping[str, MultibodyTree] = field(default_factory=dict[str, "MultibodyTree"])
    spin_asymmetry: tuple[float, ...] = ()
    rates_read: bool = True

    def geometry(self, of: str) -> Hull | None:
        """The hull of the body bound to the role ``of``, as its plan saw it."""
        return self.hulls.get(of)

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


class Role(Enum):
    """What a setting is (design #191): ``design``, chosen by the experimenter (a command, an initial state, a
    window); ``physical``, a fact about the setup (a geometry, a fixed apparatus's placement, local gravity);
    ``numerical``, a solver's choice (a sample spacing, a regulariser, a step); ``standard``, a constant whose
    definition MIME builds in. A value may be given as the experimenter's own choice (``reason=chosen(...)``), which
    makes it design in that experiment whatever its role."""

    DESIGN = "design"
    PHYSICAL = "physical"
    NUMERICAL = "numerical"
    STANDARD = "standard"


class DefaultKind(Enum):
    """What a setting's default claims (design #191 item 4): ``determined``, the one right answer from other inputs
    (the wall kind from the region the medium fills), resolved in every profile and reported; ``convention``, what
    anyone assumes untold (the window from the run's start), the setting's written default, taken in every profile and
    reported; ``numerical``, a stated rule with its own basis (a sample spacing from the body's size), taken in
    exploratory and stamped pending, its value required in research; ``none``, no default: the setting is required.
    ``undeclared`` is not declared: it is how the build reports a written default whose setting declares no role (a
    lab's entry), stamped pending in exploratory and refused in research."""

    DETERMINED = "determined"
    CONVENTION = "convention"
    NUMERICAL = "numerical"
    NONE = "none"
    UNDECLARED = "undeclared"


@dataclass(frozen=True)
class SettingRole:
    """A setting's role and its default's kind, with the default's rule and basis in words. A ``determined`` default,
    and a ``numerical`` one that depends on other inputs, is computed by the entry's method ``by`` from the plan's
    context (``default_spacing(self, ctx) -> value | None``, ``None`` where it does not apply, as on a rung that reads
    no such setting), and the setting's written default is then ``None``, meaning not given. A numerical default that
    is one value, and a ``convention``, is the setting's written default; it is taken when the value is that default
    and carries no source (the saved form cannot tell the two apart either)."""

    role: Role
    default: DefaultKind = DefaultKind.NONE
    rule: str = ""
    basis: str = ""
    by: str | None = None
    #: What was measured of a numerical default, in numbers and on what (the bem rung's spacing's miss against a
    #: reference): carried beside the item on every stamp and by ``mime pending``, never into the saved form.
    measured: str = ""


DESIGN = SettingRole(Role.DESIGN)
PHYSICAL = SettingRole(Role.PHYSICAL)
NUMERICAL = SettingRole(Role.NUMERICAL)
STANDARD = SettingRole(Role.STANDARD)


def convention(role: SettingRole, basis: str) -> SettingRole:
    """``role`` with its written default declared a convention, and why anyone would assume it untold."""
    return SettingRole(role.role, DefaultKind.CONVENTION, "the setting's written default", basis)


def determined(role: SettingRole, rule: str, by: str) -> SettingRole:
    """``role`` with a default determined by other inputs: its rule, and the entry's method computing it."""
    return SettingRole(role.role, DefaultKind.DETERMINED, rule, "it is the one value the other inputs allow", by)


def numerical_default(rule: str, basis: str, by: str | None = None, measured: str = "") -> SettingRole:
    """A numerical setting with a default: its rule, its basis, the entry's method computing it (``None``: the written
    default is the value), and what was measured of it."""
    return SettingRole(Role.NUMERICAL, DefaultKind.NUMERICAL, rule, basis, by, measured)


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


@dataclass(frozen=True, kw_only=True)
class GradientRecord:
    """What an entry whose status is not ``exact`` declares beside it: the ``switches`` its law evaluates, each with its
    orders and its route to ``exact`` (``mime.core.switch``); and, for a status no switch gives (``fixed_key``,
    ``unverified``, ``cut``), ``where`` the derivative is not the exact derivative of what it computes, the ``reason``
    and the ``route`` to ``exact``. A ``piecewise_kink`` or ``piecewise_jump`` entry's switches are its record."""

    where: str | NotEstablished | None = None
    reason: str | NotEstablished | None = None
    route: str | NotEstablished | None = None
    switches: tuple[switch.Declaration, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RungGradient:
    """One rung's own status and gradient record, for an entry whose rungs differ: what a user differentiates through
    is the rung they attached. ``gradient`` is as an entry's, its switches those the rung's own nodes evaluate;
    ``None`` for an ``exact`` rung."""

    differentiability: GradientStatus
    gradient: GradientRecord | None = None


class OutputKind(Enum):
    """When an observation's output has a value: at every step (a series, kept when a run records it), or at the end
    of a run, reduced by the graph from what it observed (always there)."""

    PER_STEP = "per_step"
    AT_END = "at_end"


class Reduction(Enum):
    """How the graph reduces a per-step output over a window: the node's own rule, declared beside the output it
    gives, so that a series taken over the same window and reduced by the same rule gives the same value. A window is
    the steps whose end lies after its start and not after its end.

    A closed set: a mean and an integral. An observation whose graph reduces by another rule (a maximum, a root mean
    square) needs a MIME edit today; the route is a rule registered with its series-side reduction beside it, which is
    not done yet."""

    #: The mean of the step's values over the steps in the window (each step the same length).
    MEAN = "mean"
    #: The sum over the steps in the window of each step's value times its length.
    INTEGRAL = "integral"


@dataclass(frozen=True)
class Output:
    """An output an observation keeps in its node's state: the field ``field`` of its node ``node`` (``""`` for its
    only node), of dimension ``dimension`` and shape ``shape``, with what it is (``description``). ``kind`` says when it
    has a value. An output at the end that the graph reduces from a per-step one names it (``reduces``) and its rule
    (``rule``); ``after`` names the setting, a time, whose value starts its window, which is then read from the node
    when the experiment is built (none: the window starts at the run's start)."""

    field: str
    dimension: Dimension
    description: str
    kind: OutputKind = OutputKind.PER_STEP
    shape: tuple[int, ...] = ()
    node: str = ""
    reduces: str | None = None
    rule: Reduction | None = None
    after: str | None = None


@dataclass(frozen=True)
class Flag:
    """A flag an entry's node raises: its state field ``field`` (of node ``node``, ``""`` for the entry's only node),
    nonzero at a step where what ``meaning`` says happened, and zero otherwise. ``rung`` is the rung whose node raises
    it, for an entry with rungs. A run reports the flags raised over each window it reads, and a flag never blocks a
    read."""

    field: str
    meaning: str
    node: str = ""
    rung: str | None = None


@dataclass(frozen=True)
class ReynoldsRange:
    """The Reynolds number up to which an entry's model is declared (``upper``, by the diameter of the body's hull, its
    speed and the medium's values) and the ``basis`` of that bound, for the rung ``rung`` (every rung when ``None``).
    A declaration, not a fit: it is the line the post-run watch compares a run's Reynolds number with, and the watch
    raises a flag, never a refusal, when a run lies past it. An entry of the hydrodynamic family without one is
    watched all the same, and the watch says it has no range to compare with."""

    upper: float
    basis: str
    rung: str | None = None


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
    (``""`` for an entry without rungs), which the node walk checks against what a case builds.

    An entry whose rungs differ in how a derivative passes through them declares ``rung_gradients``, each built rung's
    own status and record (:class:`RungGradient`): the build records the attached rung's, and the pages print each
    rung's beside it. Its ``differentiability`` is then the least of its rungs', and its ``gradient`` holds every
    rung's switches, so that each owner is still declared once, by the entry. An entry without it is one status for
    every rung."""

    name: ClassVar[str]
    version: ClassVar[str]
    differentiability: ClassVar[GradientStatus]
    gradient: ClassVar[GradientRecord | None] = None
    references: ClassVar[tuple[str, ...] | NotEstablished] = ()
    builds: ClassVar[Mapping[str, tuple[type[SimulationNode], ...]]] = MappingProxyType({})
    rung_gradients: ClassVar[Mapping[str, RungGradient]] = MappingProxyType({})
    validity: ClassVar[tuple[str, ...]] = ()
    #: The flags its nodes raise (:class:`Flag`), which a run reports on every window it reads.
    raises: ClassVar[tuple[Flag, ...]] = ()
    #: The Reynolds numbers its rungs are declared up to (:class:`ReynoldsRange`), which the post-run watch compares a
    #: run's with: for an entry of the hydrodynamic family, whose model neglects the fluid's inertia.
    reynolds: ClassVar[tuple[ReynoldsRange, ...]] = ()
    #: Whether the entry follows geometry that moves: a thing's surface, region or direction it refers to is read
    #: through the thing's pose every step. Without it, a reference to the geometry of a thing with a dynamics is
    #: refused, since the entry takes geometry once, at construction, and would keep it as it was at t = 0.
    moving: ClassVar[bool] = False
    #: Each setting's role and its default's kind (:class:`SettingRole`), merged along the class's bases: a setting
    #: that declares none is required, with no default (design #191 item 1). Registration refuses a declaration that
    #: contradicts itself (a default on a physical or design setting that is not a convention or determined, a
    #: numerical default without its rule and basis).
    setting_roles: ClassVar[Mapping[str, SettingRole]] = MappingProxyType({})

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
    #: Each of the group's settings' roles, as an entry's (:attr:`Entry.setting_roles`); a setting of the group that
    #: declares none takes the role the entry declares for the group.
    setting_roles: ClassVar[Mapping[str, SettingRole]] = MappingProxyType({})


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

    A dynamics with ``rungs`` has a ``rung`` setting with no default, as an interaction has; ``reserved_rungs`` maps a
    rung whose name and meaning are fixed but which is not built to the reason, and the build refuses it with that
    reason. ``limits`` states what its model leaves out, each line naming the anomaly that records it; its capability
    record carries them.
    """

    applies_to: ClassVar[frozenset[ThingKind]]
    flags: ClassVar[frozenset[Capability]] = frozenset()
    publishes: ClassVar[Mapping[str, Port]] = _EMPTY
    accepts: ClassVar[Mapping[str, Port]] = _EMPTY
    registration_kinds: ClassVar[frozenset[str]] = frozenset()
    initial_state: ClassVar[Mapping[str, Dimension]] = MappingProxyType({})
    values: ClassVar[Mapping[str, str]] = MappingProxyType({})
    nodes: ClassVar[tuple[str, ...]] = ("",)
    rungs: ClassVar[tuple[str, ...]] = ()
    reserved_rungs: ClassVar[Mapping[str, str]] = MappingProxyType({})
    limits: ClassVar[tuple[str, ...]] = ()
    setting_roles: ClassVar[Mapping[str, SettingRole]] = MappingProxyType({
        "rate": convention(NUMERICAL, "a node steps every base step unless the experiment chooses a slower rate")})
    rate: Quantity | None = None

    def capabilities(self) -> CapabilityRecord:
        return CapabilityRecord(self.label(), frozenset(self.flags), dict(self.publishes), dict(self.accepts),
                                frozenset(self.registration_kinds), tuple(self.limits))

    def initial_problems(self, initial: Mapping[str, Magnitude]) -> list[str]:
        """What is wrong with an initial state whose dimensions are right (a quaternion that is not a unit one, a
        vector of the wrong length), one line each."""
        return []

    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Segment:
    """A chain segment as its settings make it, read once in phase 1: the names of its frames, in the order of the
    rows of the frame poses it publishes; the commands it takes, by name; the classification of the description it was
    read from, if any, which the build report and the saved form record; the problems found, each refused; and the
    pose of each frame in the segment's base frame at its initial state, ``(x, y, z, qw, qx, qy, qz)``, which places
    what mounts on it before the first step, and its angular velocity there relative to the frame the segment mounts
    on, in its base frame's axes (none given: at rest); and whether its own description gives mass that its joints
    move (``massive``: an arm's links), which a dynamic segment it is mounted on would have to carry."""

    frames: tuple[str, ...]
    commands: Mapping[str, Command] = field(default_factory=dict[str, Command])
    description: DescriptionClassification | None = None
    problems: tuple[Problem, ...] = ()
    initial: tuple[tuple[float, ...], ...] = ()
    initial_rates: tuple[tuple[float, float, float], ...] = ()
    massive: bool = False


@dataclass(frozen=True)
class Spinner:
    """A rotor a chain segment carries by its absolute spin: the ``frame`` its parts mount on, its ``axis`` in the
    segment's base frame (a unit vector through the frame's origin), its own moment of inertia about that axis beyond
    its parts' (``inertia``, kg m^2), and the most its spin reaches where that is declared (``top_speed``, rad/s)."""

    frame: str
    axis: tuple[float, float, float]
    inertia: float
    top_speed: float | None = None


@dataclass(frozen=True, kw_only=True)
class ChainSegment(Dynamics):
    """A segment of an apparatus's chain (``Apparatus(..., chain={"arm": ..., "motor": ...})``): the dynamics of one
    mechanism, a robot arm or a motor, which publishes the world pose of every frame it names as
    ``apparatus.frame_poses``, under every rung alike, so that what mounts on it reads it the same way.

    Segments mount in chain order. The first is placed in the world by ``at`` (a translation) and ``orientation`` (a
    unit quaternion ``(w, x, y, z)``); each later one on the frame ``mount`` of the segment before it, offset by
    ``at`` and ``orientation`` in that frame. The build derives the edge that carries the mount frame's pose; the
    segment's node takes every frame of the segment before it as ``mount_poses`` and reads its row (``ctx.mount``,
    ``b.mount``).

    A segment says what its settings make it in :meth:`segment` (its frames and commands, the classification of its
    description), and with :meth:`load_refusal` why it takes no load. Closed chains, which URDF cannot describe, are
    out of scope until a downstream user needs one: ask at https://github.com/Microrobotics-Simulation-Framework/MIME/
    issues. Until then a lab can register its own segment, a ``ChainSegment`` that takes its own description of the
    mechanism and publishes the same frame poses, and mounts and everything downstream work unchanged."""

    applies_to = frozenset({ThingKind.APPARATUS})
    publishes = MappingProxyType({FRAME_POSES.name: Port("frame_poses")})
    setting_roles: ClassVar[Mapping[str, SettingRole]] = MappingProxyType({
        "mount": PHYSICAL,
        "at": convention(PHYSICAL, "a segment sits at the origin of the frame it is placed in unless offset"),
        "orientation": convention(PHYSICAL, "a segment is not turned from the frame it is placed in unless turned")})
    mount: str | None = None
    at: Annotated[Quantity | None, Length] = None
    orientation: tuple[float, float, float, float] | None = None

    def segment(self, settings: Mapping[str, Magnitude]) -> Segment:
        """Its frames and commands as its settings (in SI) make them, the classification of what it read, and the
        problems found."""
        raise NotImplementedError

    def load_refusal(self) -> str | None:
        """Why this segment takes no load delivered to it, or ``None`` when it takes the loads it accepts."""
        return None

    def carries_mass(self) -> bool:
        """Whether mass mounted on its frames loads its dynamics (a dynamic arm's links): the build grafts each rigid
        body mounted there onto its tree (``ctx.mounted``). A segment whose motion is prescribed carries none and
        needs none, and every mounted mass is accepted on it."""
        return False

    def riding(self) -> tuple[str, ...]:
        """Its frames fixed to its base, so to the frame it mounts on: what is mounted on one rides there."""
        return ()

    def spinner(self, settings: Mapping[str, Magnitude]) -> Spinner | None:
        """The rotor it carries by its absolute spin about an axis fixed in its base, as its settings (in SI) make it
        (a spinner): the parts mounted on its frame are its rotor, whose mass and inertia across the axis the frame
        the segment mounts on carries. ``None`` for a segment that spins none."""
        return None

    def carried_model(self, mounted: Sequence[Payload]) -> MultibodyTree | None:
        """The tree its dynamics carry, ``mounted`` grafted on: what a model-based controller reads. ``None`` for a
        segment with no tree."""
        return None

    def reads_mount_rates(self, settings: Mapping[str, Magnitude], published: bool) -> bool:
        """Whether its nodes read the angular velocities of the frame it mounts on (``b.mount_rates``), given whether
        anything reads its own frames' (``published``, ``ctx.rates_read``): a spinner always does, and a segment
        that adds them to its own only to publish those, only where those are read. By default it does, wherever
        they are published, so that the segment it mounts on publishes them for it."""
        return True

    def placement(self, settings: Mapping[str, Magnitude]) -> tuple[float, float, float, float, float, float, float]:
        """Its base's pose in the frame it mounts on (in the world for the first segment): ``at`` (in SI) and
        ``orientation``."""
        at = settings.get("at", (0.0, 0.0, 0.0))
        x, y, z = at if isinstance(at, tuple) and len(at) == 3 else (0.0, 0.0, 0.0)
        w, i, j, k = self.orientation if self.orientation is not None else (1.0, 0.0, 0.0, 0.0)
        return (x, y, z, float(w), float(i), float(j), float(k))


@dataclass(frozen=True, kw_only=True)
class FieldProvider(Entry):
    """What produces a field. ``publishes`` maps each field interface to the prefix of its output ports: a reader
    slot ``s`` reads its points into ``points_<s>`` (unless the provider claims ``Capability.UNIFORM``) and the field
    out of ``<prefix>_<s>``. ``commands`` are the inputs a drive or the controller writes, by command name.

    ``parts`` names the settings that each name a part of the provider's apparatus (``part="rotor_magnet"``), each
    with the role the part must have (its role's ``model``, or ``None`` for any part); the provider reads the part's
    pose through its mount (``ctx.parts``, ``b.part``). ``values`` reads values of its apparatus (label to value
    name), as a dynamics' do. ``outside_parts`` says the field is evaluated outside the parts it names (a point dipole's
    is singular at its centre): the build refuses a reader whose initial position lies inside the smallest sphere
    that holds one."""

    flags: ClassVar[frozenset[Capability]]
    publishes: ClassVar[Mapping[str, str]]
    commands: ClassVar[Mapping[str, Command]] = MappingProxyType({})
    parts: ClassVar[Mapping[str, str | None]] = MappingProxyType({})
    values: ClassVar[Mapping[str, str]] = MappingProxyType({})
    outside_parts: ClassVar[bool] = False

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
    bindings are its fields named after its roles (``Recorder(body=probe)``). ``outputs`` declares what it gives a
    run's result (:class:`Output`), each with its dimension, per step or at the end: the pages and a run's manifest
    are generated from them, and a run's result reads each by its name."""

    roles: ClassVar[Mapping[str, RoleKind]]
    id_roles: ClassVar[tuple[str, ...] | None] = None
    reads: ClassVar[Mapping[str, Read]] = _EMPTY
    outputs: ClassVar[tuple[Output, ...]] = ()
    setting_roles: ClassVar[Mapping[str, SettingRole]] = MappingProxyType({
        "rate": convention(NUMERICAL, "an observation reads every base step unless the experiment chooses a slower "
                                      "rate")})
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
    """A geometry kind tessellated: one closed polygon mesh of its body, in the kind's own frame (its prim's), in
    metres and near its origin, each face's vertices counter-clockwise seen from outside. The prim's transform
    places it: USD stores points as float32, so the points stay small and local and the placement is written in double
    precision. ``representation`` says in words what the mesh is, for the stage's readers."""

    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    representation: str


@dataclass(frozen=True)
class Native:
    """A geometry kind written as one of USD's own primitives (UsdGeom's ``Sphere``, ``Capsule`` or ``Cylinder``): the
    primitive's type, each of its length attributes with the kind's setting it holds (in metres, written through the
    stage's ``metersPerUnit``), and its token attributes. The primitive's axis, where it has one, is the kind's own +Z:
    the prim's transform turns it onto the kind's ``axis``."""

    primitive: str
    lengths: Mapping[str, str]
    tokens: Mapping[str, str] = field(default_factory=dict[str, str])


#: The primitives of OpenMaterica's scene manifest (its ``geometry.md`` section 3.1), with each one's fields.
REGION_PRIMITIVES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "sphere": ("centre", "radius"),
    "cylinder": ("centre", "direction", "radius", "length"),
    "slab": ("point", "normal", "thickness"),
})


@dataclass(frozen=True)
class RegionPrimitive:
    """A region of a geometry kind as one of OpenMaterica's analytic primitives (:data:`REGION_PRIMITIVES`): its inside
    test, a point on its surface counted inside, is the region's, exactly. Where a region has no end along an axis (the
    tube's lumen, which a law reads as a uniform tube), the primitive is the region clipped at the kind's own extent
    along it, and the kind's ``region_primitive`` says so. Its fields are in metres in the kind's own frame, its points
    as three numbers and its ``direction`` or ``normal`` a unit vector."""

    primitive: str
    fields: Mapping[str, float | tuple[float, float, float]]


@dataclass(frozen=True)
class NoPrimitive:
    """A region no OpenMaterica primitive gives exactly, and why: refused, never approximated."""

    reason: str


@dataclass(frozen=True)
class Samples:
    """A surface patch sampled: points on it, the outward unit normal at each, and each point's quadrature weight (an
    area), as arrays ``(n, 3)``, ``(n, 3)`` and ``(n,)`` in the kind's own frame; and, for a kind whose arithmetic
    rounds a point at a scale below its distance from the kind's origin, that scale, ``(n,)`` m (``rounding_length``:
    the helix's, which carries its turn along its axis exact, is the distance from the axis). A check judges a point's
    rounding on it, and on the distance from the origin where it is ``None``."""

    points: Any
    normals: Any
    weights: Any
    rounding_length: Any = None


@dataclass(frozen=True, kw_only=True)
class Geometry(Entry):
    """A geometry kind: a thing's shape as a script declares it, with its settings in SI handed to its methods. It
    names its ``regions`` (``lumen``), its ``directions`` (``axis``) and its ``surfaces``, each surface mapped to the
    shape in ``mime.core`` that its free side is evaluated with, and each region in ``region_shapes`` to the shape it
    is; typed paths to them come from the thing's handle (``tube.inner_surface``). A reader takes a surface's or a
    region's shape name and parameters into its node at construction.

    **Build-side capabilities**, each an optional method: a kind implements those it has (:func:`implements`), and a
    consumer that needs one its kind lacks is refused when it plans, naming both. ``patches`` names the patches its
    surface is sampled by, each with the region whose boundary it lies on (a closed kind's patches are its whole
    surface, and a solid's body is the intersection of its regions): :meth:`samples` by patch at a count of samples a consumer chooses (:meth:`sample_counts` for a spacing),
    :meth:`area` by patch, :meth:`family_parameters`, :meth:`radial_reach` (the hull's), :meth:`tangent`,
    :meth:`drawing` and :meth:`region_primitive`. Its run-time capabilities (signed distance, outward normal, nearest
    point) are its regions' shapes.

    **Its own frame** is its prim's: the settings ``placement`` names place the kind rather than shape it (``"origin"``
    a point its origin sits at, ``"z_axis"`` a direction its own +Z is turned onto), so on a stage they are the prim's
    transform and never attributes. Its drawing, native form and region primitives are in that frame; its region
    parameters and samples are in the frame its settings are written in. ``native`` is its native form, USD's own
    primitive, chosen by kind and never by value: a kind with one is written as that primitive, and one without
    (``None``) in MIME's schema with a drawn mesh."""

    regions: ClassVar[tuple[str, ...]] = ()
    region_shapes: ClassVar[Mapping[str, str]] = MappingProxyType({})
    surfaces: ClassVar[Mapping[str, str]] = MappingProxyType({})
    directions: ClassVar[tuple[str, ...]] = ()
    patches: ClassVar[Mapping[str, str]] = MappingProxyType({})
    placement: ClassVar[Mapping[str, str]] = MappingProxyType({})
    native: ClassVar[Native | None] = None

    def surface_parameters(self, surface: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        raise NotImplementedError

    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        """The parameters of the shape ``region_shapes`` names for ``region``, in SI."""
        raise NotImplementedError

    def direction(self, name: str, settings: Mapping[str, Magnitude]) -> tuple[float, float, float]:
        raise NotImplementedError

    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        """The kind tessellated in its own frame at the writer's resolution: ``segments`` facets round a full turn of
        each curved direction, every vertex on the surface."""
        raise NotImplementedError

    def region_primitive(self, region: str, settings: Mapping[str, Magnitude]) -> RegionPrimitive | NoPrimitive:
        """``region`` as the OpenMaterica primitive whose inside test is exactly the region's, in the kind's own
        frame, or why there is none."""
        raise NotImplementedError

    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        """The counts of samples along each of ``patch``'s coordinates that keep neighbouring samples about
        ``spacing`` (m) apart at most: what :meth:`samples` takes."""
        raise NotImplementedError

    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        """``patch`` sampled at ``counts``: a JAX function of ``parameters`` (its settings in SI, numbers or traced
        arrays), so that a study's derivatives in them flow through the samples."""
        raise NotImplementedError

    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        """The area of ``patch``, m^2."""
        raise NotImplementedError

    def family_parameters(self, settings: Mapping[str, Magnitude]) -> dict[str, float | int]:
        """The parameters of the family the kind is (a surrogate trained on one family reads them), by name."""
        raise NotImplementedError

    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        """How far the kind reaches from its own axis, through its centroid: the radius of the coaxial cylinder that
        holds it (any axis through a sphere's centre). A body whose hull it is takes it as its ``extent``."""
        raise NotImplementedError

    def tangent(self, parameters: Mapping[str, Any], point: Any) -> Any:
        """The unit tangent the kind declares at its surface point nearest ``point`` (in its frame): a direction a
        law may be anisotropic in, which no kind has unless it declares one."""
        raise NotImplementedError


#: The build-side capabilities a kind may implement, by method name; ``native`` is the class variable, and
#: ``mass_properties`` a solid's ``volume`` and ``inertia_per_mass``.
CAPABILITIES = ("samples", "area", "family_parameters", "radial_reach", "tangent", "mass_properties", "drawing",
                "native", "region_primitive")


def implements(kind: type[Geometry] | Geometry, capability: str) -> bool:
    """Whether ``kind`` implements the build-side ``capability`` (one of :data:`CAPABILITIES`), rather than inheriting
    ``Geometry``'s refusal of it."""
    if capability not in CAPABILITIES:
        raise KeyError(f"{capability!r} is no capability of a geometry kind; they are {CAPABILITIES}")
    held = kind if isinstance(kind, type) else type(kind)
    if capability == "native":
        return held.native is not None
    if capability == "mass_properties":
        return issubclass(held, Solid) and held.volume is not Solid.volume
    return getattr(held, capability) is not getattr(Geometry, capability)


def declared_switches(cls: type[Entry]) -> tuple[switch.Declaration, ...]:
    """The switches an entry declares: its gradient record's, and for a geometry kind those of each shape its
    surfaces and regions are evaluated with, which the shape declares beside itself."""
    own = cls.gradient.switches if cls.gradient is not None else ()
    if not issubclass(cls, Geometry):
        return own
    shapes = sorted({*cls.surfaces.values(), *cls.region_shapes.values()})
    inherited = [declared for name in shapes for declared in shape_switches(name)]
    return (*own, *(declared for declared in inherited if declared not in own))


def rung_gradient(cls: type[Entry], rung: str | None) -> RungGradient:
    """The status and record of what a user differentiates through when ``cls`` is attached with ``rung`` (``None``
    or ``""`` for an entry without rungs): the rung's own where the entry declares one, and the entry's otherwise."""
    held = cls.rung_gradients.get(rung or "")
    return held if held is not None else RungGradient(differentiability=cls.differentiability, gradient=cls.gradient)


def placed_gradient(cls: type[Entry], *, callback: bool = False) -> RungGradient:
    """The status and record of what a user differentiates through when ``cls`` is attached: for a controller placed as
    a callback (``callback``), the binding's ``cut``; for any other attachment, and a controller inside the compiled
    step, the entry's own."""
    if callback and issubclass(cls, Controller):
        return cls.callback_gradient
    return rung_gradient(cls, None)


def rung_gradient_problems(cls: type[Entry]) -> list[str]:
    """What is wrong with an entry's ``rung_gradients``: a rung it does not build, or one it builds left out; a status
    that is not a ``GradientStatus``; an entry status that is not the least of its rungs'; and a rung's switch that
    is not the entry's, or an entry's switch that is no rung's."""
    held = cast(Mapping[str, object], cls.rung_gradients)
    if not held:
        return []
    rungs = tuple(cast(tuple[str, ...], getattr(cls, "rungs", ())))
    built = [rung for rung in rungs if rung not in cast(Mapping[str, str], getattr(cls, "reserved_rungs", {}))]
    if sorted(held) != sorted(built):
        return [f"rung_gradients names the rungs {sorted(held)}, and it builds {sorted(built)}: declare one for each "
                f"built rung, or none"]
    found: list[str] = []
    statuses: list[GradientStatus] = []
    switches: list[switch.Declaration] = []
    for rung, record in held.items():
        if not (isinstance(record, RungGradient) and isinstance(cast(object, record.differentiability),
                                                                 GradientStatus)):
            found.append(f"rung_gradients[{rung!r}] is not a RungGradient with a mime.gradients.GradientStatus")
            continue
        statuses.append(record.differentiability)
        switches += record.gradient.switches if record.gradient is not None else ()
    least = min(statuses, key=RANK.__getitem__, default=None)
    if least is not None and RANK[least] != RANK[cls.differentiability]:
        found.append(f"it is declared {cls.differentiability.value}, and the least of its rungs' statuses is "
                     f"{least.value}: an entry with rungs of their own is as good as its least rung")
    own = cls.gradient.switches if cls.gradient is not None else ()
    if {d.owner for d in switches} != {d.owner for d in own} or any(d not in own for d in switches):
        found.append(f"its rungs declare the switches {sorted(d.owner for d in switches)}, and its gradient record "
                     f"{sorted(d.owner for d in own)}: the entry's record holds every rung's switches, each once")
    return found


class WallForm(Enum):
    """How a wall kind gives a solver the wall's effect on the flow. ``ANALYTIC``: a correction to the free-space
    kernel for an idealised wall (its image system, or a table of it), so that only the body's samples are unknowns;
    free space is its trivial case, with no correction. ``MESHED``: the wall's own surface samples join the unknowns
    with the free-space kernel, for any wall a geometry kind can sample; none is built yet."""

    ANALYTIC = "analytic"
    MESHED = "meshed"


@dataclass(frozen=True, kw_only=True)
class WallKind(Entry):
    """A wall kind: how a confining wall reaches a solver of the flow, declared apart from any one solver. It declares
    ``matches``, the shape of the region a medium fills that it matches exactly, by the name the geometry kind's region
    already exposes (a tube's lumen is ``cylinder``), or ``None`` for a medium that fills no region; ``reads``, the
    parameters of that shape it reads (it never defines the surface: the geometry kind stays its one authority); its
    ``form``; its validity lines about the wall; and its gradient status, as any entry does.

    What a solver does with it is that solver's, registered by the solver under the wall kind's name and version (the
    boundary-element method's correction: ``mime.nodes.hydro.bem``), so the declaration is shared by every solver
    family and a solver's node receives its own correction, never the geometry kind. A run uses one wall kind: a wall
    model is never switched mid-run."""

    form: ClassVar[WallForm]
    matches: ClassVar[str | None]
    reads: ClassVar[tuple[str, ...]] = ()


type Tensor = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class Solid(Geometry):
    """A geometry kind with a closed-form volume and inertia: the solid a part is. In the part's own frame, centred on
    its centroid, it gives its ``volume`` (m^3), ``inertia_per_mass``, the inertia tensor about its centroid divided by
    its mass (m^2), for a uniform density, and ``enclosing_radius``, given its settings in SI."""

    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        raise NotImplementedError

    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        raise NotImplementedError

    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        """The radius of the smallest sphere about its centroid that holds it (m): the distance a point field's
        model is judged against, as a magnet's near field is."""
        raise NotImplementedError


@dataclass(frozen=True)
class Signal:
    """A value a controller reads or writes under one of its labels: its units, spelt as the vocabulary spells them,
    and its shape, ``-1`` for a size the experiment's binding decides (an arm's number of joints)."""

    units: str
    shape: tuple[int, ...] = ()


type Arrays = Mapping[str, Any]
type ControlStep = Callable[[Arrays, Arrays], tuple[Arrays, Arrays]]


@dataclass(frozen=True)
class ControlLaw:
    """A controller made ready to run from its settings: ``state``, what it carries from one tick to the next, as it
    stands before the first; ``held``, its writes in force before its first write lands, under a declared latency;
    and ``step``, ``(state, reads) -> (state, writes)``, each a mapping of arrays by label. The run calls ``step`` once
    every control period with what the experiment binds to its read labels, and feeds each write to the command bound
    to its label.

    Inside the compiled step (mode 1) the very same ``step`` is traced once, so it must be a pure function of its
    arguments: whatever changes between ticks lives in ``state``. A step that reads Python state its own calls change
    (a counter, a global, a random generator) traces to a different program each time, and is refused for mode 1
    (``controller.impure``). A step with a side effect on the world (``print``, a file, a list it appends to) is not
    seen by any trace: mode 1 runs the effect once, at the build, and not at its ticks, where mode 2 runs it at every
    tick, so the two modes differ without any refusal."""

    state: Arrays
    held: Arrays
    step: ControlStep


#: What a controller placed as a callback (mode 2) gives, whatever its law: the binding's, not the entry's. The run
#: calls the step on the host between compiled chunks, so a chunk takes its writes as fixed inputs.
CALLBACK_GRADIENT = RungGradient(
    differentiability=GradientStatus.CUT,
    gradient=GradientRecord(
        where="the controller's writes: a compiled chunk takes them as fixed inputs, so a derivative of the run "
              "stops at them (it is zero through them)",
        reason="the run calls the controller's step on the host between compiled chunks, where no derivative passes "
               "through a call",
        route="place the controller inside the compiled step, where its status is its law's own, derived from "
              "the switches it declares"))


@dataclass(frozen=True, kw_only=True)
class Controller(Entry):
    """What closes a loop: it reads what the experiment binds to its read labels and writes the apparatus commands
    bound to its write labels, once every control period. Declared with ``Experiment.set_controller``, which binds the
    labels and gives its mode, rate and latency; its fields are its settings, as an interaction's are.

    ``reads`` and ``writes`` declare its labels, each a :class:`Signal`. :meth:`law` makes it ready to run from its
    settings in SI (``ctx.scalar``, ``ctx.vector``; ``ctx.timestep`` is its control period): its state at the start,
    the writes held before its first lands, and its step. In the callback mode (mode 2) the run calls the step on the
    host between compiled chunks with concrete arrays, so it may be any Python. In the mode inside the compiled step
    (mode 1) the very same step is traced, so it must be written in ``jax.numpy`` and be a pure function of its state
    and reads (see :class:`ControlLaw`); a step that cannot be traced is refused for that mode and stays a callback.
    The two modes also differ in what a run may be: mode 1 runs any number of steps, mode 2 only whole control
    periods."""

    reads: ClassVar[Mapping[str, Signal]] = _EMPTY
    writes: ClassVar[Mapping[str, Signal]] = _EMPTY
    #: The status of the controller placed as a callback, recorded beside ``differentiability``, which is its law's
    #: own, inside the compiled step. The binding's: no entry declares it.
    callback_gradient: ClassVar[RungGradient] = CALLBACK_GRADIENT

    def law(self, ctx: BuildContext) -> ControlLaw:
        raise NotImplementedError
