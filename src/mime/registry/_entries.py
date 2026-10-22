from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, override

from mime.gradients import GradientStatus
from mime.things import ThingKind
from mime.units import Dimension, Quantity
from mime.vocabulary import Capability

if TYPE_CHECKING:
    from mime.build import (
        DynamicsBuilder,
        InteractionBuilder,
        ObservationBuilder,
        ProviderBuilder,
        ResolvedRegistration,
        Slot,
    )

_EMPTY: Mapping[str, Any] = MappingProxyType({})


class RoleKind(Enum):
    """What a role of an interaction or observation binds."""

    BODY = "body"
    MEDIUM = "medium"
    STRUCTURE = "structure"
    APPARATUS = "apparatus"
    FIELD = "field"

    @property
    def thing_kind(self) -> ThingKind | None:
        return None if self is RoleKind.FIELD else ThingKind(self.value)


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
class GroupRequest:
    """A request that the dynamics of the bound ``roles`` (and the entry's own nodes, with ``own_nodes``) be solved
    together in one coupling group. Overlapping requests are merged; every request states its tolerance and cap."""

    roles: tuple[str, ...]
    reason: str
    tolerance: float
    max_iterations: int
    own_nodes: bool = False


@dataclass(frozen=True)
class Plan:
    """What an interaction or observation will build, as data: its nodes by local name, its reads and deliveries by
    label, its registrations, its solve-together requests and its random streams."""

    nodes: tuple[str, ...] = ("",)
    reads: Mapping[str, Read] = field(default_factory=dict[str, Read])
    delivers: Mapping[str, Delivery] = field(default_factory=dict[str, Delivery])
    registrations: tuple[Registration, ...] = ()
    groups: tuple[GroupRequest, ...] = ()
    streams: tuple[str, ...] = ()


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
class PlanContext:
    """What an entry may consult while planning: run settings in SI, its bindings by role, and the capability record
    of each bound thing's dynamics or bound field's provider."""

    dt: float
    up: tuple[float, float, float]
    gravity: tuple[float, float, float]
    bindings: Mapping[str, str]
    capabilities: Mapping[str, CapabilityRecord | None]


@dataclass(frozen=True)
class BuildContext:
    """What an entry may consult while building. Every node it adds is named ``node_name(local)`` and stepped at
    ``timestep``; ``seeds`` holds one derived seed per declared random stream."""

    id: str
    dt: float
    timestep: float
    prefix: str
    bindings: Mapping[str, str] = field(default_factory=dict[str, str])
    seeds: Mapping[str, int] = field(default_factory=dict[str, int])
    initial: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def node_name(self, local: str = "") -> str:
        return self.prefix if not local else f"{self.prefix}.{local}"


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


@dataclass(frozen=True, kw_only=True)
class Entry:
    """Every registered kind. An entry is a frozen dataclass whose fields are its settings; ``name`` and ``version``
    are its registry key, and the build records both. ``validity`` holds declared validity lines, listed in the
    build report and not yet checked."""

    name: ClassVar[str]
    version: ClassVar[str]
    differentiability: ClassVar[GradientStatus]
    validity: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def label(cls) -> str:
        return f"{cls.name} {cls.version}"


@dataclass(frozen=True, kw_only=True)
class Dynamics(Entry):
    """How one aspect of one thing evolves. The experiment builds it once, with every registration addressed to it.

    ``publishes`` maps state interfaces to the node fields that hold them, ``accepts`` target interfaces to the node
    inputs that take them, and ``registration_kinds`` names the registration records it can take. ``initial_state``
    maps each value it takes from a body's ``State`` to that value's dimension.
    """

    applies_to: ClassVar[frozenset[ThingKind]]
    flags: ClassVar[frozenset[Capability]] = frozenset()
    publishes: ClassVar[Mapping[str, Port]] = _EMPTY
    accepts: ClassVar[Mapping[str, Port]] = _EMPTY
    registration_kinds: ClassVar[frozenset[str]] = frozenset()
    initial_state: ClassVar[Mapping[str, Dimension]] = MappingProxyType({})
    nodes: ClassVar[tuple[str, ...]] = ("",)
    rate: Quantity | None = None

    def capabilities(self) -> CapabilityRecord:
        return CapabilityRecord(self.label(), frozenset(self.flags), dict(self.publishes), dict(self.accepts),
                                frozenset(self.registration_kinds))

    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class FieldProvider(Entry):
    """What produces a field. ``publishes`` maps each field interface to the prefix of its output ports: a reader
    slot ``s`` reads its points into ``points_<s>`` and the field out of ``<prefix>_<s>``."""

    flags: ClassVar[frozenset[Capability]]
    publishes: ClassVar[Mapping[str, str]]

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
    """

    roles: ClassVar[Mapping[str, RoleKind]]
    id_roles: ClassVar[tuple[str, ...] | None] = None
    rungs: ClassVar[tuple[str, ...]] = ()
    support: ClassVar[tuple[SupportRow, ...]] = ()
    reads: ClassVar[Mapping[str, Read]] = _EMPTY
    delivers: ClassVar[Mapping[str, Delivery]] = _EMPTY

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers))

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
class Controller(Entry):
    """Reads observations and writes apparatus commands; registered here, declared on an experiment once the
    controller's modes are built."""

    rate: Quantity | None = None
