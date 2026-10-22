from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_NAME = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
MIME_NAMESPACE = "mime"


class TargetKind(Enum):
    """How the deliveries into one target of a dynamics combine.

    Load, kinematic and operator targets sum every delivery. An equality constraint takes at most one. An inequality
    constraint and a registration are not delivered by an edge: the receiving dynamics takes them when it is
    constructed.
    """

    LOAD = "load"
    KINEMATIC = "kinematic"
    OPERATOR = "operator"
    EQUALITY_CONSTRAINT = "equality constraint"
    INEQUALITY_CONSTRAINT = "inequality constraint"
    REGISTRATION = "registration"

    @property
    def additive(self) -> bool:
        return self in (TargetKind.LOAD, TargetKind.KINEMATIC, TargetKind.OPERATOR)

    @property
    def exclusive(self) -> bool:
        return self is TargetKind.EQUALITY_CONSTRAINT

    @property
    def delivered_by_edge(self) -> bool:
        return self.additive or self.exclusive


class ReadKind(Enum):
    """What a reader reads: a dynamics' published state, or a field evaluated at points."""

    STATE = "state"
    FIELD = "field"


class Capability(Enum):
    """The closed list of flags a dynamics or field provider may claim. Support tables match on these, never on model
    names, so a lab's own dynamics is supported wherever its flags are. MIME maintains the list; a flag is added here,
    not by a lab."""

    RIGID = "rigid"
    INERTIAL = "inertial"
    ACCEPTS_PART_TARGETS = "accepts_part_targets"
    PROJECTS_CONFINEMENT = "projects_confinement"
    HYDROSTATIC = "hydrostatic"
    RESISTANCE_IMPLICIT = "resistance_integration=implicit"
    RESISTANCE_EXPLICIT = "resistance_integration=explicit"
    PROVISION_EVALUATED = "provision=evaluated"
    PROVISION_SOLVED = "provision=solved"
    PROVISION_TABULATED = "provision=tabulated"


@dataclass(frozen=True)
class Interface:
    """One named port that dynamics publish or accept and that entries read or deliver into.

    ``units`` are SI symbols, for the report and for readers: arrays crossing an edge carry SI values and no units.
    ``dtype`` is ``"run"`` (the run's precision) or ``"float64"``. Every interface that carries values declares a
    typical magnitude, which a coupling group's acceleration weighs fields by. A reserved entry fixes its kind,
    units, frame and sign now and leaves its shape open (``None``) until it is built; using it is refused until
    then.
    """

    name: str
    kind: TargetKind | ReadKind
    units: str
    frame: str | None
    shape: tuple[int, ...] | None
    sign: str = ""
    dtype: str = "run"
    magnitude: float | None = None
    reserved: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"interface name {self.name!r}: lower-case words joined by dots, such as 'body.pose'")
        if not isinstance(self.kind, (TargetKind, ReadKind)):
            raise TypeError(f"interface {self.name!r}: kind is a TargetKind or a ReadKind, not {self.kind!r}")
        if self.dtype not in ("run", "float64"):
            raise ValueError(f"interface {self.name!r}: dtype is 'run' or 'float64', not {self.dtype!r}")
        if self.reserved:
            if self.shape is not None:
                raise ValueError(f"reserved interface {self.name!r}: its shape stays open (None) until it is built")
        else:
            if self.shape is None:
                raise ValueError(f"interface {self.name!r}: only a reserved entry leaves its shape open")
            carries_values = isinstance(self.kind, ReadKind) or self.kind.delivered_by_edge
            if carries_values and not (isinstance(self.magnitude, (int, float)) and self.magnitude > 0):
                raise ValueError(f"interface {self.name!r}: declare a typical magnitude above zero")

    @property
    def namespace(self) -> str:
        return namespace_of(self.name)


_namespaces: dict[str, str] = {}
_entries: dict[str, Interface] = {}


def register_namespace(name: str, version: str) -> None:
    """Declare a namespace other than MIME's and its vocabulary version. Its names are prefixed ``<name>.``."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name == MIME_NAMESPACE:
        raise ValueError(f"namespace {name!r}: one lower-case word, and not {MIME_NAMESPACE!r}")
    held = _namespaces.get(name)
    if held is not None and held != version:
        raise ValueError(f"namespace {name!r} is registered at version {held}, not {version}")
    _namespaces[name] = version


def namespace_of(name: str) -> str:
    """The namespace a vocabulary or registry name belongs to: its first word when that is a registered namespace,
    and MIME's otherwise (MIME's own names are unprefixed)."""
    first = name.split(".", 1)[0]
    return first if first in _namespaces else MIME_NAMESPACE


def versions() -> dict[str, str]:
    """The vocabulary version of every namespace, MIME's included."""
    from mime import __version__

    return {MIME_NAMESPACE: __version__, **dict(sorted(_namespaces.items()))}


def define(interface: Interface) -> Interface:
    """Add an entry. Defining the same entry again changes nothing; a different entry under a held name is refused."""
    held = _entries.get(interface.name)
    if held is not None and held != interface:
        raise ValueError(f"interface {interface.name!r} is already defined differently: {held}")
    _entries[interface.name] = interface
    return interface


def lookup(name: str) -> Interface | None:
    return _entries.get(name)


def names() -> list[str]:
    return sorted(_entries)
