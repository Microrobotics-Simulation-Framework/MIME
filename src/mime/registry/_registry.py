from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from mime.gradients import GradientStatus
from mime.registry._entries import (
    Controller,
    Dynamics,
    Entry,
    FieldProvider,
    Interaction,
    Observation,
    RoleKind,
)
from mime.things import ThingKind
from mime.vocabulary import MIME_NAMESPACE, Capability, namespace_of

_NAME = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
_PROVISIONS = frozenset({Capability.PROVISION_EVALUATED, Capability.PROVISION_SOLVED, Capability.PROVISION_TABULATED})


class EntryKind(Enum):
    INTERACTION = "interaction"
    OBSERVATION = "observation"
    DYNAMICS = "dynamics"
    FIELD_PROVIDER = "field provider"
    CONTROLLER = "controller"


_BASES: dict[EntryKind, type[Entry]] = {
    EntryKind.INTERACTION: Interaction,
    EntryKind.OBSERVATION: Observation,
    EntryKind.DYNAMICS: Dynamics,
    EntryKind.FIELD_PROVIDER: FieldProvider,
    EntryKind.CONTROLLER: Controller,
}


@dataclass(frozen=True)
class Registered:
    kind: EntryKind
    name: str
    version: str
    cls: type[Entry]

    @property
    def namespace(self) -> str:
        return namespace_of(self.name)

    @property
    def defined_in(self) -> str:
        return f"{self.cls.__module__}.{self.cls.__qualname__}"


_by_key: dict[tuple[EntryKind, str, str], Registered] = {}
_by_class: dict[type, Registered] = {}

def _problems(kind: EntryKind, cls: type) -> list[str]:
    base = _BASES[kind]
    if not (isinstance(cls, type) and issubclass(cls, base)):
        return [f"a registered {kind.value} subclasses mime.registry.{base.__name__}"]
    found = []
    if "__dataclass_params__" not in vars(cls) or not cls.__dataclass_params__.frozen:  # type: ignore[attr-defined]
        found.append("an entry is a frozen dataclass, its fields its settings: decorate the class itself with "
                     "@dataclass(frozen=True, kw_only=True)")
    name, version = getattr(cls, "name", None), getattr(cls, "version", None)
    if not (isinstance(name, str) and _NAME.fullmatch(name)):
        found.append(f"name {name!r}: lower-case words joined by dots, such as 'drag' or 'labx.slender_tail'")
    elif namespace_of(name) == MIME_NAMESPACE and not cls.__module__.split(".")[0] == "mime":
        found.append(f"name {name!r} is in MIME's namespace; register your own with "
                     f"mime.vocabulary.register_namespace and prefix the name with it")
    if not (isinstance(version, str) and version):
        found.append(f"version {version!r}: a non-empty string")
    if not isinstance(getattr(cls, "differentiability", None), GradientStatus):
        found.append("differentiability: declare a mime.gradients.GradientStatus")
    if kind in (EntryKind.INTERACTION, EntryKind.OBSERVATION):
        roles = getattr(cls, "roles", None)
        if not (isinstance(roles, Mapping) and roles and all(isinstance(k, RoleKind) for k in roles.values())):
            found.append("roles: a non-empty mapping of role names to RoleKind")
        else:
            fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
            if kind is EntryKind.OBSERVATION and not set(roles) <= fields:
                found.append(f"an observation's bindings are fields named after its roles; missing: "
                             f"{sorted(set(roles) - fields)}")
            id_roles = getattr(cls, "id_roles", None)
            if id_roles is not None and not set(id_roles) <= set(roles):
                found.append(f"id_roles {id_roles} name roles it does not have")
    if kind is EntryKind.INTERACTION:
        rungs = tuple(getattr(cls, "rungs", ()))
        fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
        if rungs and "rung" not in fields:
            found.append("an interaction with rungs has a 'rung' setting, with no default")
        for row in getattr(cls, "support", ()):
            if row.rung is not None and row.rung not in rungs:
                found.append(f"support row for rung {row.rung!r}, which is not one of {rungs}")
            if not set(row.when) <= set(getattr(cls, "roles", {})):
                found.append(f"support row names roles it does not have: {sorted(set(row.when))}")
    if kind is EntryKind.DYNAMICS:
        applies = getattr(cls, "applies_to", None)
        if not (isinstance(applies, frozenset) and applies and all(isinstance(k, ThingKind) for k in applies)):
            found.append("applies_to: a non-empty frozenset of ThingKind")
        if not all(isinstance(flag, Capability) for flag in getattr(cls, "flags", ())):
            found.append("flags: members of mime.vocabulary.Capability, the closed list")
    if kind is EntryKind.FIELD_PROVIDER:
        flags = getattr(cls, "flags", frozenset())
        if not all(isinstance(flag, Capability) for flag in flags) or len(set(flags) & _PROVISIONS) != 1:
            found.append("flags: Capability members, with exactly one provision (evaluated, solved or tabulated)")
        if not isinstance(getattr(cls, "publishes", None), Mapping) or not cls.publishes:  # type: ignore[attr-defined]
            found.append("publishes: a non-empty mapping of field interfaces to port prefixes")
    return found


def _register[E: type[Entry]](kind: EntryKind, cls: E) -> E:
    found = _problems(kind, cls)
    if found:
        raise TypeError(f"cannot register {getattr(cls, '__qualname__', cls)!r} as a {kind.value}: " + "; ".join(found))
    key = (kind, cls.name, cls.version)
    held = _by_key.get(key)
    if held is not None and held.cls is not cls:
        if held.defined_in != f"{cls.__module__}.{cls.__qualname__}":
            raise ValueError(f"{kind.value} {cls.name} {cls.version} is already registered by {held.defined_in}")
        del _by_class[held.cls]
    record = Registered(kind, cls.name, cls.version, cls)
    _by_key[key] = record
    _by_class[cls] = record
    return cls


def register_interaction[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.INTERACTION, cls)


def register_observation[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.OBSERVATION, cls)


def register_dynamics[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.DYNAMICS, cls)


def register_field_provider[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.FIELD_PROVIDER, cls)


def register_controller[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.CONTROLLER, cls)


def registration_of(entry: Entry | type) -> Registered | None:
    """The registration of an entry or entry class, or ``None`` when it was never registered."""
    return _by_class.get(entry if isinstance(entry, type) else type(entry))


def lookup(kind: EntryKind, name: str, version: str) -> Registered | None:
    return _by_key.get((kind, name, version))


def versions_of(kind: EntryKind, name: str) -> list[str]:
    return sorted(v for (k, n, v) in _by_key if k is kind and n == name)


def registered(kind: EntryKind | None = None) -> list[Registered]:
    return sorted((r for r in _by_key.values() if kind is None or r.kind is kind),
                  key=lambda r: (r.kind.value, r.name, r.version))
