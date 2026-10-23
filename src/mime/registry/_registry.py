from __future__ import annotations

import dataclasses
import re
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, cast

from mime.gradients import GradientStatus
from mime.registry._entries import (
    Controller,
    DimensionedSetting,
    Dynamics,
    Entry,
    FieldProvider,
    Interaction,
    Observation,
    RoleKind,
    SupportRow,
)
from mime.things import ThingKind
from mime.units import Dimension, Quantity, UnitError, si
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
    """A registered entry class. ``settings`` holds the dimension each of its dimensioned settings declares."""

    kind: EntryKind
    name: str
    version: str
    cls: type[Entry]
    settings: Mapping[str, DimensionedSetting] = field(default_factory=dict[str, DimensionedSetting])

    @property
    def namespace(self) -> str:
        return namespace_of(self.name)

    @property
    def defined_in(self) -> str:
        return f"{self.cls.__module__}.{self.cls.__qualname__}"


_by_key: dict[tuple[EntryKind, str, str], Registered] = {}
_by_class: dict[type, Registered] = {}

#: The run setting an entry may take, a period or a frequency, which phase 1 checks against ``dt`` by its own rule.
_RATE = "rate"


def _settings(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()


def _members(hint: object) -> tuple[object, ...]:
    """The types a union annotation admits, or the annotation alone."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return typing.get_args(hint)
    return (hint,)


def dimensioned_settings(cls: type) -> tuple[dict[str, DimensionedSetting], list[str]]:
    """The dimension each setting of ``cls`` declares in its annotation, and every problem with those declarations:
    a marker on a setting that does not hold a :class:`Quantity`, a :class:`Quantity` with no marker, a default of
    another dimension."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception as error:  # noqa: BLE001 - a lab's annotation that does not resolve becomes a refusal naming it
        return {}, [f"its annotations do not resolve at registration ({error!r}); define every name they use at "
                    f"module level"]
    found: dict[str, DimensionedSetting] = {}
    problems: list[str] = []
    for setting in dataclasses.fields(cls):
        hint: object = hints.get(setting.name)
        if typing.get_origin(hint) is Annotated:
            held, *metadata = typing.get_args(hint)
            markers = [m for m in metadata if isinstance(m, Dimension)]
            if not markers:
                continue
            members = set(_members(held))
            if len(markers) > 1 or Quantity not in members or not members <= {Quantity, type(None)}:
                problems.append(f"setting {setting.name!r} is annotated {hint!r}: a dimensioned setting holds a "
                                f"Quantity (or None) and declares one dimension, as Annotated[Quantity, Length]")
                continue
            declared = DimensionedSetting(markers[0], optional=type(None) in members)
            default: object = setting.default
            if default is not dataclasses.MISSING and not (default is None and declared.optional):
                try:
                    si(default, declared.dimension, what=f"its default {setting.name}")
                except (UnitError, TypeError) as error:
                    problems.append(str(error))
            found[setting.name] = declared
        elif setting.name != _RATE and Quantity in _members(hint):
            problems.append(f"setting {setting.name!r} holds a Quantity and declares no dimension: annotate it with "
                            f"its marker from mime.units, as {setting.name}: Annotated[Quantity, Length]")
    return found, problems


def _problems(kind: EntryKind, candidate: object) -> list[str]:
    """Everything that keeps ``candidate`` from being registered as a ``kind``; read from a class the registry has
    not checked yet, so every attribute is asked for rather than assumed."""
    base = _BASES[kind]
    if not (isinstance(candidate, type) and issubclass(candidate, base)):
        return [f"a registered {kind.value} subclasses mime.registry.{base.__name__}"]
    cls: type[Entry] = candidate
    held: Any = cls
    found: list[str] = []
    params: Any = vars(cls).get("__dataclass_params__")
    if params is None or not params.frozen:
        found.append("an entry is a frozen dataclass, its fields its settings: decorate the class itself with "
                     "@dataclass(frozen=True, kw_only=True)")
    name: object = getattr(held, "name", None)
    version: object = getattr(held, "version", None)
    if not (isinstance(name, str) and _NAME.fullmatch(name)):
        found.append(f"name {name!r}: lower-case words joined by dots, such as 'drag' or 'labx.slender_tail'")
    elif namespace_of(name) == MIME_NAMESPACE and not cls.__module__.split(".")[0] == "mime":
        found.append(f"name {name!r} is in MIME's namespace; register your own with "
                     f"mime.vocabulary.register_namespace and prefix the name with it")
    if not (isinstance(version, str) and version):
        found.append(f"version {version!r}: a non-empty string")
    if not isinstance(getattr(held, "differentiability", None), GradientStatus):
        found.append("differentiability: declare a mime.gradients.GradientStatus")
    roles: object = getattr(held, "roles", None)
    role_names: set[str] = set(cast(Mapping[str, object], roles)) if isinstance(roles, Mapping) else set()
    if kind in (EntryKind.INTERACTION, EntryKind.OBSERVATION):
        kinds = list(cast(Mapping[str, object], roles).values()) if isinstance(roles, Mapping) else []
        if not (kinds and all(isinstance(k, RoleKind) for k in kinds)):
            found.append("roles: a non-empty mapping of role names to RoleKind")
        else:
            fields = _settings(cls)
            if kind is EntryKind.OBSERVATION and not role_names <= fields:
                found.append(f"an observation's bindings are fields named after its roles; missing: "
                             f"{sorted(role_names - fields)}")
            id_roles: object = getattr(held, "id_roles", None)
            if id_roles is not None and not set(cast(tuple[str, ...], id_roles)) <= role_names:
                found.append(f"id_roles {id_roles} name roles it does not have")
    if kind is EntryKind.INTERACTION:
        rungs = tuple(cast(tuple[str, ...], getattr(held, "rungs", ())))
        if rungs and "rung" not in _settings(cls):
            found.append("an interaction with rungs has a 'rung' setting, with no default")
        for row in cast(tuple[SupportRow, ...], getattr(held, "support", ())):
            if row.rung is not None and row.rung not in rungs:
                found.append(f"support row for rung {row.rung!r}, which is not one of {rungs}")
            if not set(row.when) <= role_names:
                found.append(f"support row names roles it does not have: {sorted(set(row.when))}")
    if kind is EntryKind.DYNAMICS:
        applies: object = getattr(held, "applies_to", None)
        members = list(cast(frozenset[object], applies)) if isinstance(applies, frozenset) else []
        if not (members and all(isinstance(k, ThingKind) for k in members)):
            found.append("applies_to: a non-empty frozenset of ThingKind")
        if not all(isinstance(flag, Capability) for flag in cast(frozenset[object], getattr(held, "flags", ()))):
            found.append("flags: members of mime.vocabulary.Capability, the closed list")
    if kind is EntryKind.FIELD_PROVIDER:
        flags = set(cast(frozenset[object], getattr(held, "flags", frozenset[object]())))
        if not all(isinstance(flag, Capability) for flag in flags) or len(flags & _PROVISIONS) != 1:
            found.append("flags: Capability members, with exactly one provision (evaluated, solved or tabulated)")
        publishes: object = getattr(held, "publishes", None)
        if not (isinstance(publishes, Mapping) and publishes):
            found.append("publishes: a non-empty mapping of field interfaces to port prefixes")
    return found


def _register[E: type[Entry]](kind: EntryKind, cls: E) -> E:
    found = _problems(kind, cls)
    settings: dict[str, DimensionedSetting] = {}
    if dataclasses.is_dataclass(cls):
        settings, problems = dimensioned_settings(cls)
        found += problems
    if found:
        raise TypeError(f"cannot register {getattr(cls, '__qualname__', cls)!r} as a {kind.value}: " + "; ".join(found))
    key = (kind, cls.name, cls.version)
    held = _by_key.get(key)
    if held is not None and held.cls is not cls:
        if held.defined_in != f"{cls.__module__}.{cls.__qualname__}":
            raise ValueError(f"{kind.value} {cls.name} {cls.version} is already registered by {held.defined_in}")
        del _by_class[held.cls]
    record = Registered(kind, cls.name, cls.version, cls, settings)
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
