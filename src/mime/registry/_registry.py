from __future__ import annotations

import dataclasses
import functools
import inspect
import math
import re
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, cast

import numpy as np
from numpy.typing import NDArray

from mime.core import shape_named, shape_names
from mime.gradients import GradientStatus
from mime.registry._entries import (
    Commanded,
    CommandedRate,
    Controller,
    DefaultKind,
    DimensionedSetting,
    Direction,
    Drive,
    Dynamics,
    Entry,
    FieldProvider,
    Flag,
    Geometry,
    Interaction,
    LiveInput,
    Observation,
    Output,
    OutputKind,
    Reduction,
    ReynoldsRange,
    Role,
    RoleKind,
    SettingGroup,
    SettingKind,
    SettingRole,
    Signal,
    SupportRow,
    WallForm,
    WallKind,
    rung_gradient_problems,
)
from mime.things import ThingKind
from mime.units import SAY_THE_ANGLE, TIME, Dimension, Quantity, UnitError, says_its_angle, si
from mime.vocabulary import MIME_NAMESPACE, Capability, EnergyClass, ReadKind, namespace_of, port_kind
from mime.vocabulary import lookup as interface_named

_NAME = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
#: A controller's read or write label.
_LABEL = re.compile(r"[a-z][a-z0-9_]*")
_PROVISIONS = frozenset({Capability.PROVISION_EVALUATED, Capability.PROVISION_SOLVED, Capability.PROVISION_TABULATED})
#: An output's name and a flag's field: a result reads an output as an attribute (``result[swim].speed``).
_FIELD = re.compile(r"[a-z][a-z0-9_]*")


class EntryKind(Enum):
    INTERACTION = "interaction"
    OBSERVATION = "observation"
    DYNAMICS = "dynamics"
    FIELD_PROVIDER = "field provider"
    CONTROLLER = "controller"
    DRIVE = "drive"
    GEOMETRY = "geometry"
    WALL_KIND = "wall kind"


_BASES: dict[EntryKind, type[Entry]] = {
    EntryKind.INTERACTION: Interaction,
    EntryKind.OBSERVATION: Observation,
    EntryKind.DYNAMICS: Dynamics,
    EntryKind.FIELD_PROVIDER: FieldProvider,
    EntryKind.CONTROLLER: Controller,
    EntryKind.DRIVE: Drive,
    EntryKind.GEOMETRY: Geometry,
    EntryKind.WALL_KIND: WallKind,
}


@dataclass(frozen=True)
class Registered:
    """A registered entry class. ``settings`` holds the dimension each of its dimensioned settings declares."""

    kind: EntryKind
    name: str
    version: str
    cls: type[Entry]
    settings: Mapping[str, DimensionedSetting] = field(default_factory=dict[str, DimensionedSetting])
    #: Each setting's declared role and default kind (merged along the class's bases); a setting absent here
    #: declares none, and is required.
    roles: Mapping[str, SettingRole] = field(default_factory=dict[str, SettingRole])

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


def _groups_of(hint: object) -> list[type[SettingGroup]]:
    """The setting groups a setting's annotation holds (``Lubrication | None``; a flow's ``PeriodicBox |
    VesselRegion``), if any."""
    return [m for m in _members(hint) if isinstance(m, type) and issubclass(m, SettingGroup)]


def dimensioned_settings(cls: type) -> tuple[dict[str, DimensionedSetting], list[str]]:
    """The dimension each setting of ``cls`` declares in its annotation, and every problem with those declarations:
    a marker on a setting that does not hold a :class:`Quantity`, a :class:`Quantity` with no marker, a default of
    another dimension. A setting holding a :class:`SettingGroup` contributes its group's dimensioned settings as
    ``<setting>.<field>``, optional where the setting may be ``None`` or another group it may hold has no such field
    (a flow's ``domain.length``, a periodic box's and not a vessel region's)."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception as error:  # noqa: BLE001 - a lab's annotation that does not resolve becomes a refusal naming it
        return {}, [f"its annotations do not resolve at registration ({error!r}); define every name they use at "
                    f"module level"]
    found: dict[str, DimensionedSetting] = {}
    problems: list[str] = []
    for setting in dataclasses.fields(cls):
        hint: object = hints.get(setting.name)
        groups = _groups_of(hint)
        if groups:
            if issubclass(cls, SettingGroup):
                problems.append(f"setting {setting.name!r} holds the group {groups[0].__name__}: a group holds no "
                                f"group")
                continue
            optional = type(None) in _members(hint)
            for group in groups:
                inner, inner_problems = dimensioned_settings(group)
                problems += [f"in its group {setting.name!r}: {problem}" for problem in inner_problems]
                for name, declared in inner.items():
                    key = f"{setting.name}.{name}"
                    held = found.get(key)
                    if held is not None and held.dimension != declared.dimension:
                        problems.append(f"setting {key!r} is of two dimensions in the groups {setting.name!r} holds")
                    everywhere = all(name in _settings(other) for other in groups)
                    found[key] = dataclasses.replace(declared, optional=declared.optional or optional or not everywhere)
            continue
        if typing.get_origin(hint) is Annotated:
            held, *metadata = typing.get_args(hint)
            kinds = [m for m in metadata if isinstance(m, SettingKind)]
            if kinds:
                if len(kinds) > 1 or any(isinstance(m, Dimension) for m in metadata):
                    problems.append(f"setting {setting.name!r} is annotated {hint!r}: one marker, a dimension or "
                                    f"one of {Direction!r}, {Commanded!r} and {CommandedRate!r}")
                elif kinds[0] in (Commanded, CommandedRate) and Quantity not in _members(held):
                    problems.append(f"setting {setting.name!r} is marked {kinds[0]!r} and holds no Quantity")
                else:
                    found[setting.name] = DimensionedSetting(None, optional=type(None) in _members(held),
                                                             kind=kinds[0])
                continue
            markers = [m for m in metadata if isinstance(m, Dimension)]
            if not markers:
                continue
            members = set(_members(held))
            if len(markers) > 1 or Quantity not in members or not members <= {Quantity, type(None)}:
                problems.append(f"setting {setting.name!r} is annotated {hint!r}: a dimensioned setting holds a "
                                f"Quantity (or None) and declares one dimension, as Annotated[Quantity, Length]")
                continue
            if not says_its_angle(markers[0]):
                problems.append(f"setting {setting.name!r} is annotated with a rate per second that says neither "
                                f"turns nor radians: {SAY_THE_ANGLE}")
                continue
            declared = DimensionedSetting(markers[0], optional=type(None) in members)
            default: object = setting.default
            if default is not dataclasses.MISSING and not (default is None and declared.optional):
                try:
                    si(default, markers[0], what=f"its default {setting.name}")
                except (UnitError, TypeError) as error:
                    problems.append(str(error))
            found[setting.name] = declared
        elif setting.name != _RATE and Quantity in _members(hint):
            problems.append(f"setting {setting.name!r} holds a Quantity and declares no dimension: annotate it with "
                            f"its marker from mime.units, as {setting.name}: Annotated[Quantity, Length]")
    return found, problems


def setting_names(cls: type) -> list[str]:
    """An entry class's settings: its dataclass fields, less an observation's bindings (fields named after its
    roles)."""
    if not dataclasses.is_dataclass(cls):
        return []
    bindings: set[str] = set(cast(Mapping[str, object], getattr(cls, "roles", {}))) if issubclass(cls, Observation) \
        else set()
    return [f.name for f in dataclasses.fields(cls) if f.name not in bindings]


def declared_roles(cls: type) -> dict[str, SettingRole]:
    """Each setting's role as ``cls`` and its bases declare it (``setting_roles``), a subclass's winning."""
    found: dict[str, SettingRole] = {}
    for held in reversed(cls.__mro__):
        found.update(cast(Mapping[str, SettingRole], vars(held).get("setting_roles", {})))
    return found


def _written_default(cls: type, name: str) -> object:
    held = next(f for f in dataclasses.fields(cls) if f.name == name)
    if held.default is not dataclasses.MISSING:
        return cast(object, held.default)
    if held.default_factory is not dataclasses.MISSING:
        return cast(object, held.default_factory())
    return dataclasses.MISSING


def role_problems(cls: type) -> list[str]:
    """Every way ``cls``'s role declarations contradict themselves (the consistency of the roles): a role for
    something that is no setting; a default on a physical or design setting that is not a convention or determined;
    a written default with no declared kind; a numerical default on a setting that is not numerical, or without its
    rule, its basis or the method computing it; a convention without its basis or a value."""
    found: list[str] = []
    names = set(setting_names(cls))
    for name, held in sorted(declared_roles(cls).items()):
        where = f"setting {name!r}"
        if not isinstance(cast(object, held), SettingRole):
            found.append(f"setting_roles[{name!r}] is {held!r}, not a mime.registry.SettingRole")
            continue
        if name not in names:
            found.append(f"setting_roles names {name!r}, which is not one of its settings ({', '.join(sorted(names))})")
            continue
        written = _written_default(cls, name)
        given = written is not dataclasses.MISSING and written is not None
        kind = held.default
        if kind is DefaultKind.UNDECLARED:
            found.append(f"{where} declares the kind undeclared, which is how the build reports a default with no "
                         f"declared kind: declare determined, convention, numerical or none")
        if kind is DefaultKind.NONE and given:
            found.append(f"{where} is {held.role.value} and has the default {written!r} with no declared kind: a "
                         f"{held.role.value} setting's default is a claim, so declare it a convention with its basis "
                         f"(convention(...)) or a determined value, or give it none and make it required")
        if kind is DefaultKind.NUMERICAL and held.role is not Role.NUMERICAL:
            found.append(f"{where} is {held.role.value} and declares a numerical default: only a numerical setting "
                         f"takes one; a physical or design setting is required, a convention or determined")
        if kind in (DefaultKind.NUMERICAL, DefaultKind.DETERMINED):
            if not (held.rule.strip() and held.basis.strip()):
                found.append(f"{where} declares a {kind.value} default without its rule and its basis")
            if held.by is None and (kind is DefaultKind.DETERMINED or not given):
                found.append(f"{where} declares a {kind.value} default with neither the method that computes it "
                             f"(by=) nor a written default to be it")
            elif held.by is not None and not callable(getattr(cls, held.by, None)):
                found.append(f"{where} declares a {kind.value} default computed by {held.by!r}, which is no method "
                             f"of the entry")
            elif held.by is not None and given:
                found.append(f"{where}'s {kind.value} default is computed by {held.by!r}, so its written default is "
                             f"None (not given), not {written!r}")
            if held.by is not None and not issubclass(cls, (Interaction, Observation)):
                found.append(f"{where} declares a default computed by {held.by!r}, and the build computes one from the "
                             f"plan's context for an interaction's or an observation's setting only: declare a "
                             f"convention, or a numerical default that is its written value")
        if kind is DefaultKind.CONVENTION and (not held.basis.strip() or written is dataclasses.MISSING):
            found.append(f"{where} declares a convention without its basis, or without a written default to be it")
    found += _group_default_problems(cls)
    return found


def _group_default_problems(cls: type) -> list[str]:
    """A default computed for a setting of a group the entry holds (its role's ``by``) is computed by the entry's own
    method, from the plan's context, as an interaction's or an observation's own setting's is; its written default is
    None (not given)."""
    if not dataclasses.is_dataclass(cls):
        return []
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:  # noqa: BLE001 - dimensioned_settings reports an annotation that does not resolve
        return []
    found: list[str] = []
    for setting in dataclasses.fields(cls):
        group = _group_of(hints.get(setting.name))
        if group is None:
            continue
        for name, held in sorted(declared_roles(group).items()):
            if held.by is None:
                continue
            where = f"setting {setting.name}.{name}"
            if not issubclass(cls, (Interaction, Observation)) or not callable(getattr(cls, held.by, None)):
                found.append(f"{where} declares a default computed by {held.by!r}, which is no method of the entry, "
                             f"an interaction or an observation, that holds its group")
            elif _written_default(group, name) is not None:
                found.append(f"{where}'s default is computed by {held.by!r}, so its written default is None")
    return found


def _output_problems(cls: type[Entry]) -> list[str]:
    """What is wrong with an observation's ``outputs``: none declared, a name declared twice or not a lower-case word,
    a reduction that names no per-step output of its node or no rule, a window that starts at a setting that is not a
    time."""
    held: object = getattr(cls, "outputs", None)
    if not (isinstance(held, tuple) and held and all(isinstance(o, Output) for o in cast(tuple[object, ...], held))):
        return ["outputs: a non-empty tuple of mime.registry.Output, each output its node keeps for a run's result"]
    outputs = cast(tuple[Output, ...], held)
    found: list[str] = []
    names = [output.field for output in outputs]
    found += [f"outputs: {name!r} is declared twice; a result reads each output by its name"
              for name in sorted({n for n in names if names.count(n) > 1})]
    settings, _ = dimensioned_settings(cls)
    per_step = {(o.node, o.field) for o in outputs if o.kind is OutputKind.PER_STEP}
    for output in outputs:
        what = f"output {output.field!r}"
        if not _FIELD.fullmatch(output.field):
            found.append(f"{what}: its name is a lower-case word joined by underscores, its node's state field")
        if not isinstance(cast(object, output.dimension), Dimension) or not isinstance(cast(object, output.kind),
                                                                                      OutputKind):
            found.append(f"{what}: dimension is a mime.units dimension and kind a mime.registry.OutputKind")
        if not all(isinstance(cast(object, size), int) and size >= 0 for size in output.shape):
            found.append(f"{what}: shape {output.shape!r} is a tuple of sizes")
        if (output.reduces is None) != (output.rule is None) or (
                output.rule is not None and not isinstance(cast(object, output.rule), Reduction)):
            found.append(f"{what}: an output the graph reduces names the per-step output it reduces and its rule "
                         f"(a mime.registry.Reduction), and one it does not names neither")
        if output.reduces is not None and (output.kind is not OutputKind.AT_END
                                           or (output.node, output.reduces) not in per_step):
            found.append(f"{what}: it reduces {output.reduces!r}, which is to be a per-step output of its node, and "
                         f"it is then an output at the end")
        if output.after is not None:
            declared = settings.get(output.after)
            if output.kind is not OutputKind.AT_END or declared is None or declared.dimension != TIME:
                found.append(f"{what}: its window starts at the setting {output.after!r}, which is to be a setting "
                             f"of dimension time, and it is then an output at the end")
    return found


def _field_problems(cls: type[Entry], applies: list[object]) -> list[str]:
    """What is wrong with a dynamics' solved-field declarations (``Dynamics.field_declarations``): a dynamics a medium
    carries is a field kind and declares at least one, each with no problem of its own (SF-2's among them: it provides
    at least one quantity) and each quantity an interface of the vocabulary a reading gives; any other declares none."""
    from mime.fields._solved import SolvedField

    try:
        declared: object = getattr(cls, "field_declarations")()
    except Exception as error:  # noqa: BLE001 - a kind's own code, refused naming it
        return [f"field_declarations() raised {error!r}"]
    if not (isinstance(declared, tuple)
            and all(isinstance(d, SolvedField) for d in cast(tuple[object, ...], declared))):
        return ["field_declarations(): a tuple of mime.fields.SolvedField"]
    held = cast(tuple[SolvedField, ...], declared)
    if ThingKind.MEDIUM not in applies:
        return ["field_declarations(): only a field kind a medium carries declares itself under the solved-field "
                "contract"] if held else []
    if not held:
        return ["a dynamics a medium carries is a field kind: it declares itself under the solved-field contract "
                "(field_declarations(), each a mime.fields.SolvedField naming in provides what a law finds it by)"]
    found = [f"its solved-field declaration: {problem}" for d in held for problem in d.problems()]
    for name in dict.fromkeys(name for d in held for name in d.provides):
        interface = interface_named(name)
        if interface is None or not isinstance(interface.kind, ReadKind):
            found.append(f"provides {name!r}, which is not an interface of the vocabulary that a reading gives")
    return list(dict.fromkeys(found))


def _flag_problems(cls: type[Entry]) -> list[str]:
    held: object = getattr(cls, "raises", ())
    if not (isinstance(held, tuple) and all(isinstance(f, Flag) for f in cast(tuple[object, ...], held))):
        return ["raises: a tuple of mime.registry.Flag, each a state field its node raises"]
    rungs = tuple(cast(tuple[str, ...], getattr(cls, "rungs", ())))
    found: list[str] = []
    for flag in cast(tuple[Flag, ...], held):
        if not _FIELD.fullmatch(flag.field) or not flag.meaning.strip():
            found.append(f"raises {flag.field!r}: a state field's name, with what it means when it is raised")
        if flag.rung is not None and flag.rung not in rungs:
            found.append(f"raises {flag.field!r}: on rung {flag.rung!r}, which is not one of its rungs {rungs}")
    return found


def _reynolds_problems(cls: type[Entry]) -> list[str]:
    held: object = getattr(cls, "reynolds", ())
    if not (isinstance(held, tuple) and all(isinstance(r, ReynoldsRange) for r in cast(tuple[object, ...], held))):
        return ["reynolds: a tuple of mime.registry.ReynoldsRange, each the Reynolds number a rung is declared up to"]
    rungs = tuple(cast(tuple[str, ...], getattr(cls, "rungs", ())))
    found: list[str] = []
    seen: set[str | None] = set()
    for declared in cast(tuple[ReynoldsRange, ...], held):
        if not (isinstance(declared.upper, float) and math.isfinite(declared.upper) and declared.upper > 0.0):
            found.append(f"reynolds: the bound {declared.upper!r} is not a number above zero")
        if not declared.basis.strip():
            found.append(f"reynolds: the bound {declared.upper!r} declares no basis, and a bound without one is a "
                         f"guess")
        if declared.rung is not None and declared.rung not in rungs:
            found.append(f"reynolds: a bound on rung {declared.rung!r}, which is not one of its rungs {rungs}")
        if declared.rung in seen:
            found.append(f"reynolds: two bounds for {'every rung' if declared.rung is None else declared.rung}")
        seen.add(declared.rung)
    return found


def _energy_problems(kind: EntryKind, held: Any) -> list[str]:
    """What is wrong with what an interaction or a dynamics says of itself for the energy ledger: an interaction's
    class an ``EnergyClass`` or none, a dynamics' port kinds a tuple of names (no two of them pairing one interface: a
    delivery is paired by the one kind whose deliveries name its interface), and either's reason for being unmetered a
    string, stated only where it declares no class or port kind."""
    if kind not in (EntryKind.INTERACTION, EntryKind.DYNAMICS):
        return []
    found: list[str] = []
    reason: object = getattr(held, "unmetered", "")
    if not isinstance(reason, str):
        found.append("unmetered: the reason the energy ledger cannot meter it, a string")
    declared: object = getattr(held, "energy_class" if kind is EntryKind.INTERACTION else "port_kinds", None)
    if kind is EntryKind.INTERACTION and declared is not None and not isinstance(declared, EnergyClass):
        found.append("energy_class: a mime.vocabulary.EnergyClass, or None for a law the energy ledger cannot meter")
    if kind is EntryKind.DYNAMICS:
        names = cast(tuple[object, ...], declared) if isinstance(declared, tuple) else None
        if names is None or not all(isinstance(name, str) and name for name in names):
            found.append("port_kinds: a tuple of the names of registered port kinds (mime.vocabulary.port_kinds), or "
                         "empty")
        else:
            paired: dict[str, str] = {}
            for name in cast(tuple[str, ...], names):
                held_kind = port_kind(name)
                for interface in held_kind.deliveries if held_kind is not None else ():
                    if interface in paired:
                        found.append(f"port_kinds: {paired[interface]!r} and {name!r} both pair {interface!r}; a "
                                     f"delivery is paired by one kind")
                    paired[interface] = name
    if declared and reason:
        found.append("unmetered states a reason for being unmetered, and it declares "
                     + ("its class" if kind is EntryKind.INTERACTION else "a port kind"))
    return found


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
    else:
        found += rung_gradient_problems(cls)
    roles: object = getattr(held, "roles", None)
    role_names: set[str] = set(cast(Mapping[str, object], roles)) if isinstance(roles, Mapping) else set()
    switch_only = kind is EntryKind.OBSERVATION and getattr(held, "meters_energy", False) is True
    if switch_only and (role_names or getattr(held, "outputs", ()) or getattr(held, "reads", {})):
        found.append("an observation that meters energy is a switch: it binds no role, reads nothing and declares no "
                     "output (its account is the metered receivers' records)")
    elif kind in (EntryKind.INTERACTION, EntryKind.OBSERVATION) and not switch_only:
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
    found += _flag_problems(cls)
    found += _reynolds_problems(cls)
    found += _energy_problems(kind, held)
    if kind is EntryKind.OBSERVATION and not switch_only:
        found += _output_problems(cls)
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
        rungs = tuple(cast(tuple[str, ...], getattr(held, "rungs", ())))
        if rungs and "rung" not in _settings(cls):
            found.append("a dynamics with rungs has a 'rung' setting, with no default")
        reserved = cast(Mapping[str, str], getattr(held, "reserved_rungs", {}))
        if not set(reserved) <= set(rungs):
            found.append(f"reserved_rungs names rungs it does not have: {sorted(set(reserved) - set(rungs))}")
        applies: object = getattr(held, "applies_to", None)
        members = list(cast(frozenset[object], applies)) if isinstance(applies, frozenset) else []
        if not (members and all(isinstance(k, ThingKind) for k in members)):
            found.append("applies_to: a non-empty frozenset of ThingKind")
        if not all(isinstance(flag, Capability) for flag in cast(frozenset[object], getattr(held, "flags", ()))):
            found.append("flags: members of mime.vocabulary.Capability, the closed list")
        found += _field_problems(cls, members)
    if kind is EntryKind.GEOMETRY:
        surfaces: object = getattr(held, "surfaces", None)
        for name in ("regions", "directions"):
            names: object = getattr(held, name, None)
            if not (isinstance(names, tuple) and all(isinstance(n, str) for n in cast(tuple[object, ...], names))):
                found.append(f"{name}: a tuple of names")
        if not (isinstance(surfaces, Mapping) and all(isinstance(k, str) and isinstance(v, str) for k, v in
                                                      cast(Mapping[object, object], surfaces).items())):
            found.append("surfaces: a mapping of surface names to the names of the shapes they are evaluated with")
        shaped: object = getattr(held, "region_shapes", None)
        regions: object = getattr(held, "regions", ())
        named = set(cast(tuple[object, ...], regions)) if isinstance(regions, tuple) else set[object]()
        if not (isinstance(shaped, Mapping) and all(k in named and isinstance(v, str) for k, v in
                                                    cast(Mapping[object, object], shaped).items())):
            found.append("region_shapes: a mapping of some of its regions to the names of the shapes they are")
        openings: object = getattr(held, "openings", None)
        surfaced = set(cast(Mapping[object, object], surfaces)) if isinstance(surfaces, Mapping) else set[object]()
        if not (isinstance(openings, Mapping) and all(
                k in named and isinstance(v, tuple) and all(o in surfaced for o in cast(tuple[object, ...], v))
                for k, v in cast(Mapping[object, object], openings).items())):
            found.append("openings: a mapping of some of its regions to the tuple of its surfaces each may be opened "
                         "at, in order along its declared axis")
    if kind is EntryKind.WALL_KIND:
        found += _wall_kind_problems(held)
    if kind is EntryKind.FIELD_PROVIDER:
        flags = set(cast(frozenset[object], getattr(held, "flags", frozenset[object]())))
        if not all(isinstance(flag, Capability) for flag in flags) or len(flags & _PROVISIONS) != 1:
            found.append("flags: Capability members, with exactly one provision (evaluated, solved or tabulated)")
        publishes: object = getattr(held, "publishes", None)
        if not (isinstance(publishes, Mapping) and publishes):
            found.append("publishes: a non-empty mapping of field interfaces to port prefixes")
    if kind is EntryKind.CONTROLLER:
        for name, required in (("reads", False), ("writes", True)):
            labels: object = getattr(held, name, None)
            signals = list(cast(Mapping[object, object], labels).items()) if isinstance(labels, Mapping) else None
            if signals is None or (required and not signals) or not all(
                    isinstance(label, str) and _LABEL.fullmatch(label) and isinstance(signal, Signal)
                    for label, signal in signals):
                found.append(f"{name}: a {'non-empty ' if required else ''}mapping of labels (lower-case words joined "
                             f"by underscores) to mime.registry.Signal")
        found += _live_problems(held)
    return found


def _live_problems(held: Any) -> list[str]:
    """What keeps a controller's live inputs from being read: each a :class:`LiveInput` under a label that is none of
    its reads', its default finite numbers of its shape (a number stands for every element)."""
    live: object = getattr(held, "live", {})
    inputs = list(cast(Mapping[object, object], live).items()) if isinstance(live, Mapping) else None
    reads: object = getattr(held, "reads", {})
    if inputs is None or not all(isinstance(label, str) and _LABEL.fullmatch(label) and isinstance(value, LiveInput)
                                 for label, value in inputs):
        return ["live: a mapping of labels (lower-case words joined by underscores) to mime.registry.LiveInput"]
    found: list[str] = []
    for label, value in cast(list[tuple[str, LiveInput]], inputs):
        if isinstance(reads, Mapping) and label in reads:
            found.append(f"live: {label!r} is also a read label, and the step reads both under their labels")
        if live_default(value) is None:
            found.append(f"live: {label!r}'s default {value.default!r} is not finite numbers of its shape "
                         f"{tuple(value.shape)}")
    return found


def live_default(value: LiveInput) -> NDArray[np.float64] | None:
    """A live input's default as the value a session hands its step, float64 numbers of its shape (a number stands for
    every element), or ``None`` when it is not finite numbers of that shape."""
    try:
        held = np.asarray(value.default, dtype=np.float64)
        shaped = np.broadcast_to(held, tuple(value.shape)) if held.ndim == 0 else held
    except (TypeError, ValueError):
        return None
    if shaped.shape != tuple(value.shape) or not np.isfinite(shaped).all():
        return None
    return np.array(shaped, dtype=np.float64)


def _wall_kind_problems(held: Any) -> list[str]:
    """What keeps a wall kind's declaration from being read: its form, the shape it matches (registered, or ``None``)
    and the parameters it reads (that shape's)."""
    found: list[str] = []
    if not isinstance(getattr(held, "form", None), WallForm):
        found.append("form: a mime.registry.WallForm, analytic or meshed")
    matches: object = getattr(held, "matches", ())
    reads: object = getattr(held, "reads", ())
    if not (isinstance(reads, tuple) and all(isinstance(r, str) for r in cast(tuple[object, ...], reads))):
        found.append("reads: a tuple of the names of the parameters it reads from the shape it matches")
    elif matches is None:
        if reads:
            found.append(f"reads {reads}: a wall kind that matches no region reads no parameters")
    elif not isinstance(matches, str):
        found.append("matches: the name of the registered shape of the region it matches, or None for a medium that "
                     "fills no region")
    elif matches not in shape_names():
        found.append(f"matches {matches!r}, which is no registered shape; the shapes are {shape_names()}")
    elif not set(cast(tuple[str, ...], reads)) <= set(shape_named(matches).parameter_names):
        found.append(f"reads {reads}, and the shape {matches!r} has the parameters "
                     f"{shape_named(matches).parameter_names}")
    return found


def rate_problems(cls: type) -> list[str]:
    """Every command (``commands``) and initial state (``initial_state``) ``cls`` declares with a rate per second that
    says neither turns nor radians: a frequency and an angular velocity differ by 2 pi, and a site that takes either
    would take each as the other. Its settings are checked with their dimensions (:func:`dimensioned_settings`)."""
    found: list[str] = []
    commands: object = getattr(cls, "commands", None)
    if isinstance(commands, Mapping):
        for name, command in cast(Mapping[str, object], commands).items():
            dimension = getattr(command, "dimension", None)
            if isinstance(dimension, Dimension) and not says_its_angle(dimension):
                found.append(f"its command {name!r} is declared with a rate per second that says neither turns nor "
                             f"radians: {SAY_THE_ANGLE}")
    initial: object = getattr(cls, "initial_state", None)
    if isinstance(initial, Mapping):
        for name, dimension in cast(Mapping[str, object], initial).items():
            if isinstance(dimension, Dimension) and not says_its_angle(dimension):
                found.append(f"its initial {name!r} is declared with a rate per second that says neither turns nor "
                             f"radians: {SAY_THE_ANGLE}")
    return found


#: The key of a setting's ``dataclasses.field(metadata=...)`` that says why it has no default, in the words a refusal
#: of a construction without it prints (``starts: int = field(metadata={WHY_REQUIRED: "its number of starts, ..."})``).
WHY_REQUIRED = "mime_why_required"
#: Why a required setting of each role has no default, where its field does not say.
_ROLE_REQUIRES = {Role.PHYSICAL: "a physical setting, a fact about the setup, which MIME does not assume",
                  Role.DESIGN: "a design setting, the experimenter's choice, which MIME does not make",
                  Role.NUMERICAL: "a numerical setting with no stated rule for a default",
                  Role.STANDARD: "a standard setting with no default written"}


def _setting_written(held: dataclasses.Field[Any], hint: object) -> str:
    """How a script writes the setting ``held``, its kind of value in angle brackets: ``radius=<a quantity of length
    (m)>``, ``starts=<an int>``."""
    if typing.get_origin(hint) is Annotated:
        held_type, *metadata = typing.get_args(hint)
        marker = next((m for m in metadata if isinstance(m, Dimension)), None)
        if marker is not None:
            return f"{held.name}=<a quantity of {marker}>"
        if Direction in metadata:
            return f"{held.name}=<a direction, such as X or (1.0, 0.0, 0.0)>"
        hint = held_type
    named = getattr(hint, "__name__", None)
    return f"{held.name}=<{'an' if str(named)[:1] in 'aeiou' else 'a'} {named}>" if isinstance(named, str) else \
        f"{held.name}=..."


def missing_settings_refusal(cls: type, missing: list[str]) -> tuple[str, str]:
    """What a construction of ``cls`` without its required settings ``missing`` is told: each setting with why it has
    no default (its field's :data:`WHY_REQUIRED`, else its role's), and the settings as the call writes them."""
    roles = declared_roles(cls)
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:  # noqa: BLE001 - a hint that does not resolve is refused at registration, never here
        hints = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    reasons: list[str] = []
    for name in missing:
        why = fields[name].metadata.get(WHY_REQUIRED)
        role = roles.get(name)
        if not isinstance(why, str):
            why = _ROLE_REQUIRES[role.role] if role is not None else "a setting its entry gives no default"
        reasons.append(f"{name}, {why}")
    return "; ".join(reasons), ", ".join(_setting_written(fields[name], hints.get(name)) for name in missing)


def _missing(signature: inspect.Signature, args: tuple[object, ...], kwargs: dict[str, object]) -> list[str]:
    """The required parameters a call of ``signature`` with ``args`` and ``kwargs`` leaves out, where that is all that
    is wrong with its arguments; empty otherwise (an unknown keyword, one too many, or nothing missing)."""
    names = list(signature.parameters)
    if not set(kwargs) <= set(names) or len(args) > len(names):
        return []
    given = set(kwargs) | set(names[:len(args)])
    return [name for name in names if name not in given
            and signature.parameters[name].default is inspect.Parameter.empty]


def _naming_missing_settings(cls: type) -> None:
    """Make ``cls``'s own dataclass constructor refuse a call without a required setting by name, with why it is
    required (an ``ArgumentTypeRefused``, ``call.missing-setting``), where Python's own error names the argument
    alone; any other error of the call is left as it is."""
    init = vars(cls).get("__init__")
    if init is None or getattr(init, "__mime_names_missing__", False):
        return
    signature = inspect.signature(init)

    @functools.wraps(init)
    def constructed(self: object, *args: object, **kwargs: object) -> None:
        try:
            init(self, *args, **kwargs)
        except TypeError:
            missing = _missing(signature, (self, *args), kwargs)
            if not missing:
                raise
            from mime.build._refusals import ArgumentTypeRefused, Refusal, RefusalClass

            why, written = missing_settings_refusal(cls, missing)
            label = cls.__qualname__
            raise ArgumentTypeRefused([Refusal(
                "call.missing-setting", RefusalClass.USER,
                f"{label}(...) is not given {', '.join(map(repr, missing))}, and "
                f"{'they have' if len(missing) > 1 else 'it has'} "
                f"no default: {why}",
                f"write {label}(..., {written}), with the value of your own setup")]) from None

    setattr(constructed, "__mime_names_missing__", True)
    setattr(cls, "__init__", constructed)


def _register[E: type[Entry]](kind: EntryKind, cls: E) -> E:
    found = _problems(kind, cls)
    found += rate_problems(cls)
    settings: dict[str, DimensionedSetting] = {}
    if dataclasses.is_dataclass(cls):
        settings, problems = dimensioned_settings(cls)
        found += problems
        found += role_problems(cls)
    if found:
        raise TypeError(f"cannot register {getattr(cls, '__qualname__', cls)!r} as a {kind.value}: " + "; ".join(found))
    key = (kind, cls.name, cls.version)
    held = _by_key.get(key)
    if held is not None and held.cls is not cls:
        if held.defined_in != f"{cls.__module__}.{cls.__qualname__}":
            raise ValueError(f"{kind.value} {cls.name} {cls.version} is already registered by {held.defined_in}")
        del _by_class[held.cls]
    record = Registered(kind, cls.name, cls.version, cls, settings, declared_roles(cls))
    _by_key[key] = record
    _by_class[cls] = record
    _naming_missing_settings(cls)
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


def register_drive[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.DRIVE, cls)


def register_geometry[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.GEOMETRY, cls)


def register_wall_kind[E: type[Entry]](cls: E) -> E:
    return _register(EntryKind.WALL_KIND, cls)


def wall_kinds_matching(shape: str | None) -> list[Registered]:
    """The registered wall kinds that match a region of the shape ``shape`` exactly (``None``: a medium that fills no
    region)."""
    return [record for record in registered(EntryKind.WALL_KIND)
            if cast(type[WallKind], record.cls).matches == shape]


_groups: dict[str, type[SettingGroup]] = {}
_GROUP = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")


def register_setting_group[G: type[SettingGroup]](cls: G) -> G:
    """Make a group of settings writable in the saved form under its ``group`` name, once its fields are checked as
    an entry's settings are. Registering the same class again changes nothing."""
    found: list[str] = []
    params: Any = vars(cls).get("__dataclass_params__")
    if params is None or not params.frozen:
        found.append("a setting group is a frozen dataclass: decorate it with @dataclass(frozen=True, kw_only=True)")
    name: object = getattr(cls, "group", None)
    if not (isinstance(name, str) and _GROUP.fullmatch(name)):
        found.append(f"group {name!r}: lower-case words joined by dots, such as 'lubrication'")
    if dataclasses.is_dataclass(cls):
        found += dimensioned_settings(cls)[1]
    if found:
        raise TypeError(f"cannot register {cls.__qualname__!r} as a setting group: " + "; ".join(found))
    held = _groups.get(cls.group)
    if held is not None and held is not cls and f"{held.__module__}.{held.__qualname__}" != \
            f"{cls.__module__}.{cls.__qualname__}":
        raise ValueError(f"setting group {cls.group!r} is already registered by {held.__module__}.{held.__qualname__}")
    _groups[cls.group] = cls
    return cls


def setting_group_named(name: str) -> type[SettingGroup]:
    if name not in _groups:
        raise KeyError(f"no setting group {name!r} is registered; the groups are {sorted(_groups)}")
    return _groups[name]


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
