from __future__ import annotations

import dataclasses
import re
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, cast

from mime.core import shape_named, shape_names
from mime.gradients import GradientStatus
from mime.registry._entries import (
    Commanded,
    Controller,
    DimensionedSetting,
    Direction,
    Drive,
    Dynamics,
    Entry,
    FieldProvider,
    Flag,
    Geometry,
    Interaction,
    Observation,
    Output,
    OutputKind,
    Reduction,
    RoleKind,
    SettingGroup,
    SettingKind,
    Signal,
    SupportRow,
    WallForm,
    WallKind,
    rung_gradient_problems,
)
from mime.things import ThingKind
from mime.units import TIME, Dimension, Quantity, UnitError, si
from mime.vocabulary import MIME_NAMESPACE, Capability, namespace_of

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


def _group_of(hint: object) -> type[SettingGroup] | None:
    """The setting group a setting's annotation holds (``Lubrication | None``), if any."""
    groups = [m for m in _members(hint) if isinstance(m, type) and issubclass(m, SettingGroup)]
    return groups[0] if len(groups) == 1 else None


def dimensioned_settings(cls: type) -> tuple[dict[str, DimensionedSetting], list[str]]:
    """The dimension each setting of ``cls`` declares in its annotation, and every problem with those declarations:
    a marker on a setting that does not hold a :class:`Quantity`, a :class:`Quantity` with no marker, a default of
    another dimension. A setting holding a :class:`SettingGroup` contributes its group's dimensioned settings as
    ``<setting>.<field>``, optional where the setting may be ``None``."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception as error:  # noqa: BLE001 - a lab's annotation that does not resolve becomes a refusal naming it
        return {}, [f"its annotations do not resolve at registration ({error!r}); define every name they use at "
                    f"module level"]
    found: dict[str, DimensionedSetting] = {}
    problems: list[str] = []
    for setting in dataclasses.fields(cls):
        hint: object = hints.get(setting.name)
        group = _group_of(hint)
        if group is not None:
            if issubclass(cls, SettingGroup):
                problems.append(f"setting {setting.name!r} holds the group {group.__name__}: a group holds no group")
                continue
            inner, inner_problems = dimensioned_settings(group)
            problems += [f"in its group {setting.name!r}: {problem}" for problem in inner_problems]
            optional = type(None) in _members(hint)
            for name, declared in inner.items():
                found[f"{setting.name}.{name}"] = dataclasses.replace(declared,
                                                                      optional=declared.optional or optional)
            continue
        if typing.get_origin(hint) is Annotated:
            held, *metadata = typing.get_args(hint)
            kinds = [m for m in metadata if isinstance(m, SettingKind)]
            if kinds:
                if len(kinds) > 1 or any(isinstance(m, Dimension) for m in metadata):
                    problems.append(f"setting {setting.name!r} is annotated {hint!r}: one marker, a dimension or "
                                    f"one of {Direction!r} and {Commanded!r}")
                elif kinds[0] is Commanded and Quantity not in _members(held):
                    problems.append(f"setting {setting.name!r} is marked {Commanded!r} and holds no Quantity")
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
    found += _flag_problems(cls)
    if kind is EntryKind.OBSERVATION:
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
    return found


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
