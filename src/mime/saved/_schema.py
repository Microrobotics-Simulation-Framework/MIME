"""The saved form's two files as closed TypedDicts, and the run-time check a loader makes of them.

A TypedDict does nothing at run time, so :func:`shape_problems` checks a value read from disk against the same
annotations a type checker reads: an unknown key, a missing key or a value of another type is a problem, named by its
path. The check is driven by the annotations themselves, and refuses a typing form it does not know, so the static
schema and the run-time check cannot drift apart without a failure.
"""


import types
import typing
from collections.abc import Sequence
from typing import Any, Literal, Never, TypeAliasType, cast

from typing_extensions import NotRequired, ReadOnly, Required, TypedDict, get_type_hints, is_typeddict

DOCUMENT_SCHEMA = "mime.document/1"
LOCK_SCHEMA = "mime.lock/1"

type SettingValue = None | bool | int | float | str | list[SettingValue] | QuantityDoc
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


class QuantityDoc(TypedDict, closed=True):
    """A quantity as SI value and SI unit symbol (``{value: 0.02, unit: s}``)."""

    value: ReadOnly[float | list[float]]
    unit: ReadOnly[str]


class EntryDoc(TypedDict, closed=True):
    """A registered entry by name and version, with the settings written differently from its defaults. An alias
    such as ``ToyDrag.Quadratic()`` saves as its family with the rung among the settings."""

    name: ReadOnly[str]
    version: ReadOnly[str]
    settings: NotRequired[dict[str, SettingValue]]


class BoundEntryDoc(TypedDict, closed=True):
    """An interaction or an observation: an entry with its id and the things or fields bound to its roles, by
    name."""

    id: ReadOnly[str]
    name: ReadOnly[str]
    version: ReadOnly[str]
    binds: dict[str, str]
    settings: NotRequired[dict[str, SettingValue]]


class DynamicsDoc(TypedDict, closed=True):
    """A thing's dynamics, one entry per aspect: a body's or a structure's motion, a medium's flow."""

    motion: NotRequired[EntryDoc]
    flow: NotRequired[EntryDoc]


class MagnetDoc(TypedDict, closed=True):
    """A part's role as a magnet: its model, and its direction in the part's frame when the experiment declares it
    rather than the part's prim."""

    model: ReadOnly[Literal["permanent_magnet"]]
    direction: NotRequired[list[float]]


class ThingDoc(TypedDict, closed=True):
    """A thing: its kind, its prim (a scene path, an experiment name like the thing's own) and dynamics; a body's
    initial state and the roles of its parts; a structure's geometry kind; the region a medium fills
    (``tube.lumen``); an apparatus's fields and the drive of each command."""

    kind: ReadOnly[Literal["structure", "medium", "body", "apparatus"]]
    prim: ReadOnly[str]
    dynamics: NotRequired[DynamicsDoc]
    initial: NotRequired[dict[str, SettingValue]]
    fields: NotRequired[dict[str, EntryDoc]]
    geometry: NotRequired[EntryDoc]
    fills: NotRequired[str]
    commands: NotRequired[dict[str, EntryDoc]]
    magnets: NotRequired[dict[str, MagnetDoc]]


class DeclaredDoc(TypedDict, closed=True):
    """A value declared in the experiment (``umr.mass``): in SI, with its unit and its reason."""

    value: ReadOnly[float | list[float]]
    unit: ReadOnly[str]
    reason: ReadOnly[str]


class RunDoc(TypedDict, closed=True):
    dt: QuantityDoc
    seed: int


class ReferenceDoc(TypedDict, closed=True):
    """An asset layer referenced into the scene: the prim it is referenced at, its file below the document's
    folder, and its pin."""

    at: ReadOnly[str]
    layer: ReadOnly[str]
    sha256: ReadOnly[str]


class SceneDoc(TypedDict, closed=True):
    """The scene: up, once, the unit its stage is written in, and the asset layers it references."""

    up: list[float]
    units: str
    references: list[ReferenceDoc]


class ExperimentDocument(TypedDict, closed=True):
    """What the experiment declares, written by registered name and version, settings and bindings, and nothing
    else: never code, never a copy of a record's values. ``declared`` holds the values declared in the experiment,
    each with its reason; ``gravity`` is its magnitude, along ``-up``. ``materials`` and ``controller`` are slots
    kept empty until those land."""

    schema: ReadOnly[Literal["mime.document/1"]]
    mime: ReadOnly[str]
    experiment: ReadOnly[str]
    run: RunDoc
    scene: SceneDoc
    gravity: QuantityDoc
    things: dict[str, ThingDoc]
    interactions: list[BoundEntryDoc]
    observations: list[BoundEntryDoc]
    declared: dict[str, DeclaredDoc]
    materials: dict[str, Never]
    controller: None


class EnvironmentRecord(TypedDict, closed=True):
    """What a build ran under. A load compares it and reports every difference; it never refuses on one."""

    mime: ReadOnly[str]
    maddening: ReadOnly[str]
    openmaterica: ReadOnly[str]
    usd_core: ReadOnly[str]
    jax: ReadOnly[str]
    jaxlib: ReadOnly[str]
    python: ReadOnly[str]
    platform: ReadOnly[str]
    device: ReadOnly[str]
    x64: ReadOnly[bool]
    prng: ReadOnly[str]


class EntryRecord(TypedDict, closed=True):
    """One entry the experiment uses, and the distribution it came from, so that a lab's entry pins its own
    package."""

    kind: ReadOnly[Literal["interaction", "observation", "dynamics", "field provider", "controller", "drive",
                           "geometry"]]
    name: ReadOnly[str]
    version: ReadOnly[str]
    distribution: ReadOnly[str]
    distribution_version: ReadOnly[str]


class HashRecord(TypedDict, closed=True):
    """``document``: the document in canonical JSON. ``input``: the document with the resolved record, the entries
    and the canonical order, which a load must reproduce exactly. ``derived`` and ``observations``: the graph built,
    physics and observations apart."""

    recipe: ReadOnly[str]
    document: ReadOnly[str]
    input: ReadOnly[str]
    derived: ReadOnly[str]
    observations: ReadOnly[str]


class ResolvedRun(TypedDict, closed=True):
    """The run settings in SI, the world frame with gravity's vector derived, and how a run executes."""

    experiment: ReadOnly[str]
    dt: ReadOnly[float]
    seed: ReadOnly[int]
    up: ReadOnly[list[float]]
    gravity: ReadOnly[list[float]]
    structure: ReadOnly[str]


class ResolvedEntry(TypedDict, closed=True):
    """One declared entry as the build resolved it: every setting, defaults included; a body's initial state in SI;
    its declared gradient status; the nodes it built and how many base steps each takes."""

    kind: ReadOnly[str]
    name: ReadOnly[str]
    version: ReadOnly[str]
    differentiability: ReadOnly[str]
    binds: ReadOnly[dict[str, str]]
    settings: ReadOnly[dict[str, SettingValue]]
    initial: ReadOnly[dict[str, SettingValue]]
    nodes: ReadOnly[list[str]]
    every: ReadOnly[int]


class ValueRecord(TypedDict, closed=True):
    """A value of a thing the entries read: in SI, where it came from (``declared in the experiment``) and why, and
    the ids that read it."""

    value: ReadOnly[float | list[float]]
    unit: ReadOnly[str]
    source: ReadOnly[str]
    reason: ReadOnly[str]
    read_by: ReadOnly[list[str]]


class GeometryRecord(TypedDict, closed=True):
    """A structure's geometry kind as the build resolved it: every setting, defaults included."""

    name: ReadOnly[str]
    version: ReadOnly[str]
    settings: ReadOnly[dict[str, SettingValue]]


class RegistrationRecord(TypedDict, closed=True):
    id: ReadOnly[str]
    kind: ReadOnly[str]
    subject: ReadOnly[str]


class SlotRecord(TypedDict, closed=True):
    name: ReadOnly[str]
    reader: ReadOnly[str]
    interfaces: ReadOnly[list[str]]


class GroupRecord(TypedDict, closed=True):
    nodes: ReadOnly[list[str]]
    tolerance: ReadOnly[float]
    max_iterations: ReadOnly[int]
    requested_by: ReadOnly[list[str]]


class MagnetRecord(TypedDict, closed=True):
    """A body's magnet as resolved: its part's prim, its model, its direction in the part's frame and in the
    body's, and where the direction came from."""

    part: ReadOnly[str]
    model: ReadOnly[str]
    in_part: ReadOnly[list[float]]
    in_body: ReadOnly[list[float]]
    source: ReadOnly[str]


class ResolvedScene(TypedDict, closed=True):
    """The scene as resolved: up, the stage's unit and metres per unit, every thing's prim, the things bound to a
    prim an asset brings in, and each body's magnets by their key."""

    up: ReadOnly[list[float]]
    units: ReadOnly[str]
    meters_per_unit: ReadOnly[float]
    prims: ReadOnly[dict[str, str]]
    bound: ReadOnly[list[str]]
    magnets: ReadOnly[dict[str, dict[str, MagnetRecord]]]


class ResolvedRecord(TypedDict, closed=True):
    """What the build derived from the document: what a reader needs to see what ran without that version's code,
    and what a load compares item by item."""

    vocabulary: ReadOnly[dict[str, str]]
    run: ReadOnly[ResolvedRun]
    things: ReadOnly[dict[str, str]]
    entries: ReadOnly[dict[str, ResolvedEntry]]
    order: ReadOnly[list[str]]
    registrations: ReadOnly[dict[str, list[RegistrationRecord]]]
    slots: ReadOnly[dict[str, list[SlotRecord]]]
    groups: ReadOnly[list[GroupRecord]]
    values: ReadOnly[dict[str, ValueRecord]]
    geometry: ReadOnly[dict[str, GeometryRecord]]
    fills: ReadOnly[dict[str, str]]
    scene: ReadOnly[ResolvedScene]


class DerivedRecord(TypedDict, closed=True):
    """The derived graph, saved with its hash: MADDENING's description of it, arrays by content hash."""

    physics: ReadOnly[dict[str, Json]]
    observations: ReadOnly[dict[str, Json]]


class ReferenceRecord(TypedDict, closed=True):
    """The golden outputs ``verify`` reruns against: every field of every node over the first ``steps`` steps, in
    ``file``, with the tolerance used across environments (within one environment they match bit for bit)."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]
    steps: ReadOnly[int]
    rtol: ReadOnly[float]
    atol: ReadOnly[float]


class StageRecord(TypedDict, closed=True):
    """The stage the build wrote: its file below the folder and its SHA-256, and every layer it references, by its
    file below the folder, with its SHA-256. The stage's identity."""

    file: ReadOnly[str]
    sha256: ReadOnly[str]
    layers: ReadOnly[dict[str, str]]


class Lock(TypedDict, closed=True):
    """What a build wrote beside the document: the environment, the entries and their distributions, the derived
    seeds, the hashes, the resolved record, the derived graph, the stage and the reference outputs."""

    schema: ReadOnly[Literal["mime.lock/1"]]
    environment: ReadOnly[EnvironmentRecord]
    entries: ReadOnly[list[EntryRecord]]
    seeds: ReadOnly[dict[str, dict[str, int]]]
    hashes: ReadOnly[HashRecord]
    resolved: ReadOnly[ResolvedRecord]
    derived: ReadOnly[DerivedRecord]
    stage: ReadOnly[StageRecord]
    reference: ReadOnly[ReferenceRecord]


_WRAPPERS = (ReadOnly, NotRequired, Required)


def _unwrap(annotation: Any) -> Any:
    while typing.get_origin(annotation) in _WRAPPERS:
        annotation = typing.get_args(annotation)[0]
    return annotation


def typed_dict_fields(cls: type) -> dict[str, Any]:
    """A TypedDict's keys and their value types, with ``ReadOnly`` and ``NotRequired`` taken off."""
    return {key: _unwrap(value) for key, value in get_type_hints(cls, include_extras=True).items()}


def _name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def shape_problems(value: object, annotation: Any, path: str = "$") -> list[str]:
    """Every way ``value`` differs from ``annotation``, each with its path; empty when it conforms."""
    annotation = _unwrap(annotation)
    if isinstance(annotation, TypeAliasType):
        return shape_problems(value, annotation.__value__, path)
    if annotation is None or annotation is type(None):
        return [] if value is None else [f"{path}: {value!r} is not null"]
    if annotation is Never:
        return [f"{path}: nothing may be written here yet (the slot is kept empty until it is built)"]
    if annotation is bool:
        return [] if isinstance(value, bool) else [f"{path}: {value!r} is not a boolean"]
    if annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        return [] if ok else [f"{path}: {value!r} is not an integer"]
    if annotation is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        return [] if ok else [f"{path}: {value!r} is not a number"]
    if annotation is str:
        return [] if isinstance(value, str) else [f"{path}: {value!r} is not a string"]
    origin = typing.get_origin(annotation)
    if origin is Literal:
        allowed = typing.get_args(annotation)
        return [] if any(type(value) is type(a) and value == a for a in allowed) else [
            f"{path}: {value!r} is not one of {', '.join(repr(a) for a in allowed)}"]
    if origin in (types.UnionType, typing.Union):
        members = typing.get_args(annotation)
        found = [shape_problems(value, member, path) for member in members]
        if any(not problems for problems in found):
            return []
        return [f"{path}: {value!r} is none of {' | '.join(_name(m) for m in members)}"]
    if origin is list:
        if not isinstance(value, list):
            return [f"{path}: {value!r} is not a list"]
        (item,) = typing.get_args(annotation)
        return [p for i, element in enumerate(cast(list[object], value)) for p in shape_problems(element, item,
                                                                                               f"{path}[{i}]")]
    if origin is dict:
        if not isinstance(value, dict):
            return [f"{path}: {value!r} is not a mapping"]
        key_type, item = typing.get_args(annotation)
        if key_type is not str:
            raise TypeError(f"the saved form's mappings have string keys, not {key_type!r}")
        problems: list[str] = []
        for key, element in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                problems.append(f"{path}: the key {key!r} is not a string")
            else:
                problems += shape_problems(element, item, f"{path}.{key}")
        return problems
    if isinstance(annotation, type) and is_typeddict(annotation):
        return _typed_dict_problems(value, annotation, path)
    raise TypeError(f"the saved form's check does not know the typing form {annotation!r}")


def _typed_dict_problems(value: object, cls: type, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: {value!r} is not a mapping"]
    held = cast(dict[object, object], value)
    fields = typed_dict_fields(cls)
    required: frozenset[str] = getattr(cls, "__required_keys__")
    problems = [f"{path}: unknown key {key!r} ({cls.__name__} holds only {', '.join(sorted(fields))})"
                for key in held if key not in fields]
    problems += [f"{path}: missing key {key!r}" for key in sorted(required) if key not in held]
    for key, annotation in fields.items():
        if key in held:
            problems += shape_problems(held[key], annotation, f"{path}.{key}")
    return problems


def reachable_typed_dicts(root: type) -> list[type]:
    """Every TypedDict ``root``'s annotations reach, ``root`` first: what the check and its test walk."""
    found: list[type] = []
    aliases: set[TypeAliasType] = set()

    def visit(annotation: Any) -> None:
        annotation = _unwrap(annotation)
        if isinstance(annotation, TypeAliasType):
            if annotation not in aliases:
                aliases.add(annotation)
                visit(annotation.__value__)
        elif isinstance(annotation, type) and is_typeddict(annotation):
            if annotation not in found:
                found.append(annotation)
                for item in typed_dict_fields(annotation).values():
                    visit(item)
        else:
            for argument in typing.get_args(annotation):
                if not isinstance(argument, (str, int, bool)) and argument is not None:
                    visit(argument)

    visit(root)
    return found


def conforms[T](value: object, schema: type[T]) -> tuple[T | None, Sequence[str]]:
    """``value`` as a ``schema`` when it conforms, and every problem otherwise."""
    problems = shape_problems(value, schema)
    return (cast(T, value) if not problems else None), problems


__all__ = [
    "DOCUMENT_SCHEMA", "LOCK_SCHEMA", "BoundEntryDoc", "DeclaredDoc", "DerivedRecord", "DynamicsDoc", "EntryDoc",
    "EntryRecord", "EnvironmentRecord", "ExperimentDocument", "GeometryRecord", "GroupRecord", "HashRecord", "Json",
    "Lock", "MagnetDoc", "MagnetRecord", "QuantityDoc", "ReferenceDoc", "ReferenceRecord", "RegistrationRecord",
    "ResolvedEntry", "ResolvedRecord", "ResolvedRun", "ResolvedScene", "RunDoc", "SceneDoc", "SettingValue",
    "SlotRecord", "StageRecord", "ThingDoc", "ValueRecord", "conforms", "reachable_typed_dicts", "shape_problems",
    "typed_dict_fields",
]
