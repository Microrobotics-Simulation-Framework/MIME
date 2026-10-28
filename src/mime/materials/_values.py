"""The values of things that laws and dynamics read (a body's mass and moment, a medium's density), and the values of
a body's parts that the part resolver reads (a part's density, a magnet's remanence): what each is, the record field
it is read from when a record provides it, and a value declared in the experiment in place of one."""

from __future__ import annotations

import math
from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from mime.things import ThingKind, did_you_mean
from mime.units import (
    DENSITY,
    DIMENSIONLESS,
    DYNAMIC_VISCOSITY,
    LENGTH,
    MAGNETIC_FLUX_DENSITY,
    MAGNETIC_MOMENT,
    MASS,
    MOMENT_OF_INERTIA,
    VOLUME,
    Dimension,
    Magnitude,
    Quantity,
    UnitError,
    si,
    unit_symbol,
)

#: How the build report and the saved form label a value that came from a declaration.
DECLARED_IN_THE_EXPERIMENT = "declared in the experiment"
#: How they label a value the part resolver derived.
DERIVED_FROM_PARTS = "derived from its parts"

#: A value whose direction comes from design data (a magnet's on its part's prim), by path, with the unit vector and
#: where it came from: the experiment then declares the value's magnitude alone.
type Directed = Mapping[str, tuple[tuple[float, float, float], str]]

#: The path segment between a body and its part's name: ``umr.parts.shell.density``.
PARTS = "parts"


def _no_tokens() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({})


@dataclass(frozen=True)
class RecordField:
    """Where a value is read from in an OpenMaterica record: the field (``bulk.density``), the unit the record stores
    it in, which is SI and so MIME's, and the token conditions the value's meaning fixes, each with the tokens it
    accepts (a medium's density for buoyancy is the ``apparent`` density, of the whole specimen)."""

    field: str
    unit: str
    require: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=_no_tokens)


@dataclass(frozen=True)
class ValueKind:
    """What a value of a thing is: its dimension, its shape (``()`` or ``(3,)``) and what it means, and the record
    field it is read from when the thing's material is a record (``None``: no record holds it)."""

    dimension: Dimension
    shape: tuple[int, ...]
    meaning: str
    record: RecordField | None = None


_APPARENT: Mapping[str, tuple[str, ...]] = MappingProxyType({"density_kind": ("apparent",)})

#: Every value a thing has, by the kind of thing. A body's are derived from its parts when it has parts; a medium's
#: are read from its record when it has one; either may be declared in the experiment.
VALUE_KINDS: dict[ThingKind, dict[str, ValueKind]] = {
    ThingKind.BODY: {
        "mass": ValueKind(MASS, (), "the body's mass"),
        "inertia": ValueKind(MOMENT_OF_INERTIA, (3,), "the principal moments of inertia about the centre of mass, "
                                                      "along the body frame's axes"),
        "buoyant_volume": ValueKind(VOLUME, (), "the volume of fluid the body displaces"),
        "centre_of_buoyancy": ValueKind(LENGTH, (3,), "the centre of the displaced volume, from the centre of mass, "
                                                       "in the body frame"),
        "magnetic_moment": ValueKind(MAGNETIC_MOMENT, (3,), "the body's permanent magnetic moment, in the body "
                                                            "frame"),
        "magnet_centre": ValueKind(LENGTH, (3,), "the centroid of the body's magnet, from the centre of mass, in the "
                                                 "body frame: where a field's gradient pulls on its moment"),
        "extent": ValueKind(LENGTH, (), "the radial reach of the body's hull about its axis: the radius of the "
                                        "cylinder about the axis through the centre of mass that holds the body, "
                                        "derived from a hull whose kind gives it"),
        "radius": ValueKind(LENGTH, (), "the body's reference radius: a sphere's radius; for a screw-shaped body, the "
                                        "radius of the cylinder its cross-section is modulated about"),
        "length": ValueKind(LENGTH, (), "the body's length along its axis"),
        "helix_wavenumber": ValueKind(DIMENSIONLESS, (), "the normalised wavenumber nu of a screw-shaped body: its "
                                                         "cross-section turns by nu / R radians per metre along its "
                                                         "axis, R its reference radius (de Jongh et al. 2025, Eq. 1)"),
    },
    ThingKind.APPARATUS: {
        "magnetic_moment": ValueKind(MAGNETIC_MOMENT, (3,), "the permanent moment of the apparatus's magnet part, in "
                                                            "that part's frame"),
    },
    ThingKind.MEDIUM: {
        "density": ValueKind(DENSITY, (), "the medium's mass density", RecordField("bulk.density", "kg/m3", _APPARENT)),
        "viscosity": ValueKind(DYNAMIC_VISCOSITY, (), "the medium's dynamic viscosity",
                               RecordField("fluid.dynamic_viscosity", "Pa.s")),
    },
}

#: Every value of a body's part, read from the part's material or declared in the experiment
#: (``umr.parts.shell.density``): what the part resolver derives the body's values from.
PART_VALUE_KINDS: dict[str, ValueKind] = {
    "density": ValueKind(DENSITY, (), "the part's mass density", RecordField("bulk.density", "kg/m3", _APPARENT)),
    "remanence": ValueKind(MAGNETIC_FLUX_DENSITY, (), "the remanence B_r of a magnet part's material",
                           RecordField("magnetic.remanence", "T")),
}


@dataclass(frozen=True)
class ValuePath:
    """A value's path, parsed: ``umr.mass`` is the thing ``umr``'s value ``mass``; ``umr.parts.shell.density`` the
    part ``shell`` of body ``umr``, its value ``density``."""

    thing: str
    name: str
    part: str | None = None

    @classmethod
    def parse(cls, path: str) -> ValuePath:
        pieces = path.split(".")
        if len(pieces) == 4 and pieces[1] == PARTS:
            return cls(pieces[0], pieces[3], pieces[2])
        thing, _, name = path.partition(".")
        return cls(thing, name)


@dataclass(frozen=True)
class Declared:
    """A value declared in the experiment: ``path`` is ``<thing>.<value>`` (``umr.mass``) or
    ``<body>.parts.<part>.<value>`` (``umr.parts.shell.density``)."""

    path: str
    value: Quantity
    reason: str


@dataclass(frozen=True)
class ResolvedValue:
    """A value as an entry receives it: in SI, with where it came from and why, for the build report and the saved
    form, and one line per input it was read or derived from."""

    path: str
    value: Magnitude
    unit: str
    source: str
    reason: str
    details: tuple[str, ...] = ()


def value_kind(kind: ThingKind, name: str) -> ValueKind | None:
    return VALUE_KINDS.get(kind, {}).get(name)


def kind_of_path(path: ValuePath, things: Mapping[str, ThingKind]) -> ValueKind | None:
    if path.part is not None:
        return PART_VALUE_KINDS.get(path.name) if things.get(path.thing) is ThingKind.BODY else None
    held = things.get(path.thing)
    return value_kind(held, path.name) if held is not None else None


def _si(declared: Declared, kind: ValueKind) -> tuple[Magnitude | None, str]:
    try:
        value = si(declared.value, kind.dimension, what=f"the declared value {declared.path}")
    except (UnitError, TypeError) as error:
        return None, str(error)
    components = value if isinstance(value, tuple) else (value,)
    shape = (len(value),) if isinstance(value, tuple) else ()
    if shape != kind.shape:
        expected = "one number" if not kind.shape else f"{kind.shape[0]} components"
        return None, f"the declared value {declared.path} is {kind.meaning}: {expected}, not {declared.value}"
    if not all(math.isfinite(c) for c in components):
        return None, f"the declared value {declared.path} is not finite: {declared.value}"
    return value, ""


def _magnitude(declared: Declared, kind: ValueKind) -> tuple[float | None, str]:
    try:
        value = si(declared.value, kind.dimension, what=f"the declared value {declared.path}")
    except (UnitError, TypeError) as error:
        return None, str(error)
    if isinstance(value, tuple) or not (math.isfinite(value) and value >= 0.0):
        return None, f"the declared value {declared.path} is not a finite magnitude >= 0: {declared.value}"
    return value, ""


def declaration_problems(declared: Declared, things: Mapping[str, ThingKind], directed: Directed | None = None,
                         parts: Mapping[str, frozenset[str]] | None = None) -> list[tuple[str, str, str]]:
    """Every way ``declared`` is not a value of a declared thing or of a declared body's part (``parts``: each
    body's part names), as ``(code, message, action)``. A value in ``directed`` takes its direction from design
    data, so it is declared as a magnitude, and a vector is its direction declared twice."""
    path = ValuePath.parse(declared.path)
    reason = cast(object, declared.reason)  # a script is not type-checked
    if not isinstance(reason, str) or not reason.strip():
        return [("value.reason", f"the declared value {declared.path} gives no reason",
                 f"declare it with its reason: declare({declared.path!r}, ..., reason=\"where the number comes "
                 f"from\")")]
    if path.thing not in things:
        return [("value.unknown-thing", f"the declared value {declared.path} names no thing",
                 did_you_mean(path.thing, things))]
    if path.part is not None:
        held = (parts or {}).get(path.thing, frozenset[str]())
        if things[path.thing] is not ThingKind.BODY or path.part not in held:
            return [("value.unknown-part", f"the declared value {declared.path} names no part of a body",
                     did_you_mean(path.part, held) if held else f"{path.thing} declares no parts")]
        kinds = PART_VALUE_KINDS
    else:
        kinds = VALUE_KINDS.get(things[path.thing], {})
    if path.name not in kinds:
        owner = "a part" if path.part is not None else f"a {things[path.thing].value}"
        return [("value.unknown", f"{owner} has no value {path.name!r}", did_you_mean(path.name, kinds))]
    kind = kinds[path.name]
    if directed is not None and declared.path in directed:
        source = directed[declared.path][1]
        if isinstance(declared.value.value, tuple):
            return [("magnet.direction-twice", f"the declared value {declared.path} is a vector, and its direction is "
                                               f"{source}: a direction is declared once",
                     f"declare its magnitude alone: declare({declared.path!r}, <a quantity of "
                     f"{kind.dimension}>, reason=...)")]
        _, problem = _magnitude(declared, kind)
        return [("value.units", problem, f"declare {declared.path}'s magnitude in units of {kind.dimension}")
                ] if problem else []
    _, problem = _si(declared, kind)
    if problem:
        return [("value.units", problem, f"declare {declared.path} as {kind.meaning}, in units of {kind.dimension}")]
    return []


def resolve_declared(held: Declared, kind: ValueKind, directed: Directed | None = None) -> ResolvedValue | None:
    """The declared value ``held`` in SI, labelled "declared in the experiment" with its reason; a value in
    ``directed`` is its declared magnitude along the given direction. ``None`` when it does not convert, which
    :func:`declaration_problems` refuses."""
    if directed is not None and held.path in directed:
        magnitude, problem = _magnitude(held, kind)
        if magnitude is None or problem:
            return None
        direction, source = directed[held.path]
        return ResolvedValue(held.path, tuple(magnitude * c for c in direction), unit_symbol(kind.dimension),
                             f"{DECLARED_IN_THE_EXPERIMENT} (its magnitude), along {source}", held.reason)
    value, problem = _si(held, kind)
    if value is None or problem:
        return None
    return ResolvedValue(held.path, value, unit_symbol(kind.dimension), DECLARED_IN_THE_EXPERIMENT, held.reason)


def resolve(path: str, kind: ThingKind, declared: Mapping[str, Declared],
            directed: Directed | None = None) -> ResolvedValue | None:
    """The value at ``path`` (``umr.mass``) of a thing of ``kind`` as the experiment declares it, or ``None`` when
    it declares none."""
    held = declared.get(path)
    value_of = value_kind(kind, path.partition(".")[2])
    if held is None or value_of is None:
        return None
    return resolve_declared(held, value_of, directed)
