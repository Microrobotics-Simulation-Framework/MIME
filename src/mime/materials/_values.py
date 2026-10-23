"""The values of things that laws and dynamics read: a body's mass and moment, a medium's density. The one place that
reads them. Today the only source is a value declared in the experiment with its reason; OpenMaterica's records (a
medium's density) and the part resolver (a body's mass from its parts and their materials) replace most of those
declarations, here, without the entries that read the values changing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from mime.things import ThingKind, did_you_mean
from mime.units import (
    DENSITY,
    DIMENSIONLESS,
    DYNAMIC_VISCOSITY,
    LENGTH,
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

#: A value whose direction comes from design data (a magnet's on its part's prim), by path, with the unit vector and
#: where it came from: the experiment then declares the value's magnitude alone.
type Directed = Mapping[str, tuple[tuple[float, float, float], str]]


@dataclass(frozen=True)
class ValueKind:
    """What a value of a thing is: its dimension, its shape (``()`` or ``(3,)``) and what it means."""

    dimension: Dimension
    shape: tuple[int, ...]
    meaning: str


#: Every value a thing has, by the kind of thing. A body's values are derived from its parts in a later phase; a
#: medium's come from its record.
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
        "extent": ValueKind(LENGTH, (), "the radius of a sphere about the centre of mass that holds the body"),
        "radius": ValueKind(LENGTH, (), "the body's reference radius: a sphere's radius; for a screw-shaped body, the "
                                        "radius of the cylinder its cross-section is modulated about"),
        "length": ValueKind(LENGTH, (), "the body's length along its axis"),
        "helix_wavenumber": ValueKind(DIMENSIONLESS, (), "the normalised wavenumber nu of a screw-shaped body: its "
                                                         "cross-section turns by nu / R radians per metre along its "
                                                         "axis, R its reference radius (de Jongh et al. 2025, Eq. 1)"),
    },
    ThingKind.MEDIUM: {
        "density": ValueKind(DENSITY, (), "the medium's mass density"),
        "viscosity": ValueKind(DYNAMIC_VISCOSITY, (), "the medium's dynamic viscosity"),
    },
}


@dataclass(frozen=True)
class Declared:
    """A value declared in the experiment: ``path`` is ``<thing>.<value>`` (``umr.mass``)."""

    path: str
    value: Quantity
    reason: str


@dataclass(frozen=True)
class ResolvedValue:
    """A value as an entry receives it: in SI, with where it came from, for the build report and the saved form."""

    path: str
    value: Magnitude
    unit: str
    source: str
    reason: str


def value_kind(kind: ThingKind, name: str) -> ValueKind | None:
    return VALUE_KINDS.get(kind, {}).get(name)


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


def declaration_problems(declared: Declared, things: Mapping[str, ThingKind],
                         directed: Directed | None = None) -> list[tuple[str, str, str]]:
    """Every way ``declared`` is not a value of a declared thing, as ``(code, message, action)``. A value in
    ``directed`` takes its direction from design data, so it is declared as a magnitude, and a vector is its
    direction declared twice."""
    thing, _, name = declared.path.partition(".")
    reason = cast(object, declared.reason)  # a script is not type-checked
    if not isinstance(reason, str) or not reason.strip():
        return [("value.reason", f"the declared value {declared.path} gives no reason",
                 f"declare it with its reason: declare({declared.path!r}, ..., reason=\"where the number comes "
                 f"from\")")]
    if thing not in things:
        return [("value.unknown-thing", f"the declared value {declared.path} names no thing",
                 did_you_mean(thing, things))]
    kinds = VALUE_KINDS.get(things[thing], {})
    if name not in kinds:
        return [("value.unknown", f"a {things[thing].value} has no value {name!r}",
                 did_you_mean(name, kinds))]
    if directed is not None and declared.path in directed:
        source = directed[declared.path][1]
        if isinstance(declared.value.value, tuple):
            return [("magnet.direction-twice", f"the declared value {declared.path} is a vector, and its direction is "
                                               f"{source}: a direction is declared once",
                     f"declare its magnitude alone: declare({declared.path!r}, <a quantity of "
                     f"{kinds[name].dimension}>, reason=...)")]
        _, problem = _magnitude(declared, kinds[name])
        return [("value.units", problem, f"declare {declared.path}'s magnitude in units of {kinds[name].dimension}")
                ] if problem else []
    _, problem = _si(declared, kinds[name])
    if problem:
        return [("value.units", problem, f"declare {declared.path} as {kinds[name].meaning}, in units of "
                                        f"{kinds[name].dimension}")]
    return []


def resolve(path: str, kind: ThingKind, declared: Mapping[str, Declared],
            directed: Directed | None = None) -> ResolvedValue | None:
    """The value at ``path`` (``umr.mass``) of a thing of ``kind``, or ``None`` when nothing provides it. Only a
    declaration provides one today; a value in ``directed`` is its declared magnitude along the given direction."""
    held = declared.get(path)
    value_of = value_kind(kind, path.partition(".")[2])
    if held is None or value_of is None:
        return None
    if directed is not None and path in directed:
        magnitude, problem = _magnitude(held, value_of)
        if magnitude is None or problem:
            return None
        direction, source = directed[path]
        return ResolvedValue(path, tuple(magnitude * c for c in direction), unit_symbol(value_of.dimension),
                             f"{DECLARED_IN_THE_EXPERIMENT} (its magnitude), along {source}", held.reason)
    value, problem = _si(held, value_of)
    if value is None or problem:
        return None
    return ResolvedValue(path, value, unit_symbol(value_of.dimension), DECLARED_IN_THE_EXPERIMENT, held.reason)
