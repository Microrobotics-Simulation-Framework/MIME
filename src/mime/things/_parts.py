"""A body's parts, and the part resolver: the body's mass, inertia, buoyant volume, centre of buoyancy and magnetic
moment, derived from each part's solid, placement and material, each with its derivation.

The resolver is arithmetic on numbers the build hands it (each part's volume, centroid and inertia per unit mass in
the body's frame, its density and its magnetisation), so it reads no record and no geometry kind itself. The body's
frame has its origin at the body's centre of mass: parts whose centre of mass lies elsewhere are refused, not
shifted."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mime.units import VACUUM_PERMEABILITY, Magnitude

if TYPE_CHECKING:
    from mime.materials import Record
    from mime.registry import Solid, Tensor
    from mime.units import Quantity

type Vector = tuple[float, float, float]
type Problem = tuple[str, str, str]

#: The values the resolver derives, by name: what each body value is made of.
DERIVED = ("mass", "inertia", "buoyant_volume", "centre_of_buoyancy", "magnetic_moment", "extent")


@dataclass(frozen=True)
class Part:
    """A part of a body: its solid (a registered ``mime.registry.Solid``, in the part's own frame), its material,
    where it sits, and the part it lies inside, if any.

    ``at`` places a part of a body written Python-first: its centroid in the body's frame, whose origin is the body's
    centre of mass; such a part's frame is turned with the body's. A part of a body bound to an asset's prim is
    placed by the asset's transforms, so ``at`` is refused there. ``inside`` names the part this one lies within: its
    volume is part of that part's and its mass is counted in that part's density, so it adds to neither the body's
    mass nor its buoyant volume (a magnet embedded in a shell whose density is the body's mean)."""

    geometry: Solid
    material: Record | None = None
    at: Quantity | None = None
    inside: str | None = None


@dataclass(frozen=True)
class PartMass:
    """A part as the resolver takes it: its volume (m^3), its centroid and its inertia tensor per unit mass about
    the centroid (m^2), both in the body's frame, the part it lies inside, and its kind's radial reach (m) where the
    kind gives one."""

    name: str
    solid: str
    volume: float
    centre: Vector
    inertia_per_mass: Tensor
    inside: str | None = None
    reach: float | None = None


@dataclass(frozen=True)
class Derived:
    """A value derived from a body's parts, in SI, with one line per input, or the problem that keeps it from being
    derived."""

    value: Magnitude | None
    details: tuple[str, ...] = ()
    problems: tuple[Problem, ...] = field(default_factory=tuple[Problem, ...])


def _vec(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(c) + 0.0:.6g}" for c in values) + ")"


def _sum(vectors: Sequence[Vector]) -> Vector:
    return (math.fsum(v[0] for v in vectors), math.fsum(v[1] for v in vectors), math.fsum(v[2] for v in vectors))


def _scaled(vector: Vector, factor: float) -> Vector:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _shifted(inertia: Tensor, mass: float, offset: Vector) -> list[list[float]]:
    """``inertia`` (about a part's centroid) about the body's origin, by the parallel-axis theorem."""
    square = offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2
    return [[mass * inertia[i][j] + mass * ((square if i == j else 0.0) - offset[i] * offset[j]) for j in range(3)]
            for i in range(3)]


def _bulk(parts: Sequence[PartMass]) -> list[PartMass]:
    return [part for part in parts if part.inside is None]


def _inside_line(parts: Sequence[PartMass]) -> tuple[str, ...]:
    inner = [part for part in parts if part.inside is not None]
    return tuple(f"{part.name}: inside {part.inside}, its volume and mass counted in {part.inside}'s" for part in inner)


def derive(body: str, parts: Sequence[PartMass], density: Mapping[str, float],
           magnetisation: Mapping[str, Vector]) -> dict[str, Derived]:
    """Each of ``DERIVED`` for ``body`` from its ``parts``, given each part's ``density`` (kg/m^3, by part name)
    and each magnet part's ``magnetisation`` (A/m, in the body's frame). A value whose inputs are missing is not
    derived, and says which input is missing; the body's centre of mass off its origin, or an inertia whose
    principal axes are not the body frame's, is a problem of the values that depend on it."""
    bulk = _bulk(parts)
    found: dict[str, Derived] = {}
    volume = math.fsum(part.volume for part in bulk)
    volume_lines = tuple(f"{part.name}: {part.volume:.6g} m^3 ({part.solid}) at {_vec(part.centre)} m"
                         for part in bulk) + _inside_line(parts)
    if not bulk or volume <= 0.0:
        missing = (("parts.no-bulk", f"body {body!r} has no part outside another, so no volume of its own",
                    "give the body a part that is inside no other"),)
        return {name: Derived(None, problems=missing) for name in DERIVED}
    found["buoyant_volume"] = Derived(volume, volume_lines)
    found["centre_of_buoyancy"] = Derived(_scaled(_sum([_scaled(p.centre, p.volume) for p in bulk]), 1.0 / volume),
                                          volume_lines)
    lacking = [part.name for part in bulk if part.name not in density]
    if lacking:
        problem = ("parts.density", f"body {body!r}'s mass and inertia are derived from its parts, and "
                                    f"{', '.join(repr(n) for n in lacking)} has no density",
                   f"give {body}.parts.<part>.density: a record that holds it, or declare it")
        found["mass"] = found["inertia"] = Derived(None, problems=(problem,))
    else:
        masses = {part.name: density[part.name] * part.volume for part in bulk}
        mass = math.fsum(masses.values())
        mass_lines = tuple(f"{part.name}: {density[part.name]:.6g} kg/m^3 x {part.volume:.6g} m^3 = "
                           f"{masses[part.name]:.6g} kg" for part in bulk) + _inside_line(parts)
        moments = [_scaled(part.centre, masses[part.name]) for part in bulk]
        centre = _scaled(_sum(moments), 1.0 / mass)
        scale = math.fsum(abs(c) for moment in moments for c in moment) / mass
        off = math.sqrt(sum(c * c for c in centre))
        if off > 1e-12 * scale:
            problem = ("parts.centre-of-mass", f"body {body!r}'s parts put its centre of mass at {_vec(centre)} m in "
                                               f"its frame, whose origin is its centre of mass",
                       "place the parts about the body's centre of mass")
            found["mass"] = Derived(mass, mass_lines, (problem,))
            found["inertia"] = Derived(None, problems=(problem,))
        else:
            found["mass"] = Derived(mass, mass_lines)
            tensor = [[math.fsum(_shifted(part.inertia_per_mass, masses[part.name], part.centre)[i][j]
                                 for part in bulk) for j in range(3)] for i in range(3)]
            diagonal = (tensor[0][0], tensor[1][1], tensor[2][2])
            largest = max(abs(c) for c in diagonal)
            across = max(abs(tensor[i][j]) for i in range(3) for j in range(3) if i != j)
            inertia_lines = tuple(f"{part.name}: {part.solid} of {masses[part.name]:.6g} kg about its centroid "
                                  f"{_vec(part.centre)} m, by the parallel-axis theorem" for part in bulk)
            if across > 1e-12 * largest:
                found["inertia"] = Derived(None, problems=((
                    "parts.inertia-axes", f"body {body!r}'s parts give an inertia tensor whose principal axes are "
                                          f"not the body frame's (a product of inertia of {across:.3g} kg m^2 "
                                          f"against {largest:.3g})",
                    "place the parts symmetrically about the body frame's axes, or declare the inertia with its "
                    "reason"),))
            else:
                found["inertia"] = Derived(diagonal, inertia_lines)
    found["extent"] = _extent(body, bulk)
    magnets = [part for part in parts if part.name in magnetisation]
    if not magnets:
        found["magnetic_moment"] = Derived(None, problems=((
            "parts.no-magnet", f"body {body!r} has parts and no magnet among them, so no moment to derive",
            "give a part a magnet's role: magnets={\"<part>\": PermanentMagnet(...)}"),))
    else:
        moment = _sum([_scaled(magnetisation[part.name], part.volume) for part in magnets])
        found["magnetic_moment"] = Derived(moment, tuple(
            f"{part.name}: M {_vec(magnetisation[part.name])} A/m x {part.volume:.6g} m^3, M = B_r / mu_0 along its "
            f"direction (mu_0 = {VACUUM_PERMEABILITY!r} N A^-2), a rigid magnetisation" for part in magnets))
    return found


def _extent(body: str, bulk: Sequence[PartMass]) -> Derived:
    """The body's extent: its hull's radial reach about its axis, the hull being its one part outside every other,
    centred on the body's origin, of a kind that gives its reach."""
    action = ("give the body one part outside every other, of a kind that gives its radial reach (a sphere, a capsule "
              "or a de Jongh helix), or declare the extent with its reason where the hull's kind gives none")
    if len(bulk) != 1:
        return Derived(None, problems=(("parts.extent", f"body {body!r} has {len(bulk)} parts outside one another: its "
                                                        f"hull would be their union, which is not built", action),))
    part = bulk[0]
    if part.reach is None:
        return Derived(None, problems=(("parts.extent", f"body {body!r}'s hull, {part.name!r}, is a {part.solid}, "
                                                        f"which gives no radial reach", action),))
    off = math.sqrt(sum(c * c for c in part.centre))
    if off > 1e-12 * part.reach:
        return Derived(None, problems=(("parts.extent", f"body {body!r}'s hull, {part.name!r}, is centred "
                                                        f"{off:.6g} m off the body's origin, so its reach about the "
                                                        f"body's axis is not its own", action),))
    return Derived(part.reach, (f"{part.name}: the radial reach of its {part.solid} about its axis, "
                                f"{part.reach:.6g} m",))


def magnetisation(remanence: float, direction: Vector) -> Vector:
    """``B_r / mu_0`` along ``direction`` (a unit vector in the body's frame), in A/m: a permanent magnet's
    magnetisation from its material's remanence, with no recoil permeability and no demagnetisation."""
    return _scaled(direction, remanence / VACUUM_PERMEABILITY)


__all__ = ["DERIVED", "Derived", "Part", "PartMass", "derive", "magnetisation"]
