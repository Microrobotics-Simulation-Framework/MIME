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
DERIVED = ("mass", "inertia", "buoyant_volume", "centre_of_buoyancy", "magnetic_moment", "extent", "magnet_centre")


@dataclass(frozen=True)
class Part:
    """A part of a body or an apparatus: its solid (a registered ``mime.registry.Solid``, in the part's own frame),
    its material, where it sits, and the part it lies inside, if any.

    ``at`` places a part of a body written Python-first: its centroid in the body's frame, whose origin is the body's
    centre of mass; such a part's frame is turned with the body's. A part of a body bound to an asset's prim is
    placed by the asset's transforms, so ``at`` is refused there. ``inside`` names the part this one lies within: its
    volume is part of that part's and its mass is counted in that part's density, so it adds to neither the body's
    mass nor its buoyant volume (a magnet embedded in a shell whose density is the body's mean).

    ``mount`` fixes a part of an apparatus to a frame of a segment of its chain, ``<segment>.<frame>``
    (``"motor.rotor"``), and ``at`` places its centroid in that frame, which its own frame is not turned from; with
    no ``mount`` it is fixed in the world, ``at`` its centroid there. An apparatus's part has a mass where its material
    gives it a density, or where the experiment declares its density or its mass
    (``<apparatus>.parts.<part>.density``, ``.mass``), a solid of uniform density; a dynamic segment carries that mass,
    and a part with none changes nothing. A body's part takes no ``mount``."""

    geometry: Solid
    material: Record | None = None
    at: Quantity | None = None
    inside: str | None = None
    mount: str | None = None


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
           magnetisation: Mapping[str, Vector], magnet_parts: Sequence[str] = ()) -> dict[str, Derived]:
    """Each of ``DERIVED`` for ``body`` from its ``parts``, given each part's ``density`` (kg/m^3, by part name)
    and each magnet part's ``magnetisation`` (A/m, in the body's frame); ``magnet_parts`` names the magnet parts, whose
    centroid is ``magnet_centre`` when there is one. A value whose inputs are missing is not derived, and says which
    input is missing; the body's centre of mass off its origin, or an inertia whose principal axes are not the body
    frame's, is a problem of the values that depend on it."""
    bulk = _bulk(parts)
    found: dict[str, Derived] = {}
    volume = math.fsum(part.volume for part in bulk)
    volume_lines = tuple(f"{part.name}: {part.volume:.6g} m^3 ({part.solid}) at {_vec(part.centre)} m"
                         for part in bulk) + _inside_line(parts)
    if not bulk or volume <= 0.0:
        missing = (("parts.no-bulk", f"body {body!r} has no part outside another, so no volume of its own",
                    "give the body a part that is inside no other: parts={\"shell\": Part(...)}"),)
        return {name: Derived(None, problems=missing) for name in DERIVED}
    found["buoyant_volume"] = Derived(volume, volume_lines)
    found["centre_of_buoyancy"] = Derived(_scaled(_sum([_scaled(p.centre, p.volume) for p in bulk]), 1.0 / volume),
                                          volume_lines)
    lacking = [part.name for part in bulk if part.name not in density]
    if lacking:
        problem = ("parts.density", f"body {body!r}'s mass and inertia are derived from its parts, and "
                                    f"{', '.join(repr(n) for n in lacking)} has no density",
                   f"give {body}.parts.<part>.density: a record that holds it, Part(..., material=Record(...)), or "
                   "declare it, exp.declare(...)")
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
                       "place the parts about the body's centre of mass: Part(..., at=...) for each, balanced about it")
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
                found["inertia"] = Derived(None, problems=(_inertia_axes_problem(body, tensor),))
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
    found["magnet_centre"] = _magnet_centre(body, [part for part in parts if part.name in magnet_parts])
    return found


#: The product of inertia, as a fraction of the largest diagonal moment, above which the refusal offers placing the
#: parts symmetrically before dropping the products: a convention for the order of the advice, not a threshold of
#: accuracy (the effect of dropping products on a run is not measured here).
LARGE_PRODUCT = 0.01


def _inertia_axes_problem(body: str, tensor: Sequence[Sequence[float]]) -> Problem:
    """The refusal of an inertia tensor whose principal axes are not the body frame's: the tensor the parts give,
    in the body's frame with its units, its principal moments and axes, how far the axes lie from the body's, and the
    declaration that states the diagonal with its products dropped, written out to paste. The body's ``inertia`` is
    the three moments about the body's axes, so a tensor with products can only be declared as its diagonal. The
    action is ordered by the products' size: declaring the diagonal first where they are small beside the moments, and
    placing the parts symmetrically first (with how far the diagonal is from the tensor) where they are not."""
    import numpy as np

    held = np.array(tensor, dtype=np.float64)
    diagonal = (float(held[0, 0]), float(held[1, 1]), float(held[2, 2]))
    largest = max(abs(c) for c in diagonal)
    across = float(max(abs(held[i, j]) for i in range(3) for j in range(3) if i != j))
    moments, axes = np.linalg.eigh(held)
    turn = float(np.degrees(max(np.arccos(min(1.0, float(np.abs(axes[:, k]).max()))) for k in range(3))))
    shift = float(np.abs(moments - np.sort(np.array(diagonal))).max()) / largest
    rows = "\n".join("      [" + "  ".join(f"{c:+.6e}" for c in row) + " ]" for row in held)
    principal = "\n".join(f"      {m:.6e} kg m^2 about {_vec(tuple(float(c) for c in axes[:, k]))}"
                          for k, m in enumerate(moments))
    reason = (f"the parts' inertia tensor about the body's axes, its products of inertia (the largest {across:.3g} kg "
              f"m^2, {100.0 * across / largest:.2g}% of the largest moment) dropped: a modelling choice, not a "
              f"measurement")
    message = (f"body {body!r}'s parts give an inertia tensor whose principal axes are not the body frame's: a "
               f"product of inertia of {across:.3g} kg m^2 against {largest:.3g} kg m^2 on the diagonal\n"
               f"    the tensor about the body's centre of mass, in the body's frame (kg m^2):\n{rows}\n"
               f"    its principal moments, with their axes in the body's frame:\n{principal}\n"
               f"    its axes lie at most {turn:.3g} degrees from the body's (axes of moments that nearly agree turn "
               f"easily), and its principal moments differ from the diagonal's by at most {100.0 * shift:.2g}% of "
               f"the largest")
    if across > LARGE_PRODUCT * largest:
        action = (f"place the parts symmetrically about the body frame's axes, Part(..., at=...), since the products are "
                  f"{100.0 * across / largest:.2g}% of the largest moment, above the {100.0 * LARGE_PRODUCT:.0f}% past "
                  f"which dropping them is not offered first (the diagonal's moments lie {100.0 * shift:.2g}% of the "
                  f"largest from the tensor's principal moments, and the tensor's axes {turn:.3g} degrees from the "
                  f"body's); or, to use the diagonal all the same, declare the moments about the body's axes with the "
                  f"products dropped, exp.declare(\"{body}.inertia\", kg_m2(({diagonal[0]!r}, {diagonal[1]!r}, "
                  f"{diagonal[2]!r})), reason=\"{reason}\")")
    else:
        action = (f"declare the moments about the body's axes with the products dropped, "
                  f"exp.declare(\"{body}.inertia\", kg_m2(({diagonal[0]!r}, {diagonal[1]!r}, {diagonal[2]!r})), "
                  f"reason=\"{reason}\"), or place the parts symmetrically about the body frame's axes, "
                  f"Part(..., at=...)")
    return ("parts.inertia-axes", message, action)


def _extent(body: str, bulk: Sequence[PartMass]) -> Derived:
    """The body's extent: its hull's radial reach about its axis, the hull being its one part outside every other,
    centred on the body's origin, of a kind that gives its reach."""
    action = ("give the body one part outside every other, of a kind that gives its radial reach (a sphere, a capsule "
              "or a screw), or declare the extent with its reason where the hull's kind gives none: "
              "exp.declare(\"<body>.extent\", mm(...), reason=\"...\")")
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


def _magnet_centre(body: str, magnets: Sequence[PartMass]) -> Derived:
    if len(magnets) == 1:
        (magnet,) = magnets
        return Derived(magnet.centre, (f"{magnet.name}: its centroid, {_vec(magnet.centre)} m in the body's frame",))
    if not magnets:
        return Derived(None, problems=((
            "parts.no-magnet", f"body {body!r} has parts and no magnet among them, so no magnet's centroid",
            "give a part a magnet's role: magnets={\"<part>\": PermanentMagnet(...)}"),))
    return Derived(None, problems=((
        "parts.several-magnets", f"body {body!r} has {len(magnets)} magnets ({', '.join(m.name for m in magnets)}): a "
                                 f"field's gradient pulls on each where it sits, which one centroid does not hold",
        "keep one magnet in the body, or declare its magnet_centre with its reason, the lever the force acts at: "
        "exp.declare(\"<body>.magnet_centre\", m((...)), reason=\"...\")"),))


def magnetisation(remanence: float, direction: Vector) -> Vector:
    """``B_r / mu_0`` along ``direction`` (a unit vector in the body's frame), in A/m: a permanent magnet's
    magnetisation from its material's remanence, with no recoil permeability and no demagnetisation."""
    return _scaled(direction, remanence / VACUUM_PERMEABILITY)


__all__ = ["DERIVED", "Derived", "Part", "PartMass", "derive", "magnetisation"]
