"""The screw: the screw-shaped body of de Jongh et al. (2025), Eq. 1, as a geometry kind."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

import numpy as np

from mime.core import SCREW, jnp, switch
from mime.gradients import GradientStatus
from mime.registry import (
    Drawing,
    GradientRecord,
    NoPrimitive,
    RegionPrimitive,
    Samples,
    Solid,
    Tensor,
    register_geometry,
)
from mime.scene._caps import any_closer, cap_rings, ring_floor
from mime.scene._kinds import NO_PRIMITIVE_ADVICE, banded_faces, closed_drawing
from mime.units import DIMENSIONLESS, Length, Magnitude, Quantity

#: The starts, the modulation and the wavenumber the kind accepts: those its screw's root find was checked over
#: against a brute-force minimum (``tests/scene/test_screw.py``). One start puts the centroid off the axis, and a
#: modulation of zero makes every point of the circle nearest the axis.
STARTS = (2, 3, 4)
LARGEST_MODULATION = 0.5
LARGEST_WAVENUMBER = 4.0
#: Points round a turn for the lateral patch's area, a periodic integral the trapezoid rule takes to rounding.
AREA_POINTS = 4096
#: No two samples of a cap lie closer than this many spacings. The polar grid of version 0.1.0 put its innermost
#: ring's samples far closer, which made a BEM's matrix singular in float64 once a wall's correction was added
#: (MIME-ANO-119). A declaration, above the closest pair of the lateral patch at FL-9's working spacing, on which a
#: BEM's derivative agrees with finite differences, and above the pairs whose removal restored the closed surface's
#: (the anomaly's record has both).
CAP_SEPARATION = 0.3


def _held(settings: Mapping[str, Magnitude], name: str) -> float:
    value = settings[name]
    if isinstance(value, tuple):
        raise TypeError(f"setting {name!r} is one number, not {value!r}")
    return value


def _turned(angle: np.ndarray, turn: tuple[Any, Any]) -> tuple[Any, Any]:
    """The cosine and sine of the section's ``angle`` turned by the screw's ``turn`` (its cosine and sine): the section
    turned whole, so that its rounding is the section's, not the turn's size."""
    cos, sin = jnp.asarray(np.cos(angle)), jnp.asarray(np.sin(angle))
    return cos * turn[0] - sin * turn[1], sin * turn[0] + cos * turn[1]


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Screw(Solid):
    """A screw-shaped body (after de Jongh et al. 2025, Eq. 1): a cylinder of ``radius`` ``R`` and ``length`` ``L``
    along its frame's x, its cross-section ``rho(theta) = R (1 + modulation sin(starts theta))`` turned along the axis
    by ``wavenumber / R`` radians per metre (``nu``, de Jongh's normalised wavenumber; its sign the handedness), closed
    by flat caps. Its frame's origin is its centroid, the middle of its axis; the 0.2 line's body ran from 0 to ``L``,
    so this is that body moved half its length along the axis, the cross-section's angle still measured from the
    frame's y at the end ``x = -L / 2`` (MIME-ANO-112's lesson: a reference point is stated, never assumed).

    Its patches are the ``lateral`` screw surface, the boundary of its region ``screw`` (the unbounded screw,
    evaluated by the shape ``screw``), and the two flat ``caps``, the boundary of its region ``span`` (the
    slab between the end planes, the shape ``slab``); the body is the two regions' intersection. Its tangent is the
    thread's direction.

    Its volume is ``pi R^2 L (1 + modulation^2 / 2)``, each cross-section's area. Its inertia about the centroid, per
    unit volume of a uniform density, is the cross-section's second moments integrated along the axis: with ``P0 =
    pi R^4 (2 + 6 eps^2 + 3 eps^4 / 4)`` the integral of ``rho^4`` round a turn, ``L P0 / 4`` about the axis; across
    it, ``A L^3 / 12`` plus ``L P0 / 8``, and for two starts a twist term: ``rho^4`` then has the component ``pi R^4 (4
    eps + 3 eps^3) sin(2 theta)``, so the second moments across the axis differ by, and couple through, its integral
    along the turning section.

    Its scope is this one. The thread's profile is sinusoidal (a modulated cylinder: the radius is modulated
    by ``modulation sin(starts theta)``), and its ends are flat caps (the screw region intersected with the span between
    two planes). De Jongh et al.'s design (the FL-9) is one setting of ``radius``, ``length``, ``wavenumber``,
    ``modulation`` and ``starts``; the paper is cited here and in ``references``."""

    name = "screw"
    version = "0.2.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord()
    references = ("deJongh2025NonlinDyn",)
    builds = {"": ()}
    regions = ("screw", "span")
    region_shapes = {"screw": "screw", "span": "slab"}
    patches = {"lateral": "screw", "caps": "span"}
    radius: Annotated[Quantity, Length]
    length: Annotated[Quantity, Length]
    wavenumber: Annotated[Quantity, DIMENSIONLESS]
    modulation: Annotated[Quantity, DIMENSIONLESS]
    starts: int = 2

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        for name in ("radius", "length"):
            value = settings[name]
            if not (isinstance(value, float) and math.isfinite(value) and value > 0.0):
                found.append(f"{name} is {value!r} m, not a length above zero")
        nu, epsilon = settings["wavenumber"], settings["modulation"]
        if not (isinstance(nu, float) and abs(nu) <= LARGEST_WAVENUMBER):
            found.append(f"wavenumber is {nu!r}, not a number within +-{LARGEST_WAVENUMBER:g}, the range its screw's "
                         f"root find was checked over")
        if not (isinstance(epsilon, float) and 0.0 < epsilon <= LARGEST_MODULATION):
            found.append(f"modulation is {epsilon!r}, not above 0 and at most {LARGEST_MODULATION:g}: at 0 every point "
                         f"of the circle is equally near the axis, so the screw's nearest point has no derivative "
                         f"there, and above {LARGEST_MODULATION:g} lies outside the range its root find was checked "
                         f"over")
        starts: object = self.starts
        if isinstance(starts, bool) or starts not in STARTS:
            found.append(f"starts is {starts!r}, not one of {STARTS}: one start puts the centroid off the axis, and "
                         f"more than four lie outside the range its screw's root find was checked over")
        return found

    def _section(self, settings: Mapping[str, Magnitude]) -> tuple[float, float, float, float]:
        return _held(settings, "radius"), _held(settings, "length"), _held(settings, "wavenumber"), \
            _held(settings, "modulation")

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        radius, length, _, epsilon = self._section(settings)
        return math.pi * radius * radius * length * (1.0 + epsilon * epsilon / 2.0)

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        radius, length, nu, epsilon = self._section(settings)
        area = math.pi * radius * radius * (1.0 + epsilon * epsilon / 2.0)
        fourth = math.pi * radius ** 4 * (2.0 + 6.0 * epsilon ** 2 + 0.75 * epsilon ** 4)
        twist = math.pi * radius ** 4 * (4.0 * epsilon + 3.0 * epsilon ** 3) if self.starts == 2 else 0.0
        k = nu / radius
        sine = length if k == 0.0 else math.sin(2.0 * k * length) / (2.0 * k)
        versine = 0.0 if k == 0.0 else (1.0 - math.cos(2.0 * k * length)) / (2.0 * k)
        along_y = length * fourth / 8.0 - twist * versine / 8.0
        along_z = length * fourth / 8.0 + twist * versine / 8.0
        product = twist * sine / 8.0
        axial = area * length ** 3 / 12.0
        volume = area * length
        return ((length * fourth / 4.0 / volume, 0.0, 0.0),
                (0.0, (axial + along_z) / volume, -product / volume),
                (0.0, -product / volume, (axial + along_y) / volume))

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        radius, length, nu, epsilon = self._section(settings)
        if region == "screw":
            return {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "radius": radius,
                    "length": length, "wavenumber": nu, "modulation": epsilon, "starts": self.starts}
        if region == "span":
            return {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "half_thickness": length / 2.0}
        raise KeyError(region)

    @override
    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        radius, length, _, epsilon = self._section(settings)
        around = max(4 * self.starts + 1, math.ceil(2.0 * math.pi * radius * (1.0 + epsilon) / spacing))
        if patch == "lateral":
            return around, max(1, math.ceil(length / spacing))
        if patch == "caps":
            outward = max(1, math.ceil(radius * (1.0 + epsilon) / spacing))
            while (outward > 1 and (outward - 1) * around > ring_floor(self.starts)
                   and any_closer(self._cap_plane(radius, epsilon, outward * around), CAP_SEPARATION * spacing)):
                outward -= 1
            return outward, around
        raise KeyError(patch)

    def _cap_plane(self, radius: float, epsilon: float, total: int) -> np.ndarray:
        """A cap's samples in its own plane, unturned (a turn moves none nearer another)."""
        rings = cap_rings(total, self.starts)
        reach = rings.fraction * radius * (1.0 + epsilon * np.sin(self.starts * rings.angle))
        return np.stack([reach * np.cos(rings.angle), reach * np.sin(rings.angle)], axis=-1)

    @override
    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        """The lateral patch by the trapezoid rule round the turn and Gauss-Legendre nodes along the axis, weighted by
        the surface element ``|r_theta x r_zeta|``. Each cap in rings (``mime.scene._caps``): its ``outward x around``
        samples are a sample at its centre and rings out to its rim, each a homothetic copy of the section at a fraction
        ``s`` of the way out, its count growing with its radius (at least ``4 starts + 3``, which the second moments
        need round it), weighted by ``s rho^2 ds dtheta``, so that the area and the first and second moments are exact
        and no two samples lie closer than ``CAP_SEPARATION`` spacings (:meth:`sample_counts` lowers ``outward`` where
        the section's troughs and flanks are too narrow for that many). Each section is turned whole by the screw's own
        turn at its axial position (``screw``'s, carried exact), so that a sample rounds at the section's radius, its
        ``rounding_length``, however far along the axis it lies."""
        radius, length = parameters["radius"], parameters["length"]
        nu, epsilon, starts = parameters["wavenumber"], parameters["modulation"], self.starts
        k, half = nu / radius, length / 2.0
        if patch == "lateral":
            around, along = counts
            node, weight = np.polynomial.legendre.leggauss(along)
            angle = np.repeat(2.0 * math.pi * (np.arange(around) + 0.5) / around, along)
            theta = jnp.asarray(angle)
            zeta = half * jnp.asarray(np.tile(node, around))
            rho = radius * (1.0 + epsilon * jnp.sin(starts * theta))
            slope = radius * epsilon * starts * jnp.cos(starts * theta)
            c, s = _turned(angle, SCREW.turn(parameters, zeta))
            points = jnp.stack([zeta, rho * c, rho * s], axis=-1)
            normal = jnp.stack([k * rho * slope, rho * c + slope * s, rho * s - slope * c], axis=-1)
            element = switch.nonzero_norm(normal, axis=-1)
            areas = jnp.asarray(np.tile(weight, around)) * half * (2.0 * math.pi / around)
            return Samples(points, normal / element[:, None], element * areas, rounding_length=rho)
        if patch == "caps":
            rings = cap_rings(math.prod(counts), starts)
            angle = rings.angle
            fraction = jnp.asarray(rings.fraction)
            rho = radius * (1.0 + epsilon * jnp.sin(starts * jnp.asarray(angle)))
            # The centre's share is of the section's mean square radius, R^2 (1 + eps^2 / 2); at its angle, 0, rho is R.
            centre = jnp.asarray(np.where(rings.fraction == 0.0, 1.0, 0.0))
            square = rho ** 2 * (1.0 + centre * epsilon ** 2 / 2.0)
            found: list[Samples] = []
            for end in (-1.0, 1.0):
                c, s = _turned(angle, SCREW.turn(parameters, end * half))
                points = jnp.stack([jnp.full_like(rho, end * half), fraction * rho * c, fraction * rho * s], axis=-1)
                normals = jnp.zeros_like(points).at[:, 0].set(end)
                found.append(Samples(points, normals, jnp.asarray(rings.share) * square, rho))
            return Samples(*(jnp.concatenate([getattr(s, name) for s in found]) for name in
                             ("points", "normals", "weights", "rounding_length")))
        raise KeyError(patch)

    @override
    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        """The caps' ``2 pi R^2 (1 + eps^2 / 2)``; the lateral patch's ``L R`` times the integral round a turn of
        ``sqrt(g^2 + g'^2 (1 + nu^2 g^2))``, ``g = 1 + eps sin(N theta)``, which has no closed form and is taken by
        the trapezoid rule at ``AREA_POINTS``, to rounding."""
        radius, length, nu, epsilon = self._section(settings)
        if patch == "caps":
            return 2.0 * math.pi * radius * radius * (1.0 + epsilon * epsilon / 2.0)
        if patch == "lateral":
            theta = 2.0 * math.pi * np.arange(AREA_POINTS) / AREA_POINTS
            g = 1.0 + epsilon * np.sin(self.starts * theta)
            slope = epsilon * self.starts * np.cos(self.starts * theta)
            turn = float(np.sum(np.sqrt(g * g + slope * slope * (1.0 + nu * nu * g * g)))) * 2.0 * math.pi / AREA_POINTS
            return length * radius * turn
        raise KeyError(patch)

    @override
    def family_parameters(self, settings: Mapping[str, Magnitude]) -> dict[str, float | int]:
        radius, length, nu, epsilon = self._section(settings)
        return {"radius": radius, "length": length, "wavenumber": nu, "modulation": epsilon, "starts": self.starts}

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        """``R (1 + modulation)``: the crests' radius."""
        radius, _, _, epsilon = self._section(settings)
        return radius * (1.0 + epsilon)

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        """To a crest at a cap's rim: ``sqrt((R (1 + modulation))^2 + (L/2)^2)``."""
        radius, length, _, epsilon = self._section(settings)
        return math.hypot(radius * (1.0 + epsilon), length / 2.0)

    @override
    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        """Sections along the axis, ``segments`` points round each at the angles the body's formula takes, as many
        sections as keep each turned from the next by at most a ``segments``-th of a turn (one span for an untwisted
        body, whose sides are straight along it), the two ends on the caps; closed by a fan across each cap from its
        centre, which every point of a section sees, since the section's radius is a function of its angle."""
        radius, length, nu, epsilon = self._section(settings)
        k, half = nu / radius, length / 2.0
        spans = max(1, math.ceil(abs(k) * length * segments / (2.0 * math.pi)))
        theta = 2.0 * math.pi * np.arange(segments) / segments
        zeta = np.concatenate([-half + length * np.arange(spans) / spans, [half]])
        rho = radius * (1.0 + epsilon * np.sin(self.starts * theta))
        alpha = theta[None, :] + k * (zeta[:, None] + half)
        sections = np.stack([np.broadcast_to(zeta[:, None], alpha.shape), rho * np.cos(alpha), rho * np.sin(alpha)],
                            axis=-1).reshape(-1, 3)
        points = np.concatenate([sections, [(-half, 0.0, 0.0), (half, 0.0, 0.0)]])
        low, high = len(sections), len(sections) + 1
        faces = [face[::-1] for face in banded_faces(spans + 1, segments, low, high, start=0)]
        return closed_drawing(points, faces, f"the screw-shaped body as {spans + 1} sections of {segments} points "
                                             f"along its axis, the prim's +X, closed by a fan across each cap")

    @override
    def region_primitive(self, region: str, settings: Mapping[str, Magnitude]) -> RegionPrimitive | NoPrimitive:
        """The span between the caps as a slab; the screw has none."""
        if region == "span":
            return RegionPrimitive("slab", {"point": (0.0, 0.0, 0.0), "normal": (1.0, 0.0, 0.0),
                                            "thickness": _held(settings, "length")})
        if region == "screw":
            return NoPrimitive(f"the screw's surface is no primitive. {NO_PRIMITIVE_ADVICE}")
        raise KeyError(region)

    @override
    def tangent(self, parameters: Mapping[str, Any], point: Any) -> Any:
        """The thread's direction at the lateral surface's point nearest ``point``: along the surface at constant
        angle, towards +x (``screw``'s tangent, with its seam)."""
        screw = {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "starts": self.starts,
                 **{name: parameters[name] for name in ("radius", "length", "wavenumber", "modulation")}}
        return SCREW.tangent(screw, point)


__all__ = ["LARGEST_MODULATION", "LARGEST_WAVENUMBER", "STARTS", "Screw"]
