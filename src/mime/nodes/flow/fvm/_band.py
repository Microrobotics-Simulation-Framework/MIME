"""The wall on the grid: a solid's signed distance lifted onto the faces by a smooth band, and the property-field seam
the solid reaches the step through.

**The band** (F0's B3). A solid's indicator rises from 0 in the fluid to 1 in the solid across a band ``W`` wide,
centred on the wall, as a sine ramp in the signed distance; the penalty there is a resistivity ``nu / K`` with
``1 / K = chi / (delta h)^2``. Beyond the band a stiffer interior (``delta_in``) adds ``(delta / delta_in)^2`` times as
much again, over a second ramp a band wide starting at the band's outer edge, so the Darcy flow through the solid is
``(delta_in / delta)^2`` of the band's. Only smooth bands are registered (:data:`BANDS`): the wall's shift is then
smooth in the wall's sub-cell position, which the wall plant's control needs. The band is an accuracy-bearing
numerical setting; its measured shift is its validity line.

**The seam** (B5). A property field is a function of points, SI in and SI out, with its quantity, units and region;
the grid kind decides where it samples (each velocity component's faces). F1a builds one source, a region's
assignment: the wall's solid, from the wall kind's signed distance. Regions combine by adding their resistivities, so
the step reads ``nu x sum(1/K)``: the viscosity multiplies in the step and is never folded into a value made at build.
The geometry enters as data (the tube's centre, axis and radius are the step's parameters), so the fields are made in
the program from them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
from jax import Array

from mime.core import jnp, switch
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.walls import TubeWall

#: The indicator's switch: the band's clip of the signed distance to its half-width.
BAND_OWNER = "fvm.band"
#: The solid's images in a periodic box: the nearest one's distance is the solid's.
IMAGE_OWNER = "fvm.image"


@dataclass(frozen=True)
class Band:
    """A registered band: its width ``W`` and penalty depth ``delta``, each in cells, and its measured validity."""

    name: str
    width: float
    depth: float
    validity: str


#: The registered bands. (2h, 0.3h) is the one smooth sweet spot (S1: shift +0.01 cells over sub-cell positions; the
#: quarter-cell wall plant caught on every run).
BANDS = MappingProxyType({
    "smooth_2h_0.3h": Band("smooth_2h_0.3h", 2.0, 0.3,
                           "a first-order wall: its effective position shifted from the signed distance's zero by a "
                           "planar calibration's offset, constant in cells; calibrated on straight tubes at D/h 16 to "
                           "256, aligned and tilted"),
})
DEFAULT_BAND = "smooth_2h_0.3h"

#: The stiff interior's offered depths, in cells: 0.1h by default, 0.01h where the solid's bypass flux must be smaller
#: (a clot's pressure drop), at about ten times the iterations.
INTERIORS = MappingProxyType({"0.1h": 0.1, "0.01h": 0.01})
DEFAULT_INTERIOR = "0.1h"


def band_named(name: str) -> Band:
    if name not in BANDS:
        raise ValueError(f"no band {name!r} is registered: only smooth bands are offered, {sorted(BANDS)}")
    return BANDS[name]


def interior_depth(name: str) -> float:
    if name not in INTERIORS:
        raise ValueError(f"the stiff interior's depth is one of {sorted(INTERIORS)}, not {name!r}")
    return INTERIORS[name]


def ramp(phi: Array, width: Any) -> Array:
    """The band's indicator: 0 at ``phi <= -width/2``, 1 at ``phi >= width/2``, a sine ramp between."""
    s = phi / width
    clipped = switch.clip(s, -0.5, 0.5, owner=BAND_OWNER, resolution=lambda: switch.rounding(s), indicator=s)
    return 0.5 * (1.0 + jnp.sin(jnp.pi * clipped))


@dataclass(frozen=True)
class TubeSolid:
    """The solid around a straight circular tube's lumen, read from the ``tube`` wall kind's signed distance (positive
    in the solid), at the nearest of its periodic images: the lattice shifts ``images`` (m) are the box's periods that
    can bring an image within reach of a face."""

    centre: Any
    axis: Any
    radius: Any
    images: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)

    region: str = "wall"

    def signed_distance(self, points: Array) -> Array:
        parameters = {"centre": jnp.asarray(self.centre), "axis": jnp.asarray(self.axis),
                      "radius": jnp.asarray(self.radius)}
        wall = TubeWall()
        nearest: Array | None = None
        with jax.default_matmul_precision("highest"):
            for shift in self.images:
                found = wall.distance(parameters, points - jnp.asarray(shift, dtype=points.dtype))
                assert found is not None
                phi = -found.distance
                nearest = phi if nearest is None else switch.minimum(
                    nearest, phi, owner=IMAGE_OWNER, resolution=lambda: switch.rounding(phi),
                    indicator=lambda: phi - nearest)
        assert nearest is not None
        return nearest


@dataclass(frozen=True)
class WallFields:
    """What the step reads of a solid, sampled on each velocity component's faces: the band's indicator ``chi`` (the
    body force acts on ``1 - chi``), the inverse permeability ``1/K`` (m^-2) of every region summed, and the signed
    distance, from which the lumen's and the solid's fluxes are taken."""

    chi: Velocity
    inverse_permeability: Velocity
    signed_distance: Velocity


def wall_fields(grid: MACGrid, solids: Sequence[TubeSolid], band: Band, interior: float, dtype: Any,
                offset: Callable[[Array], Array] | None = None) -> WallFields:
    """Each solid's fields on ``grid``'s faces, the regions' resistivities added. The band's width and depth scale
    with the spacing; the interior adds ``(delta / delta_in)^2`` over a second ramp beyond the band."""
    h = grid.spacing
    width = band.width * h
    stiffening = (band.depth / interior) ** 2
    chis: list[Array] = []
    inverse: list[Array] = []
    distances: list[Array] = []
    for component in range(3):
        points = grid.face_points(component, dtype)
        if offset is not None:
            points = offset(points)
        phis = [solid.signed_distance(points) for solid in solids]
        phi = phis[0]
        for other in phis[1:]:
            phi = switch.minimum(phi, other, owner=IMAGE_OWNER, resolution=lambda: switch.rounding(other),
                                 indicator=lambda: other - phi)
        chi = ramp(phi, width)
        resistivity = jnp.zeros_like(phi)
        for p in phis:
            band_part = ramp(p, width)
            interior_part = ramp(p - width, max(band.width, 1.0) * h) * stiffening
            resistivity = resistivity + (band_part + interior_part) / (band.depth * h) ** 2
        chis.append(chi)
        inverse.append(resistivity)
        distances.append(phi)
    return WallFields((chis[0], chis[1], chis[2]), (inverse[0], inverse[1], inverse[2]),
                      (distances[0], distances[1], distances[2]))


def negative_resistivity(fields: WallFields) -> Array:
    """Whether any region's sampled resistivity is negative: the positivity a source must keep, checked at sampling
    (the band's form keeps it by construction; a later source may not)."""
    return jnp.any(jnp.stack([jnp.any(k < 0.0) for k in fields.inverse_permeability]))


__all__ = ["BANDS", "DEFAULT_BAND", "DEFAULT_INTERIOR", "INTERIORS", "Band", "TubeSolid", "WallFields", "band_named",
           "interior_depth", "negative_resistivity", "ramp", "wall_fields"]
