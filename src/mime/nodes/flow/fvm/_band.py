"""The wall on the grid: a solid's signed distance lifted onto the faces by a smooth band, and the property-field seam
the solid reaches the step through.

**The band** (F0's B3). A solid's indicator rises from 0 in the fluid to 1 in the solid across a band ``W`` wide,
centred on the wall, as a sine ramp in the signed distance; the penalty there is a resistivity ``nu / K`` with
``1 / K = chi / (delta h)^2``. Beyond the band a stiffer interior (``delta_in``) adds ``(delta / delta_in)^2`` times as
much again, over a second ramp from the band's outer edge (``h`` into the solid) to ``INTERIOR_RAMP_END`` (3h), so the
Darcy flow through the solid is ``(delta_in / delta)^2`` of the band's. A ramp ending at 1.5h (row #315) was measured
and not adopted: it halves the bypass at the 0.01h interior and an occlusion's leak falls, but through a 75 %
stenosis at the 0.1h interior its bypass rose 2.3 % and its flux fell 0.44 %, past the gate's line. Only smooth bands are registered (:data:`BANDS`): the wall's shift is then
smooth in the wall's sub-cell position, which the wall plant's control needs. The band is an accuracy-bearing
numerical setting; its measured shift is its validity line.

**The seam** (B5). A property field is a function of points, SI in and SI out, with its quantity, units and region;
the grid kind decides where it samples (each velocity component's faces). F1a builds one source, a region's
assignment: the wall's solid, from the wall kind's signed distance. Regions combine by adding their resistivities, so
the step reads ``nu x sum(1/K)``: the viscosity multiplies in the step and is never folded into a value made at build.
The geometry enters as data (the tube's centre, axis and radius are the step's parameters; a tube tree's points,
radii and fillet), so the fields are made in the program from them.

**A tube tree's wall** (:class:`TreeSolid`) is its lumen's union's exact distance (``mime.scene.ExactUnion``) wherever
a field the step reads is not flat, made in the program from the tree's leaves: the plain minimum on every face, exact
in the solid and never deeper than the union in the fluid, and the union's candidates where its check fails within the
band's fluid half (deeper, every field is flat). Where they run is chosen in the program at the run's values; only how
many may run is held at build (:func:`plan_rung`, a rung above the count the build's values need, as the cell counts
are held). The union's counts (``UnionDistance``: failing faces past that size, chosen faces whose candidates did not
stand, pairs of tubes the run's values bring within reach that the union does not hold as meeting) travel beside the
fields, for the step's report: the band reads only the distance. The wall is constant over a run, so a chunk makes it
once (the node's run-constant)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.scene import ExactUnion
from mime.walls import TubeWall



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
#: Where the stiff interior's ramp ends, in cells into the solid: it starts at the band's outer edge (half the band's
#: width, h) and reaches the interior's full stiffness here.
INTERIOR_RAMP_END = 3.0


def band_named(name: str) -> Band:
    if name not in BANDS:
        raise ValueError(f"no band {name!r} is registered: only smooth bands are offered, {sorted(BANDS)}")
    return BANDS[name]


def interior_depth(name: str) -> float:
    if name not in INTERIORS:
        raise ValueError(f"the stiff interior's depth is one of {sorted(INTERIORS)}, not {name!r}")
    return INTERIORS[name]


def ramp(phi: Array, width: Any) -> Array:
    """The band's indicator: 0 at ``phi <= -width/2``, 1 at ``phi >= width/2``, a sine ramp between. The ramp meets the
    flats with zero slope, so it is continuously differentiable across its clip: no derivative jumps there."""
    return 0.5 * (1.0 + jnp.sin(jnp.pi * jnp.clip(phi / width, -0.5, 0.5)))


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
                # Two images are equidistant only deep in the solid, beyond the band and the interior's ramp, where
                # every field the step reads of the solid is flat: the minimum's kink reaches nothing.
                nearest = phi if nearest is None else jnp.minimum(nearest, phi)
        assert nearest is not None
        return nearest


#: The least of the plan's sizes (:func:`plan_rung`), in faces.
PLAN_QUANTUM = 256


def plan_rung(count: int) -> int:
    """How many faces a tube tree's node may run the union's candidates at, held at build for ``count`` (the faces the
    build's values choose): the least rung ``PLAN_QUANTUM x ceil(2^(k/2))`` at or above it, so between ``count`` and
    about 1.41 times it, and a move of the walls that changes ``count`` by a few faces keeps one program; zero for
    none. A declaration, not a fit."""
    if count <= 0:
        return 0
    k = 0
    while PLAN_QUANTUM * math.ceil(2.0 ** (k / 2.0)) < count:
        k += 1
    return PLAN_QUANTUM * math.ceil(2.0 ** (k / 2.0))


@dataclass(frozen=True)
class TreeSolid:
    """The solid around a tube tree's lumen (``mime.scene.TubeTree``), its signed distance the union's (positive in
    the solid) from ``parameters`` (the tree's leaves, traced): the union's candidates run at up to ``planned`` faces
    where its check fails shallower than ``floor`` (``-W h / 2``, the band's fluid half; deeper every field the step
    reads is flat), the plain minimum elsewhere (``ExactUnion.evaluate_chosen``). ``given`` is the wall a chunk made
    once (the node's run-constant: the distance on the three face sets' faces in order, and the counts), read in place
    of making it."""

    union: ExactUnion
    parameters: Mapping[str, Any]
    floor: Any
    planned: int
    given: tuple[Array, Array] | None = None
    region: str = "wall"

    def made(self, points: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        """The distance on the three face sets' ``points``, flat in their order, and ``(unrefined, unconverged,
        unplanned)``."""
        flat = jnp.concatenate([held.reshape(-1, 3) for held in points])
        found = self.union.evaluate_chosen(self.parameters, flat, self.planned, self.floor)
        counts = jnp.stack([jnp.asarray(n).astype(flat.dtype)
                            for n in (found.unrefined, found.unconverged, found.unplanned)])
        return found.distance, counts

    def face_distances(self, points: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array, Array], Array]:
        """The distance on each face set's ``points`` (as given, or made: :meth:`made`), and the counts."""
        distance, counts = self.given if self.given is not None else self.made(points)
        ends = np.cumsum([0] + [int(np.prod(held.shape[:-1])) for held in points])
        phis = [distance[ends[k]:ends[k + 1]].reshape(points[k].shape[:-1]) for k in range(3)]
        return (phis[0], phis[1], phis[2]), counts


@dataclass(frozen=True)
class WallFields:
    """What the step reads of a solid, sampled on each velocity component's faces: the band's indicator ``chi`` (the
    body force acts on ``1 - chi``), the inverse permeability ``1/K`` (m^-2) of every region summed, and the signed
    distance, from which the lumen's and the solid's fluxes are taken."""

    chi: Velocity
    inverse_permeability: Velocity
    signed_distance: Velocity
    #: A tube tree's counts, summed over the face sets (:class:`TreeSolid`): ``(unrefined, unconverged, unplanned)``;
    #: ``None`` for a wall whose distance is exact by construction (a straight tube's).
    counts: Array | None = None


@runtime_checkable
class FacedSolid(Protocol):
    """A solid whose distance is made on the three face sets' points together (a tube tree's, one union over them all),
    with the counts of where it is not exact (:class:`TreeSolid`)."""

    @property
    def region(self) -> str: ...

    def face_distances(self, points: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array, Array], Array]: ...


def interior_ramp(phi: Array, band: Band, h: Any, end: float) -> Array:
    """The stiff interior's indicator: 0 to the band's outer edge (``phi = W/2``), 1 from ``end`` cells into the solid,
    a sine ramp between."""
    start = 0.5 * band.width
    return ramp(phi - 0.5 * (start + end) * h, (end - start) * h)


def wall_fields(grid: MACGrid, solids: Sequence[TubeSolid | FacedSolid], band: Band, interior: float, dtype: Any,
                offset: Callable[[Array], Array] | None = None) -> WallFields:
    """Each solid's fields on ``grid``'s faces, the regions' resistivities added. The band's width and depth scale
    with the spacing; the interior adds ``(delta / delta_in)^2`` over a second ramp beyond the band, ending
    :data:`INTERIOR_RAMP_END` cells into the solid. A solid held per face set (:class:`FacedSolid`, one a wall) gives
    its counts too."""
    h = grid.spacing
    width = band.width * h
    stiffening = (band.depth / interior) ** 2
    end = INTERIOR_RAMP_END
    faces: list[Array] = []
    for component in range(3):
        points = grid.face_points(component, dtype)
        faces.append(points if offset is None else offset(points))
    held = (faces[0], faces[1], faces[2])
    faced: dict[int, tuple[Array, Array, Array]] = {}
    counts: Array | None = None
    for index, solid in enumerate(solids):
        if isinstance(solid, FacedSolid):
            faced[index], counts = solid.face_distances(held)
    chis: list[Array] = []
    inverse: list[Array] = []
    distances: list[Array] = []
    for component in range(3):
        phis = [faced[index][component] if index in faced else cast(TubeSolid, solid).signed_distance(faces[component])
                for index, solid in enumerate(solids)]
        phi = phis[0]
        for other in phis[1:]:
            phi = jnp.where(other < phi, other, phi)  # the nearer solid, by one comparison (its derivative the chosen's)
        chi = ramp(phi, width)
        resistivity = jnp.zeros_like(phi)
        for p in phis:
            band_part = ramp(p, width)
            interior_part = interior_ramp(p, band, h, end) * stiffening
            resistivity = resistivity + (band_part + interior_part) / (band.depth * h) ** 2
        chis.append(chi)
        inverse.append(resistivity)
        distances.append(phi)
    return WallFields((chis[0], chis[1], chis[2]), (inverse[0], inverse[1], inverse[2]),
                      (distances[0], distances[1], distances[2]), counts)


def negative_resistivity(fields: WallFields) -> Array:
    """Whether any region's sampled resistivity is negative: the positivity a source must keep, checked at sampling
    (the band's form keeps it by construction; a later source may not)."""
    return jnp.any(jnp.stack([jnp.any(k < 0.0) for k in fields.inverse_permeability]))


__all__ = ["BANDS", "DEFAULT_BAND", "DEFAULT_INTERIOR", "INTERIORS", "INTERIOR_RAMP_END", "PLAN_QUANTUM", "Band",
           "FacedSolid", "TreeSolid", "TubeSolid", "WallFields", "band_named", "interior_depth", "interior_ramp",
           "negative_resistivity", "plan_rung", "ramp", "wall_fields"]
