"""The wall on the grid: a solid's signed distance lifted onto the faces by a smooth band, and the property-field seam
the solid reaches the step through.

**The band** (F0's B3). A solid's indicator rises from 0 in the fluid to 1 in the solid across a band ``W`` wide,
centred on the wall, as a sine ramp in the signed distance; the penalty there is a resistivity ``nu / K`` with
``1 / K = chi / (delta h)^2``. Beyond the band a stiffer interior (``delta_in``) adds ``(delta / delta_in)^2`` times as
much again, over a second ramp from the band's outer edge (``h`` into the solid) to ``INTERIOR_RAMP_END`` (1.5h), so the
Darcy flow through the solid is ``(delta_in / delta)^2`` of the band's. The ramp ended at 3h until F2a measured the
reshaped one (row #315): starting at the band's edge and ending at 1.5h, it cuts the bypass an open vessel leaks and
leaves the wall's shift and the wall plant's detection where they were. Only smooth bands are registered (:data:`BANDS`): the wall's shift is then
smooth in the wall's sub-cell position, which the wall plant's control needs. The band is an accuracy-bearing
numerical setting; its measured shift is its validity line.

**The seam** (B5). A property field is a function of points, SI in and SI out, with its quantity, units and region;
the grid kind decides where it samples (each velocity component's faces). F1a builds one source, a region's
assignment: the wall's solid, from the wall kind's signed distance. Regions combine by adding their resistivities, so
the step reads ``nu x sum(1/K)``: the viscosity multiplies in the step and is never folded into a value made at build.
The geometry enters as data (the tube's centre, axis and radius are the step's parameters; a tube tree's points,
radii and fillet), so the fields are made in the program from them.

**A tube tree's wall** (:class:`TreeSolid`) reads the exact distance of its lumen's union (``mime.scene.ExactUnion``) on
the faces near the wall, the faces it was made for when the node was built (a set held with the plan of where the
candidates run): the band's contract, exact to the 0.05-cell line or counted. The faces beyond, deep in the fluid or the
solid, read the plain minimum, exact in the solid and, in the fluid, where every field the step reads is flat. The
union's counts (``UnionDistance``: failing points outside the plan, planned points whose candidates did not stand, pairs
of tubes the run's values bring within reach that the union does not hold as meeting) and the faces a moved wall has
brought into the band outside the held set travel beside the fields, for the step's report: the band reads only the
distance."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast

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
#: width, h) and reaches the interior's full stiffness here (row #315; it ended at 3h before).
INTERIOR_RAMP_END = 1.5


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


#: The faces a tube tree's exact distance is held on, beyond the band's fluid half and the fillet's radius, in cells:
#: the room a wall has to move (a study, a central difference) before a face it brings into the band lies outside the
#: held set and is counted. A declaration, not a fit.
NEAR_MARGIN_CELLS = 0.5
#: The margin, in cells, of the union's plan (``ExactUnion.plan``) on the held faces: a face whose deepest lumen's
#: nearest wall point lies inside another lumen or within this of one runs the candidates. A declaration, not a fit: it
#: covers a central difference's step and a small study's move; a wall moved further is counted (``unrefined``).
PLAN_MARGIN_CELLS = 0.25


@dataclass(frozen=True)
class TreeSolid:
    """The solid around a tube tree's lumen (``mime.scene.TubeTree``), its signed distance the union's (positive in
    the solid), from ``parameters`` (the tree's leaves, traced): exact on the faces ``near`` holds for each face set (a
    flat index into its points), its candidates run at ``plans`` (indices into those faces), and the plain minimum
    elsewhere: in the solid beyond the fillet's radius it is exact, and in the fluid it is never deeper than the
    union's depth, so a face it puts deeper than the band's fluid half is at least as deep, where every field the step
    reads is flat (the band's ramp meets its flat with zero slope there). ``reach`` is that interval, ``(below,
    above)`` (m) about the wall, for the count of faces a moved wall brings into it outside the held set."""

    union: ExactUnion
    parameters: Mapping[str, Any]
    near: tuple[np.ndarray, np.ndarray, np.ndarray]
    plans: tuple[np.ndarray, np.ndarray, np.ndarray]
    reach: tuple[float, float]
    region: str = "wall"

    def face_distances(self, points: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array, Array], Array]:
        """The distance on each face set's ``points``, the three sets' held faces evaluated together (one union, one
        program), and ``(unrefined, unconverged, unplanned)``: the union's counts, with the faces within ``reach`` of
        the wall outside the held set added to the first."""
        flats = [held.reshape(-1, 3) for held in points]
        sizes = np.cumsum([0] + [len(n) for n in self.near])
        plan = np.concatenate([self.plans[k] + sizes[k] for k in range(3)])
        found = self.union.evaluate(self.parameters, jnp.concatenate([flats[k][self.near[k]] for k in range(3)]),
                                    plan)
        missed = jnp.zeros((), dtype=jnp.int32)
        phis: list[Array] = []
        for k in range(3):
            minimum = self.union.minimum(self.parameters, flats[k])
            phi = minimum.at[self.near[k]].set(found.distance[sizes[k]:sizes[k + 1]])
            outside = np.ones(flats[k].shape[0], dtype=bool)
            outside[self.near[k]] = False
            within = (minimum > -self.reach[0]) & (minimum <= self.reach[1])
            missed = missed + jnp.sum(jnp.asarray(outside) & within)
            phis.append(phi.reshape(points[k].shape[:-1]))
        dtype = points[0].dtype
        counts = jnp.stack([(found.unrefined + missed).astype(dtype), jnp.asarray(found.unconverged).astype(dtype),
                            jnp.asarray(found.unplanned).astype(dtype)])
        return (phis[0], phis[1], phis[2]), counts


def held_faces(union: ExactUnion, parameters: Mapping[str, Any], grid: MACGrid, band: Band, fillet: float,
               dtype: Any) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray],
                                    tuple[float, float]]:
    """What a :class:`TreeSolid` holds for ``grid``, made from the tree's values as built: on each face set, the faces
    whose plain minimum lies within the band's fluid half (``W/2``) on the fluid side and within the fillet's radius
    ``fillet`` (m) on the solid side, each widened by :data:`NEAR_MARGIN_CELLS`, and the union's plan on them
    (:data:`PLAN_MARGIN_CELLS`); and that interval unwidened, which a face outside them is counted within. NumPy, fixed
    for the program."""
    h = float(grid.spacing)
    reach = (0.5 * band.width * h, fillet)
    margin = NEAR_MARGIN_CELLS * h
    near: list[np.ndarray] = []
    plans: list[np.ndarray] = []
    for component in range(3):
        flat = np.asarray(grid.face_points(component, dtype)).reshape(-1, 3)
        minimum = np.asarray(union.minimum(parameters, jnp.asarray(flat)))
        index = np.nonzero((minimum > -(reach[0] + margin)) & (minimum <= reach[1] + margin))[0]
        near.append(index)
        plans.append(union.plan(parameters, flat[index], PLAN_MARGIN_CELLS * h))
    return (near[0], near[1], near[2]), (plans[0], plans[1], plans[2]), reach


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


class FacedSolid(Protocol):
    """A solid whose distance is held per face set (a tube tree's: its held faces and plan are each set's): its signed
    distance on the three face sets' points, together, and the counts of where it is not exact (:class:`TreeSolid`)."""

    @property
    def region(self) -> str: ...

    def face_distances(self, points: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array, Array], Array]: ...


def interior_ramp(phi: Array, band: Band, h: Any, end: float) -> Array:
    """The stiff interior's indicator: 0 to the band's outer edge (``phi = W/2``), 1 from ``end`` cells into the solid,
    a sine ramp between."""
    start = 0.5 * band.width
    return ramp(phi - 0.5 * (start + end) * h, (end - start) * h)


def wall_fields(grid: MACGrid, solids: Sequence[TubeSolid | FacedSolid], band: Band, interior: float, dtype: Any,
                offset: Callable[[Array], Array] | None = None, ramp_end: float | None = None) -> WallFields:
    """Each solid's fields on ``grid``'s faces, the regions' resistivities added. The band's width and depth scale
    with the spacing; the interior adds ``(delta / delta_in)^2`` over a second ramp beyond the band, ending
    ``ramp_end`` cells into the solid (:data:`INTERIOR_RAMP_END` unless given). A solid held per face set
    (:class:`FacedSolid`, one a wall) gives its counts too."""
    h = grid.spacing
    width = band.width * h
    stiffening = (band.depth / interior) ** 2
    end = INTERIOR_RAMP_END if ramp_end is None else ramp_end
    faces: list[Array] = []
    for component in range(3):
        points = grid.face_points(component, dtype)
        faces.append(points if offset is None else offset(points))
    held = (faces[0], faces[1], faces[2])
    faced: dict[int, tuple[Array, Array, Array]] = {}
    counts: Array | None = None
    for index, solid in enumerate(solids):
        if not isinstance(solid, TubeSolid):
            faced[index], counts = solid.face_distances(held)
    chis: list[Array] = []
    inverse: list[Array] = []
    distances: list[Array] = []
    for component in range(3):
        phis = [faced[index][component] if index in faced else cast(TubeSolid, solid).signed_distance(faces[component])
                for index, solid in enumerate(solids)]
        phi = phis[0]
        for other in phis[1:]:
            phi = jnp.minimum(phi, other)
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


__all__ = ["BANDS", "DEFAULT_BAND", "DEFAULT_INTERIOR", "INTERIORS", "INTERIOR_RAMP_END", "NEAR_MARGIN_CELLS",
           "PLAN_MARGIN_CELLS", "Band", "FacedSolid", "TreeSolid", "TubeSolid", "WallFields", "band_named",
           "held_faces", "interior_depth", "interior_ramp", "negative_resistivity", "ramp", "wall_fields"]
