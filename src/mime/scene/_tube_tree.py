"""A vessel as a tree of straight tubes (``TubeTree``): points, the segments between them with a radius at each end,
and a fillet, from which come its centreline (``mime.registry.Centreline``: its segments, and a junction where two
segment ends or more meet) and its lumen's exact signed distance (``mime.scene.ExactUnion``). A Y is its first case,
a periodic split-and-merge loop its second; a curve is a chain of short segments, and a network may hold a loop.

**The lumen.** Each segment is the union of the balls along it, its radius changing linearly from one end to the
other: a round cone (a capsule where the radii agree), so a segment's radius is the inscribed ball's at each point of
its centreline. The lumen is their union, its distance the seam-aware rule's, so a carina and a daughter's root are
exact to rounding, not the minimum's false wall. A free end (one no other segment meets) closes with its end ball:
whatever opens it (a port, a periodic box) cuts it.

**The fillet** is one radius for the tree, ``fillet`` times the smallest radius any segment has at a junction (the
smaller daughter's at a Y), on every edge the solid has between two walls: the carina's and each root's. It is a
parameter of the geometry, traced like the radii, and never set from a grid. Its default is ``FILLET_SHARE``, a
declaration: no measurement of vessels gives it yet. One radius, not one a junction: the exact rule is the closing by
one ball.

**Its values are parameters** (:meth:`TubeTree.parameters`: the points, the radii and the fillet's share, in SI), read
by the centreline and the lumen as JAX functions, so a derivative in any of them flows; its structure (which points
the segments join, so which tubes meet) is fixed. Tubes that do not share a point but overlap (a tight bend's
non-neighbours, a loop closing on itself) meet as well, found at the values it is made with.

**Tilted vessels** are its points turned (:meth:`TubeTree.turned`): nothing in it is aligned with an axis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.core import jnp
from mime.registry import Centreline, CentrelineFrame, CentrelineSegment, Junction
from mime.scene._centreline import across, transported
from mime.scene._union import ExactUnion, RoundCone

type Parameters = Mapping[str, Any]

#: The fillet's radius at a vessel's junctions, as a share of the smallest radius a segment has at one (the smaller
#: daughter's at a Y): an anatomical baseline declared for vessels, not fitted to any measurement.
FILLET_SHARE = 0.1
#: How many points along each of two tubes that share no point their nearest approach is read at, when the tree is
#: made, to find whether they overlap.
APPROACH_POINTS = 257


def _refused(message: str) -> Exception:
    """The refusal of a tree's declaration (``call.tube_tree``): the ``ValueError`` a call's argument raises, coded."""
    from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass

    return ArgumentRefused([Refusal("call.tube_tree", RefusalClass.USER, message,
                                    "give TubeTree(points, segments, radii) three numbers a point, two distinct "
                                    "points' indices a segment, two radii above zero a segment (at its start and its "
                                    "end) differing by less than its length, every point on a segment, and a fillet "
                                    "share from zero on")])


@dataclass(frozen=True)
class WallFeatures:
    """What a wall's geometry gives the resolution report, in metres and degrees: its smallest radius of curvature (a
    sharp seam's is zero), its narrowest lumen (a diameter), the narrowest solid between two tubes that do not meet
    (infinite where none come close), and the sharpest wedge of solid between two tubes leaving a junction (its
    opening, degrees; 180 where none)."""

    smallest_radius: float
    narrowest_lumen: float
    narrowest_solid: float
    sharpest_wedge: float


@dataclass(frozen=True)
class TubeTree:
    """A vessel as straight tubes between ``points`` (m, three numbers each, in the world's frame): ``segments`` joins
    two of them by index (from its start to its end), ``radii`` gives its radius at its start and at its end (m), and
    ``fillet`` is the fillet's radius as a share of the smallest radius at a junction (zero: sharp). The module's
    docstring says what it is."""

    points: tuple[tuple[float, float, float], ...]
    segments: tuple[tuple[int, int], ...]
    radii: tuple[tuple[float, float], ...]
    fillet: float = FILLET_SHARE

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
            raise _refused(f"a tube tree's points are two or more, each three finite numbers; it was given shape "
                           f"{points.shape}")
        if not self.segments or any(len(pair) != 2 for pair in self.segments):
            raise _refused(f"a tube tree has one segment or more, each two points' indices: {self.segments}")
        for start, end in self.segments:
            if not (0 <= start < len(points) and 0 <= end < len(points)) or start == end:
                raise _refused(f"segment ({start}, {end}) joins two distinct points among {len(points)}")
        if len({frozenset(pair) for pair in self.segments}) != len(self.segments):
            raise _refused(f"two segments join the same two points: {self.segments}")
        if {index for pair in self.segments for index in pair} != set(range(len(points))):
            raise _refused("every point of a tube tree is an end of a segment")
        radii = np.asarray(self.radii, dtype=np.float64)
        if radii.shape != (len(self.segments), 2) or not np.all(np.isfinite(radii) & (radii > 0.0)):
            raise _refused(f"a tube tree's radii are two above zero a segment, {len(self.segments)} segments; it was "
                           f"given shape {radii.shape}")
        lengths = self.lengths()
        if np.any(np.abs(radii[:, 0] - radii[:, 1]) >= lengths):
            raise _refused("a segment's radii differ by less than its length, or one end's ball holds the other's")
        if not (math.isfinite(self.fillet) and self.fillet >= 0.0):
            raise _refused(f"a tube tree's fillet share is {self.fillet!r}, not a number from zero on")

    # -- its structure, fixed -------------------------------------------------------------------------------------
    def lengths(self) -> np.ndarray:
        points = np.asarray(self.points, dtype=np.float64)
        return np.asarray([np.linalg.norm(points[end] - points[start]) for start, end in self.segments])

    @property
    def junctions(self) -> tuple[Junction, ...]:
        """A junction at each point two segment ends or more meet at, named ``j<point>``, its ends in segment order."""
        ends: dict[int, list[tuple[int, int]]] = {}
        for index, (start, end) in enumerate(self.segments):
            ends.setdefault(start, []).append((index, 0))
            ends.setdefault(end, []).append((index, 1))
        return tuple(Junction(f"j{point}", tuple(held)) for point, held in sorted(ends.items()) if len(held) >= 2)

    def _junction_ends(self) -> tuple[np.ndarray, np.ndarray]:
        """The (segment, end) pairs that lie at a junction, as two index arrays into the radii."""
        held = [end for junction in self.junctions for end in junction.ends]
        return np.asarray([s for s, _ in held], dtype=np.int64), np.asarray([e for _, e in held], dtype=np.int64)

    def meets(self) -> tuple[tuple[int, int], ...]:
        """The pairs of segments whose tubes meet: those sharing a point, and those that overlap (with the fillet's
        widening) at the values the tree is made with."""
        shared = {(i, j) for i in range(len(self.segments)) for j in range(i + 1, len(self.segments))
                  if set(self.segments[i]) & set(self.segments[j])}
        points = np.asarray(self.points, dtype=np.float64)
        radii = np.asarray(self.radii, dtype=np.float64)
        widening = 2.0 * self.fillet_radius_value()
        t = np.linspace(0.0, 1.0, APPROACH_POINTS)
        lines = [points[a][None] + t[:, None] * (points[b] - points[a])[None] for a, b in self.segments]
        steps = [float(np.linalg.norm(points[b] - points[a])) / (APPROACH_POINTS - 1) for a, b in self.segments]
        for i in range(len(self.segments)):
            for j in range(i + 1, len(self.segments)):
                if (i, j) in shared:
                    continue
                apart = float(np.min(np.linalg.norm(lines[i][:, None] - lines[j][None], axis=-1)))
                if apart < radii[i].max() + radii[j].max() + widening + steps[i] + steps[j]:
                    shared.add((i, j))
        return tuple(sorted(shared))

    # -- its values, parameters ---------------------------------------------------------------------------------------
    def parameters(self) -> dict[str, np.ndarray]:
        """Its values in SI by name, the leaves its centreline and lumen read: ``points`` ``(k, 3)``, ``radii``
        ``(n, 2)`` and ``fillet`` (the share)."""
        return {"points": np.asarray(self.points, dtype=np.float64),
                "radii": np.asarray(self.radii, dtype=np.float64),
                "fillet": np.asarray(self.fillet, dtype=np.float64)}

    def fillet_radius(self, parameters: Parameters) -> Any:
        """The fillet's radius (m): its share times the smallest radius a segment has at a junction; zero with no
        junction."""
        segments, ends = self._junction_ends()
        if not segments.size:
            return jnp.asarray(0.0) * jnp.asarray(parameters["fillet"])
        radii = jnp.asarray(parameters["radii"])
        return jnp.asarray(parameters["fillet"]) * jnp.min(radii[segments, ends])

    def fillet_radius_value(self) -> float:
        return float(np.asarray(self.fillet_radius(self.parameters())))

    def lumen(self) -> ExactUnion:
        """Its lumen: the round cones' union, filleted, its exact signed distance a JAX function of
        :meth:`parameters`."""
        regions = tuple(
            RoundCone(start=lambda p, a=a: jnp.asarray(p["points"])[a], end=lambda p, b=b: jnp.asarray(p["points"])[b],
                      start_radius=lambda p, i=i: jnp.asarray(p["radii"])[i, 0],
                      end_radius=lambda p, i=i: jnp.asarray(p["radii"])[i, 1])
            for i, (a, b) in enumerate(self.segments))
        return ExactUnion(regions, self.meets(), float(np.max(np.asarray(self.radii))), fillet=self.fillet_radius)

    def centreline(self) -> Centreline:
        """Its centreline: a straight segment for each, its local radius linear between its ends' (the inscribed
        ball's), and a junction where two ends or more meet. Each segment's normal is held as made: across its
        tangent, carried round the corner from the first segment that ends where it starts (``mime.scene.
        transported``), so the frame is continuous through every junction a parent reaches."""
        points = np.asarray(self.points, dtype=np.float64)
        tangents = [(points[b] - points[a]) / np.linalg.norm(points[b] - points[a]) for a, b in self.segments]
        normals: dict[int, np.ndarray] = {}
        incoming = {end: index for index, (_, end) in reversed(list(enumerate(self.segments)))}
        for _ in range(len(self.segments)):
            for index, (start, _) in enumerate(self.segments):
                if index in normals:
                    continue
                parent = incoming.get(start)
                if parent is None or parent == index:
                    normals[index] = across(tangents[index])
                elif parent in normals:
                    normals[index] = transported(normals[parent], tangents[parent], tangents[index])
        for index in range(len(self.segments)):
            normals.setdefault(index, across(tangents[index]))
        return Centreline(tuple(_segment(index, pair, normals[index]) for index, pair in enumerate(self.segments)),
                          self.junctions)

    # -- placed ------------------------------------------------------------------------------------------------------
    def turned(self, rotation: Sequence[Sequence[float]] | np.ndarray, offset: Sequence[float] = (0.0, 0.0, 0.0)
               ) -> TubeTree:
        """The tree turned by ``rotation`` (a rotation matrix acting on column vectors) and then moved by
        ``offset`` (m): a tilted or displaced vessel."""
        turn = np.asarray(rotation, dtype=np.float64)
        moved = np.asarray(self.points, dtype=np.float64) @ turn.T + np.asarray(offset, dtype=np.float64)
        return TubeTree(tuple((float(p[0]), float(p[1]), float(p[2])) for p in moved), self.segments, self.radii,
                        self.fillet)

    # -- what the resolution report reads ------------------------------------------------------------------------------
    def features(self) -> WallFeatures:
        """Its :class:`WallFeatures` at the values it is made with."""
        radii = np.asarray(self.radii, dtype=np.float64)
        junctions = self.junctions
        smallest = float(radii.min())
        if junctions:
            smallest = min(smallest, self.fillet_radius_value())
        points = np.asarray(self.points, dtype=np.float64)
        wedge = 180.0
        for junction in junctions:
            directions = []
            for segment, end in junction.ends:
                a, b = (points[i] for i in self.segments[segment])
                away = (b - a) if end == 0 else (a - b)
                directions.append(away / np.linalg.norm(away))
            for i in range(len(directions)):
                for j in range(i + 1, len(directions)):
                    cosine = float(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
                    wedge = min(wedge, math.degrees(math.acos(cosine)))
        meets = set(self.meets())
        apart = math.inf
        t = np.linspace(0.0, 1.0, APPROACH_POINTS)
        for i in range(len(self.segments)):
            for j in range(i + 1, len(self.segments)):
                if (i, j) in meets:
                    continue
                a, b = (points[k] for k in self.segments[i])
                c, d = (points[k] for k in self.segments[j])
                along_i = a[None] + t[:, None] * (b - a)[None]
                along_j = c[None] + t[:, None] * (d - c)[None]
                ri = radii[i, 0] + t * (radii[i, 1] - radii[i, 0])
                rj = radii[j, 0] + t * (radii[j, 1] - radii[j, 0])
                gap = np.linalg.norm(along_i[:, None] - along_j[None], axis=-1) - ri[:, None] - rj[None]
                apart = min(apart, float(gap.min()))
        return WallFeatures(smallest, 2.0 * float(radii.min()), apart, wedge)


def _segment(index: int, pair: tuple[int, int], normal: np.ndarray) -> CentrelineSegment:
    """Segment ``index`` of a tree from point ``pair[0]`` to ``pair[1]``, read from the tree's parameters: its
    tangent the unit chord, its normal ``normal`` held across it."""
    start_point, end_point = pair

    def ends(parameters: Parameters, dtype: Any) -> tuple[Any, Any]:
        points = jnp.asarray(parameters["points"], dtype=dtype)
        return points[start_point], points[end_point]

    def length(parameters: Parameters) -> Any:
        a, b = ends(parameters, jnp.asarray(parameters["points"]).dtype)
        return jnp.sqrt(jnp.sum((b - a) ** 2))

    def frame(parameters: Parameters, s: Any) -> CentrelineFrame:
        at = jnp.asarray(s)
        a, b = ends(parameters, at.dtype)
        chord = b - a
        tangent = chord / jnp.sqrt(jnp.sum(chord * chord))
        held = jnp.asarray(normal, dtype=at.dtype)
        across_it = held - jnp.dot(held, tangent) * tangent
        across_it = across_it / jnp.sqrt(jnp.sum(across_it * across_it))
        count = at.shape[0]
        return CentrelineFrame(a[None, :] + at[:, None] * tangent[None, :], jnp.broadcast_to(tangent, (count, 3)),
                               jnp.broadcast_to(across_it, (count, 3)))

    def radius(parameters: Parameters, s: Any) -> Any:
        at = jnp.asarray(s)
        radii = jnp.asarray(parameters["radii"], dtype=at.dtype)
        return radii[index, 0] + (radii[index, 1] - radii[index, 0]) * at / length(parameters).astype(at.dtype)

    return CentrelineSegment(length=length, frame=frame, radius=radius)


__all__ = ["APPROACH_POINTS", "FILLET_SHARE", "TubeTree", "WallFeatures"]
