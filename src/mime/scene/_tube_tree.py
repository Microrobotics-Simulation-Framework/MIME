"""A vessel as a tree of straight tubes (``TubeTree``, the geometry kind ``tube_tree``): points, the segments between
them with a radius at each end, and a fillet, from which come its centreline (``mime.registry.Centreline``: its
segments, and a junction where two segment ends or more meet) and its lumen's exact signed distance
(``mime.scene.ExactUnion``). A Y is its first case, a periodic split-and-merge loop its second; a curve is a chain of
short segments, and a network may hold a loop.

**As a kind** it is declared in quantities, ``TubeTree(points=mm(0, 0, -3, 0, 0, 0, ...), segments=((0, 1), ...),
radii=mm(0.5, 0.5, ...), fillet=0.1, ends=("left", "right"))``: three numbers a point and two radii a segment (at its
start and its end), each a flat quantity; ``TubeTree.si(points, segments, radii, ...)`` takes them in SI as nested
tuples. Its regions are the ``lumen`` (the shape ``tube_tree``, the union's distance) and the ``wall``; its **ends**
are named by geometry, never by the role a law gives them (row #336): the first segment's start is ``root`` where no
other segment meets it, and every other free end (a point one segment reaches) takes the name ``ends`` gives it, in the
order of its points, ``end_0``, ``end_1``, ... by default (:meth:`TubeTree.end_names`).

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
the segments join, so which tubes meet) is fixed. Tubes that do not share a point but come within reach of each other
(their walls within two fillet radii and a margin: a tight bend's non-neighbours, a loop closing on itself, a hairpin's
legs) meet as well, found at the values it is made with; at a run's values the lumen counts the pairs that come within
reach without meeting (``UnionDistance.unplanned``: a study that moves the radii or the points far enough), where its
distance is not exact. Tubes that share no point and come within ``2 r`` across solid would be joined by the fillet
(its closing fills the wall): a tree that does so is refused when planned (row #353), naming a smaller fillet or a
thicker wall (:meth:`TubeTree.setting_problems`, from :meth:`TubeTree.wall_features`).

**Tilted vessels** are its points turned (:meth:`TubeTree.turned`): nothing in it is aligned with an axis."""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

import jax
import numpy as np

from mime.core import jnp, register_shape
from mime.gradients import GradientStatus
from mime.registry import PHYSICAL, Centreline, CentrelineFrame, CentrelineSegment, Geometry, GradientRecord, \
    Junction, convention, register_geometry
from mime.scene._centreline import across, transported
from mime.scene._union import ExactUnion, RoundCone
from mime.units import LENGTH, Length, Magnitude, Quantity, m

type Parameters = Mapping[str, Any]

#: The fillet's radius at a vessel's junctions, as a share of the smallest radius a segment has at one (the smaller
#: daughter's at a Y): an anatomical baseline declared for vessels, not fitted to any measurement.
FILLET_SHARE = 0.1
#: How far, as a share of the largest radius, two tubes that share no point may lie outside each other's reach (their
#: walls two fillet radii apart) and still be made to meet: the room a study has before the lumen counts the pair.
MEETING_MARGIN = 1.0 / 16.0
#: How many points along each of two tubes that share no point the solid between them is read at, for the report.
APPROACH_POINTS = 257
#: The name of the free end at the first segment's start.
ROOT = "root"
#: An end's name: lower-case words joined by underscores.
_END_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """``|v|`` over ``axis``, in NumPy."""
    found: np.ndarray = np.sqrt(np.sum(v * v, axis=axis))
    return found


def _refused(message: str) -> Exception:
    """The refusal of a tree's declaration (``call.tube_tree``): the ``ValueError`` a call's argument raises, coded."""
    from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass

    return ArgumentRefused([Refusal("call.tube_tree", RefusalClass.USER, message,
                                    "give TubeTree(points=mm(...), segments=..., radii=mm(...)) three lengths a point, "
                                    "two distinct points' indices a segment, two radii above zero a segment (at its "
                                    "start and its end) differing by less than its length, every point on a segment, "
                                    "a fillet share from zero on, and a name for each free end but the root's "
                                    "(lower-case words, none repeated)")])


def _si_lengths(value: object, what: str, width: int) -> np.ndarray:
    """A flat quantity of lengths ``width`` a row as SI rows, or the refusal naming ``what``."""
    if not (isinstance(value, Quantity) and value.dimension == LENGTH and isinstance(value.value, tuple)):
        raise _refused(f"a tube tree's {what} is a quantity of lengths, {width} numbers a row, as mm(...); it was "
                       f"given {value!r}")
    held = np.asarray(value.value, dtype=np.float64)
    if held.size % width:
        rows = "two or more, each three finite numbers" if what == "points" else "two above zero a segment"
        raise _refused(f"a tube tree's {what} are {rows}; it was given {held.size} numbers, not {width} a row")
    return held.reshape(-1, width)


@dataclass(frozen=True)
class WallFeatures:
    """What a wall's geometry gives the resolution report, in metres and degrees: its smallest radius of curvature (a
    sharp seam's is zero), its narrowest lumen (a diameter), the narrowest solid between two tubes that share no point
    (infinite where none has solid between them), the sharpest wedge of solid between two tubes leaving a junction (its
    opening, degrees; 180 where none), and the pairs of tubes sharing no point whose solid the fillet fills (thinner
    than its diameter: the closing joins them)."""

    smallest_radius: float
    narrowest_lumen: float
    narrowest_solid: float
    sharpest_wedge: float
    fused: tuple[tuple[int, int], ...] = ()


class TubeTreeShape:
    """The lumen of a tube tree as a registered shape (``tube_tree``): its union's exact signed distance and outward
    normal (``ExactUnion``, every point's candidates run) at an array of points, its parameters the tree's in SI, its
    ``segments`` among them its structure (two indices a segment, integers, or numbers a region's parameters hold),
    read when the shape is called and never traced. The tubes it holds as meeting are those sharing a point; a pair
    that shares none and comes within reach is counted (``UnionDistance.unplanned``), where a tree built from its
    values would hold it too. A reader that steps a tree holds the tree whole (the FVM's wall), not this shape."""

    parameter_names: tuple[str, ...] = ("points", "segments", "radii", "fillet")
    takes_many = True

    @staticmethod
    def union(parameters: Mapping[str, Any]) -> ExactUnion:
        flat = [int(round(float(i))) for i in np.asarray(parameters["segments"], dtype=np.float64).reshape(-1)]
        return _structural_lumen(tuple((flat[k], flat[k + 1]) for k in range(0, len(flat), 2)))

    def signed_distance(self, parameters: Mapping[str, Any], point: Any) -> Any:
        held = jnp.asarray(point)
        found = self.union(parameters).evaluate(parameters, held.reshape(-1, 3), None)
        return found.distance.reshape(held.shape[:-1])

    def outward_normal(self, parameters: Mapping[str, Any], point: Any) -> Any:
        held = jnp.asarray(point)
        union = self.union(parameters)

        def total(at: Any) -> Any:
            return jnp.sum(union.evaluate(parameters, at, None).distance)

        return cast(Any, jax).grad(total)(held.reshape(-1, 3)).reshape(held.shape)

    def axis(self, parameters: Mapping[str, Any]) -> Any:
        return None

    def axis_point(self, parameters: Mapping[str, Any]) -> Any:
        return None


@register_geometry
@dataclass(frozen=True, kw_only=True)
class TubeTree(Geometry):
    """A vessel as straight tubes between ``points`` (three lengths each, in the world's frame, flat): ``segments``
    joins two of them by index (from its start to its end), ``radii`` gives its radius at its start and at its end
    (two lengths a segment, flat), ``fillet`` is the fillet's radius as a share of the smallest radius at a junction
    (zero: sharp), and ``ends`` names its free ends but the root, in the order of their points. The module's docstring
    says what it is."""

    name = "tube_tree"
    version = "0.1.0"
    setting_roles = {
        **dict.fromkeys(("points", "segments", "radii"), PHYSICAL),
        "fillet": convention(PHYSICAL, "an anatomical baseline declared for vessels, a tenth of the smallest radius "
                                       "at a junction (the smaller daughter's at a Y), not fitted to any measurement"),
        "ends": convention(PHYSICAL, "a tree's free ends but the root are end_0, end_1, ... in the order of their "
                                     "points unless named"),
    }
    differentiability = GradientStatus.UNVERIFIED
    gradient = GradientRecord(
        where="in its points, radii and fillet, through the lumen's distance and normal",
        reason="the union's selections (the deepest lumen, the least of its candidates, whether a candidate stands) are "
               "plain JAX, declared as no switch: the distance has a kink where two candidates tie and its normal a "
               "jump; a seam's and a corner's point carry the implicit function theorem's derivative, which the "
               "union's gate holds to central differences, and a flow's derivative through the band is held by the "
               "FVM's",
        route="the union's selections written through mime.core.switch with owners, when a contact or a drag reads a "
              "tree's surface")
    references = ()
    builds = {"": ()}
    regions = ("lumen", "wall")
    region_shapes = {"lumen": "tube_tree"}
    points: Annotated[Quantity, Length]
    segments: tuple[tuple[int, int], ...]
    radii: Annotated[Quantity, Length]
    fillet: float = FILLET_SHARE
    ends: tuple[str, ...] = ()

    @classmethod
    def si(cls, points: Any, segments: Sequence[Sequence[int]], radii: Any, fillet: float = FILLET_SHARE,
           ends: Sequence[str] = ()) -> TubeTree:
        """The tree from its points ``(k, 3)`` and radii ``(n, 2)`` in metres (nested sequences or arrays)."""
        flat_points = tuple(float(c) for c in np.asarray(points, dtype=np.float64).reshape(-1))
        flat_radii = tuple(float(c) for c in np.asarray(radii, dtype=np.float64).reshape(-1))
        return cls(points=m(flat_points), segments=tuple((int(a), int(b)) for a, b in segments),
                   radii=m(flat_radii), fillet=float(fillet), ends=tuple(ends))

    def _points(self) -> np.ndarray:
        return _si_lengths(self.points, "points", 3)

    def _radii(self) -> np.ndarray:
        return _si_lengths(self.radii, "radii", 2)

    def __post_init__(self) -> None:
        points = self._points()
        if len(points) < 2 or not np.all(np.isfinite(points)):
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
        radii = self._radii()
        if radii.shape != (len(self.segments), 2) or not np.all(np.isfinite(radii) & (radii > 0.0)):
            raise _refused(f"a tube tree's radii are two above zero a segment, {len(self.segments)} segments; it was "
                           f"given shape {radii.shape}")
        lengths = self.lengths()
        if np.any(np.abs(radii[:, 0] - radii[:, 1]) >= lengths):
            raise _refused("a segment's radii differ by less than its length, or one end's ball holds the other's")
        fillet = cast(object, self.fillet)
        if not (isinstance(fillet, (int, float)) and math.isfinite(fillet) and fillet >= 0.0):
            raise _refused(f"a tube tree's fillet share is {self.fillet!r}, not a number from zero on")
        free = len(self._free_ends()) - (1 if self._rooted() else 0)
        if self.ends and (len(self.ends) != free or len(set(self.ends)) != len(self.ends) or ROOT in self.ends
                          or any(e in self.regions for e in self.ends)
                          or not all(isinstance(e, str) and _END_NAME.fullmatch(e) for e in self._end_settings())):
            raise _refused(f"a tube tree with {free} free ends besides its root names them all or none, each in "
                           f"lower-case words and none of them {ROOT!r}, a region's name ({', '.join(self.regions)}) "
                           f"or repeated; it was given {self.ends!r}")

    # -- as a kind --------------------------------------------------------------------------------------------------
    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        """Its rows, and the fillet's fusion refused (row #353): tubes that share no point with solid between them
        thinner than the fillet's diameter, which its closing would fill, joining them."""
        found: list[str] = []
        for name, width in (("points", 3), ("radii", 2)):
            value = settings[name]
            if not (isinstance(value, tuple) and len(value) % width == 0 and all(math.isfinite(v) for v in value)):
                found.append(f"{name} is {value!r} m, not {width} finite lengths a row")
        if found:
            return found
        tree = TubeTree.si(np.asarray(settings["points"], dtype=np.float64).reshape(-1, 3), self.segments,
                           np.asarray(settings["radii"], dtype=np.float64).reshape(-1, 2), self.fillet, self.ends)
        features = tree.wall_features()
        if features.fused:
            radius = tree.fillet_radius_value()
            pairs = ", ".join(f"segments {i} and {j}" for i, j in features.fused)
            found.append(f"the fillet fuses tubes that share no point ({pairs}): the solid between them is "
                         f"{features.narrowest_solid * 1e3:.4g} mm thick at its thinnest, under the fillet's diameter "
                         f"{2.0 * radius * 1e3:.4g} mm, and the fillet's closing would fill it, joining their lumens, so "
                         f"write a smaller fillet (TubeTree(fillet=...) under "
                         f"{self.fillet * features.narrowest_solid / (2.0 * radius):.3g}) or a thicker wall between "
                         f"those tubes (their points further apart, or their radii smaller)")
        return found

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        """The lumen's: its points and radii in SI (flat), its segments (two indices a segment, as numbers) and its
        fillet's share, the shape ``tube_tree``'s parameters."""
        if region != "lumen":
            raise KeyError(region)
        return {"points": settings["points"], "segments": tuple(float(i) for pair in self.segments for i in pair),
                "radii": settings["radii"], "fillet": float(self.fillet)}

    @override
    def surface_names(self) -> Mapping[str, str]:
        """Its free ends (:meth:`end_names`), each the plane normal to its segment at its end, evaluated as the
        half-space the lumen lies in."""
        return {**self.surfaces, **{name: "half_space" for name, _ in self.end_names()}}

    @override
    def opening_names(self) -> Mapping[str, tuple[str, ...]]:
        """The lumen opens at its free ends (:meth:`end_names`), the root first: each an opening a flow's law may be
        attached at (row #336)."""
        return {"lumen": tuple(name for name, _ in self.end_names())}

    @override
    def surface_parameters(self, surface: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        """A free end's plane: normal to its segment at the end's point, its normal into the segment (the half-space
        the lumen lies in, ``x . n >= offset``)."""
        ends = dict(self.end_names())
        if surface not in ends:
            raise KeyError(surface)
        point = ends[surface]
        points = np.asarray(settings["points"], dtype=np.float64).reshape(-1, 3)
        start, end = next((a, b) for a, b in self.segments if point in (a, b))
        inward = points[end] - points[start] if point == start else points[start] - points[end]
        normal = inward / _norm(inward)
        return {"normal": (float(normal[0]), float(normal[1]), float(normal[2])),
                "offset": float(np.dot(normal, points[point]))}

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        """The hull query: the tubes' end balls' farthest reach along ``direction``, ``max(c . d + r |d|)`` (a round
        cone is the hull of its two end balls)."""
        d = jnp.asarray(direction)
        points = jnp.reshape(jnp.asarray(parameters["points"], dtype=d.dtype), (-1, 3))
        radii = jnp.reshape(jnp.asarray(parameters["radii"], dtype=d.dtype), (-1, 2))
        starts = np.asarray([a for a, _ in self.segments])
        stops = np.asarray([b for _, b in self.segments])
        centres = jnp.concatenate([points[starts], points[stops]])
        balls = jnp.concatenate([radii[:, 0], radii[:, 1]])
        return jnp.max(centres @ d + balls * jnp.sqrt(jnp.sum(d * d)))

    # -- its ends, named by geometry ------------------------------------------------------------------------------
    def _end_settings(self) -> tuple[object, ...]:
        """``ends`` as given, before its names are known to be strings."""
        return tuple(self.ends)

    def _free_ends(self) -> list[int]:
        """The points one segment end reaches, by index."""
        reached: dict[int, int] = {}
        for start, end in self.segments:
            reached[start] = reached.get(start, 0) + 1
            reached[end] = reached.get(end, 0) + 1
        return sorted(point for point, count in reached.items() if count == 1)

    def _rooted(self) -> bool:
        return self.segments[0][0] in self._free_ends()

    def end_names(self) -> tuple[tuple[str, int], ...]:
        """Its free ends, each ``(name, point)``: ``root`` at the first segment's start where that is free, then the
        others in the order of their points, named by ``ends`` or ``end_0``, ``end_1``, ..."""
        rooted = self._rooted()
        root = self.segments[0][0]
        others = [point for point in self._free_ends() if not (rooted and point == root)]
        names = self.ends or tuple(f"end_{k}" for k in range(len(others)))
        return ((ROOT, root),) * rooted + tuple(zip(names, others, strict=True))

    # -- its structure, fixed -------------------------------------------------------------------------------------
    def lengths(self) -> np.ndarray:
        points = self._points()
        return np.asarray([float(_norm(points[end] - points[start])) for start, end in self.segments])

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
        """The pairs of segments whose tubes meet: those sharing a point, and those within reach of each other (their
        walls within two fillet radii, and ``MEETING_MARGIN``) at the values the tree is made with."""
        margin = MEETING_MARGIN * float(np.max(self._radii()))
        apart, reach = self._approach(self.parameters(), np)
        shared = {pair for pair in self._pairs() if set(self.segments[pair[0]]) & set(self.segments[pair[1]])}
        near = {pair for pair, gap, within in zip(self._pairs(), apart, reach, strict=True) if gap < within + margin}
        return tuple(sorted(shared | near))

    def _pairs(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(len(self.segments)) for j in range(i + 1, len(self.segments))]

    def _approach(self, parameters: Parameters, xp: Any, pairs: Sequence[tuple[int, int]] | None = None
                  ) -> tuple[Any, Any]:
        """For each pair (every pair by default): the least distance between the two segments' axes, and their reach
        (each tube's largest radius and two fillet radii), in ``xp`` (NumPy or JAX)."""
        held = list(self._pairs()) if pairs is None else list(pairs)
        first = np.asarray([self.segments[i] for i, _ in held], dtype=np.int64).reshape(-1, 2)
        second = np.asarray([self.segments[j] for _, j in held], dtype=np.int64).reshape(-1, 2)
        points = xp.reshape(xp.asarray(parameters["points"]), (-1, 3))
        radii = xp.max(xp.reshape(xp.asarray(parameters["radii"]), (-1, 2)), axis=1)
        apart = _segments_apart(xp, points[first[:, 0]], points[first[:, 1]], points[second[:, 0]],
                                points[second[:, 1]])
        indices = np.asarray(held, dtype=np.int64).reshape(-1, 2)
        reach = radii[indices[:, 0]] + radii[indices[:, 1]] + 2.0 * self.fillet_radius(parameters, xp)
        return apart, reach

    def unplanned(self, parameters: Parameters) -> Any:
        """How many pairs of tubes that do not meet (:meth:`meets`) come within reach of each other at
        ``parameters``: where the lumen's check is blind and its distance not exact."""
        meeting = set(self.meets())
        held = [pair for pair in self._pairs() if pair not in meeting]
        if not held:
            return jnp.asarray(0)
        apart, reach = self._approach(parameters, jnp, held)
        return jnp.sum(apart < reach)

    # -- its values, parameters ---------------------------------------------------------------------------------------
    def parameters(self) -> dict[str, np.ndarray]:
        """Its values in SI by name, the leaves its centreline and lumen read: ``points`` ``(k, 3)``, ``radii``
        ``(n, 2)`` and ``fillet`` (the share)."""
        return {"points": self._points(), "radii": self._radii(), "fillet": np.asarray(self.fillet, dtype=np.float64)}

    def fillet_radius(self, parameters: Parameters, xp: Any = jnp) -> Any:
        """The fillet's radius (m): its share times the smallest radius a segment has at a junction; zero with no
        junction."""
        segments, ends = self._junction_ends()
        if not segments.size:
            return xp.asarray(0.0) * xp.asarray(parameters["fillet"])
        radii = xp.reshape(xp.asarray(parameters["radii"]), (-1, 2))
        held = radii[segments, ends]
        # The least by its index, its derivative the selected radius's (``mime.scene._union._lesser``'s reason).
        return xp.asarray(parameters["fillet"]) * held[xp.argmin(held)]

    def fillet_radius_value(self) -> float:
        return float(self.fillet_radius(self.parameters(), np))

    def lumen(self) -> ExactUnion:
        """Its lumen: the round cones' union, filleted, its exact signed distance a JAX function of
        :meth:`parameters`, written about the points' centroid; one union for equal trees (so one compiled
        program)."""
        return _lumen(self)

    @override
    def centreline(self, settings: Mapping[str, Magnitude] | None = None) -> Centreline:
        """Its centreline: a straight segment for each, its local radius linear between its ends' (the inscribed
        ball's), and a junction where two ends or more meet. Each segment's normal is held as made: across its
        tangent, carried round the corner from the first segment that ends where it starts (``mime.scene.
        transported``), so the frame is continuous through every junction a parent reaches. Its values are read from
        the parameters, the settings in SI; the structure from the tree, whatever ``settings`` holds."""
        del settings
        points = self._points()
        tangents: list[np.ndarray] = [(points[b] - points[a]) / _norm(points[b] - points[a]) for a, b in self.segments]
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
        moved = self._points() @ turn.T + np.asarray(offset, dtype=np.float64)
        return TubeTree.si(moved, self.segments, self._radii(), self.fillet, self.ends)

    # -- what the resolution report reads ------------------------------------------------------------------------------
    def wall_features(self) -> WallFeatures:
        """Its :class:`WallFeatures` at the values it is made with."""
        radii = self._radii()
        junctions = self.junctions
        fillet = self.fillet_radius_value()
        smallest = float(radii.min())
        if junctions:
            smallest = min(smallest, fillet)
        points = self._points()
        wedge = 180.0
        for junction in junctions:
            directions: list[np.ndarray] = []
            for segment, end in junction.ends:
                a, b = (points[i] for i in self.segments[segment])
                away = (b - a) if end == 0 else (a - b)
                directions.append(away / _norm(away))
            for i in range(len(directions)):
                for j in range(i + 1, len(directions)):
                    cosine = float(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
                    wedge = min(wedge, math.degrees(math.acos(cosine)))
        walls = {pair: self._solid_between(*pair) for pair in self._pairs()
                 if not set(self.segments[pair[0]]) & set(self.segments[pair[1]])}
        fused = tuple(pair for pair, wall in walls.items() if wall < 2.0 * fillet)
        return WallFeatures(smallest, 2.0 * float(radii.min()), min(walls.values(), default=math.inf), wedge, fused)

    def _solid_between(self, i: int, j: int) -> float:
        """The thinnest solid between tubes ``i`` and ``j`` (sharing no point): the least gap between their walls
        along the line joining two points of their axes (``APPROACH_POINTS`` each) whose middle lies outside every
        tube; infinite where no such gap is solid (the tubes overlap, or what lies between them is lumen)."""
        points = self._points()
        radii = self._radii()
        t = np.linspace(0.0, 1.0, APPROACH_POINTS)

        def along(k: int) -> tuple[np.ndarray, np.ndarray]:
            a, b = (points[m] for m in self.segments[k])
            return a[None] + t[:, None] * (b - a)[None], radii[k, 0] + t * (radii[k, 1] - radii[k, 0])

        (axis_i, radius_i), (axis_j, radius_j) = along(i), along(j)
        offset = axis_j[None] - axis_i[:, None]
        apart = _norm(offset)
        gap = apart - radius_i[:, None] - radius_j[None]
        share = (radius_i[:, None] + 0.5 * gap) / np.where(apart > 0.0, apart, 1.0)
        middle = axis_i[:, None] + offset * share[..., None]
        flat = middle.reshape(-1, 3)
        solid = np.ones(len(flat), dtype=bool)
        for k, (a, b) in enumerate(self.segments):
            solid &= _round_cone_distance(flat, points[a], points[b], radii[k, 0], radii[k, 1]) > 0.0
        held = gap.reshape(-1)[solid & (gap.reshape(-1) >= 0.0)]
        return float(held.min()) if held.size else math.inf


@functools.lru_cache(maxsize=64)
def _lumen(tree: TubeTree) -> ExactUnion:
    """``tree``'s lumen (:meth:`TubeTree.lumen`), made once for equal trees."""

    def points(parameters: Parameters) -> Any:
        return jnp.reshape(jnp.asarray(parameters["points"]), (-1, 3))

    def radii(parameters: Parameters) -> Any:
        return jnp.reshape(jnp.asarray(parameters["radii"]), (-1, 2))

    def local(parameters: Parameters) -> Any:
        held = points(parameters)
        return held - jnp.mean(held, axis=0)

    regions = tuple(
        RoundCone(start=lambda p, a=a: local(p)[a], end=lambda p, b=b: local(p)[b],
                  start_radius=lambda p, i=i: radii(p)[i, 0], end_radius=lambda p, i=i: radii(p)[i, 1])
        for i, (a, b) in enumerate(tree.segments))
    return ExactUnion(regions, tree.meets(), size=lambda p: jnp.max(radii(p)), fillet=tree.fillet_radius,
                      origin=lambda p: jnp.mean(points(p), axis=0), unplanned=tree.unplanned)


@functools.lru_cache(maxsize=64)
def _structural_lumen(segments: tuple[tuple[int, int], ...]) -> ExactUnion:
    """A tree's lumen from its structure alone (:class:`TubeTreeShape`'s): its tubes, those sharing a point held as
    meeting, its fillet and its other pairs within reach counted, every value read from the parameters."""
    def points(parameters: Parameters) -> Any:
        return jnp.reshape(jnp.asarray(parameters["points"]), (-1, 3))

    def radii(parameters: Parameters) -> Any:
        return jnp.reshape(jnp.asarray(parameters["radii"]), (-1, 2))

    def local(parameters: Parameters) -> Any:
        held = points(parameters)
        return held - jnp.mean(held, axis=0)

    reached: dict[int, list[tuple[int, int]]] = {}
    for index, (start, end) in enumerate(segments):
        reached.setdefault(start, []).append((index, 0))
        reached.setdefault(end, []).append((index, 1))
    at_junctions = [end for held in reached.values() if len(held) >= 2 for end in held]
    rows = np.asarray([k for k, _ in at_junctions], dtype=np.int64)
    columns = np.asarray([e for _, e in at_junctions], dtype=np.int64)
    pairs = [(i, j) for i in range(len(segments)) for j in range(i + 1, len(segments))]
    shared = tuple(pair for pair in pairs if set(segments[pair[0]]) & set(segments[pair[1]]))
    apart = [pair for pair in pairs if pair not in shared]

    def fillet(parameters: Parameters) -> Any:
        if not rows.size:
            return 0.0 * jnp.asarray(parameters["fillet"])
        held = radii(parameters)[rows, columns]
        return jnp.asarray(parameters["fillet"]) * held[jnp.argmin(held)]

    def unplanned(parameters: Parameters) -> Any:
        if not apart:
            return jnp.asarray(0)
        first = np.asarray([segments[i] for i, _ in apart], dtype=np.int64)
        second = np.asarray([segments[j] for _, j in apart], dtype=np.int64)
        held, widest = points(parameters), jnp.max(radii(parameters), axis=1)
        gap = _segments_apart(jnp, held[first[:, 0]], held[first[:, 1]], held[second[:, 0]], held[second[:, 1]])
        index = np.asarray(apart, dtype=np.int64)
        return jnp.sum(gap < widest[index[:, 0]] + widest[index[:, 1]] + 2.0 * fillet(parameters))

    regions = tuple(
        RoundCone(start=lambda p, a=a: local(p)[a], end=lambda p, b=b: local(p)[b],
                  start_radius=lambda p, i=i: radii(p)[i, 0], end_radius=lambda p, i=i: radii(p)[i, 1])
        for i, (a, b) in enumerate(segments))
    return ExactUnion(regions, shared, size=lambda p: jnp.max(radii(p)), fillet=fillet,
                      origin=lambda p: jnp.mean(points(p), axis=0), unplanned=unplanned)


def _segments_apart(xp: Any, a: Any, b: Any, c: Any, d: Any) -> Any:
    """The least distance between segments ``a``-``b`` and ``c``-``d`` (each ``(..., 3)``), in ``xp``: from an end of
    one to the other, or between interior points where the joining line is square to both (not parallel)."""

    def to_segment(p: Any, start: Any, end: Any) -> Any:
        span = end - start
        along = xp.clip(xp.sum((p - start) * span, axis=-1) / xp.sum(span * span, axis=-1), 0.0, 1.0)
        return xp.sqrt(xp.sum((p - start - along[..., None] * span) ** 2, axis=-1))

    ends = xp.minimum(xp.minimum(to_segment(a, c, d), to_segment(b, c, d)),
                      xp.minimum(to_segment(c, a, b), to_segment(d, a, b)))
    u, v, w = b - a, d - c, a - c
    uu, vv, uv = xp.sum(u * u, axis=-1), xp.sum(v * v, axis=-1), xp.sum(u * v, axis=-1)
    uw, vw = xp.sum(u * w, axis=-1), xp.sum(v * w, axis=-1)
    determinant = uu * vv - uv * uv
    crossing = determinant > 1e-12 * uu * vv
    safe = xp.where(crossing, determinant, 1.0)
    s, t = (uv * vw - vv * uw) / safe, (uu * vw - uv * uw) / safe
    inside = crossing & (s > 0.0) & (s < 1.0) & (t > 0.0) & (t < 1.0)
    interior = xp.sqrt(xp.sum((w + s[..., None] * u - t[..., None] * v) ** 2, axis=-1))
    return xp.where(inside, xp.minimum(ends, interior), ends)


def _round_cone_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray, ra: float, rb: float) -> np.ndarray:
    """A round cone's distance, in NumPy, near enough for the report: the distance from its segment less the radius at
    the nearest point's share of it (exact for a capsule)."""
    span = b - a
    along = np.clip((points - a) @ span / (span @ span), 0.0, 1.0)
    return _norm(points - (a + along[:, None] * span)) - (ra + along * (rb - ra))


def _segment(index: int, pair: tuple[int, int], normal: np.ndarray) -> CentrelineSegment:
    """Segment ``index`` of a tree from point ``pair[0]`` to ``pair[1]``, read from the tree's parameters: its
    tangent the unit chord, its normal ``normal`` held across it."""
    start_point, end_point = pair

    def ends(parameters: Parameters, dtype: Any) -> tuple[Any, Any]:
        points = jnp.reshape(jnp.asarray(parameters["points"], dtype=dtype), (-1, 3))
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
        radii = jnp.reshape(jnp.asarray(parameters["radii"], dtype=at.dtype), (-1, 2))
        return radii[index, 0] + (radii[index, 1] - radii[index, 0]) * at / length(parameters).astype(at.dtype)

    return CentrelineSegment(length=length, frame=frame, radius=radius)


TUBE_TREE = register_shape("tube_tree", TubeTreeShape())

__all__ = ["APPROACH_POINTS", "FILLET_SHARE", "MEETING_MARGIN", "ROOT", "TUBE_TREE", "TubeTree", "TubeTreeShape",
           "WallFeatures"]
