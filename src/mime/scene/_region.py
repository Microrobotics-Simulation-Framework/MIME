"""The region builder (F0's B6 and B7, decisions 1 to 5; round 5, items 11 and 14): a region cut from a vessel's
centreline by planes, its cuts continued by flow extensions and turned by bends to the faces a grid kind requires,
field-agnostic geometry (``mime.scene``): the vessel's own part between its cuts a tube tree, each opening's extension,
bend and run a ball swept along a tangent-continuous path (:func:`path_distance`), whose distance is exact.

**The cut.** A vessel (a straight tube, or a tube tree) is held as points joined by straight segments, a radius at each
segment's ends. ``around`` names a junction of its centreline (``junction_0``, ..., ``TubeTree.junction_names``) and ``reach`` the arc
length from it along every branch to the branch's cut, the plane normal to the branch there; with neither, the region
is the whole vessel, cut at its own free ends. A branch whose end lies within reach is cut at its end. **Each cut is
named after the one free end of the vessel beyond it** (``root``, ``end_0``, ...; a tube's ``start`` and ``end``), so a
law attached at a vessel's end binds the cut that leads to it whether the region reaches that end or not; a cut with
several ends beyond it (a tree dividing again outside the region) is refused, the network that reduces them being
F2c's.

**The upstream opening** is the cut that leads to the vessel's root (a tree's ``root``, a tube's ``start``): the role
the centreline gives it, never the law attached there (round 5, item 14). A tree with no root (its first segment's
start meets another segment) has no upstream end, and is refused.

**The extension and the bend** (decisions 3 and 4; T4). Each cut is continued straight along the branch's tangent at
the cut's radius for ``extension`` diameters (the section the cut has), then turned by a circular bend of ``bend``
diameters' radius onto the face axis a grid kind requires (``axis``: the nearer of its two faces, by the sign of the
tangent along it), a chain of chords no more than :data:`CHORD_DEGREES` apart, then run straight to that face's plane
and one diameter past it. Each plane lies one diameter past every bend's end on its face, at or past every other
cut's extension, and past every point of the vessel's own part along the axis, so it cuts straight runs only. A cut whose tangent lies across the axis cannot be turned to either
face by one bend and is refused; so are openings that all turn to one face, and two openings on one plane whose
lumens cross the line midway between them.

**What it reports** (round 5, item 11): each opening's plane, its centre there and its neighbours on that plane; each
opening's extension, bend (its angle and arc) and run, their length the series resistance a law at the face adds to
the vessel's at the cut (Poiseuille at the cut's radius, ``8 mu L / (pi r^4)``, :meth:`RegionOpening.series_resistance`;
R5-C-01: the law acts at the face, its numbers describe the vessel at the cut), and the Reynolds number to which an
extension develops a laminar inflow (an entrance length of about 0.06 Re diameters [recalled])."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.core import jnp

#: The angle between the points a bend's extent is bounded at (degrees).
CHORD_DEGREES = 7.5
#: The numbers of a path's row (``BuiltRegion.paths``).
PATH_ROW = 26
#: The laminar entrance length, in diameters per unit Reynolds number [recalled, B6 and B7, 4].
ENTRANCE_PER_REYNOLDS = 0.06
#: How far past each bend's end each face's plane lies at least, in the cut's diameters (a cut along the axis needs
#: none); and how far the runs reach past the plane, in the cut's diameters.
FACE_CLEARANCE = 1.0
PAST_FACE = 1.0


class RegionRefused(ValueError):
    """A region the builder cannot make: its kind (``cut``, a cut that cannot be made; ``face``, a cut that cannot reach
    a face, or openings that cannot share one; ``upstream``, a vessel with no upstream end), the message and the route.
    A field kind that builds regions refuses it under its own code."""

    def __init__(self, kind: str, message: str, action: str) -> None:
        super().__init__(message)
        self.kind, self.message, self.action = kind, message, action


@dataclass(frozen=True)
class RegionOpening:
    """One opening of a built region: the vessel end it leads to (``name``), the face it turned to (``face``: -1 the
    low plane along the axis, +1 the high one), where its run crosses that plane (``centre``, m), the cut's centre on
    the vessel and its radius (m), and the lengths of its extension, bend and run (m), the bend's angle (degrees)."""

    name: str
    face: int
    centre: tuple[float, float, float]
    cut: tuple[float, float, float]
    radius: float
    extension: float
    bend: float
    turn: float
    run: float
    upstream: bool

    @property
    def series_length(self) -> float:
        """The length from the cut to the face (m): the extension, the bend's arc and the run."""
        return self.extension + self.bend + self.run

    def series_resistance(self, viscosity: float) -> float:
        """The Poiseuille resistance (Pa s/m^3) of the length from the cut to the face at the cut's radius, which a law
        at the face adds in series to what it states of the vessel at the cut."""
        return 8.0 * viscosity * self.series_length / (math.pi * self.radius ** 4)

    def developed_reynolds(self) -> float:
        """The Reynolds number by the cut's diameter up to which its extension is a laminar entrance length."""
        return self.extension / (2.0 * self.radius) / ENTRANCE_PER_REYNOLDS


@dataclass(frozen=True)
class BuiltRegion:
    """A region cut from a vessel in the world's frame: the vessel's own part between its cuts as a tube tree (its
    points ``(k, 3)``, segments, radii ``(n, 2)`` and fillet share), its openings in the vessel's order of ends with
    each one's path to its face, and the two faces' planes along ``axis`` (m)."""

    points: np.ndarray
    segments: tuple[tuple[int, int], ...]
    radii: np.ndarray
    fillet: float
    openings: tuple[RegionOpening, ...]
    #: Each opening's path from its cut to past its face, a row of ``PATH_ROW`` numbers (``mime.nodes.flow.fvm.
    #: PathSolid`` reads them): the cut's centre and tangent, the extension's length; the bend's centre, its start
    #: direction from the centre, its normal, its radius and angle (zero where none); the run's start, direction and
    #: length; the radius.
    paths: np.ndarray
    faces: tuple[float, float]
    axis: tuple[float, float, float]

    def signed_distance(self, points: Any) -> Any:
        """The region's lumen's signed distance (m, negative inside) at ``points`` ``(N, 3)``: the least of the tree's
        (its union's plain minimum) and the paths'."""
        from mime.scene._tube_tree import TubeTree

        tree = TubeTree.si(self.points, self.segments, self.radii, self.fillet)
        held = jnp.asarray(points)
        return jnp.minimum(tree.lumen().minimum(tree.parameters(), held), path_distance(self.paths, held))

    def extent_tree(self) -> tuple[np.ndarray, tuple[tuple[int, int], ...], np.ndarray]:
        """The tree's points, segments and radii with each path's pieces added as straight segments between its cut,
        its bend's ends and middle, and its run's end: what a box holding the whole region bounds (the arc within its
        chords' sagitta, under a tenth of its radius for a bend to 90 degrees, which the box's margin covers)."""
        points = [*self.points]
        segments = [*self.segments]
        radii = [*self.radii]
        for row in self.paths:
            cut, extended = row[0:3], row[0:3] + row[6] * row[3:6]
            half = row[17] / 2.0
            middle = row[7:10] + row[16] * (row[10:13] * math.cos(half) + np.cross(row[13:16], row[10:13]) *
                                            math.sin(half))
            bent = (middle,) if row[17] > 0.0 else ()
            ends = (cut, extended, *bent, row[18:21], row[18:21] + row[24] * row[21:24])
            for start, end in zip(ends[:-1], ends[1:], strict=True):
                points += [start, end]
                segments.append((len(points) - 2, len(points) - 1))
                radii.append((row[25], row[25]))
        return np.asarray(points), tuple(segments), np.asarray(radii)

    def upstream(self) -> int:
        return next(k for k, opening in enumerate(self.openings) if opening.upstream)

    def neighbours(self, index: int) -> tuple[tuple[str, float], ...]:
        """The other openings on ``index``'s plane and the distance between their centres in its radii."""
        here = self.openings[index]
        return tuple((other.name, float(np.linalg.norm(np.subtract(other.centre, here.centre))) / here.radius)
                     for k, other in enumerate(self.openings) if k != index and other.face == here.face)

    def lines(self, viscosity: float | None = None) -> tuple[str, ...]:
        """What the build report shows of the region: each opening's plane and neighbours, its extension, bend and
        run, the series resistance they add (with ``viscosity``) and the Reynolds number its extension develops."""
        found: list[str] = []
        for k, opening in enumerate(self.openings):
            plane = "low" if opening.face < 0 else "high"
            near = self.neighbours(k)
            beside = "; ".join(f"{name} {distance:.3g} radii away" for name, distance in near) or "alone there"
            if near:
                beside += (" (the solid between their patches exchanges flow, opening_exchange_flagged watching it: a "
                           "stiffer interior, FVM(wall=Band(delta_in=0.01)); the ends further apart, "
                           "VesselRegion(extension_diameters=..., bend_diameters=...); or a geometry putting them on "
                           "separate planes)")
            role = "the upstream opening (by the centreline, no backflow term)" if opening.upstream else \
                "an outlet by the centreline (the backflow term on its half cell)"
            series = (f", a series resistance of {opening.series_resistance(viscosity):.4g} Pa s/m^3 (Poiseuille at "
                      f"the cut's radius) added to the law's at the face") if viscosity is not None else ""
            found.append(f"opening {opening.name}: {role}; on the {plane} plane, {beside}; from its cut "
                         f"{2.0 * opening.radius * 1e3:.4g} mm across: an extension of "
                         f"{opening.extension / (2.0 * opening.radius):.3g} diameters (a laminar entrance length up "
                         f"to Re {opening.developed_reynolds():.3g}), a bend of {opening.turn:.3g} degrees "
                         f"({opening.bend * 1e3:.4g} mm of arc) and a run of {opening.run * 1e3:.4g} mm{series}")
        return tuple(found)


@dataclass(frozen=True)
class Vessel:
    """A vessel as the builder reads it: points ``(k, 3)`` (m), segments, radii ``(n, 2)`` at each segment's start and
    end (m), the fillet share, and its free ends ``(name, point)``, the upstream one first."""

    points: np.ndarray
    segments: tuple[tuple[int, int], ...]
    radii: np.ndarray
    fillet: float
    ends: tuple[tuple[str, int], ...]


def _unit(v: np.ndarray) -> np.ndarray:
    found: np.ndarray = v / np.linalg.norm(v)
    return found


def _ends_at(segments: Sequence[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    """Each point's segment ends ``(segment, 0 start or 1 end)``."""
    held: dict[int, list[tuple[int, int]]] = {}
    for index, (start, end) in enumerate(segments):
        held.setdefault(start, []).append((index, 0))
        held.setdefault(end, []).append((index, 1))
    return held


def _beyond(vessel: Vessel, at: Mapping[int, list[tuple[int, int]]], segment: int, towards: int) -> list[int]:
    """The free ends reached from the point ``towards`` without passing back along ``segment``."""
    free = {point for _, point in vessel.ends}
    found: list[int] = []
    stack, seen = [(segment, towards)], {segment}
    while stack:
        _, point = stack.pop()
        if point in free:
            found.append(point)
            continue
        for other, _ in at.get(point, []):
            if other not in seen:
                seen.add(other)
                a, b = vessel.segments[other]
                stack.append((other, b if a == point else a))
    return found


def build_region(vessel: Vessel, *, around: int | None, reach: float | None, extension: float, bend: float,
                 axis: Sequence[float] = (0.0, 0.0, 1.0)) -> BuiltRegion:
    """The region of ``vessel`` around the junction at point ``around`` within arc length ``reach`` (m) of it (both
    ``None``: the whole vessel), each cut continued ``extension`` diameters and bent at ``bend`` diameters' radius onto
    ``axis`` (the module says how)."""
    a = _unit(np.asarray(axis, dtype=np.float64))
    points, radii, segments = vessel.points, vessel.radii, vessel.segments
    at = _ends_at(segments)
    names = {point: name for name, point in vessel.ends}
    upstream = vessel.ends[0][1]
    if (around is None) != (reach is None):
        raise RegionRefused("cut", "a region is cut around a junction within a reach of it, both given, "
                                               "or is the whole vessel, neither given",
                            "write VesselRegion(around='junction_0', reach=mm(3)), or VesselRegion() for the whole vessel")
    if around is not None and len(at.get(around, [])) < 2:
        junctions = ", ".join(f"j{p}" for p, held in sorted(at.items()) if len(held) >= 2) or "none"
        raise RegionRefused("cut", f"the vessel has no junction at point {around} (its junctions: "
                                               f"{junctions})", "write VesselRegion(around=<one of its junctions>, ...)")
    kept: list[tuple[np.ndarray, np.ndarray, float, float]] = []  # each kept piece: start, end, their radii
    cuts: list[tuple[str, np.ndarray, np.ndarray, float]] = []  # each cut: its end's name, centre, outward tangent, radius
    if around is None:
        for index, (start, end) in enumerate(segments):
            kept.append((points[start], points[end], float(radii[index, 0]), float(radii[index, 1])))
        for name, point in vessel.ends:
            index, side = at[point][0]
            other = segments[index][1 - side]
            cuts.append((name, points[point], _unit(points[point] - points[other]), float(radii[index, side])))
    else:
        assert reach is not None
        stack = [(index, around, 0.0) for index, _ in at[around]]
        seen: set[int] = set()
        while stack:
            index, start, travelled = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            a_end, b_end = segments[index]
            forward = a_end == start
            end = b_end if forward else a_end
            r_start, r_end = (float(radii[index, 0]), float(radii[index, 1])) if forward else \
                (float(radii[index, 1]), float(radii[index, 0]))
            gap: np.ndarray = points[end] - points[start]
            length = float(np.linalg.norm(gap))
            if travelled + length <= reach:
                kept.append((points[start], points[end], r_start, r_end))
                if end in names:
                    cuts.append((names[end], points[end], _unit(points[end] - points[start]), r_end))
                for other, _ in at[end]:
                    if other != index:
                        stack.append((other, end, travelled + length))
                continue
            share = (reach - travelled) / length
            centre = points[start] + share * (points[end] - points[start])
            radius = r_start + share * (r_end - r_start)
            kept.append((points[start], centre, r_start, radius))
            beyond = _beyond(vessel, at, index, end)
            if len(beyond) != 1:
                raise RegionRefused(
                    "cut", f"the cut {reach * 1e3:.4g} mm from j{around} along segment {index} has "
                                       f"{len(beyond)} of the vessel's ends beyond it "
                                       f"({', '.join(names[p] for p in beyond)}): an opening leads to one end, where "
                                       f"the law attached there binds", "reach past the next junction "
                                                                         "(VesselRegion(reach=...) larger), or reduce "
                                                                         "the tree beyond the cut to its own law "
                                                                         "(the build-made network comes later)")
            cuts.append((names[beyond[0]], centre, _unit(points[end] - points[start]), radius))
    order = {name: k for k, (name, _) in enumerate(vessel.ends)}
    cuts.sort(key=lambda cut: order[cut[0]])
    reached = {cut[0] for cut in cuts}
    if names[upstream] not in reached or len(reached) != len(cuts):
        raise RegionRefused("cut", "the region's cuts do not lead to the vessel's upstream end once each",
                            "write VesselRegion(around=..., reach=...) with a reach that keeps the root's branch")
    # Each cut's extension and bend, then the faces (past every run's start and every point of the vessel's own part
    # along the axis), then each run to its face.
    bounded: list[tuple[np.ndarray, float]] = [(p, 0.0) for piece in kept for p in piece[:2]]
    turned: list[tuple[str, np.ndarray, np.ndarray, float, float, float, float, int, np.ndarray]] = []
    for name, centre, tangent, radius in cuts:
        along = float(np.dot(tangent, a))
        if abs(along) < 1e-9:
            raise RegionRefused("face", f"the cut leading to {name} faces across the grid's axis {tuple(a)}"
                                                    f", so no bend of one turn reaches either face",
                                "move the cut (VesselRegion(reach=...)) or turn the vessel so that its branch is not "
                                "square to the axis")
        face = 1 if along > 0.0 else -1
        normal = face * a
        extended = centre + extension * 2.0 * radius * tangent
        angle = math.acos(min(1.0, max(-1.0, float(np.dot(tangent, normal)))))
        curve = bend * 2.0 * radius
        row = np.zeros(PATH_ROW)
        row[0:3], row[3:6], row[6] = centre, tangent, extension * 2.0 * radius
        bent, arc = extended, 0.0
        if angle > 1e-9:
            across = normal - float(np.dot(normal, tangent)) * tangent
            across = across / np.linalg.norm(across)
            pivot = extended + curve * across
            for k in range(1, max(1, math.ceil(math.degrees(angle) / CHORD_DEGREES)) + 1):
                phi = angle * k / max(1, math.ceil(math.degrees(angle) / CHORD_DEGREES))
                bounded.append((pivot + curve * (-across * math.cos(phi) + tangent * math.sin(phi)), radius))
            bent = pivot + curve * (-across * math.cos(angle) + tangent * math.sin(angle))
            arc = curve * angle
            row[7:10], row[10:13], row[13:16] = pivot, -across, np.cross(-across, tangent)
            row[16], row[17] = curve, angle
        row[21:24], row[25] = normal, radius
        turned.append((name, centre, bent, radius, extension * 2.0 * radius, arc, math.degrees(angle), face, row))
    faces_held = {t[7] for t in turned}
    if faces_held != {-1, 1}:
        raise RegionRefused("face", f"every opening of the region turns to the {'high' if 1 in faces_held else 'low'} "
                                                f"face of the grid's axis; the grid kind opens both faces of one axis",
                            "cut the region so that its upstream cut and its outlets face opposite ways along the axis, "
                            "or turn the vessel")
    def run_start(t: tuple[str, np.ndarray, np.ndarray, float, float, float, float, int, np.ndarray]) -> float:
        """Where along the axis a cut's run may meet its face: its bend's end and a diameter past it, or the end of its
        straight extension where it needs no bend."""
        return float(np.dot(t[2], a)) + (t[7] * FACE_CLEARANCE * 2.0 * t[3] if t[5] > 0.0 else 0.0)

    high = max(max(float(np.dot(p, a)) + r for p, r in bounded), max(run_start(t) for t in turned if t[7] == 1))
    low = min(min(float(np.dot(p, a)) - r for p, r in bounded), min(run_start(t) for t in turned if t[7] == -1))
    openings: list[RegionOpening] = []
    rows: list[np.ndarray] = []
    for name, centre, bent, radius, extended, arc, turn, face, row in turned:
        plane = high if face > 0 else low
        run = abs(plane - float(np.dot(bent, a)))
        on_plane = bent + run * face * a
        row[18:21], row[24] = bent, run + PAST_FACE * 2.0 * radius
        rows.append(row)
        openings.append(RegionOpening(name, face, (float(on_plane[0]), float(on_plane[1]), float(on_plane[2])),
                                      (float(centre[0]), float(centre[1]), float(centre[2])), radius, extended, arc,
                                      turn, run, name == names[upstream]))
    for i, first in enumerate(openings):
        for second in openings[i + 1:]:
            if first.face == second.face:
                apart = float(np.linalg.norm(np.subtract(first.centre, second.centre)))
                if apart <= 2.0 * max(first.radius, second.radius):
                    raise RegionRefused(
                        "face", f"the openings {first.name} and {second.name} share a plane with their "
                                            f"centres {apart * 1e3:.4g} mm apart: a lumen crosses the line midway "
                                            f"between them, where the plane is split between its openings",
                        "lengthen the extensions (VesselRegion(extension_diameters=...)) so the branches part further before "
                        "they turn, or cut the region nearer the junction (reach=...)")
    return _as_tree(kept, vessel, tuple(openings), np.asarray(rows), (low, high), a)


def _as_tree(pieces: Sequence[tuple[np.ndarray, np.ndarray, float, float]], vessel: Vessel,
             openings: tuple[RegionOpening, ...], paths: np.ndarray, faces: tuple[float, float],
             axis: np.ndarray) -> BuiltRegion:
    """The vessel's own part as a tube tree: points shared where pieces meet (to a part in 1e9 of the region's size),
    and the fillet share that keeps the vessel's own fillet radius; each opening's path beside it."""
    scale = max(float(np.ptp(np.asarray([p for piece in pieces for p in piece[:2]]), axis=0).max()), 1e-300)
    held: list[np.ndarray] = []

    def index_of(point: np.ndarray) -> int:
        for k, other in enumerate(held):
            if float(np.linalg.norm(other - point)) <= 1e-9 * scale:
                return k
        held.append(np.asarray(point, dtype=np.float64))
        return len(held) - 1

    segments: list[tuple[int, int]] = []
    radii: list[tuple[float, float]] = []
    for start, end, r_start, r_end in pieces:
        if float(np.linalg.norm(end - start)) <= 1e-9 * scale:
            continue
        segments.append((index_of(start), index_of(end)))
        radii.append((r_start, r_end))
    segments, radii = _merged(held, segments, radii)
    used = sorted({k for pair in segments for k in pair})
    renumbered = {old: new for new, old in enumerate(used)}
    held = [held[k] for k in used]
    segments = [(renumbered[a], renumbered[b]) for a, b in segments]
    given = _junction_least(vessel.segments, vessel.radii)
    made = _junction_least(segments, np.asarray(radii))
    share = vessel.fillet if given is None or made is None else vessel.fillet * given / made
    return BuiltRegion(np.asarray(held), tuple(segments), np.asarray(radii), float(share), openings, paths, faces,
                       (float(axis[0]), float(axis[1]), float(axis[2])))


def _merged(points: Sequence[np.ndarray], segments: list[tuple[int, int]], radii: list[tuple[float, float]]
            ) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    """``segments`` with every two that meet alone at a point, along one line and of one radius throughout, made one
    (a cut's extension and run along its branch): the same lumen, fewer seams for the union to resolve. Points left on
    no segment are kept out by the caller's renumbering."""
    while True:
        at = _ends_at(segments)
        for point, held in at.items():
            if len(held) != 2:
                continue
            (i, side_i), (j, side_j) = held
            far_i, far_j = segments[i][1 - side_i], segments[j][1 - side_j]
            a = _unit(points[point] - points[far_i])
            b = _unit(points[far_j] - points[point])
            widths = (*radii[i], *radii[j])
            if float(np.dot(a, b)) < 1.0 - 1e-12 or max(widths) - min(widths) > 1e-12 * max(widths):
                continue
            segments = [(far_i, far_j) if k == i else segments[k] for k in range(len(segments)) if k != j]
            radii = [(widths[0], widths[0]) if k == i else radii[k] for k in range(len(radii)) if k != j]
            break
        else:
            return segments, radii


def _junction_least(segments: Sequence[tuple[int, int]], radii: np.ndarray) -> float | None:
    """The smallest radius a segment has at a point two segment ends or more meet at; ``None`` where none meet."""
    at = _ends_at(segments)
    found = [float(radii[index, side]) for held in at.values() if len(held) >= 2 for index, side in held]
    return min(found) if found else None


def path_distance(paths: Any, points: Any) -> Any:
    """The signed distance (m, positive outside) from ``points`` ``(..., 3)`` to the lumens the paths ``(openings,
    PATH_ROW)`` sweep: for each, the distance to its extension, its bend and its run less its radius (exact where the
    bend's radius exceeds the lumen's, the path's tangent continuous), the least over the paths. In JAX, so a reader's
    step traces it with the rows a leaf."""
    rows = jnp.asarray(paths)
    held = jnp.asarray(points, dtype=rows.dtype)
    nearest: Any = None
    for k in range(rows.shape[0]):
        row = rows[k]
        straight = jnp.minimum(_to_segment(held, row[0:3], row[3:6], row[6]),
                               _to_segment(held, row[18:21], row[21:24], row[24]))
        centre, start, normal, curve, angle = row[7:10], row[10:13], row[13:16], row[16], row[17]
        offset = held - centre
        a = jnp.sum(offset * start, axis=-1)
        b = jnp.sum(offset * jnp.cross(normal, start), axis=-1)
        c = jnp.sum(offset * normal, axis=-1)
        within = (b >= 0.0) & (a * jnp.sin(angle) - b * jnp.cos(angle) >= 0.0) & (angle > 0.0)
        square = a * a + b * b
        planar = jnp.where(square > 0.0, jnp.sqrt(jnp.where(square > 0.0, square, 1.0)), 0.0)
        arc = (planar - curve) ** 2 + c * c
        arc = jnp.where(arc > 0.0, jnp.sqrt(jnp.where(arc > 0.0, arc, 1.0)), 0.0)
        found = jnp.where(within, jnp.minimum(straight, arc), straight) - row[25]
        nearest = found if nearest is None else jnp.minimum(nearest, found)
    return nearest


def _to_segment(points: Any, start: Any, direction: Any, length: Any) -> Any:
    """The distance from ``points`` to the segment from ``start`` along the unit ``direction`` for ``length``."""
    offset = points - start
    along = jnp.clip(jnp.sum(offset * direction, axis=-1), 0.0, length)
    gap = offset - along[..., None] * direction
    square = jnp.sum(gap * gap, axis=-1)
    return jnp.where(square > 0.0, jnp.sqrt(jnp.where(square > 0.0, square, 1.0)), 0.0)


def moved_paths(paths: np.ndarray, by: Sequence[float]) -> np.ndarray:
    """``paths`` with their points (the cut's centre, the bend's centre and the run's start) moved by ``by`` (m)."""
    moved = np.array(paths, dtype=np.float64)
    for start in (0, 7, 18):
        moved[:, start:start + 3] += np.asarray(by, dtype=np.float64)
    return moved


def tube_vessel(start: Sequence[float] | np.ndarray, end: Sequence[float] | np.ndarray, radius: float, fillet: float,
                names: tuple[str, str] = ("start", "end")) -> Vessel:
    """A straight tube from ``start`` to ``end`` (m) of ``radius`` (m), its ends named ``start`` (upstream) and
    ``end``."""
    return Vessel(np.asarray([start, end], dtype=np.float64), ((0, 1),), np.asarray([[radius, radius]]), fillet,
                  ((names[0], 0), (names[1], 1)))


__all__ = ["CHORD_DEGREES", "ENTRANCE_PER_REYNOLDS", "PATH_ROW", "BuiltRegion", "RegionOpening", "RegionRefused", "Vessel",
           "moved_paths", "path_distance",
           "build_region", "tube_vessel"]
