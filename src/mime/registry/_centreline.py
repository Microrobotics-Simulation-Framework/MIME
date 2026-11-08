"""A geometry kind's centreline: a curve by arc length with its frame, shared by a slender body and a vessel.

**The contract.** A :class:`Centreline` is a tree of :class:`CentrelineSegment` joined at named :class:`Junction`
(one segment is the common case: a slender robot's axis, a straight tube). A segment gives, as JAX functions of the kind's
parameters (its settings in SI, numbers or traced arrays, as ``Geometry.samples`` takes them), its length and, at arc
lengths ``s`` along it, the point, the unit tangent (the point's derivative in ``s``) and a unit normal across it,
**the rotation-minimising frame's** (Bishop's parallel transport): the normal turns about the tangent only as the
tangent turns, ``dn/ds = -(dt/ds . n) t``, from the normal the segment declares at its start. That frame is defined on a
straight segment, where Frenet's is not, and it does not twist along a curve, so a camera or a field path carried
along it does not spin. Optional: the local radius along it (a vessel's lumen, a filament's section), the one radius a
slender law's coefficients take (``filament_radius``) and the arc length of one period of a periodic segment
(``wavelength``, a helix's turn). Points are in the frame the kind's samples are in."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from mime.core import jnp

#: A kind's parameters: its settings in SI by name, numbers or traced arrays.
type Parameters = Mapping[str, Any]
#: The fewest points a consumer reads a segment at, whatever the spacing: two, so that a straight segment's points
#: span it.
LEAST_POINTS = 2


class CentrelineFrame(NamedTuple):
    """A segment at arc lengths ``s`` ``(n,)``: the points ``(n, 3)``, the unit tangents ``(n, 3)`` and the
    rotation-minimising frame's unit normals ``(n, 3)``."""

    point: Any
    tangent: Any
    normal: Any


@dataclass(frozen=True, eq=False)
class CentrelineSegment:
    """One arc of a centreline, each field a JAX function of the kind's parameters: its ``length`` (m), its
    :class:`CentrelineFrame` at arc lengths ``s`` from its start (``frame(parameters, s)``), and, where the segment
    has them, its local ``radius(parameters, s)`` (m, ``(n,)``), the ``filament_radius(parameters)`` a slender law's
    coefficients take (one number, m: a filament's section), and its ``wavelength(parameters)`` (m of arc length per
    period, a helix's turn; ``None`` on a segment that is not periodic)."""

    length: Callable[[Parameters], Any]
    frame: Callable[[Parameters, Any], CentrelineFrame]
    radius: Callable[[Parameters, Any], Any] | None = None
    filament_radius: Callable[[Parameters], Any] | None = None
    wavelength: Callable[[Parameters], Any] | None = None


@dataclass(frozen=True)
class Junction:
    """A named point where segments meet: ``ends`` are ``(segment, end)`` pairs, the segment's index in its
    centreline and ``0`` for its start or ``1`` for its end."""

    name: str
    ends: tuple[tuple[int, int], ...]


@dataclass(frozen=True, eq=False)
class Centreline:
    """A tree of :class:`CentrelineSegment` joined at :class:`Junction`: its structure is fixed when the kind gives
    it (its segments, which ends meet), its values functions of the kind's parameters."""

    segments: tuple[CentrelineSegment, ...]
    junctions: tuple[Junction, ...] = ()

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a centreline has one segment or more")
        names = [junction.name for junction in self.junctions]
        if len(set(names)) != len(names):
            raise ValueError(f"a centreline's junctions are named once each: {names}")
        for junction in self.junctions:
            if len(junction.ends) < 2:
                raise ValueError(f"junction {junction.name!r} joins two segment ends or more: {junction.ends}")
            for segment, end in junction.ends:
                if not (0 <= segment < len(self.segments)) or end not in (0, 1):
                    raise ValueError(f"junction {junction.name!r} names ({segment}, {end}): a segment's index below "
                                     f"{len(self.segments)} and its end, 0 or 1")

    def junction(self, name: str) -> Junction:
        for junction in self.junctions:
            if junction.name == name:
                return junction
        raise KeyError(f"no junction {name!r}; the centreline's junctions are {[j.name for j in self.junctions]}")


class CentrelinePoints(NamedTuple):
    """A centreline read at points: each point ``(n, 3)``, its unit tangent and normal ``(n, 3)``, the arc length it
    stands for ``(n,)`` (its segment's length over its count), its local radius where the segment gives one (else
    ``None``), and the index of its segment ``(n,)`` (NumPy, fixed)."""

    point: Any
    tangent: Any
    normal: Any
    arc: Any
    radius: Any
    segment: np.ndarray


def centreline_counts(centreline: Centreline, parameters: Parameters, spacing: float) -> tuple[int, ...]:
    """The count rule: each segment read at ``ceil(length / spacing)`` points (at least :data:`LEAST_POINTS`), its
    length evaluated at the value the experiment is built with; the count is held for the build, so a run never moves
    it, and a declared range of the geometry (when one is declared) fixes it instead, here."""
    found: list[int] = []
    for segment in centreline.segments:
        length = float(segment.length(parameters))
        if not (math.isfinite(length) and length > 0.0):
            raise ValueError(f"a centreline's segment has length {length!r} m, not a length above zero")
        found.append(max(LEAST_POINTS, math.ceil(length / spacing)))
    return tuple(found)


def centreline_points(centreline: Centreline, parameters: Parameters, counts: Sequence[int], dtype: Any) -> \
        CentrelinePoints:
    """``centreline`` read at ``counts`` points a segment, each at the middle of its share of the segment's length
    (``(i + 1/2) L / N``), from the traced length: a JAX function of the parameters at fixed counts."""
    if len(counts) != len(centreline.segments):
        raise ValueError(f"one count a segment: {len(centreline.segments)} segments, counts {tuple(counts)}")
    points: list[Any] = []
    tangents: list[Any] = []
    normals: list[Any] = []
    arcs: list[Any] = []
    radii: list[Any] = []
    owner: list[np.ndarray] = []
    every = True
    for index, (segment, count) in enumerate(zip(centreline.segments, counts, strict=True)):
        length = jnp.asarray(segment.length(parameters), dtype=dtype)
        s = (jnp.arange(count, dtype=dtype) + 0.5) * length / count
        frame = segment.frame(parameters, s)
        points.append(jnp.asarray(frame.point, dtype=dtype))
        tangents.append(jnp.asarray(frame.tangent, dtype=dtype))
        normals.append(jnp.asarray(frame.normal, dtype=dtype))
        arcs.append(jnp.broadcast_to(length / count, (count,)))
        if segment.radius is None:
            every = False
        else:
            radii.append(jnp.asarray(segment.radius(parameters, s), dtype=dtype) * jnp.ones((count,), dtype=dtype))
        owner.append(np.full(count, index, dtype=np.int32))
    return CentrelinePoints(jnp.concatenate(points), jnp.concatenate(tangents), jnp.concatenate(normals),
                            jnp.concatenate(arcs), jnp.concatenate(radii) if every else None, np.concatenate(owner))


__all__ = ["LEAST_POINTS", "Centreline", "CentrelineFrame", "CentrelinePoints", "CentrelineSegment", "Junction",
           "Parameters", "centreline_counts", "centreline_points"]
