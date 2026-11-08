"""The centreline harness: does a kind's centreline keep its contract (``mime.registry.Centreline``)?

On each segment at points through its arc: the tangent is a unit vector and the derivative of the point in arc length
(by forward-mode autodiff, so to rounding: a curve given by another parameter than arc length fails it); the frame's
normal is a unit vector across the tangent, and turns only as the tangent turns (its derivative along the tangent, the
rotation-minimising frame's property); a local radius and the filament's radius, where given, are above zero; and the
ends a junction names meet. Each is judged at ``CENTRELINE_ROUNDING_ULPS`` units in the last place of float64 on the
scale it came from: a unit vector's on one, a point's on the segment's reach from the kind's origin, a derivative's on
its own scale times the curvature's (``1 + |t'| scale``)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import jax
import numpy as np

from mime.core import jnp
from mime.registry import Centreline, Geometry, implements
from mime.units import Magnitude

#: The harness's tolerance in units in the last place of float64, on each quantity's own scale (the module's docstring).
CENTRELINE_ROUNDING_ULPS = 64.0
#: Points a segment is read at, through its arc and short of its ends.
CENTRELINE_POINTS = 17
_EPSILON = float(np.finfo(np.float64).eps)
_jacfwd: Any = cast(Any, jax).jacfwd
_vmap: Any = cast(Any, jax).vmap


def _rows(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _norms(rows: np.ndarray) -> np.ndarray:
    return np.asarray(np.sqrt(np.sum(rows * rows, axis=-1)), dtype=np.float64)


def _largest(rows: np.ndarray) -> np.ndarray:
    return np.asarray(np.max(np.abs(rows), axis=-1), dtype=np.float64)


def _reads(frame: Any, parameters: Mapping[str, Any], s: Any) -> tuple[Any, Any, Any, Any]:
    """The segment's frame at ``s``, and the derivatives in arc length of its point, its normal and its tangent."""

    def point(at: Any) -> Any:
        return frame(parameters, at[None])[0][0]

    def normal(at: Any) -> Any:
        return frame(parameters, at[None])[2][0]

    def tangent(at: Any) -> Any:
        return frame(parameters, at[None])[1][0]

    held = frame(parameters, s)
    return (jnp.stack([held[0], held[1], held[2]]), _vmap(_jacfwd(point))(s), _vmap(_jacfwd(normal))(s),
            _vmap(_jacfwd(tangent))(s))


#: ``_reads`` compiled once per segment function (a static argument: a kind whose segment functions are module-level
#: functions, as the helix's, compiles once for every geometry drawn).
_read: Any = cast(Any, jax).jit(_reads, static_argnums=0)


def centreline_problems(kind: Geometry, settings: Mapping[str, Magnitude],
                        centreline: Centreline | None = None) -> list[str]:
    """Every way ``kind``'s centreline (or ``centreline``, given), at its ``settings`` in SI, fails its contract."""
    if centreline is None:
        if not implements(kind, "centreline"):
            return [f"{kind.label()} declares no centreline"]
        centreline = kind.centreline(settings)
    parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                  for name, value in settings.items()}
    label = kind.label()
    found: list[str] = []
    ends: dict[tuple[int, int], np.ndarray] = {}
    for index, segment in enumerate(centreline.segments):
        length = float(segment.length(parameters))
        if not (math.isfinite(length) and length > 0.0):
            found.append(f"{label}: segment {index}'s length is {length!r} m, not a length above zero")
            continue
        s = jnp.asarray(length * (np.arange(CENTRELINE_POINTS) + 0.5) / CENTRELINE_POINTS, dtype=jnp.float64)
        read = cast(tuple[Any, Any, Any, Any], _read(segment.frame, parameters, s))
        frame: np.ndarray = np.asarray(read[0], dtype=np.float64)
        derivative: np.ndarray = np.asarray(read[1], dtype=np.float64)
        turning: np.ndarray = np.asarray(read[2], dtype=np.float64)
        bend: np.ndarray = np.asarray(read[3], dtype=np.float64)
        points, tangents, normals = _rows(frame[0]), _rows(frame[1]), _rows(frame[2])
        scale = max(1.0, float(np.max(_norms(points))) / length) * length
        unit = CENTRELINE_ROUNDING_ULPS * _EPSILON
        curved = unit * (1.0 + _norms(bend) * scale)
        across = turning - np.sum(turning * tangents, axis=1)[:, None] * tangents
        checks: tuple[tuple[str, np.ndarray, np.ndarray], ...] = (
            ("its tangent is no unit vector", np.abs(_norms(tangents) - 1.0), np.full(len(s), unit)),
            ("its tangent is not its point's derivative in arc length", _largest(derivative - tangents), curved),
            ("its normal is no unit vector", np.abs(_norms(normals) - 1.0), np.full(len(s), unit)),
            ("its normal is not across its tangent", np.abs(np.sum(normals * tangents, axis=1)),
             np.full(len(s), unit)),
            ("its normal turns about its tangent (not the rotation-minimising frame)", _largest(across),
             curved / scale),
        )
        for what, apart, allowed in checks:
            if not bool(np.all(apart <= allowed)):
                worst = int(np.argmax(apart / allowed))
                found.append(f"{label}: segment {index}: {what}, by {float(apart[worst]):.3g} at s = "
                             f"{float(s[worst]):.6g} m, beyond {float(allowed[worst]):.3g}")
        if segment.radius is not None:
            radii = np.asarray(segment.radius(parameters, s), dtype=np.float64)
            if not np.all(np.isfinite(radii) & (radii > 0.0)):
                found.append(f"{label}: segment {index}'s local radius is not above zero along it: {radii.tolist()}")
        if segment.filament_radius is not None:
            radius = float(segment.filament_radius(parameters))
            if not (math.isfinite(radius) and radius > 0.0):
                found.append(f"{label}: segment {index}'s filament radius is {radius!r} m, not a length above zero")
        tips = np.asarray(segment.frame(parameters, jnp.asarray([0.0, length], dtype=jnp.float64))[0], dtype=np.float64)
        ends[(index, 0)], ends[(index, 1)] = tips[0], tips[1]
    for junction in centreline.junctions:
        held = [ends[end] for end in junction.ends if end in ends]
        if held:
            apart = max(float(np.linalg.norm(p - held[0])) for p in held)
            reach = max(float(np.linalg.norm(p)) for p in held)
            if not apart <= CENTRELINE_ROUNDING_ULPS * _EPSILON * max(reach, 1e-300):
                found.append(f"{label}: junction {junction.name!r}'s ends lie {apart:.3g} m apart")
    return found


__all__ = ["CENTRELINE_POINTS", "CENTRELINE_ROUNDING_ULPS", "centreline_problems"]
