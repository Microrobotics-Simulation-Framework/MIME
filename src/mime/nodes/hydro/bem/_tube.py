"""The straight circular tube's correction for the boundary-element method: the image system of a Stokeslet inside an
infinite tube with a no-slip wall (Liron & Shahar 1978), read from a table of it through ``mime.core.tables``.

**The table** is the 2026-05-21 one the v0.2 line built by the direct Fourier-Bessel code, at a tube radius ``R_cyl``
and viscosity 1: for a source at ``(rho_s, 0, 0)`` and a target at ``(rho_t, dphi, dz)`` (cylindrical), the image's
``3 x 3`` block, its rows the velocity's cylindrical components at the target (radial, azimuthal, axial) and its
columns the force's at the source, on knots ``rho_t`` and ``rho_s`` (the first two axes, in that order), ``dphi`` on
``[0, pi]`` and ``dz`` on ``[0, 5 R_cyl]``. The image system scales with the tube: in a tube of radius ``R`` and
viscosity ``mu`` it is ``G(x / R) / (R mu)``, with ``G`` the image of the tube of radius 1 at viscosity 1, so the
table is read as that one, its lengths over ``R_cyl`` and its values times ``R_cyl``, whichever radius it was built
at.

**Its parity.** Each component is even or odd in ``dphi`` and in ``dz`` (``ODD_IN_ANGLE``, ``ODD_ALONG``): the table
holds half of each period, and a read at ``|dphi|`` (folded into ``[0, pi]``) and ``|dz|`` takes the sign of the odd
components from the sign of ``sin(dphi)`` and of ``dz``. So that the read is continuously differentiable across
``dphi = 0``, ``dphi = pi`` and ``dz = 0``, as the image is, the angle and the axial offset each gain two knots
mirrored past each end that is a symmetry (their values the parity's), so that the cubic's slope at the end is the
symmetric one (zero for an even component), and each odd component is set to zero exactly where its parity makes it
zero (it is zero there to the table's rounding).

**The read.** The body's samples, in the tube's frame (its axis along z through its centre, the frame across the axis
held as built) and over its radius: each sample's radius once, as a column and a row (the pairwise form), the angle
between each pair from the cosines and sines of their own angles, and their axial offset. The block is turned from
each sample's cylindrical components into the world's and scaled by ``1 / (R mu)``. Where a read lies outside the
table's readable range (a sample within 0.02 R of the axis or the wall, or a pair further apart along the axis than
the table reaches) its value is NaN and it is not accepted: the node publishes whether every read was."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from jax import Array

from mime.core import jnp, switch
from mime.core.tables import Axis, Scheme, Table, TableArrays, evaluate, query_range_problems
from mime.nodes.hydro.bem._solve import PRECISION
from mime.nodes.hydro.bem._walls import Corrected, WallCorrection

#: The components odd in the angle between the source and the target, and odd in their axial offset: rows the
#: velocity's cylindrical component at the target, columns the force's at the source (radial, azimuthal, axial).
ODD_IN_ANGLE = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
ODD_ALONG = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool)
#: The owners of the read's switches, which the tube's wall kind declares: locating each query among the table's
#: knots, the angle's fold (its absolute value and the odd components' sign, at 0 and at pi) and the axial offset's.
TABLE = "tube.table"
ANGLE = "tube.angle"
ALONG = "tube.axial"
#: Knots mirrored past each symmetric end of the angle's and the axial offset's axes.
MIRRORED = 2


def _parity(odd: np.ndarray) -> np.ndarray:
    """``-1`` for each odd component and ``+1`` for each even one, as a ``3 x 3`` block."""
    return np.where(odd, -1.0, 1.0)


def unit_tube_table(radius: float, rho: np.ndarray, angle: np.ndarray, along: np.ndarray, data: np.ndarray, *,
                    scheme: Scheme | tuple[Scheme, ...] = "cubic", float32_values: bool = False,
                    readable_rho: tuple[float, float] | None = None) -> Table:
    """The table of the image of the tube of radius 1 at viscosity 1, from one built at ``radius`` (``rho``,
    ``angle`` from 0, to pi for a whole table, ``along`` from 0, ``data`` of shape ``(rho_t, rho_s, angle, along, 3,
    3)``), with the angle mirrored past 0 (and past pi where it reaches it) and the axial offset past 0;
    ``readable_rho`` narrows the radii's readable range, in the tube's radius. A slice of a table (a block of its
    knots from the first angle and the first offset) is read the same way."""
    if not (angle[0] == 0.0 and along[0] == 0.0):
        raise ValueError("the table's angle and its axial offset run from 0")
    wraps = math.isclose(float(angle[-1]), math.pi, rel_tol=1e-15)
    by_angle, by_along = _parity(ODD_IN_ANGLE), _parity(ODD_ALONG)
    count, steps, past = len(angle), len(along), MIRRORED if wraps else 0
    values = np.empty((len(rho), len(rho), MIRRORED + count + past, MIRRORED + steps, 3, 3))
    inside = values[:, :, MIRRORED:MIRRORED + count, MIRRORED:]
    np.multiply(data, radius, out=inside)
    inside[:, :, 0][..., ODD_IN_ANGLE] = 0.0
    inside[:, :, :, 0][..., ODD_ALONG] = 0.0
    values[:, :, :MIRRORED, MIRRORED:] = inside[:, :, MIRRORED:0:-1] * by_angle
    ends = [-angle[MIRRORED:0:-1], angle]
    if wraps:
        inside[:, :, -1][..., ODD_IN_ANGLE] = 0.0
        values[:, :, MIRRORED + count:, MIRRORED:] = inside[:, :, -2:-2 - MIRRORED:-1] * by_angle
        ends.append(2.0 * math.pi - angle[-2:-2 - MIRRORED:-1])
    values[:, :, :, :MIRRORED] = values[:, :, :, 2 * MIRRORED:MIRRORED:-1] * by_along
    angles = np.concatenate(ends)
    offsets = np.concatenate([-along[MIRRORED:0:-1], along]) / radius
    radii = tuple(float(r) for r in rho / radius)
    axes = (Axis("target radius over the tube's", "1", radii, source_range=readable_rho),
            Axis("source radius over the tube's", "1", radii, source_range=readable_rho),
            Axis("angle from the source to the target", "rad", tuple(float(a) for a in angles)),
            Axis("axial offset over the tube's radius", "1", tuple(float(z) for z in offsets)))
    return Table(axes, values, scheme=scheme, float32_values=float32_values)


@cache
def tube_table_from(path: str, *, scheme: Scheme | tuple[Scheme, ...] = "cubic", float32_values: bool = False,
                    readable_rho: tuple[float, float] | None = None) -> Table:
    """The unit tube's table from a table file of the 2026-05-21 form (``R_cyl``, ``rho_grid``, ``dphi_grid``,
    ``dz_grid``, ``data``), read once per process for each file and setting."""
    with np.load(Path(path)) as held:
        return unit_tube_table(float(held["R_cyl"]), held["rho_grid"], held["dphi_grid"], held["dz_grid"],
                               held["data"], scheme=scheme, float32_values=float32_values, readable_rho=readable_rho)


def frame_across(axis: Any) -> tuple[float, ...]:
    """The tube's frame as rows, the last its axis: a frame across an axis cannot be continuous in every axis, so it
    is taken here, as built, from the coordinate direction least along the axis."""
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    first = np.cross(unit, np.eye(3)[int(np.argmin(np.abs(unit)))])
    first /= np.linalg.norm(first)
    return tuple(float(v) for v in np.stack([first, np.cross(unit, first), unit]).ravel())


def tube_correction(points: Array, parameters: Mapping[str, Any], viscosity: Any) -> Corrected:
    """The tube's image system at the samples ``points`` (world frame, m): its ``centre``, ``frame`` (rows, the last
    its axis), ``radius`` and its unit tube's compiled ``table``."""
    table: TableArrays = parameters["table"]
    frame = jnp.asarray(parameters["frame"], dtype=points.dtype).reshape(3, 3)
    radius = jnp.asarray(parameters["radius"], dtype=points.dtype)
    local = jnp.matmul(points - jnp.asarray(parameters["centre"], dtype=points.dtype), frame.T,
                       precision=PRECISION) / radius
    across = local[:, :2]
    rho = switch.nonzero_norm(across, axis=-1)
    cos, sin = across[:, 0] / rho, across[:, 1] / rho
    sine = sin[:, None] * cos[None, :] - cos[:, None] * sin[None, :]
    cosine = cos[:, None] * cos[None, :] + sin[:, None] * sin[None, :]
    folded = jnp.arctan2(switch.absolute(sine, owner=ANGLE, resolution=lambda: switch.rounding(1.0),
                                         indicator=sine), cosine)
    offset = local[:, None, 2] - local[None, :, 2]
    along = switch.absolute(offset, owner=ALONG, resolution=lambda: switch.rounding(1.0), indicator=offset)
    read = evaluate(table, rho[:, None], rho[None, :], folded, along, owner=TABLE)
    by_angle = switch.where(sine >= 0, 1.0, -1.0, owner=ANGLE, resolution=lambda: switch.rounding(1.0),
                            indicator=sine)
    by_along = switch.where(offset >= 0, 1.0, -1.0, owner=ALONG, resolution=lambda: switch.rounding(1.0),
                            indicator=offset)
    odd_angle, odd_along = jnp.asarray(ODD_IN_ANGLE, dtype=points.dtype), jnp.asarray(ODD_ALONG, dtype=points.dtype)
    block = read.value * (1.0 + (by_angle[..., None, None] - 1.0) * odd_angle) \
        * (1.0 + (by_along[..., None, None] - 1.0) * odd_along)
    basis = jnp.stack([cos[:, None] * frame[0] + sin[:, None] * frame[1],
                       -sin[:, None] * frame[0] + cos[:, None] * frame[1],
                       jnp.broadcast_to(frame[2], (points.shape[0], 3))], axis=1)
    turned = jnp.einsum("tsab,sbj->tsaj", block, basis, precision=PRECISION)
    world = jnp.einsum("tai,tsaj->tsij", basis, turned, precision=PRECISION)
    count = 3 * points.shape[0]
    return Corrected(world.transpose(0, 2, 1, 3).reshape(count, count) / (radius * viscosity),
                     jnp.all(read.accepted))


def tube_preparation(table: Callable[[], Table]) -> Callable[[Mapping[str, Any], Any], dict[str, object]]:
    """The tube's ``prepare``: the frame across its axis, held as built, and ``table()`` compiled for the run's
    dtype."""

    def prepare(parameters: Mapping[str, Any], dtype: Any) -> dict[str, object]:
        return {"centre": tuple(float(c) for c in parameters["centre"]), "frame": frame_across(parameters["axis"]),
                "radius": float(parameters["radius"]), "table": table().arrays(dtype)}

    return prepare


#: The tube's constants: its centre and radius (a study's leaves), its frame and its table (held as built).
TUBE_PARAMETERS = ("centre", "frame", "radius", "table")
TUBE_UNITS = {"centre": "m", "frame": "1", "radius": "m", "table": "1"}


def axial_reach(table: Callable[[], Table]) -> Callable[[Mapping[str, Any], float], list[tuple[str, str]]]:
    """The tube's ``beyond``: a body whose extent along the tube's axis (at most twice its reach from its centre)
    exceeds the axial offset the table reaches, over the tube's radius, is read outside it at every pose."""

    def beyond(parameters: Mapping[str, Any], reach: float) -> list[tuple[str, str]]:
        held = table()
        span = 2.0 * reach / float(parameters["radius"])
        low = [axis.readable_range[0] for axis in held.axes]
        high = [axis.readable_range[1] for axis in held.axes]
        found = query_range_problems(held, low, [*high[:3], max(span, low[3])])
        return [(f"the body reaches {reach!r} m from its centre, so its samples may lie {span!r} of the tube's "
                 f"radius apart along its axis, and the tube's table reaches {high[3]!r}: {problem}",
                 "use a body shorter than the table reaches along the tube's axis, or a wider tube")
                for problem in found]

    return beyond


def tube_wall(name: str, version: str, table: Callable[[], Table]) -> WallCorrection:
    """The tube's correction for a wall kind ``name`` ``version`` reading ``table()``, the unit tube's table: what a
    tube's wall kind registers with the boundary-element method (``register_wall_correction``)."""
    return WallCorrection(name, version, TUBE_PARAMETERS, tube_correction, prepare=tube_preparation(table),
                          units=TUBE_UNITS, held=frozenset({"frame", "table"}), beyond=axial_reach(table))


__all__ = ["ALONG", "ANGLE", "MIRRORED", "ODD_ALONG", "ODD_IN_ANGLE", "TABLE", "TUBE_PARAMETERS", "TUBE_UNITS",
           "axial_reach", "frame_across", "tube_correction", "tube_preparation", "tube_table_from", "tube_wall",
           "unit_tube_table"]
