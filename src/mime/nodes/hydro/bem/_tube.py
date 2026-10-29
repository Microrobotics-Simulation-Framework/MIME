"""The straight circular tube's correction for the boundary-element method: the image system of a Stokeslet inside an
infinite tube with a no-slip wall (Liron & Shahar 1978), read from a table of it through ``mime.core.tables``.

**The table** is a table file of the 2026-05-21 form, at a tube radius ``R_cyl`` and viscosity 1: for a source at
``(rho_s, 0, 0)`` and a target at ``(rho_t, dphi, dz)`` (cylindrical), the image's ``3 x 3`` block, its rows the
velocity's cylindrical components at the target (radial, azimuthal, axial) and its columns the force's at the source,
on knots ``rho_t`` and ``rho_s`` (the first two axes, in that order), ``dphi`` on ``[0, pi]`` and ``dz`` from 0. The
image system scales with the tube: in a tube of radius ``R`` and viscosity ``mu`` it is ``G(x / R) / (R mu)``, with
``G`` the image of the tube of radius 1 at viscosity 1, so the table is read as that one, its lengths over ``R_cyl``
and its values times ``R_cyl``, whichever radius it was built at. **Its readable range** is its own: the file's
``readable_rho`` (two radii, in its lengths) where it states one, its radii's knots otherwise; a read outside it is
NaN and not accepted, and the node publishes whether every read was.

**Held in the source's frame.** The table is held with each block's rows turned from the target's cylindrical
components into the source's (``x`` along the source's direction from the axis, ``y`` across it, ``z`` along the
axis): ``H = R(dphi)^T B``. On the axis a target has no angle, and there the image the source sees is the same from
every direction, so ``H`` at ``rho_t = 0`` is one block at every angle (the series' limit on the axis, which the table's
knot there holds), and the cubic, which reproduces a constant exactly, reads that block at every angle between the
knots too. Turning keeps each component's parity in the angle (the rows' parities are each other's opposite).

**Its parity.** Each component is even or odd in ``dphi`` and in ``dz`` (``ODD_IN_ANGLE``, ``ODD_ALONG``): the table
holds half of each period, and a read at ``|dphi|`` (folded into ``[0, pi]``) and ``|dz|`` takes the sign of the odd
components from the sign of ``sin(dphi)`` and of ``dz``. So that the read is continuously differentiable across
``dphi = 0``, ``dphi = pi`` and ``dz = 0``, as the image is, the angle and the axial offset each gain two knots
mirrored past each end that is a symmetry (their values the parity's), so that the cubic's slope at the end is the
symmetric one (zero for an even component), and each odd component is set to zero exactly where its parity makes it
zero (it is zero there to the table's rounding).

**The read.** The body's samples, in the tube's frame (its axis along z through its centre, the frame across the axis
held as built) and over its radius: each sample's distance from the axis and its direction across it, the angle
between each pair from those directions, and their axial offset. A pair read as above, the target's rows in the
source's frame, is turned into the world's by the source's frame alone, ``W(t, s) = R_s^T H R_s``, so a target on the
axis needs no angle: the direction it is given there (the tube frame's first, ``AXIS``) reads the same block. A source
on the axis has no frame, and by reciprocity (``G(x, y) = G(y, x)^T``) the pair is also the transposed read with the
two samples' roles exchanged, in which it is the target. **The rule on the axis:** each pair is the blend
``w W(t, s) + (1 - w) W(s, t)^T``, with ``w = rho_s^2 / (rho_t^2 + rho_s^2)``: the read in which the sample nearer
the axis is the target weighs more, wholly so on the axis, so the Cartesian result is continuous through the axis at
either end of a pair, and the matrix is reciprocal exactly. A pair both on the axis takes ``w = 1/2`` (``AXIS``); its
two reads are the axis's own block, the same from every direction. Exactly on the axis a sample's derivative across it
is the norm's there, zero (one point). The block is scaled by ``1 / (R mu)``. Where a read lies outside the table's
readable range (a sample further from the axis than it reads, or a pair further apart along the axis than it reaches)
its value is NaN and it is not accepted: the node publishes whether every read was."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, cast

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
#: The owner of the rule on the axis: a sample on it takes the tube frame's first direction across it, and a pair both
#: on it weighs its two reads equally.
AXIS = "tube.axis"
#: Knots mirrored past each symmetric end of the angle's and the axial offset's axes.
MIRRORED = 2


def _parity(odd: np.ndarray) -> np.ndarray:
    """``-1`` for each odd component and ``+1`` for each even one, as a ``3 x 3`` block."""
    return np.where(odd, -1.0, 1.0)


def into_the_source_frame(blocks: np.ndarray, angle: np.ndarray) -> None:
    """``blocks`` (``(rho_t, rho_s, angle, along, 3, 3)``, the target's rows cylindrical), each row turned in place
    into the source's frame at its ``angle`` from the source to the target: ``H = R(dphi)^T B``, ``x = cos B_rho - sin
    B_phi`` and ``y = sin B_rho + cos B_phi``. One angle at a time, so a large table is never copied whole."""
    for index, at in enumerate(angle):
        c, s = math.cos(float(at)), math.sin(float(at))
        held = blocks[:, :, index]
        radial, azimuthal = held[..., 0, :].copy(), held[..., 1, :].copy()
        held[..., 0, :] = c * radial - s * azimuthal
        held[..., 1, :] = s * radial + c * azimuthal


def unit_tube_table(radius: float, rho: np.ndarray, angle: np.ndarray, along: np.ndarray, data: np.ndarray, *,
                    scheme: Scheme | tuple[Scheme, ...] = "cubic", float32_values: bool = False,
                    readable_rho: tuple[float, float] | None = None) -> Table:
    """The table of the image of the tube of radius 1 at viscosity 1, from one built at ``radius`` (``rho``,
    ``angle`` from 0, to pi for a whole table, ``along`` from 0, ``data`` of shape ``(rho_t, rho_s, angle, along, 3,
    3)``), with the angle mirrored past 0 (and past pi where it reaches it) and the axial offset past 0;
    ``readable_rho`` narrows the radii's readable range, in the tube's radius. Held with each block's rows in the
    source's frame (``into_the_source_frame``). A slice of a table (a block of its knots from the first angle and the
    first offset) is read the same way."""
    if not (angle[0] == 0.0 and along[0] == 0.0):
        raise ValueError("the table's angle and its axial offset run from 0")
    wraps = math.isclose(float(angle[-1]), math.pi, rel_tol=1e-15)
    by_angle, by_along = _parity(ODD_IN_ANGLE), _parity(ODD_ALONG)
    count, steps, past = len(angle), len(along), MIRRORED if wraps else 0
    values = np.empty((len(rho), len(rho), MIRRORED + count + past, MIRRORED + steps, 3, 3))
    inside = values[:, :, MIRRORED:MIRRORED + count, MIRRORED:]
    np.multiply(data, radius, out=inside)
    into_the_source_frame(inside, angle)
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
def tube_table_from(path: str, *, scheme: Scheme | tuple[Scheme, ...] = "cubic",
                    float32_values: bool = False) -> Table:
    """The unit tube's table from a table file of the 2026-05-21 form (``R_cyl``, ``rho_grid``, ``dphi_grid``,
    ``dz_grid``, ``data``), read once per process for each file and setting. Its radii's readable range is the file's
    own: its ``readable_rho`` (two radii, in the file's lengths) where it states one, which narrows its knots' range and
    never widens it, and its knots' range otherwise."""
    with np.load(Path(path)) as held:
        radius = float(held["R_cyl"])
        stated = held["readable_rho"] if "readable_rho" in held.files else None
        readable = None if stated is None else cast(tuple[float, float],
                                                    tuple(float(r) / radius for r in np.asarray(stated).ravel()))
        return unit_tube_table(radius, held["rho_grid"], held["dphi_grid"], held["dz_grid"], held["data"],
                               scheme=scheme, float32_values=float32_values, readable_rho=readable)


def frame_across(axis: Any) -> tuple[float, ...]:
    """The tube's frame as rows, the last its axis: a frame across an axis cannot be continuous in every axis, so it
    is taken here, as built, from the coordinate direction least along the axis."""
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    first = np.cross(unit, np.eye(3)[int(np.argmin(np.abs(unit)))])
    first /= np.linalg.norm(first)
    return tuple(float(v) for v in np.stack([first, np.cross(unit, first), unit]).ravel())


def across_the_axis(across: Array) -> tuple[Array, Array, Array]:
    """Each sample's squared distance from the tube's axis, its distance and its direction across the axis (rows,
    ``x`` and ``y`` of the tube's frame), from its coordinates across it: on the axis, where it has none, the frame's
    first direction (``AXIS``), which a read there does not depend on."""
    squared = jnp.sum(across * across, axis=-1)
    off_axis = squared > 0
    distance = switch.where(off_axis, jnp.sqrt(switch.domain_guard(off_axis, squared, 1.0)), 0.0, owner=AXIS,
                            resolution=lambda: switch.rounding(1.0), indicator=squared)
    first = jnp.zeros_like(across).at[:, 0].set(1.0)
    direction = switch.where(off_axis[:, None], across / switch.domain_guard(off_axis, distance, 1.0)[:, None], first,
                             owner=AXIS, resolution=lambda: switch.rounding(1.0), indicator=squared[:, None])
    return squared, distance, direction


def target_weight(squared: Array) -> Array:
    """``(t, s)``: the weight of each pair's read with ``t`` the target, ``rho_s^2 / (rho_t^2 + rho_s^2)``, from the
    samples' squared distances from the axis; ``1/2`` for a pair both on it (``AXIS``)."""
    total = squared[:, None] + squared[None, :]
    held = total > 0
    return switch.where(held, squared[None, :] / switch.domain_guard(held, total, 1.0), 0.5, owner=AXIS,
                        resolution=lambda: switch.rounding(1.0), indicator=total)


def tube_correction(points: Array, parameters: Mapping[str, Any], viscosity: Any) -> Corrected:
    """The tube's image system at the samples ``points`` (world frame, m): its ``centre``, ``frame`` (rows, the last
    its axis), ``radius`` and its unit tube's compiled ``table``."""
    table: TableArrays = parameters["table"]
    frame = jnp.asarray(parameters["frame"], dtype=points.dtype).reshape(3, 3)
    radius = jnp.asarray(parameters["radius"], dtype=points.dtype)
    local = jnp.matmul(points - jnp.asarray(parameters["centre"], dtype=points.dtype), frame.T,
                       precision=PRECISION) / radius
    squared, rho, direction = across_the_axis(local[:, :2])
    cos, sin = direction[:, 0], direction[:, 1]
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
    world = jnp.einsum("sai,tsaj->tsij", basis, turned, precision=PRECISION)
    weight = target_weight(squared)[..., None, None]
    blended = weight * world + (1.0 - weight) * world.transpose(1, 0, 3, 2)
    count = 3 * points.shape[0]
    return Corrected(blended.transpose(0, 2, 1, 3).reshape(count, count) / (radius * viscosity),
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


__all__ = ["ALONG", "ANGLE", "AXIS", "MIRRORED", "ODD_ALONG", "ODD_IN_ANGLE", "TABLE", "TUBE_PARAMETERS", "TUBE_UNITS",
           "across_the_axis", "axial_reach", "frame_across", "into_the_source_frame", "target_weight", "tube_correction",
           "tube_preparation", "tube_table_from", "tube_wall", "unit_tube_table"]
