"""The moving refinement patch (the patch phase; F0's C4, its window decisions and Bundle 2's measured defaults): a fixed
fine box near a body by a wall, which hands back its near-wall reflection of the cluster's forces in place of the coarse
grid's under-resolved one in what the near field reads. The coarse solve is untouched.

**What it solves** (the patch spike's form, F2): the reflection field ``v = u - u_free`` of the cluster, steady
Stokes-Brinkman on the box, ``-L v + G p + (1/K) v = -(1/K) u_free``, ``D v = 0``, every face given by the coarse grid's
reflection (:func:`patch_boundary`), ``1/K`` the walls' band resampled from the same signed distance at the box's faces,
and ``u_free`` the forces' own regularised Stokeslets (no blob in the patch). The viscosity divides out of the steady
operator; it enters through ``u_free`` and the given values. The near field reads ``v`` at its points (trilinear: the
measured default), or each force point's cell average (:func:`cell_read`).

**Quasi-steady** (the first step's proposal, taken by the coordinator): the patch holds no field of its own between
steps, only its warm start; the near field it serves is steady Stokes.

**Snapped, never recompiling** (C4 decided 2): the box's shape is the kind's; its corner (``origin``) is whole fine
cells on the fine lattice from the flow grid's corner, state, so a move changes values and never a shape.

**The forces' own Stokeslets on a shell.** ``u_free`` enters through ``(1/K) u_free``, nonzero in the band and the
solid. It is evaluated on the faces from the band's fluid edge to :data:`SHELL_DEPTH` cells into the solid, at a
capacity the kind holds (its overflow counted, never dropped silently): deeper, the stiff interior screens a face's
source over about ``delta_in h`` (0.1 cell), so its effect on the band is about e^-10 of it [arithmetic]. A capacity of
zero takes every interior face (exact, the measured reference).

**One uniform dilation per patch** (decided 4 (5)), by whole fine cells (decided 4 (4)): the walls' signed distance is
moved into the solid by ``l`` cells; :func:`blended_reflection` blends the two neighbouring whole-cell evaluations
linearly in the seam's continuous dilation, the second solve taken only where its weight is nonzero.

**The transpose.** The patch's CG is its own transpose solve (the operator is symmetric, its start linear in its
right-hand side): :func:`patch_reflection_transpose` reads back through the read's transpose, solves once, and
transposes the solve-free remainder (the lift and the source) by JAX; it is the transpose to the CG's tolerance."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._band import DEFAULT_BAND, DEFAULT_INTERIOR, INTERIOR_RAMP_END, FacedSolid, band_named, \
    interior_depth, wall_fields
from mime.nodes.flow.fvm._grid import Velocity
from mime.nodes.flow.fvm._nested import NestedMACGrid
from mime.nodes.flow.fvm._patch_transfers import Given, Quadrature, cell_read, cell_read_transpose, patch_read, \
    patch_read_transpose
from mime.nodes.flow.fvm._solve import AT_FLOOR_BY_DEFAULT, REPORT_SIZE, Jacobi, Stopping, default_tolerance, \
    projected_solve, stagnation_window

_lax: Any = jax.lax
_linear_transpose: Any = cast(Any, jax).linear_transpose

#: How the near field reads the patch's field: at its force points, or each force point's cell average.
READS = ("point", "cell")
#: How far into the solid, in fine cells, the forces' own Stokeslets are evaluated: the stiff interior's ramp end and
#: one cell beyond it.
SHELL_DEPTH = INTERIOR_RAMP_END + 1.0
#: The bytes one chunk of the Stokeslets' pairs (faces against force points) may hold.
CHUNK_BYTES = 1 << 26
#: The state the host node stores for the patch (the kind owns it; MADDENING 0.4 keeps state only in nodes): the
#: corner in whole fine cells, the warm start (a column per component) and the last solve's report.
STATE_FIELDS = ("patch_origin", "patch_start_x", "patch_start_y", "patch_start_z", "patch_report")
#: The given values' net flux over their absolute flux above which a solve is flagged.
FLUX_SHARE_LINE = 1e-2


@dataclass(frozen=True)
class RefinementPatch:
    """The kind: a box of ``cells`` fine cells of side ``spacing`` (m), the coarse grid's band and interior by name, the
    near field's read (:data:`READS`), the forces' own Stokeslets' capacity in faces (zero: every face) and depth into
    the solid in cells (``None``: every face where the band or the solid is)."""

    cells: tuple[int, int, int]
    spacing: float
    band: str = DEFAULT_BAND
    interior: str = DEFAULT_INTERIOR
    read: str = "point"
    shell_capacity: int = 0
    shell_depth: float | None = SHELL_DEPTH

    def __post_init__(self) -> None:
        if self.read not in READS:
            raise ValueError(f"the patch's read is one of {READS}, not {self.read!r}")
        if len(self.cells) != 3 or min(self.cells) < 4:
            raise ValueError(f"the patch is a box of at least 4 cells a side, not {self.cells}")
        if not self.spacing > 0.0:
            raise ValueError(f"the patch's fine cell is positive, not {self.spacing}")
        band_named(self.band)
        interior_depth(self.interior)

    def grid(self) -> NestedMACGrid:
        return NestedMACGrid(self.cells, self.spacing)

    def face_count(self) -> int:
        grid = self.grid()
        return sum(int(np.prod(grid.interior_shape(c))) for c in range(3))

    def capacity(self) -> int:
        return self.shell_capacity if self.shell_capacity > 0 else self.face_count()

    def initial_state(self, origin: Array, dtype: Any) -> dict[str, Array]:
        zero = self.grid().zeros(dtype)
        return {"patch_origin": jnp.asarray(origin, dtype=jnp.int32), "patch_start_x": zero[0],
                "patch_start_y": zero[1], "patch_start_z": zero[2], "patch_report": jnp.zeros(REPORT_SIZE, dtype=dtype)}

    def start_at(self, state: dict[str, Array], origin: Array) -> Velocity:
        """The held warm start moved to ``origin`` (whole cells; the slab the move uncovers zero)."""
        held = (state["patch_start_x"], state["patch_start_y"], state["patch_start_z"])
        return shifted_start(held, jnp.asarray(origin, jnp.int32) - jnp.asarray(state["patch_origin"], jnp.int32))

    def advanced_state(self, origin: Array, solved: PatchSolve) -> dict[str, Array]:
        """The state after a solve at ``origin``."""
        return {"patch_origin": jnp.asarray(origin, dtype=jnp.int32), "patch_start_x": solved.start[0],
                "patch_start_y": solved.start[1], "patch_start_z": solved.start[2], "patch_report": solved.report}


class PatchPlacement(NamedTuple):
    """What a step freezes of the patch, made once a step (:func:`placed`): the corner in whole fine cells; the walls'
    inverse permeability on the interior faces (m^-2), resampled there and dilated; the forces' own Stokeslets' faces
    (flat interior indices, the face count where unused), their points (m), components and ``1/K``, and how many faces
    wanted one; the near field's points ``(n, 3)`` and, for the cell read, its quadrature; the blob (``eps^2``, m^2) and
    the viscosity (Pa s) of the forces' own Stokeslets."""

    origin: Array
    resistivity: Velocity
    shell: Array
    shell_points: Array
    shell_components: Array
    shell_weight: Array
    shell_needed: Array
    points: Array
    quadrature: Quadrature | None
    blob: Array
    viscosity: Array


class PatchSolve(NamedTuple):
    """One patch solve: the reflection at the read points ``(n, 3)``, the solved interior remainder (the next warm
    start), the CG's report, the given values' net flux over their absolute flux before it was removed, and how many
    reads or given values fell outside their lattice."""

    values: Array
    start: Velocity
    report: Array
    flux_share: Array
    outside: Array


# -- placing the patch ------------------------------------------------------------------------------------------------
def snapped_origin(patch: RefinementPatch, centre: Array) -> Array:
    """The corner (whole fine cells, int32) of the patch centred on ``centre`` (m, flow frame), snapped to the fine
    lattice."""
    held = _lax.stop_gradient(jnp.asarray(centre))
    cells = jnp.asarray(patch.cells, dtype=held.dtype)
    return jnp.round(held / patch.spacing - 0.5 * cells).astype(jnp.int32)


def whole_cell_dilations(patch: RefinementPatch, dilation: Array) -> tuple[Array, Array]:
    """The lower of the two whole-cell dilations around ``dilation`` (m) and the upper's blend weight (decided 4 (4)):
    the dilation is ``(lower + weight) h``, continuous in ``dilation`` through the weight."""
    cells = jnp.asarray(dilation) / patch.spacing
    lower = jnp.floor(_lax.stop_gradient(cells))
    return lower.astype(jnp.int32), cells - lower


@dataclass(frozen=True)
class _Dilated:
    """A solid moved into itself by ``by`` (m): its signed distance (positive in the solid) less ``by``."""

    solid: Any
    by: Any

    @property
    def region(self) -> str:
        return str(self.solid.region)

    def signed_distance(self, points: Array) -> Array:
        return self.solid.signed_distance(points) - self.by


@dataclass(frozen=True)
class _DilatedFaced:
    """:class:`_Dilated` for a solid held per face set (``FacedSolid``)."""

    solid: Any
    by: Any

    @property
    def region(self) -> str:
        return str(self.solid.region)

    def face_distances(self, points: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array, Array], Array]:
        phis, counts = self.solid.face_distances(points)
        return (phis[0] - self.by, phis[1] - self.by, phis[2] - self.by), counts


def placed(patch: RefinementPatch, solids: Sequence[Any], origin: Array, dilation: Array, points: Array, blob: Any,
           viscosity: Any, quadrature: Quadrature | None = None) -> PatchPlacement:
    """The patch at ``origin`` (whole fine cells) with the walls of ``solids`` resampled at its faces and dilated by
    ``dilation`` whole fine cells; the near field's ``points`` (m, flow frame), its quadrature for the cell read, and
    the forces' own Stokeslets' ``blob`` (``eps^2``, m^2: one value or one per force point, the MRS's own) and
    ``viscosity`` (Pa s)."""
    grid = patch.grid()
    dtype = points.dtype
    h = jnp.asarray(patch.spacing, dtype=dtype)
    origin = jnp.asarray(origin, dtype=jnp.int32)
    corner = origin.astype(dtype) * h
    by = jnp.asarray(dilation).astype(dtype) * h
    opened = [_DilatedFaced(solid, by) if isinstance(solid, FacedSolid) else _Dilated(solid, by) for solid in solids]
    fields = wall_fields(grid, cast(Any, opened), band_named(patch.band), interior_depth(patch.interior), dtype,
                         offset=lambda at: at + corner)
    resistivity = fields.inverse_permeability
    flat_k = jnp.concatenate([k.reshape(-1) for k in resistivity])
    flat_phi = jnp.concatenate([phi.reshape(-1) for phi in fields.signed_distance])
    wanted = flat_k > 0.0
    if patch.shell_depth is not None:
        wanted = wanted & (flat_phi <= patch.shell_depth * h)
    total = flat_k.shape[0]
    (index,) = jnp.nonzero(wanted, size=patch.capacity(), fill_value=total)
    used = index < total
    safe = jnp.where(used, index, 0)
    face_points = jnp.concatenate([grid.face_points(c, dtype).reshape(-1, 3) for c in range(3)]) + corner
    components = np.concatenate([np.full(int(np.prod(grid.interior_shape(c))), c, dtype=np.int32) for c in range(3)])
    return PatchPlacement(origin, resistivity, index.astype(jnp.int32), face_points[safe],
                          jnp.asarray(components)[safe], jnp.where(used, flat_k[safe], 0.0),
                          jnp.sum(wanted.astype(jnp.int32)), points, quadrature, jnp.asarray(blob, dtype=dtype),
                          jnp.asarray(viscosity, dtype=dtype))


# -- the forces' own Stokeslets ---------------------------------------------------------------------------------------
def own_stokeslets(placement: PatchPlacement, forces: Array) -> Array:
    """The forces' own regularised Stokeslets at the shell's faces, each face its component, ``(capacity,)``: MIME's
    MRS kernel, ``((r^2 + 2 eps^2) F + (r . F) r) / (8 pi mu (r^2 + eps^2)^(3/2))``, summed over the force points in
    chunks of faces. Linear in ``forces``."""
    at = placement.shell_points
    count = at.shape[0]
    sources = placement.points.astype(at.dtype)
    forces = forces.astype(at.dtype)
    size = max(1, min(count, CHUNK_BYTES // (max(1, sources.shape[0]) * 8 * at.dtype.itemsize)))
    chunks = -(-count // size)
    pad = chunks * size - count
    held_at = jnp.concatenate([at, jnp.broadcast_to(at[:1], (pad, 3))]).reshape(chunks, size, 3)
    held_c = jnp.concatenate([placement.shell_components, jnp.zeros(pad, jnp.int32)]).reshape(chunks, size)
    eps2 = jnp.broadcast_to(placement.blob, (sources.shape[0],))[None, :]
    scale = 1.0 / (8.0 * math.pi * placement.viscosity)

    def chunk(pair: tuple[Array, Array]) -> Array:
        x, component = pair
        r = x[:, None, :] - sources[None, :, :]
        r2 = jnp.sum(r * r, axis=-1)
        s = scale / (r2 + eps2) ** 1.5
        along = jnp.sum(r * forces[None, :, :], axis=-1)
        force_c = forces.T[component]
        r_c = jnp.take_along_axis(r, component[:, None, None], axis=2)[..., 0]
        return jnp.sum(s * ((r2 + 2.0 * eps2) * force_c + r_c * along), axis=1)

    return _lax.map(chunk, (held_at, held_c)).reshape(-1)[:count]


def source(patch: RefinementPatch, placement: PatchPlacement, forces: Array) -> Velocity:
    """``-(1/K) u_free`` on the interior faces: the forces' own Stokeslets at the shell, zero elsewhere."""
    grid = patch.grid()
    shapes = [grid.interior_shape(c) for c in range(3)]
    total = sum(int(np.prod(s)) for s in shapes)
    held = own_stokeslets(placement, forces)
    flat = jnp.zeros(total, dtype=held.dtype).at[placement.shell].add(-placement.shell_weight * held, mode="drop")
    ends = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    parts = [flat[ends[c]:ends[c + 1]].reshape(shapes[c]) for c in range(3)]
    return (parts[0], parts[1], parts[2])


# -- the solve --------------------------------------------------------------------------------------------------------
def _operators(grid: NestedMACGrid, resistivity: Velocity) -> tuple[Callable[[Velocity], Velocity],
                                                                    Callable[[Velocity], Velocity], Velocity]:
    """``-L + 1/K`` on the interior (given values zero), its magnitude, and its diagonal."""
    k = resistivity

    def apply(x: Velocity) -> Velocity:
        lap = grid.laplacian(x)
        return (-lap[0] + k[0] * x[0], -lap[1] + k[1] * x[1], -lap[2] + k[2] * x[2])

    def magnitude(x: Velocity) -> Velocity:
        lap = grid.laplacian_magnitude(x)
        return (lap[0] + k[0] * x[0], lap[1] + k[1] * x[1], lap[2] + k[2] * x[2])

    base = -grid.laplacian_diagonal()
    return apply, magnitude, (base + k[0], base + k[1], base + k[2])


def lifted_and_drive(patch: RefinementPatch, placement: PatchPlacement, boundary: Given, forces: Array
                     ) -> tuple[Velocity, Velocity]:
    """The solve-free part, linear in ``(boundary, forces)``: the given values made compatible and lifted (padded,
    divergence-free), and the interior's right-hand side, the forces' source less the operator on the lift."""
    grid = patch.grid()
    lift = grid.lifted(grid.compatible(boundary))
    drive = source(patch, placement, forces)
    lap = grid.laplacian_padded(lift)
    inside = grid.interior_of(lift)
    k = placement.resistivity
    rhs = (drive[0] + lap[0] - k[0] * inside[0], drive[1] + lap[1] - k[1] * inside[1],
           drive[2] + lap[2] - k[2] * inside[2])
    return lift, rhs


def patch_stopping(patch: RefinementPatch, dtype: Any, tolerance: Any = None, cap: int = 5000,
                   check_every: int = 20) -> Stopping:
    """The patch CG's stop: the run's dtype's default tolerance, at float32's rounding floor, the flow's stagnation
    window for the box."""
    name = str(jnp.dtype(dtype))
    return Stopping(tolerance=default_tolerance(dtype) if tolerance is None else tolerance,
                    at_floor=name in AT_FLOOR_BY_DEFAULT, cap=cap, check_every=check_every,
                    window=stagnation_window(patch.cells, check_every))


def _read(patch: RefinementPatch, placement: PatchPlacement, field: Velocity) -> tuple[Array, Array]:
    grid = patch.grid()
    if patch.read == "cell":
        assert placement.quadrature is not None, "the cell read takes the body's quadrature"
        return cell_read(grid, placement.origin, field, placement.quadrature)
    return patch_read(grid, placement.origin, field, placement.points)


def _read_transpose(patch: RefinementPatch, placement: PatchPlacement, cotangent: Array) -> Velocity:
    grid = patch.grid()
    if patch.read == "cell":
        assert placement.quadrature is not None, "the cell read takes the body's quadrature"
        return cell_read_transpose(grid, placement.origin, placement.quadrature, cotangent)
    return patch_read_transpose(grid, placement.origin, placement.points, cotangent)


def patch_reflection(patch: RefinementPatch, placement: PatchPlacement, boundary: Given, forces: Array,
                     start: Velocity | None, stopping: Stopping, tap: Array) -> PatchSolve:
    """The patch's reflection of ``forces`` ``(n, 3)`` (N, at the placement's points) at the read points, its given
    values ``boundary`` (the coarse grid's reflection at them, :func:`patch_boundary`), warm-started at ``start``
    (``None``: cold), through the FVM's projected CG (``tap``: its report block's input, zeros). Linear in
    ``(boundary, forces)`` to the CG's tolerance."""
    grid = patch.grid()
    lift, rhs = lifted_and_drive(patch, placement, boundary, forces)
    apply, magnitude, diagonal = _operators(grid, placement.resistivity)
    remainder, report = projected_solve(grid, apply, diagonal, rhs, start, stopping, tap, Jacobi(), magnitude)
    padded = grid.padded(remainder)
    field = (lift[0] + padded[0], lift[1] + padded[1], lift[2] + padded[2])
    values, outside = _read(patch, placement, field)
    return PatchSolve(values, remainder, report, grid.flux_share(boundary), outside)


def patch_reflection_transpose(patch: RefinementPatch, placement: PatchPlacement, cotangent: Array,
                               start: Velocity | None, stopping: Stopping, tap: Array
                               ) -> tuple[Given, Array, Array]:
    """:func:`patch_reflection`'s transpose: a cotangent ``(n, 3)`` at the read points to the given values' and the
    forces' cotangents, and the transpose solve's report. One CG solve (the operator's own transpose)."""
    grid = patch.grid()
    dtype = cotangent.dtype
    back = _read_transpose(patch, placement, cotangent)
    apply, magnitude, diagonal = _operators(grid, placement.resistivity)
    solved, report = projected_solve(grid, apply, diagonal, grid.interior_of(back), start, stopping, tap, Jacobi(),
                                     magnitude)
    shapes = [jax.ShapeDtypeStruct((grid.given_count(c),), dtype) for c in range(3)]
    forces = jax.ShapeDtypeStruct(placement.points.shape, dtype)

    def linear(boundary: Given, f: Array) -> tuple[Velocity, Velocity]:
        return lifted_and_drive(patch, placement, boundary, f)

    transposed = _linear_transpose(linear, (shapes[0], shapes[1], shapes[2]), forces)
    given, force = cast(tuple[Given, Array], transposed((back, solved)))
    return (given[0], given[1], given[2]), force, report


def blended_reflection(patch: RefinementPatch, placements: tuple[PatchPlacement, PatchPlacement], weight: Array,
                       boundary: Given, forces: Array, starts: tuple[Velocity | None, Velocity | None],
                       stopping: Stopping, tap: Array) -> tuple[Array, PatchSolve, PatchSolve]:
    """The two neighbouring whole-cell dilations' reflections blended by ``weight`` (the upper's, decided 4 (4)): the
    second solve only where the weight is nonzero."""
    lower = patch_reflection(patch, placements[0], boundary, forces, starts[0], stopping, tap)
    dtype = lower.values.dtype

    def upper(_: Any) -> PatchSolve:
        return patch_reflection(patch, placements[1], boundary, forces, starts[1], stopping, tap)

    def skipped(_: Any) -> PatchSolve:
        held = starts[1] if starts[1] is not None else patch.grid().zeros(dtype)
        return PatchSolve(jnp.zeros_like(lower.values), held, jnp.zeros(REPORT_SIZE, dtype=dtype),
                          jnp.zeros((), dtype=dtype), jnp.zeros((), dtype=lower.outside.dtype))

    higher = _lax.cond(weight > 0.0, upper, skipped, None)
    w = jnp.asarray(weight, dtype=dtype)
    return (1.0 - w) * lower.values + w * higher.values, lower, higher


def shifted_start(start: Velocity, by: Array) -> Velocity:
    """A warm start moved by ``by`` whole cells (the new corner less the old): each entry taken from ``i + by``, the
    slab the move uncovers zero."""
    held: list[Array] = []
    for component in start:
        moved = component
        for axis in range(3):
            moved = jnp.roll(moved, -by[axis], axis=axis)
            index = jnp.arange(component.shape[axis]) + by[axis]
            inside = (index >= 0) & (index < component.shape[axis])
            shape = [1, 1, 1]
            shape[axis] = component.shape[axis]
            moved = jnp.where(inside.reshape(shape), moved, 0.0)
        held.append(moved)
    return (held[0], held[1], held[2])


__all__ = ["CHUNK_BYTES", "FLUX_SHARE_LINE", "READS", "SHELL_DEPTH", "STATE_FIELDS", "PatchPlacement", "PatchSolve",
           "RefinementPatch", "blended_reflection", "lifted_and_drive", "own_stokeslets", "patch_reflection",
           "patch_reflection_transpose", "patch_stopping", "placed", "shifted_start", "snapped_origin", "source",
           "whole_cell_dilations"]
