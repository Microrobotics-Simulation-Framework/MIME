"""The moving refinement patch (the patch phase; F0's C4, its window decisions and Bundle 2's measured defaults): a fixed
fine box near a body by a wall, which hands back its near-wall reflection of the cluster's forces in place of the coarse
grid's under-resolved one in what the near field reads. The coarse solve is untouched.

**What it solves** (the patch spike's form, F2, under the step operator of the F5-patch interface): the reflection
field ``v = u - u_own`` of the cluster, ``(shift - mu L + mu/K) v + G q = shift v_prev - (mu/K) u_own``, ``D v = 0``,
every face given by the coarse grid's reflection (:func:`patch_boundary`); ``shift = rho/dt`` (zero: the steady limit),
``1/K`` the walls' band resampled from the same signed distance at the box's faces, ``u_own`` the cluster's own field
under the same operator, nonzero only where ``1/K`` is. The near field reads ``v`` at its points (trilinear: the
measured default), or each force point's cell average (:func:`cell_read`).

**The own field, in two parts** (the interface's (a) and (b)). The current step's: linear in the forces, the step
operator's regularised Stokeslet, MIME's regularised Stokeslet (the MRS's own blob) plus the Brinkman Stokeslet's
difference from the Stokes one at ``alpha^2 = shift / mu``, which is regular at the force point (:func:`own_response`;
the steady limit's is the regularised Stokeslet alone). The history's: the free field the cluster's past forces left,
which has no closed form; the flow's own box hands it as a lattice field, sampled at the shell's faces
(:func:`shell_sample`).

**Columns and the base.** The patch's functions take one column of forces and are vmappable over a column axis (no
branch on a value, no shape from data). A column takes no history: it is linear in its given values and its forces. The
base column carries the history (:class:`PatchHistory`): the previous step's reflection field, moved with the patch and
its uncovered slab filled from the coarse grid's previous reflection, and the own field's history at the shell.

**Snapped, never recompiling** (C4 decided 2): the box's shape is the kind's; its corner (``origin``) is whole fine
cells on the fine lattice from the flow grid's corner, state, so a move changes values and never a shape.

**The own field on a shell.** ``u_own`` enters through ``(mu/K) u_own``, nonzero in the band and the solid. It is evaluated
on the faces from the band's fluid edge to :data:`SHELL_DEPTH` cells into the solid, at a capacity the kind holds (its
overflow counted, never dropped silently): deeper, the stiff interior screens a face's source over about ``delta_in h``
(0.1 cell), so its effect on the band is about e^-10 of it [arithmetic]. A capacity of zero takes every interior face.

**One uniform dilation per patch** (decided 4 (5)), by whole fine cells (decided 4 (4)): the walls' signed distance is
moved into the solid by ``l`` cells; :func:`blended_reflection` blends the two neighbouring whole-cell evaluations
linearly in the seam's continuous dilation, the second solve taken only where its weight is nonzero.

**The transpose.** The patch's CG is its own transpose solve (the operator is symmetric, its start linear in its
right-hand side): :func:`patch_reflection_transpose` reads back through the read's transpose, solves once, and
transposes the solve-free remainder (the lift and the column's source) by JAX; it is the transpose to the CG's
tolerance."""

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
#: How far into the solid, in fine cells, the own field is evaluated: the stiff interior's ramp end and one cell beyond.
SHELL_DEPTH = INTERIOR_RAMP_END + 1.0
#: The bytes one chunk of the own response's pairs (faces against force points) may hold.
CHUNK_BYTES = 1 << 26
#: The state the host node stores for the patch (the kind owns it; MADDENING 0.4 keeps state only in nodes): the
#: corner in whole fine cells, the last step's reflection field (a column per component: its history and the next
#: step's start) and the last solve's report.
STATE_FIELDS = ("patch_origin", "patch_field_x", "patch_field_y", "patch_field_z", "patch_report")
#: The given values' net flux over their absolute flux above which a solve is flagged.
FLUX_SHARE_LINE = 1e-2
#: Below ``alpha r`` of this the Brinkman difference is summed as its series (the closed form cancels there).
SERIES_EDGE = 1.0
#: The series' terms: the last is under ``1 / 20!`` of the first at the edge.
SERIES_TERMS = 20


@dataclass(frozen=True)
class RefinementPatch:
    """The kind: a box of ``cells`` fine cells of side ``spacing`` (m), the coarse grid's band and interior by name, the
    near field's read (:data:`READS`), the own field's capacity in faces (zero: every face) and depth into the solid in
    cells (``None``: every face where the band or the solid is)."""

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
        """The state at a run's start: the corner, a field at rest, an empty report."""
        zero = self.grid().zeros(dtype)
        return {"patch_origin": jnp.asarray(origin, dtype=jnp.int32), "patch_field_x": zero[0],
                "patch_field_y": zero[1], "patch_field_z": zero[2], "patch_report": jnp.zeros(REPORT_SIZE, dtype=dtype)}

    def held_field(self, state: dict[str, Array], origin: Array, fill: Velocity) -> Velocity:
        """The last step's reflection field moved to ``origin`` (whole cells), the slab the move uncovers taken from
        ``fill`` (the coarse grid's previous reflection at the new faces, :func:`patch_faces`): the base column's history
        and every column's start."""
        held = (state["patch_field_x"], state["patch_field_y"], state["patch_field_z"])
        return moved(held, jnp.asarray(origin, jnp.int32) - jnp.asarray(state["patch_origin"], jnp.int32), fill)

    def advanced_state(self, origin: Array, field: Velocity, report: Array) -> dict[str, Array]:
        """The state after a step at ``origin``: its reflection field (the composition's, not a product's) and report."""
        return {"patch_origin": jnp.asarray(origin, dtype=jnp.int32), "patch_field_x": field[0],
                "patch_field_y": field[1], "patch_field_z": field[2], "patch_report": report}


class PatchPlacement(NamedTuple):
    """What a step freezes of the patch, made once a step (:func:`placed`): the corner in whole fine cells; the walls'
    inverse permeability on the interior faces (m^-2), resampled there and dilated; the own field's faces (flat interior
    indices, the face count where unused), their points (m), components and ``1/K``, and how many faces wanted one; the
    near field's points ``(n, 3)`` and, for the cell read, its quadrature; the blob (``eps^2``, m^2), the viscosity
    ``mu`` (Pa s) and the step's ``shift`` ``rho / dt`` (kg/m^3/s; zero: the steady limit)."""

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
    shift: Array


class PatchHistory(NamedTuple):
    """The base column's history: the last step's reflection field on the interior faces at this step's corner
    (:meth:`RefinementPatch.held_field`), and the own field's history at the shell's faces ``(capacity,)``
    (:func:`shell_sample` of the flow's own box)."""

    field: Velocity
    own: Array


class PatchSolve(NamedTuple):
    """One patch solve: the reflection at the read points ``(n, 3)``, the reflection field on the interior faces, the
    CG's report, the given values' net flux over their absolute flux before it was removed, and how many reads fell
    outside the box."""

    values: Array
    field: Velocity
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
           viscosity: Any, shift: Any = 0.0, quadrature: Quadrature | None = None) -> PatchPlacement:
    """The patch at ``origin`` (whole fine cells) with the walls of ``solids`` resampled at its faces and dilated by
    ``dilation`` whole fine cells; the near field's ``points`` (m, flow frame), its quadrature for the cell read; the
    own response's ``blob`` (``eps^2``, m^2: one value or one per force point, the MRS's own), the ``viscosity`` (Pa s)
    and the step's ``shift`` (``rho / dt``; zero, the steady limit)."""
    grid = patch.grid()
    dtype = points.dtype
    h = jnp.asarray(patch.spacing, dtype=dtype)
    origin = jnp.asarray(origin, dtype=jnp.int32)
    corner = origin.astype(dtype) * h
    by = jnp.asarray(dilation).astype(dtype) * h
    opened = [_DilatedFaced(solid, by) if isinstance(solid, FacedSolid) else _Dilated(solid, by) for solid in solids]
    if opened:
        fields = wall_fields(grid, cast(Any, opened), band_named(patch.band), interior_depth(patch.interior), dtype,
                             offset=lambda at: at + corner)
        resistivity, distances = fields.inverse_permeability, fields.signed_distance
    else:
        resistivity = grid.zeros(dtype)
        distances = tuple(jnp.full(grid.interior_shape(c), -jnp.inf, dtype=dtype) for c in range(3))
    flat_k = jnp.concatenate([k.reshape(-1) for k in resistivity])
    flat_phi = jnp.concatenate([phi.reshape(-1) for phi in distances])
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
                          jnp.asarray(viscosity, dtype=dtype), jnp.asarray(shift, dtype=dtype))


# -- the own field ----------------------------------------------------------------------------------------------------
def _series_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """The Brinkman difference's series in ``R = alpha r``: ``(A - 1)/R = 2 sum c_m R^(m-1)`` and ``(B - 1)/R = -2 sum
    d_m R^(m-1)``, ``c_m = (-1)^m (1/m! - 1/(m+1)! + 1/(m+2)!)``, ``d_m = (-1)^m (1/m! - 3/(m+1)! + 3/(m+2)!)``, from
    ``A = 2 e^-R (1 + 1/R + 1/R^2) - 2/R^2`` and ``B = -2 e^-R (1 + 3/R + 3/R^2) + 6/R^2``."""
    m = np.arange(1, SERIES_TERMS + 1)
    f = np.array([math.factorial(int(k)) for k in range(SERIES_TERMS + 3)], dtype=np.float64)
    sign = (-1.0) ** m
    c = sign * (1.0 / f[m] - 1.0 / f[m + 1] + 1.0 / f[m + 2])
    d = sign * (1.0 / f[m] - 3.0 / f[m + 1] + 3.0 / f[m + 2])
    return 2.0 * c, -2.0 * d


def brinkman_difference(big_r: Array) -> tuple[Array, Array]:
    """``((A - 1)/R, (B - 1)/R)`` of the Brinkman Stokeslet against the Stokes one, ``8 pi mu u = [A F / r + B r (r.F) /
    r^3]``, at ``R = alpha r``: the series below :data:`SERIES_EDGE`, the closed form above (where it does not cancel).
    Both are regular at ``R = 0`` (``-4/3`` and ``0``)."""
    a_terms, b_terms = _series_coefficients()
    dtype = big_r.dtype
    small = jnp.minimum(big_r, SERIES_EDGE)
    # Horner's rule: a power's derivative at R = 0 (0 times 0^-1) would be NaN where the series is taken.
    series_a = jnp.zeros_like(small)
    series_b = jnp.zeros_like(small)
    for coefficient_a, coefficient_b in zip(a_terms[::-1], b_terms[::-1], strict=True):
        series_a = series_a * small + coefficient_a
        series_b = series_b * small + coefficient_b
    big = jnp.maximum(big_r, SERIES_EDGE)
    decay = jnp.exp(-big)
    closed_a = (2.0 * decay * (1.0 + 1.0 / big + 1.0 / big ** 2) - 2.0 / big ** 2 - 1.0) / big
    closed_b = (-2.0 * decay * (1.0 + 3.0 / big + 3.0 / big ** 2) + 6.0 / big ** 2 - 1.0) / big
    below = big_r < SERIES_EDGE
    return jnp.where(below, series_a, closed_a), jnp.where(below, series_b, closed_b)


def screening(placement: PatchPlacement) -> Array:
    """``alpha = sqrt(shift / mu)`` (1/m), zero in the steady limit (its derivative held finite there)."""
    shift = placement.shift
    positive = shift > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, shift, 1.0) / placement.viscosity), 0.0)


def own_response(placement: PatchPlacement, forces: Array) -> Array:
    """The current step's own field of ``forces`` ``(n, 3)`` at the shell's faces, each its component, ``(capacity,)``:
    MIME's regularised Stokeslet ``((r^2 + 2 eps^2) F + (r . F) r) / (8 pi mu (r^2 + eps^2)^(3/2))`` plus the Brinkman
    difference ``alpha [f_A F + f_B rhat (rhat . F)] / (8 pi mu)`` (:func:`brinkman_difference`), summed over the force
    points in chunks of faces. Linear in ``forces``."""
    at = placement.shell_points
    count = at.shape[0]
    sources = placement.points.astype(at.dtype)
    forces = forces.astype(at.dtype)
    size = max(1, min(count, CHUNK_BYTES // (max(1, sources.shape[0]) * 12 * at.dtype.itemsize)))
    chunks = -(-count // size)
    pad = chunks * size - count
    held_at = jnp.concatenate([at, jnp.broadcast_to(at[:1], (pad, 3))]).reshape(chunks, size, 3)
    held_c = jnp.concatenate([placement.shell_components, jnp.zeros(pad, jnp.int32)]).reshape(chunks, size)
    eps2 = jnp.broadcast_to(placement.blob, (sources.shape[0],))[None, :]
    scale = 1.0 / (8.0 * math.pi * placement.viscosity)
    alpha = screening(placement)

    def chunk(pair: tuple[Array, Array]) -> Array:
        x, component = pair
        r = x[:, None, :] - sources[None, :, :]
        r2 = jnp.sum(r * r, axis=-1)
        s = scale / (r2 + eps2) ** 1.5
        along = jnp.sum(r * forces[None, :, :], axis=-1)
        force_c = forces.T[component]
        r_c = jnp.take_along_axis(r, component[:, None, None], axis=2)[..., 0]
        steady = s * ((r2 + 2.0 * eps2) * force_c + r_c * along)
        distance = jnp.sqrt(jnp.maximum(r2, jnp.finfo(at.dtype).tiny))
        part_a, part_b = brinkman_difference(alpha * distance)
        screened = alpha * scale * (part_a * force_c
                                    + part_b * r_c * along / jnp.maximum(r2, jnp.finfo(at.dtype).tiny))
        return jnp.sum(steady + screened, axis=1)

    return _lax.map(chunk, (held_at, held_c)).reshape(-1)[:count]


def source(patch: RefinementPatch, placement: PatchPlacement, forces: Array, history: PatchHistory | None = None
           ) -> Velocity:
    """The interior's drive: ``-(mu/K) u_own`` at the shell (the current step's own response, and the history's where
    given), zero elsewhere, and the history's ``shift v_prev``."""
    grid = patch.grid()
    shapes = [grid.interior_shape(c) for c in range(3)]
    total = sum(int(np.prod(s)) for s in shapes)
    own = own_response(placement, forces)
    if history is not None:
        own = own + history.own
    weight = placement.viscosity * placement.shell_weight
    flat = jnp.zeros(total, dtype=own.dtype).at[placement.shell].add(-weight * own, mode="drop")
    ends = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    parts = [flat[ends[c]:ends[c + 1]].reshape(shapes[c]) for c in range(3)]
    if history is not None:
        parts = [parts[c] + placement.shift * history.field[c] for c in range(3)]
    return (parts[0], parts[1], parts[2])


# -- the solve --------------------------------------------------------------------------------------------------------
def operators(grid: NestedMACGrid, placement: PatchPlacement) -> tuple[Callable[[Velocity], Velocity],
                                                                       Callable[[Velocity], Velocity], Velocity]:
    """``shift - mu L + mu/K`` on the interior (given values zero), its magnitude, and its diagonal."""
    mu, shift = placement.viscosity, placement.shift
    k = placement.resistivity

    def apply(x: Velocity) -> Velocity:
        lap = grid.laplacian(x)
        held = [shift * x[c] - mu * lap[c] + mu * k[c] * x[c] for c in range(3)]
        return (held[0], held[1], held[2])

    def magnitude(x: Velocity) -> Velocity:
        lap = grid.laplacian_magnitude(x)
        held = [shift * x[c] + mu * lap[c] + mu * k[c] * x[c] for c in range(3)]
        return (held[0], held[1], held[2])

    base = shift - mu * grid.laplacian_diagonal()
    return apply, magnitude, (base + mu * k[0], base + mu * k[1], base + mu * k[2])


def lifted_and_drive(patch: RefinementPatch, placement: PatchPlacement, boundary: Given, forces: Array,
                     history: PatchHistory | None = None) -> tuple[Velocity, Velocity]:
    """The solve-free part, linear in ``(boundary, forces)`` without history: the given values made compatible and
    lifted (padded, divergence-free), and the interior's right-hand side, the drive less the operator on the lift."""
    grid = patch.grid()
    lift = grid.lifted(grid.compatible(boundary))
    drive = source(patch, placement, forces, history)
    lap = grid.laplacian_padded(lift)
    inside = grid.interior_of(lift)
    mu, shift, k = placement.viscosity, placement.shift, placement.resistivity
    rhs = [drive[c] + mu * lap[c] - (shift + mu * k[c]) * inside[c] for c in range(3)]
    return lift, (rhs[0], rhs[1], rhs[2])


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
                     start: Velocity | None, stopping: Stopping, tap: Array, history: PatchHistory | None = None
                     ) -> PatchSolve:
    """One column's reflection at the read points: ``forces`` ``(n, 3)`` (N, at the placement's points), its given
    values ``boundary`` (the coarse grid's reflection there, :func:`patch_boundary`), the CG started from ``start``
    (the step's frozen field, or ``None``: zero) through the FVM's projected CG (``tap``: its report block's input).
    A column (``history`` ``None``) is linear in ``(boundary, forces)`` to the CG's tolerance; the base column takes the
    step's ``history``."""
    grid = patch.grid()
    lift, rhs = lifted_and_drive(patch, placement, boundary, forces, history)
    apply, magnitude, diagonal = operators(grid, placement)
    remainder, report = projected_solve(grid, apply, diagonal, rhs, start, stopping, tap, Jacobi(), magnitude)
    padded = grid.padded(remainder)
    field = (lift[0] + padded[0], lift[1] + padded[1], lift[2] + padded[2])
    values, outside = _read(patch, placement, field)
    return PatchSolve(values, grid.interior_of(field), report, grid.flux_share(boundary), outside)


def patch_reflection_transpose(patch: RefinementPatch, placement: PatchPlacement, cotangent: Array,
                               start: Velocity | None, stopping: Stopping, tap: Array
                               ) -> tuple[Given, Array, Array]:
    """A column's :func:`patch_reflection` transposed: a cotangent ``(n, 3)`` at the read points to the given values'
    and the forces' cotangents, and the transpose solve's report. One CG solve (the operator's own transpose)."""
    grid = patch.grid()
    dtype = cotangent.dtype
    back = _read_transpose(patch, placement, cotangent)
    apply, magnitude, diagonal = operators(grid, placement)
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
                       boundary: Given, forces: Array, start: Velocity | None, stopping: Stopping, tap: Array,
                       history: tuple[PatchHistory, PatchHistory] | None = None
                       ) -> tuple[Array, PatchSolve, PatchSolve]:
    """The two neighbouring whole-cell dilations' reflections blended by ``weight`` (the upper's, decided 4 (4)): the
    second solve only where the weight is nonzero (a step's value, the same for every column)."""
    lower = patch_reflection(patch, placements[0], boundary, forces, start, stopping, tap,
                             None if history is None else history[0])
    dtype = lower.values.dtype

    def upper(_: Any) -> PatchSolve:
        return patch_reflection(patch, placements[1], boundary, forces, start, stopping, tap,
                                None if history is None else history[1])

    def skipped(_: Any) -> PatchSolve:
        return PatchSolve(jnp.zeros_like(lower.values), lower.field, jnp.zeros(REPORT_SIZE, dtype=dtype),
                          jnp.zeros((), dtype=dtype), jnp.zeros((), dtype=lower.outside.dtype))

    higher = _lax.cond(weight > 0.0, upper, skipped, None)
    w = jnp.asarray(weight, dtype=dtype)
    return (1.0 - w) * lower.values + w * higher.values, lower, higher


def moved(field: Velocity, by: Array, fill: Velocity) -> Velocity:
    """A field on the interior faces moved by ``by`` whole cells (the new corner less the old): each entry taken from
    ``i + by``, the slab the move uncovers from ``fill`` (the same faces at the new corner)."""
    held: list[Array] = []
    for component, refill in zip(field, fill, strict=True):
        shifted = component
        inside = jnp.ones(component.shape, dtype=bool)
        for axis in range(3):
            shifted = jnp.roll(shifted, -by[axis], axis=axis)
            index = jnp.arange(component.shape[axis]) + by[axis]
            shape = [1, 1, 1]
            shape[axis] = component.shape[axis]
            inside = inside & ((index >= 0) & (index < component.shape[axis])).reshape(shape)
        held.append(jnp.where(inside, shifted, refill))
    return (held[0], held[1], held[2])


# -- the numbers F5's build report reads (scope 7) --------------------------------------------------------------------
#: The near-contact window's lines (decided 4 (2)): the contact's local radius at least this many fine cells (about
#: 7 % on the lift), the hand-over gap between these fine cells and at most this share of the local radius.
WINDOW_RADIUS_CELLS = 20.0
WINDOW_HANDOVER_CELLS = (2.0, 3.0)
WINDOW_HANDOVER_SHARE = 0.1
#: The coarse grid's own error the patch's faces carry in (its floor), measured on the axis of the straight tube by the
#: patch spike.
COARSE_FLOOR_MEASURED = (
    "measured by the patch spike on the straight periodic tube's axis, MIME's 338-point sphere, the patch's faces given "
    "by the coarse grid: wall-normal -0.8, -1.9, -3.5 % at a/R 0.1, 0.2, 0.3 for D/h 16, -0.5 and -0.9 % at a/R 0.2 and "
    "0.3 for D/h 32; wall-parallel -1.2, -3.0, -5.8 % (D/h 16) and -0.8, -1.5 % (D/h 32)")


class PatchNumbers(NamedTuple):
    """The patch's numbers at a build (scope 7): its fine cell (m) and the coarse-to-fine ratio; how far its box reaches
    beyond the body, in coarse cells (the least over the axes); the near-contact window's numbers (the contact's local
    radius in fine cells, the hand-over gap in fine cells, in local radii and in force spacings); and whether each of the
    window's lines holds (a flag, never a refusal: item 59's Q2)."""

    fine_cell: float
    ratio: float
    beyond_body_coarse_cells: float
    local_radius_cells: float
    handover_cells: float
    handover_radii: float
    handover_spacings: float
    radius_holds: bool
    handover_holds: bool


def rule_two_cell(local_radius: float) -> float:
    """The fine cell rule 2 gives for the smallest local radius served (decided 4 (6)): the radius over
    :data:`WINDOW_RADIUS_CELLS`."""
    return local_radius / WINDOW_RADIUS_CELLS


def patch_numbers(patch: RefinementPatch, coarse_spacing: float, body_reach: float, local_radius: float,
                  handover: float, force_spacing: float) -> PatchNumbers:
    """The numbers for a body of ``body_reach`` (m, its extent from its centre) with a contact of ``local_radius`` (m),
    the hand-over gap ``handover`` (m) and the MRS's ``force_spacing`` (m), on a coarse grid of ``coarse_spacing``."""
    h = patch.spacing
    beyond = (0.5 * min(patch.cells) * h - body_reach) / coarse_spacing
    radius_cells = local_radius / h
    handover_cells = handover / h
    low, high = WINDOW_HANDOVER_CELLS
    return PatchNumbers(h, coarse_spacing / h, beyond, radius_cells, handover_cells, handover / local_radius,
                        handover / force_spacing, radius_cells >= WINDOW_RADIUS_CELLS,
                        low <= handover_cells <= high and handover <= WINDOW_HANDOVER_SHARE * local_radius * (1 + 1e-12))


__all__ = ["CHUNK_BYTES", "COARSE_FLOOR_MEASURED", "FLUX_SHARE_LINE", "READS", "WINDOW_HANDOVER_CELLS",
           "WINDOW_HANDOVER_SHARE", "WINDOW_RADIUS_CELLS", "PatchNumbers", "patch_numbers", "rule_two_cell", "SERIES_EDGE", "SERIES_TERMS", "SHELL_DEPTH", "STATE_FIELDS",
           "PatchHistory", "PatchPlacement", "PatchSolve", "RefinementPatch", "blended_reflection",
           "brinkman_difference", "lifted_and_drive", "moved", "operators", "own_response", "patch_reflection",
           "patch_reflection_transpose", "patch_stopping", "placed", "screening", "snapped_origin", "source",
           "whole_cell_dilations"]
