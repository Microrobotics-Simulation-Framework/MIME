"""The moving refinement patch (the patch phase; F0's C4, its window decisions and Bundle 2's measured defaults): a fixed
fine box near a body by a wall, which hands back its near-wall reflection of the cluster's forces in place of the coarse
grid's under-resolved one in what the near field reads. The coarse solve is untouched.

**What it solves** (the patch spike's form, F2, under the step operator of the F5-patch interface): the reflection
field ``v = u - u_own`` of the cluster, ``(shift - mu L + mu/K) v + G q = shift v_prev - (mu/K) u_own``, ``D v = 0``,
every face given by the coarse grid's reflection (:func:`patch_boundary`); ``shift = rho/dt`` (zero: the steady limit),
``1/K`` the walls' band resampled from the same signed distance at the box's faces, ``u_own`` the cluster's own field
under the same operator, nonzero only where ``1/K`` is. The near field reads ``v`` over each force point's cell (its
average over the body's quadrature, :func:`cell_read`: the default, decided 4 (3)'s first route), or at the force
points (trilinear). Measured on the plane oracle at 20 fine cells across the radius and MIME's default force spacing
(0.25 radii): read at the points, the MRS's own normal overshoot at a gap of 0.1 radii is +47 % and the patch arm's
+15 %; read over the cells +2.3 % and -2.8 %, and +2.8 % and +1.4 % at 0.15 radii.

**The own field, in two parts** (the interface's (a) and (b)). The current step's: linear in the forces, the step
operator's regularised Stokeslet, MIME's regularised Stokeslet (the MRS's own blob) plus the Brinkman Stokeslet's
difference from the Stokes one at ``alpha^2 = shift / mu``, which is regular at the force point (:func:`own_response`;
the steady limit's is the regularised Stokeslet alone). The history's: the free field the cluster's past forces left,
which has no closed form; the flow's own box hands it as a lattice field, sampled at the shell's faces
(:func:`shell_sample`).

**In time: backward Euler with state** (the signed interface's default, under "exact by default": the form consistent
with the grid's own time scheme, not a measured preference). Under a time-stepped flow the patch takes the step's shift
and carries its last reflection field. What gate 10 measured in F5's structure (stage 1's corners, ``lambda = a /
sqrt(nu dt)`` 0.79 to 4.97, the patch at 10 fine cells across the radius in a periodic box of 24 coarse cells): an
agreement with a fine monolithic backward-Euler reference at the same box and coarse cell, not an accuracy. Where the
forms settle (lambda 0.79; 1.57 to 1e-4 a step), the BE patch reads +0.04 to +3.2 % and the quasi-steady patch (zero
shift, no history) +3.7 to +6.1 %; but settled, the two differ only by the own field's history (its periodic own box
against the free-space closed form), and at lambda 0.79 that gap falls with the box (2.09 points at 36 coarse cells,
1.67 at 48), while a reference at 15 fine cells across the radius moves +2.76 %, against which the quasi-steady patch
reads +0.71 %, inside the BE patch's -0.94 %. At lambda 2.48 and 4.97 no form had settled by its last step (the
reference still moving 1.6e-3 to 2.9e-3 a step), so those readings are transients. So the quasi-steady opt-in is
undecided by this evidence, not refuted: at a runner's size the time form cannot be separated from the box and the
reference's own cell. Dropping the history (the shift kept) moves the BE patch's force by -4.1 % at lambda 0.79 and
-12.9 % at 4.97: the history is load-bearing. The steady limit is the zero shift.

**Columns and the base** (the signed interface's rules, held by the functions). The patch's functions take one column of
forces and are vmappable over a column axis (no branch on a value, no shape from data). A column takes no history and
starts at zero: it is linear in its given values and its forces. The base column carries the history
(:class:`PatchHistory`): the previous step's reflection field, moved with the patch and its uncovered slab filled from
the coarse grid's previous reflection, and the own field's history at the shell; it starts at zero or from that field.
Any other start (a product's solution among them) is refused. Vmapped, the columns are the loop's program batched: they
agree with the loop as the loop agrees with itself under a one-ulp change of its input (to its tolerance at 1e-8 in
float64, to rounding at the floor).

**Snapped, never recompiling** (C4 decided 2): the box's shape is the kind's; its corner (``origin``) is whole fine
cells on the fine lattice from the flow grid's corner, state, so a move changes values and never a shape.

**The own field where the walls are.** ``u_own`` enters through ``(mu/K) u_own``, nonzero in the band and the solid, and
is evaluated on every such face (the shell), at a capacity the host plans (:func:`planned_capacity`: the same values as
every interior face at about 2.4 times fewer pairs), an overflow flagged (:func:`patch_flags`). The penalty and the drive
cover the whole solid by default. They may be cut together at a depth (an opt-in, ``solid_depth``), the solid deeper then
left to its given values, which hold it at ``v = -u_own`` only where they are consistent with the own field
(:data:`SOLID_DEPTH_MEASURED`). The drive alone may not be cut short with the penalty kept deeper: the penalty screens a
face's velocity over about ``delta_in h``, but not the pressure, which carries the solid's divergence-free constraint to
the fluid (17 % of the reflection at 4 cells [measured: ``tests/patch/test_nested_grid.py``'s shell test]).

**One uniform dilation per patch** (decided 4 (5)), by whole fine cells (decided 4 (4)): the walls' signed distance is
moved into the solid by ``l`` cells; :func:`blended_reflection` blends the two neighbouring whole-cell evaluations
linearly in the seam's continuous dilation, the second solve taken only where its weight is nonzero.

**The transpose.** The patch's CG is its own transpose solve (the operator is symmetric, its start linear in its
right-hand side): :func:`patch_reflection_transpose` reads back through the read's transpose, solves once, and
transposes the solve-free remainder (the lift and the column's source); it is the transpose to the CG's tolerance. The
own response's transpose is written by hand (:func:`own_response_transpose`, the same kernel summed over the faces into
the forces, chunk by chunk), and both that and JAX's reverse mode take it: transposing the chunked map would hold every
face-point pair's kernel, 83 GiB at stage 1's 107^3 box. A derivative in the placement (the force points, the
viscosity, the shift) goes through the map by forward mode, its chunk checkpointed, so reverse mode recomputes it."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._band import DEFAULT_BAND, DEFAULT_INTERIOR, FacedSolid, band_named, interior_depth, \
    wall_fields
from mime.nodes.flow.fvm._grid import Velocity
from mime.nodes.flow.fvm._nested import NestedMACGrid
from mime.nodes.flow.fvm._patch_transfers import Given, Quadrature, cell_read, cell_read_transpose, patch_read, \
    patch_read_transpose
from mime.nodes.flow.fvm._solve import AT_FLOOR_BY_DEFAULT, REPORT_SIZE, Jacobi, Stopping, default_tolerance, \
    projected_solve, stagnation_window

_lax: Any = jax.lax
_linear_transpose: Any = cast(Any, jax).linear_transpose
_jvp: Any = cast(Any, jax).jvp

#: How the near field reads the patch's field: at its force points, or each force point's cell average.
READS = ("point", "cell")
#: The bytes one chunk of the own response's pairs (faces against force points) may hold.
CHUNK_BYTES = 1 << 26
#: The state the host node stores for the patch (the kind owns it; MADDENING 0.4 keeps state only in nodes): the
#: corner in whole fine cells, the last step's reflection field (a column per component: its history and the next
#: step's start) and the last solve's report.
STATE_FIELDS = ("patch_origin", "patch_field_x", "patch_field_y", "patch_field_z", "patch_report")
#: The given values' net flux over their absolute flux above which a solve is flagged.
FLUX_SHARE_LINE = 1e-2
#: How deep into the solid (fine cells past the walls' dilated surface) the penalty and the own field's drive are kept,
#: by default: the whole solid. A finite depth is an opt-in (``RefinementPatch.solid_depth``): deeper, the solid is left
#: unpenalised and its velocity set by its given values, which must then be consistent with ``v = -u_own``; it saves
#: the own field's faces but costs the CG iterations, and was slower overall at stage 1's size
#: (:data:`SOLID_DEPTH_MEASURED`).
SOLID_DEPTH_CELLS = math.inf
SOLID_DEPTH_MEASURED = (
    "measured against the penalty and the drive on the whole solid: on the plane oracle (10 fine cells across the "
    "radius, a gap of 0.2 radii, 14 cells of solid in a 52^3 box, the cell read, steady) a depth of 4 cells reads "
    "-0.335 % normal and -0.037 % along the wall, 8 cells -0.051 % and -0.010 %; in the 5 mm tube at stage 1's box "
    "(Landers' sphere 0.1 a from the wall, h = a / 20, 107^3 and 120^3) a depth of 8 with given values consistent in "
    "the solid moves the reflection's metric -0.006 to -0.16 % (steady to lambda 4.97), but with the steady "
    "Stokeslets' negative as given values under a shift (inconsistent there) -10.5 %: the deep solid then follows its "
    "given values, where the whole solid's penalty pins it; and it takes 17 to 28 % more CG iterations (126 to 149 at "
    "lambda 4.97, 283 to 360 at 0.79), so a product took 0.97 to 1.15 times as long as the whole solid's at a planned "
    "capacity. The drive alone cut at d cells with the penalty kept deeper is wrong: -53.3 and -32.5 % normal at 4 and "
    "8 on the plane")
#: The margin a planned capacity holds over the most faces counted at the poses planned for.
PLAN_MARGIN = 1.1
#: Below ``alpha r`` of this the Brinkman difference is summed as its series (the closed form cancels there).
SERIES_EDGE = 1.0
#: The series' terms: the last is under ``1 / 20!`` of the first at the edge.
SERIES_TERMS = 20


@dataclass(frozen=True)
class RefinementPatch:
    """The kind: a box of ``cells`` fine cells of side ``spacing`` (m), the coarse grid's band and interior by name, the
    near field's read (:data:`READS`; the cell read takes the body's quadrature in its placement), how deep into the
    solid the penalty and the drive are kept (:data:`SOLID_DEPTH_CELLS`, the whole solid; a finite depth an opt-in) and
    the own field's capacity in faces: planned by the host at its build (:func:`planned_capacity`), an overflow flagged
    (:func:`patch_flags`); zero takes every interior face, fluid ones included (the same values, 2.4 times the pairs at
    stage 1's box). A box side with a large prime factor makes the cosine transform slow: 107 against 120, about twice
    the time a cell and iteration (two runners)."""

    cells: tuple[int, int, int]
    spacing: float
    band: str = DEFAULT_BAND
    interior: str = DEFAULT_INTERIOR
    read: str = "cell"
    shell_capacity: int = 0
    solid_depth: float = SOLID_DEPTH_CELLS

    def __post_init__(self) -> None:
        if self.read not in READS:
            raise ValueError(f"the patch's read is one of {READS}, not {self.read!r}")
        if len(self.cells) != 3 or min(self.cells) < 4:
            raise ValueError(f"the patch is a box of at least 4 cells a side, not {self.cells}")
        if not self.spacing > 0.0:
            raise ValueError(f"the patch's fine cell is positive, not {self.spacing}")
        if not self.solid_depth > 0.5 * band_named(self.band).width:
            raise ValueError(f"the patch keeps its penalty past the band's inner edge, not {self.solid_depth} cells")
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
        depth = patch.solid_depth * h
        k = fields.inverse_permeability
        phi = fields.signed_distance
        resistivity = (jnp.where(phi[0] <= depth, k[0], 0.0), jnp.where(phi[1] <= depth, k[1], 0.0),
                       jnp.where(phi[2] <= depth, k[2], 0.0))
    else:
        resistivity = grid.zeros(dtype)
    flat_k = jnp.concatenate([k.reshape(-1) for k in resistivity])
    wanted = flat_k > 0.0
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


def planned_capacity(patch: RefinementPatch, solids: Sequence[Any], origins: Sequence[Array], dilation: int = 0,
                     margin: float = PLAN_MARGIN) -> int:
    """The own field's capacity a host plans at its build: the most faces that want it (``1/K > 0`` to the patch's
    solid depth) over the corners ``origins`` it plans for, times ``margin``; a pose that needs more is flagged
    (:func:`patch_flags`), never silently short."""
    unplanned = RefinementPatch(patch.cells, patch.spacing, patch.band, patch.interior, patch.read, 0,
                                patch.solid_depth)
    points = jnp.zeros((1, 3))
    most = max(int(placed(unplanned, solids, origin, jnp.asarray(dilation), points, 0.0, 1.0).shell_needed)
               for origin in origins)
    return max(1, min(unplanned.face_count(), int(math.ceil(margin * most))))


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
    small = jnp.minimum(big_r, SERIES_EDGE)
    # Horner's rule: a power's derivative at R = 0 (0 times 0^-1) would be NaN where the series is taken.
    series_a = jnp.zeros_like(small)
    series_b = jnp.zeros_like(small)
    for coefficient_a, coefficient_b in zip(a_terms[::-1], b_terms[::-1], strict=True):
        series_a = series_a * small + float(coefficient_a)
        series_b = series_b * small + float(coefficient_b)
    big = jnp.maximum(big_r, SERIES_EDGE)
    decay = jnp.exp(-big)
    closed_a = (2.0 * decay * (1.0 + 1.0 / big + 1.0 / big ** 2) - 2.0 / big ** 2 - 1.0) / big
    closed_b = (-2.0 * decay * (1.0 + 3.0 / big + 3.0 / big ** 2) + 6.0 / big ** 2 - 1.0) / big
    below = big_r < SERIES_EDGE
    return jnp.where(below, series_a, closed_a), jnp.where(below, series_b, closed_b)


def _screening(shift: Array, viscosity: Array) -> Array:
    positive = shift > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, shift, 1.0) / viscosity), 0.0)


def screening(placement: PatchPlacement) -> Array:
    """``alpha = sqrt(shift / mu)`` (1/m), zero in the steady limit (its derivative held finite there)."""
    return _screening(placement.shift, placement.viscosity)


class _OwnPlace(NamedTuple):
    """What the own response reads of a placement: the shell's points and components, the force points, the blob,
    the viscosity and the shift."""

    at: Array
    components: Array
    sources: Array
    blob: Array
    viscosity: Array
    shift: Array


def _own_place(placement: PatchPlacement) -> _OwnPlace:
    at = placement.shell_points
    return _OwnPlace(at, placement.shell_components, placement.points.astype(at.dtype), placement.blob.astype(at.dtype),
                     placement.viscosity.astype(at.dtype), placement.shift.astype(at.dtype))


def _chunked(place: _OwnPlace) -> tuple[Array, Array, int]:
    """The shell's points and components in chunks of faces (padded with the first face), and the chunk count."""
    at = place.at
    count = at.shape[0]
    size = max(1, min(count, CHUNK_BYTES // (max(1, place.sources.shape[0]) * 12 * at.dtype.itemsize)))
    chunks = -(-count // size)
    pad = chunks * size - count
    held_at = jnp.concatenate([at, jnp.broadcast_to(at[:1], (pad, 3))]).reshape(chunks, size, 3)
    held_c = jnp.concatenate([place.components, jnp.zeros(pad, jnp.int32)]).reshape(chunks, size)
    return held_at, held_c, chunks


def _kernel(place: _OwnPlace, x: Array) -> tuple[Array, Array, Array]:
    """A chunk's kernel against every force point: ``r`` ``(size, n, 3)`` and the coefficients of ``F_c`` and of
    ``r_c (r . F)``, ``(size, n)``: the regularised Stokeslet's and the Brinkman difference's together."""
    dtype = x.dtype
    tiny = jnp.finfo(dtype).tiny
    eps2 = jnp.broadcast_to(place.blob, (place.sources.shape[0],))[None, :]
    scale = 1.0 / (8.0 * math.pi * place.viscosity)
    alpha = _screening(place.shift, place.viscosity)
    r = x[:, None, :] - place.sources[None, :, :]
    r2 = jnp.sum(r * r, axis=-1)
    s = scale / (r2 + eps2) ** 1.5
    part_a, part_b = brinkman_difference(alpha * jnp.sqrt(jnp.maximum(r2, tiny)))
    return r, s * (r2 + 2.0 * eps2) + alpha * scale * part_a, s + alpha * scale * part_b / jnp.maximum(r2, tiny)


def _own_values(place: _OwnPlace, forces: Array) -> Array:
    """The own response computed: the kernel summed over the force points, chunk by chunk of faces. The chunk is
    checkpointed, so that a derivative in the placement taken in reverse recomputes each chunk's kernel rather than
    holding every face-point pair's."""
    held_at, held_c, _ = _chunked(place)

    @cast(Any, jax).checkpoint
    def chunk(pair: tuple[Array, Array]) -> Array:
        x, component = pair
        r, diagonal, outer = _kernel(place, x)
        r_c = jnp.take_along_axis(r, component[:, None, None], axis=2)[..., 0]
        along = jnp.sum(r * forces[None, :, :], axis=-1)
        return jnp.sum(diagonal * forces.T[component] + outer * r_c * along, axis=1)

    return _lax.map(chunk, (held_at, held_c)).reshape(-1)[:place.at.shape[0]]


def _own_transposed(place: _OwnPlace, cotangent: Array) -> Array:
    """The own response's transpose, by hand: each chunk's kernel against the force points, weighted by the chunk's
    cotangent and summed over its faces into the forces' cotangent ``(n, 3)``, carried from chunk to chunk (the
    kernel is symmetric in its two components, so it is the same kernel)."""
    held_at, held_c, chunks = _chunked(place)
    count = place.at.shape[0]
    held_t = jnp.concatenate([cotangent, jnp.zeros(chunks * held_at.shape[1] - count, cotangent.dtype)]
                             ).reshape(chunks, -1)

    def chunk(total: Array, triple: tuple[Array, Array, Array]) -> tuple[Array, None]:
        x, component, weight = triple
        r, diagonal, outer = _kernel(place, x)
        r_c = jnp.take_along_axis(r, component[:, None, None], axis=2)[..., 0]
        picked = (component[:, None] == jnp.arange(3)[None, :]).astype(x.dtype)
        part = (diagonal * weight[:, None]).T @ picked + jnp.einsum("fp,fpj->pj", outer * r_c * weight[:, None], r)
        return total + part, None

    total, _ = _lax.scan(chunk, jnp.zeros(place.sources.shape, cotangent.dtype), (held_at, held_c, held_t))
    return total


def _own_linear(place: _OwnPlace, forces: Array) -> Array:
    """The own response as a linear map of ``forces`` whose transpose is :func:`_own_transposed`: a linear solve whose
    operator is the identity on ``(forces, values)`` pairs and whose solve is the response, so that JAX transposes it
    by the hand-written transpose and never transposes the chunked map."""
    # The pair's second part is the forces' too (an exact zero times one of them): a transposed solve's whole
    # right-hand side is linear in what is transposed.
    pair = (forces, jnp.zeros(place.at.shape[0], forces.dtype) + 0.0 * forces[0, 0])

    def solve(_: Any, held: tuple[Array, Array]) -> tuple[Array, Array]:
        return jnp.zeros_like(held[0]), _own_values(place, held[0])

    def transpose_solve(_: Any, held: tuple[Array, Array]) -> tuple[Array, Array]:
        return _own_transposed(place, held[1]), jnp.zeros_like(held[1])

    def identity(held: tuple[Array, Array]) -> tuple[Array, Array]:
        return held

    return _lax.custom_linear_solve(identity, pair, solve, transpose_solve)[1]


@cast(Any, jax).custom_jvp
def _own(place: _OwnPlace, forces: Array) -> Array:
    return _own_linear(place, forces)


def _own_jvp(primals: tuple[_OwnPlace, Array], tangents: tuple[Any, Any]) -> tuple[Array, Array]:
    """The forces' tangent through the linear map (so reverse mode takes the hand-written transpose); the placement's
    by forward mode through the checkpointed map."""
    place, forces = primals
    d_place, d_forces = tangents
    values = _own_linear(place, forces)
    zero = cast(Any, jax).custom_derivatives.SymbolicZero
    tangent = jnp.zeros_like(values) if isinstance(d_forces, zero) else _own_linear(place, d_forces)
    moved = [k for k, t in enumerate(d_place)
             if not isinstance(t, zero) and jnp.issubdtype(place[k].dtype, jnp.floating)]
    if moved:
        def of(*held: Array) -> Array:
            parts = list(place)
            for k, value in zip(moved, held, strict=True):
                parts[k] = value
            return _own_values(_OwnPlace(*parts), forces)

        _, by_place = _jvp(of, tuple(place[k] for k in moved), tuple(d_place[k] for k in moved))
        tangent = tangent + by_place
    return values, tangent


_own.defjvp(_own_jvp, symbolic_zeros=True)


def own_response(placement: PatchPlacement, forces: Array) -> Array:
    """The current step's own field of ``forces`` ``(n, 3)`` at the shell's faces, each its component, ``(capacity,)``:
    MIME's regularised Stokeslet ``((r^2 + 2 eps^2) F + (r . F) r) / (8 pi mu (r^2 + eps^2)^(3/2))`` plus the Brinkman
    difference ``alpha [f_A F + f_B rhat (rhat . F)] / (8 pi mu)`` (:func:`brinkman_difference`), summed over the force
    points in chunks of faces. Linear in ``forces``, its transpose by hand (:func:`own_response_transpose`), which
    reverse mode and :func:`patch_reflection_transpose` take: transposing the chunked map would hold every
    face-point pair's kernel (83 GiB at 107^3 cells)."""
    place = _own_place(placement)
    return cast(Array, _own(place, forces.astype(place.at.dtype)))


def own_response_transpose(placement: PatchPlacement, cotangent: Array) -> Array:
    """:func:`own_response`'s transpose: a cotangent at the shell's faces ``(capacity,)`` to the forces' ``(n, 3)``."""
    place = _own_place(placement)
    return _own_transposed(place, cotangent.astype(place.at.dtype))


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


def _start(start: Velocity | None, history: PatchHistory | None) -> Velocity | None:
    """The F5-patch interface's starts, held by the function: a column starts at zero; the base column at zero or from
    its own history's field (the frozen step-start field, moved with the patch), passed as ``history.field``; any other
    start, a product's solution among them, is refused."""
    if start is None:
        return None
    if history is None:
        raise ValueError("a column of the patch starts at zero: only the base column, with its history, takes a start")
    if start is not history.field:
        raise ValueError("the base column starts only from its history's own field (history.field, the frozen "
                         "step-start field), never from another field such as a product's solution")
    return history.field


def patch_reflection(patch: RefinementPatch, placement: PatchPlacement, boundary: Given, forces: Array,
                     start: Velocity | None, stopping: Stopping, tap: Array, history: PatchHistory | None = None
                     ) -> PatchSolve:
    """One column's reflection at the read points: ``forces`` ``(n, 3)`` (N, at the placement's points), its given
    values ``boundary`` (the coarse grid's reflection there, :func:`patch_boundary`), through the FVM's projected CG
    (``tap``: its report block's input). A column (``history`` ``None``) starts at zero (``start`` ``None``; any other
    is refused) and is linear in ``(boundary, forces)`` to the CG's tolerance. The base column takes the step's
    ``history``, and starts at zero or from ``history.field`` (``start=history.field``; :func:`_start`)."""
    start = _start(start, history)
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
    and the forces' cotangents, and the transpose solve's report. One CG solve (the operator's own transpose), started
    at zero as a column is (``start`` ``None``; any other is refused); the own response's transpose is the hand-written
    one (:func:`own_response_transpose`)."""
    _start(start, None)
    grid = patch.grid()
    dtype = cotangent.dtype
    back = _read_transpose(patch, placement, cotangent)
    apply, magnitude, diagonal = operators(grid, placement)
    solved, report = projected_solve(grid, apply, diagonal, grid.interior_of(back), None, stopping, tap, Jacobi(),
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
    second solve only where the weight is nonzero (a step's value, the same for every column); where it is skipped,
    the upper hands back the lower's values, so the blend's derivative in the weight is zero there, not ``-R(l)``.
    Starts as :func:`patch_reflection`'s: ``start`` ``None``, or the base column's ``history[0].field`` (each
    dilation then starts from its own history's field)."""
    lower = patch_reflection(patch, placements[0], boundary, forces, start, stopping, tap,
                             None if history is None else history[0])
    dtype = lower.values.dtype

    def upper(_: Any) -> PatchSolve:
        held = None if history is None else history[1]
        return patch_reflection(patch, placements[1], boundary, forces, None if start is None or held is None
                                else held.field, stopping, tap, held)

    def skipped(_: Any) -> PatchSolve:
        return PatchSolve(lower.values, lower.field, jnp.zeros(REPORT_SIZE, dtype=dtype), jnp.zeros((), dtype=dtype),
                          jnp.zeros((), dtype=lower.outside.dtype))

    higher = _lax.cond(weight > 0.0, upper, skipped, None)
    w = jnp.asarray(weight, dtype=dtype)
    return (1.0 - w) * lower.values + w * higher.values, lower, higher


class PatchFlags(NamedTuple):
    """What a step flags of a patch solve (never a refusal): the own field's faces past the capacity (dropped, the
    reflection then wrong by up to tens of per cent), the given values' net flux over their absolute flux above
    :data:`FLUX_SHARE_LINE`, and read points outside the box."""

    overflow: Array
    flux: Array
    outside: Array


def patch_flags(patch: RefinementPatch, placement: PatchPlacement, solve: PatchSolve) -> PatchFlags:
    """The three flags of a solve at a placement."""
    return PatchFlags(placement.shell_needed > patch.capacity(), jnp.abs(solve.flux_share) > FLUX_SHARE_LINE,
                      solve.outside > 0)


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
    window's lines holds (a flag, never a refusal: item 59's Q2), the read's among them: the point read needs a
    hand-over gap of a force spacing or more (:data:`POINT_READ_SPACINGS`), the cell read holds to under half of one."""

    fine_cell: float
    ratio: float
    beyond_body_coarse_cells: float
    local_radius_cells: float
    handover_cells: float
    handover_radii: float
    handover_spacings: float
    radius_holds: bool
    handover_holds: bool
    read_holds: bool


#: The hand-over gap, in force spacings, below which the point read is flagged: at 0.4 spacings (0.1 radii at MIME's
#: default spacing) it overshoots the normal entry by 47 %, at 0.6 by 7 %; the cell read, +2.25 % at 0.4.
POINT_READ_SPACINGS = 1.0


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
                        low <= handover_cells <= high
                        and handover <= WINDOW_HANDOVER_SHARE * local_radius * (1 + 1e-12),
                        patch.read != "point" or handover / force_spacing >= POINT_READ_SPACINGS)


__all__ = ["CHUNK_BYTES", "COARSE_FLOOR_MEASURED", "FLUX_SHARE_LINE", "PLAN_MARGIN", "POINT_READ_SPACINGS", "READS",
           "SERIES_EDGE", "SERIES_TERMS", "SOLID_DEPTH_CELLS", "SOLID_DEPTH_MEASURED", "STATE_FIELDS",
           "WINDOW_HANDOVER_CELLS", "WINDOW_HANDOVER_SHARE", "WINDOW_RADIUS_CELLS", "PatchFlags", "PatchHistory",
           "PatchNumbers", "PatchPlacement", "PatchSolve", "RefinementPatch", "blended_reflection",
           "brinkman_difference", "lifted_and_drive", "moved", "operators", "own_response", "own_response_transpose",
           "patch_flags", "patch_numbers", "patch_reflection", "patch_reflection_transpose", "patch_stopping",
           "placed", "planned_capacity", "rule_two_cell", "screening", "snapped_origin", "source",
           "whole_cell_dilations"]
