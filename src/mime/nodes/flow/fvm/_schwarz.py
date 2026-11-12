"""An MRS body coupled to the flow's grid by Schwarz (F5; the record's C4 and C1): the body's near field in free space
(the joint near field's matrix, W3a's), the walls carried by the grid, the body's own field on the grid removed by a
free box, and the coupled response handed to the body (#351).

**The system a step solves.** The body's tractions ``t`` (``3N``: each cell's force per area, its force at its force
point ``F = a t``) solve, for a twist ``V'`` of the body,

    ``K t + W t = M V' - u_b``,

``K`` the near field's matrix at the step's pose and viscosity (free space), ``M`` the body's rigid motions at its force
points, ``W t`` the walls' reflection of the tractions read at the force points and ``u_b`` the flow without the body
there. The reflection is the walled grid's response to the forces less the body's own field in a free box:
``W t = G u_w(S F) - G u_o(S F) - c(F)``, ``S`` and ``G`` the exchange kernel's scatter and gather
(``mime.nodes.flow.fvm._exchange``, its average), ``u_w`` one solve of the flow's own step from rest
(:func:`prepared`'s ``walled``: the walls, the solid and, in an open vessel, the openings), ``u_o`` the free periodic box's
response by the same step operator (:func:`own_response`: the grid's spacing and kernel, aligned with the grid's lattice,
so that the two discretise the body's own field alike) and ``c`` the box's periodic images, Hasimoto's term relaxed at the
box's slowest diffusion rate (:func:`image_terms`: route 2's, as the point body's own field holds it). ``u_b`` is the
same difference for the flow the step solves without the body's forces: the walled base field less the box's own
field carried from the steps before.

**Seven columns.** The body's six rigid motions and its load at zero twist (``-u_b``), solved together by the shared
GMRES's operator form (``mime.nodes.hydro.near_field._gmres.reported_operator_solve``), right-preconditioned by the near
field's own inverse, each column to its tolerance on its true residual; the products' own solves (the walled CG) report
into the solve's second block. The tractions are ``t = T V' + t0``, ``T`` the six columns and ``t0`` the seventh: an
``Eliminated`` (``mime.core.elimination``) over the body's twist, the free unknown. The body is handed its load and
response, ``F_b = A0 - R V'`` with ``R = W^T T`` and ``A0 = -W^T t0`` (``W`` the body's wrench of tractions), which its
own step solves through with its other loads, its contact and its friction (the dynamics owns its step).

**The composition** (:func:`composed`): once the body has taken its step, its twist closes the tractions, and one more
walled solve adds their field to the base: the flow's stored field is the solved one. The free box's own field and its
image term advance by the same forces. The near field's equations are then read at the composed field, at every force
point (:func:`composed`'s residual): every unknown of the step, never a subset of them (the 0.2 line's Schwarz stopped
on an interface norm over a subset and sat 2.6e-4 off its fixed point while reporting arrival).

**What is frozen within a step** (one function of the tractions for GMRES and its transpose): the body's pose and force
points, the stencils on the grid and the box, the step's shift and viscosity, the base field, the box's carried field
and the near field's matrix. The tractions, the twist and (in an open vessel) the openings' pressures float."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
from jax import Array

from mime.core import jnp
from mime.core.elimination import Eliminated
from mime.core.solves import SolveSite, register, solving
from mime.nodes.flow.fvm._exchange import OwnBox, gather, image_term, own_box, scatter
from mime.nodes.flow.fvm._grid import ExchangeStencil, MACGrid, Velocity
from mime.nodes.hydro.mrs._solve import PRECISION, force_sums
from mime.nodes.hydro.near_field._contract import Placed
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Operator, Stopping, reduced, reported_operator_solve

#: The free box's transforms: the body's own field under the step operator, and its carried field's decay (a transform
#: that inverts an operator diagonal in it).
FREE_BOX_SITE = register(SolveSite("schwarz_free_box", "transform", "mime.nodes.flow.fvm._schwarz.own_response",
                                   nodes=("ImplicitFlowNode",)))

#: The columns a step solves: the body's six rigid motions, then its load at zero twist.
COLUMNS = 7
#: The free box's least side in cells, and its side in the body's extent plus the kernel's reach on each side: a power of
#: two at least this many times that (the coupling spike's box was twice the tube's diameter, 64 cells at D/h 32).
LEAST_SIDE = 32
SIDE_EXTENTS = 2
#: The inner solves' tolerance as a share of the outer GMRES's: the walled CG within each product (C1 point 2), so that
#: the inexactness it puts in the operator, about ``tau_in |W| |t|``, stays two decades under the outer tolerance.
#: Provisional: the solve-mask spike's arm B sets it.
INNER_SHARE = 1e-2
#: The composed residual's line, in the outer tolerance: above it a step is flagged (``schwarz_flagged``).
RESIDUAL_LINE = 10.0
#: The columns' GMRES tolerance by the run's dtype where none is written: float64's the coupling spike's (its
#: products two decades below at the inner share); float32's above ten times the CG's reachable floor (about 1e-5).
#: Provisional with :data:`INNER_SHARE`.
SCHWARZ_TOLERANCE = {"float64": 1e-8, "float32": 1e-4}


def box_side(extent: float, spacing: float, reach_cells: int) -> int:
    """The free box's side in cells for a body ``extent`` across (m): a power of two, at least :data:`LEAST_SIDE` and at
    least :data:`SIDE_EXTENTS` times the body's extent in cells plus the kernel's reach each side."""
    cells = extent / spacing + 2 * reach_cells
    return max(LEAST_SIDE, int(2 ** math.ceil(math.log2(SIDE_EXTENTS * cells))))


def own_response(box: OwnBox, density: Velocity, shift: Any, nu: Any, rho: Any) -> Velocity:
    """The free periodic box's response to a force density (N/m^3) on its three face sets under the step operator,
    ``(shift - nu L) u + grad p = f / rho``, ``div u = 0``, its mean removed: the transform of each face set times
    the projection over ``rho (shift + nu lambda)`` (the MAC grid's Laplacian and projector are diagonal in it)."""
    with solving(FREE_BOX_SITE):
        n = box.side
        lam = box.eigenvalue
        safe = jnp.where(lam == 0.0, 1.0, lam)
        hats = [jnp.fft.rfftn(d) for d in density]
        divergence = sum(s * f for s, f in zip(box.symbols, hats, strict=True)) / safe
        denominator = rho * (shift + nu * safe)
        out = [jnp.fft.irfftn(jnp.where(lam == 0.0, 0.0, (f - jnp.conj(s) * divergence) / denominator), s=(n, n, n))
               for s, f in zip(box.symbols, hats, strict=True)]
        return (out[0].astype(density[0].dtype), out[1].astype(density[0].dtype), out[2].astype(density[0].dtype))


def own_decayed(box: OwnBox, field: Velocity, shift: Any, nu: Any) -> Velocity:
    """The box's carried field one step on with no force: each wavenumber times ``shift / (shift + nu lambda)`` (zero in
    the steady limit, where nothing is carried)."""
    with solving(FREE_BOX_SITE):
        n = box.side
        lam = box.eigenvalue
        decay = jnp.where(lam == 0.0, 0.0, shift / (shift + nu * jnp.where(lam == 0.0, 1.0, lam)))
        out = [jnp.fft.irfftn(decay * jnp.fft.rfftn(f), s=(n, n, n)).astype(f.dtype) for f in field]
        return (out[0], out[1], out[2])


def image_terms(side: int, spacing: Any, shift: Any, nu: Any, mu: Any) -> tuple[Any, Any]:
    """The box's periodic images as one mode (route 2's, ``mime.nodes.flow.fvm._exchange.image_mode``): the velocity
    per unit net force the step's own force gives at the body (``now``, m/(N s)), and the decay a step of what the earlier
    steps' forces left (``decay``); in the steady limit Hasimoto's whole term, ``c / (6 pi mu L)``, and nothing
    carried."""
    smallest = 4.0 * math.sin(math.pi / side) ** 2 / spacing ** 2
    rate = nu * smallest
    return image_term(side) / (mu * spacing) * rate / (shift + rate), shift / (shift + rate)


class SchwarzBody(NamedTuple):
    """The body as the step holds it, frozen at the step's start: its placement in the world (force points, cells'
    centroids and areas, centre), its near field's matrix at the step's viscosity (``3N x 3N``, free space) and that
    matrix's inverse as a preconditioner and its transpose."""

    placed: Placed
    matrix: Array
    precondition: Operator
    transposed: Operator


class Stencils(NamedTuple):
    """The body's force points' exchange stencils on the grid's face sets and on the free box's."""

    grid: tuple[ExchangeStencil, ExchangeStencil, ExchangeStencil]
    box: tuple[ExchangeStencil, ExchangeStencil, ExchangeStencil]


class Prepared(NamedTuple):
    """A step's preparation: the tractions of the seven columns (``3N x 7``), the response and load handed to the body
    (``R`` and ``A0``), the GMRES's two report blocks, and the body's elimination over its twist (``t = t0 + T V'``)."""

    columns: Array
    resistance: Array
    load: Array
    report: Array

    def eliminated(self) -> Eliminated:
        return Eliminated(self.columns[:, COLUMNS - 1], self.columns[:, :COLUMNS - 1])


class Composed(NamedTuple):
    """A step composed from the body's twist: the tractions, the walled field, the box's field and its image term at
    the step's end (neither decayed: the next step's preparation decays them, :func:`own_decayed`), the composition's
    solve report, the near field's residual at the composed field (relative to
    the equations' scale) and the forces' field on each face set (N/m^3, for the step's identity)."""

    tractions: Array
    field: Velocity
    own: Velocity
    image: Array
    report: Array
    residual: Array
    density: Velocity


def stencils(grid: MACGrid, box: OwnBox, points: Array, width: float) -> Stencils:
    """The kernel's stencils at ``points`` (m, the grid's frame) on the grid and on the box (``points`` taken modulo the
    box, whose lattice the grid's shares)."""
    held = MACGrid((box.side, box.side, box.side), box.spacing)
    on_grid = tuple(grid.exchange_stencil(c, points, width) for c in range(3))
    on_box = tuple(held.exchange_stencil(c, points, width) for c in range(3))
    return Stencils((on_grid[0], on_grid[1], on_grid[2]), (on_box[0], on_box[1], on_box[2]))


def _scattered(grid: MACGrid, held: Sequence[ExchangeStencil], forces: Array) -> Velocity:
    out = [scatter(grid, c, forces[:, c], held[c]) for c in range(3)]
    return (out[0], out[1], out[2])


def _gathered(grid: MACGrid, held: Sequence[ExchangeStencil], field: Velocity) -> Array:
    return jnp.stack([gather(grid, c, field[c], held[c]) for c in range(3)], axis=1)


def _box_grid(box: OwnBox) -> MACGrid:
    return MACGrid((box.side, box.side, box.side), box.spacing)


def _forces(placed: Placed, tractions: Array) -> Array:
    """Each force point's force (N, ``(N, 3)``, a trailing column axis kept) of tractions ``(3N, ...)``."""
    held = tractions.reshape(placed.count, 3, *tractions.shape[1:])
    return placed.area.reshape(placed.count, 1, *([1] * (tractions.ndim - 1))) * held


def reflection(grid: MACGrid, box: OwnBox, held: Stencils, walled: Callable[[Velocity], tuple[Velocity, Array]],
               forces: Array, shift: Any, nu: Any, rho: Any, image_now: Any) -> tuple[Array, Array]:
    """The walls' reflection of one column of forces ``(N, 3)`` at the force points (``(N, 3)``, m/s): the walled
    step's response from rest less the box's own and its images', and the walled solve's report (the coarse form of
    the reflection seam; the patch replaces it near a wall)."""
    walls, report = walled(tuple(d / rho for d in _scattered(grid, held.grid, forces)))
    boxed = own_response(box, _scattered(_box_grid(box), held.box, forces), shift, nu, rho)
    images = image_now * jnp.sum(forces, axis=0)
    return _gathered(grid, held.grid, walls) - _gathered(_box_grid(box), held.box, boxed) - images[None, :], report


def background(grid: MACGrid, box: OwnBox, held: Stencils, base: Velocity, carried: Velocity,
               image: Array) -> Array:
    """``u_b`` at the force points (``3N``): the walled base field less the box's carried field and image history."""
    return (_gathered(grid, held.grid, base) - _gathered(_box_grid(box), held.box, carried)
            - image[None, :]).reshape(-1)


def _reduced_all(reports: Array) -> Array:
    total = reports[0]
    for k in range(1, reports.shape[0]):
        total = reduced(total, reports[k])
    return total


def prepared(grid: MACGrid, box: OwnBox, held: Stencils, body: SchwarzBody,
             walled: Callable[[Velocity], tuple[Velocity, Array]], base: Velocity, carried: Velocity, image: Array,
             shift: Any, nu: Any, rho: Any, image_now: Any, stopping: Stopping, tap: Array,
             magnitude: Operator | None = None) -> Prepared:
    """The step's seven columns by GMRES (the module states the system), and what the body is handed. ``walled`` is
    one walled solve of the step from rest (its inner tolerance the caller's), ``carried`` the box's field and ``image``
    its image history at the step's start (each decayed a step), ``tap`` the report blocks' input (``2 x``
    :data:`REPORT_SIZE` zeros)."""
    placed = body.placed

    def product(t: Array) -> tuple[Array, Array]:
        def column(forces: Array) -> tuple[Array, Array]:
            found, report = reflection(grid, box, held, walled, forces, shift, nu, rho, image_now)
            return found.reshape(-1), report

        reflected, reports = jax.vmap(column, in_axes=2, out_axes=(1, 0))(_forces(placed, t))
        return jnp.matmul(body.matrix, t, precision=PRECISION) + reflected, _reduced_all(reports)

    right = jnp.concatenate([placed.motions(), -background(grid, box, held, base, carried, image)[:, None]], axis=1)
    columns, report = reported_operator_solve(product, right, tap, body.precondition, body.transposed, stopping,
                                              magnitude)
    sums = force_sums(placed.wrench(), columns)
    return Prepared(columns, sums[:, :COLUMNS - 1], -sums[:, COLUMNS - 1], report)


def composed(grid: MACGrid, box: OwnBox, held: Stencils, body: SchwarzBody, step: Prepared, twist: Array,
             walled: Callable[[Velocity], tuple[Velocity, Array]], base: Velocity, carried: Velocity, image: Array,
             shift: Any, nu: Any, rho: Any, image_now: Any) -> Composed:
    """The step closed by the body's ``twist``: its tractions ``t0 + T V'``, the walled field ``u_base + u_w(S F)`` (one
    walled solve, ``walled`` at the flow's own tolerance), the box's field and image term after the step, and the
    near field's equations read at the composed field, ``K t + G u - G u_o - c - M V'``, over the larger of ``|K t|``
    and ``|M V'|`` (every force point, every component)."""
    placed = body.placed
    tractions = step.columns[:, COLUMNS - 1] + jnp.matmul(step.columns[:, :COLUMNS - 1], twist, precision=PRECISION)
    forces = _forces(placed, tractions)
    density = _scattered(grid, held.grid, forces)
    response, report = walled(tuple(d / rho for d in density))
    field = (base[0] + response[0], base[1] + response[1], base[2] + response[2])
    boxed = own_response(box, _scattered(_box_grid(box), held.box, forces), shift, nu, rho)
    own = (carried[0] + boxed[0], carried[1] + boxed[1], carried[2] + boxed[2])
    total = jnp.sum(forces, axis=0)
    after = image + image_now * total
    near = jnp.matmul(body.matrix, tractions, precision=PRECISION)
    motion = jnp.matmul(placed.motions(), twist, precision=PRECISION)
    remainder = near + (_gathered(grid, held.grid, field) - _gathered(_box_grid(box), held.box, own)
                        - after[None, :]).reshape(-1) - motion
    scale = jnp.maximum(jnp.linalg.norm(near), jnp.linalg.norm(motion))
    residual = jnp.where(scale > 0.0, jnp.linalg.norm(remainder) / jnp.where(scale > 0.0, scale, 1.0), 0.0)
    return Composed(tractions, field, own, after, report, residual, density)


def free_box(extent: float, spacing: float, reach_cells: int, dtype: Any) -> OwnBox:
    """The free box for a body ``extent`` across (m) on a grid of ``spacing``: route 2's box (its transform's half
    spectrum) of :func:`box_side` cells."""
    return own_box(box_side(extent, spacing, reach_cells), spacing, dtype)


__all__ = ["COLUMNS", "FREE_BOX_SITE", "INNER_SHARE", "LEAST_SIDE", "REPORT_SIZE", "RESIDUAL_LINE", "SCHWARZ_TOLERANCE",
           "SIDE_EXTENTS",
           "Composed", "Prepared", "SchwarzBody", "Stencils", "background", "box_side", "composed", "free_box",
           "image_terms", "own_decayed", "own_response", "prepared", "reflection", "stencils"]
