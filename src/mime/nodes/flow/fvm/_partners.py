"""Point bodies as partners of the flow's one elimination (F3's second PR; C5 point 3; the record's B8, body half): a
point body's drag law held in the flow's own step, its force and couple solved with the field (one pass), the
openings beside it where the vessel is open.

**A body's deliveries and readings** are the exchange law's (``mime.nodes.flow.fvm._exchange``): its force ``f`` and
couple ``T`` reach the grid through the Gaussian kernel's transpose pair, ``S_F f = S f + L S (c f)`` with Faxén's
coefficient ``c = a^2/6 - sigma^2/2`` and the kernel's antisymmetric dipole for the couple; the field is read back
through their transposes, ``G_F u = G u + c G L u`` and ``Omega = (1/2) G curl u`` (:func:`delivered`, :func:`read`),
so the exchange is workless at one field: ``<S_F f, u> + <dipole T, u> = f . G_F u + T . Omega``.

**The step** (:func:`partnered_step`): the base field (every body's delivery zero, each opening at the pressure the
last step solved for), one unit field a delivery (each opening but the reference's move, each body's six), every unit
field read by every partner (the cross terms), and one elimination (``mime.core.elimination``):

- the openings' rows (``mime.nodes.flow.fvm._ports.opening_rows``);
- each body's drag law, ``f = R0 (V' - u_d)`` with ``u_d = G_F u' - M_inf f`` (the constant correction of the
  kernel's own steady field), its couple likewise: ``(M0 - M_inf) f + G_F u' = V'``, ``M0 = 1/R0`` (the law's block,
  :class:`BodyPartner`'s ``law``), its twist ``V'`` left free;
- each body's own step, which closes its twist: ``S V' + f = r``, ``S = M'/dt + R`` (the body's stepping matrix and
  the resistances of its other laws), ``r = M' V / dt + g + F`` (its free load, its gyroscopic term with it, and its
  other loads), as the body publishes and its other laws deliver them (:class:`BodyPartner`'s ``system``,
  ``right``).

The field is then the base plus each unit field times its solved delivery: its stored field and kinetic energy are
the solved ones, within the step. Each body is handed its load and response with the twist left free (the
elimination's sensitivity: ``F = A0 - R_h V'``, ``R_h`` positive definite where the coupled one-step margin is), which
its own step folds into the matrix it solves through (its own integrator: the dynamics owns its step), so its twist is
the field's to rounding where the body's own step is the row the field closed, and the exchange is workless.

**What it does not do.** A body whose own step is not linear in its twist (a hosted contact, a confinement) has no row
the field can close: the plan gives it the lagged exchange. Several bodies' joint step is F5's (the body-set host); the
step here closes every body's row the field holds, each handed its own response with the others' twists at their
solved values."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from jax import Array

from mime.core import jnp
from mime.core.elimination import PartnerRows, closed, exchange_system
from mime.nodes.flow.fvm._exchange import gather, scatter
from mime.nodes.flow.fvm._grid import ExchangeStencil, MACGrid, Velocity
from mime.nodes.flow.fvm._ports import PortSolution, RowKind, opening_rows, openings_solved

#: A point body's deliveries into its field: its force's three components, then its couple's three.
DELIVERIES = 6
#: A partner's solution as the flow holds it (``flow.partner_solution``), a row a body: its force and couple (N; N m),
#: its twist as the field closed it (m/s; rad/s) and its kernel's reading of the solved field (:func:`kernel_reading`).
SOLUTION = ("force", "twist", "kernel")
#: The columns of a row of the solution.
SOLUTION_COLUMNS = 2 * DELIVERIES + 9


class BodyPartner(NamedTuple):
    """One point body's law and own row over the step: its point (m, in the grid's frame), its kernel's width (cells,
    structural), the Faxén coefficients of its reading and its delivery (m^2), its law's block ``diag(M0 - M_inf,
    K0 - K_inf)`` (six, ``(m/s)/N`` and ``(rad/s)/(N m)``), and its own step's matrix and right-hand side."""

    point: Array
    width: float
    reading: Array
    delivery: Array
    law: Array
    system: Array
    right: Array


def stencils(grid: MACGrid, point: Array, width: float) -> tuple[ExchangeStencil, ...]:
    """The exchange kernel's stencil at ``point`` on each face set."""
    return tuple(grid.exchange_stencil(component, point[None, :], width) for component in range(3))


def delivered(grid: MACGrid, held: Sequence[ExchangeStencil], values: Array, delivery: Any) -> Velocity:
    """The force density (N/m^3) on each face set of a body's force and couple ``values`` (six): ``S f + L S (c f)``
    and the couple's dipole, the transposes of :func:`read`'s."""
    force, couple = values[:3], values[3:]
    out: list[Array] = []
    for component in range(3):
        stencil = held[component]
        density = scatter(grid, component, force[component:component + 1], stencil, "average")
        density = density + scatter(grid, component, delivery * force[component:component + 1], stencil, "laplacian")
        out.append(density + scatter(grid, component, couple[None, :], stencil, "curl"))
    return (out[0], out[1], out[2])


def kernel_reading(grid: MACGrid, held: Sequence[ExchangeStencil], velocity: Velocity) -> Array:
    """The kernel's reading of ``velocity`` (nine): its average ``G u`` and its Laplacian's ``G L u`` on each face set
    (m/s; 1/(m s)), then the fluid's rotation ``(1/2) G curl u`` (rad/s), each face set's share summed."""
    average = [gather(grid, component, velocity[component], held[component], "average")[0] for component in range(3)]
    laplacian = [gather(grid, component, velocity[component], held[component], "laplacian")[0] for component in range(3)]
    rotation = sum((gather(grid, component, velocity[component], held[component], "curl")[0] for component in range(3)),
                   start=jnp.zeros(3, dtype=velocity[0].dtype))
    return jnp.concatenate([jnp.stack(average), jnp.stack(laplacian), rotation])


def read(grid: MACGrid, held: Sequence[ExchangeStencil], velocity: Velocity, reading: Any) -> Array:
    """A body's reading of ``velocity`` (six): ``G u + c G L u`` on each face set (m/s), then the fluid's rotation
    ``(1/2) G curl u`` (rad/s): the transposes of :func:`delivered`'s."""
    kernel = kernel_reading(grid, held, velocity)
    return jnp.concatenate([kernel[:3] + reading * kernel[3:6], kernel[6:]])


class Openings(NamedTuple):
    """The openings as the step's partner: their row kinds, the laws' ``a`` and blocks ``b`` over the stage, the
    relative pressures the step's base was solved at, the gauge's reference, which of ``b``'s entries a law places, and
    each unit pressure's force per unit mass (one a non-reference opening)."""

    rows: Sequence[RowKind]
    a: Array
    blocks: Array
    predicted: Array
    reference: int
    coupling: Sequence[Sequence[bool]]
    unit_forces: Sequence[Velocity]


class Partnered(NamedTuple):
    """What the step's one elimination solved: the field at the solution and its base, every unit field (the openings'
    then each body's six), every partner's deliveries, each body's twist as its row closed it, its law's reading and
    its kernel's reading (:func:`kernel_reading`) at the solution, the load and response it is handed (``F = load - resistance V'``), each body's solved force per unit mass
    on the face sets, the openings' answer (``None`` without openings) and the step's solves' report."""

    solved: Velocity
    base: Velocity
    units: tuple[Velocity, ...]
    deliveries: Array
    twists: Array
    readings: Array
    kernels: Array
    handed_resistance: Array
    handed_load: Array
    forces: tuple[Velocity, ...]
    openings: PortSolution | None
    report: Array


def combined(reports: Sequence[Array]) -> Array:
    """One report for a step's solves: the iterations and counts summed, the residual the worst."""
    total = sum(reports[1:], start=reports[0])
    worst = jnp.max(jnp.stack([report[1] for report in reports]))
    return total.at[1].set(worst)


def _add(a: Velocity, b: Velocity, scale: Any = 1.0) -> Velocity:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


def _scaled(a: Velocity, scale: Any) -> Velocity:
    return (scale * a[0], scale * a[1], scale * a[2])


def partnered_step(one: Callable[[Velocity, Velocity | None], tuple[Velocity, Array]], grid: Any, rhs: Velocity,
                   guess: Velocity | None, bodies: Sequence[BodyPartner], held_units: Sequence[Velocity],
                   density: Any, openings: Openings | None = None) -> Partnered:
    """The step's solve with its partners (the module states it): ``one(rhs, start)`` the field's own solve, ``rhs``
    its right-hand side with each opening at its predicted pressure, ``held_units`` the last step's unit fields (the
    openings' then each body's six), each warm-starting its own."""
    dtype = rhs[0].dtype
    base, report = one(rhs, guess)
    reports = [report]
    units: list[Velocity] = []
    unit_forces = list(openings.unit_forces) if openings is not None else []
    held = [stencils(grid, body.point, body.width) for body in bodies]
    for body, stencil in zip(bodies, held, strict=True):
        for k in range(DELIVERIES):
            unit = jnp.zeros(DELIVERIES, dtype=dtype).at[k].set(1.0)
            unit_forces.append(_scaled(delivered(grid, stencil, unit, body.delivery), 1.0 / density))
    for force, start in zip(unit_forces, held_units, strict=True):
        unit, unit_report = one(force, start)
        units.append(unit)
        reports.append(unit_report)
    count = len(units)
    partners: list[PartnerRows] = []
    opened: tuple[tuple[Array, Array], bool, Array, Array] | None = None
    if openings is not None:
        fluxes = grid.opening_fluxes(base)
        response = jnp.stack([grid.opening_fluxes(unit) for unit in units], axis=1)
        rows, held_row, gauge = opening_rows(openings.rows, openings.a, openings.blocks, fluxes, response,
                                             openings.predicted, openings.reference, openings.coupling)
        partners.append(rows)
        opened = (held_row, gauge, fluxes, response)
    first = count - DELIVERIES * len(bodies)
    for b, (body, stencil) in enumerate(zip(bodies, held, strict=True)):
        start = first + DELIVERIES * b
        responses = jnp.stack([read(grid, stencil, unit, body.reading) for unit in units], axis=1)
        matrix = responses.at[:, start:start + DELIVERIES].add(jnp.diag(body.law))
        partners.append(PartnerRows(matrix, -read(grid, stencil, base, body.reading),
                                    jnp.eye(DELIVERIES, dtype=dtype)))
    eliminated = exchange_system(partners)
    width = DELIVERIES * len(bodies)
    own = jnp.zeros((width, width), dtype=dtype)
    selected = jnp.zeros((width, count), dtype=dtype)
    right = jnp.zeros(width, dtype=dtype)
    for b, body in enumerate(bodies):
        rows_b = slice(DELIVERIES * b, DELIVERIES * (b + 1))
        own = own.at[rows_b, rows_b].set(body.system)
        selected = selected.at[rows_b, first + DELIVERIES * b:first + DELIVERIES * (b + 1)].set(
            jnp.eye(DELIVERIES, dtype=dtype))
        right = right.at[rows_b].set(body.right)
    free, deliveries = closed(eliminated, own, selected, right)
    sensitivity = eliminated.sensitivity
    assert sensitivity is not None
    solved = base
    for j, unit in enumerate(units):
        solved = _add(solved, unit, deliveries[j])
    twists = jnp.reshape(free, (len(bodies), DELIVERIES))
    readings: list[Array] = []
    kernels: list[Array] = []
    forces: list[Velocity] = []
    resistances: list[Array] = []
    loads: list[Array] = []
    for b, (body, stencil) in enumerate(zip(bodies, held, strict=True)):
        cols = slice(first + DELIVERIES * b, first + DELIVERIES * (b + 1))
        own_cols = slice(DELIVERIES * b, DELIVERIES * (b + 1))
        response = sensitivity[cols, own_cols]
        force = deliveries[cols]
        resistances.append(response)
        loads.append(-(force - response @ twists[b]))
        kernel = kernel_reading(grid, stencil, solved)
        kernels.append(kernel)
        readings.append(jnp.concatenate([kernel[:3] + body.reading * kernel[3:6], kernel[6:]]))
        forces.append(_scaled(delivered(grid, stencil, force, body.delivery), 1.0 / density))
    solution = None
    if openings is not None and opened is not None:
        held_row, gauge, fluxes, response = opened
        solution = openings_solved(len(openings.rows), openings.reference, held_row, gauge, deliveries, fluxes,
                                   response, openings.predicted)
    total = combined(reports)
    empty = jnp.zeros((0, DELIVERIES), dtype=dtype)
    return Partnered(solved, base, tuple(units), deliveries, twists,
                     jnp.stack(readings) if readings else empty,
                     jnp.stack(kernels) if kernels else jnp.zeros((0, 9), dtype=dtype),
                     jnp.stack(resistances) if resistances else jnp.zeros((0, DELIVERIES, DELIVERIES), dtype=dtype),
                     jnp.stack(loads) if loads else empty, tuple(forces), solution, total)


__all__ = ["DELIVERIES", "SOLUTION", "SOLUTION_COLUMNS", "BodyPartner", "Openings", "Partnered", "combined", "delivered", "kernel_reading",
           "partnered_step", "read", "stencils"]
