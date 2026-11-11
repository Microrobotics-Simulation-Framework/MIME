"""Drawn open tubes for the open box's gates (F2b's PR 2): the node built as the build makes it, the correct runs the
lines are measured on and the planted bugs that prove them, shared by the CI tests and the detection-rate probes so
that both measure the same thing.

**A draw** (:func:`draw`, from a seed): IMEX Euler with convection on, D/h 8, 12 or 16, two diameters long, in units
where D, nu and rho are one; an ``Inflow`` at the low plane with mean speed U (the Reynolds number by U and D
log-uniform in [1, 100]) and a pulsatile part of 0.3 of it over 40 steps a period; a ``Windkessel`` at the high plane
(R_p 0.1 to 3 R_tube, R_d log-uniform 1 to 1e4 R_tube: a segment two diameters long has a downstream bed 1e2 to 1e4
times its own resistance, a carotid's about 2e3; dt/(R_d C) 0.1 to 3); a narrowing of 0 to 0.4 at mid-length (the
interior spike's stenosis); a swirl over the lumen of 0.05 to 0.5 of U in the start; the step a fraction 0.1 to 1 of
the CFL line at the larger of the start's peak and the throat's at the inflow's peak (a run past the line is held, so
no gate judges one). :data:`STEPS` steps, metered as the build meters it.

**The plants** (:data:`PLANTS`, each applied by ``setattr``, as ``flag_cases``' are): the port faces at full weight;
each open plane's half cell's viscous stencil, or one plane's convective one, at a full cell; an inflow 1e-3 above its
waveform; a port gift in the Windkessel's row; the unit field scaled by 1.1 or negated; an interior gift (an unbooked
body force on the fluid); the distal loss written into ``N_WK``.

**The open planes' operator** (:func:`operator_draw`, :func:`operator_errors`): the open grid's Laplacian and
convection against closed forms, exact to rounding on drawn fields (the module of ``test_fvm_open_operator.py``)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from fvm_cases import Stenosis

from mime.core import ledger as books
from mime.nodes.flow.fvm import ImplicitFlowNode, OpenFlowNode, OpenMACGrid, periodic_tube_layout
from mime.nodes.flow.fvm import _open as open_module
from mime.nodes.flow.fvm._ports import Inflow, Windkessel

STEPS = 60
_POINT_SEED = 20_000
INFLOW_LAW, WINDKESSEL_LAW, OPENINGS_LAW = "inflow:tube.start", "windkessel:tube.end", "flow.openings"
#: The deliveries a metered open node takes as the build makes them (the drive's none: no BodyForce).
METERED = (
    books.LedgerDelivery("viscous", "flow.viscosity", books.DISSIPATIVE, books.HOSTED).encoded(),
    books.LedgerDelivery("penalty", "tube.wall", books.DISSIPATIVE, books.HOSTED).encoded(),
    books.LedgerDelivery("opening_0", INFLOW_LAW, books.UNMETERED, books.HOSTED, "", INFLOW_LAW).encoded(),
    books.LedgerDelivery("opening_1", WINDKESSEL_LAW, books.DISSIPATIVE, books.HOSTED).encoded(),
    books.LedgerDelivery("kinetic_flux", OPENINGS_LAW, books.UNMETERED, books.HOSTED).encoded(),
)
Setter = Callable[[Any, str, Any], None]


class NarrowedOpenNode(OpenFlowNode):
    """The open node with the interior spike's stenosis in place of its straight tube where ``narrowing`` is above
    zero (its throat at mid-length, the full radius at both planes)."""

    def solids(self, p: Any, grid: Any) -> Any:
        narrowing = float(self.constants().get("narrowing", 0.0))
        if narrowing == 0.0:
            return super().solids(p, grid)
        centre = self.constants()["tube_centre"]
        return [Stenosis(float(centre[0]), float(centre[1]), 0.5, grid.cells[2] * grid.spacing, narrowing)]


@dataclass(frozen=True)
class Point:
    cells: int
    reynolds: float
    swirl: float
    narrowing: float
    proximal: float
    distal: float
    capacitor: float
    fraction: float
    seed: int


def draw(seed: int) -> Point:
    rng = np.random.default_rng(_POINT_SEED + seed)
    return Point(int(rng.choice([8, 12, 16])), float(math.exp(rng.uniform(0.0, math.log(100.0)))),
                 float(rng.uniform(0.05, 0.5)), float(rng.uniform(0.0, 0.4)), float(rng.uniform(0.1, 3.0)),
                 float(math.exp(rng.uniform(0.0, math.log(1e4)))), float(math.exp(rng.uniform(math.log(0.1), math.log(3.0)))),
                 float(rng.uniform(0.1, 1.0)), seed)


def tube_resistance(length: float) -> float:
    """Poiseuille's resistance of a tube of D = 1 (nu = rho = 1) ``length`` long: ``8 mu L / (pi R^4)``."""
    return 8.0 * length / (math.pi * 0.5 ** 4)


def open_node(cells: int, *, length: float = 2.0, laws: tuple[str, str] = ("inflow", "windkessel"),
              parameters: Any = None, dt: float = 0.01, narrowing: float = 0.0, **settings: Any) -> OpenFlowNode:
    """The open tube of D = 1 along z, ``length`` diameters long, as the build lays out its box."""
    layout = periodic_tube_layout(0.5, length, cells_across=cells)
    if parameters is None:
        resistance = tube_resistance(length)
        parameters = [[0.1 * math.pi / 4.0, 0.0, 0.0, 0.0], [0.3 * resistance, 1.0 / resistance, 3.0 * resistance, 0.0]]
    extra = {"narrowing": narrowing} if narrowing else {}
    return NarrowedOpenNode("flow", dt, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                            spacing=layout.spacing, viscosity=1.0, density=1.0, tube_centre=layout.centre,
                            tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=("tube.start", "tube.end"),
                            opening_laws=laws, opening_parameters=parameters,
                            opening_law_ids=(INFLOW_LAW, WINDKESSEL_LAW) if laws == ("inflow", "windkessel") else (),
                            **extra, **settings)


def point_node(point: Point, *, ledger: bool = True, dt_scale: float = 1.0, **settings: Any) -> OpenFlowNode:
    """The drawn point's node, its start a Poiseuille profile at U with the swirl, projected on the open grid; with
    ``dt_scale`` its step that times the drawn one, every law's parameter as drawn (an order check's refinement)."""
    speed = point.reynolds
    flux = speed * math.pi / 4.0
    probe = open_node(point.cells, narrowing=point.narrowing)
    grid = probe.open_grid(probe._step_constants(None))  # noqa: SLF001
    c = probe.constants()
    rng = np.random.default_rng(point.seed)
    parts = []
    for k in range(3):
        at = np.asarray(grid.face_points(k, "float64"))
        across = ((at[..., 0] - c["tube_centre"][0]) ** 2 + (at[..., 1] - c["tube_centre"][1]) ** 2) / 0.25
        swirl = point.swirl * speed * rng.standard_normal(grid.shape(k)) * (across < 1.0)
        parts.append(swirl + (2.0 * speed * np.clip(1.0 - across, 0.0, None) if k == 2 else 0.0))
    projected = grid.project(tuple(jnp.asarray(x) for x in parts))
    start = {f"initial_velocity_{x}": np.asarray(projected[k]) for k, x in enumerate("xyz")}
    # The step from the larger of the start's peak and the throat's: Poiseuille's centreline at the inflow's peak (1.3 U)
    # through the narrowing's area, so the run stays inside the CFL line as the flow develops (a run past it is held).
    peak = max(max(float(np.max(np.abs(v))) for v in start.values()), 2.6 * speed / (1.0 - point.narrowing) ** 2)
    dt = point.fraction * 0.4 * grid.spacing / peak
    resistance = tube_resistance(2.0)
    distal = point.distal * resistance
    period = 40.0 * dt
    parameters = [[flux, 0.3 * flux, 1.0 / period, 0.0],
                  [point.proximal * resistance, dt / (point.capacitor * distal), distal, 0.0]]
    return open_node(point.cells, narrowing=point.narrowing, dt=dt * dt_scale, parameters=parameters,
                     **{"convection": True, **settings}, **({"ledger": METERED} if ledger else {}), **start)


def n_wk_sum(point: Point, steps: int = 20) -> Callable[[float, float], dict[str, float]]:
    """``term_orders``' sums for one drawn case: ``N_WK`` summed over ``steps`` of the drawn step (the same time at
    any ``dt_scale``)."""

    def sums(dt_scale: float, tolerance_scale: float) -> dict[str, float]:
        del tolerance_scale
        node = point_node(point, dt_scale=dt_scale)
        final, _ = scan(node, int(round(steps / dt_scale)))
        return {"windkessel": abs(books_of(final)["n_wk"])}

    return sums


def scan(node: OpenFlowNode, steps: int, reading: Callable[[Any, Any, Any], Any] | None = None,
         state: Any = None) -> tuple[dict[str, Any], Any]:
    """``steps`` of the node's own step under one compiled scan (no input fed: an open box with no drive takes
    none), with ``reading(p, before, after)`` at each step where given."""
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    dt = node.delta_t

    def body(s: Any, _: Any) -> tuple[Any, Any]:
        ended = node.advance(p, s, {}, dt)
        return ended, (reading(p, s, ended) if reading is not None else None)

    start = node.initial_state() if state is None else state
    ended, readings = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))(dict(start))
    return dict(ended), readings


# -- the plants -------------------------------------------------------------------------------------------------

def _uniform_weights(setter: Setter) -> None:
    setter(open_module, "_end_weight", lambda shape, dtype: jnp.ones((1, 1, shape[2]), dtype=dtype))


def _full_cell_end(setter: Setter) -> None:
    """The end plane's half-cell convective flux at a full cell, ``(w_b^2 - w_c^2) / h`` for ``2 (w_b^2 - w_c^2) / h``."""
    real = OpenMACGrid.convection

    def convection(self: Any, velocity: Any) -> Any:
        c = real(self, velocity)
        w = velocity[2]
        centre = 0.5 * (w[..., -1] + w[..., -2])
        drop = (w[..., -1] ** 2 - centre ** 2) / self.spacing
        return (c[0], c[1], c[2].at[..., -1].add(drop))

    setter(OpenMACGrid, "convection", convection)


def _full_cell_start(setter: Setter) -> None:
    """The start plane's half-cell convective flux at a full cell, ``(w_c^2 - w_b^2) / h`` for ``2 (w_c^2 - w_b^2) / h``."""
    real = OpenMACGrid.convection

    def convection(self: Any, velocity: Any) -> Any:
        c = real(self, velocity)
        w = velocity[2]
        centre = 0.5 * (w[..., 0] + w[..., 1])
        rise = (centre ** 2 - w[..., 0] ** 2) / self.spacing
        return (c[0], c[1], c[2].at[..., 0].add(rise))

    setter(OpenMACGrid, "convection", convection)


def _laplacian_full_cell(setter: Setter) -> None:
    """The reviewer's mutant: each open plane's half cell takes the z-Laplacian of a full cell, ``(w_1 - w_0) / h^2``
    for ``2 (w_1 - w_0) / h^2`` (the half-weight faces' viscous stencil wrong, the operator no longer self-adjoint)."""

    def along_faces(w: Any, h2: Any, magnitude: bool) -> Any:
        sign = 1.0 if magnitude else -1.0
        inside = (w[..., 2:] + w[..., :-2] + sign * 2.0 * w[..., 1:-1]) / h2
        low = (w[..., 1:2] + sign * w[..., :1]) / h2
        high = (w[..., -2:-1] + sign * w[..., -1:]) / h2
        return jnp.concatenate([low, inside, high], axis=-1)

    setter(OpenMACGrid, "_along_faces", staticmethod(along_faces))


def _gradient_full_cell(setter: Setter) -> None:
    """The planes' faces' gradient at a full cell, ``p_0 / h`` for ``2 p_0 / h``: the projector's end faces wrong."""
    real = OpenMACGrid.gradient

    def gradient(self: Any, pressure: Any) -> Any:
        g = real(self, pressure)
        return (g[0], g[1], g[2].at[..., 0].multiply(0.5).at[..., -1].multiply(0.5))

    setter(OpenMACGrid, "gradient", gradient)


def _inflow_scaled(setter: Setter) -> None:
    real = Inflow.affine
    setter(Inflow, "affine", lambda self, parameters, state, time, dt: tuple(
        1.001 * x for x in real(self, parameters, state, time, dt)))


#: The port gift's size, a share of the Windkessel's row's block ``b`` (the stage's whole resistance, which a downstream
#: resistance many times the vessel's makes far larger than ``R_p``).
GIFT_SHARE = 0.05


def _port_gift(setter: Setter) -> None:
    """A negative resistance in the Windkessel's row, ``GIFT_SHARE`` of its block, that its books do not hold."""
    real = Windkessel.affine

    def affine(self: Any, parameters: Any, state: Any, time: Any, dt: Any) -> Any:
        a, b = real(self, parameters, state, time, dt)
        return a, (1.0 - GIFT_SHARE) * b

    setter(Windkessel, "affine", affine)


def _unit(factor: float) -> Callable[[Setter], None]:
    def planted(setter: Setter) -> None:
        def unit_field(self: Any, solve: Any, force: Any, start: Any) -> Any:
            del self
            unit, report = solve(force, start)
            return (factor * unit[0], factor * unit[1], factor * unit[2]), report

        setter(OpenFlowNode, "unit_field", unit_field)

    return planted


def _interior_gift(setter: Setter) -> None:
    real = OpenFlowNode.forcing

    def forcing(self: Any, fields: Any, boundary_inputs: Any, scheme: Any, dtype: Any) -> Any:
        delivered = real(self, fields, boundary_inputs, scheme, dtype)
        size = GIFT[str(np.dtype(dtype))]
        speed = float(np.asarray(self.constants()["opening_parameters"])[0, 0]) / (math.pi / 4.0)
        gift = size * 32.0 * speed  # a share of the Poiseuille gradient at the inflow's mean speed

        def force(stage: int, time: Any) -> Any:
            held = delivered(stage, time)
            return (held[0], held[1], held[2] + gift * (1.0 - fields.chi[2]))

        return force

    setter(OpenFlowNode, "forcing", forcing)


#: The interior gift's size, as a share of the Poiseuille gradient at the inflow's mean speed, by dtype: about 25 of
#: the identity's floors on draw 0 (274 floors at 1e-8 in float64, 0.51 at 1e-2 in float32, measured while writing).
GIFT = {"float64": 1e-9, "float32": 0.5}


def _kinetic_in_n_conv(setter: Setter) -> None:
    """The kinetic energy convection carries through the openings left inside ``N_conv`` (its line booked as zero,
    ``N_conv`` as ``-dt rho <C(u), u'>``): the step identity still closes; only the boundary flag can see it."""
    real = OpenFlowNode.open_energy_terms

    def planted(self: Any, *args: Any, **kwargs: Any) -> Any:
        terms = dict(real(self, *args, **kwargs))
        flux = terms["kinetic_flux"]
        terms["convection"] = terms["convection"] - flux
        terms["kinetic_flux"] = jnp.zeros_like(flux)
        return terms

    setter(OpenFlowNode, "open_energy_terms", planted)


def _loss_in_n_wk(setter: Setter) -> None:
    real = Windkessel.books

    def planted(self: Any, parameters: Any, state: Any, ended: Any, flux: Any, pressure: Any, dt: Any) -> Any:
        held = real(self, parameters, state, ended, flux, pressure, dt)
        distal = dt * ended[0] * ended[0] / parameters[2]
        return type(held)(held.storage_start, held.storage_end, held.loss - distal, held.numerical + distal)

    setter(Windkessel, "books", planted)


def _gauge_off(setter: Setter) -> None:
    """FB12's control: the gauge off, each step's relative pressures formed as differences of the absolute ones (so the
    field's unknowns and the moves carry the absolute pressures' rounding, ``eps |level|``), the moves and the rows'
    fluxes from them."""
    from mime.nodes.flow.fvm._ports import PortSolution

    real = OpenFlowNode.solve_ports

    def solve_ports(self: Any, rows: Any, a: Any, blocks: Any, base: Any, response: Any, predicted: Any,
                    reference: int) -> Any:
        solution = real(self, rows, a, blocks, base, response, predicted, reference)
        relative = solution.pressures - solution.pressures[reference]
        moved = relative - jnp.asarray(predicted)
        others = [j for j in range(len(rows)) if j != reference]
        fluxes = base + (response @ jnp.stack([moved[j] for j in others]) if others else 0.0)
        return PortSolution(solution.pressures, relative, moved, fluxes)

    setter(OpenFlowNode, "solve_ports", solve_ports)


PLANTS: dict[str, Callable[[Setter], None]] = {
    "gradient_full_cell": _gradient_full_cell, "weights_uniform": _uniform_weights,
    "laplacian_full_cell": _laplacian_full_cell, "end_full_cell": _full_cell_end, "start_full_cell": _full_cell_start,
    "inflow_scaled": _inflow_scaled, "port_gift": _port_gift, "unit_scaled": _unit(1.1), "unit_negated": _unit(-1.0),
    "interior_gift": _interior_gift, "loss_in_n_wk": _loss_in_n_wk, "kinetic_in_n_conv": _kinetic_in_n_conv,
    "gauge_off": _gauge_off,
}


#: FB1's line on the arms' difference over the largest speed, by dtype (clean: up to 3.1e-13 and 6.6e-7 at D/h 8 and
#: 16 while writing; the planted end faces 0.084 to 0.34).
FB1_LINE = {"float64": 1e-9, "float32": 1e-4}

__all__ = ["FB1_LINE", "GIFT", "GIFT_SHARE", "INFLOW_LAW", "INTERIOR_FROM", "METERED", "OPENINGS_LAW", "PLANTS", "STEPS",
           "WINDKESSEL_LAW", "FA1_PER_DEPTH_SQUARED", "NarrowedOpenNode", "OperatorDraw", "Point", "bypass_arm",
           "bypass_reading", "fa1_line",
           "darcy_permeability", "draw", "n_wk_sum", "open_node", "operator_draw", "operator_errors", "point_node",
           "scan", "tube_resistance"]


# -- the gates' readings, a step each (traced) ------------------------------------------------------------------

def _velocity(state: Any) -> Any:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


def inflow_reading(node: OpenFlowNode) -> Callable[[Any, Any, Any], Any]:
    """FB2: the start's face flux against the inflow's waveform at the step's end (nominal, from the node's own
    parameters), over its floor, derived from the two roundings it carries (never fitted): the face flux's summation,
    ``(N + 2) eps S`` (the field's face flux and the row's both sums over the plane's ``N`` faces of the superposed field,
    the base and the unit field: three sums of ``N`` terms and the faces' own rounding, :class:`OpenFlowNode`'s
    ``superposed_scale``), ``S`` bounded from the step's states as ``h^2 (sum |w| + 2 |m| sum |w_unit|)``, ``m`` the
    inflow's relative pressure's move; and the waveform's own, ``4 eps (|steady| + |amplitude| (1 + 2 pi |f| t))``, the
    node's instant and the reading's differing by a rounding of ``t``. No pressure level enters: a move is never formed
    from absolute pressures. Above one it has failed."""

    def reading(p: Any, before: Any, after: Any) -> Any:
        grid = node.open_grid(p)
        row = jnp.asarray(p["opening_parameters"])[0]
        eps = jnp.finfo(row.dtype).eps
        time = jnp.asarray(after["steps"]).astype(row.dtype) * node.delta_t
        phase = 2.0 * jnp.pi * row[2] * time
        flux = row[0] + row[1] * jnp.cos(phase + row[3])
        w = jnp.asarray(after["velocity_z"])[..., 0]
        others = node.others()
        moved = jnp.asarray(after["opening_pressure_relative"])[0] - jnp.asarray(before["opening_pressure_relative"])[0]
        unit = jnp.sum(jnp.abs(jnp.asarray(after["unit_z"])[others.index(0)][..., 0])) if 0 in others else 0.0
        scale = grid.spacing ** 2 * (jnp.sum(jnp.abs(w)) + 2.0 * jnp.abs(moved) * unit)
        faces = grid.cells[0] * grid.cells[1]
        floor = (faces + 2) * eps * scale + 4.0 * eps * (jnp.abs(row[0]) + jnp.abs(row[1]) * (1.0 + jnp.abs(phase)))
        return jnp.abs(jnp.asarray(after["opening_flux"])[0] + flux) / floor

    return reading


def superposed_reading(node: OpenFlowNode) -> Callable[[Any, Any, Any], Any]:
    """FB4: the step's solution's projected true residual at the openings' solved pressures, over the projected
    right-hand side's norm (the solve's own measure), against ten times the larger of the tolerance and the step's
    reported residual: above one it has failed."""

    def reading(p: Any, before: Any, after: Any) -> Any:
        grid = node.open_grid(p)
        fields = node.fields(p, grid)
        sigma = node.resistivity(p, fields)
        shift = 1.0 / node.delta_t if node.scheme.stores_inertia else 0.0
        velocity, solved = _velocity(before), _velocity(after)
        convection = grid.convection(velocity) if node.constants()["convection"] else grid.zeros(velocity[0].dtype)
        force = grid.port_force(jnp.asarray(after["opening_pressure_relative"]), jnp.asarray(p["density"]))
        rhs = tuple(shift * velocity[k] + convection[k] + force[k] for k in range(3))
        applied = node.operator_of(p, grid, sigma)(shift)(solved)
        residual = grid.project(tuple(rhs[k] - applied[k] for k in range(3)))
        projected = grid.project(rhs)
        relative = jnp.sqrt(grid.inner(residual, residual) / grid.inner(projected, projected))
        bar = 10.0 * jnp.maximum(jnp.asarray(p["tolerance"]), jnp.asarray(after["solve_report"])[1])
        return relative / bar

    return reading


def books_of(final: Any) -> dict[str, Any]:
    """The run's last records: the law identity's worst over its floor, N_WK summed, the step identity's worst."""
    dtype = np.asarray(final[f"{books.RECORDS}work"]).dtype
    floor = books.IDENTITY_ULPS * float(np.finfo(dtype).eps)
    rows = books.laws(books.deliveries({"ledger": METERED}))
    return {"law_identity": float(final[f"{books.RECORDS}law_identity_worst"]) / floor,
            "n_wk": float(np.asarray(final[f"{books.RECORDS}law_numerical"])[rows.index(WINDKESSEL_LAW)]),
            "identity": float(final[f"{books.RECORDS}identity_worst"]) / floor,
            "kinetic_flux": float(np.asarray(final[f"{books.RECORDS}work"])[rows.index(OPENINGS_LAW)])}


class DrivenEverywhereNode(ImplicitFlowNode):
    """The periodic node with its body force on the solid as well as the fluid: what a pressure drop across a period
    is (a uniform force on everything), FB1's periodic arm. MIME's own drive acts on the fluid alone (round 4, item 6),
    which differs from a pressure drop in the band by about the band's share of the flux."""

    def drive(self, fields: Any, delivered: Any) -> Any:
        return tuple(delivered[k] * jnp.ones_like(fields.chi[k]) for k in range(3))


def fb1_pair(cells: int, *, offset: tuple[float, float] = (0.0, 0.0), length_cells: int = 16,
             tolerance: float = 1e-11) -> tuple[Any, Any]:
    """FB1's two arms, steady Stokes at the same layout: the open tube with its ends at given pressures ``G L`` apart
    and the periodic tube driven by the body force ``G`` (D = nu = rho = 1, G = 32 so the mean speed is one), each
    after one steady step: the open arm's velocity and the periodic arm's."""
    from fvm_cases import driven, run, Waveform

    gradient, length = 32.0, length_cells / cells
    layout = periodic_tube_layout(0.5, length, cells_across=cells, offset_cells=offset)
    opened = NarrowedOpenNode("flow", 1.0, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                              spacing=layout.spacing, viscosity=1.0, density=1.0, tube_centre=layout.centre,
                              tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=("tube.start", "tube.end"),
                              opening_laws=("pressure", "pressure"),
                              opening_parameters=[[gradient * length, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                              scheme="steady", convection=False, tolerance=tolerance)
    periodic = driven(DrivenEverywhereNode(
        "flow", 1.0, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2], spacing=layout.spacing,
        viscosity=1.0, density=1.0, tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=0.5,
        scheme="steady", convection=False, tolerance=tolerance), Waveform(steady=gradient))
    open_final, _ = scan(opened, 2)
    periodic_final = run(periodic, 2)
    return open_final, periodic_final


def fb1_difference(open_final: Any, periodic_final: Any) -> float:
    """The largest difference between the arms' velocities over the largest speed (``w``'s planes' faces against the
    periodic box's first face, the same plane)."""
    opened = [np.asarray(open_final[f"velocity_{x}"]) for x in "xyz"]
    periodic = [np.asarray(periodic_final[f"velocity_{x}"]) for x in "xyz"]
    opened[2] = opened[2][..., :-1]
    speed = max(float(np.max(np.abs(c))) for c in periodic)
    return max(float(np.max(np.abs(a - b))) for a, b in zip(opened, periodic, strict=True)) / speed


#: The stiff interior's depth, in band widths from the wall, from which every face is fully stiff (``_flags.INSIDE``).
INTERIOR_FROM = 1.5


def darcy_permeability(spacing: float, interior: float, band_depth: float = 0.3) -> float:
    """The permeability (m^2) deep in the solid, beyond the band and its stiffening ramp: the band's and the interior's
    resistivities in series, ``(delta h)^2 / (1 + (delta / delta_in)^2)`` (``_band.wall_fields``), ``delta_in``
    the arm's own ``interior`` (in cells)."""
    return (band_depth * spacing) ** 2 / (1.0 + (band_depth / interior) ** 2)


def bypass_arm(cells: int, interior: str, *, length: float = 2.0, offset: tuple[float, float] = (0.0, 0.0),
               tolerance: float = 1e-11) -> tuple[OpenFlowNode, dict[str, Any]]:
    """FA1 = FB14's arm: the open tube with its ends at given pressures ``G L`` apart (G = 32: mean speed one), steady
    Stokes at the stiff interior ``interior``, after its steady steps."""
    gradient = 32.0
    layout = periodic_tube_layout(0.5, length, cells_across=cells, offset_cells=offset)
    node = NarrowedOpenNode("flow", 1.0, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                            spacing=layout.spacing, viscosity=1.0, density=1.0, tube_centre=layout.centre,
                            tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=("tube.start", "tube.end"),
                            opening_laws=("pressure", "pressure"),
                            opening_parameters=[[gradient * layout.cells[2] * layout.spacing, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0]],
                            scheme="steady", convection=False, tolerance=tolerance, interior=interior)
    final, _ = scan(node, 2)
    return node, final


def bypass_reading(node: OpenFlowNode, final: Any, interior: float) -> dict[str, Any]:
    """Each opening's flux through its faces deep in the solid against Darcy's at the run's own pressure drop and the
    oracle's ``interior`` (``u = K (p_start - p_end) / (mu L)`` on each such face), over Darcy's; and each opening's
    flux through the solid beyond the band's half width, a share of the opening's flux (the share an ``Inflow``'s
    pinned flux carries outside the lumen)."""
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.open_grid(p)
    fields = node.fields(p, grid)
    h = float(grid.spacing)
    velocity = (jnp.asarray(final["velocity_x"]), jnp.asarray(final["velocity_y"]), jnp.asarray(final["velocity_z"]))
    deep = INTERIOR_FROM * 2.0 * h
    found = np.asarray(node.opening_solid_fluxes(p, velocity, deep), dtype=np.float64)
    pressure = np.asarray(final["opening_pressure"], dtype=np.float64)
    length = grid.cells[2] * h
    speed = darcy_permeability(h, interior) * (pressure[0] - pressure[1]) / length
    distance = np.asarray(fields.signed_distance[2], dtype=np.float64)
    faces = [int(np.sum(distance[..., plane] >= deep)) for plane in (0, -1)]
    oracle = np.asarray([-faces[0], faces[1]], dtype=np.float64) * h * h * speed
    shares = np.asarray(node.opening_solid_fluxes(p, velocity), dtype=np.float64) / \
        np.asarray(final["opening_flux"], dtype=np.float64)
    carried = float(np.max(np.abs(np.asarray(final["opening_flux"], dtype=np.float64)) / np.abs(oracle)))
    return {"error": float(np.max(np.abs(found - oracle) / np.abs(oracle))), "found": found, "oracle": oracle,
            "solid_share": shares, "carried": carried}


#: FA1 = FB14's line on the deep solid's flux against Darcy's, over Darcy's: a share of ``delta_in^2`` (the Brinkman
#: screening length over the spacing, squared: how far the band's faster flow reaches into the stiff interior's first
#: faces), and the solve's tolerance carried to so small a part of the field (its relative residual is the whole
#: field's, so it reaches the deep solid's flux scaled by the opening's flux over the solid's).
FA1_PER_DEPTH_SQUARED = 0.2


def fa1_line(found: dict[str, Any], interior: float, tolerance: float) -> float:
    """FA1's line for ``found`` (:func:`bypass_reading`) at the arm's ``interior`` (cells) and its solve's ``tolerance``."""
    return FA1_PER_DEPTH_SQUARED * interior ** 2 + tolerance * found["carried"]


def monolithic_distance(node: OpenFlowNode, before: Any, after: Any) -> float:
    """SF-12's affine case on the open node (FB4's monolithic reference): the step's superposed solution against one
    projected solve of the step's right-hand side with the openings' forces at their solved pressures, cold-started,
    at a tolerance a thousand times tighter than the step's (float64; float32 its own), over the largest speed."""
    from mime.nodes.flow.fvm import Jacobi, Stopping, projected_solve
    from mime.nodes.flow.fvm._solve import stagnation_window

    p = node._step_constants(None)  # noqa: SLF001
    grid = node.open_grid(p)
    fields = node.fields(p, grid)
    sigma = node.resistivity(p, fields)
    shift = 1.0 / node.delta_t if node.scheme.stores_inertia else 0.0
    velocity = _velocity(before)
    dtype = velocity[0].dtype
    convection = grid.convection(velocity) if node.constants()["convection"] else grid.zeros(dtype)
    force = grid.port_force(jnp.asarray(after["opening_pressure_relative"]), jnp.asarray(p["density"]))
    rhs = tuple(shift * velocity[k] + convection[k] + force[k] for k in range(3))
    tight = 1e-3 * float(p["tolerance"]) if np.dtype(dtype) == np.float64 else float(p["tolerance"])
    stopping = Stopping(tolerance=jnp.asarray(tight, dtype=dtype), at_floor=np.dtype(dtype) == np.float32,
                        window=stagnation_window(grid.cells, 20))
    solved, _ = jax.jit(lambda b: projected_solve(
        grid, node.operator_of(p, grid, sigma)(shift), node.diagonal_of(p, grid, sigma, shift), b, None, stopping,
        jnp.zeros(6, dtype=dtype), Jacobi(), node.magnitude_of(p, grid, sigma)(shift)))(rhs)
    speed = max(float(np.max(np.abs(np.asarray(c)))) for c in solved)
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
               for a, b in zip(_velocity(after), solved, strict=True)) / speed


# -- the open planes' operator against closed forms (B2) --------------------------------------------------------

#: The operator gate's lines, in units of the run's eps: the viscous stencil's on each face's own coefficient magnitude
#: (``laplacian_magnitude`` of the field in magnitude: seven terms and the field's own rounding, at most 10 u); the
#: convective one's on ``M^2 / h``, ``M`` the field's largest speed (a face sums at most three flux differences, each a
#: product of two face averages, under 40 u with the field's own rounding).
OPERATOR_ULPS = {"viscous": 16.0, "convective": 32.0}


@dataclass(frozen=True)
class OperatorDraw:
    """A drawn open grid and the fields its operator is judged on: ``cells`` and ``spacing`` (m); ``modes``, a cosine
    mode a velocity component, ``(kx, ky, m, amplitude, phase_x, phase_y)``, ``m >= 1`` so that each varies along z at
    the planes; ``linear``, ``(a0, a1, b0, b1, kx, ky, phase_a, phase_b)``: ``w = a(x, y) + b(x, y) z`` with ``u = v =
    0``; ``uniform``, ``(U, V, a, b)``: ``w = a + b z`` with ``u = U`` and ``v = V``."""

    cells: tuple[int, int, int]
    spacing: float
    modes: tuple[tuple[float, ...], ...]
    linear: tuple[float, ...]
    uniform: tuple[float, ...]
    seed: int


def operator_draw(seed: int) -> OperatorDraw:
    """Physical scales: cells of 10 um to 1 mm, speeds of 1 mm/s to 1 m/s, 4 to 12 cells across and 4 to 32 along;
    the convective fields kept above a third of their speed (``w > 0``, ``|b| L`` at most 0.4 of ``a``)."""
    rng = np.random.default_rng(40_000 + seed)
    nx, ny, nz = int(rng.integers(4, 13)), int(rng.integers(4, 13)), int(rng.integers(4, 33))
    spacing = float(math.exp(rng.uniform(math.log(1e-5), math.log(1e-3))))
    speed = float(math.exp(rng.uniform(math.log(1e-3), 0.0)))
    two_pi = 2.0 * math.pi
    modes = tuple((float(rng.integers(0, nx // 2)), float(rng.integers(0, ny // 2)), float(rng.integers(1, nz)),
                   speed * float(rng.uniform(0.2, 1.0)) * float(rng.choice([-1.0, 1.0])),
                   float(rng.uniform(0.0, two_pi)), float(rng.uniform(0.0, two_pi))) for _ in range(3))
    length = nz * spacing
    slope = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.1, 0.4)) * speed / length
    linear = (speed, float(rng.uniform(-0.2, 0.2)) * speed, slope, float(rng.uniform(-0.2, 0.2)) * slope,
              float(rng.integers(0, nx // 2)), float(rng.integers(0, ny // 2)), float(rng.uniform(0.0, two_pi)),
              float(rng.uniform(0.0, two_pi)))
    uniform = (speed * float(rng.uniform(-1.0, 1.0)), speed * float(rng.uniform(-1.0, 1.0)), speed,
               float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.1, 0.4)) * speed / length)
    return OperatorDraw((nx, ny, nz), spacing, modes, linear, uniform, seed)


def _points(grid: OpenMACGrid, component: int) -> list[np.ndarray]:
    """The points (m, float64) the ``component``-th velocity is sampled at, as ``face_points`` places them."""
    h = grid.spacing
    axes = [np.arange(n) * h if axis == component else (np.arange(n) + 0.5) * h
            for axis, n in enumerate(grid.shape(component))]
    return list(np.meshgrid(*axes, indexing="ij"))


def _half_cells(b: np.ndarray, w: Any, z: np.ndarray, h: float, length: float) -> np.ndarray:
    """``-d(w^2)/dz = -2 b w`` at each ``w`` face of a field linear in z, each plane's half cell its average over the
    half cell (the value at its centroid, a quarter cell in)."""
    held = z.copy()
    held[..., 0], held[..., -1] = 0.25 * h, length - 0.25 * h
    return -2.0 * b * w(held)


def operator_errors(drawn: OperatorDraw, dtype: str) -> dict[str, float]:
    """The open grid's Laplacian and convection on the drawn fields (in ``dtype``) against their closed forms (float64),
    each face's error over its line (:data:`OPERATOR_ULPS`), the worst over every face and over the planes' faces alone.

    **Viscous**: on a cosine mode ``cos(2 pi kx x / Lx) cos(2 pi ky y / Ly) cos(pi m z / Lz)`` the discrete Laplacian
    is its eigenvalue times the mode exactly, ``-(4 / h^2) (sin^2(pi kx / nx) + sin^2(pi ky / ny) + sin^2(pi m /
    (2 nz)))``, on every face the planes' included: the mode's zero gradient at a plane makes the half cell's stencil
    ``2 (w_1 - w_0) / h^2`` the interior one with its reflected ghost. **Convective**: on ``w`` linear in z the face
    averages are exact and ``w^2`` is quadratic, so each face's divergence-form difference is ``-2 b w`` at the face and
    each plane's half cell the average over it; with ``u = v = 0``, ``a`` and ``b`` vary across the planes; with ``u = U``
    and ``v = V``, constants, the tangential fluxes at the planes' edges give ``-U b`` and ``-V b``."""
    grid = OpenMACGrid(drawn.cells, drawn.spacing)
    nx, ny, nz = drawn.cells
    h = drawn.spacing
    lx, ly, lz = nx * h, ny * h, nz * h
    eps = float(np.finfo(dtype).eps)
    two_pi = 2.0 * math.pi

    def judged(found: Any, exact: np.ndarray, floor: np.ndarray) -> tuple[float, float]:
        error = np.abs(np.asarray(found, dtype=np.float64) - exact)
        ratio = np.where(floor > 0.0, error / np.where(floor > 0.0, floor, 1.0), np.where(error > 0.0, np.inf, 0.0))
        return float(np.max(ratio)), float(max(np.max(ratio[..., 0]), np.max(ratio[..., -1])))

    # The modes and their eigenvalues in extended precision (``np.longdouble``), so that the closed form carries no
    # rounding of its phases (``eps |theta|`` in float64, up to some 100 eps at the drawn modes) beyond its own: its
    # precision's eps times the phase, kept in the line, which is rounding where the platform has extended precision.
    wide = np.longdouble
    reference = float(np.finfo(wide).eps)
    hw, pi = wide(h), np.arccos(wide(-1.0))

    def symbol(kx: float, ky: float, m: float) -> Any:
        return -(wide(4.0) / hw ** 2) * (np.sin(pi * wide(kx) / nx) ** 2 + np.sin(pi * wide(ky) / ny) ** 2
                                         + np.sin(pi * wide(m) / (2 * nz)) ** 2)

    viscous = []
    for k, (kx, ky, m, amplitude, px, py) in enumerate(drawn.modes):
        # Each point's place along each axis as a share of the box, from its index (``face_points``' places), so that
        # no rounding of a length enters a phase.
        x, y, z = np.meshgrid(*[(np.arange(n, dtype=wide) + (wide(0) if axis == k else wide(0.5))) / wide(cells)
                                for axis, (n, cells) in enumerate(zip(grid.shape(k), drawn.cells, strict=True))],
                              indexing="ij")
        mode = wide(amplitude) * np.cos(2 * pi * wide(kx) * x + wide(px)) * np.cos(2 * pi * wide(ky) * y + wide(py)) \
            * np.cos(pi * wide(m) * z)
        alone = tuple(jnp.asarray(mode.astype(np.float64), dtype=dtype) if c == k else
                      jnp.zeros(grid.shape(c), dtype=dtype) for c in range(3))
        found = grid.laplacian(alone)[k]
        magnitude = np.asarray(grid.laplacian_magnitude(tuple(jnp.abs(c) for c in alone))[k], np.float64)
        phases = 2.0 * math.pi * (kx + ky + 2.0) + math.pi * m
        floor = OPERATOR_ULPS["viscous"] * eps * magnitude + 12.0 / h ** 2 * abs(amplitude) * 4.0 * reference * phases
        viscous.append(judged(found, (symbol(kx, ky, m) * mode).astype(np.float64), floor))

    a0, a1, b0, b1, kx, ky, pa, pb = drawn.linear
    x, y, z = _points(grid, 2)
    a = a0 + a1 * np.cos(two_pi * (kx * x / lx + ky * y / ly) + pa)
    b = b0 + b1 * np.cos(two_pi * (kx * x / lx + ky * y / ly) + pb)
    linear_w = a + b * z
    shapes = [grid.shape(k) for k in range(3)]
    linear = (np.zeros(shapes[0]), np.zeros(shapes[1]), linear_w)
    linear_exact = (np.zeros(shapes[0]), np.zeros(shapes[1]), _half_cells(b, lambda at: a + b * at, z, h, lz))
    big_u, big_v, ua, ub = drawn.uniform
    uniform = (np.full(shapes[0], big_u), np.full(shapes[1], big_v), ua + ub * z)
    uniform_exact = (np.full(shapes[0], -big_u * ub), np.full(shapes[1], -big_v * ub),
                     _half_cells(np.full(shapes[2], ub), lambda at: ua + ub * at, z, h, lz))
    convective = []
    for field, exact in ((linear, linear_exact), (uniform, uniform_exact)):
        held = tuple(jnp.asarray(f, dtype=dtype) for f in field)
        speed = max(float(np.max(np.abs(f))) for f in field)
        found = grid.convection(held)
        convective += [judged(found[k], exact[k], np.full(shapes[k], OPERATOR_ULPS["convective"] * eps * speed ** 2 / h))
                       for k in range(3)]
    return {"viscous": max(v[0] for v in viscous), "viscous_planes": max(v[1] for v in viscous),
            "convective": max(c[0] for c in convective), "convective_planes": max(c[1] for c in convective)}
