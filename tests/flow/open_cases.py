"""Drawn open tubes for the open box's gates (F2b's PR 2): the node built as the build makes it, the correct runs the
lines are measured on and the planted bugs that prove them, shared by the CI tests and the detection-rate probes so
that both measure the same thing.

**A draw** (:func:`draw`, from a seed): IMEX Euler with convection on, D/h 8, 12 or 16, two diameters long, in units
where D, nu and rho are one; an ``Inflow`` at the low plane with mean speed U (the Reynolds number by U and D
log-uniform in [1, 100]) and a pulsatile part of 0.3 of it over 40 steps a period; a ``Windkessel`` at the high plane
(R_p 0.1 to 3 R_tube, R_d 1 to 10 R_tube, dt/(R_d C) 0.1 to 3); a narrowing of 0 to 0.4 at mid-length (the
interior spike's stenosis); a swirl over the lumen of 0.05 to 0.5 of U in the start; the step a fraction 0.1 to 1 of
the CFL line at the larger of the start's peak and the throat's at the inflow's peak (a run past the line is held, so
no gate judges one). :data:`STEPS` steps, metered as the build meters it.

**The plants** (:data:`PLANTS`, each applied by ``setattr``, as ``flag_cases``' are): the port faces at full weight;
the outlet's half-cell convective flux at full weight; an inflow 1e-3 above its waveform; a port gift in the
Windkessel's block; the unit field scaled by 1.1 or negated; an interior gift (an unbooked body force on the fluid);
the distal loss written into ``N_WK``."""

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
INFLOW_LAW, WINDKESSEL_LAW, OPENINGS_LAW = "inflow:tube.inlet", "windkessel:tube.outlet", "flow.openings"
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
                 float(rng.uniform(1.0, 10.0)), float(math.exp(rng.uniform(math.log(0.1), math.log(3.0)))),
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
                            tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=("tube.inlet", "tube.outlet"),
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


def _full_cell_outlet(setter: Setter) -> None:
    real = OpenMACGrid.convection

    def convection(self: Any, velocity: Any) -> Any:
        c = real(self, velocity)
        w = velocity[2]
        centre = 0.5 * (w[..., -1] + w[..., -2])
        drop = (w[..., -1] ** 2 - centre ** 2) / self.spacing
        return (c[0], c[1], c[2].at[..., -1].add(drop))

    setter(OpenMACGrid, "convection", convection)


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


def _port_gift(setter: Setter) -> None:
    real = Windkessel.affine

    def affine(self: Any, parameters: Any, state: Any, time: Any, dt: Any) -> Any:
        a, b = real(self, parameters, state, time, dt)
        return a, b - 0.05 * parameters[0]

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


def _loss_in_n_wk(setter: Setter) -> None:
    real = Windkessel.books

    def planted(self: Any, parameters: Any, state: Any, ended: Any, flux: Any, pressure: Any, dt: Any) -> Any:
        held = real(self, parameters, state, ended, flux, pressure, dt)
        distal = dt * ended[0] * ended[0] / parameters[2]
        return type(held)(held.storage_start, held.storage_end, held.loss - distal, held.numerical + distal)

    setter(Windkessel, "books", planted)


def _kinetic_flux_in_n_conv(setter: Setter) -> None:
    """The periodic box's convection term at open faces: the kinetic energy's flux left inside ``N_conv``."""
    real = OpenFlowNode.open_energy_terms

    def terms(self: Any, *args: Any) -> Any:
        held = dict(real(self, *args))
        held["convection"] = held["convection"] - held["kinetic_flux"]
        held["kinetic_flux"] = jnp.zeros_like(held["kinetic_flux"])
        return held

    setter(OpenFlowNode, "open_energy_terms", terms)


PLANTS: dict[str, Callable[[Setter], None]] = {
    "kinetic_flux_in_n_conv": _kinetic_flux_in_n_conv, "gradient_full_cell": _gradient_full_cell,
    "weights_uniform": _uniform_weights, "outlet_full_cell": _full_cell_outlet, "inflow_scaled": _inflow_scaled,
    "port_gift": _port_gift, "unit_scaled": _unit(1.1), "unit_negated": _unit(-1.0), "interior_gift": _interior_gift,
    "loss_in_n_wk": _loss_in_n_wk,
}


#: FB1's line on the arms' difference over the largest speed, by dtype (clean: up to 3.1e-13 and 6.6e-7 at D/h 8 and
#: 16 while writing; the planted end faces 0.084 to 0.34).
FB1_LINE = {"float64": 1e-9, "float32": 1e-4}

__all__ = ["FB1_LINE", "GIFT", "INFLOW_LAW", "METERED", "OPENINGS_LAW", "PLANTS", "STEPS", "WINDKESSEL_LAW", "NarrowedOpenNode", "Point",
           "draw", "n_wk_sum", "open_node", "point_node", "scan", "tube_resistance"]


# -- the gates' readings, a step each (traced) ------------------------------------------------------------------

def _velocity(state: Any) -> Any:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


#: FB2's line, in units of the run's eps on the face's magnitude: the prediction's 64 missed in float32 on 2 of 190 clean
#: drawn runs (85 eps at worst; float64's worst 22), so the line is three times float32's worst.
FB2_ULPS = 256.0


def inflow_reading(node: OpenFlowNode) -> Callable[[Any, Any, Any], Any]:
    """FB2: the inlet's face flux against the inflow's waveform at the step's end (nominal, from the node's own
    parameters), over :data:`FB2_ULPS` eps of the face's magnitude ``h^2 sum |w|``: above one it has failed."""

    def reading(p: Any, before: Any, after: Any) -> Any:
        del before
        grid = node.open_grid(p)
        row = jnp.asarray(p["opening_parameters"])[0]
        time = jnp.asarray(after["steps"]).astype(row.dtype) * node.delta_t
        flux = row[0] + row[1] * jnp.cos(2.0 * jnp.pi * row[2] * time + row[3])
        w = jnp.asarray(after["velocity_z"])[..., 0]
        size = grid.spacing ** 2 * jnp.sum(jnp.abs(w))
        return jnp.abs(jnp.asarray(after["opening_flux"])[0] + flux) / (FB2_ULPS * jnp.finfo(row.dtype).eps * size)

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
        pressures = jnp.asarray(after["opening_pressure"])
        force = grid.port_force(pressures - pressures[node.reference()], jnp.asarray(p["density"]))
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
                              tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=("tube.inlet", "tube.outlet"),
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
    pressures = jnp.asarray(after["opening_pressure"])
    force = grid.port_force(pressures - pressures[node.reference()], jnp.asarray(p["density"]))
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
