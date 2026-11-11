"""Drawn open tubes stepped by RK3 (F2b PR 4b): the open node of the explicit stage kind
(``mime.nodes.flow.fvm.OpenExplicitFlowNode``) built as the build makes it, its correct runs and the planted bugs that
prove its gates, shared by the CI tests and the probes so that both measure the same thing.

**A draw** is ``open_cases``' (D/h 8, 12 or 16, two diameters long, D = nu = rho = 1, an ``Inflow`` at the low plane at
Re 1 to 100 with a pulsatile part of 0.3 over 40 steps a period, a ``Windkessel`` at the high plane, a narrowing of 0
to 0.4 at mid-length, a swirl in the start), stepped by RK3 at a fraction 0.1 to 0.9 of the smaller of its CFL line
at the larger of the start's peak and the throat's and its viscous number; the backflow term on the outlet, as the
build places it.

**The plants** (:data:`PLANTS`, each applied by ``setattr``): the law stepped once a step (FB11's control); a gift on
one step, twice that step's dissipation, the laws' losses included (route 1's control); the unit fields scaled by 1.1
(the affine case's); the unit fields divided by the penalty (the start's transverse kick); the inflow 1e-3 above its waveform; the openings' forces at the drive's weight ``gamma_k dt`` in
place of the pressure's ``alpha_k dt`` (the open tube's fixed point's)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from open_cases import INFLOW_LAW, METERED, WINDKESSEL_LAW, Point, draw, tube_resistance

from mime.core import ledger as books
from mime.nodes.flow.fvm import OpenExplicitFlowNode, periodic_tube_layout
from mime.nodes.flow.fvm import _open_explicit, _rk3
from mime.nodes.flow.fvm._ports import Inflow

STEPS = 30
#: The deliveries a metered open RK3 node takes as the build makes them, the backflow term on the outlet among them.
METERED_RK3 = (*METERED, books.LedgerDelivery("backflow", "flow.backflow", books.DISSIPATIVE, books.HOSTED).encoded())
#: The step the one-step gift is given on.
GIFT_STEP = 10
Setter = Callable[[Any, str, Any], None]


@dataclasses.dataclass(frozen=True)
class LocalStenosis:
    """A cosine narrowing of a straight tube along z, ``length`` long, centred at ``centre`` (m): the radius ``R (1 - s
    (1 + cos(2 pi x)) / 2)``, ``x = (z - centre) / length`` clipped to a half, with the first-order signed distance
    ``(r - R(z)) / sqrt(1 + R'(z)^2)``, positive in the solid (the open-ports spike's stenosis, its ``s`` 0.5 a 50 %
    narrowing of the radius)."""

    centre_x: float
    centre_y: float
    radius: float
    centre: float
    length: float
    narrowing: float
    region: str = "wall"

    def signed_distance(self, points: Any) -> Any:
        r = jnp.hypot(points[..., 0] - self.centre_x, points[..., 1] - self.centre_y)
        x = jnp.clip((points[..., 2] - self.centre) / self.length, -0.5, 0.5)
        local = self.radius * (1.0 - self.narrowing * (1.0 + jnp.cos(2.0 * jnp.pi * x)) / 2.0)
        slope = self.radius * self.narrowing * jnp.pi / self.length * jnp.sin(2.0 * jnp.pi * x)
        return (r - local) / jnp.sqrt(1.0 + slope ** 2)


class ShapedOpenExplicitNode(OpenExplicitFlowNode):
    """The open RK3 node with a stenosis where one is given: ``narrowing`` (``open_cases``' throat at mid-length, over
    the box's length) or ``stenosis`` ``(s, length, centre)`` (:class:`LocalStenosis`)."""

    def solids(self, p: Any, grid: Any) -> Any:
        c = self.constants()
        centre = c["tube_centre"]
        if "stenosis" in c:
            s, length, at = (float(v) for v in c["stenosis"])
            return [LocalStenosis(float(centre[0]), float(centre[1]), float(c["tube_radius"]), at, length, s)]
        if float(c.get("narrowing", 0.0)):
            from fvm_cases import Stenosis

            return [Stenosis(float(centre[0]), float(centre[1]), float(c["tube_radius"]), grid.cells[2] * grid.spacing,
                             float(c["narrowing"]))]
        return super().solids(p, grid)


def rk3_node(cells: int, *, length: float = 2.0, laws: tuple[str, str] = ("inflow", "windkessel"),
             parameters: Any = None, dt: float | None = None, radius: float = 0.5, viscosity: float = 1.0,
             **settings: Any) -> OpenExplicitFlowNode:
    """The open tube along z, ``length`` long, of ``radius`` (D/h ``cells``), as the build lays out its box, stepped by
    RK3 (the backflow term on the outlet unless ``backflow_openings`` says otherwise)."""
    layout = periodic_tube_layout(radius, length, cells_across=cells)
    h = layout.spacing
    if parameters is None:
        resistance = tube_resistance(length)
        parameters = [[0.1 * math.pi / 4.0, 0.0, 0.0, 0.0], [0.3 * resistance, 1.0 / resistance, 3.0 * resistance, 0.0]]
    ids = (INFLOW_LAW, WINDKESSEL_LAW) if laws == ("inflow", "windkessel") else ()
    return ShapedOpenExplicitNode("flow", 0.1 * h * h / viscosity if dt is None else dt, cells_x=layout.cells[0],
                                  cells_y=layout.cells[1], cells_z=layout.cells[2], spacing=h, viscosity=viscosity,
                                  density=1.0, tube_centre=layout.centre, tube_axis=(0.0, 0.0, 1.0), tube_radius=radius,
                                  openings=("tube.start", "tube.end"), opening_laws=laws,
                                  opening_parameters=parameters, opening_law_ids=ids, scheme="rk3",
                                  **{"backflow_openings": (1,), **settings})


def rk3_step(cells: int, peak: float, fraction: float) -> float:
    """``fraction`` of the smaller of RK3's two bounds on a tube of D = nu = 1 at D/h ``cells``: its CFL line at
    ``peak`` (1.7 along z) and its viscous number (0.15)."""
    h = 1.0 / cells
    return fraction * min(_rk3_line() * h / peak, 0.15 * h * h)


def _rk3_line() -> float:
    from mime.nodes.flow.fvm import RK3

    line = RK3.cfl_line
    assert line is not None
    return line


def point_node(point: Point, *, ledger: bool = True, dt_scale: float = 1.0, **settings: Any) -> OpenExplicitFlowNode:
    """The drawn point's open RK3 node (``open_cases.point_node``'s start, laws and narrowing), its step a fraction 0.1
    to 0.9 of RK3's bound (the drawn fraction times 0.9), times ``dt_scale`` (an order check's refinement)."""
    speed = point.reynolds
    flux = speed * math.pi / 4.0
    probe = rk3_node(point.cells, narrowing=point.narrowing)
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
    peak = max(max(float(np.max(np.abs(v))) for v in start.values()), 2.6 * speed / (1.0 - point.narrowing) ** 2)
    dt = rk3_step(point.cells, peak, 0.9 * point.fraction)
    resistance = tube_resistance(2.0)
    distal = point.distal * resistance
    parameters = [[flux, 0.3 * flux, 1.0 / (40.0 * dt), 0.0],
                  [point.proximal * resistance, dt / (point.capacitor * distal), distal, 0.0]]
    return rk3_node(point.cells, narrowing=point.narrowing, dt=dt * dt_scale, parameters=parameters,
                    **{"convection": True, **settings}, **({"ledger": METERED_RK3} if ledger else {}), **start)


def drawn(seed: int, **settings: Any) -> OpenExplicitFlowNode:
    return point_node(draw(seed), **settings)


def scan(node: OpenExplicitFlowNode, steps: int, reading: Callable[[Any, Any, Any], Any] | None = None,
         state: Any = None) -> tuple[dict[str, Any], Any]:
    """``steps`` of the node's own step under one compiled scan (no input fed: no drive), with ``reading(p, before,
    after)`` at each step where given."""
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    dt = node.delta_t

    def body(s: Any, _: Any) -> tuple[Any, Any]:
        ended = node.advance(p, s, {}, dt)
        return ended, (reading(p, s, ended) if reading is not None else None)

    start = node.initial_state() if state is None else state
    ended, readings = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))(dict(start))
    return dict(ended), readings


def books_of(final: Any) -> dict[str, float]:
    """The run's last records: the law identity's worst and the claimed identity's over the identity's floor, route
    1's negative steps and its worst step, the unattributed work's worst."""
    r = books.RECORDS
    dtype = np.asarray(final[f"{r}work"]).dtype
    floor = books.IDENTITY_ULPS * float(np.finfo(dtype).eps)
    return {"law_identity": float(final[f"{r}law_identity_worst"]) / floor,
            "identity": float(final[f"{r}identity_worst"]) / floor,
            "negative_steps": float(final[f"{r}route_one_negative_steps"]),
            "route_one_worst": float(final[f"{r}route_one_worst"]),
            "unattributed": float(final[f"{r}unattributed_worst"])}


def inflow_reading(node: OpenExplicitFlowNode) -> Callable[[Any, Any, Any], Any]:
    """FB2 under RK3: the step's face flux at the start plane against the inflow's waveform at the step's end (the
    last stage's end instant), over its floor: the face flux's summation, ``(N + 2) eps S``, ``S`` bounded by four times
    the plane's ``h^2 sum |w|`` (the base and the unit field superposed at a move of the flux's own order), and the
    waveform's, ``4 eps (|steady| + |amplitude| (1 + 2 pi |f| t))``. Above one it has failed."""

    def reading(p: Any, before: Any, after: Any) -> Any:
        del before
        grid = node.open_grid(p)
        row = jnp.asarray(p["opening_parameters"])[0]
        eps = jnp.finfo(row.dtype).eps
        time = jnp.asarray(after["steps"]).astype(row.dtype) * node.delta_t
        phase = 2.0 * jnp.pi * row[2] * time
        flux = row[0] + row[1] * jnp.cos(phase + row[3])
        scale = 4.0 * grid.spacing ** 2 * jnp.sum(jnp.abs(jnp.asarray(after["velocity_z"])[..., 0]))
        faces = grid.cells[0] * grid.cells[1]
        floor = (faces + 2) * eps * scale + 4.0 * eps * (jnp.abs(row[0]) + jnp.abs(row[1]) * (1.0 + jnp.abs(phase)))
        return jnp.abs(jnp.asarray(after["opening_flux"])[0] + flux) / floor

    return reading


def stage_reference_distance(node: OpenExplicitFlowNode, before: Any) -> tuple[float, Any]:
    """SF-12's affine case under RK3: the step from ``before`` (whose pressure is made) against the same step taken with
    each stage's moves applied as forces, ``P[held + alpha_k dt F(m_k)]``, one projection of the stage's whole field
    (no unit field, no port system: the moves the stages solved, the prediction's forces as the stage takes them),
    the largest difference over the largest speed; and the stepped state."""
    p = node._step_constants(None)  # noqa: SLF001
    dt = node.delta_t
    grid = node.open_grid(p)
    fields = node.fields(p, grid)
    sigma = node.resistivity(p, fields)
    rho = jnp.asarray(p["density"])
    velocity = (jnp.asarray(before["velocity_x"]), jnp.asarray(before["velocity_y"]), jnp.asarray(before["velocity_z"]))
    time = jnp.asarray(before["steps"]).astype(velocity[0].dtype) * dt
    after = node.advance(p, dict(before), {}, dt)
    convection = bool(node.constants()["convection"])

    def drive(k: int, t: Any) -> Any:
        return grid.zeros(velocity[0].dtype)

    def stepped(state: Any) -> Any:
        stages = node.stages(p, grid, fields, sigma, state, time, dt)
        _rk3.step(grid, velocity, jnp.asarray(state["pressure"]), node.kinematic(p), sigma, drive, convection, time, dt,
                  explicit=stages.explicit, held_force=stages.held_force, project=stages.project)
        return [r.solution.moved for r in stages.records]

    moves = jax.jit(stepped)(dict(before))

    def plain(state: Any) -> Any:
        stages = node.stages(p, grid, fields, sigma, state, time, dt)

        def project(k: int, held: Any, stage_dt: Any) -> Any:
            force = grid.port_force(moves[k], rho)
            projected, phi = _open_explicit.projection(grid, tuple(held[c] + stage_dt * force[c] for c in range(3)),
                                                       stage_dt)
            stages.predicted = stages.predicted + moves[k]
            return projected, phi, held

        ended, _, _ = _rk3.step(grid, velocity, jnp.asarray(state["pressure"]), node.kinematic(p), sigma, drive,
                                convection, time, dt, explicit=stages.explicit, held_force=stages.held_force,
                                project=project)
        return ended

    reference = jax.jit(plain)(dict(before))
    solved = (after["velocity_x"], after["velocity_y"], after["velocity_z"])
    speed = max(float(np.max(np.abs(np.asarray(c)))) for c in reference)
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(solved, reference, strict=True)) \
        / speed, after


def start_node(proximal: float, cells: int = 16) -> tuple[OpenExplicitFlowNode, float]:
    """The open-ports spike's question 7 on this node: a pressure inlet ``P0 (1 + 0.3 cos w t)``, an RCR outlet with
    ``R_p`` ``proximal`` times the vessel's resistance, C 0.4 and R_d 5, S1's units (nu 0.005, R 0.5, alpha 6), D/h
    ``cells``, 4 D long, the stiff interior 0.01h, 350 steps a period at D/h 16, Poiseuille's profile at the mean speed
    0.5 in the start; and that speed."""
    nu, radius, speed = 0.005, 0.5, 0.5
    omega = 36.0 * nu / radius ** 2
    period = 2.0 * math.pi / omega
    h = 1.0 / cells
    dt = period / math.ceil(period / min(0.15 * h * h / nu, 0.8 * h / 2.0))
    flux = math.pi * radius ** 2 * speed
    vessel = 8.0 * nu / (math.pi * radius ** 4) * 4.0
    distal = 5.0
    level = flux * (vessel + proximal * vessel + distal)
    parameters = [[level, 0.3 * level, omega / (2.0 * math.pi), 0.0], [proximal * vessel, 0.4, distal, 0.0]]
    settings: dict[str, Any] = dict(length=4.0, laws=("pressure", "windkessel"), parameters=parameters, dt=dt,
                                    viscosity=nu, interior="0.01h", backflow_openings=())
    probe = rk3_node(cells, **settings)
    grid = probe.open_grid(probe._step_constants(None))  # noqa: SLF001
    c = probe.constants()
    at = np.asarray(grid.face_points(2, "float64"))
    across = ((at[..., 0] - c["tube_centre"][0]) ** 2 + (at[..., 1] - c["tube_centre"][1]) ** 2) / radius ** 2
    return rk3_node(cells, **settings, initial_velocity_z=2.0 * speed * np.clip(1.0 - across, 0.0, None),
                    initial_opening_state=[[0.0], [flux * distal]]), speed


def transverse_kick(node: OpenExplicitFlowNode, speed: float, steps: int = 5, zero: bool = False) -> float:
    """The largest transverse speed in the fluid (``chi`` under a half) over the first ``steps`` steps, over ``speed``;
    ``zero``: from the zero start (the field's pressure and the openings' prediction zero, marked made)."""
    p = node._step_constants(None)  # noqa: SLF001
    chi = node.fields(p, node.open_grid(p)).chi[0]
    start = dict(node.initial_state())
    if zero:
        start["pressure_ready"] = jnp.ones_like(start["pressure_ready"])
    _, found = scan(node, steps, lambda p, s, e: jnp.max(jnp.where(chi < 0.5, jnp.abs(e["velocity_x"]), 0.0)), start)
    return float(np.max(np.asarray(found))) / speed


def restart_distance(node: OpenExplicitFlowNode, arm: str, at: int = 10) -> float:
    """A state given from outside at step ``at``, one step on, against the uninterrupted run's step, over its largest
    speed: ``consistent``, its pressure marked unmade (the node makes it consistent with the velocity and the openings);
    ``mismatched``, the field's pressure at zero with the openings' prediction kept (the spike's interior pressure at
    zero against the end pressures); ``zero``, both at zero, marked made."""
    middle, _ = scan(node, at)
    reference, _ = scan(node, 1, state=middle)
    state = dict(middle)
    if arm == "consistent":
        state["pressure_ready"] = jnp.zeros_like(state["pressure_ready"])
    else:
        state["pressure"] = jnp.zeros_like(state["pressure"])
        if arm == "zero":
            state["opening_pressure_relative"] = jnp.zeros_like(state["opening_pressure_relative"])
    stepped, _ = scan(node, 1, state=state)
    speed = max(float(np.max(np.abs(np.asarray(reference[f"velocity_{x}"])))) for x in "xyz")
    return max(float(np.max(np.abs(np.asarray(stepped[f"velocity_{x}"]) - np.asarray(reference[f"velocity_{x}"]))))
               for x in "xyz") / speed


def fixed_point_pair(cells: int = 8, *, narrowing: float = 0.4, length: float = 2.0, steps: int = 1500,
                     tolerance: float = 1e-11) -> tuple[dict[str, Any], dict[str, Any]]:
    """The open tube's two arms through a narrowing, Stokes, its ends at given pressures ``G L`` apart (G = 32: a mean
    speed about one in the open tube): RK3 after ``steps`` steps of dt 0.15 h^2 from rest, and the steady open solve
    (``open_cases.NarrowedOpenNode``, the projected CG at ``tolerance``); each final state."""
    from open_cases import NarrowedOpenNode
    from open_cases import scan as open_scan

    gradient = 32.0
    layout = periodic_tube_layout(0.5, length, cells_across=cells)
    parameters = [[gradient * layout.cells[2] * layout.spacing, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    steady = NarrowedOpenNode("flow", 1.0, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                              spacing=layout.spacing, viscosity=1.0, density=1.0, tube_centre=layout.centre,
                              tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=("tube.start", "tube.end"),
                              opening_laws=("pressure", "pressure"), opening_parameters=parameters, scheme="steady",
                              convection=False, tolerance=tolerance, narrowing=narrowing)
    stepped = rk3_node(cells, length=length, laws=("pressure", "pressure"), parameters=parameters,
                       dt=0.15 * layout.spacing ** 2, convection=False, narrowing=narrowing, backflow_openings=())
    return scan(stepped, steps)[0], open_scan(steady, 2)[0]


def fixed_point_distance(stepped: Any, steady: Any) -> float:
    """The largest difference between the two arms' velocities over the largest speed."""
    pairs = [(np.asarray(stepped[f"velocity_{x}"]), np.asarray(steady[f"velocity_{x}"])) for x in "xyz"]
    speed = max(float(np.max(np.abs(b))) for _, b in pairs)
    return max(float(np.max(np.abs(a - b))) for a, b in pairs) / speed


#: The affine case's line under RK3, over the largest speed: the stages' own rounding (fast solves, no tolerance);
#: 1e-12 in float64 and 1e-5 in float32 (measured while writing on draw 0: TBD).
AFFINE_LINE = {"float64": 1e-11, "float32": 3e-6}


# -- the plants ---------------------------------------------------------------------------------------------------

def _law_once(setter: Setter) -> None:
    """FB11's control: each law that stores energy (the Windkessel) held over the whole step at its start, its rows
    backward Euler over ``dt`` at the step's end at every stage, advanced once with the last stage's flux and pressure
    (its books over ``dt``); the other laws (the inflow) at their stages as built, and the flow still booking its
    stages' ``sum alpha_k dt p_k q_k``. (Every law held at the step's end, the inflow too, pins the same flux at each
    stage, and the residual vanishes: measured while writing, 0.03 of the floor.)"""
    stages = _open_explicit.OpeningStages
    real_rows = stages.rows_at

    def stored_openings(self: Any) -> list[int]:
        return [o for law, held in zip(self.laws, self.bound, strict=True) if law.stores for o in held]

    def rows_at(self: Any, k: int, stage_dt: Any) -> Any:
        start = self.__dict__.setdefault("step_start", self.law_state)
        a, block = real_rows(self, k, stage_dt)
        once_a, once_block = self.node.law_rows(self.parameters, start, self.time + self.dt, self.dt)
        for o in stored_openings(self):
            a, block = a.at[o].set(once_a[o]), block.at[o].set(once_block[o])
        return a, block

    def advance_laws(self: Any, flux: Any, flux_size: Any, solution: Any, blocks: Any, superposed: Any,
                     stage_dt: Any) -> None:
        start = self.__dict__.setdefault("step_start", self.law_state)
        node, pressures = self.node, solution.pressures
        last = len(self.records) == 2
        ended, own = [], []
        for i, (law, held) in enumerate(zip(self.laws, self.bound, strict=True)):
            q, p = node.per_law(law, held, flux), node.per_law(law, held, pressures)
            if not law.stores:
                ended.append(law.advanced(self.parameters[i], self.law_state[i], q, p, stage_dt))
                own.append(law.books(self.parameters[i], self.law_state[i], ended[-1], q, p, stage_dt))
                continue
            # A stage before the last moves the law not at all: its books are its own at rest (no step).
            step = self.dt if last else 0.0
            ended.append(law.advanced(self.parameters[i], start[i], q, p, self.dt) if last else start[i])
            own.append(law.books(self.parameters[i], start[i], ended[-1], q, p, step))
        self.records.append(_open_explicit.StageRecord(solution, flux, flux_size, superposed, blocks, tuple(own),
                                                       stage_dt))
        self.law_state = jnp.stack(ended)

    setter(stages, "rows_at", rows_at)
    setter(stages, "advance_laws", advance_laws)


def _gift_once(setter: Setter) -> None:
    """Route 1's control: on :data:`GIFT_STEP` alone, the stepped velocity scaled so the kinetic energy rises by twice
    the step's own dissipation by formula (the viscosity's, the wall's, the backflow term's and the laws' losses and
    numerical terms), unbooked."""
    real = OpenExplicitFlowNode.stepped

    def stepped(self: Any, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        ended, parts = real(self, p, state, boundary_inputs, dt)
        stage, energy = parts.stage, parts.energy
        rho = jnp.asarray(p["density"])
        lost = rho * (stage.viscous + stage.penalty - (stage.explicit if stage.explicit is not None else 0.0)) + \
            jnp.sum(energy["law_books"][:, 2] + energy["law_books"][:, 3])
        kinetic = self.storage(p, ended)
        factor = jnp.sqrt(1.0 + 2.0 * lost / kinetic)
        factor = jnp.where(jnp.asarray(state["steps"]) == GIFT_STEP, factor, 1.0)
        held = dict(ended)
        for x in "xyz":
            held[f"velocity_{x}"] = factor * held[f"velocity_{x}"]
        return held, parts

    setter(OpenExplicitFlowNode, "stepped", stepped)


def _unit_scaled(setter: Setter) -> None:
    real = _open_explicit.OpeningStages.units

    def units(self: Any, stage_dt: Any) -> Any:
        return [(tuple(1.1 * c for c in unit), 1.1 * phi) for unit, phi in real(self, stage_dt)]

    setter(_open_explicit.OpeningStages, "units", units)


def _unit_penalised(setter: Setter) -> None:
    """The moves taken before the penalty's solve: each unit field the projection of its force divided by the
    penalty's pointwise solve (the plane's solid faces damped, the solid behind them not)."""

    def units(self: Any, stage_dt: Any) -> Any:
        return [_open_explicit.projection(self.grid, tuple(stage_dt * force[c] / (1.0 + stage_dt * self.sigma[c])
                                                           for c in range(3)), stage_dt)
                for force in self.unit_forces]

    setter(_open_explicit.OpeningStages, "units", units)


def _inflow_scaled(setter: Setter) -> None:
    real = Inflow.affine
    setter(Inflow, "affine", lambda self, parameters, state, time, dt: tuple(
        1.001 * x for x in real(self, parameters, state, time, dt)))


def _forces_at_gamma(setter: Setter) -> None:
    """The openings' forces taken with the drive's weight ``gamma_k dt`` in place of the pressure's ``alpha_k dt``."""
    real = _open_explicit.OpeningStages.held_force

    def held_force(self: Any, k: int) -> Any:
        return tuple(_rk3.GAMMA[k] / _rk3.ALPHA[k] * c for c in real(self, k))

    setter(_open_explicit.OpeningStages, "held_force", held_force)


PLANTS: dict[str, Callable[[Setter], None]] = {
    "law_once": _law_once, "gift_once": _gift_once, "unit_scaled": _unit_scaled, "unit_penalised": _unit_penalised,
    "inflow_scaled": _inflow_scaled,
    "forces_at_gamma": _forces_at_gamma,
}

__all__ = ["AFFINE_LINE", "GIFT_STEP", "fixed_point_distance", "fixed_point_pair", "LocalStenosis", "METERED_RK3", "PLANTS", "STEPS", "ShapedOpenExplicitNode",
           "books_of", "drawn", "inflow_reading", "open_tube", "restart_distance", "start_node", "transverse_kick", "point_node", "rk3_node", "rk3_step", "scan",
           "stage_reference_distance"]


def open_tube(scheme: str = "rk3", opened: bool = True, cells_across: int = 16, length_mm: float = 2.0,
              speed: float = 0.01, dt_ms: float = 0.5) -> Any:
    """An experiment of a straight tube of radius 0.5 mm, ``length_mm`` long, water, at ``cells_across``: open at its
    two ends (an ``Inflow`` of mean speed ``speed`` m/s with a pulsatile part of 0.3 at 2 Hz, a ``Windkessel``) or on a
    periodic box driven by a body force giving about that speed; stepped by ``scheme``. The probes' memory estimates
    (``mem_estimate.py --case``) and FB19's budget (``node_measure.py``)."""
    from mime.dynamics import FVM, Scheme
    from mime.experiment import Experiment
    from mime.interactions import BodyForce, Inflow, Windkessel
    from mime.scene import PeriodicBox, Scene, Tube, VesselRegion, Z
    from mime.things import Medium, Structure
    from mime.units import Hz, Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_Pa, m3_per_s, m_per_s2, mm, ms

    radius, mu = 0.5e-3, 1e-3
    length = length_mm * 1e-3
    exp = Experiment("open_tube_budget", scene=Scene.new(up=Z, units="m"), dt=ms(dt_ms))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(length_mm))))
    chosen = Scheme.RK3() if scheme == "rk3" else Scheme.IMEXEuler()
    domain = VesselRegion() if opened else PeriodicBox(length=mm(length_mm))
    exp.add_medium(Medium("water", fills="tube.lumen",
                          fields={"flow": FVM(domain=domain, cells_across=cells_across, scheme=chosen)}))
    exp.declare("water.viscosity", Pa_s(mu), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    flux = speed * math.pi * radius ** 2
    resistance = 8.0 * mu * length / (math.pi * radius ** 4)
    if opened:
        exp.attach(Inflow(steady=m3_per_s(flux), amplitude=m3_per_s(0.3 * flux), frequency=Hz(2.0)), medium="water",
                   surface="tube.start")
        exp.attach(Windkessel(proximal=Pa_s_per_m3(0.3 * resistance), compliance=m3_per_Pa(1e-12),
                              distal=Pa_s_per_m3(3.0 * resistance)), medium="water", surface="tube.end")
    else:
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(8.0 * mu / 1e3 * speed / radius ** 2)),
                   medium="water")
    return exp


def gate_reading(seed: int) -> dict[str, float]:
    """FB11's and route 1's oracle on one draw (``drawn(seed)``, :data:`STEPS` steps): the books, the inflow's worst
    reading, and the affine case's distance after them."""
    node = drawn(seed)
    final, inflow = scan(node, STEPS, inflow_reading(node))
    affine, _ = stage_reference_distance(node, final)
    return {**books_of(final), "inflow": float(np.max(np.asarray(inflow))), "affine": affine}


def gate_fails(reading: dict[str, float], dtype: str) -> bool:
    return reading["law_identity"] > 1.0 or reading["negative_steps"] != 0.0 or reading["unattributed"] != 0.0 or \
        reading["inflow"] > 1.0 or reading["affine"] > AFFINE_LINE[dtype]


def main() -> int:
    """The probe's oracle (``detection_rate.py --script``): one draw at ``DETECTION_SEED``, its plant from
    ``RK3_PLANT`` (:data:`PLANTS`); exits 1 when the gate fails."""
    import json
    import os

    seed = int(os.environ.get("DETECTION_SEED", "0"))
    plant = os.environ.get("RK3_PLANT") or None
    if plant:
        PLANTS[plant](setattr)
    dtype = "float64" if jax.config.read("jax_enable_x64") else "float32"
    reading = gate_reading(seed)
    print(json.dumps({"seed": seed, "plant": plant, "dtype": dtype, **reading}))
    return 1 if gate_fails(reading, dtype) else 0


if __name__ == "__main__":
    raise SystemExit(main())
