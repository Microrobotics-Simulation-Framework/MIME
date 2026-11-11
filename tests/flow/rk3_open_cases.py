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
(the affine case's); the inflow 1e-3 above its waveform; the openings' forces at the drive's weight ``gamma_k dt`` in
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
    """SF-12's affine case under RK3: the step from ``before`` (whose pressure is made) against the same step taken
    with each stage's openings' forces held at the relative pressures that stage solved and projected plainly (no unit
    field, no port system: the stage's own projection of its whole right-hand side), the largest difference over the
    largest speed; and the stepped state."""
    p = node._step_constants(None)  # noqa: SLF001
    dt = node.delta_t
    grid = node.open_grid(p)
    fields = node.fields(p, grid)
    sigma = node.resistivity(p, fields)
    velocity = (jnp.asarray(before["velocity_x"]), jnp.asarray(before["velocity_y"]), jnp.asarray(before["velocity_z"]))
    time = jnp.asarray(before["steps"]).astype(velocity[0].dtype) * dt
    after = node.advance(p, dict(before), {}, dt)

    def stepped(state: Any) -> Any:
        stages = node.stages(p, grid, fields, sigma, state, time, dt)
        ended, _, _ = _rk3.step(grid, velocity, jnp.asarray(state["pressure"]), node.kinematic(p), sigma,
                                lambda k, t: grid.zeros(velocity[0].dtype), bool(node.constants()["convection"]), time,
                                dt, explicit=stages.explicit, held_force=stages.held_force, project=stages.project)
        return ended, [r.solution.relative for r in stages.records]

    _, relative = jax.jit(stepped)(dict(before))

    def plain(state: Any) -> Any:
        stages = node.stages(p, grid, fields, sigma, state, time, dt)
        ended, _, _ = _rk3.step(grid, velocity, jnp.asarray(state["pressure"]), node.kinematic(p), sigma,
                                lambda k, t: grid.zeros(velocity[0].dtype), bool(node.constants()["convection"]), time,
                                dt, explicit=stages.explicit,
                                held_force=lambda k: grid.port_force(relative[k], jnp.asarray(p["density"])))
        return ended

    reference = jax.jit(plain)(dict(before))
    solved = (after["velocity_x"], after["velocity_y"], after["velocity_z"])
    speed = max(float(np.max(np.abs(np.asarray(c)))) for c in reference)
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(solved, reference, strict=True)) \
        / speed, after


#: The affine case's line under RK3, over the largest speed: the stages' own rounding (fast solves, no tolerance);
#: 1e-12 in float64 and 1e-5 in float32 (measured while writing on draw 0: TBD).
AFFINE_LINE = {"float64": 1e-11, "float32": 1e-4}


# -- the plants ---------------------------------------------------------------------------------------------------

def _law_once(setter: Setter) -> None:
    """FB11's control: each law held over the whole step at its start, its rows backward Euler over ``dt`` at the
    step's end at every stage, advanced once with the last stage's flux and pressure (its books over ``dt``); the flow
    still books the stages' ``sum alpha_k dt p_k q_k``."""
    stages = _open_explicit.OpeningStages

    def rows_at(self: Any, k: int, stage_dt: Any) -> Any:
        del k, stage_dt
        start = self.__dict__.setdefault("step_start", self.law_state)
        return self.node.law_rows(self.parameters, start, self.time + self.dt, self.dt)

    def advance_laws(self: Any, flux: Any, flux_size: Any, solution: Any, blocks: Any, superposed: Any,
                     stage_dt: Any) -> None:
        start = self.__dict__.setdefault("step_start", self.law_state)
        node, pressures = self.node, solution.pressures
        last = len(self.records) == 2
        ended = jnp.stack([law.advanced(self.parameters[i], start[i], node.per_law(law, held, flux),
                                        node.per_law(law, held, pressures), self.dt) if last else start[i]
                           for i, (law, held) in enumerate(zip(self.laws, self.bound, strict=True))])
        # A stage before the last moves no law: its books are the law's at rest (no step, its storage at the start).
        own = tuple(law.books(self.parameters[i], start[i], ended[i], node.per_law(law, held, flux),
                              node.per_law(law, held, pressures), self.dt if last else 0.0)
                    for i, (law, held) in enumerate(zip(self.laws, self.bound, strict=True)))
        self.records.append(_open_explicit.StageRecord(solution, flux, flux_size, superposed, blocks, own, stage_dt))
        self.law_state = ended

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
        return [(tuple(1.1 * c for c in unit), 1.1 * phi, tuple(1.1 * c for c in held))
                for unit, phi, held in real(self, stage_dt)]

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
    "law_once": _law_once, "gift_once": _gift_once, "unit_scaled": _unit_scaled, "inflow_scaled": _inflow_scaled,
    "forces_at_gamma": _forces_at_gamma,
}

__all__ = ["AFFINE_LINE", "GIFT_STEP", "LocalStenosis", "METERED_RK3", "PLANTS", "STEPS", "ShapedOpenExplicitNode",
           "books_of", "drawn", "inflow_reading", "point_node", "rk3_node", "rk3_step", "scan",
           "stage_reference_distance"]
