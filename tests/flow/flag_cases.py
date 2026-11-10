"""Drawn tube flows for the flow's ledger flags (``mime.nodes.flow.fvm._flags``): the correct runs their lines are
measured on and the planted bugs that prove them, shared by the CI tests and the detection-rate probes so that both
measure the same thing.

**A draw** (:func:`draw`, from a seed): IMEX Euler with convection on, D/h 8, 12 or 16, the Reynolds number by the
centreline speed and the diameter log-uniform in [1, 200] (in units where D, nu and rho are one, so the Poiseuille
drive is ``g = 16 U``), a swirl over the lumen of 0.05 to 0.5 of U, and the step a fraction 0.1 to 1 of the CFL line
at the start's peak speed; :data:`STEPS` steps from the Poiseuille profile plus the swirl, projected, with the build's
deliveries (``METERED``). Cell Reynolds numbers 0.06 to 25, CFL 0.04 to 0.4, tau/dt about 4 to 3e3.

**The plants** (:data:`PLANTS`, each applied by ``setattr`` so that pytest's ``monkeypatch`` and a probe's own process
apply it alike): a solve stopped short (two iterations; a node tolerance 1e4 looser; or the CG's stop 1e4 looser than the
node's tolerance, which the node declares at its default: each reporting arrival as its stopping rule allows); a
convective term formed from the run's starting velocity (a stale level) or from the velocity's components permuted (the
wrong field), in the advective form, or at four times the step with the CFL line lifted; the convective operator itself
scaled by 1.5 or negated (still skew: an operator error); and the fluid's mask lost beyond the band (the drive reaching
into the solid) or the solid's interior a thousand times less stiff."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from fvm_cases import METERED, tube_node

from mime.core import ledger as books
from mime.nodes.flow.fvm import IMEXEuler, ImplicitFlowNode, MACGrid
from mime.nodes.flow.fvm import _node as node_module
from mime.nodes.flow.fvm._band import WallFields
from mime.nodes.flow.fvm._schemes import StageParts
from mime.testing import run_flags

#: The steps each drawn run takes.
STEPS = 60
#: The seed a draw's operating point is drawn from is offset from the swirl's own, so the two are independent.
_POINT_SEED = 10_000

Setter = Callable[[Any, str, Any], None]


@dataclass(frozen=True)
class Point:
    """A drawn operating point: cells across, the Reynolds number, the swirl's share of the centreline speed, the step
    as a fraction of the CFL line, and the swirl's seed."""

    cells: int
    reynolds: float
    swirl: float
    fraction: float
    seed: int


def draw(seed: int) -> Point:
    rng = np.random.default_rng(_POINT_SEED + seed)
    return Point(int(rng.choice([8, 12, 16])), float(math.exp(rng.uniform(0.0, math.log(200.0)))),
                 float(rng.uniform(0.05, 0.5)), float(rng.uniform(0.1, 1.0)), seed)


#: The run's starting velocity, which the stale-level plant forms its convection from.
_STARTED: dict[str, Any] = {}


def _stale(setter: Setter) -> None:
    def step(self: Any, context: Any, state: Any, time: Any, dt: Any) -> Any:
        grid = context.grid
        velocity = state["velocity"]
        shift = 1.0 / dt
        force = context.force(0, time + self.drive_instants[0] * dt)
        convection = grid.convection(tuple(jnp.asarray(c, dtype=velocity[0].dtype) for c in _STARTED["velocity"]))
        rhs = tuple(shift * velocity[k] + force[k] + convection[k] for k in range(3))
        solved, report = context.solve(shift, rhs, velocity)
        return {"velocity": solved}, StageParts(force, convection, rhs, solved, shift, report)

    setter(IMEXEuler, "step", step)


def _permuted(setter: Setter) -> None:
    real = MACGrid.convection
    setter(MACGrid, "convection", lambda self, velocity: real(self, (velocity[1], velocity[2], velocity[0])))


def _operator(factor: float) -> Callable[[Setter], None]:
    """The convective operator times ``factor`` (1.5: scaled; -1: negated): still skew, an operator error."""

    def planted(setter: Setter) -> None:
        real = MACGrid.convection

        def convection(self: Any, velocity: Any) -> Any:
            c = real(self, velocity)
            return (factor * c[0], factor * c[1], factor * c[2])

        setter(MACGrid, "convection", convection)

    return planted


def _advective(setter: Setter) -> None:
    def advective(self: Any, velocity: Any) -> Any:
        """``-(u . grad) u`` on each face set, central differences and four-point averages: not skew."""
        u, v, w = velocity
        h = self.spacing

        def centred(f: Any, axis: int) -> Any:
            return (jnp.roll(f, -1, axis) - jnp.roll(f, 1, axis)) / (2.0 * h)

        def onto(f: Any, own: int, other: int) -> Any:
            centre = 0.5 * (f + jnp.roll(f, -1, own))
            return 0.5 * (centre + jnp.roll(centre, 1, other))

        cu = u * centred(u, 0) + onto(v, 1, 0) * centred(u, 1) + onto(w, 2, 0) * centred(u, 2)
        cv = onto(u, 0, 1) * centred(v, 0) + v * centred(v, 1) + onto(w, 2, 1) * centred(v, 2)
        cw = onto(u, 0, 2) * centred(w, 0) + onto(v, 1, 2) * centred(w, 1) + w * centred(w, 2)
        return (-cu, -cv, -cw)

    setter(MACGrid, "convection", advective)


def _past_the_line(setter: Setter) -> None:
    setter(IMEXEuler, "cfl_line", 1e9)


def _wall(kind: str) -> Callable[[Setter], None]:
    def planted(setter: Setter) -> None:
        real = node_module.wall_fields

        def fields(grid: Any, solids: Any, band: Any, interior: Any, dtype: Any, offset: Any = None) -> WallFields:
            held = real(grid, solids, band, interior, dtype, offset)
            beyond = [held.signed_distance[c] > 0.5 * band.width * grid.spacing for c in range(3)]
            if kind == "mask":
                chi = tuple(jnp.where(beyond[c], 0.0, held.chi[c]) for c in range(3))
                return WallFields((chi[0], chi[1], chi[2]), held.inverse_permeability, held.signed_distance)
            inverse = tuple(jnp.where(beyond[c], 1e-3 * held.inverse_permeability[c], held.inverse_permeability[c])
                            for c in range(3))
            return WallFields(held.chi, (inverse[0], inverse[1], inverse[2]), held.signed_distance)

        setter(node_module, "wall_fields", fields)

    return planted


def _none(setter: Setter) -> None:
    del setter


def stops_short(factor: float) -> Callable[[Setter], None]:
    """The CG stops at a tolerance ``factor`` times the node's own, the node declaring (and reporting) its default."""

    def planted(setter: Setter) -> None:
        real = node_module.Stopping

        def stopping(**given: Any) -> Any:
            return real(**{**given, "tolerance": factor * given["tolerance"]})

        setter(node_module, "Stopping", stopping)

    return planted


#: Each plant by name: what it patches (two of the solve's take node settings instead, :func:`settings_of`).
PLANTS: dict[str, Callable[[Setter], None]] = {
    "solve_two_iterations": _none, "solve_tolerance_1e4": _none, "solve_stops_short_1e4": stops_short(1e4),
    "convection_stale": _stale,
    "convection_permuted": _permuted, "convection_advective": _advective, "convection_past_the_line": _past_the_line,
    "convection_scaled": _operator(1.5), "convection_negated": _operator(-1.0),
    "leak_mask": _wall("mask"), "leak_weak_interior": _wall("weak"),
}
#: The flag each plant is aimed at.
AIMED = {"solve_two_iterations": "stall", "solve_tolerance_1e4": "stall", "solve_stops_short_1e4": "stall",
         "convection_stale": "convection",
         "convection_permuted": "convection", "convection_advective": "convection",
         "convection_past_the_line": "convection", "convection_scaled": "convection",
         "convection_negated": "convection", "leak_mask": "leak", "leak_weak_interior": "leak"}


def settings_of(plant: str | None, dtype: str) -> dict[str, Any]:
    if plant == "solve_two_iterations":
        return {"cap": 2, "check_every": 1}
    if plant == "solve_tolerance_1e4":
        return {"tolerance": 1e4 * (1e-8 if dtype == "float64" else 1e-5)}
    return {}


def readings(point: Point, dtype: str, plant: str | None = None, flags: Any = None,
             settings: dict[str, Any] | None = None,
             ended: list[Any] | None = None) -> tuple[ImplicitFlowNode, dict[str, np.ndarray]]:
    """The point's node and its run (``plant`` applied already, by :data:`PLANTS`), each flag of the node (its active
    ones, or ``flags``) read at every step: a row a step. ``settings`` override the node's (a plant's size); the run's
    ended state is appended to ``ended`` where given (its books). The run's dtype is the caller's."""
    reynolds = point.reynolds
    drive = 16.0 * reynolds
    probe = tube_node(point.cells, g=drive, length_cells=4, dt=1e-3)
    grid = probe.grid(probe._step_constants(None))  # noqa: SLF001
    c = probe.constants()
    rng = np.random.default_rng(point.seed)
    parts = []
    for k in range(3):
        at = np.asarray(grid.face_points(k, "float64"))
        across = ((at[..., 0] - c["tube_centre"][0]) ** 2 + (at[..., 1] - c["tube_centre"][1]) ** 2) / 0.25
        swirl = point.swirl * reynolds * rng.standard_normal(grid.cells) * (across < 1.0)
        parts.append(swirl + (reynolds * np.clip(1.0 - across, 0.0, None) if k == 2 else 0.0))
    projected = grid.project(tuple(jnp.asarray(x) for x in parts))
    start = {f"initial_velocity_{x}": np.asarray(projected[k]) for k, x in enumerate("xyz")}
    peak = max(float(np.max(np.abs(v))) for v in start.values())
    line = 0.4  # IMEX Euler's line, read before any plant lifts it
    dt = (4.0 if plant == "convection_past_the_line" else 1.0) * point.fraction * line * grid.spacing / peak
    node = tube_node(point.cells, g=drive, length_cells=4, dt=dt, convection=True, ledger=METERED, **start,
                     **{**settings_of(plant, dtype), **(settings or {})})
    _STARTED["velocity"] = tuple(jnp.asarray(start[f"initial_velocity_{x}"]) for x in "xyz")
    p = node._step_constants(None)  # noqa: SLF001
    held = books.deliveries(node.constants())
    given = {books.law_input(d.field, i): jnp.asarray([0.0, 0.0, drive]) if d.field == "body_force"
             else jnp.zeros(7) for i, d in enumerate(held) if d.kind == books.LOAD}
    final, found = run_flags(node, lambda s: node.advance(p, s, given, dt), node.initial_state(), STEPS, p, flags)
    if ended is not None:
        ended.append(final)
    return node, found


__all__ = ["AIMED", "PLANTS", "STEPS", "Point", "draw", "readings", "settings_of", "stops_short"]
