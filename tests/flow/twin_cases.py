"""Two openings at one plane (F2b's PR 3), at the node level: two parallel tubes from one inlet plane, the open node
built as the build makes it, the correct runs the gates' lines are measured on and the planted bugs that prove them,
shared by the CI tests and the detection-rate probes.

**The rig** (:func:`twin_node`): two straight tubes of D = 1 along z (nu = rho = 1), mirror images of each other across
the box's middle plane ``x = Lx / 2`` (a draw moves them apart or together by a sub-cell ``spread`` and both across y
by a sub-cell ``lift``, never off the mirror), each in its own half of a box periodic across x and y, open at both z
planes: the low plane one opening (``start``, both tubes' inlets at one pressure), the high plane two (``end_0`` and
``end_1``, its halves, each tube's outlet with the solid around it). The grid's mirror maps each tube's faces onto the
other's, so the two tubes' discrete resistances are equal, and a 0-D network of two equal resistances from one node is
the rig's own exactly, but for the solid's Darcy flow between them: the network's resistance ``R_t`` is measured on the
same rig with given pressures (:func:`tube_resistance_of`), never Poiseuille's. **Tube 1 plugged**
(``plug=``): solid in tube 1 from that height to the high plane, the outlet ``end_1`` walled off (the valve's
reference, a monolithic run with two openings).

**The draws** (:func:`draw`): D/h 8 or 12, two to three diameters long, a sub-cell spread and lift, the outlets'
resistances 0.3 to 10 R_t with one at least twice the other (a split away from a half, which a swapped control needs).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from open_cases import scan, tube_resistance

from mime.core import ledger as books
from mime.nodes.flow.fvm import OpenFlowNode, periodic_tube_layout
from mime.nodes.flow.fvm._ports import LawBooks, RowKind, SharedDistal, Valve, port_matrix

_SEED = 60_000
START, END_0, END_1 = "tube.start", "tube.end_0", "tube.end_1"


@dataclass(frozen=True)
class Twins:
    """The solid around two parallel tubes along z of ``radius``, at ``x0 -+ offset`` and ``y0``, the second solid
    from ``plug`` up where given: the signed distance ``r - R`` of each (exact for a cylinder), the plug's
    ``max(r - R, z - plug)``, the union's the lesser (exact where the lumens are apart)."""

    x0: float
    y0: float
    offset: float
    radius: float
    plug: float | None = None
    region: str = "wall"

    def signed_distance(self, points: Any) -> Any:
        first = jnp.hypot(points[..., 0] - (self.x0 - self.offset), points[..., 1] - self.y0) - self.radius
        second = jnp.hypot(points[..., 0] - (self.x0 + self.offset), points[..., 1] - self.y0) - self.radius
        if self.plug is not None:
            second = jnp.maximum(second, points[..., 2] - self.plug)
        return jnp.minimum(first, second)


class TwinNode(OpenFlowNode):
    """The open node on :class:`Twins` (its constants ``twin_offset`` and ``twin_plug``)."""

    def solids(self, p: Any, grid: Any) -> Any:
        c = self.constants()
        centre = c["tube_centre"]
        plug = c.get("twin_plug", -1.0)
        return [Twins(float(centre[0]), float(centre[1]), float(c["twin_offset"]), float(c["tube_radius"]),
                      None if plug < 0.0 else float(plug))]


@dataclass(frozen=True)
class Point:
    cells: int
    length: float
    spread: float
    lift: float
    first: float
    second: float
    seed: int


def draw(seed: int) -> Point:
    rng = np.random.default_rng(_SEED + seed)
    first = float(math.exp(rng.uniform(math.log(0.3), math.log(10.0))))
    ratio = float(math.exp(rng.uniform(math.log(2.0), math.log(10.0))))
    second = first * ratio if rng.uniform() < 0.5 else first / ratio
    return Point(int(rng.choice([8, 12])), float(rng.choice([2.0, 3.0])), float(rng.uniform(-0.4, 0.4)),
                 float(rng.uniform(-0.4, 0.4)), first, second, seed)


def layout_of(point: Point) -> dict[str, Any]:
    """The rig's box: each half the one-tube box's width (the build's layout, its margin), the tubes at its quarters."""
    one = periodic_tube_layout(0.5, point.length, cells_across=point.cells)
    h = one.spacing
    nx, ny, nz = 2 * one.cells[0], one.cells[1], one.cells[2]
    x0, y0 = 0.5 * nx * h, (0.5 * ny + point.lift) * h
    offset = (0.25 * nx + point.spread) * h
    centres = ((x0 - offset) / h, y0 / h), ((x0 + offset) / h, y0 / h)
    return {"cells_x": nx, "cells_y": ny, "cells_z": nz, "spacing": h, "tube_centre": (x0, y0, 0.0),
            "twin_offset": offset, "centres": centres}


def twin_node(point: Point, laws: tuple[str, ...], parameters: Any, *, law_openings: Any = None, dt: float = 1.0,
              plug: float | None = None, openings: tuple[str, ...] = (START, END_0, END_1), ids: tuple[str, ...] = (),
              **settings: Any) -> TwinNode:
    """The rig's node with ``laws`` (rows of ``parameters``) at its openings: by default ``start`` and the two ends,
    a law each in their order; ``openings`` two (``start``, ``tube.end``) for the plugged reference, its end the whole
    high plane."""
    held = layout_of(point)
    three = len(openings) == 3
    return TwinNode("flow", dt, cells_x=held["cells_x"], cells_y=held["cells_y"], cells_z=held["cells_z"],
                    spacing=held["spacing"], viscosity=1.0, density=1.0, tube_centre=held["tube_centre"],
                    tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=openings, opening_laws=laws,
                    opening_parameters=parameters, opening_law_ids=ids,
                    opening_planes=("low", "high", "high") if three else ("low", "high"),
                    opening_centres=((held["centres"][0][0], held["centres"][0][1]), *held["centres"]) if three
                    else (), law_openings=law_openings, twin_offset=held["twin_offset"],
                    **({"twin_plug": plug} if plug is not None else {}), **settings)


def _row(*values: float) -> list[float]:
    return [*values, *([0.0] * (4 - len(values)))]


#: The steady inflow (m^3/s at mean speed one through one tube's area, both tubes' together).
FLUX = 2.0 * math.pi / 4.0


def resistor(resistance: float) -> list[float]:
    """A Windkessel whose capacitor follows its distal resistance within a step (``dt / (R_d C) = 1e6`` at the steady
    scheme's step of one): a resistance ``resistance`` at steady state, all of it distal."""
    return _row(0.0, 1e-6 / resistance, resistance)


def steady(node: OpenFlowNode, steps: int = 3) -> dict[str, Any]:
    final, _ = scan(node, steps)
    return final


def runs(node: OpenFlowNode, rows: list[Any], steps: int = 3) -> list[dict[str, Any]]:
    """``steps`` of ``node`` from its start at each row of its laws' parameters in ``rows``, under one compiled program
    (the parameters are its leaves)."""
    held = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    dt = node.delta_t

    def run(parameters: Any) -> Any:
        p = {**held, "opening_parameters": parameters}

        def body(s: Any, _: Any) -> tuple[Any, None]:
            return node.advance(p, s, {}, dt), None

        return jax.lax.scan(body, dict(node.initial_state()), None, length=steps)[0]

    compiled = jax.jit(run)
    dtype = np.asarray(held["opening_parameters"]).dtype
    return [dict(compiled(jnp.asarray(np.asarray(row, dtype=np.float64), dtype=dtype))) for row in rows]


def conductance(point: Point, tolerance: float, **settings: Any) -> np.ndarray:
    """The rig's conductance over its three openings, steady Stokes (m^3/s per Pa): column ``j`` the outward fluxes
    a unit pressure at opening ``j`` drives with the others at zero, each from its own given-pressure run (three
    solves of one compiled program, never the law run's own unit fields)."""
    node = twin_node(point, ("pressure", "pressure", "pressure"), [_row(0.0)] * 3, scheme="steady",
                     convection=False, tolerance=tolerance, **settings)
    finals = runs(node, [[_row(1.0 if k == j else 0.0) for k in range(3)] for j in range(3)])
    return np.stack([np.asarray(final["opening_flux"], dtype=np.float64) for final in finals], axis=1)


def tube_resistance_of(point: Point, tolerance: float) -> float:
    """``R_t`` on the rig itself: a given pressure at the start and zero at both ends, steady Stokes, ``p / q`` of
    the first tube (the second's equal by the mirror)."""
    return 1.0 / float(conductance(point, tolerance)[1, 0])


def network_flows(conductances: np.ndarray, flux: float, law: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
                  ) -> np.ndarray:
    """The outward fluxes of the rig's 0-D network on its own conductances: the start's flux ``-flux`` given, the two
    ends' pressures ``law``'s, ``p = A q`` (``A`` a 2 x 2 matrix, by ``law(conductances)``'s ``(A, offset)``), the
    field ``q = G p``."""
    resistance, offset = law(conductances)
    system = np.zeros((3, 3))
    right = np.zeros(3)
    system[0] = conductances[0]
    right[0] = -flux
    for i in (1, 2):
        system[i] = -resistance[i - 1, 0] * conductances[1] - resistance[i - 1, 1] * conductances[2]
        system[i, i] += 1.0
        right[i] = offset[i - 1]
    return conductances @ np.linalg.solve(system, right)


def network_split(resistance: float, first: float, second: float) -> float:
    """The first outlet's share of the inflow by the 0-D network: two equal tubes ``resistance`` from one node, the
    outlets' ``first`` and ``second`` after them."""
    return (resistance + second) / (2.0 * resistance + first + second)


def shared_split(resistance: float, first: float, second: float, distal: float, dropped: bool = False) -> float:
    """The first outlet's share with a shared distal resistance after both outlets: the network's own (``R_d`` drops
    out: both outlets see the same distal pressure), or with the off-diagonals dropped (``R_d`` folded into each)."""
    return network_split(resistance, first + distal, second + distal) if dropped else \
        network_split(resistance, first, second)


def split_of(final: dict[str, Any]) -> float:
    q = np.asarray(final["opening_flux"], dtype=np.float64)
    return float(q[1] / (q[1] + q[2]))


# -- the gates' readings -------------------------------------------------------------------------------------------

def outlets(point: Point, first: float, second: float, tolerance: float, *, shared: float | None = None,
            **settings: Any) -> TwinNode:
    """The rig with an inflow at the start and, at the two ends, a resistance each (``first``, ``second``) or, with
    ``shared``, the shared distal law over both (its distal resistance ``shared``), steady Stokes."""
    if shared is None:
        return twin_node(point, ("inflow", "windkessel", "windkessel"),
                         [_row(FLUX), resistor(first), resistor(second)], scheme="steady", convection=False,
                         tolerance=tolerance, **settings)
    return twin_node(point, ("inflow", "shared_distal"), [_row(FLUX), _row(first, second, shared)],
                     law_openings=((0,), (1, 2)), scheme="steady", convection=False, tolerance=tolerance, **settings)


def split_reading(point: Point, tolerance: float, *, swapped: bool = False, distal: tuple[float, ...] = ()
                  ) -> dict[str, Any]:
    """FB5 (``distal`` empty) and the shared distal gate (``distal``, its distal resistances in ``R_t``): the first
    outlet's share of the inflow against the 0-D network's on the rig's own conductances, each split's difference; the
    network with two equal tubes beside it (what the conductances' cross terms add, the solid's flow between the
    outlets' patches). ``swapped``: the solver's outlet laws swapped, never the network's."""
    g = conductance(point, tolerance)
    resistance = 1.0 / float(g[1, 0])
    first, second = point.first * resistance, point.second * resistance
    flows = network_flows(g, FLUX, lambda _: (np.diag([first, second]), np.zeros(2)))
    oracle = float(flows[1] / (flows[1] + flows[2]))
    given = (second, first) if swapped else (first, second)
    if distal:
        node = outlets(point, *given, tolerance, shared=distal[0] * resistance)
        finals = runs(node, [[_row(FLUX), _row(*given, d * resistance)] for d in distal])
    else:
        finals = runs(outlets(point, *given, tolerance), [[_row(FLUX), resistor(given[0]), resistor(given[1])]])
    splits = [split_of(final) for final in finals]
    return {"splits": splits, "oracle": oracle, "errors": [abs(x - oracle) for x in splits],
            "equal_tubes": network_split(resistance, first, second), "resistance": resistance,
            "cross": float(g[1, 2] / g[1, 1])}


#: The valve gate's stiff interior: the walled-off reference's cap leaks the solid's Darcy flow, a hundred times less
#: at 0.01h than at the default 0.1h (``delta_in^2``).
VALVE_INTERIOR = "0.01h"


def valve_reading(point: Point, tolerance: float) -> dict[str, Any]:
    """The closed valve against the walled-off vessel: the rig with the second outlet closed by a valve against the
    monolithic run with the second tube capped by solid from mid-length (two openings: the start and the whole high
    plane), at :data:`VALVE_INTERIOR`. The reference's own leak, its flux through the capped tube's half of the high
    plane (the cap's Darcy flow and the solid's around it), moves its open tube's flux by that much, so the inlet
    pressures differ by ``R_t`` times it (``R_t`` the open tube's, on the rig's conductances); the reading is what is
    left, over the inlet pressure; and the valve's flux, over the inflow."""
    g = conductance(point, tolerance, interior=VALVE_INTERIOR)
    resistance = 1.0 / float(g[1, 0])
    first = point.first * resistance
    valve = steady(twin_node(point, ("inflow", "windkessel", "valve"), [_row(FLUX), resistor(first), _row()],
                             scheme="steady", convection=False, tolerance=tolerance, interior=VALVE_INTERIOR))
    walled = steady(twin_node(point, ("inflow", "windkessel"), [_row(FLUX), resistor(first)],
                              openings=(START, "tube.end"), plug=0.5 * point.length, scheme="steady",
                              convection=False, tolerance=tolerance, interior=VALVE_INTERIOR))
    held = layout_of(point)
    w = np.asarray(walled["velocity_z"], dtype=np.float64)[..., -1]
    leak = held["spacing"] ** 2 * float(np.sum(w[held["cells_x"] // 2:, :]))
    at_valve = np.asarray(valve["opening_pressure"], dtype=np.float64)
    at_wall = np.asarray(walled["opening_pressure"], dtype=np.float64)
    moved = at_valve[0] - at_wall[0]
    return {"pressure": abs(moved - resistance * leak) / at_wall[0], "moved": moved / at_wall[0],
            "leak": leak / FLUX, "valve_flux": float(np.asarray(valve["opening_flux"])[2]) / FLUX,
            "outlet": (at_valve[1] - at_wall[1]) / at_wall[1], "resistance": resistance}


#: The valve gate's lines: the cap's leak explains the inlet pressures' difference to this share of it, beyond the
#: solves' own (``SOLVES`` tolerances of the inlet pressure); the valve's flux is zero to its summation floor over the
#: inflow, ``(N + 2) eps S``, with ``N`` under 2,000 faces a patch on the rig's draws and ``S`` under twice the inflow.
EXPLAINED, SOLVES, SHUT_ULPS = 0.1, 100.0, 4e3


def valve_passes(found: dict[str, Any], tolerance: float, dtype: str) -> bool:
    """Whether the valve gate holds on ``found`` (:func:`valve_reading`) at the solves' ``tolerance`` in ``dtype``."""
    return found["pressure"] <= EXPLAINED * abs(found["moved"]) + SOLVES * tolerance \
        and abs(found["valve_flux"]) <= SHUT_ULPS * float(np.finfo(dtype).eps)


def stand_in_reading(point: Point, tolerance: float, stiffness: tuple[float, ...] = (1e2, 1e4, 1e6, 1e8)
                     ) -> list[dict[str, float]]:
    """The stiff resistance that stands in for the closed valve (impedance-only's route), at each ``stiffness`` times
    ``R_t``: its leak over the inflow, its inlet pressure against the valve's, the port system's condition number
    (the matrix it solves, ``port_matrix``) and the run's CG iterations; the valve's own first, its stiffness zero."""
    g = conductance(point, tolerance, interior=VALVE_INTERIOR)
    resistance = 1.0 / float(g[1, 0])
    first = point.first * resistance
    found: list[dict[str, float]] = []
    reference = 0.0
    for stiff in (0.0, *stiffness):
        laws = ("inflow", "windkessel", "valve" if stiff == 0.0 else "windkessel")
        node = twin_node(point, laws, [_row(FLUX), resistor(first), _row() if stiff == 0.0 else resistor(stiff * resistance)],
                         scheme="steady", convection=False, tolerance=tolerance, interior=VALVE_INTERIOR)
        final = steady(node)
        p = node._step_constants(None)  # noqa: SLF001
        grid = node.open_grid(p)
        units = [tuple(jnp.asarray(final[f"unit_{x}"])[k] for x in "xyz") for k in range(len(node.others()))]
        response = jnp.stack([grid.opening_fluxes(unit) for unit in units], axis=1)
        _, block = node.law_rows(jnp.asarray(p["opening_parameters"]), jnp.asarray(final["opening_state"]), 0.0,
                                 node.delta_t)
        matrix = np.asarray(port_matrix(node.rows(), block, response, node.reference(), node.coupling()), np.float64)
        pressure = float(np.asarray(final["opening_pressure"], dtype=np.float64)[0])
        if stiff == 0.0:
            reference = pressure
        found.append({"stiffness": stiff, "leak": float(np.asarray(final["opening_flux"])[2]) / FLUX,
                      "pressure": (pressure - reference) / reference, "condition": float(np.linalg.cond(matrix)),
                      "iterations": float(np.asarray(final["solve_report_sum"])[0])})
    return found


def response_gain(point: Point, dt: float, tolerance: float) -> float:
    """The largest flux a unit pressure at an outlet drives over one IMEX Euler step on the rig (m^3/s per Pa): what
    ``R_p q``, FB6's loop gain, is measured in."""
    node = twin_node(point, ("inflow", "windkessel", "windkessel"), [_row(FLUX), resistor(1.0), resistor(1.0)], dt=dt,
                     scheme="imex_euler", convection=False, tolerance=tolerance)
    final, _ = scan(node, 1)
    grid = node.open_grid(node._step_constants(None))  # noqa: SLF001
    units = [tuple(jnp.asarray(final[f"unit_{x}"])[k] for x in "xyz") for k in range(2)]
    return float(np.max(np.abs(np.asarray(jnp.stack([grid.opening_fluxes(u) for u in units], axis=1)))))


#: FB6's step and run: 40 steps a period of the pulsatile inflow, two periods.
FB6_DT, FB6_STEPS = 0.05, 80


def fb6_reading(point: Point, gain: float, tolerance: float, explicit: bool) -> dict[str, float]:
    """FB6: the rig's two outlets, each an RCR whose proximal resistance is ``gain`` over the step's response
    (``R_p q``), its distal resistance 3 and 6 times a tube's Poiseuille resistance and ``dt / (R_d C)`` one, under a
    pulsatile inflow, 80 IMEX Euler steps (Stokes): the largest outlet flux over the inflow's mean, or
    infinity where a flux is not finite; ``explicit``, the test-only explicit arm (each pressure row lagged whole)."""
    proximal = gain / response_gain(point, FB6_DT, tolerance)
    distal = tube_resistance(point.length)
    period = 40.0 * FB6_DT
    rows = [_row(FLUX, 0.3 * FLUX, 1.0 / period), _row(proximal, FB6_DT / (3.0 * distal), 3.0 * distal),
            _row(proximal, FB6_DT / (6.0 * distal), 6.0 * distal)]
    node = twin_node(point, ("inflow", "windkessel", "windkessel"), rows, dt=FB6_DT, scheme="imex_euler",
                     convection=False, tolerance=tolerance)
    if explicit:
        node.__class__ = ExplicitNode
    _, fluxes = scan(node, FB6_STEPS, lambda p, before, after: after["opening_flux"])
    held = np.asarray(fluxes, dtype=np.float64)[:, 1:]
    finite = bool(np.all(np.isfinite(held)))
    return {"largest": float(np.max(np.abs(held))) / FLUX if finite else float("inf"), "finite": finite}


#: FB6's line: a stable run's outlet fluxes stay within three times the inflow's mean.
FB6_LINE = 3.0


# -- the plants ---------------------------------------------------------------------------------------------------

Setter = Callable[[Any, str, Any], None]


def _dropped(setter: Setter) -> None:
    """The shared distal law's off-diagonals dropped (each R_d folded into its own row), its books consistent with
    the dropped row: what only the split can see (the record's control (a))."""

    def affine(self: Any, parameters: Any, state: Any, time: Any, dt: Any) -> Any:
        distal = parameters[2]
        return jnp.zeros(2, dtype=parameters.dtype), jnp.diag(jnp.stack([parameters[0] + distal,
                                                                         parameters[1] + distal]))

    def booked(self: Any, parameters: Any, state: Any, ended: Any, flux: Any, pressure: Any, dt: Any) -> Any:
        zero = jnp.zeros((), dtype=flux.dtype)
        loss = dt * ((parameters[0] + parameters[2]) * flux[0] ** 2 + (parameters[1] + parameters[2]) * flux[1] ** 2)
        return LawBooks(zero, zero, loss, zero)

    setter(SharedDistal, "affine", affine)
    setter(SharedDistal, "books", booked)


class LaggedNode(TwinNode):
    """The rig with each law over several openings coupled through its off-diagonals at the last step's fluxes (the
    record's control (b)): the row's ``a`` takes ``B_ij q_j`` of the step before, its block the diagonal alone."""

    def stepped(self, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        self._lagged = jnp.asarray(state["opening_flux"])  # noqa: SLF001 - read by law_rows within this trace
        return super().stepped(p, state, boundary_inputs, dt)

    def law_rows(self, parameters: Any, law_state: Any, instant: Any, dt: Any) -> Any:
        a, block = super().law_rows(parameters, law_state, instant, dt)
        diagonal = jnp.diag(jnp.diag(block))
        return a + (block - diagonal) @ self._lagged, diagonal

    def coupling(self) -> list[list[bool]]:
        count = self.opening_count()
        return [[i == j for j in range(count)] for i in range(count)]


class ExplicitNode(TwinNode):
    """FB6's test-only explicit arm: each pressure row lagged whole, its pressure ``a + B q`` at the last step's fluxes
    given to the field (no unit field's response through it), the flux rows still held by the port system."""

    def stepped(self, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        self._lagged = jnp.asarray(state["opening_flux"])  # noqa: SLF001
        return super().stepped(p, state, boundary_inputs, dt)

    def law_rows(self, parameters: Any, law_state: Any, instant: Any, dt: Any) -> Any:
        a, block = super().law_rows(parameters, law_state, instant, dt)
        return a + block @ self._lagged, jnp.zeros_like(block)


def law_books(final: dict[str, Any]) -> float:
    """The run's worst law identity over its floor (``IDENTITY_ULPS`` eps of the law's own scale)."""
    dtype = np.asarray(final[f"{books.RECORDS}work"]).dtype
    return float(final[f"{books.RECORDS}law_identity_worst"]) / (books.IDENTITY_ULPS * float(np.finfo(dtype).eps))


#: The leaking valve's flux, a share of the rig's inflow.
LEAK = 2e-2


def _leaking(setter: Setter) -> None:
    """The valve's row holding a flux of :data:`LEAK` of the inflow out, not zero."""

    def affine(self: Any, parameters: Any, state: Any, time: Any, dt: Any) -> Any:
        return jnp.asarray(LEAK * FLUX, dtype=parameters.dtype), jnp.zeros((), dtype=parameters.dtype)

    setter(Valve, "affine", affine)


def _opened(setter: Setter) -> None:
    """The valve's row a pressure row of zero: an outlet left open at the distal pressure, not closed."""
    setter(Valve, "row", RowKind.PRESSURE)


PLANTS: dict[str, Callable[[Setter], None]] = {"dropped": _dropped, "leaking": _leaking, "opened": _opened}

__all__ = ["END_0", "END_1", "FB6_DT", "FB6_LINE", "FB6_STEPS", "FLUX", "LEAK", "PLANTS", "START", "VALVE_INTERIOR",
           "ExplicitNode", "LaggedNode", "Point", "TwinNode", "Twins", "conductance", "draw", "fb6_reading", "law_books",
           "layout_of", "network_flows", "network_split", "outlets", "resistor", "response_gain", "runs",
           "shared_split", "split_of", "split_reading", "stand_in_reading", "steady", "tube_resistance_of", "twin_node",
           "valve_passes", "valve_reading"]
