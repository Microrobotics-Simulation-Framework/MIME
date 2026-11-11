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

**Every run is metered** (the record's rung 1: the ledger on in every lane F gate), its deliveries as the build makes them
(:func:`metered`), and every gate's reading carries whether each of its runs' books closed at every step (``closes``,
``mime.testing.ledger_closes``), which its pass requires: so a force at an opening that disagrees with the pressure its
law holds there (the reviewer's mutant, :data:`PLANTS`' ``patch_force``) fails the step identity on every run, whatever
the gate's own reading sees.

**The mirror's limit:** the rig is invariant under swapping the two outlets' patches, so a mix-up between an opening's
name and its patch reads the same as the right one; a rig with unequal branches (the tube tree's, through the
experiment) is what can see it.
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
from mime.nodes.flow.fvm import _open as open_module
from mime.nodes.flow.fvm._ports import LawBooks, RowKind, SharedDistal, Valve, port_matrix
from mime.testing import ledger_closes

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


def metered(node: OpenFlowNode) -> tuple[str, ...]:
    """The deliveries the rig's node takes metered, as the build makes them (``mime.build._ledger``): each law it holds
    in its step under its id and class (the viscosity's, the wall's, each opening's law's, the kinetic flux's line), a
    law at an opening that declares no class a source supplying its own work."""
    laws = set(node.constants()["opening_law_ids"])
    return tuple(books.LedgerDelivery(role, law, energy_class, books.HOSTED, "",
                                      law if law in laws and energy_class == books.UNMETERED else "").encoded()
                 for role, law, energy_class in node.hosted_laws())


def twin_node(point: Point, laws: tuple[str, ...], parameters: Any, *, law_openings: Any = None, dt: float = 1.0,
              plug: float | None = None, openings: tuple[str, ...] = (START, END_0, END_1), ids: tuple[str, ...] = (),
              **settings: Any) -> TwinNode:
    """The rig's node with ``laws`` (rows of ``parameters``) at its openings: by default ``start`` and the two ends,
    a law each in their order; ``openings`` two (``start``, ``tube.end``) for the plugged reference, its end the whole
    high plane. Metered (:func:`metered`) unless ``ledger=`` is given."""
    held = layout_of(point)
    three = len(openings) == 3

    def made(**extra: Any) -> TwinNode:
        return TwinNode("flow", dt, cells_x=held["cells_x"], cells_y=held["cells_y"], cells_z=held["cells_z"],
                        spacing=held["spacing"], viscosity=1.0, density=1.0, tube_centre=held["tube_centre"],
                        tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5, openings=openings, opening_laws=laws,
                        opening_parameters=parameters, opening_law_ids=ids,
                        opening_planes=("low", "high", "high") if three else ("low", "high"),
                        opening_centres=((held["centres"][0][0], held["centres"][0][1]), *held["centres"]) if three
                        else (), law_openings=law_openings, twin_offset=held["twin_offset"],
                        **({"twin_plug": plug} if plug is not None else {}), **settings, **extra)

    return made() if "ledger" in settings else made(ledger=metered(made()))


def closes(*finals: dict[str, Any]) -> bool:
    """Whether each run's books closed at every step (``ledger_closes``: the step identity, each law's own books)."""
    return all(ledger_closes(final).passed for final in finals)


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


def conductance_runs(point: Point, tolerance: float, columns: tuple[int, ...] = (0, 1, 2), **settings: Any
                     ) -> tuple[np.ndarray, bool]:
    """The rig's conductance over its three openings, steady Stokes (m^3/s per Pa): column ``j`` (of ``columns``) the
    outward fluxes a unit pressure at opening ``j`` drives with the others at zero, each from its own given-pressure
    run (solves of one compiled program, never the law run's own unit fields); and whether the runs' books closed."""
    node = twin_node(point, ("pressure", "pressure", "pressure"), [_row(0.0)] * 3, scheme="steady",
                     convection=False, tolerance=tolerance, **settings)
    finals = runs(node, [[_row(1.0 if k == j else 0.0) for k in range(3)] for j in columns])
    return np.stack([np.asarray(final["opening_flux"], dtype=np.float64) for final in finals], axis=1), closes(*finals)


def conductance(point: Point, tolerance: float, **settings: Any) -> np.ndarray:
    return conductance_runs(point, tolerance, **settings)[0]


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
    outlets' patches); and whether every run's books closed. ``swapped``: the solver's outlet laws swapped, never the
    network's."""
    g, closed = conductance_runs(point, tolerance)
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
            "cross": float(g[1, 2] / g[1, 1]), "closes": closed and closes(*finals)}


#: The splits' line, in the solves' tolerance: the law run's split against the network on the rig's conductances, each a
#: steady solve to that tolerance.
SPLIT_PER_TOLERANCE = 100.0


def split_passes(found: dict[str, Any], tolerance: float) -> bool:
    """Whether FB5 or the shared distal gate holds on ``found`` (:func:`split_reading`): every split within its line of
    the network's, and every run's books closed."""
    return found["closes"] and all(error <= SPLIT_PER_TOLERANCE * tolerance for error in found["errors"])


#: The stiff interiors the valve is gated at: the default (``Band()``'s 0.1h), where the solid's Darcy flow at a closed
#: outlet is a few per cent of the inflow, and 0.01h (``Band(delta_in=0.01)``), where it is a hundred times less (it
#: falls as ``delta_in^2``).
VALVE_INTERIORS = ("0.1h", "0.01h")


def walled_reference(point: Point, tolerance: float, interior: str) -> dict[str, Any]:
    """What the closed valve is judged against at ``interior``: the rig's conductances (``R_t`` and the open outlet's
    resistance from them) and the monolithic run with the second tube capped by solid from mid-length (two openings:
    the start and the whole high plane), its leak (its flux through the capped tube's half of the high plane, the cap's
    Darcy flow and the solid's around it), and whether those runs' books closed (``R_t`` from the start's column)."""
    g, closed = conductance_runs(point, tolerance, (0,), interior=interior)
    resistance = 1.0 / float(g[1, 0])
    first = point.first * resistance
    walled = steady(twin_node(point, ("inflow", "windkessel"), [_row(FLUX), resistor(first)],
                              openings=(START, "tube.end"), plug=0.5 * point.length, scheme="steady",
                              convection=False, tolerance=tolerance, interior=interior))
    held = layout_of(point)
    w = np.asarray(walled["velocity_z"], dtype=np.float64)[..., -1]
    leak = held["spacing"] ** 2 * float(np.sum(w[held["cells_x"] // 2:, :]))
    return {"resistance": resistance, "first": first, "walled": walled, "leak": leak,
            "closes": closed and closes(walled)}


def valve_node(point: Point, tolerance: float, interior: str, first: float) -> TwinNode:
    """The rig with the second outlet closed by a valve, the first a resistance ``first``, at ``interior``."""
    return twin_node(point, ("inflow", "windkessel", "valve"), [_row(FLUX), resistor(first), _row()], scheme="steady",
                     convection=False, tolerance=tolerance, interior=interior)


def valve_judged(reference: dict[str, Any], node: TwinNode, valve: dict[str, Any]) -> dict[str, Any]:
    """The valve run ``valve`` of ``node`` against ``reference`` (:func:`walled_reference`): the inlet pressures'
    difference (``moved``, over the reference's), the bound the two runs' own solid flows set on it (``bound``: the
    reference's leak and the valve's exchange, the flux its closed patch's lumen carries out while its solid faces carry
    it back in from the neighbouring patch, carried through the open tube's resistance ``R_t``), the valve's flux over
    the inflow, and whether every run's books closed."""
    resistance, walled, leak = reference["resistance"], reference["walled"], reference["leak"]
    velocity = (jnp.asarray(valve["velocity_x"]), jnp.asarray(valve["velocity_y"]), jnp.asarray(valve["velocity_z"]))
    exchange = float(np.asarray(node.opening_solid_fluxes(node._step_constants(None), velocity))[2])  # noqa: SLF001
    at_valve = np.asarray(valve["opening_pressure"], dtype=np.float64)
    at_wall = np.asarray(walled["opening_pressure"], dtype=np.float64)
    moved = at_valve[0] - at_wall[0]
    return {"moved": moved / at_wall[0], "bound": resistance * (abs(leak) + abs(exchange)) / at_wall[0],
            "pressure": abs(moved - resistance * leak) / at_wall[0], "leak": leak / FLUX, "exchange": exchange / FLUX,
            "valve_flux": float(np.asarray(valve["opening_flux"])[2]) / FLUX,
            "outlet": (at_valve[1] - at_wall[1]) / at_wall[1], "resistance": resistance,
            "open_share": resistance / (resistance + reference["first"]),
            "closes": reference["closes"] and closes(valve)}


def valve_reading(point: Point, tolerance: float, interior: str = VALVE_INTERIORS[0]) -> dict[str, Any]:
    """The closed valve against the walled-off vessel at ``interior``. Each run carries a flow of the solid's own at the
    closed outlet, which moves its open tube's flux by its size: the reference's leak through its cap and the solid
    around it; the valve's exchange (the valve holds its patch's net flux at zero, not each face's). Both are the
    solid's Darcy flow, falling as ``delta_in^2``. The inlet pressures' difference is judged against those two flows
    carried through the open tube's resistance (:func:`valve_judged`)."""
    reference = walled_reference(point, tolerance, interior)
    node = valve_node(point, tolerance, interior, reference["first"])
    return valve_judged(reference, node, steady(node))


def valve_runs(point: Point, tolerance: float, interior: str, plants: dict[str, Callable[[Setter], None] | None],
               context: Callable[[], Any]) -> dict[str, dict[str, Any]]:
    """The closed valve with each of ``plants`` (a name and its plant, ``None`` for the clean valve) against one
    walled-off reference at ``interior`` (its runs made once): each reading (:func:`valve_judged`), each plant applied
    within ``context()`` (a ``MonkeyPatch.context``, undone before the next)."""
    reference = walled_reference(point, tolerance, interior)
    found: dict[str, dict[str, Any]] = {}
    for name, plant in plants.items():
        with context() as patch:
            if plant is not None:
                plant(patch.setattr)
            node = valve_node(point, tolerance, interior, reference["first"])
            found[name] = valve_judged(reference, node, steady(node))
    return found


#: The valve gate's lines: the inlet pressures' difference within ``MARGIN`` times the two runs' own solid flows carried
#: through the open tube's resistance, beyond the solves' own (``SOLVES`` tolerances of the inlet pressure); the
#: valve's flux zero to its summation floor over the inflow, ``(N + 2) eps S``, with ``N`` under 2,000 faces a patch on
#: the rig's draws and ``S`` under twice the inflow. (The first statement, the reference's leak alone explaining the
#: difference to a tenth of it, missed on every float64 draw: the valve's own exchange through the solid is as large.)
MARGIN, SOLVES, SHUT_ULPS = 1.1, 100.0, 4e3


def valve_line(found: dict[str, Any], tolerance: float) -> float:
    """The first clause's line on ``found`` (:func:`valve_reading`): what the inlet pressures may differ by."""
    return MARGIN * found["bound"] + SOLVES * tolerance


def valve_line_holds(found: dict[str, Any], tolerance: float) -> bool:
    """The first clause alone: the inlet pressures within the line the two runs' solid flows set."""
    return abs(found["moved"]) <= valve_line(found, tolerance)


def valve_passes(found: dict[str, Any], tolerance: float, dtype: str) -> bool:
    """Whether the valve gate holds on ``found`` (:func:`valve_reading`) at the solves' ``tolerance`` in ``dtype``: the
    first clause, the valve's flux zero to its floor, and every run's books closed."""
    return valve_line_holds(found, tolerance) and abs(found["valve_flux"]) <= SHUT_ULPS * float(np.finfo(dtype).eps) \
        and found["closes"]


#: The stand-in's stiff interior (where the valve's exchange is small beside its leak: 0.01h).
STAND_IN_INTERIOR = "0.01h"


def scaled_condition(matrix: np.ndarray) -> float:
    """The condition number of ``matrix`` with each row divided by its 2-norm: a flux row is in m^3/(s Pa) and a
    pressure row is dimensionless, so the raw number mixes units and moves with them."""
    return float(np.linalg.cond(matrix / np.linalg.norm(matrix, axis=1)[:, None]))


def stand_in_reading(point: Point, tolerance: float, stiffness: tuple[float, ...] = (1e2, 1e4, 1e6, 1e8)
                     ) -> list[dict[str, float]]:
    """The stiff resistance that stands in for the closed valve (impedance-only's route), at each ``stiffness`` times
    ``R_t``: its leak over the inflow, its inlet pressure against the valve's, the port system's condition number with
    its rows scaled (:func:`scaled_condition`, of the matrix it solves, ``port_matrix``), the run's CG iterations, and
    whether its books closed; the valve's own first, its stiffness zero."""
    g = conductance_runs(point, tolerance, (0,), interior=STAND_IN_INTERIOR)[0]
    resistance = 1.0 / float(g[1, 0])
    first = point.first * resistance
    found: list[dict[str, float]] = []
    valve = twin_node(point, ("inflow", "windkessel", "valve"), [_row(FLUX), resistor(first), _row()],
                      scheme="steady", convection=False, tolerance=tolerance, interior=STAND_IN_INTERIOR)
    stiff_rows = [[_row(FLUX), resistor(first), resistor(stiff * resistance)] for stiff in stiffness]
    stand_in = twin_node(point, ("inflow", "windkessel", "windkessel"), stiff_rows[0], scheme="steady",
                         convection=False, tolerance=tolerance, interior=STAND_IN_INTERIOR)
    held = [(0.0, valve, np.asarray(valve.constants()["opening_parameters"]), steady(valve))]
    held += [(stiff, stand_in, np.asarray(rows, dtype=np.float64), final)
             for stiff, rows, final in zip(stiffness, stiff_rows, runs(stand_in, stiff_rows), strict=True)]
    reference = 0.0
    for stiff, node, parameters, final in held:
        p = node._step_constants(None)  # noqa: SLF001
        grid = node.open_grid(p)
        units = [tuple(jnp.asarray(final[f"unit_{x}"])[k] for x in "xyz") for k in range(len(node.others()))]
        response = jnp.stack([grid.opening_fluxes(unit) for unit in units], axis=1)
        _, block = node.law_rows(jnp.asarray(parameters, dtype=response.dtype), jnp.asarray(final["opening_state"]),
                                 0.0, node.delta_t)
        matrix = np.asarray(port_matrix(node.rows(), block, response, node.reference(), node.coupling()), np.float64)
        pressure = float(np.asarray(final["opening_pressure"], dtype=np.float64)[0])
        if stiff == 0.0:
            reference = pressure
        found.append({"stiffness": stiff, "leak": float(np.asarray(final["opening_flux"])[2]) / FLUX,
                      "pressure": (pressure - reference) / reference, "scaled_condition": scaled_condition(matrix),
                      "iterations": float(np.asarray(final["solve_report_sum"])[0]), "closes": float(closes(final))})
    return found


def response_gain(point: Point, dt: float, tolerance: float) -> float:
    """The largest flux a unit pressure at an outlet drives over one IMEX Euler step on the rig (m^3/s per Pa): what
    ``R_p q``, FB6's loop gain, is measured in."""
    # A measurement of the rig's response that sets FB6's gain, not a gate's run: unmetered.
    node = twin_node(point, ("inflow", "windkessel", "windkessel"), [_row(FLUX), resistor(1.0), resistor(1.0)], dt=dt,
                     scheme="imex_euler", convection=False, tolerance=tolerance, ledger=())
    final, _ = scan(node, 1)
    grid = node.open_grid(node._step_constants(None))  # noqa: SLF001
    units = [tuple(jnp.asarray(final[f"unit_{x}"])[k] for x in "xyz") for k in range(2)]
    return float(np.max(np.abs(np.asarray(jnp.stack([grid.opening_fluxes(u) for u in units], axis=1)))))


#: FB6's step and run: 40 steps a period of the pulsatile inflow, two periods; ``dt / (R_d C)`` a tenth (the spike's
#: two-outlet case: 0.13), where only the proximal coupling's loop gain decides the explicit arm's stability.
FB6_DT, FB6_STEPS, FB6_RATIO = 0.05, 80, 0.1


def fb6_reading(point: Point, gain: float, tolerance: float, explicit: bool, ratio: float = FB6_RATIO
                ) -> dict[str, float]:
    """FB6: the rig's two outlets, each an RCR whose proximal resistance is ``gain`` over the step's response
    (``R_p q``), its distal resistance 3 and 6 times a tube's Poiseuille resistance and ``dt / (R_d C)`` ``ratio``,
    under a pulsatile inflow, 80 IMEX Euler steps (Stokes): the largest outlet flux over the inflow's mean, or infinity
    where a flux is not finite, and whether the run's books closed (the implicit arm's; the explicit arm runs
    unmetered, its pressures not the field's response, so its books would be no evidence); ``explicit``, the test-only
    explicit arm (the spike's: the capacitor stepped forward,
    the proximal drop lagged a step). At ``dt / (R_d C)`` of one the explicit arm's forward capacitor alone is
    unstable whatever ``R_p`` (its amplification ``-R_d q``): the first probe's arm, unstable at every gain from
    0.34."""
    proximal = gain / response_gain(point, FB6_DT, tolerance)
    distal = tube_resistance(point.length)
    period = 40.0 * FB6_DT
    rows = [_row(FLUX, 0.3 * FLUX, 1.0 / period), _row(proximal, FB6_DT / (3.0 * distal * ratio), 3.0 * distal),
            _row(proximal, FB6_DT / (6.0 * distal * ratio), 6.0 * distal)]
    # The explicit arm runs unmetered: its pressures are not the field's response, so its books are no evidence.
    node = twin_node(point, ("inflow", "windkessel", "windkessel"), rows, dt=FB6_DT, scheme="imex_euler",
                     convection=False, tolerance=tolerance, **({"ledger": ()} if explicit else {}))
    if explicit:
        node.__class__ = ExplicitNode
    final, fluxes = scan(node, FB6_STEPS, lambda p, before, after: after["opening_flux"])
    held = np.asarray(fluxes, dtype=np.float64)[:, 1:]
    finite = bool(np.all(np.isfinite(held)))
    return {"largest": float(np.max(np.abs(held))) / FLUX if finite else float("inf"), "finite": finite,
            "closes": (not explicit) and closes(final)}


def fb6_passes(found: dict[str, Any]) -> bool:
    """FB6 on ``found`` (:func:`fb6_reading`): the outlets' fluxes within :data:`FB6_LINE` and the books closed."""
    return found["largest"] <= FB6_LINE and found["closes"]


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
    """FB6's test-only explicit arm, the open-ports spike's: each Windkessel's pressure its capacitor's at the step's
    start plus its proximal resistance times the last step's flux, given to the field (no unit field's response through
    it), its capacitor stepped by forward Euler on that flux; the flux rows still held by the port system."""

    def stepped(self, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        self._lagged = jnp.asarray(state["opening_flux"])  # noqa: SLF001
        held, parts = super().stepped(p, state, boundary_inputs, dt)
        parameters, law_state = jnp.asarray(p["opening_parameters"]), jnp.asarray(state["opening_state"])
        ended = jnp.asarray(held["opening_state"])
        for index, (law, openings) in enumerate(zip(self.laws(), self.law_openings(), strict=True)):
            if law.name == "windkessel":
                row, capacitor = parameters[index], law_state[index, 0]
                flux = self._lagged[openings[0]]
                ended = ended.at[index, 0].set(capacitor + dt * (flux - capacitor / row[2]) / row[1])
        return {**held, "opening_state": ended}, parts

    def law_rows(self, parameters: Any, law_state: Any, instant: Any, dt: Any) -> Any:
        a, block = super().law_rows(parameters, law_state, instant, dt)
        for index, (law, openings) in enumerate(zip(self.laws(), self.law_openings(), strict=True)):
            if law.name == "windkessel":
                k = openings[0]
                a = a.at[k].set(law_state[index, 0] + parameters[index, 0] * self._lagged[k])
                block = block.at[k, k].set(0.0)
        return a, block


def law_books(final: dict[str, Any]) -> float:
    """The run's worst law identity over its floor (``IDENTITY_ULPS`` eps of the law's own scale)."""
    dtype = np.asarray(final[f"{books.RECORDS}work"]).dtype
    return float(final[f"{books.RECORDS}law_identity_worst"]) / (books.IDENTITY_ULPS * float(np.finfo(dtype).eps))


#: The leaking valve's flux, a share of the rig's inflow (the valve's shut clause catches it).
LEAK = 2e-2
#: The leak the first clause alone is judged on, at each interior: above what it resolves there (the two runs' solid
#: flows, through the open tube's share of the inlet pressure; a leak lowers the inlet pressure by about its own share
#: of the inflow).
LINE_LEAK = {"0.1h": 0.2, "0.01h": 3e-3}


def leaking(share: float) -> Callable[[Setter], None]:
    """The plant: the valve's row holding a flux of ``share`` of the inflow out, not zero."""

    def planted(setter: Setter) -> None:
        def affine(self: Any, parameters: Any, state: Any, time: Any, dt: Any) -> Any:
            return jnp.asarray(share * FLUX, dtype=parameters.dtype), jnp.zeros((), dtype=parameters.dtype)

        setter(Valve, "affine", affine)

    return planted


def _leaking_by_row(setter: Setter) -> None:
    """The valve's row holding the flux its first parameter gives (a leaf: one compiled program for every size)."""

    def affine(self: Any, parameters: Any, state: Any, time: Any, dt: Any) -> Any:
        return parameters[0], jnp.zeros((), dtype=parameters.dtype)

    setter(Valve, "affine", affine)


def _opened(setter: Setter) -> None:
    """The valve's row a pressure row of zero: an outlet left open at the distal pressure, not closed."""
    setter(Valve, "row", RowKind.PRESSURE)


#: The reviewer's mutant's size: the force's share above its pressure.
PATCH_FORCE = 0.01


def _patch_force(setter: Setter) -> None:
    """The reviewer's mutant: where a plane holds several openings, the last one's patch takes the force of
    ``1 + PATCH_FORCE`` times its pressure, so the force a patch applies is not the pressure its law holds on the faces
    its flux is read from (the field does work its books do not hold)."""
    real = open_module.OpenMACGrid.port_force

    def port_force(self: Any, pressures: Any, density: Any) -> Any:
        force = real(self, pressures, density)
        members = [k for k, held in enumerate(self.layout.planes) if held == open_module.HIGH]
        if len(members) < 2:
            return force
        k = members[-1]
        p = jnp.asarray(pressures)
        mask = jnp.asarray(self.layout.masks(self.cells)[k], dtype=p.dtype)
        extra = -PATCH_FORCE * (2.0 / (density * self.spacing)) * p[k] * mask
        return (force[0], force[1], force[2].at[..., -1].add(extra))

    setter(open_module.OpenMACGrid, "port_force", port_force)


PLANTS: dict[str, Callable[[Setter], None]] = {"dropped": _dropped, "leaking": leaking(LEAK), "opened": _opened,
                                               "patch_force": _patch_force, "leaking_by_row": _leaking_by_row}

__all__ = ["END_0", "END_1", "FB6_DT", "FB6_LINE", "FB6_RATIO", "FB6_STEPS", "FLUX", "LEAK", "LINE_LEAK", "MARGIN",
           "PATCH_FORCE", "PLANTS", "SHUT_ULPS", "SOLVES", "SPLIT_PER_TOLERANCE", "START", "STAND_IN_INTERIOR",
           "VALVE_INTERIORS", "ExplicitNode", "LaggedNode", "Point", "TwinNode", "Twins", "closes", "conductance",
           "conductance_runs", "draw", "fb6_passes", "fb6_reading", "law_books", "layout_of", "leaking", "metered",
           "network_flows", "network_split", "outlets", "resistor", "response_gain", "runs", "scaled_condition",
           "shared_split", "split_of", "split_passes", "split_reading", "stand_in_reading", "steady",
           "tube_resistance_of", "twin_node", "valve_judged", "valve_line", "valve_line_holds", "valve_node", "valve_runs",
           "valve_passes", "valve_reading", "walled_reference"]
