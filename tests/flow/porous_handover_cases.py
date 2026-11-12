"""Shared set-ups for the porous plug's hand-over gates (``tests/flow/predictions_porous_handover.json``): a plug below
the line held at the line's permeability, its throughflow read one way from its face pressures, and the forced-flow
test. Units where D, nu and rho are one (``porous_cases``, ``open_cases``, ``twin_cases``), but for the experiments.
Each plant is switched by ``MIME_PLANT_<NAME>`` (:func:`planted`), each draw seeded by ``DETECTION_SEED``, for
``.github/probe/detection_rate.py``."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from open_cases import open_node, scan, tube_resistance
from porous_cases import R, plant, plug_node, seed, series_flux, steady
from twin_cases import FLUX, Point, TwinNode, layout_of, resistor
from twin_cases import draw as twin_draw

import mime.nodes.flow.fvm._node as flow_node
import mime.nodes.flow.fvm._porous as porous
from mime.nodes.flow.fvm import ImplicitFlowNode, OpenFlowNode
from mime.nodes.flow.fvm._node import porous_leaf, porous_leaves
from mime.nodes.flow.fvm._solve import DEFAULT_CAP

LAW = "darcy:clot"
#: The plants, each by its variable's name: the reading's permeability doubled; the face pressures' signs flipped; the
#: forced-flow comparison disabled; the topology test answering "forced" for every region; the face pressures, or the
#: permeability in the reading, held from the derivative; a porous flow's solve gone not finite left unnamed; plugs in
#: series left unrefused; each region's mid-section flux 5 % high; a non-finite step counted as the solve's own whatever
#: its start.
PLANTS = ("READING_K2", "FACE_SIGN", "FORCED_OFF", "FORCED_EVERYWHERE", "FACE_STOP", "K_STOP", "NOT_FINITE_SILENT",
          "SERIES_SILENT", "SECTION_HIGH", "NOT_FINITE_ANY")


@contextlib.contextmanager
def planted() -> Iterator[None]:
    """The plants switched on in the environment, applied for a block (each read where the step is traced)."""
    reading, faces = porous.darcy_reading, ImplicitFlowNode.porous_face_pressures
    sections, finite = ImplicitFlowNode.porous_section_fluxes, flow_node.all_finite
    with pytest.MonkeyPatch.context() as patched:
        if plant("READING_K2"):
            patched.setattr(porous, "darcy_reading", lambda k, *rest: reading(2.0 * k, *rest))
        if plant("K_STOP"):
            patched.setattr(porous, "darcy_reading", lambda k, *rest: reading(jax.lax.stop_gradient(k), *rest))
        if plant("FACE_SIGN"):
            patched.setattr(ImplicitFlowNode, "porous_face_pressures", lambda self, *a: -faces(self, *a))
        if plant("FACE_STOP"):
            patched.setattr(ImplicitFlowNode, "porous_face_pressures",
                            lambda self, *a: jax.lax.stop_gradient(faces(self, *a)))
        if plant("FORCED_OFF"):
            patched.setattr(porous, "forced_through", lambda *a: jnp.asarray(False))
        if plant("FORCED_EVERYWHERE"):
            for owner in (ImplicitFlowNode, OpenFlowNode):
                patched.setattr(owner, "through_plug", lambda self, closing: [jnp.asarray(True) for _ in closing])
        if plant("NOT_FINITE_SILENT"):
            patched.setattr(ImplicitFlowNode, "porous_not_finite", lambda self, state: None)
        if plant("SERIES_SILENT"):
            patched.setattr(ImplicitFlowNode, "porous_series_problem", lambda self, state: None)
        if plant("SECTION_HIGH"):
            patched.setattr(ImplicitFlowNode, "porous_section_fluxes",
                            lambda self, *a: [1.05 * flux for flux in sections(self, *a)])
        if plant("NOT_FINITE_ANY"):
            # The step's start, drive and resistivity read finite whatever they are: only the end is judged.
            patched.setattr(flow_node, "all_finite", lambda *arrays: finite(*arrays) if len(arrays) == 3 else
                            jnp.asarray(True))
        yield


@contextlib.contextmanager
def line_at(line: float) -> Iterator[None]:
    """The hand-over's line moved to ``line`` (``sqrt(K)/h``, every box type and interior) for a block: the overlap
    gate's hand-over arm, a plug above the real line read as the route below it reads one; the open line's probe."""
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(porous, "HANDOVER_LINE", {box: {"0.1h": ((math.inf, line),), "0.01h": ((math.inf, line),)}
                                                  for box in porous.HANDOVER_LINE})
        patched.setattr(porous, "HANDOVER_FROM", {box: 0.0 for box in porous.HANDOVER_FROM})
        yield


def line_of(dh: float, interior: str, box: str) -> float:
    """The line (``sqrt(K)/h``) at D/h ``dh``."""
    handover = porous.handover_of(box, interior, dh)
    assert handover is not None
    return handover.line


def line_permeability(dh: int, interior: str, box: str) -> float:
    """The line's permeability in a tube of D = 1 at D/h ``dh``."""
    return (line_of(dh, interior, box) / dh) ** 2


# -- gates 1 and 3: the periodic tube ------------------------------------------------------------------------------

def tube_reading(dh: int, root: float, interior: str, *, length: float = 0.5, period: float = 2.0,
                 line: float | None = None, cap: int = DEFAULT_CAP, steps: int = 1, **settings: Any) -> dict[str, float]:
    """A plug ``length`` long closing the periodic tube in a period ``period`` (diameters), ``sqrt(K)/h = root``,
    driven by ``g``, steady Stokes (or ``settings``' scheme, ``steps`` steps): the run's ``lumen_flux``, ``porous_flux``
    and flags, against the series closed form; with ``line`` the hand-over's line moved there (:func:`line_at`)."""
    from fvm_cases import run

    K = (root / dh) ** 2
    with planted(), (line_at(line) if line is not None else contextlib.nullcontext()):
        node = plug_node(dh, [dict(radius=R, length=length, permeability=K)], length_cells=int(period * dh),
                         interior=interior, cap=cap, **settings)
        state = steady(node) if steps == 1 else run(node, steps)
    closed = series_flux(K, length, period)
    report = np.asarray(state["solve_report"], np.float64)
    return {"lumen": float(state["lumen_flux"]) / closed, "reading": float(np.asarray(state["porous_flux"])[0]) / closed,
            "series": closed, "iterations": float(report[0]), "residual": float(report[1]),
            "solve_flagged": float(state["solve_flagged"]), "porous_flagged": float(state["porous_flagged"]),
            "forced_flagged": float(state["forced_flagged"])}


def overlap(dh: int, root: float, interior: str, **settings: Any) -> dict[str, float]:
    """Gate 1: the resolved route's ``lumen_flux`` at ``sqrt(K)/h = root`` and the hand-over route's reading of the same
    plug, its line moved to three times ``root`` (the solve holding it at nine times its permeability, a wall to within
    that plug's flux), and their ratio."""
    resolved = tube_reading(dh, root, interior, **settings)
    handed = tube_reading(dh, root, interior, line=3.0 * root, **settings)
    return {"resolved": resolved["lumen"], "reading": handed["reading"], "ratio": handed["reading"] / resolved["lumen"],
            "flagged": resolved["solve_flagged"] + handed["solve_flagged"]}


# -- gate 2: the forced-flow test ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Forced:
    """A drawn plug below the line: its permeability over the line's, its length and centre along the vessel (in
    diameters), and the inflow (in units of a mean speed of one)."""

    ratio: float
    length: float
    at: float
    inflow: float


def forced_draw(index: int) -> Forced:
    rng = np.random.default_rng(90_000 + index)
    return Forced(float(math.exp(rng.uniform(math.log(1e-4), math.log(0.5)))), float(rng.uniform(0.3, 0.8)),
                  float(rng.uniform(0.7, 1.3)), float(math.exp(rng.uniform(math.log(0.01), math.log(10.0)))))


def vessel(drawn: Forced, *, cells: int = 16, radius: float = R, interior: str = "0.1h",
           laws: tuple[str, str] = ("inflow", "windkessel"), steps: int = 2, **settings: Any) -> dict[str, Any]:
    """An open tube two diameters long, the drawn inflow at its start (or a given pressure, ``laws`` ``("pressure",
    "windkessel")``) and a resistance of three tubes' at its end, steady Stokes, no body force, a plug of ``radius`` (the lumen's: closing it) on its axis."""
    dh = cells
    K = drawn.ratio * line_permeability(dh, interior, "open")
    resistance = 3.0 * tube_resistance(2.0)
    flux = drawn.inflow * math.pi / 4.0
    # A given pressure that would drive the inflow through the open vessel and its outlet, where the start holds one.
    first = [flux, 0.0, 0.0, 0.0] if laws[0] == "inflow" else [flux * 4.0 * resistance / 3.0, 0.0, 0.0, 0.0]
    probe = open_node(cells, laws=laws, parameters=[first, resistor(resistance)])
    centre = (*probe.constants()["tube_centre"][:2], drawn.at)
    with planted():
        node = open_node(cells, laws=laws, parameters=[first, resistor(resistance)], dt=1.0, scheme="steady",
                         convection=False, interior=interior, porous_laws=[LAW],
                         **porous_leaves([dict(centre=centre, axis=(0.0, 0.0, 1.0), radius=radius, length=drawn.length,
                                               permeability=K)]), **settings)
        final, _ = scan(node, steps)
    return final


class PluggedTwins(TwinNode):
    """The twin tubes' node (``twin_cases``), each porous region's gap read against the tube it sits in, the nearer
    across x of the two at ``x0 -+ offset`` (the node's own straight tube is the box's middle, between them)."""

    def porous_sections(self, p: Any, grid: Any) -> Any:
        c = self.constants()
        centre, offset = c["tube_centre"], float(c["twin_offset"])
        tubes = [(float(centre[0]) + side * offset, float(centre[1]), 0.0) for side in (-1.0, 1.0)]
        read = [super(PluggedTwins, self).porous_sections({**p, "tube_centre": jnp.asarray(tube)}, grid)
                for tube in tubes]
        nearer = [int(float(c[porous_leaf(i, "centre")][0]) > float(centre[0])) for i in range(len(self.porous_laws))]
        return [read[k][0][i] for i, k in enumerate(nearer)], [read[k][1][i] for i, k in enumerate(nearer)]


def occluded_twin(point: Point, ratio: float, *, interior: str = "0.1h", **settings: Any) -> dict[str, Any]:
    """Two parallel tubes from one inlet plane (``twin_cases``): a fixed inflow at the start, a resistance of three
    tubes' at each outlet, steady Stokes, the second tube closed at mid-length by a plug below the line; the flow
    takes the first. Two steady steps' final state (``settings``: :func:`twin_node`'s)."""
    with planted():
        final, _ = scan(twin_node(point, ratio, interior=interior, **settings), 2)
    return final


#: The twin rig's outlet laws: each a resistance (the flow takes the open tube), or the first tube's outlet a closed
#: valve (every bit of the inflow is the plugged second tube's).
TWIN_LAWS = {"open": ("inflow", "windkessel", "windkessel"), "valve": ("inflow", "valve", "windkessel")}


def twin_node(point: Point, ratio: float, *, interior: str = "0.1h", outlets: str = "open",
              tubes: tuple[int, ...] = (1,)) -> PluggedTwins:
    """:func:`occluded_twin`'s node, a plug at ``ratio`` of the open line's permeability at mid-length of each tube in
    ``tubes`` (0 the first, 1 the second), its outlets' laws :data:`TWIN_LAWS`'s ``outlets`` (the plants act where its
    step is traced, under :func:`planted`)."""
    held = layout_of(point)
    h = held["spacing"]
    K = ratio * (line_of(point.cells, interior, "open") * h) ** 2
    resistance = 3.0 * tube_resistance(point.length)
    laws = TWIN_LAWS[outlets]
    rows = {"inflow": [FLUX, 0.0, 0.0, 0.0], "windkessel": resistor(resistance), "valve": [0.0, 0.0, 0.0, 0.0]}
    plugs = [dict(centre=(held["tube_centre"][0] + (2 * tube - 1) * held["twin_offset"], held["tube_centre"][1],
                          0.5 * held["cells_z"] * h), axis=(0.0, 0.0, 1.0), radius=0.5, length=0.5, permeability=K)
             for tube in tubes]
    return PluggedTwins(
        "flow", 1.0, cells_x=held["cells_x"], cells_y=held["cells_y"], cells_z=held["cells_z"], spacing=h,
        viscosity=1.0, density=1.0, tube_centre=held["tube_centre"], tube_axis=(0.0, 0.0, 1.0), tube_radius=0.5,
        openings=("tube.start", "tube.end_0", "tube.end_1"), opening_laws=laws,
        opening_parameters=[rows[law] for law in laws], opening_planes=("low", "high", "high"),
        opening_centres=((held["centres"][0][0], held["centres"][0][1]), *held["centres"]),
        twin_offset=held["twin_offset"], scheme="steady", convection=False, interior=interior,
        porous_laws=[LAW if i == 0 else f"{LAW}_{i}" for i in range(len(plugs))], **porous_leaves(plugs))


def quiet_draw(index: int) -> tuple[Point, float]:
    """The twin rig's drawn point and a plug's permeability over the line's (log-uniform 1e-4 to 0.5)."""
    rng = np.random.default_rng(91_000 + index)
    return twin_draw(index), float(math.exp(rng.uniform(math.log(1e-4), math.log(0.5))))


def not_finite_draw(index: int) -> tuple[Point, float]:
    """The ``index``-th of :func:`quiet_draw`'s first 50 draws at D/h 12 (30 of them, cycled), where the twin rig's held
    solve at the periodic box's 0.1h line, sqrt(K)/h 0.003, went not finite on every one (run 43201390806)."""
    twelve = [seed for seed in range(50) if twin_draw(seed).cells == 12]
    return quiet_draw(twelve[index % len(twelve)])


def seeded() -> int:
    return seed()


# -- plugs in series ------------------------------------------------------------------------------------------------

def series_closed(plugs: list[tuple[float, float]], period: float) -> float:
    """The periodic tube's flux through plugs ``(K, length)`` closing it in series in a period ``period``: the drive
    over the open part's Poiseuille resistance and each plug's Darcy resistance with Brinkman's wall factor."""
    from porous_cases import G, open_resistance, plug_resistance

    open_length = period - sum(length for _, length in plugs)
    return G * period / (open_resistance(open_length) + sum(plug_resistance(K, length) for K, length in plugs))


def series_tube(dh: int, roots: tuple[float, float], interior: str = "0.01h", length: float = 0.5,
                period: float = 4.0) -> tuple[ImplicitFlowNode, dict[str, Any], float]:
    """Two plugs ``length`` long closing a periodic tube at a quarter and three quarters of a period ``period``, their
    ``sqrt(K)/h`` ``roots``, one cold steady step at the default cap, under :func:`planted`: the node, its final state
    and the series closed form."""
    Ks = [(root / dh) ** 2 for root in roots]
    with planted():
        node = plug_node(dh, [dict(radius=R, length=length, permeability=K, at=at) for K, at in zip(Ks, (0.25, 0.75))],
                         length_cells=int(period * dh), interior=interior, cap=DEFAULT_CAP)
        final = steady(node)
    return node, final, series_closed([(K, length) for K in Ks], period)


# -- gate 4: the reading's derivative in the open vessel -----------------------------------------------------------

def gradient_vessel(cells: int = 8, ratio: float = 0.3, interior: str = "0.01h", tolerance: float = 1e-12,
                    steps: int = 2) -> tuple[Any, Any, Any]:
    """The open vessel of :func:`vessel` at D/h ``cells`` (an inflow of mean speed one, a plug closing it at ``ratio``
    of the line's permeability): ``outputs(x)``, its plug's ``porous_flux`` and two face pressures after ``steps``
    steady steps as a function of ``x = (K, inflow)``, under :func:`planted`, and ``x`` at the case's values."""
    resistance = 3.0 * tube_resistance(2.0)
    K = ratio * line_permeability(cells, interior, "open")
    flux = math.pi / 4.0
    probe = open_node(cells)
    centre = (*probe.constants()["tube_centre"][:2], 1.0)
    node = open_node(cells, parameters=[[flux, 0.0, 0.0, 0.0], resistor(resistance)], dt=1.0, scheme="steady",
                     convection=False, interior=interior, tolerance=tolerance, porous_laws=[LAW],
                     **porous_leaves([dict(centre=centre, axis=(0.0, 0.0, 1.0), radius=R, length=0.5,
                                           permeability=K)]))
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants

    def outputs(x: Any) -> Any:
        rows = jnp.asarray(p["opening_parameters"]).at[0, 0].set(x[1])
        held = {**p, "porous_0_permeability": x[0], "opening_parameters": rows}

        def body(s: Any, _: Any) -> tuple[Any, None]:
            return node.advance(held, s, {}, node.delta_t), None

        ended = jax.lax.scan(body, dict(node.initial_state()), None, length=steps)[0]
        return jnp.concatenate([ended["porous_flux"], ended["porous_face_pressure"][0]])

    return outputs, jnp.asarray([K, flux], dtype=jnp.float64), node


# -- the plug beside a resolved body -------------------------------------------------------------------------------

def beside_a_body(permeability: float, *, bead: bool = True, period: float = 2.0, delta_in: float = 0.1) -> Any:
    """A periodic tube of 1 mm, ``period`` mm long at a spacing of 50 um (D/h 20, the bead's three cells a radius, with
    or without it), its wall's interior ``delta_in``, steady Stokes driven by a body force of 1 m/s^2: a plug 0.5 mm
    long closing it at z 1.25 mm of declared ``permeability``, and with ``bead`` a sphere of 150 um resolved on the grid
    (``Drag.Penalty``), 1.2 times water's density, on the axis at z 0.25 mm, at rest at the first step; metered."""
    from mime.dynamics import FVM, Band, RigidInertial, Scheme
    from mime.experiment import Experiment
    from mime.interactions import BodyForce, Darcy, Drag
    from mime.observations import EnergyLedger
    from mime.scene import PeriodicBox, Plug, Scene, Sphere, Tube, Z
    from mime.things import Body, Medium, Part, State, Structure
    from mime.units import Pa_s, kg, kg_m2, kg_per_m3, m2, m_per_s, m_per_s2, mm, ms, rad_per_s, um

    exp = Experiment("plug_beside_a_body", scene=Scene.new(up=Z, units="m"), dt=ms(0.25))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=mm(period), margin=4), spacing=um(50.0), stokes=True, scheme=Scheme.Steady(),
               wall=Band(delta_in=delta_in))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.add_structure(Structure("clot", geometry=Plug(axis=+Z, radius=mm(0.5), length=mm(0.5),
                                                      centre=mm((0.0, 0.0, 1.25)))))
    exp.declare("clot.permeability", m2(permeability), reason="a gate's value")
    exp.attach(Darcy(), medium="water", structure="clot")
    if bead:
        radius = 150e-6
        mass = 1.2 * 1000.0 * 4.0 / 3.0 * math.pi * radius ** 3
        exp.add_body(Body("bead", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=um(150.0)))},
                          initial=State(position=mm((0.0, 0.0, 0.25)), orientation=(1.0, 0.0, 0.0, 0.0),
                                        velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        exp.declare("bead.mass", kg(mass), reason="a bead 1.2 times water's density")
        exp.declare("bead.inertia", kg_m2((0.4 * mass * radius ** 2,) * 3), reason="a uniform sphere")
        exp.attach(Drag.Penalty(feature_cells=3), body="bead", medium="water")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    exp.observe(EnergyLedger())
    return exp


@contextlib.contextmanager
def body_twice() -> Iterator[None]:
    """The resolved node's books as before this PR where ``MIME_PLANT_BODY_TWICE`` is set: each body's own loss taken off
    the walls' penalty though, with porous regions, the walls' penalty never held it."""
    from mime.nodes.flow.fvm import _resolved_node as resolved

    original = resolved.ResolvedFlowNode.energy_terms
    with pytest.MonkeyPatch.context() as patched:
        if plant("BODY_TWICE"):
            def twice(self: Any, p: Any, grid: Any, before: Any, stage: Any, sigma: Any, fields: Any, dt: float) -> Any:
                terms = original(self, p, grid, before, stage, sigma, fields, dt)
                own = sum(self.penalty_work(p, fields, stage.solved, index, dt)[0] for index in range(self.count))
                return {**terms, "penalty": terms["penalty"] - own}

            patched.setattr(resolved.ResolvedFlowNode, "energy_terms", twice)
        yield
