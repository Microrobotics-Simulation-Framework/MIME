"""Shared set-ups for the porous plug's gates: a plug in the periodic tube's box built through the flow node (D = 1,
nu = rho = 1, g = 8, as ``fvm_cases``), the experiment form (a ``Plug`` structure, ``Darcy`` attached), and the closed
forms: Brinkman's flow in a tube filled with a porous medium, the Brinkman core matched to a Stokes annulus, and the
series of the plug's resistance with the open tube's. Each plant is switched by ``MIME_PLANT_<NAME>``, each draw
seeded by ``DETECTION_SEED``, for ``.github/probe/detection_rate.py``."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from fvm_cases import run, tube_node
from scipy.special import i0e, i1e

from mime.dynamics import FVM, Band, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce, Darcy
from mime.nodes.flow.fvm import ImplicitFlowNode, periodic_tube_layout
from mime.nodes.flow.fvm._node import porous_leaves
from mime.scene import PeriodicBox, Plug, Scene, Tube, Z
from mime.things import Medium, Structure
from mime.units import kg_per_m3, m2, m_per_s2, mm, mPa_s, ms

G, MU, R = 8.0, 1.0, 0.5
LAW = "darcy:clot"


def plant(name: str) -> str:
    """The plant switched on for this run (``MIME_PLANT_<NAME>``'s value), or an empty string."""
    return os.environ.get(f"MIME_PLANT_{name}", "")


def seed() -> int:
    return int(os.environ.get("DETECTION_SEED", "0"))


def brinkman_flux(K: float, radius: float = R, g: float = G, mu: float = MU) -> float:
    """Brinkman's flux in a tube filled with a porous medium, ``(G K pi R^2 / mu)(1 - 2 sqrt(K) I1 / (R I0))``,
    ``I0, I1`` at ``R / sqrt(K)``."""
    s = math.sqrt(K)
    return g * K * math.pi * radius ** 2 / mu * (1.0 - 2.0 * s / radius * i1e(radius / s) / i0e(radius / s))


def brinkman_profile(K: float, r: Any, radius: float = R, g: float = G, mu: float = MU) -> Any:
    """Brinkman's profile in the tube, ``(G K / mu)(1 - I0(r / sqrt K) / I0(R / sqrt K))``."""
    s = math.sqrt(K)
    r = np.asarray(r, dtype=np.float64)
    return g * K / mu * (1.0 - i0e(r / s) / i0e(radius / s) * np.exp((r - radius) / s))


def composite(K: float, a: float, radius: float = R, g: float = G, mu: float = MU) -> tuple[float, float]:
    """The total flux and the core's of a coaxial Brinkman core of radius ``a`` in a tube of radius ``R``, both driven
    by ``G``: inside ``u = G K / mu + A I0(r / sqrt K)``, in the annulus ``u = -G r^2 / (4 mu) + B ln r + C``, the
    velocity and its derivative continuous at ``a`` and zero at ``R``."""
    s = math.sqrt(K)
    ratio = i1e(a / s) / i0e(a / s)
    matrix = np.array([[1.0, -math.log(a), -1.0], [ratio / s, -1.0 / a, 0.0], [0.0, math.log(radius), 1.0]])
    rhs = np.array([-g * K / mu - g * a ** 2 / (4 * mu), -g * a / (2 * mu), g * radius ** 2 / (4 * mu)])
    A, B, C = np.linalg.solve(matrix, rhs)
    core = 2 * math.pi * (g * K * a ** 2 / (2 * mu) + A * s * a * ratio)

    def ring(r: float) -> float:
        return r ** 2 / 2 * math.log(r) - r ** 2 / 4

    annulus = 2 * math.pi * (-g * (radius ** 4 - a ** 4) / (16 * mu) + B * (ring(radius) - ring(a))
                             + C * (radius ** 2 - a ** 2) / 2)
    return core + annulus, core


def open_resistance(length: float, radius: float = R, mu: float = MU) -> float:
    """Poiseuille's resistance of an open tube ``length`` long, ``8 mu L / (pi R^4)``."""
    return 8 * mu * length / (math.pi * radius ** 4)


def plug_resistance(K: float, length: float, radius: float = R, mu: float = MU) -> float:
    """A plug's resistance filling the tube, Darcy's ``mu L / (K pi R^2)`` over Brinkman's wall factor ``F_B = 1 - 2
    sqrt(K) I1 / (R I0)``."""
    factor = brinkman_flux(K, radius, 1.0, mu) / (K * math.pi * radius ** 2 / mu)
    return mu * length / (K * math.pi * radius ** 2 * factor)


def series_flux(K: float, length: float, period: float, radius: float = R, g: float = G, mu: float = MU) -> float:
    """The tube's flux through a plug filling it, ``length`` long in a period ``period``: the drive ``rho g period``
    over the open part's Poiseuille resistance and the plug's in series."""
    return g * period / (open_resistance(period - length, radius, mu) + plug_resistance(K, length, radius, mu))


def plug_node(dh: int, regions: list[dict[str, Any]], *, length_cells: int = 2, interior: str = "0.1h",
              **settings: Any) -> ImplicitFlowNode:
    """A periodic tube's box (``fvm_cases.tube_node``, steady Stokes) holding ``regions``, each a mapping of its leaves
    with its centre given across the axis only (``offset``, m, default on the axis) and along it as a share of the
    period (``at``, default the middle)."""
    layout = periodic_tube_layout(R, length_cells / dh, cells_across=dh)
    placed = []
    for region in regions:
        held = dict(region)
        offset = held.pop("offset", (0.0, 0.0))
        at = held.pop("at", 0.5)
        held.setdefault("axis", (0.0, 0.0, 1.0))
        held["centre"] = (layout.centre[0] + offset[0], layout.centre[1] + offset[1],
                          at * layout.cells[2] * layout.spacing)
        placed.append(held)
    settings.setdefault("scheme", "steady")
    settings.setdefault("dt", 0.5)
    return tube_node(dh, convection=False, length_cells=length_cells, interior=interior,
                     porous_laws=[LAW if i == 0 else f"darcy:clot_{i}" for i in range(len(placed))],
                     **porous_leaves(placed), **settings)


def box_flux(node: ImplicitFlowNode, state: Any) -> float:
    """The box's whole axial flux (every face, the lumen's and the solid's), the same through every cross-section."""
    c = node.constants()
    h = c["spacing"]
    return float(np.sum(np.asarray(state["velocity_z"], np.float64)) * h ** 3 / (c["cells_z"] * h))


def steady(node: ImplicitFlowNode) -> dict[str, Any]:
    return run(node, 1)


def experiment(*, permeability: float = 1e-10, radius: float = 0.5, length: float = 1.0, centre: tuple = (0, 0, 0),
               period: float = 4.0, cells_across: int = 16, delta_in: float = 0.1, scheme: Any = "steady",
               plug: bool = True, axis: Any = +Z) -> Experiment:
    """A plug in a periodic water-filled tube of 1 mm (lengths in mm), driven by a body force of 1 m/s^2, through the
    experiment: ``Structure("clot", geometry=Plug(...))``, its permeability declared, ``Darcy`` attached."""
    exp = Experiment("porous_plug", scene=Scene.new(up=Z, units="m"), dt=ms(20.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=axis, inner_radius=mm(0.5), wall=mm(0.2),
                                                      length=mm(10.0))))
    chosen = {"steady": Scheme.Steady(), "imex": Scheme.IMEXEuler(), "rk3": Scheme.RK3(), None: None}[scheme]
    flow = FVM(domain=PeriodicBox(length=mm(period)), cells_across=cells_across, stokes=scheme != "rk3",
               scheme=chosen, wall=Band(delta_in=delta_in))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(1.0), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    if plug:
        exp.add_structure(Structure("clot", geometry=Plug(axis=+Z, radius=mm(radius), length=mm(length),
                                                          centre=mm(centre))))
        exp.declare("clot.permeability", m2(permeability), reason="a gate's value")
        exp.attach(Darcy(), medium="water", structure="clot")
    return exp
