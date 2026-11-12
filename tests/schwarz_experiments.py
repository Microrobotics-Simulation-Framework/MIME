"""An MRS sphere in a medium the FVM fills (F5's Schwarz coupling): a small periodic tube, IMEX Euler, Stokes, driven
along its axis; with a second body, a wall kind written, an open vessel or the drag's settings, the cases its plan
refuses. The switch scan traces its flow's step (``tests/saved/switch_configurations.py``)."""

from __future__ import annotations

import math

from mime.dynamics import FVM, RigidInertial
from mime.dynamics.flow._fvm import Scheme
from mime.experiment import Experiment
from mime.interactions import Drag, Inflow, Windkessel
from mime.interactions.flow import BodyForce
from mime.scene import PeriodicBox, Scene, Sphere, Tube, VesselRegion, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import Pa_s, Pa_s_per_m3, kg, kg_m2, kg_per_m3, m3_per_Pa, m_per_s, m_per_s2, mL_per_s, mm, rad_per_s, \
    s, um


def schwarz_sphere(*, second: str = "", wall_kind: str | None = None, opened: bool = False,
               settings: dict | None = None) -> Experiment:
    """A neutral MRS sphere, ``ball``, 0.05 mm off a 1 mm tube's axis in water the FVM solves (8 cells across);
    ``second`` another body (``pebble`` a point body, else an MRS one), ``wall_kind`` written on the drag, ``opened``
    the vessel open at its ends, ``settings`` the drag's others."""
    exp = Experiment("schwarz_sphere", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(1e-3))
    exp.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(0.5), wall=mm(0.1),
                                                      length=mm(4.0))))
    domain = VesselRegion(margin=4) if opened else PeriodicBox(length=mm(1.0))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": FVM(domain=domain, stokes=True, cells_across=8,
                                                                          scheme=Scheme.IMEXEuler())}))
    exp.declare("water.viscosity", Pa_s(1e-3), reason="water")
    exp.declare("water.density", kg_per_m3(1e3), reason="water")
    radius = 150e-6
    mass = 1e3 * 4.0 / 3.0 * math.pi * radius ** 3
    bodies = ["ball", *([second] if second else [])]
    for k, name in enumerate(bodies):
        exp.add_body(Body(name, dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=um(radius * 1e6)))},
                          initial=State(position=mm((0.05, 0.0, 0.4 * k)), orientation=(1.0, 0.0, 0.0, 0.0),
                                        velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        exp.declare(f"{name}.mass", kg(mass), reason="neutral")
        exp.declare(f"{name}.inertia", kg_m2((0.4 * mass * radius ** 2,) * 3), reason="uniform")
    written = {"wall_kind": wall_kind} if wall_kind else {}
    exp.attach(Drag.MRS(spacing=um(100.0), **written, **(settings or {})), body="ball", medium="water")
    if second == "pebble":
        exp.attach(Drag.Stokes(), body="pebble", medium="water")
    elif second:
        exp.attach(Drag.MRS(spacing=um(100.0)), body=second, medium="water")
    if opened:
        exp.attach(Inflow(steady=mL_per_s(1e-3)), medium="water", surface="tube.start")
        exp.attach(Windkessel(proximal=Pa_s_per_m3(1e8), compliance=m3_per_Pa(1e-13), distal=Pa_s_per_m3(1e9)),
                   medium="water", surface="tube.end")
    else:
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    return exp


__all__ = ["schwarz_sphere"]
