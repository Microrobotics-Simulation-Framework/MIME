"""An MRS sphere in a medium the FVM fills (F5's Schwarz coupling): a small periodic tube, IMEX Euler, Stokes, driven
along its axis; with a second body, a wall kind written, an open vessel or the drag's settings, the cases its plan
refuses. The switch scan traces its flow's step (``tests/saved/switch_configurations.py``)."""

from __future__ import annotations

import math

from mime.dynamics import FVM, RigidInertial
from mime.dynamics.flow._fvm import Scheme
from mime.experiment import Experiment
from mime.fields import UniformRotating
from mime.interactions import Drag, Inflow, Magnetic, Windkessel
from mime.interactions.flow import BodyForce
from mime.scene import PeriodicBox, Scene, Sphere, Tube, VesselRegion, Z
from mime.things import Apparatus, Body, Medium, Part, State, Structure
from mime.units import A_m2, Hz, Pa_s, Pa_s_per_m3, kg, kg_m2, kg_per_m3, m3_per_Pa, m_per_s, m_per_s2, mL_per_s, mm, \
    mT, rad_per_s, s, um


def schwarz_sphere(*, second: str = "", wall_kind: str | None = None, opened: bool = False,
                   settings: dict | None = None, viscosity: float = 1e-3, flowing: bool = True,
                   offset: float = 0.05, length: float = 4.0, magnetic: bool = False,
                   amplitude: float = 1.0, drive: float = 1.0) -> Experiment:
    """A neutral MRS sphere, ``ball``, 0.05 mm off a 1 mm tube's axis in water the FVM solves (8 cells across);
    ``second`` another body (``pebble`` a point body, else an MRS one), ``wall_kind`` written on the drag, ``opened``
    the vessel open at its ends, ``settings`` the drag's others, ``viscosity`` the water's (Pa s); ``flowing=False``
    the water at rest, no flow solved (no Schwarz coupling); ``offset`` the sphere's distance from the axis (mm);
    ``length`` the tube's (mm); ``magnetic`` the ball a magnet (1e-11 A m^2 along its x) in a field turning about the
    tube's axis at 0.1 Hz (``field``, 1 mT; the rotation's Reynolds number 0.014), its torque about the axis moving
    the off-axis ball along the wall only through the wall's reflection; ``amplitude`` the field's (mT); ``drive``
    the periodic water's body force along the axis (m/s^2), none at zero."""
    exp = Experiment("schwarz_sphere", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(1e-3))
    exp.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(0.5), wall=mm(0.1),
                                                      length=mm(length))))
    domain = VesselRegion(margin=4) if opened else PeriodicBox(length=mm(1.0))
    fields = {"flow": FVM(domain=domain, stokes=True, cells_across=8, scheme=Scheme.IMEXEuler())} if flowing else {}
    exp.add_medium(Medium("water", fills="tube.lumen", fields=fields))
    exp.declare("water.viscosity", Pa_s(viscosity), reason="water")
    exp.declare("water.density", kg_per_m3(1e3), reason="water")
    radius = 150e-6
    mass = 1e3 * 4.0 / 3.0 * math.pi * radius ** 3
    bodies = ["ball", *([second] if second else [])]
    for k, name in enumerate(bodies):
        exp.add_body(Body(name, dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=um(radius * 1e6)))},
                          initial=State(position=mm((offset, 0.0, 0.4 * k)), orientation=(1.0, 0.0, 0.0, 0.0),
                                        velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        exp.declare(f"{name}.mass", kg(mass), reason="neutral")
        exp.declare(f"{name}.inertia", kg_m2((0.4 * mass * radius ** 2,) * 3), reason="uniform")
    if magnetic:
        field = exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(
            rotation_axis=(0.0, 0.0, 1.0), initial_direction=(0.0, 1.0, 0.0))},
            commands={"frequency": Hz(0.1), "amplitude": mT(amplitude)}))
        exp.declare("ball.magnetic_moment", A_m2((1e-11, 0.0, 0.0)), reason="a weak magnet: the torque's rotation small")
        exp.attach(Magnetic.Torque(), body="ball", field=field.magnetic)
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
    elif flowing and drive:
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(drive)), medium="water")
    return exp


__all__ = ["schwarz_sphere"]
