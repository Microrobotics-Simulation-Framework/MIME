"""Script A0 through the public API, with the values phase 0's axis-fixed reference ran with: the AR4 arm carrying a
motor whose rotor carries de Jongh's 35 x 20 mm N45 magnet (a point dipole), 15 cm above the body in script A's tube.
The body is script A's (FL-9, in the tube along x, its drag surrogate with lubrication on the weights the reference
used, ``dejongh_mlp_v2``). The arm's joint positions are written every step by ``arm_tracking`` with the 0.2 line's
settings, and the rotor turns at the field's frequency.

The reference's frame had the tube along x and gravity along -z, as here; its body started at (0, 0, -1) mm with its
axis along the tube. The helix's axis is body x in 0.3, and the moment body y, as script A's.

The surrogate's weights are the reference's, and are not validated here: MIME-ANO-117 records that they inherit the old
wall tables' defect. A0 compares the framework with its reference on the same weights."""

from __future__ import annotations

import math
from typing import Any

from chain_cases import AR4_BASE, AR4_HOME, MAGNET_MOMENT, TRACKING_READS, TRACKING_WRITES, rpm, tracking
from script_a import (
    EPSILON,
    INERTIA,
    MASS,
    MOMENT,
    REASON,
    REFERENCE_START,
    VISCOSITY,
    VOLUME,
    WATER,
    fl9,
)

from mime import Body, Experiment, Medium, State, Structure
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Scene, Tube, X, Z
from mime.things import Part
from mime.units import A_m2, Hz, m, m_per_s, m_per_s2, mm, ms, rad_per_s

__all__ = ["a0"]


def a0(name: str = "script_a0", *, frequency=Hz(3.0), dt=ms(0.5), mode: Any = "graph", rate=None, latency=None,
       controller=None, joints=AR4_HOME, moment: float = MAGNET_MOMENT, servo=None, magnet_at=None,
       arm_at=AR4_BASE, start=REFERENCE_START, **settings) -> Experiment:
    """A0: ``rpm`` closed by ``tracking`` every ``rate`` (every step when not given), the body script A's. ``frequency``
    is the rotor's. The keywords after it are for the controls' planted faults: ``controller`` replaces the
    controller, ``moment`` is the rotor magnet's, ``magnet_at`` its place on the rotor, ``joints`` the arm's initial
    positions, ``servo`` its time constant, ``arm_at`` the arm's base (the controller still believes it at
    ``AR4_BASE``), ``latency`` the controller's, ``settings`` the controller's own."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt)
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                             length=mm(100.0))))
    water = exp.add_medium(Medium("water", fills=tube.lumen))
    exp.add_apparatus(rpm(speed=2 * math.pi * frequency.value, controlled=("arm.joint_positions",), joints=joints,
                          magnet_at=magnet_at, servo=servo, arm_at=arm_at))
    exp.declare("rpm.magnetic_moment", A_m2(moment), reason="de Jongh's 35 x 20 mm N45 rotor magnet, 18.89 A m^2")
    umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=start, orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(fl9())}))
    for path, value in (("umr.inertia", INERTIA), ("umr.mass", MASS), ("umr.buoyant_volume", VOLUME),
                        ("umr.magnetic_moment", MOMENT), ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))),
                        ("umr.magnet_centre", m((0.0, 0.0, 0.0))), ("water.density", WATER),
                        ("water.viscosity", VISCOSITY)):
        exp.declare(path, value, reason=REASON)
    exp.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=EPSILON)),
               body=umr, medium=water)
    exp.attach(Magnetic.Torque(), body=umr, field="rpm.magnetic")
    exp.attach(Magnetic.Force(), body=umr, field="rpm.magnetic")
    exp.attach(Gravity(), body=umr, medium=water)
    exp.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    exp.observe(SwimSpeed(umr, along=tube.axis))
    exp.set_controller(controller if controller is not None else tracking(**settings), reads=TRACKING_READS,
                       writes=TRACKING_WRITES, mode=mode, rate=rate if rate is not None else dt, latency=latency)
    return exp
